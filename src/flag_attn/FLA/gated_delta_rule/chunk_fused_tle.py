# Copyright (c) 2026 FlagAttention contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

TLE_CHUNK_GDR_BLOCK_S = 64
TLE_CHUNK_GDR_HEAD_DIM = 128
TLE_CHUNK_GDR_PIPE_CAPACITY = 2
TLE_CHUNK_GDR_TARGET_NUM_CTAS = 128


def _tle_chunk_gdr_target_num_ctas(device):
    try:
        props = torch.cuda.get_device_properties(device)
        return max(1, int(props.multi_processor_count * 0.7))
    except Exception:
        return TLE_CHUNK_GDR_TARGET_NUM_CTAS


def _tle_chunk_gdr_block_dv(batch_size, num_value_heads, target_num_ctas):
    grid_size = batch_size * num_value_heads
    if grid_size >= target_num_ctas:
        return 128
    if grid_size * 2 >= target_num_ctas:
        return 64
    return 32


def _resolve_tle_chunk_gdr_block_dv(batch_size, num_value_heads, value_dim, device):
    block_dv = _tle_chunk_gdr_block_dv(batch_size, num_value_heads, _tle_chunk_gdr_target_num_ctas(device))
    return block_dv if value_dim % block_dv == 0 else value_dim


def _set_tle_descriptor_allocator(device):
    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


@triton.jit
def _tle_chunk_gdr_linear_pids(
    H: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    bbhv = tl.program_id(0)
    num_v_blocks: tl.constexpr = tl.cdiv(DV, BLOCK_DV)
    bbh = bbhv // num_v_blocks
    pid_v = bbhv - bbh * num_v_blocks
    pid_b = bbh // H
    pid_h = bbh - pid_b * H
    return pid_v, pid_b, pid_h


@triton.jit
def _tle_chunk_gdr_qk_producer(
    qk_writer,
    q_desc,
    k_desc,
    pid_b,
    pid_hg,
    T: tl.constexpr,
    DK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)

    for chunk in tl.range(num_chunks):
        qk_slot = qk_writer.acquire(chunk)
        left = chunk * CHUNK
        tle.gpu.copy(
            q_desc,
            qk_slot.q,
            [1, 1, CHUNK, DK],
            [pid_b, pid_hg, left, 0],
        )
        tle.gpu.copy(
            k_desc,
            qk_slot.k,
            [1, 1, CHUNK, DK],
            [pid_b, pid_hg, left, 0],
        )
        qk_writer.commit(chunk)


@triton.jit
def _tle_chunk_gdr_vbeta_producer(
    vbeta_writer,
    v_desc,
    beta,
    pid_b,
    pid_h,
    block_v,
    T: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    offs_s = tl.arange(0, CHUNK)
    num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)
    gh_batch_base: tl.constexpr = T * H

    for chunk in tl.range(num_chunks):
        vbeta_slot = vbeta_writer.acquire(chunk)
        left = chunk * CHUNK
        token = left + offs_s
        tle.gpu.copy(
            v_desc,
            vbeta_slot.v,
            [1, 1, CHUNK, BLOCK_DV],
            [pid_b, pid_h, left, block_v],
        )
        beta_ptrs = beta + pid_b * gh_batch_base + token * H + pid_h
        beta_blk = tl.load(beta_ptrs)
        tl.store(tle.gpu.local_ptr(vbeta_slot.beta), beta_blk)
        vbeta_writer.commit(chunk)


@triton.jit
def _tle_chunk_gdr_ag_producer(
    agamma_writer,
    a_desc,
    g,
    pid_b,
    pid_h,
    T: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
):
    offs_s = tl.arange(0, CHUNK)
    num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)
    gh_batch_base: tl.constexpr = T * H

    for chunk in tl.range(num_chunks):
        agamma_slot = agamma_writer.acquire(chunk)
        left = chunk * CHUNK
        token = left + offs_s
        tle.gpu.copy(
            a_desc,
            agamma_slot.a,
            [1, 1, CHUNK, CHUNK],
            [pid_b, pid_h, left, 0],
        )
        g_ptrs = g + pid_b * gh_batch_base + token * H + pid_h
        g_blk = tl.load(g_ptrs)
        tl.store(tle.gpu.local_ptr(agamma_slot.g), g_blk)
        agamma_writer.commit(chunk)


@triton.jit
def _tle_chunk_gdr_store_worker(
    o_reader,
    o_desc,
    pid_b,
    pid_h,
    block_v,
    T: tl.constexpr,
    H: tl.constexpr,
    DV: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)

    for chunk in tl.range(1, num_chunks):
        store_chunk = chunk - 1
        wait = o_reader.wait(store_chunk)
        slot = wait.slot
        left = store_chunk * CHUNK
        values = tl.load(tle.gpu.local_ptr(slot.o))
        rows = left + tl.arange(0, CHUNK)
        cols = block_v + tl.arange(0, BLOCK_DV)
        tl.store(o_desc + ((pid_b * T + rows[:, None]) * H + pid_h) * DV + cols[None, :], values)
        o_reader.release(store_chunk)

    last_chunk: tl.constexpr = ((T + CHUNK - 1) // CHUNK) - 1
    wait = o_reader.wait(last_chunk)
    slot = wait.slot
    left = last_chunk * CHUNK
    values = tl.load(tle.gpu.local_ptr(slot.o))
    rows = left + tl.arange(0, CHUNK)
    cols = block_v + tl.arange(0, BLOCK_DV)
    tl.store(o_desc + ((pid_b * T + rows[:, None]) * H + pid_h) * DV + cols[None, :], values)
    o_reader.release(last_chunk)


@triton.jit
def _recurrence_worker(
    qk_reader,
    vb_reader,
    ag_reader,
    h_writer,
    vd_writer,
    final_state,
    pid_b,
    pid_h,
    block_v,
    T: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    rows = tl.arange(0, CHUNK)
    h = tl.zeros((DK, BLOCK_DV), tl.float32)
    for chunk in tl.range(T // CHUNK):
        hs = h_writer.acquire(chunk)
        tl.store(tle.gpu.local_ptr(hs.h), h.to(hs.h.dtype))
        h_writer.commit(chunk)
        qs = qk_reader.wait(chunk).slot
        vs = vb_reader.wait(chunk).slot
        gs = ag_reader.wait(chunk).slot
        k = tl.load(tle.gpu.local_ptr(qs.k))
        v = tl.load(tle.gpu.local_ptr(vs.v))
        beta = tl.load(tle.gpu.local_ptr(vs.beta)).to(tl.float32)
        a = tl.load(tle.gpu.local_ptr(gs.a)).to(tl.float32)
        g = tl.load(tle.gpu.local_ptr(gs.g)).to(tl.float32)
        last = tl.sum(tl.where(rows == CHUNK - 1, g, 0.0))
        ge = tl.exp(g)
        lower = rows[:, None] >= rows[None, :]
        gate = tl.where(lower, tl.exp(tl.where(lower, g[:, None] - g[None, :], 0.0)), 0.0)
        ag = (a * gate * beta[None, :]).to(k.dtype)
        u = tl.dot(k, h.to(k.dtype))
        w = (v.to(tl.float32) - ge[:, None] * u).to(k.dtype)
        vd = tl.dot(ag, w)
        ds = vd_writer.acquire(chunk)
        tl.store(tle.gpu.local_ptr(ds.vd), vd.to(k.dtype))
        vd_writer.commit(chunk)
        h *= tl.exp(last)
        h += tl.dot(tl.trans(k), (tl.exp(last - g)[:, None] * vd).to(k.dtype))
        qk_reader.release(chunk)
        vb_reader.release(chunk)
        ag_reader.release(chunk)
    kk = tl.arange(0, DK)
    vv = block_v + tl.arange(0, BLOCK_DV)
    tl.store(final_state + ((pid_b * H + pid_h) * DK + kk[:, None]) * DV + vv[None, :], h)


@triton.jit
def _output_worker(
    qk_reader, ag_reader, h_reader, vd_reader, o_writer, T: tl.constexpr, CHUNK: tl.constexpr, scale: tl.constexpr
):
    rows = tl.arange(0, CHUNK)
    for chunk in tl.range(T // CHUNK):
        qs = qk_reader.wait(chunk).slot
        gs = ag_reader.wait(chunk).slot
        hs = h_reader.wait(chunk).slot
        q = tl.load(tle.gpu.local_ptr(qs.q))
        k = tl.load(tle.gpu.local_ptr(qs.k))
        g = tl.load(tle.gpu.local_ptr(gs.g)).to(tl.float32)
        h = tl.load(tle.gpu.local_ptr(hs.h))
        lower = rows[:, None] >= rows[None, :]
        gate = tl.where(lower, tl.exp(tl.where(lower, g[:, None] - g[None, :], 0.0)), 0.0)
        p = tl.dot(q, tl.trans(k))
        out = tl.dot(q, h) * (scale * tl.exp(g)[:, None])
        ds = vd_reader.wait(chunk).slot
        vd = tl.load(tle.gpu.local_ptr(ds.vd))
        out += tl.dot((p * (scale * gate)).to(q.dtype), vd)
        os = o_writer.acquire(chunk)
        tl.store(tle.gpu.local_ptr(os.o), out.to(q.dtype))
        o_writer.commit(chunk)
        qk_reader.release(chunk)
        ag_reader.release(chunk)
        h_reader.release(chunk)
        vd_reader.release(chunk)


@triton.jit
def _tle_chunk_gdr_fwd_kernel(
    q_desc,
    k_desc,
    v_desc,
    a_desc,
    g,
    beta,
    o_desc,
    final_state,
    T: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    PIPE_CAPACITY: tl.constexpr,
    scale: tl.constexpr,
):
    pid_v, pid_b, pid_h = _tle_chunk_gdr_linear_pids(H, DV, BLOCK_DV)
    block_v = pid_v * BLOCK_DV
    pid_hg = pid_h // (H // HG)
    qs = tle.gpu.alloc([PIPE_CAPACITY, CHUNK, DK], q_desc.dtype, scope=tle.gpu.smem)
    ks = tle.gpu.alloc([PIPE_CAPACITY, CHUNK, DK], k_desc.dtype, scope=tle.gpu.smem)
    vs = tle.gpu.alloc([PIPE_CAPACITY, CHUNK, BLOCK_DV], v_desc.dtype, scope=tle.gpu.smem)
    bs = tle.gpu.alloc([PIPE_CAPACITY, CHUNK], beta.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    aa = tle.gpu.alloc([PIPE_CAPACITY, CHUNK, CHUNK], a_desc.dtype, scope=tle.gpu.smem)
    gs = tle.gpu.alloc([PIPE_CAPACITY, CHUNK], g.dtype.element_ty, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    os = tle.gpu.alloc([1, CHUNK, BLOCK_DV], q_desc.dtype, scope=tle.gpu.smem)
    hh = tle.gpu.alloc([1, DK, BLOCK_DV], q_desc.dtype, scope=tle.gpu.smem)
    dd = tle.gpu.alloc([1, CHUNK, BLOCK_DV], q_desc.dtype, scope=tle.gpu.smem)
    qp = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="qk", readers=("rec", "out"), q=qs, k=ks)
    vp = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="vb", v=vs, beta=bs)
    ap = tle.pipe(capacity=PIPE_CAPACITY, scope="cta", name="ag", readers=("rec", "out"), a=aa, g=gs)
    hp = tle.pipe(capacity=1, scope="cta", name="h", h=hh)
    dp = tle.pipe(capacity=1, scope="cta", name="vd", vd=dd)
    op = tle.pipe(capacity=1, scope="cta", name="out", o=os)
    tle.gpu.warp_specialize(
        [
            (
                _recurrence_worker,
                (
                    qp.reader("rec", fields=("k",)),
                    vp.reader(),
                    ap.reader("rec"),
                    hp.writer(),
                    dp.writer(),
                    final_state,
                    pid_b,
                    pid_h,
                    block_v,
                    T,
                    H,
                    DK,
                    DV,
                    CHUNK,
                    BLOCK_DV,
                ),
            ),
            (
                _output_worker,
                (
                    qp.reader("out"),
                    ap.reader("out", fields=("g",)),
                    hp.reader(),
                    dp.reader(),
                    op.writer(),
                    T,
                    CHUNK,
                    scale,
                ),
            ),
            (_tle_chunk_gdr_qk_producer, (qp.writer(), q_desc, k_desc, pid_b, pid_hg, T, DK, CHUNK)),
            (_tle_chunk_gdr_vbeta_producer, (vp.writer(), v_desc, beta, pid_b, pid_h, block_v, T, H, CHUNK, BLOCK_DV)),
            (_tle_chunk_gdr_ag_producer, (ap.writer(), a_desc, g, pid_b, pid_h, T, H, CHUNK)),
            (_tle_chunk_gdr_store_worker, (op.reader(), o_desc, pid_b, pid_h, block_v, T, H, DV, CHUNK, BLOCK_DV)),
        ],
        [4, 1, 1, 1, 1],
        [128, 32, 32, 32, 32],
    )


def _tle_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    block_dv = _resolve_tle_chunk_gdr_block_dv(B, H, V, q.device)

    _set_tle_descriptor_allocator(q.device)
    o = torch.empty_like(v)
    final_state = torch.empty((B, H, K, V), device=v.device, dtype=torch.float32)

    q_desc = TensorDescriptor(
        q,
        shape=[B, Hg, T, K],
        strides=[T * Hg * K, K, Hg * K, 1],
        block_shape=[1, 1, TLE_CHUNK_GDR_BLOCK_S, K],
    )
    k_desc = TensorDescriptor(
        k,
        shape=[B, Hg, T, K],
        strides=[T * Hg * K, K, Hg * K, 1],
        block_shape=[1, 1, TLE_CHUNK_GDR_BLOCK_S, K],
    )
    v_desc = TensorDescriptor(
        v,
        shape=[B, H, T, V],
        strides=[T * H * V, V, H * V, 1],
        block_shape=[1, 1, TLE_CHUNK_GDR_BLOCK_S, block_dv],
    )
    a_desc = TensorDescriptor(
        a,
        shape=[B, H, T, TLE_CHUNK_GDR_BLOCK_S],
        strides=[
            T * H * TLE_CHUNK_GDR_BLOCK_S,
            TLE_CHUNK_GDR_BLOCK_S,
            H * TLE_CHUNK_GDR_BLOCK_S,
            1,
        ],
        block_shape=[1, 1, TLE_CHUNK_GDR_BLOCK_S, TLE_CHUNK_GDR_BLOCK_S],
    )
    grid = (triton.cdiv(V, block_dv) * B * H,)
    _tle_chunk_gdr_fwd_kernel[grid](
        q_desc,
        k_desc,
        v_desc,
        a_desc,
        g_cumsum,
        beta,
        o,
        final_state,
        T,
        H,
        Hg,
        K,
        V,
        TLE_CHUNK_GDR_BLOCK_S,
        block_dv,
        TLE_CHUNK_GDR_PIPE_CAPACITY,
        scale,
        num_warps=4,
    )
    return o, final_state
