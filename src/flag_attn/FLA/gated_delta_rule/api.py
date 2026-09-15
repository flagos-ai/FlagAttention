import os

import torch
import triton
import triton.language as tl

from flag_attn.FLA.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from flag_attn.FLA.gated_delta_rule.chunk_gated_delta_direct import (
    can_use_chunk_gated_delta_rule_direct,
    chunk_gated_delta_rule_direct_fwd,
)
from flag_attn.FLA.gated_delta_rule.chunk_fused_forward import can_use_two_kernel_fused_forward
from flag_attn.FLA.compat import libentry
from flag_attn.utils import has_triton_tle
from flag_attn.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_CHUNK_GATED_DELTA_RULE = True
    except ImportError:
        tle = None
        HAS_TLE_CHUNK_GATED_DELTA_RULE = False
else:
    tle = None
    HAS_TLE_CHUNK_GATED_DELTA_RULE = False


TLE_CHUNK_GDR_BLOCK_S = 64
TLE_CHUNK_GDR_HEAD_DIM = 128
TLE_CHUNK_GDR_PIPE_CAPACITY = 2
TLE_CHUNK_GDR_TARGET_NUM_CTAS = 128


@libentry()
@triton.jit
def _l2_normalize_last_dim_kernel(
    x,
    out,
    n_rows: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    h = row % H
    row_bt = row // H
    t = row_bt % n_rows
    b = row_bt // n_rows
    x_base = x + b * stride_x_b + t * stride_x_t + h * stride_x_h
    values = tl.load(x_base + offs * stride_x_k, mask=mask, other=0.0).to(tl.float32)
    inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(values * values, axis=0)), 1e-6)
    tl.store(out + row * K + offs, values * inv_norm, mask=mask)


def _tle_chunk_gdr_enabled() -> bool:
    value = os.environ.get("FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}


def _tle_chunk_gdr_target_num_ctas(device: torch.device) -> int:
    try:
        props = torch.cuda.get_device_properties(device)
        return max(1, int(props.multi_processor_count * 0.7))
    except Exception:
        return TLE_CHUNK_GDR_TARGET_NUM_CTAS


def _tle_chunk_gdr_block_dv(
    batch_size: int, num_value_heads: int, target_num_ctas: int
) -> int:
    grid_size = batch_size * num_value_heads
    if grid_size >= target_num_ctas:
        return 128
    if grid_size * 2 >= target_num_ctas:
        return 64
    return 32


def _resolve_tle_chunk_gdr_block_dv(
    batch_size: int,
    num_value_heads: int,
    value_dim: int,
    device: torch.device,
) -> int:
    target_num_ctas = _tle_chunk_gdr_target_num_ctas(device)
    block_dv = _tle_chunk_gdr_block_dv(batch_size, num_value_heads, target_num_ctas)
    if value_dim % block_dv != 0:
        return value_dim
    return block_dv


def _set_tle_descriptor_allocator(device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def _can_use_tle_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    if not (HAS_TLE_CHUNK_GATED_DELTA_RULE and _tle_chunk_gdr_enabled()):
        return False
    if not output_final_state:
        return False
    if initial_state is not None or cu_seqlens is not None:
        return False
    if q.device.type != "cuda":
        return False
    if not all(x.device == q.device for x in (k, v, beta, g)):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not all(x.dtype == q.dtype for x in (k, v, beta, g)):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if beta.ndim != 3 or g.ndim != 3:
        return False

    B, T, Hg, K = q.shape
    H = v.shape[2]
    return (
        k.shape == (B, T, Hg, K)
        and v.shape[:2] == (B, T)
        and beta.shape == (B, T, H)
        and g.shape == (B, T, H)
        and H % Hg == 0
        and K == TLE_CHUNK_GDR_HEAD_DIM
        and v.shape[3] == TLE_CHUNK_GDR_HEAD_DIM
        # Keep low-workload shapes on the numerically stable native path.
        and B * H >= 32
        and T >= 2048
        and T % TLE_CHUNK_GDR_BLOCK_S == 0
    )


if HAS_TLE_CHUNK_GATED_DELTA_RULE:

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
    def _tle_chunk_gdr_state_worker(
        data_reader,
        h_writer,
        gate_reader,
        vn_reader,
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
        offs_k = tl.arange(0, DK)
        offs_v = tl.arange(0, BLOCK_DV)
        num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)
        h_batch_base: tl.constexpr = H * DK * DV
        h_state = tl.zeros((DK, BLOCK_DV), dtype=tl.float32)
        for chunk in tl.range(num_chunks):
            h_slot = h_writer.acquire(chunk)
            tl.store(tle.gpu.local_ptr(h_slot.h), h_state.to(h_slot.h.dtype))
            h_writer.commit(chunk)

            gate_wait = gate_reader.wait(chunk)
            gate_slot = gate_wait.slot
            g_last = tl.load(tle.gpu.local_ptr(gate_slot.g_exp, (CHUNK - 1,))).to(
                tl.float32
            )
            h_state *= g_last
            gate_reader.release(chunk)

            data_wait = data_reader.wait(chunk)
            data_slot = data_wait.slot
            vn_wait = vn_reader.wait(chunk)
            vn_slot = vn_wait.slot
            k_blk = tl.load(tle.gpu.local_ptr(data_slot.k))
            vn_blk = tl.load(tle.gpu.local_ptr(vn_slot.vn))
            h_state += tl.dot(tl.trans(k_blk), vn_blk, out_dtype=tl.float32)
            vn_reader.release(chunk)
            data_reader.release(chunk)

        h_ptrs = (
            final_state
            + pid_b * h_batch_base
            + pid_h * (DK * DV)
            + offs_k[:, None] * DV
            + block_v
            + offs_v[None, :]
        )
        tl.store(h_ptrs, h_state)

    @triton.jit
    def _tle_chunk_gdr_value_worker(
        qk_reader,
        vbeta_reader,
        agamma_reader,
        h_reader,
        gate_writer,
        computed_ag_reader,
        vd_writer,
        vn_writer,
        T: tl.constexpr,
        DK: tl.constexpr,
        CHUNK: tl.constexpr,
    ):
        num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)
        for chunk in tl.range(num_chunks):
            agamma_wait = agamma_reader.wait(chunk)
            agamma_slot = agamma_wait.slot

            g_vec = tl.load(tle.gpu.local_ptr(agamma_slot.g)).to(tl.float32)
            g_exp = tl.math.exp2(g_vec * 1.4426950408889634)
            g_last = tl.load(tle.gpu.local_ptr(agamma_slot.g, (CHUNK - 1,))).to(
                tl.float32
            )
            g_rev_exp = tl.math.exp2((g_last - g_vec) * 1.4426950408889634)

            gate_slot = gate_writer.acquire(chunk)
            tl.store(tle.gpu.local_ptr(gate_slot.g_exp), g_exp)
            tl.store(tle.gpu.local_ptr(gate_slot.g_rev), g_rev_exp)
            gate_writer.commit(chunk)

            h_wait = h_reader.wait(chunk)
            h_slot = h_wait.slot
            qk_wait = qk_reader.wait(chunk)
            qk_slot = qk_wait.slot
            vbeta_wait = vbeta_reader.wait(chunk)
            vbeta_slot = vbeta_wait.slot
            k_blk = tl.load(tle.gpu.local_ptr(qk_slot.k))
            h_blk = tl.load(tle.gpu.local_ptr(h_slot.h))
            v_blk = tl.load(tle.gpu.local_ptr(vbeta_slot.v))
            u = tl.dot(k_blk, h_blk, out_dtype=tl.float32)
            h_reader.release(chunk)

            w = v_blk.to(tl.float32) - g_exp[:, None] * u
            tl.store(tle.gpu.local_ptr(vbeta_slot.v), w.to(v_blk.dtype))

            ag_wait = computed_ag_reader.wait(chunk)
            ag_slot = ag_wait.slot
            ag_blk = tl.load(tle.gpu.local_ptr(ag_slot.ag))
            w_blk = tl.load(tle.gpu.local_ptr(vbeta_slot.v))
            vd = tl.dot(ag_blk, w_blk, out_dtype=tl.float32)
            vn = g_rev_exp[:, None] * vd
            qk_reader.release(chunk)
            vbeta_reader.release(chunk)
            agamma_reader.release(chunk)
            computed_ag_reader.release(chunk)

            vd_slot = vd_writer.acquire(chunk)
            tl.store(tle.gpu.local_ptr(vd_slot.vd), vd.to(ag_blk.dtype))
            vd_writer.commit(chunk)

            vn_slot = vn_writer.acquire(chunk)
            tl.store(tle.gpu.local_ptr(vn_slot.vn), vn.to(ag_blk.dtype))
            vn_writer.commit(chunk)

    @triton.jit
    def _tle_chunk_gdr_output_worker(
        qk_reader,
        vbeta_reader,
        agamma_reader,
        h_reader,
        gate_reader,
        ag_writer,
        vd_reader,
        p_smem,
        o_writer,
        T: tl.constexpr,
        DK: tl.constexpr,
        CHUNK: tl.constexpr,
        scale: tl.constexpr,
    ):
        rows = tl.arange(0, CHUNK)[:, None]
        cols = tl.arange(0, CHUNK)[None, :]
        num_chunks: tl.constexpr = tl.cdiv(T, CHUNK)
        for chunk in tl.range(num_chunks):
            qk_wait = qk_reader.wait(chunk)
            qk_slot = qk_wait.slot
            q_blk = tl.load(tle.gpu.local_ptr(qk_slot.q))
            k_blk = tl.load(tle.gpu.local_ptr(qk_slot.k))
            p = tl.dot(q_blk, tl.trans(k_blk), out_dtype=tl.float32)

            vbeta_wait = vbeta_reader.wait(chunk)
            vbeta_slot = vbeta_wait.slot
            agamma_wait = agamma_reader.wait(chunk)
            agamma_slot = agamma_wait.slot
            a_blk = tl.load(tle.gpu.local_ptr(agamma_slot.a)).to(tl.float32)
            beta_vec = tl.load(tle.gpu.local_ptr(vbeta_slot.beta)).to(tl.float32)
            g_vec = tl.load(tle.gpu.local_ptr(agamma_slot.g)).to(tl.float32)
            lower = rows >= cols
            g_mat = tl.where(
                lower,
                tl.math.exp2((g_vec[:, None] - g_vec[None, :]) * 1.4426950408889634),
                0.0,
            )
            ag = a_blk * g_mat * beta_vec[None, :]
            ag_slot = ag_writer.acquire(chunk)
            tl.store(tle.gpu.local_ptr(ag_slot.ag), ag.to(q_blk.dtype))
            ag_writer.commit(chunk)
            qk_reader.release(chunk)
            vbeta_reader.release(chunk)
            agamma_reader.release(chunk)

            h_wait = h_reader.wait(chunk)
            h_slot = h_wait.slot
            gate_wait = gate_reader.wait(chunk)
            gate_slot = gate_wait.slot
            h_blk = tl.load(tle.gpu.local_ptr(h_slot.h))
            g_exp = tl.load(tle.gpu.local_ptr(gate_slot.g_exp)).to(tl.float32)
            o_blk = tl.dot(q_blk, h_blk, out_dtype=tl.float32)
            h_reader.release(chunk)

            pg = p * (scale * g_mat)
            tl.store(tle.gpu.local_ptr(p_smem), pg.to(q_blk.dtype))
            p_blk = tl.load(tle.gpu.local_ptr(p_smem))
            o_blk *= scale * g_exp[:, None]
            gate_reader.release(chunk)

            vd_wait = vd_reader.wait(chunk)
            vd_slot = vd_wait.slot
            vd_blk = tl.load(tle.gpu.local_ptr(vd_slot.vd))
            o_blk += tl.dot(p_blk, vd_blk, out_dtype=tl.float32)
            vd_reader.release(chunk)

            o_slot = o_writer.acquire(chunk)
            tl.store(tle.gpu.local_ptr(o_slot.o), o_blk.to(q_blk.dtype))
            o_writer.commit(chunk)

    @triton.jit
    def _tle_chunk_gdr_store_worker(
        o_reader,
        o_desc,
        pid_b,
        pid_h,
        block_v,
        T: tl.constexpr,
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
            tle.gpu.copy(
                slot.o,
                o_desc,
                [1, 1, CHUNK, BLOCK_DV],
                [pid_b, pid_h, left, block_v],
            )
            o_reader.release(store_chunk)

        last_chunk: tl.constexpr = ((T + CHUNK - 1) // CHUNK) - 1
        wait = o_reader.wait(last_chunk)
        slot = wait.slot
        left = last_chunk * CHUNK
        tle.gpu.copy(
            slot.o,
            o_desc,
            [1, 1, CHUNK, BLOCK_DV],
            [pid_b, pid_h, left, block_v],
        )
        o_reader.release(last_chunk)

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
        if H != HG:
            pid_hg = pid_h // (H // HG)
        else:
            pid_hg = pid_h

        q_smem = tle.gpu.alloc(
            [PIPE_CAPACITY, CHUNK, DK],
            dtype=q_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        k_smem = tle.gpu.alloc(
            [PIPE_CAPACITY, CHUNK, DK],
            dtype=k_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        v_smem = tle.gpu.alloc(
            [PIPE_CAPACITY, CHUNK, BLOCK_DV],
            dtype=v_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        a_smem = tle.gpu.alloc(
            [PIPE_CAPACITY, CHUNK, CHUNK],
            dtype=a_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        g_smem = tle.gpu.alloc(
            [PIPE_CAPACITY, CHUNK],
            dtype=g.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        beta_smem = tle.gpu.alloc(
            [PIPE_CAPACITY, CHUNK],
            dtype=beta.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        h_smem = tle.gpu.alloc(
            [1, DK, BLOCK_DV],
            dtype=q_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        g_exp_smem = tle.gpu.alloc(
            [1, CHUNK],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        g_rev_exp_smem = tle.gpu.alloc(
            [1, CHUNK],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        vd_smem = tle.gpu.alloc(
            [1, CHUNK, BLOCK_DV],
            dtype=q_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        vn_smem = tle.gpu.alloc(
            [1, CHUNK, BLOCK_DV],
            dtype=q_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        p_smem = tle.gpu.alloc(
            [CHUNK, CHUNK],
            dtype=q_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )
        o_smem = tle.gpu.alloc(
            [1, CHUNK, BLOCK_DV],
            dtype=q_desc.dtype,
            layout=None,
            scope=tle.gpu.smem,
        )

        qk_pipe = tle.pipe(
            capacity=PIPE_CAPACITY,
            scope="cta",
            name="gdr_qk",
            readers=("state", "value", "output"),
            q=q_smem,
            k=k_smem,
        )
        vbeta_pipe = tle.pipe(
            capacity=PIPE_CAPACITY,
            scope="cta",
            name="gdr_vbeta",
            readers=("value", "output"),
            v=v_smem,
            beta=beta_smem,
        )
        agamma_pipe = tle.pipe(
            capacity=PIPE_CAPACITY,
            scope="cta",
            name="gdr_agamma",
            readers=("value", "output"),
            a=a_smem,
            g=g_smem,
        )
        h_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="gdr_h",
            readers=("value", "output"),
            h=h_smem,
        )
        gate_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="gdr_gate",
            readers=("state", "output"),
            g_exp=g_exp_smem,
            g_rev=g_rev_exp_smem,
        )
        ag_pipe = tle.pipe(
            capacity=PIPE_CAPACITY,
            scope="cta",
            name="gdr_ag",
            ag=a_smem,
        )
        vd_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="gdr_vd",
            readers=("output",),
            vd=vd_smem,
        )
        vn_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="gdr_vn",
            readers=("state",),
            vn=vn_smem,
        )
        o_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="gdr_o",
            readers=("store",),
            o=o_smem,
        )

        tle.gpu.warp_specialize(
            [
                (
                    _tle_chunk_gdr_state_worker,
                    (
                        qk_pipe.reader("state", fields=("k",)),
                        h_pipe.writer(),
                        gate_pipe.reader("state", fields=("g_exp",)),
                        vn_pipe.reader("state"),
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
                    _tle_chunk_gdr_value_worker,
                    (
                        qk_pipe.reader("value", fields=("k",)),
                        vbeta_pipe.reader("value", fields=("v",)),
                        agamma_pipe.reader("value", fields=("g",)),
                        h_pipe.reader("value"),
                        gate_pipe.writer(),
                        ag_pipe.reader(),
                        vd_pipe.writer(),
                        vn_pipe.writer(),
                        T,
                        DK,
                        CHUNK,
                    ),
                ),
                (
                    _tle_chunk_gdr_output_worker,
                    (
                        qk_pipe.reader("output", fields=("q", "k")),
                        vbeta_pipe.reader("output", fields=("beta",)),
                        agamma_pipe.reader("output", fields=("a", "g")),
                        h_pipe.reader("output"),
                        gate_pipe.reader("output", fields=("g_exp",)),
                        ag_pipe.writer(),
                        vd_pipe.reader("output"),
                        p_smem,
                        o_pipe.writer(),
                        T,
                        DK,
                        CHUNK,
                        scale,
                    ),
                ),
                (
                    _tle_chunk_gdr_qk_producer,
                    (
                        qk_pipe.writer(),
                        q_desc,
                        k_desc,
                        pid_b,
                        pid_hg,
                        T,
                        DK,
                        CHUNK,
                    ),
                ),
                (
                    _tle_chunk_gdr_vbeta_producer,
                    (
                        vbeta_pipe.writer(),
                        v_desc,
                        beta,
                        pid_b,
                        pid_h,
                        block_v,
                        T,
                        H,
                        CHUNK,
                        BLOCK_DV,
                    ),
                ),
                (
                    _tle_chunk_gdr_ag_producer,
                    (
                        agamma_pipe.writer(),
                        a_desc,
                        g,
                        pid_b,
                        pid_h,
                        T,
                        H,
                        CHUNK,
                    ),
                ),
                (
                    _tle_chunk_gdr_store_worker,
                    (
                        o_pipe.reader("store"),
                        o_desc,
                        pid_b,
                        pid_h,
                        block_v,
                        T,
                        DV,
                        CHUNK,
                        BLOCK_DV,
                    ),
                ),
            ],
            [4, 4, 1, 1, 1, 1],
            [128, 128, 32, 32, 32, 32],
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
        from triton.tools.tensor_descriptor import TensorDescriptor

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
        o_desc = TensorDescriptor(
            o,
            shape=[B, H, T, V],
            strides=[T * H * V, V, H * V, 1],
            block_shape=[1, 1, TLE_CHUNK_GDR_BLOCK_S, block_dv],
        )

        grid = (triton.cdiv(V, block_dv) * B * H,)
        _tle_chunk_gdr_fwd_kernel[grid](
            q_desc,
            k_desc,
            v_desc,
            a_desc,
            g_cumsum,
            beta,
            o_desc,
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


def _as_seq_first(
    x: torch.Tensor,
    *,
    name: str,
    head_first: bool,
    expected_ndim: int,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != expected_ndim:
        raise ValueError(f"{name} must be {expected_ndim}D, got shape {tuple(x.shape)}")
    if head_first:
        return x.transpose(1, 2)
    return x


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    B, T, Hg, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, H, V = v.shape

    tensors = {"k": k, "v": v, "beta": beta, "g": g}
    for name, tensor in tensors.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype != q.dtype:
            raise ValueError(f"{name} must have the same dtype as q")

    if (Bk, Tk, Hk, Kk) != (B, T, Hg, K):
        raise ValueError(
            "q and k must have matching [B, T, Hq, K] shapes after layout conversion"
        )
    if (Bv, Tv) != (B, T):
        raise ValueError("v must have matching B and T dimensions with q/k")
    if H % Hg != 0:
        raise ValueError("the q/k head count must divide the v head count")
    if beta.shape != (B, T, H):
        raise ValueError(
            f"beta must have shape {(B, T, H)} after layout conversion, got {tuple(beta.shape)}"
        )
    if g.shape != (B, T, H):
        raise ValueError(
            f"g must have shape {(B, T, H)} after layout conversion, got {tuple(g.shape)}"
        )
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if B != 1:
            raise ValueError("cu_seqlens packed varlen inputs must use B=1")

    if initial_state is not None:
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        if initial_state.dtype != q.dtype:
            raise ValueError("initial_state must have the same dtype as q")
        expected_n = B if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_shape = (expected_n, H, K, V)
        if initial_state.shape != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}"
            )


def _direct_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _l2_normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    B, T, H, K = x.shape
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    block_k = triton.next_power_of_2(K)
    _l2_normalize_last_dim_kernel[(B * T * H,)](
        x=x,
        out=out,
        n_rows=T,
        H=H,
        K=K,
        stride_x_b=x.stride(0),
        stride_x_t=x.stride(1),
        stride_x_h=x.stride(2),
        stride_x_k=x.stride(3),
        BLOCK_K=block_k,
    )
    return out


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Public wrapper for the chunk gated delta rule forward operator.

    Inputs follow common FLA layouts:
    - ``head_first=True``: q/k/v are ``[B, H, T, D]`` and beta/g are ``[B, H, T]``.
    - ``head_first=False``: q/k/v are ``[B, T, H, D]`` and beta/g are ``[B, T, H]``.

    q/k may use fewer heads than v when the q/k head count divides the v head count.
    """
    if BT != 64:
        raise ValueError("chunk_gated_delta_rule currently supports only BT=64")

    q_seq = _as_seq_first(q, name="q", head_first=head_first, expected_ndim=4)
    k_seq = _as_seq_first(k, name="k", head_first=head_first, expected_ndim=4)
    v_seq = _as_seq_first(v, name="v", head_first=head_first, expected_ndim=4)
    beta_seq = _as_seq_first(beta, name="beta", head_first=head_first, expected_ndim=3)
    g_seq = _as_seq_first(g, name="g", head_first=head_first, expected_ndim=3)

    _validate_inputs(q_seq, k_seq, v_seq, beta_seq, g_seq, initial_state, cu_seqlens)

    if scale is None:
        scale = k_seq.shape[-1] ** -0.5

    B, T, Hg, K = q_seq.shape
    H, V = v_seq.shape[2], v_seq.shape[3]
    if (
        initial_state is None
        and cu_seqlens is None
        and T <= 128
        and K <= 128
        and V <= 128
        and H % Hg == 0
    ):
        q_direct = _direct_contiguous(q_seq)
        k_direct = _direct_contiguous(k_seq)
        v_direct = _direct_contiguous(v_seq)
        g_direct = _direct_contiguous(g_seq)
        beta_direct = _direct_contiguous(beta_seq)
        if can_use_chunk_gated_delta_rule_direct(
            q=q_direct,
            k=k_direct,
            v=v_direct,
            g=g_direct,
            beta=beta_direct,
            initial_state=None,
            cu_seqlens=None,
        ):
            o, final_state = chunk_gated_delta_rule_direct_fwd(
                q=q_direct,
                k=k_direct,
                v=v_direct,
                g=g_direct,
                beta=beta_direct,
                scale=float(scale),
                initial_state=None,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            if head_first:
                o = o.transpose(1, 2)
            return o, final_state

    if use_qk_l2norm_in_kernel:
        q_seq = _l2_normalize_last_dim(q_seq)
        k_seq = _l2_normalize_last_dim(k_seq)

    use_two_kernel_fused_forward = can_use_two_kernel_fused_forward(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        beta=beta_seq,
        g=g_seq,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    if not use_two_kernel_fused_forward and _can_use_tle_chunk_gated_delta_rule(
        q_seq,
        k_seq,
        v_seq,
        beta_seq,
        g_seq,
        initial_state,
        output_final_state,
        cu_seqlens,
    ):
        q_tle = _direct_contiguous(q_seq)
        k_tle = _direct_contiguous(k_seq)
        v_tle = _direct_contiguous(v_seq)
        beta_tle = _direct_contiguous(beta_seq)
        g_tle = _direct_contiguous(g_seq)
        _set_tle_descriptor_allocator(q_tle.device)
        g_cumsum, a = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
            g=g_tle,
            k=k_tle,
            beta=beta_tle,
            cu_seqlens=None,
            chunk_size=TLE_CHUNK_GDR_BLOCK_S,
            output_dtype=k_tle.dtype,
            use_g_in_kkt=False,
        )
        o, final_state = _tle_chunk_gated_delta_rule_fwd(
            q=q_tle,
            k=k_tle,
            v=v_tle,
            g_cumsum=g_cumsum,
            beta=beta_tle,
            a=a,
            scale=float(scale),
        )
        if head_first:
            o = o.transpose(1, 2)
        return o, final_state if output_final_state else None

    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        g=g_seq,
        beta=beta_seq,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    if head_first:
        o = o.transpose(1, 2)
    return o, final_state
