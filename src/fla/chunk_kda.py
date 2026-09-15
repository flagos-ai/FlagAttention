# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the source
# repository: https://github.com/fla-org/flash-linear-attention

"""Optimized TLE forward path for KDA chunk inference (BT=16).

K2 uses a four-warp load partition, a four-warp MMA worker, and a one-warp
store worker.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices, prepare_chunk_offsets

__all__ = ["chunk_kda_fwd_infer"]

RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


def _allocate_triton_workspace(size: int, _alignment: int, _stream) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _input_guard(fn):
    """Make tensor inputs contiguous and launch under their CUDA device context."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args)
        kwargs = {
            name: value.contiguous() if isinstance(value, torch.Tensor) else value
            for name, value in kwargs.items()
        }
        tensor = next(
            (value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)),
            None,
        )
        if tensor is not None and tensor.is_cuda:
            with torch.cuda.device(tensor.device):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


def _tle_input_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    cu_seqlens: torch.LongTensor | None,
    safe_gate: bool,
    lower_bound: float | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
) -> str | None:
    inputs = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    invalid_dtypes = {
        name: tensor.dtype for name, tensor in inputs.items() if tensor.dtype != torch.bfloat16
    }
    if invalid_dtypes:
        details = ", ".join(f"{name}={dtype}" for name, dtype in invalid_dtypes.items())
        return f"TLE KDA requires bfloat16 inputs, got {details}"
    if any(not tensor.is_cuda for tensor in inputs.values()):
        return "TLE KDA requires CUDA inputs"
    if any(tensor.device != q.device for tensor in inputs.values()):
        return "TLE KDA requires all inputs on the same device"
    if any(not tensor.is_contiguous() for tensor in inputs.values()):
        return "TLE KDA requires contiguous inputs"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        return "TLE KDA expects q/k/v/g with rank 4 and beta with rank 3"

    B, T, H, D = q.shape
    if T == 0:
        return "TLE KDA requires a non-empty sequence"
    if D != 128:
        return f"TLE KDA requires K=128, got {D}"
    if v.shape[-1] != 128:
        return f"TLE KDA requires V=128, got {v.shape[-1]}"
    if v.shape[2] != H:
        return f"TLE KDA does not support GVA (HV={v.shape[2]} != H={H})"
    if k.shape != q.shape or v.shape != q.shape or g.shape != q.shape:
        return "TLE KDA requires q, k, v, and g shape [B, T, H, 128]"
    if beta.shape != (B, T, H):
        return f"TLE KDA requires beta shape {(B, T, H)}, got {tuple(beta.shape)}"
    if not use_qk_l2norm_in_kernel:
        return "TLE KDA requires use_qk_l2norm_in_kernel=True"
    if not use_gate_in_kernel:
        return "TLE KDA requires use_gate_in_kernel=True"
    if not use_beta_sigmoid_in_kernel:
        return "TLE KDA requires use_beta_sigmoid_in_kernel=True"
    if allow_neg_eigval:
        return "TLE KDA does not support allow_neg_eigval=True"
    if not safe_gate:
        return "TLE KDA requires safe_gate=True"
    if lower_bound is None or not -5 <= lower_bound < 0:
        return f"TLE KDA requires -5 <= lower_bound < 0, got {lower_bound}"
    if not state_v_first:
        return "TLE KDA requires state_v_first=True"
    if chunk_size != 16:
        return f"TLE KDA requires chunk_size=16, got {chunk_size}"

    if A_log is None or A_log.dtype != torch.float32 or A_log.shape != (H,):
        actual = None if A_log is None else (tuple(A_log.shape), A_log.dtype)
        return f"TLE KDA requires float32 A_log with shape {(H,)}, got {actual}"
    if dt_bias is None or dt_bias.dtype != torch.float32 or dt_bias.shape != (H, D):
        actual = None if dt_bias is None else (tuple(dt_bias.shape), dt_bias.dtype)
        return f"TLE KDA requires float32 dt_bias with shape {(H, D)}, got {actual}"
    if A_log.device != q.device or dt_bias.device != q.device:
        return "TLE KDA requires A_log and dt_bias on the input device"
    if not A_log.is_contiguous() or not dt_bias.is_contiguous():
        return "TLE KDA requires contiguous A_log and dt_bias"

    N = B
    if cu_seqlens is not None:
        if B != 1:
            return "TLE KDA requires B=1 when cu_seqlens is provided"
        if cu_seqlens.device != q.device or cu_seqlens.dtype != torch.long or cu_seqlens.ndim != 1:
            return "TLE KDA requires a 1D int64 cu_seqlens tensor on the input device"
        if cu_seqlens.numel() < 2:
            return "TLE KDA requires cu_seqlens to contain at least two elements"
        N = cu_seqlens.numel() - 1

    if initial_state is not None:
        expected = (N, H, D, D)
        if initial_state.dtype not in (torch.bfloat16, torch.float32) or initial_state.shape != expected:
            return f"TLE KDA requires bfloat16 or float32 initial_state with shape {expected}"
        if initial_state.device != q.device or not initial_state.is_contiguous():
            return "TLE KDA requires contiguous initial_state on the input device"
    return None


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages, maxnreg=maxnreg)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 4, 8]
        for maxnreg in [None, 32, 64, 72]
    ],
    key=["H", "HV", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_intra_kernel(
    q,
    k,
    g,
    beta,
    ws,
    Aqk,
    Akk,
    g_last,
    A_log,
    dt_bias,
    lower_bound,
    scale,
    g_scale,
    l2norm_eps,
    cu_seqlens,
    chunk_indices,
    T,
    NT_TOTAL,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_bh = tl.program_id(0), tl.program_id(1)
    i_t = i_tg
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_b = 0
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_b = i_bh // HV
        bos = i_b.to(tl.int64) * T
        i_tg = i_b * tl.cdiv(T, BT) + i_t

    if i_t * BT >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    g_last += (i_tg * HV + i_hv).to(tl.int64) * K
    if IS_VARLEN:
        a_chunk = i_hv * NT_TOTAL + i_tg
    else:
        a_chunk = (i_b * HV + i_hv) * NT_TOTAL + i_t
    Aqk += a_chunk.to(tl.int64) * BT * BT
    Akk += a_chunk.to(tl.int64) * BT * BT
    ws += (bos * HV + i_hv) * 3 * K
    beta += bos * HV + i_hv

    o_i = tl.arange(0, BT)
    token_start = i_t * BT
    o_c = token_start + o_i
    m_c = o_c < T

    q_buf = tle.gpu.alloc([BT, K], dtype=q.dtype.element_ty, scope=tle.gpu.smem)
    k_buf = tle.gpu.alloc([BT, K], dtype=k.dtype.element_ty, scope=tle.gpu.smem)
    gc_buf = tle.gpu.alloc([BT, K], dtype=tl.float32, scope=tle.gpu.smem)

    rows = tl.broadcast_to(tl.arange(0, BT)[:, None], (BT, K))
    cols = tl.broadcast_to(tl.arange(0, K)[None, :], (BT, K))
    q_sp = tle.gpu.local_ptr(q_buf, (rows, cols))
    k_sp = tle.gpu.local_ptr(k_buf, (rows, cols))
    gc_sp = tle.gpu.local_ptr(gc_buf, (rows, cols))

    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (token_start, 0), (BT, K), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (token_start, 0), (BT, K), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (HV * K, 1), (token_start, 0), (BT, K), (1, 0))
    b_q = tle.load(p_q, boundary_check=(0, 1), is_async=True)
    b_k = tle.load(p_k, boundary_check=(0, 1), is_async=True)
    tl.store(q_sp, b_q)
    tl.store(k_sp, b_k)

    b_qf = b_q.to(tl.float32)
    b_kf = b_k.to(tl.float32)

    b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
    b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

    b_g = tle.load(p_g, boundary_check=(0, 1), is_async=True).to(tl.float32)
    b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
    p_dt = tl.make_block_ptr(dt_bias + i_hv * K, (K,), (1,), (0,), (K,), (0,))
    b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
    b_g = b_g + b_bias[None, :]
    b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
    tl.store(gc_sp, b_g)
    one_row = tl.broadcast_to(tl.arange(0, 1)[:, None], (1, K))
    col_row = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
    b_acc = tl.zeros([1, K], dtype=tl.float32)
    for r in tl.static_range(BT):
        rp = tle.gpu.local_ptr(gc_buf, (tl.broadcast_to(one_row + r, (1, K)), col_row))
        b_acc = b_acc + tl.load(rp)
        tl.store(rp, b_acc)

    b_gq = tl.where(m_c[:, None], exp2(tl.load(gc_sp)), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(-tl.load(gc_sp)), 0.0)

    b_kgt = tl.trans(b_kf * b_gk).to(b_k.dtype)
    b_Aqk = tl.dot((b_qf * b_gq).to(b_q.dtype), b_kgt, out_dtype=tl.float32)
    b_Akk = tl.dot((b_kf * b_gq).to(b_k.dtype), b_kgt, out_dtype=tl.float32)

    b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
    b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (token_start,), (BT,), (0,))
    b_beta = tl.sigmoid(tl.load(p_beta, boundary_check=(0,)).to(tl.float32))

    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))

    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)

    p_Akk_out = tl.make_block_ptr(Akk, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty))

    b_k3 = tl.load(k_sp).to(tl.float32) * b_k_rstd[:, None]
    b_gk3 = tl.load(gc_sp)
    b_kb = b_k3 * b_beta[:, None] * exp2(b_gk3)
    p_w = tl.make_block_ptr(
        ws, (T, 3 * K), (HV * 3 * K, 1), (token_start, 0), (BT, K), (1, 0)
    )
    tl.store(p_w, b_kb.to(ws.dtype.element_ty), boundary_check=(0, 1))

    b_q3 = tl.load(q_sp).to(tl.float32) * b_q_rstd[:, None]
    b_qg_val = b_q3 * exp2(b_gk3)
    p_qg = tl.make_block_ptr(
        ws, (T, 3 * K), (HV * 3 * K, 1), (token_start, K), (BT, K), (1, 0)
    )
    tl.store(p_qg, b_qg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

    last_local = (tl.minimum(BT, T - token_start) - 1).to(tl.int32)
    gn_rows = tl.broadcast_to(last_local + tl.zeros([1, K], dtype=tl.int32), (1, K))
    gn_cols = tl.broadcast_to(tl.arange(0, K)[None, :], (1, K))
    b_gn = tl.load(tle.gpu.local_ptr(gc_buf, (gn_rows, gn_cols)))
    p_g_last = tl.make_block_ptr(g_last, (1, K), (K, 1), (0, 0), (1, K), (1, 0))
    tl.store(p_g_last, b_gn.to(g_last.dtype.element_ty), boundary_check=(0, 1))
    b_kg_val = b_k3 * tl.where(m_c[:, None], exp2(b_gn - b_gk3), 0)
    p_kg = tl.make_block_ptr(
        ws, (T, 3 * K), (HV * 3 * K, 1), (token_start, 2 * K), (BT, K), (1, 0)
    )
    tl.store(p_kg, b_kg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))


def _kda_fwd_intra(
    q,
    k,
    g,
    beta,
    scale,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=16,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
):
    B, T_len, H, K = q.shape
    HV = g.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HV)

    T_padded = NT * BT
    g_last = torch.empty(B * NT, HV, K, device=q.device, dtype=torch.float32)
    ws = torch.empty(B, T_padded, HV, 3 * K, device=q.device, dtype=q.dtype)
    Aqk = torch.empty(B, HV, NT, BT, BT, device=q.device, dtype=q.dtype)
    Akk = torch.empty(B, HV, NT, BT, BT, device=q.device, dtype=q.dtype)

    _kda_fwd_intra_kernel[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        ws=ws,
        Aqk=Aqk,
        Akk=Akk,
        g_last=g_last,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        scale=scale,
        g_scale=RCP_LN2,
        l2norm_eps=1e-6,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_len,
        NT_TOTAL=NT,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
    )
    return ws, Aqk, Akk, g_last


@triton.jit
def _kda_state_output_load_producer(
    writer,
    ws_desc,
    v_ptr,
    beta_ptr,
    gk_desc,
    Aqk_desc,
    Akk_desc,
    K: tl.constexpr,
    T,
    HV: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT,
    i_v,
    USE_HOST_DESCRIPTORS: tl.constexpr,
    state_row_start,
    chunk_start,
    i_h,
):
    for i_t in tl.range(NT):
        slot = writer.acquire(i_t)

        ws_row = i_t * BT
        ws_col = 0
        A_row = i_t * BT
        A_col = 0
        gk_row = i_t
        gk_col = 0
        if USE_HOST_DESCRIPTORS:
            ws_row += state_row_start
            ws_col = i_h * 3 * K
            A_row = state_row_start * HV + i_h * NT * BT + i_t * BT
            gk_row += chunk_start
            gk_col = i_h * K

        ws_row = ws_row.to(tl.int32)
        ws_col = ws_col.to(tl.int32)
        A_row = A_row.to(tl.int32)
        A_col = A_col.to(tl.int32)
        gk_row = gk_row.to(tl.int32)
        gk_col = gk_col.to(tl.int32)

        tle.gpu.copy(ws_desc, slot.w, [BT, K], [ws_row, ws_col])
        tle.gpu.copy(ws_desc, slot.qg, [BT, K], [ws_row, ws_col + K])
        tle.gpu.copy(ws_desc, slot.kg, [BT, K], [ws_row, ws_col + 2 * K])
        tle.gpu.copy(Aqk_desc, slot.Aqk, [BT, BT], [A_row, A_col])
        tle.gpu.copy(Akk_desc, slot.Akk, [BT, BT], [A_row, A_col])
        tle.gpu.copy(gk_desc, slot.gk, [1, K], [gk_row, gk_col])

        p_v = tl.make_block_ptr(
            v_ptr, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v_raw = tl.load(p_v, boundary_check=(0, 1))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_beta_f = tl.sigmoid(b_beta.to(tl.float32))
        b_v = (b_v_raw.to(tl.float32) * b_beta_f[:, None]).to(b_v_raw.dtype)
        tl.store(tle.gpu.local_ptr(slot.v), b_v)

        writer.commit(i_t)


@triton.jit
def _kda_state_output_mma_consumer(
    load_reader,
    store_writer,
    h0,
    ht,
    scale,
    i_v,
    i_nh,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    NT,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    state_dtype: tl.constexpr = tl.bfloat16
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
        )
        b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
    else:
        b_h = tl.zeros([K, BV], dtype=tl.float32)

    for i_t in tl.range(NT):
        wait = load_reader.wait(i_t)
        slot = wait.slot

        b_w = tl.load(tle.gpu.local_ptr(slot.w))
        b_v_raw = tl.load(tle.gpu.local_ptr(slot.v))
        b_qg = tl.load(tle.gpu.local_ptr(slot.qg))
        b_kg = tl.load(tle.gpu.local_ptr(slot.kg))
        b_Aqk = tl.load(tle.gpu.local_ptr(slot.Aqk))
        b_Akk = tl.load(tle.gpu.local_ptr(slot.Akk))
        b_gk = tl.load(tle.gpu.local_ptr(slot.gk)).reshape([K])

        b_h_bf = b_h.to(state_dtype)

        b_kh = tl.dot(b_w, b_h_bf).to(tl.float32)
        b_diff = b_v_raw.to(tl.float32) - b_kh
        b_v = tl.dot(b_Akk, b_diff.to(state_dtype)).to(tl.float32)

        b_qh = tl.dot(b_qg, b_h_bf).to(tl.float32)
        b_o = scale * b_qh
        b_v_cast = b_v.to(state_dtype)
        b_o += tl.dot(b_Aqk, b_v_cast).to(tl.float32)

        out_slot = store_writer.acquire(i_t)
        tl.store(tle.gpu.local_ptr(out_slot.output), b_o)
        store_writer.commit(i_t)

        load_reader.release(i_t)

        b_h = b_h * exp2(b_gk)[:, None] + tl.dot(tl.trans(b_kg), b_v_cast).to(tl.float32)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
        )
        tl.store(p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _kda_state_output_store_consumer(
    store_reader,
    store_target,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT,
    i_v,
    USE_HOST_DESCRIPTORS: tl.constexpr,
    output_row_start,
    output_col_start,
):
    for i_t in tl.range(NT):
        store_wait = store_reader.wait(i_t)
        slot = store_wait.slot
        output_row = i_t * BT
        output_col = i_v * BV
        if USE_HOST_DESCRIPTORS:
            output_row += output_row_start
            output_col += output_col_start
        output_row = output_row.to(tl.int32)
        output_col = output_col.to(tl.int32)
        tle.gpu.copy(slot.output, store_target, [BT, BV], [output_row, output_col])
        store_reader.release(i_t)


PIPE_STAGES = tl.constexpr(4)


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
        "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_state_output_kernel(
    v,
    beta,
    gk,
    Aqk,
    Akk,
    o,
    ws,
    h0,
    ht,
    ws_host_desc,
    gk_host_desc,
    Aqk_host_desc,
    Akk_host_desc,
    output_host_desc,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    NT_TOTAL,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_HOST_DESCRIPTORS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
        chunk_start = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos = i_n.to(tl.int64) * T
        NT = tl.cdiv(T, BT)
        chunk_start = i_n * NT

    v += (bos * HV + i_h) * V
    beta += bos * HV + i_h
    gk += (chunk_start * HV + i_h).to(tl.int64) * K
    o += (bos * HV + i_h) * V
    ws_base = ws + (bos * HV + i_h) * 3 * K

    if IS_VARLEN:
        state_row_start = bos.to(tl.int64)
    else:
        state_row_start = i_n.to(tl.int64) * NT * BT
    if USE_HOST_DESCRIPTORS:
        ws_desc = ws_host_desc
        gk_desc = gk_host_desc
        Aqk_desc = Aqk_host_desc
        Akk_desc = Akk_host_desc
    else:
        if IS_VARLEN:
            a_chunk = (i_h * NT_TOTAL + chunk_start) * BT * BT
        else:
            a_chunk = (i_n * HV + i_h) * NT_TOTAL * BT * BT
        Aqk += a_chunk.to(tl.int64)
        Akk += a_chunk.to(tl.int64)
        ws_desc = tl.make_tensor_descriptor(
            ws_base, shape=[T, 3 * K], strides=[HV * 3 * K, 1], block_shape=[BT, K]
        )
        gk_desc = tl.make_tensor_descriptor(
            gk, shape=[NT, K], strides=[HV * K, 1], block_shape=[1, K]
        )
        Aqk_desc = tl.make_tensor_descriptor(
            Aqk, shape=[NT * BT, BT], strides=[BT, 1], block_shape=[BT, BT]
        )
        Akk_desc = tl.make_tensor_descriptor(
            Akk, shape=[NT * BT, BT], strides=[BT, 1], block_shape=[BT, BT]
        )

    w_smem = tle.gpu.alloc([PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem)
    v_smem = tle.gpu.alloc([PIPE_STAGES, BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem)
    qg_smem = tle.gpu.alloc([PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem)
    kg_smem = tle.gpu.alloc([PIPE_STAGES, BT, K], dtype=tl.bfloat16, scope=tle.gpu.smem)
    Aqk_smem = tle.gpu.alloc([PIPE_STAGES, BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem)
    Akk_smem = tle.gpu.alloc([PIPE_STAGES, BT, BT], dtype=tl.bfloat16, scope=tle.gpu.smem)
    gk_smem = tle.gpu.alloc([PIPE_STAGES, 1, K], dtype=tl.float32, scope=tle.gpu.smem)
    out_smem = tle.gpu.alloc([PIPE_STAGES, BT, BV], dtype=tl.bfloat16, scope=tle.gpu.smem)
    if USE_HOST_DESCRIPTORS:
        output_store_target = output_host_desc
    else:
        output_store_target = tl.make_tensor_descriptor(
            o, shape=[T, V], strides=[HV * V, 1], block_shape=[BT, BV]
        )

    load_pipe = tle.pipe(
        capacity=PIPE_STAGES,
        scope="cta",
        name="kda_load",
        w=w_smem,
        v=v_smem,
        qg=qg_smem,
        kg=kg_smem,
        Aqk=Aqk_smem,
        Akk=Akk_smem,
        gk=gk_smem,
    )
    store_pipe = tle.pipe(
        capacity=PIPE_STAGES,
        scope="cta",
        name="kda_store",
        output=out_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _kda_state_output_load_producer,
                (
                    load_pipe.writer(),
                    ws_desc,
                    v,
                    beta,
                    gk_desc,
                    Aqk_desc,
                    Akk_desc,
                    K,
                    T,
                    HV,
                    V,
                    BT,
                    BV,
                    NT,
                    i_v,
                    USE_HOST_DESCRIPTORS,
                    state_row_start,
                    chunk_start,
                    i_h,
                ),
            ),
            (
                _kda_state_output_mma_consumer,
                (
                    load_pipe.reader(),
                    store_pipe.writer(),
                    h0,
                    ht,
                    scale,
                    i_v,
                    i_nh,
                    K,
                    V,
                    BV,
                    NT,
                    USE_INITIAL_STATE,
                    STORE_FINAL_STATE,
                ),
            ),
            (
                _kda_state_output_store_consumer,
                (
                    store_pipe.reader(),
                    output_store_target,
                    BT,
                    BV,
                    NT,
                    i_v,
                    USE_HOST_DESCRIPTORS,
                    bos.to(tl.int64),
                    i_h.to(tl.int64) * V,
                ),
            ),
        ],
        [4, 1],
        [240, 32],
    )


def _kda_fwd_state_output(
    v: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    gk: torch.Tensor,
    Aqk: torch.Tensor,
    ws: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, _, HV, packed_K = ws.shape
    K = packed_K // 3
    T_actual = v.shape[1]
    V = v.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        N = B
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    final_state = None
    if output_final_state:
        final_state = ws.new_empty(N, HV, V, K, dtype=torch.float32)

    o = torch.empty(B, T_actual, HV, V, device=ws.device, dtype=v.dtype)

    h0_arg = initial_state if initial_state is not None else ws.new_empty(1, dtype=torch.float32)
    ht_arg = final_state if final_state is not None else ws.new_empty(1, dtype=torch.float32)

    use_host_descriptors = cu_seqlens is None and T_actual % BT == 0
    if use_host_descriptors:
        NT_total = ws.shape[1] // BT
        descriptor_rows = B * HV * NT_total * BT
        ws_desc_arg = TensorDescriptor(
            ws,
            shape=[descriptor_rows, HV * 3 * K],
            strides=[HV * 3 * K, 1],
            block_shape=[BT, K],
        )
        gk_desc_arg = TensorDescriptor(
            gk,
            shape=[gk.shape[0], HV * K],
            strides=[HV * K, 1],
            block_shape=[1, K],
        )
        Aqk_desc_arg = TensorDescriptor(
            Aqk,
            shape=[descriptor_rows, BT],
            strides=[BT, 1],
            block_shape=[BT, BT],
        )
        Akk_desc_arg = TensorDescriptor(
            Akk,
            shape=[descriptor_rows, BT],
            strides=[BT, 1],
            block_shape=[BT, BT],
        )
        output_desc_arg = TensorDescriptor(
            o,
            shape=[B * T_actual, HV * V],
            strides=[HV * V, 1],
            block_shape=[BT, 128],
        )
    else:
        ws_desc_arg = ws
        gk_desc_arg = gk
        Aqk_desc_arg = Aqk
        Akk_desc_arg = Akk
        output_desc_arg = o

    _kda_fwd_state_output_kernel[(1, N * HV)](
        v=v,
        beta=beta,
        gk=gk,
        Aqk=Aqk,
        Akk=Akk,
        o=o,
        ws=ws,
        h0=h0_arg,
        ht=ht_arg,
        ws_host_desc=ws_desc_arg,
        gk_host_desc=gk_desc_arg,
        Aqk_host_desc=Aqk_desc_arg,
        Akk_host_desc=Akk_desc_arg,
        output_host_desc=output_desc_arg,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T_actual,
        NT_TOTAL=ws.shape[1] // BT,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BV=128,
        USE_HOST_DESCRIPTORS=use_host_descriptors,
        num_warps=4,
    )

    return o, final_state


@_input_guard
def chunk_kda_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.is_grad_enabled():
        raise RuntimeError("TLE KDA only supports inference mode")
    reason = _tle_input_error(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        chunk_size=chunk_size,
    )
    if reason is not None:
        raise ValueError(reason)

    triton.set_allocator(_allocate_triton_workspace)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    ws, Aqk, Akk, g_last = _kda_fwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    return _kda_fwd_state_output(
        v=v,
        beta=beta,
        Akk=Akk,
        gk=g_last,
        Aqk=Aqk,
        ws=ws,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
