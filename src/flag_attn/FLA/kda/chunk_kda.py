# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the source
# repository: https://github.com/fla-org/flash-linear-attention

"""Pure-Triton forward path for KDA chunk inference.

This is the generic implementation: it uses only ``triton.language`` and runs
on any accelerator with a working Triton, including machines whose Triton does
not ship ``triton.experimental.tle``. The warp-specialized Hopper kernels live
under ``runtime/backend/_nvidia/hopper`` and are selected there.

The chunked formulation follows the KDA recurrence. Each chunk computes its
intra-chunk gate cumulative sums and the two triangular score matrices, then a
second pass carries the recurrent state across chunks and writes the output.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices, prepare_chunk_offsets

__all__ = ["chunk_kda_fwd_infer", "chunk_kda_fwd_infer_triton"]

RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


def input_guard(fn):
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


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_intra_core_triton_kernel(
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
    """TLE-free K1 for the S60 K=V=128, BT=32 production shape."""
    i_tg, i_bh = tl.program_id(0), tl.program_id(1)
    i_t = i_tg
    i_hv = i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_b = 0
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        ws_bos = bos
    else:
        i_b = i_bh // HV
        bos = i_b.to(tl.int32) * T
        i_tg = i_b * tl.cdiv(T, BT) + i_t
        ws_bos = i_b * NT_TOTAL * BT

    if i_t * BT >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    g_last += (i_tg * HV + i_hv).to(tl.int32) * K
    if IS_VARLEN:
        a_chunk = i_hv * NT_TOTAL + i_tg
    else:
        a_chunk = (i_b * HV + i_hv) * NT_TOTAL + i_t
    Aqk += a_chunk.to(tl.int32) * BT * BT
    Akk += a_chunk.to(tl.int32) * BT * BT
    ws += (ws_bos * HV + i_hv) * 3 * K
    beta += bos * HV + i_hv

    o_i = tl.arange(0, BT)
    token_start = i_t * BT
    o_c = token_start + o_i
    m_c = o_c < T

    p_q = tl.make_block_ptr(
        q, (T, K), (H * K, 1), (token_start, 0), (BT, K), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k, (T, K), (H * K, 1), (token_start, 0), (BT, K), (1, 0)
    )
    p_g = tl.make_block_ptr(
        g, (T, K), (HV * K, 1), (token_start, 0), (BT, K), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.where(m_c[:, None], b_q, 0.0)
    b_k = tl.where(m_c[:, None], b_k, 0.0)
    b_qf = b_q.to(tl.float32)
    b_kf = b_k.to(tl.float32)
    b_q_rstd = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf, 1) + l2norm_eps)
    b_k_rstd = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf, 1) + l2norm_eps)

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)
    p_dt = tl.make_block_ptr(
        dt_bias + i_hv * K, (K,), (1,), (0,), (K,), (0,)
    )
    b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
    b_g = b_g + b_bias[None, :]
    b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
    b_gc = tl.cumsum(tl.where(m_c[:, None], b_g, 0.0), axis=0)

    # A single exponent shift keeps every operand representable. Only the
    # three blocks retained by the lower-triangular mask are formed, avoiding
    # the unused overflow-prone quadrant of the BT32 product.
    split_row: tl.constexpr = BT // 2
    b_shift = tl.sum(
        tl.where(o_i[:, None] == split_row - 1, b_gc, 0.0), axis=0
    )[None, :]
    b_gq = tl.where(m_c[:, None], exp2(b_gc - b_shift), 0.0)
    b_gk = tl.where(m_c[:, None], exp2(b_shift - b_gc), 0.0)
    b_k_left = (b_kf * b_gk).to(b_k.dtype)
    b_q_scaled = (b_qf * b_gq).to(b_q.dtype)
    b_k_scaled = (b_kf * b_gq).to(b_k.dtype)
    m_lo = o_i < split_row
    b_k_left_lo = tl.where(m_lo[:, None], b_k_left, 0.0)
    b_k_left_hi = tl.where(~m_lo[:, None], b_k_left, 0.0)
    b_q_right_lo = tl.trans(tl.where(m_lo[:, None], b_q_scaled, 0.0))
    b_q_right_hi = tl.trans(tl.where(~m_lo[:, None], b_q_scaled, 0.0))
    b_k_right_lo = tl.trans(tl.where(m_lo[:, None], b_k_scaled, 0.0))
    b_k_right_hi = tl.trans(tl.where(~m_lo[:, None], b_k_scaled, 0.0))

    b_Aqk_t = tl.dot(b_k_left_lo, b_q_right_lo, out_dtype=tl.float32)
    b_Aqk_t += tl.dot(b_k_left_lo, b_q_right_hi, out_dtype=tl.float32)
    b_Aqk_t += tl.dot(b_k_left_hi, b_q_right_hi, out_dtype=tl.float32)
    b_Akk_t = tl.dot(b_k_left_lo, b_k_right_lo, out_dtype=tl.float32)
    b_Akk_t += tl.dot(b_k_left_lo, b_k_right_hi, out_dtype=tl.float32)
    b_Akk_t += tl.dot(b_k_left_hi, b_k_right_hi, out_dtype=tl.float32)
    b_Aqk = tl.trans(b_Aqk_t)
    b_Akk = tl.trans(b_Akk_t)
    b_Aqk = b_Aqk * b_q_rstd[:, None] * b_k_rstd[None, :]
    b_Akk = b_Akk * b_k_rstd[:, None] * b_k_rstd[None, :]

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (token_start,), (BT,), (0,))
    b_beta = tl.sigmoid(tl.load(p_beta, boundary_check=(0,)).to(tl.float32))
    b_beta = tl.where(m_c, b_beta, 0.0)

    m_valid = m_c[:, None] & m_c[None, :]
    m_Aqk = (o_i[:, None] >= o_i[None, :]) & m_valid
    m_Akk = (o_i[:, None] > o_i[None, :]) & m_valid
    m_I = o_i[:, None] == o_i[None, :]
    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    p_Aqk = tl.make_block_ptr(Aqk, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty))

    b_q3 = b_qf * b_q_rstd[:, None]
    b_qg_val = b_q3 * exp2(b_gc)
    p_qg = tl.make_block_ptr(
        ws,
        (T, 3 * K),
        (HV * 3 * K, 1),
        (token_start, K),
        (BT, K),
        (1, 0),
    )
    tl.store(p_qg, b_qg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

    b_k3 = b_kf * b_k_rstd[:, None]
    b_kb = b_k3 * b_beta[:, None] * exp2(b_gc)
    p_w = tl.make_block_ptr(
        ws,
        (T, 3 * K),
        (HV * 3 * K, 1),
        (token_start, 0),
        (BT, K),
        (1, 0),
    )
    tl.store(p_w, b_kb.to(ws.dtype.element_ty), boundary_check=(0, 1))

    last_local = (tl.minimum(BT, T - token_start) - 1).to(tl.int32)
    b_gn = tl.sum(
        tl.where(o_i[:, None] == last_local, b_gc, 0.0), axis=0
    )[None, :]
    p_g_last = tl.make_block_ptr(
        g_last, (1, K), (K, 1), (0, 0), (1, K), (1, 0)
    )
    tl.store(p_g_last, b_gn.to(g_last.dtype.element_ty), boundary_check=(0, 1))
    b_kg_val = b_k3 * tl.where(m_c[:, None], exp2(b_gn - b_gc), 0.0)
    p_kg = tl.make_block_ptr(
        ws,
        (T, 3 * K),
        (HV * 3 * K, 1),
        (token_start, 2 * K),
        (BT, K),
        (1, 0),
    )
    tl.store(p_kg, b_kg_val.to(ws.dtype.element_ty), boundary_check=(0, 1))

    # (I-L)(I+L^2)(I+L^4)(I+L^8)(I+L^16) = I for a strictly
    # lower-triangular 32x32 L.
    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)
    b_L16 = tl.dot(b_L8, b_L8, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L16, out_dtype=tl.float16)
    p_Akk = tl.make_block_ptr(Akk, (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty))


def _kda_fwd_intra_core_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 32,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    _kda_fwd_intra_core_triton_kernel[grid](
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
        num_warps=1,
        num_stages=2,
    )
    return ws, Aqk, Akk, g_last


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"].numel() > 1,
        "STORE_FINAL_STATE": lambda args: args["ht"].numel() > 1,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_state_output_core_triton_kernel(
    v,
    beta,
    gk,
    Aqk,
    Akk,
    o,
    ws,
    h0,
    ht,
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
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """TLE-free K2; one S60 program owns a complete 128x128 head state."""
    i_pid = tl.program_id(0)
    n_v_tiles = tl.cdiv(V, BV)
    i_v = i_pid % n_v_tiles
    i_nh = i_pid // n_v_tiles
    i_n = i_nh // HV
    i_h = i_nh % HV

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        chunk_start = tl.load(chunk_offsets + i_n).to(tl.int32)
        a_chunk_start = i_h * NT_TOTAL + chunk_start
        ws_bos = bos
    else:
        bos = i_n * T
        NT = tl.cdiv(T, BT)
        chunk_start = i_n * NT
        a_chunk_start = (i_n * HV + i_h) * NT_TOTAL
        ws_bos = i_n * NT * BT

    v += (bos * HV + i_h) * V
    beta += bos * HV + i_h
    o += (bos * HV + i_h) * V
    ws += (ws_bos * HV + i_h) * 3 * K
    gk += (chunk_start * HV + i_h) * K
    Aqk += a_chunk_start * BT * BT
    Akk += a_chunk_start * BT * BT

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_nh * V * K,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, K),
            (1, 0),
        )
        b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
    else:
        b_h = tl.zeros([K, BV], dtype=tl.float32)

    for i_t in tl.range(NT):
        token_start = i_t * BT
        o_t = tl.arange(0, BT)
        m_t = token_start + o_t < T

        p_w = tl.make_block_ptr(
            ws,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (token_start, 0),
            (BT, K),
            (1, 0),
        )
        p_v = tl.make_block_ptr(
            v,
            (T, V),
            (HV * V, 1),
            (token_start, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_beta = tl.make_block_ptr(
            beta, (T,), (HV,), (token_start,), (BT,), (0,)
        )
        p_Akk = tl.make_block_ptr(
            Akk + i_t * BT * BT,
            (BT, BT),
            (BT, 1),
            (0, 0),
            (BT, BT),
            (1, 0),
        )

        b_v_raw = tl.load(p_v, boundary_check=(0, 1))
        b_v_raw = tl.where(m_t[:, None], b_v_raw, 0.0)
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
        b_beta = tl.where(m_t, b_beta, 0.0)
        b_vb = (b_v_raw.to(tl.float32) * tl.sigmoid(b_beta)[:, None]).to(
            tl.bfloat16
        ).to(tl.float32)
        b_Akk = tl.load(p_Akk)

        b_h_bf = b_h.to(tl.bfloat16)
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_w = tl.where(m_t[:, None], b_w, 0.0)
        b_kh = tl.dot(b_w, b_h_bf, out_dtype=tl.float32)
        b_diff = b_vb - b_kh
        b_v_new = tl.dot(
            b_Akk, b_diff.to(tl.bfloat16), out_dtype=tl.float32
        )
        b_v_new = tl.where(m_t[:, None], b_v_new, 0.0)
        b_v_new_bf = b_v_new.to(tl.bfloat16)

        p_qg = tl.make_block_ptr(
            ws,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (token_start, K),
            (BT, K),
            (1, 0),
        )
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        b_qg = tl.where(m_t[:, None], b_qg, 0.0)
        b_qh = tl.dot(b_qg, b_h_bf, out_dtype=tl.float32)
        b_o = scale * b_qh
        p_Aqk = tl.make_block_ptr(
            Aqk + i_t * BT * BT,
            (BT, BT),
            (BT, 1),
            (0, 0),
            (BT, BT),
            (1, 0),
        )
        b_Aqk = tl.load(p_Aqk)
        b_o += tl.dot(b_Aqk, b_v_new_bf, out_dtype=tl.float32)
        p_o = tl.make_block_ptr(
            o,
            (T, V),
            (HV * V, 1),
            (token_start, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        o_k = tl.arange(0, K)
        b_gk = tl.load(
            gk + i_t * HV * K + o_k, mask=o_k < K, other=0.0
        ).to(tl.float32)
        b_h_next = b_h * exp2(b_gk)[:, None]
        p_kg = tl.make_block_ptr(
            ws,
            (T, 3 * K),
            (HV * 3 * K, 1),
            (token_start, 2 * K),
            (BT, K),
            (1, 0),
        )
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        b_kg = tl.where(m_t[:, None], b_kg, 0.0)
        b_h_next += tl.dot(
            tl.trans(b_kg), b_v_new_bf, out_dtype=tl.float32
        )
        b_h = b_h_next

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_nh * V * K,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, K),
            (1, 0),
        )
        tl.store(
            p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1)
        )


def _kda_fwd_state_output_core_triton(
    v: torch.Tensor,
    beta: torch.Tensor,
    Akk: torch.Tensor,
    gk: torch.Tensor,
    Aqk: torch.Tensor,
    ws: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 32,
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
    h0_arg = (
        initial_state
        if initial_state is not None
        else ws.new_empty(1, dtype=torch.float32)
    )
    ht_arg = (
        final_state
        if final_state is not None
        else ws.new_empty(1, dtype=torch.float32)
    )

    BV = 128
    _kda_fwd_state_output_core_triton_kernel[
        (triton.cdiv(V, BV) * N * HV,)
    ](
        v=v,
        beta=beta,
        gk=gk,
        Aqk=Aqk,
        Akk=Akk,
        o=o,
        ws=ws,
        h0=h0_arg,
        ht=ht_arg,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T_actual,
        NT_TOTAL=ws.shape[1] // BT,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return o, final_state


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_QG": lambda args: args["qg"] is not None,
        "STORE_KG": lambda args: args["kg"] is not None,
        "USE_GATE_IN_KERNEL": lambda args: args["A_log"] is not None,
        "USE_QK_L2NORM": lambda args: args["use_qk_l2norm"],
        "APPLY_BETA_SIGMOID": lambda args: args["apply_beta_sigmoid"],
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "HAS_DT_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BK": 32, "BV": 32}, num_warps=1, num_stages=1)],
    key=["H", "HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_intra_triton_kernel(
    q,
    k,
    v,
    g,
    beta,
    w,
    u,
    qg,
    kg,
    Aqk,
    Akk,
    g_out,
    A_log,
    dt_bias,
    lower_bound,
    scale,
    g_scale,
    l2norm_eps,
    use_qk_l2norm,
    apply_beta_sigmoid,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    USE_GATE_IN_KERNEL: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    APPLY_BETA_SIGMOID: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    g_out += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    w += (bos * HV + i_hv) * K
    u += (bos * HV + i_hv) * V
    beta += bos * HV + i_hv
    if STORE_QG:
        qg += (bos * HV + i_hv) * K
    if STORE_KG:
        kg += (bos * HV + i_hv) * K

    o_i = tl.arange(0, BT)
    o_c = i_t * BT + o_i
    m_c = o_c < T

    # Phase 0: L2 norm on q/k (optional) + beta sigmoid (optional)
    if USE_QK_L2NORM:
        b_q_ss = tl.zeros([BT], dtype=tl.float32)
        b_k_ss = tl.zeros([BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_k = tl.make_block_ptr(
                k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
            b_q_ss += tl.sum(b_q * b_q, 1)
            b_k_ss += tl.sum(b_k * b_k, 1)

        b_q_rstd = 1.0 / tl.sqrt(b_q_ss + l2norm_eps)
        b_k_rstd = 1.0 / tl.sqrt(b_k_ss + l2norm_eps)

    p_beta = tl.make_block_ptr(beta, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    if APPLY_BETA_SIGMOID:
        b_beta = tl.sigmoid(b_beta)

    # Phase 1: cumsum(g) + intra-chunk Aqk/Akk
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Akk = tl.zeros([BT, BT], dtype=tl.float32)

    if USE_GATE_IN_KERNEL:
        b_A = exp2(tl.load(A_log + i_hv).to(tl.float32) * g_scale)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_g = tl.make_block_ptr(
            g, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )

        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * b_q_rstd[:, None]
            b_k = b_k * b_k_rstd[:, None]
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        if USE_GATE_IN_KERNEL:
            if HAS_DT_BIAS:
                p_dt = tl.make_block_ptr(
                    dt_bias + i_hv * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
                )
                b_bias = tl.load(p_dt, boundary_check=(0,)).to(tl.float32)
                b_g = b_g + b_bias[None, :]
            if USE_LOWER_BOUND:
                b_g = (lower_bound * g_scale) * tl.sigmoid(b_A * b_g)
            else:
                b_g = -b_A * _softplus(b_g) * g_scale
        else:
            b_g = b_g * g_scale
        b_g = tl.cumsum(b_g, axis=0)

        p_g_out = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        tl.store(p_g_out, b_g.to(g_out.dtype.element_ty), boundary_check=(0, 1))

        b_gq = tl.where(m_c[:, None], exp2(b_g), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_g), 0.0)

        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_k * b_gq, b_kgt)

    # Causal mask
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk * scale, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk * b_beta[:, None], 0.0)

    p_Aqk = tl.make_block_ptr(
        Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 2: Solve (I + L)^{-1} via parallel prefix
    b_L = b_Akk.to(tl.float16)
    b_Ai = m_I.to(tl.float16) - b_L
    b_L2 = tl.dot(b_L, b_L, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L2, out_dtype=tl.float16)
    b_L4 = tl.dot(b_L2, b_L2, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L4, out_dtype=tl.float16)
    b_L8 = tl.dot(b_L4, b_L4, out_dtype=tl.float16)
    b_Ai = b_Ai + tl.dot(b_Ai, b_L8, out_dtype=tl.float16)
    p_Akk_out = tl.make_block_ptr(
        Akk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_Akk_out, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))

    # Phase 3: w, u, qg, kg
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_Ai.to(b_vb.dtype), b_vb)
        tl.store(p_u, b_u.to(u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        p_gk = tl.make_block_ptr(
            g_out, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32) * b_k_rstd[:, None]
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kb = b_k * b_beta[:, None] * exp2(b_gk)

        if STORE_QG:
            p_q = tl.make_block_ptr(
                q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            p_qg_out = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32) * b_q_rstd[:, None]
            b_qg_val = b_q * exp2(b_gk)
            tl.store(p_qg_out, b_qg_val.to(qg.dtype.element_ty), boundary_check=(0, 1))

        if STORE_KG:
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            last_idx = tl.minimum(i_t * BT + BT, T) - 1
            b_gn = tl.load(g_out + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )
            b_kg_val = b_k * tl.where(m_c[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg_out = tl.make_block_ptr(
                kg, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
            )
            tl.store(p_kg_out, b_kg_val.to(kg.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_w = tl.dot(b_Ai.to(b_kb.to(b_k.dtype).dtype), b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(w.dtype.element_ty), boundary_check=(0, 1))


def _kda_fwd_intra_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 16,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    apply_beta_sigmoid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused intra-chunk computation. Returns (w, u, qg, kg, Aqk, Akk, g_cumsum)."""
    B, T_len, H, K = q.shape
    HV = g.shape[2]
    V = v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T_len, BT) if cu_seqlens is None else len(chunk_indices)
    grid = (NT, B * HV)

    g_out = torch.empty(B, T_len, HV, K, device=q.device, dtype=torch.float32)
    w = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    u = torch.empty(B, T_len, HV, V, device=q.device, dtype=q.dtype)
    qg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    kg = torch.empty(B, T_len, HV, K, device=q.device, dtype=q.dtype)
    Aqk = torch.empty(B, T_len, HV, BT, device=q.device, dtype=q.dtype)
    Akk = torch.zeros(B, T_len, HV, BT, device=q.device, dtype=q.dtype)

    _kda_fwd_intra_triton_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        w=w,
        u=u,
        qg=qg,
        kg=kg,
        Aqk=Aqk,
        Akk=Akk,
        g_out=g_out,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        scale=scale,
        g_scale=RCP_LN2,
        l2norm_eps=1e-6,
        use_qk_l2norm=use_qk_l2norm,
        apply_beta_sigmoid=apply_beta_sigmoid,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T_len,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
    )
    return w, u, qg, kg, Aqk, Akk, g_out


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BV": 32}, num_warps=2)],
    key=["HV", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _kda_fwd_h_o_triton_kernel(
    kg,
    w,
    u,
    gk,
    qg,
    Aqk,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        i_n = i_nh // HV
        i_h = i_nh % HV
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)

    kg += (bos * HV + i_h).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    u += (bos * HV + i_h).to(tl.int64) * V
    gk += (bos * HV + i_h).to(tl.int64) * K
    qg += (bos * HV + i_h).to(tl.int64) * K
    Aqk += (bos * HV + i_h).to(tl.int64) * BT
    o += (bos * HV + i_h).to(tl.int64) * V

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_h0_1 = tl.make_block_ptr(
                h0 + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0 + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # v_new = u - w @ h
        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_u = tl.make_block_ptr(
            u, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

        # output = scale * qg @ h + Aqk @ v_new
        p_qg = tl.make_block_ptr(
            qg, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_o = tl.dot(b_qg, tl.trans(b_h1).to(b_qg.dtype))
        else:
            b_o = tl.dot(b_qg, b_h1.to(b_qg.dtype))
        if K > 64:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h2).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h2.to(b_qg.dtype))
        if K > 128:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h3).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h3.to(b_qg.dtype))
        if K > 192:
            p_qg = tl.make_block_ptr(
                qg, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h4).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h4.to(b_qg.dtype))
        b_o *= scale

        p_Aqk = tl.make_block_ptr(
            Aqk, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
        b_o += tl.dot(b_Aqk.to(b_v.dtype), b_v)

        p_o = tl.make_block_ptr(
            o, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # decay: h *= exp2(gk_last)
        last_idx = tl.minimum(i_t * BT + BT, T) - 1
        o_k1 = tl.arange(0, 64)
        b_gk_last1 = tl.load(
            gk + last_idx * HV * K + o_k1, mask=(o_k1 < K), other=0.0
        ).to(tl.float32)
        if STATE_V_FIRST:
            b_h1 *= exp2(b_gk_last1)[None, :]
        else:
            b_h1 *= exp2(b_gk_last1)[:, None]
        if K > 64:
            o_k2 = 64 + o_k1
            b_gk_last2 = tl.load(
                gk + last_idx * HV * K + o_k2, mask=(o_k2 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h2 *= exp2(b_gk_last2)[None, :]
            else:
                b_h2 *= exp2(b_gk_last2)[:, None]
        if K > 128:
            o_k3 = 128 + o_k1
            b_gk_last3 = tl.load(
                gk + last_idx * HV * K + o_k3, mask=(o_k3 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h3 *= exp2(b_gk_last3)[None, :]
            else:
                b_h3 *= exp2(b_gk_last3)[:, None]
        if K > 192:
            o_k4 = 192 + o_k1
            b_gk_last4 = tl.load(
                gk + last_idx * HV * K + o_k4, mask=(o_k4 < K), other=0.0
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h4 *= exp2(b_gk_last4)[None, :]
            else:
                b_h4 *= exp2(b_gk_last4)[:, None]

        # state update: h += kg^T @ v_new
        b_v = b_v.to(kg.dtype.element_ty)
        p_kg = tl.make_block_ptr(
            kg, (K, T), (1, HV * K), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_kg, b_v))
        else:
            b_h1 += tl.dot(b_kg, b_v)
        if K > 64:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h2 += tl.dot(b_kg, b_v)
        if K > 128:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h3 += tl.dot(b_kg, b_v)
        if K > 192:
            p_kg = tl.make_block_ptr(
                kg, (K, T), (1, HV * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_kg, b_v))
            else:
                b_h4 += tl.dot(b_kg, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht + i_nh * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _kda_fwd_h_o_triton(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    qg: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused state propagation + output."""
    B, T, HV, K = kg.shape
    V = u.shape[-1]
    BT = chunk_size

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    final_state = None
    if output_final_state:
        if state_v_first:
            final_state = kg.new_zeros(N, HV, V, K, dtype=torch.float32)
        else:
            final_state = kg.new_zeros(N, HV, K, V, dtype=torch.float32)

    o = torch.zeros(B, T, HV, V, device=kg.device, dtype=u.dtype)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(V, meta["BV"]), N * HV)

    _kda_fwd_h_o_triton_kernel[grid](
        kg=kg,
        w=w,
        u=u,
        gk=gk,
        qg=qg,
        Aqk=Aqk,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o, final_state


def _validate(q, k, v, chunk_size):
    """Return why these inputs cannot run, or None when they can.

    The generic path keeps the same contract as the Hopper kernels, so a call
    that runs on one runs on the other with the same result.
    """
    if q.shape[-1] != 128:
        return f"chunk KDA requires K=128, got {q.shape[-1]}"
    if v.shape[-1] != 128:
        return f"chunk KDA requires V=128, got {v.shape[-1]}"
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return "chunk KDA requires bfloat16 q/k/v"
    if q.shape[2] != v.shape[2]:
        return f"chunk KDA does not support GVA (HV={v.shape[2]} != H={q.shape[2]})"
    if chunk_size != 16:
        return f"generic chunk KDA supports chunk_size=16, got {chunk_size}"
    if q.shape[1] == 0:
        return "chunk KDA requires a non-empty sequence"
    return None


@input_guard
def chunk_kda_fwd_infer_triton(
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
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run KDA chunk inference with plain Triton kernels.

    Arguments and returns match the Hopper entry point. Inference only: the
    kernels keep no saved tensors, so a grad-enabled call is rejected rather
    than silently producing an output that cannot be differentiated.
    """
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)
    ):
        raise RuntimeError("chunk KDA supports inference only")

    BT = chunk_size
    reason = _validate(q, k, v, BT)
    if reason is not None:
        raise ValueError(reason)

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    w, u, qg, kg, Aqk, Akk, g_cumsum = _kda_fwd_intra_triton(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=BT,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        use_qk_l2norm=True,
        apply_beta_sigmoid=True,
    )
    return _kda_fwd_h_o_triton(
        kg=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        qg=qg,
        Aqk=Aqk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
    )


# The operator registry resolves "chunk_kda_fwd_infer"; on Hopper with TLE it
# finds the warp-specialized kernels under runtime/backend/_nvidia/hopper, and
# everywhere else this pure-Triton implementation.
chunk_kda_fwd_infer = chunk_kda_fwd_infer_triton
