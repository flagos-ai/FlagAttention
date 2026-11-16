# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernel for the Wall attention forward pass.

Score for pair (i, j) is ``sum_n q_in k_jn exp2(P_in - P_jn) * scale * log2(e)``
with ``P`` the per-channel log2 prefix of the gate.  As in the upstream Triton
kernel the decay is folded into the operands per BT block: with an anchor
``R = P[i_t*BT]``, ``q_til = q*exp2(P_q - R)`` and ``k_til = k*exp2(R - P_k)``,
so the score is a plain dot product of rescaled operands.  The diagonal blocks
use their own local anchor to keep exp2 bounded.

Note: ``BS <= BT`` is required so that an off-diagonal KV block never reaches
into the query rows of the same program (those need the causal mask).
"""

import triton
import triton.language as tl

_RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.exp2(x)


@triton.jit
def log2(x):
    return tl.log2(x)


@triton.jit
def _wall_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    g_scalar_cumsum,
    sink_bias,
    lse,
    scale,
    T,
    W,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    USE_SINK: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // HQ
    i_hq = i_bh - i_b * HQ
    i_h = i_hq // (HQ // H)
    bos = i_b * T

    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (1, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    # Fold the constant scale into the per-program Q rescale so the inner loop
    # needs one multiply less per KV block.
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R) * (scale * RCP_LN2)).to(b_q.dtype)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_cq = tl.load(p_cq, boundary_check=(0,)).to(tl.float32)

    if USE_SINK:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = 0.0

    o_q = i_t * BT + tl.arange(0, BT)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0
    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1))
        p_R_local = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_s, 0), (1, BK), (1, 0))

        b_R_local = tl.load(p_R_local, boundary_check=(0, 1)).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)

        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = (b_k.to(tl.float32) * exp2(b_exp_k)).to(b_k.dtype)

        b_exp_q = tl.minimum(b_pq - b_R_local_bc, tl.zeros([BT, BK], dtype=tl.float32))
        b_q_til_local = (b_q.to(tl.float32) * exp2(b_exp_q) * (scale * RCP_LN2)).to(b_q.dtype)
        b_s = tl.dot(b_q_til_local, b_k_til)

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)

    if USE_SINK:
        b_m = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_acc += exp2(b_sink_bias - b_m)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))
