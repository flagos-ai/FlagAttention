# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Upstream: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/wall_attn
# The general implementation is preserved from the accepted 13d0e0f snapshot.
# Wall attention was contributed upstream by Tilde Research (Timor Averbuch, Dhruv Pai).
"""T-Head Wall-Attention prefill/training API.

Uses generic Triton forward/backward kernels without NVIDIA-specific dispatch.
"""

# This module has no runtime dependency on the source FLA repository.

import os
from contextvars import ContextVar

import torch
import triton
import triton.language as tl

from flag_attn.FLA.cumsum import chunk_global_cumsum
from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.parallel_nsa.bwd_preprocess import parallel_attn_bwd_preprocess
from flag_attn.utils import input_guard

RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.exp2(x)


@triton.jit
def log2(x):
    return tl.log2(x)


# Set WALL_ATTN_DKV_DIAG_BF16=0 to force fp32 tensor cores in the diagonal loop
# of parallel_wall_attn_bwd_kernel_dkv (precision over speed).
_DKV_DIAG_BF16 = os.environ.get("WALL_ATTN_DKV_DIAG_BF16", "1") == "1"

WALL_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS}, num_warps=nw, num_stages=ns)
    for BT in (64, 128)
    for BS in (32, 64)
    for nw in (2, 4, 8)
    for ns in (2, 3)
]

WALL_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({'BT': BT, 'BS': BS}, num_warps=nw, num_stages=ns)
    for BT in (64, 128)
    for BS in (32, 64)
    for nw in (2, 4, 8)
    for ns in (2, 3)
]


@triton.autotune(
    configs=WALL_FWD_AUTOTUNE_CONFIGS,
    key=['T', 'K', 'V', 'HQ', 'H'],
)
@triton.heuristics({
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit
def parallel_wall_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    g_scalar_cumsum,
    sink_bias,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    # int64 tile indices reduce spills in the PPU forward kernel.
    # Interleave heads in the fastest grid dimension and issue costly causal
    # query tiles first. This spreads independent heads across the 64 PPU SMs
    # and leaves only cheap tiles in the last CTA waves.
    pid = tl.program_id(0).to(tl.int64)
    nv = tl.cdiv(V, BV)
    i_v = pid % nv
    i_bh = pid // nv % (B * HQ)
    nt = tl.num_programs(0) // (nv * B * HQ)
    i_t = nt - 1 - pid // (nv * B * HQ)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    m_qk = m_q[:, None] & (o_d[None, :] < K)
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_lse = lse + bos * HQ + i_hq + o_q * HQ

    p_pq = g_cumsum + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    o_r = i_t * BT + tl.arange(0, 1)
    p_R = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_pq = tl.load(p_pq, mask=m_qk, other=0.0).to(tl.float32)
    b_R = tl.load(p_R, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
    # b_R is at i_t*BT; the same value is used in both off-diag and diag loops,
    # since |b_pq - b_R| within a BT-chunk is bounded by BT*|g_max|*RCP_LN2.
    # Off-diagonal: P_q - R <= 0 and R - P_k <= 0 -> exp2 <= 1 -> bf16 safe.
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = g_scalar_cumsum + bos * HQ + i_hq + o_q * HQ
        b_cq = tl.load(p_cq, mask=m_q, other=0.0).to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ*K)

        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_pk = tl.load(p_pk, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0).to(tl.float32)
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_mp = b_m

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        p_pk = g_cumsum + (bos * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ*K)

        # Per-sub-block local reference to prevent exp2 overflow.
        o_r = i_s + tl.arange(0, 1)
        p_R_local = g_cumsum + (bos * HQ + i_hq) * K + o_r[:, None] * (HQ*K) + o_d[None, :]
        b_R_local = tl.load(p_R_local, mask=(o_r[:, None] < T) & (o_d[None, :] < K), other=0.0).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_pk = tl.load(p_pk, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0).to(tl.float32)

        # k_til in local frame: exp2 bounded by BS positions <= 110 (fp32-safe with BK dot).
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame; clamp exp for Q before sub-block (masked by causality).
        b_exp_q = tl.minimum(b_pq - b_R_local_bc, tl.zeros([BT, BK], dtype=tl.float32))
        b_q_til_local = b_q.to(tl.float32) * exp2(b_exp_q)
        b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_mp = b_m

    if USE_SINK_BIAS:
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        b_acc += exp2(b_sink_bias - b_m)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), mask=m_q)


@triton.autotune(
    configs=WALL_BWD_AUTOTUNE_CONFIGS,
    key=['T', 'K', 'V', 'HQ', 'H'],
)
@triton.heuristics({
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_wall_attn_bwd_kernel_dq(
    q,
    k,
    v,
    g_cumsum,
    g_scalar_cumsum,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    dg_scalar_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (1, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    # Off-diagonal q_til (bf16 tensor cores, exp2 <= 1).
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq_til = tl.zeros([BT, BK], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_cq = tl.load(p_cq, boundary_check=(0,)).to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0
    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq_til += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k_til))
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Diagonal: use per-sub-block local reference to prevent exp2 overflow.
    # Accumulate dq_diag directly in output space to avoid large frame corrections.
    b_dq_diag = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_s), (BV, BS), (0, 1))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        p_R_local = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        b_R_local = tl.load(p_R_local, boundary_check=(0, 1)).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))

        # Local-ref k_til: exp2 arg bounded by BS positions <= 110 (fp32-safe with BK dot).
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame for computing scores.
        b_q_til_local = b_q.to(tl.float32) * exp2(b_pq - b_R_local_bc)
        b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        if USE_WINDOW:
            b_p = tl.where(
                (o_q[:, None] >= o_k[None, :]) & (o_q[:, None] - o_k[None, :] < W) & m_k[None, :],
                exp2(b_s - b_lse[:, None]), 0
            )
        else:
            b_p = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], exp2(b_s - b_lse[:, None]), 0)

        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # dq_sub in local frame; convert to output space: dq += dq_sub * scale * exp2(Pq - R_local).
        # For Q < sub-block start: causal mask makes ds=0, but mask explicitly to kill fp noise.
        b_dq_sub = tl.dot(b_ds.to(tl.float32), tl.trans(b_k_til))
        m_q_causal = (o_q >= i_s)[:, None]
        b_dq_diag += tl.where(m_q_causal, b_dq_sub * exp2(b_pq - b_R_local_bc), 0.0)
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Off-diagonal (chunk-ref) + diagonal (direct output space).
    b_dq = (b_dq_til * scale) * exp2(b_pq - b_R) + b_dq_diag * scale

    # Structural: b_dg derived from b_dq (saves [BT,BK] fp32 accumulator).
    b_dg = LN2 * b_q.to(tl.float32) * b_dq
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if USE_SCALAR_G:
        p_dc = tl.make_block_ptr(dg_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=WALL_BWD_AUTOTUNE_CONFIGS,
    key=['T', 'K', 'V', 'HQ', 'H'],
)
@triton.heuristics({
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_wall_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    g_scalar_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    dg_scalar_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
    DIAG_BF16: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)

    if USE_SCALAR_G:
        p_ck = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_ck = tl.load(p_ck, boundary_check=(0,)).to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_Rq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_Rq = tl.load(p_Rq, boundary_check=(0, 1)).to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
        b_exp_k = b_Rq_bc_k - b_pk
        # Non-causal keys (key_pos > last query in sub-block) can have
        # exp_k >> 110, overflowing the gradient headroom.  The causal mask
        # zeros their contribution in b_p, but inf*0 = NaN poisons b_dk/b_dg.
        # Clamp to 110 (fp32-safe with BK/BS dot accumulation) so products
        # in dg accumulation stay well within fp32.
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_exp_k_val = exp2(b_exp_k)  # CSE: reused for k_til and final dk scaling
        if DIAG_BF16:
            # Fast path: bf16 tensor cores, fp32 accumulation.
            b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
            b_k_til = (b_k.to(tl.float32) * b_exp_k_val).to(b_k.dtype)
        else:
            # Precise path: fp32 tensor cores.
            b_q_til = b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)
            b_k_til = b_k.to(tl.float32) * b_exp_k_val
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(tl.float32)
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where(
                (o_k[:, None] <= o_q[None, :]) & (o_q[None, :] - o_k[:, None] < W) & m_q[None, :],
                exp2(b_s - b_lse[None, :]), 0
            )
        else:
            b_p = tl.where((o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        if DIAG_BF16:
            b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        else:
            b_dk_til = tl.dot(b_ds.to(tl.float32), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    i_end = min(tl.cdiv(T, BS) * BS, (i_t + 1) * BT + W - 1) if USE_WINDOW else tl.cdiv(T, BS) * BS

    for i_s in range((i_t + 1) * BT, i_end, BS):
        p_Rq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (BS, BK), (1, 0))
        p_do = tl.make_block_ptr(do + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_Rq = tl.load(p_Rq, boundary_check=(0, 1)).to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
        b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
        b_exp_k_off = b_Rq_bc_k - b_pk
        b_exp_k_off_val = exp2(b_exp_k_off)  # CSE: reused for k_til and final dk scaling
        b_k_til = (b_k.to(tl.float32) * b_exp_k_off_val).to(b_k.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(tl.float32)
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where((o_q[None, :] - o_k[:, None] < W) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        else:
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_off_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    # Structural: b_dg derived from b_dk (saves [BT,BK] fp32 accumulator).
    b_dg = -LN2 * b_k.to(tl.float32) * b_dk
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if USE_SCALAR_G:
        p_dc = tl.make_block_ptr(dg_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), boundary_check=(0,))


def parallel_wall_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    sink_bias: torch.Tensor | None,
    scale: float,
    g_scalar_cumsum: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    if g_cumsum.dim() != 4 or g_cumsum.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError(
            f"`g_cumsum` must be [B, T, HQ, K] matching `q`; got {g_cumsum.shape} vs q {q.shape}"
        )
    if g_cumsum.stride(-1) != 1:
        raise ValueError("`g_cumsum` must be contiguous in the last (K) dimension")
    g_cumsum = g_cumsum.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BK = min(256, max(16, triton.next_power_of_2(K)))
    BV = min(64, max(16, triton.next_power_of_2(V)))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    if cu_seqlens is not None:
        # Autotuning sweeps BT in {64,128}, but `chunk_indices` is precomputed for a
        # single BT. Pin BT=128 and bypass the autotuner (calling the heuristics-wrapped
        # jit directly) so the varlen chunk map stays valid.
        BT = 128
        BS = min(32, max(16, triton.next_power_of_2(T)))
        num_warps = 2
        num_stages = 2
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
        lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        parallel_wall_attn_fwd_kernel.fn[(NV * NT * B * HQ,)](
            q=q, k=k, v=v, o=o,
            g_cumsum=g_cumsum, g_scalar_cumsum=g_scalar_cumsum, sink_bias=sink_bias,
            lse=lse, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            B=B, T=T, W=window_size, H=H, HQ=HQ, G=G, K=K, V=V,
            BT=BT, BS=BS, BK=BK, BV=BV, num_warps=num_warps, num_stages=num_stages,
        )
        return o, lse

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    def grid(meta):
        return (NV * triton.cdiv(T, meta['BT']) * B * HQ,)
    parallel_wall_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        sink_bias=sink_bias,
        lse=lse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        W=window_size,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
    )
    return o, lse


def parallel_wall_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    g_scalar_cumsum: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    if g_cumsum.dim() != 4 or g_cumsum.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError("`g_cumsum` must be [B, T, HQ, K] matching `q`")
    g_cumsum = g_cumsum.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    # dq/dk are reduced over the full value dim in one program (no cross-program accumulation),
    # so BV must span all of V (NV == 1). Don't cap it here -- the forward can, the backward can't.
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)

    NV = triton.cdiv(V, BV)

    # Varlen: pin BT=128 and bypass the autotuner (BT sweep would invalidate the
    # precomputed chunk map). `extra` injects the fixed launch config via `.fn`.
    is_varlen = cu_seqlens is not None
    extra = {}
    if is_varlen:
        BT = 128
        BS = min(32, max(16, triton.next_power_of_2(T)))
        num_warps = 2
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        grid = (NV, NT, B * HQ)
        extra = dict(BT=BT, BS=BS, num_warps=num_warps, num_stages=2)
    else:
        def grid(meta):
            return (NV, triton.cdiv(T, meta['BT']), B * HQ)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    dq_kernel = parallel_wall_attn_bwd_kernel_dq.fn if is_varlen else parallel_wall_attn_bwd_kernel_dq
    dkv_kernel = parallel_wall_attn_bwd_kernel_dkv.fn if is_varlen else parallel_wall_attn_bwd_kernel_dkv

    dg_cumsum = torch.empty(B, T, HQ, K, dtype=torch.float, device=q.device)
    dg_cumsum_k = torch.empty_like(dg_cumsum)

    if g_scalar_cumsum is not None:
        dg_scalar_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_scalar_cumsum_k = torch.empty_like(dg_scalar_cumsum)
    else:
        dg_scalar_cumsum = None
        dg_scalar_cumsum_k = None

    dq_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        dg_scalar_cumsum=dg_scalar_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        **extra,
    )
    dkv_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum_k,
        dg_scalar_cumsum=dg_scalar_cumsum_k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        DIAG_BF16=_DKV_DIAG_BF16,
        **extra,
    )
    dk = dk.view(B, T, H, G, K).sum(3)
    dv = dv.view(B, T, H, G, V).sum(3)
    dg_cumsum.add_(dg_cumsum_k)

    if g_scalar_cumsum is not None:
        dg_scalar_cumsum.add_(dg_scalar_cumsum_k)

    dsink_bias = None
    if sink_bias is not None:
        p_sink_bias = torch.exp2(sink_bias[None, None, :] - lse)
        dsink_bias = -(p_sink_bias * delta).sum((0, 1))

    return dq, dk, dv, dg_cumsum, dsink_bias, dg_scalar_cumsum


class WallParallelAttentionFunction(torch.autograd.Function):
    r"""Wall Attention: per-channel prefix :math:`P` (log_2); scores use
    :math:`\sum_n q_{in} k_{jn} \exp_2(P_{in}-P_{jn})` via per-Q-block reference
    :math:`R_n=P_{i_t\cdot BT,n}` inside :func:`parallel_wall_attn_fwd_kernel`."""

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar=None, chunk_indices=None):
        if g.shape[-1] != q.shape[-1] or g.shape[2] != q.shape[2]:
            raise ValueError(
                f"`g` must be [B, T, HQ, K] with same HQ and K as `q`; got {g.shape} vs q {q.shape}"
            )
        if q.shape[2] % k.shape[2] != 0:
            raise ValueError("HQ must be divisible by `k.shape[2]` for GQA")


        P = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2)

        if g_scalar is not None:
            c = chunk_global_cumsum(g_scalar, cu_seqlens=cu_seqlens, scale=RCP_LN2)
        else:
            c = None

        sink_bias_scaled = sink_bias * RCP_LN2 if sink_bias is not None else None
        o, lse = parallel_wall_attn_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=P,
            sink_bias=sink_bias_scaled,
            scale=scale,
            g_scalar_cumsum=c,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        ctx.save_for_backward(q, k, v, o, P, lse, sink_bias_scaled, c if c is not None else torch.empty(0))
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.cu_seqlens = cu_seqlens
        ctx.has_scalar_g = g_scalar is not None
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, k, v, o, P, lse, sink_bias_scaled, c_or_empty = ctx.saved_tensors
        c = c_or_empty if ctx.has_scalar_g else None


        dq, dk, dv, dP, dsink_bias, dc = parallel_wall_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=P,
            lse=lse,
            do=do,
            sink_bias=sink_bias_scaled,
            scale=ctx.scale,
            g_scalar_cumsum=c,
            window_size=ctx.window_size,
            cu_seqlens=ctx.cu_seqlens,
        )

        # The kernel emits dP with an LN2 factor (b_dg = LN2 * q * dq); the forward set
        # P = cumsum(g) * RCP_LN2, so dL/dg = RCP_LN2 * reverse_cumsum(dP) (nets the LN2).
        dg = chunk_global_cumsum(dP, cu_seqlens=ctx.cu_seqlens, reverse=True, scale=RCP_LN2)

        if dc is not None:
            dg_scalar = chunk_global_cumsum(dc, cu_seqlens=ctx.cu_seqlens, reverse=True)
        else:
            dg_scalar = None

        return dq.to(q), dk.to(k), dv.to(v), dg, dsink_bias, None, None, None, dg_scalar, None


_LAST_ROUTE: ContextVar[str | None] = ContextVar("wall_attn_route", default=None)


def get_last_route() -> str | None:
    """Return the route used by the T-Head implementation."""
    return _LAST_ROUTE.get()


def select_route(*args, **kwargs) -> str:
    """Select the generic Triton implementation."""
    return "thead_triton_generic"


def parallel_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""Wall parallel attention (training / prefill forward + backward).

    Scores use a per-channel multiplicative decay: with :math:`P = \mathrm{cumsum}(g)`
    in :math:`\log_2` space, the logit for pair :math:`(i, j)` is
    :math:`\mathrm{scale} \sum_n q_{in} k_{jn} \exp_2(P_{in} - P_{jn})`.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, HQ, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]`` (``H`` may be < ``HQ`` for GQA).
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            Per-channel log-decay of shape ``[B, T, HQ, K]`` (cumsum'd internally).
        g_scalar (torch.Tensor, Optional):
            FoX-style additive scalar gate of shape ``[B, T, HQ]``. Default: `None`.
        sink_bias (torch.Tensor, Optional):
            Attention-sink logit of shape ``[HQ]``. Default: `None`.
        scale (float, Optional):
            Softmax scale. If `None`, defaults to ``K ** -0.5``. Default: `None`.
        window_size (int, Optional):
            Sliding-window width (causal). Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative seqlens of shape ``[N + 1]`` for varlen packing
            (requires ``B == 1``). Default: `None`.

    Returns:
        Attention output of shape ``[B, T, HQ, V]``.
    """
    _LAST_ROUTE.set("thead_triton_generic")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    return WallParallelAttentionFunction.apply(
        q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar, None
    )


__all__ = ["parallel_wall_attn", "select_route", "get_last_route"]
