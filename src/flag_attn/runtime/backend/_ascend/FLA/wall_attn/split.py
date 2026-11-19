# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Split forward kernels for dense causal Wall attention on Ascend.

A prefix scan produces FP32 log2 gate prefixes. QK rescales query/key tiles
around an anchor and stores causal FP16 scores with padded row strides. PV
exponentiates the scores in UB, computes row sums and weighted values, and
normalizes the output without allocating a separate probabilities tensor.

The even-length path shares an anchor across two query tiles and uses unmasked
loads for fully covered KV tiles. Other lengths use masked loads. Sink/scalar
gates, sliding windows and packed sequences use another implementation.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as raw

from .cumsum import gate_prefix_sum
from .raw_exp_sum import exp_sum_op

_RCP_LN2 = 1.4426950216
#: extra columns per score row so the row stride is never a 4 KiB multiple.
_ROW_PAD = 128


@triton.jit
def _wall_qk_kernel(
    q,
    k,
    P,
    scores,
    lse_dummy,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    LD_S: tl.constexpr,
    COV: tl.constexpr,
    EVEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_hq = tl.program_id(1)
    i_b = tl.program_id(2)
    i_h = i_hq // (HQ // H)
    bos = i_b * T
    RCP_LN2: tl.constexpr = 1.4426950216

    rm = i_t * BT + tl.arange(0, BT)
    rn = tl.arange(0, BK)
    msk_n = rn < K

    # q tile and its gate prefix, rescaled once with the program anchor R
    q_row = ((bos + rm) * HQ + i_hq) * K
    r_row = ((bos + i_t * BT) * HQ + i_hq) * K
    b_R = tl.load(P + r_row + rn, mask=msk_n, other=0.0).to(tl.float32)
    if EVEN:
        b_q = tl.load(q + q_row[:, None] + rn[None, :])
        b_pq = tl.load(P + q_row[:, None] + rn[None, :]).to(tl.float32)
    else:
        msk_m = rm < T
        b_q = tl.load(q + q_row[:, None] + rn[None, :],
                      mask=msk_m[:, None] & msk_n[None, :], other=0.0)
        b_pq = tl.load(P + q_row[:, None] + rn[None, :],
                       mask=msk_m[:, None] & msk_n[None, :], other=0.0).to(tl.float32)
    b_qt = (b_q.to(tl.float32) * exp2(b_pq - b_R[None, :]) * (scale * RCP_LN2)).to(b_q.dtype)

    # rm is already the global row, so the tile start must not be added again
    s_base = scores + (i_b * HQ + i_hq) * (T * LD_S)
    o_rs = tl.arange(0, BS)
    # the PV kernel reads whole PV_BT-wide tiles, so the diagonal over-cover
    # must be written too (all of those columns are future rows -> -inf)
    n_tiles = (((i_t * BT + BT + COV - 1) // COV) * COV) // BS
    if EVEN:
        # bulk: every column of these tiles is <= the first query row, so the
        # causal mask is a no-op and nothing can be out of bounds.
        n_full = (i_t * BT) // BS
        for i_s in tl.range(0, n_full):
            o_k = i_s * BS + o_rs
            b_k = tl.load(k + (((bos + o_k) * H + i_h) * K)[None, :] + rn[:, None])
            b_pk = tl.load(P + (((bos + o_k) * HQ + i_hq) * K)[None, :] + rn[:, None]).to(tl.float32)
            b_ku = (b_k.to(tl.float32) * exp2(tl.minimum(b_R[:, None] - b_pk, 76.0))).to(b_k.dtype)
            b_s = tl.dot(b_qt, b_ku)
            tl.store(s_base + rm[:, None] * LD_S + o_k[None, :], b_s.to(tl.float16))
        for i_s in tl.range(n_full, n_tiles):
            o_k = i_s * BS + o_rs
            b_k = tl.load(k + (((bos + o_k) * H + i_h) * K)[None, :] + rn[:, None])
            b_pk = tl.load(P + (((bos + o_k) * HQ + i_hq) * K)[None, :] + rn[:, None]).to(tl.float32)
            b_ku = (b_k.to(tl.float32) * exp2(tl.minimum(b_R[:, None] - b_pk, 76.0))).to(b_k.dtype)
            b_s = tl.dot(b_qt, b_ku)
            # -inf is exact in fp16 and exp2(-inf) == 0, so PV needs no mask
            b_s = tl.where(rm[:, None] >= o_k[None, :], b_s, float("-inf"))
            tl.store(s_base + rm[:, None] * LD_S + o_k[None, :], b_s.to(tl.float16))
    else:
        msk_m = rm < T
        for i_s in tl.range(0, n_tiles):
            o_k = i_s * BS + o_rs
            msk_s = o_k < T
            b_k = tl.load(k + ((bos + o_k) * H + i_h) * K + rn[:, None],
                          mask=msk_s[None, :] & msk_n[:, None], other=0.0)
            b_pk = tl.load(P + ((bos + o_k) * HQ + i_hq) * K + rn[:, None],
                           mask=msk_s[None, :] & msk_n[:, None], other=0.0).to(tl.float32)
            b_ku = (b_k.to(tl.float32) * exp2(tl.minimum(b_R[:, None] - b_pk, 76.0))).to(b_k.dtype)
            b_s = tl.dot(b_qt, b_ku)
            b_s = tl.where((rm[:, None] >= o_k[None, :]) & msk_s[None, :], b_s, float("-inf"))
            tl.store(s_base + rm[:, None] * LD_S + o_k[None, :], b_s.to(tl.float16),
                     mask=msk_m[:, None] & msk_s[None, :])


@triton.jit
def exp2(x):
    return tl.exp2(x)


@triton.jit
def log2(x):
    return tl.log2(x)


@triton.jit
def _wall_qk2_kernel(
    q,
    k,
    P,
    scores,
    scale,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    COV: tl.constexpr,
    LD_S: tl.constexpr,
):
    """Two query tiles per program, sharing one anchor and one tilted K tile.

    Halving the number of q-tile programs halves the k/P loads, the exp2 of the
    decay factor and the loop bookkeeping per score element (measured -16% at
    T=4096).  The anchor is the *second* tile's first row, so the score range
    is halved as well, which is also slightly more accurate.
    """
    i_t = tl.program_id(0)
    i_hq = tl.program_id(1)
    i_b = tl.program_id(2)
    i_h = i_hq // (HQ // H)
    bos = i_b * T
    RCP_LN2: tl.constexpr = 1.4426950216
    q0 = i_t * 2 * BT
    rn = tl.arange(0, K)
    b_R = tl.load(P + ((bos + q0 + BT) * HQ + i_hq) * K + rn).to(tl.float32)
    rm1 = q0 + tl.arange(0, BT)
    rm2 = rm1 + BT
    b_q1 = tl.load(q + (((bos + rm1) * HQ + i_hq) * K)[:, None] + rn[None, :])
    b_q2 = tl.load(q + (((bos + rm2) * HQ + i_hq) * K)[:, None] + rn[None, :])
    p1 = tl.load(P + (((bos + rm1) * HQ + i_hq) * K)[:, None] + rn[None, :]).to(tl.float32)
    p2 = tl.load(P + (((bos + rm2) * HQ + i_hq) * K)[:, None] + rn[None, :]).to(tl.float32)
    b_qt1 = (b_q1.to(tl.float32) * exp2(p1 - b_R[None, :]) * (scale * RCP_LN2)).to(b_q1.dtype)
    b_qt2 = (b_q2.to(tl.float32) * exp2(p2 - b_R[None, :]) * (scale * RCP_LN2)).to(b_q2.dtype)

    s_base = scores + (i_b * HQ + i_hq) * (T * LD_S)
    o_rs = tl.arange(0, BS)
    n_tiles = (((q0 + 2 * BT + COV - 1) // COV) * COV) // BS
    n_full = q0 // BS
    for i_s in tl.range(0, n_full):
        o_k = i_s * BS + o_rs
        b_k = tl.load(k + (((bos + o_k) * H + i_h) * K)[:, None] + rn[None, :])
        b_pk = tl.load(P + (((bos + o_k) * HQ + i_hq) * K)[:, None] + rn[None, :]).to(tl.float32)
        b_ku = (b_k.to(tl.float32) * exp2(tl.minimum(b_R[None, :] - b_pk, 76.0))).to(b_k.dtype)
        tl.store(s_base + rm1[:, None] * LD_S + o_k[None, :], tl.dot(b_qt1, tl.trans(b_ku)).to(tl.float16))
        tl.store(s_base + rm2[:, None] * LD_S + o_k[None, :], tl.dot(b_qt2, tl.trans(b_ku)).to(tl.float16))
    for i_s in tl.range(n_full, n_tiles):
        o_k = i_s * BS + o_rs
        b_k = tl.load(k + (((bos + o_k) * H + i_h) * K)[:, None] + rn[None, :])
        b_pk = tl.load(P + (((bos + o_k) * HQ + i_hq) * K)[:, None] + rn[None, :]).to(tl.float32)
        b_ku = (b_k.to(tl.float32) * exp2(tl.minimum(b_R[None, :] - b_pk, 76.0))).to(b_k.dtype)
        # -inf is exact in fp16 and exp2(-inf) == 0, so PV needs no mask
        s1 = tl.where(rm1[:, None] >= o_k[None, :], tl.dot(b_qt1, tl.trans(b_ku)), float("-inf"))
        s2 = tl.where(rm2[:, None] >= o_k[None, :], tl.dot(b_qt2, tl.trans(b_ku)), float("-inf"))
        tl.store(s_base + rm1[:, None] * LD_S + o_k[None, :], s1.to(tl.float16))
        tl.store(s_base + rm2[:, None] * LD_S + o_k[None, :], s2.to(tl.float16))


@triton.jit
def _wall_pv_kernel(
    scores,
    v,
    o,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BV: tl.constexpr,
    LD_S: tl.constexpr,
    EVEN: tl.constexpr,
    DOTDEN: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_hq = tl.program_id(1)
    i_b = tl.program_id(2)
    i_h = i_hq // (HQ // H)
    bos = i_b * T

    rm = i_t * BT + tl.arange(0, BT)
    rv = tl.arange(0, BV)
    o_rs = tl.arange(0, BS)
    msk_v = rv < V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_den = tl.zeros([BT], dtype=tl.float32)
    # BF16 uses Cube row sums; FP16 uses the in-place Raw exp/sum operation.
    b_den16 = tl.zeros([BT, 16], dtype=tl.float32)
    b_ones = tl.full([BS, 16], 1.0, dtype=tl.float16)

    # rm is already the global row, so the tile start must not be added again
    s_base = scores + (i_b * HQ + i_hq) * (T * LD_S)
    n_tiles = (i_t * BT + BT + BS - 1) // BS
    if EVEN:
        # the QK kernel already wrote every column of every tile (future
        # columns are -inf and exp2(-inf) == 0), so no mask is needed here
        for i_s in tl.range(0, n_tiles):
            o_k = i_s * BS + o_rs
            b_s = tl.load(s_base + rm[:, None] * LD_S + o_k[None, :])
            if DOTDEN:
                b_p = tl.exp2(b_s)
                b_den16 += tl.dot(b_p, b_ones)
            else:
                tl.static_assert(BS == 256 and BT % 16 == 0)
                sums = tl.full((1, BT), 0, tl.float32)
                scratch = tl.full((16, 256), 0, tl.float32)
                b_p, sums, scratch = raw.call(
                    exp_sum_op, [b_s, sums, scratch], output_indices=[0, 1, 2]
                )
                b_sum = tl.sum(sums, 0)
                b_den += b_sum
            b_v = tl.load(v + (((bos + o_k) * H + i_h) * V)[:, None] + rv[None, :])
            b_o += tl.dot(b_p, b_v.to(b_p.dtype))
    else:
        msk_m = rm < T
        for i_s in tl.range(0, n_tiles):
            o_k = i_s * BS + o_rs
            msk_s = o_k < T
            b_s = tl.load(s_base + rm[:, None] * LD_S + o_k[None, :],
                          mask=msk_m[:, None] & msk_s[None, :], other=float("-inf"))
            if DOTDEN:
                b_p = tl.exp2(b_s.to(scores.dtype.element_ty))
                b_den16 += tl.dot(b_p, b_ones)
            else:
                tl.static_assert(BS == 256 and BT % 16 == 0)
                sums = tl.full((1, BT), 0, tl.float32)
                scratch = tl.full((16, 256), 0, tl.float32)
                b_p, sums, scratch = raw.call(
                    exp_sum_op, [b_s, sums, scratch], output_indices=[0, 1, 2]
                )
                b_sum = tl.sum(sums, 0)
                b_den += b_sum
            b_v = tl.load(v + (((bos + o_k) * H + i_h) * V)[:, None] + rv[None, :],
                          mask=msk_s[:, None] & msk_v[None, :], other=0.0)
            b_o += tl.dot(b_p, b_v.to(b_p.dtype))

    if DOTDEN:
        b_den = tl.sum(b_den16, 1) * 0.0625  # the 16 columns are identical
    b_o = b_o / tl.where(b_den[:, None] > 0.0, b_den[:, None], 1.0)
    q_row = ((bos + rm) * HQ + i_hq) * V
    if EVEN:
        tl.store(o + q_row[:, None] + rv[None, :], b_o.to(o.dtype.element_ty))
    else:
        msk_m = rm < T
        tl.store(o + q_row[:, None] + rv[None, :], b_o.to(o.dtype.element_ty),
                 mask=msk_m[:, None] & msk_v[None, :])


def _wall_split_impl(q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar):
    """Dense non-varlen split forward (the benchmark contract)."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    # the split kernels implement plain causal attention only: a sliding window
    # or a sink/scalar gate must go to the generic fused kernel, which has the
    # masking for them.
    if (cu_seqlens is not None or sink_bias is not None or g_scalar is not None
            or window_size is not None):
        return None
    if K != 64 or V != 64:
        return None
    P = gate_prefix_sum(g, scale=_RCP_LN2)
    ld_s = T + _ROW_PAD
    scores = torch.empty((B * HQ * T, ld_s), dtype=torch.float16, device=q.device)
    o = torch.empty((B, T, HQ, V), dtype=q.dtype, device=q.device)
    # Align QK coverage with PV tiles so diagonal padding is initialized.
    qk_bt, qk_bs, qk_bk = 128, 128, 64
    pv_bt, pv_bs, pv_bv = 128, 256, 64
    #: the wide-tile path needs T divisible by 2*qk_bt == pv_bs == 256
    even = (T % (2 * qk_bt) == 0) and (T % pv_bs == 0)
    if even:
        # two query tiles per program: shared K tilt, half the loop overhead
        _wall_qk2_kernel[(T // (2 * qk_bt), HQ, B)](
            q, k, P, scores, scale, T, H=H, HQ=HQ, K=K, BT=qk_bt, BS=qk_bs,
            COV=pv_bs, LD_S=ld_s)
    else:
        _wall_qk_kernel[(triton.cdiv(T, qk_bt), HQ, B)](
            q, k, P, scores, o, scale, T,
            B=B, H=H, HQ=HQ, K=K, BT=qk_bt, BS=qk_bs, BK=qk_bk,
            LD_S=ld_s, COV=pv_bs, EVEN=even)
    pv_grid = (triton.cdiv(T, pv_bt), HQ, B)
    _wall_pv_kernel[pv_grid](scores, v, o, T,
                             B=B, H=H, HQ=HQ, K=K, V=V, BT=pv_bt, BS=pv_bs, BV=pv_bv,
                             LD_S=ld_s, EVEN=even, DOTDEN=q.dtype == torch.bfloat16)
    return o, None
