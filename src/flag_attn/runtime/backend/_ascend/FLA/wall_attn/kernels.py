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

"""Fused Wall attention forward kernel for Ascend.

The first query row's gate prefix anchors the Q/K rescaling for every KV
tile. The kernel applies causal and optional window masks before accumulating
exponential weights and weighted values.
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
    PV_FP32: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b = i_bh // HQ
    i_hq = i_bh - i_b * HQ
    i_h = i_hq // (HQ // H)
    bos = i_b * T

    RCP_LN2: tl.constexpr = 1.4426950216

    rm = i_t * BT + tl.arange(0, BT)
    rn = tl.arange(0, BK)
    rs = tl.arange(0, BS)
    rv = i_v * BV + tl.arange(0, BV)
    msk_m = rm < T
    msk_n = rn < K
    msk_v = rv < V

    q_row = ((bos + rm) * HQ + i_hq) * K
    b_q = tl.load(q + q_row[:, None] + rn[None, :], mask=msk_m[:, None] & msk_n[None, :], other=0.0)
    b_pq = tl.load(g_cumsum + q_row[:, None] + rn[None, :],
                   mask=msk_m[:, None] & msk_n[None, :], other=0.0).to(tl.float32)
    # Program anchor: the first query row of this block.
    r_row = ((bos + i_t * BT) * HQ + i_hq) * K
    b_R = tl.load(g_cumsum + r_row + rn, mask=msk_n, other=0.0).to(tl.float32)
    b_qt = (b_q.to(tl.float32) * exp2(b_pq - b_R[None, :]) * (scale * RCP_LN2)).to(b_q.dtype)

    if USE_SCALAR_G:
        b_cq = tl.load(g_scalar_cumsum + (bos + rm) * HQ + i_hq, mask=msk_m, other=0.0).to(tl.float32)

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0
    i_end = min((i_t + 1) * BT, T)

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    for i_s in range(i_start, i_end, BS):
        o_k = i_s + rs
        msk_s = o_k < T
        k_col = ((bos + o_k) * H + i_h) * K
        pk_col = ((bos + o_k) * HQ + i_hq) * K
        v_row = ((bos + o_k) * H + i_h) * V
        b_k = tl.load(k + k_col[None, :] + rn[:, None],
                      mask=msk_s[None, :] & msk_n[:, None], other=0.0)
        b_v = tl.load(v + v_row[:, None] + rv[None, :],
                      mask=msk_s[:, None] & msk_v[None, :], other=0.0)
        b_pk = tl.load(g_cumsum + pk_col[None, :] + rn[:, None],
                       mask=msk_s[None, :] & msk_n[:, None], other=0.0).to(tl.float32)

        # Cap the key exponent before rescaling in the shared anchor frame.
        b_k_use = (b_k.to(tl.float32)
                   * exp2(tl.minimum(b_R[:, None] - b_pk, 76.0))).to(b_k.dtype)
        b_s = tl.dot(b_qt, b_k_use)
        b_s = tl.where((rm[:, None] >= o_k[None, :]) & msk_s[None, :], b_s, float("-inf"))
        if USE_WINDOW:
            b_s = tl.where(rm[:, None] - o_k[None, :] < W, b_s, float("-inf"))

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq,
                           mask=msk_s, other=0.0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        b_p = exp2(b_s)
        b_acc += tl.sum(b_p, 1)
        if PV_FP32:
            # Keep unnormalized probabilities in FP32 for FP16 inputs.
            b_o += tl.dot(b_p, b_v.to(tl.float32))
        else:
            b_o += tl.dot(b_p.to(b_v.dtype), b_v)

    if USE_SINK:
        b_acc += exp2(tl.load(sink_bias + i_hq).to(tl.float32))

    b_o = b_o / b_acc[:, None]
    tl.store(o + q_row[:, None] + rv[None, :], b_o.to(o.dtype.element_ty),
             mask=msk_m[:, None] & msk_v[None, :])
    tl.store(lse + bos * HQ + i_hq + rm * HQ, log2(b_acc), mask=msk_m)
