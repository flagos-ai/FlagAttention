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

@triton.jit
def _wall_rescale_bounds(
    Q, K_IN, P, SAFE, T, W,
    H: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
    USE_WINDOW: tl.constexpr,
):
    MAX_RESCALE_LOG2: tl.constexpr = 60.0 * 1.4426950216
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HQ
    head = bh % HQ
    kv_head = head // (HQ // H)
    first = tile * BT
    end = tl.minimum(first + BT, T)
    begin = tl.maximum(first - W + 1, 0) if USE_WINDOW else 0
    d = tl.arange(0, BK)
    r = first + tl.arange(0, BT)
    anchor_ptr = ((batch * T + first) * HQ + head) * K + d
    anchor = tl.load(P + anchor_ptr, d < K, 0)
    q_ptr = ((batch * T + r[:, None]) * HQ + head) * K + d[None, :]
    q_mask = (r[:, None] < end) & (d[None, :] < K)
    pq = tl.load(P + q_ptr, q_mask, 0)
    q_max = tl.max(tl.max(tl.where(q_mask, pq - anchor[None, :], 0.0), 0), 0)
    safe = q_max <= MAX_RESCALE_LOG2
    if Q.dtype.element_ty != tl.float16:
        q = tl.load(Q + q_ptr, q_mask, 0).to(tl.float32)
        safe = safe & (tl.max(tl.max(tl.abs(q), 0), 0) <= 1e6)

    start = begin
    while (start < end) & safe:
        c = start + tl.arange(0, BS)
        mask = (c[:, None] < end) & (d[None, :] < K)
        p_ptr = ((batch * T + c[:, None]) * HQ + head) * K + d[None, :]
        pk = tl.load(P + p_ptr, mask, 0)
        k_max = tl.max(tl.max(tl.where(mask, anchor[None, :] - pk, 0.0), 0), 0)
        safe = k_max <= MAX_RESCALE_LOG2
        if K_IN.dtype.element_ty != tl.float16:
            k_ptr = ((batch * T + c[:, None]) * H + kv_head) * K + d[None, :]
            k = tl.load(K_IN + k_ptr, mask, 0).to(tl.float32)
            safe = safe & (tl.max(tl.max(tl.abs(k), 0), 0) <= 1e6)
        start += BS
    tl.store(SAFE + bh * tl.num_programs(0) + tile, safe.to(tl.int32))


@triton.jit
def _wall_stable_tile(
    Q, K_IN, V_IN, P, C, SINK, O, LSE, scale, T, W, first, batch, head, v_start,
    H: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SINK: tl.constexpr, USE_SCALAR_G: tl.constexpr,
    DIRECT: tl.constexpr, SAVE_LSE: tl.constexpr,
):
    kv_head = head // (HQ // H)
    r = first + tl.arange(0, BT)
    d = tl.arange(0, BK)
    v_cols = v_start + tl.arange(0, BV)
    row_mask = r < T
    q_base = (batch * T + r) * HQ + head
    if not DIRECT:
        q_ptr = q_base[:, None] * K + d[None, :]
        q_mask = row_mask[:, None] & (d[None, :] < K)
        anchor = tl.load(P + ((batch * T + first) * HQ + head) * K + d, d < K, 0)
        q = tl.load(Q + q_ptr, q_mask, 0).to(tl.float32)
        pq = tl.load(P + q_ptr, q_mask, 0)
        q = q * tl.exp2(tl.where(q_mask, pq - anchor[None, :], 0.0))

    if USE_SCALAR_G:
        cq = tl.load(C + q_base, row_mask, 0)

    start = tl.maximum(first - W + 1, 0) if USE_WINDOW else 0
    end = tl.minimum(first + BT, T)
    acc = tl.zeros((BT, BV), tl.float32)
    if USE_SINK:
        mx = tl.full((BT,), 0.0, tl.float32) + tl.load(SINK + head).to(tl.float32) * 1.4426950216
        den = tl.full((BT,), 1.0, tl.float32)
    else:
        mx = tl.full((BT,), -float("inf"), tl.float32)
        den = tl.zeros((BT,), tl.float32)

    for block in range(start, end, BS):
        c = block + tl.arange(0, BS)
        col_mask = c < end
        valid = row_mask[:, None] & col_mask[None, :] & (c[None, :] <= r[:, None])
        if USE_WINDOW:
            valid = valid & ((r[:, None] - c[None, :]) < W)
        k_base = (batch * T + c) * H + kv_head
        p_base = (batch * T + c) * HQ + head
        if DIRECT:
            # Use the same gated score without forming separate exponential factors.
            score = tl.zeros((BT, BS), tl.float32)
            for channel in range(tl.cdiv(K, 16)):
                f = channel * 16 + tl.arange(0, 16)
                q_ptr = q_base[:, None] * K + f[None, :]
                k_ptr = k_base[:, None] * K + f[None, :]
                p_ptr = p_base[:, None] * K + f[None, :]
                q_mask = row_mask[:, None] & (f[None, :] < K)
                k_mask = col_mask[:, None] & (f[None, :] < K)
                q = tl.load(Q + q_ptr, q_mask, 0).to(tl.float32)
                k = tl.load(K_IN + k_ptr, k_mask, 0).to(tl.float32)
                pq = tl.load(P + q_ptr, q_mask, 0)
                pk = tl.load(P + p_ptr, k_mask, 0)
                delta = tl.where(valid[:, :, None], pq[:, None, :] - pk[None, :, :], 0.0)
                score += tl.sum(q[:, None, :] * k[None, :, :] * tl.exp2(delta), 2)
        else:
            k_mask = col_mask[:, None] & (d[None, :] < K)
            k = tl.load(K_IN + k_base[:, None] * K + d[None, :], k_mask, 0).to(tl.float32)
            pk = tl.load(P + p_base[:, None] * K + d[None, :], k_mask, 0)
            k = k * tl.exp2(tl.where(k_mask, anchor[None, :] - pk, 0.0))
            score = tl.dot(q, tl.trans(k), input_precision="hf32")

        score = score * (scale * 1.4426950216)
        if USE_SCALAR_G:
            ck = tl.load(C + p_base, col_mask, 0)
            score += cq[:, None] - ck[None, :]
        score = tl.where(valid, score, -float("inf"))

        next_mx = tl.maximum(mx, tl.max(score, 1))
        shift = tl.where(next_mx == -float("inf"), 0.0, next_mx)
        old = den * tl.exp2(mx - shift)
        prob = tl.exp2(score - shift[:, None])
        den = old + tl.sum(prob, 1)
        inv = 1.0 / tl.where(den > 0.0, den, 1.0)
        value = tl.load(
            V_IN + k_base[:, None] * V + v_cols[None, :],
            col_mask[:, None] & (v_cols[None, :] < V), 0,
        ).to(tl.float32)
        acc = acc * (old * inv)[:, None] + tl.dot(prob * inv[:, None], value, input_precision="hf32")
        mx = next_mx

    tl.store(
        O + q_base[:, None] * V + v_cols[None, :], acc,
        row_mask[:, None] & (v_cols[None, :] < V),
    )
    if SAVE_LSE:
        if v_start == 0:
            tl.store(LSE + q_base, mx + tl.log2(den), row_mask)


@triton.jit
def _wall_stable_fwd(
    Q, K_IN, V_IN, P, C, SINK, SAFE, O, LSE, scale, T, W,
    H: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    USE_WINDOW: tl.constexpr, USE_SINK: tl.constexpr, USE_SCALAR_G: tl.constexpr, SAVE_LSE: tl.constexpr,
):
    tile = tl.program_id(1)
    bh = tl.program_id(2)
    batch = bh // HQ
    head = bh % HQ
    v_start = tl.program_id(0) * BV
    first = tile * BT
    if tl.load(SAFE + bh * tl.num_programs(1) + tile) != 0:
        _wall_stable_tile(
            Q, K_IN, V_IN, P, C, SINK, O, LSE, scale, T, W, first, batch, head, v_start,
            H, HQ, K, V, BT, BS, BK, BV, USE_WINDOW, USE_SINK, USE_SCALAR_G, False, SAVE_LSE,
        )
    else:
        for sub in range(BT // 16):
            _wall_stable_tile(
                Q, K_IN, V_IN, P, C, SINK, O, LSE, scale, T, W, first + sub * 16, batch, head, v_start,
                H, HQ, K, V, 16, 16, BK, BV, USE_WINDOW, USE_SINK, USE_SCALAR_G, True, SAVE_LSE,
            )
