# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-length chunked GLA forward for Ascend.

Gates are cumsummed in log2 units within each chunk (BT=64). The state scan
computes h_next = exp2(g_last) * h + (k * exp2(g_last - g_cum))^T @ v;
output combines the incoming state projection with causal A @ v.
Dots and running states use FP32; stored chunk states use the input dtype.
Both [K,V] and state_v_first [V,K] state layouts are supported.

Intra-chunk A normally uses BC=16 panels with midpoint exponent references
and a capped query factor to limit overflow under strong gate decay.
For K=V=128 and T>=1024, A and output are fused. Chunks with midpoint gate
span <=60 log2 units and sufficient q/k exponent headroom use a full-width
dot; others consume panels immediately. Each branch stores independently
to avoid a CANN 9.1 mixed-core accumulator-merge compilation failure.
Training saves A for backward dv; fused inference omits its allocation.

Chunk-parallel launches are grouped with a 512-program budget to bound
runtime scratch. The backward lives in chunk_bwd.py; packed and host-split
varlen paths are dispatched separately by api.py."""

import torch
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950216)
# State-scan chunk size and intra-chunk panel width. Midpoint references
# limit each gate factor's exponent span; PANEL_LOG_MAX caps the query factor.
BT = 64
BC = 16
# Cap on the panel exponent, in log2 units.  It keeps rows above the panel
# finite (their dA / q values are zero) without touching the in-panel factors.
PANEL_LOG_MAX = tl.constexpr(100.0)
# Keep both whole-chunk gate factors representable with headroom for operands.
WIDE_LOG_SPAN_MAX = tl.constexpr(60.0)
# Launch tiles.  The state scan and the output projection use much wider tiles
# than the intra-chunk matrix: both amortise redundant k/v (resp. h) loads.
BK = 64
BK_W = 256
BV = 256
STATE_BK = 128
STATE_BV = 128
MAX_HEAD_DIM = 512
# Cap on the number of programs of one chunk-parallel launch; see the module
# docstring.  Small problems stay below it and run a single launch.
MAX_PROGRAMS = 1024
# Fixed-length launches use a smaller program budget to reduce peak scratch
# memory. Keep the packed varlen default separate: its short batches are
# launch-bound.
FIXED_MAX_PROGRAMS = 512


def _chunks_per_launch(programs: int, nt: int, max_programs: int | None = None) -> int:
    """Chunks covered by one launch of a kernel whose full grid is ``programs``.

    A full grid at or below the program budget runs as a single launch; wider
    grids are split into ``ceil(programs / budget)`` groups of contiguous
    chunks. The default budget is ``MAX_PROGRAMS``; fixed-length callers use
    ``FIXED_MAX_PROGRAMS`` to reduce launch scratch.
    """
    budget = MAX_PROGRAMS if max_programs is None else max_programs
    groups = max(1, triton.cdiv(programs, budget))
    return triton.cdiv(nt, groups)


@triton.jit
def exp2(x):
    """Base-2 exponential in float32 (gates are pre-scaled by log2(e))."""
    return tl.exp2(x.to(tl.float32))


def _state_tiles(K: int, V: int) -> tuple[int, int]:
    """Tile sizes of the sequential state scan, clamped to the head dims."""
    return min(STATE_BK, triton.next_power_of_2(K)), min(STATE_BV, triton.next_power_of_2(V))


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_cumsum_kernel(
    g,
    g_cum,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """Per-chunk inclusive cumsum of the gate, in base-2 log units."""
    i_t = tl.program_id(0)
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m = (o_t[:, None] < T) & (o_k[None, :] < K)

    off = i_b * T * H * K + i_h * K
    b_g = tl.load(g + off + o_t[:, None] * (H * K) + o_k[None, :], mask=m, other=0.0)
    b_g = tl.cumsum(b_g.to(tl.float32) * RCP_LN2, axis=0)
    tl.store(g_cum + off + o_t[:, None] * (H * K) + o_k[None, :], b_g, mask=m)


@triton.jit
def _wide_operand_safe(x):
    """Leave headroom for the gate factors; biased exponents 67/187 encode +/-60."""
    if x.dtype == tl.float16:
        magnitude = x.to(tl.uint16, bitcast=True) & 0x7fff
        return magnitude < 0x7c00
    elif x.dtype == tl.bfloat16:
        magnitude = x.to(tl.uint16, bitcast=True) & 0x7fff
        return (magnitude == 0) | ((magnitude >= (67 << 7)) & (magnitude < (187 << 7)))
    else:
        magnitude = x.to(tl.int32, bitcast=True) & 0x7fffffff
        return (magnitude == 0) | ((magnitude >= (67 << 23)) & (magnitude < (187 << 23)))


@triton.jit
def _wide_block_unsafe(q, k, offsets, mask):
    b_q = tl.load(q + offsets, mask=mask, other=0.0)
    b_k = tl.load(k + offsets, mask=mask, other=0.0)
    unsafe = mask & ~(_wide_operand_safe(b_q) & _wide_operand_safe(b_k))
    return tl.max(tl.max(unsafe.to(tl.int32), axis=0), axis=0)


@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_cumsum_wide_kernel(
    q, k, g, g_cum, wide_ok, T,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
):
    """Compute gates and one device-side wide-dot eligibility flag per chunk/head."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m = (o_t[:, None] < T) & (o_k[None, :] < K)
    off = i_b * T * H * K + i_h * K + o_t[:, None] * (H * K) + o_k[None, :]
    b_g = tl.load(g + off, mask=m, other=0.0).to(tl.float32)
    b_g = tl.cumsum(b_g * RCP_LN2, axis=0)
    tl.store(g_cum + off, b_g, mask=m)

    mid = tl.minimum(BT // 2, T - 1 - i_t * BT)
    b_ref = tl.sum(tl.where(tl.arange(0, BT)[:, None] == mid, b_g, 0.0), axis=0)
    span = tl.max(tl.max(tl.where(m, tl.abs(b_g - b_ref[None, :]), 0.0), axis=0), axis=0)
    p_flag = wide_ok + (i_b * tl.cdiv(T, BT) + i_t) * H + i_h
    if span <= WIDE_LOG_SPAN_MAX:
        # fp16 finite values already have sufficient headroom. For bf16/fp32,
        # nonzero magnitudes must lie in [2^-60, 2^60), so multiplying by
        # either gate factor cannot overflow or underflow an operand to zero.
        if q.dtype.element_ty == tl.float32 or k.dtype.element_ty == tl.float32:
            # Scan fp32 operands in halves so they fit beside the cumsum tile.
            any_unsafe = tl.full((), 0, tl.int32)
            for i_k in range(tl.cdiv(K, 64)):
                cols = i_k * 64 + tl.arange(0, 64)
                op_mask = (o_t[:, None] < T) & (cols[None, :] < K)
                offsets = i_b * T * H * K + i_h * K + o_t[:, None] * (H * K) + cols[None, :]
                any_unsafe = tl.maximum(any_unsafe, _wide_block_unsafe(q, k, offsets, op_mask))
        else:
            any_unsafe = _wide_block_unsafe(q, k, off, m)
        tl.store(p_flag, any_unsafe == 0)
        return
    tl.store(p_flag, False)


@triton.jit(do_not_specialize=["T", "NT"])
def chunk_gla_fwd_h_kernel(
    k,
    v,
    g_cum,
    h,
    h0,
    ht,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    """Sequential inter-chunk state scan; stores the state entering each chunk."""
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V
        if STATE_V_FIRST:
            b_h = tl.trans(
                tl.load(p_h0 + o_v[:, None].to(tl.int64) * K + o_k[None, :], mask=m_v[:, None] & m_k[None, :],
                        other=0.0))
        else:
            b_h = tl.load(p_h0 + o_k[:, None].to(tl.int64) * V + o_v[None, :], mask=m_h, other=0.0)
        b_h = b_h.to(tl.float32)

    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    for i_t in range(NT):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_kt = m_k[:, None] & m_t[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        o_h = ((i_b * NT + i_t) * H + i_h) * K * V
        if STATE_V_FIRST:
            p_h = h + o_h + o_v[:, None].to(tl.int64) * K + o_k[None, :]
            tl.store(p_h, tl.trans(b_h).to(h.dtype.element_ty),
                     mask=m_v[:, None] & m_k[None, :])
        else:
            p_h = h + o_h + o_k[:, None].to(tl.int64) * V + o_v[None, :]
            tl.store(p_h, b_h.to(h.dtype.element_ty), mask=m_h)

        last_idx = i_t * BT + BT - 1
        last_idx = tl.where(last_idx < T, last_idx, T - 1)
        b_gl = tl.load(g_cum + base_k + last_idx * (H * K) + o_k, mask=m_k, other=0.0)
        b_k = tl.load(k + base_k + o_t[None, :] * (H * K) + o_k[:, None], mask=m_kt, other=0.0)
        b_g = tl.load(g_cum + base_k + o_t[None, :] * (H * K) + o_k[:, None], mask=m_kt, other=0.0)
        b_v = tl.load(v + base_v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)

        b_h = b_h * exp2(b_gl)[:, None]
        b_ke = b_k.to(tl.float32) * exp2(b_gl[:, None] - b_g)
        b_h = tl.dot(b_ke, b_v.to(tl.float32), b_h)

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V
        if STATE_V_FIRST:
            tl.store(p_ht + o_v[:, None].to(tl.int64) * K + o_k[None, :], tl.trans(b_h),
                     mask=m_v[:, None] & m_k[None, :])
        else:
            tl.store(p_ht + o_k[:, None].to(tl.int64) * V + o_v[None, :], b_h, mask=m_h)


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_fwd_A_kernel(
    q,
    k,
    g_cum,
    A,
    scale,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """Intra-chunk causal matrix: A[i, j] = scale * q_i.k_j * exp2(g_i - g_j).

    A is built in ``BC``-wide column panels.  The panel *midpoint* is the
    exponent reference, so the key and the query factor each span at most
    ``BC // 2`` tokens and the row factor is capped at ``PANEL_LOG_MAX``.  A
    single 64-token reference, or even a 16-token one anchored at the panel
    start, lets those exponentials run past the float32 range for strongly
    decaying gates.
    """
    i_t = tl.program_id(0) + I_T_LO
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_j = tl.arange(0, BT)
    o_t = i_t * BT + o_j
    o_r = tl.arange(0, BC)
    m_t = o_t < T
    base_k = i_b * T * H * K + i_h * K

    for i_p in tl.static_range(BT // BC):
        c0 = i_t * BT + i_p * BC
        if c0 < T:
            o_c = c0 + o_r
            m_c = o_c < T
            c_mid = tl.minimum(c0 + BC // 2, T - 1)
            # Rows above the panel are never needed for these columns; dropping
            # them from the load also keeps their exponent inside float32.
            m_q = m_t & (o_t >= c0)
            b_A = tl.zeros([BT, BC], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                o_k = i_k * BK + tl.arange(0, BK)
                m_k = o_k < K
                b_ref = tl.load(g_cum + base_k + c_mid * (H * K) + o_k, mask=m_k, other=0.0)
                b_q = tl.load(q + base_k + o_t[:, None] * (H * K) + o_k[None, :],
                              mask=m_q[:, None] & m_k[None, :], other=0.0)
                b_g = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :],
                              mask=m_q[:, None] & m_k[None, :], other=0.0)
                b_qg = b_q.to(tl.float32) * exp2(tl.minimum(b_g - b_ref[None, :], PANEL_LOG_MAX))
                b_kt = tl.load(k + base_k + o_c[None, :] * (H * K) + o_k[:, None],
                               mask=m_k[:, None] & m_c[None, :], other=0.0)
                b_gt = tl.load(g_cum + base_k + o_c[None, :] * (H * K) + o_k[:, None],
                               mask=m_k[:, None] & m_c[None, :], other=0.0)
                b_kg = b_kt.to(tl.float32) * exp2(b_ref[:, None] - b_gt)
                b_A = tl.dot(b_qg, b_kg, b_A)
            b_A = tl.where(o_t[:, None] >= o_c[None, :], b_A * scale, 0.0)
            p_A = A + (i_b * T + o_t)[:, None] * (H * BT) + i_h * BT + (o_c - i_t * BT)[None, :]
            tl.store(p_A, b_A, mask=m_t[:, None])


@triton.jit(do_not_specialize=["T", "NT", "I_T_LO"])
def chunk_gla_fwd_o_kernel(
    q,
    v,
    g_cum,
    h,
    A,
    o,
    scale,
    T,
    NT,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    """o = scale * (q * exp2(g_cum)) @ h_prev + A @ v."""
    i_v = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]

    base_q = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    o_h = ((i_b * NT + i_t) * H + i_h) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_q = tl.load(q + base_q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_g = tl.load(g_cum + base_q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        # Fold the scale into the operand: the accumulator stays untouched.
        b_qg = b_q * exp2(b_g) * scale
        if STATE_V_FIRST:
            b_h = tl.load(h + o_h + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                          mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_o = tl.dot(b_qg, tl.trans(b_h), b_o)
        else:
            b_h = tl.load(h + o_h + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                          mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            b_o = tl.dot(b_qg, b_h, b_o)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_A = tl.load(A + (i_b * T + o_t)[:, None] * (H * BT) + i_h * BT + tl.arange(0, BT)[None, :],
                  mask=m_t[:, None] & m_s, other=0.0)
    b_v = tl.load(v + base_v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
    # Accumulate the intra-chunk projection in float32.
    b_o = tl.dot(tl.where(m_s, b_A, 0.0), b_v.to(tl.float32), b_o)
    tl.store(o + base_v + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def _chunk_gla_A_panel(
    q, k, g_cum, scale, T, base_k, i_t, i_p,
    H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BC: tl.constexpr, BK: tl.constexpr,
):
    """One causal panel with the same midpoint reference as the standalone A kernel."""
    o_t = i_t * BT + tl.arange(0, BT)
    o_r = tl.arange(0, BC)
    m_t = o_t < T
    c0 = i_t * BT + i_p * BC
    o_c = c0 + o_r
    m_c = o_c < T
    c_mid = tl.minimum(c0 + BC // 2, T - 1)
    # Rows above the panel are never needed for these columns; dropping
    # them from the load also keeps their exponent inside float32.
    m_q = m_t & (o_t >= c0)
    b_A = tl.zeros([BT, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_ref = tl.load(g_cum + base_k + c_mid * (H * K) + o_k, mask=m_k, other=0.0)
        b_q = tl.load(q + base_k + o_t[:, None] * (H * K) + o_k[None, :],
                      mask=m_q[:, None] & m_k[None, :], other=0.0)
        b_g = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :],
                      mask=m_q[:, None] & m_k[None, :], other=0.0)
        b_qg = b_q.to(tl.float32) * exp2(tl.minimum(b_g - b_ref[None, :], PANEL_LOG_MAX))
        b_kt = tl.load(k + base_k + o_c[None, :] * (H * K) + o_k[:, None],
                       mask=m_k[:, None] & m_c[None, :], other=0.0)
        b_gt = tl.load(g_cum + base_k + o_c[None, :] * (H * K) + o_k[:, None],
                       mask=m_k[:, None] & m_c[None, :], other=0.0)
        b_kg = b_kt.to(tl.float32) * exp2(b_ref[:, None] - b_gt)
        b_A = tl.dot(b_qg, b_kg, b_A)
    b_A = tl.where(o_t[:, None] >= o_c[None, :], b_A * scale, 0.0)
    return b_A


@triton.jit
def _chunk_gla_wide_output(
    q,
    k,
    v,
    g_cum,
    A,
    scale,
    T,
    i_t,
    i_bh,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BKA: tl.constexpr,
    BV: tl.constexpr,
    SAVE_A: tl.constexpr,
):
    """Intra-chunk output under the whole-chunk gate-span guard."""
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    o_k = tl.arange(0, BKA)
    mk = o_k < K
    mt = m_t[:, None] & mk[None, :]
    b_ref = tl.load(g_cum + base_k + tl.minimum(i_t * BT + BT // 2, T - 1) * (H * K) + o_k, mask=mk, other=0.0)
    bg = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=mt, other=0.0)
    qq = tl.load(q + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=mt, other=0.0).to(tl.float32)
    kk = tl.load(k + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=mt, other=0.0).to(tl.float32)
    qg = qq * exp2(bg - b_ref[None, :])
    kg = kk * exp2(b_ref[None, :] - bg)
    b_A = tl.dot(qg, tl.trans(kg))
    rr = tl.arange(0, BT)
    b_A = tl.where(rr[:, None] >= rr[None, :], b_A * scale, 0.0)
    if SAVE_A:
        p_A = A + (i_b * T + o_t)[:, None] * (H * BT) + i_h * BT + rr[None, :]
        tl.store(p_A, b_A, mask=m_t[:, None])
    vv = tl.load(v + base_v + o_t[:, None] * (H * V) + o_v[None, :],
                 mask=m_tv, other=0.0).to(tl.float32)
    b_o = tl.dot(b_A, vv, b_o)
    return b_o


@triton.jit
def _chunk_gla_panel_output(
    q,
    k,
    v,
    g_cum,
    A,
    scale,
    T,
    i_t,
    i_bh,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BKA: tl.constexpr,
    BV: tl.constexpr,
    SAVE_A: tl.constexpr,
):
    """Intra-chunk output with safe column-panel exponent references."""
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_p in range(BT // BC):
        b_A = _chunk_gla_A_panel(q, k, g_cum, scale, T, base_k, i_t, i_p, H, K, BT, BC, BKA)
        o_c = i_t * BT + i_p * BC + tl.arange(0, BC)
        if SAVE_A:
            p_A = A + (i_b * T + o_t)[:, None] * (H * BT) + i_h * BT + (o_c - i_t * BT)[None, :]
            tl.store(p_A, b_A, mask=m_t[:, None])
        b_v = tl.load(v + base_v + o_c[:, None] * (H * V) + o_v[None, :],
                      mask=(o_c[:, None] < T) & m_v[None, :], other=0.0)
        b_o = tl.dot(b_A, b_v.to(tl.float32), b_o)
    return b_o


@triton.jit
def _chunk_gla_inter_output(
    b_o,
    q,
    g_cum,
    h,
    scale,
    T,
    NT,
    i_t,
    i_bh,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    """Add the inter-chunk state projection to the local output."""
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    base_k = i_b * T * H * K + i_h * K
    base_q = base_k
    base_v = i_b * T * H * V + i_h * V
    o_h = ((i_b * NT + i_t) * H + i_h) * K * V

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_q = tl.load(q + base_q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_g = tl.load(g_cum + base_q + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        # Fold the scale into the operand: the accumulator stays untouched.
        b_qg = b_q * exp2(b_g) * scale
        if STATE_V_FIRST:
            b_h = tl.load(h + o_h + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                          mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_o = tl.dot(b_qg, tl.trans(b_h), b_o)
        else:
            b_h = tl.load(h + o_h + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                          mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            b_o = tl.dot(b_qg, b_h, b_o)


    return b_o


@triton.jit(do_not_specialize=["T", "NT", "I_T_LO"])
def chunk_gla_fwd_Ao_kernel(
    q, k, v, g_cum, h, A, o, wide_ok, scale, T, NT, I_T_LO,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BKA: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    SAVE_A: tl.constexpr, STATE_V_FIRST: tl.constexpr,
):
    """Dispatch safe chunks to a wide dot and retain panels for larger spans.

    Each path ends at its own global store. CANN 9.1 fails to compile the
    shared-accumulator merge form, so keep the mixed-core output paths
    independent, including inference without A storage.
    """
    i_t = tl.program_id(0) + I_T_LO
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    base_v = i_b * T * H * V + i_h * V

    use_wide = tl.load(wide_ok + (i_b * NT + i_t) * H + i_h)
    if use_wide:
        b_o = _chunk_gla_wide_output(
            q, k, v, g_cum, A, scale, T, i_t, i_bh,
            H, K, V, BT, BKA, BV, SAVE_A,
        )
        b_o = _chunk_gla_inter_output(
            b_o, q, g_cum, h, scale, T, NT, i_t, i_bh,
            H, K, V, BT, BK, BV, STATE_V_FIRST,
        )
        tl.store(o + base_v + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)
        return
    b_o = _chunk_gla_panel_output(
        q, k, v, g_cum, A, scale, T, i_t, i_bh,
        H, K, V, BT, BC, BKA, BV, SAVE_A,
    )
    b_o = _chunk_gla_inter_output(
        b_o, q, g_cum, h, scale, T, NT, i_t, i_bh,
        H, K, V, BT, BK, BV, STATE_V_FIRST,
    )
    tl.store(o + base_v + o_t[:, None] * (H * V) + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)


def _use_fused_output(K: int, V: int, T: int, bt: int = BT) -> bool:
    # The fused kernel uses one 128-wide value tile and a 64-token chunk.
    return K == V == 128 and T >= 1024 and bt == BT


def chunk_gla_fwd_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int | None = None,
    return_intermediates: bool = False,
):
    """Chunked GLA forward.

    Shapes: ``q/k/g`` are ``[B, T, H, K]``, ``v`` and ``o`` are
    ``[B, T, H, V]``.  ``g`` holds the per-key log gate in natural log units.
    ``initial_state`` is float32 ``[N, H, K, V]`` (``[N, H, V, K]`` when
    ``state_v_first``).  Returns ``(o, ht)`` where ``ht`` is the float32 final
    state (``None`` unless ``output_final_state``).  With
    ``return_intermediates=True`` a third element ``dict`` carries ``g_cum``
    and ``A`` (float32) and ``h`` (input dtype). For K=V=128, T>=1024 and 64-token chunks,
    A is fused into the output kernel and is only stored when intermediates
    are requested for backward.

    Variable-length inputs are not supported yet: ``cu_seqlens`` (and the
    ``cu_seqlens_cpu`` twin kept for API compatibility) must be ``None``.
    """
    if cu_seqlens is not None or cu_seqlens_cpu is not None:
        raise NotImplementedError("chunk_gla_fwd_chunked does not support variable-length inputs yet.")
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape[:3] == q.shape[:3], "v must share batch, sequence and head dimensions."
    B, T, H, K = q.shape
    V = v.shape[-1]
    if K > MAX_HEAD_DIM or V > MAX_HEAD_DIM:
        raise ValueError(f"chunked GLA supports K,V <= {MAX_HEAD_DIM}, got K={K}, V={V}.")
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."
    if scale is None:
        scale = K ** -0.5
    # Chunk size is a multiple of the panel width, at most BT, and never below
    # it so that short sequences still use one full panel.
    bt = chunk_size if chunk_size is not None else min(BT, max(BC, triton.next_power_of_2(T)))
    bt = (bt + BC - 1) // BC * BC
    assert chunk_size is None or bt == chunk_size, f"chunk_size must be a multiple of {BC}, got {chunk_size}."
    assert BC <= bt <= BT, f"chunk_size must be in [{BC}, {BT}], got {bt}."
    nt = triton.cdiv(T, bt)

    bkc = min(BK, triton.next_power_of_2(K))
    bkw = min(BK_W, triton.next_power_of_2(K))
    state_shape = (V, K) if state_v_first else (K, V)
    # The per-chunk states are the dominant intermediate, so they are stored in
    # the input dtype; the kernels upcast them before the float32 dots.
    h = torch.empty((B, nt, H, *state_shape), device=q.device, dtype=q.dtype)
    g_cum = torch.empty((B, T, H, K), device=q.device, dtype=torch.float32)
    fused_output = _use_fused_output(K, V, T, bt)
    wide_ok = torch.empty((B, nt, H), device=q.device, dtype=torch.bool) if fused_output else None
    A = None
    if not fused_output or return_intermediates:
        A = torch.empty((B, T, H, bt), device=q.device, dtype=torch.float32)
    o = torch.empty_like(v)
    ht = None
    if output_final_state:
        ht = torch.empty((B, H, *state_shape), device=q.device, dtype=torch.float32)

    if fused_output:
        chunk_gla_fwd_cumsum_wide_kernel[(nt, B * H)](
            q=q, k=k, g=g, g_cum=g_cum, wide_ok=wide_ok, T=T,
            H=H, K=K, BT=bt, BK=bkw,
        )
    else:
        chunk_gla_fwd_cumsum_kernel[(nt, triton.cdiv(K, bkw), B * H)](
            g=g,
            g_cum=g_cum,
            T=T,
            H=H,
            K=K,
            BT=bt,
            BK=bkw,
        )


    bk, bv = _state_tiles(K, V)
    chunk_gla_fwd_h_kernel[(triton.cdiv(K, bk), triton.cdiv(V, bv), B * H)](
        k=k,
        v=v,
        g_cum=g_cum,
        h=h,
        h0=initial_state,
        ht=ht,
        T=T,
        NT=nt,
        H=H,
        K=K,
        V=V,
        BT=bt,
        BK=bk,
        BV=bv,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=ht is not None,
        STATE_V_FIRST=state_v_first,
    )

    if fused_output:
        cpb = _chunks_per_launch(nt * B * H, nt, FIXED_MAX_PROGRAMS)
        for i_lo in range(0, nt, cpb):
            chunk_gla_fwd_Ao_kernel[(min(cpb, nt - i_lo), B * H)](
                q=q, k=k, v=v, g_cum=g_cum, h=h, A=A, o=o, wide_ok=wide_ok,
                scale=scale, T=T, NT=nt, I_T_LO=i_lo,
                H=H, K=K, V=V, BT=bt, BC=BC, BKA=128, BK=bkc, BV=128,
                SAVE_A=return_intermediates, STATE_V_FIRST=state_v_first,
            )
    else:
        cpb = _chunks_per_launch(nt * B * H, nt, FIXED_MAX_PROGRAMS)
        for i_lo in range(0, nt, cpb):
            chunk_gla_fwd_A_kernel[(min(cpb, nt - i_lo), B * H)](
                q=q,
                k=k,
                g_cum=g_cum,
                A=A,
                scale=scale,
                T=T,
                I_T_LO=i_lo,
                H=H,
                K=K,
                BT=bt,
                BC=BC,
                BK=bkw,
            )

        bov = min(BV, triton.next_power_of_2(V))
        n_vo = triton.cdiv(V, bov)
        cpb = _chunks_per_launch(n_vo * nt * B * H, nt, FIXED_MAX_PROGRAMS)
        for i_lo in range(0, nt, cpb):
            chunk_gla_fwd_o_kernel[(n_vo, min(cpb, nt - i_lo), B * H)](
                q=q,
                v=v,
                g_cum=g_cum,
                h=h,
                A=A,
                o=o,
                scale=scale,
                T=T,
                NT=nt,
                I_T_LO=i_lo,
                H=H,
                K=K,
                V=V,
                BT=bt,
                BK=bkc,
                BV=bov,
                STATE_V_FIRST=state_v_first,
            )

    if return_intermediates:
        return o, ht, {"g_cum": g_cum, "h": h, "A": A}
    return o, ht
