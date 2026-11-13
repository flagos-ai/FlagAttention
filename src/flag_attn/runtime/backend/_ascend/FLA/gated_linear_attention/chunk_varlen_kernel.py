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

"""Packed variable-length chunked GLA kernels for Ascend.

Inputs are [1,T,H,*], with one state per sequence. Kernels read bos/eos from
cu_seqlens and derive compact state slots by summing preceding chunk counts;
no host chunk_indices tensor is built. State scans stay within each sequence,
and empty sequences preserve their initial state and its gradient.

Forward has three kernel stages: gate cumsum, state scan, and fused A/output.
Backward has four: dh, fused dA/intra dqk, inter dqkg, and dv. A and dA are
not stored in GM; dv rebuilds A. Grouped launches bound runtime scratch.
Dots and running states use FP32; stored chunk states use the input dtype.

Short batches use padded grids and an upper bound on state slots. CPU
boundaries, when supplied, give exact sizes; otherwise T>=4096 triggers one
host boundary read. Backward reuses the forward plan without another read.
Programs outside their sequence return early, though highly uneven batches
can still create many padded programs.

Gate factors use a whole-chunk midpoint reference and exponent caps;
the host-split path uses BC=16 panel references.

Separate state-dependent value loops avoid state-tile clobbering. Loading v
in transposed order directly avoids transposing a loaded tile on this backend. K/V<=512; the public autograd entry supports only [K,V]
state layout, while the low-level forward also accepts state_v_first."""

import torch
import triton
import triton.language as tl

from .chunk import (
    BC,
    BT,
    MAX_HEAD_DIM,
    PANEL_LOG_MAX,
    RCP_LN2,
    STATE_BK,
    STATE_BV,
    _chunks_per_launch,
    exp2,
)

__all__ = [
    "chunk_gla_varlen_kernel",
    "chunk_gla_varlen_kernel_fwd",
    "chunk_gla_varlen_kernel_bwd",
]

# Tiles of the fused output / intra-chunk kernels.  Narrower than the fixed
# length ones: ``A`` (or ``dA``) and the ``dq``/``dk`` accumulators are live at
# the same time as the ``v``/``do``/``h``/``dh`` tiles.
FWD_BK = 64
FWD_BV = 128
BWD_BK = 64
BWD_BV = 128
BWD_BV_W = 128
# Same cap as ``chunk.PANEL_LOG_MAX``; see the module docstring.
LOG_CAP = PANEL_LOG_MAX
# The sequential scans keep the cumsum tile, the k tile and the v tile live next
# to the running state; cap the state tile area (elements) so the widest tile
# combination still fits the 192 KiB unified buffer.
STATE_AREA = 16384
# Read host boundaries only for long batches to size grids and state arrays
# exactly. Short batches use upper bounds without device-to-host synchronization;
# kernels read their sequence boundaries directly on device.
HOST_READ_MIN_TOKENS = 4096


def _varlen_state_tiles(K: int, V: int) -> tuple[int, int]:
    """State tile of the sequential scans, clamped to the head dims and to ``STATE_AREA``."""
    bk = min(STATE_BK, triton.next_power_of_2(K))
    bv = min(STATE_BV, triton.next_power_of_2(V))
    while bk * bv > STATE_AREA:
        if bv >= bk:
            bv //= 2
        else:
            bk //= 2
    return max(bk, 16), max(bv, 16)


@triton.jit
def _chunk_offset(cu_seqlens, i_n, BT: tl.constexpr):
    """Global chunk slot of the first chunk of sequence ``i_n``.

    ``sum_{m < i_n} cdiv(len_m, BT)``, read from ``cu_seqlens`` only, so the
    kernels need no host-built chunk index.  The loop is short (``i_n`` is at
    most the number of packed sequences).
    """
    off = 0
    for m in range(i_n):
        bos = tl.load(cu_seqlens + m).to(tl.int32)
        eos = tl.load(cu_seqlens + m + 1).to(tl.int32)
        off += (eos - bos + BT - 1) // BT
    return off


@triton.jit
def _seq_bounds(cu_seqlens, i_n, T):
    """Inclusive start / exclusive (clamped) end token of sequence ``i_n``."""
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.minimum(tl.load(cu_seqlens + i_n + 1).to(tl.int32), T)
    return bos, eos


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_varlen_fwd_cumsum_kernel(
    g,
    g_cum,
    cu_seqlens,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """Per-chunk inclusive gate cumsum, in base-2 log units, over the packed batch."""
    i_t = tl.program_id(0) + I_T_LO
    i_k = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_n = i_bh // H
    i_h = i_bh % H

    bos, eos = _seq_bounds(cu_seqlens, i_n, T)
    start = bos + i_t * BT
    if start >= eos:
        return

    o_t = start + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_t = o_t < eos
    m_tk = m_t[:, None] & m_k[None, :]

    base = i_h * K
    b_g = tl.load(g + o_t[:, None] * (H * K) + base + o_k[None, :], mask=m_tk, other=0.0)
    b_g = tl.cumsum(b_g.to(tl.float32) * RCP_LN2, axis=0)
    tl.store(g_cum + o_t[:, None] * (H * K) + base + o_k[None, :], b_g, mask=m_tk)


@triton.jit(do_not_specialize=["T", "N"])
def chunk_gla_varlen_fwd_h_kernel(
    k,
    v,
    g_cum,
    h,
    h0,
    ht,
    cu_seqlens,
    T,
    N,
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
    """Sequential inter-chunk state scan over the packed batch.

    One program per state tile; it walks every sequence in order and stores the
    state entering each chunk.  The chunk slot advances with the same
    ``cdiv(len_n, BT)`` rule as ``_chunk_offset``, so it matches the slot the
    chunk-parallel kernels compute independently.
    """
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_h = tl.program_id(2)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    o_i64 = o_k[:, None].to(tl.int64)

    off = 0
    for i_n in range(N):
        bos, eos = _seq_bounds(cu_seqlens, i_n, T)
        nt = (eos - bos + BT - 1) // BT

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            p_h0 = h0 + (i_n * H + i_h) * K * V
            if STATE_V_FIRST:
                # [V, K] tile, materialised before the transpose (see the module
                # docstring on transposing loaded tiles).
                b_h = tl.trans(
                    tl.load(p_h0 + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                            mask=m_v[:, None] & m_k[None, :], other=0.0) * 1.0
                ).to(tl.float32)
            else:
                b_h = tl.load(p_h0 + o_i64 * V + o_v[None, :], mask=m_h, other=0.0).to(tl.float32)

        base_k = i_h * K
        base_v = i_h * V
        for i_t in range(nt):
            o_t = bos + i_t * BT + tl.arange(0, BT)
            m_t = o_t < eos
            m_kt = m_t[None, :] & m_k[:, None]
            m_tv = m_t[:, None] & m_v[None, :]

            o_h = ((off + i_t).to(tl.int64) * H + i_h) * K * V
            # The per-chunk states always use the internal [K, V] layout, whatever
            # the caller's state layout is: only ``h0``/``ht`` are transposed.
            tl.store(h + o_h + o_i64 * V + o_v[None, :], b_h.to(h.dtype.element_ty), mask=m_h)

            # [BK, BT] orientation: the state update contracts the tokens.
            last = tl.minimum(bos + i_t * BT + BT - 1, eos - 1)
            b_gl = tl.load(g_cum + last * (H * K) + base_k + o_k, mask=m_k, other=0.0)
            b_cg = tl.load(g_cum + o_t[None, :] * (H * K) + base_k + o_k[:, None], mask=m_kt, other=0.0)
            b_k = tl.load(k + o_t[None, :] * (H * K) + base_k + o_k[:, None], mask=m_kt, other=0.0)
            b_v = tl.load(v + o_t[:, None] * (H * V) + base_v + o_v[None, :], mask=m_tv, other=0.0)

            b_h = b_h * exp2(b_gl)[:, None]
            b_h = tl.dot(b_k.to(tl.float32) * exp2(b_gl[:, None] - b_cg), b_v.to(tl.float32), b_h)

        if STORE_FINAL_STATE:
            p_ht = ht + (i_n * H + i_h) * K * V
            if STATE_V_FIRST:
                tl.store(
                    p_ht + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                    tl.trans(b_h),
                    mask=m_v[:, None] & m_k[None, :],
                )
            else:
                tl.store(p_ht + o_i64 * V + o_v[None, :], b_h, mask=m_h)

        off += nt


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_varlen_fwd_o_kernel(
    q,
    k,
    v,
    g_cum,
    h,
    o,
    scale,
    cu_seqlens,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Intra-chunk matrix ``A`` plus output ``o`` for one chunk of one sequence.

    One program per (chunk, head): ``A`` is built in registers over the head
    dimension (one reference per key tile, see ``LOG_CAP``), then the value loop
    adds ``q_g @ h + A @ v``.  ``A`` never reaches global memory.
    """
    i_t = tl.program_id(0) + I_T_LO
    i_bh = tl.program_id(1)
    i_n = i_bh // H
    i_h = i_bh % H

    bos, eos = _seq_bounds(cu_seqlens, i_n, T)
    start = bos + i_t * BT
    if start >= eos:
        return

    o_t = start + tl.arange(0, BT)
    m_t = o_t < eos
    r = tl.arange(0, BT)
    m_causal = r[:, None] >= r[None, :]

    base_k = i_h * K
    base_v = i_h * V
    c_mid = tl.minimum(start + BT // 2, eos - 1)

    # --- intra-chunk matrix, [BT, BT] in float32 registers ---
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_ref = tl.load(g_cum + c_mid * (H * K) + base_k + o_k, mask=m_k, other=0.0)
        b_cg = tl.load(g_cum + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_row = exp2(tl.minimum(b_cg - b_ref[None, :], LOG_CAP))
        b_col = exp2(tl.minimum(b_ref[None, :] - b_cg, LOG_CAP))
        b_q = tl.load(q + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_k = tl.load(k + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_A = tl.dot(b_q.to(tl.float32) * b_row, tl.trans(b_k.to(tl.float32) * b_col), b_A)

    b_A = tl.where(m_causal, b_A * scale, 0.0)

    o_state = ((_chunk_offset(cu_seqlens, i_n, BT) + i_t) * H + i_h) * K * V
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_tk = m_t[:, None] & m_k[None, :]
            b_cg = tl.load(g_cum + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
            b_q = tl.load(q + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
            b_h = tl.load(
                h + o_state + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                mask=m_k[:, None] & m_v[None, :],
                other=0.0,
            ).to(tl.float32)
            b_o = tl.dot(b_q.to(tl.float32) * exp2(b_cg) * scale, b_h, b_o)
        b_v = tl.load(v + o_t[:, None] * (H * V) + base_v + o_v[None, :], mask=m_tv, other=0.0)
        b_o = tl.dot(b_A, b_v.to(tl.float32), b_o)
        tl.store(o + o_t[:, None] * (H * V) + base_v + o_v[None, :], b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit(do_not_specialize=["T", "N"])
def chunk_gla_varlen_bwd_dh_kernel(
    q,
    do,
    g_cum,
    dh,
    dh0,
    dht,
    cu_seqlens,
    scale,
    T,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
):
    """Reverse inter-chunk state scan over the packed batch (mirror of ``fwd_h``)."""
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_h = tl.program_id(2)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    o_i64 = o_k[:, None].to(tl.int64)

    # Total number of chunks of the packed batch; the reverse walk subtracts one
    # sequence at a time so ``off + i_t`` is the compact chunk slot.
    off = _chunk_offset(cu_seqlens, N, BT)
    for i_n in range(N - 1, -1, -1):
        bos, eos = _seq_bounds(cu_seqlens, i_n, T)
        nt = (eos - bos + BT - 1) // BT
        off -= nt

        b_dh = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_FINAL_STATE_GRADIENT:
            b_dh = tl.load(
                dht + (i_n * H + i_h) * K * V + o_i64 * V + o_v[None, :], mask=m_h, other=0.0
            ).to(tl.float32)

        base_k = i_h * K
        base_v = i_h * V
        for i_t in range(nt - 1, -1, -1):
            o_t = bos + i_t * BT + tl.arange(0, BT)
            m_t = o_t < eos
            m_kt = m_t[None, :] & m_k[:, None]
            m_tv = m_t[:, None] & m_v[None, :]

            o_h = ((off + i_t).to(tl.int64) * H + i_h) * K * V
            tl.store(dh + o_h + o_i64 * V + o_v[None, :], b_dh.to(dh.dtype.element_ty), mask=m_h)

            last = tl.minimum(bos + i_t * BT + BT - 1, eos - 1)
            b_gl = tl.load(g_cum + last * (H * K) + base_k + o_k, mask=m_k, other=0.0)
            b_cg = tl.load(g_cum + o_t[None, :] * (H * K) + base_k + o_k[:, None], mask=m_kt, other=0.0)
            b_q = tl.load(q + o_t[None, :] * (H * K) + base_k + o_k[:, None], mask=m_kt, other=0.0)
            b_do = tl.load(do + o_t[:, None] * (H * V) + base_v + o_v[None, :], mask=m_tv, other=0.0)

            b_dh = b_dh * exp2(b_gl)[:, None]
            b_dh = tl.dot(b_q.to(tl.float32) * exp2(b_cg) * scale, b_do.to(tl.float32), b_dh)

        if STORE_INITIAL_STATE_GRADIENT:
            tl.store(dh0 + (i_n * H + i_h) * K * V + o_i64 * V + o_v[None, :], b_dh, mask=m_h)


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_varlen_bwd_dqk_intra_kernel(
    q,
    k,
    v,
    g_cum,
    do,
    dq,
    dk,
    scale,
    cu_seqlens,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Intra-chunk ``dq`` / ``dk`` for one chunk / head tile.

    ``dA = tril(do @ v^T) * scale`` is accumulated over the value dimension and
    kept in registers, so the intra-chunk matmul never lands in global memory.
    This kernel writes the intra part only: ``chunk_gla_varlen_bwd_dqkg_kernel``
    adds the inter-chunk state terms on top.
    """
    i_k = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2)
    i_n = i_bh // H
    i_h = i_bh % H

    bos, eos = _seq_bounds(cu_seqlens, i_n, T)
    start = bos + i_t * BT
    if start >= eos:
        return

    o_t = start + tl.arange(0, BT)
    m_t = o_t < eos
    r = tl.arange(0, BT)
    m_causal = r[:, None] >= r[None, :]

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    base_k = i_h * K
    base_v = i_h * V

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_vt = m_t[:, None] & m_v[None, :]
        b_do = tl.load(do + o_t[:, None] * (H * V) + base_v + o_v[None, :], mask=m_vt, other=0.0)
        # ``v`` is read transposed ([BV, BT]) straight from global memory: the
        # dot contracts on the value dimension and no loaded tile is transposed
        # in registers (see the module docstring).
        b_vt = tl.load(v + o_v[:, None] + o_t[None, :] * (H * V) + base_v,
                       mask=m_v[:, None] & m_t[None, :], other=0.0)
        b_dA = tl.dot(b_do.to(tl.float32), b_vt.to(tl.float32), b_dA)

    b_dA = tl.where(m_causal, b_dA * scale, 0.0)

    c_mid = tl.minimum(start + BT // 2, eos - 1)
    b_ref = tl.load(g_cum + c_mid * (H * K) + base_k + o_k, mask=m_k, other=0.0)
    b_cg = tl.load(g_cum + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
    b_row = exp2(tl.minimum(b_cg - b_ref[None, :], LOG_CAP))
    b_col = exp2(tl.minimum(b_ref[None, :] - b_cg, LOG_CAP))
    b_q = tl.load(q + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
    b_k = tl.load(k + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)

    b_dq = tl.dot(b_dA, b_k.to(tl.float32) * b_col) * b_row
    b_dk = tl.dot(tl.trans(b_dA), b_q.to(tl.float32) * b_row) * b_col

    p_dq = dq + o_t[:, None] * (H * K) + base_k + o_k[None, :]
    p_dk = dk + o_t[:, None] * (H * K) + base_k + o_k[None, :]
    tl.store(p_dq, b_dq, mask=m_tk)
    tl.store(p_dk, b_dk, mask=m_tk)


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_varlen_bwd_dqkg_kernel(
    q,
    k,
    v,
    g_cum,
    h,
    dh,
    do,
    dq,
    dk,
    dg,
    scale,
    cu_seqlens,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Add inter-chunk dq/dk to the intra gradients and compute dg.

    Separate value loops keep transposed h/dh tiles out of each other's dot
    lifetimes, avoiding clobbered operands on this backend."""
    i_k = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2)
    i_n = i_bh // H
    i_h = i_bh % H

    bos, eos = _seq_bounds(cu_seqlens, i_n, T)
    start = bos + i_t * BT
    if start >= eos:
        return

    o_t = start + tl.arange(0, BT)
    m_t = o_t < eos

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]
    base_k = i_h * K
    base_v = i_h * V
    o_state = ((_chunk_offset(cu_seqlens, i_n, BT) + i_t) * H + i_h) * K * V

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_do = tl.load(do + o_t[:, None] * (H * V) + base_v + o_v[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_h = tl.load(
            h + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
            mask=m_v[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dq = tl.dot(b_do.to(tl.float32), b_h, b_dq)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_v = tl.load(v + o_t[:, None] * (H * V) + base_v + o_v[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0)
        b_dh = tl.load(
            dh + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
            mask=m_v[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dk = tl.dot(b_v.to(tl.float32), b_dh, b_dk)

    # dL/dg_last of the state decay: h_next = exp2(g_last) * h.
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_h = tl.load(
            h + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
            mask=m_v[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dh = tl.load(
            dh + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
            mask=m_v[:, None] & m_k[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dgk += tl.sum(b_h * b_dh, axis=0)

    last = tl.minimum(start + BT - 1, eos - 1)
    b_gl = tl.load(g_cum + last * (H * K) + base_k + o_k, mask=m_k, other=0.0)
    b_cg = tl.load(g_cum + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
    b_q = tl.load(q + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
    b_k = tl.load(k + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
    p_dq = dq + o_t[:, None] * (H * K) + base_k + o_k[None, :]
    p_dk = dk + o_t[:, None] * (H * K) + base_k + o_k[None, :]

    b_dgk = b_dgk * exp2(b_gl)
    b_dq = b_dq * scale * exp2(b_cg) + tl.load(p_dq, mask=m_tk, other=0.0)
    b_dk = b_dk * exp2(b_gl[None, :] - b_cg)
    # The k_e factor carries g_last; only the inter-chunk term contributes here.
    b_dgk += tl.sum(b_dk * b_k.to(tl.float32), axis=0)
    b_dk += tl.load(p_dk, mask=m_tk, other=0.0)

    b_dg = b_q.to(tl.float32) * b_dq - b_k.to(tl.float32) * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]

    p_dg = dg + o_t[:, None] * (H * K) + base_k + o_k[None, :]
    tl.store(p_dq, b_dq, mask=m_tk)
    tl.store(p_dk, b_dk, mask=m_tk)
    tl.store(p_dg, b_dg.to(dg.dtype.element_ty), mask=m_tk)


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_varlen_bwd_dv_kernel(
    q,
    k,
    v,
    g_cum,
    dh,
    do,
    dv,
    scale,
    cu_seqlens,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """``dv = A^T @ do + k_e @ dh`` for one chunk / head / value tile.

    ``A`` is rebuilt from ``q``/``k``/``g_cum`` (the forward does not store it).
    """
    i_v = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2)
    i_n = i_bh // H
    i_h = i_bh % H

    bos, eos = _seq_bounds(cu_seqlens, i_n, T)
    start = bos + i_t * BT
    if start >= eos:
        return

    o_t = start + tl.arange(0, BT)
    m_t = o_t < eos
    r = tl.arange(0, BT)
    m_causal = r[:, None] >= r[None, :]

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]

    base_k = i_h * K
    base_v = i_h * V
    o_state = ((_chunk_offset(cu_seqlens, i_n, BT) + i_t) * H + i_h) * K * V
    c_mid = tl.minimum(start + BT // 2, eos - 1)

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_ref = tl.load(g_cum + c_mid * (H * K) + base_k + o_k, mask=m_k, other=0.0)
        b_cg = tl.load(g_cum + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_row = exp2(tl.minimum(b_cg - b_ref[None, :], LOG_CAP))
        b_col = exp2(tl.minimum(b_ref[None, :] - b_cg, LOG_CAP))
        b_q = tl.load(q + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_k = tl.load(k + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_A = tl.dot(b_q.to(tl.float32) * b_row, tl.trans(b_k.to(tl.float32) * b_col), b_A)

    b_A = tl.where(m_causal, b_A * scale, 0.0)
    b_do = tl.load(do + o_t[:, None] * (H * V) + base_v + o_v[None, :], mask=m_tv, other=0.0)
    b_dv = tl.dot(tl.trans(b_A), b_do.to(tl.float32))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        last = tl.minimum(start + BT - 1, eos - 1)
        b_gl = tl.load(g_cum + last * (H * K) + base_k + o_k, mask=m_k, other=0.0)
        b_cg = tl.load(g_cum + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_k = tl.load(k + o_t[:, None] * (H * K) + base_k + o_k[None, :], mask=m_tk, other=0.0)
        b_dh = tl.load(
            dh + o_state + o_k[:, None].to(tl.int64) * V + o_v[None, :],
            mask=m_k[:, None] & m_v[None, :],
            other=0.0,
        ).to(tl.float32)
        b_dv = tl.dot(b_k.to(tl.float32) * exp2(b_gl[None, :] - b_cg), b_dh, b_dv)

    tl.store(dv + o_t[:, None] * (H * V) + base_v + o_v[None, :], b_dv, mask=m_tv)


def _prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens,
    cu_seqlens_cpu,
    chunk_size: int | None,
    plan: dict | None = None,
) -> tuple[torch.Tensor, int, int, int, int, int, int, int, int]:
    """Validate the packed call and derive every host-side constant.

    Returns ``(cu_seqlens, T, N, H, K, V, bt, max_nt, nt_upper)``, with
    ``cu_seqlens`` the device tensor the kernels read.  ``nt_upper`` bounds the
    number of chunks of the packed batch, so the state arrays need no host read
    of ``cu_seqlens``.
    """
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape[:3] == q.shape[:3], "v must share batch, sequence and head dimensions."
    B, T, H, K = q.shape
    V = v.shape[-1]
    if B != 1:
        raise ValueError(
            f"the batch size is expected to be 1 when using cu_seqlens, got {B}; "
            "flatten variable-length inputs before processing."
        )
    if K > MAX_HEAD_DIM or V > MAX_HEAD_DIM:
        raise ValueError(f"chunked GLA supports K,V <= {MAX_HEAD_DIM}, got K={K}, V={V}.")
    if cu_seqlens is None:
        if cu_seqlens_cpu is None:
            raise ValueError(
                "the variable-length chunked GLA needs cu_seqlens (or cu_seqlens_cpu)."
            )
        # Only the CPU copy was supplied: the kernels read cu_seqlens on device.
        cu_seqlens = cu_seqlens_cpu.to(q.device)
    elif not isinstance(cu_seqlens, torch.Tensor):
        # Convert list boundaries to a device tensor for kernel indexing.
        cu_seqlens = torch.tensor(list(cu_seqlens), dtype=torch.int32, device=q.device)
    elif cu_seqlens.device != q.device:
        cu_seqlens = cu_seqlens.to(q.device)
    N = cu_seqlens.shape[0] - 1
    if N <= 0:
        raise ValueError(
            f"cu_seqlens must hold at least two entries, got {cu_seqlens.shape[0]}."
        )
    if chunk_size is not None:
        if chunk_size % BC or not (BC <= chunk_size <= BT):
            raise ValueError(
                f"chunk_size must be a multiple of {BC} in [{BC}, {BT}], got {chunk_size}."
            )
        bt = chunk_size
    else:
        # Same rule as the fixed-length entry: the packed batch is chunked at one
        # size, never narrower than one panel.
        bt = min(BT, max(BC, triton.next_power_of_2(T)))

    if plan is not None:
        # The backward of a forward call reuses the plan, so no second host read.
        return cu_seqlens, T, N, H, K, V, bt, plan["max_nt"], plan["nt_upper"]

    bounds = None
    if cu_seqlens_cpu is not None:
        bounds = cu_seqlens_cpu
    elif T >= HOST_READ_MIN_TOKENS:
        bounds = cu_seqlens
    if bounds is not None:
        values = [int(v) for v in (bounds.tolist() if isinstance(bounds, torch.Tensor) else bounds)]
        if len(values) < 2:
            raise ValueError(f"cu_seqlens must hold at least two entries, got {len(values)}.")
        if values[0] != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {values[0]}.")
        if any(b < a for a, b in zip(values, values[1:])):
            raise ValueError(f"cu_seqlens must be non-decreasing, got {values}.")
        if values[-1] != T:
            raise ValueError(
                f"the packed sequence length T={T} must equal cu_seqlens[-1]={values[-1]}."
            )
        # Exact grid bound and an exact (not padded) state-array size.
        max_nt = max(1, max((e - s + bt - 1) // bt for s, e in zip(values, values[1:])))
        nt_upper = max(1, sum((e - s + bt - 1) // bt for s, e in zip(values, values[1:])))
        return cu_seqlens, T, N, H, K, V, bt, max_nt, nt_upper

    # Bound each sequence by the packed length without reading boundaries
    # on the host. This pads the grid when segments are shorter than T.
    max_nt = triton.cdiv(T, bt)
    return cu_seqlens, T, N, H, K, V, bt, max_nt, max_nt + N


def chunk_gla_varlen_kernel_fwd(
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
    """Chunked GLA forward for a packed variable-length batch, one launch group.

    ``q/k/g`` are ``[1, T, H, K]``, ``v``/``o`` are ``[1, T, H, V]`` and
    ``cu_seqlens`` (``N + 1`` entries, on the device) delimits the sequences.
    ``initial_state`` is float32 ``[N, H, K, V]`` (``[N, H, V, K]`` for
    ``state_v_first``).  Returns ``(o, ht)``; ``ht`` is float32 and ``None``
    unless ``output_final_state``.  With ``return_intermediates=True`` the third
    element is the dict consumed by :func:`chunk_gla_varlen_kernel_bwd`.
    """
    cu_seqlens, T, N, H, K, V, bt, max_nt, nt_upper = _prepare(
        q, k, v, g, cu_seqlens, cu_seqlens_cpu, chunk_size
    )
    if scale is None:
        scale = K ** -0.5

    state_shape = (V, K) if state_v_first else (K, V)
    h = torch.empty((nt_upper, H, K, V), device=q.device, dtype=q.dtype)
    g_cum = torch.empty((1, T, H, K), device=q.device, dtype=torch.float32)
    o = torch.empty_like(v)
    ht = None
    if output_final_state:
        ht = torch.empty((N, H, *state_shape), device=q.device, dtype=torch.float32)

    bkw = min(128, triton.next_power_of_2(K))
    cpb = _chunks_per_launch(triton.cdiv(K, bkw) * max_nt * N * H, max_nt)
    for i_lo in range(0, max_nt, cpb):
        n = min(cpb, max_nt - i_lo)
        chunk_gla_varlen_fwd_cumsum_kernel[(n, triton.cdiv(K, bkw), N * H)](
            g=g,
            g_cum=g_cum,
            cu_seqlens=cu_seqlens,
            T=T,
            I_T_LO=i_lo,
            H=H,
            K=K,
            BT=bt,
            BK=bkw,
        )

    bk, bv = _varlen_state_tiles(K, V)
    chunk_gla_varlen_fwd_h_kernel[(triton.cdiv(K, bk), triton.cdiv(V, bv), H)](
        k=k,
        v=v,
        g_cum=g_cum,
        h=h,
        h0=initial_state,
        ht=ht,
        cu_seqlens=cu_seqlens,
        T=T,
        N=N,
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

    bkc = min(FWD_BK, triton.next_power_of_2(K))
    bov = min(FWD_BV, triton.next_power_of_2(V))
    cpb = _chunks_per_launch(max_nt * N * H, max_nt)
    for i_lo in range(0, max_nt, cpb):
        chunk_gla_varlen_fwd_o_kernel[(min(cpb, max_nt - i_lo), N * H)](
            q=q,
            k=k,
            v=v,
            g_cum=g_cum,
            h=h,
            o=o,
            scale=scale,
            cu_seqlens=cu_seqlens,
            T=T,
            I_T_LO=i_lo,
            H=H,
            K=K,
            V=V,
            BT=bt,
            BK=bkc,
            BV=bov,
        )

    if return_intermediates:
        return o, ht, {
            "h": h,
            "g_cum": g_cum,
            "bt": bt,
            "cu_seqlens": cu_seqlens,
            "max_nt": max_nt,
            "nt_upper": nt_upper,
        }
    return o, ht


def chunk_gla_varlen_kernel_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor | None,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int | None = None,
    parts: dict | None = None,
):
    """Chunked GLA backward for a packed variable-length batch.

    ``do`` is the output gradient (``None`` means zeros), ``dht`` the gradient of
    the final state, ``[N, H, K, V]``.  ``parts`` is the forward dictionary with
    ``return_intermediates=True``; when it is ``None`` the forward is re-run to
    rebuild the per-chunk states.  Returns ``(dq, dk, dv, dg, dh0)`` exactly as
    ``chunk_bwd.chunk_gla_bwd_chunked`` does (float32 ``dq``/``dk``/``dh0``).
    """
    if parts is None:
        _, _, parts = chunk_gla_varlen_kernel_fwd(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=dht is not None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            return_intermediates=True,
        )
    if cu_seqlens is None:
        cu_seqlens = parts["cu_seqlens"]

    cu_seqlens, T, N, H, K, V, bt, max_nt, _ = _prepare(
        q, k, v, g, cu_seqlens, cu_seqlens_cpu, chunk_size, plan=parts
    )
    h = parts["h"]
    g_cum = parts["g_cum"]
    if do is None:
        do = torch.zeros_like(v)
    elif not do.is_contiguous():
        do = do.contiguous()
    if dht is not None and not dht.is_contiguous():
        dht = dht.contiguous()

    bsk, bsv = _varlen_state_tiles(K, V)
    dh = torch.empty_like(h)
    dh0 = None
    if initial_state is not None:
        dh0 = torch.empty((N, H, K, V), device=q.device, dtype=torch.float32)
    dq = torch.empty((1, T, H, K), device=q.device, dtype=torch.float32)
    dk = torch.empty((1, T, H, K), device=q.device, dtype=torch.float32)
    dv = torch.empty((1, T, H, V), device=q.device, dtype=v.dtype)
    dg = torch.empty((1, T, H, K), device=q.device, dtype=g.dtype)

    chunk_gla_varlen_bwd_dh_kernel[(triton.cdiv(K, bsk), triton.cdiv(V, bsv), H)](
        q=q,
        do=do,
        g_cum=g_cum,
        dh=dh,
        dh0=dh0,
        dht=dht,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        N=N,
        H=H,
        K=K,
        V=V,
        BT=bt,
        BK=bsk,
        BV=bsv,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        STORE_INITIAL_STATE_GRADIENT=dh0 is not None,
    )

    bkc = min(BWD_BK, triton.next_power_of_2(K))
    nk = triton.cdiv(K, bkc)
    bv = min(BWD_BV, triton.next_power_of_2(V))
    cpb = _chunks_per_launch(nk * max_nt * N * H, max_nt)
    for i_lo in range(0, max_nt, cpb):
        n = min(cpb, max_nt - i_lo)
        chunk_gla_varlen_bwd_dqk_intra_kernel[(nk, n, N * H)](
            q=q,
            k=k,
            v=v,
            g_cum=g_cum,
            do=do,
            dq=dq,
            dk=dk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            T=T,
            I_T_LO=i_lo,
            H=H,
            K=K,
            V=V,
            BT=bt,
            BK=bkc,
            BV=bv,
        )
    for i_lo in range(0, max_nt, cpb):
        n = min(cpb, max_nt - i_lo)
        chunk_gla_varlen_bwd_dqkg_kernel[(nk, n, N * H)](
            q=q,
            k=k,
            v=v,
            g_cum=g_cum,
            h=h,
            dh=dh,
            do=do,
            dq=dq,
            dk=dk,
            dg=dg,
            scale=scale,
            cu_seqlens=cu_seqlens,
            T=T,
            I_T_LO=i_lo,
            H=H,
            K=K,
            V=V,
            BT=bt,
            BK=bkc,
            BV=bv,
        )

    bvw = min(BWD_BV_W, triton.next_power_of_2(V))
    n_vd = triton.cdiv(V, bvw)
    cpb = _chunks_per_launch(n_vd * max_nt * N * H, max_nt)
    for i_lo in range(0, max_nt, cpb):
        n = min(cpb, max_nt - i_lo)
        chunk_gla_varlen_bwd_dv_kernel[(n_vd, n, N * H)](
            q=q,
            k=k,
            v=v,
            g_cum=g_cum,
            dh=dh,
            do=do,
            dv=dv,
            scale=scale,
            cu_seqlens=cu_seqlens,
            T=T,
            I_T_LO=i_lo,
            H=H,
            K=K,
            V=V,
            BT=bt,
            BK=bkc,
            BV=bvw,
        )

    return dq, dk, dv, dg, dh0


def _contiguous(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return tensors with contiguous packed-row strides.

    Already contiguous tensors pass through unchanged; kernels require the
    H*K and H*V row strides."""
    return tuple(t if t.is_contiguous() else t.contiguous() for t in tensors)


class _ChunkGLAVarlenKernelFunction(torch.autograd.Function):
    """Autograd wrapper around the packed-batch chunked forward/backward."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        chunk_size,
        cu_seqlens,
    ):
        o, ht, parts = chunk_gla_varlen_kernel_fwd(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            return_intermediates=True,
        )
        ctx.save_for_backward(q, k, v, g, initial_state, parts["h"], parts["g_cum"])
        ctx.bt = parts["bt"]
        ctx.max_nt = parts["max_nt"]
        ctx.nt_upper = parts["nt_upper"]
        ctx.cu_seqlens = parts["cu_seqlens"]
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o, ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, initial_state, h, g_cum = ctx.saved_tensors
        parts = {
            "h": h,
            "g_cum": g_cum,
            "bt": ctx.bt,
            "cu_seqlens": ctx.cu_seqlens,
            "max_nt": ctx.max_nt,
            "nt_upper": ctx.nt_upper,
        }
        dq, dk, dv, dg, dh0 = chunk_gla_varlen_kernel_bwd(
            q,
            k,
            v,
            g,
            do,
            dht,
            ctx.scale,
            initial_state=initial_state,
            cu_seqlens=ctx.cu_seqlens,
            chunk_size=ctx.chunk_size,
            parts=parts,
        )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(g.dtype),
            None,
            dh0,
            None,
            None,
            None,
            None,
        )


def chunk_gla_varlen_kernel(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-capable chunked GLA for a packed variable-length batch.

    Drop-in replacement for the varlen branch of ``api.chunk_gla``: same
    arguments, same ``(o, ht)`` outputs, one state per sequence, one launch
    group for the whole packed batch.  ``state_v_first`` is forward-only (the
    backward assumes the ``[K, V]`` layout), so keep those calls recurrent.
    """
    if cu_seqlens is None and cu_seqlens_cpu is None:
        raise ValueError(
            "chunk_gla_varlen_kernel requires cu_seqlens (or cu_seqlens_cpu)."
        )
    if state_v_first:
        raise NotImplementedError(
            "the chunked variable-length backward only supports the [K, V] state "
            "layout; keep state_v_first on the recurrent implementation."
        )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _ChunkGLAVarlenKernelFunction.apply(
        *_contiguous(q, k, v, g),
        scale,
        None if initial_state is None else _contiguous(initial_state)[0],
        output_final_state,
        state_v_first,
        chunk_size,
        cu_seqlens if cu_seqlens is not None else cu_seqlens_cpu,
    )
