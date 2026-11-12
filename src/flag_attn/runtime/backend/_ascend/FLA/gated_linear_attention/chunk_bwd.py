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

"""Fixed-length chunked GLA backward for Ascend.

Stages: reverse state scan dh, causal dA, intra-chunk dq/dk, inter-chunk
dq/dk/dg, and dv. The forward supplies log2 gate cumsums, A, and chunk states.
Gate gradients use inclusive suffix sums of q*dq-k*dk plus state-decay carry.
Final-state gradients seed dh; the scan also produces initial-state gradients.
Both state layouts are addressed in-kernel without transpose copies.

Dots use FP32; stored dh and final dv/dg use their corresponding input dtypes.
Chunk-parallel launches use the fixed-length program budget to bound scratch.
Large fp16/bf16 K=V=128 workloads use split state/dq/dk phases and optional
TLE Raw factors/reductions. Guarded wide intra dots retain a panel fallback;
Other configurations use panel intra dots and standard state projections.
Varlen wrappers live separately."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from .chunk import BC, BK, FIXED_MAX_PROGRAMS, PANEL_LOG_MAX, _chunks_per_launch, _state_tiles, exp2

# Tile of the V loop in the dv and dq/dk/dg kernels.  Narrower than the forward
# output tile: the backward stages keep several float32 accumulators live.
BWD_BV = 128
BWD_BK = 64
BWD_BK_W = 128
BWD_BV_W = 256


@lru_cache(maxsize=1)
def _raw_intra_launcher():
    # Load lazily so builds without TLE Raw retain the existing kernels.
    try:
        from .chunk_bwd_raw import launch_dqk_intra
    except ImportError:
        return None
    return launch_dqk_intra


@lru_cache(maxsize=1)
def _raw_inter_reductions():
    try:
        from .chunk_bwd_raw import _reverse_sum, _state_product
    except ImportError:
        return None, None
    return _state_product, _reverse_sum


@triton.jit(do_not_specialize=["T", "NT"])
def chunk_gla_bwd_dh_kernel(
    q,
    do,
    g_cum,
    dh,
    dh0,
    dht,
    scale,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    """Reverse inter-chunk state scan.

    Mirrors the forward state pass with ``dht`` as the seed: the state gradient
    leaving chunk ``i`` is stored before the update that folds chunk ``i`` in,
    and the value left after the last iteration is ``dh0``.

    ``STATE_V_FIRST`` selects the ``[V, K]`` state layout, exactly as in the
    forward state kernel: the accumulator stays ``[BK, BV]`` and only the
    load/store of the ``[K, V]`` tile is transposed.
    """
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_bh * K * V
        if STATE_V_FIRST:
            b_dh = tl.trans(
                tl.load(p_dht + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                        mask=m_v[:, None] & m_k[None, :], other=0.0)
            )
        else:
            b_dh = tl.load(p_dht + o_k[:, None].to(tl.int64) * V + o_v[None, :], mask=m_h, other=0.0)
        b_dh = b_dh.to(tl.float32)

    base_q = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    for i_t in range(NT - 1, -1, -1):
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        m_kt = m_k[:, None] & m_t[None, :]
        m_tv = m_t[:, None] & m_v[None, :]

        o_dh = ((i_b * NT + i_t) * H + i_h) * K * V
        if STATE_V_FIRST:
            tl.store(dh + o_dh + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                     tl.trans(b_dh).to(dh.dtype.element_ty), mask=m_v[:, None] & m_k[None, :])
        else:
            tl.store(dh + o_dh + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                     b_dh.to(dh.dtype.element_ty), mask=m_h)

        last_idx = i_t * BT + BT - 1
        last_idx = tl.where(last_idx < T, last_idx, T - 1)
        b_gl = tl.load(g_cum + base_q + last_idx * (H * K) + o_k, mask=m_k, other=0.0)
        b_q = tl.load(q + base_q + o_t[None, :] * (H * K) + o_k[:, None], mask=m_kt, other=0.0)
        b_g = tl.load(g_cum + base_q + o_t[None, :] * (H * K) + o_k[:, None], mask=m_kt, other=0.0)
        b_do = tl.load(do + base_v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)

        b_dh = b_dh * exp2(b_gl)[:, None]
        b_dh = tl.dot(b_q.to(tl.float32) * exp2(b_g) * scale, b_do.to(tl.float32), b_dh)

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_bh * K * V
        if STATE_V_FIRST:
            tl.store(p_dh0 + o_v[:, None].to(tl.int64) * K + o_k[None, :], tl.trans(b_dh),
                     mask=m_v[:, None] & m_k[None, :])
        else:
            tl.store(p_dh0 + o_k[:, None].to(tl.int64) * V + o_v[None, :], b_dh, mask=m_h)


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_bwd_dA_kernel(
    v,
    do,
    dA,
    scale,
    T,
    I_T_LO,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """dA = tril(do @ v^T) * scale, the gradient of the intra-chunk matrix."""
    i_t = tl.program_id(0) + I_T_LO
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    base_v = i_b * T * H * V + i_h * V

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        b_do = tl.load(do + base_v + o_t[:, None] * (H * V) + o_v[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0)
        # v is read transposed so the dot contracts over the value dimension.
        b_v = tl.load(v + base_v + o_v[:, None] + o_t[None, :] * (H * V),
                      mask=m_v[:, None] & m_t[None, :], other=0.0)
        b_dA = tl.dot(b_do.to(tl.float32), b_v.to(tl.float32), b_dA)

    b_dA = tl.where(o_i[:, None] >= o_i[None, :], b_dA * scale, 0.0)
    p_dA = dA + (i_b * T + o_t)[:, None] * (H * BT) + i_h * BT + o_i[None, :]
    tl.store(p_dA, b_dA, mask=m_t[:, None])


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def chunk_gla_bwd_dqk_intra_kernel(
    q,
    k,
    g_cum,
    dA,
    dq,
    dk,
    T,
    I_T_LO,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
):
    """Intra-chunk dq and dk from dA, in ``BC``-wide column panels.

    The panel midpoint is the exponent reference, exactly as in the forward A
    kernel: the key and query factors each span at most ``BC // 2`` tokens and
    the query factor is capped at ``PANEL_LOG_MAX``, so both stay inside the
    float32 range for strongly decaying gates.  Rows above the panel have
    ``dA == 0`` and only pick up the cap.
    """
    i_k = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_k = i_k * BK + tl.arange(0, BK)
    o_t = i_t * BT + tl.arange(0, BT)
    m_k = o_k < K
    m_t = o_t < T
    m_tk = m_t[:, None] & m_k[None, :]

    base_k = i_b * T * H * K + i_h * K
    b_g = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    b_q = tl.load(q + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    for i_p in tl.static_range(BT // BC):
        c0 = i_t * BT + i_p * BC
        if c0 < T:
            o_c = c0 + tl.arange(0, BC)
            m_ck = (o_c < T)[:, None] & m_k[None, :]
            c_mid = tl.minimum(c0 + BC // 2, T - 1)
            b_ref = tl.load(g_cum + base_k + c_mid * (H * K) + o_k, mask=m_k, other=0.0)
            b_kc = tl.load(k + base_k + o_c[:, None] * (H * K) + o_k[None, :], mask=m_ck, other=0.0)
            b_gc = tl.load(g_cum + base_k + o_c[:, None] * (H * K) + o_k[None, :], mask=m_ck, other=0.0)
            b_row = exp2(tl.minimum(b_g - b_ref[None, :], PANEL_LOG_MAX))
            b_col = exp2(b_ref[None, :] - b_gc)
            o_i = i_p * BC + tl.arange(0, BC)
            p_dA = dA + (i_b * T + o_t)[:, None] * (H * BT) + i_h * BT + o_i[None, :]
            b_dA = tl.load(p_dA, mask=m_t[:, None], other=0.0)
            b_dq += tl.dot(b_dA, b_kc.to(tl.float32) * b_col) * b_row
            b_dkj = tl.dot(tl.trans(b_dA), b_q.to(tl.float32) * b_row) * b_col
            p_dk = dk + base_k + o_c[:, None] * (H * K) + o_k[None, :]
            tl.store(p_dk, b_dkj, mask=m_ck)

    tl.store(dq + base_k + o_t[:, None] * (H * K) + o_k[None, :], b_dq, mask=m_tk)


@triton.jit(do_not_specialize=["T", "NT", "I_T_LO"])
def chunk_gla_bwd_dqkg_kernel(
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
    """Inter-chunk dq/dk, added on top of the intra passes, plus dg.

    ``dq``/``dk`` are read back from the intra kernel, extended with the
    inter-chunk terms and written out final; the ``g_last`` term of ``dg`` only
    uses the inter-chunk ``dk``.

    The state tiles are always materialised as ``[BV, BK]`` so both dots
    contract on V; ``STATE_V_FIRST`` only changes the addressing (``[V, K]`` is
    the natural layout of that tile, ``[K, V]`` is read transposed).
    """
    i_k = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_k = i_k * BK + tl.arange(0, BK)
    o_t = i_t * BT + tl.arange(0, BT)
    m_k = o_k < K
    m_t = o_t < T
    m_tk = m_t[:, None] & m_k[None, :]

    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    o_state = ((i_b * NT + i_t) * H + i_h) * K * V

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        b_v = tl.load(v + base_v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
        b_do = tl.load(do + base_v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
        # State tiles are read transposed ([BV, BK]) so both dots contract on V.
        # h/dh are stored in the input dtype; upcast for the float32 dots.
        if STATE_V_FIRST:
            # [V, K] layout: the o_v-major tile is already [BV, BK].
            b_h = tl.load(h + o_state + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                          mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_dh = tl.load(dh + o_state + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                           mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        else:
            b_h = tl.load(h + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
                          mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_dh = tl.load(dh + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
                           mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        # dL/dg_last from the state decay: h_next = exp2(g_last) * h.
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq = tl.dot(b_do.to(tl.float32), b_h, b_dq)
        b_dk = tl.dot(b_v.to(tl.float32), b_dh, b_dk)

    last_idx = i_t * BT + BT - 1
    last_idx = tl.where(last_idx < T, last_idx, T - 1)
    b_gl = tl.load(g_cum + base_k + last_idx * (H * K) + o_k, mask=m_k, other=0.0)
    b_g = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    b_q = tl.load(q + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    b_k = tl.load(k + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    p_dq = dq + base_k + o_t[:, None] * (H * K) + o_k[None, :]
    p_dk = dk + base_k + o_t[:, None] * (H * K) + o_k[None, :]

    b_dgk = b_dgk * exp2(b_gl)
    b_dq = b_dq * scale * exp2(b_g) + tl.load(p_dq, mask=m_tk, other=0.0)
    b_dk = b_dk * exp2(b_gl[None, :] - b_g)
    # The k_e factor carries g_last; only the inter-chunk term contributes here.
    b_dgk += tl.sum(b_dk * b_k.to(tl.float32), axis=0)
    b_dk += tl.load(p_dk, mask=m_tk, other=0.0)

    b_dg = b_q.to(tl.float32) * b_dq - b_k.to(tl.float32) * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]

    tl.store(p_dq, b_dq, mask=m_tk)
    tl.store(p_dk, b_dk, mask=m_tk)
    tl.store(dg + base_k + o_t[:, None] * (H * K) + o_k[None, :], b_dg, mask=m_tk)


@triton.jit(do_not_specialize=["T", "NT", "I_T_LO"])
def chunk_gla_bwd_dqkg_split_kernel(
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
    STATE_PRODUCT: tl.constexpr = None,
    REVERSE_SUM: tl.constexpr = None,
):
    """K projection with separate state-product, dq, and dk phases.

    Reload h/dh in each phase to limit live state tiles for BK=128.
    The state-decay reduction uses 64-element value tiles."""
    i_k = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_k = i_k * BK + tl.arange(0, BK)
    o_t = i_t * BT + tl.arange(0, BT)
    m_k = o_k < K
    m_t = o_t < T
    m_tk = m_t[:, None] & m_k[None, :]

    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    o_state = ((i_b * NT + i_t) * H + i_h) * K * V

    # State-decay reduction first: its two states expire before either dot.
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, 64)):
        o_v = i_v * 64 + tl.arange(0, 64)
        m_v = o_v < V
        if STATE_PRODUCT is not None and not STATE_V_FIRST:
            # V is contiguous in [K,V] states; reduce each row directly in UB.
            b_h = tl.load(h + o_state + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                          mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            b_dh = tl.load(dh + o_state + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                           mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            b_dgk += STATE_PRODUCT(b_h, b_dh)
        else:
            if STATE_V_FIRST:
                b_h = tl.load(h + o_state + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                    mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                b_dh = tl.load(dh + o_state + o_v[:, None].to(tl.int64) * K + o_k[None, :],
                    mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            else:
                b_h = tl.load(h + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
                    mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
                b_dh = tl.load(dh + o_state + o_k[None, :].to(tl.int64) * V + o_v[:, None],
                    mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_dgk += tl.sum(b_h * b_dh, axis=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    for j_v in range(tl.cdiv(V, BV)):
        v_q = j_v * BV + tl.arange(0, BV)
        mv_q = v_q < V
        b_do = tl.load(do + base_v + o_t[:, None] * (H * V) + v_q[None, :],
            mask=m_t[:, None] & mv_q[None, :], other=0.0).to(tl.float32)
        if STATE_V_FIRST:
            h_q = tl.load(h + o_state + v_q[:, None].to(tl.int64) * K + o_k[None, :],
                mask=mv_q[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        else:
            h_q = tl.load(h + o_state + o_k[None, :].to(tl.int64) * V + v_q[:, None],
                mask=mv_q[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_dq = tl.dot(b_do, h_q, b_dq)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for j_v in range(tl.cdiv(V, BV)):
        v_k = j_v * BV + tl.arange(0, BV)
        mv_k = v_k < V
        b_v = tl.load(v + base_v + o_t[:, None] * (H * V) + v_k[None, :],
            mask=m_t[:, None] & mv_k[None, :], other=0.0).to(tl.float32)
        if STATE_V_FIRST:
            dh_k = tl.load(dh + o_state + v_k[:, None].to(tl.int64) * K + o_k[None, :],
                mask=mv_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        else:
            dh_k = tl.load(dh + o_state + o_k[None, :].to(tl.int64) * V + v_k[:, None],
                mask=mv_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_dk = tl.dot(b_v, dh_k, b_dk)
    last_idx = i_t * BT + BT - 1
    last_idx = tl.where(last_idx < T, last_idx, T - 1)
    b_gl = tl.load(g_cum + base_k + last_idx * (H * K) + o_k, mask=m_k, other=0.0)
    b_g = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    b_q = tl.load(q + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    b_k = tl.load(k + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
    p_dq = dq + base_k + o_t[:, None] * (H * K) + o_k[None, :]
    p_dk = dk + base_k + o_t[:, None] * (H * K) + o_k[None, :]

    b_dgk = b_dgk * exp2(b_gl)
    b_dq = b_dq * scale * exp2(b_g) + tl.load(p_dq, mask=m_tk, other=0.0)
    b_dk = b_dk * exp2(b_gl[None, :] - b_g)
    # The k_e factor carries g_last; only the inter-chunk term contributes here.
    b_dgk += tl.sum(b_dk * b_k.to(tl.float32), axis=0)
    b_dk += tl.load(p_dk, mask=m_tk, other=0.0)

    b_dg = b_q.to(tl.float32) * b_dq - b_k.to(tl.float32) * b_dk
    if REVERSE_SUM is None:
        b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    else:
        b_dg = REVERSE_SUM(b_dg, b_dgk)

    tl.store(p_dq, b_dq, mask=m_tk)
    tl.store(p_dk, b_dk, mask=m_tk)
    tl.store(dg + base_k + o_t[:, None] * (H * K) + o_k[None, :], b_dg, mask=m_tk)


@triton.jit(do_not_specialize=["T", "NT", "I_T_LO"])
def chunk_gla_bwd_dv_kernel(
    k,
    g_cum,
    A,
    do,
    dh,
    dv,
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
    """dv = A^T @ do + k_e @ dh."""
    i_v = tl.program_id(0)
    i_t = tl.program_id(1) + I_T_LO
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]

    base_k = i_b * T * H * K + i_h * K
    base_v = i_b * T * H * V + i_h * V
    o_state = ((i_b * NT + i_t) * H + i_h) * K * V

    # A is read transposed ([j, t]) so the dot contracts on the token axis and
    # yields dv = A^T @ do; the rows above the diagonal are zero in A.
    b_A = tl.load(A + (i_b * T + o_t)[None, :] * (H * BT) + i_h * BT + o_i[:, None],
                  mask=m_t[None, :], other=0.0)
    b_do = tl.load(do + base_v + o_t[:, None] * (H * V) + o_v[None, :], mask=m_tv, other=0.0)
    b_dv = tl.dot(b_A, b_do.to(tl.float32))

    last_idx = i_t * BT + BT - 1
    last_idx = tl.where(last_idx < T, last_idx, T - 1)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_gl = tl.load(g_cum + base_k + last_idx * (H * K) + o_k, mask=m_k, other=0.0)
        b_g = tl.load(g_cum + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        b_k = tl.load(k + base_k + o_t[:, None] * (H * K) + o_k[None, :], mask=m_tk, other=0.0)
        if STATE_V_FIRST:
            # [V, K] layout: same [BK, BV] tile, read transposed.
            b_dh = tl.load(dh + o_state + o_v[None, :].to(tl.int64) * K + o_k[:, None],
                           mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        else:
            b_dh = tl.load(dh + o_state + o_k[:, None].to(tl.int64) * V + o_v[None, :],
                           mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_dv = tl.dot(b_k.to(tl.float32) * exp2(b_gl[None, :] - b_g), b_dh, b_dv)

    tl.store(dv + base_v + o_t[:, None] * (H * V) + o_v[None, :], b_dv, mask=m_tv)


def chunk_gla_bwd_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    g_cum: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None = None,
    state_v_first: bool = False,
):
    """Chunked GLA backward from the forward intermediates.

    ``h`` is the ``[B, NT, H, K, V]`` state array in the input dtype (``[B, NT,
    H, V, K]`` when ``state_v_first``, matching the forward), ``A`` the fp32
    ``[B, T, H, BT]`` intra-chunk matrix and ``g_cum`` the fp32 base-2 gate
    cumsum, all as returned by ``chunk_gla_fwd_chunked``.  ``initial_state`` and
    ``dht`` must use the same layout as ``h``'s last two dims.  Returns
    ``(dq, dk, dv, dg, dh0)``: ``dq``/``dk``/``dh0`` in float32, ``dv``/``dg``
    in the input dtype (the caller casts them to it anyway).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    bt = A.shape[-1]
    nt = h.shape[1]
    state_shape = (V, K) if state_v_first else (K, V)
    state_layout = "V, K" if state_v_first else "K, V"
    assert tuple(h.shape) == (B, nt, H, *state_shape), (
        f"h must be the forward [B, NT, H, {state_layout}] state array."
    )
    assert g_cum.shape == (B, T, H, K), "g_cum must be the forward [B, T, H, K] cumsum."
    if dht is not None:
        assert tuple(dht.shape) == (B, H, *state_shape), f"dht must be [B, H, {state_layout}]."
    if initial_state is not None:
        assert tuple(initial_state.shape) == (B, H, *state_shape), (
            f"initial_state must be [B, H, {state_layout}]."
        )

    bkc = min(BWD_BK, triton.next_power_of_2(K))
    bkw = min(BWD_BK_W, triton.next_power_of_2(K))
    bsk, bsv = _state_tiles(K, V)
    bv = min(BWD_BV, triton.next_power_of_2(V))
    bvw = min(BWD_BV_W, triton.next_power_of_2(V))
    nk = triton.cdiv(K, bkc)
    nkw = triton.cdiv(K, bkw)

    # dh mirrors h (input dtype); dv/dg are produced in one pass and are cast to
    # the input dtype by the caller anyway, so they are allocated in it directly.
    dh = torch.empty_like(h)
    dh0 = torch.empty_like(initial_state) if initial_state is not None else None
    dA = torch.empty((B, T, H, bt), device=q.device, dtype=torch.float32)
    dq = torch.empty((B, T, H, K), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dv = torch.empty((B, T, H, V), device=q.device, dtype=v.dtype)
    dg = torch.empty((B, T, H, K), device=q.device, dtype=g.dtype)

    chunk_gla_bwd_dh_kernel[(triton.cdiv(K, bsk), triton.cdiv(V, bsv), B * H)](
        q=q,
        do=do,
        g_cum=g_cum,
        dh=dh,
        dh0=dh0,
        dht=dht,
        scale=scale,
        T=T,
        NT=nt,
        H=H,
        K=K,
        V=V,
        BT=bt,
        BK=bsk,
        BV=bsv,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        STORE_INITIAL_STATE_GRADIENT=dh0 is not None,
        STATE_V_FIRST=state_v_first,
    )
    # Each kernel is launched once per group of chunks: the full grid is just
    # split, which bounds the per-program launch scratch of each launch.
    cpb = _chunks_per_launch(nt * B * H, nt, FIXED_MAX_PROGRAMS)
    for i_lo in range(0, nt, cpb):
        chunk_gla_bwd_dA_kernel[(min(cpb, nt - i_lo), B * H)](
            v=v,
            do=do,
            dA=dA,
            scale=scale,
            T=T,
            I_T_LO=i_lo,
            H=H,
            V=V,
            BT=bt,
            BV=bv,
        )
    raw_intra = None
    if (q.dtype in (torch.float16, torch.bfloat16) and K == V == 128
            and bt == 64 and T >= 1024 and nt * B * H >= 1024):
        raw_intra = _raw_intra_launcher()
    if raw_intra is not None:
        raw_intra(q, k, g_cum, dA, dq, dk)
    else:
        cpb_kw = _chunks_per_launch(nkw * nt * B * H, nt, FIXED_MAX_PROGRAMS)
        for i_lo in range(0, nt, cpb_kw):
            n = min(cpb_kw, nt - i_lo)
            chunk_gla_bwd_dqk_intra_kernel[(nkw, n, B * H)](
                q=q,
                k=k,
                g_cum=g_cum,
                dA=dA,
                dq=dq,
                dk=dk,
                T=T,
                I_T_LO=i_lo,
                H=H,
                K=K,
                BT=bt,
                BC=BC,
                BK=bkw,
            )
    # Separate phases bound live storage for BK=128/BV=64. FP32 inputs
    # exceed the UB capacity of this tile and use the standard projection.
    split_inter = (
        q.dtype in (torch.float16, torch.bfloat16)
        and K == V == 128 and bt == 64 and T >= 1024 and nt * B * H >= 1024
    )
    if split_inter:
        inter_bk, inter_bv = 128, 64
        state_product, reverse_sum = _raw_inter_reductions()
        inter_nk = triton.cdiv(K, inter_bk)
        cpb_k = _chunks_per_launch(inter_nk * nt * B * H, nt, FIXED_MAX_PROGRAMS)
        for i_lo in range(0, nt, cpb_k):
            n = min(cpb_k, nt - i_lo)
            chunk_gla_bwd_dqkg_split_kernel[(inter_nk, n, B * H)](
                q=q, k=k, v=v, g_cum=g_cum, h=h, dh=dh, do=do,
                dq=dq, dk=dk, dg=dg, scale=scale,
                T=T, NT=nt, I_T_LO=i_lo, H=H, K=K, V=V,
                BT=bt, BK=inter_bk, BV=inter_bv, STATE_V_FIRST=state_v_first,
                STATE_PRODUCT=state_product, REVERSE_SUM=reverse_sum,
            )
    else:
        cpb_k = _chunks_per_launch(nk * nt * B * H, nt, FIXED_MAX_PROGRAMS)
        for i_lo in range(0, nt, cpb_k):
            n = min(cpb_k, nt - i_lo)
            chunk_gla_bwd_dqkg_kernel[(nk, n, B * H)](
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
                T=T,
                NT=nt,
                I_T_LO=i_lo,
                H=H,
                K=K,
                V=V,
                BT=bt,
                BK=bkc,
                BV=bv,
                STATE_V_FIRST=state_v_first,
            )

    n_vd = triton.cdiv(V, bvw)
    cpb_v = _chunks_per_launch(n_vd * nt * B * H, nt, FIXED_MAX_PROGRAMS)
    for i_lo in range(0, nt, cpb_v):
        chunk_gla_bwd_dv_kernel[(n_vd, min(cpb_v, nt - i_lo), B * H)](
            k=k,
            g_cum=g_cum,
            A=A,
            do=do,
            dh=dh,
            dv=dv,
            T=T,
            NT=nt,
            I_T_LO=i_lo,
            H=H,
            K=K,
            V=V,
            BT=bt,
            BK=bkw,
            BV=bvw,
            STATE_V_FIRST=state_v_first,
        )
    return dq, dk, dv, dg, dh0
