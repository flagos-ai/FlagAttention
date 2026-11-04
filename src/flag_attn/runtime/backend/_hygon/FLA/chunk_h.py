# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_offsets
from .utils import check_shared_mem

# The Hygon BW runtime does not expose shared-memory properties through
# PyTorch, so check_shared_mem() conservatively returns False there. BW has
# sufficient SRAM for the larger state tiles; using them halves the number of
# K/V tile programs in the recurrent state scan.
_is_hygon_bw = False
if torch.cuda.is_available():
    try:
        _is_hygon_bw = torch.cuda.get_device_name(torch.cuda.current_device()) == "BW"
    except (AssertionError, RuntimeError):
        pass
BKV_LIST = [32, 64] if check_shared_mem() or _is_hygon_bw else [16, 32]


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "USE_G", "USE_GK", "USE_GV", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] accumulator; STATE_V_FIRST only flips the stored state's HBM layout to [V, K]
    # applied at the load/store below.
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = tl.make_block_ptr(
                h0 + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
        else:
            p_h0 = tl.make_block_ptr(
                h0 + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (K, T),
            (1, H * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(
                h + o_h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_h = tl.make_block_ptr(
                h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )

        if i_t % NTS == 0:
            tl.store(
                p_h,
                (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty),
                boundary_check=(0, 1),
            )
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.0)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(
                gk + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            p_gk_last = (
                gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            )

            b_gk_last = tl.load(
                p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.0
            )
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_h *= exp2(b_gk_last)[:, None]
            b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(
                gv + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_gv_last = (
                gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            )

            b_gv_last = tl.load(
                p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.0
            )
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_h *= exp2(b_gv_last)[None, :]
            b_v = (b_v * exp2(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            tl.store(
                p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_h_v2(
    k,
    v,
    gk,
    h,
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    V_GROUP_OFFSET: tl.constexpr,
    K_PRETRANSFORMED: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Forward-only H scan processing two adjacent V tiles per program.

    The K tile and its gate decay are shared by both V tiles.  This removes
    one K/gate load and halves the program count for wide states.  The
    specialized path is used only for the common no-initial/final-state,
    fixed-length GLA forward call; the general kernel above remains the
    fallback for backward and variable-length APIs.
    """
    i_k, i_vg, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_v = (i_vg + V_GROUP_OFFSET) * 2
    NT = tl.cdiv(T, BT)
    bos = i_b * T

    b_h0 = tl.zeros([BK, BV], dtype=tl.float32)
    b_h1 = tl.zeros([BK, BV], dtype=tl.float32)

    for i_t in range(NT):
        o_h = ((i_b * NT + i_t) * H + i_h).to(tl.int64) * K * V
        p_h0 = tl.make_block_ptr(
            h + o_h,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_h1 = tl.make_block_ptr(
            h + o_h,
            (K, V),
            (V, 1),
            (i_k * BK, (i_v + 1) * BV),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_h0, b_h0.to(p_h0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (K, T),
            (1, H * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_gk = tl.make_block_ptr(
            gk + (bos * H + i_h) * K,
            (K, T),
            (1, H * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_v0 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_v1 = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, (i_v + 1) * BV),
            (BT, BV),
            (1, 0),
        )

        last_idx = min((i_t + 1) * BT, T) - 1
        p_gk_last = (
            gk
            + (bos + last_idx) * H * K
            + i_h * K
            + i_k * BK
            + tl.arange(0, BK)
        )
        b_gk_last = tl.load(
            p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.0
        )
        if K_PRETRANSFORMED:
            b_k = tl.load(p_k, boundary_check=(0, 1))
        else:
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)
        decay = exp2(b_gk_last)[:, None]
        b_h0 *= decay
        b_h1 *= decay
        b_v0 = tl.load(p_v0, boundary_check=(0, 1))
        b_v1 = tl.load(p_v1, boundary_check=(0, 1))
        b_h0 += tl.dot(b_k, b_v0)
        b_h1 += tl.dot(b_k, b_v1)


@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_h_v2_cumsum(
    k,
    v,
    g,
    gk_out,
    kbar,
    h,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_KBAR: tl.constexpr,
):
    """H state scan for V-group zero; computes and publishes local gk."""
    i_k, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    NT = tl.cdiv(T, BT)
    bos = i_b * T
    i_v = 0
    b_h0 = tl.zeros([BK, BV], dtype=tl.float32)
    b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT):
        o_h = ((i_b * NT + i_t) * H + i_h).to(tl.int64) * K * V
        p_h0 = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, 0), (BK, BV), (1, 0))
        p_h1 = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, BV), (BK, BV), (1, 0))
        tl.store(p_h0, b_h0.to(p_h0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gk_out = tl.make_block_ptr(gk_out + (bos * H + i_h) * K, (K, T), (1, H * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v0 = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
        p_v1 = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, BV), (BT, BV), (1, 0))

        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_gk = tl.cumsum(b_g, axis=1) * 1.4426950216
        tl.store(p_gk_out, b_gk.to(p_gk_out.dtype.element_ty), boundary_check=(0, 1))
        # Reuse the tile-local cumulative decay; keep publishing gk_out above.
        last_col = min((i_t + 1) * BT, T) - 1 - i_t * BT
        b_gk_last = tl.sum(
            tl.where(tl.arange(0, BT)[None, :] == last_col, b_gk, 0.0), axis=1
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)
        if STORE_KBAR:
            p_kbar = tl.make_block_ptr(
                kbar + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            tl.store(p_kbar, b_k.to(p_kbar.dtype.element_ty), boundary_check=(0, 1))
        decay = exp2(b_gk_last)[:, None]
        b_h0 *= decay
        b_h1 *= decay
        b_v0 = tl.load(p_v0, boundary_check=(0, 1))
        b_v1 = tl.load(p_v1, boundary_check=(0, 1))
        b_h0 += tl.dot(b_k, b_v0)
        b_h1 += tl.dot(b_k, b_v1)

@triton.heuristics(
    {
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "USE_G", "USE_GK", "USE_GV", "STATE_V_FIRST"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dh(
    q,
    g,
    g_gamma,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    cu_seqlens,
    split_offsets,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] accumulator; STATE_V_FIRST only flips the stored state's HBM layout to [V, K]
    # applied at the load/store below.
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht = tl.make_block_ptr(
                dht + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            b_dh += tl.trans(tl.load(p_dht, boundary_check=(0, 1))).to(tl.float32)
        else:
            p_dht = tl.make_block_ptr(
                dht + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_dh = tl.make_block_ptr(
                dh + o_dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_dh = tl.make_block_ptr(
                dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )

        if i_t % (BS // BT) == 0:
            tl.store(
                p_dh,
                (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh.dtype.element_ty),
                boundary_check=(0, 1),
            )
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        p_q = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K,
            (K, T),
            (1, HQ * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V,
            (T, V),
            (HQ * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.0)
            b_q = (b_q * exp2(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp2(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = (b_q * exp2(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp2(b_g_last)

        if USE_GK:
            p_gk = tl.make_block_ptr(
                gk + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            p_gk_last = (
                gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            )

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_gk_last = tl.load(
                p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.0
            )
            b_q = (b_q * exp2(b_gk)).to(b_q.dtype)
            b_dh *= exp2(b_gk_last)[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(
                gv + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_gv_last = (
                gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            )

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_gv_last = tl.load(
                p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.0
            )
            b_do = b_do * exp2(b_gv)
            b_dh *= exp2(b_gv_last)[None, :]

        b_dh += tl.dot(b_q, b_do.to(b_q.dtype))

    if STORE_INITIAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dh0 = tl.make_block_ptr(
                dh0 + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            tl.store(
                p_dh0, tl.trans(b_dh).to(p_dh0.dtype.element_ty), boundary_check=(0, 1)
            )
        else:
            p_dh0 = tl.make_block_ptr(
                dh0 + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert (
        BS % BT == 0
    ), f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()

    # `state_v_first` stores the states in V-first `[V, K]` layout instead of `[K, V]`
    state_shape = (V, K) if state_v_first else (K, V)
    h = k.new_empty(
        B, NS, H, *state_shape, dtype=k.dtype if not states_in_fp32 else torch.float
    )
    ht = (
        k.new_empty(N, H, *state_shape, dtype=torch.float)
        if output_final_state
        else None
    )

    # For the forward-only GLA path, process two adjacent V tiles together.
    # The shared K/gate tile is loaded once, reducing state-scan program count
    # and global traffic for the wide states that dominate Hygon latency.
    use_v2 = (
        g is None
        and g_gamma is None
        and gk is not None
        and gv is None
        and h0 is None
        and not output_final_state
        and not state_v_first
        and not states_in_fp32
        and cu_seqlens is None
        and V >= 256
        and os.environ.get("FLAG_ATTN_GLA_H_V2", "1") != "0"
    )

    if use_v2:
        grid_v2 = lambda meta: (
            triton.cdiv(K, meta["BK"]),
            triton.cdiv(V, 2 * meta["BV"]),
            N * H,
        )
        chunk_fwd_kernel_h_v2[grid_v2](
            k=k,
            v=v,
            gk=gk,
            h=h,
            cu_seqlens=cu_seqlens,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=32,
            BV=64,
            V_GROUP_OFFSET=0,
            K_PRETRANSFORMED=False,
        )
        return h, ht

    def grid(meta):
        return (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), N * H)

    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        STATE_V_FIRST=state_v_first,
    )
    return h, ht

def chunk_fwd_h_fused_cumsum(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local gk and the wide-state H scan in one forward pipeline."""
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    # The fused path is register-sensitive. For V=128/256, use four warps;
    # for wider states, use eight warps to parallelize the long K reduction
    # without the register pressure observed with BK=64. These are the
    # measured production tiles; runtime tile overrides are intentionally not
    # exposed.
    if V >= 512:
        # On BW, BK=32 with eight warps is faster than widening BK to 64:
        # it preserves occupancy while parallelizing the long K reduction.
        BK, BV, num_warps = 32, 64, 8
    else:
        BK, BV, num_warps = 32, 64, 4
    NT = triton.cdiv(T, BT)
    h = k.new_empty(B, NT, H, K, V)
    gk_out = torch.empty_like(g, dtype=torch.float)
    n_groups = triton.cdiv(V, 2 * BV)
    # Reuse the gate-transformed K tile across the remaining V groups only
    # when there are at least four groups.  On gfx936 this avoids repeated
    # K/gk traffic and gate scaling for wide states; narrower states do not
    # amortize the extra Kbar write/read.  Keep an opt-out for A/B diagnosis.
    use_pretransformed_k = (
        os.environ.get("FLAG_ATTN_GLA_H_PRETRANSFORMED_K", "1") == "1"
        and n_groups >= 4
    )
    kbar = torch.empty_like(k) if use_pretransformed_k else k
    chunk_fwd_kernel_h_v2_cumsum[(triton.cdiv(K, BK), B * H)](
        k=k,
        v=v,
        g=g,
        gk_out=gk_out,
        kbar=kbar,
        h=h,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        STORE_KBAR=use_pretransformed_k,
        num_warps=num_warps,
    )
    if n_groups > 1:
        chunk_fwd_kernel_h_v2[(triton.cdiv(K, BK), n_groups - 1, B * H)](
            k=kbar,
            v=v,
            gk=gk_out,
            h=h,
            cu_seqlens=None,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            V_GROUP_OFFSET=1,
            K_PRETRANSFORMED=use_pretransformed_k,
        )
    return gk_out, h



def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    assert (
        BS % BT == 0
    ), f"The `split_size` (got {BS}) must be a multiple of `chunk_size` {BT}"
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NG: number of groups in GQA
    if cu_seqlens is None:
        N, NS, split_offsets = B, triton.cdiv(T, BS), None
    else:
        split_offsets = prepare_chunk_offsets(cu_seqlens, BS)
        N, NS = len(cu_seqlens) - 1, split_offsets[-1].item()
    NG = HQ // H

    # `state_v_first` stores the states in V-first `[V, K]` layout instead of `[K, V]`
    state_shape = (V, K) if state_v_first else (K, V)
    dh = k.new_empty(
        B, NS, HQ, *state_shape, dtype=k.dtype if not states_in_fp32 else torch.float
    )
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta):
        return (triton.cdiv(K, meta["BK"]), triton.cdiv(V, meta["BV"]), N * H)

    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        NG=NG,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        STATE_V_FIRST=state_v_first,
    )
    return dh, dh0
