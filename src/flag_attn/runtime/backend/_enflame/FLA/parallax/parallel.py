# Copyright 2026 FlagOS Contributors
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy

import torch
import triton
import triton.language as tl

from ..compat import exp2, get_tuned_config, libtuner
from ..index import prepare_chunk_indices_enflame as prepare_chunk_indices
from ..utils import get_num_sms, input_guard, make_grid_2d, requires_grid_remap


# Dense and packed paths share the same query-row tile.
_BLOCK_T = 64


_SHORT_MULTI_MAX_TILES = 2
_DENSE_ALIGNED_MAX_T = 2048
_GROUPED_DKV_G2_MIN_T = 8192


def _make_parallax_configs(tile_pairs):
    """Build the Enflame fallback autotune space."""
    return [
        triton.Config(
            {"BK": block_k, "BS": block_s},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_k, block_s in tile_pairs
        for num_warps, num_stages in ((1, 1), (1, 2), (2, 1))
    ]


# BT fixes the query-row grid. BS is tuned independently because FWD/DQR
# accumulate over query rows, while DKV carries two [BS, BK] fp32 accumulators.
_PARALLAX_DQR_TILES = (
    (16, 64),
    (16, 128),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 16),
    (256, 32),
    (256, 64),
    (256, 128),
)

_PARALLAX_FWD_TILES = tuple(tile for tile in _PARALLAX_DQR_TILES if tile != (64, 16))

_PARALLAX_DKV_TILES = (
    (16, 64),
    (16, 128),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 16),
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 16),
    (256, 32),
    (256, 64),
    (256, 128),
)

PARALLAX_FWD_CONFIGS = get_tuned_config(
    "parallel_parallax_fwd_kernel",
    _make_parallax_configs(_PARALLAX_FWD_TILES),
    fast_autotune_key="BK",
)
PARALLAX_DQR_CONFIGS = get_tuned_config(
    "parallel_parallax_bwd_kernel_dqr",
    _make_parallax_configs(_PARALLAX_DQR_TILES),
    fast_autotune_key="BK",
)
PARALLAX_DQR_SPLIT_CONFIGS = get_tuned_config(
    "parallel_parallax_bwd_kernel_dqr_split",
    _make_parallax_configs(_PARALLAX_DQR_TILES),
    fast_autotune_key="BK",
)
PARALLAX_DKV_CONFIGS = get_tuned_config(
    "parallel_parallax_bwd_kernel_dkv",
    _make_parallax_configs(_PARALLAX_DKV_TILES),
    fast_autotune_key="BK",
)

PARALLAX_PREPROCESS_CONFIGS = get_tuned_config(
    "parallel_parallax_bwd_kernel_preprocess",
    [
        triton.Config(
            {"BP": block_p},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_p in (32, 64, 128, 256)
        for num_warps in (1, 2)
        for num_stages in (1, 2)
    ],
    fast_autotune_key="BP",
)


def _autotune_arg(named_args, kwargs, name):
    value = named_args.get(name)
    # FlagTree may keep constexpr names in `named_args` with a None value and
    # pass their actual launch value through kwargs.  `dict.get(key, default)`
    # does not use the default when such a placeholder exists.
    return kwargs.get(name) if value is None else value


def _copy_autotune_configs(configs):
    """Protect config templates from mutation by FlagTree's autotuner."""
    return copy.deepcopy(list(configs))


def parallax_prune_configs(configs, named_args, **kwargs):
    """Keep the matching head tile and preserve one fixed varlen config."""
    head_dim = _autotune_arg(named_args, kwargs, "K")
    if head_dim is None:
        q = _autotune_arg(named_args, kwargs, "q")
        if q is not None and hasattr(q, "shape"):
            head_dim = q.shape[-1]
    if head_dim is None:
        return _copy_autotune_configs(configs)

    expected_bk = triton.next_power_of_2(int(head_dim))
    valid = [cfg for cfg in configs if cfg.kwargs["BK"] == expected_bk]

    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(valid)

    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [cfg for cfg in valid if cfg.kwargs["BS"] == block_t]
    preferred = [cfg for cfg in tile_matches if cfg.num_warps == 1 and cfg.num_stages == 1]
    return _copy_autotune_configs(preferred or tile_matches)


def parallax_prune_preprocess_configs(configs, named_args, **kwargs):
    """Varlen chunk indices require preprocess BP to remain equal to BT."""
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(configs)
    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [cfg for cfg in configs if cfg.kwargs["BP"] == block_t]
    preferred = [cfg for cfg in tile_matches if cfg.num_warps == 1 and cfg.num_stages == 1]
    return _copy_autotune_configs(preferred or tile_matches)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel_short(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
):
    """One-tile dense forward used when the complete sequence fits in SRAM."""
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int32)

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_r = tl.make_block_ptr(
        r + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

    offs_q = tl.arange(0, BT)
    offs_kv = tl.arange(0, BT)
    row_mask = offs_q[:, None] < T
    mask = (offs_q[:, None] >= offs_kv[None, :]) & row_mask & (offs_kv[None, :] < T)
    if WINDOW_SIZE_LEFT >= 0:
        mask = mask & (offs_kv[None, :] >= offs_q[:, None] - WINDOW_SIZE_LEFT + 1)

    scale_log2 = scale * 1.4426950216
    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
    qk = tl.where(mask, qk, -float("inf"))
    m_acc = tl.max(qk, axis=1, keep_dims=True)
    safe_m = tl.where(m_acc == -float("inf"), 0.0, m_acc)
    w = exp2(qk - safe_m)
    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    wr = w * rk

    d1_acc = tl.sum(w, axis=1, keep_dims=True)
    d2_acc = tl.sum(wr, axis=1, keep_dims=True)
    barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32)
    rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32)
    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - rv_acc * inv_d1

    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_barv = tl.make_block_ptr(
        barv + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_d1 = tl.make_block_ptr(
        d1 + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_bart = tl.make_block_ptr(
        bart + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_m = tl.make_block_ptr(
        m + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_d1, d1_acc, boundary_check=(0, 1))
    tl.store(p_bart, b_bart, boundary_check=(0, 1))
    tl.store(p_m, m_acc, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel_short_multi(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    """Dense forward for a small number of query tiles."""
    i_t = tl.program_id(0).to(tl.int32)
    i_bh = tl.program_id(1).to(tl.int32)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int32)
    offs_k = tl.arange(0, BK).to(tl.int32)
    row_mask = offs_t[:, None] < T
    head_mask = offs_k[None, :] < K
    q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)

    m_acc = tl.full((BT, 1), -float("inf"), dtype=tl.float32)
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_s in range(0, tl.minimum(i_t + 1, NT)):
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int32)
        col_mask = offs_s[:, None] < T
        kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
        b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        wr = w * tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)

        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        rv_acc = alpha * rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=rv_acc)
        m_acc = m_new

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - rv_acc * inv_d1

    scalar_offsets = (bos + offs_t[:, None]) * HQ + i_hq
    tl.store(o + q_offsets, b_o.to(o.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(barv + q_offsets, b_barv.to(barv.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(d1 + scalar_offsets, d1_acc, mask=row_mask)
    tl.store(bart + scalar_offsets, b_bart, mask=row_mask)
    tl.store(m + scalar_offsets, m_acc, mask=row_mask)


@libtuner(
    configs=PARALLAX_FWD_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int32), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int32)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_o = o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_barv = barv + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id.to(tl.int32) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        # Complete the w path before creating rk/wr so fewer [BT, BS]
        # temporaries compete with the two persistent output accumulators.
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        Rv_acc = alpha * Rv_acc
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    # Phase A: safe blocks (no mask).
    for _safe in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        o_kv = _safe.to(tl.int32) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        Rv_acc = alpha * Rv_acc
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int32) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        Rv_acc = alpha * Rv_acc
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_qk)
    tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), mask=m_qk)
    tl.store(p_d1, d1_acc, mask=row_mask)
    tl.store(p_bart, b_bart, mask=row_mask)
    tl.store(p_m, m_acc, mask=row_mask)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_short(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    grad_q,
    grad_r,
    grad_k_buf,
    grad_v_buf,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
):
    """Fuse preprocess, dQ/dR, and dK/dV for a one-tile sequence."""
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int32)

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_r = tl.make_block_ptr(
        r + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_barv = tl.make_block_ptr(
        barv + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_o = tl.make_block_ptr(
        grad_o + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_d1 = tl.make_block_ptr(
        d1 + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_bart = tl.make_block_ptr(
        bart + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_m = tl.make_block_ptr(
        m + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_barv = tl.load(p_barv, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_grad_o = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")
    b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
    b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
    b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")

    delta_t = tl.sum(b_grad_o.to(tl.float32) * b_o, axis=1, keep_dims=True)
    delta_b = tl.sum(b_grad_o.to(tl.float32) * b_barv, axis=1, keep_dims=True)

    offs_q = tl.arange(0, BT)
    offs_kv = tl.arange(0, BT)
    row_mask = offs_q[:, None] < T
    mask = (offs_q[:, None] >= offs_kv[None, :]) & row_mask & (offs_kv[None, :] < T)
    if WINDOW_SIZE_LEFT >= 0:
        mask = mask & (offs_kv[None, :] >= offs_q[:, None] - WINDOW_SIZE_LEFT + 1)

    scale_log2 = scale * 1.4426950216
    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
    qk = tl.where(mask, qk, -float("inf"))
    w = exp2(qk - b_m)
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
    p = w * inv_d1

    a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    delta = a - delta_b
    bart_minus_rk = b_bart - rk
    gl = p * (a - delta_t + bart_minus_rk * delta)
    gu = -p * delta

    grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32) * scale
    grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32)
    grad_k_acc = tl.dot(tl.trans(gl * scale).to(b_q.dtype), b_q, out_dtype=tl.float32)
    grad_k_acc += tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32)
    weights = p * (1.0 + bart_minus_rk)
    grad_v_acc = tl.dot(tl.trans(weights).to(b_grad_o.dtype), b_grad_o, out_dtype=tl.float32)

    p_grad_q = tl.make_block_ptr(
        grad_q + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_r = tl.make_block_ptr(
        grad_r + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_k = tl.make_block_ptr(
        grad_k_buf + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_v = tl.make_block_ptr(
        grad_v_buf + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr_short_multi(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    grad_q,
    grad_r,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    """Compute dQ/dR for a small fixed number of dense tiles."""
    i_t = tl.program_id(0).to(tl.int32)
    i_bh = tl.program_id(1).to(tl.int32)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int32)
    offs_k = tl.arange(0, BK).to(tl.int32)
    row_mask = offs_t[:, None] < T
    head_mask = offs_k[None, :] < K
    q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]

    b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_o = tl.load(o + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
    b_barv = tl.load(barv + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
    b_grad_o = tl.load(grad_o + q_offsets, mask=row_mask & head_mask, other=0.0)
    scalar_offsets = (bos + offs_t) * HQ + i_hq
    b_d1 = tl.load(d1 + scalar_offsets, mask=offs_t < T, other=1.0)[:, None]
    b_bart = tl.load(bart + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]
    b_m = tl.load(m + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]

    delta_t = tl.sum(b_grad_o.to(tl.float32) * b_o, axis=1, keep_dims=True)
    delta_b = tl.sum(b_grad_o.to(tl.float32) * b_barv, axis=1, keep_dims=True)
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_s in range(0, tl.minimum(i_t + 1, NT)):
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int32)
        col_mask = offs_s[:, None] < T
        kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
        b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)
        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - delta_b
        bart_minus_rk = b_bart - tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - delta_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_r_acc)

    tl.store(grad_q + q_offsets, grad_q_acc.to(grad_q.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(grad_r + q_offsets, grad_r_acc.to(grad_r.dtype.element_ty), mask=row_mask & head_mask)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dkv_short_multi(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    grad_k_buf,
    grad_v_buf,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    """Compute dK/dV without atomics for a small fixed number of dense tiles."""
    i_s = tl.program_id(0).to(tl.int32)
    i_bh = tl.program_id(1).to(tl.int32)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_s = i_s * BT + tl.arange(0, BT).to(tl.int32)
    offs_k = tl.arange(0, BK).to(tl.int32)
    col_mask = offs_s[:, None] < T
    head_mask = offs_k[None, :] < K
    kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
    b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
    b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

    grad_k_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_t in range(i_s, NT):
        offs_t = (i_t * BT + tl.arange(0, BT)).to(tl.int32)
        row_mask = offs_t[:, None] < T
        q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
        b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
        b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)
        b_o = tl.load(o + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
        b_barv = tl.load(barv + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
        b_grad_o = tl.load(grad_o + q_offsets, mask=row_mask & head_mask, other=0.0)
        scalar_offsets = (bos + offs_t) * HQ + i_hq
        b_d1 = tl.load(d1 + scalar_offsets, mask=offs_t < T, other=1.0)[:, None]
        b_bart = tl.load(bart + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]
        b_m = tl.load(m + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        delta_t = tl.sum(b_grad_o.to(tl.float32) * b_o, axis=1, keep_dims=True)
        delta_b = tl.sum(b_grad_o.to(tl.float32) * b_barv, axis=1, keep_dims=True)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * tl.where(row_mask, 1.0 / b_d1, 0.0)
        a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - delta_b
        bart_minus_rk = b_bart - tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - delta_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1.0 + bart_minus_rk)
        grad_v_acc = tl.dot(
            tl.trans(weights).to(b_grad_o.dtype),
            b_grad_o,
            out_dtype=tl.float32,
            acc=grad_v_acc,
        )

    grad_offsets = ((bos + offs_s[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    tl.store(
        grad_k_buf + grad_offsets,
        grad_k_acc.to(grad_k_buf.dtype.element_ty),
        mask=col_mask & head_mask,
    )
    tl.store(
        grad_v_buf + grad_offsets,
        grad_v_acc.to(grad_v_buf.dtype.element_ty),
        mask=col_mask & head_mask,
    )


@libtuner(
    configs=PARALLAX_PREPROCESS_CONFIGS,
    key=["T", "K", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_preprocess_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_preprocess(
    grad_o,
    o,
    barv,
    delta_t,
    delta_b,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BP: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int32), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int32)

    row_offset = i_t * BP
    o_t = row_offset + tl.arange(0, BP)
    o_k = tl.arange(0, BK)
    m_tk = (o_t[:, None] < T) & (o_k[None, :] < K)
    m_t = o_t[:, None] < T
    p_grad_o = grad_o + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_o = o + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_barv = barv + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_t = delta_t + bos * HQ + i_hq + o_t[:, None] * HQ
    p_b = delta_b + bos * HQ + i_hq + o_t[:, None] * HQ

    b_grad_o = tl.load(p_grad_o, mask=m_tk, other=0.0).to(tl.float32)
    b_o = tl.load(p_o, mask=m_tk, other=0.0).to(tl.float32)
    b_barv = tl.load(p_barv, mask=m_tk, other=0.0).to(tl.float32)

    b_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
    b_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)

    tl.store(p_t, b_t, mask=m_t)
    tl.store(p_b, b_b, mask=m_t)


@triton.jit(do_not_specialize=["T"])
def _parallel_parallax_bwd_kernel_dqr(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FUSED_PREPROCESS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int32), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int32)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_q = grad_q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_r = grad_r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
    b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
    b_m = tl.load(p_m, mask=row_mask, other=0.0)
    grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)
    if FUSED_PREPROCESS:
        p_o = o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_barv = barv + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        grad_o_tile_fp32 = grad_o_tile.to(tl.float32)
        b_o = tl.load(p_o, mask=m_qk, other=0.0).to(tl.float32)
        b_barv = tl.load(p_barv, mask=m_qk, other=0.0).to(tl.float32)
        b_t = tl.sum(grad_o_tile_fp32 * b_o, axis=1, keep_dims=True)
        b_b = tl.sum(grad_o_tile_fp32 * b_barv, axis=1, keep_dims=True)
        tl.store(p_t, b_t, mask=row_mask)
        tl.store(p_b, b_b, mask=row_mask)
    else:
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)

    # Phase 0: left-border blocks (SWA only).
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id.to(tl.int32) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        # Finish dR before materializing rk/gl. This shortens the live ranges
        # of several [BT, BS] matrices and substantially reduces register
        # pressure in the long-sequence kernel.
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    # Phase A: safe blocks (no mask).
    for _ in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        o_kv = _.to(tl.int32) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int32) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    grad_q_acc = scale * grad_q_acc

    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), mask=m_qk)
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), mask=m_qk)


@triton.jit
def parallel_parallax_bwd_kernel_dqr_dense_aligned(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Pure-Triton dense DQR specialization for aligned causal D=128."""
    tl.static_assert(T % BT == 0)
    tl.static_assert(BT % BS == 0)
    tl.static_assert(K == BK)

    i_t = tl.program_id(0).to(tl.int32)
    i_bh = tl.program_id(1).to(tl.int32)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT).to(tl.int32)
    o_k = tl.arange(0, BK).to(tl.int32)

    q_offsets = ((bos + row_indices[:, None]) * HQ + i_hq) * K + o_k[None, :]
    scalar_offsets = (bos + row_indices[:, None]) * HQ + i_hq
    b_q = tl.load(q + q_offsets)
    b_r = tl.load(r + q_offsets)
    b_d1 = tl.load(d1 + scalar_offsets)
    b_bart = tl.load(bart + scalar_offsets)
    b_m = tl.load(m + scalar_offsets)
    b_t = tl.load(delta_t + scalar_offsets)
    b_b = tl.load(delta_b + scalar_offsets)
    grad_o_tile = tl.load(grad_o + q_offsets)

    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    inv_d1 = 1.0 / b_d1
    scale_log2 = scale * 1.4426950216

    # All blocks strictly left of this query tile are fully dense.  Alignment
    # proves that the q/r/scalar/K/V loads have neither row nor head tails, so
    # the generic masks only add predicates and dependency-chain pressure.
    num_safe_blocks = row_offset // BS
    for col_block_id in range(0, num_safe_blocks):
        col_indices = col_block_id.to(tl.int32) * BS + tl.arange(0, BS).to(tl.int32)
        kv_offsets = ((bos + col_indices[:, None]) * H + i_h) * K + o_k[None, :]
        b_k = tl.load(k + kv_offsets)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        p = tl.math.exp2(qk - b_m) * inv_d1
        b_v = tl.load(v + kv_offsets)
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(
            gl.to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_q_acc,
        )

    # BT=64 and BS=32 leave exactly two diagonal-intersecting blocks.  Their
    # loads remain aligned; only the causal score predicate is required.
    for border_offset in tl.static_range(0, BT // BS):
        col_indices = (num_safe_blocks + border_offset).to(tl.int32) * BS + tl.arange(0, BS).to(tl.int32)
        kv_offsets = ((bos + col_indices[:, None]) * H + i_h) * K + o_k[None, :]
        b_k = tl.load(k + kv_offsets)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(
            row_indices[:, None] >= col_indices[None, :],
            qk,
            -float("inf"),
        )
        p = tl.math.exp2(qk - b_m) * inv_d1
        b_v = tl.load(v + kv_offsets)
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(
            gl.to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_q_acc,
        )

    tl.store(
        grad_q + q_offsets,
        (scale * grad_q_acc).to(grad_q.dtype.element_ty),
    )
    tl.store(grad_r + q_offsets, grad_r_acc.to(grad_r.dtype.element_ty))


@libtuner(
    configs=PARALLAX_DQR_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr_fused(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    _parallel_parallax_bwd_kernel_dqr(
        q,
        r,
        k,
        v,
        o,
        barv,
        d1,
        bart,
        m,
        delta_t,
        delta_b,
        grad_o,
        grad_q,
        grad_r,
        scale,
        cu_seqlens,
        chunk_indices,
        T,
        HQ,
        H,
        G,
        K,
        BK,
        WINDOW_SIZE_LEFT,
        BT,
        BS,
        IS_VARLEN,
        True,
    )


@libtuner(
    configs=PARALLAX_DQR_SPLIT_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    _parallel_parallax_bwd_kernel_dqr(
        q,
        r,
        k,
        v,
        q,
        q,
        d1,
        bart,
        m,
        delta_t,
        delta_b,
        grad_o,
        grad_q,
        grad_r,
        scale,
        cu_seqlens,
        chunk_indices,
        T,
        HQ,
        H,
        G,
        K,
        BK,
        WINDOW_SIZE_LEFT,
        BT,
        BS,
        IS_VARLEN,
        False,
    )


@triton.jit
def _parallel_parallax_dkv_dense_aligned_accumulate_row(
    q,
    r,
    grad_o,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    b_k,
    b_v,
    bos,
    i_hq,
    row_indices,
    col_indices,
    scale,
    grad_k_acc,
    grad_v_acc,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    CAUSAL_BORDER: tl.constexpr,
):
    """One aligned DKV row tile with no dense-tail load masks."""
    o_k = tl.arange(0, BK)
    q_offsets = ((bos + row_indices[:, None]) * HQ + i_hq) * K + o_k[None, :]
    scalar_offsets = (bos + row_indices[:, None]) * HQ + i_hq

    b_q = tl.load(q + q_offsets)
    b_r = tl.load(r + q_offsets)
    grad_o_tile = tl.load(grad_o + q_offsets)
    b_d1 = tl.load(d1 + scalar_offsets)
    b_bart = tl.load(bart + scalar_offsets)
    b_m = tl.load(m + scalar_offsets)
    b_t = tl.load(delta_t + scalar_offsets)
    b_b = tl.load(delta_b + scalar_offsets)

    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32)
    qk *= scale * 1.4426950216
    if CAUSAL_BORDER:
        qk = tl.where(
            row_indices[:, None] >= col_indices[None, :],
            qk,
            -float("inf"),
        )
    p = exp2(qk - b_m) / b_d1
    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
    delta = a - b_b
    bart_minus_rk = b_bart - rk
    gl = p * (a - b_t + bart_minus_rk * delta) * scale
    gu = -p * delta
    grad_k_acc = tl.dot(
        tl.trans(gl).to(b_q.dtype),
        b_q,
        out_dtype=tl.float32,
        acc=grad_k_acc,
    )
    grad_k_acc = tl.dot(
        tl.trans(gu).to(b_r.dtype),
        b_r,
        out_dtype=tl.float32,
        acc=grad_k_acc,
    )
    weights = p * (1.0 + bart_minus_rk)
    grad_v_acc = tl.dot(
        tl.trans(weights).to(grad_o_tile.dtype),
        grad_o_tile,
        out_dtype=tl.float32,
        acc=grad_v_acc,
    )
    return grad_k_acc, grad_v_acc


@triton.jit
def parallel_parallax_bwd_kernel_dkv_dense_aligned(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """Pure-Triton dense causal DKV specialization for aligned D=128."""
    tl.static_assert(K == BK)
    tl.static_assert(BT == BS)
    tl.static_assert(T % BT == 0)

    i_col = tl.program_id(0).to(tl.int32)
    i_bh = tl.program_id(1).to(tl.int32)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    col_indices = i_col * BS + tl.arange(0, BS).to(tl.int32)
    o_k = tl.arange(0, BK).to(tl.int32)
    kv_offsets = ((bos + col_indices[:, None]) * H + i_h) * K + o_k[None, :]
    out_offsets = ((bos + col_indices[:, None]) * HQ + i_hq) * K + o_k[None, :]
    b_k = tl.load(k + kv_offsets)
    b_v = tl.load(v + kv_offsets)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)

    # BT == BS gives one causal-border row tile. Every later row tile is
    # provably dense and aligned, so all row/head load masks are omitted.
    border_rows = i_col * BT + tl.arange(0, BT).to(tl.int32)
    grad_k_acc, grad_v_acc = _parallel_parallax_dkv_dense_aligned_accumulate_row(
        q,
        r,
        grad_o,
        d1,
        bart,
        m,
        delta_t,
        delta_b,
        b_k,
        b_v,
        bos,
        i_hq,
        border_rows,
        col_indices,
        scale,
        grad_k_acc,
        grad_v_acc,
        HQ,
        K,
        BK,
        True,
    )
    for row_block in range(i_col + 1, T // BT):
        row_indices = row_block * BT + tl.arange(0, BT).to(tl.int32)
        grad_k_acc, grad_v_acc = _parallel_parallax_dkv_dense_aligned_accumulate_row(
            q,
            r,
            grad_o,
            d1,
            bart,
            m,
            delta_t,
            delta_b,
            b_k,
            b_v,
            bos,
            i_hq,
            row_indices,
            col_indices,
            scale,
            grad_k_acc,
            grad_v_acc,
            HQ,
            K,
            BK,
            False,
        )

    tl.store(
        grad_k + out_offsets,
        grad_k_acc.to(grad_k.dtype.element_ty),
    )
    tl.store(
        grad_v + out_offsets,
        grad_v_acc.to(grad_v.dtype.element_ty),
    )


@libtuner(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dkv(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int32)
    i_bh = tl.program_id(1).to(tl.int32)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int32)
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = i_t * BS
    col_indices = col_offset + tl.arange(0, BS)

    start_row_block = col_offset // BT

    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_k = grad_k + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id.to(tl.int32) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        # Consume the independent R contribution before forming RK/GL.
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        bart_minus_rk = b_bart - rk
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id.to(tl.int32) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        # Consume the independent R contribution before forming RK/GL.
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        bart_minus_rk = b_bart - rk
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)

    # Phase C: window-border row blocks (SWA only).
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id.to(tl.int32) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        # Consume the independent R contribution before forming RK/GL.
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        bart_minus_rk = b_bart - rk
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@libtuner(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dkv_grouped(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Accumulate all query heads in a GQA group directly into one KV head."""
    i_t, i_bh = tl.program_id(0).to(tl.int32), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int32)
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = i_t * BS
    col_indices = col_offset + tl.arange(0, BS)
    start_row_block = col_offset // BT
    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_k = grad_k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)

    for i_g in range(0, G):
        i_hq = i_h * G + i_g

        # Phase A: causal-border row blocks.
        for row_block_id in range(start_row_block, causal_end):
            row_offset = row_block_id.to(tl.int32) * BT
            row_indices = row_offset + tl.arange(0, BT)
            row_mask = row_indices[:, None] < T
            m_qk = row_mask & m_k[None, :]
            p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_qk, other=0.0)
            b_r = tl.load(p_r, mask=m_qk, other=0.0)
            b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
            b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
            b_m = tl.load(p_m, mask=row_mask, other=0.0)
            b_t = tl.load(p_t, mask=row_mask, other=0.0)
            b_b = tl.load(p_b, mask=row_mask, other=0.0)
            grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

            qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            if WINDOW_SIZE_LEFT >= 0:
                mask = (
                    (row_indices[:, None] >= col_indices[None, :])
                    & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                    & row_mask
                    & (col_indices[None, :] < T)
                )
            else:
                mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
            qk = tl.where(mask, qk, -float("inf"))
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            # Consume the independent R contribution before forming RK/GL.
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            bart_minus_rk = b_bart - rk
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)

        # Phase B: safe row blocks (no causal/col/window mask).
        for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
            row_offset = row_block_id.to(tl.int32) * BT
            row_indices = row_offset + tl.arange(0, BT)
            row_mask = row_indices[:, None] < T
            m_qk = row_mask & m_k[None, :]
            p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_qk, other=0.0)
            b_r = tl.load(p_r, mask=m_qk, other=0.0)
            b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
            b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
            b_m = tl.load(p_m, mask=row_mask, other=0.0)
            b_t = tl.load(p_t, mask=row_mask, other=0.0)
            b_b = tl.load(p_b, mask=row_mask, other=0.0)
            grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

            qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            # Consume the independent R contribution before forming RK/GL.
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            bart_minus_rk = b_bart - rk
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)

        # Phase C: window-border row blocks (SWA only).
        for row_block_id in range(window_border_start, num_row_blocks):
            row_offset = row_block_id.to(tl.int32) * BT
            row_indices = row_offset + tl.arange(0, BT)
            row_mask = row_indices[:, None] < T
            m_qk = row_mask & m_k[None, :]
            p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_qk, other=0.0)
            b_r = tl.load(p_r, mask=m_qk, other=0.0)
            b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
            b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
            b_m = tl.load(p_m, mask=row_mask, other=0.0)
            b_t = tl.load(p_t, mask=row_mask, other=0.0)
            b_b = tl.load(p_b, mask=row_mask, other=0.0)
            grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

            qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            mask = (
                (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
            qk = tl.where(mask, qk, -float("inf"))
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            # Consume the independent R contribution before forming RK/GL.
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            bart_minus_rk = b_bart - rk
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.jit(do_not_specialize=["N"])
def parallel_parallax_bwd_kernel_reduce_gqa(
    grad_k_buf,
    grad_v_buf,
    grad_k,
    grad_v,
    N,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    USE_GRID_REMAP: tl.constexpr,
):
    """Fold each scalar-selected head row through regular D segments."""
    if USE_GRID_REMAP:
        program_id = tl.program_id(0) + tl.num_programs(0) * (tl.program_id(1) + tl.num_programs(1) * tl.program_id(2))
    else:
        program_id = tl.program_id(0)
    dims = tl.arange(0, BLOCK).to(tl.int32)
    # Retain the serial group accumulation order and bounded output work.
    for row in range(ROWS):
        row_head = program_id * ROWS + row
        base = (row_head // H * HQ + row_head % H * G) * K
        offsets = row_head * K + dims
        mask = (dims < K) & (offsets < N)
        grad_k_acc = tl.zeros((BLOCK,), tl.float32)
        grad_v_acc = tl.zeros((BLOCK,), tl.float32)
        for group in range(G):
            source = base + group * K + dims
            grad_k_acc += tl.load(grad_k_buf + source, mask, other=0.0)
            grad_v_acc += tl.load(grad_v_buf + source, mask, other=0.0)
        tl.store(grad_k + offsets, grad_k_acc, mask)
        tl.store(grad_v + offsets, grad_v_acc, mask)


def parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax forward (Triton). `(B, T, HQ, D)` / packed `(1, T_total, HQ, D)` inputs.

    Returns `(o, barv, d1, bart, m)`: `o`/`barv` in the input dtype and layout;
    `d1`/`bart`/`m` are fp32 per-(position, query-head) scalars `(B, T, HQ)`.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _BLOCK_T
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    stats = torch.empty((3, B, T, HQ), device=q.device, dtype=torch.float32)
    d1, bart, m = stats.unbind(0)

    if cu_seqlens is None:
        # BT only partitions query rows.  The long kernel autotunes BS
        # independently for the key/value columns it scans.
        NQ = triton.cdiv(T, BT)
        query_grid = (NQ, B * HQ)
        if T <= BT:
            parallel_parallax_fwd_kernel_short[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                num_warps=4,
                num_stages=1,
            )
        elif NQ <= _SHORT_MULTI_MAX_TILES:
            parallel_parallax_fwd_kernel_short_multi[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                NT=NQ,
                num_warps=1,
                num_stages=2,
            )
        else:
            parallel_parallax_fwd_kernel[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                scale,
                cu_seqlens,
                chunk_indices,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
            )
    else:
        # Packed chunks are created with BT, so the autotune pruner keeps
        # BS == BT for this path.  Dense inputs above remain fully decoupled.
        chunk_grid = (len(chunk_indices), B * HQ)
        # FlagTree's autotuner mutates constexpr launch kwargs on this packed
        # path and can pass BT twice on GCU.  Bypass only the autotuner layer;
        # keep the heuristic and launch the S60-safe fixed tile explicitly.
        parallel_parallax_fwd_kernel.fn[chunk_grid](
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            scale,
            cu_seqlens,
            chunk_indices,
            T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            BK=BK,
            WINDOW_SIZE_LEFT=window_size_left,
            BT=BT,
            BS=BT,
            num_warps=1,
            num_stages=1,
        )
    return o, barv, d1, bart, m


def parallel_parallax_bwd(
    q, r, k, v, o, barv, d1, bart, m, grad_o, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1
):
    """Parallax backward (Triton). Returns grads matching `q, r, k, v`."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _BLOCK_T
    NQ = triton.cdiv(T, BT)
    use_dense_aligned = (
        q.device.type == "gcu"
        and q.dtype in (torch.bfloat16, torch.float16)
        and cu_seqlens is None
        and K == 128
        and T % BT == 0
        and NQ > 16
        and T <= _DENSE_ALIGNED_MAX_T
        and window_size_left < 0
    )
    dkv_programs = (len(chunk_indices) if cu_seqlens is not None else NQ) * B * H
    num_sms = get_num_sms(q.device.index)
    group_dkv_heads = G in (4, 8) or (G == 2 and (cu_seqlens is not None or T >= _GROUPED_DKV_G2_MIN_T))
    use_grouped_dkv = (
        # Packed T is the total token count, so varlen stays on the grouped
        # path whenever the device has enough work to occupy its clusters.
        group_dkv_heads
        # Only the generic dense branch launches grouped DKV. Fixed short
        # kernels still require per-query-head gradient buffers.
        and (cu_seqlens is not None or NQ > _SHORT_MULTI_MAX_TILES)
        and num_sms > 0
        and dkv_programs >= num_sms
    )

    if cu_seqlens is None and NQ <= _SHORT_MULTI_MAX_TILES and G == 1:
        # Fixed short MHA kernels write four equally sized, disjoint outputs.
        # Share one allocation while retaining contiguous per-input gradients.
        grad_q, grad_r, grad_k, grad_v = torch.empty((4, B, T, HQ, K), device=q.device, dtype=q.dtype).unbind(0)
        grad_k_buf, grad_v_buf = grad_k, grad_v
    else:
        grad_q = torch.empty_like(q)
        grad_r = torch.empty_like(r)
        if use_grouped_dkv or G == 1:
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
            grad_k_buf, grad_v_buf = grad_k, grad_v
        else:
            # Short sequences retain one CTA per query head for occupancy, then
            # fold their private dK/dV tiles back to the shared KV-head axis.
            grad_k = grad_v = None
            grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
            grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    if cu_seqlens is None:
        query_grid = (NQ, B * HQ)
        if T <= BT:
            parallel_parallax_bwd_kernel_short[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                grad_o,
                grad_q,
                grad_r,
                grad_k_buf,
                grad_v_buf,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                num_warps=4,
                num_stages=1,
            )
        elif NQ <= _SHORT_MULTI_MAX_TILES:
            parallel_parallax_bwd_kernel_dqr_short_multi[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                grad_o,
                grad_q,
                grad_r,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                NT=NQ,
                num_warps=1,
                num_stages=2,
            )
            parallel_parallax_bwd_kernel_dkv_short_multi[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                grad_o,
                grad_k_buf,
                grad_v_buf,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                NT=NQ,
                num_warps=1,
                num_stages=2,
            )
        else:
            deltas = torch.empty((2, B, T, HQ), device=q.device, dtype=torch.float32)
            delta_t, delta_b = deltas.unbind(0)

            # dQ/dR owns BT query rows, whereas dK/dV owns BS key/value
            # columns.  Its grid must therefore be derived from tuned BS.
            def dkv_grid(meta):
                return triton.cdiv(T, meta["BS"]), B * HQ

            def grouped_dkv_grid(meta):
                return triton.cdiv(T, meta["BS"]), B * H

            # Small heads can keep the two scalar reductions in the DQR
            # program without excessive register pressure.  Larger heads
            # retain the separately autotuned preprocess pass.
            use_split_dqr = K >= 128
            if use_split_dqr:

                def preprocess_grid(meta):
                    return triton.cdiv(T, meta["BP"]), B * HQ

                parallel_parallax_bwd_kernel_preprocess[preprocess_grid](
                    grad_o,
                    o,
                    barv,
                    delta_t,
                    delta_b,
                    cu_seqlens,
                    chunk_indices,
                    T,
                    HQ=HQ,
                    K=K,
                    BK=BK,
                    BT=BT,
                )
                if use_dense_aligned:
                    parallel_parallax_bwd_kernel_dqr_dense_aligned[query_grid](
                        q,
                        r,
                        k,
                        v,
                        d1,
                        bart,
                        m,
                        delta_t,
                        delta_b,
                        grad_o,
                        grad_q,
                        grad_r,
                        scale,
                        T,
                        HQ=HQ,
                        H=H,
                        G=G,
                        K=K,
                        BK=BK,
                        BT=BT,
                        BS=32,
                        num_warps=1,
                        num_stages=2,
                    )
                else:
                    parallel_parallax_bwd_kernel_dqr[query_grid](
                        q,
                        r,
                        k,
                        v,
                        d1,
                        bart,
                        m,
                        delta_t,
                        delta_b,
                        grad_o,
                        grad_q,
                        grad_r,
                        scale,
                        cu_seqlens,
                        chunk_indices,
                        T,
                        HQ=HQ,
                        H=H,
                        G=G,
                        K=K,
                        WINDOW_SIZE_LEFT=window_size_left,
                        BT=BT,
                    )
            else:
                parallel_parallax_bwd_kernel_dqr_fused[query_grid](
                    q,
                    r,
                    k,
                    v,
                    o,
                    barv,
                    d1,
                    bart,
                    m,
                    delta_t,
                    delta_b,
                    grad_o,
                    grad_q,
                    grad_r,
                    scale,
                    cu_seqlens,
                    chunk_indices,
                    T,
                    HQ=HQ,
                    H=H,
                    G=G,
                    K=K,
                    WINDOW_SIZE_LEFT=window_size_left,
                    BT=BT,
                )
            if use_dense_aligned and not use_grouped_dkv:
                parallel_parallax_bwd_kernel_dkv_dense_aligned[(T // BT, B * HQ)](
                    q,
                    r,
                    k,
                    v,
                    d1,
                    bart,
                    m,
                    delta_t,
                    delta_b,
                    grad_o,
                    grad_k_buf,
                    grad_v_buf,
                    scale,
                    T,
                    HQ=HQ,
                    H=H,
                    G=G,
                    K=K,
                    BK=BK,
                    BT=BT,
                    BS=64,
                    num_warps=1,
                    num_stages=1,
                )
            elif use_grouped_dkv:
                parallel_parallax_bwd_kernel_dkv_grouped[grouped_dkv_grid](
                    q,
                    r,
                    k,
                    v,
                    d1,
                    bart,
                    m,
                    delta_t,
                    delta_b,
                    grad_o,
                    grad_k,
                    grad_v,
                    scale,
                    cu_seqlens,
                    chunk_indices,
                    T,
                    HQ=HQ,
                    H=H,
                    G=G,
                    K=K,
                    WINDOW_SIZE_LEFT=window_size_left,
                    BT=BT,
                )
            else:
                parallel_parallax_bwd_kernel_dkv[dkv_grid](
                    q,
                    r,
                    k,
                    v,
                    d1,
                    bart,
                    m,
                    delta_t,
                    delta_b,
                    grad_o,
                    grad_k_buf,
                    grad_v_buf,
                    scale,
                    cu_seqlens,
                    chunk_indices,
                    T,
                    HQ=HQ,
                    H=H,
                    G=G,
                    K=K,
                    WINDOW_SIZE_LEFT=window_size_left,
                    BT=BT,
                )
    else:
        # chunk_indices is indexed in BT-sized units; the autotune pruners
        # intentionally pin BP and BS to BT for packed variable lengths.
        chunk_grid = (len(chunk_indices), B * HQ)
        grouped_chunk_grid = (len(chunk_indices), B * H)
        deltas = torch.empty((2, B, T, HQ), device=q.device, dtype=torch.float32)
        delta_t, delta_b = deltas.unbind(0)
        parallel_parallax_bwd_kernel_preprocess.fn[chunk_grid](
            grad_o,
            o,
            barv,
            delta_t,
            delta_b,
            cu_seqlens,
            chunk_indices,
            T,
            HQ=HQ,
            K=K,
            BK=BK,
            BT=BT,
            BP=BT,
            num_warps=1,
            num_stages=1,
        )
        parallel_parallax_bwd_kernel_dqr.fn[chunk_grid](
            q,
            r,
            k,
            v,
            d1,
            bart,
            m,
            delta_t,
            delta_b,
            grad_o,
            grad_q,
            grad_r,
            scale,
            cu_seqlens,
            chunk_indices,
            T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            BK=BK,
            WINDOW_SIZE_LEFT=window_size_left,
            BT=BT,
            BS=BT,
            num_warps=1,
            num_stages=1,
        )
        if use_grouped_dkv:
            parallel_parallax_bwd_kernel_dkv_grouped.fn[grouped_chunk_grid](
                q,
                r,
                k,
                v,
                d1,
                bart,
                m,
                delta_t,
                delta_b,
                grad_o,
                grad_k,
                grad_v,
                scale,
                cu_seqlens,
                chunk_indices,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                BS=BT,
                num_warps=1,
                num_stages=1,
            )
        else:
            parallel_parallax_bwd_kernel_dkv.fn[chunk_grid](
                q,
                r,
                k,
                v,
                d1,
                bart,
                m,
                delta_t,
                delta_b,
                grad_o,
                grad_k_buf,
                grad_v_buf,
                scale,
                cu_seqlens,
                chunk_indices,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                BS=BT,
                num_warps=1,
                num_stages=1,
            )
    if G > 1 and not use_grouped_dkv:
        if grad_k is None:
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
        num_grad_k_elements = grad_k.numel()
        reduction_block = triton.next_power_of_2(K)
        reduction_rows = max(1, 256 // reduction_block)
        reduction_programs = triton.cdiv(num_grad_k_elements // K, reduction_rows)
        use_reduction_grid_remap = requires_grid_remap(reduction_programs)
        reduction_grid = make_grid_2d(
            reduction_programs,
            1,
            remap=use_reduction_grid_remap,
        )
        parallel_parallax_bwd_kernel_reduce_gqa[reduction_grid](
            grad_k_buf,
            grad_v_buf,
            grad_k,
            grad_v,
            num_grad_k_elements,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            BLOCK=reduction_block,
            ROWS=reduction_rows,
            USE_GRID_REMAP=use_reduction_grid_remap,
            num_warps=1,
        )
    return grad_q, grad_r, grad_k, grad_v


def _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens):
    tensors = {"q": q, "r": r, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, T, H, D], got {tuple(tensor.shape)}")
        if tensor.device.type != "gcu":
            raise ValueError(f"{name} must be an Enflame GCU tensor, got device={tensor.device}")

    if r.shape != q.shape:
        raise ValueError(f"r shape {tuple(r.shape)} must match q shape {tuple(q.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}")
    if any(tensor.device != q.device for tensor in (r, k, v)):
        raise ValueError("q, r, k, and v must be on the same GCU device")
    if any(tensor.dtype != q.dtype for tensor in (r, k, v)):
        raise TypeError("q, r, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")

    B, T, HQ, K = q.shape
    if k.shape[0] != B or k.shape[1] != T or k.shape[3] != K:
        raise ValueError(
            f"k/v shape {tuple(k.shape)} is incompatible with q/r shape {tuple(q.shape)}; B, T, and D must match"
        )
    H = k.shape[2]
    if T <= 0 or H <= 0 or HQ <= 0:
        raise ValueError("sequence length and query/KV head counts must be positive")
    if HQ % H != 0:
        raise ValueError(f"GQA requires HQ % H == 0, got HQ={HQ}, H={H}")
    if not 16 <= K <= 256:
        raise ValueError(f"head dimension must be in [16, 256], got D={K}")
    if window_size is not None and (not isinstance(window_size, int) or window_size <= 0):
        raise ValueError(f"window_size must be a positive integer or None, got {window_size!r}")

    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor or None")
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using cu_seqlens. "
                "Flatten variable-length inputs before processing."
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same GCU device as q")


class ParallaxFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = prepare_chunk_indices(cu_seqlens, _BLOCK_T) if cu_seqlens is not None else None
        o, barv, d1, bart, m = parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @input_guard
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        gq, gr, gk, gv = parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            do,
            ctx.scale,
            ctx.cu_seqlens,
            ctx.chunk_indices,
            ctx.window_size_left,
        )
        return gq, gr, gk, gv, None, None, None


def parallel_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Causal Parallax with autograd, implemented with Enflame Triton kernels.

    Args:
        q, r: Query tensors [B, T, HQ, D]; r must not be pre-scaled.
        k, v: Key/value tensors [B, T, H, D], with HQ divisible by H.
        scale: Applied only to QK logits; None uses 1 / sqrt(D).
        window_size: Attend to [i - window_size + 1, i]; None or -1 uses full causal attention.
        cu_seqlens: Optional int32/int64 offsets [N+1] for packed sequences with B=1.

    Returns:
        Output tensor of shape [B, T, HQ, D].
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"parallel_parallax got unexpected keyword argument(s): {unexpected}")
    if isinstance(window_size, int) and window_size == -1:
        window_size = None
    _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens)
    if cu_seqlens is not None and cu_seqlens.dtype != torch.int32:
        # S60 does not lower int64 index arithmetic.  Sequence offsets used by
        # this backend are bounded by the int32 address range.
        cu_seqlens = cu_seqlens.to(dtype=torch.int32)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
