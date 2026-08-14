# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import copy
from contextlib import nullcontext
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
from flaggems_vllm.ops.FLA.utils import input_guard

try:
    import triton.experimental.tle.language as tle
except ImportError:
    # TLE is an optional FlagTree extension.  Keep the regular Triton kernels
    # importable so unsupported installations can use the generic path.
    tle = None

HAS_TLE = tle is not None
HAS_TLE_CLUSTER = bool(
    HAS_TLE
    and hasattr(tle, "device_mesh")
    and hasattr(tle, "shard_id")
    and hasattr(tle, "remote")
    and hasattr(tle, "distributed_barrier")
)


@lru_cache(maxsize=None)
def _is_nvidia_blackwell(device_index: int | None) -> bool:
    try:
        return torch.cuda.get_device_capability(device_index)[0] in (10, 12)
    except Exception:
        return False


@lru_cache(maxsize=None)
def _is_nvidia_hopper(device_index: int | None) -> bool:
    try:
        return torch.cuda.get_device_capability(device_index)[0] == 9
    except Exception:
        return False


@lru_cache(maxsize=None)
def _get_num_sms(device_index: int | None) -> int:
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        return 0


@lru_cache(maxsize=None)
def _get_parallax_bwd_streams(
    device_index: int,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    with torch.cuda.device(device_index):
        return (
            torch.cuda.Stream(device=device_index),
            torch.cuda.Stream(device=device_index),
        )


@lru_cache(maxsize=1)
def _get_parallax_gqa_cluster_mesh_2():
    if not HAS_TLE_CLUSTER:
        raise RuntimeError("The installed Triton/TLE build lacks cluster support")
    return tle.device_mesh({"block_cluster": [("query_head", 2)]})


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


_SHARED_MEMORY_BY_ARCH = {
    'ada': 101376,
    'ampere': 166912,
    'hopper': 232448,
}


@lru_cache(maxsize=None)
def _check_shared_mem(arch: str = 'none', tensor_idx: int = 0) -> bool:
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            tensor_idx,
        )
        max_shared_memory = properties['max_shared_mem']
        required_shared_memory = _SHARED_MEMORY_BY_ARCH.get(
            arch.lower(),
            102400,
        )
        return max_shared_memory >= required_shared_memory
    except Exception:
        return False


def _block_size(head_dim: int, device_index: int) -> int:
    # The fused short backward keeps attention scores and all four gradient
    # accumulators live at once, so Hopper uses a smaller tile than the baseline
    # kernel. Varlen still uses this same BT to keep chunk_indices consistent.
    if _is_nvidia_blackwell(device_index):
        return 128
    if _check_shared_mem('hopper', device_index):
        return 64 if head_dim <= 128 else 32
    return 64


def _make_parallax_configs(tile_pairs):
    """Build independent autotune spaces for each long-sequence pass."""
    configs = []
    for block_k, block_s in tile_pairs:
        if block_k == 64 and block_s == 16:
            # A narrow score tile is specifically useful when long-sequence
            # kernels hit the 255-register ceiling. One-stage variants keep
            # its shared-memory footprint low enough for more resident CTAs.
            launch_configs = ((2, 1), (2, 2), (4, 1), (4, 2))
        elif block_k == 64:
            launch_configs = ((2, 2), (4, 1), (4, 2), (8, 2))
        else:
            launch_configs = ((4, 1), (4, 2), (8, 1), (8, 2))
        for num_warps, num_stages in launch_configs:
            configs.append(
                triton.Config(
                    {"BK": block_k, "BS": block_s},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


# BT fixes the query-row grid. BS is tuned independently because FWD/DQR
# accumulate over query rows, while DKV carries two [BS, BK] fp32 accumulators.
# DQR additionally tries BS=16 for D=64 to relieve its short-lived score-tile
# pressure. Forward retains BS>=32: its two persistent output accumulators make
# the extra loop iterations of BS=16 a net loss on Hopper.
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

_PARALLAX_FWD_TILES = tuple(
    tile
    for tile in _PARALLAX_DQR_TILES
    if tile != (64, 16)
)

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

PARALLAX_FWD_CONFIGS = _make_parallax_configs(_PARALLAX_FWD_TILES)
PARALLAX_DQR_CONFIGS = _make_parallax_configs(_PARALLAX_DQR_TILES)
PARALLAX_DQR_SPLIT_CONFIGS = _make_parallax_configs(_PARALLAX_DQR_TILES)
# DKV's triangular row traversal is substantially longer than the FWD/DQR
# inner loops.  On H100, D=128/BS=64 with eight warps benefits from a third
# software-pipeline stage (9.7% on B2/T8192/H8/HQ16), while the same launch
# parameters with two stages are the current autotuned baseline.  Keep this in
# the DKV-only search space: the autotune key includes T/K/G/window/varlen, so
# unrelated shapes retain it only when their own timing also wins.
PARALLAX_DKV_CONFIGS = _make_parallax_configs(_PARALLAX_DKV_TILES) + [
    triton.Config(
        {"BK": 128, "BS": 64},
        num_warps=8,
        num_stages=3,
    )
]


# Specialized D=128 DQR launch for dense T=2048 and T=4096. Repeated H100
# measurements show that BK=128, BS=32, 4 warps, 3 stages is about 7.3% and
# 7.0% faster, respectively, than the generic DQR+preprocess path. It also
# matches the best v9 autotuned result, so fixing the measured winner avoids
# compiling and benchmarking the other candidates on first production use.
def _make_tle_v9_dqr_configs():
    return [
        triton.Config(
            {"BK": 128, "BS": 32},
            num_warps=4,
            num_stages=3,
        )
    ]


PARALLAX_TLE_V9_DQR_CONFIGS = _make_tle_v9_dqr_configs()

PARALLAX_PREPROCESS_CONFIGS = [
    triton.Config(
        {"BP": block_p},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_p in (32, 64, 128, 256)
    for num_warps in (4, 8)
    for num_stages in (2, 4)
]


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

    mesh = _autotune_arg(named_args, kwargs, "mesh")
    if mesh is not None:
        cluster_configs = [
            cfg
            for cfg in valid
            if cfg.kwargs["BS"] == 128
            and cfg.num_warps == 8
            and cfg.num_stages == 2
        ]
        return _copy_autotune_configs(cluster_configs)

    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(valid)

    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [
        cfg
        for cfg in valid
        if cfg.kwargs["BS"] == block_t
    ]
    preferred = [
        cfg
        for cfg in tile_matches
        if cfg.num_warps == 8 and cfg.num_stages == 2
    ]
    return _copy_autotune_configs(preferred or tile_matches)


def parallax_prune_preprocess_configs(configs, named_args, **kwargs):
    """Varlen chunk indices require preprocess BP to remain equal to BT."""
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(configs)
    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [
        cfg
        for cfg in configs
        if cfg.kwargs["BP"] == block_t
    ]
    preferred = [
        cfg
        for cfg in tile_matches
        if cfg.num_warps == 4 and cfg.num_stages == 2
    ]
    return _copy_autotune_configs(preferred or tile_matches)


@triton.jit(do_not_specialize=['T'])
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
    bos = (i_b * T).to(tl.int64)

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


@triton.jit(do_not_specialize=['T'])
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
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
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
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int64)
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


@triton.autotune(
    configs=PARALLAX_FWD_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
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
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
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
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
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
        o_kv = _safe.to(tl.int64) * BS + tl.arange(0, BS)
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
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
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
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
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


@triton.jit(do_not_specialize=['T'])
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
    bos = (i_b * T).to(tl.int64)

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


@triton.jit(do_not_specialize=['T'])
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
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
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
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int64)
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


@triton.jit(do_not_specialize=['T'])
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
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_s = i_s * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    col_mask = offs_s[:, None] < T
    head_mask = offs_k[None, :] < K
    kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
    b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
    b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

    grad_k_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_t in range(i_s, NT):
        offs_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
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


@triton.autotune(
    configs=PARALLAX_PREPROCESS_CONFIGS,
    key=["T", "K", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_preprocess_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
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
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)

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


@triton.jit(do_not_specialize=['T'])
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
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
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
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
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
        o_kv = _.to(tl.int64) * BS + tl.arange(0, BS)
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
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
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
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
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


@triton.jit(do_not_specialize=['T'])
def _parallel_parallax_bwd_kernel_dqr_tle_v9(
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
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
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
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tle.load(p_k, mask=m_kv, other=0.0, is_async=True)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
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
        o_kv = _.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tle.load(p_k, mask=m_kv, other=0.0, is_async=True)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        p = exp2(qk - b_m) * inv_d1
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
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
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tle.load(p_k, mask=m_kv, other=0.0, is_async=True)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
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
    """Hopper dense DQR specialization for aligned T=8192, D=128."""
    tl.static_assert(T % BT == 0)
    tl.static_assert(BT % BS == 0)
    tl.static_assert(K == BK)

    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)

    q_offsets = (
        ((bos + row_indices[:, None]) * HQ + i_hq) * K
        + o_k[None, :]
    )
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
        col_indices = (
            col_block_id.to(tl.int64) * BS
            + tl.arange(0, BS).to(tl.int64)
        )
        kv_offsets = (
            ((bos + col_indices[:, None]) * H + i_h) * K
            + o_k[None, :]
        )
        b_k = tle.load(k + kv_offsets, is_async=True)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        p = tl.math.exp2(qk - b_m) * inv_d1
        b_v = tle.load(v + kv_offsets)
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
        col_indices = (
            (num_safe_blocks + border_offset).to(tl.int64) * BS
            + tl.arange(0, BS).to(tl.int64)
        )
        kv_offsets = (
            ((bos + col_indices[:, None]) * H + i_h) * K
            + o_k[None, :]
        )
        b_k = tle.load(k + kv_offsets, is_async=True)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(
            row_indices[:, None] >= col_indices[None, :],
            qk,
            -float("inf"),
        )
        p = tl.math.exp2(qk - b_m) * inv_d1
        b_v = tle.load(v + kv_offsets)
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


@triton.autotune(
    configs=PARALLAX_DQR_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
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
        q, r, k, v, o, barv, d1, bart, m,
        delta_t, delta_b, grad_o, grad_q, grad_r, scale,
        cu_seqlens, chunk_indices, T,
        HQ, H, G, K, BK, WINDOW_SIZE_LEFT, BT, BS, IS_VARLEN, True,
    )


@triton.autotune(
    configs=PARALLAX_DQR_SPLIT_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
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
        q, r, k, v, q, q, d1, bart, m,
        delta_t, delta_b, grad_o, grad_q, grad_r, scale,
        cu_seqlens, chunk_indices, T,
        HQ, H, G, K, BK, WINDOW_SIZE_LEFT, BT, BS, IS_VARLEN, False,
    )


@triton.autotune(
    configs=PARALLAX_TLE_V9_DQR_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_parallax_bwd_kernel_dqr_tle_v9(
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
    _parallel_parallax_bwd_kernel_dqr_tle_v9(
        q, r, k, v, q, q, d1, bart, m,
        delta_t, delta_b, grad_o, grad_q, grad_r, scale,
        cu_seqlens, chunk_indices, T,
        HQ, H, G, K, BK, WINDOW_SIZE_LEFT, BT, BS, IS_VARLEN, False,
    )


# -----------------------------------------------------------------------------
# Experimental TLE-Struct warp-specialized DKV kernels.
#
# These kernels are intentionally narrow and excluded from production. They target
# the v9 production shape B2/T2048/H2/HQ8/D128 (dense, causal, G=4), where the
# baseline DKV autotuner selects BK=128, BS=32, 4 warps, 2 stages and NCU shows
# register pressure at the architectural limit.  The producer partition stages
# Q/R/dO and scalar state through a two-slot SMEM pipe.  The consumers own the
# fp32 gradient accumulators and therefore receive explicit register budgets.
#
# Measurements showed both variants regress the validated target, so dispatch
# is fixed to the regular DKV path rather than exposing a runtime tuning knob.
# -----------------------------------------------------------------------------

_DKV_WS_PIPE_CAPACITY = tl.constexpr(2)
_DKV_WS_BS = 32


@triton.jit
def _parallel_parallax_dkv_ws_producer(
    writer,
    q,
    r,
    grad_o,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    bos,
    i_hq,
    start_row_block,
    n_row_blocks,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Stage one query-row tile per pipe iteration.

    The specialized dispatch guarantees T=2048, K=BK=128 and BT=64 on H100,
    so every produced row tile is full.  Matrix fields use tle.gpu.copy so the
    backend can lower them to its preferred asynchronous GMEM->SMEM transport;
    scalar fields share the same logical pipe slot through local stores.
    """
    offs_k = tl.arange(0, BK).to(tl.int64)
    for pipe_iter in tl.range(0, n_row_blocks):
        pipe_stage = pipe_iter.to(tl.int32)
        row_block = start_row_block + pipe_iter
        row_indices = row_block.to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)
        q_offsets = (
            ((bos + row_indices[:, None]) * HQ + i_hq) * K
            + offs_k[None, :]
        )
        scalar_offsets = (bos + row_indices) * HQ + i_hq

        slot = writer.acquire(pipe_stage)
        tle.gpu.copy(q + q_offsets, slot.q, [BT, BK])
        tle.gpu.copy(r + q_offsets, slot.r, [BT, BK])
        tle.gpu.copy(grad_o + q_offsets, slot.go, [BT, BK])

        tl.store(
            tle.gpu.local_ptr(slot.d1),
            tl.load(d1 + scalar_offsets),
        )
        tl.store(
            tle.gpu.local_ptr(slot.bart),
            tl.load(bart + scalar_offsets),
        )
        tl.store(
            tle.gpu.local_ptr(slot.m),
            tl.load(m + scalar_offsets),
        )
        tl.store(
            tle.gpu.local_ptr(slot.dt),
            tl.load(delta_t + scalar_offsets),
        )
        tl.store(
            tle.gpu.local_ptr(slot.db),
            tl.load(delta_b + scalar_offsets),
        )
        writer.commit(pipe_stage)


@triton.jit
def _parallel_parallax_dkv_ws_joint_consumer(
    reader,
    k,
    v,
    grad_k,
    grad_v,
    bos,
    i_h,
    i_hq,
    col_offset,
    start_row_block,
    n_row_blocks,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """One compute worker owns both dK and dV accumulators.

    This variant avoids recomputing qk/rk/probabilities but keeps two
    [BS, BK] fp32 accumulators in the same worker partition.
    """
    col_indices = col_offset + tl.arange(0, BS).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    m_ck = (col_indices[:, None] < T) & (offs_k[None, :] < K)

    kv_offsets = (
        ((bos + col_indices[:, None]) * H + i_h) * K
        + offs_k[None, :]
    )
    b_k = tl.load(k + kv_offsets, mask=m_ck, other=0.0)
    b_v = tl.load(v + kv_offsets, mask=m_ck, other=0.0)

    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for pipe_iter in tl.range(0, n_row_blocks):
        pipe_stage = pipe_iter.to(tl.int32)
        ready = reader.wait(pipe_stage)
        row_block = start_row_block + pipe_iter
        row_indices = row_block.to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)

        b_q = tl.load(tle.gpu.local_ptr(ready.slot.q))
        b_r = tl.load(tle.gpu.local_ptr(ready.slot.r))
        b_go = tl.load(tle.gpu.local_ptr(ready.slot.go))
        b_d1 = tl.load(tle.gpu.local_ptr(ready.slot.d1))[:, None]
        b_bart = tl.load(tle.gpu.local_ptr(ready.slot.bart))[:, None]
        b_m = tl.load(tle.gpu.local_ptr(ready.slot.m))[:, None]
        b_t = tl.load(tle.gpu.local_ptr(ready.slot.dt))[:, None]
        b_b = tl.load(tle.gpu.local_ptr(ready.slot.db))[:, None]

        mask = row_indices[:, None] >= col_indices[None, :]
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = exp2(qk - b_m) / b_d1
        a = tl.dot(b_go, tl.trans(b_v), out_dtype=tl.float32)
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
            tl.trans(weights).to(b_go.dtype),
            b_go,
            out_dtype=tl.float32,
            acc=grad_v_acc,
        )
        reader.release(pipe_stage)

    out_offsets = (
        ((bos + col_indices[:, None]) * HQ + i_hq) * K
        + offs_k[None, :]
    )
    tl.store(
        grad_k + out_offsets,
        grad_k_acc.to(grad_k.dtype.element_ty),
        mask=m_ck,
    )
    tl.store(
        grad_v + out_offsets,
        grad_v_acc.to(grad_v.dtype.element_ty),
        mask=m_ck,
    )


@triton.jit
def _parallel_parallax_dkv_ws_dk_consumer(
    reader,
    k,
    v,
    grad_k,
    bos,
    i_h,
    i_hq,
    col_offset,
    start_row_block,
    n_row_blocks,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """dK-only consumer: one fp32 [BS, BK] accumulator."""
    col_indices = col_offset + tl.arange(0, BS).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    m_ck = (col_indices[:, None] < T) & (offs_k[None, :] < K)
    kv_offsets = (
        ((bos + col_indices[:, None]) * H + i_h) * K
        + offs_k[None, :]
    )
    b_k = tl.load(k + kv_offsets, mask=m_ck, other=0.0)
    b_v = tl.load(v + kv_offsets, mask=m_ck, other=0.0)

    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for pipe_iter in tl.range(0, n_row_blocks):
        pipe_stage = pipe_iter.to(tl.int32)
        ready = reader.wait(pipe_stage)
        row_block = start_row_block + pipe_iter
        row_indices = row_block.to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)

        b_q = tl.load(tle.gpu.local_ptr(ready.slot.q))
        b_r = tl.load(tle.gpu.local_ptr(ready.slot.r))
        b_go = tl.load(tle.gpu.local_ptr(ready.slot.go))
        b_d1 = tl.load(tle.gpu.local_ptr(ready.slot.d1))[:, None]
        b_bart = tl.load(tle.gpu.local_ptr(ready.slot.bart))[:, None]
        b_m = tl.load(tle.gpu.local_ptr(ready.slot.m))[:, None]
        b_t = tl.load(tle.gpu.local_ptr(ready.slot.dt))[:, None]
        b_b = tl.load(tle.gpu.local_ptr(ready.slot.db))[:, None]

        mask = row_indices[:, None] >= col_indices[None, :]
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = exp2(qk - b_m) / b_d1
        a = tl.dot(b_go, tl.trans(b_v), out_dtype=tl.float32)
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
        reader.release(pipe_stage)

    out_offsets = (
        ((bos + col_indices[:, None]) * HQ + i_hq) * K
        + offs_k[None, :]
    )
    tl.store(
        grad_k + out_offsets,
        grad_k_acc.to(grad_k.dtype.element_ty),
        mask=m_ck,
    )


@triton.jit
def _parallel_parallax_dkv_ws_dv_consumer(
    reader,
    k,
    grad_v,
    bos,
    i_h,
    i_hq,
    col_offset,
    start_row_block,
    n_row_blocks,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    """dV-only consumer: lower accumulator pressure, duplicated score math."""
    col_indices = col_offset + tl.arange(0, BS).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    m_ck = (col_indices[:, None] < T) & (offs_k[None, :] < K)
    kv_offsets = (
        ((bos + col_indices[:, None]) * H + i_h) * K
        + offs_k[None, :]
    )
    b_k = tl.load(k + kv_offsets, mask=m_ck, other=0.0)

    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for pipe_iter in tl.range(0, n_row_blocks):
        pipe_stage = pipe_iter.to(tl.int32)
        ready = reader.wait(pipe_stage)
        row_block = start_row_block + pipe_iter
        row_indices = row_block.to(tl.int64) * BT + tl.arange(0, BT).to(tl.int64)

        b_q = tl.load(tle.gpu.local_ptr(ready.slot.q))
        b_r = tl.load(tle.gpu.local_ptr(ready.slot.r))
        b_go = tl.load(tle.gpu.local_ptr(ready.slot.go))
        b_d1 = tl.load(tle.gpu.local_ptr(ready.slot.d1))[:, None]
        b_bart = tl.load(tle.gpu.local_ptr(ready.slot.bart))[:, None]
        b_m = tl.load(tle.gpu.local_ptr(ready.slot.m))[:, None]

        mask = row_indices[:, None] >= col_indices[None, :]
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        p = exp2(qk - b_m) / b_d1
        weights = p * (1.0 + b_bart - rk)

        # dV does not depend on V, delta_t, or delta_b. The split consumer
        # therefore avoids the dO@V path entirely and keeps only one fp32
        # [BS, BK] accumulator. Its cost is duplicated qk/rk/p computation.
        grad_v_acc = tl.dot(
            tl.trans(weights).to(b_go.dtype),
            b_go,
            out_dtype=tl.float32,
            acc=grad_v_acc,
        )
        reader.release(pipe_stage)

    out_offsets = (
        ((bos + col_indices[:, None]) * HQ + i_hq) * K
        + offs_k[None, :]
    )
    tl.store(
        grad_v + out_offsets,
        grad_v_acc.to(grad_v.dtype.element_ty),
        mask=m_ck,
    )


@triton.jit
def parallel_parallax_bwd_kernel_dkv_ws_joint(
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
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_col = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    col_offset = i_col * BS
    start_row_block = col_offset // BT
    n_row_blocks = tl.cdiv(T, BT) - start_row_block

    q_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT, BK],
        dtype=q.dtype.element_ty,
        scope=tle.gpu.smem,
    )
    r_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT, BK],
        dtype=r.dtype.element_ty,
        scope=tle.gpu.smem,
    )
    go_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT, BK],
        dtype=grad_o.dtype.element_ty,
        scope=tle.gpu.smem,
    )
    d1_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    bart_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    m_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dt_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    db_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=_DKV_WS_PIPE_CAPACITY,
        scope="cta",
        name="parallax_dkv_joint",
        q=q_smem,
        r=r_smem,
        go=go_smem,
        d1=d1_smem,
        bart=bart_smem,
        m=m_smem,
        dt=dt_smem,
        db=db_smem,
    )

    tle.gpu.warp_specialize(
        [
            (
                _parallel_parallax_dkv_ws_producer,
                (
                    pipe.writer(), q, r, grad_o, d1, bart, m, delta_t,
                    delta_b, bos, i_hq, start_row_block, n_row_blocks,
                    HQ, K, BK, BT,
                ),
            ),
            (
                _parallel_parallax_dkv_ws_joint_consumer,
                (
                    pipe.reader(), k, v, grad_k, grad_v, bos, i_h, i_hq,
                    col_offset, start_row_block, n_row_blocks, scale, T,
                    HQ, H, K, BK, BT, BS,
                ),
            ),
        ],
        [4],
        [224],
    )


@triton.jit
def parallel_parallax_bwd_kernel_dkv_ws_split(
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
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    i_col = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T
    col_offset = i_col * BS
    start_row_block = col_offset // BT
    n_row_blocks = tl.cdiv(T, BT) - start_row_block

    q_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT, BK],
        dtype=q.dtype.element_ty,
        scope=tle.gpu.smem,
    )
    r_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT, BK],
        dtype=r.dtype.element_ty,
        scope=tle.gpu.smem,
    )
    go_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT, BK],
        dtype=grad_o.dtype.element_ty,
        scope=tle.gpu.smem,
    )
    d1_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    bart_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    m_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    dt_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    db_smem = tle.gpu.alloc(
        [_DKV_WS_PIPE_CAPACITY, BT],
        dtype=tl.float32,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(
        capacity=_DKV_WS_PIPE_CAPACITY,
        scope="cta",
        name="parallax_dkv_split",
        readers=("dk", "dv"),
        q=q_smem,
        r=r_smem,
        go=go_smem,
        d1=d1_smem,
        bart=bart_smem,
        m=m_smem,
        dt=dt_smem,
        db=db_smem,
    )

    tle.gpu.warp_specialize(
        [
            (
                _parallel_parallax_dkv_ws_producer,
                (
                    pipe.writer(), q, r, grad_o, d1, bart, m, delta_t,
                    delta_b, bos, i_hq, start_row_block, n_row_blocks,
                    HQ, K, BK, BT,
                ),
            ),
            (
                _parallel_parallax_dkv_ws_dk_consumer,
                (
                    pipe.reader("dk"), k, v, grad_k, bos, i_h, i_hq,
                    col_offset, start_row_block, n_row_blocks, scale, T,
                    HQ, H, K, BK, BT, BS,
                ),
            ),
            (
                _parallel_parallax_dkv_ws_dv_consumer,
                (
                    pipe.reader(
                        "dv",
                        fields=("q", "r", "go", "d1", "bart", "m"),
                    ),
                    k, grad_v, bos, i_h, i_hq,
                    col_offset, start_row_block, n_row_blocks, scale, T,
                    HQ, H, K, BK, BT, BS,
                ),
            ),
        ],
        [4, 4],
        [176, 160],
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
    q_offsets = (
        ((bos + row_indices[:, None]) * HQ + i_hq) * K
        + o_k[None, :]
    )
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
    """Dense causal DKV specialization for aligned D=128 Hopper tiles."""
    tl.static_assert(K == BK)
    tl.static_assert(BT == BS)
    tl.static_assert(T % BT == 0)

    i_col = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    col_indices = i_col * BS + tl.arange(0, BS).to(tl.int64)
    o_k = tl.arange(0, BK).to(tl.int64)
    kv_offsets = (
        ((bos + col_indices[:, None]) * H + i_h) * K
        + o_k[None, :]
    )
    out_offsets = (
        ((bos + col_indices[:, None]) * HQ + i_hq) * K
        + o_k[None, :]
    )
    b_k = tl.load(k + kv_offsets)
    b_v = tl.load(v + kv_offsets)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)

    # BT == BS gives one causal-border row tile. Every later row tile is
    # provably dense and aligned, so all row/head load masks are omitted.
    border_rows = i_col * BT + tl.arange(0, BT).to(tl.int64)
    grad_k_acc, grad_v_acc = (
        _parallel_parallax_dkv_dense_aligned_accumulate_row(
            q, r, grad_o, d1, bart, m, delta_t, delta_b,
            b_k, b_v, bos, i_hq, border_rows, col_indices, scale,
            grad_k_acc, grad_v_acc,
            HQ, K, BK, True,
        )
    )
    for row_block in range(i_col + 1, T // BT):
        row_indices = row_block * BT + tl.arange(0, BT).to(tl.int64)
        grad_k_acc, grad_v_acc = (
            _parallel_parallax_dkv_dense_aligned_accumulate_row(
                q, r, grad_o, d1, bart, m, delta_t, delta_b,
                b_k, b_v, bos, i_hq, row_indices, col_indices, scale,
                grad_k_acc, grad_v_acc,
                HQ, K, BK, False,
            )
        )

    tl.store(
        grad_k + out_offsets,
        grad_k_acc.to(grad_k.dtype.element_ty),
    )
    tl.store(
        grad_v + out_offsets,
        grad_v_acc.to(grad_v.dtype.element_ty),
    )


@triton.autotune(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'mesh': lambda args: args.get('mesh'),
    'CLUSTER_REDUCE': lambda args: args.get('mesh') is not None,
})
@triton.jit(do_not_specialize=['T'])
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
    mesh: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CLUSTER_REDUCE: tl.constexpr,
):
    physical_i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    if CLUSTER_REDUCE:
        cluster_rank = tle.shard_id(mesh, "query_head")
        i_t = (physical_i_t - cluster_rank) // G
        i_b, i_h = i_bh // H, i_bh % H
        i_hq = i_h * G + cluster_rank
    else:
        i_t = physical_i_t
        i_b, i_hq = i_bh // HQ, i_bh % HQ
        i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
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
    if CLUSTER_REDUCE:
        p_grad_k = grad_k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_grad_v = grad_v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    else:
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
        row_offset = row_block_id.to(tl.int64) * BT
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
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & row_mask
                & (col_indices[None, :] < T)
            )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id.to(tl.int64) * BT
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
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

    # Phase C: window-border row blocks (SWA only).
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id.to(tl.int64) * BT
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
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
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
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

    if CLUSTER_REDUCE:
        grad_k_smem = tle.gpu.alloc(
            [BS, BK],
            dtype=grad_k.dtype.element_ty,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        grad_v_smem = tle.gpu.alloc(
            [BS, BK],
            dtype=grad_v.dtype.element_ty,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_rows = tl.arange(0, BS)[:, None] + 0 * o_k[None, :]
        smem_cols = 0 * tl.arange(0, BS)[:, None] + o_k[None, :]
        local_grad_k = tle.gpu.local_ptr(
            grad_k_smem,
            (smem_rows, smem_cols),
        )
        local_grad_v = tle.gpu.local_ptr(
            grad_v_smem,
            (smem_rows, smem_cols),
        )
        # Rank 0 already owns its partials in registers. Only peer ranks need
        # to publish through DSM; this halves epilogue shared stores and remote
        # reads for the measured G=2 cluster.
        if cluster_rank != 0:
            tl.store(
                local_grad_k,
                grad_k_acc.to(grad_k.dtype.element_ty),
                mask=m_ck,
            )
            tl.store(
                local_grad_v,
                grad_v_acc.to(grad_v.dtype.element_ty),
                mask=m_ck,
            )
        tle.distributed_barrier(mesh)

        if cluster_rank == 0:
            # Match the existing global reduction's bf16 partial precision.
            reduced_k = grad_k_acc.to(grad_k.dtype.element_ty).to(tl.float32)
            reduced_v = grad_v_acc.to(grad_v.dtype.element_ty).to(tl.float32)
            for peer_rank in tl.static_range(1, G):
                peer_k_smem = tle.remote(
                    grad_k_smem,
                    peer_rank,
                    scope=mesh,
                )
                peer_v_smem = tle.remote(
                    grad_v_smem,
                    peer_rank,
                    scope=mesh,
                )
                peer_k = tle.gpu.local_ptr(
                    peer_k_smem,
                    (smem_rows, smem_cols),
                )
                peer_v = tle.gpu.local_ptr(
                    peer_v_smem,
                    (smem_rows, smem_cols),
                )
                reduced_k += tl.load(peer_k, mask=m_ck, other=0.0)
                reduced_v += tl.load(peer_v, mask=m_ck, other=0.0)
            tl.store(
                p_grad_k,
                reduced_k.to(p_grad_k.dtype.element_ty),
                mask=m_ck,
            )
            tl.store(
                p_grad_v,
                reduced_v.to(p_grad_v.dtype.element_ty),
                mask=m_ck,
            )
        # Keep every peer CTA and its DSM allocation alive until rank 0 has
        # completed all remote reads and final stores.
        tle.distributed_barrier(mesh)
    else:
        tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
        tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.autotune(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
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
    """Accumulate all query heads in a GQA group directly into one KV head.

    The decode kernel packs 2/4/8 query heads that share K/V into one CTA. For
    training, packing the forward rows would create prohibitively large score
    tiles, but dK/dV already has exactly one destination per KV head. Owning
    that destination here removes the [B, T, HQ, K] staging tensors and their
    separate reduction while loading each K/V tile only once per group.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
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
            row_offset = row_block_id.to(tl.int64) * BT
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
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            if WINDOW_SIZE_LEFT >= 0:
                mask = (
                    (row_indices[:, None] >= col_indices[None, :])
                    & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                    & row_mask
                    & (col_indices[None, :] < T)
                )
            else:
                mask = (
                    (row_indices[:, None] >= col_indices[None, :])
                    & row_mask
                    & (col_indices[None, :] < T)
                )
            qk = tl.where(mask, qk, -float("inf"))
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            bart_minus_rk = b_bart - rk
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

        # Phase B: safe row blocks (no causal/col/window mask).
        for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
            row_offset = row_block_id.to(tl.int64) * BT
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
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            bart_minus_rk = b_bart - rk
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

        # Phase C: window-border row blocks (SWA only).
        for row_block_id in range(window_border_start, num_row_blocks):
            row_offset = row_block_id.to(tl.int64) * BT
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
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
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
            bart_minus_rk = b_bart - rk
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

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
):
    """Fold per-query-head dK/dV into their shared GQA heads in one pass."""
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < N

    head_dim_offset = offsets % K
    row_head = offsets // K
    kv_head = row_head % H
    batch_row = row_head // H
    input_offsets = (
        (batch_row * HQ + kv_head * G) * K + head_dim_offset
    )

    grad_k_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    grad_v_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i_g in range(0, G):
        group_offsets = input_offsets + i_g * K
        grad_k_acc += tl.load(grad_k_buf + group_offsets, mask=mask, other=0.0)
        grad_v_acc += tl.load(grad_v_buf + group_offsets, mask=mask, other=0.0)

    tl.store(grad_k + offsets, grad_k_acc, mask=mask)
    tl.store(grad_v + offsets, grad_v_acc, mask=mask)


def parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax forward (Triton). `(B, T, HQ, D)` / packed `(1, T_total, HQ, D)` inputs.

    Returns `(o, barv, d1, bart, m)`: `o`/`barv` in the input dtype and layout;
    `d1`/`bart`/`m` are fp32 per-(position, query-head) scalars `(B, T, HQ)`.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
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
                q, r, k, v, o, barv, d1, bart, m, scale, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                num_warps=4, num_stages=1,
            )
        elif NQ <= 8:
            # B3/T111/H2/HQ2/D100 on H100 is too small to fill the device with
            # the default four-warp launch.  Eight warps provides the large
            # gain over that old path; within the validated eight-warp path,
            # one stage is another ~2.3% faster than two stages.  Both keep
            # the same BT=64 math and exact output.  Keep this launch override
            # narrowly scoped to the measured shape.
            use_d100_short_w8 = (
                _is_nvidia_hopper(q.device.index)
                and B == 3
                and T == 111
                and H == 2
                and HQ == 2
                and K == 100
                and window_size_left < 0
            )
            parallel_parallax_fwd_kernel_short_multi[query_grid](
                q, r, k, v, o, barv, d1, bart, m, scale, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT, NT=NQ,
                num_warps=(
                    8
                    if use_d100_short_w8
                    else (4 if BT <= 64 else 8)
                ),
                num_stages=1 if use_d100_short_w8 else 2,
            )
        else:
            parallel_parallax_fwd_kernel[query_grid](
                q, r, k, v, o, barv, d1, bart, m,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT,
            )
    else:
        # Packed chunks are created with BT, so the autotune pruner keeps
        # BS == BT for this path.  Dense inputs above remain fully decoupled.
        chunk_grid = (len(chunk_indices), B * HQ)
        parallel_parallax_fwd_kernel[chunk_grid](
            q, r, k, v, o, barv, d1, bart, m,
            scale, cu_seqlens, chunk_indices, T,
            HQ=HQ, H=H, G=G, K=K,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT,
        )
    return o, barv, d1, bart, m


def parallel_parallax_bwd(q, r, k, v, o, barv, d1, bart, m, grad_o, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Parallax backward (Triton). Returns grads matching `q, r, k, v`."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    NQ = triton.cdiv(T, BT)
    # Enable the measured v9 DQR path only for validated dense D=128 shapes.
    # T8192 is intentionally an exact shape match: on H100, BS=32/W4/S3 is
    # 1.012x faster than the generic autotuner winner (BS=32/W4/S2), while the
    # TLE and Triton S3 lowerings are effectively tied.  Fixing the TLE launch
    # avoids the noisy autotune choice without changing unrelated long shapes.
    use_tle_v9_dqr = (
        HAS_TLE
        and _is_nvidia_hopper(q.device.index)
        and cu_seqlens is None
        and K == 128
        and (
            T in (2048, 4096)
            or (
                B == 2
                and T == 8192
                and H == 8
                and HQ == 16
                and G == 2
            )
        )
        and NQ > 16
    )
    # Exact dense-aligned Hopper specialization.  For this shape every BT=64
    # query tile and BK=128 head tile is complete; BS=32 creates two causal
    # border blocks and all earlier blocks are mask-free.  Stable paired H100
    # measurements select TLE K prefetch with W4/S3: 2.812 ms versus 2.927 ms
    # for production auto (1.041x), with zero observed output mismatches.
    use_dense_aligned_dqr = (
        HAS_TLE
        and _is_nvidia_hopper(q.device.index)
        and q.dtype == torch.bfloat16
        and cu_seqlens is None
        and B == 2
        and T == 8192
        and H == 8
        and HQ == 16
        and G == 2
        and K == 128
        and BK == 128
        and BT == 64
        and window_size_left < 0
        and not _is_cuda_graph_capturing()
    )
    dkv_programs = (len(chunk_indices) if cu_seqlens is not None else NQ) * B * H
    num_sms = _get_num_sms(q.device.index)
    group_dkv_heads = (
        G in (4, 8)
        or (
            G == 2
            and (cu_seqlens is not None or T > 8192)
        )
    )
    # On the measured H100 long-sequence D=128 shapes below, the grouped
    # kernel is 14-26% slower.  Its long-lived accumulators outweigh K/V reuse,
    # while the per-query-head path exposes more CTAs and its final reduction
    # is comparatively cheap.  Keep these exceptions architecture- and
    # shape-specific because the per-head path uses larger staging buffers.
    prefer_per_head_dkv = (
        _is_nvidia_hopper(q.device.index)
        and cu_seqlens is None
        and K == 128
        and (
            (
                B == 2
                and T == 8192
                and HQ == 16
                and G in (4, 8)
            )
            or (
                B == 4
                and T == 16384
                and HQ == 32
                and G == 2
            )
        )
    )
    # TLE cluster epilogue validated on H100: the two query-head CTAs publish
    # bf16 partial dK/dV through DSM and rank 0 writes the final KV-head result.
    # This removes the HQ-sized global buffers and reduction launch (1.053x).
    use_tle_cluster_reduce_dkv = (
        HAS_TLE_CLUSTER
        and _is_nvidia_hopper(q.device.index)
        and cu_seqlens is None
        and B == 4
        and T == 16384
        and H == 16
        and HQ == 32
        and G == 2
        and K == 128
        and window_size_left < 0
        and not _is_cuda_graph_capturing()
    )
    # Dense aligned DKV specialization validated on H100 BF16.  T=8192 is
    # exactly divisible by BT=BS=64 and D=BK=128, so only the first row tile
    # intersects the causal diagonal; every later tile can omit all tail-load
    # masks.  Stable paired measurements improve DKV+GQA reduction by 3.9%.
    use_dense_aligned_dkv = (
        _is_nvidia_hopper(q.device.index)
        and q.dtype == torch.bfloat16
        and cu_seqlens is None
        and B == 2
        and T == 8192
        and H == 8
        and HQ == 16
        and G == 2
        and K == 128
        and BT == 64
        and window_size_left < 0
    )
    use_grouped_dkv = (
        # Grouped K/V reuse is the general long-sequence default.  The measured
        # dense Hopper exceptions above retain separate CTAs; varlen stays
        # grouped because T is the packed total rather than one sequence.
        group_dkv_heads
        and not prefer_per_head_dkv
        and (cu_seqlens is not None or NQ > 16)
        and num_sms > 0
        and dkv_programs >= num_sms
    )
    # The two-stream schedule has been validated on H100 for the dense D=128
    # shapes below.  It overlaps the independent TLE DQR and DKV branches
    # after preprocess.  Keep an explicit measured whitelist: larger grouped
    # workloads can saturate the SMs and leave no useful overlap.
    concurrent_shape_validated = (
        (
            B == 2
            and HQ == 8
            and T in (2048, 4096)
            and G in (2, 4, 8)
        )
        or (
            B == 1
            and HQ == 8
            and T in (2048, 4096)
            and G == 4
        )
        or (
            B == 4
            and HQ == 8
            and T == 2048
            and G == 4
        )
        or (
            B == 2
            and HQ == 16
            and T == 2048
            and G == 4
        )
    )
    use_concurrent_bwd = (
        use_tle_v9_dqr
        and cu_seqlens is None
        and K == 128
        and concurrent_shape_validated
        and window_size_left < 0
        and not _is_cuda_graph_capturing()
    )
    parent_stream = None
    dqr_stream = None
    dkv_stream = None
    if use_concurrent_bwd:
        parent_stream = torch.cuda.current_stream(q.device)
        dqr_stream, dkv_stream = _get_parallax_bwd_streams(q.device.index)

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    if use_grouped_dkv or use_tle_cluster_reduce_dkv or G == 1:
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_k_buf, grad_v_buf = grad_k, grad_v
    else:
        # Short sequences retain one CTA per query head for occupancy, then
        # fold their private dK/dV tiles back to the shared KV-head axis.
        if use_concurrent_bwd:
            # Allocate final reduction outputs before the side streams wait on
            # the parent, so their storage is visible to the DKV stream.
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
        else:
            grad_k = grad_v = None
        grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
        grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    if cu_seqlens is None:
        query_grid = (NQ, B * HQ)
        if T <= BT:
            parallel_parallax_bwd_kernel_short[query_grid](
                q, r, k, v, o, barv, d1, bart, m, grad_o,
                grad_q, grad_r, grad_k_buf, grad_v_buf, scale, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                num_warps=4, num_stages=1,
            )
        elif NQ <= 16:
            # The same under-filled D100 workload as the forward special case
            # benefits from eight warps in both backward passes.  DQR prefers
            # two stages, while DKV is substantially faster with one stage.
            use_d100_short_bwd = (
                _is_nvidia_hopper(q.device.index)
                and B == 3
                and T == 111
                and H == 2
                and HQ == 2
                and K == 100
                and window_size_left < 0
            )
            parallel_parallax_bwd_kernel_dqr_short_multi[query_grid](
                q, r, k, v, o, barv, d1, bart, m, grad_o,
                grad_q, grad_r, scale, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT, NT=NQ,
                num_warps=(
                    8
                    if use_d100_short_bwd
                    else (4 if BT <= 64 else 8)
                ),
                num_stages=2,
            )
            parallel_parallax_bwd_kernel_dkv_short_multi[query_grid](
                q, r, k, v, o, barv, d1, bart, m, grad_o,
                grad_k_buf, grad_v_buf, scale, T,
                HQ=HQ, H=H, G=G, K=K, BK=BK,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT, NT=NQ,
                num_warps=(
                    8
                    if use_d100_short_bwd
                    else (4 if BT <= 64 else 8)
                ),
                num_stages=1 if use_d100_short_bwd else 2,
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
                    grad_o, o, barv, delta_t, delta_b,
                    cu_seqlens, chunk_indices, T,
                    HQ=HQ, K=K, BK=BK, BT=BT,
                )
                if use_concurrent_bwd:
                    dqr_stream.wait_stream(parent_stream)
                    dkv_stream.wait_stream(parent_stream)
                dqr_context = (
                    torch.cuda.stream(dqr_stream)
                    if use_concurrent_bwd
                    else nullcontext()
                )
                with dqr_context:
                    if use_dense_aligned_dqr:
                        parallel_parallax_bwd_kernel_dqr_dense_aligned[
                            query_grid
                        ](
                            q, r, k, v, d1, bart, m,
                            delta_t, delta_b, grad_o, grad_q, grad_r,
                            scale, T,
                            HQ=HQ, H=H, G=G, K=K, BK=BK,
                            BT=BT, BS=32,
                            num_warps=4, num_stages=3,
                        )
                    elif use_tle_v9_dqr:
                        parallel_parallax_bwd_kernel_dqr_tle_v9[query_grid](
                            q, r, k, v, d1, bart, m,
                            delta_t, delta_b, grad_o, grad_q, grad_r,
                            scale, cu_seqlens, chunk_indices, T,
                            HQ=HQ, H=H, G=G, K=K,
                            WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                        )
                    else:
                        parallel_parallax_bwd_kernel_dqr[query_grid](
                            q, r, k, v, d1, bart, m,
                            delta_t, delta_b, grad_o, grad_q, grad_r,
                            scale, cu_seqlens, chunk_indices, T,
                            HQ=HQ, H=H, G=G, K=K,
                            WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                        )
            else:
                parallel_parallax_bwd_kernel_dqr_fused[query_grid](
                    q, r, k, v, o, barv, d1, bart, m,
                    delta_t, delta_b, grad_o, grad_q, grad_r,
                    scale, cu_seqlens, chunk_indices, T,
                    HQ=HQ, H=H, G=G, K=K,
                    WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                )
            dkv_context = (
                torch.cuda.stream(dkv_stream)
                if use_concurrent_bwd
                else nullcontext()
            )
            with dkv_context:
                if use_dense_aligned_dkv:
                    parallel_parallax_bwd_kernel_dkv_dense_aligned[
                        (T // BT, B * HQ)
                    ](
                        q, r, k, v, d1, bart, m, delta_t, delta_b,
                        grad_o, grad_k_buf, grad_v_buf, scale, T,
                        HQ=HQ, H=H, G=G, K=K, BK=BK, BT=BT, BS=64,
                        num_warps=8, num_stages=3,
                    )
                elif use_tle_cluster_reduce_dkv:
                    parallel_parallax_bwd_kernel_dkv[grouped_dkv_grid](
                        q, r, k, v, d1, bart, m, delta_t, delta_b,
                        grad_o, grad_k, grad_v,
                        scale, cu_seqlens, chunk_indices, T,
                        mesh=_get_parallax_gqa_cluster_mesh_2(),
                        HQ=HQ, H=H, G=G, K=K,
                        WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                    )
                elif use_grouped_dkv:
                    parallel_parallax_bwd_kernel_dkv_grouped[grouped_dkv_grid](
                        q, r, k, v, d1, bart, m, delta_t, delta_b,
                        grad_o, grad_k, grad_v,
                        scale, cu_seqlens, chunk_indices, T,
                        HQ=HQ, H=H, G=G, K=K,
                        WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                    )
                else:
                    parallel_parallax_bwd_kernel_dkv[dkv_grid](
                        q, r, k, v, d1, bart, m, delta_t, delta_b,
                        grad_o, grad_k_buf, grad_v_buf,
                        scale, cu_seqlens, chunk_indices, T,
                        HQ=HQ, H=H, G=G, K=K,
                        WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                    )
    else:
        # chunk_indices is indexed in BT-sized units; the autotune pruners
        # intentionally pin BP and BS to BT for packed variable lengths.
        chunk_grid = (len(chunk_indices), B * HQ)
        grouped_chunk_grid = (len(chunk_indices), B * H)
        deltas = torch.empty((2, B, T, HQ), device=q.device, dtype=torch.float32)
        delta_t, delta_b = deltas.unbind(0)
        parallel_parallax_bwd_kernel_preprocess[chunk_grid](
            grad_o, o, barv, delta_t, delta_b,
            cu_seqlens, chunk_indices, T,
            HQ=HQ, K=K, BK=BK, BT=BT,
        )
        parallel_parallax_bwd_kernel_dqr[chunk_grid](
            q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
            scale, cu_seqlens, chunk_indices, T,
            HQ=HQ, H=H, G=G, K=K,
            WINDOW_SIZE_LEFT=window_size_left, BT=BT,
        )
        if use_grouped_dkv:
            parallel_parallax_bwd_kernel_dkv_grouped[grouped_chunk_grid](
                q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k, grad_v,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT,
            )
        else:
            parallel_parallax_bwd_kernel_dkv[chunk_grid](
                q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
                scale, cu_seqlens, chunk_indices, T,
                HQ=HQ, H=H, G=G, K=K,
                WINDOW_SIZE_LEFT=window_size_left, BT=BT,
            )
    if G > 1 and not use_grouped_dkv and not use_tle_cluster_reduce_dkv:
        if grad_k is None:
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
        num_grad_k_elements = grad_k.numel()
        reduction_block = 256
        reduction_grid = (triton.cdiv(num_grad_k_elements, reduction_block),)
        reduction_context = (
            torch.cuda.stream(dkv_stream)
            if use_concurrent_bwd
            else nullcontext()
        )
        with reduction_context:
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
                num_warps=4,
            )
    if use_concurrent_bwd:
        parent_stream.wait_stream(dqr_stream)
        parent_stream.wait_stream(dkv_stream)
    return grad_q, grad_r, grad_k, grad_v


def _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens):
    tensors = {"q": q, "r": r, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, T, H, D], got {tuple(tensor.shape)}")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got device={tensor.device}")

    if r.shape != q.shape:
        raise ValueError(f"r shape {tuple(r.shape)} must match q shape {tuple(q.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}")
    if any(tensor.device != q.device for tensor in (r, k, v)):
        raise ValueError("q, r, k, and v must be on the same CUDA device")
    if any(tensor.dtype != q.dtype for tensor in (r, k, v)):
        raise TypeError("q, r, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")

    B, T, HQ, K = q.shape
    if k.shape[0] != B or k.shape[1] != T or k.shape[3] != K:
        raise ValueError(
            f"k/v shape {tuple(k.shape)} is incompatible with q/r shape {tuple(q.shape)}; "
            "B, T, and D must match"
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
            raise ValueError("cu_seqlens must be on the same CUDA device as q")


class ParallaxFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = prepare_chunk_indices(cu_seqlens, _block_size(q.shape[-1], q.device.index)) \
            if cu_seqlens is not None else None
        o, barv, d1, bart, m = parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        gq, gr, gk, gv = parallel_parallax_bwd(
            q, r, k, v, o, barv, d1, bart, m, do,
            ctx.scale, ctx.cu_seqlens, ctx.chunk_indices, ctx.window_size_left,
        )
        return gq.to(q), gr.to(r), gk.to(k), gv.to(v), None, None, None


def parallel_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Causal Parallax (parameterized local linear attention) with autograd,
    backed by Triton kernels. See `fla.ops.parallax.naive.naive_parallax` for
    the reference math.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            secondary queries of shape `[B, T, HQ, D]` (same shape as `q`). NOTE:
            `r` is *not* scaled by `scale`; pass it un-pre-scaled.
        k (torch.Tensor):
            keys of shape `[B, T, H, D]`. GQA is applied when `HQ` is divisible by `H`.
        v (torch.Tensor):
            values of shape `[B, T, H, D]`.
        scale (float, Optional):
            Scale applied to the `q @ k^T` logits only. If `None`, defaults to `1 / sqrt(D)`.
            Default: `None`.
        window_size (int, Optional):
            Sliding-window length. If provided, each query at position `i` only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training
            (FlashAttention convention). The batch size must be 1 when packing. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"parallel_parallax got unexpected keyword argument(s): {unexpected}")
    _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
