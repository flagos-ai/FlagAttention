# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
#
# Derived from the Parallax implementation in FlagAttention PR #57.
# Modified for MetaX C550 adaptation and optimization.
# Original contributor information:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# MIT License
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

"""Training-time Parallax attention implemented with Triton kernels."""

from __future__ import annotations

import copy
import warnings
from functools import wraps

import torch
import triton
import triton.language as tl
from triton.testing import do_bench as _triton_do_bench

from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.FLA.utils import (
    get_num_sms,
    has_shared_memory,
    is_nvidia_blackwell_device,
    is_nvidia_hopper_device,
)


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


def _use_c550_short_tiles(q, cu_seqlens) -> bool:
    # Use smaller tiles for short sequences with padded head dimensions.
    if cu_seqlens is not None or not (0 < q.shape[1] <= 256 and 64 < q.shape[-1] <= 128):
        return False
    name = torch.cuda.get_device_name(q.device).upper()
    return "METAX" in name and "C550" in name


def _block_size(head_dim: int, device_index: int, cu_seqlens=None) -> int:
    # Packed forward, backward and chunk generation must share this tile size.
    if cu_seqlens is not None and 64 < head_dim <= 128:
        name = torch.cuda.get_device_name(device_index).upper()
        if "METAX" in name and "C550" in name:
            return 32
    # The fused short backward keeps attention scores and all four gradient
    # accumulators live at once, so Hopper uses a smaller tile than the baseline
    # kernel. Varlen still uses this same BT to keep chunk_indices consistent.
    if is_nvidia_blackwell_device(device_index):
        return 128
    if has_shared_memory(232_448, device_index):
        return 64 if head_dim <= 128 else 32
    return 64


def _make_parallax_configs(tile_pairs):
    """Build independent autotune spaces for each long-sequence pass."""
    configs = []
    for block_k, block_s in tile_pairs:
        if block_k == 64 and block_s == 16:
            # Narrow score tiles bound intermediate storage; one-stage variants reduce shared-memory use.
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
# DQR also tries BS=16 for D=64; forward retains BS>=32 to limit loop overhead.
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

PARALLAX_FWD_CONFIGS = _make_parallax_configs(_PARALLAX_FWD_TILES)
PARALLAX_DQR_CONFIGS = _make_parallax_configs(_PARALLAX_DQR_TILES)
PARALLAX_DQR_SPLIT_CONFIGS = _make_parallax_configs(_PARALLAX_DQR_TILES)
# Keep the additional pipeline-stage configuration in the DKV-only tuning space.
PARALLAX_DKV_CONFIGS = _make_parallax_configs(_PARALLAX_DKV_TILES) + [
    triton.Config(
        {"BK": 128, "BS": 64},
        num_warps=8,
        num_stages=3,
    )
]


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


def _bench_with_maca_resource_limit(fn, quantiles=None):
    """Reject unlaunchable candidates without shrinking the tuning space.

    MACA reports private-memory exhaustion as RuntimeError, whereas Triton's
    autotuner normally recognizes only its own OutOfResources exception.
    Keep all other runtime failures visible. If every candidate fails, the
    final selected launch still raises instead of returning an invalid result.
    """
    try:
        return _triton_do_bench(fn, quantiles=quantiles)
    except RuntimeError as error:
        if str(error) != "Triton Error [MACA]: memory size or pointer value too large to fit in 32 bit":
            raise
        warnings.warn(f"Parallax autotune candidate exceeds the MACA launch memory limit: {error}", stacklevel=2)
        return [float("inf")] * len(quantiles) if quantiles is not None else float("inf")


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
    q = _autotune_arg(named_args, kwargs, "q")
    if q is not None and expected_bk == 128:
        name = torch.cuda.get_device_name(q.device).upper()
        if "METAX" in name and "C550" in name:
            selected = [cfg for cfg in tile_matches if cfg.num_warps == 4 and cfg.num_stages == 1]
            if not selected:
                raise RuntimeError("C550 packed resource candidate is missing")
            return _copy_autotune_configs(selected)
    preferred = [cfg for cfg in tile_matches if cfg.num_warps == 8 and cfg.num_stages == 2]
    return _copy_autotune_configs(preferred or tile_matches)


def parallax_prune_fwd_configs(configs, named_args, **kwargs):
    valid = parallax_prune_configs(configs, named_args, **kwargs)
    q = _autotune_arg(named_args, kwargs, "q")
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if q is not None and cu_seqlens is not None and 64 < q.shape[-1] <= 128:
        name = torch.cuda.get_device_name(q.device).upper()
        if "METAX" in name and "C550" in name:
            # Preserve BS == BT so packed chunk ownership is unchanged.
            # Select from templates: the generic pruner already prefers W8/S2.
            block_t = _autotune_arg(named_args, kwargs, "BT")
            selected = [
                cfg
                for cfg in configs
                if cfg.kwargs["BK"] == 128
                and cfg.kwargs["BS"] == block_t
                and cfg.num_warps == 4
                and cfg.num_stages == 1
            ]
            if not selected:
                raise RuntimeError("C550 varlen forward resource candidate is missing")
            return _copy_autotune_configs(selected)
    if q is not None and cu_seqlens is None and 32 < q.shape[-1] <= 64:
        name = torch.cuda.get_device_name(q.device).upper()
        if "METAX" in name and "C550" in name:
            # Bound launch resources for the C550 private-memory limit.
            selected = [
                cfg
                for cfg in valid
                if cfg.kwargs["BK"] == 64 and cfg.kwargs["BS"] == 32 and cfg.num_warps == 4 and cfg.num_stages == 1
            ]
            if not selected:
                raise RuntimeError("C550 forward resource candidate is missing")
            return selected
    return valid


def parallax_prune_dqr_fused_configs(configs, named_args, **kwargs):
    valid = parallax_prune_configs(configs, named_args, **kwargs)
    q = _autotune_arg(named_args, kwargs, "q")
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if q is not None and cu_seqlens is None and 32 < q.shape[-1] <= 64:
        name = torch.cuda.get_device_name(q.device).upper()
        if "METAX" in name and "C550" in name:
            # Bound launch resources for the C550 private-memory limit.
            selected = [
                cfg
                for cfg in valid
                if cfg.kwargs["BK"] == 64 and cfg.kwargs["BS"] == 16 and cfg.num_warps == 4 and cfg.num_stages == 1
            ]
            if not selected:
                raise RuntimeError("C550 fused DQR resource candidate is missing")
            return selected
    return valid


def parallax_prune_dkv_grouped_configs(configs, named_args, **kwargs):
    valid = parallax_prune_configs(configs, named_args, **kwargs)
    q = _autotune_arg(named_args, kwargs, "q")
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if q is not None and cu_seqlens is None and 32 < q.shape[-1] <= 64:
        name = torch.cuda.get_device_name(q.device).upper()
        if "METAX" in name and "C550" in name:
            # Bound grouped DKV launch resources on C550.
            selected = [
                cfg
                for cfg in valid
                if cfg.kwargs["BK"] == 64 and cfg.kwargs["BS"] == 32 and cfg.num_warps == 4 and cfg.num_stages == 1
            ]
            if not selected:
                raise RuntimeError("C550 grouped DKV resource candidate is missing")
            return selected
    return valid


def parallax_prune_dqr_varlen_configs(configs, named_args, **kwargs):
    q = _autotune_arg(named_args, kwargs, "q")
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if q is not None and cu_seqlens is not None and 64 < q.shape[-1] <= 128:
        name = torch.cuda.get_device_name(q.device).upper()
        if "METAX" in name and "C550" in name:
            # Preserve packed chunk ownership; only reduce launch resources.
            block_t = _autotune_arg(named_args, kwargs, "BT")
            selected = [
                cfg
                for cfg in configs
                if cfg.kwargs["BK"] == 128
                and cfg.kwargs["BS"] == block_t
                and cfg.num_warps == 4
                and cfg.num_stages == 1
            ]
            if not selected:
                raise RuntimeError("C550 varlen DQR resource candidate is missing")
            return _copy_autotune_configs(selected)
    return parallax_prune_configs(configs, named_args, **kwargs)


def parallax_prune_preprocess_configs(configs, named_args, **kwargs):
    """Varlen chunk indices require preprocess BP to remain equal to BT."""
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(configs)
    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [cfg for cfg in configs if cfg.kwargs["BP"] == block_t]
    preferred = [cfg for cfg in tile_matches if cfg.num_warps == 4 and cfg.num_stages == 2]
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


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel_c550_tiled(
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
    # Larger head tiles use transposed output accumulators to reduce spilling.
    if BK == 128:
        barv_acc = tl.zeros((BK, BT), dtype=tl.float32)
        rv_acc = tl.zeros((BK, BT), dtype=tl.float32)
    else:
        barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
        rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    start_s = 0
    if WINDOW_SIZE_LEFT > 0:
        start_s = tl.maximum(0, i_t * BT - WINDOW_SIZE_LEFT + 1) // BT
    for i_s in range(start_s, tl.minimum(i_t + 1, NT)):
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
        if BK == 128:
            barv_acc = tl.trans(alpha) * barv_acc
            rv_acc = tl.trans(alpha) * rv_acc
            # Split the reduction dimension into two contiguous halves.
            v0, v1 = tl.split(tl.permute(tl.reshape(tl.trans(b_v), (BK, 2, BT // 2)), (0, 2, 1)))
            w0, w1 = tl.split(tl.permute(tl.reshape(tl.trans(w.to(b_v.dtype)), (2, BT // 2, BT)), (1, 2, 0)))
            barv_acc = tl.dot(v0, w0, out_dtype=tl.float32, acc=barv_acc)
            barv_acc = tl.dot(v1, w1, out_dtype=tl.float32, acc=barv_acc)
            wr0, wr1 = tl.split(tl.permute(tl.reshape(tl.trans(wr.to(b_v.dtype)), (2, BT // 2, BT)), (1, 2, 0)))
            rv_acc = tl.dot(v0, wr0, out_dtype=tl.float32, acc=rv_acc)
            rv_acc = tl.dot(v1, wr1, out_dtype=tl.float32, acc=rv_acc)
        else:
            barv_acc = alpha * barv_acc
            rv_acc = alpha * rv_acc
            barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
            rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=rv_acc)
        m_acc = m_new

    if BK == 128:
        barv_acc = tl.trans(barv_acc)
        rv_acc = tl.trans(rv_acc)
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
    prune_configs_by={"early_config_prune": parallax_prune_fwd_configs},
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
    grad_q_acc = tl.zeros((BK, BT), dtype=tl.float32)
    grad_r_acc = tl.zeros((BK, BT), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    first_key_tile = 0
    if WINDOW_SIZE_LEFT >= 0:
        first_key_tile = tl.maximum(0, (i_t * BT - WINDOW_SIZE_LEFT + 1) // BT)
    for i_s in range(first_key_tile, tl.minimum(i_t + 1, NT)):
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
        grad_q_acc = tl.dot(tl.trans(b_k), tl.trans(gl.to(b_k.dtype)), out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(tl.trans(b_k), tl.trans(gu.to(b_k.dtype)), out_dtype=tl.float32, acc=grad_r_acc)

    tl.store(grad_q + q_offsets, tl.trans(grad_q_acc).to(grad_q.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(grad_r + q_offsets, tl.trans(grad_r_acc).to(grad_r.dtype.element_ty), mask=row_mask & head_mask)


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

    grad_k_acc = tl.zeros((BK, BT), dtype=tl.float32)
    grad_v_acc = tl.zeros((BK, BT), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    query_tile_end = NT
    if WINDOW_SIZE_LEFT >= 0:
        query_tile_end = tl.minimum(NT, i_s + 1 + tl.cdiv(WINDOW_SIZE_LEFT - 1, BT))
    for i_t in range(i_s, query_tile_end):
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
        grad_k_acc = tl.dot(tl.trans(b_q), gl.to(b_q.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(b_r), gu.to(b_r.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1.0 + bart_minus_rk)
        grad_v_acc = tl.dot(
            tl.trans(b_grad_o),
            weights.to(b_grad_o.dtype),
            out_dtype=tl.float32,
            acc=grad_v_acc,
        )

    grad_offsets = ((bos + offs_s[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    tl.store(
        grad_k_buf + grad_offsets,
        tl.trans(grad_k_acc).to(grad_k_buf.dtype.element_ty),
        mask=col_mask & head_mask,
    )
    tl.store(
        grad_v_buf + grad_offsets,
        tl.trans(grad_v_acc).to(grad_v_buf.dtype.element_ty),
        mask=col_mask & head_mask,
    )


@triton.autotune(
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
        # Finish dR before materializing rk/gl to shorten intermediate live ranges.
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


@triton.autotune(
    configs=PARALLAX_DQR_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_dqr_fused_configs},
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


@triton.autotune(
    configs=PARALLAX_DQR_SPLIT_CONFIGS,
    do_bench=_bench_with_maca_resource_limit,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_dqr_varlen_configs},
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
    kv_offsets = ((bos + col_indices[:, None]) * H + i_h) * K + o_k[None, :]
    out_offsets = ((bos + col_indices[:, None]) * HQ + i_hq) * K + o_k[None, :]
    b_k = tl.load(k + kv_offsets)
    b_v = tl.load(v + kv_offsets)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)

    # BT == BS gives one causal-border row tile. Every later row tile is
    # provably dense and aligned, so all row/head load masks are omitted.
    border_rows = i_col * BT + tl.arange(0, BT).to(tl.int64)
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
        row_indices = row_block * BT + tl.arange(0, BT).to(tl.int64)
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


@triton.autotune(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
    do_bench=_bench_with_maca_resource_limit,
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
    physical_i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_t = physical_i_t
    if not IS_VARLEN and WINDOW_SIZE_LEFT < 0:
        # Change logical task order without changing each CTA's computation.
        linear_id = physical_i_t + i_bh * tl.num_programs(0)
        num_head_tasks = tl.num_programs(1)
        i_t = linear_id // num_head_tasks
        i_bh = linear_id % num_head_tasks
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
    p_grad_k = grad_k + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BK, BS), dtype=tl.float32)
    grad_v_acc = tl.zeros((BK, BS), dtype=tl.float32)
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
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(b_q), gl.to(b_q.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(b_r), gu.to(b_r.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(grad_o_tile), weights.to(grad_o_tile.dtype), out_dtype=tl.float32, acc=grad_v_acc)

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
        grad_k_acc = tl.dot(tl.trans(b_q), gl.to(b_q.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(b_r), gu.to(b_r.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(grad_o_tile), weights.to(grad_o_tile.dtype), out_dtype=tl.float32, acc=grad_v_acc)

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
        grad_k_acc = tl.dot(tl.trans(b_q), gl.to(b_q.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(b_r), gu.to(b_r.dtype), out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(grad_o_tile), weights.to(grad_o_tile.dtype), out_dtype=tl.float32, acc=grad_v_acc)

    tl.store(p_grad_k, tl.trans(grad_k_acc).to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, tl.trans(grad_v_acc).to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.autotune(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_dkv_grouped_configs},
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
                mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
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
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )

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
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )

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
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )

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
    input_offsets = (batch_row * HQ + kv_head * G) * K + head_dim_offset

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
    use_c550_short = _use_c550_short_tiles(q, cu_seqlens)
    # Use fully masked traversal for dense padded-head inputs on C550.
    use_c550_padded_fwd = False
    if cu_seqlens is None and 256 < T <= 1024 and 32 < K < 64:
        name = torch.cuda.get_device_name(q.device).upper()
        use_c550_padded_fwd = "METAX" in name and "C550" in name
    BT = 32 if (use_c550_short or use_c550_padded_fwd) else _block_size(K, q.device.index, cu_seqlens)
    o = torch.empty_like(q)
    barv = torch.empty_like(q)
    stats = torch.empty((3, B, T, HQ), device=q.device, dtype=torch.float32)
    d1, bart, m = stats.unbind(0)

    if cu_seqlens is None and T > 8 * 32 and 32 < K <= 128 and "C550" in torch.cuda.get_device_name(q.device).upper():
        # Small query/KV tiles bound the two output accumulators; skip KV
        # blocks that cannot contribute to any row of a sliding window.
        c550_bt = 32
        parallel_parallax_fwd_kernel_c550_tiled[(triton.cdiv(T, c550_bt), B * HQ)](
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
            BT=c550_bt,
            NT=triton.cdiv(T, c550_bt),
            num_warps=2,
            num_stages=1,
        )
        return o, barv, d1, bart, m

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
        elif NQ <= 8 or use_c550_padded_fwd:
            # Retain the inherited Hopper launch override for this short-sequence shape.
            use_d100_short_w8 = (
                is_nvidia_hopper_device(q.device.index)
                and B == 3
                and T == 111
                and H == 2
                and HQ == 2
                and K == 100
                and window_size_left < 0
            )
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
                num_warps=(8 if use_d100_short_w8 else (4 if BT <= 64 else 8)),
                num_stages=1 if (use_c550_short or use_c550_padded_fwd or use_d100_short_w8) else 2,
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
        parallel_parallax_fwd_kernel[chunk_grid](
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
    return o, barv, d1, bart, m


def parallel_parallax_bwd(
    q, r, k, v, o, barv, d1, bart, m, grad_o, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1
):
    """Parallax backward (Triton). Returns grads matching `q, r, k, v`."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    use_c550_short = _use_c550_short_tiles(q, cu_seqlens)
    if cu_seqlens is None and 256 < T <= 1024 and 64 < K <= 128:
        name = torch.cuda.get_device_name(q.device).upper()
        use_c550_short = "METAX" in name and "C550" in name
    BT = 32 if use_c550_short else _block_size(K, q.device.index, cu_seqlens)
    # C550 D64 GQA spills with 64 query rows; D128 long DKV requires
    # 96 KiB shared memory at that tile size. Halving the query tile bounds
    # liveness without changing KV ownership or packed chunk mapping.
    if cu_seqlens is None and T > 256 and (64 < K <= 128 or (32 < K <= 64 and G in (4, 8))):
        if "C550" in torch.cuda.get_device_name(q.device).upper():
            BT = 32
    NQ = triton.cdiv(T, BT)
    dkv_programs = (len(chunk_indices) if cu_seqlens is not None else NQ) * B * H
    num_sms = get_num_sms(q.device.index)
    group_dkv_heads = G in (4, 8) or (G == 2 and (cu_seqlens is not None or T > 8192))
    # Retain inherited Hopper exceptions that use per-query-head CTAs and larger staging buffers.
    prefer_per_head_dkv = (
        is_nvidia_hopper_device(q.device.index)
        and cu_seqlens is None
        and K == 128
        and ((B == 2 and T == 8192 and HQ == 16 and G in (4, 8)) or (B == 4 and T == 16384 and HQ == 32 and G == 2))
    )
    # For this aligned Hopper specialization, only the first row tile intersects the causal diagonal.
    use_dense_aligned_dkv = (
        is_nvidia_hopper_device(q.device.index)
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
    # Parallelize query-head contributions instead of serializing G heads
    # inside one CTA; existing reduction combines per-head dK/dV buffers.
    # Keep dense D=128 on the same per-head path with the bounded query tile
    # above, without removing tuning candidates.
    if cu_seqlens is None and T > 256 and 32 < K <= 128 and G in (4, 8):
        if "C550" in torch.cuda.get_device_name(q.device).upper():
            prefer_per_head_dkv = True
    use_grouped_dkv = (
        # Grouped K/V reuse is the general long-sequence default.  The measured
        # dense Hopper exceptions above retain separate CTAs; varlen stays
        # grouped because T is the packed total rather than one sequence.
        group_dkv_heads
        # Short-multi DKV writes per-query-head partials before GQA reduction.
        # Keep HQ-sized buffers even when BT=32 raises NQ above 16.
        and not use_c550_short
        and not prefer_per_head_dkv
        and (cu_seqlens is not None or NQ > 16)
        and num_sms > 0
        and dkv_programs >= num_sms
    )

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
        elif NQ <= 16 or (use_c550_short and NQ <= 32):
            # Retain the inherited Hopper launch override for the short backward path.
            use_d100_short_bwd = (
                is_nvidia_hopper_device(q.device.index)
                and B == 3
                and T == 111
                and H == 2
                and HQ == 2
                and K == 100
                and window_size_left < 0
            )
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
                num_warps=(8 if use_d100_short_bwd else (4 if BT <= 64 else 8)),
                num_stages=1 if use_c550_short else 2,
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
                num_warps=(8 if use_d100_short_bwd else (4 if BT <= 64 else 8)),
                num_stages=1 if (use_c550_short or use_d100_short_bwd) else 2,
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
            if use_dense_aligned_dkv:
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
                    num_warps=8,
                    num_stages=3,
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
        parallel_parallax_bwd_kernel_preprocess[chunk_grid](
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
        parallel_parallax_bwd_kernel_dqr[chunk_grid](
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
        if use_grouped_dkv:
            parallel_parallax_bwd_kernel_dkv_grouped[grouped_chunk_grid](
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
            parallel_parallax_bwd_kernel_dkv[chunk_grid](
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
    if G > 1 and (not use_grouped_dkv):
        if grad_k is None:
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
        num_grad_k_elements = grad_k.numel()
        reduction_block = 256
        reduction_grid = (triton.cdiv(num_grad_k_elements, reduction_block),)
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
            raise ValueError("cu_seqlens must be on the same CUDA device as q")


def _parallax_input_guard(fn):
    """Preserve contiguous/device semantics without redundant device switches."""

    @wraps(fn)
    def wrapper(ctx, *args):
        # Both autograd entrypoints receive a tensor as their first data arg.
        tensor = args[0]
        packed_args = tuple(x.contiguous() if isinstance(x, torch.Tensor) else x for x in args)
        if tensor.device.index == torch.cuda.current_device():
            return fn(ctx, *packed_args)
        with torch.cuda.device(tensor.device):
            return fn(ctx, *packed_args)

    return wrapper


class ParallaxFunction(torch.autograd.Function):
    @staticmethod
    @_parallax_input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, _block_size(q.shape[-1], q.device.index, cu_seqlens))
            if cu_seqlens is not None
            else None
        )
        o, barv, d1, bart, m = parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @_parallax_input_guard
    @torch.amp.custom_bwd(device_type="cuda")
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
            keys in `[i - window_size + 1, i]`. If `None` or `-1`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training
            (FlashAttention convention). The batch size must be 1 when packing. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"parallel_parallax got unexpected keyword argument(s): {unexpected}")
    # Public callers may use the same unlimited-window sentinel as decode.
    if isinstance(window_size, int) and window_size == -1:
        window_size = None
    _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
