# Copyright (c) 2026 Parallax authors.
# SPDX-License-Identifier: MIT
"""Single-token Parallax decode written in Triton with TLE async loads.

This is the TLE counterpart of :mod:`flag_attn.parallax.cute.parallax_decode`.  It uses
the same ``(B, 1, H_q, D)`` query and ``(B, L, H_kv, D)`` cache layouts and
evaluates the same composite Parallax attention formula::

    s1_j = scale * dot(q, k_j)
    s2_j = dot(r, k_j)
    p1   = exp(s1)
    p2   = p1 * s2
    out  = O1 / d1 * (1 + d2 / d1) - O2 / d1

where ``d1=sum(p1)``, ``d2=sum(p2)``, ``O1=sum(p1*v)`` and
``O2=sum(p2*v)``.  All online-softmax state and the split merge are fp32.

The implementation has two forward-only paths:

* one CTA per ``(batch, query-head)`` for short caches, sliding windows and
  runtime ``seqused_k``;
* split-KV partial reductions followed by a log-sum-exp merge for long-cache,
  low-occupancy decode.

K and V tiles are fetched with ``tle.load(..., is_async=True)``.  Therefore a
FlagTree build containing ``triton.experimental.tle`` is required; upstream
Triton alone is deliberately not treated as a silent fallback.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
except ImportError as exc:
    tle = None
    _TLE_IMPORT_ERROR = exc
else:
    _TLE_IMPORT_ERROR = None

HAS_TLE = tle is not None and hasattr(tle, "load")
# Triton 3.6 rejects ordinary Python globals referenced by @triton.jit.  This
# must be an instantiated tl.constexpr (a type annotation is not sufficient).
_LOG2E = tl.constexpr(1.4426950408889634)
_BLOCK_N: int = 128
_MAX_SPLITS: int = 32
_NUM_WARPS: int = 4
_NUM_STAGES: int = 3


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics({"USE_SEQLEN": lambda args: args["seqused_k"] is not None})
@triton.jit(do_not_specialize=["Skv"])
def _parallax_decode_tle_kernel(
    q,
    r,
    k,
    v,
    out,
    scale,
    seqused_k,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_SEQLEN: tl.constexpr,
):
    """One online-softmax reduction per batch/query-head."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    kv_hi = Skv
    if USE_SEQLEN:
        kv_hi = tl.minimum(tl.maximum(tl.load(seqused_k + i_b), 0), Skv)
    kv_lo = 0
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_hi - WINDOW_SIZE_LEFT, 0)

    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    q_offset = i_bh * K + offs_d
    b_q = tl.load(q + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    b_r = tl.load(r + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    scale_log2 = scale * _LOG2E

    m = tl.full((1,), -float("inf"), tl.float32)
    d1 = tl.zeros((1,), tl.float32)
    d2 = tl.zeros((1,), tl.float32)
    o1 = tl.zeros((BK,), tl.float32)
    o2 = tl.zeros((BK,), tl.float32)

    first_block = kv_lo // BN
    last_block = tl.cdiv(kv_hi, BN)
    for block_id in range(first_block, last_block):
        cols = block_id * BN + tl.arange(0, BN)
        mask_n = (cols >= kv_lo) & (cols < kv_hi)
        kv_offsets = (
            ((i_b * Skv + cols[:, None]) * H + i_h) * K
            + offs_d[None, :]
        )
        mask_kv = mask_n[:, None] & mask_d[None, :]
        # TLE lowers these copies through its asynchronous load pipeline.
        b_k = tle.load(k + kv_offsets, mask=mask_kv, other=0.0, is_async=True)

        s1 = tl.sum(b_k.to(tl.float32) * b_q[None, :], axis=1) * scale_log2
        s2 = tl.sum(b_k.to(tl.float32) * b_r[None, :], axis=1)
        s1 = tl.where(mask_n, s1, -float("inf"))

        m_new = tl.maximum(m, tl.max(s1, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - m_safe)
        p1 = _exp2(s1 - m_safe)
        p2 = p1 * s2
        # Start V only after the score consumes K. This keeps the two full
        # [BN, BK] tiles from having overlapping live ranges.
        b_v = tle.load(v + kv_offsets, mask=mask_kv, other=0.0, is_async=True)
        b_v = b_v.to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=0)
        d2 = d2 * alpha + tl.sum(p2, axis=0)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(out + q_offset, result, mask=mask_d)


@triton.jit(do_not_specialize=["Skv"])
def _parallax_decode_tle_split_kernel(
    q,
    r,
    k,
    v,
    partial_m,
    partial_d1,
    partial_d2,
    partial_o1,
    partial_o2,
    scale,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Compute one fp32 online-softmax state per KV split."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_split = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    num_blocks = tl.cdiv(Skv, BN)
    # Proportional boundaries keep all splits non-empty and balanced even when
    # the number of KV tiles is not divisible by NUM_SPLITS.
    first_block = (i_split * num_blocks) // NUM_SPLITS
    last_block = ((i_split + 1) * num_blocks) // NUM_SPLITS

    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    q_offset = i_bh * K + offs_d
    b_q = tl.load(q + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    b_r = tl.load(r + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    scale_log2 = scale * _LOG2E

    m = tl.full((1,), -float("inf"), tl.float32)
    d1 = tl.zeros((1,), tl.float32)
    d2 = tl.zeros((1,), tl.float32)
    o1 = tl.zeros((BK,), tl.float32)
    o2 = tl.zeros((BK,), tl.float32)

    for block_id in range(first_block, last_block):
        cols = block_id * BN + tl.arange(0, BN)
        mask_n = cols < Skv
        kv_offsets = (
            ((i_b * Skv + cols[:, None]) * H + i_h) * K
            + offs_d[None, :]
        )
        mask_kv = mask_n[:, None] & mask_d[None, :]
        b_k = tle.load(k + kv_offsets, mask=mask_kv, other=0.0, is_async=True)

        s1 = tl.sum(b_k.to(tl.float32) * b_q[None, :], axis=1) * scale_log2
        s2 = tl.sum(b_k.to(tl.float32) * b_r[None, :], axis=1)
        s1 = tl.where(mask_n, s1, -float("inf"))

        m_new = tl.maximum(m, tl.max(s1, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - m_safe)
        p1 = _exp2(s1 - m_safe)
        p2 = p1 * s2
        b_v = tle.load(v + kv_offsets, mask=mask_kv, other=0.0, is_async=True)
        b_v = b_v.to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=0)
        d2 = d2 * alpha + tl.sum(p2, axis=0)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new

    state_offset = i_bh * NUM_SPLITS + i_split
    # m/d1/d2 deliberately retain shape (1,) throughout the online reduction.
    # Triton 3.6 requires a block pointer tensor when storing a block value;
    # a scalar pointer plus a one-element block is rejected.
    state_ptr_offset = state_offset + tl.arange(0, 1)
    tl.store(partial_m + state_ptr_offset, m)
    tl.store(partial_d1 + state_ptr_offset, d1)
    tl.store(partial_d2 + state_ptr_offset, d2)
    vector_offset = state_offset * K + offs_d
    tl.store(partial_o1 + vector_offset, o1, mask=mask_d)
    tl.store(partial_o2 + vector_offset, o2, mask=mask_d)


@triton.jit
def _parallax_decode_tle_merge_kernel(
    partial_m,
    partial_d1,
    partial_d2,
    partial_o1,
    partial_o2,
    out,
    K: tl.constexpr,
    BK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BSPLIT: tl.constexpr,
):
    """Log-sum-exp merge of the split-local Parallax states."""
    i_bh = tl.program_id(0).to(tl.int64)
    offs_s = tl.arange(0, BSPLIT)
    mask_s = offs_s < NUM_SPLITS
    state_offsets = i_bh * NUM_SPLITS + offs_s

    m = tl.load(partial_m + state_offsets, mask=mask_s, other=-float("inf"))
    m_global = tl.max(m, axis=0)
    m_safe = tl.where(m_global == -float("inf"), 0.0, m_global)
    alpha = _exp2(m - m_safe)
    d1 = tl.sum(
        tl.load(partial_d1 + state_offsets, mask=mask_s, other=0.0) * alpha,
        axis=0,
    )
    d2 = tl.sum(
        tl.load(partial_d2 + state_offsets, mask=mask_s, other=0.0) * alpha,
        axis=0,
    )

    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    vector_offsets = state_offsets[:, None] * K + offs_d[None, :]
    vector_mask = mask_s[:, None] & mask_d[None, :]
    o1 = tl.sum(
        tl.load(partial_o1 + vector_offsets, mask=vector_mask, other=0.0)
        * alpha[:, None],
        axis=0,
    )
    o2 = tl.sum(
        tl.load(partial_o2 + vector_offsets, mask=vector_mask, other=0.0)
        * alpha[:, None],
        axis=0,
    )
    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(out + i_bh * K + offs_d, result, mask=mask_d)


@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _env_choice(name: str, default: int, choices: tuple[int, ...]) -> int:
    value = int(os.environ.get(name, default))
    if value not in choices:
        allowed = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {{{allowed}}}, got {value}")
    return value


def _launch_config(kv_len: int, head_dim: int) -> tuple[int, int, int, int]:
    """Select a shape-aware default while retaining explicit tuning overrides."""
    # H100 CUDA-graph measurements show that D=128/L=512 improves from
    # 7.36 us at BN=128 to 6.42 us at BN=64.  Keep BN=128 elsewhere until
    # those shape families have equivalent tuning evidence.
    default_block_n = 64 if head_dim == 128 and kv_len <= 512 else _BLOCK_N
    block_n = _env_choice(
        "PARALLAX_TLE_BLOCK_N", default_block_n, (64, 128)
    )
    num_warps = _env_choice("PARALLAX_TLE_NUM_WARPS", _NUM_WARPS, (2, 4, 8))
    num_stages = _env_choice("PARALLAX_TLE_NUM_STAGES", _NUM_STAGES, (1, 2, 3, 4))
    max_splits = _env_choice(
        "PARALLAX_TLE_MAX_SPLITS", _MAX_SPLITS, (8, 16, 32, 64)
    )
    return block_n, num_warps, num_stages, max_splits


def _choose_num_splits(
    batch_heads: int,
    kv_len: int,
    device_index: int,
    block_n: int,
    max_splits: int,
) -> int:
    """Choose enough split CTAs to fill an under-occupied GPU."""
    override = os.environ.get("PARALLAX_TLE_DECODE_SPLITS")
    num_blocks = triton.cdiv(kv_len, block_n)
    if override is not None:
        requested = int(override)
        if requested < 1:
            raise ValueError("PARALLAX_TLE_DECODE_SPLITS must be >= 1")
        return min(requested, num_blocks, max_splits)
    # Keep a single CTA/head only below 512 keys. On H100, B=1/Hq=8/L=512
    # improves from 14.56 us at one split to 7.47 us at four splits: the
    # second merge launch is much cheaper than leaving 124 of 132 SMs idle.
    # The num_blocks clamp below naturally selects four splits for L=512.
    if kv_len < 512 or batch_heads >= _num_sms(device_index):
        return 1
    # Aim for two waves, capped so every split owns at least one KV tile.
    requested = triton.cdiv(2 * _num_sms(device_index), batch_heads)
    return max(1, min(requested, num_blocks, max_splits))


_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _split_workspace(
    device: torch.device, batch_heads: int, num_splits: int, head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Return stable split buffers (also avoids allocations during timing)."""
    key = (device.type, device.index, batch_heads, num_splits, head_dim)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        scalar_shape = (batch_heads, num_splits)
        vector_shape = (batch_heads, num_splits, head_dim)
        ws = (
            torch.empty(scalar_shape, device=device, dtype=torch.float32),
            torch.empty(scalar_shape, device=device, dtype=torch.float32),
            torch.empty(scalar_shape, device=device, dtype=torch.float32),
            torch.empty(vector_shape, device=device, dtype=torch.float32),
            torch.empty(vector_shape, device=device, dtype=torch.float32),
        )
        _WORKSPACE_CACHE[key] = ws
    return ws


def _require_tle() -> None:
    if not HAS_TLE:
        raise ImportError(
            "Parallax TLE decode requires a FlagTree Triton build exposing "
            "triton.experimental.tle.language. Upstream Triton does not ship "
            "this extension."
        ) from _TLE_IMPORT_ERROR


def _validate_inputs(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor | None,
    out: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or r.shape != q.shape:
        raise ValueError("q and r must have the same (B, 1, H_q, D) shape")
    if q.shape[1] != 1:
        raise ValueError(f"TLE decode requires seqlen_q=1, got {q.shape[1]}")
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("k and v must have the same (B, L, H_kv, D) shape")
    B, _, HQ, D = q.shape
    Bk, L, H, Dk = k.shape
    if B != Bk or D != Dk:
        raise ValueError("q/r and k/v batch size and head dimension must match")
    if L < 1:
        raise ValueError("KV cache length must be positive")
    if HQ < 1 or H < 1 or D < 1:
        raise ValueError("head counts and head dimension must be positive")
    if HQ % H:
        raise ValueError(f"H_q ({HQ}) must be divisible by H_kv ({H}) for GQA")
    if D not in (64, 128):
        raise ValueError(f"TLE decode supports head_dim in {{64, 128}}, got {D}")
    tensors = (q, r, k, v)
    if not all(x.is_cuda for x in tensors):
        raise ValueError("q, r, k and v must be CUDA tensors")
    if not all(x.device == q.device for x in tensors):
        raise ValueError("q, r, k and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/r/k/v must be fp16 or bf16, got {q.dtype}")
    if not all(x.dtype == q.dtype for x in tensors):
        raise TypeError("q, r, k and v must have the same dtype")
    if seqused_k is not None:
        if seqused_k.shape != (B,) or seqused_k.dtype != torch.int32:
            raise ValueError(f"seqused_k must be int32 with shape ({B},)")
        if not seqused_k.is_cuda or seqused_k.device != q.device:
            raise ValueError("seqused_k must be on the same CUDA device as q")
        if not torch.cuda.is_current_stream_capturing():
            lo = int(seqused_k.min().item())
            hi = int(seqused_k.max().item())
            if lo < 1 or hi > L:
                raise ValueError(f"seqused_k values must be in [1, {L}], got [{lo}, {hi}]")
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match q's shape, dtype and device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    return B, L, HQ, H, D


def _window_size_to_left(window_size) -> int:
    if window_size is None:
        return -1
    if isinstance(window_size, int):
        return int(window_size)
    try:
        left, right = window_size
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "window_size must be None, an int, or a (left, right) pair"
        ) from exc
    if right is not None and right > 0:
        raise ValueError("single-token causal decode requires window_size right <= 0")
    return int(left)


def parallax_attn_with_kvcache(
    q: torch.Tensor,
    r: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    window_size=None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """TLE Parallax decode with the CuTe baseline's FA-style interface.

    Dense KV, MHA/GQA, fp16/bf16, runtime ``seqused_k``, sliding windows and a
    caller-owned output are supported.  Paged KV is not implemented.
    """
    _require_tle()
    if page_table is not None:
        raise NotImplementedError("paged KV (page_table) is not implemented")
    B, L, HQ, H, D = _validate_inputs(q, r, k_cache, v_cache, seqused_k, out)
    window_left = _window_size_to_left(window_size)
    if window_left < -1:
        raise ValueError("window_size left must be -1 or non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q, r, k_cache, v_cache = (
        x if x.is_contiguous() else x.contiguous() for x in (q, r, k_cache, v_cache)
    )
    if seqused_k is not None and not seqused_k.is_contiguous():
        seqused_k = seqused_k.contiguous()
    result = torch.empty_like(q) if out is None else out
    BK = triton.next_power_of_2(D)
    batch_heads = B * HQ
    block_n, num_warps, num_stages, max_splits = _launch_config(L, D)
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    # Runtime lengths and windows use the one-CTA path so no split scans cache
    # positions which are known to be inactive.
    num_splits = 1
    if seqused_k is None and window_left < 0:
        num_splits = _choose_num_splits(
            batch_heads, L, device_index, block_n, max_splits
        )

    if num_splits == 1:
        _parallax_decode_tle_kernel[(batch_heads,)](
            q,
            r,
            k_cache,
            v_cache,
            result,
            float(scale),
            seqused_k,
            L,
            HQ=HQ,
            H=H,
            G=HQ // H,
            K=D,
            BK=BK,
            BN=block_n,
            WINDOW_SIZE_LEFT=window_left,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return result

    pm, pd1, pd2, po1, po2 = _split_workspace(
        q.device, batch_heads, num_splits, D
    )
    _parallax_decode_tle_split_kernel[(batch_heads, num_splits)](
        q,
        r,
        k_cache,
        v_cache,
        pm,
        pd1,
        pd2,
        po1,
        po2,
        float(scale),
        L,
        HQ=HQ,
        H=H,
        G=HQ // H,
        K=D,
        BK=BK,
        BN=block_n,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _parallax_decode_tle_merge_kernel[(batch_heads,)](
        pm,
        pd1,
        pd2,
        po1,
        po2,
        result,
        K=D,
        BK=BK,
        NUM_SPLITS=num_splits,
        BSPLIT=triton.next_power_of_2(num_splits),
        num_warps=4,
        num_stages=1,
    )
    return result


def parallax_decode(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float | None = None,
    *,
    window_size_left: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Back-compatible TLE decode entry matching the CuTe decode alias."""
    window_size = None if window_size_left < 0 else (window_size_left, 0)
    return parallax_attn_with_kvcache(
        q,
        r,
        k,
        v,
        window_size=window_size,
        scale=qk_scale,
        out=out,
    )


__all__ = ["HAS_TLE", "parallax_attn_with_kvcache", "parallax_decode"]
