# Copyright (c) 2026 Parallax authors.
# Copyright (c) 2026 Yifei Zuo
# SPDX-License-Identifier: MIT
#
# See LICENSE in this directory for the full license text.

from __future__ import annotations

import math
import os
from functools import lru_cache, wraps

import torch
import triton
import triton.language as tl

from flag_attn.FLA.utils import get_num_sms, is_cuda_graph_capturing


# Compatibility alias; this backend uses ordinary Triton.
HAS_TRITON_DECODE = True
HAS_TLE = HAS_TRITON_DECODE

_LOG2E = tl.constexpr(1.4426950408889634)
_DECODE_CONFIG = {
    "BLOCK_N": (128, (64, 128)),
    "NUM_WARPS": (4, (2, 4, 8)),
    "NUM_STAGES": (3, (1, 2, 3, 4)),
    "MAX_SPLITS": (32, (8, 16, 32, 64)),
}


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics({"USE_SEQLEN": lambda args: args["seqused_k"] is not None})
@triton.jit(do_not_specialize=["Skv"])
def _parallax_decode_kernel(
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
        kv_offsets = ((i_b * Skv + cols[:, None]) * H + i_h) * K + offs_d[None, :]
        mask_kv = mask_n[:, None] & mask_d[None, :]
        b_k = tl.load(k + kv_offsets, mask=mask_kv, other=0.0)

        s1 = tl.sum(b_k.to(tl.float32) * b_q[None, :], axis=1) * scale_log2
        s2 = tl.sum(b_k.to(tl.float32) * b_r[None, :], axis=1)
        s1 = tl.where(mask_n, s1, -float("inf"))

        m_new = tl.maximum(m, tl.max(s1, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - m_safe)
        p1 = _exp2(s1 - m_safe)
        p2 = p1 * s2
        b_v = tl.load(v + kv_offsets, mask=mask_kv, other=0.0)
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
def _parallax_decode_split_kernel(
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
    i_bh = tl.program_id(0).to(tl.int64)
    i_split = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    num_blocks = tl.cdiv(Skv, BN)
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
        kv_offsets = ((i_b * Skv + cols[:, None]) * H + i_h) * K + offs_d[None, :]
        mask_kv = mask_n[:, None] & mask_d[None, :]
        b_k = tl.load(k + kv_offsets, mask=mask_kv, other=0.0)

        s1 = tl.sum(b_k.to(tl.float32) * b_q[None, :], axis=1) * scale_log2
        s2 = tl.sum(b_k.to(tl.float32) * b_r[None, :], axis=1)
        s1 = tl.where(mask_n, s1, -float("inf"))

        m_new = tl.maximum(m, tl.max(s1, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - m_safe)
        p1 = _exp2(s1 - m_safe)
        p2 = p1 * s2
        b_v = tl.load(v + kv_offsets, mask=mask_kv, other=0.0)
        b_v = b_v.to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=0)
        d2 = d2 * alpha + tl.sum(p2, axis=0)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new

    state_offset = i_bh * NUM_SPLITS + i_split
    state_ptr_offset = state_offset + tl.arange(0, 1)
    tl.store(partial_m + state_ptr_offset, m)
    tl.store(partial_d1 + state_ptr_offset, d1)
    tl.store(partial_d2 + state_ptr_offset, d2)
    vector_offset = state_offset * K + offs_d
    tl.store(partial_o1 + vector_offset, o1, mask=mask_d)
    tl.store(partial_o2 + vector_offset, o2, mask=mask_d)


@triton.jit
def _parallax_decode_merge_kernel(
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
        tl.load(partial_o1 + vector_offsets, mask=vector_mask, other=0.0) * alpha[:, None],
        axis=0,
    )
    o2 = tl.sum(
        tl.load(partial_o2 + vector_offsets, mask=vector_mask, other=0.0) * alpha[:, None],
        axis=0,
    )
    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(out + i_bh * K + offs_d, result, mask=mask_d)


@lru_cache(maxsize=None)
def _env_choice(name: str, default: int, choices: tuple[int, ...]) -> int:
    legacy_name = name.replace("PARALLAX_DECODE_", "PARALLAX_TLE_", 1)
    value = int(os.environ.get(name, os.environ.get(legacy_name, default)))
    if value not in choices:
        allowed = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {{{allowed}}}, got {value}")
    return value


def _launch_config(kv_len: int, head_dim: int) -> tuple[int, int, int, int]:
    default_block_n, block_n_choices = _DECODE_CONFIG["BLOCK_N"]
    if head_dim == 128 and kv_len <= 512:
        default_block_n = 64
    block_n = _env_choice("PARALLAX_DECODE_BLOCK_N", default_block_n, block_n_choices)
    num_warps = _env_choice("PARALLAX_DECODE_NUM_WARPS", *_DECODE_CONFIG["NUM_WARPS"])
    num_stages = _env_choice("PARALLAX_DECODE_NUM_STAGES", *_DECODE_CONFIG["NUM_STAGES"])
    max_splits = _env_choice("PARALLAX_DECODE_MAX_SPLITS", *_DECODE_CONFIG["MAX_SPLITS"])
    return block_n, num_warps, num_stages, max_splits


def _choose_num_splits(
    batch_heads: int,
    kv_len: int,
    device_index: int,
    block_n: int,
    max_splits: int,
) -> int:
    override = os.environ.get("PARALLAX_TLE_DECODE_SPLITS")
    num_blocks = triton.cdiv(kv_len, block_n)
    if override is not None:
        requested = int(override)
        if requested < 1:
            raise ValueError("PARALLAX_TLE_DECODE_SPLITS must be >= 1")
        return min(requested, num_blocks, max_splits)
    if kv_len < 512 or batch_heads >= get_num_sms(device_index):
        return 1
    requested = triton.cdiv(2 * get_num_sms(device_index), batch_heads)
    return max(1, min(requested, num_blocks, max_splits))


def _split_workspace(
    device: torch.device, batch_heads: int, num_splits: int, head_dim: int
) -> tuple[torch.Tensor, ...]:
    scalar_shape = (batch_heads, num_splits)
    vector_shape = (batch_heads, num_splits, head_dim)
    return (
        torch.empty(scalar_shape, device=device, dtype=torch.float32),
        torch.empty(scalar_shape, device=device, dtype=torch.float32),
        torch.empty(scalar_shape, device=device, dtype=torch.float32),
        torch.empty(vector_shape, device=device, dtype=torch.float32),
        torch.empty(vector_shape, device=device, dtype=torch.float32),
    )


def _window_size_to_left(window_size) -> int:
    if window_size is None:
        return -1
    if isinstance(window_size, int):
        return int(window_size)
    try:
        left, right = window_size
    except (TypeError, ValueError) as exc:
        raise ValueError("window_size must be None, an int, or a (left, right) pair") from exc
    if right is not None and right > 0:
        raise ValueError("single-token causal decode requires window_size right <= 0")
    return int(left)


def _decode_device_guard(fn):
    @wraps(fn)
    def wrapper(q, *args, **kwargs):
        if not isinstance(q, torch.Tensor) or q.device.type != "cuda":
            return fn(q, *args, **kwargs)
        if q.device.index == torch.cuda.current_device():
            return fn(q, *args, **kwargs)
        with torch.cuda.device(q.device):
            return fn(q, *args, **kwargs)

    return wrapper


@_decode_device_guard
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
    """Single-token causal Parallax attention over a dense KV cache.

    Dense KV, MHA/GQA, fp16/bf16, runtime ``seqused_k``, sliding windows and a
    caller-owned output are supported.  Paged KV is not implemented.
    """
    if page_table is not None:
        raise NotImplementedError("paged KV (page_table) is not implemented")
    B, Sq, HQ, D = q.shape
    Bk, L, H, Dk = k_cache.shape
    if Sq != 1 or r.shape != q.shape:
        raise ValueError("q and r must have the same (B, 1, H_q, D) shape")
    if v_cache.shape != k_cache.shape or (Bk, Dk) != (B, D):
        raise ValueError("k/v must have matching shapes and share q's batch size and head dimension")
    if min(L, HQ, H) < 1 or HQ % H:
        raise ValueError("KV length and head counts must be positive, with H_q divisible by H_kv")
    if D not in (64, 128):
        raise ValueError(f"Triton decode supports head_dim in {{64, 128}}, got {D}")
    if not q.is_cuda or any(x.device != q.device for x in (r, k_cache, v_cache)):
        raise ValueError("q, r, k and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype for x in (r, k_cache, v_cache)):
        raise TypeError("q, r, k and v must have the same fp16 or bf16 dtype")
    if seqused_k is not None:
        if seqused_k.shape != (B,) or seqused_k.dtype != torch.int32 or seqused_k.device != q.device:
            raise ValueError("seqused_k must be int32 with shape (B,) on q's device")
        if not is_cuda_graph_capturing():
            lo, hi = int(seqused_k.min().item()), int(seqused_k.max().item())
            if lo < 1 or hi > L:
                raise ValueError(f"seqused_k values must be in [1, {L}], got [{lo}, {hi}]")
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or not out.is_contiguous():
            raise ValueError("out must be contiguous and match q's shape, dtype and device")
    window_left = _window_size_to_left(window_size)
    if window_left < -1:
        raise ValueError("window_size left must be -1 or non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q, r, k_cache, v_cache = (x.contiguous() for x in (q, r, k_cache, v_cache))
    if seqused_k is not None and not seqused_k.is_contiguous():
        seqused_k = seqused_k.contiguous()
    result = torch.empty_like(q) if out is None else out
    BK = triton.next_power_of_2(D)
    batch_heads = B * HQ
    block_n, num_warps, num_stages, max_splits = _launch_config(L, D)
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    num_splits = 1
    if seqused_k is None and window_left < 0:
        num_splits = _choose_num_splits(batch_heads, L, device_index, block_n, max_splits)

    if num_splits == 1:
        _parallax_decode_kernel[(batch_heads,)](
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

    pm, pd1, pd2, po1, po2 = _split_workspace(q.device, batch_heads, num_splits, D)
    _parallax_decode_split_kernel[(batch_heads, num_splits)](
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
    _parallax_decode_merge_kernel[(batch_heads,)](
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
    """Back-compatible Triton decode entry matching the CuTe decode alias."""
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
