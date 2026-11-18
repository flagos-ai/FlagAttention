# Copyright 2026 FlagOS Contributors
# Copyright (c) 2026 Parallax authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from ..compat import exp2 as _exp2
from ..utils import get_num_sms

# Triton 3.6 rejects ordinary Python globals referenced by @triton.jit.  This
# must be an instantiated tl.constexpr (a type annotation is not sufficient).
_LOG2E = tl.constexpr(1.4426950408889634)
_BLOCK_N: int = 128
_MAX_SPLITS: int = 32
_NUM_WARPS: int = 1
_NUM_STAGES: int = 3


@triton.jit(do_not_specialize=["Skv"])
def _parallax_decode_grouped_matrix_kernel(
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
    BG: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_SEQLEN: tl.constexpr,
):
    """Share a regular KV tile across the query heads of one GQA group."""
    bh = tl.program_id(0).to(tl.int32)
    batch, head = bh // H, bh % H
    rows = tl.arange(0, BG)
    dims = tl.arange(0, BK)
    q_offsets = (batch * HQ + head * G + rows[:, None]) * K + dims[None, :]
    q_mask = (rows[:, None] < G) & (dims[None, :] < K)
    bq = tl.load(q + q_offsets, q_mask, other=0)
    br = tl.load(r + q_offsets, q_mask, other=0)
    hi = Skv
    if USE_SEQLEN:
        hi = tl.minimum(tl.maximum(tl.load(seqused_k + batch), 0), Skv)
    lo = 0
    if WINDOW_SIZE_LEFT >= 0:
        lo = tl.maximum(hi - WINDOW_SIZE_LEFT, 0)
    m = tl.full((BG,), -float("inf"), tl.float32)
    d1 = tl.zeros((BG,), tl.float32)
    d2 = tl.zeros((BG,), tl.float32)
    o1 = tl.zeros((BG, BK), tl.float32)
    o2 = tl.zeros((BG, BK), tl.float32)
    for block in range(lo // BN, tl.cdiv(hi, BN)):
        cols = block * BN + tl.arange(0, BN)
        valid = (cols >= lo) & (cols < hi)
        kv_offsets = ((batch * Skv + cols[:, None]) * H + head) * K + dims[None, :]
        kv_mask = valid[:, None] & (dims[None, :] < K)
        bk = tl.load(k + kv_offsets, kv_mask, other=0)
        s1 = tl.dot(bq, tl.trans(bk), out_dtype=tl.float32) * (scale * _LOG2E)
        s2 = tl.dot(br, tl.trans(bk), out_dtype=tl.float32)
        s1 = tl.where(valid[None, :], s1, -float("inf"))
        m_new = tl.maximum(m, tl.max(s1, axis=1))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - safe_m)
        p1 = _exp2(s1 - safe_m[:, None])
        p2 = p1 * s2
        bv = tl.load(v + kv_offsets, kv_mask, other=0).to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=1)
        d2 = d2 * alpha + tl.sum(p2, axis=1)
        # Keep weights and V in FP32, matching the vector decode's precision.
        o1 = tl.dot(p1, bv, acc=o1 * alpha[:, None], input_precision="ieee")
        o2 = tl.dot(p2, bv, acc=o2 * alpha[:, None], input_precision="ieee")
        m = m_new
    inv = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv[:, None] * (1.0 + d2[:, None] * inv[:, None]) - o2 * inv[:, None]
    tl.store(out + q_offsets, result, q_mask)


@triton.jit
def _parallax_decode_accumulate(
    q,
    r,
    k,
    v,
    scale,
    Skv,
    kv_lo,
    kv_hi,
    first_block,
    last_block,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    MASK_LOWER_BOUND: tl.constexpr,
):
    """Accumulate an FP32 Parallax state over a range of KV tiles."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // HQ
    i_h = (i_bh % HQ) // G
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
        mask_n = cols < kv_hi
        if MASK_LOWER_BOUND:
            mask_n &= cols >= kv_lo
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
        # Start V only after the score consumes K. This keeps the two full
        # [BN, BK] tiles from having overlapping live ranges.
        b_v = tl.load(v + kv_offsets, mask=mask_kv, other=0.0)
        b_v = b_v.to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=0)
        d2 = d2 * alpha + tl.sum(p2, axis=0)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new
    return m, d1, d2, o1, o2


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
    """One online-softmax reduction per batch/query-head."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // HQ

    kv_hi = Skv
    if USE_SEQLEN:
        kv_hi = tl.minimum(tl.maximum(tl.load(seqused_k + i_b), 0), Skv)
    kv_lo = 0
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_hi - WINDOW_SIZE_LEFT, 0)

    _, d1, d2, o1, o2 = _parallax_decode_accumulate(
        q,
        r,
        k,
        v,
        scale,
        Skv,
        kv_lo,
        kv_hi,
        kv_lo // BN,
        tl.cdiv(kv_hi, BN),
        HQ,
        H,
        G,
        K,
        BK,
        BN,
        True,
    )
    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    q_offset = i_bh * K + offs_d

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
    """Compute one fp32 online-softmax state per KV split."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_split = tl.program_id(1)

    num_blocks = tl.cdiv(Skv, BN)
    # Proportional boundaries keep all splits non-empty and balanced even when
    # the number of KV tiles is not divisible by NUM_SPLITS.
    first_block = (i_split * num_blocks) // NUM_SPLITS
    last_block = ((i_split + 1) * num_blocks) // NUM_SPLITS

    m, d1, d2, o1, o2 = _parallax_decode_accumulate(
        q,
        r,
        k,
        v,
        scale,
        Skv,
        0,
        Skv,
        first_block,
        last_block,
        HQ,
        H,
        G,
        K,
        BK,
        BN,
        False,
    )
    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K

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
    value = int(os.environ.get(name, default))
    if value not in choices:
        allowed = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {{{allowed}}}, got {value}")
    return value


def _launch_config(kv_len: int, head_dim: int) -> tuple[int, int, int, int]:
    default_block_n = 64 if head_dim == 128 and kv_len <= 512 else _BLOCK_N
    block_n = _env_choice("PARALLAX_BLOCK_N", default_block_n, (64, 128))
    num_warps = _env_choice("PARALLAX_NUM_WARPS", _NUM_WARPS, (1, 2, 4, 8))
    num_stages = _env_choice("PARALLAX_NUM_STAGES", _NUM_STAGES, (1, 2, 3, 4))
    max_splits = _env_choice("PARALLAX_MAX_SPLITS", _MAX_SPLITS, (8, 16, 32, 64))
    return block_n, num_warps, num_stages, max_splits


def _choose_num_splits(
    batch_heads: int,
    kv_len: int,
    device_index: int,
    block_n: int,
    max_splits: int,
) -> int:
    """Choose enough split programs for the available GCU multiprocessors."""
    override = os.environ.get("PARALLAX_DECODE_SPLITS")
    num_blocks = triton.cdiv(kv_len, block_n)
    if override is not None:
        requested = int(override)
        if requested < 1:
            raise ValueError("PARALLAX_DECODE_SPLITS must be >= 1")
        return min(requested, num_blocks, max_splits)
    # Short caches and sufficiently many heads use one program per head.
    # Otherwise, split the cache to expose more parallel work on the GCU.
    if kv_len < 512 or batch_heads >= get_num_sms(device_index):
        return 1
    # Aim for two waves, capped so every split owns at least one KV tile.
    requested = triton.cdiv(2 * get_num_sms(device_index), batch_heads)
    return max(1, min(requested, num_blocks, max_splits))


@lru_cache(maxsize=None)
def _split_workspace(
    device: torch.device, batch_heads: int, num_splits: int, head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Reuse FP32 split-state buffers for the same device and shape."""
    scalar_shape = (batch_heads, num_splits)
    vector_shape = (batch_heads, num_splits, head_dim)
    return tuple(
        torch.empty(shape, device=device, dtype=torch.float32)
        for shape in (scalar_shape, scalar_shape, scalar_shape, vector_shape, vector_shape)
    )


def _is_gcu_graph_capturing() -> bool:
    gcu = getattr(torch, "gcu", None)
    is_capturing = getattr(gcu, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except RuntimeError:
        return False


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
        raise ValueError(f"Parallax decode requires seqlen_q=1, got {q.shape[1]}")
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("k and v must have the same (B, L, H_kv, D) shape")
    B, _, HQ, D = q.shape
    Bk, L, H, Dk = k.shape
    if B != Bk or D != Dk:
        raise ValueError("q/r and k/v batch size and head dimension must match")
    if L < 1:
        raise ValueError("KV cache length must be positive")
    if HQ < 1 or H < 1:
        raise ValueError("head counts must be positive")
    if HQ % H:
        raise ValueError(f"H_q ({HQ}) must be divisible by H_kv ({H}) for GQA")
    if D not in (64, 128):
        raise ValueError(f"Parallax decode supports head_dim in {{64, 128}}, got {D}")
    tensors = (q, r, k, v)
    if not all(x.device.type == "gcu" for x in tensors):
        raise ValueError("q, r, k and v must be GCU tensors")
    if not all(x.device == q.device for x in tensors):
        raise ValueError("q, r, k and v must be on the same GCU device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/r/k/v must be fp16 or bf16, got {q.dtype}")
    if not all(x.dtype == q.dtype for x in tensors):
        raise TypeError("q, r, k and v must have the same dtype")
    if seqused_k is not None:
        if seqused_k.shape != (B,) or seqused_k.dtype != torch.int32:
            raise ValueError(f"seqused_k must be int32 with shape ({B},)")
        if seqused_k.device != q.device:
            raise ValueError("seqused_k must be on the same GCU device as q")
        if not _is_gcu_graph_capturing():
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
        raise ValueError("window_size must be None, an int, or a (left, right) pair") from exc
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
    """Enflame Parallax decode with an FA-style KV-cache interface.

    Dense KV, MHA/GQA, fp16/bf16, runtime ``seqused_k``, sliding windows and a
    caller-owned output are supported.  Paged KV is not implemented.
    """
    if page_table is not None:
        raise NotImplementedError("paged KV (page_table) is not implemented")
    B, L, HQ, H, D = _validate_inputs(q, r, k_cache, v_cache, seqused_k, out)
    window_left = _window_size_to_left(window_size)
    if window_left < -1:
        raise ValueError("window_size left must be -1 or non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q, r, k_cache, v_cache = (x if x.is_contiguous() else x.contiguous() for x in (q, r, k_cache, v_cache))
    if seqused_k is not None and not seqused_k.is_contiguous():
        seqused_k = seqused_k.contiguous()
    result = torch.empty_like(q) if out is None else out
    BK = triton.next_power_of_2(D)
    batch_heads = B * HQ
    block_n, num_warps, num_stages, max_splits = _launch_config(L, D)
    # One regular KV tile serves all query heads sharing this KV head.
    # Bound the live row tile; larger groups retain the existing decode path.
    group_size = HQ // H
    if 1 < group_size <= 16:
        _parallax_decode_grouped_matrix_kernel[(B * H,)](
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
            G=group_size,
            K=D,
            BK=BK,
            BN=block_n,
            BG=max(16, triton.next_power_of_2(group_size)),
            WINDOW_SIZE_LEFT=window_left,
            USE_SEQLEN=seqused_k is not None,
            num_warps=1,
            num_stages=1,
        )
        return result
    device_index = q.device.index
    if device_index is None:
        device_index = torch.gcu.current_device()

    # Runtime lengths and windows use the one-CTA path so no split scans cache
    # positions which are known to be inactive.
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
            G=group_size,
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
        G=group_size,
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
    """Parallax decode alias for the single-query Enflame serving interface."""
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


__all__ = ["parallax_attn_with_kvcache", "parallax_decode"]
