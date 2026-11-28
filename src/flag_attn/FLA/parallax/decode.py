# Copyright (c) 2026 Parallax authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Portable single-token Parallax attention using online softmax.

For p = softmax(scale * q @ k.T) and t = r @ k.T, return
(p @ v) * (1 + sum(p * t)) - (p * t) @ v. Accumulation is FP32.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG2E = tl.constexpr(1.4426950408889634)


def __getattr__(name):
    # Compatibility metadata is queried lazily; generic kernels need no extension.
    if name == "HAS_TLE":
        from flag_attn.utils import has_triton_tle

        return has_triton_tle(required_api=("load",))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_graph_capturing(device):
    backend = torch.get_device_module(device)
    capturing = getattr(backend, "is_current_stream_capturing", None)
    return bool(capturing()) if capturing is not None else False

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

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(out + q_offset, result, mask=mask_d)

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

def _split_workspace(
    device: torch.device, batch_heads: int, num_splits: int, head_dim: int
) -> tuple[torch.Tensor, ...]:
    """Invocation-local scratch also remains stable under graph replay."""
    scalar_shape = (batch_heads, num_splits)
    vector_shape = (batch_heads, num_splits, head_dim)
    return tuple(
        torch.empty(shape, device=device, dtype=torch.float32)
        for shape in (scalar_shape, scalar_shape, scalar_shape, vector_shape, vector_shape)
    )

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
    if HQ < 1 or H < 1 or D < 1:
        raise ValueError("head counts and head dimension must be positive")
    if HQ % H:
        raise ValueError(f"H_q ({HQ}) must be divisible by H_kv ({H}) for GQA")
    if D not in (64, 128):
        raise ValueError(f"Parallax decode supports head_dim in {{64, 128}}, got {D}")
    tensors = (q, r, k, v)
    if not all(x.device.type != "cpu" for x in tensors):
        raise ValueError("q, r, k and v must be accelerator tensors")
    if not all(x.device == q.device for x in tensors):
        raise ValueError("q, r, k and v must be on the same device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/r/k/v must be fp16 or bf16, got {q.dtype}")
    if not all(x.dtype == q.dtype for x in tensors):
        raise TypeError("q, r, k and v must have the same dtype")
    if seqused_k is not None:
        if seqused_k.shape != (B,) or seqused_k.dtype != torch.int32:
            raise ValueError(f"seqused_k must be int32 with shape ({B},)")
        if seqused_k.device != q.device:
            raise ValueError("seqused_k must be on the same device as q")
        if not _is_graph_capturing(q.device):
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
    """Parallax decode with the CuTe baseline's FA-style interface.

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
        BN=32,
        WINDOW_SIZE_LEFT=window_left,
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
    """Back-compatible decode entry matching the CuTe decode alias."""
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
