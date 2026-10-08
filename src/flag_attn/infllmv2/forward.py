# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Forward implementation for continuous-packed InfLLM-V2 attention.

The standard Triton path is complete on its own. Hopper TLE kernels live in
this module as optional specializations and are selected only when the local
Triton installation provides ``triton.experimental.tle``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor
except (ImportError, AttributeError):
    tle = None
    TensorDescriptor = None
    _TLE_AVAILABLE = False
else:
    _TLE_AVAILABLE = True


def compressed_lengths(
    cu_seqlens: torch.Tensor,
    kernel_size: int,
    stride: int,
) -> torch.Tensor:
    """Return packed cumulative lengths after sliding-window compression."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.where(lengths >= kernel_size, (lengths - kernel_size) // stride + 1, 0)
    out = torch.zeros_like(cu_seqlens)
    out[1:] = torch.cumsum(counts, dim=0)
    return out


def _check_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("InfLLM-V2 Triton kernels require CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("InfLLM-V2 Triton kernels require contiguous tensors")


@triton.jit
def _compress_k_kernel(
    k_ptr,
    out_ptr,
    cu_in_ptr,
    cu_out_ptr,
    hkv: tl.constexpr,
    d: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    block_window: tl.constexpr,
    block_d: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    in_start = tl.load(cu_in_ptr + batch)
    in_end = tl.load(cu_in_ptr + batch + 1)
    out_start = tl.load(cu_out_ptr + batch)
    out_end = tl.load(cu_out_ptr + batch + 1)
    if out_start + chunk >= out_end:
        return
    offs_w = tl.arange(0, block_window)
    offs_d = tl.arange(0, block_d)
    token = in_start + chunk * stride + offs_w
    ptrs = k_ptr + token[:, None] * hkv * d + head * d + offs_d[None, :]
    values = tl.load(
        ptrs,
        mask=(offs_w[:, None] < kernel_size) & (token[:, None] < in_end) & (offs_d[None, :] < d),
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / kernel_size
    out_ptrs = out_ptr + (out_start + chunk) * hkv * d + head * d + offs_d
    tl.store(out_ptrs, mean, mask=offs_d < d)


def compress_k(
    k: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    kernel_size: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton mean-pooling compression for packed ``[Tk, Hkv, D]`` keys."""
    _check_cuda_contiguous(k, cu_seqlens_k)
    if cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_k must be int32")
    cu_out = compressed_lengths(cu_seqlens_k, kernel_size, stride)
    total_out = int(cu_out[-1].item())
    out = torch.empty((total_out, k.shape[1], k.shape[2]), device=k.device, dtype=k.dtype)
    if total_out == 0:
        return out, cu_out
    counts = cu_out[1:] - cu_out[:-1]
    max_chunks = int(counts.max().item())
    block_window = triton.next_power_of_2(kernel_size)
    block_d = triton.next_power_of_2(k.shape[2])
    _compress_k_kernel[(max_chunks, k.shape[1], cu_seqlens_k.numel() - 1)](
        k,
        out,
        cu_seqlens_k,
        cu_out,
        hkv=k.shape[1],
        d=k.shape[2],
        kernel_size=kernel_size,
        stride=stride,
        block_window=block_window,
        block_d=block_d,
        num_warps=4,
    )
    return out, cu_out


@triton.autotune(
    configs=[
        triton.Config({"block_q": 1, "block_n": 128}, num_warps=4, num_stages=2),
        triton.Config({"block_q": 4, "block_n": 64}, num_warps=4, num_stages=2),
        triton.Config({"block_q": 8, "block_n": 128}, num_warps=4, num_stages=2),
    ],
    key=["d", "group", "tune_key"],
    cache_results=True,
)
@triton.jit
def _stage1_kernel(
    q_ptr,
    k1_ptr,
    k2_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k1_ptr,
    cu_k2_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    max_k1,
    scale,
    k1_stride: tl.constexpr,
    k2_stride: tl.constexpr,
    causal: tl.constexpr,
    tune_key: tl.constexpr,
    block_q: tl.constexpr,
    block_g: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
):
    q_local_start = tl.program_id(0) * block_q
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local_start >= q_end:
        return
    k1_start = tl.load(cu_k1_ptr + batch)
    k1_end = tl.load(cu_k1_ptr + batch + 1)
    k2_start = tl.load(cu_k2_ptr + batch)
    k2_end = tl.load(cu_k2_ptr + batch + 1)
    k1_len = k1_end - k1_start
    k2_len = k2_end - k2_start
    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
    offs_q = tl.arange(0, block_q)
    q_local = q_local_start + offs_q
    q_index = q_start + q_local
    q_mask = q_index < q_end

    offs_g = tl.arange(0, block_g)
    offs_d = tl.arange(0, block_d)
    q_ptrs = (
        q_ptr
        + q_index[:, None, None] * hq * d
        + (hk * group + offs_g[None, :, None]) * d
        + offs_d[None, None, :]
    )
    q = tl.load(
        q_ptrs,
        mask=q_mask[:, None, None]
        & (offs_g[None, :, None] < group)
        & (offs_d[None, None, :] < d),
        other=0.0,
    )
    q = tl.reshape(q, [block_q * block_g, block_d])

    m = tl.full((block_q, block_g), float("-inf"), tl.float32)
    l = tl.zeros((block_q, block_g), tl.float32)
    # Align in COMPRESSED coordinates. Integer divisions here match positive
    # sequence lengths with C++ truncation toward zero, including Lq < stride.
    coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
    coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
    coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
    offs_n = tl.arange(0, block_n)
    for start_n in tl.range(0, k2_len, block_n):
        pos = start_n + offs_n
        k_ptrs = k2_ptr + (k2_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k2_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid = q_mask[:, None] & (pos[None, :] < k2_len)
        if causal:
            valid = valid & (pos[None, :] < coarse_right[:, None])
        logits = tl.where(
            valid[:, None, :] & (offs_g[None, :, None] < group),
            logits,
            float("-inf"),
        )
        tile_m = tl.max(logits, axis=2)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        p = tl.where(
            valid[:, None, :] & (offs_g[None, :, None] < group),
            tl.exp(logits - new_m[:, :, None]),
            0.0,
        )
        l = l * alpha + tl.sum(p, axis=2)
        m = new_m
    lse = m + tl.log(l)

    fine_q_len = tl.maximum(0, q_len - k1_stride + 1) // k1_stride
    fine_right = tl.maximum(0, (q_local + 1) // k1_stride - 1 + k1_len - fine_q_len)
    for start_n in tl.range(0, k1_len, block_n):
        pos = start_n + offs_n
        k_ptrs = k1_ptr + (k1_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k1_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid_n = q_mask[:, None] & (pos[None, :] < k1_len)
        if causal:
            valid_n = valid_n & (pos[None, :] < fine_right[:, None])
        valid_g = offs_g < group
        probs = tl.where(
            valid_g[None, :, None] & valid_n[:, None, :] & (l[:, :, None] > 0.0),
            tl.exp(logits - lse[:, :, None]),
            0.0,
        )
        score = tl.sum(probs, axis=1).to(q_ptr.dtype.element_ty).to(tl.float32)
        out_ptrs = out_ptr + hk * total_q * max_k1 + q_index[:, None] * max_k1 + pos[None, :]
        tl.store(out_ptrs, score, mask=q_mask[:, None] & (pos[None, :] < k1_len))


def stage1(
    q: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    k1_stride: int = 16,
    k2_stride: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    _allow_tle: bool = True,
) -> torch.Tensor:
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check_cuda_contiguous(q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    max_k1 = int(torch.max(cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).item())
    # On Hopper, a double-buffered TMA/WGMMA implementation begins to amortize
    # its TensorMap/shared-memory setup at 65K full-prefill. Shorter prefill and
    # decode retain the lower-overhead standard Triton tl.dot kernel.
    use_tle_hopper = _allow_tle and _TLE_AVAILABLE and (
        max_seqlen_q >= 65536
        and q.dtype == torch.bfloat16
        and k1.dtype == q.dtype
        and k2.dtype == q.dtype
        and group == 16
        and d in (64, 128)
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    )
    if use_tle_hopper:
        return stage1_tle_hopper(
            q,
            k1,
            k2,
            cu_seqlens_q,
            cu_seqlens_k1,
            cu_seqlens_k2,
            max_seqlen_q,
            k1_stride,
            k2_stride,
            softmax_scale,
            causal,
            cu_seqlens_k,
        )
    # Stage1 probabilities are already rounded to the Q dtype in the kernel.
    # Storing them as FP32 only doubles traffic and capacity without retaining
    # additional information.
    out = torch.zeros((hkv, total_q, max_k1), dtype=q.dtype, device=q.device)
    if max_k1 == 0:
        return out
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    # Short chunks need the untiled Q=1 grid to expose enough independent CTAs;
    # direct A/B shows Q=4 is about 20% slower at Q=128 even when the kernel
    # autotuner picks it.  Autotune only the long-prefill regime where K reuse
    # amortizes the smaller grid.
    tune_key = triton.next_power_of_2(max_seqlen_q)
    if max_seqlen_q < 1024:
        block_q = 1
        _stage1_kernel.fn[(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)](
            q,
            k1,
            k2,
            out,
            cu_seqlens_q,
            cu_seqlens_k1,
            cu_seqlens_k2,
            cu_seqlens_k,
            hq=hq,
            hkv=hkv,
            group=group,
            d=d,
            total_q=total_q,
            max_k1=max_k1,
            scale=scale,
            k1_stride=k1_stride,
            k2_stride=k2_stride,
            causal=causal,
            tune_key=tune_key,
            block_q=block_q,
            block_g=max(16, triton.next_power_of_2(group)),
            block_d=triton.next_power_of_2(d),
            block_n=128,
            num_warps=4,
            num_stages=2,
        )
        return out
    grid = lambda meta: (
        triton.cdiv(max_seqlen_q, meta["block_q"]),
        hkv,
        cu_seqlens_q.numel() - 1,
    )
    _stage1_kernel[grid](
        q,
        k1,
        k2,
        out,
        cu_seqlens_q,
        cu_seqlens_k1,
        cu_seqlens_k2,
        cu_seqlens_k,
        hq=hq,
        hkv=hkv,
        group=group,
        d=d,
        total_q=total_q,
        max_k1=max_k1,
        scale=scale,
        k1_stride=k1_stride,
        k2_stride=k2_stride,
        causal=causal,
        tune_key=tune_key,
        block_g=max(16, triton.next_power_of_2(group)),
        block_d=triton.next_power_of_2(d),
    )
    return out


@triton.jit
def _pool_scores_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k1_ptr,
    cu_k_ptr,
    total_q,
    max_k1,
    max_blocks,
    hkv: tl.constexpr,
    block_size: tl.constexpr,
    block_stride: tl.constexpr,
    pool_window: tl.constexpr,
    pad: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    block_b: tl.constexpr,
    valid_only: tl.constexpr,
):
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k1_len = tl.load(cu_k1_ptr + batch + 1) - tl.load(cu_k1_ptr + batch)
    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
    q_position = k_len - q_len + q_local
    offs_b = tl.arange(0, block_b)
    active = offs_b < max_blocks
    if valid_only:
        # Production TopK never reads blocks to the right of the current
        # causal position. Avoid pooling and writing that dead suffix. Debug
        # keeps the complete tensor for private debug/reference checks.
        valid_blocks = tl.minimum(
            max_blocks,
            tl.minimum(
                (k_len + block_size - 1) // block_size,
                (q_position + block_size) // block_size,
            ),
        )
        active = active & (offs_b < valid_blocks)
    q_block = q_position // block_size
    forced = (offs_b < init_blocks) | ((q_block >= offs_b) & (q_block <= offs_b + local_blocks))
    pooled = tl.full((block_b,), float("-inf"), tl.float32)
    for j in tl.static_range(0, pool_window):
        fine = offs_b * block_stride - pad + j
        ptrs = score_ptr + hk * total_q * max_k1 + (q_start + q_local) * max_k1 + fine
        value = tl.load(
            ptrs,
            mask=active & (~forced) & (fine >= 0) & (fine < k1_len),
            other=float("-inf"),
        )
        pooled = tl.maximum(pooled, value)
    pooled = tl.where(forced, float("inf"), pooled)
    out_ptrs = out_ptr + hk * total_q * max_blocks + (q_start + q_local) * max_blocks + offs_b
    tl.store(out_ptrs, pooled, mask=active)


def pool_scores(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    max_seqlen_q: int,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    init_blocks: int = 1,
    local_blocks: int = 32,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    *,
    _valid_only: bool = False,
) -> torch.Tensor:
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    max_seqlen_k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    _check_cuda_contiguous(score, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k)
    hkv, total_q, max_k1 = score.shape
    max_blocks = math.ceil(max_seqlen_k / block_size)
    out = torch.empty((hkv, total_q, max_blocks), dtype=score.dtype, device=score.device)
    if max_blocks == 0:
        return out
    pool_window = kernel_size // kernel_stride + block_size // kernel_stride - 1
    _pool_scores_kernel[(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)](
        score,
        out,
        cu_seqlens_q,
        cu_seqlens_k1,
        cu_seqlens_k,
        total_q,
        max_k1,
        max_blocks,
        hkv=hkv,
        block_size=block_size,
        block_stride=block_size // kernel_stride,
        pool_window=pool_window,
        pad=kernel_size // kernel_stride - 1,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_b=triton.next_power_of_2(max_blocks),
        valid_only=_valid_only,
        num_warps=8,
    )
    return out


@triton.jit
def _topk_float_to_ordered_key(x_bits):
    sign_bit = tl.full(x_bits.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(x_bits.shape, 0xFFFFFFFF, dtype=tl.uint32)
    return x_bits ^ tl.where((x_bits & sign_bit) != 0, full_mask, sign_bit)


@triton.jit
def _topk_index_to_key(index):
    max_u16 = tl.full(index.shape, 0xFFFF, dtype=tl.uint32)
    return max_u16 - index.to(tl.uint32)


@triton.jit
def _topk_key_to_index(index_key):
    max_u16 = tl.full(index_key.shape, 0xFFFF, dtype=tl.uint32)
    return (max_u16 - index_key.to(tl.uint32)).to(tl.int32)


@triton.jit
def _select_blocks_streaming_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    total_q,
    max_blocks,
    hkv: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block_b: tl.constexpr,
):
    """Packed-key exact TopK with a streaming tile merge."""
    tl.static_assert(block_b >= topk)
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return

    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
    q_position = k_len - q_len + q_local
    valid_blocks = tl.minimum(
        max_blocks,
        tl.minimum(
            (k_len + block_size - 1) // block_size,
            (q_position + block_size) // block_size,
        ),
    )
    q_index = q_start + q_local
    offs_t = tl.arange(0, topk)
    out_ptrs = out_ptr + hk * total_q * topk + q_index * topk + offs_t

    # Causal rows with N <= K select every visible block.  Their canonical
    # output is already the ascending identity, so scores need not be loaded.
    if valid_blocks <= topk:
        selected = tl.where(offs_t < valid_blocks, offs_t, -1)
        tl.store(out_ptrs, selected)
        return

    offs_b = tl.arange(0, block_b)
    num_tiles = tl.cdiv(valid_blocks, block_b)
    tile_start = (num_tiles - 1) * block_b
    block_idx = tile_start + offs_b
    valid = block_idx < valid_blocks
    score_row = score_ptr + hk * total_q * max_blocks + q_index * max_blocks
    score = tl.load(score_row + block_idx, mask=valid, other=float("-inf")).to(tl.float32)
    score = tl.where(score == score, score, float("-inf"))
    score_key = _topk_float_to_ordered_key(score.to(tl.uint32, bitcast=True))
    index_key = _topk_index_to_key(block_idx)
    packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
    packed = tl.where(valid, packed, tl.zeros_like(packed))
    selected_keys = tl.topk(packed, topk)

    # Merge one sorted TopK list per score tile.  Packing score in the high
    # bits and the reversed block id in the low bits gives deterministic,
    # stable ties: equal scores prefer the smaller logical block id.
    for _ in tl.range(0, num_tiles - 1):
        selected_keys = tl.bitonic_merge(selected_keys)
        tile_start -= block_b
        block_idx = tile_start + offs_b
        score = tl.load(score_row + block_idx).to(tl.float32)
        score = tl.where(score == score, score, float("-inf"))
        score_key = _topk_float_to_ordered_key(score.to(tl.uint32, bitcast=True))
        index_key = _topk_index_to_key(block_idx)
        packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
        selected_keys = tl.maximum(selected_keys, tl.topk(packed, topk))

    # Rotate the 16-bit index key above the score key and sort once more so
    # Stage2 receives the same ascending block-id order as the baseline.
    selected_keys = (selected_keys << 48) | (selected_keys >> 16)
    selected_keys = tl.sort(selected_keys, descending=True)
    selected = _topk_key_to_index((selected_keys >> 48).to(tl.uint32))
    selected = tl.where(selected < valid_blocks, selected, -1)
    tl.store(out_ptrs, selected)


def select_blocks(
    block_score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    block_size: int = 64,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    _allow_tle: bool = True,
) -> torch.Tensor:
    """Exact TopK with a shape-selected TLE radix or standard streaming kernel."""
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check_cuda_contiguous(block_score, cu_seqlens_q, cu_seqlens_k)
    hkv, total_q, max_blocks = block_score.shape
    if max_blocks > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk > 2048 or (topk & (topk - 1)) != 0:
        raise ValueError("TopK must be a power of two no greater than 2048")
    out = torch.full((hkv, total_q, topk), -1, dtype=torch.int32, device=block_score.device)
    if max_blocks == 0:
        return out
    max_q = int(torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).item())
    grid = (max_q, hkv, cu_seqlens_q.numel() - 1)
    # Wide full-prefill rows expose enough CTAs to amortize eight radix passes;
    # TLE CTA-shared histograms avoid the expensive 64-bit bitonic TopK network.
    # Short chunks/decode and <=256-block rows stay on streaming tl.topk.  On
    # H100 the streaming bitonic tile doubles at 257 blocks; the radix path is
    # already faster from that exact boundary in the measured BF16 shape.
    use_tle_radix = _allow_tle and _TLE_AVAILABLE and (
        max_q >= 1024
        and max_blocks > 256
        and topk <= 64
        and block_score.dtype == torch.bfloat16
        and torch.cuda.get_device_capability(block_score.device) == (9, 0)
    )
    if use_tle_radix:
        return select_blocks_tle_radix(
            block_score, cu_seqlens_q, topk, block_size, cu_seqlens_k
        )
    block_b = max(64, topk, triton.next_power_of_2(min(max_blocks, 1024)))
    num_warps = 2 if block_b <= 64 else (4 if block_b <= 128 else 8)
    _select_blocks_streaming_kernel[grid](
        block_score,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        max_blocks,
        hkv=hkv,
        block_size=block_size,
        topk=topk,
        block_b=block_b,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


@triton.jit
def _sparse_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_d: tl.constexpr,
):
    q_local = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    hk = head // group
    offs_d = tl.arange(0, block_d)
    offs_n = tl.arange(0, sparse_block)
    qv = tl.load(q_ptr + (q_start + q_local) * hq * d + head * d + offs_d, mask=offs_d < d, other=0.0)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros((block_d,), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    for rank in tl.static_range(0, topk):
        block = tl.load(selected_ptr + hk * total_q * topk + (q_start + q_local) * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        valid = (block >= 0) & (pos < k_len)
        if causal:
            valid = valid & (pos < causal_limit)
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=valid[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)
        logits = tl.sum(kval * qv[None, :], axis=1) * scale
        logits = tl.where(valid, logits, float("-inf"))
        tile_m = tl.max(logits, axis=0)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        probs = tl.where(valid, tl.exp(logits - new_m), 0.0)
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        vval = tl.load(v_ptrs, mask=valid[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * vval, axis=0)
        l = l * alpha + tl.sum(probs, axis=0)
        m = new_m
    result = tl.where(l > 0.0, acc / l, 0.0)
    tl.store(out_ptr + (q_start + q_local) * hq * d + head * d + offs_d, result, mask=offs_d < d)
    tl.store(lse_ptr + (q_start + q_local) * hq + head, tl.where(l > 0.0, m + tl.log(l), float("-inf")))


@triton.jit
def _sparse_attention_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_d: tl.constexpr,
):
    q_local = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local
    hk = head // group
    offs_d = tl.arange(0, block_d)
    offs_n = tl.arange(0, sparse_block)
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + head * d + offs_d
    qv = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    outv = tl.load(out_ptr + q_index * hq * d + head * d + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    dov = tl.load(do_ptr + q_index * hq * d + head * d + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + q_index * hq + head)
    delta = tl.sum(outv * dov, axis=0)
    causal_limit = k_len - q_len + q_local + 1
    dq = tl.zeros((block_d,), tl.float32)

    for rank in tl.static_range(0, topk):
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        valid = (block >= 0) & (pos < k_len)
        if causal:
            valid = valid & (pos < causal_limit)
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        mask = valid[:, None] & d_mask[None, :]
        kval = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)
        vval = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
        logits = tl.sum(kval * qv[None, :], axis=1) * scale
        probs = tl.where(valid & (lse != float("-inf")), tl.exp(logits - lse), 0.0)
        dp = tl.sum(vval * dov[None, :], axis=1)
        ds = probs * (dp - delta) * scale
        dq += tl.sum(ds[:, None] * kval, axis=0)
        tl.atomic_add(dk_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :], ds[:, None] * qv[None, :], mask=mask)
        tl.atomic_add(dv_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :], probs[:, None] * dov[None, :], mask=mask)

    tl.store(dq_ptr + q_index * hq * d + head * d + offs_d, dq, mask=d_mask)


@triton.jit
def _sparse_attention_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    store_lse: tl.constexpr,
):
    """Tensor-Core Stage-2: one query token and one complete GQA group."""
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local

    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    heads = hk * group + offs_m
    m_mask = offs_m < group
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
    qv = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full((block_m,), float("-inf"), tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, block_d), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    rank_count = topk
    if causal:
        # The selector emits every visible block followed by -1 padding while
        # fewer than TopK blocks exist. Avoid running Tensor-Core work on those
        # padded ranks (notably the first 4096 tokens of full prefill).
        rank_count = tl.minimum(topk, (causal_limit + sparse_block - 1) // sparse_block)

    # Use a regular loop rather than static_range: topk=64 should not be fully
    # unrolled into a huge, register-heavy program.
    for rank in tl.range(0, rank_count):
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        n_mask = (block >= 0) & (pos < k_len)
        if causal:
            n_mask = n_mask & (pos < causal_limit)

        kv_mask = n_mask[:, None] & d_mask[None, :]
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
        logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, float("-inf"))

        tile_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, tile_m)
        alpha = tl.where(new_m == float("-inf"), 0.0, tl.exp(m_i - new_m))
        probs = tl.where(
            m_mask[:, None] & n_mask[None, :],
            tl.exp(logits - new_m[:, None]),
            0.0,
        )
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        vval = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(vval.dtype), vval, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = new_m

    result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(out_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :], result,
             mask=m_mask[:, None] & d_mask[None, :])
    if store_lse:
        tl.store(lse_ptr + q_index * hq + heads, tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf")),
                 mask=m_mask)


@triton.jit
def _sparse_attention_gqa_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Tensor-Core backward, reducing a full GQA group before dK/dV atomics."""
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local

    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    heads = hk * group + offs_m
    m_mask = offs_m < group
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
    md_mask = m_mask[:, None] & d_mask[None, :]
    qv = tl.load(q_ptrs, mask=md_mask, other=0.0)
    outv = tl.load(out_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :],
                   mask=md_mask, other=0.0)
    dov = tl.load(do_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :],
                  mask=md_mask, other=0.0)
    lse = tl.load(lse_ptr + q_index * hq + heads, mask=m_mask, other=float("-inf"))
    delta = tl.sum(outv.to(tl.float32) * dov.to(tl.float32), axis=1)
    dq = tl.zeros((block_m, block_d), tl.float32)
    causal_limit = k_len - q_len + q_local + 1

    for rank in tl.range(0, topk):
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        n_mask = (block >= 0) & (pos < k_len)
        if causal:
            n_mask = n_mask & (pos < causal_limit)
        nd_mask = n_mask[:, None] & d_mask[None, :]

        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=nd_mask, other=0.0)
        vval = tl.load(v_ptrs, mask=nd_mask, other=0.0)
        logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
        valid = m_mask[:, None] & n_mask[None, :] & (lse[:, None] != float("-inf"))
        probs = tl.where(valid, tl.exp(logits - lse[:, None]), 0.0)
        dp = tl.dot(dov, tl.trans(vval), out_dtype=tl.float32)
        ds = probs * (dp - delta[:, None]) * scale

        dq += tl.dot(ds.to(kval.dtype), kval, out_dtype=tl.float32)
        dk_tile = tl.dot(tl.trans(ds.to(qv.dtype)), qv, out_dtype=tl.float32)
        dv_tile = tl.dot(tl.trans(probs.to(dov.dtype)), dov, out_dtype=tl.float32)
        grad_ptrs = (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        tl.atomic_add(dk_ptr + grad_ptrs, dk_tile, mask=nd_mask)
        tl.atomic_add(dv_ptr + grad_ptrs, dv_tile, mask=nd_mask)

    tl.store(dq_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :], dq, mask=md_mask)


@triton.jit
def _cast_fp32_kernel(src_ptr, dst_ptr, n_elements, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(src_ptr + offsets, mask=offsets < n_elements, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=offsets < n_elements)


def _sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    store_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_cuda_contiguous(q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    if selected_blocks.ndim != 3 or selected_blocks.shape[:2] != (hkv, total_q):
        raise ValueError(
            "selected_blocks must have [Hkv, total_q_tokens, topk] layout; "
            "all query heads in one GQA group share the per-token Hkv selection"
        )
    topk = selected_blocks.shape[-1]
    out = torch.empty_like(q)
    # Inference does not consume LSE. Reuse the already allocated output as an
    # inert pointer for that constexpr-disabled store instead of allocating a
    # separate zero-sized CUDA tensor.
    lse = torch.empty((total_q, hq), dtype=torch.float32, device=q.device) if store_lse else out
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    block_m = max(16, triton.next_power_of_2(hq // hkv))
    # Stage2 launch autotuning found no meaningful D128 change.  D64 long
    # prefill consistently preferred one extra software stage by 0.5%-1.1%, so
    # keep that small winner without paying runtime autotune overhead.
    stage2_num_warps = 2 if max_seqlen_q >= 1024 else (4 if max_seqlen_q >= 128 else 8)
    if max_seqlen_q >= 1024:
        stage2_num_stages = 3 if d == 64 else 2
    else:
        stage2_num_stages = 2 if max_seqlen_q >= 128 else 3
    _sparse_attention_gqa_kernel[(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)](
        q,
        k,
        v,
        selected_blocks,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        hq=hq,
        hkv=hkv,
        group=hq // hkv,
        d=d,
        total_q=total_q,
        scale=scale,
        topk=topk,
        sparse_block=block_size,
        causal=causal,
        block_m=block_m,
        block_n=block_size,
        block_d=triton.next_power_of_2(d),
        store_lse=store_lse,
        num_warps=stage2_num_warps,
        num_stages=stage2_num_stages,
    )
    return out, lse


class _SparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        block_size: int,
        softmax_scale: float | None,
        causal: bool,
    ) -> torch.Tensor:
        out, lse = _sparse_attention_forward(
            q,
            k,
            v,
            selected_blocks,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            block_size,
            softmax_scale,
            causal,
        )
        ctx.save_for_backward(q, k, v, selected_blocks, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, selected, out, lse, cu_q, cu_k = ctx.saved_tensors
        dout = dout.contiguous()
        if ctx.max_seqlen_q >= 64:
            # Keep short decode/chunks on the lower-overhead fused path below.
            # Long queries amortize reverse-index construction and benefit from
            # grouping dK/dV contributions before updating shared KV gradients.
            from .backward import sparse_backward

            dq, dk, dv = sparse_backward(
                q, k, v, out, dout, lse, selected, cu_q, cu_k,
                ctx.max_seqlen_q, ctx.block_size, ctx.softmax_scale, ctx.causal,
            )
            return dq, dk, dv, None, None, None, None, None, None, None
        total_q, hq, d = q.shape
        hkv = k.shape[1]
        scale = ctx.softmax_scale if ctx.softmax_scale is not None else d**-0.5
        dq = torch.empty_like(q)
        dk_acc = torch.zeros_like(k, dtype=torch.float32)
        dv_acc = torch.zeros_like(v, dtype=torch.float32)
        topk = selected.shape[-1]
        block_m = max(16, triton.next_power_of_2(hq // hkv))
        _sparse_attention_gqa_backward_kernel[(ctx.max_seqlen_q, hkv, cu_q.numel() - 1)](
            q,
            k,
            v,
            selected,
            out,
            dout,
            lse,
            dq,
            dk_acc,
            dv_acc,
            cu_q,
            cu_k,
            hq=hq,
            hkv=hkv,
            group=hq // hkv,
            d=d,
            total_q=total_q,
            scale=scale,
            topk=topk,
            sparse_block=ctx.block_size,
            causal=ctx.causal,
            block_m=block_m,
            block_n=ctx.block_size,
            block_d=triton.next_power_of_2(d),
            num_warps=8,
            num_stages=3,
        )
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        for source, target in ((dk_acc, dk), (dv_acc, dv)):
            n_elements = source.numel()
            _cast_fp32_kernel[(triton.cdiv(n_elements, 256),)](source, target, n_elements, block=256)
        return dq, dk, dv, None, None, None, None, None, None, None


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    *,
    _allow_tle: bool = True,
) -> torch.Tensor:
    """Differentiable sparse attention over selected blocks of continuous packed K/V."""
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, k, v)):
        # The Hopper TLE path pays for Tensor Maps, mbarriers, one warpgroup per
        # CTA and one launch per KV head. Measurements put its crossover above
        # short chunks, so keep decode/Q=128 on the lower-overhead tl.dot path.
        hkv = k.shape[1]
        use_tle_hopper = _allow_tle and _TLE_AVAILABLE and (
            max_seqlen_q >= 1024
            and q.dtype == torch.bfloat16
            and k.dtype == q.dtype
            and v.dtype == q.dtype
            and q.shape[1] // hkv == 16
            # D64 stays on standard Triton: on H100 its lower arithmetic load
            # does not amortize TensorMap, warpgroup and shared-memory setup.
            and q.shape[2] == 128
            and block_size == 64
            and torch.cuda.get_device_capability(q.device) == (9, 0)
        )
        if use_tle_hopper:
            return sparse_attention_tle_hopper(
                q,
                k,
                v,
                selected_blocks,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                block_size,
                softmax_scale,
                causal,
            )
        return _sparse_attention_forward(
            q,
            k,
            v,
            selected_blocks,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            block_size,
            softmax_scale,
            causal,
            store_lse=False,
        )[0]
    return _SparseAttentionFunction.apply(
        q,
        k,
        v,
        selected_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        block_size,
        softmax_scale,
        causal,
    )


# ---------------------------------------------------------------------------
# Optional Hopper TLE specializations
# ---------------------------------------------------------------------------


def _descriptor_allocator(size: int, align: int, stream: Optional[int]):
    del align, stream
    return torch.empty(size, dtype=torch.int8, device=triton.runtime.driver.active.get_active_torch_device())


@triton.autotune(
    configs=[
        triton.Config({"block_q": 4, "block_n": 128}, num_warps=4, num_stages=1),
        triton.Config({"block_q": 8, "block_n": 128}, num_warps=4, num_stages=1),
    ],
    key=["d", "tune_key"],
    cache_results=True,
)
@triton.jit
def _stage1_tle_hopper_kernel(
    q_ptr,
    desc_k1,
    desc_k2,
    out_ptr,
    cu_q_ptr,
    cu_k1_ptr,
    cu_k2_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    max_k1,
    scale,
    k1_stride: tl.constexpr,
    k2_stride: tl.constexpr,
    causal: tl.constexpr,
    tune_key: tl.constexpr,
    block_q: tl.constexpr,
    block_n: tl.constexpr,
):
    """Long-prefill Stage1: register Q, TMA K tiles and Hopper WGMMA."""
    q_local_start = tl.program_id(0) * block_q
    batch = tl.program_id(1)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local_start >= q_end:
        return
    k1_start = tl.load(cu_k1_ptr + batch)
    k1_end = tl.load(cu_k1_ptr + batch + 1)
    k2_start = tl.load(cu_k2_ptr + batch)
    k2_end = tl.load(cu_k2_ptr + batch + 1)
    k1_len = k1_end - k1_start
    k2_len = k2_end - k2_start
    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)

    offs_q = tl.arange(0, block_q)
    offs_g = tl.arange(0, group)
    offs_d = tl.arange(0, d)
    q_local = q_local_start + offs_q
    q_index = q_start + q_local
    q_mask = q_index < q_end
    qv = tl.load(
        q_ptr
        + q_index[:, None, None] * hq * d
        + offs_g[None, :, None] * d
        + offs_d[None, None, :],
        mask=q_mask[:, None, None],
        other=0.0,
    )
    q_flat = tl.reshape(qv, [block_q * group, d])

    # Both passes reuse a two-slot K pipeline. Distinct barrier sets restart
    # phase numbering between K2 and K1; the 32 KiB data allocation is shared.
    k_smem = tle.gpu.alloc([2, block_n, d], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k2_empty = tle.gpu.alloc_barriers(2, arrive_count=1, init=tle.gpu.READY)
    k2_full = tle.gpu.alloc_barriers(2, arrive_count=1, expect_bytes=block_n * d * 2)
    k1_empty = tle.gpu.alloc_barriers(2, arrive_count=1, init=tle.gpu.READY)
    k1_full = tle.gpu.alloc_barriers(2, arrive_count=1, expect_bytes=block_n * d * 2)

    m = tl.full((block_q, group), float("-inf"), tl.float32)
    l = tl.zeros((block_q, group), tl.float32)
    coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
    coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
    coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
    offs_n = tl.arange(0, block_n)
    tle.gpu.barrier_wait(k2_empty[0], phaseIdx=0)
    tle.gpu.copy(desc_k2, k_smem.slot(0), [block_n, d], [k2_start, 0], barrier=k2_full[0])
    for start_n in tl.range(0, k2_len, block_n):
        tile = start_n // block_n
        buf = tile % 2
        phase = tile // 2
        tle.gpu.barrier_wait(k2_full[buf], phaseIdx=phase)
        if start_n + block_n < k2_len:
            next_tile = tile + 1
            next_buf = next_tile % 2
            next_phase = next_tile // 2
            tle.gpu.barrier_wait(k2_empty[next_buf], phaseIdx=next_phase)
            tle.gpu.copy(
                desc_k2, k_smem.slot(next_buf), [block_n, d],
                [k2_start + start_n + block_n, 0], barrier=k2_full[next_buf],
            )
        logits = tle.gpu.wgmma(q_flat, k_smem.slot(buf), out_dtype=tl.float32, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits) * scale
        tle.gpu.barrier_arrive(k2_empty[buf], phaseIdx=phase)
        logits = tl.reshape(logits, [block_q, group, block_n])
        pos = start_n + offs_n
        valid = q_mask[:, None] & (pos[None, :] < k2_len)
        if causal:
            valid = valid & (pos[None, :] < coarse_right[:, None])
        logits = tl.where(valid[:, None, :], logits, float("-inf"))
        tile_m = tl.max(logits, axis=2)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        p = tl.where(valid[:, None, :], tl.exp(logits - new_m[:, :, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=2)
        m = new_m
    lse = m + tl.log(l)

    fine_q_len = tl.maximum(0, q_len - k1_stride + 1) // k1_stride
    fine_right = tl.maximum(0, (q_local + 1) // k1_stride - 1 + k1_len - fine_q_len)
    tle.gpu.barrier_wait(k1_empty[0], phaseIdx=0)
    tle.gpu.copy(desc_k1, k_smem.slot(0), [block_n, d], [k1_start, 0], barrier=k1_full[0])
    for start_n in tl.range(0, k1_len, block_n):
        tile = start_n // block_n
        buf = tile % 2
        phase = tile // 2
        tle.gpu.barrier_wait(k1_full[buf], phaseIdx=phase)
        if start_n + block_n < k1_len:
            next_tile = tile + 1
            next_buf = next_tile % 2
            next_phase = next_tile // 2
            tle.gpu.barrier_wait(k1_empty[next_buf], phaseIdx=next_phase)
            tle.gpu.copy(
                desc_k1, k_smem.slot(next_buf), [block_n, d],
                [k1_start + start_n + block_n, 0], barrier=k1_full[next_buf],
            )
        logits = tle.gpu.wgmma(q_flat, k_smem.slot(buf), out_dtype=tl.float32, trans_b=True)
        logits = tle.gpu.wgmma_wait(0, logits) * scale
        tle.gpu.barrier_arrive(k1_empty[buf], phaseIdx=phase)
        logits = tl.reshape(logits, [block_q, group, block_n])
        pos = start_n + offs_n
        valid_n = q_mask[:, None] & (pos[None, :] < k1_len)
        if causal:
            valid_n = valid_n & (pos[None, :] < fine_right[:, None])
        probs = tl.where(
            valid_n[:, None, :] & (l[:, :, None] > 0.0),
            tl.exp(logits - lse[:, :, None]),
            0.0,
        )
        score = tl.sum(probs, axis=1).to(tl.bfloat16)
        tl.store(
            out_ptr + q_index[:, None] * max_k1 + pos[None, :],
            score,
            mask=q_mask[:, None] & (pos[None, :] < k1_len),
        )


def stage1_tle_hopper(
    q: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    k1_stride: int = 16,
    k2_stride: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Production BF16 H100 Stage1 path for optimized GQA16 shapes."""
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    tensors = (q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_seqlens_k)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("TLE Stage1 requires contiguous CUDA tensors")
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    group = hq // hkv
    if q.dtype != torch.bfloat16 or k1.dtype != q.dtype or k2.dtype != q.dtype:
        raise ValueError("the TLE Stage1 path currently supports BF16 only")
    if group != 16 or d not in (64, 128) or max_seqlen_q < 1024:
        raise ValueError("the TLE Stage1 path requires group=16, D in {64,128} and Q>=1024")
    max_k1 = int(torch.max(cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).item())
    out = torch.zeros((hkv, total_q, max_k1), dtype=q.dtype, device=q.device)
    if max_k1 == 0:
        return out

    triton.set_allocator(_descriptor_allocator)
    total_k1, total_k2 = k1.shape[0], k2.shape[0]
    batch = cu_seqlens_q.numel() - 1
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    for hk in range(hkv):
        desc_k1 = TensorDescriptor(
            k1[:, hk, :], shape=[total_k1, d], strides=[hkv * d, 1], block_shape=[128, d]
        )
        desc_k2 = TensorDescriptor(
            k2[:, hk, :], shape=[total_k2, d], strides=[hkv * d, 1], block_shape=[128, d]
        )
        tune_key = triton.next_power_of_2(max_seqlen_q)
        grid = lambda meta: (triton.cdiv(max_seqlen_q, meta["block_q"]), batch)
        _stage1_tle_hopper_kernel[grid](
            q[:, hk * group :, :], desc_k1, desc_k2, out[hk],
            cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_seqlens_k,
            hq=hq, group=group, d=d, total_q=total_q, max_k1=max_k1, scale=scale,
            k1_stride=k1_stride, k2_stride=k2_stride, causal=causal,
            tune_key=tune_key,
        )
    return out


@triton.jit
def _tle_topk_float_key(x_bits):
    sign = tl.full(x_bits.shape, 0x80000000, tl.uint32)
    full = tl.full(x_bits.shape, 0xFFFFFFFF, tl.uint32)
    return x_bits ^ tl.where((x_bits & sign) != 0, full, sign)


@triton.jit
def _select_blocks_tle_radix_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    total_q,
    max_blocks,
    hkv: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block_k: tl.constexpr,
    block_t: tl.constexpr,
):
    """TLE shared-memory radix select adapted to InfLLM-V2 packed rows."""
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
    q_position = k_len - q_len + q_local
    valid_blocks = tl.minimum(
        max_blocks,
        tl.minimum((k_len + block_size - 1) // block_size, (q_position + block_size) // block_size),
    )
    q_index = q_start + q_local
    off_t = tl.arange(0, block_t)
    out_row = out_ptr + hk * total_q * topk + q_index * topk
    if valid_blocks <= topk:
        tl.store(out_row + off_t, tl.where(off_t < valid_blocks, off_t, -1), mask=off_t < topk)
        return

    lane = tl.arange(0, block_k)
    bins = tl.arange(0, 16)
    counts_smem = tle.gpu.alloc(
        [16], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    count_ptrs = tle.gpu.local_ptr(counts_smem, (bins,))
    score_row = score_ptr + hk * total_q * max_blocks + q_index * max_blocks
    n_tiles = tl.cdiv(valid_blocks, block_k)
    desired = tl.full((), 0, tl.uint32)
    desired_mask = tl.full((), 0, tl.uint32)
    k_to_find = tl.full((), topk, tl.int32)
    for digit_pos in tl.static_range(28, -1, -4):
        tl.store(count_ptrs, tl.zeros([16], tl.int32))
        tl.debug_barrier()
        for tile in tl.range(0, n_tiles):
            block_idx = tile * block_k + lane
            valid = block_idx < valid_blocks
            score = tl.load(score_row + block_idx, mask=valid, other=float("-inf")).to(tl.float32)
            score = tl.where(score == score, score, float("-inf"))
            key = _tle_topk_float_key(score.to(tl.uint32, bitcast=True))
            matches = (key & desired_mask) == desired
            digit = ((key >> digit_pos) & 15).to(tl.int32)
            tl.atomic_add(
                tle.gpu.local_ptr(counts_smem, (digit,)),
                tl.full([block_k], 1, tl.int32),
                mask=valid & matches,
                sem="relaxed",
                scope="cta",
            )
        tl.debug_barrier()
        counts = tl.load(count_ptrs)
        suffix = tl.cumsum(counts, axis=0, reverse=True)
        selected_mask = suffix >= k_to_find
        selected = tl.max(tl.where(selected_mask, bins, 0), axis=0).to(tl.int32)
        greater = tl.max(tl.where(bins == selected + 1, suffix, 0), axis=0)
        desired |= selected.to(tl.uint32) << digit_pos
        desired_mask |= tl.full((), 15, tl.uint32) << digit_pos
        k_to_find -= greater

    written = tl.full((), 0, tl.int32)
    equal_seen = tl.full((), 0, tl.int32)
    for tile in tl.range(0, n_tiles):
        block_idx = tile * block_k + lane
        valid = block_idx < valid_blocks
        score = tl.load(score_row + block_idx, mask=valid, other=float("-inf")).to(tl.float32)
        score = tl.where(score == score, score, float("-inf"))
        key = _tle_topk_float_key(score.to(tl.uint32, bitcast=True))
        take_gt = valid & (key > desired)
        equal = valid & (key == desired)
        equal_rank = tl.cumsum(equal.to(tl.int32), axis=0)
        take = take_gt | (equal & (equal_seen + equal_rank <= k_to_find))
        take_rank = tl.cumsum(take.to(tl.int32), axis=0)
        tl.store(out_row + written + take_rank - 1, block_idx, mask=take)
        written += tl.sum(take.to(tl.int32), axis=0)
        equal_seen += tl.sum(equal.to(tl.int32), axis=0)


def select_blocks_tle_radix(
    block_score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    block_size: int = 64,
    cu_seqlens_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Production TLE radix TopK for wide full-prefill selector rows."""
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    tensors = (block_score, cu_seqlens_q, cu_seqlens_k)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("TLE radix TopK requires contiguous CUDA tensors")
    hkv, total_q, max_blocks = block_score.shape
    if topk > 64 or max_blocks > 4096:
        raise ValueError("TLE radix TopK requires topk<=64 and max_blocks<=4096")
    out = torch.full((hkv, total_q, topk), -1, dtype=torch.int32, device=block_score.device)
    if max_blocks == 0:
        return out
    max_q = int(torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).item())
    block_k = max(32, triton.next_power_of_2(min(max_blocks, 1024)))
    _select_blocks_tle_radix_kernel[(max_q, hkv, cu_seqlens_q.numel() - 1)](
        block_score, out, cu_seqlens_q, cu_seqlens_k, total_q, max_blocks,
        hkv=hkv, block_size=block_size, topk=topk, block_k=block_k,
        block_t=triton.next_power_of_2(topk), num_warps=8, num_stages=1,
    )
    return out


@triton.jit
def _sparse_attention_tle_hopper_kernel(
    q_ptr,
    desc_k,
    desc_v,
    selected_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
):
    """One query token/GQA group using TMA and transpose-mapped WGMMA."""
    q_local = tl.program_id(0)
    batch = tl.program_id(1)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local

    offs_g = tl.arange(0, group)
    offs_d = tl.arange(0, d)
    qv = tl.load(q_ptr + q_index * hq * d + offs_g[:, None] * d + offs_d[None, :])

    q_smem = tle.gpu.alloc([group, d], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    kv_smem = tle.gpu.alloc([sparse_block, d], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([sparse_block, group], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    tl.store(tle.gpu.local_ptr(q_smem), qv)

    kv_empty = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, init=tle.gpu.READY)
    kv_full = tle.gpu.alloc_barriers(
        num_barriers=1,
        arrive_count=1,
        expect_bytes=sparse_block * d * 2,
    )

    m_i = tl.full((group,), float("-inf"), tl.float32)
    l_i = tl.zeros((group,), tl.float32)
    acc_t = tl.zeros((d, group), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    rank_count = topk
    if causal:
        rank_count = tl.minimum(topk, (causal_limit + sparse_block - 1) // sparse_block)

    offs_n = tl.arange(0, sparse_block)
    for rank in tl.range(0, rank_count):
        block = tl.load(selected_ptr + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        block_start = k_start + safe_block * sparse_block
        pos = safe_block * sparse_block + offs_n
        n_mask = (block >= 0) & (pos < k_len)
        if causal:
            n_mask = n_mask & (pos < causal_limit)

        k_phase = rank * 2
        tle.gpu.barrier_wait(kv_empty[0], phaseIdx=k_phase)
        tle.gpu.copy(
            desc_k,
            kv_smem,
            [sparse_block, d],
            [block_start, 0],
            barrier=kv_full[0],
        )
        tle.gpu.barrier_wait(kv_full[0], phaseIdx=k_phase)
        qk_t = tle.gpu.wgmma(kv_smem, q_smem, out_dtype=tl.float32, trans_b=True)
        qk_t = tle.gpu.wgmma_wait(0, qk_t)
        tle.gpu.barrier_arrive(kv_empty[0], phaseIdx=k_phase)

        v_phase = k_phase + 1
        tle.gpu.barrier_wait(kv_empty[0], phaseIdx=v_phase)
        tle.gpu.copy(
            desc_v,
            kv_smem,
            [sparse_block, d],
            [block_start, 0],
            barrier=kv_full[0],
        )

        logits = tl.trans(qk_t) * scale
        logits = tl.where(n_mask[None, :], logits, float("-inf"))
        tile_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, tile_m)
        alpha = tl.where(new_m == float("-inf"), 0.0, tl.exp(m_i - new_m))
        probs = tl.where(n_mask[None, :], tl.exp(logits - new_m[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = new_m
        tl.store(tle.gpu.local_ptr(p_smem), tl.trans(probs).to(tl.bfloat16))

        tle.gpu.barrier_wait(kv_full[0], phaseIdx=v_phase)
        acc_t *= alpha[None, :]
        acc_t = tle.gpu.wgmma(kv_smem, p_smem, acc_t, trans_a=True)
        acc_t = tle.gpu.wgmma_wait(0, acc_t)
        tle.gpu.barrier_arrive(kv_empty[0], phaseIdx=v_phase)

    result = tl.where(l_i[:, None] > 0.0, tl.trans(acc_t) / l_i[:, None], 0.0)
    tl.store(out_ptr + q_index * hq * d + offs_g[:, None] * d + offs_d[None, :], result)


def sparse_attention_tle_hopper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Production BF16 H100 Stage2 path for the optimized GQA16 D128 shape."""
    if not all(t.is_cuda for t in (q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k)):
        raise ValueError("TLE Hopper Stage2 requires CUDA tensors")
    if not all(t.is_contiguous() for t in (q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k)):
        raise ValueError("TLE Hopper Stage2 requires contiguous tensors")
    total_q, hq, d = q.shape
    total_k, hkv, _ = k.shape
    group = hq // hkv
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("the TLE Stage2 path currently supports BF16 only")
    if group != 16 or d != 128 or block_size != 64:
        raise ValueError("the TLE Stage2 path requires group=16, D=128 and block_size=64")
    if selected_blocks.shape[:2] != (hkv, total_q):
        raise ValueError("selected_blocks must have [Hkv, total_q, topk] layout")

    triton.set_allocator(_descriptor_allocator)
    out = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    topk = selected_blocks.shape[-1]
    batch = cu_seqlens_q.numel() - 1
    for hk in range(hkv):
        k_h = k[:, hk, :]
        v_h = v[:, hk, :]
        desc_k = TensorDescriptor(
            k_h,
            shape=[total_k, d],
            strides=[hkv * d, 1],
            block_shape=[block_size, d],
        )
        desc_v = TensorDescriptor(
            v_h,
            shape=[total_k, d],
            strides=[hkv * d, 1],
            block_shape=[block_size, d],
        )
        _sparse_attention_tle_hopper_kernel[(max_seqlen_q, batch)](
            q[:, hk * group :, :],
            desc_k,
            desc_v,
            selected_blocks[hk],
            out[:, hk * group :, :],
            cu_seqlens_q,
            cu_seqlens_k,
            hq=hq,
            group=group,
            d=d,
            total_q=total_q,
            scale=scale,
            topk=topk,
            sparse_block=block_size,
            causal=causal,
            num_warps=4,
            num_stages=1,
        )
    return out
