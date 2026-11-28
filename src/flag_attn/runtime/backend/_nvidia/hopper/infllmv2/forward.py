# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Hopper TMA/WGMMA and shared-memory radix kernels for InfLLM-V2."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor


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
        def grid(meta):
            return (triton.cdiv(max_seqlen_q, meta["block_q"]), batch)
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
