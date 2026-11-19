# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Streaming rounded Stage1 scores directly into block pooling.

This preserves the original coarse normalization and fine dot tiling. A single
rounded score is carried across fine tiles; no complete token-score tensor is
written. The internal launcher is used only for its supported compression geometry.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def stage1_pool_kernel(
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
    max_blocks,
    scale,
    k1_stride: tl.constexpr,
    k2_stride: tl.constexpr,
    causal: tl.constexpr,
    block_q: tl.constexpr,
    block_g: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    max_q: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    wide_offsets: tl.constexpr,
):
    tl.static_assert(k1_stride == 16 and k2_stride == 64)
    tl.static_assert(block_n % 4 == 0)
    index_ty: tl.constexpr = tl.int64 if wide_offsets else tl.int32
    for virtual_query in tl.range(tl.program_id(0), tl.cdiv(max_q, block_q), tl.num_programs(0)):
        q_local_start = virtual_query * block_q
        hk, batch = tl.program_id(1), tl.program_id(2)
        q_start, q_end = tl.load(cu_q_ptr + batch), tl.load(cu_q_ptr + batch + 1)
        if q_start + q_local_start < q_end:
            k1_start, k1_end = tl.load(cu_k1_ptr + batch), tl.load(cu_k1_ptr + batch + 1)
            k2_start, k2_end = tl.load(cu_k2_ptr + batch), tl.load(cu_k2_ptr + batch + 1)
            k1_len, k2_len = k1_end - k1_start, k2_end - k2_start
            k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
            q_len = q_end - q_start
            q_local = q_local_start + tl.arange(0, block_q)
            q_index = (q_start + q_local).to(index_ty)
            q_mask = q_index < q_end
            offs_g, offs_d = tl.arange(0, block_g), tl.arange(0, block_d)
            q = tl.load(
                q_ptr
                + q_index[:, None, None] * hq * d
                + (hk * group + offs_g[None, :, None]) * d
                + offs_d[None, None, :],
                q_mask[:, None, None] & (offs_g[None, :, None] < group) & (offs_d[None, None, :] < d),
                other=0.0,
            )
            q = tl.reshape(q, [block_q * block_g, block_d])
            m = tl.full((block_q, block_g), float("-inf"), tl.float32)
            l = tl.zeros((block_q, block_g), tl.float32)
            coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
            coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
            coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
            offs_n = tl.arange(0, block_n)
            q_block = (k_len - q_len + q_local) // 64
            pool_limit = max_blocks
            score_limit = max_blocks
            if causal:
                current = tl.max(tl.where(q_mask, q_block, -1), axis=0)
                pool_limit = tl.minimum(max_blocks, tl.maximum(current + 1, 0))
                score_limit = pool_limit
                if block_q == 1:
                    # The local suffix is forced to +inf. No later visible
                    # pool window can consume its scores, including the halo.
                    score_limit = tl.minimum(pool_limit, tl.maximum(current - local_blocks, 0))
                    score_limit = tl.where(score_limit <= init_blocks, 0, score_limit)
            score_tiles = tl.cdiv(score_limit, block_n // 4)

            coarse_limit = k2_len
            if causal:
                coarse_limit = tl.minimum(k2_len, tl.max(tl.where(q_mask, coarse_right, 0), axis=0))
            for start_n in tl.range(0, tl.where(score_tiles > 0, coarse_limit, 0), block_n):
                pos = start_n + offs_n
                kval = tl.load(
                    k2_ptr + (k2_start.to(index_ty) + pos[None, :]) * hkv * d + hk * d + offs_d[:, None],
                    (offs_d[:, None] < d) & (pos[None, :] < k2_len),
                    other=0.0,
                )
                logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
                valid = q_mask[:, None] & (pos[None, :] < k2_len)
                if causal:
                    valid = valid & (pos[None, :] < coarse_right[:, None])
                logits = tl.where(valid[:, None, :] & (offs_g[None, :, None] < group), logits, float("-inf"))
                tile_m = tl.max(logits, axis=2)
                new_m = tl.maximum(m, tile_m)
                alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
                p = tl.where(
                    valid[:, None, :] & (offs_g[None, :, None] < group), tl.exp(logits - new_m[:, :, None]), 0.0
                )
                l = l * alpha + tl.sum(p, axis=2)
                m = new_m
            lse = m + tl.log(l)
            fine_q_len = tl.maximum(0, q_len - k1_stride + 1) // k1_stride
            fine_right = tl.maximum(0, (q_local + 1) // k1_stride - 1 + k1_len - fine_q_len)
            offs_b = tl.arange(0, block_n // 4)
            previous = tl.full((block_q, block_n // 4), float("-inf"), tl.float32)
            for tile in tl.range(0, score_tiles):
                start_n = tile * block_n
                pos = start_n + offs_n
                kval = tl.load(
                    k1_ptr + (k1_start.to(index_ty) + pos[None, :]) * hkv * d + hk * d + offs_d[:, None],
                    (offs_d[:, None] < d) & (pos[None, :] < k1_len),
                    other=0.0,
                )
                logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
                valid = q_mask[:, None] & (pos[None, :] < k1_len)
                if causal:
                    valid = valid & (pos[None, :] < fine_right[:, None])
                allowed = valid[:, None, :] & (offs_g[None, :, None] < group) & (l[:, :, None] > 0.0)
                probabilities = tl.exp(logits - lse[:, :, None])
                # The reference discards nonfinite probabilities before GQA accumulation.
                probabilities = tl.where(allowed & (probabilities < float("inf")), probabilities, 0.0)
                # The original rounding boundary precedes every pooling comparison.
                score = tl.sum(probabilities, axis=1).to(q_ptr.dtype.element_ty).to(tl.float32)
                score = tl.where(pos[None, :] < k1_len, score, float("-inf"))
                # Use aligned 128-element reductions. The compact four-element
                # reduction faults in the installed Vector lowering.
                start = offs_b * 4
                in_window = (offs_n[None, :] >= start[:, None] - 1) & (offs_n[None, :] < start[:, None] + 4)
                pooled = tl.max(tl.where(in_window[None, :, :], score[:, None, :], float("-inf")), axis=2)
                pooled = tl.where(offs_b[None, :] == 0, tl.maximum(pooled, previous), pooled)
                last = tl.max(tl.where(offs_n[None, :] == block_n - 1, score, float("-inf")), axis=1)
                previous = tl.broadcast_to(last[:, None], [block_q, block_n // 4])
                block = tile * (block_n // 4) + offs_b
                forced = (block[None, :] < init_blocks) | (
                    (q_block[:, None] >= block[None, :]) & (q_block[:, None] <= block[None, :] + local_blocks)
                )
                pooled = tl.where(forced, float("inf"), pooled)
                pooled = tl.where(block[None, :] < tl.cdiv(k_len, 64), pooled, float("-inf"))
                offsets = hk.to(index_ty) * total_q * max_blocks + q_index[:, None] * max_blocks + block[None, :]
                tl.store(out_ptr + offsets, pooled, q_mask[:, None] & (block[None, :] < max_blocks))

            # The last computed tile may already cover some of the suffix.
            for tile in tl.range(score_tiles, tl.cdiv(pool_limit, block_n // 4)):
                block = tile * (block_n // 4) + offs_b
                offsets = hk.to(index_ty) * total_q * max_blocks + q_index[:, None] * max_blocks + block[None, :]
                tl.store(out_ptr + offsets, float("inf"), q_mask[:, None] & (block[None, :] < pool_limit))


def stage1_pool(q, k1, k2, cu_q, cu_k1, cu_k2, cu_k, max_q, max_k, scale, causal, init_blocks, local_blocks):
    """Internal valid-block scores for the 32/16,128/64 compression geometry."""
    from .forward import _query_grid

    hq, d = q.shape[1:]
    hkv = k1.shape[1]
    # Align each private score row to a complete Vector store tile.
    # Selection still clips to each sequence's real KV block count.
    max_blocks = triton.cdiv(triton.cdiv(max_k, 64), 32) * 32
    out = torch.empty((hkv, q.shape[0], max_blocks), device=q.device, dtype=q.dtype)
    grid = _query_grid(max_q, hkv, cu_q.numel() - 1)
    stage1_pool_kernel[grid](
        q,
        k1,
        k2,
        out,
        cu_q,
        cu_k1,
        cu_k2,
        cu_k,
        hq=hq,
        hkv=hkv,
        group=hq // hkv,
        d=d,
        total_q=q.shape[0],
        max_blocks=max_blocks,
        scale=d**-0.5 if scale is None else scale,
        k1_stride=16,
        k2_stride=64,
        causal=causal,
        block_q=1,
        block_g=max(16, triton.next_power_of_2(hq // hkv)),
        block_d=triton.next_power_of_2(d),
        block_n=128,
        max_q=max_q,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        wide_offsets=max(q.numel(), k1.numel(), k2.numel(), out.numel()) > 2**31 - 1,
        num_warps=4,
        num_stages=2,
    )
    return out
