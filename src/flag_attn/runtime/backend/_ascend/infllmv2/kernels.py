# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Triton kernels for packed InfLLM-V2 on Ascend."""

import triton
import triton.language as tl


@triton.jit
def make_tasks_kernel(TaskCu, Tasks, OWNERS: tl.constexpr, NUM_TASKS, QUERY_SPLIT: tl.constexpr, TILE: tl.constexpr):
    # Own contiguous, aligned output tiles. Tiny per-owner strided stores can
    # share an Ascend store granule and overwrite a neighbouring owner's task.
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    task = offset // 2
    lo = tl.full((TILE,), 0, tl.int32)
    hi = tl.full((TILE,), OWNERS, tl.int32)
    for _ in tl.static_range(0, (OWNERS + 1).bit_length()):
        mid = (lo + hi) // 2
        end = tl.load(TaskCu + tl.minimum(mid + 1, OWNERS))
        right = task >= end
        active = lo < hi
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    owner = tl.minimum(lo, OWNERS - 1)
    start = tl.load(TaskCu + owner)
    value = tl.where(offset % 2 == 0, owner, (task - start) * QUERY_SPLIT)
    tl.store(Tasks + offset, value, offset < NUM_TASKS * 2)


@triton.jit
def reverse_index_kernel(
    Selected,
    CuQ,
    Counts,
    Queries,
    TOTAL_Q: tl.constexpr,
    TOPK: tl.constexpr,
    HKV: tl.constexpr,
    MAX_Q: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    owner = tl.program_id(0)
    block = owner % MAX_BLOCKS
    hk = owner // MAX_BLOCKS % HKV
    batch = owner // (MAX_BLOCKS * HKV)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    qi, rank = tl.arange(0, BQ), tl.arange(0, BK)
    selected = tl.load(
        Selected + (hk.to(tl.int64) * TOTAL_Q + qs + qi[:, None]) * TOPK + rank[None, :],
        (qi[:, None] < qe - qs) & (rank[None, :] < TOPK),
        other=-1,
    )
    occurrences = tl.sum((selected == block).to(tl.int32), axis=1)
    present = (occurrences > 0) & (qi < qe - qs)
    offset = tl.cumsum(present.to(tl.int32), axis=0) - 1
    # Keep one slot per query and preserve the multiplicity of direct selections.
    tl.store(Queries + owner.to(tl.int64) * MAX_Q + offset, (occurrences - 1) * BQ + qi, present)
    tl.store(Counts + owner, tl.sum(present.to(tl.int32), axis=0))


@triton.jit
def stage1_kernel(
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
    block_q: tl.constexpr,
    block_g: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    max_q: tl.constexpr,
):
    for virtual_query in tl.range(tl.program_id(0), tl.cdiv(max_q, block_q), tl.num_programs(0)):
        q_local_start = virtual_query * block_q
        hk = tl.program_id(1)
        batch = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + batch)
        q_end = tl.load(cu_q_ptr + batch + 1)
        if not q_start + q_local_start >= q_end:
            k1_start = tl.load(cu_k1_ptr + batch)
            k1_end = tl.load(cu_k1_ptr + batch + 1)
            k2_start = tl.load(cu_k2_ptr + batch)
            k2_end = tl.load(cu_k2_ptr + batch + 1)
            k1_len = k1_end - k1_start
            k2_len = k2_end - k2_start
            q_len = q_end - q_start
            offs_q = tl.arange(0, block_q)
            q_local = q_local_start + offs_q
            q_index = (q_start + q_local).to(tl.int64)
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
                mask=q_mask[:, None, None] & (offs_g[None, :, None] < group) & (offs_d[None, None, :] < d),
                other=0.0,
            )
            q = tl.reshape(q, [block_q * block_g, block_d])
            m = tl.full((block_q, block_g), float("-inf"), tl.float32)
            l = tl.zeros((block_q, block_g), tl.float32)
            coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
            coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
            coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
            offs_n = tl.arange(0, block_n)
            for start_n in tl.range(0, k2_len, block_n):
                pos = start_n + offs_n
                k_ptrs = k2_ptr + (k2_start.to(tl.int64) + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
                kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k2_len), other=0.0)
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
            for start_n in tl.range(0, k1_len, block_n):
                pos = start_n + offs_n
                k_ptrs = k1_ptr + (k1_start.to(tl.int64) + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
                kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k1_len), other=0.0)
                logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
                valid_n = q_mask[:, None] & (pos[None, :] < k1_len)
                if causal:
                    valid_n = valid_n & (pos[None, :] < fine_right[:, None])
                valid_g = offs_g < group
                allowed = valid_g[None, :, None] & valid_n[:, None, :] & (l[:, :, None] > 0.0)
                probs = tl.exp(logits - lse[:, :, None])
                probs = tl.where(allowed & (probs < float("inf")), probs, 0.0)
                score = tl.sum(probs, axis=1).to(q_ptr.dtype.element_ty).to(tl.float32)
                out_ptrs = out_ptr + hk.to(tl.int64) * total_q * max_k1 + q_index[:, None] * max_k1 + pos[None, :]
                tl.store(out_ptrs, score, mask=q_mask[:, None] & (pos[None, :] < k1_len))


@triton.jit
def pool_scores_kernel(
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
    max_q: tl.constexpr,
):
    for virtual_query in tl.range(tl.program_id(0), max_q, tl.num_programs(0)):
        q_local = virtual_query
        hk = tl.program_id(1)
        batch = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + batch)
        q_end = tl.load(cu_q_ptr + batch + 1)
        if not q_start + q_local >= q_end:
            k1_len = tl.load(cu_k1_ptr + batch + 1) - tl.load(cu_k1_ptr + batch)
            q_len = q_end - q_start
            k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
            q_position = k_len - q_len + q_local
            offs_b = tl.arange(0, block_b)
            active = offs_b < max_blocks
            if valid_only:
                valid_blocks = tl.minimum(
                    max_blocks,
                    tl.minimum((k_len + block_size - 1) // block_size, (q_position + block_size) // block_size),
                )
                active = active & (offs_b < valid_blocks)
            q_block = q_position // block_size
            forced = (offs_b < init_blocks) | (q_block >= offs_b) & (q_block <= offs_b + local_blocks)
            pooled = tl.full((block_b,), float("-inf"), tl.float32)
            for j in tl.static_range(0, pool_window):
                fine = offs_b * block_stride - pad + j
                load_mask = active & ~forced & (fine >= 0) & (fine < k1_len)
                safe_fine = tl.minimum(tl.maximum(fine, 0), tl.maximum(k1_len - 1, 0))
                ptrs = score_ptr + hk.to(tl.int64) * total_q * max_k1 + (q_start.to(tl.int64) + q_local) * max_k1 + safe_fine
                value = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)
                value = tl.where(load_mask, value, float("-inf"))
                pooled = tl.maximum(pooled, value)
            pooled = tl.where(forced, float("inf"), pooled)
            out_ptrs = out_ptr + hk.to(tl.int64) * total_q * max_blocks + (q_start.to(tl.int64) + q_local) * max_blocks + offs_b
            tl.store(out_ptrs, pooled, mask=active)


@triton.jit
def _unsafe_for_paged(x):
    # D <= 128 and |scale| <= 1 keep a dot of values below 2**60
    # within FP32. FP16 already bounds finite inputs much more tightly.
    limit: tl.constexpr = 0x5D80 if x.dtype == tl.bfloat16 else 0x7C00
    return (x.to(tl.int16, bitcast=True) & 0x7FFF) >= limit


@triton.jit
def _paged_pair_needs_retry(
    Q, Selected, PageFirst, Flags, CuK, q_index, q_start, q_end, q_local, batch, hk, total_q,
    HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
    TOPK: tl.constexpr, BS: tl.constexpr, MAX_BLOCKS: tl.constexpr, PROGRAMS: tl.constexpr,
):
    k_len = tl.load(CuK + batch + 1) - tl.load(CuK + batch)
    visible = tl.maximum(0, tl.minimum(k_len, k_len - (q_end - q_start) + q_local + 1))
    ranks = tl.arange(0, TOPK)
    block = tl.load(Selected + (hk.to(tl.int64) * total_q + q_index) * TOPK + ranks)
    legal = (ranks < tl.cdiv(visible, BS)) & (block >= 0) & (block < tl.minimum(MAX_BLOCKS, tl.cdiv(visible, BS)))
    safe = tl.where(legal, block, 0)
    page = (batch * HKV + hk) * MAX_BLOCKS + safe
    # A pack program writes page metadata only if its KV flag is set.
    resolved = legal & ((tl.load(Flags + page % PROGRAMS, legal, 0) & 2) != 0)
    first = tl.load(PageFirst + page, resolved, BS)
    kv_bad = tl.max((resolved & (first < tl.minimum(BS, visible - safe * BS))).to(tl.int32), 0) != 0
    # A causal page exposes a prefix, so its first unsafe token is sufficient.
    dims = tl.arange(0, triton.next_power_of_2(G * D))
    q = tl.load(Q + q_index.to(tl.int64) * HQ * D + hk * G * D + dims, dims < G * D, 0)
    q_bad = tl.max(_unsafe_for_paged(q).to(tl.int32), 0) != 0
    return kv_bad | (q_bad & (tl.max(legal.to(tl.int32), 0) != 0))


@triton.jit
def attention_kernel(
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
    ranks_per_tile: tl.constexpr,
    max_q: tl.constexpr,
    dense: tl.constexpr = False,
    selection_is_compact: tl.constexpr = False,
    wide_offsets: tl.constexpr = False,
    exception_flags=None,
    flag_count: tl.constexpr = 0,
    exception_pages=None,
    exception_max_blocks: tl.constexpr = 0,
    value_scale: tl.constexpr = 1,
    replay: tl.constexpr = False,
):
    index_ty: tl.constexpr = tl.int64 if wide_offsets else tl.int32
    q_limit = max_q
    if exception_flags is not None:
        i = tl.arange(0, triton.next_power_of_2(flag_count))
        bits = tl.load(exception_flags + i, i < flag_count, 0)
        q_limit = tl.where(tl.sum(bits, 0) != 0, max_q, 0)
    for q_local in tl.range(tl.program_id(0), q_limit, tl.num_programs(0)):
        hk = tl.program_id(1)
        batch = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + batch)
        q_end = tl.load(cu_q_ptr + batch + 1)
        q_index = (q_start + q_local).to(index_ty)
        active = q_index < q_end
        if replay:
            lanes = tl.arange(0, triton.next_power_of_2(group * d))
            prior_ptr = out_ptr + q_index * hq * d + hk * group * d + lanes
            prior = tl.load(prior_ptr, active & (lanes < group * d), 0.0)
            nonfinite = (prior.to(tl.int16, bitcast=True) & 0x7F80) == 0x7F80
            active = active & (tl.max(nonfinite.to(tl.int32), 0) != 0)
        if exception_pages is not None:
            if active:
                active = _paged_pair_needs_retry(
                    q_ptr, selected_ptr, exception_pages, exception_flags, cu_k_ptr,
                    q_index, q_start, q_end, q_local, batch, hk, total_q,
                    hq, hkv, group, d, topk, sparse_block, exception_max_blocks, flag_count,
                )
        if active:
            k_start = tl.load(cu_k_ptr + batch)
            k_end = tl.load(cu_k_ptr + batch + 1)
            q_len = q_end - q_start
            k_len = k_end - k_start
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
            if dense:
                nonfinite_v = tl.full((block_d,), False, tl.int1)
            if causal and (dense or selection_is_compact):
                # Direct selections can contain repeated blocks or padding before valid ranks.
                rank_count = tl.minimum(rank_count, (causal_limit + sparse_block - 1) // sparse_block)
            block_limit = tl.cdiv(k_len, sparse_block)
            for rank_start in tl.range(0, rank_count, ranks_per_tile):
                if ranks_per_tile == 1:
                    block = tl.load(selected_ptr + hk.to(index_ty) * total_q * topk + q_index * topk + rank_start)
                    legal = (block >= 0) & (block < block_limit)
                    pos = tl.where(legal, block, 0) * sparse_block + offs_n
                    n_mask = legal & (pos < k_len)
                else:
                    rank = rank_start + offs_n // sparse_block
                    block = tl.load(
                        selected_ptr + hk.to(index_ty) * total_q * topk + q_index * topk + tl.minimum(rank, topk - 1),
                        rank < rank_count,
                        other=-1,
                    )
                    legal = (block >= 0) & (block < block_limit)
                    pos = tl.where(legal, block, 0) * sparse_block + offs_n % sparse_block
                    n_mask = (rank < rank_count) & legal & (pos < k_len)
                if causal:
                    n_mask = n_mask & (pos < causal_limit)
                kv_mask = n_mask[:, None] & d_mask[None, :]
                k_ptrs = k_ptr + (k_start.to(index_ty) + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
                kval = tl.load(k_ptrs, mask=kv_mask, other=0.0)
                logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
                logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, float("-inf"))
                tile_m = tl.max(logits, axis=1)
                new_m = tl.maximum(m_i, tile_m)
                alpha = tl.where(new_m == float("-inf"), 0.0, tl.exp(m_i - new_m))
                # An all -inf tile has zero mass until a finite logit is seen.
                center = tl.where(new_m == float("-inf"), 0.0, new_m)
                probs = tl.where(m_mask[:, None] & n_mask[None, :], tl.exp(logits - center[:, None]), 0.0)
                v_ptrs = v_ptr + (k_start.to(index_ty) + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
                vval = tl.load(v_ptrs, mask=kv_mask, other=0.0)
                if dense:
                    bad_v = ~(tl.abs(vval.to(tl.float32)) < float("inf"))
                    nonfinite_v = nonfinite_v | (tl.sum(bad_v.to(tl.int32), 0) != 0)
                if value_scale > 1:
                    # Bound the unnormalized BF16 P@V sum within FP32 range.
                    vval = (vval.to(tl.float32) * (1.0 / value_scale)).to(vval.dtype)
                acc = acc * alpha[:, None] + tl.dot(probs.to(vval.dtype), vval, out_dtype=tl.float32)
                l_i = l_i * alpha + tl.sum(probs, axis=1)
                m_i = new_m
            if dense:
                invalid = tl.where(nonfinite_v[None, :], float("nan"), 0.0)
            else:
                # A zero denominator requires inspecting selected blocks only
                # when distinguishing an empty row from all -inf logits.
                has_tokens = False
                if tl.sum(((l_i == 0.0) & m_mask).to(tl.int32), 0) != 0:
                    r = tl.arange(0, triton.next_power_of_2(topk))
                    base = (hk.to(index_ty) * total_q + q_index) * topk
                    picked = tl.load(selected_ptr + base + r, r < tl.minimum(rank_count, topk), -1)
                    visible = tl.minimum(k_len, causal_limit) if causal else k_len
                    has_tokens = tl.sum(((picked >= 0) & (picked < tl.cdiv(visible, sparse_block))).to(tl.int32), 0) != 0
                nan_mass = (l_i.to(tl.int32, bitcast=True) & 2147483647) > 2139095040
                invalid = tl.where(nan_mass[:, None] | has_tokens, float("nan"), 0.0)
            normalized = acc / l_i[:, None]
            result = tl.where(l_i[:, None] > 0.0, normalized * value_scale, invalid).to(out_ptr.dtype.element_ty)
            if value_scale > 1:
                # A finite weighted mean cannot exceed the BF16 input range.
                # The bit below either infinity is the signed maximum finite value.
                mean_is_finite = (normalized.to(tl.int32, bitcast=True) & 0x7FFFFFFF) < 0x7F800000
                result_bits = result.to(tl.int16, bitcast=True)
                overflow = mean_is_finite & ((result_bits & 0x7FFF) == 0x7F80)
                result = (result_bits - overflow.to(tl.int16)).to(tl.bfloat16, bitcast=True)
            dst = out_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
            output_mask = m_mask[:, None] & d_mask[None, :]
            if replay:
                # Other channels already have a finite unscaled result.
                previous = tl.load(dst, output_mask, 0.0)
                previous_is_finite = (previous.to(tl.int16, bitcast=True) & 0x7FFF) < 0x7F80
                result = tl.where(previous_is_finite, previous, result)
            tl.store(dst, result, mask=output_mask)
            if store_lse:
                tl.store(
                    lse_ptr + q_index * hq + heads, tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf")), mask=m_mask
                )


@triton.jit
def select_blocks_persistent_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    total_q,
    max_blocks,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block_b: tl.constexpr,
    max_q: tl.constexpr,
):
    for virtual_query in tl.range(tl.program_id(0), max_q, tl.num_programs(0)):
        qi, hk, batch = (virtual_query + 0, tl.program_id(1), tl.program_id(2))
        qs, qe = (tl.load(cu_q_ptr + batch), tl.load(cu_q_ptr + batch + 1))
        if not qs + qi >= qe:
            klen = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
            qpos = klen - (qe - qs) + qi
            valid_blocks = tl.minimum(
                max_blocks, tl.minimum(tl.cdiv(klen, block_size), (qpos + block_size) // block_size)
            )
            rank = tl.arange(0, topk)
            base = out_ptr + (hk.to(tl.int64) * total_q + qs + qi) * topk
            if valid_blocks <= topk:
                tl.store(base + rank, tl.where(rank < valid_blocks, rank, -1))
            else:
                lane = tl.arange(0, block_b)
                score = tl.load(
                    score_ptr + (hk.to(tl.int64) * total_q + qs + qi) * max_blocks + lane, lane < valid_blocks, other=float("-inf")
                ).to(tl.float32)
                score = tl.where(score == score, score, float("-inf"))
                if score_ptr.dtype.element_ty == tl.bfloat16:
                    bits16 = score.to(tl.int32, bitcast=True)
                    magnitude = (bits16 & 2147483647) >> 16
                    ordered = tl.where(bits16 < 0, 32767 - magnitude, 32768 + magnitude).to(tl.uint32)
                    key = ((ordered << 12) | (4095 - lane).to(tl.uint32)).to(tl.int32)
                    key = tl.where(lane < valid_blocks, key, -1)
                else:
                    bits = score.to(tl.uint32, bitcast=True)
                    key = (
                        bits ^ tl.where(bits & 2147483648 != 0, 4294967295, 2147483648).to(tl.uint32) ^ 2147483648
                    ).to(tl.int32)
                    key = tl.where(lane < valid_blocks, key, -2147483648)
                selected = tl.full((topk,), 2147483647, tl.int32)
                for r in tl.range(0, topk):
                    largest = tl.max(key, axis=0)
                    if score_ptr.dtype.element_ty == tl.bfloat16:
                        chosen = 4095 - (largest & 4095)
                    else:
                        chosen = tl.min(tl.where(key == largest, lane, 2147483647), axis=0)
                    selected = tl.where(rank == r, chosen, selected)
                    if score_ptr.dtype.element_ty == tl.bfloat16:
                        key = tl.where(lane == chosen, -1, key)
                    else:
                        key = tl.where(lane == chosen, -2147483648, key)
                if topk <= 64:
                    position = tl.sum((selected[:, None] > selected[None, :]).to(tl.int32), axis=1)
                    tl.store(base + position, selected)
                else:
                    for r in tl.range(0, topk):
                        chosen = tl.min(selected, axis=0)
                        tl.store(base + r, chosen)
                        selected = tl.where(selected == chosen, 2147483647, selected)
