# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Device-side page routing for sparse inference and exact causal tail masking."""

import torch
import torch_npu
import triton
import triton.language as tl
from .forward import _query_grid, sparse_attention_forward
from .kernels import _unsafe_for_paged


@triton.jit
def causal_tail_attention_kernel(
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
    block_m: tl.constexpr,
    max_q: tl.constexpr,
    wide_offsets: tl.constexpr = False,
):
    index_ty: tl.constexpr = tl.int64 if wide_offsets else tl.int32
    for q_local in tl.range(tl.program_id(0), max_q, tl.num_programs(0)):
        hk = tl.program_id(1)
        batch = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + batch)
        q_end = tl.load(cu_q_ptr + batch + 1)
        if q_start + q_local < q_end:
            k_start = tl.load(cu_k_ptr + batch)
            k_end = tl.load(cu_k_ptr + batch + 1)
            q_len = q_end - q_start
            k_len = k_end - k_start
            q_index = (q_start + q_local).to(index_ty)
            offs_m = tl.arange(0, block_m)
            offs_n = tl.arange(0, sparse_block)
            offs_d = tl.arange(0, d)
            heads = hk * group + offs_m
            m_mask = offs_m < group
            q_ptrs = q_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
            qv = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
            causal_limit = k_len - q_len + q_local + 1
            visible = tl.maximum(0, tl.minimum(k_len, causal_limit))
            # At most one selected page can intersect the causal boundary.
            tail_id, tail_len = visible // sparse_block, visible % sparse_block
            ranks = tl.arange(0, topk)
            picks = tl.load(selected_ptr + (hk.to(tl.int64) * total_q + q_index) * topk + ranks)
            found = tl.sum(((picks == tail_id) & (ranks < tl.cdiv(visible, sparse_block))).to(tl.int32), 0) > 0
            n_mask = found & (offs_n < tail_len)
            pos = tl.where(n_mask, tail_id * sparse_block + offs_n, 0)
            offset = ((k_start.to(tl.int64) + pos[:, None]) * hkv + hk) * d + offs_d[None, :]
            kval = tl.load(k_ptr + offset, n_mask[:, None], other=0.0)
            vval = tl.load(v_ptr + offset, n_mask[:, None], other=0.0)
            # Zero masked K/V before the dot so future nonfinite values cannot leak.
            logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
            valid = m_mask[:, None] & n_mask[None, :]
            logits = tl.where(valid, logits, float("-inf"))
            m_i = tl.max(logits, axis=1)
            probs = tl.where(valid, tl.exp(logits - m_i[:, None]), 0.0)
            l_i = tl.sum(probs, axis=1)
            acc = tl.dot(probs.to(vval.dtype), vval, out_dtype=tl.float32)
            result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
            tl.store(
                out_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :],
                result,
                mask=m_mask[:, None],
            )
            lse = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))
            tl.store(lse_ptr + q_index * hq + heads, lse, mask=m_mask)


@triton.jit
def _load_kv_page(K, V, CuK, task, n, d, HKV: tl.constexpr, D: tl.constexpr,
                  MAX_BLOCKS: tl.constexpr, BS: tl.constexpr):
    batch = task // (HKV * MAX_BLOCKS)
    hk = task // MAX_BLOCKS % HKV
    block = task % MAX_BLOCKS
    start, end = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    pos = block * BS + n
    valid = pos < end - start
    safe = tl.where(valid, pos, 0)
    addr = (start.to(tl.int64) + safe[:, None]) * HKV * D + hk * D + d[None, :]
    if end > start:
        kval = tl.load(K + addr, valid[:, None], other=0.0)
        vval = tl.load(V + addr, valid[:, None], other=0.0)
    else:
        kval = tl.full((BS, D), 0, K.dtype.element_ty)
        vval = tl.full((BS, D), 0, V.dtype.element_ty)
    return kval, vval


@triton.jit
def pack_kv_pages_kernel(
    Q,
    K,
    V,
    CuK,
    SafeQ,
    PK,
    PV,
    Flags,
    FirstUnsafe,
    NQ,
    HKV: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BATCH: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    n, d = tl.arange(0, BS), tl.arange(0, D)
    blocks = BATCH * HKV * MAX_BLOCKS
    qsize: tl.constexpr = BS * D
    lanes = tl.arange(0, qsize)
    bad_q = tl.full((qsize,), False, tl.int1)
    for tile in tl.range(pid, tl.cdiv(NQ, qsize), programs):
        pos = tile.to(tl.int64) * qsize + lanes
        value = tl.load(Q + pos, pos < NQ, 0.0)
        unsafe = _unsafe_for_paged(value)
        bad_q = bad_q | unsafe
        tl.store(SafeQ + pos, tl.where(unsafe, 0.0, value), pos < NQ)
    bad_kv = tl.full((BS, D), False, tl.int1)
    for task in tl.range(pid, blocks + 1, programs):
        if task < blocks:
            kval, vval = _load_kv_page(K, V, CuK, task, n, d, HKV, D, MAX_BLOCKS, BS)
        else:
            kval = tl.full((BS, D), 0, K.dtype.element_ty)
            vval = tl.full((BS, D), 0, V.dtype.element_ty)
        unsafe_k = _unsafe_for_paged(kval)
        unsafe_v = _unsafe_for_paged(vval)
        bad_kv = bad_kv | unsafe_k | unsafe_v
        kval = tl.where(unsafe_k, 0.0, kval)
        vval = tl.where(unsafe_v, 0.0, vval)
        dest = (task.to(tl.int64) * BS + n[:, None]) * D + d[None, :]
        tl.store(PK + dest, kval)
        tl.store(PV + dest, vval)
    kv_exception = tl.sum(tl.sum(bad_kv.to(tl.int32), 1), 0) != 0
    # Reduce across pages first. Resolve token positions only on programs
    # that observed exceptional KV values.
    if kv_exception:
        for task in tl.range(pid, blocks, programs):
            kval, vval = _load_kv_page(K, V, CuK, task, n, d, HKV, D, MAX_BLOCKS, BS)
            unsafe = _unsafe_for_paged(kval) | _unsafe_for_paged(vval)
            token_unsafe = tl.max(unsafe.to(tl.int32), 1) != 0
            first = tl.min(tl.where(token_unsafe, n, BS), 0)
            tl.store(FirstUnsafe + task, first)
    query_exception = tl.sum(bad_q.to(tl.int32), 0) != 0
    tl.store(Flags + pid, query_exception.to(tl.int32) | (kv_exception.to(tl.int32) << 1))


@triton.jit
def page_table_kernel(
    Selected,
    CuQ,
    CuK,
    Table,
    Mask,
    Meta,
    TOTAL_Q: tl.constexpr,
    ROWS: tl.constexpr,
    OFFSET,
    HKV: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BATCH: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    rank, n = tl.arange(0, TOPK), tl.arange(0, BS)
    sentinel = BATCH * HKV * MAX_BLOCKS
    for pair in tl.range(tl.program_id(0), ROWS * HKV, tl.num_programs(0)):
        hk = pair % HKV
        qi = OFFSET + pair // HKV
        if BATCH == 1:
            batch = 0
        else:
            lo = 0
            hi = BATCH
            for _ in tl.static_range(0, (BATCH + 1).bit_length()):
                mid = (lo + hi) // 2
                end = tl.load(CuQ + tl.minimum(mid + 1, BATCH))
                right = qi >= end
                active = lo < hi
                lo = tl.where(active & right, mid + 1, lo)
                hi = tl.where(active & ~right, mid, hi)
            batch = tl.minimum(lo, BATCH - 1)
        qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
        ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
        selected = tl.load(Selected + (hk.to(tl.int64) * TOTAL_Q + qi) * TOPK + rank)
        # Internal selections are unique, increasing and padding-last, so the
        # causal rank bound is also the number of visible blocks.
        legal = (selected >= 0) & (selected < MAX_BLOCKS) & (qi >= qs) & (qi < qe)
        limit = ke - ks
        if CAUSAL:
            limit = ke - ks - (qe - qs) + qi - qs + 1
            legal = legal & (rank < tl.minimum(TOPK, (limit + BS - 1) // BS))
        # A page is complete iff its exclusive end is within the visible prefix.
        # Clamp for noncausal or empty rows, before the page-index comparison.
        visible = tl.maximum(0, tl.minimum(limit, ke - ks))
        complete = visible // BS
        partial = legal & (selected == complete) & (visible % BS != 0)
        # A fully masked page must point to the zero sentinel too. A mask
        # alone does not prevent nonfinite V from entering CANN's PV dot.
        full = legal & (selected < complete)
        page = tl.where(full, (batch * HKV + hk) * MAX_BLOCKS + selected, sentinel)
        tl.store(Table + pair * TOPK + rank, page)
        tl.store(Mask + (pair.to(tl.int64) * TOPK + rank[:, None]) * BS + n[None, :], ~full[:, None])
        # Keep the 32-byte owner stride; only two words are consumed by merge.
        tl.store(Meta + pair * 4 + 2, tl.sum(partial.to(tl.int32), 0))
        tl.store(Meta + pair * 4 + 3, tl.sum(full.to(tl.int32), 0))


@triton.jit
def merge_page_tail_kernel(
    Full,
    FullLSE,
    PartialLSE,
    Meta,
    Out,
    ROWS: tl.constexpr,
    OFFSET,
    HKV: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
):
    g, d = tl.arange(0, GP), tl.arange(0, D)
    for pair in tl.range(tl.program_id(0), ROWS * HKV, tl.num_programs(0)):
        qi = OFFSET + pair // HKV
        hk = pair % HKV
        sg = tl.minimum(g, G - 1)
        count = tl.load(Meta + pair * 4 + 2)
        full_count = tl.load(Meta + pair * 4 + 3)
        lf = tl.load(FullLSE + pair * G + sg)
        lp = tl.load(PartialLSE + qi.to(tl.int64) * HKV * G + hk * G + sg)
        good_f = (full_count == 0) | ((lf == lf) & (lf != float("inf")) & (lf != float("-inf")))
        good_p = (count == 0) | ((lp == lp) & (lp != float("inf")) & (lp != float("-inf")))
        lf = tl.where(full_count > 0, lf, float("-inf"))
        lp = tl.where(count > 0, lp + tl.log(count.to(tl.float32)), float("-inf"))
        base = tl.maximum(lf, lp)
        wf = tl.where(full_count > 0, tl.exp(lf - base), 0.0)
        wp = tl.where(count > 0, tl.exp(lp - base), 0.0)
        dest = (qi.to(tl.int64) * HKV * G + hk * G + sg[:, None]) * D + d[None, :]
        op = tl.load(Out + dest).to(tl.float32)
        of = tl.load(Full + (pair.to(tl.int64) * G + sg[:, None]) * D + d[None, :]).to(tl.float32)
        of = tl.where(full_count > 0, of, 0.0)
        op = tl.where(count > 0, op, 0.0)
        denom = wf + wp
        result = tl.where(
            (denom[:, None] > 0) & good_f[:, None] & good_p[:, None],
            (of * wf[:, None] + op * wp[:, None]) / denom[:, None],
            0.0,
        )
        tl.store(Out + dest, result, g[:, None] < G)


def sparse_paged_inference(q, k, v, selected, cuq, cuk, max_q, max_k, softmax_scale, *, dense=False):
    total, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    topk = selected.shape[-1]
    # The caller bounds geometry and NPU capability.
    block_size, tile_q, causal = 64, 512, True
    if total == 0 or k.shape[0] == 0:
        return torch.zeros_like(q)
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    if not -1.0 <= scale <= 1.0:
        return sparse_attention_forward(
            q, k, v, selected, cuq, cuk, max_q, block_size, softmax_scale, store_lse=False, dense=dense,
            selection_is_compact=True,
        )[0]
    cores = triton.runtime.driver.active.utils.get_aivector_core_num()
    batch = cuq.numel() - 1
    maxblocks = triton.cdiv(max_k, block_size)
    pages = batch * hkv * maxblocks + 1
    tokens = topk * block_size
    capacity = min(tile_q, total) * hkv
    # BSH uses a flat GQA head dimension and a [page, token, D] KV cache.
    # Every pair has Q length one and KV head count one, so the views
    # preserve the existing token order without a transpose or a copy.
    pk = torch.empty((pages, block_size, d), device=q.device, dtype=q.dtype)
    pv = torch.empty_like(pk)
    safe_q = torch.empty_like(q)
    flags = torch.empty(min(cores, pages), dtype=torch.int32, device=q.device)
    first_unsafe = torch.empty(pages - 1, dtype=torch.int32, device=q.device)
    pack_kv_pages_kernel[(flags.numel(),)](
        q, k, v, cuk, safe_q, pk, pv, flags, first_unsafe, q.numel(),
        HKV=hkv, D=d, MAX_BLOCKS=maxblocks, BATCH=batch, BS=block_size, num_warps=4,
    )
    out = torch.empty_like(q)
    partial_lse = torch.empty((total, hq), device=q.device, dtype=torch.float32)
    grid = _query_grid(max_q, hkv, batch)
    causal_tail_attention_kernel[grid](
        q,
        k,
        v,
        selected,
        out,
        partial_lse,
        cuq,
        cuk,
        hq=hq,
        hkv=hkv,
        group=g,
        d=d,
        total_q=total,
        scale=scale,
        topk=topk,
        sparse_block=block_size,
        block_m=max(16, triton.next_power_of_2(g)),
        max_q=max_q,
        wide_offsets=q.numel() > 2**31 - 1,
        num_warps=4,
        num_stages=2,
    )
    table = torch.empty((capacity, topk), device=q.device, dtype=torch.int32)
    mask = torch.empty((capacity, tokens), device=q.device, dtype=torch.bool)
    meta = torch.empty((capacity, 4), device=q.device, dtype=torch.int64)
    for offset in range(0, total, tile_q):
        rows = min(tile_q, total - offset)
        pairs = rows * hkv
        page_table_kernel[(min(cores, pairs),)](
            selected,
            cuq,
            cuk,
            table,
            mask,
            meta,
            TOTAL_Q=total,
            ROWS=rows,
            OFFSET=offset,
            HKV=hkv,
            TOPK=topk,
            BS=block_size,
            MAX_BLOCKS=maxblocks,
            BATCH=batch,
            CAUSAL=causal,
            num_warps=4,
            num_stages=1,
            multibuffer=False,
        )
        # Dense and sparse producers both emit increasing unique block IDs.
        # A complete visible block therefore occupies a rank smaller than the
        # number of complete pages visible to the last query in this batch.
        # For one packed sequence, tensor extents give that bound without a
        # device read. Keep one zero-sentinel page for an all-masked batch.
        page_tokens = tokens
        if batch == 1:
            visible = k.shape[0] - total + offset + rows
            page_tokens = min(tokens, max(block_size, visible // block_size * block_size))
        result = torch_npu.npu_fused_infer_attention_score(
            safe_q[offset : offset + rows].view(pairs, 1, g * d),
            pk,
            pv,
            num_heads=g,
            num_key_value_heads=1,
            input_layout="BSH",
            block_table=table[:pairs],
            block_size=block_size,
            actual_seq_lengths_kv=[page_tokens] * pairs,
            atten_mask=mask[:pairs, None, None, :],
            scale=scale,
            sparse_mode=0,
            inner_precise=0,
            softmax_lse_flag=True,
        )
        norm = result[1].contiguous()
        merge_page_tail_kernel[(min(cores, pairs),)](
            result[0].contiguous(),
            norm,
            partial_lse,
            meta,
            out,
            ROWS=rows,
            OFFSET=offset,
            HKV=hkv,
            G=g,
            GP=triton.next_power_of_2(g),
            D=d,
            num_warps=4,
        )
    # CANN consumed sanitized inputs. Replace its result only when the
    # original inputs require native softmax semantics, on the same stream.
    return sparse_attention_forward(
        q, k, v, selected, cuq, cuk, max_q, block_size, scale,
        store_lse=False, dense=dense, out=out, exception_flags=flags,
        exception_pages=first_unsafe, exception_max_blocks=maxblocks, selection_is_compact=True,
    )[0]
