# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA MSA routing for Hopper TLE, PDL, FP8 and Blackwell persistent/NVFP4."""

from math import prod

import torch
import triton
import triton.language as tl
from flag_attn.utils import is_sm90_device, is_sm100_device
from flag_attn.minimax_sparse_attention import sparse_attn as generic
from . import decode as cuda_decode
from .prefill import _gqa_sparse_fwd_kernel
from .dispatch import force_generic, hopper_prefill
from .utils import _packed_bytes, _sm_count, _tensor_metadata, _cache_plan, _can_reuse_launch, _KernelLaunch

# SM100 workload thresholds.
_PERSISTENT_MIN_TOKENS = 512
_PACKED_FP8_MIN_TOKENS = 128
_LONG_QUERY_MIN = 2048
_NVFP4_STAGING_MIN_TOKENS = 2048
_FP8_NARROW_OUTPUT_MIN_QUERY = 4096
_DENSE_MAX_QUERY = 1024
_DENSE_MAX_PAGES = 128
_PERSISTENT_MAX_WORKSPACE = 8 * 1024**3
_DECODE_PAIR_MIN_PARALLELISM = 64


# Device helpers: arithmetic, packed storage, bucketing and reduction


@triton.jit
def _fp8_pv(probability, value):
    # MiniMax's default FP8 policy: cast FP32 P directly to E4M3 for PV.
    return tl.dot(probability.to(tl.float8e4nv), value.to(tl.float8e4nv), max_num_imprecise_acc=0)


@triton.jit
def _nvfp4_scale_offset(row, group):
    # Logical [rows, 8] -> cuBLAS 128x4 layout; rows flatten page/head/token.
    return ((row // 128) * 2 + group // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4


@triton.jit
def _load_nvfp4_page(
    KV,
    Scales,
    page,
    head,
    kvs0: tl.constexpr,
    kvs1: tl.constexpr,
    kvs2: tl.constexpr,
    HEADS_K: tl.constexpr,
    OFFSET: tl.constexpr,
):
    ns, packed_d, groups = tl.arange(0, 128), tl.arange(0, 64), tl.arange(0, 8)
    packed = tl.load(KV + page * kvs0 + head * kvs1 + ns[:, None] * kvs2 + OFFSET + packed_d[None, :])
    # Decode both E2M1 values in each byte with one conversion.
    lo, hi = tl.inline_asm_elementwise(
        """{
        .reg .b8 b0, b1, b2, b3;
        .reg .b32 pair;
        mov.b32 {b0, b1, b2, b3}, $2;
        cvt.rn.f16x2.e2m1x2 pair, b0;
        mov.b32 {$0, $1}, pair;
        }""",
        constraints="=h,=h,r",
        args=[packed.to(tl.int32)],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    values = tl.interleave(lo, hi).to(tl.float32)
    row = (page * HEADS_K + head) * 128 + ns[:, None]
    offsets = _nvfp4_scale_offset(row, groups[None, :])
    scales = tl.load(Scales + offsets).to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    values = tl.reshape(values, (128, 8, 16)) * scales[:, :, None]
    values = tl.reshape(values, (128, 128))
    # FP4 times a finite E4M3 block scale is exact in BF16. FP8-Q callers
    # normalize this decoded value before rounding it to E4M3.
    return values.to(tl.bfloat16)


@triton.jit
def _nvfp4_to_fp8(value):
    """Normalize finite E2M1*E4M3 into E4M3; consumers compensate by 8."""
    # |E2M1 * E4M3| <= 6 * 448 = 2688; /8 bounds the operand by 336.
    # A fixed power of two avoids saturation without data-dependent routing.
    return (value.to(tl.float32) * 0.125).to(tl.float8e4nv)


@triton.jit
def _stage_nvfp4_kv_fp8(
    KV,
    KS,
    VS,
    K,
    V,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    H: tl.constexpr,
    HO: tl.constexpr,
    ORIGINAL_H: tl.constexpr,
):
    page, head = tl.program_id(0), tl.program_id(1)
    r, d = tl.arange(0, 128), tl.arange(0, 128)
    off = ((page * H + head) * 128 + r[:, None]) * 128 + d[None, :]
    k = _load_nvfp4_page(KV, KS, page, head + HO, S0, S1, S2, ORIGINAL_H, 0)
    v = _load_nvfp4_page(KV, VS, page, head + HO, S0, S1, S2, ORIGINAL_H, 64)
    tl.store(K + off, _nvfp4_to_fp8(k))
    tl.store(V + off, _nvfp4_to_fp8(v))


@triton.jit
def _init_persistent_workspace(Counts, LSE, COUNTS: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """Reset CSR counts and inactive partial rows on every invocation."""
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Counts + offsets, 0, offsets < COUNTS)
    tl.store(LSE + offsets, float("-inf"), offsets < ROWS)


@triton.jit
def _bucket_queries(
    TopK,
    CuQ,
    Seq,
    Prefix,
    Counts,
    RowPtr,
    QueryIds,
    ts0: tl.constexpr,
    ts1: tl.constexpr,
    ts2: tl.constexpr,
    BATCH: tl.constexpr,
    PAGES: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile, hb = tl.program_id(0), tl.program_id(1)
    head, batch = hb // BATCH, hb % BATCH
    start = tl.load(CuQ + batch)
    qlen = tl.load(CuQ + batch + 1) - start
    seq, prefix = tl.load(Seq + batch), tl.load(Prefix + batch)
    pair = tile * BLOCK + tl.arange(0, BLOCK)
    token, slot = pair // TOPK, pair % TOPK
    block = tl.load(TopK + head * ts0 + (start + token) * ts1 + slot * ts2, token < qlen, -1)
    valid = (token < qlen) & (block >= 0) & (block < PAGES)
    valid &= block * 128 < tl.minimum(seq, prefix + token + 1)
    position = tl.atomic_add(Counts + hb * PAGES + block, 1, valid, sem="relaxed")
    if SCATTER:
        base = tl.load(RowPtr + hb * (PAGES + 1) + block, valid, 0)
        tl.store(QueryIds + hb * CAPACITY + base + position, (start + token) * TOPK + slot, valid)


@triton.jit
def _dense_page_buckets(
    TopK,
    CuQ,
    Seq,
    Prefix,
    Counts,
    Ids,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BATCH: tl.constexpr,
    PAGES: tl.constexpr,
    TOPK: tl.constexpr,
    MAX_Q: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
):
    page, hb = tl.program_id(0), tl.program_id(1)
    head, batch = hb // BATCH, hb % BATCH
    start = tl.load(CuQ + batch)
    count = tl.load(CuQ + batch + 1) - start
    seq, prefix = tl.load(Seq + batch), tl.load(Prefix + batch)
    pairs = tl.arange(0, Q * K)
    rows, slots = pairs // K, pairs % K
    selected = tl.load(TopK + head * S0 + (start + rows) * S1 + slots * S2, (rows < count) & (slots < TOPK), -1)
    hit = (selected == page) & (rows < count) & (slots < TOPK)
    hit &= page * 128 < tl.minimum(seq, prefix + rows + 1)
    offset = tl.cumsum(hit.to(tl.int32)) - 1
    tl.store(Ids + hb * PAGES * MAX_Q + page * MAX_Q + offset, (start + rows) * TOPK + slots, hit)
    tl.store(Counts + hb * PAGES + page, tl.sum(hit.to(tl.int32), 0))


@triton.jit
def _bucket_segment_sum(ka, va, kb, vb):
    return kb, tl.where(ka == kb, va + vb, vb)


@triton.jit
def _bucket_segment_max(ka, va, kb, vb):
    return kb, tl.where(ka == kb, tl.maximum(va, vb), vb)


@triton.jit
def _aggregate_page_buckets(
    TopK,
    CuQ,
    Seq,
    Prefix,
    Counts,
    RowPtr,
    QueryIds,
    ts0: tl.constexpr,
    ts1: tl.constexpr,
    ts2: tl.constexpr,
    BATCH: tl.constexpr,
    PAGES: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCATTER: tl.constexpr,
    BLOCK: tl.constexpr,
    AGGREGATE_SCATTER: tl.constexpr = True,
):
    tile, hb = tl.program_id(0), tl.program_id(1)
    head, batch = hb // BATCH, hb % BATCH
    start = tl.load(CuQ + batch)
    qlen = tl.load(CuQ + batch + 1) - start
    seq, prefix = tl.load(Seq + batch), tl.load(Prefix + batch)
    lane = tl.arange(0, BLOCK)
    pair = tile * BLOCK + lane
    token, slot = pair // TOPK, pair % TOPK
    block = tl.load(TopK + head * ts0 + (start + token) * ts1 + slot * ts2, token < qlen, -1)
    valid = (token < qlen) & (block >= 0) & (block < PAGES)
    valid &= block * 128 < tl.minimum(seq, prefix + token + 1)
    if not SCATTER:
        histogram = tl.histogram(block, triton.next_power_of_2(PAGES), mask=valid)
        bins = tl.arange(0, triton.next_power_of_2(PAGES))
        tl.atomic_add(Counts + hb * PAGES + bins, histogram, (bins < PAGES) & (histogram > 0), sem="relaxed")
    else:
        if AGGREGATE_SCATTER:
            key = tl.sort(tl.where(valid, block, PAGES) * BLOCK + lane, descending=False)
            block = key // BLOCK
            pair = tile * BLOCK + key % BLOCK
            token, slot = pair // TOPK, pair % TOPK
            valid = block < PAGES
            _, rank = tl.associative_scan((block, tl.full((BLOCK,), 1, tl.int32)), 0, _bucket_segment_sum)
            next_block = tl.gather(block, tl.minimum(lane + 1, BLOCK - 1), 0)
            last = (lane == BLOCK - 1) | (block != next_block)
            offset = tl.atomic_add(Counts + hb * PAGES + block, rank, valid & last, sem="relaxed")
            offset = tl.where(last & valid, offset, 0)
            _, offset = tl.associative_scan((block, offset), 0, _bucket_segment_max, reverse=True)
            position = offset + rank - 1
        else:
            position = tl.atomic_add(Counts + hb * PAGES + block, 1, valid, sem="relaxed")
        base = tl.load(RowPtr + hb * (PAGES + 1) + block, valid, 0)
        tl.store(QueryIds + hb * CAPACITY + base + position, (start + token) * TOPK + slot, valid)


@triton.jit
def _prefix_buckets(Counts, RowPtr, TilePtr, PAGES: tl.constexpr, Q_PER_TILE: tl.constexpr, BLOCK: tl.constexpr):
    hb = tl.program_id(0)
    pages = tl.arange(0, BLOCK)
    counts = tl.load(Counts + hb * PAGES + pages, pages < PAGES, 0)
    ends = tl.cumsum(counts)
    tiles = tl.cdiv(counts, Q_PER_TILE)
    tile_ends = tl.cumsum(tiles)
    tl.store(RowPtr + hb * (PAGES + 1) + pages, ends - counts, pages < PAGES)
    tl.store(TilePtr + hb * (PAGES + 1) + pages, tile_ends - tiles, pages < PAGES)
    tl.store(RowPtr + hb * (PAGES + 1) + PAGES, tl.sum(counts, 0))
    tl.store(TilePtr + hb * (PAGES + 1) + PAGES, tl.sum(tiles, 0))
    tl.store(Counts + hb * PAGES + pages, 0, pages < PAGES)


@triton.jit
def _merge_fp8_splits(
    Partial,
    LSE,
    Out,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    splits, ds = tl.arange(0, SPLITS), tl.arange(0, BLOCK_D)
    indices = (row * HEADS + head) * SPLITS + splits
    lse = tl.load(LSE + indices)
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp2(lse - maximum)
    values = tl.load(
        Partial + indices[:, None] * D + ds[None, :], (lse[:, None] != float("-inf")) & (ds[None, :] < D), 0
    )
    result = tl.sum(values.to(tl.float32) * weights[:, None], 0)
    result /= tl.maximum(tl.sum(weights, 0), 1.0e-20)
    tl.store(Out + row * OS0 + head * OS1 + ds, result, ds < D)


@triton.jit
def _combine_splits(
    Partial,
    LSE,
    Out,
    HEADS_Q: tl.constexpr,
    TOPK: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token, head_tile = tl.program_id(0), tl.program_id(1)
    slots, ds = tl.arange(0, BLOCK_K), tl.arange(0, 128)
    heads = head_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    rows = (token * HEADS_Q + heads[None, :]) * TOPK + slots[:, None]
    mask = (slots[:, None] < TOPK) & (heads[None, :] < HEADS_Q)
    lse = tl.load(LSE + rows, mask, float("-inf"))
    valid = mask & (lse != float("-inf"))
    max_lse = tl.max(lse, 0)
    max_lse = tl.where(max_lse == float("-inf"), 0.0, max_lse)
    weight = tl.exp2(lse - max_lse[None, :])
    denominator = tl.sum(weight, 0)
    value = tl.load(Partial + rows[:, :, None] * 128 + ds[None, None, :], valid[:, :, None], 0.0)
    result = tl.sum(value.to(tl.float32) * weight[:, :, None], 0) / tl.maximum(denominator[:, None], 1.0e-20)
    tl.store(Out + token * os0 + heads[:, None] * os1 + ds[None, :], result, heads[:, None] < HEADS_Q)


@triton.jit
def _merge_tiled(
    P,
    L,
    O,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
):
    rows = tl.program_id(0) * M + tl.arange(0, M)
    ds = tl.program_id(1) * D + tl.arange(0, D)
    slots = tl.arange(0, K)
    ids = rows[:, None] * TOPK + slots[None, :]
    mask = (rows[:, None] < ROWS) & (slots[None, :] < TOPK)
    lse = tl.load(L + ids, mask, float("-inf"))
    maximum = tl.max(lse, 1)
    maximum = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp2(lse - maximum[:, None])
    denom = tl.maximum(tl.sum(weights, 1), 1.0e-20)
    values = tl.load(
        P + ids[:, :, None] * 128 + ds[None, None, :],
        (mask & (lse != float("-inf")))[:, :, None],
        0.0,
        # Partial rows are consumed once; avoid retaining them in L1.
        eviction_policy="evict_first",
    )
    result = tl.sum(values.to(tl.float32) * weights[:, :, None], 1) / denom[:, None]
    tl.store(
        O + (rows // HEADS)[:, None] * OS0 + (rows % HEADS)[:, None] * OS1 + ds[None, :], result, rows[:, None] < ROWS
    )


# Attention kernels: six implementation families


@triton.jit
def _gqa_sparse_q_fp8_kv_fp8_kernel(
    Q,
    KV,
    Selected,
    Pages,
    CuQ,
    Seq,
    Prefix,
    KS,
    VS,
    Out,
    LSE,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    TS2: tl.constexpr,
    PS0: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    SKH: tl.constexpr,
    SKT: tl.constexpr,
    SVH: tl.constexpr,
    SVT: tl.constexpr,
    HEADS: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    DECODE_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    task, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    token, split = task // SPLITS, task % SPLITS
    seq = tl.load(Seq + batch)
    if DECODE_Q:
        start, qlen, prefix = batch * DECODE_Q, DECODE_Q, seq - DECODE_Q
    else:
        start = tl.load(CuQ + batch)
        qlen = tl.load(CuQ + batch + 1) - start
        prefix = tl.load(Prefix + batch)
    if token >= qlen:
        return
    row, position = start + token, prefix + token
    hs, ds, ns = tl.arange(0, BLOCK_H), tl.arange(0, BLOCK_D), tl.arange(0, 128)
    q = tl.load(
        Q + row * QS0 + (head * GQA + hs[:, None]) * QS1 + ds[None, :], (hs[:, None] < GQA) & (ds[None, :] < D), 0.0
    )
    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    kg, vg = 1.0, 1.0
    if SCALE_MODE == 1:
        kg, vg = tl.load(KS), tl.load(VS)
    for local in range(tl.cdiv(TOPK, SPLITS)):
        slot = split * tl.cdiv(TOPK, SPLITS) + local
        block = tl.load(Selected + head * TS0 + row * TS1 + slot * TS2, slot < TOPK, -1)
        if (block >= 0) & (block * 128 < tl.minimum(seq, position + 1)):
            page = tl.load(Pages + batch * PS0 + block).to(tl.int64)
            base = KV + page * KS0 + head * KS1 + ns[:, None] * KS2
            k = tl.load(base + ds[None, :], ds[None, :] < D, 0.0)
            logits = tl.dot(q, tl.trans(k), max_num_imprecise_acc=0)
            if SCALE_MODE == 2:
                kg_token = tl.load(KS + head * SKH + (page * 128 + ns) * SKT)
                logits *= kg_token[None, :]
            logits *= SCALE * 1.4426950408889634 * kg
            visible = (block * 128 + ns < seq) & (block * 128 + ns <= position)
            logits = tl.where(visible[None, :] & (hs[:, None] < GQA), logits, float("-inf"))
            next_m = tl.maximum(m, tl.max(logits, 1))
            safe_m = tl.where(next_m == float("-inf"), 0.0, next_m)
            alpha = tl.exp2(m - safe_m)
            p = tl.exp2(logits - safe_m[:, None])
            denominator = denominator * alpha + tl.sum(p, 1)
            acc *= alpha[:, None]
            v = tl.load(base + D + ds[None, :], ds[None, :] < D, 0.0)
            if SCALE_MODE == 2:
                vs = tl.load(VS + head * SVH + (page * 128 + ns) * SVT)
                weighted = p * vs[None, :]
            else:
                weighted = p
            acc += _fp8_pv(weighted, v)
            m = next_m
    result = acc * (vg / tl.where(denominator > 0, denominator, 1.0))[:, None]
    if SPLITS == 1:
        target = Out + row * OS0 + (head * GQA + hs[:, None]) * OS1 + ds[None, :]
    else:
        out_row = (row * HEADS + head * GQA + hs) * SPLITS + split
        target = Out + out_row[:, None] * D + ds[None, :]
        tl.store(LSE + out_row, m + tl.log2(denominator), hs < GQA)
    tl.store(target, result, (hs[:, None] < GQA) & (ds[None, :] < D))


@triton.jit
def _gqa_sparse_decode_q_fp8_kv_fp8_kernel(
    Q,
    KV,
    Selected,
    Pages,
    CuQ,
    Seq,
    Prefix,
    KS,
    VS,
    Out,
    LSE,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    TS0: tl.constexpr,
    TS1: tl.constexpr,
    TS2: tl.constexpr,
    PS0: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    SKH: tl.constexpr,
    SKT: tl.constexpr,
    SVH: tl.constexpr,
    SVT: tl.constexpr,
    HEADS: tl.constexpr,
    GQA: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    DECODE_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_TILE: tl.constexpr = 1,
):
    PAGES_PER_TILE: tl.constexpr = PAGE_TILE
    task, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    token, split = task // SPLITS, task % SPLITS
    tl.static_assert(DECODE_Q > 0)
    seq = tl.load(Seq + batch)
    start, prefix = batch * DECODE_Q, seq - DECODE_Q
    if token >= DECODE_Q:
        return
    row, position = start + token, prefix + token
    hs, ds = tl.arange(0, BLOCK_H), tl.arange(0, BLOCK_D)
    ns = tl.arange(0, 128 * PAGES_PER_TILE)
    q = tl.load(
        Q + row * QS0 + (head * GQA + hs[:, None]) * QS1 + ds[None, :], (hs[:, None] < GQA) & (ds[None, :] < D), 0.0
    )
    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    kg, vg = 1.0, 1.0
    if SCALE_MODE == 1:
        kg, vg = tl.load(KS), tl.load(VS)
    PER_SPLIT: tl.constexpr = (TOPK + SPLITS - 1) // SPLITS
    for tile in range(tl.cdiv(PER_SPLIT, PAGES_PER_TILE)):
        local = tile * PAGES_PER_TILE + ns // 128
        slot = split * PER_SPLIT + local
        block = tl.load(Selected + head * TS0 + row * TS1 + slot * TS2, (local < PER_SPLIT) & (slot < TOPK), -1)
        valid = (block >= 0) & (block * 128 < tl.minimum(seq, position + 1))
        page = tl.load(Pages + batch * PS0 + block, valid, 0).to(tl.int64)
        base = KV + page[:, None] * KS0 + head * KS1 + (ns[:, None] % 128) * KS2
        k = tl.load(base + ds[None, :], valid[:, None] & (ds[None, :] < D), 0.0)
        logits = tl.dot(q, tl.trans(k), max_num_imprecise_acc=0)
        if SCALE_MODE == 2:
            kt = tl.load(KS + head * SKH + (page * 128 + ns % 128) * SKT, valid, 0.0)
            logits *= kt[None, :]
        logits *= SCALE * 1.4426950408889634 * kg
        visible = valid & (block * 128 + ns % 128 < seq) & (block * 128 + ns % 128 <= position)
        logits = tl.where(visible[None, :] & (hs[:, None] < GQA), logits, float("-inf"))
        next_m = tl.maximum(m, tl.max(logits, 1))
        safe_m = tl.where(next_m == float("-inf"), 0.0, next_m)
        alpha = tl.exp2(m - safe_m)
        probability = tl.exp2(logits - safe_m[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        acc *= alpha[:, None]
        v = tl.load(base + D + ds[None, :], valid[:, None] & (ds[None, :] < D), 0.0)
        if SCALE_MODE == 2:
            vt = tl.load(VS + head * SVH + (page * 128 + ns % 128) * SVT, valid, 0.0)
            weighted = probability * vt[None, :]
        else:
            weighted = probability
        acc += _fp8_pv(weighted, v)
        m = next_m
    result = acc * (vg / tl.where(denominator > 0, denominator, 1.0))[:, None]
    if SPLITS == 1:
        target = Out + row * OS0 + (head * GQA + hs[:, None]) * OS1 + ds[None, :]
    else:
        out_row = (row * HEADS + head * GQA + hs) * SPLITS + split
        target = Out + out_row[:, None] * D + ds[None, :]
        tl.store(LSE + out_row, m + tl.log2(denominator), hs < GQA)
    tl.store(target, result, (hs[:, None] < GQA) & (ds[None, :] < D))


@triton.jit
def _gqa_sparse_prefill_persistent_kernel(
    Q,
    KV,
    Pages,
    CuQ,
    Seq,
    Prefix,
    RowPtr,
    TilePtr,
    QueryIds,
    Partial,
    LSE,
    KG,
    VG,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    kvs0: tl.constexpr,
    kvs1: tl.constexpr,
    kvs2: tl.constexpr,
    ps0: tl.constexpr,
    HEADS_Q: tl.constexpr,
    BATCH: tl.constexpr,
    PAGES: tl.constexpr,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    GQA: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILES_PER_CTA: tl.constexpr,
    GLOBAL_SCALE: tl.constexpr,
    PACKED_KV: tl.constexpr = False,
    DENSE_BUCKETS: tl.constexpr = False,
    MAX_QUERY: tl.constexpr = 0,
    KS=None,
    VS=None,
    ORIGINAL_HEADS: tl.constexpr = 0,
    HEAD_OFFSET: tl.constexpr = 0,
    PIPE_STAGES: tl.constexpr = 1,
):
    group_id, hb = tl.program_id(0), tl.program_id(1)
    head, batch = hb // BATCH, hb % BATCH
    if DENSE_BUCKETS:
        groups: tl.constexpr = tl.cdiv(MAX_QUERY, TILES_PER_CTA * (BLOCK_M // GQA))
        block, group_start = group_id // groups, group_id % groups
        row_start = block * MAX_QUERY
        row_end = row_start + tl.load(RowPtr + hb * PAGES + block)
        if group_start * TILES_PER_CTA * (BLOCK_M // GQA) >= row_end - row_start:
            return
    else:
        tiles = TilePtr + hb * (PAGES + 1)
        if group_id >= tl.load(tiles + PAGES):
            return
        lo, hi = 0, PAGES
        while lo < hi:
            mid = (lo + hi) // 2
            right = group_id >= tl.load(tiles + mid + 1)
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(right, hi, mid)
        block = lo
        row_start = tl.load(RowPtr + hb * (PAGES + 1) + block)
        row_end = tl.load(RowPtr + hb * (PAGES + 1) + block + 1)
        group_start = group_id - tl.load(tiles + block)
    ms, ns, ds = tl.arange(0, BLOCK_M), tl.arange(0, 128), tl.arange(0, 128)
    qhead = head * GQA + ms % GQA
    page = tl.load(Pages + batch * ps0 + block).to(tl.int64)
    base = KV + page * kvs0 + head * kvs1
    Q_FP8: tl.constexpr = Q.dtype.element_ty == tl.float8e4nv
    if PACKED_KV:
        k = _load_nvfp4_page(KV, KS, page, head + HEAD_OFFSET, kvs0, kvs1, kvs2, ORIGINAL_HEADS, 0)
        v = _load_nvfp4_page(KV, VS, page, head + HEAD_OFFSET, kvs0, kvs1, kvs2, ORIGINAL_HEADS, 64)
        if Q_FP8:
            k = _nvfp4_to_fp8(k)
            v = _nvfp4_to_fp8(v)
    else:
        k = tl.load(base + ns[:, None] * kvs2 + ds[None, :]).to(Q.dtype.element_ty)
        v = tl.load(base + ns[:, None] * kvs2 + 128 + ds[None, :]).to(Q.dtype.element_ty)
    q_start = tl.load(CuQ + batch)
    prefix, seq = tl.load(Prefix + batch), tl.load(Seq + batch)
    qk_scale = SCALE * 1.4426950408889634
    v_scale = 1.0
    if GLOBAL_SCALE:
        qk_scale *= tl.load(KG)
        v_scale = tl.load(VG)
    if PACKED_KV and Q_FP8:
        # Undo _nvfp4_to_fp8's /8 after each FP8 dot, in FP32.
        qk_scale *= 8.0
        v_scale *= 8.0
    positions = block * 128 + ns
    group_row = row_start + group_start * TILES_PER_CTA * (BLOCK_M // GQA)
    for tile in tl.range(
        0, tl.minimum(TILES_PER_CTA, tl.cdiv(row_end - group_row, BLOCK_M // GQA)), num_stages=PIPE_STAGES
    ):
        row = group_row + tile * (BLOCK_M // GQA) + ms // GQA
        active = row < row_end
        encoded = tl.load(QueryIds + hb * CAPACITY + row, active, 0)
        token, slot = encoded // TOPK, encoded % TOPK
        q = tl.load(Q + token[:, None] * qs0 + qhead[:, None] * qs1 + ds[None, :], active[:, None], 0.0)
        scores = tl.dot(q, tl.trans(k), max_num_imprecise_acc=0)
        scores *= qk_scale
        qpos = prefix + token - q_start
        scores = tl.where(
            active[:, None] & (positions[None, :] <= qpos[:, None]) & (positions[None, :] < seq), scores, float("-inf")
        )
        maximum = tl.where(active, tl.max(scores, 1), 0.0)
        probability = tl.exp2(scores - maximum[:, None])
        denominator = tl.sum(probability, 1)
        if Q_FP8:
            acc = _fp8_pv(probability, v)
        else:
            acc = tl.dot(probability.to(tl.bfloat16), v)
        result = acc * (v_scale / tl.where(denominator > 0, denominator, 1.0))[:, None]
        output_row = (token * HEADS_Q + qhead) * TOPK + slot
        tl.store(
            Partial + output_row[:, None] * 128 + ds[None, :], result, active[:, None], eviction_policy="evict_first"
        )
        tl.store(LSE + output_row, maximum + tl.log2(denominator), active)


@triton.jit
def _gqa_sparse_kv_nvfp4_kernel(
    Q,
    KV,
    KS,
    VS,
    KG,
    VG,
    TopK,
    Pages,
    CuQ,
    Seq,
    Prefix,
    Out,
    LSE,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    kvs0: tl.constexpr,
    kvs1: tl.constexpr,
    kvs2: tl.constexpr,
    ts0: tl.constexpr,
    ts1: tl.constexpr,
    ts2: tl.constexpr,
    ps0: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    HEADS_K: tl.constexpr,
    GQA: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Q_FP8: tl.constexpr,
    SPLITS: tl.constexpr,
    HEADS_Q: tl.constexpr,
    DECODE_Q: tl.constexpr = 0,
):
    task, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    token, split = task // SPLITS, task % SPLITS
    seq_len = tl.load(Seq + batch)
    if DECODE_Q:
        start, qlen, prefix = batch * DECODE_Q, DECODE_Q, seq_len - DECODE_Q
    else:
        start = tl.load(CuQ + batch)
        qlen = tl.load(CuQ + batch + 1) - start
        prefix = tl.load(Prefix + batch)
    if token >= qlen:
        return
    query_id = start + token
    qpos = prefix + token
    hs, ds, ns = tl.arange(0, BLOCK_H), tl.arange(0, 128), tl.arange(0, 128)
    q = tl.load(Q + query_id * qs0 + (head * GQA + hs[:, None]) * qs1 + ds[None, :], hs[:, None] < GQA, 0.0)
    kg, vg = tl.load(KG), tl.load(VG)
    if Q_FP8:
        # K/V are normalized by _nvfp4_to_fp8 before the FP8 dot.
        kg *= 8.0
        vg *= 8.0
    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, 128), tl.float32)
    for local_slot in range(tl.cdiv(TOPK, SPLITS)):
        slot = split * tl.cdiv(TOPK, SPLITS) + local_slot
        block = tl.load(TopK + head * ts0 + query_id * ts1 + slot * ts2, slot < TOPK, -1)
        if (block >= 0) & (block * 128 <= qpos) & (block * 128 < seq_len):
            page = tl.load(Pages + batch * ps0 + block).to(tl.int64)
            k = _load_nvfp4_page(KV, KS, page, head, kvs0, kvs1, kvs2, HEADS_K, 0)
            if Q_FP8:
                k = _nvfp4_to_fp8(k)
            logits = tl.dot(q, tl.trans(k), max_num_imprecise_acc=0)
            logits *= SCALE * 1.4426950408889634 * kg
            positions = block * 128 + ns
            logits = tl.where((positions[None, :] <= qpos) & (positions[None, :] < seq_len), logits, float("-inf"))
            next_m = tl.maximum(m, tl.max(logits, 1))
            alpha = tl.exp2(m - next_m)
            probability = tl.exp2(logits - next_m[:, None])
            acc *= alpha[:, None]
            v = _load_nvfp4_page(KV, VS, page, head, kvs0, kvs1, kvs2, HEADS_K, 64)
            if Q_FP8:
                acc += _fp8_pv(probability, _nvfp4_to_fp8(v))
            else:
                acc = tl.dot(probability.to(tl.bfloat16), v, acc)
            l = l * alpha + tl.sum(probability, 1)
            m = next_m
    result = acc * (vg / tl.where(l > 0, l, 1.0))[:, None]
    if SPLITS == 1:
        tl.store(Out + query_id * os0 + (head * GQA + hs[:, None]) * os1 + ds[None, :], result, hs[:, None] < GQA)
    else:
        rows = (query_id * HEADS_Q + head * GQA + hs) * SPLITS + split
        tl.store(Out + rows[:, None] * 128 + ds[None, :], result, hs[:, None] < GQA)
        tl.store(LSE + rows, m + tl.log2(l), hs < GQA)


# Host policy: admission, resource budget and workload thresholds


def _persistent_kind(q, packed, max_query_len):
    """Ordinary page reuse depends on per-request length; staging on total work."""
    if q.dtype == torch.float8_e4m3fn:
        if packed and q.shape[0] >= _NVFP4_STAGING_MIN_TOKENS:
            return "nvfp4"
        if not packed and max_query_len >= _LONG_QUERY_MIN:
            return "fp8"
    return "general"


def _persistent_workspace_layout(q, kv, selected, pages, seq, maxq, total_heads, packed, heads=1):
    """Describe one KV-head group's scratch layout for budgeting and allocation."""
    kind = _persistent_kind(q, packed, maxq)
    dense = kind != "fp8" and maxq <= _DENSE_MAX_QUERY and pages.shape[1] <= _DENSE_MAX_PAGES
    capacity = pages.shape[1] * maxq if dense else maxq * selected.shape[-1]
    batch, page_count, topk = seq.numel(), pages.shape[1], selected.shape[-1]
    gqa = q.shape[1] // total_heads
    layout = dict(
        kind=kind,
        dense=dense,
        capacity=capacity,
        gqa=gqa,
        counts_shape=(heads * batch, page_count),
        ids_shape=(heads * batch, capacity),
        rows_shape=(heads * batch, page_count + 1),
        partial_shape=(q.shape[0], heads * gqa, topk, 128),
    )
    if kind == "nvfp4":
        layout["stage_shape"] = (kv.shape[0], heads, 128, 128)
    return layout


def _persistent_workspace_bytes(q, kv, selected, pages, seq, maxq, heads, packed):
    """Return scratch bytes, partial elements and a budgeted KV-head group."""
    layout = _persistent_workspace_layout(q, kv, selected, pages, seq, maxq, heads, packed)
    partial_elements = prod(layout["partial_shape"])
    per_head = partial_elements * 2 + prod(layout["partial_shape"][:-1]) * 4
    # Budget CSR storage even when dense buckets alias rows.
    per_head += 4 * (prod(layout["counts_shape"]) + prod(layout["ids_shape"]) + 2 * prod(layout["rows_shape"]))
    if "stage_shape" in layout:
        per_head += 2 * prod(layout["stage_shape"])  # Separate E4M3 K and V.
    # More heads amortize launches, but must not disqualify inputs that fit
    # with smaller groups. The caller rejects even a single-head overflow.
    for group in (min(4, heads), min(2, heads), 1):
        scratch, elements = group * per_head, group * partial_elements
        if scratch <= _PERSISTENT_MAX_WORKSPACE and elements <= 2**31:
            break
    return scratch, elements, group


def _persistent_admissible(
    q, kv, selected, pages, cu, seq, prefix, maxq, heads, output, kg=None, vg=None, *, packed=False
):
    if not is_sm100_device(q.device) or q.ndim != 3 or q.shape[-1] != 128 or heads <= 0:
        return False
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or q.shape[1] % heads:
        return False
    if q.shape[1] // heads not in (1, 2, 4, 8, 16):
        return False
    minimum = _PACKED_FP8_MIN_TOKENS if packed and q.dtype == torch.float8_e4m3fn else _PERSISTENT_MIN_TOKENS
    if q.shape[0] < minimum or maxq < 1 or maxq > q.shape[0]:
        return False
    width = 128 if packed else 256
    if kv.ndim != 4 or tuple(kv.shape[1:]) != (heads, 128, width):
        return False
    if not packed and kv.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        return False
    if not packed and q.dtype == torch.float8_e4m3fn and kv.dtype != q.dtype:
        return False
    if selected.ndim != 3 or selected.shape[:2] != (heads, q.shape[0]) or selected.shape[-1] not in (4, 8, 16, 32):
        return False
    if output.shape != q.shape or output.dtype != torch.bfloat16:
        return False
    if cu is None or cu.ndim != 1 or cu.numel() < 2 or pages.ndim != 2 or pages.shape[0] != cu.numel() - 1:
        return False
    if seq.shape != (cu.numel() - 1,) or prefix.shape != seq.shape or not 0 < pages.shape[1] <= 65536:
        return False
    if any(t.device != q.device for t in (kv, selected, pages, cu, seq, prefix, output)):
        return False
    if any(t.stride(-1) != 1 for t in (q, kv, output, pages, cu, seq, prefix)):
        return False
    if any(t.dtype not in (torch.int32, torch.int64) for t in (selected, pages, cu, seq, prefix)):
        return False
    if (kg is None) != (vg is None):
        return False
    if kg is not None and any(t.numel() != 1 or t.dtype != torch.float32 or t.device != q.device for t in (kg, vg)):
        return False
    if packed and _persistent_kind(q, True, maxq) == "nvfp4":
        # Do not expand an entire serving pool for a small active selection.
        if kv.shape[0] * 4 > q.shape[0] * selected.shape[-1]:
            return False
    scratch, elements, _ = _persistent_workspace_bytes(q, kv, selected, pages, seq, maxq, heads, packed)
    return (
        scratch <= _PERSISTENT_MAX_WORKSPACE
        # The maximum offset is element_count - 1, not element_count.
        and elements <= 2**31
        and max(q.numel(), kv.numel(), q.shape[0] * selected.shape[-1]) < 2**31
    )


def _can_use_persistent(
    q,
    kv,
    selected,
    pages,
    cu,
    seq,
    prefix,
    max_query_len,
    heads,
    output,
    k_scale=None,
    v_scale=None,
):
    """Shared SM100 admission for ordinary BF16/FP8 prefill."""
    return _persistent_admissible(
        q, kv, selected, pages, cu, seq, prefix, max_query_len, heads, output, k_scale, v_scale
    )


# Execution families: persistent prefill and FP8 QKV


_PERSISTENT_LAUNCH_CACHE = {}


def _build_persistent_plans(
    q, kv, selected, pages, seq, maxq, total_heads, scale, output, kg, vg, packed
):
    """Build bucket, persistent attention and merge launches; do not execute."""
    layout = _persistent_workspace_layout(q, kv, selected, pages, seq, maxq, total_heads, packed)
    kind = layout["kind"]
    if kind in ("fp8", "nvfp4"):
        from .sparse_attn_gluon import (
            _gqa_sparse_prefill_persistent_q_fp8_kv_fp8_kernel,
            _gqa_sparse_prefill_persistent_q_fp8_kv_nvfp4_kernel,
        )
    _, _, group = _persistent_workspace_bytes(q, kv, selected, pages, seq, maxq, total_heads, packed)
    batch, page_count, topk = seq.numel(), pages.shape[1], selected.shape[-1]
    total, gqa = q.shape[0], layout["gqa"]
    dense, capacity = layout["dense"], layout["capacity"]
    result = []
    for lo in range(0, total_heads, group):
        heads = min(group, total_heads - lo)
        plan = dict(
            lo=lo,
            heads=heads,
            **_persistent_workspace_layout(q, kv, selected, pages, seq, maxq, total_heads, packed, heads),
            stage=None,
        )
        cache_stride = kv.stride()[:3]
        if kind == "nvfp4":
            plan["stage"] = _KernelLaunch(
                _stage_nvfp4_kv_fp8, (kv.shape[0], heads), (*cache_stride, heads, lo, total_heads), num_warps=4
            )
            cache_stride = (heads * 128 * 128, 128 * 128, 128)
        if dense:
            plan["dense_bucket"] = _KernelLaunch(
                _dense_page_buckets,
                (page_count, heads * batch),
                (
                    *selected.stride(),
                    batch,
                    page_count,
                    topk,
                    maxq,
                    triton.next_power_of_2(maxq),
                    triton.next_power_of_2(topk),
                ),
                num_warps=8,
            )
            groups = max(1, 16 // (batch * heads))
            if packed and q.dtype == torch.float8_e4m3fn and batch > 1:
                groups = 1
            tiles = triton.cdiv(maxq, groups * (64 // gqa))
            bound = page_count * triton.cdiv(maxq, tiles * (64 // gqa))
        else:
            # Reuse each page across more query tiles as the head group grows.
            # Bound that reuse by an estimated two CTAs per SM, so small
            # workloads retain parallelism. Bucket skew may lower actual work.
            parallel = capacity * heads * batch // (2 * _sm_count(q.device) * (64 // gqa))
            parallel = 1 << (max(1, parallel).bit_length() - 1)
            budget = (16 if kind == "fp8" else 20) * heads
            # Keep enough independent CTAs for the Gluon execution families.
            if kind != "general":
                budget = min(budget, 32)
            tiles = min(budget, max(16, parallel))
            count_elements = heads * batch * page_count
            lse_elements = total * heads * gqa * topk
            plan["initialize"] = _KernelLaunch(
                _init_persistent_workspace,
                (triton.cdiv(max(count_elements, lse_elements), 1024),),
                (count_elements, lse_elements, 1024),
                num_warps=4,
            )
            # Amortize CSR aggregation across more entries.
            bucket_size = 512
            grid = (triton.cdiv(capacity, bucket_size), heads * batch)
            tail = (*selected.stride(), batch, page_count, topk, capacity)
            bucket = _aggregate_page_buckets if page_count <= 1024 and maxq >= _LONG_QUERY_MIN else _bucket_queries
            suffix = (True,) if bucket is _aggregate_page_buckets else ()
            plan["count"] = _KernelLaunch(bucket, grid, (*tail, False, bucket_size, *suffix), num_warps=8)
            plan["prefix"] = _KernelLaunch(
                _prefix_buckets,
                (heads * batch,),
                (page_count, tiles * (64 // gqa), triton.next_power_of_2(page_count)),
                num_warps=4,
            )
            plan["scatter"] = _KernelLaunch(bucket, grid, (*tail, True, bucket_size, *suffix), num_warps=8)
            bound = triton.cdiv(capacity, tiles * (64 // gqa)) + page_count
        tail = (
            *q.stride()[:2],
            *cache_stride,
            pages.stride(0),
            heads * gqa,
            batch,
            page_count,
            topk,
            capacity,
            gqa,
            scale,
            64,
            tiles,
            kg is not None and vg is not None,
        )
        grid = (bound, heads * batch)
        if kind == "fp8":
            plan["attention"] = _KernelLaunch(
                _gqa_sparse_prefill_persistent_q_fp8_kv_fp8_kernel,
                grid,
                (*tail, maxq >= _FP8_NARROW_OUTPUT_MIN_QUERY),
                num_warps=4,
                num_stages=1,
            )
        elif kind == "nvfp4":
            plan["attention"] = _KernelLaunch(
                _gqa_sparse_prefill_persistent_q_fp8_kv_nvfp4_kernel,
                grid,
                (*tail, dense, maxq if dense else 0),
                num_warps=4,
                num_stages=1,
            )
        else:
            fp8_packed = packed and q.dtype == torch.float8_e4m3fn
            plan["attention"] = _KernelLaunch(
                _gqa_sparse_prefill_persistent_kernel,
                grid,
                (*tail, packed, dense, maxq if dense else 0),
                suffix=(total_heads if packed else 0, lo if packed else 0, 2 if fp8_packed or not dense else 3),
                num_warps=8 if fp8_packed and maxq >= 256 else 4,
                num_stages=1,
            )
        plan["merge"] = _KernelLaunch(
            _merge_tiled,
            (triton.cdiv(total * heads * gqa, 8), 1),
            (total * heads * gqa, heads * gqa, topk, *output.stride()[:2], 8, 128, triton.next_power_of_2(topk)),
            num_warps=4,
            num_stages=1,
        )
        result.append(plan)
    return tuple(result)


def _run_persistent_prefill(
    q,
    kv,
    selected,
    pages,
    cu_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    *,
    k_global_scale=None,
    v_global_scale=None,
    packed_k_scale=None,
    packed_v_scale=None,
):
    """Run the three persistent kernel variants via `_build_persistent_plans`.

    Cache only explicit launch metadata and compiled kernels. Every call
    rebuilds buckets and reads the current scale values on device.
    """
    tensors = (
        q,
        kv,
        selected,
        pages,
        cu_q,
        seq_lens,
        prefix_lens,
        output,
        k_global_scale,
        v_global_scale,
        packed_k_scale,
        packed_v_scale,
    )
    key = tuple(_tensor_metadata(t) for t in tensors) + (
        max_query_len,
        num_kv_heads,
        sm_scale,
    )
    plans = _PERSISTENT_LAUNCH_CACHE.get(key)
    if plans is None:
        plans = _build_persistent_plans(
            q,
            kv,
            selected,
            pages,
            seq_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_global_scale,
            v_global_scale,
            packed_k_scale is not None,
        )
        _cache_plan(_PERSISTENT_LAUNCH_CACHE, key, plans)
    for plan in plans:
        lo, size, gqa = plan["lo"], plan["heads"], plan["gqa"]
        query = q[:, lo * gqa : (lo + size) * gqa]
        result = output[:, lo * gqa : (lo + size) * gqa]
        chosen = selected[lo : lo + size]
        cache = kv if packed_k_scale is not None else kv[:, lo : lo + size]
        ks, vs = packed_k_scale, packed_v_scale
        if plan["stage"] is not None:
            cache = torch.empty(plan["stage_shape"], device=q.device, dtype=torch.float8_e4m3fn)
            vs = torch.empty_like(cache)
            plan["stage"](kv, packed_k_scale, packed_v_scale, cache, vs)
        counts = torch.empty(plan["counts_shape"], device=q.device, dtype=torch.int32)
        partial = torch.empty(plan["partial_shape"], device=q.device, dtype=torch.bfloat16)
        lse = torch.empty(plan["partial_shape"][:-1], device=q.device, dtype=torch.float32)
        ids = torch.empty(plan["ids_shape"], device=q.device, dtype=torch.int32)
        if plan["dense"]:
            lse.fill_(float("-inf"))
            plan["dense_bucket"](chosen, cu_q, seq_lens, prefix_lens, counts, ids)
            rows, tile_ptr = counts, counts
        else:
            plan["initialize"](counts, lse)
            rows = torch.empty(plan["rows_shape"], device=q.device, dtype=torch.int32)
            tile_ptr = torch.empty_like(rows)
            plan["count"](chosen, cu_q, seq_lens, prefix_lens, counts, rows, ids)
            plan["prefix"](counts, rows, tile_ptr)
            plan["scatter"](chosen, cu_q, seq_lens, prefix_lens, counts, rows, ids)
        operands = (
            query,
            cache,
            pages,
            cu_q,
            seq_lens,
            prefix_lens,
            rows,
            tile_ptr,
            ids,
            partial,
            lse,
            query if k_global_scale is None else k_global_scale,
            query if v_global_scale is None else v_global_scale,
        )
        extra = () if plan["kind"] == "fp8" else ((vs,) if plan["kind"] == "nvfp4" else (ks, vs))
        plan["attention"](*operands, extra=extra)
        plan["merge"](partial, lse, result)
        del operands, partial, lse, ids, rows, tile_ptr, counts, cache, ks, vs


_FP8_LAUNCH_CACHE = {}


def _build_fp8_attention_plan(
    q,
    kv,
    selected,
    pages,
    cu_q,
    seq,
    prefix,
    max_query_len,
    heads,
    sm_scale,
    output,
    k_scale=None,
    v_scale=None,
    *,
    decode_query_len=0,
):
    """Validate metadata and prepare explicit shared-FP8 launch arguments."""
    if not (is_sm90_device(q.device) or is_sm100_device(q.device)):
        raise ValueError("FP8 QKV MSA currently targets NVIDIA SM90 and SM100")
    if q.dtype != torch.float8_e4m3fn or kv.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 QKV attention requires E4M3FN Q and KV")
    if q.ndim != 3 or q.shape[2] not in (64, 96, 128, 256) or heads <= 0:
        raise ValueError("FP8 QKV requires Q [tokens, heads, D], D in {64,96,128,256}")
    total, qheads, dim = q.shape
    if qheads == 0 or qheads % heads or qheads // heads > 64:
        raise ValueError("FP8 QKV requires integral GQA in [1,64]")
    if kv.ndim != 4 or tuple(kv.shape[1:]) != (heads, 128, 2 * dim):
        raise ValueError("FP8 KV must have shape [pages, Hkv, 128, 2*D]")
    if selected.ndim != 3 or selected.shape[:2] != (heads, total) or selected.shape[2] < 1:
        raise ValueError("Selections must have shape [Hkv, total_q, positive_topk]")
    if seq.ndim != 1 or pages.ndim != 2 or pages.shape[0] != seq.numel() or pages.shape[1] < 1:
        raise ValueError("Invalid sequence lengths or page table")
    if output.shape != q.shape or output.dtype != torch.bfloat16:
        raise ValueError("FP8 attention output must be BF16 with Q's shape")
    metadata = [selected, pages, seq]
    if decode_query_len:
        if decode_query_len < 1 or total != seq.numel() * decode_query_len:
            raise ValueError("Decode Q must have batch * decode_query_len tokens")
        max_query_len = decode_query_len
    else:
        if cu_q is None or prefix is None or cu_q.shape != (seq.numel() + 1,) or prefix.shape != seq.shape:
            raise ValueError("Prefill requires cu_seqlens_q and prefix_lens")
        metadata += [cu_q, prefix]
    if max_query_len < 0 or (total and max_query_len < 1):
        raise ValueError("Invalid maximum query length")
    if any(t.device != q.device for t in [kv, output] + metadata):
        raise ValueError("All attention tensors must be on the query device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise TypeError("Attention metadata must be int32 or int64")
    if any(t.stride(-1) != 1 for t in [q, kv, output, pages, seq] + ([] if decode_query_len else [cu_q, prefix])):
        raise ValueError("Q/KV/output and sequence metadata require contiguous last dimensions")
    mode, skh, skt, svh, svt = 0, 0, 0, 0, 0
    if (k_scale is None) != (v_scale is None):
        raise ValueError("k_scale and v_scale must both be supplied or both omitted")
    if k_scale is not None:
        if any(t.device != q.device or t.dtype != torch.float32 for t in (k_scale, v_scale)):
            raise ValueError("K/V scales must be FP32 on the query device")
        if k_scale.numel() == v_scale.numel() == 1:
            mode = 1
        elif (
            k_scale.ndim == 2
            and k_scale.shape == v_scale.shape
            and k_scale.shape[0] == heads
            and k_scale.shape[1] >= kv.shape[0] * 128
        ):
            mode = 2
            skh, skt = k_scale.stride()
            svh, svt = v_scale.stride()
        else:
            raise ValueError("K/V scales must be scalars or [Hkv, physical_cache_tokens]")
    if total == 0:
        return
    topk, gqa = selected.shape[-1], qheads // heads
    # Split Top-K pages to fill idle SMs; cap partial storage by a device-sized grid.
    target_grid = 256 if decode_query_len else 2 * _sm_count(q.device)
    target = min(topk, max(1, target_grid // max(1, total * heads)), 16)
    splits = 1 << (target.bit_length() - 1)
    optimized_decode = bool(decode_query_len) and is_sm100_device(q.device)
    parallelism = total * heads
    kernel = _gqa_sparse_decode_q_fp8_kv_fp8_kernel if optimized_decode else _gqa_sparse_q_fp8_kv_fp8_kernel
    tail = (
        q.stride(0),
        q.stride(1),
        *kv.stride()[:3],
        *selected.stride(),
        pages.stride(0),
        output.stride(0),
        output.stride(1),
        skh,
        skt,
        svh,
        svt,
        qheads,
        gqa,
        dim,
        topk,
        sm_scale,
        mode,
        decode_query_len,
        splits,
        max(16, triton.next_power_of_2(gqa)),
        triton.next_power_of_2(dim),
    )
    if optimized_decode:
        tail += (2 if parallelism >= _DECODE_PAIR_MIN_PARALLELISM else 1,)
    attend = _KernelLaunch(kernel, (max_query_len * splits, heads, seq.numel()), tail, num_warps=4, num_stages=1)
    merge = None
    if splits > 1:
        merge = _KernelLaunch(
            _merge_fp8_splits,
            (total, qheads),
            (qheads, dim, splits, output.stride(0), output.stride(1), triton.next_power_of_2(dim)),
            num_warps=4,
        )
    return attend, merge, (total, qheads, splits, dim)


def _run_fp8_qkv_attention(
    q,
    kv,
    selected,
    pages,
    cu_q,
    seq,
    prefix,
    max_query_len,
    heads,
    sm_scale,
    output,
    k_scale=None,
    v_scale=None,
    *,
    decode_query_len=0,
):
    """Run `_gqa_sparse_q_fp8_kv_fp8_kernel` or its decode variant, then merge."""
    tensors = (q, kv, selected, pages, cu_q, seq, prefix, output, k_scale, v_scale)
    key = tuple(_tensor_metadata(t) for t in tensors) + (
        max_query_len,
        heads,
        sm_scale,
        decode_query_len,
    )
    plan = _FP8_LAUNCH_CACHE.get(key)
    if plan is None:
        plan = _build_fp8_attention_plan(
            q,
            kv,
            selected,
            pages,
            cu_q,
            seq,
            prefix,
            max_query_len,
            heads,
            sm_scale,
            output,
            k_scale,
            v_scale,
            decode_query_len=decode_query_len,
        )
        if plan is None:
            return
        _cache_plan(_FP8_LAUNCH_CACHE, key, plan)
    attend, merge, shape = plan
    if merge is None:
        partial, lse = output, output
    else:
        partial = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
        lse = torch.empty(shape[:-1], device=q.device, dtype=torch.float32)
    attend(
        q,
        kv,
        selected,
        pages,
        seq if cu_q is None else cu_q,
        seq,
        seq if prefix is None else prefix,
        output if k_scale is None else k_scale,
        output if v_scale is None else v_scale,
        partial,
        lse,
    )
    if merge is not None:
        merge(partial, lse, output)


# Execution family: NVFP4 KV (packed input, optional per-call unpacking)


_NVFP4_LAUNCH_CACHE = {}


def _nvfp4_launch_tail(
    q, kv, topk, pages, heads, sm_scale, output, splits, decode_query_len
):
    q_fp8 = q.dtype == torch.float8_e4m3fn
    gqa = q.shape[1] // heads
    return (
        q.stride(0),
        q.stride(1),
        *kv.stride()[:3],
        *topk.stride(),
        pages.stride(0),
        output.stride(0),
        output.stride(1),
        heads,
        gqa,
        topk.shape[-1],
        sm_scale,
        max(64 if q_fp8 else 16, triton.next_power_of_2(gqa)),
        q_fp8,
        splits,
        q.shape[1],
        decode_query_len,
    )


def _run_nvfp4_plan(plan, q, kv, topk, pages, cu, seq, prefix, output, ks, vs, kg, vg):
    """Run `_run_persistent_prefill` or packed NVFP4 attention and optional merge."""
    if plan["mode"] == "persistent":
        return _run_persistent_prefill(
            q,
            kv,
            topk,
            pages,
            cu,
            seq,
            prefix,
            plan["max_query_len"],
            plan["heads"],
            plan["scale"],
            output,
            k_global_scale=kg,
            v_global_scale=vg,
            packed_k_scale=ks,
            packed_v_scale=vs,
        )
    splits, attend, merge = plan["splits"], plan["attend"], plan["merge"]
    if splits == 1:
        target, lse = output, output
    else:
        target = torch.empty((*q.shape[:2], splits, 128), device=q.device, dtype=torch.bfloat16)
        lse = torch.empty((*q.shape[:2], splits), device=q.device, dtype=torch.float32)
    attend(
        q,
        kv,
        ks,
        vs,
        kg,
        vg,
        topk,
        pages,
        seq if cu is None else cu,
        seq,
        seq if prefix is None else prefix,
        target,
        lse,
    )
    if merge is not None:
        merge(target, lse, output)


def _run_nvfp4_kv_attention(
    q,
    kv,
    topk,
    pages,
    cu_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    *,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    decode_query_len=0,
):
    """Run `_run_nvfp4_plan`, `_run_persistent_prefill`, or
    `_gqa_sparse_kv_nvfp4_kernel` for BF16/FP8 Q with packed NVFP4 KV.
    """
    tensors = (
        q,
        kv,
        topk,
        pages,
        cu_q,
        seq_lens,
        prefix_lens,
        output,
        k_scale,
        v_scale,
        k_global_scale,
        v_global_scale,
    )
    cache_key = tuple(_tensor_metadata(t) for t in tensors) + (
        num_kv_heads,
        sm_scale,
        max_query_len,
        decode_query_len,
    )
    plan = _NVFP4_LAUNCH_CACHE.get(cache_key)
    if plan is not None:
        return _run_nvfp4_plan(
            plan,
            q,
            _packed_bytes(kv),
            topk,
            pages,
            cu_q,
            seq_lens,
            prefix_lens,
            output,
            k_scale,
            v_scale,
            k_global_scale,
            v_global_scale,
        )
    if not is_sm100_device(q.device):
        raise ValueError("NVFP4 MSA requires an NVIDIA SM100 device")
    kv = _packed_bytes(kv)
    if q.ndim != 3 or q.shape[-1] != 128 or q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("NVFP4 attention requires BF16/FP8 Q with head_dim=128")
    if num_kv_heads <= 0 or q.shape[1] == 0:
        raise ValueError("Query and KV head counts must be positive")
    if kv.dtype != torch.uint8 or kv.ndim != 4 or tuple(kv.shape[1:]) != (num_kv_heads, 128, 128):
        raise ValueError("NVFP4 KV must be uint8 [pages, Hkv, 128, 128], packed K then V")
    if q.shape[1] % num_kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if output.shape != q.shape or output.dtype != torch.bfloat16:
        raise ValueError("NVFP4 attention output must be BF16 with the query shape")
    required_scales = kv.shape[0] * num_kv_heads * 128 * 8
    for scale in (k_scale, v_scale):
        if (
            scale is None
            or scale.numel() != required_scales
            or scale.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or not scale.is_contiguous()
            or scale.device != q.device
        ):
            raise ValueError("K/V scales require contiguous E4M3 128x4-swizzled storage")
    for scale in (k_global_scale, v_global_scale):
        if scale is None or scale.numel() != 1 or scale.dtype != torch.float32 or scale.device != q.device:
            raise ValueError("NVFP4 global scales must be CUDA FP32 scalar tensors")
    metadata = (topk, pages, seq_lens) if decode_query_len else (topk, pages, cu_q, seq_lens, prefix_lens)
    if any(t.device != q.device for t in (kv, output) + metadata):
        raise ValueError("All NVFP4 attention tensors must share the query device")
    if any(t.stride(-1) != 1 for t in (q, kv, output)):
        raise ValueError("Q, KV and output require contiguous last dimensions")
    batch = seq_lens.numel() if decode_query_len else cu_q.numel() - 1
    if decode_query_len:
        if decode_query_len < 1 or q.shape[0] != batch * decode_query_len:
            raise ValueError("Decode Q must have batch * decode_query_len tokens")
        max_query_len = decode_query_len
    elif cu_q.ndim != 1 or prefix_lens.shape != (batch,):
        raise ValueError("Invalid prefill sequence metadata")
    if (
        seq_lens.shape != (batch,)
        or pages.ndim != 2
        or pages.shape[0] != batch
        or pages.shape[1] < 1
        or pages.stride(-1) != 1
        or topk.ndim != 3
        or topk.shape[:2] != (num_kv_heads, q.shape[0])
        or topk.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Invalid sparse selection or ragged sequence metadata layout")
    if any(t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise TypeError("NVFP4 metadata must be int32 or int64")
    if seq_lens.stride(0) != 1 or (not decode_query_len and (cu_q.stride(0) != 1 or prefix_lens.stride(0) != 1)):
        raise ValueError("Sequence metadata must be contiguous")
    if topk.shape[-1] < 1 or (q.shape[0] and max_query_len < 1):
        raise ValueError("TopK and maximum query length must be positive")
    if q.shape[0] == 0:
        return
    if not decode_query_len and _persistent_admissible(
        q,
        kv,
        topk,
        pages,
        cu_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        output,
        k_global_scale,
        v_global_scale,
        packed=True,
    ):
        plan = dict(
            mode="persistent", max_query_len=max_query_len,
            heads=num_kv_heads, scale=sm_scale,
        )
    else:
        # Process selected packed pages without staging the whole KV pool.
        target = max(1, min(topk.shape[-1], 2 * _sm_count(q.device) // max(1, q.shape[0] * num_kv_heads)))
        splits = min(16, 1 << (target.bit_length() - 1))
        attend = _KernelLaunch(
            _gqa_sparse_kv_nvfp4_kernel,
            (max_query_len * splits, num_kv_heads, batch),
            _nvfp4_launch_tail(q, kv, topk, pages, num_kv_heads, sm_scale, output, splits, decode_query_len),
            num_warps=4,
            num_stages=1,
        )
        merge = None
        if splits > 1:
            merge = _KernelLaunch(
                _combine_splits,
                (q.shape[0], triton.cdiv(q.shape[1], 4)),
                (q.shape[1], splits, output.stride(0), output.stride(1), triton.next_power_of_2(splits), 4),
                num_warps=4,
            )
        plan = dict(mode="query", splits=splits, attend=attend, merge=merge)
    _cache_plan(_NVFP4_LAUNCH_CACHE, cache_key, plan)
    return _run_nvfp4_plan(
        plan, q, kv, topk, pages, cu_q, seq_lens, prefix_lens,
        output, k_scale, v_scale, k_global_scale, v_global_scale,
    )


# Common-kernel adapters: BF16 Q with ordinary BF16/FP8 KV


def _run_common_bf16_q_prefill(
    q,
    kv_cache,
    topk_idx,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    num_kv_heads,
    sm_scale,
    output,
    k_scale=None,
    v_scale=None,
):
    """Run common `_gqa_sparse_fwd_fallback_kernel` or `_gqa_sparse_fwd_kernel`.

    Q is BF16; KV is BF16 or FP8, with the common kernels' scale handling.
    """
    _, num_heads, head_dim = q.shape
    gqa_group_size = num_heads // num_kv_heads
    block_size_h = triton.next_power_of_2(gqa_group_size)
    use_fp8 = kv_cache.dtype in generic._FP8_DTYPES
    k_scale_arg, v_scale_arg, stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t, kv_scale_mode = (
        generic._kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (output, output, 0, 0, 0, 0, generic._KV_SCALE_NONE)
    )
    if not use_fp8 and block_size_h >= 8:
        kernel = generic._gqa_sparse_fwd_fallback_kernel
        launch_options = {}
    else:
        kernel = _gqa_sparse_fwd_kernel
        launch_options = dict(num_warps=4, num_stages=3)
    grid = (max_query_len, num_kv_heads, cu_seqlens_q.shape[0] - 1)
    kernel[grid](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk_idx.shape[-1],
        1,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=generic.SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_QH=block_size_h,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        **launch_options,
    )


def _validate_attention_dtypes(q, kv, kv_format, k_global_scale, v_global_scale):
    """Reject unsupported input contracts before choosing an execution path."""
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("NVIDIA MSA requires BF16 or E4M3FN Q; packed FP4 Q is not supported")
    if kv_format not in (None, "nvfp4"):
        raise ValueError("Only kv_format='nvfp4' is supported")
    if kv_format == "nvfp4":
        # The NVFP4 launcher validates SM100, packed storage and its scales.
        return
    if (
        kv.dtype in (torch.uint8, torch.int8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
        or k_global_scale is not None
        or v_global_scale is not None
    ):
        raise ValueError("Packed NVFP4 KV/global scales require kv_format='nvfp4'")
    if q.dtype == torch.float8_e4m3fn:
        if kv.dtype != torch.float8_e4m3fn:
            raise TypeError("FP8 QKV attention requires E4M3FN Q and KV")
    elif kv.dtype not in (torch.bfloat16,) + generic._FP8_DTYPES:
        raise TypeError("BF16 Q requires BF16 or FP8 KV")


_DECODE_LAUNCH_CACHE = {}


def _run_common_bf16_q_decode(q, kv, selected, pages, seq, heads, scale, output, qlen, k_scale, v_scale):
    """Run common `_gqa_sparse_decode_kernel` and `_merge_topk_attn_out_kernel`,
    or delegate to `cuda_decode.minimax_m3_sparse_attn_decode`.

    Cache launch metadata and compiled code; allocate workspace per call.
    The fused/high-parallelism path uses the common launcher.
    """
    attention = cuda_decode._gqa_sparse_decode_kernel
    reduction = cuda_decode._merge_topk_attn_out_kernel
    # This adapter supports the current single Heuristics -> JIT wrapper.
    # If upstream changes that interface, use its ordinary launcher.
    if not all(isinstance(k, triton.runtime.autotuner.Heuristics) for k in (attention, reduction)):
        return cuda_decode.minimax_m3_sparse_attn_decode(
            q, kv, selected, pages, seq, heads, scale, output, qlen, k_scale, v_scale
        )
    kernel, merge = attention.fn, reduction.fn
    n, hq, dim = q.shape
    tensors = (q, kv, selected, pages, seq, output, k_scale, v_scale)
    if (
        n * heads >= cuda_decode._MSA_DECODE_FUSED_MIN_PARALLELISM
        or not _can_reuse_launch(kernel, merge)
        or type(scale) not in (float, int)
        or any(t is not None and type(t) is not torch.Tensor for t in tensors)
    ):
        return cuda_decode.minimax_m3_sparse_attn_decode(
            q, kv, selected, pages, seq, heads, scale, output, qlen, k_scale, v_scale
        )
    key = tuple(_tensor_metadata(t) for t in tensors) + (heads, scale, qlen)
    plan = _DECODE_LAUNCH_CACHE.get(key)
    if plan is None:
        # Device is part of the recipe key. Validate it when preparing a new
        # recipe, not on every launch of an already validated one.
        if not is_sm100_device(q.device):
            return cuda_decode.minimax_m3_sparse_attn_decode(
                q, kv, selected, pages, seq, heads, scale, output, qlen, k_scale, v_scale
            )
        assert n == seq.shape[0] * qlen
        fp8 = kv.dtype in generic._FP8_DTYPES
        scales = (
            generic._kv_scale_args(output, heads, k_scale, v_scale)
            if fp8
            else (output, output, 0, 0, 0, 0, generic._KV_SCALE_NONE)
        )
        _, _, ksh, kst, vsh, vst, mode = scales
        p, topk = max(1, n * heads), selected.shape[-1]
        target_grid = (
            3 * cuda_decode._sm_count(q.device)
            if cuda_decode._MSA_DECODE_GRID_BAND_LO <= p <= cuda_decode._MSA_DECODE_GRID_BAND_HI
            else cuda_decode._MSA_DECODE_TARGET_GRID
        )
        if not fp8 and p <= cuda_decode._MSA_DECODE_TARGET_GRID:
            skip = cuda_decode._MSA_DECODE_TARGET_GRID - 1
            if max(1, min(topk, skip // p)) == 1 and max(1, min(topk, target_grid // p)) > 1:
                target_grid = skip
        target = max(1, min(topk, target_grid // p))
        chunks = 1 << (target.bit_length() - 1)
        pdl = cuda_decode._pdl_supported()
        shape_o, shape_l = (chunks, n, hq, dim), (chunks, n, hq)
        stride_o = (n * hq * dim, hq * dim, dim, 1) if chunks > 1 else (0,) + output.stride()
        stride_l = (n * hq, hq, 1)
        # Use upstream's tile heuristics rather than duplicating their formulas.
        tile_args = dict(gqa_group_size=hq // heads, head_dim=dim)
        block_h = attention.values["BLOCK_SIZE_H"](tile_args)
        block_d = attention.values["BLOCK_SIZE_D"](tile_args)
        merge_d = reduction.values["BLOCK_SIZE_D"](tile_args)
        tail = (
            n,
            hq // heads,
            dim,
            topk,
            scale,
            qlen,
            *q.stride(),
            *kv.stride(),
            ksh,
            kst,
            vsh,
            vst,
            *selected.stride(),
            *stride_o,
            *stride_l,
            pages.stride(0),
            generic.SPARSE_BLOCK_SIZE,
            chunks,
            block_h,
            block_d,
            fp8,
            mode,
            pdl,
        )
        merge_tail = (dim, *stride_o, *stride_l, *output.stride(), chunks, merge_d, pdl)
        options = {"launch_pdl": True} if pdl and chunks > 1 else {}
        plan = dict(
            chunks=chunks, shape_o=shape_o, shape_l=shape_l, fp8=fp8,
            attention=_KernelLaunch(kernel, (n * chunks, heads), tail, **options),
            merge=_KernelLaunch(merge, (n, hq), merge_tail, **options) if chunks > 1 else None,
        )
        _cache_plan(_DECODE_LAUNCH_CACHE, key, plan)
    chunks = plan["chunks"]
    partial = torch.empty(plan["shape_o"], dtype=q.dtype, device=q.device) if chunks > 1 else output
    lse = torch.empty(plan["shape_l"], dtype=torch.float32, device=q.device)
    ks = k_scale if plan["fp8"] and k_scale is not None else output
    vs = v_scale if plan["fp8"] and v_scale is not None else output
    plan["attention"](q, kv, ks, vs, selected, partial, lse, pages, seq)
    if plan["merge"] is not None:
        plan["merge"](partial, lse, output)


# NVIDIA operator entries: public signatures remain unchanged


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    *,
    kv_format: str | None = None,
    k_global_scale: torch.Tensor | None = None,
    v_global_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention over selected blocks, with one query per row.

    ``kv_format="nvfp4"`` uses packed E2M1 K/V with shape
    ``[pages, Hkv, 128, 128]`` (64 K bytes followed by 64 V bytes),
    E4M3 block scales in cuBLAS 128x4 storage (``k_scale``/``v_scale``),
    and FP32 scalar global dequant scales. Q is BF16 or E4M3; output is BF16.
    No new arguments are required for existing BF16/FP8 callers.
    """
    _validate_attention_dtypes(q, kv_cache, kv_format, k_global_scale, v_global_scale)
    if force_generic() and kv_format is None:
        return generic.minimax_m3_sparse_attn(
            q, kv_cache, topk_idx, block_table, cu_seqlens_q, seq_lens,
            prefix_lens, max_query_len, num_kv_heads, sm_scale, output, k_scale, v_scale,
        )

    # 1. Packed NVFP4 KV: SM100 only, never a common-kernel fallback.
    if kv_format == "nvfp4":
        return _run_nvfp4_kv_attention(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_scale=k_scale,
            v_scale=v_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
        )

    # Preserve the common API's ignored-scale contract for unquantized BF16 KV.
    if kv_cache.dtype == torch.bfloat16:
        k_scale = v_scale = None

    # 2. SM100 persistent for eligible BF16/FP8 inputs.
    if _can_use_persistent(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        output,
        k_scale,
        v_scale,
    ):
        return _run_persistent_prefill(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_global_scale=k_scale,
            v_global_scale=v_scale,
        )

    # 3. Remaining FP8 QKV: shared SM90/SM100 implementation.
    if q.dtype == torch.float8_e4m3fn:
        return _run_fp8_qkv_attention(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_scale,
            v_scale,
        )

    # 4. Only SM90 with compatible TLE may enter the Hopper implementation.
    # Other devices use the ordinary NVIDIA or portable Triton kernels.
    if is_sm90_device(q.device) and hopper_prefill() is not None:
        return hopper_prefill().minimax_m3_sparse_attn(
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_scale,
            v_scale,
        )
    return _run_common_bf16_q_prefill(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    num_kv_heads,
    sm_scale,
    output,
    decode_query_len,
    k_scale=None,
    v_scale=None,
    *,
    kv_format=None,
    k_global_scale=None,
    v_global_scale=None,
):
    """Shared FP8/NVFP4 decode; BF16 Q reuses the common compute kernels."""
    if decode_query_len <= 0:
        raise ValueError("decode_query_len must be positive")
    _validate_attention_dtypes(q, kv_cache, kv_format, k_global_scale, v_global_scale)
    if force_generic() and kv_format is None:
        return generic.minimax_m3_sparse_attn_decode(
            q, kv_cache, topk_idx, block_table, seq_lens, num_kv_heads,
            sm_scale, output, decode_query_len, k_scale, v_scale,
        )

    # 1. Packed NVFP4 KV: SM100 only.
    if kv_format == "nvfp4":
        return _run_nvfp4_kv_attention(
            q,
            kv_cache,
            topk_idx,
            block_table,
            None,
            seq_lens,
            None,
            decode_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_scale=k_scale,
            v_scale=v_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
            decode_query_len=decode_query_len,
        )

    # 2. FP8 QKV: shared SM90/SM100 decode, no persistent prefill policy.
    if q.dtype == torch.float8_e4m3fn:
        return _run_fp8_qkv_attention(
            q,
            kv_cache,
            topk_idx,
            block_table,
            None,
            seq_lens,
            None,
            decode_query_len,
            num_kv_heads,
            sm_scale,
            output,
            k_scale,
            v_scale,
            decode_query_len=decode_query_len,
        )

    # 3. BF16 Q: preserve the common launcher on every architecture.
    return _run_common_bf16_q_decode(
        q,
        kv_cache,
        topk_idx,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        k_scale,
        v_scale,
    )
