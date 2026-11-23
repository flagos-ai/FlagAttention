# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Blackwell MSA index optimizations using the common Top-K selectors."""

import torch
import triton
import triton.language as tl
from flag_attn.utils import current_platform, is_sm100_device, round_up
from flag_attn.minimax_sparse_attention import index_topk as generic
from flag_attn.minimax_sparse_attention.index_topk import (
    _decode_index_score_kernel,
    _decode_topk_identity_kernel,
    _topk_index_partial_kernel,
    _topk_index_merge_kernel,
    _bitonic_merge,
)
from .utils import _packed_bytes, _sm_count, _tensor_metadata, _can_reuse_launch, _cache_plan, _KernelLaunch

SPARSE_BLOCK_SIZE = generic.SPARSE_BLOCK_SIZE
_DECODE_SINGLE_PASS_TOPK_MAX_BLOCKS = 1024
_DECODE_INDEX_LAUNCH_CACHE = {}


def _decode_score_chunks(dtype, batch, max_seq_len, heads, blocks):
    """One SM100 score-grid policy shared by regular and planned submission."""
    if dtype == torch.float8_e4m3fn:
        target_grid = 2048
    elif batch >= 16 and (max_seq_len >= 65536 or heads > 1):
        target_grid = 4096
    else:
        target_grid = 1024
    limit = min(256, triton.next_power_of_2(max(1, blocks)))
    target = max(1, min(limit, target_grid // max(1, batch)))
    return 1 << (target.bit_length() - 1)


def _decode_score_launch_config(q, k, pages, seq, maxseq, init, local, heads, qlen, maxqlen, score_stride, pdl):
    """Share score grid, kernel arguments and launch options across submission modes."""
    blocks = triton.cdiv(maxseq, SPARSE_BLOCK_SIZE)
    chunks = _decode_score_chunks(q.dtype, seq.numel(), maxseq, heads, blocks)
    tail = (
        heads,
        q.shape[-1],
        init,
        local,
        qlen,
        *q.stride(),
        *k.stride(),
        *score_stride,
        pages.stride(0),
        SPARSE_BLOCK_SIZE,
        triton.next_power_of_2(maxqlen),
        chunks,
        pdl,
    )
    options = {"launch_pdl": True} if pdl else {}
    # Multi-head, multi-position tiles need fewer stages to limit resource pressure.
    if heads > 1 and maxqlen > 1:
        options.update(num_warps=4, num_stages=2)
    return (seq.numel(), chunks), tail, options


def _decode_index_planned(q, k, pages, seq, maxseq, topk, init, local, heads, qlen, maxqlen):
    """Prepare score/selection metadata once; recompute both outputs every call."""
    key = tuple(_tensor_metadata(t) for t in (q, k, pages, seq)) + (
        maxseq,
        topk,
        init,
        local,
        heads,
        qlen,
        maxqlen,
    )
    plan = _DECODE_INDEX_LAUNCH_CACHE.get(key)
    if plan is None:
        # Device is included in the key; a cached recipe is already admitted.
        if not is_sm100_device(q.device):
            return None
        pdl = current_platform.is_arch_support_pdl()
        total = q.shape[0]
        blocks = triton.cdiv(maxseq, SPARSE_BLOCK_SIZE)
        width = round_up(blocks, 16)
        score_shape, selected_shape = (heads, total, width), (heads, total, topk)
        score_stride, selected_stride = (total * width, width, 1), (total * topk, topk, 1)
        score_grid, score_tail, score_options = _decode_score_launch_config(
            q, k, pages, seq, maxseq, init, local, heads, qlen, maxqlen, score_stride, pdl
        )
        select_tail = (*score_stride, *selected_stride, qlen, blocks, topk, triton.next_power_of_2(blocks), pdl)
        options = {"launch_pdl": True} if pdl else {}
        plan = dict(
            score_shape=score_shape,
            selected_shape=selected_shape,
            score=_KernelLaunch(_decode_index_score_kernel, score_grid, score_tail, **score_options),
            select=_KernelLaunch(_decode_topk_single_pass_kernel, (total, heads), select_tail,
                                 **dict(options, num_warps=4)),
        )
        _cache_plan(_DECODE_INDEX_LAUNCH_CACHE, key, plan)
    score = torch.empty(plan["score_shape"], dtype=torch.float32, device=q.device)
    selected = torch.empty(plan["selected_shape"], dtype=torch.int32, device=q.device)
    plan["score"](q, k, score, pages, seq)
    plan["select"](score, selected, seq)
    return selected


@triton.jit
def _decode_topk_single_pass_kernel(
    Score,
    Selected,
    Seq,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    QLEN: tl.constexpr,
    BLOCKS: tl.constexpr,
    TOPK: tl.constexpr,
    WIDTH: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Existing bitonic ordering, without partial/merge for bounded rows."""
    row, head = tl.program_id(0), tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    batch, offset = row // QLEN, row % QLEN
    visible = tl.maximum(tl.load(Seq + batch) - QLEN + offset + 1, 0)
    pages = tl.cdiv(visible, 128)
    ids = tl.arange(0, WIDTH)
    valid = (ids < BLOCKS) & (ids < pages)
    values = tl.load(Score + head * SS0 + row * SS1 + ids * SS2, valid, -1.0e30)
    values = tl.where(values == values, values, -1.0e30)
    indices = tl.where(valid, ids + 1, 0)
    levels: tl.constexpr = tl.standard._log2(WIDTH)
    for stage in tl.static_range(1, levels):
        values, indices = _bitonic_merge(values, indices, stage, 2, levels)
    values, indices = _bitonic_merge(values, indices, levels, True, levels)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(Selected + head * OS0 + row * OS1 + ids * OS2, tl.where(ids < pages, indices - 1, -1), ids < TOPK)


def use_page_parallel_scores(q, max_query_len, batch, heads):
    """Keep the serial-page BF16 path when it already fills the device."""
    return not (
        q.dtype == torch.bfloat16
        and max_query_len >= 4096
        and triton.cdiv(max_query_len, 64) * batch * heads >= 2 * _sm_count(q.device)
    )


@triton.jit
def _dense_index_scores(
    Q,
    K,
    Scores,
    Pages,
    CuQ,
    Seq,
    Prefix,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ps0: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    tile, block, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh // HEADS, bh % HEADS
    start = tl.load(CuQ + batch)
    qlen = tl.load(CuQ + batch + 1) - start
    seq = tl.load(Seq + batch)
    prefix = tl.load(Prefix + batch)
    rows = tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    ns, ds = tl.arange(0, 128), tl.arange(0, 128)
    scores = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    if block * 128 < tl.minimum(seq, prefix + tl.minimum(qlen, (tile + 1) * BLOCK_Q)):
        page = tl.load(Pages + batch * ps0 + block).to(tl.int64)
        q = tl.load(Q + (start + rows[:, None]) * qs0 + head * qs1 + ds[None, :], rows[:, None] < qlen, 0.0)
        k = tl.load(K + page * ks0 + ns[:, None] * ks1 + ds[None, :])
        # Native FP8 uses precise accumulation; other FP8 formats widen to FP16.
        if q.dtype.is_fp8() and not NATIVE_FP8:
            q, k = q.to(tl.float16), k.to(tl.float16)
        logits = tl.dot(q, tl.trans(k), max_num_imprecise_acc=0)
        positions = block * 128 + ns
        logits = tl.where(
            (positions[None, :] <= prefix + rows[:, None]) & (positions[None, :] < seq), logits, float("-inf")
        )
        scores = tl.max(logits, 1)
    tl.store(Scores + head * ss0 + (start + rows) * ss1 + block, scores, rows < qlen)


def dense_index_scores(
    q, k, pages, cu_q, seq_lens, prefix_lens, max_query_len, max_seq_len, num_kv_heads, *, block_q=None
):
    """Parallelize independent page scores to expose the full SM100 grid."""
    blocks = triton.cdiv(triton.cdiv(max_seq_len, 128), 16) * 16
    if block_q is None:
        grid_128 = triton.cdiv(max_query_len, 128) * blocks * (cu_q.numel() - 1) * num_kv_heads
        # Small tiles expose more CTAs; larger BF16 tiles improve page reuse.
        if max_query_len <= 32 or grid_128 < _sm_count(q.device):
            block_q = 32
        elif q.dtype == torch.bfloat16 and max_query_len >= 256:
            block_q = 256
        else:
            block_q = 128
    scores = torch.empty((num_kv_heads, q.shape[0], blocks), device=q.device, dtype=torch.float32)
    _dense_index_scores[(triton.cdiv(max_query_len, block_q), blocks, (cu_q.numel() - 1) * num_kv_heads)](
        q,
        k,
        scores,
        pages,
        cu_q,
        seq_lens,
        prefix_lens,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        scores.stride(0),
        scores.stride(1),
        pages.stride(0),
        num_kv_heads,
        block_q,
        q.dtype == k.dtype == torch.float8_e4m3fn,
        num_warps=4,
        num_stages=1,
    )
    return scores


@triton.jit
def _fp4_index_scores(
    Q,
    K,
    QS,
    KS,
    Scores,
    Pages,
    CuQ,
    Seq,
    Prefix,
    q_s0: tl.constexpr,
    q_s1: tl.constexpr,
    k_s0: tl.constexpr,
    k_s1: tl.constexpr,
    k_s2: tl.constexpr,
    qs_s0: tl.constexpr,
    qs_s1: tl.constexpr,
    ks_s0: tl.constexpr,
    ks_s1: tl.constexpr,
    ks_s2: tl.constexpr,
    score_s0: tl.constexpr,
    score_s1: tl.constexpr,
    score_s2: tl.constexpr,
    page_s0: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    VEC: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    DECODE_QLEN: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    q_tile, block_tile, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    block = block_tile * BLOCK_PAGES
    batch, head = bh // NUM_HEADS, bh % NUM_HEADS
    kv_head = head // (NUM_HEADS // KV_HEADS)
    seq_len = tl.load(Seq + batch)
    if DECODE_QLEN:
        start = batch * DECODE_QLEN
        q_len = DECODE_QLEN
        prefix = seq_len - DECODE_QLEN
    else:
        start = tl.load(CuQ + batch)
        q_len = tl.load(CuQ + batch + 1) - start
        prefix = tl.load(Prefix + batch)
    rows = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    ns = tl.arange(0, 128 * BLOCK_PAGES)
    tokens = block * 128 + ns
    blocks = block + tl.arange(0, BLOCK_PAGES)
    values = tl.full((BLOCK_Q, BLOCK_PAGES), float("-inf"), tl.float32)
    # Skip wholly invisible blocks without touching their page-table entries.
    if block * 128 < tl.minimum(seq_len, prefix + tl.minimum(q_len, (q_tile + 1) * BLOCK_Q)):
        page_valid = (block + ns // 128) * 128 < tl.minimum(seq_len, prefix + tl.minimum(q_len, (q_tile + 1) * BLOCK_Q))
        page = tl.load(Pages + batch * page_s0 + block + ns // 128, page_valid, 0).to(tl.int64)
        packed_d = tl.arange(0, 64)
        groups = tl.arange(0, 128 // VEC)
        q = tl.load(Q + (start + rows[:, None]) * q_s0 + head * q_s1 + packed_d[None, :], rows[:, None] < q_len, 0)
        k = tl.load(
            K + page[:, None] * k_s0 + kv_head * k_s1 + (ns[:, None] % 128) * k_s2 + packed_d[None, :],
            page_valid[:, None],
            0,
        )
        qs = tl.load(QS + (start + rows[:, None]) * qs_s0 + head * qs_s1 + groups[None, :], rows[:, None] < q_len, 0.0)
        ks = tl.load(
            KS + page[:, None] * ks_s0 + kv_head * ks_s1 + (ns[:, None] % 128) * ks_s2 + groups[None, :],
            page_valid[:, None],
            0.0,
        )
        if VEC == 16:
            qs = qs.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True)
            ks = ks.to(tl.uint8, bitcast=True).to(tl.float8e4nv, bitcast=True)
        logits = tl.dot_scaled(q, qs, "e2m1", tl.trans(k), ks, "e2m1")
        logits = tl.where(
            (tokens[None, :] <= (prefix + rows[:, None])) & (tokens[None, :] < seq_len), logits, float("-inf")
        )
        values = tl.max(tl.reshape(logits, (BLOCK_Q, BLOCK_PAGES, 128)), 2)
    if DECODE_QLEN:
        visible_blocks = tl.cdiv(tl.maximum(prefix + rows + 1, 0), 128)
        visible = blocks[None, :] < visible_blocks[:, None]
        initial = (blocks[None, :] < INIT) & visible
        local = (blocks[None, :] >= tl.maximum(0, visible_blocks[:, None] - LOCAL)) & visible
        values = tl.where(local, 1.0e29, tl.where(initial, 1.0e30, values))
    tl.store(
        Scores + head * score_s0 + (start + rows[:, None]) * score_s1 + blocks[None, :] * score_s2,
        values,
        rows[:, None] < q_len,
    )


def _prepare_fp4_inputs(q, k, num_kv_heads, fp4_format, q_scale, k_scale):
    """Validate storage/scales even when identity Top-K needs no score kernel."""
    if fp4_format not in ("mxfp4", "nvfp4"):
        raise ValueError("fp4_format must be 'mxfp4' or 'nvfp4'")
    if not is_sm100_device(q.device):
        raise ValueError("FP4 MSA requires an NVIDIA SM100 CUDA device")
    q, k = _packed_bytes(q), _packed_bytes(k)
    if num_kv_heads <= 0 or q.ndim != 3 or q.shape[-1] != 64 or q.shape[1] != num_kv_heads:
        raise ValueError("FP4 queries must have shape [total_q, num_kv_heads, 64]")
    if k.ndim == 3:
        k = k.unsqueeze(1)
        if k_scale is not None:
            k_scale = k_scale.unsqueeze(1)
    if k.ndim != 4 or k.shape[-2:] != (128, 64) or k.shape[1] == 0 or num_kv_heads % k.shape[1]:
        raise ValueError("FP4 index keys must be [pages, Hk, 128, 64] or [pages, 128, 64]")
    vec = 32 if fp4_format == "mxfp4" else 16
    expected_qs = (*q.shape[:2], 128 // vec)
    expected_ks = (*k.shape[:3], 128 // vec)
    for scale, shape in ((q_scale, expected_qs), (k_scale, expected_ks)):
        if scale is None or tuple(scale.shape) != shape:
            raise ValueError(f"FP4 public scale layout requires shape {shape}")
        allowed = (
            (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
            if vec == 32
            else (torch.uint8, torch.float8_e4m3fn)
        )
        if scale.dtype not in allowed:
            raise TypeError("MXFP4 scales require uint8 E8M0; NVFP4 scales require E4M3 bytes")
        if scale.device != q.device or scale.stride(-1) != 1:
            raise ValueError("FP4 scales must be on the query device and contiguous in the last axis")
    if k.device != q.device or q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("FP4 values must share a device and have contiguous packed dimensions")
    return q, k, q_scale, k_scale, vec


def _validate_fp4_metadata(q, pages, cu_q, seq_lens, prefix_lens, max_query_len, max_seq_len, decode_query_len):
    """Check metadata on both score-computing and identity-selection paths."""
    metadata = (pages, seq_lens) if decode_query_len else (pages, cu_q, seq_lens, prefix_lens)
    if any(t.device != q.device for t in metadata):
        raise ValueError("FP4 index metadata must share the query device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in metadata):
        raise TypeError("FP4 index metadata must be int32 or int64")
    if any(t.stride(-1) != 1 for t in metadata):
        raise ValueError("FP4 index metadata must have contiguous last dimensions")
    batch = seq_lens.numel()
    if seq_lens.ndim != 1 or pages.ndim != 2 or pages.shape[0] != batch:
        raise ValueError("Invalid FP4 index page table or sequence lengths")
    if max_seq_len < 0 or pages.shape[1] < triton.cdiv(max_seq_len, 128):
        raise ValueError("FP4 page table must cover max_seq_len")
    if decode_query_len:
        if decode_query_len < 1 or q.shape[0] != batch * decode_query_len:
            raise ValueError("FP4 decode queries must contain batch * decode_query_len tokens")
    elif cu_q.shape != (batch + 1,) or prefix_lens.shape != (batch,):
        raise ValueError("Invalid FP4 prefill sequence metadata")
    if q.shape[0] and max_query_len < 1:
        raise ValueError("max_query_len must be positive for nonempty FP4 input")


def fp4_index_scores(
    q,
    k,
    pages,
    cu_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    num_kv_heads,
    *,
    fp4_format,
    q_scale,
    k_scale,
    decode_query_len=0,
    init_blocks=0,
    local_blocks=0,
    score_out=None,
    block_pages=1,
):
    """Compute block scores from public, contiguous-in-D FP4 scale layouts."""
    q, k, q_scale, k_scale, vec = _prepare_fp4_inputs(
        q, k, num_kv_heads, fp4_format, q_scale, k_scale
    )
    _validate_fp4_metadata(q, pages, cu_q, seq_lens, prefix_lens, max_query_len, max_seq_len, decode_query_len)
    # Reinterpret E8M0 storage without a conversion kernel; dot_scaled expects
    # exponent bytes for MXFP4. NVFP4 keeps E4M3's typed representation.
    if vec == 32:
        q_scale, k_scale = q_scale.view(torch.uint8), k_scale.view(torch.uint8)
    blocks = triton.cdiv(max_seq_len, 128)
    padded_blocks = triton.cdiv(blocks, 16) * 16
    if score_out is None:
        scores = torch.empty((num_kv_heads, q.shape[0], padded_blocks), device=q.device, dtype=torch.float32)
    else:
        if (
            score_out.ndim != 3
            or score_out.shape[:2] != (num_kv_heads, q.shape[0])
            or score_out.shape[2] < blocks
            or score_out.dtype != torch.float32
            or score_out.device != q.device
        ):
            raise ValueError("score_out must be FP32 [heads, total_q, >=max_blocks]")
        scores = score_out
        padded_blocks = score_out.shape[2]
    # FlagTree's SM100 scaled-MMA lowering requires M >= 128. Smaller tiles
    # decompose MXFP4 to ordinary arithmetic and cannot lower NVFP4 scales.
    if padded_blocks % block_pages:
        raise ValueError("score block count must be divisible by block_pages")
    grid = (
        triton.cdiv(max_query_len, 128),
        padded_blocks // block_pages,
        (seq_lens.numel() if decode_query_len else cu_q.numel() - 1) * num_kv_heads,
    )
    _fp4_index_scores[grid](
        q,
        k,
        q_scale,
        k_scale,
        scores,
        pages,
        cu_q,
        seq_lens,
        prefix_lens,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        pages.stride(0),
        num_kv_heads,
        k.shape[1],
        vec,
        128,
        decode_query_len,
        init_blocks,
        local_blocks,
        block_pages,
        num_warps=4,
        num_stages=1,
    )
    return scores


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    *,
    fp4_format: str | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.

    Optional FP4 inputs use packed E2M1 uint8 values and logical per-vector
    scales. ``fp4_format`` selects MXFP4 (E8M0/32) or NVFP4 (E4M3/16).
    Existing BF16/FP8 calls and their layouts are unchanged.
    """
    if fp4_format is not None:
        return fp4_index_scores(
            idx_q,
            index_kv_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            max_seq_len,
            num_kv_heads,
            fp4_format=fp4_format,
            q_scale=q_scale,
            k_scale=k_scale,
        )
    if (
        idx_q.dtype in (torch.uint8, torch.int8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
        or q_scale is not None
        or k_scale is not None
    ):
        raise ValueError("Packed FP4 inputs/scales require explicit fp4_format")
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    if (
        head_dim == 128
        and idx_q.stride(-1) == 1
        and index_kv_cache.stride(-1) == 1
        and idx_q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
    ):
        if is_sm100_device(idx_q.device) and use_page_parallel_scores(
            idx_q, max_query_len, cu_seqlens_q.numel() - 1, num_kv_heads
        ):
            return dense_index_scores(
                idx_q,
                index_kv_cache,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                max_query_len,
                max_seq_len,
                num_kv_heads,
            )
    return generic.minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        max_query_len,
        max_seq_len,
        num_kv_heads,
    )


@torch.no_grad()
def minimax_m3_index_decode_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor | None = None,
    *,
    fp4_format: str | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score (split-K, cudagraph-safe); no top-k.

    Returns score [num_kv_heads, total_q, >=max_block] (fp32; init/local blocks
    forced to 1e30/1e29). When ``score_out`` is given the scores are written into
    it (read/written by strides, so a transposed view of a unified buffer is
    accepted) instead of a fresh tensor -- used to share a unified score buffer
    with the prefill side and run a single top-k over both.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    if fp4_format is not None:
        return fp4_index_scores(
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            seq_lens,
            seq_lens,
            decode_query_len,
            max_seq_len,
            num_kv_heads,
            fp4_format=fp4_format,
            q_scale=q_scale,
            k_scale=k_scale,
            decode_query_len=decode_query_len,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            score_out=score_out,
        )
    if (
        idx_q.dtype in (torch.uint8, torch.int8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
        or q_scale is not None
        or k_scale is not None
    ):
        raise ValueError("Packed FP4 inputs/scales require explicit fp4_format")
    if not is_sm100_device(idx_q.device):
        return generic.minimax_m3_index_decode_score(
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
            init_blocks,
            local_blocks,
            num_kv_heads,
            decode_query_len,
            max_decode_query_len,
            score_out=score_out,
        )
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = current_platform.is_arch_support_pdl()
    if score_out is not None:
        score = score_out
    else:
        # Keep score strides 16-divisible to avoid Triton recompiles.
        score_block_stride = round_up(max_block, 16)
        score = torch.empty(
            (num_idx_heads, total_q, score_block_stride),
            dtype=torch.float32,
            device=idx_q.device,
        )
    # The configured maximum query length fixes codegen across decode lengths.
    grid, tail, options = _decode_score_launch_config(
        idx_q, index_kv_cache, block_table, seq_lens, max_seq_len,
        init_blocks, local_blocks, num_idx_heads, decode_query_len,
        max_decode_query_len, score.stride(), use_pdl,
    )
    _decode_index_score_kernel[grid](
        idx_q, index_kv_cache, score, block_table, seq_lens, *tail, **options,
    )
    return score


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
    *,
    fp4_format: str | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score + dispatched top-k (cudagraph-safe).

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` (stable address for cudagraph) instead of allocating.
    When ``score_out`` ([num_kv_heads, total_q, >=max_block]) is given, the block
    scores are written into it (read back by the top-k) instead of a fresh
    tensor -- used to share a unified score buffer with the prefill side. Reads
    via strides, so a transposed view of a block-major buffer is accepted.
    """
    if fp4_format is not None and fp4_format not in ("mxfp4", "nvfp4"):
        raise ValueError("fp4_format must be 'mxfp4' or 'nvfp4'")
    if fp4_format is None and (
        idx_q.dtype in (torch.uint8, torch.int8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))
        or q_scale is not None
        or k_scale is not None
    ):
        raise ValueError("Packed FP4 inputs/scales require explicit fp4_format")
    if fp4_format is not None:
        if decode_query_len < 1:
            raise ValueError("decode_query_len must be positive")
        # The scoring entry validates ordinary calls. Only identity-without-
        # scores bypasses it; validate that shortcut here, once per call.
        if score_out is None and triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE) <= topk:
            idx_q, index_kv_cache, q_scale, k_scale, _ = _prepare_fp4_inputs(
                idx_q, index_kv_cache, num_kv_heads, fp4_format, q_scale, k_scale
            )
            _validate_fp4_metadata(idx_q, block_table, None, seq_lens, None,
                                   max_decode_query_len, max_seq_len, decode_query_len)
    total_q, num_idx_heads, _ = idx_q.shape
    assert num_idx_heads == num_kv_heads, "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    if (
        fp4_format is None
        and out is None
        and score_out is None
        and idx_q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and topk < max_block <= _DECODE_SINGLE_PASS_TOPK_MAX_BLOCKS
        and _can_reuse_launch(_decode_index_score_kernel, _decode_topk_single_pass_kernel)
        and all(type(t) is torch.Tensor for t in (idx_q, index_kv_cache, block_table, seq_lens))
    ):
        selected = _decode_index_planned(
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
            topk,
            init_blocks,
            local_blocks,
            num_kv_heads,
            decode_query_len,
            max_decode_query_len,
        )
        if selected is not None:
            return selected
    if fp4_format is None and not is_sm100_device(idx_q.device):
        return generic.minimax_m3_index_decode(
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
            topk,
            init_blocks,
            local_blocks,
            num_kv_heads,
            decode_query_len,
            max_decode_query_len,
            out=out,
            score_out=score_out,
        )
    use_pdl = current_platform.is_arch_support_pdl() and fp4_format is None
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})

    block_size_t = triton.next_power_of_2(topk)
    if max_block <= topk:
        # Every visible block is selected, so scores cannot affect the result.
        # Preserve the documented score_out mutation when the caller supplies
        # that buffer; otherwise skip both score generation and Top-K sorting.
        if out is not None:
            topk_idx = out[:, :total_q, :]
        else:
            topk_idx = torch.empty(
                (num_idx_heads, total_q, topk),
                dtype=torch.int32,
                device=idx_q.device,
            )
        if score_out is not None:
            minimax_m3_index_decode_score(
                idx_q,
                index_kv_cache,
                block_table,
                seq_lens,
                max_seq_len,
                init_blocks,
                local_blocks,
                num_kv_heads,
                decode_query_len,
                max_decode_query_len,
                score_out=score_out,
                fp4_format=fp4_format,
                q_scale=q_scale,
                k_scale=k_scale,
            )
        identity_use_pdl = use_pdl and score_out is not None
        identity_pdl_kwargs = pdl_kwargs if identity_use_pdl else {}
        _decode_topk_identity_kernel[(batch, num_idx_heads)](
            topk_idx,
            seq_lens,
            SPARSE_BLOCK_SIZE,
            topk,
            decode_query_len,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            BLOCK_SIZE_T=block_size_t,
            USE_PDL=identity_use_pdl,
            **identity_pdl_kwargs,
        )
        return topk_idx

    score = minimax_m3_index_decode_score(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        score_out=score_out,
        fp4_format=fp4_format,
        q_scale=q_scale,
        k_scale=k_scale,
    )

    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=idx_q.device,
        )

    if (
        fp4_format is None
        and idx_q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and max_block <= _DECODE_SINGLE_PASS_TOPK_MAX_BLOCKS
    ):
        _decode_topk_single_pass_kernel[(total_q, num_idx_heads)](
            score,
            topk_idx,
            seq_lens,
            *score.stride(),
            *topk_idx.stride(),
            decode_query_len,
            max_block,
            topk,
            triton.next_power_of_2(max_block),
            use_pdl,
            num_warps=4,
            **pdl_kwargs,
        )
        return topk_idx

    # Chunk count is shape-constant (cudagraph-safe), capped so the merge sorts
    # pow2(num_topk_chunks * pow2(topk)) candidates.
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * num_idx_heads)))
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    topk_score_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.float32,
        device=idx_q.device,
    )
    topk_idx_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.int32,
        device=idx_q.device,
    )
    _topk_index_partial_kernel[(batch, num_idx_heads, num_topk_chunks)](
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        USE_PDL=use_pdl,
        **pdl_kwargs,
    )
    _topk_index_merge_kernel[(batch, num_idx_heads)](
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_topk_chunks=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_kwargs,
    )
    return topk_idx
