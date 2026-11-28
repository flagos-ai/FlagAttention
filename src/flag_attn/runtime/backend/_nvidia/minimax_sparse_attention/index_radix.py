# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA shared-memory radix TopK with a standard Triton fallback."""

import torch
import triton
import triton.language as tl
from triton.errors import TritonError
from flag_attn.minimax_sparse_attention.index_topk import (
    SPARSE_BLOCK_SIZE,
    _streaming_topk_fpval_to_key,
    _load_prefill_topk_score,
    _supports_prefill_selector_input,
    _can_use_streaming_prefill_topk,
    _topk_tile_num_warps,
    _streaming_prefill_launch_config,
    _topk_index_kernel_streaming,
    _topk_index_kernel_fallback,
)
from .dispatch import force_generic

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None
_HAS_TLE_RADIX = tle is not None and all(
    hasattr(getattr(tle, "gpu", None), name) for name in ("alloc", "local_ptr", "smem")
)

_RADIX_BITS = 4


_RADIX_MIN_PADDED_BLOCKS = 1024


_RADIX_MAX_TOPK = 64


_RADIX_MIN_BLOCKS_PER_TOPK = 16


_FAILED_RADIX_CONFIGS: set[tuple[int, int]] = set()


@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel_radix_tle(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    RADIX_BITS: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(RADIX_BITS == 4)
    tl.static_assert(BLOCK_SIZE_T >= topk)
    RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
    RADIX_MASK: tl.constexpr = RADIX_SIZE - 1

    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return

    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    off_t = tl.arange(0, BLOCK_SIZE_T)
    output_row = ti_ptr + (block_start + pid_q) * stride_ti_n + pid_h * stride_ti_h

    # Per-row Identity path.  It is particularly important for early
    # Prefill queries, whose causal range has not yet grown beyond K.
    if valid_blocks <= topk:
        topk_idx = tl.where(off_t < valid_blocks, off_t, -1)
        tl.store(
            output_row + off_t * stride_ti_t,
            topk_idx.to(ti_ptr.dtype.element_ty),
            mask=off_t < topk,
        )
        return

    score_row = (
        s_ptr + (seq_start + pid_q * sample_interval) * stride_s_n + pid_h * stride_s_h
    )
    lane = tl.arange(0, BLOCK_SIZE_K)
    bins = tl.arange(0, RADIX_SIZE)
    one = tl.full([BLOCK_SIZE_K], 1, tl.int32)
    n_tiles = tl.cdiv(valid_blocks, BLOCK_SIZE_K)

    # The histogram is CTA-local, exactly as in the TLE tutorial.
    smem_counts = tle.gpu.alloc(
        [RADIX_SIZE],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_count_ptrs = tle.gpu.local_ptr(smem_counts, (bins,))

    desired = tl.full((), 0, dtype=tl.uint32)
    desired_mask = tl.full((), 0, dtype=tl.uint32)
    k_to_find = tl.full((), topk, dtype=tl.int32)
    radix_mask_u32 = tl.full((), RADIX_MASK, dtype=tl.uint32)

    # Eight 4-bit MSD passes locate the exact FP32 key at the K boundary.
    for digit_pos in tl.static_range(32 - RADIX_BITS, -1, -RADIX_BITS):
        tl.store(
            smem_count_ptrs,
            tl.zeros([RADIX_SIZE], dtype=tl.int32),
        )
        tl.debug_barrier()
        for tile in tl.range(0, n_tiles):
            block_idx = tile * BLOCK_SIZE_K + lane
            score, valid = _load_prefill_topk_score(
                score_row,
                block_idx,
                valid_blocks,
                init_blocks,
                local_blocks,
                stride_s_k,
                MASK_INIT,
                MASK_LOCAL,
            )
            score_key = _streaming_topk_fpval_to_key(score.to(tl.uint32, bitcast=True))
            matches = (score_key & desired_mask) == desired
            digit = ((score_key >> digit_pos) & RADIX_MASK).to(tl.int32)
            count_ptrs = tle.gpu.local_ptr(smem_counts, (digit,))
            tl.atomic_add(
                count_ptrs,
                one,
                mask=valid & matches,
                sem="relaxed",
                scope="cta",
            )
        tl.debug_barrier()

        counts = tl.load(smem_count_ptrs)
        cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
        selected_mask = cumsum_desc >= k_to_find
        selected = tl.max(tl.where(selected_mask, bins, 0), axis=0).to(tl.int32)
        counts_gt = tl.max(tl.where(bins == selected + 1, cumsum_desc, 0), axis=0)
        selected_u32 = selected.to(tl.uint32)
        desired = desired | (selected_u32 << digit_pos)
        desired_mask = desired_mask | (radix_mask_u32 << digit_pos)
        k_to_find = k_to_find - counts_gt

    # At this point ``desired`` is the exact threshold key and
    # ``k_to_find`` is how many threshold-equal elements are still needed.
    # Scan once in ascending block-id order, emitting all greater keys and
    # the first equal keys.  This combines the tutorial's two collection
    # passes and makes ties deterministic.
    written = tl.full((), 0, dtype=tl.int32)
    equal_seen = tl.full((), 0, dtype=tl.int32)
    for tile in tl.range(0, n_tiles):
        block_idx = tile * BLOCK_SIZE_K + lane
        score, valid = _load_prefill_topk_score(
            score_row,
            block_idx,
            valid_blocks,
            init_blocks,
            local_blocks,
            stride_s_k,
            MASK_INIT,
            MASK_LOCAL,
        )
        score_key = _streaming_topk_fpval_to_key(score.to(tl.uint32, bitcast=True))
        take_gt = valid & (score_key > desired)
        take_equal = valid & (score_key == desired)
        equal_rank = tl.cumsum(take_equal.to(tl.int32), axis=0)
        take_equal = take_equal & (equal_seen + equal_rank <= k_to_find)
        take = take_gt | take_equal
        take_rank = tl.cumsum(take.to(tl.int32), axis=0)
        out_pos = written + take_rank - 1
        tl.store(
            output_row + out_pos * stride_ti_t,
            block_idx.to(ti_ptr.dtype.element_ty),
            mask=take,
        )
        written = written + tl.sum(take.to(tl.int32), axis=0)
        equal_seen = equal_seen + tl.sum(
            (valid & (score_key == desired)).to(tl.int32), axis=0
        )


def _radix_prefill_launch_config(max_score_blocks: int) -> tuple[int, int]:
    """Choose Radix resources from MSA's padded max-block dimension."""
    block_size = max(
        32,
        triton.next_power_of_2(min(max_score_blocks, 1024)),
    )
    return block_size, _topk_tile_num_warps(block_size)


def _can_use_radix_prefill_topk(
    score: torch.Tensor,
    topk: int,
    max_query_len: int,
) -> bool:
    """Use Radix for a short Prefill chunk over a wide existing context."""
    if force_generic() or not _HAS_TLE_RADIX:
        return False
    if not _supports_prefill_selector_input(score, topk):
        return False
    max_score_blocks = score.shape[2]
    block_size_k, _ = _radix_prefill_launch_config(max_score_blocks)
    return (
        # A chunk no longer than one sparse page keeps per-row valid_blocks
        # nearly constant.  Full Prefill has N growing from 1 to max_block and
        # is better served by the single-pass Streaming selector.
        0 < max_query_len <= SPARSE_BLOCK_SIZE
        and max_score_blocks >= _RADIX_MIN_PADDED_BLOCKS
        and topk <= _RADIX_MAX_TOPK
        and max_score_blocks >= _RADIX_MIN_BLOCKS_PER_TOPK * topk
        and (block_size_k, topk) not in _FAILED_RADIX_CONFIGS
    )


def _select_prefill_topk_path(
    score: torch.Tensor,
    topk: int,
    max_query_len: int,
) -> str:
    """Choose an algorithm; correctness support and performance policy differ."""
    if _can_use_radix_prefill_topk(score, topk, max_query_len):
        return "radix"
    if _can_use_streaming_prefill_topk(score, topk):
        return "streaming"
    return "fallback"


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor.

    Dispatches short Prefill chunks over wide contexts to TLE Radix Select,
    full/medium Prefill to packed-key Streaming Top-K, and unsupported inputs to
    the original Bitonic fallback.  Optimized kernels contain a per-row N <= K
    identity path.

    When ``out`` is provided (a ``[num_idx_heads, >=total_q, topk]`` buffer), the
    result is written into ``out[:, :total_q, :]`` instead of a fresh tensor --
    used to keep the top-k output at a stable address for cudagraph capture.
    """
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    kernel_args = (
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
    )
    path = _select_prefill_topk_path(score, topk, max_query_len)
    if path == "radix":
        assert _topk_index_kernel_radix_tle is not None
        block_size_k, num_warps = _radix_prefill_launch_config(score.shape[2])
        try:
            _topk_index_kernel_radix_tle[grid_topk](
                *kernel_args,
                BLOCK_SIZE_K=block_size_k,
                BLOCK_SIZE_T=triton.next_power_of_2(topk),
                RADIX_BITS=_RADIX_BITS,
                MASK_INIT=False,
                MASK_LOCAL=False,
                num_warps=num_warps,
                num_stages=1,
            )
            return topk_idx
        except TritonError:
            # Some CUDA/Triton combinations expose the TLE module but cannot
            # lower this shared-memory kernel.  Remember the failed compile
            # configuration so later calls do not repeatedly pay that cost.
            _FAILED_RADIX_CONFIGS.add((block_size_k, topk))
            path = (
                "streaming"
                if _can_use_streaming_prefill_topk(score, topk)
                else "fallback"
            )

    if path == "streaming":
        block_size_k, num_warps = _streaming_prefill_launch_config(score.shape[2], topk)
        _topk_index_kernel_streaming[grid_topk](
            *kernel_args,
            BLOCK_SIZE_K=block_size_k,
            MASK_INIT=False,
            MASK_LOCAL=False,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        _topk_index_kernel_fallback[grid_topk](
            *kernel_args,
            MASK_INIT=False,
            MASK_LOCAL=False,
        )
    return topk_idx
