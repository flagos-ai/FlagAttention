# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend M3 indexer: flattened scoring and reduction-based block selection."""
import torch
import triton
import triton.language as tl
from triton.errors import TritonError
from flag_attn.minimax_sparse_attention.index_topk import (
    _index_block_score_kernel,
    _decode_topk_identity_kernel,
)

SPARSE_BLOCK_SIZE = 128
_NPU_PREFILL_MAX_PROGRAMS = 32768


def round_up(value, multiple):
    return triton.cdiv(value, multiple) * multiple


@triton.jit
def _select_topk_pair_to_ptr(
    score,
    index,
    active,
    score_ptr,
    index_ptr,
    score_stride,
    index_stride,
    topk: tl.constexpr,
    block_size_t: tl.constexpr,
):
    """Select score/index pairs without sub-vector permutations.

    Ascend's vector core requires aligned UB accesses for vectorized
    permutation instructions.  The usual hypercube bitonic implementation
    creates short, non-contiguous subvectors (for example ``[2, 2, ...]``),
    which can lower to an unaligned vector access.  This reduction-based
    selector keeps every reduction on the original contiguous lane vector and
    writes each result through a scalar pointer offset.
    """
    for rank in tl.static_range(0, block_size_t):
        if rank < topk:
            current = tl.where(active, score, -1e30)
            max_score = tl.max(current, axis=0)
            candidate = active & (score == max_score)
            candidate_index = tl.where(candidate, -index, -1e30)
            selected_index = -tl.max(candidate_index, axis=0)
            selected = candidate & (index == selected_index)
            found = tl.max(selected.to(tl.float32), axis=0) > 0.0
            output_score = tl.where(found, max_score, -1e30)
            output_index = tl.where(found, selected_index, 0.0).to(tl.int32)
            active = active & ~selected
        else:
            output_score = -1e30
            output_index = 0
        tl.store(score_ptr + rank * score_stride, output_score)
        tl.store(index_ptr + rank * index_stride, output_index)


@triton.jit
def _select_topk_index_to_ptr(
    score,
    index,
    active,
    index_ptr,
    index_stride,
    topk: tl.constexpr,
    block_size_t: tl.constexpr,
):
    for rank in tl.static_range(0, block_size_t):
        if rank < topk:
            current = tl.where(active, score, -1e30)
            max_score = tl.max(current, axis=0)
            candidate = active & (score == max_score)
            candidate_index = tl.where(candidate, -index, -1e30)
            selected_index = -tl.max(candidate_index, axis=0)
            selected = candidate & (index == selected_index)
            found = tl.max(selected.to(tl.float32), axis=0) > 0.0
            output_index = tl.where(
                found,
                selected_index - 1.0,
                -1.0,
            ).to(tl.int32)
            active = active & ~selected
            tl.store(index_ptr + rank * index_stride, output_index)


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel_npu(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    """NPU Prefill score kernel using the official MSA row tiling.

    The official kernel treats ``(token, head)`` as one flattened M
    dimension.  Keeping those rows in one program makes a K page reusable
    across index heads.  The M tile is kept at 64 here because a 128x128
    Triton tile exceeds the 910B UB once the compiler's local buffers are
    included.
    The public score layout remains head-major, so only the address arithmetic
    differs from the legacy per-head kernel.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    rows = q_len * num_idx_heads
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < rows
    if row_start >= rows:
        return

    token_offsets = row_offsets // num_idx_heads
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n,
        shape=(rows, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(row_start, 0),
        block_shape=(BLOCK_SIZE_M, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")

    # Only pages through the last valid token in this M tile are needed.
    last_row = tl.minimum(row_start + BLOCK_SIZE_M - 1, rows - 1)
    last_token = last_row // num_idx_heads
    hi = tl.minimum(seq_len, prefix_len + last_token + 1)
    q_start = prefix_len + row_start // num_idx_heads
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_positions = prefix_len + token_offsets

    for i in tl.range(0, hi, BLOCK_SIZE_K, num_stages=2):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k)
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(
                row_mask[:, None] & (q_positions[:, None] >= pos[None, :]),
                qk,
                float("-inf"),
            )
        score = tl.max(qk, axis=1)
        s_ptrs = (
            score_ptr
            + (row_offsets - token_offsets * num_idx_heads) * stride_s_h
            + (seq_start + token_offsets) * stride_s_n
            + blk * stride_s_k
        )
        tl.store(s_ptrs, score, mask=row_mask)


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel_npu_m128_split(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """NPU M128 schedule with two UB-sized M64 matrix operations.

    A single 128x128 Triton dot exceeds the 910B UB.  Splitting M into two
    64-row dots preserves the official 128-row task granularity while keeping
    the K page loaded only once for both halves.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    rows = q_len * num_idx_heads
    row_start = pid_m * BLOCK_SIZE_M
    if row_start >= rows:
        return

    half_m: tl.constexpr = BLOCK_SIZE_M // 2
    row0 = row_start + tl.arange(0, half_m)
    row1 = row0 + half_m
    mask0 = row0 < rows
    mask1 = row1 < rows
    token0 = row0 // num_idx_heads
    token1 = row1 // num_idx_heads
    qbase = q_ptr + seq_start * stride_q_n
    q0_ptrs = tl.make_block_ptr(
        base=qbase,
        shape=(rows, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(row_start, 0),
        block_shape=(half_m, head_dim),
        order=(1, 0),
    )
    q1_ptrs = tl.make_block_ptr(
        base=qbase,
        shape=(rows, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(row_start + half_m, 0),
        block_shape=(half_m, head_dim),
        order=(1, 0),
    )
    q0 = tl.load(q0_ptrs, boundary_check=(0,), padding_option="zero")
    q1 = tl.load(q1_ptrs, boundary_check=(0,), padding_option="zero")
    qpos0 = prefix_len + token0
    qpos1 = prefix_len + token1
    qstart = prefix_len + row_start // num_idx_heads
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    hi = tl.minimum(
        seq_len,
        prefix_len + (row_start + BLOCK_SIZE_M - 1) // num_idx_heads + 1,
    )
    for i in tl.range(0, hi, BLOCK_SIZE_K, num_stages=1):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk0 = tl.dot(q0, k)
        if qstart < i + BLOCK_SIZE_K:
            qk0 = tl.where(
                mask0[:, None] & (qpos0[:, None] >= pos[None, :]),
                qk0,
                float("-inf"),
            )
        score0 = tl.max(qk0, axis=1)
        flat_base = seq_start * num_idx_heads
        s0 = score_ptr + blk * stride_s_k + flat_base + row0
        tl.store(s0, score0, mask=mask0)
        if row_start + half_m < rows:
            qk1 = tl.dot(q1, k)
            if qstart < i + BLOCK_SIZE_K:
                qk1 = tl.where(
                    mask1[:, None] & (qpos1[:, None] >= pos[None, :]),
                    qk1,
                    float("-inf"),
                )
            score1 = tl.max(qk1, axis=1)
            s1 = score_ptr + blk * stride_s_k + flat_base + row1
            tl.store(s1, score1, mask=mask1)


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(
    do_not_specialize=["query_start"],
    do_not_specialize_on_alignment=["prefix_lens"],
)
def _topk_index_kernel_fallback_npu(
    s_ptr,
    ti_ptr,
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
    score_capacity,
    query_start,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0) + query_start
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return

    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    score_capacity = tl.maximum(score_capacity, 1)
    score_row = (
        s_ptr + (seq_start + pid_q * sample_interval) * stride_s_n + pid_h * stride_s_h
    )
    off = tl.arange(0, BLOCK_SIZE_K)
    score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    index = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    local_start = tl.maximum(0, valid_blocks - local_blocks)
    for lane in tl.static_range(0, BLOCK_SIZE_K):
        lane_valid = lane < valid_blocks
        safe_lane = tl.minimum(lane, score_capacity - 1)
        lane_score = tl.load(
            score_row + safe_lane * stride_s_k,
            mask=lane_valid,
            other=-1e30,
        ).to(tl.float32)
        lane_score = tl.where(lane_score != lane_score, -1e30, lane_score)
        init_mask = lane < init_blocks
        local_mask = lane >= local_start
        if MASK_INIT:
            lane_score = tl.where(
                lane_valid & init_mask,
                lane_score - 1e29,
                lane_score,
            )
        else:
            lane_score = tl.where(
                lane_valid & init_mask,
                1e30,
                lane_score,
            )
        if MASK_LOCAL:
            lane_score = tl.where(
                lane_valid & local_mask,
                lane_score - 1e28,
                lane_score,
            )
        else:
            lane_score = tl.where(
                lane_valid & local_mask,
                1e29,
                lane_score,
            )
        score = tl.where(off == lane, lane_score, score)
        index = tl.where(
            off == lane,
            tl.where(lane_valid, lane + 1.0, 0.0),
            index,
        )

    active = (index > 0.0) & (index <= valid_blocks)
    output_base = ti_ptr + (block_start + pid_q) * stride_ti_n + pid_h * stride_ti_h
    _select_topk_index_to_ptr(
        score,
        index,
        active,
        output_base,
        stride_ti_t,
        topk,
        BLOCK_SIZE_T,
    )


@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_Q: tl.constexpr,
    num_kv_chunks,
    HEADS_PER_PROGRAM: tl.constexpr,
    IS_NPU: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = HEADS_PER_PROGRAM * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    if IS_NPU:
        h_offsets = pid_h * HEADS_PER_PROGRAM + hq_offsets % HEADS_PER_PROGRAM
        q_offsets = hq_offsets // HEADS_PER_PROGRAM
    else:
        h_offsets = pid_h * HEADS_PER_PROGRAM + hq_offsets // BLOCK_SIZE_Q
        q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = (q_offsets < decode_query_len) & (h_offsets < num_idx_heads)
    q_ids = pid_r * decode_query_len + q_offsets

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + pid_r)
    query_pos = seq_len - decode_query_len + q_offsets
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks_q = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (kv_len_max + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # block-aligned fixed-count split: grid independent of seq_len (cuda graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_r * stride_bt_b
    # Force-select init (1e30) and local (1e29, higher priority) blocks.
    local_start = tl.maximum(0, num_blocks_q - local_blocks)
    if IS_NPU:
        q = tl.load(
            q_ptr
            + q_ids[:, None] * stride_q_n
            + h_offsets[:, None] * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None],
            other=0.0,
        )
    else:
        q = tl.load(
            q_ptr
            + q_ids[None, :] * stride_q_n
            + h_offsets[None, :] * stride_q_h
            + off_d[:, None] * stride_q_d,
            mask=q_mask[None, :],
            other=0.0,
        )
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        if IS_NPU:
            pos_mask = pos[None, :] < kv_len[:, None]
            k = tl.load(
                ik_cache_ptr
                + page * stride_ik_blk
                + off_k[:, None] * stride_ik_pos
                + off_d[None, :] * stride_ik_d,
            )
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            qk = tl.where(pos_mask & q_mask[:, None], qk, float("-inf"))
            score = tl.max(qk, axis=1)
        else:
            pos_mask = pos[:, None] < kv_len[None, :]
            # we don't need masked load for K, because KV cache ensures
            # allocation is multiple of BLOCK_SIZE_K.
            # for tokens beyond seqlen, they will be masked in qk later.
            k = tl.load(
                ik_cache_ptr
                + page * stride_ik_blk
                + off_k[:, None] * stride_ik_pos
                + off_d * stride_ik_d,
            )
            # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k are
            # loaded in their stored dtype (bf16 or e4m3) and the MMA accumulates in
            # fp32 so the per-block max score is exact for the fp8 indexer too.
            kq = tl.dot(k, q, out_dtype=tl.float32)
            kq = tl.where(pos_mask & q_mask[None, :], kq, float("-inf"))
            score = tl.max(kq, axis=0)
        is_visible_block = blk < num_blocks_q
        is_init = (blk < init_blocks) & is_visible_block
        is_local = (blk >= local_start) & is_visible_block
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + h_offsets * stride_s_h + q_ids * stride_s_n + blk * stride_s_k,
            score,
            mask=q_mask,
        )


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _topk_index_partial_kernel_npu(
    s_ptr,
    ts_partial_ptr,
    ti_partial_ptr,
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    chunk_blocks,
    decode_query_len,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    score_row = s_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    aligned_start = (chunk_start // 8) * 8
    block_idx = aligned_start + off_k
    valid = (block_idx >= chunk_start) & (block_idx < chunk_end)
    score_capacity = tl.maximum(stride_s_b // stride_s_k, 1)
    safe_block_idx = tl.minimum(block_idx, score_capacity - 1)
    score = tl.load(
        score_row + safe_block_idx * stride_s_k,
        mask=valid,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)
    index = tl.where(valid, block_idx.to(tl.float32) + 1.0, 0.0)

    ts_base = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
    )
    ti_base = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
    )
    _select_topk_pair_to_ptr(
        score,
        index,
        valid,
        ts_base,
        ti_base,
        stride_ts_t,
        stride_ti_t,
        topk,
        BLOCK_SIZE_T,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit(do_not_specialize=["num_topk_chunks", "decode_query_len"])
def _topk_index_merge_kernel_npu(
    ts_partial_ptr,
    ti_partial_ptr,
    ti_final_ptr,
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    decode_query_len,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off = tl.arange(0, BLOCK_SIZE_K)
    score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    index = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    for lane in tl.static_range(0, BLOCK_SIZE_K):
        chunk_idx = lane // BLOCK_SIZE_T
        in_chunk_idx = lane % BLOCK_SIZE_T
        lane_valid = chunk_idx < num_topk_chunks
        score_value = tl.load(
            ts_partial_ptr
            + chunk_idx * stride_ts_c
            + pid_h * stride_ts_h
            + pid_b * stride_ts_b
            + in_chunk_idx * stride_ts_t,
            mask=lane_valid,
            other=-1e30,
        ).to(tl.float32)
        index_value = tl.load(
            ti_partial_ptr
            + chunk_idx * stride_ti_c
            + pid_h * stride_ti_h
            + pid_b * stride_ti_b
            + in_chunk_idx * stride_ti_t,
            mask=lane_valid,
            other=0,
        ).to(tl.float32)
        score = tl.where(off == lane, score_value, score)
        index = tl.where(off == lane, index_value, index)
    score = tl.where(score != score, -1e30, score)

    active = (index > 0.0) & (index <= num_blocks)
    tif_base = ti_final_ptr + pid_h * stride_tif_h + pid_b * stride_tif_b
    _select_topk_index_to_ptr(
        score,
        index,
        active,
        tif_base,
        stride_tif_t,
        topk,
        BLOCK_SIZE_T,
    )


def _topk_tile_num_warps(block_size: int) -> int:
    if block_size <= 64:
        return 2
    if block_size <= 128:
        return 4
    return 8


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    if idx_q.device.type != "npu":
        raise ValueError("Ascend MSA requires NPU tensors")
    (total_q, num_idx_heads, head_dim) = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    score_block_stride = round_up(max_block, 16)
    use_npu_flat_score = True and num_idx_heads > 1 and idx_q.is_contiguous()
    if use_npu_flat_score:
        score_storage = torch.empty(
            (score_block_stride, total_q * num_idx_heads),
            dtype=torch.float32,
            device=idx_q.device,
        )
        score = torch.as_strided(
            score_storage,
            (num_idx_heads, total_q, score_block_stride),
            (1, num_idx_heads, total_q * num_idx_heads),
        )
    else:
        score = torch.empty(
            (num_idx_heads, total_q, score_block_stride),
            dtype=torch.float32,
            device=idx_q.device,
        )
    if use_npu_flat_score:
        block_size_m = 128
        grid_score = (triton.cdiv(max_query_len * num_idx_heads, block_size_m), batch)
        try:
            _index_block_score_kernel_npu_m128_split[grid_score](
                idx_q,
                index_kv_cache,
                score_storage,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                num_idx_heads,
                head_dim,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                index_kv_cache.stride(0),
                index_kv_cache.stride(1),
                index_kv_cache.stride(2),
                score_storage.stride(0),
                block_table.stride(0),
                BLOCK_SIZE_M=block_size_m,
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                num_warps=8,
                num_stages=1,
            )
        except TritonError:
            fallback_m = 64
            fallback_grid = (
                triton.cdiv(max_query_len * num_idx_heads, fallback_m),
                batch,
            )
            _index_block_score_kernel_npu[fallback_grid](
                idx_q,
                index_kv_cache,
                score,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                num_idx_heads,
                head_dim,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                index_kv_cache.stride(0),
                index_kv_cache.stride(1),
                index_kv_cache.stride(2),
                score.stride(0),
                score.stride(1),
                score.stride(2),
                block_table.stride(0),
                BLOCK_SIZE_M=fallback_m,
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                num_warps=4,
                num_stages=1,
            )
    else:
        block_size_q = 64
        grid_score = (triton.cdiv(max_query_len, block_size_q), batch * num_idx_heads)
        _index_block_score_kernel[grid_score](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_idx_heads,
            head_dim,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=block_size_q,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        )
    return score


@torch.no_grad()
def minimax_m3_index_decode_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score (split-K, cudagraph-safe); no top-k.

    Returns score [num_kv_heads, total_q, >=max_block] (fp32; init/local blocks
    forced to 1e30/1e29). When ``score_out`` is given the scores are written into
    it (read/written by strides, so a transposed view of a unified buffer is
    accepted) instead of a fresh tensor -- used to share a unified score buffer
    with the prefill side and run a single top-k over both.
    """
    if idx_q.device.type != "npu":
        raise ValueError("Ascend MSA requires NPU tensors")
    (total_q, num_idx_heads, head_dim) = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = False
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})
    score_kwargs = pdl_kwargs.copy()
    score_kwargs.update({"num_stages": 1})
    if score_out is not None:
        score = score_out
    else:
        score_block_stride = round_up(max_block, 16)
        score = torch.empty(
            (num_idx_heads, total_q, score_block_stride),
            dtype=torch.float32,
            device=idx_q.device,
        )
    MAX_NUM_KV_CHUNKS = 256
    BLOCK_SIZE_Q = triton.next_power_of_2(max_decode_query_len)
    block_size_per_chunk = 32
    target_chunks = max(1, triton.cdiv(max_block, block_size_per_chunk))
    num_kv_chunks = min(MAX_NUM_KV_CHUNKS, 1 << (target_chunks - 1).bit_length())
    max_query_head_columns = 64
    heads_per_program = min(
        num_idx_heads, max(1, max_query_head_columns // BLOCK_SIZE_Q)
    )
    # The 910B compiler produces incorrect row reductions for the M16 decode
    # tile (verified with both two and four warps). Pad to M32; masks retain
    # the same logical queries and the launch grid remains unchanged.
    if heads_per_program * BLOCK_SIZE_Q == 16:
        BLOCK_SIZE_Q *= 2
    score_kwargs["num_warps"] = 4 if heads_per_program * BLOCK_SIZE_Q > 16 else 2
    head_programs = triton.cdiv(num_idx_heads, heads_per_program)
    grid_score = (seq_lens.shape[0], num_kv_chunks, head_programs)
    _decode_index_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        num_kv_chunks=num_kv_chunks,
        HEADS_PER_PROGRAM=heads_per_program,
        IS_NPU=True,
        USE_PDL=use_pdl,
        **score_kwargs,
    )
    return score


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
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
    if idx_q.device.type != "npu":
        raise ValueError("Ascend MSA requires NPU tensors")
    (total_q, num_idx_heads, _) = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = False
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})
    block_size_t = triton.next_power_of_2(topk)
    if max_block <= topk:
        if out is not None:
            topk_idx = out[:, :total_q, :]
        else:
            topk_idx = torch.empty(
                (num_idx_heads, total_q, topk), dtype=torch.int32, device=idx_q.device
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
            )
        identity_use_pdl = use_pdl and score_out is not None
        identity_pdl_kwargs = pdl_kwargs if identity_use_pdl else {}
        _decode_topk_identity_kernel[batch, num_idx_heads](
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
    )
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk), dtype=torch.int32, device=idx_q.device
        )
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * num_idx_heads))
    )
    num_topk_chunks = 1 << topk_target.bit_length() - 1
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
    partial_args = (
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
    )
    partial_grid = (batch, num_idx_heads, num_topk_chunks)
    partial_block_size = max(
        64, triton.next_power_of_2(chunk_blocks + 8), triton.next_power_of_2(topk + 1)
    )
    _topk_index_partial_kernel_npu[partial_grid](
        *partial_args,
        BLOCK_SIZE_K=partial_block_size,
        num_warps=_topk_tile_num_warps(partial_block_size),
        num_stages=1,
    )
    merge_args = (
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
    )
    merge_grid = (batch, num_idx_heads)
    _topk_index_merge_kernel_npu[merge_grid](
        *merge_args,
        num_topk_chunks=num_topk_chunks,
        num_warps=_topk_tile_num_warps(
            triton.next_power_of_2(num_topk_chunks * block_size_t)
        ),
        num_stages=1,
    )
    return topk_idx


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
    """Select blocks with the Ascend reduction kernel and bounded grids."""
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
    fallback_block_size = max(
        64,
        triton.next_power_of_2(score.shape[2]),
        triton.next_power_of_2(topk + 1),
    )
    fixed_grid_size = batch * num_idx_heads
    max_query_programs = max(
        1,
        _NPU_PREFILL_MAX_PROGRAMS // fixed_grid_size,
    )
    for query_start in range(0, max_query_len, max_query_programs):
        query_programs = min(
            max_query_programs,
            max_query_len - query_start,
        )
        npu_grid_topk = (query_programs, batch, num_idx_heads)
        _topk_index_kernel_fallback_npu[npu_grid_topk](
            *kernel_args,
            score.shape[2],
            query_start,
            BLOCK_SIZE_K=fallback_block_size,
            MASK_INIT=False,
            MASK_LOCAL=False,
            num_warps=_topk_tile_num_warps(fallback_block_size),
            num_stages=1,
        )
    return topk_idx
