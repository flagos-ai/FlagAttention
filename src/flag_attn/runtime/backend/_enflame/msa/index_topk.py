# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""S60 MSA index scoring and TopK kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


from ..utils import SPARSE_BLOCK_SIZE, round_up


@triton.jit
def _insert_topk_slot(
    candidate_score,
    candidate_id,
    candidate_valid,
    best_score,
    best_id,
):
    best_valid = best_id >= 0

    better = candidate_valid & (
        (~best_valid)
        | (candidate_score > best_score)
        | (
            (candidate_score == best_score)
            & (candidate_id < best_id)
        )
    )

    next_score = tl.where(
        better,
        best_score,
        candidate_score,
    )
    next_id = tl.where(
        better,
        best_id,
        candidate_id,
    )
    next_valid = tl.where(
        better,
        best_valid,
        candidate_valid,
    )

    best_score = tl.where(
        better,
        candidate_score,
        best_score,
    )
    best_id = tl.where(
        better,
        candidate_id,
        best_id,
    )

    return (
        best_score,
        best_id,
        next_score,
        next_id,
        next_valid,
    )


@triton.jit(
    do_not_specialize=[
        "query_tile_offset",
        "batch_head_offset",
    ]
)
def _prefill_score_topk4_enflame(
    q_ptr,
    index_kv_ptr,
    output_ptr,
    block_table_ptr,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    query_tile_offset,
    batch_head_offset,
    num_idx_heads,
    init_blocks,
    local_blocks,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_oh,
    stride_on,
    stride_ok,
    stride_bt,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    TOPK: tl.constexpr,
):
    tl.static_assert(TOPK == 4)

    query_tile = (
        tl.program_id(0)
        + query_tile_offset
    )
    batch_head = (
        tl.program_id(1)
        + batch_head_offset
    )

    request_id = batch_head // num_idx_heads
    head_id = batch_head % num_idx_heads

    query_start = tl.load(
        cu_seqlens_q + request_id
    )
    query_end = tl.load(
        cu_seqlens_q + request_id + 1
    )
    query_length = query_end - query_start

    tile_start = query_tile * BLOCK_SIZE_Q
    if tile_start >= query_length:
        return

    sequence_length = tl.load(
        seq_lens + request_id
    )
    prefix_length = tl.load(
        prefix_lens + request_id
    )

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    local_query = tile_start + off_q
    query_valid = local_query < query_length
    global_query = query_start + local_query
    absolute_query = prefix_length + local_query

    q = tl.load(
        q_ptr
        + global_query[:, None] * stride_qn
        + head_id * stride_qh
        + off_d[None, :] * stride_qd,
        mask=query_valid[:, None],
        other=0.0,
    )

    valid_blocks = (
        absolute_query + BLOCK_SIZE_K
    ) // BLOCK_SIZE_K

    sequence_blocks = (
        sequence_length + BLOCK_SIZE_K - 1
    ) // BLOCK_SIZE_K

    valid_blocks = tl.minimum(
        valid_blocks,
        sequence_blocks,
    )

    negative = float("-inf")

    best_score_0 = tl.full(
        (BLOCK_SIZE_Q,),
        negative,
        tl.float32,
    )
    best_score_1 = tl.full(
        (BLOCK_SIZE_Q,),
        negative,
        tl.float32,
    )
    best_score_2 = tl.full(
        (BLOCK_SIZE_Q,),
        negative,
        tl.float32,
    )
    best_score_3 = tl.full(
        (BLOCK_SIZE_Q,),
        negative,
        tl.float32,
    )

    best_id_0 = tl.full(
        (BLOCK_SIZE_Q,),
        -1,
        tl.int32,
    )
    best_id_1 = tl.full(
        (BLOCK_SIZE_Q,),
        -1,
        tl.int32,
    )
    best_id_2 = tl.full(
        (BLOCK_SIZE_Q,),
        -1,
        tl.int32,
    )
    best_id_3 = tl.full(
        (BLOCK_SIZE_Q,),
        -1,
        tl.int32,
    )

    block_table_row = (
        block_table_ptr
        + request_id * stride_bt
    )

    high = tl.minimum(
        sequence_length,
        prefix_length
        + (query_tile + 1) * BLOCK_SIZE_Q,
    )

    for block_position in tl.range(
        0,
        high,
        BLOCK_SIZE_K,
    ):
        block_id = (
            block_position // BLOCK_SIZE_K
        )

        page = tl.load(
            block_table_row + block_id
        ).to(tl.int32)

        k = tl.load(
            index_kv_ptr
            + page * stride_kb
            + off_d[:, None] * stride_kd
            + off_k[None, :] * stride_kn
        )

        qk = tl.dot(
            q,
            k,
            out_dtype=tl.float32,
        )

        token_position = (
            block_position + off_k
        )

        available = (
            query_valid
            & (block_id < valid_blocks)
        )

        causal_mask = (
            available[:, None]
            & (
                absolute_query[:, None]
                >= token_position[None, :]
            )
        )

        qk = tl.where(
            causal_mask,
            qk,
            negative,
        )

        candidate_score = tl.max(
            qk,
            axis=1,
        )

        candidate_score = tl.where(
            candidate_score == candidate_score,
            candidate_score,
            negative,
        )

        local_start = tl.maximum(
            valid_blocks - local_blocks,
            0,
        )

        local_mask = (
            available
            & (block_id >= local_start)
        )

        candidate_score = tl.where(
            local_mask,
            1.0e29,
            candidate_score,
        )

        init_mask = (
            available
            & (block_id < init_blocks)
            & (candidate_score < 1.0e29)
        )

        candidate_score = tl.where(
            init_mask,
            1.0e30,
            candidate_score,
        )

        candidate_id = (
            block_id
            + tl.zeros(
                (BLOCK_SIZE_Q,),
                dtype=tl.int32,
            )
        )
        candidate_valid = available

        (
            best_score_0,
            best_id_0,
            candidate_score,
            candidate_id,
            candidate_valid,
        ) = _insert_topk_slot(
            candidate_score,
            candidate_id,
            candidate_valid,
            best_score_0,
            best_id_0,
        )

        (
            best_score_1,
            best_id_1,
            candidate_score,
            candidate_id,
            candidate_valid,
        ) = _insert_topk_slot(
            candidate_score,
            candidate_id,
            candidate_valid,
            best_score_1,
            best_id_1,
        )

        (
            best_score_2,
            best_id_2,
            candidate_score,
            candidate_id,
            candidate_valid,
        ) = _insert_topk_slot(
            candidate_score,
            candidate_id,
            candidate_valid,
            best_score_2,
            best_id_2,
        )

        (
            best_score_3,
            best_id_3,
            candidate_score,
            candidate_id,
            candidate_valid,
        ) = _insert_topk_slot(
            candidate_score,
            candidate_id,
            candidate_valid,
            best_score_3,
            best_id_3,
        )

    real_selected = tl.minimum(
        valid_blocks,
        TOPK,
    )

    output_base = (
        output_ptr
        + head_id * stride_oh
        + global_query * stride_on
    )

    tl.store(
        output_base,
        tl.where(
            real_selected > 0,
            best_id_0,
            -1,
        ),
        mask=query_valid,
    )
    tl.store(
        output_base + stride_ok,
        tl.where(
            real_selected > 1,
            best_id_1,
            -1,
        ),
        mask=query_valid,
    )
    tl.store(
        output_base + 2 * stride_ok,
        tl.where(
            real_selected > 2,
            best_id_2,
            -1,
        ),
        mask=query_valid,
    )
    tl.store(
        output_base + 3 * stride_ok,
        tl.where(
            real_selected > 3,
            best_id_3,
            -1,
        ),
        mask=query_valid,
    )


@torch.no_grad()
def minimax_m3_index_score_topk(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused S60 BF16 prefill score and Top4."""

    del max_seq_len

    if idx_q.dtype != torch.bfloat16:
        raise ValueError("idx_q must use BF16")
    if index_kv_cache.dtype != torch.bfloat16:
        raise ValueError(
            "index_kv_cache must use BF16"
        )
    if idx_q.shape[-1] != 128:
        raise ValueError(
            "fused path requires index head_dim=128"
        )
    if topk != 4:
        raise ValueError(
            "fused path currently requires topk=4"
        )

    for name, tensor in (
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(
                f"{name} must use int32"
            )

    total_q, num_idx_heads, _ = idx_q.shape

    if num_idx_heads != num_kv_heads:
        raise ValueError(
            "num_idx_heads must equal num_kv_heads"
        )

    if out is None:
        result = torch.empty(
            (
                num_idx_heads,
                total_q,
                topk,
            ),
            dtype=torch.int32,
            device=idx_q.device,
        )
    else:
        if out.dtype != torch.int32:
            raise ValueError("out must use int32")
        result = out[
            :num_idx_heads,
            :total_q,
            :topk,
        ]

    if total_q == 0:
        return result

    batch = cu_seqlens_q.shape[0] - 1
    query_tiles = triton.cdiv(
        max_query_len,
        64,
    )
    batch_heads = batch * num_idx_heads

    max_grid_x = 65535
    max_grid_y = 255

    for query_offset in range(
        0,
        query_tiles,
        max_grid_x,
    ):
        query_count = min(
            max_grid_x,
            query_tiles - query_offset,
        )

        for batch_head_offset in range(
            0,
            batch_heads,
            max_grid_y,
        ):
            batch_head_count = min(
                max_grid_y,
                batch_heads
                - batch_head_offset,
            )

            grid = (
                query_count,
                batch_head_count,
            )

            _prefill_score_topk4_enflame[grid](
                idx_q,
                index_kv_cache,
                result,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                query_offset,
                batch_head_offset,
                num_idx_heads,
                init_blocks,
                local_blocks,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                index_kv_cache.stride(0),
                index_kv_cache.stride(1),
                index_kv_cache.stride(2),
                result.stride(0),
                result.stride(1),
                result.stride(2),
                block_table.stride(0),
                BLOCK_SIZE_Q=64,
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                BLOCK_SIZE_D=128,
                TOPK=4,
                num_warps=1,
                num_stages=1,
            )

    return result


@triton.jit(
    do_not_specialize=[
        "query_offset",
        "head_offset",
        "batch_offset",
    ]
)
def _prefill_topk_reduction_enflame(
    score_ptr,
    output_ptr,
    cu_seqlens_q,
    prefix_lens,
    query_offset,
    head_offset,
    batch_offset,
    init_blocks,
    local_blocks,
    max_blocks,
    stride_sh,
    stride_sn,
    stride_sk,
    stride_oh,
    stride_on,
    stride_ok,
    TOPK: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPARSE_BLOCK: tl.constexpr,
):
    query_id = (
        tl.program_id(0)
        + query_offset
    )
    head_id = (
        tl.program_id(1)
        + head_offset
    )
    request_id = (
        tl.program_id(2)
        + batch_offset
    )

    q_start = tl.load(
        cu_seqlens_q + request_id
    )
    q_end = tl.load(
        cu_seqlens_q + request_id + 1
    )
    q_len = q_end - q_start

    if query_id >= q_len:
        return

    global_query = q_start + query_id
    prefix_len = tl.load(
        prefix_lens + request_id
    )

    valid_blocks = (
        prefix_len
        + query_id
        + SPARSE_BLOCK
    ) // SPARSE_BLOCK

    valid_blocks = tl.minimum(
        tl.maximum(valid_blocks, 0),
        max_blocks,
    )
    real_selected = tl.minimum(
        valid_blocks,
        TOPK,
    )

    block_ids = tl.arange(
        0,
        BLOCK_SIZE_N,
    )
    in_storage = block_ids < max_blocks
    available = block_ids < valid_blocks

    values = tl.load(
        score_ptr
        + head_id * stride_sh
        + global_query * stride_sn
        + block_ids * stride_sk,
        mask=in_storage,
        other=float("-inf"),
    )

    values = tl.where(
        values == values,
        values,
        float("-inf"),
    )
    values = tl.where(
        available,
        values,
        float("-inf"),
    )

    if local_blocks > 0:
        local_start = tl.maximum(
            valid_blocks - local_blocks,
            0,
        )
        local_mask = (
            available
            & (block_ids >= local_start)
        )
        values = tl.where(
            local_mask,
            1.0e29,
            values,
        )

    if init_blocks > 0:
        init_mask = (
            available
            & (block_ids < init_blocks)
        )
        promote_init = (
            init_mask
            & (values < 1.0e29)
        )
        values = tl.where(
            promote_init,
            1.0e30,
            values,
        )

    for slot in tl.static_range(0, TOPK):
        maximum = tl.max(
            values,
            axis=0,
        )

        candidate_mask = (
            available
            & (values == maximum)
        )

        # Select the lowest block id for ties. This preserves
        # deterministic ordering for all-negative-infinity rows.
        negative_index = tl.where(
            candidate_mask,
            -block_ids,
            -BLOCK_SIZE_N,
        )
        selected = -tl.max(
            negative_index,
            axis=0,
        )

        slot_valid = slot < real_selected

        tl.store(
            output_ptr
            + head_id * stride_oh
            + global_query * stride_on
            + slot * stride_ok,
            tl.where(
                slot_valid,
                selected,
                -1,
            ),
        )

        available = (
            available
            & (block_ids != selected)
        )
        values = tl.where(
            available,
            values,
            float("-inf"),
        )


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """S60 device-resident reduction TopK for MSA prefill."""
    if score.ndim != 3:
        raise ValueError(
            "score must have shape "
            "[num_heads, total_q, max_blocks]"
        )
    if score.dtype != torch.float32:
        raise ValueError("score must use float32")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if max_query_len <= 0 and score.shape[1] > 0:
        raise ValueError(
            "max_query_len must be positive"
        )
    if cu_seqlens_q.dtype != torch.int32:
        raise ValueError(
            "cu_seqlens_q must use int32"
        )
    if prefix_lens.dtype != torch.int32:
        raise ValueError(
            "prefix_lens must use int32"
        )

    num_heads, total_q, max_blocks = score.shape
    batch = cu_seqlens_q.shape[0] - 1

    if max_blocks <= 0:
        raise ValueError(
            "score must contain at least one block"
        )
    if prefix_lens.shape[0] != batch:
        raise ValueError(
            "prefix_lens has an incompatible shape"
        )

    if out is None:
        topk_idx = torch.empty(
            (num_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    else:
        if out.dtype != torch.int32:
            raise ValueError("out must use int32")
        if (
            out.shape[0] != num_heads
            or out.shape[1] < total_q
            or out.shape[2] < topk
        ):
            raise ValueError(
                "out has an incompatible shape"
            )
        topk_idx = out[
            :num_heads,
            :total_q,
            :topk,
        ]

    if total_q == 0:
        return topk_idx

    block_size_n = triton.next_power_of_2(
        max_blocks
    )

    max_grid_x = 65535
    max_grid_y = 255
    max_grid_z = 255

    for batch_offset in range(
        0,
        batch,
        max_grid_z,
    ):
        batch_count = min(
            max_grid_z,
            batch - batch_offset,
        )

        for head_offset in range(
            0,
            num_heads,
            max_grid_y,
        ):
            head_count = min(
                max_grid_y,
                num_heads - head_offset,
            )

            for query_offset in range(
                0,
                max_query_len,
                max_grid_x,
            ):
                query_count = min(
                    max_grid_x,
                    max_query_len - query_offset,
                )

                grid = (
                    query_count,
                    head_count,
                    batch_count,
                )

                _prefill_topk_reduction_enflame[grid](
                    score,
                    topk_idx,
                    cu_seqlens_q,
                    prefix_lens,
                    query_offset,
                    head_offset,
                    batch_offset,
                    init_blocks,
                    local_blocks,
                    max_blocks,
                    score.stride(0),
                    score.stride(1),
                    score.stride(2),
                    topk_idx.stride(0),
                    topk_idx.stride(1),
                    topk_idx.stride(2),
                    TOPK=topk,
                    BLOCK_SIZE_N=block_size_n,
                    SPARSE_BLOCK=SPARSE_BLOCK_SIZE,
                    num_warps=1,
                    num_stages=1,
                )

    return topk_idx


@triton.jit
def _decode_topk_reduction_enflame(
    score_ptr,
    output_ptr,
    seq_lens,
    decode_query_len: tl.constexpr,
    max_blocks: tl.constexpr,
    stride_sh,
    stride_sn,
    stride_sk,
    stride_oh,
    stride_on,
    stride_ok,
    TOPK: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPARSE_BLOCK: tl.constexpr,
):
    """Select Decode pages on-device for the short S60 score rows."""
    query_id = tl.program_id(0)
    head_id = tl.program_id(1)
    request_id = query_id // decode_query_len
    query_offset = query_id - request_id * decode_query_len

    seq_len = tl.load(seq_lens + request_id).to(tl.int32)
    query_pos = seq_len - decode_query_len + query_offset
    valid_blocks = (tl.maximum(query_pos + 1, 0) + SPARSE_BLOCK - 1) // SPARSE_BLOCK
    valid_blocks = tl.minimum(valid_blocks, max_blocks)
    real_selected = tl.minimum(valid_blocks, TOPK)

    block_ids = tl.arange(0, BLOCK_SIZE_N)
    available = block_ids < valid_blocks
    values = tl.load(
        score_ptr
        + head_id * stride_sh
        + query_id * stride_sn
        + block_ids * stride_sk,
        mask=block_ids < max_blocks,
        other=float("-inf"),
    )
    values = tl.where((values == values) & available, values, float("-inf"))

    for slot in tl.static_range(0, TOPK):
        maximum = tl.max(values, axis=0)
        candidate_mask = available & (values == maximum)
        # Match deterministic page ordering by resolving ties to the lowest id.
        selected = -tl.max(
            tl.where(candidate_mask, -block_ids, -BLOCK_SIZE_N),
            axis=0,
        )
        tl.store(
            output_ptr
            + head_id * stride_oh
            + query_id * stride_on
            + slot * stride_ok,
            tl.where(slot < real_selected, selected, -1),
        )
        available = available & (block_ids != selected)
        values = tl.where(available, values, float("-inf"))


@triton.jit
def _decode_topk_identity_enflame(
    output_ptr,
    seq_lens,
    decode_query_len: tl.constexpr,
    stride_oh,
    stride_on,
    stride_ok,
    TOPK: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    SPARSE_BLOCK: tl.constexpr,
):
    """Generate all-visible Decode page ids directly on-device."""
    query_id = tl.program_id(0)
    head_id = tl.program_id(1)
    request_id = query_id // decode_query_len
    query_offset = query_id - request_id * decode_query_len

    seq_len = tl.load(seq_lens + request_id).to(tl.int32)
    query_pos = seq_len - decode_query_len + query_offset
    valid_blocks = (tl.maximum(query_pos + 1, 0) + SPARSE_BLOCK - 1) // SPARSE_BLOCK

    slots = tl.arange(0, BLOCK_SIZE_T)
    tl.store(
        output_ptr
        + head_id * stride_oh
        + query_id * stride_on
        + slots * stride_ok,
        tl.where(slots < valid_blocks, slots, -1),
        mask=slots < TOPK,
    )


@triton.jit
def _decode_index_score_kernel_enflame(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    decode_query_len: tl.constexpr,
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
    num_kv_chunks: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = num_idx_heads * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_q = tl.program_id(2)
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    h_offsets = hq_offsets // BLOCK_SIZE_Q
    q_offsets = (
        pid_q * BLOCK_SIZE_Q
        + hq_offsets % BLOCK_SIZE_Q
    )
    q_mask = q_offsets < decode_query_len
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
    # Query vectors for all index heads in a small spec-decode block.
    # Keep queries on the M axis so the dot product writes one contiguous
    # score row per query instead of using the very small HQ dimension as N.
    q = tl.load(
        q_ptr
        + q_ids[:, None] * stride_q_n
        + h_offsets[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=q_mask[:, None],
        other=0.0,
    )  # [HQ,D]
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int32)
        pos = blk * BLOCK_SIZE_K + off_k
        pos_mask = pos[None, :] < kv_len[:, None]
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_d[:, None] * stride_ik_d
            + off_k[None, :] * stride_ik_pos,
        )  # [D,N]
        # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k are
        # loaded in their stored dtype (bf16 or e4m3) and the MMA accumulates in
        # fp32 so the per-block max score is exact for the fp8 indexer too.
        kq = tl.dot(q, k, out_dtype=tl.float32)  # [HQ,N]
        kq = tl.where(pos_mask & q_mask[:, None], kq, float("-inf"))
        score = tl.max(kq, axis=1)  # [HQ]
        is_visible_block = blk < num_blocks_q
        is_init = (blk < init_blocks) & is_visible_block
        is_local = (blk >= local_start) & is_visible_block
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + h_offsets * stride_s_h + q_ids * stride_s_n + blk * stride_s_k,
            score,
            mask=q_mask,
        )


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
    total_q, num_idx_heads, head_dim = idx_q.shape

    if num_idx_heads != num_kv_heads:
        raise ValueError("num_idx_heads must equal num_kv_heads")
    if total_q != seq_lens.shape[0] * decode_query_len:
        raise ValueError("total_q is incompatible with decode metadata")

    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    if score_out is None:
        score = torch.empty(
            (
                num_idx_heads,
                total_q,
                round_up(max_block, 16),
            ),
            dtype=torch.float32,
            device=idx_q.device,
        )
    else:
        score = score_out

    max_grid_z = 255
    min_block_size_q = triton.cdiv(
        max_decode_query_len,
        max_grid_z,
    )
    block_size_q = triton.next_power_of_2(
        max(1, min_block_size_q)
    )
    num_query_tiles = triton.cdiv(
        max_decode_query_len,
        block_size_q,
    )

    # Q=1 favors one page per chunk. Speculative Decode keeps two adjacent
    # pages per chunk to balance query parallelism and program launch count.
    chunk_limit = (
        max_block
        if decode_query_len == 1
        else triton.cdiv(max_block, 2)
    )
    target = max(
        1,
        min(
            128,
            chunk_limit,
            512 // max(1, seq_lens.shape[0]),
        ),
    )
    num_kv_chunks = 1 << (target.bit_length() - 1)

    grid = (
        seq_lens.shape[0],
        num_kv_chunks,
        num_query_tiles,
    )

    _decode_index_score_kernel_enflame[grid](
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
        BLOCK_SIZE_Q=block_size_q,
        num_kv_chunks=num_kv_chunks,
        USE_PDL=False,
        num_warps=1,
        num_stages=1,
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
    total_q, num_idx_heads, _ = idx_q.shape
    max_block = triton.cdiv(
        max_seq_len,
        SPARSE_BLOCK_SIZE,
    )

    if out is None:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=idx_q.device,
        )
    else:
        topk_idx = out[:, :total_q, :topk]

    if max_block <= topk and score_out is None:
        _decode_topk_identity_enflame[(total_q, num_idx_heads)](
            topk_idx,
            seq_lens,
            decode_query_len,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            TOPK=topk,
            BLOCK_SIZE_T=triton.next_power_of_2(topk),
            SPARSE_BLOCK=SPARSE_BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
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

    # The benchmark Decode rows have at most 64 pages. Keep their TopK
    # selection entirely on GCU and retain the generic Torch fallback for
    # larger or differently configured rows.
    if max_block <= 128 and topk == 4:
        _decode_topk_reduction_enflame[(total_q, num_idx_heads)](
            score,
            topk_idx,
            seq_lens,
            decode_query_len,
            max_block,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            TOPK=topk,
            BLOCK_SIZE_N=triton.next_power_of_2(max_block),
            SPARSE_BLOCK=SPARSE_BLOCK_SIZE,
            num_warps=1,
            num_stages=1,
        )
        return topk_idx

    topk_idx.fill_(-1)
    query_ids = torch.arange(
        total_q,
        dtype=torch.int32,
        device=idx_q.device,
    )
    request_ids = query_ids // decode_query_len
    query_offsets = (
        query_ids
        - request_ids * decode_query_len
    )
    query_seq_lens = torch.index_select(
        seq_lens,
        0,
        request_ids,
    )
    kv_lens = (
        query_seq_lens
        - decode_query_len
        + query_offsets
        + 1
    ).clamp(min=0)
    valid_blocks = (
        kv_lens
        + SPARSE_BLOCK_SIZE
        - 1
    ) // SPARSE_BLOCK_SIZE

    block_ids = torch.arange(
        max_block,
        dtype=torch.int32,
        device=idx_q.device,
    ).view(1, 1, max_block)

    valid_mask = (
        block_ids
        < valid_blocks.view(1, total_q, 1)
    )

    negative = torch.full(
        (),
        -1.0e30,
        dtype=torch.float32,
        device=idx_q.device,
    )

    candidates = score[:, :, :max_block]
    candidates = torch.where(
        candidates == candidates,
        candidates,
        negative,
    )
    candidates = torch.where(
        valid_mask,
        candidates,
        negative,
    )

    selected_count = min(topk, max_block)
    selected = torch.topk(
        candidates,
        k=selected_count,
        dim=-1,
    ).indices.to(torch.int32)

    slot_ids = torch.arange(
        selected_count,
        dtype=torch.int32,
        device=idx_q.device,
    ).view(1, 1, selected_count)

    selected = torch.where(
        slot_ids
        < valid_blocks.clamp(
            max=selected_count
        ).view(1, total_q, 1),
        selected,
        torch.full_like(selected, -1),
    )

    topk_idx[:, :, :selected_count].copy_(selected)
    return topk_idx
