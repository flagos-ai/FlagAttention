# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels for Inkling FA4 paged relative attention."""

import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.jit
def _rel_attn_paged_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    rel_ptr,
    out_ptr,
    block_table_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_rel_t,
    stride_rel_h,
    stride_rel_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_bt_b,
    stride_bt_block,
    softmax_scale,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_LEFT: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    HAS_RIGHT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
):
    """One query tile of one (sequence, query head); walks the paged KV cache.

    Correctness-first form: no split-KV, no GQA packing, one row per query
    token.  The KV tile is gathered through ``block_table`` page by page, the
    relative bias is applied only inside ``[0, REL_EXTENT)`` (matching the
    reference), and the sliding window / causal mask is applied on top.
    """
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    seq = pid_s // NUM_HEADS
    head = pid_s - seq * NUM_HEADS
    if NUM_KV_HEADS == NUM_HEADS:
        kv_head = head
    else:
        kv_head = head // (NUM_HEADS // NUM_KV_HEADS)

    q_start = tl.load(cu_seqlens_q_ptr + seq)
    q_end = tl.load(cu_seqlens_q_ptr + seq + 1)
    k_len = tl.load(cache_seqlens_ptr + seq)
    q_len = q_end - q_start

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_valid = offs_m < q_len
    q_pos = offs_m + (k_len - q_len)
    rows = q_start + offs_m
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    q_tile = tl.load(
        q_ptr
        + rows[:, None] * stride_q_t
        + head * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=rows_valid[:, None] & mask_d[None, :],
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    for n0 in range(0, k_len, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        k_valid = n < k_len
        logical = n // BLOCK_SIZE
        offset = n - logical * BLOCK_SIZE
        physical = tl.load(
            block_table_ptr + seq * stride_bt_b + logical * stride_bt_block,
            mask=k_valid,
            other=0,
        )
        k_ptrs = (
            k_ptr
            + physical[:, None] * stride_k_block
            + offset[:, None] * stride_k_t
            + kv_head * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        k_tile = tl.load(k_ptrs, mask=k_valid[:, None] & mask_d[None, :], other=0.0)
        scores = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * softmax_scale

        rel_dist = q_pos[:, None] - n[None, :]
        in_range = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
        safe_idx = tl.where(in_range, rel_dist, 0)
        rel_bias = tl.load(
            rel_ptr
            + rows[:, None] * stride_rel_t
            + head * stride_rel_h
            + safe_idx * stride_rel_d,
            mask=rows_valid[:, None] & k_valid[None, :] & in_range,
            other=0.0,
        ).to(tl.float32)
        scores = scores + rel_bias

        keep = rows_valid[:, None] & k_valid[None, :]
        if CAUSAL:
            keep = keep & (n[None, :] <= q_pos[:, None])
        if HAS_LEFT:
            keep = keep & (n[None, :] >= q_pos[:, None] - WINDOW_LEFT)
        if HAS_RIGHT:
            keep = keep & (n[None, :] <= q_pos[:, None] + WINDOW_RIGHT)
        scores = tl.where(keep, scores, -float("inf"))

        v_ptrs = (
            v_ptr
            + physical[:, None] * stride_v_block
            + offset[:, None] * stride_v_t
            + kv_head * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        v_tile = tl.load(v_ptrs, mask=k_valid[:, None] & mask_d[None, :], other=0.0)

        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        row_has = m_new != -float("inf")
        m_safe = tl.where(row_has, m_new, 0.0)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
        probabilities = tl.exp(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            probabilities.to(v_tile.dtype), v_tile, out_dtype=tl.float32
        )
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    result = acc / l_safe[:, None]
    tl.store(
        out_ptr
        + rows[:, None] * stride_out_t
        + head * stride_out_h
        + offs_d[None, :] * stride_out_d,
        result.to(out_ptr.dtype.element_ty),
        mask=rows_valid[:, None] & mask_d[None, :],
    )
