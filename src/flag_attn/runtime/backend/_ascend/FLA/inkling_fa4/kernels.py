# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels for Inkling FA4 paged relative attention."""

import triton
import triton.language as tl


@triton.jit
def _rel_attn_paged_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    rel_ptr,
    out_ptr,
    partial_out_ptr,
    partial_lse_ptr,
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
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_pl_s,
    stride_pl_t,
    stride_pl_h,
    stride_bt_b,
    stride_bt_block,
    softmax_scale,
    NUM_HEADS: tl.constexpr,
    KV_GROUP: tl.constexpr,
    BIAS_ALL_IN_RANGE: tl.constexpr,
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
    num_splits,
    SPLIT_KV: tl.constexpr,
    PAGE_LOAD: tl.constexpr = False,
):
    """Relative attention over the paged KV cache.

    One program covers up to ``BLOCK_M`` consecutive query tokens of one sequence
    for one query head (GQA heads are not packed: the per-row head index made the
    Q load and the output store scalarise, which measured 4-8x slower).  The KV
    range is clipped by the causal bound and the sliding window, then split across
    ``NUM_SPLITS`` programs so long caches and short windows both get parallelism.
    """
    pid_m = tl.program_id(0)
    pid_sh = tl.program_id(1)
    pid_split = tl.program_id(2)
    seq = pid_sh // NUM_HEADS
    head = pid_sh - seq * NUM_HEADS
    kv_head = head // KV_GROUP

    q_start = tl.load(cu_seqlens_q_ptr + seq)
    q_end = tl.load(cu_seqlens_q_ptr + seq + 1)
    k_len = tl.load(cache_seqlens_ptr + seq)
    q_len = q_end - q_start

    rows = tl.arange(0, BLOCK_M)
    token = pid_m * BLOCK_M + rows
    # Rows past the sequence end duplicate its last token so that the KV range
    # stays exactly the range the valid rows need.
    token_c = tl.minimum(token, tl.maximum(q_len - 1, 0))
    row_valid = token < q_len
    q_pos = (k_len - q_len) + token_c
    q_glob = q_start + token_c
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    q_tile = tl.load(
        q_ptr
        + q_glob[:, None] * stride_q_t
        + head[:, None] * stride_q_h
        + offs_d[None, :] * stride_q_d,
        mask=row_valid[:, None] & mask_d[None, :],
        other=0.0,
    )

    if SPLIT_KV:
        per_split = (k_len + num_splits - 1) // num_splits
        kv_lo = pid_split * per_split
        kv_hi = tl.minimum(kv_lo + per_split, k_len)
    else:
        kv_lo = 0
        kv_hi = k_len
    if CAUSAL:
        kv_hi = tl.minimum(kv_hi, tl.max(q_pos) + 1)
    if HAS_LEFT:
        kv_lo = tl.maximum(kv_lo, tl.min(q_pos) - WINDOW_LEFT)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    if PAGE_LOAD:
        tl.static_assert(BLOCK_M == 1 and BLOCK_D == 128)
        tl.static_assert(BLOCK_SIZE == 16 and BLOCK_N % BLOCK_SIZE == 0)
        iter_hi = tl.cdiv(kv_hi, BLOCK_SIZE) * BLOCK_SIZE
        kv_span = tl.where(kv_hi > kv_lo, iter_hi - kv_lo, 0)
    else:
        iter_hi = kv_hi
        kv_span = kv_hi - kv_lo
    # Column offsets of the bias row slice: constant for the whole loop.
    bias_col = tl.arange(0, BLOCK_N) * stride_rel_d
    for j0 in range(0, kv_span, BLOCK_N):
        if PAGE_LOAD:
            # Load each physical page as a contiguous token/feature tile.
            n = iter_hi - 1 - j0 - tl.arange(0, BLOCK_N)
            k_valid = (n >= kv_lo) & (n < kv_hi)
            page_start = (iter_hi - j0 - BLOCK_N) // BLOCK_SIZE
            page = page_start + tl.arange(0, BLOCK_N // BLOCK_SIZE)
            page_pos = page * BLOCK_SIZE
            row = tl.arange(0, BLOCK_SIZE)
            pos = page_pos[:, None] + row[None, :]
            page_valid = (page_pos < kv_hi) & (page_pos + BLOCK_SIZE > kv_lo)
            physical_page = tl.load(block_table_ptr + seq * stride_bt_b + page * stride_bt_block,
                                    mask=page_valid, other=0)
            k_fwd = tl.load(k_ptr + physical_page[:, None, None] * stride_k_block
                            + row[None, :, None] * stride_k_t + kv_head * stride_k_h
                            + offs_d[None, None, :] * stride_k_d,
                            mask=((pos >= kv_lo) & (pos < kv_hi))[:, :, None]
                            & mask_d[None, None, :], other=0.0)
            k_tile = tl.reshape(k_fwd, [BLOCK_N, BLOCK_D])
            q_vec = tl.sum(q_tile.to(tl.float32), axis=0)
            scores_fwd = tl.sum(k_tile.to(tl.float32) * q_vec[None, :], axis=1)
            reverse = BLOCK_N - 1 - tl.arange(0, BLOCK_N)
            scores = tl.gather(scores_fwd, reverse, 0)[None, :] * softmax_scale
        else:
            j = j0 + tl.arange(0, BLOCK_N)
            k_valid = j < kv_span
            # Walk the KV window backwards: with ``pos = kv_hi - 1 - j`` the relative
            # index ``q_pos - pos`` grows by one along the column axis, so the bias
            # load becomes a contiguous ascending row slice.  With the forward walk
            # it was a descending (negative-stride) address, which the compiler
            # turned into one scalar gather per element - the dominant cost of the
            # prefill shapes.
            n = (kv_hi - 1) - j
            logical = n // BLOCK_SIZE
            offset = n - logical * BLOCK_SIZE
            physical = tl.load(
                block_table_ptr + seq * stride_bt_b + logical * stride_bt_block,
                mask=k_valid,
                other=0,
            )
            k_tile = tl.load(
                k_ptr
                + physical[:, None] * stride_k_block
                + offset[:, None] * stride_k_t
                + kv_head * stride_k_h
                + offs_d[None, :] * stride_k_d,
                mask=k_valid[:, None] & mask_d[None, :],
                other=0.0,
            )
            if BLOCK_M == 1:
                # A single query uses vector reductions without padded Cube rows.
                q_vec = tl.sum(q_tile.to(tl.float32), axis=0)
                scores = tl.sum(k_tile.to(tl.float32) * q_vec[None, :], axis=1)[None, :] * softmax_scale
            else:
                scores = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32) * softmax_scale

        # Relative index of the first column of this tile; the index grows by one
        # per column, so the whole slice is one contiguous row run and the per-row
        # base can be hoisted out of the elementwise address math.
        bias_start = q_pos - ((iter_hi - 1) - j0)
        bias_row = (
            rel_ptr
            + q_glob * stride_rel_t
            + head * stride_rel_h
            + bias_start * stride_rel_d
        )
        if BIAS_ALL_IN_RANGE:
            # causal + KV span within rel_extent: every attended key is in
            # range, so no clamp and no in-range mask are needed at all.
            rel_bias = tl.load(
                bias_row[:, None] + bias_col[None, :],
                mask=row_valid[:, None] & k_valid[None, :],
                other=0.0,
            ).to(tl.float32)
        else:
            rel_dist = bias_start[:, None] + tl.arange(0, BLOCK_N)[None, :]
            in_range = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
            clamp_lo = tl.maximum(rel_dist, 0)
            safe_idx = tl.minimum(clamp_lo, REL_EXTENT - 1)
            rel_bias = tl.load(
                rel_ptr
                + q_glob[:, None] * stride_rel_t
                + head[:, None] * stride_rel_h
                + safe_idx * stride_rel_d,
                mask=row_valid[:, None] & k_valid[None, :] & in_range,
                other=0.0,
            ).to(tl.float32)
        scores = scores + rel_bias

        keep = row_valid[:, None] & k_valid[None, :]
        if CAUSAL:
            keep = keep & (n[None, :] <= q_pos[:, None])
        if HAS_LEFT:
            keep = keep & (n[None, :] >= q_pos[:, None] - WINDOW_LEFT)
        if HAS_RIGHT:
            keep = keep & (n[None, :] <= q_pos[:, None] + WINDOW_RIGHT)
        scores = tl.where(keep, scores, -float("inf"))

        if PAGE_LOAD:
            v_fwd = tl.load(v_ptr + physical_page[:, None, None] * stride_v_block
                            + row[None, :, None] * stride_v_t + kv_head * stride_v_h
                            + offs_d[None, None, :] * stride_v_d,
                            mask=((pos >= kv_lo) & (pos < kv_hi))[:, :, None]
                            & mask_d[None, None, :], other=0.0)
            v_tile = tl.reshape(v_fwd, [BLOCK_N, BLOCK_D])
        else:
            v_tile = tl.load(
                v_ptr
                + physical[:, None] * stride_v_block
                + offset[:, None] * stride_v_t
                + kv_head * stride_v_h
                + offs_d[None, :] * stride_v_d,
                mask=k_valid[:, None] & mask_d[None, :],
                other=0.0,
            )

        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        row_has = m_new != -float("inf")
        m_safe = tl.where(row_has, m_new, 0.0)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
        probabilities = tl.exp(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        if PAGE_LOAD:
            # Reverse only score/probability vectors to keep bias loads ascending.
            p_reverse = tl.gather(tl.reshape(probabilities, [BLOCK_N]), reverse, 0)
            weighted_v = p_reverse[:, None] * v_tile.to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(weighted_v, axis=0)[None, :]
        elif BLOCK_M == 1:
            weighted_v = tl.trans(probabilities) * v_tile.to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(weighted_v, axis=0)[None, :]
        else:
            acc = acc * alpha[:, None] + tl.dot(
                probabilities.to(v_tile.dtype), v_tile, out_dtype=tl.float32
            )
        m_i = m_new

    if SPLIT_KV:
        # Store the *normalised* partial output plus the log-sum-exp.  The combine
        # weights are exp(lse - max_lse), which already contains the split's own
        # sum, so the partial must be divided by it here (an unnormalised
        # numerator would be scaled by sum_s a second time).
        l_has = l_i > 0.0
        l_safe = tl.where(l_has, l_i, 1.0)
        lse = tl.where(l_has, m_i + tl.log(l_safe), -float("inf"))
        tl.store(
            partial_out_ptr
            + pid_split * stride_po_s
            + q_glob[:, None] * stride_po_t
            + head[:, None] * stride_po_h
            + offs_d[None, :],
            acc / l_safe[:, None],
            mask=row_valid[:, None] & mask_d[None, :],
        )
        tl.store(
            partial_lse_ptr
            + pid_split * stride_pl_s
            + q_glob * stride_pl_t
            + head * stride_pl_h,
            lse,
            mask=row_valid,
        )
    else:
        result = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
        tl.store(
            out_ptr
            + q_glob[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :] * stride_out_d,
            result.to(out_ptr.dtype.element_ty),
            mask=row_valid[:, None] & mask_d[None, :],
        )


@triton.jit
def _combine_splits_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    stride_po_s,
    stride_po_t,
    stride_po_h,
    stride_pl_s,
    stride_pl_t,
    stride_pl_h,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    total_rows,
    num_splits,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Merge ``NUM_SPLITS`` partial attentions: out = sum_s w_s*acc_s / sum_s w_s."""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    row_valid = rows < total_rows
    q_idx = rows // NUM_HEADS
    head = rows - q_idx * NUM_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    m_run = tl.full([ROWS], -float("inf"), tl.float32)
    for s in range(0, num_splits):
        m_run = tl.maximum(
            m_run,
            tl.load(
                partial_lse_ptr
                + s * stride_pl_s
                + q_idx * stride_pl_t
                + head * stride_pl_h,
                mask=row_valid,
                other=-float("inf"),
            ),
        )
    den = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, BLOCK_D], tl.float32)
    for s in range(0, num_splits):
        lse_s = tl.load(
            partial_lse_ptr
            + s * stride_pl_s
            + q_idx * stride_pl_t
            + head * stride_pl_h,
            mask=row_valid,
            other=-float("inf"),
        )
        weight = tl.where(m_run == -float("inf"), 0.0, tl.exp(lse_s - m_run))
        den += weight
        part = tl.load(
            partial_out_ptr
            + s * stride_po_s
            + q_idx[:, None] * stride_po_t
            + head[:, None] * stride_po_h
            + offs_d[None, :],
            mask=row_valid[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc += weight[:, None] * part
    den_safe = tl.where(den > 0.0, den, 1.0)
    result = acc / den_safe[:, None]
    tl.store(
        out_ptr
        + q_idx[:, None] * stride_out_t
        + head[:, None] * stride_out_h
        + offs_d[None, :] * stride_out_d,
        result.to(out_ptr.dtype.element_ty),
        mask=row_valid[:, None] & mask_d[None, :],
    )
