# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""S60 MSA prefill and decode sparse-attention kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from flag_attn.minimax_sparse_attention.sparse_attn import (
    _FP8_DTYPES,
    _KV_SCALE_NONE,
    _kv_scale_args,
)
from flag_attn.runtime.backend._enflame.utils import SPARSE_BLOCK_SIZE


_PREFILL_HALF_KV_MAX_BLOCK_SIZE_QH = 8


@triton.jit
def _gqa_sparse_common_prefix(
    q,
    kv,
    ids,
    table,
    cu,
    seq,
    prefix,
    partial,
    maximum,
    denom,
    counts,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    SH: tl.constexpr,
    SD: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KN: tl.constexpr,
    KD: tl.constexpr,
    IH: tl.constexpr,
    IQ: tl.constexpr,
    IK: tl.constexpr,
    BT: tl.constexpr,
    PN: tl.constexpr,
    PH: tl.constexpr,
    PD: tl.constexpr,
    SN: tl.constexpr,
    KVH: tl.constexpr,
    NG: tl.constexpr,
    GQ: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    group = tl.program_id(0)
    head = tl.program_id(1)
    req = tl.program_id(2)
    begin = tl.load(cu + req)
    length = tl.load(cu + req + 1) - begin
    local = group * GQ
    if local >= length:
        return
    prefix_len = tl.load(prefix + req)
    seq_len = tl.load(seq + req)
    oq = tl.arange(0, GQ)
    rows = tl.arange(0, GQ * BH)
    on = tl.arange(0, 128)
    qvalid = local + oq < length
    qp = tl.make_block_ptr(
        q + begin * SQ + head * H * SH, (length, H, D), (SQ, SH, SD), (local, 0, 0), (GQ, BH, BD), (2, 1, 0)
    )
    qt = tl.load(qp, boundary_check=(0, 1, 2), padding_option="zero").reshape(GQ * BH, BD)
    m = tl.full((GQ * BH,), float("-inf"), tl.float32)
    l = tl.zeros((GQ * BH,), tl.float32)
    acc = tl.zeros((GQ * BH, BD), tl.float32)
    active = True
    count = 0
    for slot in range(min(4, K)):
        first = tl.load(ids + head * IH + (begin + local) * IQ + slot * IK)
        values = tl.load(ids + head * IH + (begin + local + oq) * IQ + slot * IK, mask=qvalid, other=-1)
        exists = slot < tl.minimum(K, (prefix_len + local + 128) // 128)
        shared = tl.sum(tl.where(qvalid, (values == first).to(tl.int32), 1), axis=0) == GQ
        active = active & exists & (first >= 0) & shared
        if active:
            count += 1
            page = tl.load(table + req * BT + first)
            base = kv + page * KB + head * KH
            kp = tl.make_block_ptr(base, (128, D), (KN, KD), (0, 0), (128, BD), (1, 0))
            kt = tl.load(kp, boundary_check=(0, 1), padding_option="zero")
            positions = first * 128 + on
            valid = (
                (rows[:, None] // BH + local < length)
                & (rows[:, None] % BH < H)
                & (positions[None, :] < seq_len)
                & (positions[None, :] <= prefix_len + local + rows[:, None] // BH)
            )
            scores = tl.dot(qt, tl.trans(kt)) * (SCALE * 1.4426950409)
            scores = tl.where(valid, scores, float("-inf"))
            updated = tl.maximum(m, tl.max(scores, axis=1))
            safe = tl.where(updated > float("-inf"), updated, 0.0)
            p = tl.where(valid, tl.exp2(scores - safe[:, None]), 0.0)
            alpha = tl.where(m > float("-inf"), tl.exp2(m - safe), 0.0)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            vp = tl.make_block_ptr(base + D * KD, (128, D), (KN, KD), (0, 0), (128, BD), (1, 0))
            vt = tl.load(vp, boundary_check=(0, 1), padding_option="zero")
            acc += tl.dot(p.to(vt.dtype), vt)
            m = updated
    ap = tl.make_block_ptr(
        partial + begin * PN + head * H * PH, (length, H, D), (PN, PH, PD), (local, 0, 0), (GQ, BH, BD), (2, 1, 0)
    )
    tl.store(ap, acc.reshape(GQ, BH, BD), boundary_check=(0, 1, 2))
    mp = tl.make_block_ptr(maximum + begin * SN + head * H, (length, H), (SN, 1), (local, 0), (GQ, BH), (1, 0))
    lp = tl.make_block_ptr(denom + begin * SN + head * H, (length, H), (SN, 1), (local, 0), (GQ, BH), (1, 0))
    tl.store(mp, m.reshape(GQ, BH), boundary_check=(0, 1))
    tl.store(lp, l.reshape(GQ, BH), boundary_check=(0, 1))
    tl.store(counts + (req * KVH + head) * NG + group, count)


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_direct(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    num_q_loop: tl.constexpr,
    sm_scale: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kv_blk: tl.constexpr,
    stride_kv_h: tl.constexpr,
    stride_kv_pos: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_ks_h: tl.constexpr,
    stride_ks_t: tl.constexpr,
    stride_vs_h: tl.constexpr,
    stride_vs_t: tl.constexpr,
    stride_th: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_bt_b: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    partial_ptr=0,
    maximum_ptr=0,
    denom_ptr=0,
    count_ptr=0,
    GROUP_Q: tl.constexpr = 1,
    NUM_GROUPS: tl.constexpr = 1,
    STRIDE_PN: tl.constexpr = 0,
    STRIDE_PH: tl.constexpr = 0,
    STRIDE_PD: tl.constexpr = 0,
    STRIDE_SN: tl.constexpr = 0,
    USE_SHARED_PREFIX: tl.constexpr = False,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    if USE_FP8 and KV_SCALE_MODE == 1:
        # Scalar scales are invariant across every selected page. Fold K into
        # the logits scale and delay V until the normalized output.
        qk_scale = sm_scale_log2e * tl.load(k_scale_ptr)
        v_scale_scalar = tl.load(v_scale_ptr)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # Valid block count from seq position (no sentinel): block_size_q == 1.
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(gqa_group_size, head_dim),
            strides=(stride_qh, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        if not USE_FP8:
            # Accumulate denominator lanes online; reduce once after all pages.
            l_acc = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
        # Keep the direct kernel's PV accumulator in [QH, D] order.  The
        # transposed [D, QH] form spills heavily for the FP8 QH=32 tile.
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        shared_count = 0
        if USE_SHARED_PREFIX:
            # Preserve FP32 online state; never round an intermediate output.
            shared_count = tl.load(count_ptr + (pid_b * num_kv_heads + pid_kh) * NUM_GROUPS + pid_q_j // GROUP_Q)
            hs = tl.arange(0, BLOCK_SIZE_QH)
            state_base = (q_start + pid_q_j) * STRIDE_SN + pid_kh * gqa_group_size
            m_i = tl.load(maximum_ptr + state_base + hs, mask=hs < gqa_group_size, other=float("-inf"))
            l_i = tl.load(denom_ptr + state_base + hs, mask=hs < gqa_group_size, other=0.0)
            pp = tl.make_block_ptr(
                partial_ptr + (q_start + pid_q_j) * STRIDE_PN + pid_kh * gqa_group_size * STRIDE_PH,
                (gqa_group_size, head_dim),
                (STRIDE_PH, STRIDE_PD),
                (0, 0),
                (BLOCK_SIZE_QH, BLOCK_SIZE_D),
                (1, 0),
            )
            acc_o = tl.load(pp, boundary_check=(0, 1), padding_option="zero")
            l_acc = tl.where(off_n[None, :] == 0, l_i[:, None], 0.0)
            t_ptr_j += shared_count * stride_tk
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        if USE_FP8:
            # Keep QK in FP8 Tensor Core form.  Q is per-head dynamically
            # scaled once, then its scale is restored on the FP32 logits.
            # This keeps the existing K cache in FP8 through tl.dot instead
            # of materializing a BF16 K tile for every selected page.
            qk_q_scale = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 448.0), 1.0e-8)
            q_qk = (q / qk_q_scale[:, None]).to(tl.float8e4nv)
        else:
            q_qk = q
        for _ in range(shared_count, real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int32)

            pos = c + off_n
            pos_mask = pos < seq_len

            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            k_ptrs = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            k = tl.load(
                k_ptrs,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr + pid_kh * stride_ks_h + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
            qk_dot = tl.dot(q_qk, tl.trans(k))
            if USE_FP8:
                qk_dot *= qk_q_scale[:, None]

            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len

            qk = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(q_abs >= pos[None, :], 0, float("-inf"))

            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            if USE_FP8:
                if KV_SCALE_MODE == 1:
                    qk += qk_dot * qk_scale
                elif KV_SCALE_MODE == 2:
                    # Q @ (K * scale[token]) is equivalent to scaling the
                    # smaller [QH, K] logits instead of the [K, D] K tile.
                    qk += qk_dot * (sm_scale_log2e * k_scale[None, :])
                else:
                    qk += qk_dot * sm_scale_log2e
            else:
                qk += qk_dot * sm_scale_log2e

            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            if USE_FP8:
                l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            if USE_FP8:
                l_i = l_i * alpha + l_ij
            else:
                l_acc = l_acc * alpha[:, None] + p

            v_base_ptr = k_base_ptr + head_dim * stride_kv_d
            v_ptrs = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v = tl.load(
                v_ptrs,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr + pid_kh * stride_vs_h + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )

            if USE_FP8 and BLOCK_SIZE_H < 32:
                # The 8/16-head tiles lower efficiently as FP8 PV MMAs.
                # Fold a per-token V scale into P before quantizing P
                # row-wise, and keep V in its FP8 cache format.
                pv_p = p
                if KV_SCALE_MODE == 2:
                    pv_p *= v_scale[None, :]
                pv_p_scale = tl.maximum(tl.max(tl.abs(pv_p), axis=1) * (1.0 / 448.0), 1.0e-8)
                p_dot = (pv_p / pv_p_scale[:, None]).to(tl.float8e4nv)
                acc_o += tl.dot(p_dot, v) * pv_p_scale[:, None]
            else:
                # QH=32 uses the native PV accumulator layout above, but its
                # FP8 PV lowering spills heavily; retain BF16 operands there.
                if USE_FP8:
                    v = v.to(q.dtype)
                p_dot = p.to(v.dtype)
                if USE_FP8 and KV_SCALE_MODE == 2:
                    p_dot = (p * v_scale[None, :]).to(v.dtype)
                acc_o += tl.dot(p_dot, v)
            m_i = m_ij

        if not USE_FP8:
            l_i = tl.sum(l_acc, axis=1)
        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        if USE_FP8 and KV_SCALE_MODE == 1:
            inv_l *= v_scale_scalar
        acc_o = acc_o * inv_l[:, None]

        o_ptrs = tl.make_block_ptr(
            base=o_ptr + (q_start + pid_q_j) * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(gqa_group_size, head_dim),
            strides=(stride_oh, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _gqa_sparse_fwd_grouped8(
    q_ptr,
    kv_cache_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    decode_query_len: tl.constexpr,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kv_blk: tl.constexpr,
    stride_kv_h: tl.constexpr,
    stride_kv_pos: tl.constexpr,
    stride_kv_d: tl.constexpr,
    stride_th: tl.constexpr,
    stride_tn: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_on: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_bt_b: tl.constexpr,
    GROUP_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    DECODE_MODE: tl.constexpr,
):
    """Process a small query group using the union of its selected pages."""
    BLOCK_SIZE_M: tl.constexpr = GROUP_SIZE_Q * BLOCK_SIZE_H
    NUM_CANDIDATES: tl.constexpr = GROUP_SIZE_Q * BLOCK_TOPK

    pid_g = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)

    seq_len = tl.load(seq_lens + pid_b)
    if DECODE_MODE:
        q_start = pid_b * decode_query_len
        q_len = decode_query_len
        prefix_len = seq_len - decode_query_len
    else:
        q_start = tl.load(cu_seqlens_q + pid_b)
        q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
        prefix_len = tl.load(prefix_lens + pid_b)

    group_start = pid_g * GROUP_SIZE_Q
    if group_start >= q_len:
        return

    off_g = tl.arange(0, GROUP_SIZE_Q)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_TOPK)

    q_local = group_start + off_g
    q_valid = q_local < q_len

    q_ptrs = tl.make_block_ptr(
        base=(q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh),
        shape=(
            q_len,
            gqa_group_size,
            head_dim,
        ),
        strides=(
            stride_qn,
            stride_qh,
            stride_qd,
        ),
        offsets=(
            group_start,
            0,
            0,
        ),
        block_shape=(
            GROUP_SIZE_Q,
            BLOCK_SIZE_H,
            BLOCK_SIZE_D,
        ),
        order=(2, 1, 0),
    )

    q = tl.load(
        q_ptrs,
        boundary_check=(0, 1, 2),
        padding_option="zero",
    )
    q = tl.reshape(
        q,
        BLOCK_SIZE_M,
        BLOCK_SIZE_D,
    )

    topk_mask = q_valid[:, None] & (off_t[None, :] < MAX_TOPK)

    topk_values = tl.load(
        t_ptr + pid_kh * stride_th + (q_start + q_local[:, None]) * stride_tn + off_t[None, :] * stride_tk,
        mask=topk_mask,
        other=-1,
    ).to(tl.int32)

    candidates = tl.reshape(
        topk_values,
        NUM_CANDIDATES,
    )
    candidate_offsets = tl.arange(
        0,
        NUM_CANDIDATES,
    )

    # Build a compact first-occurrence list for the
    # bounded candidate slots. The attention loop then runs only
    # over the actual union of selected pages.
    candidate_lhs = candidates[:, None]
    candidate_rhs = candidates[None, :]

    offset_lhs = candidate_offsets[:, None]
    offset_rhs = candidate_offsets[None, :]

    prior_matches = (candidate_lhs == candidate_rhs) & (offset_rhs < offset_lhs)

    prior_count = tl.sum(
        prior_matches,
        axis=1,
    )

    unique_mask = (candidates >= 0) & (prior_count == 0)

    # First-occurrence ranks are an inclusive prefix sum of unique flags.
    unique_rank = tl.cumsum(unique_mask.to(tl.int32), axis=0) - 1

    unique_count = tl.sum(
        unique_mask,
        axis=0,
    )

    m_i = tl.full(
        (BLOCK_SIZE_M,),
        float("-inf"),
        dtype=tl.float32,
    )
    l_i = tl.zeros(
        (BLOCK_SIZE_M,),
        dtype=tl.float32,
    )
    acc_o = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_D),
        dtype=tl.float32,
    )

    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Match the [query * padded-head, token] matrix rows directly.
    off_m = tl.arange(0, BLOCK_SIZE_M)
    row_query = group_start + off_m // BLOCK_SIZE_H
    row_valid = (row_query < q_len) & (off_m % BLOCK_SIZE_H < gqa_group_size)
    row_abs = prefix_len + row_query
    sm_scale_log2e = sm_scale * 1.4426950409

    for unique_index in range(unique_count):
        candidate = tl.sum(
            tl.where(
                unique_mask & (unique_rank == unique_index),
                candidates,
                0,
            ),
            axis=0,
        ).to(tl.int32)

        selected_q = (
            tl.sum(
                tl.where(
                    (topk_values == candidate) & topk_mask,
                    1,
                    0,
                ),
                axis=1,
            )
            > 0
        )

        page = tl.load(bt_row + candidate).to(tl.int32)

        page_position = candidate * BLOCK_SIZE_K
        positions = page_position + off_n

        kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h

        k_ptrs = tl.make_block_ptr(
            base=kv_base,
            shape=(
                BLOCK_SIZE_K,
                head_dim,
            ),
            strides=(
                stride_kv_pos,
                stride_kv_d,
            ),
            offsets=(0, 0),
            block_shape=(
                BLOCK_SIZE_K,
                BLOCK_SIZE_D,
            ),
            order=(1, 0),
        )

        k = tl.load(
            k_ptrs,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        qk_dot = tl.dot(
            q,
            tl.trans(k),
        )
        selected_rows = tl.reshape(
            tl.broadcast_to(selected_q[:, None], (GROUP_SIZE_Q, BLOCK_SIZE_H)),
            (BLOCK_SIZE_M,),
        )
        qk_mask_flat = (
            selected_rows[:, None]
            & row_valid[:, None]
            & (positions[None, :] <= row_abs[:, None])
            & (positions[None, :] < seq_len)
        )
        qk = tl.where(qk_mask_flat, qk_dot * sm_scale_log2e, float("-inf"))

        m_ij = tl.maximum(
            m_i,
            tl.max(qk, axis=1),
        )

        alpha = tl.where(
            m_i > float("-inf"),
            tl.exp2(m_i - m_ij),
            0.0,
        )

        p = tl.where(
            qk_mask_flat,
            tl.exp2(qk - m_ij[:, None]),
            0.0,
        )
        l_ij = tl.sum(p, axis=1)

        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij

        v_ptrs = tl.make_block_ptr(
            base=(kv_base + head_dim * stride_kv_d),
            shape=(
                BLOCK_SIZE_K,
                head_dim,
            ),
            strides=(
                stride_kv_pos,
                stride_kv_d,
            ),
            offsets=(0, 0),
            block_shape=(
                BLOCK_SIZE_K,
                BLOCK_SIZE_D,
            ),
            order=(1, 0),
        )

        v = tl.load(
            v_ptrs,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        acc_o += tl.dot(
            p.to(v.dtype),
            v,
        )
        m_i = m_ij

    inv_l = tl.where(
        l_i > 0,
        1.0 / l_i,
        0.0,
    )
    acc_o = acc_o * inv_l[:, None]
    acc_o = tl.reshape(
        acc_o,
        GROUP_SIZE_Q,
        BLOCK_SIZE_H,
        BLOCK_SIZE_D,
    )

    o_ptrs = tl.make_block_ptr(
        base=(o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh),
        shape=(
            q_len,
            gqa_group_size,
            head_dim,
        ),
        strides=(
            stride_on,
            stride_oh,
            stride_od,
        ),
        offsets=(
            group_start,
            0,
            0,
        ),
        block_shape=(
            GROUP_SIZE_Q,
            BLOCK_SIZE_H,
            BLOCK_SIZE_D,
        ),
        order=(2, 1, 0),
    )

    tl.store(
        o_ptrs,
        acc_o.to(o_ptr.dtype.element_ty),
        boundary_check=(0, 1, 2),
    )


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
) -> None:
    """S60 BF16 prefill using the direct Triton kernel."""

    if k_scale is not None or v_scale is not None:
        raise NotImplementedError("S60 initial prefill adaptation does not support FP8 scales")

    if q.dtype != torch.bfloat16:
        raise ValueError("S60 initial prefill path requires BF16 q")

    if kv_cache.dtype != torch.bfloat16:
        raise ValueError("S60 initial prefill path requires BF16 KV cache")

    if output.dtype != q.dtype:
        raise ValueError("output dtype must match q dtype")

    for name, tensor in (
        ("topk_idx", topk_idx),
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seq_lens", seq_lens),
        ("prefix_lens", prefix_lens),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must use int32 on S60")

    total_q, num_heads, head_dim = q.shape

    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")

    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    batch = cu_seqlens_q.shape[0] - 1
    gqa_group_size = num_heads // num_kv_heads
    block_size_h = triton.next_power_of_2(gqa_group_size)
    topk = topk_idx.shape[-1]

    # A bounded page union reuses KV payloads without scanning a large
    # unrelated context. Keep a bounded candidate pool and target wider
    # matrix tiles while preserving each query's own selected-page mask.
    grouped_query_size = 8 if block_size_h <= 8 else 4
    # Dense selected-page pools can reuse each KV page across more queries.
    # Require at least 64 nonempty query/head programs to retain parallelism.
    if (
        block_size_h <= 8
        and 0 < topk <= 16
        and block_table.shape[1] <= 2 * topk
        and total_q >= 16 * batch
        and triton.cdiv(total_q, 16) * num_kv_heads >= 64
    ):
        grouped_query_size = 16
    use_grouped8 = (
        head_dim == 128
        and max_query_len >= grouped_query_size
        and (
            (topk == 4 and block_size_h == 8)
            or (0 < topk <= 16 and block_size_h <= 32 and block_table.shape[1] <= 2 * topk)
        )
    )

    if use_grouped8:
        grouped_grid = (
            triton.cdiv(max_query_len, grouped_query_size),
            num_kv_heads,
            batch,
        )

        _gqa_sparse_fwd_grouped8[grouped_grid](
            q,
            kv_cache,
            topk_idx,
            output,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            0,
            gqa_group_size,
            head_dim,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            GROUP_SIZE_Q=grouped_query_size,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            BLOCK_SIZE_H=block_size_h,
            MAX_TOPK=topk,
            BLOCK_TOPK=triton.next_power_of_2(topk),
            DECODE_MODE=False,
            num_warps=1,
            num_stages=1,
        )
        return

    # Bound FP32 workspace and retain enough independent matrix programs.
    # The kernel verifies ordered prefix equality at runtime; no page IDs are assumed.
    prefix_group = 16
    prefix_groups = triton.cdiv(max_query_len, prefix_group)
    use_shared_prefix = (
        head_dim == 128
        and block_size_h <= 8
        and topk >= 8
        and triton.cdiv(total_q, prefix_group) * num_kv_heads >= 64
        and total_q * num_heads * (head_dim + 2) * 4 <= 64 * 1024 * 1024
    )
    partial = maximum = denom = counts = output
    if use_shared_prefix:
        partial = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        maximum = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        denom = torch.empty_like(maximum)
        counts = torch.empty((batch, num_kv_heads, prefix_groups), device=q.device, dtype=torch.int32)
        _gqa_sparse_common_prefix[(prefix_groups, num_kv_heads, batch)](
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            partial,
            maximum,
            denom,
            counts,
            gqa_group_size,
            head_dim,
            topk,
            sm_scale,
            *q.stride(),
            *kv_cache.stride(),
            *topk_idx.stride(),
            block_table.stride(0),
            *partial.stride(),
            maximum.stride(0),
            num_kv_heads,
            prefix_groups,
            prefix_group,
            block_size_h,
            triton.next_power_of_2(head_dim),
            num_warps=1,
            num_stages=1,
        )

    grid = (
        max_query_len,
        num_kv_heads,
        batch,
    )

    _gqa_sparse_fwd_direct[grid](
        q,
        kv_cache,
        output,  # unused BF16 k_scale placeholder
        output,  # unused BF16 v_scale placeholder
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
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        0,  # stride_ks_h
        0,  # stride_ks_t
        0,  # stride_vs_h
        0,  # stride_vs_t
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_QH=block_size_h,
        USE_FP8=False,
        KV_SCALE_MODE=0,
        num_warps=1,
        num_stages=1,
        partial_ptr=partial,
        maximum_ptr=maximum,
        denom_ptr=denom,
        count_ptr=counts,
        GROUP_Q=prefix_group,
        NUM_GROUPS=prefix_groups,
        STRIDE_PN=partial.stride(0),
        STRIDE_PH=partial.stride(1),
        STRIDE_PD=partial.stride(2),
        STRIDE_SN=maximum.stride(0),
        USE_SHARED_PREFIX=use_shared_prefix,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            8 if args["DIRECT_OUTPUT"] else 16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit
def _gqa_sparse_decode_kernel_enflame(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len: tl.constexpr,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    DIRECT_OUTPUT: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, blk)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int32)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                k_scale = tl.load(
                    k_scale_ptr + pid_kh * stride_ks_h + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                    mask=pos_mask,
                    other=1.0,
                )
                k = (k * k_scale[None, :]).to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp2(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                v_scale = tl.load(
                    v_scale_ptr + pid_kh * stride_vs_h + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask,
                    other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(l_i > 0, 1.0 / l_i, tl.zeros_like(l_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    if not DIRECT_OUTPUT:
        lse_i = m_i + tl.log2(l_i)
        lse_ptrs = tl.make_block_ptr(
            base=(lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h),
            shape=(gqa_group_size,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_out_kernel_enflame(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_heads_kernel_enflame(
    o_ptr,
    lse_ptr,
    out_ptr,
    head_dim,
    num_heads: tl.constexpr,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Merge adjacent heads with regular partial-output and LSE tiles."""
    query = tl.program_id(0)
    head = tl.program_id(1) * BLOCK_SIZE_H
    partial = tl.make_block_ptr(
        o_ptr + query * stride_o_b,
        (NUM_TOPK_CHUNKS, num_heads, head_dim),
        (stride_o_c, stride_o_h, stride_o_d),
        (0, head, 0),
        (NUM_TOPK_CHUNKS, BLOCK_SIZE_H, BLOCK_SIZE_D),
        (2, 1, 0),
    )
    lse_block = tl.make_block_ptr(
        lse_ptr + query * stride_l_b,
        (NUM_TOPK_CHUNKS, num_heads),
        (stride_l_c, stride_l_h),
        (0, head),
        (NUM_TOPK_CHUNKS, BLOCK_SIZE_H),
        (1, 0),
    )
    o = tl.load(partial, boundary_check=(0, 1, 2), padding_option="zero")
    lse = tl.load(lse_block, boundary_check=(0, 1), padding_option="zero")
    maximum = tl.max(lse, axis=0)
    weights = tl.exp2(lse - maximum[None, :])
    weights = weights / tl.sum(weights, axis=0)[None, :]
    merged = tl.sum(o.to(tl.float32) * weights[:, :, None], axis=0)
    output = tl.make_block_ptr(
        out_ptr + query * stride_out_n,
        (num_heads, head_dim),
        (stride_out_h, stride_out_d),
        (head, 0),
        (BLOCK_SIZE_H, BLOCK_SIZE_D),
        (1, 0),
    )
    tl.store(output, merged.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    total_q, num_heads, head_dim = q.shape

    if total_q != seq_lens.shape[0] * decode_query_len:
        raise ValueError("total_q is incompatible with decode metadata")
    if num_kv_heads > 255 or num_heads > 255:
        raise ValueError("decode head grid exceeds GCU300 limits")
    if total_q > 65535:
        raise ValueError("decode token grid exceeds GCU300 limits")

    max_topk = topk_idx.shape[-1]
    group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES

    # Adjacent speculative queries commonly select the same pages. Group them
    # so K/V pages are loaded once while preserving per-query causal masks.
    use_grouped_decode = (
        not use_fp8
        and q.dtype == torch.bfloat16
        and kv_cache.dtype == torch.bfloat16
        and max_topk == 4
        and group_size == 8
        and head_dim == 128
        and 2 <= decode_query_len <= 8
    )
    if use_grouped_decode:
        batch = seq_lens.shape[0]
        # Two-query groups give D4 enough independent programs on S60 while
        # retaining adjacent-query K/V reuse.
        grouped_query_size = 2 if decode_query_len == 4 else triton.next_power_of_2(decode_query_len)
        grouped_grid = (
            triton.cdiv(
                decode_query_len,
                grouped_query_size,
            ),
            num_kv_heads,
            batch,
        )
        _gqa_sparse_fwd_grouped8[grouped_grid](
            q,
            kv_cache,
            topk_idx,
            output,
            block_table,
            seq_lens,  # Unused cu_seqlens placeholder in decode mode.
            seq_lens,
            seq_lens,  # Unused prefix placeholder in decode mode.
            decode_query_len,
            group_size,
            head_dim,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            GROUP_SIZE_Q=grouped_query_size,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=128,
            BLOCK_SIZE_H=8,
            MAX_TOPK=4,
            BLOCK_TOPK=4,
            DECODE_MODE=True,
            num_warps=1,
            num_stages=1,
        )
        return

    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(
            output,
            num_kv_heads,
            k_scale,
            v_scale,
        )
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )

    # With enough query/head programs, process all selected pages online and
    # write the final normalized result directly, avoiding partial buffers and
    # the merge launch. This is the profitable D3 path on S60.
    direct_output = (
        not use_fp8 and max_topk <= 4 and group_size == 8 and head_dim == 128 and total_q * num_kv_heads >= 16
    )

    if direct_output:
        num_topk_chunks = 1
        partial_output = output
        partial_lse = output  # Unused in the DIRECT_OUTPUT constexpr branch.
        stride_o_c = 0
        stride_o_b = output.stride(0)
        stride_o_h = output.stride(1)
        stride_o_d = output.stride(2)
        stride_l_c = 0
        stride_l_b = 0
        stride_l_h = 0
    else:
        max_chunks_by_grid = max(
            1,
            65535 // max(1, total_q),
        )
        target = max(
            1,
            min(
                max_topk,
                max_chunks_by_grid,
                (32 if q.dtype == torch.bfloat16 and not use_fp8 else 256)
                // max(
                    1,
                    total_q * num_kv_heads,
                ),
            ),
        )
        num_topk_chunks = 1 << (target.bit_length() - 1)

        if num_topk_chunks == 1 and not use_fp8:
            # A single chunk already owns the complete normalized result.
            direct_output = True
            partial_output = output
            partial_lse = output  # Unused in the DIRECT_OUTPUT branch.
            stride_o_c = 0
            stride_o_b = output.stride(0)
            stride_o_h = output.stride(1)
            stride_o_d = output.stride(2)
            stride_l_c = 0
            stride_l_b = 0
            stride_l_h = 0
        else:
            partial_output = torch.empty(
                (
                    num_topk_chunks,
                    total_q,
                    num_heads,
                    head_dim,
                ),
                dtype=q.dtype,
                device=q.device,
            )
            partial_lse = torch.empty(
                (
                    num_topk_chunks,
                    total_q,
                    num_heads,
                ),
                dtype=torch.float32,
                device=q.device,
            )
            stride_o_c = partial_output.stride(0)
            stride_o_b = partial_output.stride(1)
            stride_o_h = partial_output.stride(2)
            stride_o_d = partial_output.stride(3)
            stride_l_c = partial_lse.stride(0)
            stride_l_b = partial_lse.stride(1)
            stride_l_h = partial_lse.stride(2)

    grid = (
        total_q * num_topk_chunks,
        num_kv_heads,
    )

    _gqa_sparse_decode_kernel_enflame[grid](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        partial_output,
        partial_lse,
        block_table,
        seq_lens,
        total_q,
        group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
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
        stride_o_c,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        stride_l_c,
        stride_l_b,
        stride_l_h,
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        DIRECT_OUTPUT=direct_output,
        num_warps=1,
        num_stages=1,
    )

    if direct_output:
        return

    # Bound the live FP32 merge tile and retain the original singleton path.
    merge_head_tile = (
        min(8, triton.next_power_of_2(num_heads), max(1, 32768 // (num_topk_chunks * triton.next_power_of_2(head_dim))))
        if not use_fp8 and q.dtype == torch.bfloat16
        else 1
    )
    # Retain at least 32 merge programs when coarsening adjacent heads.
    parallel_budget = max(1, total_q * num_heads // 32)
    parallel_tile = 1 << (parallel_budget.bit_length() - 1)
    merge_head_tile = min(merge_head_tile, parallel_tile)
    if merge_head_tile > 1:
        _merge_topk_attn_heads_kernel_enflame[(total_q, triton.cdiv(num_heads, merge_head_tile))](
            partial_output,
            partial_lse,
            output,
            head_dim,
            num_heads,
            stride_o_c,
            stride_o_b,
            stride_o_h,
            stride_o_d,
            stride_l_c,
            stride_l_b,
            stride_l_h,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            BLOCK_SIZE_H=merge_head_tile,
            num_warps=1,
            num_stages=1,
        )
        return

    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel_enflame[merge_grid](
        partial_output,
        partial_lse,
        output,
        head_dim,
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        num_warps=1,
        num_stages=1,
    )
