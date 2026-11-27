# Copyright 2026 FlagOS Contributors
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


"""PPU Parallax prefill and backward using portable Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.utils import input_guard

# Packed chunk indices and all query/KV passes use the same tile size.
_BLOCK_T = 64


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAVE_STATS: tl.constexpr = True,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_o = o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_barv = barv + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        # Complete the w path before creating rk/wr so fewer [BT, BS]
        # temporaries compete with the two persistent output accumulators.
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        Rv_acc = alpha * Rv_acc
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    # Phase A: safe blocks (no mask).
    for _safe in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        o_kv = _safe.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        Rv_acc = alpha * Rv_acc
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        Rv_acc = alpha * Rv_acc
        Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_qk)
    if SAVE_STATS:
        tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), mask=m_qk)
        tl.store(p_d1, d1_acc, mask=row_mask)
        tl.store(p_bart, b_bart, mask=row_mask)
        tl.store(p_m, m_acc, mask=row_mask)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel_split_k(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    SAVE_STATS: tl.constexpr,
):
    """Dense full-window forward that streams two 64-wide head-dim chunks.

    Accumulate the score over two head-dimension halves before normalization,
    then update and store the corresponding output halves independently.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)
    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS
    HK: tl.constexpr = BK // 2
    o_k0 = tl.arange(0, HK)
    o_k1 = HK + tl.arange(0, HK)
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K)
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K)
    b_q0 = tl.load(p_q + o_k0[None, :], mask=row_mask, other=0.0)
    b_q1 = tl.load(p_q + o_k1[None, :], mask=row_mask, other=0.0)
    b_r0 = tl.load(p_r + o_k0[None, :], mask=row_mask, other=0.0)
    b_r1 = tl.load(p_r + o_k1[None, :], mask=row_mask, other=0.0)

    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv0 = tl.zeros((BT, HK), dtype=tl.float32)
    barv1 = tl.zeros((BT, HK), dtype=tl.float32)
    Rv0 = tl.zeros((BT, HK), dtype=tl.float32)
    Rv1 = tl.zeros((BT, HK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for col_block_id in range(0, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = col_indices[:, None] < T
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K)
        b_k0 = tl.load(p_k + o_k0[None, :], mask=m_kv, other=0.0)
        b_k1 = tl.load(p_k + o_k1[None, :], mask=m_kv, other=0.0)
        qk = tl.dot(b_q0, tl.trans(b_k0), out_dtype=tl.float32)
        qk = tl.dot(b_q1, tl.trans(b_k1), out_dtype=tl.float32, acc=qk) * scale_log2
        rk = tl.dot(b_r0, tl.trans(b_k0), out_dtype=tl.float32)
        rk = tl.dot(b_r1, tl.trans(b_k1), out_dtype=tl.float32, acc=rk)
        if col_block_id >= NUM_SAFE_BLOCKS:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
            qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K)
        b_v0 = tl.load(p_v + o_k0[None, :], mask=m_kv, other=0.0)
        b_v1 = tl.load(p_v + o_k1[None, :], mask=m_kv, other=0.0)
        barv0 = alpha * barv0
        barv1 = alpha * barv1
        barv0 = tl.dot(w.to(b_v0.dtype), b_v0, out_dtype=tl.float32, acc=barv0)
        barv1 = tl.dot(w.to(b_v1.dtype), b_v1, out_dtype=tl.float32, acc=barv1)
        Rv0 = alpha * Rv0
        Rv1 = alpha * Rv1
        Rv0 = tl.dot(wr.to(b_v0.dtype), b_v0, out_dtype=tl.float32, acc=Rv0)
        Rv1 = tl.dot(wr.to(b_v1.dtype), b_v1, out_dtype=tl.float32, acc=Rv1)
        m_acc = m_new

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_bart = d2_acc * inv_d1
    b_barv0 = barv0 * inv_d1
    b_barv1 = barv1 * inv_d1
    p_o = o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K)
    o0 = b_barv0 + b_bart * b_barv0 - Rv0 * inv_d1
    o1 = b_barv1 + b_bart * b_barv1 - Rv1 * inv_d1
    tl.store(p_o + o_k0[None, :], o0.to(o.dtype.element_ty), mask=row_mask)
    tl.store(p_o + o_k1[None, :], o1.to(o.dtype.element_ty), mask=row_mask)
    if SAVE_STATS:
        p_barv = barv + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K)
        tl.store(p_barv + o_k0[None, :], b_barv0.to(barv.dtype.element_ty), mask=row_mask)
        tl.store(p_barv + o_k1[None, :], b_barv1.to(barv.dtype.element_ty), mask=row_mask)
        p_s = bos * HQ + i_hq + row_indices[:, None] * HQ
        tl.store(d1 + p_s, d1_acc, mask=row_mask)
        tl.store(bart + p_s, b_bart, mask=row_mask)
        tl.store(m + p_s, m_acc, mask=row_mask)

@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_preprocess(
    grad_o,
    o,
    barv,
    delta_t,
    delta_b,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BP: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)

    row_offset = i_t * BP
    o_t = row_offset + tl.arange(0, BP)
    o_k = tl.arange(0, BK)
    m_tk = (o_t[:, None] < T) & (o_k[None, :] < K)
    m_t = o_t[:, None] < T
    p_grad_o = grad_o + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_o = o + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_barv = barv + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_t = delta_t + bos * HQ + i_hq + o_t[:, None] * HQ
    p_b = delta_b + bos * HQ + i_hq + o_t[:, None] * HQ

    b_grad_o = tl.load(p_grad_o, mask=m_tk, other=0.0).to(tl.float32)
    b_o = tl.load(p_o, mask=m_tk, other=0.0).to(tl.float32)
    b_barv = tl.load(p_barv, mask=m_tk, other=0.0).to(tl.float32)

    b_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
    b_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)

    tl.store(p_t, b_t, mask=m_t)
    tl.store(p_b, b_b, mask=m_t)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_q = grad_q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_r = grad_r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
    b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
    b_m = tl.load(p_m, mask=row_mask, other=0.0)
    grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)
    b_t = tl.load(p_t, mask=row_mask, other=0.0)
    b_b = tl.load(p_b, mask=row_mask, other=0.0)
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)

    # Phase 0: left-border blocks (SWA only).
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        # Finish dR before materializing rk/gl. This shortens the live ranges
        # of several [BT, BS] matrices and substantially reduces register
        # pressure in the long-sequence kernel.
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    # Phase A: safe blocks (no mask).
    for _ in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        o_kv = _.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    grad_q_acc = scale * grad_q_acc

    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), mask=m_qk)
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), mask=m_qk)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def parallel_parallax_bwd_kernel_dkv(q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k, grad_v, scale, cu_seqlens, chunk_indices, T, HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, WINDOW_SIZE_LEFT: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, IS_VARLEN: tl.constexpr):
    physical_i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_t = physical_i_t
    i_b, i_hq = (i_bh // HQ, i_bh % HQ)
    i_h = i_hq // G
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    col_offset = i_t * BS
    col_indices = col_offset + tl.arange(0, BS)
    start_row_block = col_offset // BT
    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_k = grad_k + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2
    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id.to(tl.int64) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (row_indices[:, None] >= col_indices[None, :]) & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1) & row_mask & (col_indices[None, :] < T)
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float('inf'))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id.to(tl.int64) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id.to(tl.int64) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        mask = (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float('inf'))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)
    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.jit(do_not_specialize=["N"])
def parallel_parallax_bwd_kernel_reduce_gqa(
    grad_k_buf,
    grad_v_buf,
    grad_k,
    grad_v,
    N,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold per-query-head dK/dV into their shared GQA heads in one pass."""
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < N

    head_dim_offset = offsets % K
    row_head = offsets // K
    kv_head = row_head % H
    batch_row = row_head // H
    input_offsets = (batch_row * HQ + kv_head * G) * K + head_dim_offset

    grad_k_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    grad_v_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i_g in range(0, G):
        group_offsets = input_offsets + i_g * K
        grad_k_acc += tl.load(grad_k_buf + group_offsets, mask=mask, other=0.0)
        grad_v_acc += tl.load(grad_v_buf + group_offsets, mask=mask, other=0.0)

    tl.store(grad_k + offsets, grad_k_acc, mask=mask)
    tl.store(grad_v + offsets, grad_v_acc, mask=mask)


def parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1, save_stats=True):
    """Return output and FP32 statistics for dense or packed causal attention."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    BT = _BLOCK_T
    BK = triton.next_power_of_2(K)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    grid = (len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT), B * HQ)
    o = torch.empty_like(q)
    if save_stats:
        barv = torch.empty_like(q)
        d1, bart, m = torch.empty((3, B, T, HQ), device=q.device, dtype=torch.float32).unbind(0)
    else:
        barv = d1 = bart = m = None
    if cu_seqlens is None and window_size_left < 0 and K == BK and BK == 128:
        parallel_parallax_fwd_kernel_split_k[grid](
            q, r, k, v, o, barv if save_stats else q,
            d1 if save_stats else q, bart if save_stats else q, m if save_stats else q,
            scale, T,
            HQ=HQ, H=H, G=HQ // H, K=K, BK=BK, BT=BT, BS=BT,
            SAVE_STATS=save_stats, num_warps=4, num_stages=2,
        )
        return o, barv, d1, bart, m
    parallel_parallax_fwd_kernel[grid](
        q, r, k, v, o, barv if save_stats else q,
        d1 if save_stats else q, bart if save_stats else q, m if save_stats else q,
        scale, cu_seqlens, chunk_indices, T,
        HQ=HQ, H=H, G=HQ // H, K=K, BK=BK, BT=BT, BS=BT, SAVE_STATS=save_stats,
        WINDOW_SIZE_LEFT=window_size_left, num_warps=8, num_stages=1,
    )
    return o, barv, d1, bart, m


def parallel_parallax_bwd(
    q, r, k, v, o, barv, d1, bart, m, grad_o, scale,
    cu_seqlens=None, chunk_indices=None, window_size_left=-1,
):
    """Compute all four gradients, then fold shared GQA heads in FP32."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BT = _BLOCK_T
    BK = triton.next_power_of_2(K)
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    grid = (len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT), B * HQ)
    grad_q, grad_r = torch.empty_like(q), torch.empty_like(r)
    grad_k_buf, grad_v_buf = torch.empty_like(q), torch.empty_like(q)
    delta_t, delta_b = torch.empty((2, B, T, HQ), device=q.device, dtype=torch.float32).unbind(0)
    parallel_parallax_bwd_kernel_preprocess[grid](
        grad_o, o, barv, delta_t, delta_b, cu_seqlens, chunk_indices, T,
        HQ=HQ, K=K, BK=BK, BT=BT, BP=BT, num_warps=8, num_stages=1,
    )
    common = dict(HQ=HQ, H=H, G=G, K=K, BK=BK, BT=BT, BS=BT,
                  WINDOW_SIZE_LEFT=window_size_left,
                  num_warps=8, num_stages=1)
    parallel_parallax_bwd_kernel_dqr[grid](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
        scale, cu_seqlens, chunk_indices, T, **common,
    )
    parallel_parallax_bwd_kernel_dkv[grid](
        q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
        scale, cu_seqlens, chunk_indices, T, **common,
    )
    if G == 1:
        return grad_q, grad_r, grad_k_buf, grad_v_buf
    grad_k, grad_v = torch.empty_like(k), torch.empty_like(v)
    n = grad_k.numel()
    parallel_parallax_bwd_kernel_reduce_gqa[(triton.cdiv(n, 256),)](
        grad_k_buf, grad_v_buf, grad_k, grad_v, n,
        HQ=HQ, H=H, G=G, K=K, BLOCK=256, num_warps=8,
    )
    return grad_q, grad_r, grad_k, grad_v


def _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens):
    tensors = {"q": q, "r": r, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, T, H, D], got {tuple(tensor.shape)}")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got device={tensor.device}")

    if r.shape != q.shape:
        raise ValueError(f"r shape {tuple(r.shape)} must match q shape {tuple(q.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}")
    if any(tensor.device != q.device for tensor in (r, k, v)):
        raise ValueError("q, r, k, and v must be on the same CUDA device")
    if any(tensor.dtype != q.dtype for tensor in (r, k, v)):
        raise TypeError("q, r, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")

    B, T, HQ, K = q.shape
    if k.shape[0] != B or k.shape[1] != T or k.shape[3] != K:
        raise ValueError(
            f"k/v shape {tuple(k.shape)} is incompatible with q/r shape {tuple(q.shape)}; B, T, and D must match"
        )
    H = k.shape[2]
    if T <= 0 or H <= 0 or HQ <= 0:
        raise ValueError("sequence length and query/KV head counts must be positive")
    if HQ % H != 0:
        raise ValueError(f"GQA requires HQ % H == 0, got HQ={HQ}, H={H}")
    if not 16 <= K <= 256:
        raise ValueError(f"head dimension must be in [16, 256], got D={K}")
    if window_size is not None and (not isinstance(window_size, int) or window_size <= 0):
        raise ValueError(f"window_size must be a positive integer or None, got {window_size!r}")

    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor or None")
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using cu_seqlens. "
                "Flatten variable-length inputs before processing."
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same CUDA device as q")


@input_guard
def _parallel_parallax_inference(q, r, k, v, scale, window_size_left, cu_seqlens):
    # Inference needs the same online moments for the output, but has no
    # consumer for saved backward statistics or their global stores.
    return parallel_parallax_fwd(
        q, r, k, v, scale, cu_seqlens=cu_seqlens,
        window_size_left=window_size_left, save_stats=False,
    )[0]


class ParallaxFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, _BLOCK_T)
            if cu_seqlens is not None
            else None
        )
        o, barv, d1, bart, m = parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left)
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        gq, gr, gk, gv = parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            do,
            ctx.scale,
            ctx.cu_seqlens,
            ctx.chunk_indices,
            ctx.window_size_left,
        )
        return gq.to(q), gr.to(r), gk.to(k), gv.to(v), None, None, None


def parallel_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Causal Parallax (parameterized local linear attention) with autograd,
    backed by Triton kernels. See `fla.ops.parallax.naive.naive_parallax` for
    the reference math.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            secondary queries of shape `[B, T, HQ, D]` (same shape as `q`). NOTE:
            `r` is *not* scaled by `scale`; pass it un-pre-scaled.
        k (torch.Tensor):
            keys of shape `[B, T, H, D]`. GQA is applied when `HQ` is divisible by `H`.
        v (torch.Tensor):
            values of shape `[B, T, H, D]`.
        scale (float, Optional):
            Scale applied to the `q @ k^T` logits only. If `None`, defaults to `1 / sqrt(D)`.
            Default: `None`.
        window_size (int, Optional):
            Sliding-window length. If provided, each query at position `i` only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training
            (FlashAttention convention). The batch size must be 1 when packing. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"parallel_parallax got unexpected keyword argument(s): {unexpected}")
    # Backend callers also use -1 as the unrestricted causal-window sentinel.
    if isinstance(window_size, int) and window_size == -1:
        window_size = None
    _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, r, k, v)):
        return _parallel_parallax_inference(q, r, k, v, float(scale), window_size_left, cu_seqlens)
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
