# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Portable MiniMax M3 paged attention using only standard Triton."""

import torch
import triton
import triton.language as tl

SPARSE_BLOCK_SIZE = 128


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


_KV_SCALE_NONE = 0


_KV_SCALE_SCALAR = 1


_KV_SCALE_PER_TOKEN_HEAD = 2


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_fallback_kernel(
    q_ptr,
    kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
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
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < seq_len
            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            k_ptrs = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(head_dim, BLOCK_SIZE_K),
                strides=(stride_kv_d, stride_kv_pos),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            )
            k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                k = k.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k = (k * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    k = (k * k_scale[None, :]).to(q.dtype)
            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_i = l_i * alpha + l_ij
            v_base_ptr = k_base_ptr + head_dim * stride_kv_d
            v_ptrs = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                v = v.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v = (v * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    v = (v * v_scale[:, None]).to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
        acc_o = acc_o * tl.where(l_i > 0, 1.0 / l_i, 0.0)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
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
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
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
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
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
            base=q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        # Keep the tl.dot kernel's PV accumulator in [QH, D] order. The
        # transposed [D, QH] form spills heavily for the FP8 QH=32 tile.
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)

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
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
            if USE_FP8:
                # Widen storage before arithmetic; no FP8 matrix unit is required.
                qk_dot = tl.dot(
                    q.to(tl.float32), tl.trans(k.to(tl.float32)), input_precision="ieee"
                )
            else:
                qk_dot = tl.dot(q, tl.trans(k))

            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len

            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))

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
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_i = l_i * alpha + l_ij

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
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )

            if USE_FP8:
                pv_p = p
                if KV_SCALE_MODE == 2:
                    pv_p *= v_scale[None, :]
                acc_o += tl.dot(pv_p, v.to(tl.float32), input_precision="ieee")
            else:
                acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij

        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        if USE_FP8 and KV_SCALE_MODE == 1:
            inv_l *= v_scale_scalar
        acc_o = acc_o * inv_l[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


def _kv_scale_args(
    output: torch.Tensor,
    num_kv_heads: int,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    if k_scale is None and v_scale is None:
        return output, output, 0, 0, 0, 0, _KV_SCALE_NONE
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale must be both provided or both None")
    if k_scale.device != output.device or v_scale.device != output.device:
        raise ValueError("k_scale and v_scale must be on the same dvice as output")
    if k_scale.numel() == 1 and v_scale.numel() == 1:
        return k_scale, v_scale, 0, 0, 0, 0, _KV_SCALE_SCALAR
    if k_scale.dim() == 2 and v_scale.dim() == 2:
        if k_scale.shape[0] != num_kv_heads or v_scale.shape[0] != num_kv_heads:
            raise ValueError(
                "per-token/head KV scales must have shape "
                f"[{num_kv_heads}, max_kv_tokens]"
            )
        if k_scale.shape != v_scale.shape:
            raise ValueError("k_scale and v_scale must have matching shapes")
        return (
            k_scale,
            v_scale,
            k_scale.stride(0),
            k_scale.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            _KV_SCALE_PER_TOKEN_HEAD,
        )
    raise ValueError(
        "MiniMax-M3 sparse attention supports scalar KV scales or "
        "[num_kv_heads, max_kv_tokens] per-token/head scales"
    )


@triton.jit
def _k2q_count_kernel(
    topk_idx_ptr,  # [num_kv_heads, total_q, topk] int (-1 == unused)
    block_table_ptr,  # [batch, max_blocks] int32
    q_batch_ptr,  # [total_q] int32  (query -> batch)
    row_counts_ptr,  # [num_rows] int32  (atomic output)
    total_q,
    topk,
    num_kv_heads,
    max_blocks,
    stride_tk_h,
    stride_tk_q,
    stride_tk_s,
    stride_bt_b,
    stride_bt_m,
    BLOCK_SIZE_T: tl.constexpr,  # next_pow2(topk)
):
    pid = tl.program_id(0)  # flattened (kv_head, query)
    kh = pid // total_q
    q = pid % total_q
    batch_id = tl.load(q_batch_ptr + q)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    t_mask = off_t < topk
    blocks = tl.load(
        topk_idx_ptr + kh * stride_tk_h + q * stride_tk_q + off_t * stride_tk_s,
        mask=t_mask,
        other=-1,
    )
    valid = t_mask & (blocks >= 0)
    # logical block -> physical page via this query's batch row of block_table
    pages = tl.load(
        block_table_ptr + batch_id * stride_bt_b + blocks * stride_bt_m,
        mask=valid,
        other=0,
    )
    rows = pages * num_kv_heads + kh
    tl.atomic_add(row_counts_ptr + rows, 1, mask=valid)


@triton.jit
def _k2q_scatter_kernel(
    topk_idx_ptr,  # [num_kv_heads, total_q, topk] int
    block_table_ptr,  # [batch, max_blocks] int32
    q_batch_ptr,  # [total_q] int32
    cu_seqlens_ptr,  # [batch + 1] int32
    prefix_lens_ptr,  # [batch] int32
    row_ptr_ptr,  # [num_rows + 1] int32  (prefix sum, read-only)
    cursor_ptr,  # [num_rows] int32  (atomic per-row write cursor)
    out_q_global_ptr,  # [nnz_cap] int32
    out_q_abs_ptr,  # [nnz_cap] int32
    out_kv_block_ptr,  # [nnz_cap] int32
    out_slot_ptr,  # [nnz_cap] int32
    total_q,
    topk,
    num_kv_heads,
    max_blocks,
    stride_tk_h,
    stride_tk_q,
    stride_tk_s,
    stride_bt_b,
    stride_bt_m,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid = tl.program_id(0)
    kh = pid // total_q
    q = pid % total_q
    batch_id = tl.load(q_batch_ptr + q)
    batch_start = tl.load(cu_seqlens_ptr + batch_id)
    prefix = tl.load(prefix_lens_ptr + batch_id)
    q_abs = prefix + (q - batch_start)  # absolute query position
    # Scatter each valid slot. A serial loop over topk keeps the atomic cursor
    # bumps correct (vectorized atomics to the same row would race on offset).
    for t in range(topk):
        block = tl.load(
            topk_idx_ptr + kh * stride_tk_h + q * stride_tk_q + t * stride_tk_s
        )
        if block >= 0:
            page = tl.load(
                block_table_ptr + batch_id * stride_bt_b + block * stride_bt_m
            )
            row = page * num_kv_heads + kh
            base = tl.load(row_ptr_ptr + row)
            off = tl.atomic_add(cursor_ptr + row, 1)
            pos = base + off
            tl.store(out_q_global_ptr + pos, q)
            tl.store(out_q_abs_ptr + pos, q_abs)
            tl.store(out_kv_block_ptr + pos, block)
            tl.store(out_slot_ptr + pos, t)


@torch.no_grad()
def build_k2q_csr_fused(
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] int, -1 == unused
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    block_table: torch.Tensor,  # [batch, max_blocks] int32
    num_pages: int,
):
    """GPU-only k2q CSR builder (count + scatter), no host sync.

    Output layout matches ``build_k2q_csr`` EXCEPT entries within a row are in
    scatter (nondeterministic) order rather than stable (q, slot) order -- the
    LSE merge is order-independent so the attend/merge result is identical.

    The payload arrays are sized to the dense upper bound
    ``num_kv_heads * total_q * topk`` so no ``nnz`` readback is needed; the
    attend kernel only reads within ``row_ptr`` bounds, so the unused tail is
    never touched. ``nnz`` is returned as a device tensor (``nnz_dev``) to keep
    the build sync-free; it is not required by the attend/merge path.
    """
    num_kv_heads, total_q, topk = topk_idx.shape
    device = topk_idx.device
    num_rows = num_pages * num_kv_heads
    max_blocks = block_table.shape[1]
    nnz_cap = num_kv_heads * total_q * topk

    # query -> batch via searchsorted: q_batch[i] = #{b : cu[b+1] <= i}. This
    # has a FIXED output size (total_q) unlike repeat_interleave (whose output
    # size depends on the cu values and forces a device sync), so the whole
    # build is CUDA-graph capturable.
    cu = cu_seqlens_q.to(torch.int32)
    pf = prefix_lens.to(torch.int32)
    q_ids = torch.arange(total_q, device=device, dtype=torch.int32)
    q_batch = torch.searchsorted(cu[1:].contiguous(), q_ids, right=True).to(torch.int32)

    ti = topk_idx.contiguous()
    bt = block_table.to(torch.int32).contiguous()
    block_t = triton.next_power_of_2(topk)
    grid = (num_kv_heads * total_q,)

    # pass 1: count entries per physical-page row.
    row_counts = torch.zeros(num_rows, device=device, dtype=torch.int32)
    _k2q_count_kernel[grid](
        ti,
        bt,
        q_batch,
        row_counts,
        total_q,
        topk,
        num_kv_heads,
        max_blocks,
        ti.stride(0),
        ti.stride(1),
        ti.stride(2),
        bt.stride(0),
        bt.stride(1),
        BLOCK_SIZE_T=block_t,
    )

    # row_ptr = [0, cumsum(row_counts)]  (single device cumsum, no readback).
    row_ptr = torch.zeros(num_rows + 1, device=device, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(row_counts, 0, dtype=torch.int32)

    # pass 2: scatter payloads into each row's slice.
    cursor = torch.zeros(num_rows, device=device, dtype=torch.int32)
    out_q_global = torch.empty(nnz_cap, device=device, dtype=torch.int32)
    out_q_abs = torch.empty(nnz_cap, device=device, dtype=torch.int32)
    out_kv_block = torch.empty(nnz_cap, device=device, dtype=torch.int32)
    out_slot = torch.empty(nnz_cap, device=device, dtype=torch.int32)
    _k2q_scatter_kernel[grid](
        ti,
        bt,
        q_batch,
        cu,
        pf,
        row_ptr,
        cursor,
        out_q_global,
        out_q_abs,
        out_kv_block,
        out_slot,
        total_q,
        topk,
        num_kv_heads,
        max_blocks,
        ti.stride(0),
        ti.stride(1),
        ti.stride(2),
        bt.stride(0),
        bt.stride(1),
        BLOCK_SIZE_T=block_t,
    )

    return {
        "row_ptr": row_ptr,
        "q_global": out_q_global,
        "q_abs": out_q_abs,
        "kv_block": out_kv_block,
        "slot": out_slot,
        "q_batch": q_batch,
        "num_rows": num_rows,
        "num_pages": num_pages,
        "num_kv_heads": num_kv_heads,
        "nnz_cap": nnz_cap,
    }


@torch.no_grad()
def build_k2q_csr(
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk] int, -1 == unused
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    block_table: torch.Tensor,  # [batch, max_blocks] int32
    num_pages: int,
):
    """Invert per-query top-k (q2k) into a per-(page, kv_head) CSR (k2q).

    Rows are keyed by the *physical* KV page (as in the CuTe build_k2q_csr,
    which consumes the page table), so every entry in a row references exactly
    one physical KV location and the kernel can load that block once. A query's
    logical top-k block is mapped to its physical page through ``block_table``
    using the query's batch.

      row_ptr   : [num_pages * num_kv_heads + 1] int32  prefix sum of counts
      q_global  : [nnz] int32   global query id of each entry
      q_abs     : [nnz] int32   absolute query position (prefix + local)
      kv_block  : [nnz] int32   logical block index (for causal key positions)
      slot      : [nnz] int32   which of the query's top-k this block is
      num_rows  : int           num_pages * num_kv_heads

    row index == physical_page * num_kv_heads + kv_head.
    """
    num_kv_heads, total_q, topk = topk_idx.shape
    device = topk_idx.device
    num_rows = num_pages * num_kv_heads

    # Per-query absolute position and batch id, fully vectorized (no host loop
    # or per-batch .item() sync -- that construction cost dominated at small
    # query_len, dwarfing the attend/merge kernels themselves).
    batch = cu_seqlens_q.numel() - 1
    cu = cu_seqlens_q.to(torch.int64)
    pf = prefix_lens.to(torch.int64)
    counts = cu[1:] - cu[:-1]  # queries per batch [batch]
    q_batch_all = torch.repeat_interleave(
        torch.arange(batch, device=device, dtype=torch.int64), counts
    )  # [total_q]
    # within-batch local query index = global id - batch start
    q_local = torch.arange(total_q, device=device, dtype=torch.int64) - cu[q_batch_all]
    q_abs_all = pf[q_batch_all] + q_local

    ti = topk_idx.to(torch.int64)
    flat_ti = ti.reshape(-1)

    # Flat layout of topk_idx is [num_kv_heads, total_q, topk]; the (kv_head,
    # query, slot) of each entry is pure arithmetic on its flat position, so a
    # single nonzero() + 3 divmods replaces three expand+masked_select passes.
    flat_pos = (flat_ti >= 0).nonzero(as_tuple=True)[0]  # valid flat indices
    block = flat_ti[flat_pos]
    kh_e = flat_pos // (total_q * topk)
    q_global = (flat_pos // topk) % total_q
    slot_e = flat_pos % topk
    q_abs_e = q_abs_all[q_global]
    q_batch_e = q_batch_all[q_global]

    # Logical block -> physical page via the query's batch row of block_table.
    bt = block_table.to(torch.int64)
    page_e = bt[q_batch_e, block]
    row = page_e * num_kv_heads + kh_e

    # Sort entries by row so each row is contiguous; stable keeps (q, slot)
    # order. Stack the four per-entry payloads so one index_select reorders all
    # of them (four separate gathers = four launches otherwise).
    order = torch.argsort(row, stable=True)
    row_s = row[order]
    payload = torch.stack((q_global, q_abs_e, block, slot_e)).index_select(1, order)
    q_global_s = payload[0].to(torch.int32)
    q_abs_s = payload[1].to(torch.int32)
    kv_block_s = payload[2].to(torch.int32)
    slot_s = payload[3].to(torch.int32)

    counts = torch.bincount(row_s, minlength=num_rows)
    row_ptr = torch.zeros(num_rows + 1, device=device, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(counts, 0).to(torch.int32)

    return {
        "row_ptr": row_ptr,
        "q_global": q_global_s.contiguous(),
        "q_abs": q_abs_s.contiguous(),
        "kv_block": kv_block_s.contiguous(),
        "slot": slot_s.contiguous(),
        # q_batch_all is already materialized here; hand it back so the caller
        # need not recompute the same repeat_interleave via _q_batch_index.
        "q_batch": q_batch_all.to(torch.int32),
        "num_rows": num_rows,
        "num_pages": num_pages,
        "num_kv_heads": num_kv_heads,
        "nnz": int(q_global_s.numel()),
    }


@triton.jit
def _kv_block_attn_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    row_ptr,  # [num_rows + 1] int32  CSR row offsets
    csr_q_global,  # [nnz] int32
    csr_q_abs,  # [nnz] int32
    csr_kv_block,  # [nnz] int32  logical block (for causal key positions)
    csr_slot,  # [nnz] int32
    q_batch_ptr,  # [total_q] int32  batch id of each query (for seq_len)
    seq_lens,  # [batch] int32
    partial_o_ptr,  # [total_q, num_heads, max_topk, head_dim] fp32
    partial_lse_ptr,  # [total_q, num_heads, max_topk] fp32 (log2 domain)
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_po_n,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_pl_n,
    stride_pl_h,
    stride_pl_s,
    sched_row_ptr,  # [num_work + 1]: per work-tile, the CSR row it belongs to
    sched_q_lo_ptr,  # [num_work]: per work-tile, the absolute CSR entry offset
    Q_TILE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # next_pow2(gqa_group_size)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # Flattened work-tile schedule (CuTe-style): one program per (row, q_tile)
    # actually-populated work unit, so no program early-returns on an empty row
    # and long rows are split into balanced Q_TILE chunks.
    pid_w = tl.program_id(0)
    pid_row = tl.load(sched_row_ptr + pid_w)  # flattened (physical_page, kv_head)
    q_lo = tl.load(sched_q_lo_ptr + pid_w)  # absolute CSR entry offset
    page = (pid_row // num_kv_heads).to(tl.int64)
    pid_kh = pid_row % num_kv_heads
    pid_h = pid_kh * gqa_group_size

    row_end = tl.load(row_ptr + pid_row + 1)
    n_q = min(Q_TILE, row_end - q_lo)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_qt = tl.arange(0, Q_TILE)
    d_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size
    qt_mask = off_qt < n_q

    # CSR entries for this tile. All entries in a row share the physical page
    # (row key), so one KV load serves the whole tile; kv_block is the logical
    # block index and sets the key positions for causal masking.
    ent = q_lo + off_qt
    ent_mask = off_qt < n_q
    q_global = tl.load(csr_q_global + ent, mask=ent_mask, other=0).to(tl.int32)
    q_abs = tl.load(csr_q_abs + ent, mask=ent_mask, other=0).to(tl.int32)
    kv_block = tl.load(csr_kv_block + ent, mask=ent_mask, other=0).to(tl.int32)
    slot = tl.load(csr_slot + ent, mask=ent_mask, other=0).to(tl.int32)
    batch_id = tl.load(q_batch_ptr + q_global, mask=ent_mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens + batch_id, mask=ent_mask, other=0).to(tl.int32)

    # Key positions: logical block is identical across a row (same page implies
    # same logical block within a batch; across batches the block table is a
    # permutation, so a physical page maps to one logical block per batch, and
    # entries sharing a page share the logical block). Use the tile's first.
    c = kv_block * BLOCK_SIZE_K  # [Q_TILE], per-entry logical key offset
    pos = off_n  # [BLOCK_SIZE_K] within-block key position

    # Load this physical KV page once (the row key guarantees a single page).
    k = tl.load(
        kv_cache_ptr
        + page * stride_kv_blk
        + pid_kh * stride_kv_h
        + off_n[None, :] * stride_kv_pos
        + off_d[:, None] * stride_kv_d,
        mask=d_mask[:, None],
        other=0.0,
    )  # [BLOCK_SIZE_D, BLOCK_SIZE_K]
    v = tl.load(
        kv_cache_ptr
        + page * stride_kv_blk
        + pid_kh * stride_kv_h
        + off_n[:, None] * stride_kv_pos
        + (head_dim + off_d[None, :]) * stride_kv_d,
        mask=d_mask[None, :],
        other=0.0,
    )  # [BLOCK_SIZE_K, BLOCK_SIZE_D]

    # Gather Q for the query tile and every group head: [Q_TILE*H, D].
    q_rows = (q_global[:, None] * stride_qn) + (pid_h + off_h)[None, :] * stride_qh
    q_rows = q_rows.reshape(Q_TILE * BLOCK_SIZE_H)
    q = tl.load(
        q_ptr + q_rows[:, None] + off_d[None, :] * stride_qd,
        mask=(qt_mask[:, None] & h_mask[None, :]).reshape(Q_TILE * BLOCK_SIZE_H)[
            :, None
        ]
        & d_mask[None, :],
        other=0.0,
    )  # [Q_TILE*H, D]

    qk = tl.dot(q, k)  # [Q_TILE*H, BLOCK_SIZE_K]
    qk = qk.reshape(Q_TILE, BLOCK_SIZE_H, BLOCK_SIZE_K) * sm_scale_log2e

    # Causal + sequence masks per (query, key). Absolute key position is the
    # entry's logical block offset plus the within-block index. q attends key
    # pos p iff q_abs >= p and p < seq_len.
    key_abs = c[:, None] + pos[None, :]  # [Q_TILE, BLOCK_SIZE_K]
    allowed = (q_abs[:, None] >= key_abs) & (key_abs < seq_len[:, None])
    qk = tl.where(allowed[:, None, :], qk, float("-inf"))

    m = tl.max(qk, axis=2)  # [Q_TILE, H]
    p = tl.exp2(qk - m[:, :, None])  # [Q_TILE, H, K]
    l = tl.sum(p, axis=2)  # [Q_TILE, H]
    p = tl.reshape(p, Q_TILE * BLOCK_SIZE_H, BLOCK_SIZE_K)
    acc = tl.dot(p.to(v.dtype), v)  # [Q_TILE*H, D]  (unnormalized)
    acc = tl.reshape(acc, Q_TILE, BLOCK_SIZE_H, BLOCK_SIZE_D)

    # Normalize the partial by this block's l, matching the flash-decode merge
    # convention (o_chunk is softmax-normalized within the chunk; the LSE carries
    # the cross-chunk weight). Empty rows (l == 0) store 0 and an -inf LSE.
    inv_l = tl.where(l > 0, 1.0 / l, 0.0)
    acc = acc * inv_l[:, :, None]

    # log2-domain LSE for this single block; empty rows (m == -inf) -> -inf.
    lse = tl.where(l > 0, m + tl.log2(l), float("-inf"))  # [Q_TILE, H]

    # Scatter partials to [q_global, head, slot].
    po_base = (
        partial_o_ptr
        + q_global[:, None, None] * stride_po_n
        + (pid_h + off_h)[None, :, None] * stride_po_h
        + slot[:, None, None] * stride_po_s
        + off_d[None, None, :] * stride_po_d
    )
    po_mask = qt_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
    tl.store(po_base, acc, mask=po_mask)
    pl_base = (
        partial_lse_ptr
        + q_global[:, None] * stride_pl_n
        + (pid_h + off_h)[None, :] * stride_pl_h
        + slot[:, None] * stride_pl_s
    )
    tl.store(pl_base, lse, mask=qt_mask[:, None] & h_mask[None, :])


@triton.jit
def _kv_outer_merge_kernel(
    partial_o_ptr,  # [total_q, num_heads, max_topk, head_dim] fp32
    partial_lse_ptr,  # [total_q, num_heads, max_topk] fp32 log2
    out_ptr,  # [total_q, num_heads, head_dim]
    head_dim,
    max_topk,
    stride_po_n,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_pl_n,
    stride_pl_h,
    stride_pl_s,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    MAX_TOPK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_n = tl.program_id(0)  # query
    pid_h = tl.program_id(1)  # head
    off_s = tl.arange(0, MAX_TOPK)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    s_mask = off_s < max_topk
    d_mask = off_d < head_dim

    lse = tl.load(
        partial_lse_ptr
        + pid_n * stride_pl_n
        + pid_h * stride_pl_h
        + off_s * stride_pl_s,
        mask=s_mask,
        other=float("-inf"),
    )
    o = tl.load(
        partial_o_ptr
        + pid_n * stride_po_n
        + pid_h * stride_po_h
        + off_s[:, None] * stride_po_s
        + off_d[None, :] * stride_po_d,
        mask=s_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [MAX_TOPK, D]
    lse_max = tl.max(lse, axis=0)
    valid = lse > float("-inf")
    weights = tl.where(valid, tl.exp2(lse - lse_max), 0.0)
    wsum = tl.sum(weights, axis=0)
    inv = tl.where(wsum > 0, 1.0 / wsum, 0.0)
    # Zero partials from empty slots so uninitialized memory cannot create NaNs
    # (weight is 0 there, but 0 * inf == nan).
    o = tl.where(valid[:, None], o.to(tl.float32), 0.0)
    o_merged = tl.sum(o * weights[:, None], axis=0) * inv
    tl.store(
        out_ptr + pid_n * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d,
        o_merged.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


@triton.jit
def _kv_outer_merge_tiled_kernel(
    partial_o_ptr,  # [total_q, num_heads, max_topk, head_dim]
    partial_lse_ptr,  # [total_q, num_heads, max_topk] fp32 log2
    out_ptr,  # [total_q, num_heads, head_dim]
    total_q,
    head_dim,
    max_topk,
    stride_po_n,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_pl_n,
    stride_pl_h,
    stride_pl_s,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    MAX_TOPK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    N_TILE: tl.constexpr,
):
    pid_nt = tl.program_id(0)  # query tile
    pid_h = tl.program_id(1)  # head
    off_q = pid_nt * N_TILE + tl.arange(0, N_TILE)
    off_s = tl.arange(0, MAX_TOPK)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    q_mask = off_q < total_q
    s_mask = off_s < max_topk
    d_mask = off_d < head_dim

    # lse: [N_TILE, MAX_TOPK]
    lse = tl.load(
        partial_lse_ptr
        + off_q[:, None] * stride_pl_n
        + pid_h * stride_pl_h
        + off_s[None, :] * stride_pl_s,
        mask=q_mask[:, None] & s_mask[None, :],
        other=float("-inf"),
    )
    lse_max = tl.max(lse, axis=1)  # [N_TILE]
    valid = lse > float("-inf")
    weights = tl.where(valid, tl.exp2(lse - lse_max[:, None]), 0.0)  # [N_TILE, S]
    wsum = tl.sum(weights, axis=1)  # [N_TILE]
    inv = tl.where(wsum > 0, 1.0 / wsum, 0.0)

    # o: [N_TILE, MAX_TOPK, D] -> weighted sum over S
    o = tl.load(
        partial_o_ptr
        + off_q[:, None, None] * stride_po_n
        + pid_h * stride_po_h
        + off_s[None, :, None] * stride_po_s
        + off_d[None, None, :] * stride_po_d,
        mask=q_mask[:, None, None] & s_mask[None, :, None] & d_mask[None, None, :],
        other=0.0,
    )
    o = tl.where(valid[:, :, None], o.to(tl.float32), 0.0)
    o_merged = tl.sum(o * weights[:, :, None], axis=1) * inv[:, None]  # [N_TILE, D]
    tl.store(
        out_ptr
        + off_q[:, None] * stride_out_n
        + pid_h * stride_out_h
        + off_d[None, :] * stride_out_d,
        o_merged.to(out_ptr.dtype.element_ty),
        mask=q_mask[:, None] & d_mask[None, :],
    )


@torch.no_grad()
def _build_schedule(row_ptr: torch.Tensor, q_tile: int):
    """Flatten (row, q_tile-chunk) work units into two parallel arrays.

    For CSR row r with ``count`` entries, emit ``ceil(count / q_tile)`` work
    units, each tagged with its row id and absolute CSR entry offset. Empty
    rows emit nothing. Returns (sched_row [num_work] int32, sched_q_lo
    [num_work] int32).
    """
    device = row_ptr.device
    counts = (row_ptr[1:] - row_ptr[:-1]).to(torch.int64)
    tiles_per_row = (counts + q_tile - 1) // q_tile  # [num_rows]
    num_work = int(tiles_per_row.sum().item())
    if num_work == 0:
        empty = torch.zeros(0, device=device, dtype=torch.int32)
        return empty, empty
    # Row id of each work unit via repeat_interleave, then the within-row tile
    # index via a grouped arange.
    rows = torch.arange(row_ptr.numel() - 1, device=device, dtype=torch.int64)
    sched_row = torch.repeat_interleave(rows, tiles_per_row)
    # within-row tile index 0..tiles-1 for each unit
    starts = torch.cumsum(tiles_per_row, 0) - tiles_per_row
    within = (
        torch.arange(num_work, device=device, dtype=torch.int64) - starts[sched_row]
    )
    sched_q_lo = row_ptr[:-1].to(torch.int64)[sched_row] + within * q_tile
    return sched_row.to(torch.int32), sched_q_lo.to(torch.int32)


@triton.jit
def _build_schedule_kernel(
    row_ptr_ptr,  # [num_rows + 1] int32
    tile_off_ptr,  # [num_rows + 1] int32  prefix sum of tiles_per_row
    sched_row_ptr,  # [cap] int32  (over-allocated output)
    sched_q_lo_ptr,  # [cap] int32
    num_rows,
    q_tile: tl.constexpr,
):
    r = tl.program_id(0)  # one program per CSR row
    if r < num_rows:
        base = tl.load(row_ptr_ptr + r)
        row_end = tl.load(row_ptr_ptr + r + 1)
        count = row_end - base
        n_tiles = (count + q_tile - 1) // q_tile
        dst = tl.load(tile_off_ptr + r)  # where this row's tiles start
        for t in range(n_tiles):
            tl.store(sched_row_ptr + dst + t, r)
            tl.store(sched_q_lo_ptr + dst + t, base + t * q_tile)


@torch.no_grad()
def _build_schedule_fused(row_ptr: torch.Tensor, q_tile: int, nnz_cap: int):
    """Sync-free schedule build: device-only, no ``.item()`` readback.

    The work-tile count has the host-known dense upper bound
    ``num_rows + nnz_cap // q_tile`` (each row contributes at most
    ``ceil(count/q_tile) <= count/q_tile + 1`` tiles). Allocating the sched
    arrays to that cap lets the whole build stay on the GPU; ``num_work`` is
    returned as a device scalar tensor so no CPU sync is forced. The ws attend
    kernel reads ``num_work`` in-kernel and only touches valid tiles, so the
    over-allocated tail is never used. Keeping this sync-free is what lets the
    full prefill be captured in one CUDA graph.
    """
    device = row_ptr.device
    num_rows = row_ptr.numel() - 1
    counts = row_ptr[1:] - row_ptr[:-1]
    tiles_per_row = (counts + (q_tile - 1)) // q_tile  # int32
    tile_off = torch.zeros(num_rows + 1, device=device, dtype=torch.int32)
    tile_off[1:] = torch.cumsum(tiles_per_row, 0, dtype=torch.int32)
    num_work = tile_off[num_rows:]  # device scalar view [1], == total tiles

    cap = num_rows + nnz_cap // q_tile + 1
    sched_row = torch.zeros(cap, device=device, dtype=torch.int32)
    sched_q_lo = torch.zeros(cap, device=device, dtype=torch.int32)
    _build_schedule_kernel[(num_rows,)](
        row_ptr, tile_off, sched_row, sched_q_lo, num_rows, q_tile=q_tile
    )
    return sched_row, sched_q_lo, num_work


def _q_batch_index(cu_seqlens_q: torch.Tensor, total_q: int) -> torch.Tensor:
    """Map each global query id to its batch index (vectorized, no host sync)."""
    batch = cu_seqlens_q.numel() - 1
    cu = cu_seqlens_q.to(torch.int64)
    counts = cu[1:] - cu[:-1]
    return torch.repeat_interleave(
        torch.arange(batch, device=cu_seqlens_q.device, dtype=torch.int32), counts
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
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
    decode_query_len,
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
    USE_PDL: tl.constexpr,
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
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
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
        page = tl.load(bt_row + blk).to(tl.int64)
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
                    k_scale_ptr
                    + pid_kh * stride_ks_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
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
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
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
                    v_scale_ptr
                    + pid_kh * stride_vs_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask,
                    other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
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
    # The lse store is unconditional even though the merge-skip path never
    # reads it back. Gating it on a constexpr looks like free dead-code
    # removal and is not: dropping it removes a register pressure point late
    # in the kernel, the scheduler then picks a more aggressive allocation
    # (168 -> 190 regs), and at 73732 B of smem per CTA that crosses the
    # occupancy cliff from 3 CTAs/SM to 2 (65536 / (190*4*32) = 2). Measured
    # at (b=512, s=3072) fp8: 291.4 us with the store, 330.4 us without --
    # 13% slower to skip ~5 KB of writes. An explicit maxnreg=168 cap only
    # recovers half of it (316.3 us) because the cap spills instead.
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
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
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()

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
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


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
    """Run common `_gqa_sparse_fwd_fallback_kernel` or `_gqa_sparse_fwd_kernel`.

    Q is BF16; KV is BF16 or FP8, with the common kernels' scale handling.
    """
    _, num_heads, head_dim = q.shape
    gqa_group_size = num_heads // num_kv_heads
    block_size_h = triton.next_power_of_2(gqa_group_size)
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    if not use_fp8 and block_size_h >= 8:
        kernel = _gqa_sparse_fwd_fallback_kernel
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
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_QH=block_size_h,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        **launch_options,
    )


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
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    # Shape-only split count is independent of the vendor and stream.
    p = max(1, total_q * num_kv_heads)
    target = max(1, min(max_topk, 256 // p))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    use_pdl = False
    pdl_launch = {}
    o_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    _gqa_sparse_decode_kernel[total_q * num_topk_chunks, num_kv_heads](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
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
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    _merge_topk_attn_out_kernel[total_q, num_heads](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    return


@torch.no_grad()
def _minimax_m3_sparse_attn_kv_outer(
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
    *,
    q_tile: int | None = None,
    csr: dict | None = None,
    return_csr: bool = False,
    use_tle: bool | None = None,
    kernel: str | None = None,
    capacity: int = 2,
    num_ctas: int | None = None,
    maxnreg: int | None = 128,
    use_fused_csr: bool = True,
):
    """Portable KV-outer attention. Legacy kernel controls retain their arguments;
    this implementation always uses the standard Triton attend kernel."""
    if kv_cache.dtype != torch.bfloat16:
        raise NotImplementedError("kv-outer supports BF16 prefill only")
    if q_tile is None:
        q_tile = 32
    total_q, num_heads, head_dim = q.shape
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    num_pages = kv_cache.shape[0]
    if csr is None:
        builder = build_k2q_csr_fused if use_fused_csr else build_k2q_csr
        csr = builder(topk_idx, cu_seqlens_q, prefix_lens, block_table, num_pages)
    q_batch = csr.get("q_batch")
    if q_batch is None:
        q_batch = _q_batch_index(cu_seqlens_q, total_q)
    block_size_d = triton.next_power_of_2(head_dim)
    block_size_h = triton.next_power_of_2(gqa_group_size)
    partial_o = torch.empty(
        (total_q, num_heads, topk, head_dim), device=q.device, dtype=torch.bfloat16
    )
    partial_lse = torch.full(
        (total_q, num_heads, topk), float("-inf"), device=q.device, dtype=torch.float32
    )
    sched_row, sched_q_lo = _build_schedule(csr["row_ptr"], q_tile)
    num_work = sched_row.numel()
    num_work_dev = torch.tensor([num_work], device=q.device, dtype=torch.int32)
    grid = (max(1, num_work if num_work is not None else 1),)
    _kv_block_attn_kernel[grid](
        q,
        kv_cache,
        csr["row_ptr"],
        csr["q_global"],
        csr["q_abs"],
        csr["kv_block"],
        csr["slot"],
        q_batch,
        seq_lens,
        partial_o,
        partial_lse,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        partial_o.stride(0),
        partial_o.stride(1),
        partial_o.stride(2),
        partial_o.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        sched_row,
        sched_q_lo,
        Q_TILE=q_tile,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=block_size_d,
        BLOCK_SIZE_H=block_size_h,
    )
    max_topk_pow2 = triton.next_power_of_2(topk)
    n_tile = max(1, min(16, 256 // max_topk_pow2))
    merge_grid = (triton.cdiv(total_q, n_tile), num_heads)
    _kv_outer_merge_tiled_kernel[merge_grid](
        partial_o,
        partial_lse,
        output,
        total_q,
        head_dim,
        topk,
        partial_o.stride(0),
        partial_o.stride(1),
        partial_o.stride(2),
        partial_o.stride(3),
        partial_lse.stride(0),
        partial_lse.stride(1),
        partial_lse.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        MAX_TOPK=triton.next_power_of_2(topk),
        BLOCK_SIZE_D=block_size_d,
        N_TILE=n_tile,
    )
    if return_csr:
        return (output, csr)
    return output
