# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Hopper TMA/WGMMA prefill, KV-outer and warp-specialized kernels."""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from flag_attn.utils import has_triton_tle
from ...minimax_sparse_attention.decode import _sm_count
from ...minimax_sparse_attention.prefill import _gqa_sparse_fwd_kernel
from flag_attn.minimax_sparse_attention.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    _FP8_DTYPES,
    _KV_SCALE_NONE,
    _KV_SCALE_SCALAR,
    _KV_SCALE_PER_TOKEN_HEAD,
    _gqa_sparse_fwd_fallback_kernel,
    _kv_scale_args,
    _k2q_count_kernel,
    _k2q_scatter_kernel,
    build_k2q_csr_fused,
    build_k2q_csr,
    _kv_block_attn_kernel,
    _kv_outer_merge_kernel,
    _kv_outer_merge_tiled_kernel,
    _build_schedule,
    _build_schedule_kernel,
    _build_schedule_fused,
    _q_batch_index,
)

_MSA_PREFILL_UNTRANSPOSED_MIN_QH = 16


_MSA_PREFILL_FP8_MAXNREG = 128


_MSA_PREFILL_FP8_MIN_QH = 8


_PREFILL_PREFETCH_LOOP_SHAPES = frozenset(
    {
        # (batch, max_query_len, num_kv_heads)
        (1, 1024, 1),
        (4, 512, 1),
        (4, 1024, 1),
        (8, 128, 1),
        (16, 128, 1),
        (32, 128, 1),
    }
)


_PREFILL_TUNED_SM_COUNTS = frozenset({132})


_PREFILL_NO_WAR_BARRIER_SHAPES = frozenset(
    {
        # (batch, max_query_len, num_kv_heads)
        (1, 512, 1),  # 1.026x at seq 4096, 1.006x at 16384, both index patterns
    }
)


_PREFILL_HALF_KV_MAX_BLOCK_SIZE_QH = 8


_KV_OUTER_MIN_QUERY_LEN = 10240


def _prefill_drop_war_barrier(total_q, batch, max_query_len, num_kv_heads):
    """True to run upstream's half-KV loop without the kv_empty handshake."""
    if total_q != batch * max_query_len:
        return False  # only uniform batches were measured
    return (batch, max_query_len, num_kv_heads) in _PREFILL_NO_WAR_BARRIER_SHAPES


def _prefill_qh8_kernel_choice(device, total_q, batch, max_query_len, num_kv_heads):
    """'fork' | 'prefetch' | 'upstream' for the QH == 8 bf16 prefill."""
    sm = _sm_count(device)
    if sm not in _PREFILL_TUNED_SM_COUNTS:
        return "fork"
    # The launch is (max_query_len, num_kv_heads, batch), so that product -- not
    # total_q -- is what competes for SM slots. A ragged batch can hold few
    # query tokens and still launch far more than one wave of CTAs (b=4 with
    # query lengths [1, 1, 1, 125] is 128 tokens but a 500-CTA grid), and the
    # CTAs that return early still occupy a slot while they do.
    if max_query_len * num_kv_heads * batch <= sm:
        return "prefetch"
    uniform = total_q == batch * max_query_len
    if (
        uniform
        and (batch, max_query_len, num_kv_heads) in _PREFILL_PREFETCH_LOOP_SHAPES
    ):
        return "prefetch"
    return "upstream"


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,  # TMA view: [num_blocks*num_kv_heads*128, 2*head_dim]
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
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,  # == BLOCK_SIZE_H, since BLOCK_SIZE_Q == 1
    KEEP_TRANS: tl.constexpr,
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
    if pid_q >= q_block_len:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    # BLOCK_SIZE_Q == 1: one query token per program, so the valid block count
    # is a scalar rather than a 1-element tensor reduced with tl.max.
    q_abs = prefix_len + pid_q
    loop_blocks = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    causal_offsets = q_abs - off_n

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Four warps form one Hopper warpgroup. Q is staged once and reused by every
    # selected page as the transposed WGMMA B operand.
    q_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_QH, BLOCK_SIZE_D],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q)
    tl.debug_barrier()

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o_t = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_QH), dtype=tl.float32)

    topk_ptr = t_ptr + pid_kh * stride_th + (q_block_start + pid_q) * stride_tn

    # p_smem normally holds P as [BLOCK_SIZE_K, BLOCK_SIZE_QH] -- the layout
    # WGMMA already produces -- so the QK result never has to be transposed.
    # ncu attributed 99.5% of this kernel's shared-bank conflicts to that one
    # store, and removing the transpose cut instructions by 14%.
    #
    # KEEP_TRANS restores the transposed [QH, K] form for BLOCK_SIZE_QH == 8,
    # where `wgmma(v, p, acc, trans_a=True)` silently computes a WRONG result
    # (an N=8 transposed-A WGMMA returned 96.0 for an exact-128.0 reduction;
    # N>=16 is correct). Everything else in this kernel still applies there.
    if KEEP_TRANS:
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, BLOCK_SIZE_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
    else:
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_K, BLOCK_SIZE_QH],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
    kv_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
        dtype=kv_cache_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    kv_stage_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D * 2
    # Only the RAW barrier survives. Upstream also allocates a `kv_empty` WAR
    # barrier and pays 4 extra sync ops per page on it; `wgmma_wait(0)` below
    # already drains each operand read, so the slot is provably free without it.
    kv_full = tle.gpu.alloc_barriers(
        num_barriers=1,
        arrive_count=1,
        expect_bytes=kv_stage_bytes,
    )

    # Resolve page(0) before the loop; each iteration consumes the carried
    # row/offset and resolves page(i + 1) while V(i) is in flight.
    cur_kv_row = tl.full((), 0, dtype=tl.int32)
    cur_c = tl.full((), 0, dtype=tl.int32)
    if loop_blocks > 0:
        first_blk = tl.load(topk_ptr).to(tl.int32)
        first_page = tl.load(bt_row + first_blk).to(tl.int32)
        cur_kv_row = (first_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
        cur_c = first_blk * BLOCK_SIZE_K

    for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
        k_phase = block_iter * 2
        v_phase = k_phase + 1
        kv_row = cur_kv_row
        c = cur_c
        pos = c + off_n
        pos_mask = pos < seq_len

        tle.gpu.copy(
            kv_cache_desc,
            kv_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [kv_row, 0],
            barrier=kv_full[0],
        )
        tle.gpu.barrier_wait(kv_full[0], phaseIdx=k_phase)

        qk_t = tle.gpu.wgmma(
            kv_smem.slot(0),
            q_smem.slot(0),
            out_dtype=tl.float32,
            trans_b=True,
        )
        # Drains the QK WGMMA: K's smem read has completed, so the V copy below
        # cannot clobber a live operand and needs no WAR handshake.
        qk_t = tle.gpu.wgmma_wait(0, qk_t)
        qk_t *= sm_scale_log2e

        # V's origin is head_dim, not the padded BLOCK_SIZE_D; see _tma_box_width.
        tle.gpu.copy(
            kv_cache_desc,
            kv_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [kv_row, head_dim],
            barrier=kv_full[0],
        )

        if block_iter + 1 < loop_blocks:
            next_blk = tl.load(topk_ptr + (block_iter + 1) * stride_tk).to(tl.int32)
            next_page = tl.load(bt_row + next_blk).to(tl.int32)
            cur_kv_row = (next_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
            cur_c = next_blk * BLOCK_SIZE_K

        # qk_t is [BLOCK_SIZE_K, BLOCK_SIZE_QH] straight out of the WGMMA.
        # Keeping that native layout and reducing along axis 0 (the KV axis)
        # removes the transpose and the register shuffle its smem store
        # implied. Both masks are per-block scalar-guarded, so the vector
        # select is only materialized for the diagonal / tail page.
        if KEEP_TRANS:
            qk = tl.reshape(tl.trans(qk_t), (BLOCK_SIZE_QH, BLOCK_SIZE_K))
            if (c + BLOCK_SIZE_K) > q_abs:
                qk += tl.where(causal_offsets[None, :] >= c, 0, float("-inf"))
            if (c + BLOCK_SIZE_K) > seq_len:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
        else:
            qk = qk_t
            if (c + BLOCK_SIZE_K) > q_abs:
                qk += tl.where(causal_offsets[:, None] >= c, 0, float("-inf"))
            if (c + BLOCK_SIZE_K) > seq_len:
                qk += tl.where(pos_mask[:, None], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp2(qk - m_ij[None, :])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=0)
        acc_o_t *= alpha[None, :]
        tl.store(
            tle.gpu.local_ptr(p_smem.slot(0)),
            p.to(kv_cache_ptr.dtype.element_ty),
        )

        tle.gpu.barrier_wait(kv_full[0], phaseIdx=v_phase)
        tl.debug_barrier()
        # acc[D, QH] = V^T @ P, with V=[K, D] in kv_smem. In the default
        # layout p_smem is already K-major so trans_b is unnecessary; under
        # KEEP_TRANS it is [QH, K] and needs transposing, as upstream does.
        if KEEP_TRANS:
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
                trans_b=True,
            )
        else:
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
            )
        acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)

        l_i = tl.math.fma(l_i, alpha, l_ij)
        m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o_t *= inv_l[None, :]
    acc_o = tl.trans(acc_o_t)
    acc_o = tl.reshape(acc_o, 1, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_prefetch_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,  # TMA view: [num_blocks*num_kv_heads*128, 2*head_dim]
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
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,  # == BLOCK_SIZE_H, since BLOCK_SIZE_Q == 1
    KEEP_TRANS: tl.constexpr,
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
    if pid_q >= q_block_len:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    # BLOCK_SIZE_Q == 1: one query token per program, so the valid block count
    # is a scalar rather than a 1-element tensor reduced with tl.max.
    q_abs = prefix_len + pid_q
    loop_blocks = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    causal_offsets = q_abs - off_n

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Four warps form one Hopper warpgroup. Q is staged once and reused by every
    # selected page as the transposed WGMMA B operand.
    q_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_QH, BLOCK_SIZE_D],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q)
    tl.debug_barrier()

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o_t = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_QH), dtype=tl.float32)

    topk_ptr = t_ptr + pid_kh * stride_th + (q_block_start + pid_q) * stride_tn

    # p_smem normally holds P as [BLOCK_SIZE_K, BLOCK_SIZE_QH] -- the layout
    # WGMMA already produces -- so the QK result never has to be transposed.
    # ncu attributed 99.5% of this kernel's shared-bank conflicts to that one
    # store, and removing the transpose cut instructions by 14%.
    #
    # KEEP_TRANS restores the transposed [QH, K] form for BLOCK_SIZE_QH == 8,
    # where `wgmma(v, p, acc, trans_a=True)` silently computes a WRONG result
    # (an N=8 transposed-A WGMMA returned 96.0 for an exact-128.0 reduction;
    # N>=16 is correct). Everything else in this kernel still applies there.
    if KEEP_TRANS:
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, BLOCK_SIZE_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
    else:
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_K, BLOCK_SIZE_QH],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
    # Dedicated K and V slots, each one page ahead: K(i+1) is issued as soon
    # as QK(i) has read K(i), V(i+1) as soon as PV(i) has read V(i).
    k_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
        dtype=kv_cache_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
        dtype=kv_cache_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    kv_stage_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D * 2
    k_full = tle.gpu.alloc_barriers(
        num_barriers=1, arrive_count=1, expect_bytes=kv_stage_bytes
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=1, arrive_count=1, expect_bytes=kv_stage_bytes
    )

    # Prologue: page 0's K and V. From here on each iteration issues page i + 1,
    # so only the column offset has to be carried across iterations.
    cur_c = tl.full((), 0, dtype=tl.int32)
    if loop_blocks > 0:
        blk0 = tl.load(topk_ptr).to(tl.int32)
        page0 = tl.load(bt_row + blk0).to(tl.int32)
        row0 = (page0 * num_kv_heads + pid_kh) * BLOCK_SIZE_K
        cur_c = blk0 * BLOCK_SIZE_K
        tle.gpu.copy(
            kv_cache_desc,
            k_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [row0, 0],
            barrier=k_full[0],
        )
        # V's origin is head_dim, not the padded BLOCK_SIZE_D; see _tma_box_width.
        tle.gpu.copy(
            kv_cache_desc,
            v_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [row0, head_dim],
            barrier=v_full[0],
        )

    for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
        c = cur_c
        pos = c + off_n
        pos_mask = pos < seq_len
        # Next page's address, issued early so the loads overlap the K wait.
        nxt_row = tl.full((), 0, dtype=tl.int32)
        nxt_c = tl.full((), 0, dtype=tl.int32)
        if block_iter + 1 < loop_blocks:
            nblk = tl.load(topk_ptr + (block_iter + 1) * stride_tk).to(tl.int32)
            npage = tl.load(bt_row + nblk).to(tl.int32)
            nxt_row = (npage * num_kv_heads + pid_kh) * BLOCK_SIZE_K
            nxt_c = nblk * BLOCK_SIZE_K

        tle.gpu.barrier_wait(k_full[0], phaseIdx=block_iter)
        qk_t = tle.gpu.wgmma(
            k_smem.slot(0),
            q_smem.slot(0),
            out_dtype=tl.float32,
            trans_b=True,
        )
        # Drains QK: K(i) is read, so the K slot can take K(i+1) right away.
        qk_t = tle.gpu.wgmma_wait(0, qk_t)
        qk_t *= sm_scale_log2e
        if block_iter + 1 < loop_blocks:
            tle.gpu.copy(
                kv_cache_desc,
                k_smem.slot(0),
                [BLOCK_SIZE_K, BLOCK_SIZE_D],
                [nxt_row, 0],
                barrier=k_full[0],
            )

        qk = tl.reshape(tl.trans(qk_t), (BLOCK_SIZE_QH, BLOCK_SIZE_K))
        if (c + BLOCK_SIZE_K) > q_abs:
            qk += tl.where(causal_offsets[None, :] >= c, 0, float("-inf"))
        if (c + BLOCK_SIZE_K) > seq_len:
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc_o_t *= alpha[None, :]
        tl.store(
            tle.gpu.local_ptr(p_smem.slot(0)),
            p.to(kv_cache_ptr.dtype.element_ty),
        )

        tle.gpu.barrier_wait(v_full[0], phaseIdx=block_iter)
        tl.debug_barrier()
        acc_o_t = tle.gpu.wgmma(
            v_smem.slot(0),
            p_smem.slot(0),
            acc_o_t,
            trans_a=True,
            trans_b=True,
        )
        # Drains PV: V(i) is read, so the V slot can take V(i+1).
        acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)
        if block_iter + 1 < loop_blocks:
            tle.gpu.copy(
                kv_cache_desc,
                v_smem.slot(0),
                [BLOCK_SIZE_K, BLOCK_SIZE_D],
                [nxt_row, head_dim],
                barrier=v_full[0],
            )

        l_i = tl.math.fma(l_i, alpha, l_ij)
        m_i = m_ij
        cur_c = nxt_c

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o_t *= inv_l[None, :]
    acc_o = tl.trans(acc_o_t)
    acc_o = tl.reshape(acc_o, 1, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_upstream_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,  # flattened [num_blocks*num_kv_heads*128, 2*head_dim]
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
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_TL_DOT_PATH: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
    USE_HALF_KV_PIPE: tl.constexpr,
    WAR_BARRIER: tl.constexpr = True,
):
    if USE_TL_DOT_PATH:
        _gqa_sparse_fwd_kernel(
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
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            BLOCK_SIZE_D,
            BLOCK_SIZE_H,
            BLOCK_SIZE_QH,
            USE_FP8,
            KV_SCALE_MODE,
        )
        return

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

    q_tile_start = pid_q * BLOCK_SIZE_Q
    if q_tile_start >= q_block_len:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    q_abs = prefix_len + q_tile_start + off_q
    real_topk = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(q_tile_start, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Four warps form one Hopper warpgroup. Q is staged once and reused by all
    # selected pages as the transposed WGMMA B operand.
    q_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_QH, BLOCK_SIZE_D],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q)
    tl.debug_barrier()

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o_t = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_QH), dtype=tl.float32)

    loop_blocks = tl.max(real_topk, axis=0)
    topk_ptr = t_ptr + pid_kh * stride_th + (q_block_start + q_tile_start) * stride_tn

    if USE_HALF_KV_PIPE:
        HALF_K: tl.constexpr = BLOCK_SIZE_K // 2
        off_half = tl.arange(0, HALF_K)
        causal_offsets = q_abs[:, None] - off_half[None, :]
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, HALF_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_smem = tle.gpu.alloc(
            [2, HALF_K, BLOCK_SIZE_D],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_stage_bytes: tl.constexpr = HALF_K * BLOCK_SIZE_D * 2
        # WAR_BARRIER=False drops this write-after-read handshake; see the
        # per-slot argument at the two overwrite sites below. Kept as a
        # constexpr so WAR_BARRIER=True still emits upstream's exact PTX.
        if WAR_BARRIER:
            kv_empty = tle.gpu.alloc_barriers(
                num_barriers=2,
                arrive_count=1,
                init=tle.gpu.READY,
            )
        kv_full = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=1,
            expect_bytes=kv_stage_bytes,
        )
        loop_tiles = loop_blocks * 2
        # The K prefetch already resolves the next tile's logical block and
        # physical page. Keep these two scalar addresses live across the loop
        # so the V copy does not reload topk_idx and block_table for that tile.
        cur_kv_row = tl.full((), 0, dtype=tl.int32)
        cur_c = tl.full((), 0, dtype=tl.int32)

        # Prologue: stage the first K half before entering the ping-pong loop.
        if loop_tiles > 0:
            first_blk = tl.load(topk_ptr).to(tl.int32)
            first_page = tl.load(bt_row + first_blk).to(tl.int32)
            first_row = (first_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
            cur_kv_row = first_row
            cur_c = first_blk * BLOCK_SIZE_K
            if WAR_BARRIER:
                tle.gpu.barrier_wait(kv_empty[0], phaseIdx=0)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(0),
                [HALF_K, BLOCK_SIZE_D],
                [first_row, 0],
                barrier=kv_full[0],
            )

        for tile_iter in tl.range(loop_tiles, disable_licm=True, num_stages=1):
            block_iter = tile_iter // 2
            # half_idx = tile_iter % 2
            buf_idx = tile_iter % 2
            reuse_iter = tile_iter // 2
            k_phase = reuse_iter * 2
            v_phase = k_phase + 1

            kv_row = cur_kv_row
            c = cur_c
            pos = c + off_half
            pos_mask = pos < seq_len

            tle.gpu.barrier_wait(kv_full[buf_idx], phaseIdx=k_phase)
            qk_t = tle.gpu.wgmma(
                kv_smem.slot(buf_idx),
                q_smem.slot(0),
                out_dtype=tl.float32,
                trans_b=True,
            )
            qk_t = tle.gpu.wgmma_wait(0, qk_t)
            qk_t *= sm_scale_log2e
            qk = tl.reshape(
                tl.trans(qk_t),
                (BLOCK_SIZE_Q, BLOCK_SIZE_H, HALF_K),
            )

            # Reuse this slot for V while the other slot receives the next K.
            # No handshake needed: the `wgmma_wait(0)` above drained the QK that
            # read this slot as its A operand, so the V copy cannot clobber a
            # live operand. This is the same property the single-stage and
            # prefetch loops in this file already rely on.
            if WAR_BARRIER:
                tle.gpu.barrier_arrive(kv_empty[buf_idx], phaseIdx=k_phase)
                tle.gpu.barrier_wait(kv_empty[buf_idx], phaseIdx=v_phase)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(buf_idx),
                [HALF_K, BLOCK_SIZE_D],
                [kv_row, head_dim],
                barrier=kv_full[buf_idx],
            )

            if tile_iter + 1 < loop_tiles:
                next_tile = tile_iter + 1
                next_block_iter = next_tile // 2
                next_half_idx = next_tile % 2
                next_buf_idx = next_tile % 2
                next_reuse_iter = next_tile // 2
                next_k_phase = next_reuse_iter * 2
                next_blk = tl.load(topk_ptr + next_block_iter * stride_tk).to(tl.int32)
                next_page = tl.load(bt_row + next_blk).to(tl.int32)
                next_kv_row = (
                    next_page * num_kv_heads + pid_kh
                ) * BLOCK_SIZE_K + next_half_idx * HALF_K
                next_c = next_blk * BLOCK_SIZE_K + next_half_idx * HALF_K
                # The other slot was last read by the previous tile's PV, whose
                # `wgmma_wait(0)` ran before this iteration started; on the
                # first tile it has never been read at all.
                if WAR_BARRIER:
                    tle.gpu.barrier_wait(kv_empty[next_buf_idx], phaseIdx=next_k_phase)
                tle.gpu.copy(
                    kv_cache_desc,
                    kv_smem.slot(next_buf_idx),
                    [HALF_K, BLOCK_SIZE_D],
                    [next_kv_row, 0],
                    barrier=kv_full[next_buf_idx],
                )
                cur_kv_row = next_kv_row
                cur_c = next_c

            if (c + HALF_K) > (prefix_len + q_tile_start):
                qk += tl.where(causal_offsets[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, HALF_K)
            if (c + HALF_K) > seq_len:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc_o_t *= alpha[None, :]
            tl.store(
                tle.gpu.local_ptr(p_smem.slot(0)),
                p.to(kv_cache_ptr.dtype.element_ty),
            )

            tle.gpu.barrier_wait(kv_full[buf_idx], phaseIdx=v_phase)
            tl.debug_barrier()
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(buf_idx),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
                trans_b=True,
            )
            acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)
            if WAR_BARRIER:
                tle.gpu.barrier_arrive(kv_empty[buf_idx], phaseIdx=v_phase)

            l_i = tl.math.fma(l_i, alpha, l_ij)
            m_i = m_ij
    else:
        causal_offsets = q_abs[:, None] - off_n[None, :]
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, BLOCK_SIZE_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_stage_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D * 2
        kv_empty = tle.gpu.alloc_barriers(
            num_barriers=1,
            arrive_count=1,
            init=tle.gpu.READY,
        )
        kv_full = tle.gpu.alloc_barriers(
            num_barriers=1,
            arrive_count=1,
            expect_bytes=kv_stage_bytes,
        )
        # Resolve page(0) before the loop. Each iteration consumes the carried
        # row/offset, then resolves page(i + 1) while V(i) is in flight.
        cur_kv_row = tl.full((), 0, dtype=tl.int32)
        cur_c = tl.full((), 0, dtype=tl.int32)
        if loop_blocks > 0:
            first_blk = tl.load(topk_ptr).to(tl.int32)
            first_page = tl.load(bt_row + first_blk).to(tl.int32)
            cur_kv_row = (first_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
            cur_c = first_blk * BLOCK_SIZE_K

        for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
            k_phase = block_iter * 2
            v_phase = k_phase + 1
            kv_row = cur_kv_row
            c = cur_c
            pos = c + off_n
            pos_mask = pos < seq_len

            tle.gpu.barrier_wait(kv_empty[0], phaseIdx=k_phase)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(0),
                [BLOCK_SIZE_K, BLOCK_SIZE_D],
                [kv_row, 0],
                barrier=kv_full[0],
            )
            tle.gpu.barrier_wait(kv_full[0], phaseIdx=k_phase)

            qk_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                q_smem.slot(0),
                out_dtype=tl.float32,
                trans_b=True,
            )
            qk_t = tle.gpu.wgmma_wait(0, qk_t)
            qk_t *= sm_scale_log2e
            qk = tl.reshape(
                tl.trans(qk_t),
                (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K),
            )

            tle.gpu.barrier_arrive(kv_empty[0], phaseIdx=k_phase)
            tle.gpu.barrier_wait(kv_empty[0], phaseIdx=v_phase)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(0),
                [BLOCK_SIZE_K, BLOCK_SIZE_D],
                [kv_row, head_dim],
                barrier=kv_full[0],
            )

            if block_iter + 1 < loop_blocks:
                next_blk = tl.load(topk_ptr + (block_iter + 1) * stride_tk).to(tl.int32)
                next_page = tl.load(bt_row + next_blk).to(tl.int32)
                cur_kv_row = (next_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
                cur_c = next_blk * BLOCK_SIZE_K

            if (c + BLOCK_SIZE_K) > (prefix_len + q_tile_start):
                qk += tl.where(causal_offsets[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            if (c + BLOCK_SIZE_K) > seq_len:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc_o_t *= alpha[None, :]
            tl.store(
                tle.gpu.local_ptr(p_smem.slot(0)),
                p.to(kv_cache_ptr.dtype.element_ty),
            )

            tle.gpu.barrier_wait(kv_full[0], phaseIdx=v_phase)
            tl.debug_barrier()
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
                trans_b=True,
            )
            acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)
            tle.gpu.barrier_arrive(kv_empty[0], phaseIdx=v_phase)

            l_i = tl.math.fma(l_i, alpha, l_ij)
            m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o_t *= inv_l[None, :]
    acc_o = tl.trans(acc_o_t)
    acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(q_tile_start, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


def _tma_allocator(size: int, align: int, stream):
    _ = align
    _ = stream
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _tma_box_width(head_dim: int) -> int:
    """TMA box width for one K or V half of a packed KV row.

    ``TensorDescriptor`` requires power-of-two ``block_shape`` entries, and the
    kernels stage K/V into ``[*, BLOCK_SIZE_D]`` tiles. Passing ``head_dim``
    breaks at 96 twice over: the descriptor is rejected, and the box would be
    narrower than the tile copied into.

    The box is wider than ``head_dim`` when head_dim is not a power of two, so
    each tile picks up ``BLOCK_SIZE_D - head_dim`` junk columns. They are inert,
    but for two different reasons depending on the half -- the descriptor rows
    are ``2 * head_dim`` wide, so only V's overhang is actually past the row:

    - K (origin 0) overhangs into V's leading columns, which are **in bounds
      and real data**, not zero-fill.
    - V (origin ``head_dim``) overhangs past the row end, which TMA zero-fills.

    Either way nothing reaches the output: ``BLOCK_SIZE_D`` is the QK
    contraction dim and Q is loaded ``padding_option="zero"``, so every junk
    column is multiplied by a zero Q lane and adds nothing to the logits; and
    the accumulator's junk rows are dropped by the output store's
    ``boundary_check``. Verified at head_dim 96 by scaling V by 1e5, which
    leaves the K-overhang columns enormous while the result stays bit-identical
    to the non-TMA path.
    """
    return triton.next_power_of_2(head_dim)


def _tma_supports_head_dim(head_dim: int, itemsize: int) -> bool:
    """Whether a packed KV row admits a legal TMA box for both halves.

    V's box starts at column ``head_dim``, i.e. ``head_dim * itemsize`` bytes
    into the row, and TMA requires a 16-byte-aligned box origin. bf16 therefore
    needs ``head_dim % 8 == 0`` -- true for 96 and 128, false for e.g. 100,
    where the descriptor builds fine and then faults on the device with an
    illegal instruction. Callers must route the remainder to a non-TMA kernel
    rather than letting it reach a descriptor.
    """
    return (head_dim * itemsize) % 16 == 0


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_fp8_kernel(
    q_ptr,  # [total_q, num_heads, head_dim] bf16
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim] fp8
    k_desc,  # TMA view of the K half, fp8
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
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
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if pid_q >= q_len:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    q_abs = prefix_len + pid_q
    loop_blocks = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    causal_offsets = q_abs - off_n

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Scalar KV scales are invariant across pages: fold K's into the logits
    # scale and defer V's to the normalized output, exactly as upstream does.
    qk_scale = sm_scale_log2e * tl.load(k_scale_ptr)
    v_scale_scalar = tl.load(v_scale_ptr)

    # Q is quantized ONCE, outside the loop, per head. Its scale is restored on
    # the fp32 logits, so no accuracy is traded for the fp8 operand.
    q_absmax = tl.max(tl.abs(q), axis=1)
    q_scale = tl.maximum(q_absmax * (1.0 / 448.0), 1.0e-8)
    q_fp8 = (q / q_scale[:, None]).to(tl.float8e4nv)

    # Staged in smem as [QH, D]: D innermost, which is what the K-major-only
    # fp8 WGMMA requires of its B operand under trans_b.
    q_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_QH, BLOCK_SIZE_D],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q_fp8)
    tl.debug_barrier()

    # K staged by TMA, kept in fp8 -- never materialized as a register tile.
    # This is the change that is supposed to drop the register count.
    k_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D  # fp8: 1 byte/elem
    k_full = tle.gpu.alloc_barriers(
        num_barriers=1, arrive_count=1, expect_bytes=k_bytes
    )

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)

    topk_ptr = t_ptr + pid_kh * stride_th + (q_start + pid_q) * stride_tn

    cur_page = tl.full((), 0, dtype=tl.int32)
    cur_c = tl.full((), 0, dtype=tl.int32)
    if loop_blocks > 0:
        b0 = tl.load(topk_ptr).to(tl.int32)
        cur_page = tl.load(bt_row + b0).to(tl.int32)
        cur_c = b0 * BLOCK_SIZE_K

    for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
        page = cur_page
        c = cur_c
        pos = c + off_n
        pos_mask = pos < seq_len
        k_row = (page * num_kv_heads + pid_kh) * BLOCK_SIZE_K

        tle.gpu.copy(
            k_desc,
            k_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [k_row, 0],
            barrier=k_full[0],
        )
        tle.gpu.barrier_wait(k_full[0], phaseIdx=block_iter)

        # acc[K, QH] = K[K, D] @ Q[QH, D]^T -- both fp8, both D-innermost.
        qk_t = tle.gpu.wgmma(
            k_smem.slot(0),
            q_smem.slot(0),
            out_dtype=tl.float32,
            trans_b=True,
        )
        qk_t = tle.gpu.wgmma_wait(0, qk_t)

        # Resolve the next page while the WGMMA result is still being consumed.
        if block_iter + 1 < loop_blocks:
            nb = tl.load(topk_ptr + (block_iter + 1) * stride_tk).to(tl.int32)
            cur_page = tl.load(bt_row + nb).to(tl.int32)
            cur_c = nb * BLOCK_SIZE_K

        # Restore Q's quantization scale on the fp32 logits and fold in both
        # the softmax base-2 factor and the scalar K scale.
        qk = tl.trans(qk_t) * (q_scale[:, None] * qk_scale)

        if (c + BLOCK_SIZE_K) > q_abs:
            qk += tl.where(causal_offsets[None, :] >= c, 0, float("-inf"))
        if (c + BLOCK_SIZE_K) > seq_len:
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc_o *= alpha[:, None]

        # PV stays on tl.dot: it contracts over K, so fp8 WGMMA would need V as
        # [D, K] and TMA cannot produce that (see the module docstring). V is
        # loaded to registers and dequantized to Q's dtype, as upstream does.
        v = tl.load(
            kv_cache_ptr
            + page.to(tl.int64) * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None],
            other=0.0,
        ).to(q_ptr.dtype.element_ty)
        acc_o += tl.dot(p.to(v.dtype), v)

        l_i = tl.math.fma(l_i, alpha, l_ij)
        m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0) * v_scale_scalar
    acc_o *= inv_l[:, None]
    acc_o = tl.reshape(acc_o, 1, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


def _should_use_kv_outer_prefill(kv_cache, max_query_len, k_scale, v_scale) -> bool:
    """kv-outer prefill is BF16-only, unscaled, needs TLE, and only wins on long
    prompts. Anything else keeps the q-outer body."""
    if max_query_len < _KV_OUTER_MIN_QUERY_LEN:
        return False
    if kv_cache.dtype != torch.bfloat16:
        return False
    if k_scale is not None or v_scale is not None:
        return False
    return has_triton_tle()


@triton.jit
def _kv_block_attn_tle_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,  # TMA view: [num_blocks*num_kv_heads*128, 2*head_dim]
    row_ptr,  # [num_rows + 1] int32  CSR row offsets
    csr_q_global,  # [nnz] int32
    csr_q_abs,  # [nnz] int32
    csr_kv_block,  # [nnz] int32  logical block (for causal key positions)
    csr_slot,  # [nnz] int32
    q_batch_ptr,  # [total_q] int32
    seq_lens,  # [batch] int32
    partial_o_ptr,  # [total_q, num_heads, max_topk, head_dim]
    partial_lse_ptr,  # [total_q, num_heads, max_topk] fp32 (log2 domain)
    sched_row_ptr,  # [num_work]
    sched_q_lo_ptr,  # [num_work]
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_po_n,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_pl_n,
    stride_pl_h,
    stride_pl_s,
    Q_TILE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # next_pow2(gqa_group_size)
):
    sm_scale_log2e = sm_scale * 1.4426950409
    QH: tl.constexpr = Q_TILE * BLOCK_SIZE_H

    pid_w = tl.program_id(0)
    pid_row = tl.load(sched_row_ptr + pid_w)
    q_lo = tl.load(sched_q_lo_ptr + pid_w)
    page = (pid_row // num_kv_heads).to(tl.int32)
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

    # CSR entries for this tile.
    ent = q_lo + off_qt
    q_global = tl.load(csr_q_global + ent, mask=qt_mask, other=0).to(tl.int32)
    q_abs = tl.load(csr_q_abs + ent, mask=qt_mask, other=0).to(tl.int32)
    kv_block = tl.load(csr_kv_block + ent, mask=qt_mask, other=0).to(tl.int32)
    slot = tl.load(csr_slot + ent, mask=qt_mask, other=0).to(tl.int32)
    batch_id = tl.load(q_batch_ptr + q_global, mask=qt_mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens + batch_id, mask=qt_mask, other=0).to(tl.int32)

    # Gather Q [QH, D] for the (query, head) pairs and stage into SMEM once.
    q_rows = (q_global[:, None] * stride_qn) + (pid_h + off_h)[None, :] * stride_qh
    q_rows = tl.reshape(q_rows, QH)
    qh_mask = tl.reshape(qt_mask[:, None] & h_mask[None, :], QH)
    q = tl.load(
        q_ptr + q_rows[:, None] + off_d[None, :] * stride_qd,
        mask=qh_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [QH, D]

    q_smem = tle.gpu.alloc(
        [1, QH, BLOCK_SIZE_D],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q)

    # Stage the physical KV page ONCE (K then V) -- reused by every query.
    kv_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
        dtype=kv_cache_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, QH],
        dtype=kv_cache_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    kv_stage_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D * 2
    kv_full = tle.gpu.alloc_barriers(
        num_barriers=1, arrive_count=1, expect_bytes=kv_stage_bytes
    )

    kv_row = (page * num_kv_heads + pid_kh) * BLOCK_SIZE_K  # int32 for TMA offset
    # K tile.
    tle.gpu.copy(
        kv_cache_desc,
        kv_smem.slot(0),
        [BLOCK_SIZE_K, BLOCK_SIZE_D],
        [kv_row, 0],
        barrier=kv_full[0],
    )
    tle.gpu.barrier_wait(kv_full[0], phaseIdx=0)

    qk_t = tle.gpu.wgmma(
        kv_smem.slot(0), q_smem.slot(0), out_dtype=tl.float32, trans_b=True
    )  # [K, QH]
    qk_t = tle.gpu.wgmma_wait(0, qk_t)
    qk_t *= sm_scale_log2e

    # V tile into the same slot (QK's WGMMA has drained the K read).
    tle.gpu.copy(
        kv_cache_desc,
        kv_smem.slot(0),
        [BLOCK_SIZE_K, BLOCK_SIZE_D],
        [kv_row, BLOCK_SIZE_D],
        barrier=kv_full[0],
    )

    # Causal + sequence mask per (key, query). key_abs = kv_block*128 + off_n,
    # broadcast across the QH queries (each query carries its own kv_block/q_abs,
    # expanded to the group heads). qk_t is [K, QH].
    q_abs_qh = tl.reshape(
        tl.broadcast_to(q_abs[:, None], (Q_TILE, BLOCK_SIZE_H)), QH
    )  # [QH]
    seq_qh = tl.reshape(tl.broadcast_to(seq_len[:, None], (Q_TILE, BLOCK_SIZE_H)), QH)
    kb_qh = tl.reshape(tl.broadcast_to(kv_block[:, None], (Q_TILE, BLOCK_SIZE_H)), QH)
    key_abs_qh = kb_qh[None, :] * BLOCK_SIZE_K + off_n[:, None]  # [K, QH]
    allowed = (q_abs_qh[None, :] >= key_abs_qh) & (key_abs_qh < seq_qh[None, :])
    qk_t = tl.where(allowed, qk_t, float("-inf"))

    m = tl.max(qk_t, axis=0)  # [QH]
    p = tl.exp2(qk_t - m[None, :])  # [K, QH]
    l = tl.sum(p, axis=0)  # [QH]
    tl.store(tle.gpu.local_ptr(p_smem.slot(0)), p.to(kv_cache_ptr.dtype.element_ty))

    tle.gpu.barrier_wait(kv_full[0], phaseIdx=1)
    tl.debug_barrier()
    # acc[D, QH] = V^T @ P, V=[K, D] in kv_smem, P=[K, QH] in p_smem (K-major).
    acc_o_t = tl.zeros((BLOCK_SIZE_D, QH), dtype=tl.float32)
    acc_o_t = tle.gpu.wgmma(kv_smem.slot(0), p_smem.slot(0), acc_o_t, trans_a=True)
    acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)

    inv_l = tl.where(l > 0, 1.0 / l, 0.0)
    acc_o_t *= inv_l[None, :]  # [D, QH] normalized partial
    acc = tl.trans(acc_o_t)  # [QH, D]
    acc = tl.reshape(acc, Q_TILE, BLOCK_SIZE_H, BLOCK_SIZE_D)

    lse = tl.where(l > 0, m + tl.log2(l), float("-inf"))  # [QH]
    lse = tl.reshape(lse, Q_TILE, BLOCK_SIZE_H)

    # Scatter partials to [q_global, head, slot].
    po_base = (
        partial_o_ptr
        + q_global[:, None, None] * stride_po_n
        + (pid_h + off_h)[None, :, None] * stride_po_h
        + slot[:, None, None] * stride_po_s
        + off_d[None, None, :] * stride_po_d
    )
    po_mask = qt_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
    tl.store(po_base, acc.to(partial_o_ptr.dtype.element_ty), mask=po_mask)
    pl_base = (
        partial_lse_ptr
        + q_global[:, None] * stride_pl_n
        + (pid_h + off_h)[None, :] * stride_pl_h
        + slot[:, None] * stride_pl_s
    )
    tl.store(pl_base, lse, mask=qt_mask[:, None] & h_mask[None, :])


@triton.jit
def _kv_outer_ws_producer(
    kv_cache_desc,
    kv_stages,
    kv_full,
    empty,
    sched_row_ptr,
    sched_q_lo_ptr,  # unused by producer but keeps a uniform metadata view
    row_ptr,
    num_kv_heads,
    work_lo,
    work_hi,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    # Two TMA copies (K then V) per work-tile, each into its own ring slot pair.
    i = 0
    for w in range(work_lo, work_hi):
        pid_row = tl.load(sched_row_ptr + w)
        page = (pid_row // num_kv_heads).to(tl.int32)
        pid_kh = pid_row % num_kv_heads
        kv_row = (page * num_kv_heads + pid_kh) * BLOCK_SIZE_K

        k_slot = (2 * i) % CAPACITY
        gen_k = (2 * i) // CAPACITY
        tle.gpu.barrier_wait(empty[k_slot], phaseIdx=gen_k)
        tle.gpu.copy(
            kv_cache_desc,
            kv_stages.slot(k_slot),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [kv_row, 0],
            barrier=kv_full[k_slot],
        )
        v_slot = (2 * i + 1) % CAPACITY
        gen_v = (2 * i + 1) // CAPACITY
        tle.gpu.barrier_wait(empty[v_slot], phaseIdx=gen_v)
        tle.gpu.copy(
            kv_cache_desc,
            kv_stages.slot(v_slot),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [kv_row, BLOCK_SIZE_D],
            barrier=kv_full[v_slot],
        )
        i += 1


@triton.jit
def _kv_outer_ws_consumer(
    q_ptr,
    kv_stages,
    kv_full,
    empty,
    row_ptr,
    sched_row_ptr,
    sched_q_lo_ptr,
    csr_q_global,
    csr_q_abs,
    csr_kv_block,
    csr_slot,
    q_batch_ptr,
    seq_lens,
    partial_o_ptr,
    partial_lse_ptr,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_po_n,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_pl_n,
    stride_pl_h,
    stride_pl_s,
    work_lo,
    work_hi,
    Q_TILE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    QH: tl.constexpr = Q_TILE * BLOCK_SIZE_H
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_qt = tl.arange(0, Q_TILE)
    d_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size

    q_smem = tle.gpu.alloc(
        [1, QH, BLOCK_SIZE_D],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, QH],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )

    i = 0
    for w in range(work_lo, work_hi):
        pid_row = tl.load(sched_row_ptr + w)
        q_lo = tl.load(sched_q_lo_ptr + w)
        pid_kh = pid_row % num_kv_heads
        pid_h = pid_kh * gqa_group_size
        row_end = tl.load(row_ptr + pid_row + 1)
        n_q = min(Q_TILE, row_end - q_lo)
        qt_mask = off_qt < n_q

        ent = q_lo + off_qt
        q_global = tl.load(csr_q_global + ent, mask=qt_mask, other=0).to(tl.int32)
        q_abs = tl.load(csr_q_abs + ent, mask=qt_mask, other=0).to(tl.int32)
        kv_block = tl.load(csr_kv_block + ent, mask=qt_mask, other=0).to(tl.int32)
        slot_id = tl.load(csr_slot + ent, mask=qt_mask, other=0).to(tl.int32)
        batch_id = tl.load(q_batch_ptr + q_global, mask=qt_mask, other=0).to(tl.int32)
        seq_len = tl.load(seq_lens + batch_id, mask=qt_mask, other=0).to(tl.int32)

        # Gather Q [QH, D] (scattered query ids cannot be TMA-loaded).
        q_rows = (q_global[:, None] * stride_qn) + (pid_h + off_h)[None, :] * stride_qh
        q_rows = tl.reshape(q_rows, QH)
        qh_mask = tl.reshape(qt_mask[:, None] & h_mask[None, :], QH)
        q = tl.load(
            q_ptr + q_rows[:, None] + off_d[None, :] * stride_qd,
            mask=qh_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q)

        k_slot = (2 * i) % CAPACITY
        gen_k = (2 * i) // CAPACITY
        tle.gpu.barrier_wait(kv_full[k_slot], phaseIdx=gen_k)
        qk_t = tle.gpu.wgmma(
            kv_stages.slot(k_slot), q_smem.slot(0), out_dtype=tl.float32, trans_b=True
        )  # [K, QH]
        qk_t = tle.gpu.wgmma_wait(0, qk_t)
        qk_t *= sm_scale_log2e

        kb_qh = tl.reshape(
            tl.broadcast_to(kv_block[:, None], (Q_TILE, BLOCK_SIZE_H)), QH
        )
        q_abs_qh = tl.reshape(
            tl.broadcast_to(q_abs[:, None], (Q_TILE, BLOCK_SIZE_H)), QH
        )
        seq_qh = tl.reshape(
            tl.broadcast_to(seq_len[:, None], (Q_TILE, BLOCK_SIZE_H)), QH
        )
        key_abs_qh = kb_qh[None, :] * BLOCK_SIZE_K + off_n[:, None]
        allowed = (q_abs_qh[None, :] >= key_abs_qh) & (key_abs_qh < seq_qh[None, :])
        qk_t = tl.where(allowed, qk_t, float("-inf"))

        m = tl.max(qk_t, axis=0)
        p = tl.exp2(qk_t - m[None, :])
        l = tl.sum(p, axis=0)
        tl.store(tle.gpu.local_ptr(p_smem.slot(0)), p.to(q_ptr.dtype.element_ty))

        v_slot = (2 * i + 1) % CAPACITY
        gen_v = (2 * i + 1) // CAPACITY
        tle.gpu.barrier_wait(kv_full[v_slot], phaseIdx=gen_v)
        acc_o_t = tl.zeros((BLOCK_SIZE_D, QH), dtype=tl.float32)
        acc_o_t = tle.gpu.wgmma(
            kv_stages.slot(v_slot), p_smem.slot(0), acc_o_t, trans_a=True
        )
        acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)
        tle.gpu.barrier_arrive(empty[k_slot], phaseIdx=gen_k)
        tle.gpu.barrier_arrive(empty[v_slot], phaseIdx=gen_v)

        inv_l = tl.where(l > 0, 1.0 / l, 0.0)
        acc_o_t *= inv_l[None, :]
        acc = tl.reshape(tl.trans(acc_o_t), Q_TILE, BLOCK_SIZE_H, BLOCK_SIZE_D)
        lse = tl.reshape(
            tl.where(l > 0, m + tl.log2(l), float("-inf")), Q_TILE, BLOCK_SIZE_H
        )

        po_base = (
            partial_o_ptr
            + q_global[:, None, None] * stride_po_n
            + (pid_h + off_h)[None, :, None] * stride_po_h
            + slot_id[:, None, None] * stride_po_s
            + off_d[None, None, :] * stride_po_d
        )
        po_mask = qt_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
        tl.store(po_base, acc.to(partial_o_ptr.dtype.element_ty), mask=po_mask)
        pl_base = (
            partial_lse_ptr
            + q_global[:, None] * stride_pl_n
            + (pid_h + off_h)[None, :] * stride_pl_h
            + slot_id[:, None] * stride_pl_s
        )
        tl.store(pl_base, lse, mask=qt_mask[:, None] & h_mask[None, :])
        i += 1


@triton.jit
def _kv_block_attn_ws_kernel(
    q_ptr,
    kv_cache_ptr,
    kv_cache_desc,
    row_ptr,
    csr_q_global,
    csr_q_abs,
    csr_kv_block,
    csr_slot,
    q_batch_ptr,
    seq_lens,
    partial_o_ptr,
    partial_lse_ptr,
    sched_row_ptr,
    sched_q_lo_ptr,
    num_work_ptr,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_po_n,
    stride_po_h,
    stride_po_s,
    stride_po_d,
    stride_pl_n,
    stride_pl_h,
    stride_pl_s,
    Q_TILE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    CAPACITY: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    WORKER_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    # num_work arrives as a 1-element device tensor so the host never syncs on
    # it (required for CUDA-graph capture of the full build+attend+merge path).
    num_work = tl.load(num_work_ptr)
    # Static, contiguous partition of the work-tile schedule across CTAs. CTAs
    # past the work boundary get work_lo >= work_hi and fall straight through.
    per = (num_work + NUM_CTAS - 1) // NUM_CTAS
    work_lo = pid * per
    work_hi = min(work_lo + per, num_work)

    kv_stages = tle.gpu.alloc(
        (CAPACITY, BLOCK_SIZE_K, BLOCK_SIZE_D),
        dtype=kv_cache_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    kv_full = tle.gpu.alloc_barriers(
        num_barriers=CAPACITY, expect_bytes=BLOCK_SIZE_K * BLOCK_SIZE_D * 2
    )
    empty = tle.gpu.alloc_barriers(
        num_barriers=CAPACITY, arrive_count=1, init=tle.gpu.READY
    )

    tle.gpu.warp_specialize(
        [
            (
                _kv_outer_ws_consumer,
                (
                    q_ptr,
                    kv_stages,
                    kv_full,
                    empty,
                    row_ptr,
                    sched_row_ptr,
                    sched_q_lo_ptr,
                    csr_q_global,
                    csr_q_abs,
                    csr_kv_block,
                    csr_slot,
                    q_batch_ptr,
                    seq_lens,
                    partial_o_ptr,
                    partial_lse_ptr,
                    num_kv_heads,
                    gqa_group_size,
                    head_dim,
                    sm_scale,
                    stride_qn,
                    stride_qh,
                    stride_qd,
                    stride_po_n,
                    stride_po_h,
                    stride_po_s,
                    stride_po_d,
                    stride_pl_n,
                    stride_pl_h,
                    stride_pl_s,
                    work_lo,
                    work_hi,
                    Q_TILE,
                    BLOCK_SIZE_K,
                    BLOCK_SIZE_D,
                    BLOCK_SIZE_H,
                    CAPACITY,
                ),
            ),
            (
                _kv_outer_ws_producer,
                (
                    kv_cache_desc,
                    kv_stages,
                    kv_full,
                    empty,
                    sched_row_ptr,
                    sched_q_lo_ptr,
                    row_ptr,
                    num_kv_heads,
                    work_lo,
                    work_hi,
                    BLOCK_SIZE_K,
                    BLOCK_SIZE_D,
                    CAPACITY,
                ),
            ),
        ],
        worker_num_warps=[1],
        worker_num_regs=[32],
    )


@torch.no_grad()
def _minimax_m3_sparse_attn_kv_outer(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
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
    """KV-outer BF16 block-sparse prefill (block_size_q == 1).

    Pass a prebuilt ``csr`` (from ``build_k2q_csr``) to exclude the CSR build
    from timing; otherwise it is built here. Set ``return_csr`` to also return
    the CSR for reuse.

    ``kernel`` selects the attend implementation:
      - ``"ws"``   : warp-specialized + pipelined TLE (producer streams KV via
                     TMA into a multi-slot ring, consumer runs WGMMA) -- closest
                     to the CuTe fmha_sm100 mainloop. Default when TLE is present.
      - ``"tle"``  : single-stage TLE (KV staged once per work-tile, no overlap).
      - ``"plain"``: plain-Triton reference (tl.load/tl.dot), correctness oracle.
    ``use_tle`` is kept for back-compat: True -> "tle", False -> "plain".
    """
    if kv_cache.dtype != torch.bfloat16:
        raise NotImplementedError("kv-outer v1 supports BF16 prefill only")
    if kernel is None:
        # Both "tle" and "ws" are TLE kernels; "tle" is single-stage (KV staged
        # once per work-tile), "ws" adds a producer/consumer ring buffer. ncu
        # shows the ring buffer costs 64KB extra SMEM + more registers, pinning
        # occupancy at 2 block/SM (SM throughput 27%), while single-stage TLE
        # runs at 49KB/72reg -> 4-7 block/SM, 52% throughput, ~1.7x faster. The
        # ring never pays off because kv-outer work-tiles are tiny (one KV page
        # per program, nothing to pipeline across). So default to single-stage.
        if use_tle is None:
            kernel = "tle" if tle is not None else "plain"
        else:
            kernel = "tle" if use_tle else "plain"
    if kernel in ("ws", "tle") and tle is None:
        raise RuntimeError(
            f"kernel={kernel!r} needs triton TLE, which is not importable"
        )
    use_tle = kernel in ("ws", "tle")
    if q_tile is None:
        # TLE stages Q [Q_TILE*H, D] and P [K, Q_TILE*H] in SMEM; Q_TILE*H is the
        # WGMMA N and must be >= 16. ncu q_tile sweep (single-stage TLE, seq16384)
        # measured q_tile=4 as the optimum for BOTH group 8 (N=32, 1575us) and
        # group 16 (N=64, 2385us): q_tile 2 underfills the MMA, q_tile>=8 inflates
        # SMEM/registers and regresses. The plain-Triton reference has no SMEM
        # ceiling and prefers a wider tile.
        q_tile = 4 if use_tle else 32
    total_q, num_heads, head_dim = q.shape
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads

    num_pages = kv_cache.shape[0]
    if csr is None:
        builder = build_k2q_csr_fused if use_fused_csr else build_k2q_csr
        csr = builder(topk_idx, cu_seqlens_q, prefix_lens, block_table, num_pages)
    # build_k2q_csr already computes the per-query batch index; reuse it to avoid
    # a redundant repeat_interleave launch (q_batch missing on older prebuilt csr).
    q_batch = csr.get("q_batch")
    if q_batch is None:
        q_batch = _q_batch_index(cu_seqlens_q, total_q)

    block_size_d = triton.next_power_of_2(head_dim)
    block_size_h = triton.next_power_of_2(gqa_group_size)

    # partial_o is bf16 (halves the HBM round-trip of the two-pass split/merge,
    # which dominates on this problem size). The LSE gates empty slots, so
    # partial_o need not be pre-zeroed; partial_lse must be -inf-initialized.
    partial_o = torch.empty(
        (total_q, num_heads, topk, head_dim), device=q.device, dtype=torch.bfloat16
    )
    partial_lse = torch.full(
        (total_q, num_heads, topk), float("-inf"), device=q.device, dtype=torch.float32
    )

    # Flattened work-tile schedule: each (row, q_tile) chunk that actually has
    # queries becomes one program. This balances long/short rows and avoids the
    # empty-program waste of a static 2-D grid (the CuTe scheduler's role).
    #
    # The ws kernel partitions work statically across a fixed CTA grid and reads
    # num_work in-kernel, so it uses the sync-free fused schedule (device scalar
    # num_work, over-allocated arrays) -- no .item(). The tle/plain paths launch
    # a grid of num_work programs and need the host count, so they keep the
    # host schedule builder.
    if kernel == "ws" and "nnz_cap" in csr:
        sched_row, sched_q_lo, num_work_dev = _build_schedule_fused(
            csr["row_ptr"], q_tile, csr["nnz_cap"]
        )
        num_work = None  # deferred to device; do not sync
    else:
        sched_row, sched_q_lo = _build_schedule(csr["row_ptr"], q_tile)
        num_work = sched_row.numel()
        num_work_dev = torch.tensor([num_work], device=q.device, dtype=torch.int32)

    grid = (max(1, num_work if num_work is not None else 1),)
    if kernel == "ws":
        from triton.tools.tensor_descriptor import TensorDescriptor

        triton.set_allocator(_tma_allocator)
        kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
        kv_cache_desc = TensorDescriptor(
            kv_cache_2d,
            shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
            strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
            block_shape=[SPARSE_BLOCK_SIZE, head_dim],
        )
        if num_ctas is None:
            num_ctas = (
                2 * torch.cuda.get_device_properties(q.device).multi_processor_count
            )
        # Launch a fixed CTA grid; CTAs beyond the work boundary exit early.
        # Capping at num_work would force a host sync, defeating graph capture.
        num_ctas = max(1, num_ctas)
        ws_grid = (num_ctas,)
        _kv_block_attn_ws_kernel[ws_grid](
            q,
            kv_cache,
            kv_cache_desc,
            csr["row_ptr"],
            csr["q_global"],
            csr["q_abs"],
            csr["kv_block"],
            csr["slot"],
            q_batch,
            seq_lens,
            partial_o,
            partial_lse,
            sched_row,
            sched_q_lo,
            num_work_dev,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_o.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            Q_TILE=q_tile,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=block_size_d,
            BLOCK_SIZE_H=block_size_h,
            CAPACITY=capacity,
            NUM_CTAS=num_ctas,
            WORKER_WARPS=1,
            num_warps=4,
            num_stages=1,
            **({"maxnreg": maxnreg} if maxnreg is not None else {}),
        )
    elif kernel == "tle":
        from triton.tools.tensor_descriptor import TensorDescriptor

        triton.set_allocator(_tma_allocator)
        kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
        kv_cache_desc = TensorDescriptor(
            kv_cache_2d,
            shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
            strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
            block_shape=[SPARSE_BLOCK_SIZE, head_dim],
        )
        _kv_block_attn_tle_kernel[grid](
            q,
            kv_cache,
            kv_cache_desc,
            csr["row_ptr"],
            csr["q_global"],
            csr["q_abs"],
            csr["kv_block"],
            csr["slot"],
            q_batch,
            seq_lens,
            partial_o,
            partial_lse,
            sched_row,
            sched_q_lo,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            topk,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_o.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            Q_TILE=q_tile,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=block_size_d,
            BLOCK_SIZE_H=block_size_h,
            num_warps=4,
            num_stages=1,
        )
    else:
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

    # N_TILE queries x MAX_TOPK x head_dim fp32 lives in registers per program;
    # scale it down as topk grows so the tile stays register-resident.
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
        return output, csr
    return output


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
    """GQA block-sparse attention over the selected blocks. block_size_q == 1.

    For long BF16 prompts the work is routed to the kv-outer implementation
    (physical KV page loaded once per CTA and reused across the queries that
    selected it, then an LSE merge), which beats this q-outer path once the
    per-request query count makes KV reuse pay back the CSR build + merge.
    Short prompts, fp8 KV, scaled KV, and TLE-less builds keep the q-outer body
    below. The public signature and semantics are unchanged.
    """
    if _should_use_kv_outer_prefill(
        kv_cache, max_query_len, k_scale, v_scale
    ) and _tma_supports_head_dim(q.shape[-1], kv_cache.element_size()):
        _minimax_m3_sparse_attn_kv_outer(
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
        )
        return

    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
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
    grid = (max_query_len, num_kv_heads, batch)
    block_size_h = triton.next_power_of_2(gqa_group_size)

    # Head dimensions whose V box origin is not 16-byte aligned cannot form a
    # legal TMA descriptor, so they must reach a non-TMA kernel; treat them the
    # same as a TLE-less build.
    tma_ok = _tma_supports_head_dim(head_dim, kv_cache.element_size())

    # Dispatch order mirrors upstream: no-TLE fallback, then the tl.dot kernel
    # for cases that cannot form a legal WGMMA (fp8 KV has a dtype mismatch with
    # BF16 Q; a GQA tile below eight heads has an illegal WGMMA N), then TLE.
    if not use_fp8 and block_size_h >= 8 and (tle is None or not tma_ok):
        _gqa_sparse_fwd_fallback_kernel[grid](
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
            topk,
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
            USE_FP8=False,
            KV_SCALE_MODE=_KV_SCALE_NONE,
        )
        return

    # Route to the tl.dot kernel for fp8 (dtype mismatch against BF16 Q) and
    # for GQA tiles the TLE kernel cannot form a legal WGMMA for. Upstream's
    # bound is block_size_h < 8; this fork needs < 16 because storing P as
    # [BLOCK_SIZE_K, BLOCK_SIZE_QH] and issuing PV as
    # `wgmma(kv, p, acc, trans_a=True)` reads out of bounds at QH == 8
    # (measured: illegal memory access at gqa 6/7/8, exact at gqa >= 10).
    # fp8 with scalar scales goes to the TLE kernel; the tl.dot path remains
    # for per-token scales (not yet ported) and for GQA tiles too small to form
    # a legal WGMMA.
    # BLOCK_SIZE_QH >= 8 is enough here, unlike the bf16 path's >= 16. That
    # larger bound exists because the bf16 kernel's PV is a transposed-A WGMMA,
    # which silently returns a wrong result at N == 8. This kernel only uses
    # WGMMA for QK (trans_b, both operands K-major) and keeps PV on tl.dot, so
    # the transposed-A defect cannot apply. Verified directly: fp8 QK at QH = 8
    # returns the exact result (128.0 for an exact-128.0 reduction).
    use_fp8_tle = (
        use_fp8
        and tle is not None
        and tma_ok
        and kv_scale_mode == _KV_SCALE_SCALAR
        and block_size_h >= _MSA_PREFILL_FP8_MIN_QH
    )
    if use_fp8_tle:
        from triton.tools.tensor_descriptor import TensorDescriptor

        triton.set_allocator(_tma_allocator)
        kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
        k_desc = TensorDescriptor(
            kv_cache_2d,
            shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
            strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
            block_shape=[SPARSE_BLOCK_SIZE, _tma_box_width(head_dim)],
        )
        _gqa_sparse_fwd_tle_fp8_kernel[grid](
            q,
            kv_cache,
            k_desc,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            output,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
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
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
            maxnreg=_MSA_PREFILL_FP8_MAXNREG,
        )
        return

    if use_fp8 or block_size_h < 8:
        _gqa_sparse_fwd_kernel[grid](
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
            topk,
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
            num_warps=4,
            num_stages=3,
        )
        return

    from triton.tools.tensor_descriptor import TensorDescriptor

    triton.set_allocator(_tma_allocator)
    qh8_choice = (
        _prefill_qh8_kernel_choice(
            q.device, total_q, batch, max_query_len, num_kv_heads
        )
        if block_size_h <= _PREFILL_HALF_KV_MAX_BLOCK_SIZE_QH
        else "fork"
    )
    if qh8_choice == "upstream":
        # Exactly upstream's launch for this case: half-height TMA boxes for
        # the two-stage KV pipeline, and the same constexprs.
        kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
        kv_cache_desc = TensorDescriptor(
            kv_cache_2d,
            shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
            strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
            block_shape=[SPARSE_BLOCK_SIZE // 2, _tma_box_width(head_dim)],
        )
        _gqa_sparse_fwd_tle_upstream_kernel[grid](
            q,
            kv_cache,
            kv_cache_desc,
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
            USE_TL_DOT_PATH=False,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            USE_HALF_KV_PIPE=True,
            WAR_BARRIER=not _prefill_drop_war_barrier(
                total_q, batch, max_query_len, num_kv_heads
            ),
            num_warps=4,
            num_stages=1,
        )
        return
    kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
    kv_cache_desc = TensorDescriptor(
        kv_cache_2d,
        shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
        strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
        block_shape=[SPARSE_BLOCK_SIZE, _tma_box_width(head_dim)],
    )
    if qh8_choice == "prefetch":
        _gqa_sparse_fwd_tle_prefetch_kernel[grid](
            q,
            kv_cache,
            kv_cache_desc,
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
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            KEEP_TRANS=block_size_h < _MSA_PREFILL_UNTRANSPOSED_MIN_QH,
            num_warps=4,
            num_stages=1,
        )
        return
    _gqa_sparse_fwd_tle_kernel[grid](
        q,
        kv_cache,
        kv_cache_desc,
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
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        KEEP_TRANS=block_size_h < _MSA_PREFILL_UNTRANSPOSED_MIN_QH,
        num_warps=4,
        num_stages=1,
    )
