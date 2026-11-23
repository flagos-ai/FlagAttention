# Copyright 2026 FlagOS Contributors
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

"""Ascend launch and merge paths for MoBA.

The varlen and routed attention math uses the shared Triton kernels.
Ascend supplies grid partitioning, workspace limits and an FP32 merge.
The merge avoids CUDA-specific autotuning and uses int64 routed
addresses for large training buffers. Routing selection is in
``routing.py``.
"""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.moba.attention import (
    _routed_gqa_attn_fwd_kernel,
    _varlen_attn_bwd_dkdv_kernel,
    _varlen_attn_bwd_dq_kernel,
    _varlen_attn_fwd_kernel,
)
from flag_attn.FLA.moba.routing import cdiv
from flag_attn.runtime.backend._ascend.FLA.moba.routing import (
    _MAX_LAUNCH_GRID,
    _ROUTED_DESC_SLICE,
    _VARLEN_PROGRAMS_PER_SLICE,
)

# ``tl.dot`` needs a 16-wide reduction.  BLOCK_M=BLOCK_N=64 keeps the FP32
# score tile and both K/V tiles inside the 910B unified buffer with one stage.
_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_M_SMALL_D = 32
_BLOCK_N_SMALL_D = 32
_NUM_WARPS = 4
_NUM_STAGES = 1

# Merge-kernel tiling.  One program owns ``_MERGE_ROWS_PER_PROGRAM`` contiguous
# ``(token, head)`` rows, which drops the launch size by that factor (rows=32
# measured fastest on 910B4 for both small and 4.2M-row problems).
_MERGE_ROWS_PER_PROGRAM = 32
_MERGE_BLOCK_D = 128
_MERGE_NUM_WARPS = 4
# Triton-Ascend rejects launches whose *total* program count exceeds 65535
# (EE1003/coreDim).  Larger problems are covered by a grid-stride loop over row
# tiles instead of by growing the grid.
_MERGE_MAX_PROGRAMS = 65535


def _attention_launch_config(head_dim: int, device: torch.device):
    block_d = max(16, triton.next_power_of_2(head_dim))
    if head_dim <= 128:
        return _BLOCK_M, _BLOCK_N, block_d, _NUM_WARPS, _NUM_STAGES
    return _BLOCK_M_SMALL_D, _BLOCK_N_SMALL_D, block_d, _NUM_WARPS, _NUM_STAGES


# One routed program owns a row tile for one (chunk, query head) expert.
# The launcher and descriptor builder must use the same row tile.
# At D <= 128, increasing the tile from 64 to 128 roughly halves the
# descriptor count and amortizes program startup and expert K/V loads.
# The measured routed stage at 8192 tokens on 910B fell from 10.37 to
# 7.83 ms with identical output. The larger tile raises per-program
# workspace, which is bounded by the descriptor slice below.
# Larger head dimensions use the smaller shared tile.
_ROUTED_BLOCK_M = 128


def _routed_attention_launch_config(head_dim: int, device: torch.device):
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(
        head_dim, device
    )
    if head_dim <= 128:
        return _ROUTED_BLOCK_M, block_n, block_d, num_warps, num_stages
    return block_m, block_n, block_d, num_warps, num_stages


def _varlen_sequence_plan(cu_seqlens_q, cu_seqlens_k, num_blocks, num_heads):
    """Return contiguous sequence slices that keep every launch grid legal.

    The shared varlen kernels derive ``(batch, head, block)`` from
    ``program_id(0)`` and the ``cu_seqlens`` pointers, so the only legal host
    partition is over the packed sequence axis.  ``None`` means that the
    original single launch already fits and must be kept untouched (it is the
    bit-identity baseline).

    Ascend rejects a launch whose grid product exceeds 65535
    (``Invalid_Argument EE1003: ... coreDim ...``) and reserves a per-program
    workspace (measured: 56-192 KiB for these kernels), so the slice size is
    the tighter of the legality limit and ``_VARLEN_PROGRAMS_PER_SLICE``.
    """
    num_sequences = cu_seqlens_q.numel() - 1
    if cu_seqlens_k.numel() - 1 != num_sequences:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must describe the same number of sequences, "
            f"got {num_sequences} and {cu_seqlens_k.numel() - 1}"
        )
    programs_per_slice = min(_VARLEN_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID)
    if num_blocks * num_sequences * num_heads <= programs_per_slice:
        return None
    if num_blocks * num_heads > _MAX_LAUNCH_GRID:
        raise RuntimeError(
            "a single varlen sequence needs "
            f"{num_blocks} blocks x {num_heads} heads programs, which exceeds the "
            f"Ascend grid limit of {_MAX_LAUNCH_GRID}; the shared kernel derives its "
            "block index from program_id(0) so it cannot be split further"
        )
    sequences_per_launch = max(1, programs_per_slice // (num_blocks * num_heads))
    # A few dozen host integers per oversized call: the packed tensors can only
    # be sliced with concrete token bounds.
    q_bounds = cu_seqlens_q.cpu().tolist()
    k_bounds = cu_seqlens_k.cpu().tolist()
    plan = []
    for seq_start in range(0, num_sequences, sequences_per_launch):
        seq_end = min(seq_start + sequences_per_launch, num_sequences)
        plan.append(
            (
                seq_start,
                seq_end,
                q_bounds[seq_start],
                q_bounds[seq_end],
                k_bounds[seq_start],
                k_bounds[seq_end],
            )
        )
    return plan


def _triton_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    group_size = num_heads // num_kv_heads
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty((num_heads, total_q), device=q.device, dtype=torch.float32)
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    num_blocks = cdiv(max_seqlen_q, block_m)
    plan = _varlen_sequence_plan(cu_seqlens_q, cu_seqlens_k, num_blocks, num_heads)
    if plan is None:
        plan = [(0, cu_seqlens_q.numel() - 1, 0, total_q, 0, k.shape[0])]
    for seq_start, seq_end, q_start, q_end, k_start, k_end in plan:
        if q_start == 0 and q_end == total_q and k_start == 0 and k_end == k.shape[0]:
            q_slice, k_slice, v_slice, out_slice, lse_slice = q, k, v, out, lse
            cu_q_slice, cu_k_slice = cu_seqlens_q, cu_seqlens_k
        else:
            q_slice, k_slice, v_slice = q[q_start:q_end], k[k_start:k_end], v[k_start:k_end]
            out_slice, lse_slice = out[q_start:q_end], lse[:, q_start:q_end]
            # Offsets are relative to the packed slice, exactly like a fresh
            # variable-length problem covering only these sequences.
            cu_q_slice = cu_seqlens_q[seq_start : seq_end + 1] - q_start
            cu_k_slice = cu_seqlens_k[seq_start : seq_end + 1] - k_start
        grid = (num_blocks * (seq_end - seq_start) * num_heads,)
        _varlen_attn_fwd_kernel[grid](
            q_slice,
            k_slice,
            v_slice,
            out_slice,
            lse_slice,
            cu_q_slice,
            cu_k_slice,
            q_slice.stride(0),
            q_slice.stride(1),
            q_slice.stride(2),
            k_slice.stride(0),
            k_slice.stride(1),
            k_slice.stride(2),
            v_slice.stride(0),
            v_slice.stride(1),
            v_slice.stride(2),
            out_slice.stride(0),
            out_slice.stride(1),
            out_slice.stride(2),
            lse_slice.stride(0),
            lse_slice.stride(1),
            num_heads=num_heads,
            GROUP_SIZE=group_size,
            softmax_scale=softmax_scale,
            HEAD_DIM=head_dim,
            MAX_SEQLEN_K=max_seqlen_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            NUM_BLOCKS=num_blocks,
            CAUSAL=causal,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out, lse


def _triton_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    _, num_heads, head_dim = q.shape
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    num_sequences = cu_seqlens_q.numel() - 1
    num_q_blocks = cdiv(max_seqlen_q, block_m)
    num_k_blocks = cdiv(max_seqlen_k, block_n)
    dq_plan = _varlen_sequence_plan(cu_seqlens_q, cu_seqlens_k, num_q_blocks, num_heads)
    dkdv_plan = _varlen_sequence_plan(cu_seqlens_q, cu_seqlens_k, num_k_blocks, num_heads)
    if dq_plan is None:
        dq_plan = [(0, num_sequences, 0, q.shape[0], 0, k.shape[0])]
    if dkdv_plan is None:
        dkdv_plan = [(0, num_sequences, 0, q.shape[0], 0, k.shape[0])]

    def _slices(seq_start, seq_end, q_start, q_end, k_start, k_end):
        if q_start == 0 and q_end == q.shape[0] and k_start == 0 and k_end == k.shape[0]:
            return (
                q,
                k,
                v,
                out,
                dout,
                softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_k,
            )
        # Every packed tensor is sliced along its own token axis, so the views
        # keep the caller's strides and the kernels' offsets stay local.
        return (
            q[q_start:q_end],
            k[k_start:k_end],
            v[k_start:k_end],
            out[q_start:q_end],
            dout[q_start:q_end],
            softmax_lse[:, q_start:q_end],
            dq[q_start:q_end],
            dk[k_start:k_end],
            dv[k_start:k_end],
            cu_seqlens_q[seq_start : seq_end + 1] - q_start,
            cu_seqlens_k[seq_start : seq_end + 1] - k_start,
        )

    for seq_start, seq_end, q_start, q_end, k_start, k_end in dq_plan:
        (
            q_s,
            k_s,
            v_s,
            out_s,
            dout_s,
            lse_s,
            dq_s,
            _,
            _,
            cu_q_s,
            cu_k_s,
        ) = _slices(seq_start, seq_end, q_start, q_end, k_start, k_end)
        grid = (num_q_blocks * (seq_end - seq_start) * num_heads,)
        _varlen_attn_bwd_dq_kernel[grid](
            q_s,
            k_s,
            v_s,
            out_s,
            dout_s,
            lse_s,
            dq_s,
            cu_q_s,
            cu_k_s,
            q_s.stride(0),
            q_s.stride(1),
            q_s.stride(2),
            k_s.stride(0),
            k_s.stride(1),
            k_s.stride(2),
            v_s.stride(0),
            v_s.stride(1),
            v_s.stride(2),
            out_s.stride(0),
            out_s.stride(1),
            out_s.stride(2),
            dout_s.stride(0),
            dout_s.stride(1),
            dout_s.stride(2),
            lse_s.stride(0),
            lse_s.stride(1),
            dq_s.stride(0),
            dq_s.stride(1),
            dq_s.stride(2),
            num_heads=num_heads,
            softmax_scale=softmax_scale,
            HEAD_DIM=head_dim,
            MAX_SEQLEN_K=max_seqlen_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            NUM_BLOCKS=num_q_blocks,
            CAUSAL=causal,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    for seq_start, seq_end, q_start, q_end, k_start, k_end in dkdv_plan:
        (
            q_s,
            k_s,
            v_s,
            out_s,
            dout_s,
            lse_s,
            _,
            dk_s,
            dv_s,
            cu_q_s,
            cu_k_s,
        ) = _slices(seq_start, seq_end, q_start, q_end, k_start, k_end)
        grid = (num_k_blocks * (seq_end - seq_start) * num_heads,)
        _varlen_attn_bwd_dkdv_kernel[grid](
            q_s,
            k_s,
            v_s,
            out_s,
            dout_s,
            lse_s,
            dk_s,
            dv_s,
            cu_q_s,
            cu_k_s,
            q_s.stride(0),
            q_s.stride(1),
            q_s.stride(2),
            k_s.stride(0),
            k_s.stride(1),
            k_s.stride(2),
            v_s.stride(0),
            v_s.stride(1),
            v_s.stride(2),
            out_s.stride(0),
            out_s.stride(1),
            out_s.stride(2),
            dout_s.stride(0),
            dout_s.stride(1),
            dout_s.stride(2),
            lse_s.stride(0),
            lse_s.stride(1),
            dk_s.stride(0),
            dk_s.stride(1),
            dk_s.stride(2),
            dv_s.stride(0),
            dv_s.stride(1),
            dv_s.stride(2),
            num_heads=num_heads,
            softmax_scale=softmax_scale,
            HEAD_DIM=head_dim,
            MAX_SEQLEN_Q=max_seqlen_q,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            NUM_BLOCKS=num_k_blocks,
            CAUSAL=causal,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _routed_sparse_attention_triton(
    q,
    k,
    v,
    query_indices,
    expert_offsets,
    descriptor_experts,
    descriptor_starts,
    cu_seqlens,
    max_chunks_per_sequence,
    chunk_size,
):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    head_dim = q.shape[2]
    block_m, block_n, block_d, num_warps, num_stages = _routed_attention_launch_config(
        head_dim, q.device
    )
    route_out = torch.empty((query_indices.numel(), head_dim), device=q.device, dtype=q.dtype)
    route_lse = torch.empty((query_indices.numel(),), device=q.device, dtype=torch.float32)
    num_descriptors = descriptor_experts.numel()
    # Descriptors own disjoint output and LSE rows, so contiguous slices can
    # be launched independently into the same output buffers.
    #
    # Workspace scales with the number of launched programs. The 128-row
    # tile uses about 490 KiB per program on the measured 910B4 shape.
    # The one-eighth slice limits the peak on short passes, while the fixed
    # descriptor cap bounds workspace for longer passes. The grid limit
    # applies independently to each slice.
    #
    # The input descriptor capacity can include padding. The kernel masks
    # inactive descriptors, and they still count toward the launch grid.
    descriptor_slice = min(
        _ROUTED_DESC_SLICE,
        _MAX_LAUNCH_GRID,
        max(1, (num_descriptors + 7) // 8),
    )
    for descriptor_start in range(0, num_descriptors, descriptor_slice):
        descriptor_end = min(descriptor_start + descriptor_slice, num_descriptors)
        _routed_gqa_attn_fwd_kernel[(descriptor_end - descriptor_start,)](
            q,
            k,
            v,
            query_indices,
            expert_offsets,
            descriptor_experts[descriptor_start:descriptor_end],
            descriptor_starts[descriptor_start:descriptor_end],
            cu_seqlens,
            route_out,
            route_lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            route_out.stride(0),
            route_out.stride(1),
            q.shape[-1] ** -0.5,
            NUM_Q_HEADS=num_q_heads,
            GROUP_SIZE=group_size,
            MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
            CHUNK_SIZE=chunk_size,
            HEAD_DIM=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return route_out, route_lse


class _TritonVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, causal):
        scale = q.shape[-1] ** -0.5
        out, lse = _triton_varlen_forward(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
        ctx.max_q, ctx.max_k, ctx.scale, ctx.causal = max_q, max_k, scale, causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _triton_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            ctx.scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None


def _triton_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    return _TritonVarlenAttention.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


def _merge_launch_config(num_rows: int, head_dim: int):
    """Pick the row tile and launch size for the merge kernel.

    Returns ``(rows_per_program, block_d, num_tiles, grid)``.  ``grid`` equals
    ``cdiv(num_rows, rows_per_program)`` whenever that many programs are legal
    (<= 65535 on Ascend) and is capped otherwise; the kernel then grid-strides
    over the remaining row tiles.
    """
    rows = _MERGE_ROWS_PER_PROGRAM
    block_d = min(_MERGE_BLOCK_D, max(16, triton.next_power_of_2(head_dim)))
    num_tiles = cdiv(num_rows, rows)
    return rows, block_d, num_tiles, min(num_tiles, _MERGE_MAX_PROGRAMS)


def _fused_merge_softmax_triton(
    self_out,
    self_lse,
    moba_out,
    moba_lse,
    route_positions,
    final_out,
    final_lse,
):
    """Merge local and routed attention outputs over a row tile.

    Each program owns a ``[_MERGE_ROWS_PER_PROGRAM, BLOCK_D]`` slab of rows and
    computes, for every row, the numerically stable combination of one "self"
    Softmax-LSE term with up to ``topk`` routed terms::

        m = max(lse_self, lse_route_0, ...)
        lse = log(exp(lse_self - m) + sum_r exp(lse_route_r - m)) + m
        out = (exp(lse_self - m) * O_self
               + sum_r exp(lse_route_r - m) * O_route_r) / exp(lse - m)

    All arithmetic is fp32; only the final store narrows to the output dtype.
    Missing routes (``route_positions < 0``, used as padding) and empty route
    sets are skipped exactly.

    Layout (as produced by ``parallel.py``): ``self_out``/``final_out`` are
    ``(tokens * num_heads, head_dim)`` views, ``self_lse``/``final_lse`` are
    ``(num_heads, tokens)`` (so ``num_rows`` must be divisible by ``num_heads``),
    ``route_positions`` is ``(topk, num_rows)`` and ``moba_out``/``moba_lse`` are
    indexed directly by the route positions.  The grid is
    ``cdiv(num_rows, rows)`` when legal and is capped at 65535 programs, in
    which case a program loops over several row tiles (Triton-Ascend rejects
    larger launches).

    The largest official case has 8,388,608 rows and 262,144 row tiles.
    Grid striding keeps each merge pass to one launch. The kernel uses
    vector arithmetic and requires no per-program dot workspace.
    Route positions index compacted rows, whose element offset can exceed
    int32 even when the row index fits. Routed value loads widen the row
    position to int64 before multiplying by its stride.

    The shared kernel cannot be used: Triton-Ascend fails to legalize its
    ``tl.where(mask, positions, 0)`` scalar materialization, and its autotune
    key issues CUDA capability queries.
    """
    num_rows, head_dim = self_out.shape
    topk = route_positions.shape[0]
    if topk == 0 or moba_out.numel() == 0:
        # A vendor copy avoids a Triton copy grid (which itself exceeds
        # the 65535-program launch limit at long sequences).
        if final_out.data_ptr() != self_out.data_ptr():
            final_out.copy_(self_out)
        if final_lse.data_ptr() != self_lse.data_ptr():
            final_lse.copy_(self_lse)
        return

    rows, block_d, num_tiles, grid = _merge_launch_config(num_rows, head_dim)
    # With at most 8 routes the whole route axis is gathered as one
    # ``[8, ROWS]`` tile, so the max and normalizer passes share a single
    # ``route_positions``/``moba_lse`` gather.  More routes keep the shipped
    # per-route loop (the topk=33/63 tests); the kernel picks between the two
    # bodies on this constexpr.
    num_routes_pad = 8 if topk <= 8 else 0
    _fused_merge_softmax_kernel[(grid,)](
        self_out,
        self_lse,
        moba_out,
        moba_lse,
        route_positions,
        final_out,
        final_lse,
        num_rows,
        head_dim,
        num_tiles,
        grid,
        self_out.stride(0),
        self_out.stride(1),
        self_lse.stride(0),
        self_lse.stride(1),
        moba_out.stride(0),
        moba_out.stride(1),
        route_positions.stride(0),
        route_positions.stride(1),
        final_out.stride(0),
        final_out.stride(1),
        final_lse.stride(0),
        final_lse.stride(1),
        num_heads=self_lse.shape[0],
        BLOCK_D=block_d,
        ROWS=rows,
        NUM_ROUTES=topk,
        NUM_ROUTES_PAD=num_routes_pad,
        num_warps=_MERGE_NUM_WARPS,
        num_stages=1,
    )


def _fused_merge_softmax_triton_inplace(
    self_out,
    self_lse,
    moba_out,
    moba_lse,
    route_positions,
    final_out,
    final_lse,
):
    """In-place variant of :func:`_fused_merge_softmax_triton`.

    Safe when ``final_out is self_out`` (and/or ``final_lse is self_lse``), i.e.
    when the caller wants to merge a ping-pong pair without allocating a third
    buffer.  Every program reads only the rows it owns from ``self_out`` and
    ``self_lse`` and writes exactly those rows (the routed terms come from
    ``moba_out``/``moba_lse``, which must not alias ``self_out``/``final_out``).
    Row ownership is disjoint by construction - program ``p`` handles row tiles
    ``p, p + grid, ...`` and a tile covers ``[tile * rows, (tile + 1) * rows)`` -
    so no program can observe a value another program has already overwritten,
    and the fp32 weights/LSEs are materialized before the output store.  The
    same holds per ``BLOCK_D`` chunk when ``head_dim > BLOCK_D``.

    The aliasing requirement is exactly "same tensor or disjoint storage"; the
    usual ``current_out -> next_out`` ping-pong already satisfies it, and
    ``self_lse`` may alias ``final_lse`` as well.  No kernel code differs from
    the out-of-place path, so results are bit-identical to it.

    For a single pass, exact aliasing saves an output payload of
    ``num_rows * head_dim * element_size`` bytes and an FP32 LSE vector.
    Multi-pass callers can still use two disjoint buffer pairs.
    """
    _fused_merge_softmax_triton(
        self_out,
        self_lse,
        moba_out,
        moba_lse,
        route_positions,
        final_out,
        final_lse,
    )


@triton.jit
def _fused_merge_softmax_kernel(
    self_out_ptr,
    self_lse_ptr,
    moba_out_ptr,
    moba_lse_ptr,
    route_positions_ptr,
    out_ptr,
    final_lse_ptr,
    num_rows,
    head_dim,
    num_tiles,
    num_programs,
    stride_self_row,
    stride_self_d,
    stride_self_lse_head,
    stride_self_lse_token,
    stride_moba_row,
    stride_moba_d,
    stride_route_rank,
    stride_route_row,
    stride_out_row,
    stride_out_d,
    stride_final_lse_head,
    stride_final_lse_token,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
    NUM_ROUTES_PAD: tl.constexpr,
):
    """Row-tiled merge kernel; see ``_fused_merge_softmax_triton``.

    ``num_tiles = cdiv(num_rows, ROWS)``; program ``p`` processes row tiles
    ``p, p + num_programs, ...`` so the launch stays within Ascend's 65535
    program limit while each tile stays small (``[ROWS, BLOCK_D]``).  Rows
    outside ``[0, num_rows)`` (the tail of the last tile) are masked on every
    load and store.

    With ``NUM_ROUTES_PAD > 0`` (at most 8 routes) the route axis is loaded as
    one ``[NUM_ROUTES_PAD, ROWS]`` tile of positions and LSEs, shared by the max
    pass, the normalizer pass and the routed value loop.  ``tl.max`` over the
    route axis is order-free, so it equals the shipped per-route running max.
    The normalizer keeps the shipped *sequential route order* by extracting one
    lane per route (adding zeros is exact) before the running fp32 sum, and the
    value loop reuses those per-route weights while reloading only the 1-D
    positions it needs for the gather.  Results are therefore bit-identical to
    the shipped loop.  With ``NUM_ROUTES_PAD == 0`` (more than 8 routes) the
    shipped per-route loop runs verbatim.
    """
    pid = tl.program_id(0)
    offs_r = tl.arange(0, ROWS)
    offs_d = tl.arange(0, BLOCK_D)

    for tile in range(pid, num_tiles, num_programs):
        row_ids = tile * ROWS + offs_r
        row_ok = row_ids < num_rows
        # Clamped row ids keep every masked access free of out-of-range
        # offsets, which the Ascend compiler requires for masked loads.
        r = tl.where(row_ok, row_ids, 0)

        token = r // num_heads
        head = r - token * num_heads
        self_lse = tl.load(
            self_lse_ptr + head * stride_self_lse_head + token * stride_self_lse_token,
            mask=row_ok,
            other=0.0,
        ).to(tl.float32)

        if NUM_ROUTES_PAD > 0:
            k_offs = tl.arange(0, NUM_ROUTES_PAD)
            k_ok = k_offs < NUM_ROUTES
            pos2 = tl.load(
                route_positions_ptr
                + k_offs[:, None] * stride_route_rank
                + r[None, :] * stride_route_row,
                mask=k_ok[:, None] & row_ok[None, :],
                other=-1,
            )
            lse2 = tl.load(
                moba_lse_ptr + tl.maximum(pos2, 0),
                mask=pos2 >= 0,
                other=-float("inf"),
            ).to(tl.float32)

            # Pass 1: max over the route axis; padded lanes are -inf and the
            # reduction is order-free.
            max_lse = tl.maximum(self_lse, tl.max(lse2, axis=0))

            # Pass 2: extract one route lane at a time and accumulate in route
            # order, exactly like the shipped loop (the other lanes add 0.0).
            w2 = tl.where(pos2 >= 0, tl.exp(lse2 - max_lse[None, :]), 0.0)
            self_se = tl.exp(self_lse - max_lse)
            total_se = self_se
            for k in tl.static_range(NUM_ROUTES):
                total_se += tl.sum(tl.where(k_offs[:, None] == k, w2, 0.0), axis=0)

            merged_lse = tl.log(total_se) + max_lse
            tl.store(
                final_lse_ptr + head * stride_final_lse_head + token * stride_final_lse_token,
                merged_lse,
                mask=row_ok,
            )

            inv_total = 1.0 / total_se
            self_factor = self_se * inv_total

            for d_start in range(0, head_dim, BLOCK_D):
                cols = d_start + offs_d
                mask_d = cols < head_dim
                self_vals = tl.load(
                    self_out_ptr + r[:, None] * stride_self_row + cols[None, :] * stride_self_d,
                    mask=row_ok[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc = self_vals * self_factor[:, None]

                for k in tl.static_range(NUM_ROUTES):
                    # Only the position needs reloading for the gather; the LSE
                    # weight comes from the shared tile above.
                    pos = tl.load(
                        route_positions_ptr + k * stride_route_rank + r * stride_route_row,
                        mask=row_ok,
                        other=-1,
                    )
                    mask_k = pos >= 0
                    safe_pos = tl.maximum(pos, 0)
                    weight_k = tl.sum(tl.where(k_offs[:, None] == k, w2, 0.0), axis=0)
                    factor_k = weight_k * inv_total
                    # ``safe_pos * head_dim`` exceeds int32 once a pass has
                    # >= 2**31 routed elements (262144 training tokens); the
                    # int64 cast keeps the gather address exact.
                    vals = tl.load(
                        moba_out_ptr
                        + safe_pos.to(tl.int64)[:, None] * stride_moba_row
                        + cols[None, :] * stride_moba_d,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    acc += vals * factor_k[:, None]

                tl.store(
                    out_ptr + r[:, None] * stride_out_row + cols[None, :] * stride_out_d,
                    acc,
                    mask=row_ok[:, None] & mask_d[None, :],
                )
        else:
            # Shipped per-route path (more than 8 routes).
            # Pass 1: running max over the self term and every live route.
            max_lse = self_lse
            for k in tl.static_range(NUM_ROUTES):
                pos = tl.load(
                    route_positions_ptr + k * stride_route_rank + r * stride_route_row,
                    mask=row_ok,
                    other=-1,
                )
                safe_pos = tl.maximum(pos, 0)
                lse_k = tl.load(
                    moba_lse_ptr + safe_pos,
                    mask=pos >= 0,
                    other=-float("inf"),
                ).to(tl.float32)
                max_lse = tl.maximum(max_lse, lse_k)

            # Pass 2: normalizer.  Padding lanes load -inf and are forced to 0.
            self_se = tl.exp(self_lse - max_lse)
            total_se = self_se
            for k in tl.static_range(NUM_ROUTES):
                pos = tl.load(
                    route_positions_ptr + k * stride_route_rank + r * stride_route_row,
                    mask=row_ok,
                    other=-1,
                )
                safe_pos = tl.maximum(pos, 0)
                lse_k = tl.load(
                    moba_lse_ptr + safe_pos,
                    mask=pos >= 0,
                    other=-float("inf"),
                ).to(tl.float32)
                total_se += tl.where(pos >= 0, tl.exp(lse_k - max_lse), 0.0)

            merged_lse = tl.log(total_se) + max_lse
            tl.store(
                final_lse_ptr + head * stride_final_lse_head + token * stride_final_lse_token,
                merged_lse,
                mask=row_ok,
            )

            inv_total = 1.0 / total_se
            self_factor = self_se * inv_total

            for d_start in range(0, head_dim, BLOCK_D):
                cols = d_start + offs_d
                mask_d = cols < head_dim
                self_vals = tl.load(
                    self_out_ptr + r[:, None] * stride_self_row + cols[None, :] * stride_self_d,
                    mask=row_ok[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc = self_vals * self_factor[:, None]

                for k in tl.static_range(NUM_ROUTES):
                    pos = tl.load(
                        route_positions_ptr + k * stride_route_rank + r * stride_route_row,
                        mask=row_ok,
                        other=-1,
                    )
                    safe_pos = tl.maximum(pos, 0)
                    mask_k = pos >= 0
                    lse_k = tl.load(
                        moba_lse_ptr + safe_pos,
                        mask=mask_k,
                        other=-float("inf"),
                    ).to(tl.float32)
                    factor_k = tl.where(mask_k, tl.exp(lse_k - max_lse), 0.0) * inv_total
                    # ``safe_pos * head_dim`` exceeds int32 once a pass has
                    # >= 2**31 routed elements (262144 training tokens); the
                    # int64 cast keeps the gather address exact.
                    vals = tl.load(
                        moba_out_ptr
                        + safe_pos.to(tl.int64)[:, None] * stride_moba_row
                        + cols[None, :] * stride_moba_d,
                        mask=mask_k[:, None] & mask_d[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    acc += vals * factor_k[:, None]

                tl.store(
                    out_ptr + r[:, None] * stride_out_row + cols[None, :] * stride_out_d,
                    acc,
                    mask=row_ok[:, None] & mask_d[None, :],
                )
