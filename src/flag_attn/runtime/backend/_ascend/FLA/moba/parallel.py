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

"""Ascend orchestration for Triton MoBA (forward, inference, backward)."""

import os
import weakref

import torch

from flag_attn.runtime.backend._ascend.FLA.moba.attention import (
    _routed_attention_launch_config,
    _fused_merge_softmax_triton,
    _fused_merge_softmax_triton_inplace,
    _routed_sparse_attention_triton,
    _triton_varlen_backward,
    _triton_varlen_forward,
    _triton_varlen_func,
)
from flag_attn.runtime.backend._ascend.FLA.moba.routing import (
    _build_dense_chunk_metadata_triton,
    _build_sparse_routes_device_triton,
    _build_sparse_routes_triton,
    _compute_chunk_means_triton,
    _compute_dense_chunk_means_triton,
    _compute_selected_chunks_batched_streaming_triton,
    _compute_selected_chunks_streaming_triton,
    _expand_kv_heads_triton,
    _gather_moba_backward_inputs_triton,
    _gather_moba_kv_triton,
    _linear_offsets_triton,
    _target_chunk_bounds_triton,
    _validated_chunk_metadata,
    cdiv,
    gather_moba_q,
)

# Soft budget for routed-attention output payloads processed in one pass. The
# 910B has far more HBM than the smallest device this limit was chosen for, so
# the budget follows the device capacity; the CUDA-tuned 512 MiB floor made
# every long sequence use one pass per rank. The *total* capacity is used rather
# than the currently free bytes so that the pass plan (and therefore the latency
# of a repeated call) does not depend on allocator caching history.
_SPARSE_ROUTE_OUT_BUDGET_BYTES = 512 * 1024 * 1024


def _route_out_budget_bytes() -> int:
    override = os.environ.get("FLAG_ATTN_MOBA_ROUTE_BUDGET_BYTES")
    if override:
        return int(override)
    floor = _SPARSE_ROUTE_OUT_BUDGET_BYTES
    try:
        _, total_bytes = torch.npu.mem_get_info()
    except (RuntimeError, AttributeError):
        # No usable NPU memory query (CPU build or a driver error): keep the
        # conservative floor instead of failing the whole call.
        return floor
    return max(floor, int(total_bytes * 0.5))


def _uncached_sequence_lengths(cu_seqlens: torch.Tensor) -> tuple:
    bounds = cu_seqlens.tolist()
    return tuple(end - start for start, end in zip(bounds, bounds[1:]))


# Host-copy cache for the packed sequence lengths.  A tensor cannot be a dict
# key safely (``Tensor.__eq__`` returns a tensor), so the key is the storage
# address plus the version counter and the entry stores a weak reference: a
# different tensor that happens to reuse the address fails the identity check,
# and the cache never keeps a caller tensor alive.
_SEQUENCE_LENGTH_CACHE: dict = {}
_SEQUENCE_LENGTH_CACHE_SIZE = 8


def _cached_sequence_lengths(cu_seqlens: torch.Tensor, tensor_version: int) -> tuple:
    """Host copy of the packed sequence lengths, cached per tensor version.

    The inference path needs the sequence lengths twice on the host: to validate
    the caller's ``max_seqlen`` and to plan the routed rank passes.  Reading them
    once per distinct tensor version keeps both checks exact without a
    device-to-host read on every call.
    """
    key = (cu_seqlens.data_ptr(), cu_seqlens.numel(), tensor_version)
    entry = _SEQUENCE_LENGTH_CACHE.get(key)
    if entry is not None and entry[0]() is cu_seqlens:
        return entry[1]
    lengths = _uncached_sequence_lengths(cu_seqlens)
    if len(_SEQUENCE_LENGTH_CACHE) >= _SEQUENCE_LENGTH_CACHE_SIZE:
        _SEQUENCE_LENGTH_CACHE.clear()
    _SEQUENCE_LENGTH_CACHE[key] = (weakref.ref(cu_seqlens), lengths)
    return lengths


def _host_sequence_lengths(cu_seqlens: torch.Tensor) -> tuple:
    try:
        tensor_version = cu_seqlens._version
    except RuntimeError:
        # Inference-mode tensors do not track a version counter, so caching
        # could hide an in-place update; read them on every call instead.
        return _uncached_sequence_lengths(cu_seqlens)
    return _cached_sequence_lengths(cu_seqlens, tensor_version)


class MixedAttention(torch.autograd.Function):
    """Mixed chunk-local and routed sparse attention for Ascend."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        self_attn_max_seqlen,
        moba_max_seqlen_q,
        moba_chunk_size,
        moba_q_sh_indices,
        route_positions,
    ):
        ctx.self_attn_max_seqlen = self_attn_max_seqlen
        ctx.moba_max_seqlen_q = moba_max_seqlen_q
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        self_attn_out_sh, self_attn_lse_hs = _triton_varlen_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=self_attn_max_seqlen,
            max_seqlen_k=self_attn_max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
        )

        moba_attn_out, moba_attn_lse_hs = _triton_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=moba_max_seqlen_q,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
        )

        output = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)
        output_2d = output.view(-1, q.shape[2])
        self_attn_out_2d = self_attn_out_sh.view(-1, q.shape[2])
        moba_out_flat = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        moba_lse_flat = moba_attn_lse_hs.view(-1)
        mixed_attn_lse_hs = torch.empty_like(self_attn_lse_hs)

        _fused_merge_softmax_triton(
            self_attn_out_2d,
            self_attn_lse_hs,
            moba_out_flat,
            moba_lse_flat,
            route_positions,
            output_2d,
            mixed_attn_lse_hs,
        )

        ctx.save_for_backward(
            output,
            mixed_attn_lse_hs,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):
        self_attn_max_seqlen = ctx.self_attn_max_seqlen
        moba_max_seqlen_q = ctx.moba_max_seqlen_q
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale
        (
            output,
            mixed_attn_lse_hs,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        _triton_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_lse_hs,
            dq=dq,
            dk=dk,
            dv=dv,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=self_attn_max_seqlen,
            max_seqlen_k=self_attn_max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
        )

        num_selected = moba_q_sh_indices.numel()
        gathered_d_moba = torch.empty(
            (num_selected, q.shape[-1]),
            device=d_output.device,
            dtype=d_output.dtype,
        )
        gathered_moba_out = torch.empty(
            (num_selected, q.shape[-1]),
            device=output.device,
            dtype=output.dtype,
        )
        gathered_lse = torch.empty(
            (num_selected,),
            device=mixed_attn_lse_hs.device,
            dtype=mixed_attn_lse_hs.dtype,
        )

        _gather_moba_backward_inputs_triton(
            d_output,
            output,
            mixed_attn_lse_hs,
            moba_q_sh_indices,
            gathered_d_moba,
            gathered_moba_out,
            gathered_lse,
        )

        d_moba_output = gathered_d_moba.unsqueeze(1)
        moba_output = gathered_moba_out.unsqueeze(1)
        mixed_attn_vlse = gathered_lse.unsqueeze(0)

        dmq = torch.empty_like(moba_q)
        dmkv = torch.empty_like(moba_kv)
        dmk = dmkv[:, 0]
        dmv = dmkv[:, 1]

        _triton_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=dmq,
            dk=dmk,
            dv=dmv,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=moba_max_seqlen_q,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
        )

        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None, None, None


def _plan_rank_passes(
    selected_chunks,
    sequence_lengths,
    chunk_size: int,
    head_dim: int,
    element_size: int,
):
    """Group historical ranks into as few route passes as the output budget allows.

    The previous policy sized every pass by the *worst case* row count
    ``tokens * heads``, which forced one pass per rank on long sequences. The
    actual routed row count of a rank is smaller (a rank only has routes for
    tokens past its chunk). Grouping by that count reduces merge/reload work
    while targeting the same byte budget for live output and FP32 LSE rows.

    The row count is derived analytically from the host copy of ``cu_seqlens``
    (``sequence_lengths``) instead of reading ``(selected_chunks >= 0).sum()``
    back from the device: rank ``r`` is filled for a token exactly when the
    token is at least ``(r + 1) * chunk_size`` tokens into its sequence,
    whatever the top-k scores are, so::

        rows(r) = heads * sum_s max(0, length_s - (r + 1) * chunk_size)

    This is exact, needs no device readback, and preserves the previous pass
    grouping (including the one-pass-per-rank case at 262144 tokens).

    The routed forward store still multiplies an int32 route position by the
    output row stride. Keep the conservative element cap for that store. The
    merge widens routed positions to int64 before multiplying by the stride,
    which fixes the historical overflow in its gather.

    Route slots and flattened query indices have separate int32 checks in
    compaction. The routed launcher slices descriptors to bound per-launch
    workspace independently of this pass budget.
    """
    num_ranks, num_heads, _ = selected_chunks.shape
    if num_ranks <= 0:
        return []
    bytes_per_row = head_dim * element_size + 4  # output row + fp32 lse
    budget_rows = max(1, _route_out_budget_bytes() // max(bytes_per_row, 1))
    element_cap_rows = (2**31 - 1) // max(head_dim, 1)
    budget_rows = min(budget_rows, element_cap_rows)
    counts = [
        num_heads
        * sum(max(0, length - (rank + 1) * chunk_size) for length in sequence_lengths)
        for rank in range(num_ranks)
    ]
    if counts and max(counts) > element_cap_rows:
        raise ValueError(
            "a single routing rank has "
            f"{max(counts)} rows, which exceeds the int32 router output index "
            f"bound ({element_cap_rows} rows at head_dim={head_dim}); "
            "reduce the sequence length or the head count"
        )
    passes = []
    start = 0
    rows = 0
    for rank, count in enumerate(counts):
        if rank > start and rows + count > budget_rows:
            passes.append((start, rank))
            start = rank
            rows = 0
        rows += count
    passes.append((start, num_ranks))
    return passes


def _moba_attn_varlen_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """Device-side forward path with native GQA and index-only sparse routes."""
    seqlen, num_q_heads, head_dim = q.shape
    # max_seqlen bounds dense chunk capacity and selector token coverage.
    # An understated value can leave the sequence tail unwritten. Reuse the
    # host lengths from the rank planner and cache them per tensor version.
    sequence_lengths = _host_sequence_lengths(cu_seqlens)
    actual_max_seqlen = max(sequence_lengths)
    if max_seqlen < actual_max_seqlen:
        raise ValueError(
            f"max_seqlen ({max_seqlen}) is smaller than the actual maximum sequence length ({actual_max_seqlen})"
        )
    (
        cu_chunk,
        chunk_end,
        _,
        valid_historical,
        max_chunks_per_sequence,
    ) = _build_dense_chunk_metadata_triton(
        cu_seqlens,
        max_seqlen,
        moba_chunk_size,
    )

    softmax_scale = head_dim**-0.5
    local_max_seqlen = min(max_seqlen, moba_chunk_size)
    local_out, local_lse = _triton_varlen_forward(
        q,
        k,
        v,
        cu_chunk,
        cu_chunk,
        local_max_seqlen,
        local_max_seqlen,
        softmax_scale,
        causal=True,
    )

    historical_topk = min(moba_topk - 1, max_chunks_per_sequence - 1)
    if historical_topk <= 0:
        return local_out

    chunk_means = _compute_dense_chunk_means_triton(
        k,
        cu_chunk,
        valid_historical,
        moba_chunk_size,
    )
    selected_chunks = _compute_selected_chunks_batched_streaming_triton(
        q,
        chunk_means,
        chunk_end,
        cu_seqlens,
        max_seqlen,
        max_chunks_per_sequence,
        historical_topk,
        moba_chunk_size,
    )
    block_m = _routed_attention_launch_config(head_dim, q.device)[0]
    rank_passes = _plan_rank_passes(
        selected_chunks,
        sequence_lengths,
        chunk_size=moba_chunk_size,
        head_dim=head_dim,
        element_size=q.element_size(),
    )
    current_out = local_out
    current_lse = local_lse
    if len(rank_passes) == 1:
        # A single pass can reuse the local output and LSE buffers. Each
        # program owns its rows and the routed inputs are disjoint.
        next_out = current_out
        next_lse = current_lse
    else:
        next_out = torch.empty_like(q)
        next_lse = torch.empty_like(local_lse)
    for rank_start, rank_end in rank_passes:
        rank_chunks = selected_chunks[rank_start:rank_end]
        (
            query_indices,
            route_positions,
            expert_offsets,
            descriptor_experts,
            descriptor_starts,
        ) = _build_sparse_routes_device_triton(
            rank_chunks,
            valid_historical.numel(),
            num_q_heads,
            seqlen,
            block_m,
        )
        route_out, route_lse = _routed_sparse_attention_triton(
            q,
            k,
            v,
            query_indices,
            expert_offsets,
            descriptor_experts,
            descriptor_starts,
            cu_seqlens,
            max_chunks_per_sequence,
            moba_chunk_size,
        )
        _fused_merge_softmax_triton_inplace(
            current_out.view(-1, head_dim),
            current_lse,
            route_out,
            route_lse,
            route_positions,
            next_out.view(-1, head_dim),
            next_lse,
        )
        # Release routed buffers before allocating the next pass's outputs.
        del route_out, route_lse, route_positions
        del query_indices, expert_offsets, descriptor_experts, descriptor_starts
        current_out, next_out = next_out, current_out
        current_lse, next_lse = next_lse, current_lse
    return current_out


def parallel_moba(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """Ascend implementation of causal variable-length MoBA."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes, got {k.shape} and {v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k/v must have the same token count and head dimension, got q={q.shape}, k={k.shape}")
    if q.device != k.device or q.device != v.device or q.device.type != "npu":
        raise ValueError("q, k, and v must be NPU tensors on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"only float16 and bfloat16 are supported, got {q.dtype}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional tensor of length >= 2")
    if cu_seqlens.device != q.device or cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be an int32 NPU tensor on the q device")
    if not isinstance(max_seqlen, int) or max_seqlen <= 0:
        raise ValueError(f"max_seqlen must be a positive int, got {max_seqlen}")
    if not isinstance(moba_chunk_size, int) or moba_chunk_size <= 0:
        raise ValueError(f"moba_chunk_size must be a positive int, got {moba_chunk_size}")
    if not isinstance(moba_topk, int) or moba_topk < 1:
        raise ValueError(f"moba_topk must be a positive int, got {moba_topk}")
    if moba_topk > 64:
        raise ValueError(f"moba_topk must be <= 64 to match FlashMoBA, got {moba_topk}")

    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if q.shape[0] == 0 or num_q_heads == 0 or num_kv_heads == 0 or q.shape[2] == 0:
        raise ValueError("q, k, and v dimensions must be non-zero")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_q_heads}) must be divisible by kv heads ({num_kv_heads})")
    head_dim = q.shape[2]
    if head_dim > 256:
        raise ValueError(f"head_dim must be <= 256 to match FlashMoBA, got {head_dim}")

    # Strided inputs make the BiSheng compiler allocate a far larger unified
    # buffer for the chunk-mean kernels, so normalize once up front. The copy is
    # differentiable and gradients still reach the caller's original tensors.
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    if not torch.is_grad_enabled():
        return _moba_attn_varlen_inference(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            moba_chunk_size,
            moba_topk,
        )

    if num_q_heads != num_kv_heads:
        k, v = _expand_kv_heads_triton(k, v, num_q_heads)

    seqlen = q.shape[0]

    actual_max_seqlen, chunk_metadata = _validated_chunk_metadata(
        cu_seqlens,
        cu_seqlens._version,
        q.shape[0],
        moba_chunk_size,
    )
    if max_seqlen < actual_max_seqlen:
        raise ValueError(
            f"max_seqlen ({max_seqlen}) is smaller than the actual maximum sequence length ({actual_max_seqlen})"
        )
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = chunk_metadata

    max_historical_chunks = max(cdiv(actual_max_seqlen, moba_chunk_size) - 1, 0)
    moba_topk = min(
        moba_topk - 1,
        num_filtered_chunk,
        max_historical_chunks,
    )
    need_moba_attn = moba_topk > 0

    self_attn_cu_seqlen = cu_chunk
    self_attn_max_seqlen = min(max_seqlen, moba_chunk_size)

    if not need_moba_attn:
        return _triton_varlen_func(
            q,
            k,
            v,
            self_attn_cu_seqlen,
            self_attn_cu_seqlen,
            self_attn_max_seqlen,
            self_attn_max_seqlen,
            causal=True,
        )

    chunk_means = _compute_chunk_means_triton(
        k,
        cu_chunk,
        filtered_chunk_indices,
        moba_chunk_size,
    )
    chunk_end, batch_end = _target_chunk_bounds_triton(
        cu_chunk,
        filtered_chunk_indices,
        chunk_to_batch,
        cu_seqlens,
    )

    selected_chunks = _compute_selected_chunks_streaming_triton(
        q,
        chunk_means,
        chunk_end,
        batch_end,
        moba_topk,
    )

    (
        moba_q_sh_indices,
        route_positions,
        _,
        expert_offsets,
        moba_max_seqlen_q,
    ) = _build_sparse_routes_triton(
        selected_chunks,
        num_filtered_chunk,
        num_q_heads,
        seqlen,
    )

    if moba_q_sh_indices.numel() == 0:
        return _triton_varlen_func(
            q,
            k,
            v,
            self_attn_cu_seqlen,
            self_attn_cu_seqlen,
            self_attn_max_seqlen,
            self_attn_max_seqlen,
            causal=True,
        )

    moba_q = gather_moba_q(q, moba_q_sh_indices)

    moba_cu_seqlen_q = expert_offsets
    num_experts = num_filtered_chunk * num_q_heads
    moba_kv = _gather_moba_kv_triton(
        k,
        v,
        cu_chunk,
        filtered_chunk_indices,
        moba_chunk_size,
    )
    moba_cu_seqlen_kv = _linear_offsets_triton(
        num_experts + 1,
        moba_chunk_size,
        q.device,
    )

    return MixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        self_attn_max_seqlen,
        moba_max_seqlen_q,
        moba_chunk_size,
        moba_q_sh_indices,
        route_positions,
    )
