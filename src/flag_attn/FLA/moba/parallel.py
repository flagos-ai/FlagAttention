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

"""Training and inference orchestration for Triton MoBA."""

import torch

from flag_attn.FLA.moba.attention import (
    _attention_launch_config,
    _fused_merge_softmax_triton,
    _routed_sparse_attention_triton,
    _triton_varlen_backward,
    _triton_varlen_forward,
    _triton_varlen_func,
)
from flag_attn.FLA.moba.routing import (
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


# Soft budget for routed-attention output payloads processed in one pass. It
# is neither a strict peak-memory limit nor an accounting of all workspace.
_SPARSE_ROUTE_OUT_BUDGET_BYTES = 512 * 1024 * 1024


class MixedAttention(torch.autograd.Function):
    """
    Custom Autograd Function handling the mixed attention mechanism.
    Integrates Self-Attention and MoBA-Attention using Triton-optimized kernels.
    """

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

        # 1. Chunk-local causal self attention (Triton)
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

        # 2. Selected sparse chunk attention (Triton)
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

        # 3. Output merging.
        output = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)

        output_2d = output.view(-1, q.shape[2])
        self_attn_out_2d = self_attn_out_sh.view(-1, q.shape[2])
        moba_out_flat = moba_attn_out.view(-1, moba_attn_out.shape[-1])
        moba_lse_flat = moba_attn_lse_hs.view(-1)
        mixed_attn_lse_hs = torch.empty_like(self_attn_lse_hs)

        # Fused Kernel call
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

        # 1. Chunk-local self attention backward (Triton)
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

        # 2. Gather inputs for MoBA Backward
        # Optimization: Triton kernel to sparsely gather gradients/outputs needed for MoBA backward
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

        # 3. Sparse MoBA attention backward (Triton)
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

        # Return gradients in order.
        # Note: 'dmq' (sparse) will be scattered back to 'q.grad' by gather_moba_q's backward.
        return dq, dk, dv, None, dmq, dmkv, None, None, None, None, None, None, None


def _moba_attn_varlen_inference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """Device-side forward path with native GQA and index-only sparse routes.

    The training path below intentionally retains the materialized autograd
    implementation until direct routed backward kernels are available.  In
    normal inference (`torch.no_grad` / `torch.inference_mode`) this path never
    expands K/V, never copies routed Q/K/V payloads, and never reads a dynamic
    route count back to the host.

    Caller contract: ``cu_seqlens`` must start at zero, be strictly increasing,
    and end at ``q.shape[0]``; ``max_seqlen`` must be at least the actual
    maximum sequence length. These value-dependent conditions are deliberately
    not checked here because reading CUDA metadata on the host would synchronize
    the inference hot path and break CUDA Graph capture.
    """
    seqlen, num_q_heads, head_dim = q.shape
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
    )
    block_m = _attention_launch_config(head_dim, q.device)[0]
    bytes_per_rank = seqlen * num_q_heads * head_dim * q.element_size()
    ranks_per_pass = max(
        1,
        min(
            historical_topk,
            _SPARSE_ROUTE_OUT_BUDGET_BYTES // max(bytes_per_rank, 1),
        ),
    )
    current_out = local_out
    current_lse = local_lse
    next_out = torch.empty_like(q)
    next_lse = torch.empty_like(local_lse)
    for rank_start in range(0, historical_topk, ranks_per_pass):
        rank_end = min(rank_start + ranks_per_pass, historical_topk)
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
        _fused_merge_softmax_triton(
            current_out.view(-1, head_dim),
            current_lse,
            route_out,
            route_lse,
            route_positions,
            next_out.view(-1, head_dim),
            next_lse,
        )
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
    """
    Triton implementation of causal variable-length MoBA.

    ``moba_topk`` counts the current causal chunk, matching FlashMoBA. The
    implementation supports packed non-empty sequences, MHA/GQA/MQA, head
    dimensions up to 256, and top-k values up to 64.

    Inference callers must provide a valid, strictly increasing ``cu_seqlens``
    starting at zero and ending at the packed token count, and ``max_seqlen``
    must be no smaller than the actual maximum sequence length. The inference
    path intentionally does not synchronize CUDA metadata back to the host to
    validate these value-dependent conditions. The autograd path retains its
    existing host-side validation.

    Args:
        q (torch.Tensor): [seqlen, num_q_heads, head_dim]
        k (torch.Tensor): [seqlen, num_kv_heads, head_dim]
        v (torch.Tensor): [seqlen, num_kv_heads, head_dim]
        cu_seqlens (torch.Tensor): Cumulative sequence length (FlashAttention format)
        max_seqlen (int): Max sequence length in batch
        moba_chunk_size (int): Size of chunks
        moba_topk (int): Number of chunks to attend to
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes, got {k.shape} and {v.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k/v must have the same token count and head dimension, got q={q.shape}, k={k.shape}")
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"only float16 and bfloat16 are supported, got {q.dtype}")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional tensor of length >= 2")
    if cu_seqlens.device != q.device or cu_seqlens.dtype != torch.int32:
        raise TypeError("cu_seqlens must be an int32 CUDA tensor on the q device")
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

    # Inference is the deployment target and uses a separate fully device-side
    # dataflow.  Keeping the existing autograd path below avoids silently
    # weakening backward correctness while direct routed backward is pending.
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

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
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

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    max_historical_chunks = max(cdiv(actual_max_seqlen, moba_chunk_size) - 1, 0)
    moba_topk = min(
        moba_topk - 1,
        num_filtered_chunk,
        max_historical_chunks,
    )
    need_moba_attn = moba_topk > 0

    self_attn_cu_seqlen = cu_chunk
    self_attn_max_seqlen = min(max_seqlen, moba_chunk_size)

    # K=1 means current-chunk causal attention, not full-sequence attention.
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

    # Optimization: Use compact Top-K chunk indices instead of a dense gate mask.
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
        num_head,
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
    num_experts = num_filtered_chunk * num_head
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
