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

"""Routing, metadata, and sparse data-movement kernels for Triton MoBA."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

_PREFIX_SCAN_BLOCK = 1024


def cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def _fill_kernel(output_ptr, numel, value: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output_ptr + offsets, value, mask=offsets < numel)


def _fill_triton(tensor: torch.Tensor, value) -> None:
    if tensor.numel() == 0:
        return
    block = 256
    _fill_kernel[(cdiv(tensor.numel(), block),)](
        tensor,
        tensor.numel(),
        value=value,
        BLOCK=block,
    )


@triton.jit
def _cast_kernel(input_ptr, output_ptr, numel, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values, mask=mask)


def _cast_triton(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    output = torch.empty(tensor.shape, device=tensor.device, dtype=dtype)
    if tensor.numel() == 0:
        return output
    block = 256
    _cast_kernel[(cdiv(tensor.numel(), block),)](
        tensor,
        output,
        tensor.numel(),
        BLOCK=block,
    )
    return output


def _copy_triton(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    if input_tensor.numel() == 0:
        return
    block = 256
    _cast_kernel[(cdiv(input_tensor.numel(), block),)](
        input_tensor,
        output_tensor,
        input_tensor.numel(),
        BLOCK=block,
    )


@triton.jit
def _expand_kv_heads_kernel(
    k_ptr,
    v_ptr,
    expanded_k_ptr,
    expanded_v_ptr,
    total_tokens,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_ek_t,
    stride_ek_h,
    stride_ek_d,
    stride_ev_t,
    stride_ev_h,
    stride_ev_d,
    NUM_Q_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    q_head = tl.program_id(1)
    if token >= total_tokens or q_head >= NUM_Q_HEADS:
        return
    kv_head = q_head // GROUP_SIZE
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < HEAD_DIM
    k_values = tl.load(
        k_ptr + token * stride_k_t + kv_head * stride_k_h + offsets_d * stride_k_d,
        mask=mask_d,
        other=0.0,
    )
    v_values = tl.load(
        v_ptr + token * stride_v_t + kv_head * stride_v_h + offsets_d * stride_v_d,
        mask=mask_d,
        other=0.0,
    )
    tl.store(
        expanded_k_ptr + token * stride_ek_t + q_head * stride_ek_h + offsets_d * stride_ek_d,
        k_values,
        mask=mask_d,
    )
    tl.store(
        expanded_v_ptr + token * stride_ev_t + q_head * stride_ev_h + offsets_d * stride_ev_d,
        v_values,
        mask=mask_d,
    )


@triton.jit
def _reduce_expanded_kv_grads_kernel(
    d_expanded_k_ptr,
    d_expanded_v_ptr,
    dk_ptr,
    dv_ptr,
    total_tokens,
    stride_dek_t,
    stride_dek_h,
    stride_dek_d,
    stride_dev_t,
    stride_dev_h,
    stride_dev_d,
    stride_dk_t,
    stride_dk_h,
    stride_dk_d,
    stride_dv_t,
    stride_dv_h,
    stride_dv_d,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    if token >= total_tokens or kv_head >= NUM_KV_HEADS:
        return
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < HEAD_DIM
    dk_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for group_index in range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + group_index
        dk_acc += tl.load(
            d_expanded_k_ptr + token * stride_dek_t + q_head * stride_dek_h + offsets_d * stride_dek_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        dv_acc += tl.load(
            d_expanded_v_ptr + token * stride_dev_t + q_head * stride_dev_h + offsets_d * stride_dev_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
    tl.store(
        dk_ptr + token * stride_dk_t + kv_head * stride_dk_h + offsets_d * stride_dk_d,
        dk_acc,
        mask=mask_d,
    )
    tl.store(
        dv_ptr + token * stride_dv_t + kv_head * stride_dv_h + offsets_d * stride_dv_d,
        dv_acc,
        mask=mask_d,
    )


class _ExpandKVHeads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k: torch.Tensor, v: torch.Tensor, num_q_heads: int):
        total_tokens, num_kv_heads, head_dim = k.shape
        group_size = num_q_heads // num_kv_heads
        expanded_k = torch.empty((total_tokens, num_q_heads, head_dim), device=k.device, dtype=k.dtype)
        expanded_v = torch.empty_like(expanded_k)
        block_d = max(16, triton.next_power_of_2(head_dim))
        _expand_kv_heads_kernel[(total_tokens, num_q_heads)](
            k,
            v,
            expanded_k,
            expanded_v,
            total_tokens,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            expanded_k.stride(0),
            expanded_k.stride(1),
            expanded_k.stride(2),
            expanded_v.stride(0),
            expanded_v.stride(1),
            expanded_v.stride(2),
            NUM_Q_HEADS=num_q_heads,
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
        ctx.input_shape = k.shape
        ctx.num_q_heads = num_q_heads
        return expanded_k, expanded_v

    @staticmethod
    def backward(ctx, d_expanded_k, d_expanded_v):
        total_tokens, num_kv_heads, head_dim = ctx.input_shape
        dk = torch.empty(
            ctx.input_shape,
            device=d_expanded_k.device,
            dtype=d_expanded_k.dtype,
        )
        dv = torch.empty_like(dk)
        block_d = max(16, triton.next_power_of_2(head_dim))
        _reduce_expanded_kv_grads_kernel[(total_tokens, num_kv_heads)](
            d_expanded_k,
            d_expanded_v,
            dk,
            dv,
            total_tokens,
            d_expanded_k.stride(0),
            d_expanded_k.stride(1),
            d_expanded_k.stride(2),
            d_expanded_v.stride(0),
            d_expanded_v.stride(1),
            d_expanded_v.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            NUM_KV_HEADS=num_kv_heads,
            GROUP_SIZE=ctx.num_q_heads // num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
        return dk, dv, None


def _expand_kv_heads_triton(k, v, num_q_heads):
    if k.shape[1] == num_q_heads:
        return k, v
    return _ExpandKVHeads.apply(k, v, num_q_heads)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_ROWS": 4, "BYPASS_L1": False},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 4, "BYPASS_L1": True},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 8, "BYPASS_L1": False},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 8, "BYPASS_L1": True},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 16, "BYPASS_L1": False},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_ROWS": 16, "BYPASS_L1": True},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=["head_dim"],
)
@triton.jit
def _gather_moba_q_kernel(
    q_ptr,
    indices_ptr,
    output_ptr,
    num_rows,
    stride_indices: tl.constexpr,
    stride_q_token: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BYPASS_L1: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim_offsets = tl.arange(0, BLOCK_DIM)
    row_mask = row_offsets < num_rows
    source_rows = tl.load(
        indices_ptr + row_offsets * stride_indices,
        mask=row_mask,
        other=0,
    ).to(tl.int64)
    source_tokens = source_rows // num_heads
    source_heads = source_rows - source_tokens * num_heads

    q_offsets = (
        source_tokens[:, None] * stride_q_token
        + source_heads[:, None] * stride_q_head
        + dim_offsets[None, :] * stride_q_dim
    )
    mask = row_mask[:, None] & (dim_offsets[None, :] < head_dim)
    if BYPASS_L1:
        values = tl.load(
            q_ptr + q_offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        )
    else:
        values = tl.load(q_ptr + q_offsets, mask=mask, other=0.0)

    output_offsets = row_offsets[:, None] * stride_output_row + dim_offsets[None, :] * stride_output_dim
    tl.store(output_ptr + output_offsets, values, mask=mask)


def _gather_moba_q_forward(q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    head_dim = q.shape[-1]
    output = torch.empty(
        (indices.numel(), 1, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    if indices.numel() == 0:
        return output

    block_dim = triton.next_power_of_2(head_dim)

    def grid(meta):
        return (triton.cdiv(indices.numel(), meta["BLOCK_ROWS"]),)

    _gather_moba_q_kernel[grid](
        q,
        indices,
        output,
        indices.numel(),
        indices.stride(0),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        output.stride(0),
        output.stride(2),
        head_dim=head_dim,
        num_heads=q.shape[1],
        BLOCK_DIM=block_dim,
    )
    return output


@triton.jit
def _gather_moba_q_backward_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_q_ptr,
    num_rows,
    stride_indices: tl.constexpr,
    head_dim,
    num_heads: tl.constexpr,
    stride_go_row: tl.constexpr,
    stride_go_dim: tl.constexpr,
    stride_gq_token: tl.constexpr,
    stride_gq_head: tl.constexpr,
    stride_gq_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    source_row = tl.load(indices_ptr + row * stride_indices).to(tl.int64)
    source_token = source_row // num_heads
    source_head = source_row - source_token * num_heads
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim
    values = tl.load(
        grad_output_ptr + row * stride_go_row + offsets_d * stride_go_dim,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    tl.atomic_add(
        grad_q_ptr + source_token * stride_gq_token + source_head * stride_gq_head + offsets_d * stride_gq_dim,
        values,
        mask=mask_d,
    )


class _GatherMobaQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.q_shape = q.shape
        ctx.head_dim = q.shape[-1]
        return _gather_moba_q_forward(q, indices)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        grad_q = torch.empty(
            ctx.q_shape,
            device=grad_output.device,
            dtype=torch.float32,
        )
        _fill_triton(grad_q, 0.0)
        block_d = max(16, triton.next_power_of_2(ctx.head_dim))
        _gather_moba_q_backward_kernel[(indices.numel(),)](
            grad_output,
            indices,
            grad_q,
            indices.numel(),
            indices.stride(0),
            ctx.head_dim,
            num_heads=ctx.q_shape[1],
            stride_go_row=grad_output.stride(0),
            stride_go_dim=grad_output.stride(2),
            stride_gq_token=grad_q.stride(0),
            stride_gq_head=grad_q.stride(1),
            stride_gq_dim=grad_q.stride(2),
            BLOCK_D=block_d,
        )
        return _cast_triton(grad_q, grad_output.dtype), None


def gather_moba_q(q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather flattened query rows into the [selected, 1, head_dim] MoBA layout."""
    if not q.is_cuda or not indices.is_cuda:
        raise ValueError("gather_moba_q requires CUDA tensors")
    if q.ndim != 3:
        raise ValueError(f"q must have shape [seqlen, heads, head_dim], got {q.shape}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got {indices.shape}")
    if q.device != indices.device:
        raise ValueError("q and indices must be on the same CUDA device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must use int32 or int64, got {indices.dtype}")

    if torch.is_grad_enabled() and q.requires_grad:
        return _GatherMobaQ.apply(q, indices)
    return _gather_moba_q_forward(q, indices)


def _chunk_metadata_from_host(
    cu_seqlens_host: torch.Tensor,
    device: torch.device,
    chunk_size: int,
):
    """Build small immutable chunk tables on the host and upload them once."""
    boundaries = cu_seqlens_host.tolist()
    chunk_boundaries = [0]
    target_chunks = []
    chunk_to_batch = []

    for batch, (sequence_start, sequence_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        sequence_length = sequence_end - sequence_start
        num_chunks = cdiv(sequence_length, chunk_size)
        first_chunk = len(chunk_to_batch)
        for chunk in range(num_chunks):
            chunk_to_batch.append(batch)
            chunk_boundaries.append(min(sequence_start + (chunk + 1) * chunk_size, sequence_end))
        target_chunks.extend(range(first_chunk, first_chunk + num_chunks - 1))

    cu_chunk = torch.tensor(chunk_boundaries, device=device, dtype=torch.int32)
    filtered_chunk_indices = torch.tensor(target_chunks, device=device, dtype=torch.int32)
    chunk_to_batch_tensor = torch.tensor(chunk_to_batch, device=device, dtype=torch.int32)
    return (
        cu_chunk,
        filtered_chunk_indices,
        len(target_chunks),
        chunk_to_batch_tensor,
    )


def calc_chunks(cu_seqlen, moba_chunk_size):
    """Prepare packed MoBA chunk metadata without CUDA-side PyTorch ops."""
    return _chunk_metadata_from_host(
        cu_seqlen.detach().cpu(),
        cu_seqlen.device,
        moba_chunk_size,
    )


@lru_cache(maxsize=16)
def _validated_chunk_metadata(
    cu_seqlens: torch.Tensor,
    tensor_version: int,
    total_tokens: int,
    chunk_size: int,
):
    """Validate immutable packed metadata once and cache its chunk mapping."""
    del tensor_version
    cu_seqlens_host = cu_seqlens.detach().cpu()
    if cu_seqlens_host[0].item() != 0:
        raise ValueError("cu_seqlens must start at 0")
    if cu_seqlens_host[-1].item() != total_tokens:
        raise ValueError(
            f"cu_seqlens must end at the packed token count, got {cu_seqlens_host[-1].item()} and {total_tokens}"
        )
    sequence_lengths = cu_seqlens_host[1:] - cu_seqlens_host[:-1]
    if torch.any(sequence_lengths <= 0).item():
        raise ValueError("cu_seqlens must describe non-empty increasing sequences")

    actual_max_seqlen = int(sequence_lengths.max().item())
    chunk_metadata = _chunk_metadata_from_host(
        cu_seqlens_host,
        cu_seqlens.device,
        chunk_size,
    )
    return actual_max_seqlen, chunk_metadata


@triton.jit
def _dense_chunk_metadata_kernel(
    cu_seqlens_ptr,
    cu_chunk_ptr,
    chunk_end_ptr,
    valid_historical_ptr,
    num_chunk_slots,
    MAX_CHUNKS_PER_SEQUENCE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build fixed-size packed chunk metadata without a device-to-host read."""
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = slots < num_chunk_slots
    batches = slots // MAX_CHUNKS_PER_SEQUENCE
    local_chunks = slots - batches * MAX_CHUNKS_PER_SEQUENCE
    sequence_start = tl.load(cu_seqlens_ptr + batches, mask=mask, other=0)
    sequence_end = tl.load(cu_seqlens_ptr + batches + 1, mask=mask, other=0)
    chunk_start = tl.minimum(
        sequence_start + local_chunks * CHUNK_SIZE,
        sequence_end,
    )
    next_chunk_start = tl.minimum(chunk_start + CHUNK_SIZE, sequence_end)

    # The final (possibly full) chunk is handled by local causal attention and
    # must not participate in historical routing.
    is_historical = mask & (next_chunk_start < sequence_end)
    tl.store(cu_chunk_ptr + slots, chunk_start, mask=mask)
    tl.store(
        chunk_end_ptr + slots,
        tl.where(is_historical, next_chunk_start, 0),
        mask=mask,
    )
    tl.store(valid_historical_ptr + slots, is_historical.to(tl.int8), mask=mask)


@triton.jit
def _store_last_chunk_boundary_kernel(cu_seqlens_ptr, cu_chunk_ptr, batch_size, num_chunk_slots):
    total_tokens = tl.load(cu_seqlens_ptr + batch_size)
    tl.store(cu_chunk_ptr + num_chunk_slots, total_tokens)


def _build_dense_chunk_metadata_triton(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    chunk_size: int,
):
    """Return graph-safe dense chunk slots for the inference-only path."""
    batch_size = cu_seqlens.numel() - 1
    max_chunks_per_sequence = cdiv(max_seqlen, chunk_size)
    num_chunk_slots = batch_size * max_chunks_per_sequence
    cu_chunk = torch.empty((num_chunk_slots + 1,), device=cu_seqlens.device, dtype=torch.int32)
    chunk_end = torch.empty((num_chunk_slots,), device=cu_seqlens.device, dtype=torch.int32)
    valid_historical = torch.empty((num_chunk_slots,), device=cu_seqlens.device, dtype=torch.int8)
    block = 256
    _dense_chunk_metadata_kernel[(cdiv(num_chunk_slots, block),)](
        cu_seqlens,
        cu_chunk,
        chunk_end,
        valid_historical,
        num_chunk_slots,
        MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
        CHUNK_SIZE=chunk_size,
        BLOCK=block,
    )
    # A one-program launch keeps the terminal boundary on device and ordered
    # on the current stream.
    _store_last_chunk_boundary_kernel[(1,)](
        cu_seqlens,
        cu_chunk,
        batch_size,
        num_chunk_slots,
    )
    return (
        cu_chunk,
        chunk_end,
        None,
        valid_historical,
        max_chunks_per_sequence,
    )


@triton.jit
def _chunk_mean_kernel(
    k_ptr,
    cu_chunk_ptr,
    filtered_chunk_indices_ptr,
    out_ptr,
    stride_token,
    stride_head,
    stride_d,
    out_stride_chunk,
    out_stride_head,
    out_stride_d,
    num_chunks,
    num_heads,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel to compute the mean of Key vectors within each chunk.
    Replacing: filtered_kv.view(...).mean(dim=1)
    """
    chunk_id = tl.program_id(0)
    head_id = tl.program_id(1)
    if chunk_id >= num_chunks or head_id >= num_heads:
        return
    source_chunk = tl.load(filtered_chunk_indices_ptr + chunk_id)
    source_start = tl.load(cu_chunk_ptr + source_chunk)

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, HEAD_DIM, BLOCK_D):
        mask_d = (d_start + offs_d) < HEAD_DIM
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for t_start in range(0, CHUNK_SIZE, BLOCK_TOK):
            offs_t = tl.arange(0, BLOCK_TOK)
            mask_t = (t_start + offs_t) < CHUNK_SIZE

            ptrs = (
                k_ptr
                + (source_start + t_start + offs_t)[:, None] * stride_token
                + head_id * stride_head
                + (d_start + offs_d)[None, :] * stride_d
            )
            vals = tl.load(ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
            vals = vals.to(tl.float32)
            acc += tl.sum(vals, axis=0)

        acc = acc / CHUNK_SIZE
        out_ptrs = out_ptr + chunk_id * out_stride_chunk + head_id * out_stride_head + (d_start + offs_d) * out_stride_d
        tl.store(out_ptrs, acc, mask=mask_d)


@triton.jit
def _dense_chunk_mean_kernel(
    k_ptr,
    cu_chunk_ptr,
    valid_historical_ptr,
    out_ptr,
    stride_token,
    stride_head,
    stride_d,
    out_stride_chunk,
    out_stride_head,
    out_stride_d,
    num_chunk_slots,
    num_kv_heads,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    slot = tl.program_id(0)
    kv_head = tl.program_id(1)
    if slot >= num_chunk_slots or kv_head >= num_kv_heads:
        return
    is_historical = tl.load(valid_historical_ptr + slot) != 0
    source_start = tl.load(cu_chunk_ptr + slot)
    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, HEAD_DIM, BLOCK_D):
        mask_d = (d_start + offs_d) < HEAD_DIM
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t_start in range(0, CHUNK_SIZE, BLOCK_TOK):
            offs_t = t_start + tl.arange(0, BLOCK_TOK)
            mask_t = is_historical & (offs_t < CHUNK_SIZE)
            values = tl.load(
                k_ptr
                + (source_start + offs_t[:, None]) * stride_token
                + kv_head * stride_head
                + (d_start + offs_d)[None, :] * stride_d,
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(values, axis=0)
        acc /= CHUNK_SIZE
        tl.store(
            out_ptr + slot * out_stride_chunk + kv_head * out_stride_head + (d_start + offs_d) * out_stride_d,
            acc,
            mask=mask_d,
        )


def _compute_dense_chunk_means_triton(
    k: torch.Tensor,
    cu_chunk: torch.Tensor,
    valid_historical: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    num_chunk_slots = valid_historical.numel()
    _, num_kv_heads, head_dim = k.shape
    chunk_means = torch.empty(
        (num_chunk_slots, num_kv_heads, head_dim),
        device=k.device,
        dtype=k.dtype,
    )
    block_tok = min(64, triton.next_power_of_2(chunk_size))
    block_d = max(16, triton.next_power_of_2(head_dim))
    _dense_chunk_mean_kernel[(num_chunk_slots, num_kv_heads)](
        k,
        cu_chunk,
        valid_historical,
        chunk_means,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        num_chunk_slots,
        num_kv_heads,
        CHUNK_SIZE=chunk_size,
        HEAD_DIM=head_dim,
        BLOCK_TOK=block_tok,
        BLOCK_D=block_d,
        num_warps=4 if block_d >= 64 else 2,
        num_stages=2,
    )
    return chunk_means


@triton.jit
def _streaming_topk_float_to_key(values):
    value_bits = values.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(value_bits.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(value_bits.shape, 0xFFFFFFFF, dtype=tl.uint32)
    flip_mask = tl.where((value_bits & sign_mask) != 0, full_mask, sign_mask)
    return value_bits ^ flip_mask


@triton.jit
def _chunk_topk_streaming_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    selected_chunks_ptr,
    num_chunks: tl.constexpr,
    num_heads,
    seqlen,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_chunk_mean_chunk,
    stride_chunk_mean_head,
    stride_chunk_mean_d,
    stride_selected_rank,
    stride_selected_head,
    stride_selected_token,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)

    if pid_head >= num_heads:
        return
    kv_head = pid_head // GROUP_SIZE

    offs_token = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    token_mask = offs_token < seqlen
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    chunk_rows = tl.arange(0, BLOCK_CHUNK)

    q_ptrs = q_ptr + offs_token[:, None] * stride_q_seq + pid_head * stride_q_head + offs_d[None, :] * stride_q_d
    q_values = tl.load(
        q_ptrs,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    )

    neg_inf = -1.0e38
    best_packed = tl.zeros(
        [BLOCK_TOK, BLOCK_TOPK],
        dtype=tl.uint64,
    )
    max_index_key = tl.full(
        [BLOCK_CHUNK],
        0xFFFFFFFF,
        dtype=tl.uint32,
    )

    for chunk_start in range(0, num_chunks, BLOCK_CHUNK):
        chunk_offsets = chunk_start + chunk_rows
        chunk_mask = chunk_offsets < num_chunks
        safe_chunk_offsets = tl.where(chunk_mask, chunk_offsets, 0)

        chunk_ptrs = (
            chunk_means_ptr
            + safe_chunk_offsets[:, None] * stride_chunk_mean_chunk
            + kv_head * stride_chunk_mean_head
            + offs_d[None, :] * stride_chunk_mean_d
        )
        chunk_values = tl.load(
            chunk_ptrs,
            mask=chunk_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        gating = tl.dot(
            chunk_values,
            tl.trans(q_values),
            out_dtype=tl.float32,
        )

        chunk_end = tl.load(
            chunk_end_ptr + safe_chunk_offsets,
            mask=chunk_mask,
            other=0,
        )
        batch_end = tl.load(
            batch_end_ptr + safe_chunk_offsets,
            mask=chunk_mask,
            other=0,
        )
        valid = (
            chunk_mask[:, None]
            & token_mask[None, :]
            & (offs_token[None, :] >= chunk_end[:, None])
            & (offs_token[None, :] < batch_end[:, None])
        )
        gating = tl.where(valid, gating, neg_inf)

        scores = tl.trans(gating)
        score_keys = _streaming_topk_float_to_key(scores)
        index_keys = max_index_key - chunk_offsets.to(tl.uint32)
        packed = (score_keys.to(tl.uint64) << 32) | index_keys[None, :].to(tl.uint64)
        packed = tl.where(
            tl.trans(valid),
            packed,
            tl.zeros(packed.shape, dtype=tl.uint64),
        )
        local_packed = tl.topk(packed, BLOCK_TOPK)

        if chunk_start == 0:
            best_packed = local_packed
        else:
            best_packed = tl.bitonic_merge(best_packed)
            best_packed = tl.maximum(best_packed, local_packed)

    best_packed = tl.sort(best_packed, descending=True)
    best_rank_major = tl.trans(best_packed)
    selected_index_keys = best_rank_major.to(tl.uint32)
    selected_indices = (tl.full(best_rank_major.shape, 0xFFFFFFFF, dtype=tl.uint32) - selected_index_keys).to(tl.int32)
    rank_offsets = tl.arange(0, BLOCK_TOPK)
    selected_ptrs = (
        selected_chunks_ptr
        + rank_offsets[:, None] * stride_selected_rank
        + pid_head * stride_selected_head
        + offs_token[None, :] * stride_selected_token
    )
    tl.store(
        selected_ptrs,
        selected_indices,
        mask=((rank_offsets[:, None] < TOPK) & token_mask[None, :] & (best_rank_major != 0)),
    )


def _compute_chunk_means_triton(
    k: torch.Tensor,
    cu_chunk: torch.Tensor,
    filtered_chunk_indices: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute target chunk means directly from the original K tensor."""
    num_chunks = filtered_chunk_indices.numel()
    _, num_heads, head_dim = k.shape
    if num_chunks == 0:
        return torch.empty((0, num_heads, head_dim), device=k.device, dtype=k.dtype)

    chunk_means = torch.empty(
        (num_chunks, num_heads, head_dim),
        device=k.device,
        dtype=k.dtype,
    )

    block_tok = min(64, triton.next_power_of_2(chunk_size))
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_warps = 4 if block_d >= 64 else 2
    num_stages = 4 if chunk_size >= 256 else 2

    grid = (num_chunks, num_heads)

    _chunk_mean_kernel[grid](
        k,
        cu_chunk,
        filtered_chunk_indices,
        chunk_means,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        num_chunks,
        num_heads,
        CHUNK_SIZE=chunk_size,
        HEAD_DIM=head_dim,
        BLOCK_TOK=block_tok,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return chunk_means


def _compute_selected_chunks_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    batch_end: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_chunks, num_kv_heads, head_dim = chunk_means.shape
    num_heads = q.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    group_size = num_heads // num_kv_heads
    seqlen = q.shape[0]

    if topk <= 0 or num_chunks == 0:
        return torch.empty(
            (0, num_heads, seqlen),
            device=q.device,
            dtype=torch.int32,
        )
    if topk > num_chunks:
        raise ValueError(f"topk ({topk}) cannot exceed num_chunks ({num_chunks})")

    block_topk = triton.next_power_of_2(topk)
    if block_topk > 64:
        raise ValueError(f"streaming Triton Top-K currently supports topk <= 64, got topk={topk}")

    selected_chunks = torch.empty(
        (topk, num_heads, seqlen),
        device=q.device,
        dtype=torch.int32,
    )
    _fill_triton(selected_chunks, -1)

    block_d = max(16, triton.next_power_of_2(head_dim))
    # This kernel intentionally uses an explicit shape policy instead of
    # triton.autotune.  FlagTree's AABS extension mutates Config.kwargs in
    # place when a tile is larger than a runtime tensor dimension, so a small
    # first invocation can permanently shrink every candidate and break later
    # top-k calls.  NCU shows 64x32 is the best long-sequence K=8 tile on H100:
    # it reuses each chunk vector across 64 queries while keeping the dot and
    # selection working sets at 2048 values.  Wider requested top-k values
    # need a local chunk tile at least as wide as BLOCK_TOPK for tl.topk
    # correctness, so cap BLOCK_TOK to preserve that same working-set bound.
    block_chunk = max(32, block_topk)
    max_block_tok = 2048 // block_chunk
    # Query reuse pays off once routing traverses at least one full 32-chunk
    # tile.  Keying this decision to routing work (rather than a benchmark-
    # specific sequence-length threshold) also scales across chunk sizes and
    # packed batches.
    reuse_block_tok = 64 if num_chunks >= 32 else 32
    block_tok = min(reuse_block_tok, max_block_tok)
    num_stages = 2 if block_chunk >= 64 else 1
    grid = (cdiv(seqlen, block_tok), num_heads)

    _chunk_topk_streaming_kernel[grid](
        q,
        chunk_means,
        chunk_end,
        batch_end,
        selected_chunks,
        num_chunks,
        num_heads,
        seqlen,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        selected_chunks.stride(0),
        selected_chunks.stride(1),
        selected_chunks.stride(2),
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        BLOCK_TOK=block_tok,
        BLOCK_CHUNK=block_chunk,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=num_stages,
    )

    return selected_chunks


@triton.jit
def _chunk_topk_batched_streaming_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    cu_seqlens_ptr,
    selected_chunks_ptr,
    num_heads,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_chunk_mean_chunk,
    stride_chunk_mean_head,
    stride_chunk_mean_d,
    stride_selected_rank,
    stride_selected_head,
    stride_selected_token,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MAX_CHUNKS_PER_SEQUENCE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Select historical chunks without scanning chunks from other batches."""
    pid_tok = tl.program_id(0)
    pid_head = tl.program_id(1)
    batch = tl.program_id(2)

    if pid_head >= num_heads:
        return
    kv_head = pid_head // GROUP_SIZE
    sequence_start = tl.load(cu_seqlens_ptr + batch)
    sequence_end = tl.load(cu_seqlens_ptr + batch + 1)

    local_tokens = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    offs_token = sequence_start + local_tokens
    token_mask = offs_token < sequence_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    chunk_rows = tl.arange(0, BLOCK_CHUNK)

    q_ptrs = q_ptr + offs_token[:, None] * stride_q_seq + pid_head * stride_q_head + offs_d[None, :] * stride_q_d
    q_values = tl.load(
        q_ptrs,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    )

    best_packed = tl.zeros([BLOCK_TOK, BLOCK_TOPK], dtype=tl.uint64)
    max_index_key = tl.full([BLOCK_CHUNK], 0xFFFFFFFF, dtype=tl.uint32)
    chunk_base = batch * MAX_CHUNKS_PER_SEQUENCE

    for local_start in range(
        0,
        MAX_CHUNKS_PER_SEQUENCE,
        BLOCK_CHUNK,
    ):
        local_chunks = local_start + chunk_rows
        chunk_mask = local_chunks < MAX_CHUNKS_PER_SEQUENCE
        chunk_offsets = chunk_base + local_chunks
        safe_chunk_offsets = tl.where(chunk_mask, chunk_offsets, chunk_base)

        chunk_ptrs = (
            chunk_means_ptr
            + safe_chunk_offsets[:, None] * stride_chunk_mean_chunk
            + kv_head * stride_chunk_mean_head
            + offs_d[None, :] * stride_chunk_mean_d
        )
        chunk_values = tl.load(
            chunk_ptrs,
            mask=chunk_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        gating = tl.dot(
            chunk_values,
            tl.trans(q_values),
            out_dtype=tl.float32,
        )

        chunk_end = tl.load(
            chunk_end_ptr + safe_chunk_offsets,
            mask=chunk_mask,
            other=0,
        )
        valid = (
            chunk_mask[:, None]
            & token_mask[None, :]
            & (chunk_end[:, None] > sequence_start)
            & (offs_token[None, :] >= chunk_end[:, None])
        )

        scores = tl.trans(gating)
        score_keys = _streaming_topk_float_to_key(scores)
        index_keys = max_index_key - chunk_offsets.to(tl.uint32)
        packed = (score_keys.to(tl.uint64) << 32) | index_keys[None, :].to(tl.uint64)
        packed = tl.where(
            tl.trans(valid),
            packed,
            tl.zeros(packed.shape, dtype=tl.uint64),
        )
        local_packed = tl.topk(packed, BLOCK_TOPK)

        if local_start == 0:
            best_packed = local_packed
        else:
            best_packed = tl.bitonic_merge(best_packed)
            best_packed = tl.maximum(best_packed, local_packed)

    best_packed = tl.sort(best_packed, descending=True)
    best_rank_major = tl.trans(best_packed)
    selected_index_keys = best_rank_major.to(tl.uint32)
    selected_indices = (tl.full(best_rank_major.shape, 0xFFFFFFFF, dtype=tl.uint32) - selected_index_keys).to(tl.int32)
    rank_offsets = tl.arange(0, BLOCK_TOPK)
    selected_ptrs = (
        selected_chunks_ptr
        + rank_offsets[:, None] * stride_selected_rank
        + pid_head * stride_selected_head
        + offs_token[None, :] * stride_selected_token
    )
    tl.store(
        selected_ptrs,
        selected_indices,
        mask=((rank_offsets[:, None] < TOPK) & token_mask[None, :] & (best_rank_major != 0)),
    )


def _compute_selected_chunks_batched_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    max_chunks_per_sequence: int,
    topk: int,
) -> torch.Tensor:
    """Batch-aware streaming top-k for dense-slot inference metadata."""
    num_chunk_slots, num_kv_heads, head_dim = chunk_means.shape
    num_heads = q.shape[1]
    batch_size = cu_seqlens.numel() - 1
    expected_slots = batch_size * max_chunks_per_sequence
    if num_chunk_slots != expected_slots:
        raise ValueError(
            "dense chunk metadata must contain batch_size * "
            "max_chunks_per_sequence slots, got "
            f"{num_chunk_slots} and expected {expected_slots}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    if topk <= 0 or num_chunk_slots == 0:
        return torch.empty(
            (0, num_heads, q.shape[0]),
            device=q.device,
            dtype=torch.int32,
        )
    if topk >= max_chunks_per_sequence:
        raise ValueError(
            f"historical topk ({topk}) must be smaller than the per-sequence chunk capacity ({max_chunks_per_sequence})"
        )

    block_topk = triton.next_power_of_2(topk)
    if block_topk > 64:
        raise ValueError(f"streaming Triton Top-K currently supports topk <= 64, got topk={topk}")
    selected_chunks = torch.empty(
        (topk, num_heads, q.shape[0]),
        device=q.device,
        dtype=torch.int32,
    )
    _fill_triton(selected_chunks, -1)

    block_chunk = max(32, block_topk)
    max_block_tok = 2048 // block_chunk
    reuse_block_tok = 64 if max_chunks_per_sequence >= 32 else 32
    block_tok = min(reuse_block_tok, max_block_tok)
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_stages = 2 if block_chunk >= 64 else 1
    grid = (cdiv(max_seqlen, block_tok), num_heads, batch_size)
    _chunk_topk_batched_streaming_kernel[grid](
        q,
        chunk_means,
        chunk_end,
        cu_seqlens,
        selected_chunks,
        num_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        chunk_means.stride(0),
        chunk_means.stride(1),
        chunk_means.stride(2),
        selected_chunks.stride(0),
        selected_chunks.stride(1),
        selected_chunks.stride(2),
        HEAD_DIM=head_dim,
        GROUP_SIZE=num_heads // num_kv_heads,
        MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        BLOCK_TOK=block_tok,
        BLOCK_CHUNK=block_chunk,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=num_stages,
    )
    return selected_chunks


@triton.jit
def _target_chunk_bounds_kernel(
    cu_chunk_ptr,
    filtered_chunk_indices_ptr,
    chunk_to_batch_ptr,
    cu_seqlens_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    num_chunks,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_chunks
    source_chunks = tl.load(filtered_chunk_indices_ptr + offsets, mask=mask, other=0)
    batch_indices = tl.load(chunk_to_batch_ptr + source_chunks, mask=mask, other=0)
    chunk_ends = tl.load(cu_chunk_ptr + source_chunks + 1, mask=mask, other=0)
    batch_ends = tl.load(cu_seqlens_ptr + batch_indices + 1, mask=mask, other=0)
    tl.store(chunk_end_ptr + offsets, chunk_ends, mask=mask)
    tl.store(batch_end_ptr + offsets, batch_ends, mask=mask)


def _target_chunk_bounds_triton(
    cu_chunk,
    filtered_chunk_indices,
    chunk_to_batch,
    cu_seqlens,
):
    num_chunks = filtered_chunk_indices.numel()
    chunk_end = torch.empty((num_chunks,), device=cu_seqlens.device, dtype=torch.int32)
    batch_end = torch.empty_like(chunk_end)
    if num_chunks == 0:
        return chunk_end, batch_end
    block = 256
    _target_chunk_bounds_kernel[(cdiv(num_chunks, block),)](
        cu_chunk,
        filtered_chunk_indices,
        chunk_to_batch,
        cu_seqlens,
        chunk_end,
        batch_end,
        num_chunks,
        BLOCK=block,
    )
    return chunk_end, batch_end


@triton.jit
def _count_sparse_routes_kernel(
    selected_chunks_ptr,
    expert_counts_ptr,
    num_slots,
    num_heads: tl.constexpr,
    seqlen,
    BLOCK: tl.constexpr,
):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = slots < num_slots
    chunks = tl.load(selected_chunks_ptr + slots, mask=mask, other=-1)
    within_rank = slots % (num_heads * seqlen)
    heads = within_rank // seqlen
    valid = mask & (chunks >= 0)
    experts = chunks * num_heads + heads
    # The next kernel is stream-ordered; only atomicity is needed here.
    tl.atomic_add(
        expert_counts_ptr + experts,
        1,
        mask=valid,
        sem="relaxed",
    )


@triton.jit
def _exclusive_prefix_sum_kernel(
    counts_ptr,
    offsets_ptr,
    num_experts,
    BLOCK_EXPERTS: tl.constexpr,
):
    experts = tl.arange(0, BLOCK_EXPERTS)
    mask = experts < num_experts
    counts = tl.load(counts_ptr + experts, mask=mask, other=0)
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(offsets_ptr + experts + 1, inclusive, mask=mask)
    tl.store(offsets_ptr, 0)


@triton.jit
def _block_exclusive_prefix_sum_kernel(
    counts_ptr,
    offsets_ptr,
    block_sums_ptr,
    num_experts,
    BLOCK_EXPERTS: tl.constexpr,
):
    block = tl.program_id(0)
    experts = block * BLOCK_EXPERTS + tl.arange(0, BLOCK_EXPERTS)
    mask = experts < num_experts
    counts = tl.load(counts_ptr + experts, mask=mask, other=0)
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(offsets_ptr + experts + 1, inclusive, mask=mask)
    tl.store(block_sums_ptr + block, tl.sum(counts, axis=0))
    if block == 0:
        tl.store(offsets_ptr, 0)


@triton.jit
def _add_prefix_block_offsets_kernel(
    offsets_ptr,
    block_offsets_ptr,
    num_experts,
    BLOCK_EXPERTS: tl.constexpr,
):
    block = tl.program_id(0)
    experts = block * BLOCK_EXPERTS + tl.arange(0, BLOCK_EXPERTS)
    mask = experts < num_experts
    block_offset = tl.load(block_offsets_ptr + block)
    local_offsets = tl.load(offsets_ptr + experts + 1, mask=mask, other=0)
    tl.store(
        offsets_ptr + experts + 1,
        local_offsets + block_offset,
        mask=mask,
    )


def _exclusive_prefix_sum_triton(
    counts: torch.Tensor,
    offsets: torch.Tensor,
):
    """Hierarchical device-side exclusive scan with no host synchronization."""
    num_experts = counts.numel()
    if offsets.numel() != num_experts + 1:
        raise ValueError("offsets must have exactly counts.numel() + 1 entries")
    if num_experts == 0:
        _fill_triton(offsets, 0)
        return
    if num_experts <= _PREFIX_SCAN_BLOCK:
        block_experts = triton.next_power_of_2(num_experts)
        _exclusive_prefix_sum_kernel[(1,)](
            counts,
            offsets,
            num_experts,
            BLOCK_EXPERTS=block_experts,
            num_warps=4 if block_experts <= 512 else 8,
        )
        return

    num_blocks = cdiv(num_experts, _PREFIX_SCAN_BLOCK)
    block_sums = torch.empty((num_blocks,), device=counts.device, dtype=counts.dtype)
    block_offsets = torch.empty((num_blocks + 1,), device=counts.device, dtype=counts.dtype)
    _block_exclusive_prefix_sum_kernel[(num_blocks,)](
        counts,
        offsets,
        block_sums,
        num_experts,
        BLOCK_EXPERTS=_PREFIX_SCAN_BLOCK,
        num_warps=8,
    )
    _exclusive_prefix_sum_triton(block_sums, block_offsets)
    _add_prefix_block_offsets_kernel[(num_blocks,)](
        offsets,
        block_offsets,
        num_experts,
        BLOCK_EXPERTS=_PREFIX_SCAN_BLOCK,
        num_warps=8,
    )


@triton.jit
def _scatter_sparse_routes_kernel(
    selected_chunks_ptr,
    expert_offsets_ptr,
    expert_write_ptr,
    query_indices_ptr,
    route_positions_ptr,
    num_slots,
    num_heads: tl.constexpr,
    seqlen,
    BLOCK: tl.constexpr,
):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = slots < num_slots
    chunks = tl.load(selected_chunks_ptr + slots, mask=mask, other=-1)
    slots_per_rank = num_heads * seqlen
    ranks = slots // slots_per_rank
    within_rank = slots - ranks * slots_per_rank
    heads = within_rank // seqlen
    tokens = within_rank - heads * seqlen
    valid = mask & (chunks >= 0)
    experts = chunks * num_heads + heads
    # Prefix offsets are immutable and the consumer launch is stream-ordered.
    local_positions = tl.atomic_add(
        expert_write_ptr + experts,
        1,
        mask=valid,
        sem="relaxed",
    )
    positions = tl.load(expert_offsets_ptr + experts, mask=valid, other=0)
    positions += local_positions
    query_indices = tokens * num_heads + heads
    tl.store(query_indices_ptr + positions, query_indices, mask=valid)
    route_slots = ranks * slots_per_rank + tokens * num_heads + heads
    tl.store(route_positions_ptr + route_slots, positions, mask=valid)


@triton.jit
def _block_max_counts_kernel(
    counts_ptr,
    block_max_ptr,
    num_experts,
    BLOCK_EXPERTS: tl.constexpr,
):
    block = tl.program_id(0)
    experts = block * BLOCK_EXPERTS + tl.arange(0, BLOCK_EXPERTS)
    counts = tl.load(counts_ptr + experts, mask=experts < num_experts, other=0)
    tl.store(block_max_ptr + block, tl.max(counts, axis=0))


def _max_counts_triton(counts: torch.Tensor) -> torch.Tensor:
    """Hierarchically reduce route counts without an oversized CTA."""
    current = counts
    while current.numel() > 1:
        num_experts = current.numel()
        block_experts = min(
            _PREFIX_SCAN_BLOCK,
            triton.next_power_of_2(num_experts),
        )
        num_blocks = cdiv(num_experts, block_experts)
        block_max = torch.empty((num_blocks,), device=counts.device, dtype=counts.dtype)
        _block_max_counts_kernel[(num_blocks,)](
            current,
            block_max,
            num_experts,
            BLOCK_EXPERTS=block_experts,
            num_warps=4 if block_experts <= 512 else 8,
        )
        current = block_max
    return current


def _build_sparse_routes_triton(
    selected_chunks: torch.Tensor,
    num_chunks: int,
    num_heads: int,
    seqlen: int,
):
    """Group selected queries by (target chunk, head) using Triton atomics."""
    topk = selected_chunks.shape[0]
    num_experts = num_chunks * num_heads
    num_slots = selected_chunks.numel()
    expert_counts = torch.empty((num_experts,), device=selected_chunks.device, dtype=torch.int32)
    expert_offsets = torch.empty((num_experts + 1,), device=selected_chunks.device, dtype=torch.int32)
    expert_write = torch.empty_like(expert_counts)
    query_indices = torch.empty((num_slots,), device=selected_chunks.device, dtype=torch.int32)
    route_positions = torch.empty(
        (topk, seqlen * num_heads),
        device=selected_chunks.device,
        dtype=torch.int32,
    )
    _fill_triton(expert_counts, 0)
    _fill_triton(expert_write, 0)
    _fill_triton(route_positions, -1)

    block = 256
    _count_sparse_routes_kernel[(cdiv(num_slots, block),)](
        selected_chunks,
        expert_counts,
        num_slots,
        num_heads=num_heads,
        seqlen=seqlen,
        BLOCK=block,
    )
    _exclusive_prefix_sum_triton(expert_counts, expert_offsets)
    _scatter_sparse_routes_kernel[(cdiv(num_slots, block),)](
        selected_chunks,
        expert_offsets,
        expert_write,
        query_indices,
        route_positions,
        num_slots,
        num_heads=num_heads,
        seqlen=seqlen,
        BLOCK=block,
    )

    max_count_device = _max_counts_triton(expert_counts)
    num_routes = int(expert_offsets[-1].item())
    max_count = int(max_count_device.item())
    return (
        query_indices[:num_routes],
        route_positions,
        expert_counts,
        expert_offsets,
        max_count,
    )


@triton.jit
def _scatter_sparse_routes_device_kernel(
    selected_chunks_ptr,
    expert_offsets_ptr,
    expert_write_ptr,
    query_indices_ptr,
    route_experts_ptr,
    route_positions_ptr,
    num_slots,
    num_heads: tl.constexpr,
    seqlen,
    BLOCK: tl.constexpr,
):
    """Compact routes while keeping every dynamic size on the device."""
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = slots < num_slots
    chunks = tl.load(selected_chunks_ptr + slots, mask=mask, other=-1)
    slots_per_rank = num_heads * seqlen
    ranks = slots // slots_per_rank
    within_rank = slots - ranks * slots_per_rank
    heads = within_rank // seqlen
    tokens = within_rank - heads * seqlen
    valid = mask & (chunks >= 0)
    experts = chunks * num_heads + heads
    # Prefix offsets are immutable and the consumer launch is stream-ordered.
    local_positions = tl.atomic_add(
        expert_write_ptr + experts,
        1,
        mask=valid,
        sem="relaxed",
    )
    positions = tl.load(expert_offsets_ptr + experts, mask=valid, other=0)
    positions += local_positions
    query_indices = tokens * num_heads + heads
    tl.store(query_indices_ptr + positions, query_indices, mask=valid)
    tl.store(route_experts_ptr + positions, experts, mask=valid)
    route_slots = ranks * slots_per_rank + tokens * num_heads + heads
    tl.store(route_positions_ptr + route_slots, positions, mask=valid)


@triton.jit
def _build_route_block_descriptors_kernel(
    query_indices_ptr,
    route_experts_ptr,
    expert_offsets_ptr,
    descriptor_counter_ptr,
    descriptor_experts_ptr,
    descriptor_starts_ptr,
    num_slots,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    positions = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = positions < num_slots
    query_indices = tl.load(query_indices_ptr + positions, mask=mask, other=-1)
    experts = tl.load(route_experts_ptr + positions, mask=mask, other=-1)
    valid = mask & (query_indices >= 0) & (experts >= 0)
    safe_experts = tl.where(valid, experts, 0)
    expert_starts = tl.load(expert_offsets_ptr + safe_experts, mask=valid, other=0)
    local_positions = positions - expert_starts
    is_block_start = valid & ((local_positions % BLOCK_M) == 0)
    counter_ptrs = descriptor_counter_ptr + tl.zeros([BLOCK], dtype=tl.int32)
    # Each descriptor needs a unique slot; no acquire/release ordering is used.
    descriptor_positions = tl.atomic_add(
        counter_ptrs,
        1,
        mask=is_block_start,
        sem="relaxed",
    )
    tl.store(
        descriptor_experts_ptr + descriptor_positions,
        experts,
        mask=is_block_start,
    )
    tl.store(
        descriptor_starts_ptr + descriptor_positions,
        positions,
        mask=is_block_start,
    )


def _build_sparse_routes_device_triton(
    selected_chunks: torch.Tensor,
    num_chunks: int,
    num_heads: int,
    seqlen: int,
    block_m: int,
):
    """Build route metadata without reading route counts back to Python."""
    topk = selected_chunks.shape[0]
    num_experts = num_chunks * num_heads
    num_slots = selected_chunks.numel()
    expert_counts = torch.empty((num_experts,), device=selected_chunks.device, dtype=torch.int32)
    expert_offsets = torch.empty((num_experts + 1,), device=selected_chunks.device, dtype=torch.int32)
    expert_write = torch.empty_like(expert_counts)
    query_indices = torch.empty((num_slots,), device=selected_chunks.device, dtype=torch.int32)
    route_experts = torch.empty_like(query_indices)
    route_positions = torch.empty(
        (topk, seqlen * num_heads),
        device=selected_chunks.device,
        dtype=torch.int32,
    )
    _fill_triton(expert_counts, 0)
    _fill_triton(expert_write, 0)
    _fill_triton(query_indices, -1)
    _fill_triton(route_experts, -1)
    _fill_triton(route_positions, -1)

    block = 256
    _count_sparse_routes_kernel[(cdiv(num_slots, block),)](
        selected_chunks,
        expert_counts,
        num_slots,
        num_heads=num_heads,
        seqlen=seqlen,
        BLOCK=block,
    )
    _exclusive_prefix_sum_triton(expert_counts, expert_offsets)
    _scatter_sparse_routes_device_kernel[(cdiv(num_slots, block),)](
        selected_chunks,
        expert_offsets,
        expert_write,
        query_indices,
        route_experts,
        route_positions,
        num_slots,
        num_heads=num_heads,
        seqlen=seqlen,
        BLOCK=block,
    )

    # sum(ceil(count_i / block_m)) <= ceil(num_slots / block_m) + num_experts.
    # Allocate that deterministic bound and mark unused descriptors with -1.
    max_descriptors = cdiv(num_slots, block_m) + num_experts
    descriptor_experts = torch.empty((max_descriptors,), device=selected_chunks.device, dtype=torch.int32)
    descriptor_starts = torch.empty_like(descriptor_experts)
    descriptor_counter = torch.empty((1,), device=selected_chunks.device, dtype=torch.int32)
    _fill_triton(descriptor_experts, -1)
    _fill_triton(descriptor_starts, -1)
    _fill_triton(descriptor_counter, 0)
    _build_route_block_descriptors_kernel[(cdiv(num_slots, block),)](
        query_indices,
        route_experts,
        expert_offsets,
        descriptor_counter,
        descriptor_experts,
        descriptor_starts,
        num_slots,
        BLOCK_M=block_m,
        BLOCK=block,
    )
    return (
        query_indices,
        route_positions,
        expert_offsets,
        descriptor_experts,
        descriptor_starts,
    )


@triton.jit
def _gather_moba_kv_kernel(
    k_ptr,
    v_ptr,
    cu_chunk_ptr,
    filtered_chunk_indices_ptr,
    moba_kv_ptr,
    num_rows,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_mkv_t,
    stride_mkv_kv,
    stride_mkv_d,
    NUM_HEADS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    expert = row // CHUNK_SIZE
    chunk_token = row - expert * CHUNK_SIZE
    target_chunk = expert // NUM_HEADS
    head = expert - target_chunk * NUM_HEADS
    source_chunk = tl.load(filtered_chunk_indices_ptr + target_chunk)
    source_token = tl.load(cu_chunk_ptr + source_chunk) + chunk_token
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < HEAD_DIM
    k_values = tl.load(
        k_ptr + source_token * stride_k_t + head * stride_k_h + offsets_d * stride_k_d,
        mask=mask_d,
        other=0.0,
    )
    v_values = tl.load(
        v_ptr + source_token * stride_v_t + head * stride_v_h + offsets_d * stride_v_d,
        mask=mask_d,
        other=0.0,
    )
    tl.store(
        moba_kv_ptr + row * stride_mkv_t + offsets_d * stride_mkv_d,
        k_values,
        mask=mask_d,
    )
    tl.store(
        moba_kv_ptr + row * stride_mkv_t + stride_mkv_kv + offsets_d * stride_mkv_d,
        v_values,
        mask=mask_d,
    )


@triton.jit
def _scatter_moba_kv_grads_kernel(
    dmoba_kv_ptr,
    cu_chunk_ptr,
    filtered_chunk_indices_ptr,
    dk_ptr,
    dv_ptr,
    num_rows,
    stride_dmkv_t,
    stride_dmkv_kv,
    stride_dmkv_d,
    stride_dk_t,
    stride_dk_h,
    stride_dk_d,
    stride_dv_t,
    stride_dv_h,
    stride_dv_d,
    NUM_HEADS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    expert = row // CHUNK_SIZE
    chunk_token = row - expert * CHUNK_SIZE
    target_chunk = expert // NUM_HEADS
    head = expert - target_chunk * NUM_HEADS
    source_chunk = tl.load(filtered_chunk_indices_ptr + target_chunk)
    source_token = tl.load(cu_chunk_ptr + source_chunk) + chunk_token
    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < HEAD_DIM
    dk_values = tl.load(
        dmoba_kv_ptr + row * stride_dmkv_t + offsets_d * stride_dmkv_d,
        mask=mask_d,
        other=0.0,
    )
    dv_values = tl.load(
        dmoba_kv_ptr + row * stride_dmkv_t + stride_dmkv_kv + offsets_d * stride_dmkv_d,
        mask=mask_d,
        other=0.0,
    )
    tl.store(
        dk_ptr + source_token * stride_dk_t + head * stride_dk_h + offsets_d * stride_dk_d,
        dk_values,
        mask=mask_d,
    )
    tl.store(
        dv_ptr + source_token * stride_dv_t + head * stride_dv_h + offsets_d * stride_dv_d,
        dv_values,
        mask=mask_d,
    )


class _GatherMobaKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, cu_chunk, filtered_chunk_indices, chunk_size):
        total_tokens, num_heads, head_dim = k.shape
        num_experts = filtered_chunk_indices.numel() * num_heads
        num_rows = num_experts * chunk_size
        moba_kv = torch.empty((num_rows, 2, 1, head_dim), device=k.device, dtype=k.dtype)
        block_d = max(16, triton.next_power_of_2(head_dim))
        _gather_moba_kv_kernel[(num_rows,)](
            k,
            v,
            cu_chunk,
            filtered_chunk_indices,
            moba_kv,
            num_rows,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            moba_kv.stride(0),
            moba_kv.stride(1),
            moba_kv.stride(3),
            NUM_HEADS=num_heads,
            CHUNK_SIZE=chunk_size,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(cu_chunk, filtered_chunk_indices)
        ctx.input_shape = (total_tokens, num_heads, head_dim)
        ctx.chunk_size = chunk_size
        return moba_kv

    @staticmethod
    def backward(ctx, dmoba_kv):
        cu_chunk, filtered_chunk_indices = ctx.saved_tensors
        total_tokens, num_heads, head_dim = ctx.input_shape
        dk = torch.empty(ctx.input_shape, device=dmoba_kv.device, dtype=dmoba_kv.dtype)
        dv = torch.empty_like(dk)
        _fill_triton(dk, 0.0)
        _fill_triton(dv, 0.0)
        num_rows = dmoba_kv.shape[0]
        block_d = max(16, triton.next_power_of_2(head_dim))
        _scatter_moba_kv_grads_kernel[(num_rows,)](
            dmoba_kv,
            cu_chunk,
            filtered_chunk_indices,
            dk,
            dv,
            num_rows,
            dmoba_kv.stride(0),
            dmoba_kv.stride(1),
            dmoba_kv.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            NUM_HEADS=num_heads,
            CHUNK_SIZE=ctx.chunk_size,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
        )
        return dk, dv, None, None, None


def _gather_moba_kv_triton(k, v, cu_chunk, filtered_chunk_indices, chunk_size):
    return _GatherMobaKV.apply(k, v, cu_chunk, filtered_chunk_indices, chunk_size)


@triton.jit
def _linear_offsets_kernel(output_ptr, numel, multiplier, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    tl.store(output_ptr + offsets, offsets * multiplier, mask=mask)


def _linear_offsets_triton(numel, multiplier, device):
    output = torch.empty((numel,), device=device, dtype=torch.int32)
    block = 256
    _linear_offsets_kernel[(cdiv(numel, block),)](
        output,
        numel,
        multiplier,
        BLOCK=block,
    )
    return output


@triton.jit
def _gather_moba_backward_inputs_kernel(
    dout_ptr,
    out_ptr,
    lse_ptr,
    indices_ptr,
    gathered_dout_ptr,
    gathered_out_ptr,
    gathered_lse_ptr,
    num_indices,
    head_dim,
    stride_dout_token,
    stride_dout_head,
    stride_dout_d,
    stride_out_token,
    stride_out_head,
    stride_out_d,
    stride_gdout_row,
    stride_gdout_d,
    stride_gout_row,
    stride_gout_d,
    stride_lse_head,
    stride_lse_token,
    num_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Kernel to gather backward pass inputs for the MoBA branch.
    Since MoBA works on a sparse subset of Q, we need to gather gradients (dout),
    outputs (out), and LSE scores corresponding to those sparse indices.
    """
    pid = tl.program_id(0)
    if pid >= num_indices:
        return

    sh_idx = tl.load(indices_ptr + pid)
    token = sh_idx // num_heads
    head = sh_idx - token * num_heads
    lse_val = tl.load(lse_ptr + head * stride_lse_head + token * stride_lse_token)
    tl.store(gathered_lse_ptr + pid, lse_val)

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, head_dim, BLOCK_D):
        mask_d = (d_start + offs_d) < head_dim

        dout_ptrs = dout_ptr + token * stride_dout_token + head * stride_dout_head + (d_start + offs_d) * stride_dout_d
        dout_vals = tl.load(dout_ptrs, mask=mask_d, other=0.0)
        gdout_ptrs = gathered_dout_ptr + pid * stride_gdout_row + (d_start + offs_d) * stride_gdout_d
        tl.store(gdout_ptrs, dout_vals, mask=mask_d)

        out_ptrs = out_ptr + token * stride_out_token + head * stride_out_head + (d_start + offs_d) * stride_out_d
        out_vals = tl.load(out_ptrs, mask=mask_d, other=0.0)
        gout_ptrs = gathered_out_ptr + pid * stride_gout_row + (d_start + offs_d) * stride_gout_d
        tl.store(gout_ptrs, out_vals, mask=mask_d)


def _gather_moba_backward_inputs_triton(
    d_output: torch.Tensor,
    output: torch.Tensor,
    mixed_attn_vlse_flat: torch.Tensor,
    moba_indices: torch.Tensor,
    gathered_d_output: torch.Tensor,
    gathered_output: torch.Tensor,
    gathered_lse: torch.Tensor,
):
    """Wrapper for the backward gather kernel"""
    num_indices = moba_indices.numel()
    if num_indices == 0:
        return

    head_dim = d_output.shape[2]

    block_d = min(128, triton.next_power_of_2(head_dim))

    grid = (num_indices,)

    _gather_moba_backward_inputs_kernel[grid](
        d_output,
        output,
        mixed_attn_vlse_flat,
        moba_indices,
        gathered_d_output,
        gathered_output,
        gathered_lse,
        num_indices,
        head_dim,
        d_output.stride(0),
        d_output.stride(1),
        d_output.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        gathered_d_output.stride(0),
        gathered_d_output.stride(1),
        gathered_output.stride(0),
        gathered_output.stride(1),
        mixed_attn_vlse_flat.stride(0),
        mixed_attn_vlse_flat.stride(1),
        num_heads=mixed_attn_vlse_flat.shape[0],
        BLOCK_D=block_d,
    )
