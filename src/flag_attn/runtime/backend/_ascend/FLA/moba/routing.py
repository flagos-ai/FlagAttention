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

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from . import routes_raw, topk_raw

_PREFIX_SCAN_BLOCK = 1024


def _is_accelerator_tensor(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` lives on an accelerator rather than the host."""
    return tensor.device.type in ("cuda", "musa", "gcu", "npu", "mlu", "xpu")


def cdiv(x, y):
    return (x + y - 1) // y


# Cap complex launches at 32768 programs and stride over remaining work.
# Simple elementwise kernels can use the runtime limit of 65535 programs.
_MAX_CORE_DIM0 = 32768
_MAX_CORE_DIM0_SIMPLE = 65535
# Token block used by the blocked route builders.
_ROUTE_BLOCK_T = 1024


def _capped_grid(num_programs: int, dim0_cap: int = _MAX_CORE_DIM0):
    """Cap a 1D program count at the Ascend per-launch limit."""
    num_programs = max(1, int(num_programs))
    return (min(num_programs, dim0_cap),)


@triton.jit
def _fill_kernel(output_ptr, numel, value: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (numel + BLOCK - 1) // BLOCK
    iters = (num_blocks - pid + stride - 1) // stride
    for i in range(0, iters):
        offsets = (pid + i * stride) * BLOCK + tl.arange(0, BLOCK)
        tl.store(output_ptr + offsets, value, mask=offsets < numel)


def _fill_triton(tensor: torch.Tensor, value) -> None:
    if tensor.numel() == 0:
        return
    block = 256
    _fill_kernel[_capped_grid(cdiv(tensor.numel(), block), _MAX_CORE_DIM0_SIMPLE)](
        tensor,
        tensor.numel(),
        value=value,
        BLOCK=block,
    )


@triton.jit
def _cast_kernel(input_ptr, output_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (numel + BLOCK - 1) // BLOCK
    iters = (num_blocks - pid + stride - 1) // stride
    for i in range(0, iters):
        offsets = (pid + i * stride) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, values, mask=mask)


def _cast_triton(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    output = torch.empty(tensor.shape, device=tensor.device, dtype=dtype)
    if tensor.numel() == 0:
        return output
    block = 256
    _cast_kernel[_capped_grid(cdiv(tensor.numel(), block), _MAX_CORE_DIM0_SIMPLE)](
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
    _cast_kernel[_capped_grid(cdiv(input_tensor.numel(), block), _MAX_CORE_DIM0_SIMPLE)](
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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    total_jobs = total_tokens * NUM_Q_HEADS
    iters = (total_jobs - pid + stride - 1) // stride
    for i in range(0, iters):
        job = pid + i * stride
        token = job // NUM_Q_HEADS
        q_head = job - token * NUM_Q_HEADS
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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    total_jobs = total_tokens * NUM_KV_HEADS
    iters = (total_jobs - pid + stride - 1) // stride
    for i in range(0, iters):
        job = pid + i * stride
        token = job // NUM_KV_HEADS
        kv_head = job - token * NUM_KV_HEADS
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
        _expand_kv_heads_kernel[_capped_grid(total_tokens * num_q_heads)](
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
        _reduce_expanded_kv_grads_kernel[_capped_grid(total_tokens * num_kv_heads)](
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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (num_rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    iters = (num_blocks - pid + stride - 1) // stride
    for i in range(0, iters):
        row_offsets = (
            (pid + i * stride) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        )
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
        return _capped_grid(triton.cdiv(indices.numel(), meta["BLOCK_ROWS"]))

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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_rows - pid + stride - 1) // stride
    for i in range(0, iters):
        row = pid + i * stride
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
        _gather_moba_q_backward_kernel[_capped_grid(indices.numel())](
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
    if not _is_accelerator_tensor(q) or not _is_accelerator_tensor(indices):
        raise ValueError("gather_moba_q requires accelerator tensors")
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
    """Build packed chunk metadata on the host and upload it to the device."""
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
        tl.where(is_historical, next_chunk_start, 2147483647),
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


def _chunk_means_gather_torch(
    k: torch.Tensor,
    starts: torch.Tensor,
    chunk_size: int,
    valid_historical: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Gather contiguous chunk rows and reduce them with CANN mean.

    Return None when the input does not qualify for this path.
    """
    if chunk_size is None or chunk_size <= 0:
        return None
    num_chunks = starts.numel()
    if num_chunks == 0:
        return None
    total_tokens = k.shape[0]
    if total_tokens < chunk_size:
        return None
    if not k.is_contiguous():
        return None
    if k.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return None
    clamp_max = total_tokens - chunk_size
    safe_starts = starts.to(torch.int64).clamp_(0, clamp_max)
    offs = torch.arange(chunk_size, device=k.device, dtype=torch.int64)
    index = (safe_starts[:, None] + offs[None, :]).reshape(-1)
    gathered = torch.index_select(k, 0, index).view(
        num_chunks, chunk_size, k.shape[1], k.shape[2]
    )
    means = gathered.mean(dim=1, dtype=torch.float32).to(k.dtype)
    if valid_historical is not None:
        means = means * valid_historical.to(k.dtype)[:, None, None]
    return means


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
    BLOCK_HD: tl.constexpr,
    EXACT_HD: tl.constexpr,
    EXACT_T: tl.constexpr,
):
    """Compute chunk means from contiguous token rows.

    Each program reduces a [BLOCK_TOK, heads * HEAD_DIM] tile along axis 0.
    """
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_chunks - pid + stride - 1) // stride
    hd_total = num_heads * HEAD_DIM
    hd_index = tl.arange(0, BLOCK_HD)
    toks_index = tl.arange(0, BLOCK_TOK)
    for i in range(0, iters):
        chunk_id = pid + i * stride
        source_chunk = tl.load(filtered_chunk_indices_ptr + chunk_id)
        source_start = tl.load(cu_chunk_ptr + source_chunk)
        for hd_start in range(0, hd_total, BLOCK_HD):
            hd = hd_start + hd_index
            if EXACT_HD:
                mask_hd = hd_index >= 0
            else:
                mask_hd = hd < hd_total
            head = hd // HEAD_DIM
            d = hd - head * HEAD_DIM
            in_off = head * stride_head + d * stride_d
            acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
            for t_start in range(0, CHUNK_SIZE, BLOCK_TOK):
                toks = t_start + toks_index
                ptrs = (
                    k_ptr
                    + (source_start + toks)[:, None] * stride_token
                    + in_off[None, :]
                )
                if EXACT_HD and EXACT_T:
                    values = tl.load(ptrs).to(tl.float32)
                elif EXACT_HD:
                    values = tl.load(
                        ptrs, mask=(toks < CHUNK_SIZE)[:, None], other=0.0
                    ).to(tl.float32)
                elif EXACT_T:
                    values = tl.load(
                        ptrs, mask=mask_hd[None, :], other=0.0
                    ).to(tl.float32)
                else:
                    values = tl.load(
                        ptrs,
                        mask=(toks < CHUNK_SIZE)[:, None] & mask_hd[None, :],
                        other=0.0,
                    ).to(tl.float32)
                acc += tl.sum(values, axis=0)
            acc = acc / CHUNK_SIZE
            out_ptrs = (
                out_ptr
                + chunk_id * out_stride_chunk
                + head * out_stride_head
                + d * out_stride_d
            )
            if EXACT_HD:
                tl.store(out_ptrs, acc)
            else:
                tl.store(out_ptrs, acc, mask=mask_hd)


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
    BLOCK_HD: tl.constexpr,
    EXACT_HD: tl.constexpr,
    EXACT_T: tl.constexpr,
):
    """Dense-slot variant of ``_chunk_mean_kernel`` (one program per slot).

    Non-historical slots have an empty token mask and are written as zeros.
    """
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_chunk_slots - pid + stride - 1) // stride
    hd_total = num_kv_heads * HEAD_DIM
    hd_index = tl.arange(0, BLOCK_HD)
    toks_index = tl.arange(0, BLOCK_TOK)
    for i in range(0, iters):
        slot = pid + i * stride
        is_historical = tl.load(valid_historical_ptr + slot) != 0
        source_start = tl.load(cu_chunk_ptr + slot)
        for hd_start in range(0, hd_total, BLOCK_HD):
            hd = hd_start + hd_index
            if EXACT_HD:
                mask_hd = hd_index >= 0
            else:
                mask_hd = hd < hd_total
            head = hd // HEAD_DIM
            d = hd - head * HEAD_DIM
            in_off = head * stride_head + d * stride_d
            acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
            for t_start in range(0, CHUNK_SIZE, BLOCK_TOK):
                toks = t_start + toks_index
                mask_t = is_historical & (toks < CHUNK_SIZE)
                ptrs = (
                    k_ptr
                    + (source_start + toks)[:, None] * stride_token
                    + in_off[None, :]
                )
                if EXACT_HD:
                    values = tl.load(ptrs, mask=mask_t[:, None], other=0.0).to(tl.float32)
                else:
                    values = tl.load(
                        ptrs,
                        mask=mask_t[:, None] & mask_hd[None, :],
                        other=0.0,
                    ).to(tl.float32)
                acc += tl.sum(values, axis=0)
            acc = acc / CHUNK_SIZE
            out_ptrs = (
                out_ptr
                + slot * out_stride_chunk
                + head * out_stride_head
                + d * out_stride_d
            )
            if EXACT_HD:
                tl.store(out_ptrs, acc)
            else:
                tl.store(out_ptrs, acc, mask=mask_hd)


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
    fast = _chunk_means_gather_torch(
        k, cu_chunk[:num_chunk_slots], chunk_size, valid_historical
    )
    if fast is not None:
        return fast

    hd_total = num_kv_heads * head_dim
    block_hd = min(2048, triton.next_power_of_2(hd_total))
    block_tok = max(4, min(64, 8192 // block_hd))
    _dense_chunk_mean_kernel[_capped_grid(num_chunk_slots)](
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
        BLOCK_HD=block_hd,
        EXACT_HD=(hd_total % block_hd == 0),
        EXACT_T=(chunk_size % block_tok == 0),
        num_warps=8 if block_hd >= 1024 else 4,
        num_stages=2,
    )
    return chunk_means


MOBA_TOPK_MAX_PROGRAMS = 4096


@triton.jit(do_not_specialize=["num_chunks", "seqlen", "num_tiles",
                                  "i_tok_lo", "topk"])
def _chunk_topk_masked_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    selected_ptr,
    num_chunks,
    seqlen,
    num_tiles,
    i_tok_lo,
    topk,
    limits_ptr,
    limit_off,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_cm_c,
    stride_cm_h,
    stride_cm_d,
    stride_sel_r,
    stride_sel_h,
    stride_sel_t,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_LIMITS: tl.constexpr,
):
    """Exact chunk top-k with chunk-major reductions and stable index ties."""
    pid = tl.program_id(0)
    ph = tl.program_id(1)
    ot = i_tok_lo + pid * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tm = ot < seqlen
    od = tl.arange(0, BLOCK_D)
    md = od < HEAD_DIM
    qt = tl.load(
        q_ptr + od[:, None] * stride_q_d + ot[None, :] * stride_q_t
        + ph * stride_q_h,
        mask=md[:, None] & tm[None, :],
        other=0.0,
    )
    ranks = tl.arange(0, TOPK)
    cc = tl.arange(0, BLOCK_CHUNK)
    best_s = tl.full([TOPK, BLOCK_TOK], float("-inf"), tl.float32)
    best_i = tl.zeros([TOPK, BLOCK_TOK], tl.int32)
    neg_inf = float("-inf")
    if USE_LIMITS:
        tile_limit = tl.load(limits_ptr + (limit_off + pid) * 2)
        tile_start = tl.load(limits_ptr + (limit_off + pid) * 2 + 1) // BLOCK_CHUNK
        num_tiles_eff = (tile_limit + BLOCK_CHUNK - 1) // BLOCK_CHUNK
    else:
        num_tiles_eff = num_tiles
        tile_start = 0
    for tile in range(tile_start, num_tiles_eff):
        oc = tile * BLOCK_CHUNK + cc
        cml = oc < num_chunks
        # [BC, D] keeps the chunk axis first so the reductions below run on
        # axis=0.
        means_t = tl.load(
            chunk_means_ptr + oc[:, None] * stride_cm_c
            + od[None, :] * stride_cm_d
            + (ph // GROUP_SIZE) * stride_cm_h,
            mask=cml[:, None] & md[None, :],
            other=0.0,
        )
        gate = tl.dot(means_t, qt, out_dtype=tl.float32)  # [BC, BT]
        chunk_end = tl.load(chunk_end_ptr + oc, mask=cml, other=0)
        batch_end = tl.load(batch_end_ptr + oc, mask=cml, other=0)
        valid = (
            tm[None, :]
            & cml[:, None]
            & (ot[None, :] >= chunk_end[:, None])
            & (ot[None, :] < batch_end[:, None])
        )
        work = tl.where(valid, gate, neg_inf)

        tile_s = tl.full([TOPK, BLOCK_TOK], neg_inf, tl.float32)
        tile_i = tl.zeros([TOPK, BLOCK_TOK], tl.int32)
        for rank in tl.static_range(TOPK):
            row_max = tl.max(work, axis=0)
            is_max = work >= row_max[None, :]
            col_idx = tl.min(tl.where(is_max, cc[:, None], BLOCK_CHUNK), axis=0)
            tile_s = tl.where(ranks[:, None] == rank, row_max[None, :], tile_s)
            tile_i = tl.where(
                ranks[:, None] == rank,
                (col_idx + tile * BLOCK_CHUNK)[None, :],
                tile_i,
            )
            work = tl.where(cc[:, None] == col_idx[None, :], neg_inf, work)

        if not USE_LIMITS:
            # The host omits limits only when all chunks fit in one tile.
            best_s = tile_s
            best_i = tile_i
        else:
            merged_s = tl.full([TOPK, BLOCK_TOK], neg_inf, tl.float32)
            merged_i = tl.zeros([TOPK, BLOCK_TOK], tl.int32)
            for rank in tl.static_range(TOPK):
                best_max = tl.max(best_s, axis=0)
                tile_max = tl.max(tile_s, axis=0)
                take_best = best_max >= tile_max
                cur_s = tl.where(take_best, best_max, tile_max)
                best_pos = tl.min(
                    tl.where(best_s >= best_max[None, :], ranks[:, None], TOPK),
                    axis=0,
                )
                tile_pos = tl.min(
                    tl.where(tile_s >= tile_max[None, :], ranks[:, None], TOPK),
                    axis=0,
                )
                best_val = tl.sum(
                    tl.where(ranks[:, None] == best_pos[None, :], best_i, 0), axis=0
                )
                tile_val = tl.sum(
                    tl.where(ranks[:, None] == tile_pos[None, :], tile_i, 0), axis=0
                )
                merged_s = tl.where(ranks[:, None] == rank, cur_s[None, :], merged_s)
                merged_i = tl.where(
                    ranks[:, None] == rank,
                    tl.where(take_best, best_val, tile_val)[None, :],
                    merged_i,
                )
                best_s = tl.where(
                    (ranks[:, None] == best_pos[None, :]) & take_best[None, :],
                    neg_inf,
                    best_s,
                )
                tile_s = tl.where(
                    (ranks[:, None] == tile_pos[None, :]) & (~take_best)[None, :],
                    neg_inf,
                    tile_s,
                )
            best_s = merged_s
            best_i = merged_i

    tl.store(
        selected_ptr
        + ranks[:, None] * stride_sel_r
        + ph * stride_sel_h
        + ot[None, :] * stride_sel_t,
        tl.where(best_s > neg_inf, best_i, -1),
        mask=tm[None, :],
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

    hd_total = num_heads * head_dim
    block_hd = min(2048, triton.next_power_of_2(hd_total))
    starts = torch.index_select(cu_chunk, 0, filtered_chunk_indices.to(torch.int64))
    fast = _chunk_means_gather_torch(k, starts, chunk_size)
    if fast is not None:
        return fast

    block_tok = max(4, min(64, 8192 // block_hd))
    num_warps = 8 if block_hd >= 1024 else 4
    num_stages = 2

    grid = _capped_grid(num_chunks)

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
        BLOCK_HD=block_hd,
        EXACT_HD=(hd_total % block_hd == 0),
        EXACT_T=(chunk_size % block_tok == 0),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return chunk_means


# Larger rank counts need smaller tiles to keep the selection buffers in UB.
def _topk_tile_config(routed_topk: int) -> tuple[int, int]:
    if routed_topk <= 7:
        return 128, 128
    if routed_topk <= 16:
        return 128, 64
    return 64, 64


@triton.jit
def _topk_tile_limits_kernel(
    chunk_end_ptr,
    batch_end_ptr,
    limits_ptr,
    num_chunks,
    seqlen,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Conservative chunk range for each token block.

    Skip chunks ending after the last token and sequences ending at or before
    the first token. Blocks crossing a packed boundary retain both sequences.
    Each output pair is (exclusive chunk end, first possible chunk index).
    """
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (seqlen + BLOCK_T - 1) // BLOCK_T
    num_c_tiles = (num_chunks + BLOCK_C - 1) // BLOCK_C
    for i in range(0, (num_blocks - pid + stride - 1) // stride):
        blk = pid + i * stride
        tmax = tl.minimum((blk + 1) * BLOCK_T - 1, seqlen - 1)
        limit = 0
        first = num_chunks
        tmin = blk * BLOCK_T
        cc = tl.arange(0, BLOCK_C)
        for tc in range(0, num_c_tiles):
            cols = tc * BLOCK_C + cc
            mask = cols < num_chunks
            ce = tl.load(chunk_end_ptr + cols, mask=mask, other=2147483647)
            be = tl.load(batch_end_ptr + cols, mask=mask, other=0)
            first = tl.minimum(
                first, tl.min(tl.where(mask & (be > tmin), cols, num_chunks), axis=0)
            )
            ok = mask & (ce <= tmax)
            m = tl.max(tl.where(ok, cols + 1, 0), axis=0)
            limit = tl.maximum(limit, m)
        tl.store(limits_ptr + blk * 2, limit)
        tl.store(limits_ptr + blk * 2 + 1, first)


def _topk_tile_limits(
    chunk_end: torch.Tensor,
    batch_end: torch.Tensor,
    num_chunks: int,
    seqlen: int,
    num_tok_blocks: int,
    block_t: int,
):
    limits = torch.empty((num_tok_blocks * 2,), device=chunk_end.device, dtype=torch.int32)
    _topk_tile_limits_kernel[_capped_grid(num_tok_blocks)](
        chunk_end,
        batch_end,
        limits,
        num_chunks,
        seqlen,
        BLOCK_T=block_t,
        BLOCK_C=1024,
    )
    return limits


def _compute_selected_chunks_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    batch_end: torch.Tensor,
    routed_topk: int,
) -> torch.Tensor:
    """Select the top-``routed_topk`` historical chunks for every token/head."""
    seqlen, num_heads, head_dim = q.shape
    num_chunks = chunk_means.shape[0]
    group_size = num_heads // chunk_means.shape[1]
    selected = torch.full(
        (routed_topk, num_heads, seqlen),
        -1,
        device=q.device,
        dtype=torch.int32,
    )
    if num_chunks == 0 or routed_topk == 0 or seqlen == 0:
        return selected

    block_tok, block_chunk = _topk_tile_config(routed_topk)
    block_chunk = min(block_chunk, max(32, triton.next_power_of_2(num_chunks)))
    kernel = _chunk_topk_masked_kernel
    if topk_raw.is_supported(q, routed_topk) and num_chunks > topk_raw.BLOCK_CHUNKS:
        block_tok, block_chunk = topk_raw.BLOCK_TOKENS, topk_raw.BLOCK_CHUNKS
        kernel = topk_raw.chunk_topk_raw_kernel
    num_tiles = cdiv(num_chunks, block_chunk)
    block_d = triton.next_power_of_2(head_dim)
    tok_tiles = cdiv(seqlen, block_tok)
    # A single chunk tile needs no causal bound: skip both the pre-pass launch
    # and the per-program limit load (pass an arbitrary pointer, never read).
    use_limits = num_tiles > 1
    limits = (
        _topk_tile_limits(chunk_end, batch_end, num_chunks, seqlen, tok_tiles, block_tok)
        if use_limits
        else chunk_end
    )
    programs_per_launch = max(1, MOBA_TOPK_MAX_PROGRAMS // num_heads)
    for tok_lo in range(0, tok_tiles, programs_per_launch):
        n_tok = min(programs_per_launch, tok_tiles - tok_lo)
        kernel[(n_tok, num_heads)](
            q,
            chunk_means,
            chunk_end,
            batch_end,
            selected,
            num_chunks,
            seqlen,
            num_tiles,
            tok_lo * block_tok,
            routed_topk,
            limits,
            tok_lo,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            chunk_means.stride(0),
            chunk_means.stride(1),
            chunk_means.stride(2),
            selected.stride(0),
            selected.stride(1),
            selected.stride(2),
            HEAD_DIM=head_dim,
            GROUP_SIZE=group_size,
            TOPK=routed_topk,
            BLOCK_TOK=block_tok,
            BLOCK_CHUNK=block_chunk,
            BLOCK_D=block_d,
            USE_LIMITS=use_limits,
        )
    return selected


@triton.jit
def _dense_slot_batch_end_kernel(
    cu_seqlens_ptr,
    valid_historical_ptr,
    batch_end_ptr,
    num_slots,
    MAX_CHUNKS_PER_SEQUENCE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Build per-slot sequence ends for dense inference metadata.

    Historical slots receive their sequence end; other slots receive zero.
    The top-k mask uses this table to exclude slots from other sequences.
    """
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = slots < num_slots
    batch = slots // MAX_CHUNKS_PER_SEQUENCE
    sequence_end = tl.load(cu_seqlens_ptr + batch + 1, mask=mask, other=0)
    valid = tl.load(valid_historical_ptr + slots, mask=mask, other=0) != 0
    tl.store(
        batch_end_ptr + slots,
        tl.where(valid, sequence_end, 0),
        mask=mask,
    )


def _dense_slot_batch_end(
    cu_seqlens: torch.Tensor,
    valid_historical: torch.Tensor,
    max_chunks_per_sequence: int,
) -> torch.Tensor:
    """Build the dense-slot ``batch_end`` table on device (one small launch)."""
    batch_end = torch.empty_like(valid_historical).to(torch.int32)
    block = 256
    _dense_slot_batch_end_kernel[(cdiv(valid_historical.numel(), block),)](
        cu_seqlens,
        valid_historical,
        batch_end,
        valid_historical.numel(),
        MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
        BLOCK=block,
    )
    return batch_end


def _compute_selected_chunks_batched_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    max_chunks_per_sequence: int,
    topk: int,
) -> torch.Tensor:
    """Select dense historical slots through the shared chunk top-k path.

    Non-historical slots have ``batch_end == 0``. Results are int32 indices in
    ``[topk, num_heads, total_tokens]`` order, with -1 for missing entries.
    """
    del max_seqlen  # dense-slot path does not need the per-sequence maximum
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
    if topk > 64:
        raise ValueError(f"streaming Triton Top-K currently supports topk <= 64, got topk={topk}")
    valid_historical = (chunk_end < 2147483647).to(torch.int8)
    batch_end = _dense_slot_batch_end(
        cu_seqlens, valid_historical, max_chunks_per_sequence
    )
    return _compute_selected_chunks_streaming_triton(
        q, chunk_means, chunk_end, batch_end, topk
    )


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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (num_slots + BLOCK - 1) // BLOCK
    iters = (num_blocks - pid + stride - 1) // stride
    for i in range(0, iters):
        slots = (pid + i * stride) * BLOCK + tl.arange(0, BLOCK)
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


@triton.jit(do_not_specialize=["num_experts", "seqlen", "n_blocks"])
def _routes_counts_kernel(
    selected_ptr,
    counts_eb_ptr,
    chunk_end_ptr,
    num_experts,
    seqlen,
    n_blocks,
    num_heads: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_CHUNK_END: tl.constexpr,
):
    """Per (expert, token-block) route counts; each program owns its own cell."""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    total = num_experts * n_blocks
    slots_per_rank = num_heads * seqlen
    iters = (total - pid + stride - 1) // stride
    for i in range(0, iters):
        idx = pid + i * stride
        expert = idx // n_blocks
        t_block = idx - expert * n_blocks
        head = expert % num_heads
        chunk = expert // num_heads
        toks = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = toks < seqlen
        count = tl.zeros([BLOCK_T], dtype=tl.int32)
        if (not USE_CHUNK_END) or ((t_block + 1) * BLOCK_T > tl.load(chunk_end_ptr + chunk)):
            for r in tl.static_range(TOPK):
                base = r * slots_per_rank + head * seqlen
                vals = tl.load(selected_ptr + base + toks, mask=mask, other=-1)
                count += tl.where(mask & (vals == chunk), 1, 0)
        tl.store(counts_eb_ptr + idx, tl.sum(count, axis=0))


@triton.jit(do_not_specialize=["num_experts", "n_blocks", "num_heads_off"])
def _routes_bases_kernel(
    counts_eb_ptr,
    offsets_ptr,
    bases_eb_ptr,
    num_experts,
    n_blocks,
):
    """Exclusive per-block base inside each expert's compact range."""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_experts - pid + stride - 1) // stride
    for i in range(0, iters):
        expert = pid + i * stride
        running = tl.load(offsets_ptr + expert)
        for b in range(0, n_blocks):
            idx = expert * n_blocks + b
            cnt = tl.load(counts_eb_ptr + idx)
            tl.store(bases_eb_ptr + idx, running)
            running += cnt


@triton.jit(do_not_specialize=["num_experts", "seqlen", "n_blocks"])
def _routes_scatter_kernel(
    selected_ptr,
    bases_eb_ptr,
    query_indices_ptr,
    route_positions_ptr,
    chunk_end_ptr,
    num_experts,
    seqlen,
    n_blocks,
    num_heads: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_CHUNK_END: tl.constexpr,
):
    """Write each routed slot into its expert range without atomics."""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    total = num_experts * n_blocks
    slots_per_rank = num_heads * seqlen
    iters = (total - pid + stride - 1) // stride
    for i in range(0, iters):
        idx = pid + i * stride
        expert = idx // n_blocks
        t_block = idx - expert * n_blocks
        head = expert % num_heads
        chunk = expert // num_heads
        toks = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        mask = toks < seqlen
        chunk_end = tl.load(chunk_end_ptr + chunk) if USE_CHUNK_END else 0
        if (not USE_CHUNK_END) or ((t_block + 1) * BLOCK_T > chunk_end):
            hits = tl.zeros([BLOCK_T], dtype=tl.int32)
            rank_of_hit = tl.zeros([BLOCK_T], dtype=tl.int32)
            for r in tl.static_range(TOPK):
                base = r * slots_per_rank + head * seqlen
                vals = tl.load(selected_ptr + base + toks, mask=mask, other=-1)
                hit = mask & (vals == chunk) & (hits == 0)
                hits += hit.to(tl.int32)
                rank_of_hit += tl.where(hit, r, 0)
            writer = tl.load(bases_eb_ptr + idx)
            positions = writer + tl.cumsum(hits, axis=0) - 1
            keep = hits > 0
            tl.store(query_indices_ptr + positions, toks * num_heads + head, mask=keep)
            route_slot = rank_of_hit * slots_per_rank + toks * num_heads + head
            tl.store(route_positions_ptr + route_slot, positions, mask=keep)


def _compact_routes_fast(selected_chunks, num_experts, num_heads, seqlen, chunk_ends=None):
    """Return (query_indices, route_positions, expert_counts, expert_offsets).

    ``chunk_ends[k]`` is the exclusive token end of chunk ``k``. Only tokens
    at or after that end can select the chunk; earlier token blocks are skipped.
    """
    topk = selected_chunks.shape[0]
    if topk == 1:
        # Use the reference path to avoid incomplete single-rank position stores
        # in the Ascend scatter kernel.
        return _compact_routes_torch(
            selected_chunks, num_experts, num_heads, seqlen
        )
    device = selected_chunks.device
    n_blocks = triton.cdiv(seqlen, _ROUTE_BLOCK_T)
    total = num_experts * n_blocks
    grid = _capped_grid(total)
    counts_eb = torch.empty((total,), device=device, dtype=torch.int32)
    bases_eb = torch.empty((total,), device=device, dtype=torch.int32)
    expert_counts = torch.empty((num_experts,), device=device, dtype=torch.int32)
    expert_offsets = torch.empty((num_experts + 1,), device=device, dtype=torch.int32)
    query_indices = torch.full((selected_chunks.numel(),), -1, device=device, dtype=torch.int32)
    route_positions = torch.full((topk, seqlen * num_heads), -1, device=device, dtype=torch.int32)

    use_chunk_end = chunk_ends is not None
    chunk_end_ptr = chunk_ends if use_chunk_end else expert_counts
    _routes_counts_kernel[grid](
        selected_chunks, counts_eb, chunk_end_ptr, num_experts, seqlen, n_blocks,
        num_heads=num_heads, TOPK=topk, BLOCK_T=_ROUTE_BLOCK_T,
        USE_CHUNK_END=use_chunk_end,
    )
    expert_counts.copy_(counts_eb.view(num_experts, n_blocks).sum(dim=1, dtype=torch.int32))
    torch.cumsum(expert_counts, 0, dtype=torch.int32, out=expert_offsets[1:])
    expert_offsets[0] = 0
    _routes_bases_kernel[_capped_grid(num_experts)](
        counts_eb, expert_offsets, bases_eb, num_experts, n_blocks,
    )
    _routes_scatter_kernel[grid](
        selected_chunks, bases_eb, query_indices, route_positions,
        chunk_end_ptr, num_experts, seqlen, n_blocks,
        num_heads=num_heads, TOPK=topk, BLOCK_T=_ROUTE_BLOCK_T,
        USE_CHUNK_END=use_chunk_end,
    )
    return query_indices, route_positions, expert_counts, expert_offsets


def _compact_routes_torch(
    selected_chunks: torch.Tensor,
    num_experts: int,
    num_heads: int,
    seqlen: int,
):
    """Deterministically compact route slots by expert using a stable slot key.

    Return query rows, route positions, per-expert counts, and prefix offsets.
    """
    device = selected_chunks.device
    topk, heads_dim, tokens_dim = selected_chunks.shape
    num_slots = topk * heads_dim * tokens_dim
    chunks = selected_chunks.reshape(-1)
    slots = torch.arange(num_slots, device=device, dtype=torch.int64)
    slots_per_rank = num_heads * seqlen
    ranks = torch.div(slots, slots_per_rank, rounding_mode="floor")
    within = slots - ranks * slots_per_rank
    heads = torch.div(within, seqlen, rounding_mode="floor")
    tokens = within - heads * seqlen
    valid = chunks >= 0
    experts = chunks.to(torch.int64) * num_heads + heads
    sentinel = num_experts * (num_slots + 1)
    key = torch.where(valid, experts * num_slots + slots, sentinel)
    order = torch.argsort(key)
    sorted_positions = torch.arange(num_slots, device=device, dtype=torch.int64)
    slot_positions = torch.full((num_slots,), -1, device=device, dtype=torch.int64)
    slot_positions.index_copy_(0, order, sorted_positions)
    n_valid = torch.sum(valid)
    positions = torch.where(
        slot_positions < n_valid, slot_positions,
        torch.full_like(slot_positions, -1),
    )
    flat_positions = positions >= 0
    query_ids = tokens * num_heads + heads
    query_indices = torch.full((num_slots,), -1, device=device, dtype=torch.int32)
    query_indices.index_copy_(
        0, positions[flat_positions], query_ids[flat_positions].to(torch.int32)
    )
    route_positions = torch.full(
        (topk, seqlen * num_heads), -1, device=device, dtype=torch.int32
    )
    route_slots = ranks * slots_per_rank + tokens * num_heads + heads
    route_positions.view(-1).index_copy_(
        0, route_slots[flat_positions], positions[flat_positions].to(torch.int32)
    )
    expert_counts = torch.zeros((num_experts,), device=device, dtype=torch.int64)
    expert_counts.index_add_(0, experts[valid], torch.ones_like(experts[valid]))
    expert_offsets = torch.zeros((num_experts + 1,), device=device, dtype=torch.int32)
    torch.cumsum(expert_counts, 0, dtype=torch.int32, out=expert_offsets[1:])
    return (
        query_indices,
        route_positions,
        expert_counts.to(torch.int32),
        expert_offsets,
    )


# ---------------------------------------------------------------------------
# Linear-time route compaction
# ---------------------------------------------------------------------------
# Group route slots into (token tile, head, chunk) cells. Histograms and
# prefix sums give each cell a contiguous compact run; the inverse pass maps
# those runs to packed query rows and records each route slot's position.
_ROUTE_LINEAR = os.environ.get("MOBA_ROUTES_LINEAR", "1") != "0"
_ROUTE_RAW = os.environ.get("MOBA_ROUTES_RAW", "1") != "0"
_ROUTE_TILE_TOKENS = 1024
_ROUTE_TILE_TOKENS_MAX = 4096
_ROUTE_INVERSE_BLOCK = 32
# Cells expanded per program by the inverse pass.
_ROUTE_INVERSE_CELLS = 32
_ROUTE_LINEAR_MIN_SEQLEN = 0


def _route_tile_size(seqlen: int, num_chunks: int, topk: int) -> int:
    """Choose a token tile targeting 32 route slots per cell, within the tile limits."""
    target = max(1, 32 * num_chunks // max(topk, 1))
    tile = _ROUTE_TILE_TOKENS
    while tile < target and tile < _ROUTE_TILE_TOKENS_MAX and tile * 2 <= seqlen:
        tile *= 2
    return min(tile, max(seqlen, 1))


@triton.jit(do_not_specialize=["num_tokens", "num_chunks", "tiles"])
def _route_counts_cell_kernel(
    selected_ptr,
    counts_ptr,
    num_tokens,
    num_chunks,
    tiles,
    num_heads: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """`counts[(b * num_heads + h) * num_chunks + c]` for one (tile, head)."""
    pid = tl.program_id(0)
    head = pid % num_heads
    tile = pid // num_heads
    cell = pid * num_chunks
    slots_per_rank = num_heads * num_tokens
    first = tile * TILE
    for rank in tl.static_range(TOPK):
        base = rank * slots_per_rank + head * num_tokens + first
        for k in range(0, tl.cdiv(TILE, BLOCK)):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            mask = (first + offs) < num_tokens
            chunk = tl.load(selected_ptr + base + offs, mask=mask, other=-1)
            tl.atomic_add(
                counts_ptr + cell + chunk, 1, mask=mask & (chunk >= 0), sem="relaxed"
            )


def _route_cell_counts(selected_chunks, tile, num_chunks):
    """Fill `counts[b, h, c]` (int32) for every (tile, head, chunk) cell."""
    topk, num_heads, seqlen = selected_chunks.shape
    n_blocks = cdiv(seqlen, tile)
    device = selected_chunks.device
    counts = torch.zeros(
        (n_blocks * num_heads * num_chunks,), device=device, dtype=torch.int32
    )
    block = 256
    _route_counts_cell_kernel[_capped_grid(n_blocks * num_heads)](
        selected_chunks,
        counts,
        seqlen,
        num_chunks,
        n_blocks,
        num_heads=num_heads,
        TOPK=topk,
        TILE=tile,
        BLOCK=block,
        num_warps=4,
    )
    return counts.view(n_blocks, num_heads, num_chunks)


@triton.jit(do_not_specialize=["num_groups", "num_cells", "num_chunks", "tile_slots"])
def _route_inverse_kernel(
    counts_ptr,
    base_ptr,
    start_ptr,
    out_ids_ptr,
    query_ptr,
    num_groups,
    num_cells,
    num_chunks,
    tile_slots,
    BLOCK: tl.constexpr,
    CELLS: tl.constexpr,
):
    """Copy each cell's packed query rows to its compact output run.

    Each cell uses uniform source and destination bases with contiguous offsets.
    The cell loop is statically unrolled.
    """
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    off = tl.arange(0, BLOCK)
    iters = (num_groups - pid + stride - 1) // stride
    for g in range(0, iters):
        first = (pid + g * stride) * CELLS
        for i in tl.static_range(0, CELLS):
            cell = first + i
            if cell < num_cells:
                count = tl.load(counts_ptr + cell)
                cell_base = tl.load(base_ptr + cell)
                run_start = tl.load(start_ptr + cell)
                run = out_ids_ptr + (cell // num_chunks) * tile_slots + run_start
                mask = off < count
                row = tl.load(run + off, mask=mask, other=0)
                tl.store(query_ptr + cell_base + off, row, mask=mask)
                extra = count - BLOCK
                if extra > 0:
                    for k in range(1, (count + BLOCK - 1) // BLOCK):
                        offs = k * BLOCK + off
                        mask_k = offs < count
                        row_k = tl.load(run + offs, mask=mask_k, other=0)
                        tl.store(query_ptr + cell_base + offs, row_k, mask=mask_k)


def _route_inverse_expand(counts3, base3, start3, out_ids):
    """Compact position -> packed query row, from the per-cell runs."""
    n_blocks, num_heads, num_chunks = counts3.shape
    tile_slots = out_ids.shape[1]
    device = counts3.device
    query_indices = torch.empty(
        (out_ids.shape[0] * tile_slots,), device=device, dtype=torch.int32
    )
    num_cells = counts3.numel()
    cells = _ROUTE_INVERSE_CELLS
    num_groups = cdiv(num_cells, cells)
    _route_inverse_kernel[_capped_grid(num_groups)](
        counts3.reshape(-1),
        base3.reshape(-1),
        start3.reshape(-1),
        out_ids,
        query_indices,
        num_groups,
        num_cells,
        num_chunks,
        tile_slots,
        BLOCK=_ROUTE_INVERSE_BLOCK,
        CELLS=cells,
        num_warps=4,
    )
    return query_indices


def _route_bucket_tiles_torch(selected_chunks, start3, base3, tile):
    """Reference bucket pass using the same slot ordering as the Raw kernel."""
    topk, num_heads, seqlen = selected_chunks.shape
    n_blocks, _, num_chunks = start3.shape
    device = selected_chunks.device
    tile_slots = topk * tile
    rows = torch.arange(tile, device=device, dtype=torch.int64)
    ranks = torch.arange(topk, device=device, dtype=torch.int64)
    keys = torch.arange(tile_slots, device=device, dtype=torch.int64)
    sentinel = (num_chunks + 1) * tile_slots
    out_ids = torch.full(
        (n_blocks * num_heads, tile_slots), -1, device=device, dtype=torch.int32
    )
    positions = torch.full(
        (topk, num_heads * seqlen), -1, device=device, dtype=torch.int32
    )
    flat_positions = positions.view(-1)
    for b in range(n_blocks):
        first = b * tile
        width = min(tile, seqlen - first)
        for h in range(num_heads):
            chunk_ids = selected_chunks[:, h, first:first + width].to(torch.int64)
            if width < tile:
                pad = torch.full((topk, tile - width), -1, device=device,
                                 dtype=torch.int64)
                chunk_ids = torch.cat((chunk_ids, pad), dim=1)
            slots = (
                ranks[:, None] * (num_heads * seqlen)
                + (first + rows)[None, :] * num_heads + h
            )
            slots = torch.where(
                rows[None, :] < width, slots, ranks[:, None] * (num_heads * seqlen)
            )
            flat_ids = chunk_ids.reshape(-1)
            valid = flat_ids >= 0
            key = torch.where(valid, flat_ids * tile_slots + keys, sentinel)
            order = torch.argsort(key)
            keep = valid[order]
            if not bool(keep.any().item()):
                continue
            order = order[keep]
            chunk_of = flat_ids[order]
            slot_of = slots.reshape(-1)[order]
            run_start = start3[b, h].to(torch.int64)[chunk_of]
            inside = torch.arange(order.numel(), device=device) - run_start
            out_ids[b * num_heads + h].index_copy_(
                0, run_start + inside, (slot_of % (num_heads * seqlen)).to(torch.int32)
            )
            flat_positions.index_copy_(
                0, slot_of, (base3[b, h].to(torch.int64)[chunk_of] + inside).to(torch.int32)
            )
    return out_ids, positions


def _route_expert_layout(counts3):
    """Expert-major offsets and per-cell compact bases from cell counts."""
    n_blocks, num_heads, num_chunks = counts3.shape
    device = counts3.device
    # Tile-local exclusive prefix over chunks: the run start inside a tile.
    start3 = torch.cumsum(counts3, -1, dtype=torch.int32) - counts3
    # Convert head-major counts to chunk-major expert ids: chunk * num_heads + head.
    expert_counts = (
        counts3.sum(0).transpose(0, 1).reshape(-1).contiguous()
    )
    expert_offsets = torch.empty(
        (expert_counts.numel() + 1,), device=device, dtype=torch.int32
    )
    expert_offsets[0] = 0
    torch.cumsum(expert_counts, 0, dtype=torch.int32, out=expert_offsets[1:])
    offsets_hc = (
        expert_offsets[:-1]
        .view(num_chunks, num_heads)
        .transpose(0, 1)
        .reshape(-1)
        .contiguous()
    )
    flat_counts = counts3.reshape(n_blocks, num_heads * num_chunks)
    base3 = (
        torch.cumsum(flat_counts, 0, dtype=torch.int32)
        - flat_counts
        + offsets_hc[None, :]
    )
    return start3, base3.view(n_blocks, num_heads, num_chunks), expert_counts, expert_offsets


def _compact_routes_linear(
    selected_chunks: torch.Tensor,
    num_experts: int,
    num_heads: int,
    seqlen: int,
    chunk_ends: torch.Tensor | None = None,
):
    """O(topk * num_heads * seqlen) route compaction (see the banner above)."""
    del chunk_ends  # cells are visited once, so the causal pruning is moot
    topk, heads_dim, num_tokens = selected_chunks.shape
    if heads_dim != num_heads:
        raise ValueError("selected_chunks head dimension does not match num_heads")
    num_chunks = num_experts // num_heads
    device = selected_chunks.device
    if selected_chunks.dtype != torch.int32:
        selected_chunks = selected_chunks.to(torch.int32)
    selected_chunks = selected_chunks.contiguous()
    use_raw = _ROUTE_RAW

    counts3 = None
    if use_raw:
        counts3 = routes_raw.route_cell_hist(selected_chunks, num_chunks)
    if counts3 is None:
        tile = _route_tile_size(num_tokens, num_chunks, topk)
        counts3 = _route_cell_counts(selected_chunks, tile, num_chunks)
    start3, base3, expert_counts, expert_offsets = _route_expert_layout(counts3)

    bucketed = None
    if use_raw:
        bucketed = routes_raw.route_cell_bucket(selected_chunks, start3, base3)
    if bucketed is None:
        tile = num_tokens // counts3.shape[0]
        bucketed = _route_bucket_tiles_torch(selected_chunks, start3, base3, tile)
    out_rows, positions = bucketed
    query_indices = _route_inverse_expand(counts3, base3, start3, out_rows)
    return query_indices, positions, expert_counts, expert_offsets


def _compact_routes(
    selected_chunks: torch.Tensor,
    num_experts: int,
    num_heads: int,
    seqlen: int,
    chunk_ends: torch.Tensor | None = None,
):
    """Route compaction entry point (linear builder by default)."""
    if _ROUTE_LINEAR:
        return _compact_routes_linear(
            selected_chunks, num_experts, num_heads, seqlen, chunk_ends
        )
    return _compact_routes_fast(
        selected_chunks, num_experts, num_heads, seqlen, chunk_ends
    )


@triton.jit
def _descriptor_counts_kernel(
    expert_counts_ptr,
    descriptor_counts_ptr,
    expert_offsets_ptr,
    num_experts,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """ceil(count_e / BLOCK_M) descriptors per expert (for the descriptor scan)."""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (num_experts + BLOCK - 1) // BLOCK
    iters = (num_blocks - pid + stride - 1) // stride
    for i in range(0, iters):
        experts = (pid + i * stride) * BLOCK + tl.arange(0, BLOCK)
        mask = experts < num_experts
        counts = tl.load(expert_counts_ptr + experts, mask=mask, other=0)
        tl.store(
            descriptor_counts_ptr + experts,
            (counts + BLOCK_M - 1) // BLOCK_M,
            mask=mask,
        )


@triton.jit
def _descriptor_scatter_kernel(
    expert_counts_ptr,
    expert_offsets_ptr,
    descriptor_offsets_ptr,
    descriptor_experts_ptr,
    descriptor_starts_ptr,
    num_experts,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Expert-major descriptor build: no atomics, contiguous stores per expert."""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    num_blocks = (num_experts + BLOCK - 1) // BLOCK
    iters = (num_blocks - pid + stride - 1) // stride
    for i in range(0, iters):
        experts = (pid + i * stride) * BLOCK + tl.arange(0, BLOCK)
        mask = experts < num_experts
        counts = tl.load(expert_counts_ptr + experts, mask=mask, other=0)
        starts = tl.load(expert_offsets_ptr + experts, mask=mask, other=0)
        desc_offsets = tl.load(descriptor_offsets_ptr + experts, mask=mask, other=0)
        n_desc = (counts + BLOCK_M - 1) // BLOCK_M
        max_n = tl.max(n_desc, axis=0)
        for j in range(0, max_n):
            active = mask & (j < n_desc)
            tl.store(descriptor_experts_ptr + desc_offsets + j, experts, mask=active)
            tl.store(
                descriptor_starts_ptr + desc_offsets + j,
                starts + j * BLOCK_M,
                mask=active,
            )


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
    chunk_ends: torch.Tensor | None = None,
):
    """Group selected queries by (target chunk, head)."""
    num_experts = num_chunks * num_heads
    (
        query_indices,
        route_positions,
        expert_counts,
        expert_offsets,
    ) = _compact_routes(selected_chunks, num_experts, num_heads, seqlen, chunk_ends)
    max_count = int(expert_counts.max().item())
    num_routes = int(expert_offsets[-1].item())
    return (
        query_indices[:num_routes],
        route_positions,
        expert_counts,
        expert_offsets,
        max_count,
    )


def _build_sparse_routes_device_triton(
    selected_chunks: torch.Tensor,
    num_chunks: int,
    num_heads: int,
    seqlen: int,
    block_m: int,
    chunk_ends: torch.Tensor | None = None,
):
    """Device-side route metadata (descriptors are expert-major, no atomics)."""
    num_experts = num_chunks * num_heads
    num_slots = selected_chunks.numel()
    device = selected_chunks.device
    (
        query_indices,
        route_positions,
        expert_counts,
        expert_offsets,
    ) = _compact_routes(selected_chunks, num_experts, num_heads, seqlen, chunk_ends)
    block = 256
    descriptor_counts = torch.empty((num_experts,), device=device, dtype=torch.int32)
    descriptor_offsets = torch.empty((num_experts + 1,), device=device, dtype=torch.int32)
    _descriptor_counts_kernel[_capped_grid(cdiv(num_experts, block))](
        expert_counts,
        descriptor_counts,
        expert_offsets,
        num_experts,
        BLOCK_M=block_m,
        BLOCK=block,
    )
    _exclusive_prefix_sum_triton(descriptor_counts, descriptor_offsets)
    max_descriptors = cdiv(num_slots, block_m) + num_experts
    descriptor_experts = torch.empty((max_descriptors,), device=device, dtype=torch.int32)
    descriptor_starts = torch.empty_like(descriptor_experts)
    _fill_triton(descriptor_experts, -1)
    _fill_triton(descriptor_starts, -1)
    _descriptor_scatter_kernel[_capped_grid(cdiv(num_experts, block))](
        expert_counts,
        expert_offsets,
        descriptor_offsets,
        descriptor_experts,
        descriptor_starts,
        num_experts,
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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_rows - pid + stride - 1) // stride
    for i in range(0, iters):
        row = pid + i * stride
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
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_rows - pid + stride - 1) // stride
    for i in range(0, iters):
        row = pid + i * stride
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
        _gather_moba_kv_kernel[_capped_grid(num_rows)](
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
        _scatter_moba_kv_grads_kernel[_capped_grid(num_rows)](
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
    """Gather dout, output, and LSE rows for the sparse MoBA branch."""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    iters = (num_indices - pid + stride - 1) // stride
    for i in range(0, iters):
        cur = pid + i * stride
        sh_idx = tl.load(indices_ptr + cur)
        token = sh_idx // num_heads
        head = sh_idx - token * num_heads
        lse_val = tl.load(lse_ptr + head * stride_lse_head + token * stride_lse_token)
        tl.store(gathered_lse_ptr + cur, lse_val)

        offs_d = tl.arange(0, BLOCK_D)

        for d_start in range(0, head_dim, BLOCK_D):
            mask_d = (d_start + offs_d) < head_dim

            dout_ptrs = dout_ptr + token * stride_dout_token + head * stride_dout_head + (d_start + offs_d) * stride_dout_d
            dout_vals = tl.load(dout_ptrs, mask=mask_d, other=0.0)
            gdout_ptrs = gathered_dout_ptr + cur * stride_gdout_row + (d_start + offs_d) * stride_gdout_d
            tl.store(gdout_ptrs, dout_vals, mask=mask_d)

            out_ptrs = out_ptr + token * stride_out_token + head * stride_out_head + (d_start + offs_d) * stride_out_d
            out_vals = tl.load(out_ptrs, mask=mask_d, other=0.0)
            gout_ptrs = gathered_out_ptr + cur * stride_gout_row + (d_start + offs_d) * stride_gout_d
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

    grid = _capped_grid(num_indices)

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
