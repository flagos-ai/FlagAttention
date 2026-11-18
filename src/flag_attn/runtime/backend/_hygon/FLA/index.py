# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch
import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    # The FlagGems GLA wrapper forwards this optional host mirror for API
    # compatibility; index construction only needs the device tensor.
    del cu_seqlens_cpu
    chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    chunk_offsets = torch.cat([cu_seqlens.new_tensor([0]), chunk_counts]).cumsum(-1)
    chunk_arange = torch.arange(chunk_offsets[-1], device=cu_seqlens.device)
    seq_ids = torch.repeat_interleave(
        torch.arange(chunk_counts.numel(), device=cu_seqlens.device),
        chunk_counts,
    )
    chunk_ids = chunk_arange - torch.repeat_interleave(chunk_offsets[:-1], chunk_counts)
    return torch.stack([seq_ids, chunk_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(prepare_lens(cu_seqlens), chunk_size),
        ]
    ).cumsum(-1)


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    """Return a 2-D tensor ``[total_tokens, 2]`` with ``[seq_id, intra_seq_pos]``."""
    lens = prepare_lens(cu_seqlens)
    total = lens.sum().item()
    seq_ids = torch.arange(
        lens.numel(), device=cu_seqlens.device, dtype=torch.long
    ).repeat_interleave(lens)
    offsets = torch.zeros(lens.numel(), device=cu_seqlens.device, dtype=torch.long)
    offsets[1:] = lens.cumsum(0)[:-1]
    intra = (
        torch.arange(total, device=cu_seqlens.device, dtype=torch.long)
        - offsets[seq_ids]
    )
    return torch.stack([seq_ids, intra], 1)
