# Enflame FLA shared Triton index helpers.
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license.

import torch
import triton

from flag_attn.parallel_nsa.index import (
    prepare_chunk_indices as _generic_prepare_chunk_indices,
)
from flag_attn.parallel_nsa.index import (
    prepare_chunk_offsets as _generic_prepare_chunk_offsets,
)
from flag_attn.parallel_nsa.index import prepare_lens as _generic_prepare_lens
from flag_attn.parallel_nsa.index import (
    prepare_token_indices as _generic_prepare_token_indices,
)

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    chunk_offsets = torch.cat([cu_seqlens.new_tensor([0]), chunk_counts]).cumsum(
        -1,
        dtype=cu_seqlens.dtype,
    )
    chunk_arange = torch.arange(
        chunk_offsets[-1], device=cu_seqlens.device, dtype=cu_seqlens.dtype
    )
    seq_ids = torch.repeat_interleave(
        torch.arange(
            chunk_counts.numel(),
            device=cu_seqlens.device,
            dtype=cu_seqlens.dtype,
        ),
        chunk_counts,
    )
    chunk_ids = chunk_arange - torch.repeat_interleave(
        chunk_offsets[:-1], chunk_counts
    )
    return torch.stack([seq_ids, chunk_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(prepare_lens(cu_seqlens), chunk_size),
        ]
    ).cumsum(-1, dtype=cu_seqlens.dtype)


def _to_int32(value):
    if isinstance(value, torch.Tensor) and value.dtype != torch.int32:
        return value.to(dtype=torch.int32)
    return value


def prepare_lens_enflame(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return _to_int32(_generic_prepare_lens(_to_int32(cu_seqlens)))


def prepare_token_indices_enflame(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return _to_int32(_generic_prepare_token_indices(_to_int32(cu_seqlens)))


def prepare_chunk_offsets_enflame(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _to_int32(
        _generic_prepare_chunk_offsets(_to_int32(cu_seqlens), chunk_size)
    )


def prepare_chunk_indices_enflame(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return _to_int32(
        _generic_prepare_chunk_indices(_to_int32(cu_seqlens), chunk_size)
    )


__all__ = [
    "prepare_lens",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_lens_enflame",
    "prepare_token_indices_enflame",
    "prepare_chunk_offsets_enflame",
    "prepare_chunk_indices_enflame",
]
