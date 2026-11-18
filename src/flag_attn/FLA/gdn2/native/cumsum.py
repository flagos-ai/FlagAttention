# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Chunk-local vector cumsum used by the native GDN2 fallback."""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BS": block_size}, num_warps=num_warps)
        for block_size in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    offsets = tl.arange(0, BT)
    if REVERSE:
        mask = tl.where(offsets[:, None] <= offsets[None, :], 1.0, 0.0)
    else:
        mask = tl.where(offsets[:, None] >= offsets[None, :], 1.0, 0.0)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h * T) * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_o = tl.make_block_ptr(
            o + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    values = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    result = tl.dot(mask, values, allow_tf32=False)
    tl.store(p_o, result.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    """Compute cumsum independently inside each chunk of a 4-D gate tensor."""
    del kwargs
    if g.ndim != 4:
        raise ValueError(f"GDN2 gate must be 4-D, got shape {tuple(g.shape)}")
    if cu_seqlens is not None and g.shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1")
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a positive power of 2")

    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    num_chunks = triton.cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    source = g
    output = torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (triton.cdiv(S, meta["BS"]), num_chunks, B * H)

    chunk_local_cumsum_vector_kernel[grid](
        source,
        output,
        cu_seqlens,
        chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return output
