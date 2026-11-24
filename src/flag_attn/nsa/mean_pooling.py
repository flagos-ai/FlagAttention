# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices
from .triton_ops_helper import autotune_cache_kwargs
from .utils import input_guard


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def mean_pooling_fwd_kernel(
    x,
    o,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    p_x = tl.make_block_ptr(
        x + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (i_tg * H + i_h) * D, (D,), (1,), (i_d * BD,), (BD,), (0,)
    )
    # [BT, BD]
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    # [BD]
    b_o = tl.sum(b_x, axis=0) / min(BT, T - i_t * BT)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def mean_pooling_bwd_kernel(
    do,
    dx,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    p_dx = tl.make_block_ptr(
        dx + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_do = tl.make_block_ptr(
        do + (i_tg * H + i_h) * D, (D,), (1,), (i_d * BD,), (BD,), (0,)
    )
    # [BD]
    b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
    # [BT, BD]
    b_dx = b_do / tl.full((BT,), min(BT, T - i_t * BT), dtype=tl.float32)[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def mean_pooling_fwd(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, D = x.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = x.new_empty(B, NT, H, D)

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B * H)

    mean_pooling_fwd_kernel[grid](
        x,
        o,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
    )
    return o


def mean_pooling_bwd(
    do: torch.Tensor,
    batch_size: int,
    seq_len: int,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, D = batch_size, seq_len, *do.shape[-2:]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dx = do.new_empty(B, T, H, D)

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, B * H)

    mean_pooling_bwd_kernel[grid](
        do,
        dx,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        D=D,
        BT=BT,
    )
    return dx


class MeanPoolingFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, chunk_size, cu_seqlens=None):
        o = mean_pooling_fwd(x, chunk_size, cu_seqlens)
        ctx.batch_size = x.shape[0]
        ctx.seq_len = x.shape[1]
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        return o

    @staticmethod
    @input_guard
    def backward(ctx, do):
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        dx = mean_pooling_bwd(do, batch_size, seq_len, chunk_size, cu_seqlens)
        return dx, None, None


def mean_pooling(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Average-pool ``x`` along the sequence dimension with the given ``chunk_size``.

    Args:
        x: Input tensor of shape ``[B, T, H, D]``.
        chunk_size: Size of each pooling chunk.
        cu_seqlens: Optional cumulative sequence lengths for variable-length sequences.

    Returns:
        Pooled tensor of shape ``[B, NT, H, D]`` where ``NT = ceil(T / chunk_size)``.
    """
    return MeanPoolingFunction.apply(x, chunk_size, cu_seqlens)
