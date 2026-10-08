# Enflame FLA shared chunk-cumsum implementation.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import warnings

import torch
import triton
import triton.language as tl

from .compat import libentry, libtuner
from .index import prepare_chunk_indices
from .utils import input_guard, make_grid_3d, requires_grid_remap

CUMSUM_MIN_BS = 32
CUMSUM_MAX_BS = 256


def _select_cumsum_block_size(feature_dim: int) -> int:
    """Use the widest measured-safe block without padding beyond the cap."""

    return min(
        CUMSUM_MAX_BS,
        max(CUMSUM_MIN_BS, triton.next_power_of_2(feature_dim)),
    )


@libentry()
@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in (1, 2, 4, 8)
    ],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h * T,
            (T,),
            (1,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h * T,
            (T,),
            (1,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
    else:
        p_s = tl.make_block_ptr(
            s + bos * H + i_h,
            (T,),
            (H,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
        p_o = tl.make_block_ptr(
            o + bos * H + i_h,
            (T,),
            (H,),
            (i_t * BT,),
            (BT,),
            (0,),
        )
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        block_sum = tl.sum(b_s, axis=0)
        b_o = -b_o + block_sum[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    N_CHUNKS,
    N_BATCH_HEADS,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_GRID_REMAP: tl.constexpr,
):
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        grid_x = tl.cdiv(S, BS)
        i_s = linear_pid % grid_x
        linear_pid //= grid_x
        i_t = linear_pid % N_CHUNKS
        i_bh = linear_pid // N_CHUNKS
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_s = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

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

    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError("chunk_size must be a power of 2")
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    chunk_local_cumsum_scalar_kernel[(NT, B * H)](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    BS = _select_cumsum_block_size(S)
    use_grid_remap = requires_grid_remap(triton.cdiv(S, BS), NT, B * H)
    grid = make_grid_3d(
        triton.cdiv(S, BS),
        NT,
        B * H,
        remap=use_grid_remap,
    )

    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        BS=BS,
        N_CHUNKS=NT,
        N_BATCH_HEADS=B * H,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
        USE_GRID_REMAP=use_grid_remap,
        num_warps=1,
        num_stages=1,
    )
    return g


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if len(g.shape) >= 3 and not head_first and g.shape[1] < g.shape[2]:
        warnings.warn(
            "Input shape suggests that head_first may be set incorrectly: "
            f"sequence length {g.shape[1]} is smaller than heads {g.shape[2]}.",
            stacklevel=2,
        )
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    if len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    raise ValueError(
        "chunk_local_cumsum expects [B, T, H] or [B, T, H, D] "
        "input (or the corresponding head-first layout), "
        f"got {tuple(g.shape)}"
    )
