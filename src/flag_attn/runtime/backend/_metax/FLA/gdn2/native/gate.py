# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# This file is modified and supported by the Moonshot AI Team

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.FLA.utils import exp
from ...compat import autotune_cache_kwargs


@triton.jit
def softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + exp(x)))


BS_LIST = [32, 64]


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    # Apply dt_bias if exists
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias + i_h * S, (S,), (1,), (i_s * BS,), (BS,), (0,))
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Apply chunk local cumsum
    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    lower_bound: float | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert (
            g.shape[0] == 1
        ), "Only batch size 1 is supported when cu_seqlens are provided"
    assert len(g.shape) == 4
    B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2 ** (
        chunk_size.bit_length() - 1
    ), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    kda_gate_chunk_cumsum_vector_kernel[grid](
        s=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=BT,
        REVERSE=False,
    )
    return g
