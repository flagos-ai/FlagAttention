# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame-compatible attention backward preprocessing."""

import torch
import triton
import triton.language as tl


MAX_GRID_X = 65535


@triton.jit(
    do_not_specialize=["row_offset"]
)
def parallel_attn_bwd_kernel_preprocess_enflame(
    o,
    do,
    delta,
    row_offset,
    B: tl.constexpr,
    V: tl.constexpr,
):
    """Compute a grid-limited slice of attention delta."""

    i_n = (
        tl.program_id(0)
        + row_offset
    )
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(
        o + i_n * V + o_d,
        mask=m_d,
        other=0,
    )
    b_do = tl.load(
        do + i_n * V + o_d,
        mask=m_d,
        other=0,
    ).to(tl.float32)

    b_delta = tl.sum(
        b_o * b_do
    )

    tl.store(
        delta + i_n,
        b_delta.to(
            delta.dtype.element_ty
        ),
    )


def parallel_attn_bwd_preprocess_enflame(
    o: torch.Tensor,
    do: torch.Tensor,
) -> torch.Tensor:
    """Compute delta using S60-safe grid slices."""

    if o.shape != do.shape:
        raise ValueError(
            "o and do must have identical shapes"
        )

    if not o.is_contiguous():
        o = o.contiguous()

    if not do.is_contiguous():
        do = do.contiguous()

    V = o.shape[-1]

    delta = torch.empty_like(
        o[..., 0],
        dtype=torch.float,
    )

    total_rows = delta.numel()
    block_size = triton.next_power_of_2(V)

    for row_offset in range(
        0,
        total_rows,
        MAX_GRID_X,
    ):
        row_count = min(
            MAX_GRID_X,
            total_rows - row_offset,
        )

        parallel_attn_bwd_kernel_preprocess_enflame[
            (row_count,)
        ](
            o=o,
            do=do,
            delta=delta,
            row_offset=row_offset,
            B=block_size,
            V=V,
        )

    return delta


__all__ = [
    "parallel_attn_bwd_preprocess_enflame",
]


# S60-compatible public helper binding.
parallel_attn_bwd_preprocess = parallel_attn_bwd_preprocess_enflame
