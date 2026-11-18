# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dual-diagonal Raw inverse launcher for fixed 64x64 inverse tiles."""
from __future__ import annotations

from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


_CPP = str(Path(__file__).resolve().with_name("raw_inverse_outer_dual.cpp"))


@dialect(name="cann", file=_CPP,
         extern_func_name="raw_inverse32_outer_dual",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _raw_inverse32_outer_dual(*args, **kwargs):
    ...


@triton.jit
def _dual_inverse_tile(L, job, BT: tl.constexpr):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    base = job * BT * BT
    lower_a = tl.load(L + base + r[:, None] * BT + c[None, :])
    lower_d = tl.load(L + base + (r[:, None] + 32) * BT + c[None, :] + 32)
    lower_a = tl.where(r[:, None] > c[None, :], lower_a, 0.0)
    lower_d = tl.where(r[:, None] > c[None, :], lower_d, 0.0)
    eye = (r[:, None] == c[None, :]).to(tl.float32)
    offsets = (tl.arange(0, 32).to(tl.uint32) * 128)[None, :]

    # One explicit arena: lower A/D @ rows 0/32, identity A/D @ 64/96,
    # offsets @ 128, packed output @ 129, work A/D @ 195/300.
    arena = tle.dsa.alloc([405, 32], dtype=tl.uint32,
                          mem_addr_space=tle.dsa.ascend.UB)
    lower_a_view = tle.dsa.subview(arena, [0, 0], [32, 32], [1, 1])
    lower_d_view = tle.dsa.subview(arena, [32, 0], [32, 32], [1, 1])
    identity_a_view = tle.dsa.subview(arena, [64, 0], [32, 32], [1, 1])
    identity_d_view = tle.dsa.subview(arena, [96, 0], [32, 32], [1, 1])
    offsets_view = tle.dsa.subview(arena, [128, 0], [1, 32], [1, 1])
    inverse_view = tle.dsa.subview(arena, [129, 0], [66, 32], [1, 1])
    tle.dsa.to_buffer(tl.cast(lower_a, tl.uint32, bitcast=True),
                      tle.dsa.ascend.UB, bind_buffer=lower_a_view)
    tle.dsa.to_buffer(tl.cast(lower_d, tl.uint32, bitcast=True),
                      tle.dsa.ascend.UB, bind_buffer=lower_d_view)
    eye_bits = tl.cast(eye, tl.uint32, bitcast=True)
    tle.dsa.to_buffer(eye_bits, tle.dsa.ascend.UB, bind_buffer=identity_a_view)
    tle.dsa.to_buffer(eye_bits, tle.dsa.ascend.UB, bind_buffer=identity_d_view)
    tle.dsa.to_buffer(offsets, tle.dsa.ascend.UB, bind_buffer=offsets_view)
    result = tle_raw.call(
        _raw_inverse32_outer_dual,
        [tle.dsa.to_tensor(lower_a_view, writable=False),
         tle.dsa.to_tensor(lower_d_view, writable=False),
         tle.dsa.to_tensor(identity_a_view, writable=False),
         tle.dsa.to_tensor(identity_d_view, writable=False),
         tle.dsa.to_tensor(offsets_view, writable=False),
         tle.dsa.to_tensor(inverse_view, writable=True)],
        output_indices=[5],
    )
    return tl.reshape(tl.cast(result, tl.float32, bitcast=True), (2, 33, 32))


@triton.jit
def _inverse64_rawdiag_dual_kernel(L, OUT, JOBS: tl.constexpr,
                                   BT: tl.constexpr):
    job = tl.program_id(0)
    pair = _dual_inverse_tile(L, job, BT)
    r = tl.arange(0, 2)[:, None, None]
    row = tl.arange(0, 33)[None, :, None]
    col = tl.arange(0, 32)[None, None, :]
    out = OUT + job * 2 * 33 * 32 + r * 33 * 32 + row * 32 + col
    tl.store(out, pair)


def launch_inverse64_rawdiag_dual(L: torch.Tensor, diag: torch.Tensor) -> None:
    """Invert both diagonal 32x32 blocks with one CTA and one Raw call."""
    jobs, bt, _ = L.shape
    if bt != 64 or L.shape[1:] != (64, 64):
        raise ValueError("dual Raw inverse requires L shaped [jobs,64,64]")
    if diag.shape != (jobs, 2, 33, 32):
        raise ValueError("dual Raw inverse output must be [jobs,2,33,32]")
    _inverse64_rawdiag_dual_kernel[(jobs,)](
        L, diag, jobs, bt, num_warps=4, num_stages=1,
    )


__all__ = ["launch_inverse64_rawdiag_dual"]
