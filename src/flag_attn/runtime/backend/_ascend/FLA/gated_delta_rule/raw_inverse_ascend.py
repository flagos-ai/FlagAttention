# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Isolated TLE-Raw experiments for GDN block inverse."""
from __future__ import annotations

from pathlib import Path

import torch
import triton

from .fast_launch import fast_launch
import triton.language as tl
import triton.experimental.tle as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect


_RAW_CPP = str(Path(__file__).resolve().with_name("raw_inverse_diag.cpp"))
_RAW_OUTER_CPP = str(Path(__file__).resolve().with_name("raw_inverse_outer.cpp"))
_RAW_OUTER_BATCH2_CPP = str(
    Path(__file__).resolve().with_name("raw_inverse_outer_batch2.cpp")
)


@dialect(name="cann", file=_RAW_CPP,
         extern_func_name="raw_inverse32_diag",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _raw_inverse32_diag(*args, **kwargs):
    ...


@dialect(name="cann", file=_RAW_OUTER_CPP,
         extern_func_name="raw_inverse32_outer",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _raw_inverse32_outer(*args, **kwargs):
    ...


@dialect(name="cann", file=_RAW_OUTER_BATCH2_CPP,
         extern_func_name="raw_inverse32_outer_batch2",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _raw_inverse32_outer_batch2(*args, **kwargs):
    ...


@triton.jit
def _raw_inverse32_tile(L, job, block, BT: tl.constexpr):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    base = job * BT * BT + (block * 32 + r[:, None]) * BT + block * 32 + c[None, :]
    lower = tl.load(L + base).to(tl.float32)
    lower = tl.where(r[:, None] > c[None, :], lower, 0.0)
    eye = (r[:, None] == c[None, :]).to(tl.float32)
    # One explicit UB region. Slots are disjoint row bands, each beginning on
    # a 32-byte boundary: lower[0:32], eye[32:64], zero-offsets[64], output[65:98].
    arena = tle.dsa.alloc([99, 32], dtype=tl.float32,
                          mem_addr_space=tle.dsa.ascend.UB)
    lower_ub = tle.dsa.subview(arena, [0, 0], [32, 32], [1, 1])
    eye_ub = tle.dsa.subview(arena, [32, 0], [32, 32], [1, 1])
    zero_offsets = tle.dsa.subview(arena, [64, 0], [1, 32], [1, 1])
    out_ub = tle.dsa.subview(arena, [65, 0], [33, 32], [1, 1])
    tle.dsa.to_buffer(lower, tle.dsa.ascend.UB, bind_buffer=lower_ub)
    tle.dsa.to_buffer(eye, tle.dsa.ascend.UB, bind_buffer=eye_ub)
    tle.dsa.to_buffer(tl.zeros((1, 32), tl.float32), tle.dsa.ascend.UB,
                      bind_buffer=zero_offsets)
    return tle_raw.call(
        _raw_inverse32_diag,
        [tle.dsa.to_tensor(lower_ub, writable=False),
         tle.dsa.to_tensor(eye_ub, writable=False),
         tle.dsa.to_tensor(zero_offsets, writable=False),
         tle.dsa.to_tensor(out_ub, writable=True)],
        output_indices=[3],
    )


@triton.jit
def _raw_inverse32_outer_tile(L, job, block, BT: tl.constexpr,
                              PACKED_L: tl.constexpr = False):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    if PACKED_L:
        # The production opt-in stores only L's A/D diagonal and C lower-left
        # blocks in [tile, 3, 32, 32] FP32 order.
        base = job * 3 * 32 * 32 + block * 32 * 32 + r[:, None] * 32 + c[None, :]
    else:
        base = job * BT * BT + (block * 32 + r[:, None]) * BT + block * 32 + c[None, :]
    lower = tl.load(L + base).to(tl.float32)
    lower = tl.where(r[:, None] > c[None, :], lower, 0.0)
    eye = (r[:, None] == c[None, :]).to(tl.float32)

    # One UB backing region [203,32] uint32 words. Views are disjoint and all
    # starts are 32-byte aligned: L@0, I@1024, gather offsets@2048,
    # inverse@2080, work@3136. Work holds accum/column/BRCB/row/column tiles.
    arena = tle.dsa.alloc([203, 32], dtype=tl.uint32,
                          mem_addr_space=tle.dsa.ascend.UB)
    lower_view = tle.dsa.subview(arena, [0, 0], [32, 32], [1, 1])
    eye_view = tle.dsa.subview(arena, [32, 0], [32, 32], [1, 1])
    offsets_view = tle.dsa.subview(arena, [64, 0], [1, 32], [1, 1])
    inverse_view = tle.dsa.subview(arena, [65, 0], [33, 32], [1, 1])
    work_view = tle.dsa.subview(arena, [98, 0], [105, 32], [1, 1])
    tle.dsa.to_buffer(tl.cast(lower, tl.uint32, bitcast=True),
                      tle.dsa.ascend.UB, bind_buffer=lower_view)
    tle.dsa.to_buffer(tl.cast(eye, tl.uint32, bitcast=True),
                      tle.dsa.ascend.UB, bind_buffer=eye_view)
    offsets = (tl.arange(0, 32).to(tl.uint32) * 128)[None, :]
    tle.dsa.to_buffer(offsets, tle.dsa.ascend.UB, bind_buffer=offsets_view)
    lower_tensor = tle.dsa.to_tensor(lower_view, writable=False)
    eye_tensor = tle.dsa.to_tensor(eye_view, writable=False)
    offsets_tensor = tle.dsa.to_tensor(offsets_view, writable=False)
    inverse_tensor = tle.dsa.to_tensor(inverse_view, writable=True)
    result = tle_raw.call(
        _raw_inverse32_outer,
        [lower_tensor, eye_tensor, offsets_tensor, inverse_tensor],
        output_indices=[3],
    )
    return tl.cast(result, tl.float32, bitcast=True)


@triton.jit
def _inverse32_diag_kernel(L, OUT, JOBS: tl.constexpr, BT: tl.constexpr,
                           BLOCK: tl.constexpr):
    job = tl.program_id(0)
    result = _raw_inverse32_tile(L, job, BLOCK, BT)
    r = tl.arange(0, 33)
    c = tl.arange(0, 32)
    out_base = job * 33 * 32 + r[:, None] * 32 + c[None, :]
    tl.store(OUT + out_base, result)


@triton.jit
def _inverse32_diag_pair_kernel(L, OUT, JOBS: tl.constexpr, BT: tl.constexpr):
    job = tl.program_id(0)
    block = tl.program_id(1)
    result = _raw_inverse32_tile(L, job, block, BT)
    r = tl.arange(0, 33)
    c = tl.arange(0, 32)
    out_base = job * 2 * 33 * 32 + block * 33 * 32 + r[:, None] * 32 + c[None, :]
    tl.store(OUT + out_base, result)


@triton.jit
def _inverse32_diag_pair_outer_kernel(L, OUT, JOBS: tl.constexpr,
                                      BT: tl.constexpr):
    job = tl.program_id(0)
    block = tl.program_id(1)
    result = _raw_inverse32_outer_tile(L, job, block, BT)
    r = tl.arange(0, 33)
    c = tl.arange(0, 32)
    out_base = job * 2 * 33 * 32 + block * 33 * 32 + r[:, None] * 32 + c[None, :]
    tl.store(OUT + out_base, result)


@triton.jit
def _raw_inverse32_outer_batch2_tile(L, job, block, BT: tl.constexpr, arena):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    base = job * BT * BT + (block * 32 + r[:, None]) * BT + block * 32 + c[None, :]
    lower = tl.load(L + base).to(tl.float32)
    lower = tl.where(r[:, None] > c[None, :], lower, 0.0)
    eye = (r[:, None] == c[None, :]).to(tl.float32)
    lower_view = tle.dsa.subview(arena, [0, 0], [32, 32], [1, 1])
    eye_view = tle.dsa.subview(arena, [32, 0], [32, 32], [1, 1])
    offsets_view = tle.dsa.subview(arena, [64, 0], [1, 32], [1, 1])
    inverse_view = tle.dsa.subview(arena, [65, 0], [33, 32], [1, 1])
    # Kept as a descriptor for the reserved workspace tail so the shared
    # arena's intended partition remains explicit at this call site.
    work_view = tle.dsa.subview(arena, [98, 0], [105, 32], [1, 1])
    tle.dsa.to_buffer(tl.cast(lower, tl.uint32, bitcast=True),
                      tle.dsa.ascend.UB, bind_buffer=lower_view)
    tle.dsa.to_buffer(tl.cast(eye, tl.uint32, bitcast=True),
                      tle.dsa.ascend.UB, bind_buffer=eye_view)
    offsets = (tl.arange(0, 32).to(tl.uint32) * 128)[None, :]
    tle.dsa.to_buffer(offsets, tle.dsa.ascend.UB, bind_buffer=offsets_view)
    result = tle_raw.call(
        _raw_inverse32_outer_batch2,
        [tle.dsa.to_tensor(lower_view, writable=False),
         tle.dsa.to_tensor(eye_view, writable=False),
         tle.dsa.to_tensor(offsets_view, writable=False),
         tle.dsa.to_tensor(inverse_view, writable=True)],
        output_indices=[3],
    )
    return tl.cast(result, tl.float32, bitcast=True)


@triton.jit
def _inverse64_rawdiag_batch2_kernel(L, OUT, JOBS: tl.constexpr,
                                     BT: tl.constexpr):
    job = tl.program_id(0)
    r = tl.arange(0, 33)
    c = tl.arange(0, 32)
    arena = tle.dsa.alloc([203, 32], dtype=tl.uint32,
                          mem_addr_space=tle.dsa.ascend.UB)

    a = _raw_inverse32_outer_batch2_tile(L, job, 0, BT, arena)
    a_base = job * 2 * 33 * 32 + r[:, None] * 32 + c[None, :]
    tl.store(OUT + a_base, a)

    d = _raw_inverse32_outer_batch2_tile(L, job, 1, BT, arena)
    d_base = job * 2 * 33 * 32 + 33 * 32 + r[:, None] * 32 + c[None, :]
    tl.store(OUT + d_base, d)


@triton.jit
def _inverse32_diag_pair_packed_kernel(L, OUT, JOBS: tl.constexpr,
                                       BT: tl.constexpr):
    job = tl.program_id(0)
    block = tl.program_id(1)
    result = _raw_inverse32_outer_tile(L, job, block, BT, True)
    r = tl.arange(0, 33)
    c = tl.arange(0, 32)
    out_base = job * 2 * 33 * 32 + block * 33 * 32 + r[:, None] * 32 + c[None, :]
    tl.store(OUT + out_base, result)


@triton.jit
def _inverse32_stitch_kernel(L, DIAG, OUT, JOBS: tl.constexpr,
                             BT: tl.constexpr, OUT_BF16: tl.constexpr,
                             CUBE_BF16: tl.constexpr):
    job = tl.program_id(0)
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    a_base = job * 2 * 33 * 32 + r[:, None] * 32 + c[None, :]
    d_base = job * 2 * 33 * 32 + 33 * 32 + r[:, None] * 32 + c[None, :]
    a = tl.load(DIAG + a_base).to(tl.float32)
    d = tl.load(DIAG + d_base).to(tl.float32)
    c_base = job * BT * BT + (32 + r[:, None]) * BT + c[None, :]
    block_c = tl.load(L + c_base).to(tl.float32)
    if CUBE_BF16:
        x = tl.dot(block_c.to(tl.bfloat16), a.to(tl.bfloat16)).to(tl.bfloat16)
        b = -tl.dot(d.to(tl.bfloat16), x).to(tl.float32)
    else:
        x = tl.dot(block_c, a)
        b = -tl.dot(d, x)
    if OUT_BF16:
        a = a.to(tl.bfloat16).to(tl.float32)
        d = d.to(tl.bfloat16).to(tl.float32)
        b = b.to(tl.bfloat16).to(tl.float32)
    # Write only the structural zero block. Writing a full 64x64 zero tile and
    # then overwriting its diagonal/lower blocks creates overlapping stores.
    tl.store(OUT + job * BT * BT + r[:, None] * BT + (32 + c[None, :]), 0.0)
    tl.store(OUT + job * BT * BT + r[:, None] * BT + c[None, :], a)
    tl.store(OUT + job * BT * BT + (32 + r[:, None]) * BT + c[None, :], b)
    tl.store(OUT + job * BT * BT + (32 + r[:, None]) * BT + (32 + c[None, :]), d)


def launch_inverse32_diag(L: torch.Tensor, out: torch.Tensor, block: int):
    jobs, bt, _ = L.shape
    fast_launch(_inverse32_diag_kernel, (jobs,), 4, 1)(L, out, jobs, bt, block)


def launch_inverse64_rawdiag(L: torch.Tensor, diag: torch.Tensor):
    jobs, bt, _ = L.shape
    fast_launch(_inverse32_diag_pair_kernel, (jobs, 2), 4, 1)(
        L, diag, jobs, bt)


def launch_inverse64_rawdiag_outer(L: torch.Tensor, diag: torch.Tensor):
    jobs, bt, _ = L.shape
    fast_launch(_inverse32_diag_pair_outer_kernel, (jobs, 2), 4, 1)(
        L, diag, jobs, bt)


def launch_inverse64_rawdiag_outer_batch2(L: torch.Tensor, diag: torch.Tensor):
    """Invert both diagonal 32x32 blocks per CTA using one reused UB arena."""
    jobs, bt, _ = L.shape
    if bt != 64 or L.shape[1:] != (64, 64):
        raise ValueError("batched Raw inverse requires [jobs,64,64] input")
    if diag.shape != (jobs, 2, 33, 32):
        raise ValueError("batched Raw inverse output must be [jobs,2,33,32]")
    fast_launch(_inverse64_rawdiag_batch2_kernel, (jobs,), 4, 1)(
        L, diag, jobs, bt,
    )


def launch_inverse64_rawdiag_outer_packed(L: torch.Tensor, diag: torch.Tensor):
    """Invert the two diagonal 32x32 blocks in packed `[tile,3,32,32]` L."""
    jobs = L.shape[0] * L.shape[1] * L.shape[2]
    fast_launch(_inverse32_diag_pair_packed_kernel, (jobs, 2), 4, 1)(
        L, diag, jobs, 64)


def launch_inverse64_stitch(L: torch.Tensor, diag: torch.Tensor,
                            out: torch.Tensor, *, cube_bf16: bool = False):
    jobs, bt, _ = L.shape
    fast_launch(_inverse32_stitch_kernel, (jobs,), 4, 1)(
        L, diag, out, jobs, bt, out.dtype == torch.bfloat16, cube_bf16)


__all__ = ["launch_inverse32_diag", "launch_inverse64_rawdiag",
           "launch_inverse64_rawdiag_outer",
           "launch_inverse64_rawdiag_outer_batch2",
           "launch_inverse64_rawdiag_outer_packed",
           "launch_inverse64_stitch"]
