# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""TLE Raw factors/reductions for fixed 64-token, K=128 GLA backward tiles.
Guarded wide intra dots use a persistent panel fallback for unsafe chunks."""

from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as raw
from triton.experimental.tle.raw import dialect

from .chunk import FIXED_MAX_PROGRAMS, _chunks_per_launch
from .chunk_bwd import chunk_gla_bwd_dqk_intra_kernel as _panel_kernel


_FACTOR_CPP = str(Path(__file__).resolve().with_name("chunk_bwd_raw_factor.cpp"))
_REDUCE_CPP = str(Path(__file__).resolve().with_name("chunk_bwd_raw_reduce.cpp"))


@dialect(name="cann", file=_FACTOR_CPP, extern_func_name="gla_bwd_key_factor",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _key_factor(*args, **kwargs):
    ...


@dialect(name="cann", file=_FACTOR_CPP, extern_func_name="gla_bwd_query_factor",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _query_factor(*args, **kwargs):
    ...


@dialect(name="cann", file=_REDUCE_CPP, extern_func_name="gla_bwd_state_product",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _state_product_call(*args, **kwargs):
    ...


@dialect(name="cann", file=_REDUCE_CPP, extern_func_name="gla_bwd_reverse_tree",
         pipeline={"core": "vector", "pipe": "PIPE_V"})
def _reverse_tree_call(*args, **kwargs):
    ...


@triton.jit
def _state_product(h, dh):
    tl.static_assert(h.shape[0] == 128 and h.shape[1] == 64)
    # One vcadd per contiguous V row, with one scalar result per K.
    _, result = raw.call(_state_product_call, [dh, h, tl.zeros((1, 128), tl.float32)],
                         output_indices=[1, 2])
    return tl.sum(result, axis=0)


@triton.jit
def _reverse_sum(x, carry):
    tl.static_assert(x.shape[0] == 64 and x.shape[1] == 128)
    return raw.call(_reverse_tree_call, [carry[None, :], x], output_indices=[1])


@triton.jit
def _safe_operand(x):
    # A factor spans at most 2^60. Operand and dA exponents within +/-30
    # leave headroom for their product and the 64-term dot reduction.
    a = tl.abs(x.to(tl.float32))
    bad = (a != 0.0) & ~((a >= 2.0 ** -30) & (a < 2.0 ** 30))
    return tl.max(tl.max(bad.to(tl.int32), axis=0), axis=0) == 0


@triton.jit(do_not_specialize=["T"])
def _guard_kernel(q, k, g_cum, dA, wide_ok, T, H: tl.constexpr):
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    nt = tl.cdiv(T, 64)
    for i_t in range(tl.program_id(0), nt, tl.num_programs(0)):
        rows = i_t * 64 + tl.arange(0, 64)
        cols = tl.arange(0, 64)
        m = rows < T
        a = tl.load(dA + (i_b * T + rows[:, None]) * H * 64 + i_h * 64 + cols[None, :],
                    mask=m[:, None], other=0.0)
        ok = _safe_operand(a)
        for i_k in range(2):
            kk = i_k * 64 + cols
            off = (i_b * T + rows[:, None]) * H * 128 + i_h * 128 + kk[None, :]
            g = tl.load(g_cum + off, mask=m[:, None], other=0.0)
            ref = tl.load(g_cum + i_b * T * H * 128 + i_h * 128
                          + tl.minimum(i_t * 64 + 32, T - 1) * H * 128 + kk)
            span = tl.max(tl.max(tl.where(m[:, None], tl.abs(g - ref[None, :]), 0.0), axis=0), axis=0)
            qq = tl.load(q + off, mask=m[:, None], other=0.0)
            kv = tl.load(k + off, mask=m[:, None], other=0.0)
            ok = ok & (span <= 60.0) & _safe_operand(qq) & _safe_operand(kv)
        tl.store(wide_ok + (i_b * nt + i_t) * H + i_h, ok)


@triton.jit
def _factors(g, x, ref, QUERY: tl.constexpr):
    tl.static_assert(g.shape[0] == 64 and g.shape[1] == 128)
    if QUERY:
        factor, weighted, _ = raw.call(
            _query_factor, [ref[None, :], g, x, tl.zeros((64, 128), tl.float32)], output_indices=[1, 2, 3])
    else:
        factor, weighted, _ = raw.call(
            _key_factor, [ref[None, :], g, x, tl.zeros((64, 128), tl.float32)], output_indices=[1, 2, 3])
    return factor, weighted


@triton.jit
def _wide(q, k, g_cum, dA, dq, dk, T, i_t, i_bh, H: tl.constexpr):
    i_b, i_h = i_bh // H, i_bh % H
    rows = i_t * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    m = rows < T
    off = (i_b * T + rows[:, None]) * H * 128 + i_h * 128 + cols[None, :]
    ar = tl.arange(0, 64)
    a = tl.load(dA + (i_b * T + rows[:, None]) * H * 64 + i_h * 64 + ar[None, :],
                mask=m[:, None], other=0.0)
    ref = tl.load(g_cum + i_b * T * H * 128 + i_h * 128
                  + tl.minimum(i_t * 64 + 32, T - 1) * H * 128 + cols)
    g = tl.load(g_cum + off, mask=m[:, None], other=0.0)
    # Neutral padded gates prevent 0*inf in a short, strongly decaying tail.
    g = tl.where(m[:, None], g, ref[None, :])
    kv = tl.load(k + off, mask=m[:, None], other=0.0).to(tl.float32)
    f, kg = _factors(g, kv, ref, False)
    dqq = tl.dot(a, kg) * f
    tl.store(dq + off, dqq, mask=m[:, None])

    gq = tl.load(g_cum + off, mask=m[:, None], other=0.0)
    gq = tl.where(m[:, None], gq, ref[None, :])
    qq = tl.load(q + off, mask=m[:, None], other=0.0).to(tl.float32)
    fq, qg = _factors(gq, qq, ref, True)
    dkk = tl.dot(tl.trans(a), qg) * fq
    tl.store(dk + off, dkk, mask=m[:, None])


@triton.jit(do_not_specialize=["T", "I_T_LO"])
def _wide_kernel(q, k, g_cum, dA, dq, dk, wide_ok, T, I_T_LO, H: tl.constexpr):
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_t = tl.program_id(1) + I_T_LO
    nt = tl.cdiv(T, 64)
    if tl.load(wide_ok + (i_b * nt + i_t) * H + i_h):
        _wide(q, k, g_cum, dA, dq, dk, T, i_t, i_bh, H)


@triton.jit(do_not_specialize=["T"])
def _panel_kernel_persistent(q, k, g_cum, dA, dq, dk, wide_ok, T, H: tl.constexpr):
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    nt = tl.cdiv(T, 64)
    for i_t in range(tl.program_id(1), nt, tl.num_programs(1)):
        if not tl.load(wide_ok + (i_b * nt + i_t) * H + i_h):
            _panel_kernel(q, k, g_cum, dA, dq, dk, T, i_t - tl.program_id(1), H, 128, 64, 16, 128)


def launch_dqk_intra(q, k, g_cum, dA, dq, dk):
    b, t, h, _ = q.shape
    nt = triton.cdiv(t, 64)
    flags = torch.empty((b, nt, h), device=q.device, dtype=torch.bool)
    ng = min(nt, max(1, FIXED_MAX_PROGRAMS // (b * h)))
    _guard_kernel[(ng, b * h)](q, k, g_cum, dA, flags, t, h)
    cpb = _chunks_per_launch(nt * b * h, nt, FIXED_MAX_PROGRAMS)
    for lo in range(0, nt, cpb):
        _wide_kernel[(1, min(cpb, nt - lo), b * h)](q, k, g_cum, dA, dq, dk, flags, t, lo, h)
    _panel_kernel_persistent[(1, min(cpb, nt), b * h)](q, k, g_cum, dA, dq, dk, flags, t, h)
