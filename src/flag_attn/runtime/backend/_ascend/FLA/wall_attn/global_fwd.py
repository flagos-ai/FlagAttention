# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Guarded FP32 factorization for dense causal Wall attention.

With P the gate prefix and R half its final value, Qg=q*exp(P-R) and
Kg=k*exp(R-P) factor the gated score without a quadratic score allocation.
HF32 dot retains FP32 exponent range. Device metadata bounds transformed
operands and logits before using an unshifted softmax; other inputs use direct
gated scores with stable online normalization. Neither path drops KV pairs.
"""

from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language.raw as raw
from triton.experimental.tle.raw import dialect


@dialect(
    name="cann",
    file=str(Path(__file__).with_name("ops") / "wall_global_prefix.cpp"),
    extern_func_name="prefix_owned",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def prefix_op(*args, **kwargs):
    ...


@dialect(
    name="cann",
    file=str(Path(__file__).with_name("ops") / "wall_global_stats.cpp"),
    extern_func_name="stats_bounds",
    pipeline={"core": "vector", "pipe": "PIPE_V"},
)
def stats_bounds_op(*args, **kwargs):
    ...


@triton.jit
def _wall_prepare(
    Q, K, V, G, W,
    T: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, B: tl.constexpr,
    SCALE: tl.constexpr, LIMIT: tl.constexpr,
):
    size: tl.constexpr = B * T * HQ * 64
    QG = W
    KG = W + size
    DELTA = W + 2 * size
    META = tl.cast(W + 3 * size, tl.pointer_type(tl.int32))
    bh = tl.program_id(0)
    batch = bh // HQ
    head = bh % HQ
    kh = head // (HQ // H)
    r = tl.arange(0, 64)
    d = tl.program_id(1) * 16 + tl.arange(0, 16)
    field = tl.arange(0, 8)[:, None]
    # Raw owns the running sums, extrema and squared norms across blocks.
    state = tl.where(field == 3, tl.full((8, 16), -float('inf'), tl.float32), tl.full((8, 16), 0.0, tl.float32))
    for c in range(T // 64):
        rows = c * 64 + r
        x = tl.load(G + (((batch * T + rows) * HQ + head) * 64)[:, None] + d[None, :]).to(tl.float32)
        qval = tl.load(Q + (((batch * T + rows) * HQ + head) * 64)[:, None] + d[None, :]).to(tl.float32)
        kval = tl.load(K + (((batch * T + rows) * H + kh) * 64)[:, None] + d[None, :]).to(tl.float32)
        vv = tl.load(V + (((batch * T + rows) * H + kh) * 64)[:, None] + d[None, :]).to(tl.float32)
        work = tl.full((64, 16), 0.0, tl.float32)
        aux = tl.full((256,), 0.0, tl.float32)
        (x, qval, kval, vv, state, work, aux) = raw.call(
            stats_bounds_op, [x, qval, kval, vv, state, work, aux],
            output_indices=[0, 1, 2, 3, 4, 5, 6],
        )
    total = tl.sum(tl.where(field == 0, state, 0.0), 0)
    gmax = tl.max(tl.max(tl.where(field == 3, state, -float("inf")), 0), 0)
    qm = tl.max(tl.max(tl.where(field == 4, state, 0.0), 0), 0)
    km = tl.max(tl.max(tl.where(field == 5, state, 0.0), 0), 0)
    vm = tl.max(tl.max(tl.where(field == 6, state, 0.0), 0), 0)
    center = total * 0.5
    # Each channel magnitude is bounded by its group L2 norm. The margin
    # covers FP32 reduction rounding without separate amplitude reductions.
    bound = tl.abs(center) + 0.5 * tl.log(tl.maximum(1.0, tl.maximum(qm, km) + 2.0))
    # Nonpositive gates bound causal channel weights by one. The exponent
    # and V bounds protect the factorization and unnormalized FP32 numerator.
    safe = tl.min(((gmax <= 0.0) & (bound < 88.0)).to(tl.int32), 0) & (qm < 1022.0).to(tl.int32) & (km < 1022.0).to(tl.int32) & (vm < 3.402823466e+38 / (T * 1.1420073898e+26 * 32.0)).to(tl.int32)
    # Four groups bound full Q/K norms. Floor+2 conservatively covers FP32
    # reduction rounding; only the metadata is capped, never the input data.
    qi = tl.minimum(qm, 1021.0).to(tl.int32) + 2
    ki = tl.minimum(km, 1021.0).to(tl.int32) + 2
    meta = safe << 30 | (qi & 1023) << 10 | ki & 1023
    tl.store(META + bh * 4 + tl.program_id(1), meta)
    carry = tl.zeros((16,), tl.float32)
    for c in range(T // 64):
        rows = c * 64 + r
        qp = (((batch * T + rows) * HQ + head) * 64)[:, None] + d[None, :]
        kv = (((batch * T + rows) * H + kh) * 64)[:, None] + d[None, :]
        gg = tl.load(G + qp).to(tl.float32)
        work = tl.full((64, 16), 0, tl.float32)
        (cp, carry, work) = raw.call(prefix_op, [center, gg, carry, work], output_indices=[1, 2, 3])
        tl.store(DELTA + qp, cp)
        delta = tl.where(safe != 0, cp, 0.0) * 1.4426950216
        q = tl.load(Q + qp).to(tl.float32)
        tl.store(QG + qp, tl.where(safe != 0, q * tl.exp2(delta), 0.0))
        k = tl.load(K + kv).to(tl.float32)
        tl.store(KG + qp, tl.where(safe != 0, k * tl.exp2(-delta), 0.0))


@triton.jit
def _wall_global_fwd(
    Q, K, V, W, O,
    T: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, B: tl.constexpr,
    SCALE: tl.constexpr, LIMIT: tl.constexpr,
):
    size: tl.constexpr = B * T * HQ * 64
    QG = W
    KG = W + size
    D = W + 2 * size
    META = tl.cast(W + 3 * size, tl.pointer_type(tl.int32))
    BM: tl.constexpr = 128
    BN: tl.constexpr = 128
    m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HQ
    head = bh % HQ
    kh = head // (HQ // H)
    rows = m * BM + tl.arange(0, BM)
    d = tl.arange(0, 64)
    n = tl.arange(0, BN)
    # Scalar integer loads keep the predicate independent of vector reductions.
    m0 = tl.load(META + bh * 4)
    m1 = tl.load(META + bh * 4 + 1)
    m2 = tl.load(META + bh * 4 + 2)
    m3 = tl.load(META + bh * 4 + 3)
    safe = m0 & m1 & m2 & m3 & 1 << 30
    qs = (m0 >> 10 & 1023) + (m1 >> 10 & 1023) + (m2 >> 10 & 1023) + (m3 >> 10 & 1023)
    ks = (m0 & 1023) + (m1 & 1023) + (m2 & 1023) + (m3 & 1023)
    # Cauchy-Schwarz: abs(score) <= abs(scale)*sqrt(sum(qnorm)*sum(knorm)).
    fast = (safe != 0) & (qs * ks <= LIMIT)
    if fast:
        q = tl.load(QG + (((batch * T + rows) * HQ + head) * 64)[:, None] + d[None, :])
        acc = tl.zeros((BM, 64), tl.float32)
        den = tl.zeros((BM,), tl.float32)
        den16 = tl.zeros((BM, 16), tl.float32)
        ones = tl.full((BN, 16), 1.0, tl.float32)
        full = m * BM // BN
        for start in range(full):
            cols = start * BN + n
            kk = tl.load(KG + (((batch * T + cols) * HQ + head) * 64)[:, None] + d[None, :])
            s = tl.dot(q, tl.trans(kk), input_precision='hf32') * (SCALE * 1.4426950216)
            p = tl.exp2(s)
            den16 = tl.dot(p, ones, den16, input_precision='hf32')
            vv = tl.load(V + (((batch * T + cols) * H + kh) * 64)[:, None] + d[None, :]).to(tl.float32)
            acc = tl.dot(p, vv, acc, input_precision='hf32')
        for start in range(full, tl.cdiv((m + 1) * BM, BN)):
            cols = start * BN + n
            kk = tl.load(KG + (((batch * T + cols) * HQ + head) * 64)[:, None] + d[None, :])
            s = tl.dot(q, tl.trans(kk), input_precision='hf32') * (SCALE * 1.4426950216)
            s = tl.where(rows[:, None] >= cols[None, :], s, -float('inf'))
            p = tl.exp2(s)
            den16 = tl.dot(p, ones, den16, input_precision='hf32')
            vv = tl.load(V + (((batch * T + cols) * H + kh) * 64)[:, None] + d[None, :]).to(tl.float32)
            acc = tl.dot(p, vv, acc, input_precision='hf32')
        den = tl.sum(den16, 1) * 0.0625
        tl.store(O + (((batch * T + rows) * HQ + head) * 64)[:, None] + d[None, :], (acc / den[:, None]).to(O.dtype.element_ty))
    else:
        # Direct gated scores avoid overflowing the transformed operands.
        for sub in range(8):
            r = m * 128 + sub * 16 + tl.arange(0, 16)
            c = tl.arange(0, 16)
            df = tl.arange(0, 16)
            acc = tl.zeros((16, 64), tl.float32)
            den = tl.zeros((16,), tl.float32)
            mx = tl.full((16,), -float('inf'), tl.float32)
            for start in range(tl.cdiv(m * 128 + sub * 16 + 16, 16)):
                col = start * 16 + c
                s = tl.zeros((16, 16), tl.float32)
                valid = (r[:, None] < T) & (col[None, :] < T) & (col[None, :] <= r[:, None])
                for f in range(4):
                    ff = f * 16 + df
                    qo = ((batch * T + r[:, None]) * HQ + head) * 64 + ff[None, :]
                    ko = ((batch * T + col[:, None]) * H + kh) * 64 + ff[None, :]
                    po = ((batch * T + col[:, None]) * HQ + head) * 64 + ff[None, :]
                    q = tl.load(Q + qo, mask=r[:, None] < T, other=0).to(tl.float32)
                    k = tl.load(K + ko, mask=col[:, None] < T, other=0).to(tl.float32)
                    pq = tl.load(D + qo, mask=r[:, None] < T, other=0)
                    pk = tl.load(D + po, mask=col[:, None] < T, other=0)
                    delta = tl.where(valid[:, :, None], pq[:, None, :] - pk[None, :, :], 0.0)
                    s += tl.sum(q[:, None, :] * k[None, :, :] * tl.exp(delta), 2)
                s = s * (SCALE * 1.4426950408889634)
                s = tl.where(valid, s, -float('inf'))
                s = tl.where((r[:, None] >= T) & (col[None, :] == 0), 0.0, s)
                nm = tl.maximum(mx, tl.max(s, 1))
                a = tl.exp2(mx - nm)
                prob = tl.exp2(s - nm[:, None])
                newden = den * a + tl.sum(prob, 1)
                # Keep the output normalized so large finite V stays finite.
                acc = acc * (den * a / newden)[:, None]
                prob = prob / newden[:, None]
                den = newden
                dv = tl.arange(0, 64)
                vv = tl.load(V + ((batch * T + col[:, None]) * H + kh) * 64 + dv[None, :], mask=col[:, None] < T, other=0).to(tl.float32)
                acc = acc + tl.dot(prob, vv, input_precision='hf32')
                mx = nm
            tl.store(O + ((batch * T + r[:, None]) * HQ + head) * 64 + tl.arange(0, 64)[None, :], acc, mask=r[:, None] < T)


def _wall_global_impl(q, k, v, g, scale):
    (B, T, HQ, _) = q.shape
    H = k.shape[2]
    limit = int((60.0 / abs(scale)) ** 2) if abs(scale) > 60.0 / 4092 else 4092 ** 2
    # One allocation holds FP32 Qg, Kg, centered prefixes and int32 metadata.
    w = torch.empty((3 * q.numel() + B * HQ * 4,), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _wall_prepare[B * HQ, 4](q, k, v, g, w, T=T, H=H, HQ=HQ, B=B, SCALE=scale, LIMIT=limit)
    _wall_global_fwd[T // 128, B * HQ](q, k, v, w, out, T=T, H=H, HQ=HQ, B=B, SCALE=scale, LIMIT=limit)
    return out
