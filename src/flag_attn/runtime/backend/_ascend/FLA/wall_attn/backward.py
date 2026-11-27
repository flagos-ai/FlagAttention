# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tiled Wall attention gradients with FP32 accumulation on Ascend."""

import torch
import triton
import triton.language as tl

from .cumsum import gate_prefix_scan


@triton.jit
def _cast_output(X, Y, N, offset, BLOCK: tl.constexpr):
    i = (offset + tl.program_id(0)) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, i < N, 0), i < N)


@triton.jit
def _row_delta(O, DO, DELTA, N, offset, V: tl.constexpr, BM: tl.constexpr, BV: tl.constexpr):
    r = (offset + tl.program_id(0)) * BM + tl.arange(0, BM)
    d = tl.arange(0, BV)
    acc = tl.zeros((BM,), tl.float32)
    for block in range(tl.cdiv(V, BV)):
        v = block * BV + d
        p = r[:, None] * V + v[None, :]
        mask = (r[:, None] < N) & (v[None, :] < V)
        o = tl.load(O + p, mask, 0)
        do = tl.load(DO + p, mask, 0).to(tl.float32)
        acc += tl.sum(o * do, axis=1)
    tl.store(DELTA + r, acc, r < N)


@triton.jit
def _wall_grad_tile(
    Q, KEY, VALUE, P, C, STATS, DO, DELTA, OUT, DPQ, DC_OUT, scale, T, W, offset,
    H: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    MODE: tl.constexpr, HAS_WINDOW: tl.constexpr, HAS_SCALAR: tl.constexpr,
):
    pid = offset + tl.program_id(0)
    nt = tl.cdiv(T, BT)
    nv: tl.constexpr = tl.cdiv(V, BV) if MODE == 2 else 1
    v_start = (pid % nv) * BV
    tile = pid // nv % nt
    bh = pid // nv // nt
    b = bh // HQ
    hq = bh % HQ
    h = hq // (HQ // H)
    first = tile * BT
    t = tl.arange(0, BT)
    local = first + t
    d = tl.arange(0, BK)
    f = tl.arange(0, BV)

    if MODE == 0:
        start = tl.maximum(0, first - W + 1) if HAS_WINDOW else 0
        end = tl.minimum(T, first + BT)
    else:
        start = first
        end = tl.minimum(T, first + BT + W - 1) if HAS_WINDOW else T

    if MODE == 2:
        grad = tl.zeros((BT, BV), tl.float32)
    else:
        grad = tl.zeros((BT, BK), tl.float32)
    scalar_grad = tl.zeros((BT,), tl.float32)
    for block in range(start, end, BT):
        if MODE == 0:
            r, c = local, block + t
        else:
            r, c = block + t, local
        rq = (b * T + r) * HQ + hq
        ck = (b * T + c) * H + h
        cp = (b * T + c) * HQ + hq
        mr = r < T
        mc = c < T
        valid = mr[:, None] & mc[None, :] & (r[:, None] >= c[None, :])
        if HAS_WINDOW:
            valid = valid & ((r[:, None] - c[None, :]) < W)
        valid_q = mr[:, None] & (d[None, :] < K)
        valid_k = mc[:, None] & (d[None, :] < K)
        q_off = rq[:, None] * K + d[None, :]
        k_off = ck[:, None] * K + d[None, :]
        p_off = cp[:, None] * K + d[None, :]

        q = tl.load(Q + q_off, valid_q, 0).to(tl.float32)
        k = tl.load(KEY + k_off, valid_k, 0).to(tl.float32)
        pq = tl.load(P + q_off, valid_q, 0)
        pk = tl.load(P + p_off, valid_k, 0)
        exponent = tl.where(valid[:, :, None], pq[:, None, :] - pk[None, :, :], -float("inf"))
        gate = tl.exp2(exponent)
        score = tl.sum(q[:, None, :] * k[None, :, :] * gate, axis=2)
        score *= scale * 1.4426950216
        if HAS_SCALAR:
            cq = tl.load(C + rq, mr, 0)
            ck_scalar = tl.load(C + cp, mc, 0)
            score += cq[:, None] - ck_scalar[None, :]
        row_max = tl.load(STATS + 2 * rq, mr, 0)
        log_denom = tl.load(STATS + 2 * rq + 1, mr, 0)
        log_prob = (score - row_max[:, None]) - log_denom[:, None]
        prob = tl.exp2(tl.where(valid, log_prob, -float("inf")))

        if MODE == 2:
            vf = v_start + f
            do = tl.load(DO + rq[:, None] * V + vf[None, :],
                         mr[:, None] & (vf[None, :] < V), 0).to(tl.float32)
            grad += tl.sum(prob[:, :, None] * do[:, None, :], axis=0)
        else:
            dp = tl.zeros((BT, BT), tl.float32)
            for vb in range(tl.cdiv(V, BV)):
                vf = vb * BV + f
                do = tl.load(DO + rq[:, None] * V + vf[None, :],
                             mr[:, None] & (vf[None, :] < V), 0).to(tl.float32)
                v = tl.load(VALUE + ck[:, None] * V + vf[None, :],
                            mc[:, None] & (vf[None, :] < V), 0).to(tl.float32)
                dp += tl.sum(do[:, None, :] * v[None, :, :], axis=2)
            delta = tl.load(DELTA + rq, mr, 0)
            ds = prob * (dp - delta[:, None])
            weight = ds[:, :, None] * gate
            if MODE == 0:
                grad += tl.sum(weight * k[None, :, :], axis=1)
                if HAS_SCALAR:
                    scalar_grad += tl.sum(ds, axis=1)
            else:
                grad += tl.sum(weight * q[:, None, :], axis=0)
                if HAS_SCALAR:
                    scalar_grad += tl.sum(ds, axis=0)

    row = (b * T + local) * HQ + hq
    if MODE == 2:
        vf = v_start + f
        tl.store(OUT + row[:, None] * V + vf[None, :], grad,
                 (local[:, None] < T) & (vf[None, :] < V))
    else:
        p = row[:, None] * K + d[None, :]
        valid_out = (local[:, None] < T) & (d[None, :] < K)
        grad *= scale
        tl.store(OUT + p, grad, valid_out)
        if MODE == 0:
            q = tl.load(Q + p, valid_out, 0).to(tl.float32)
            tl.store(DPQ + p, q * grad, valid_out)
        if HAS_SCALAR:
            tl.store(DC_OUT + row, scalar_grad, local < T)


@triton.jit
def _gate_grad(Q_SIDE, K_SIDE, KEY, N, offset, HQ: tl.constexpr, H: tl.constexpr,
               K: tl.constexpr, VECTOR: tl.constexpr, BLOCK: tl.constexpr):
    i = (offset + tl.program_id(0)) * BLOCK + tl.arange(0, BLOCK)
    q_part = tl.load(Q_SIDE + i, i < N, 0)
    k_part = tl.load(K_SIDE + i, i < N, 0)
    if VECTOR:
        head = i // K % HQ
        kv_head = head // (HQ // H)
        ki = (i // (K * HQ) * H + kv_head) * K + i % K
        k_part *= tl.load(KEY + ki, i < N, 0).to(tl.float32)
    tl.store(Q_SIDE + i, q_part - k_part, i < N)


@triton.jit
def _reduce_kv_heads(PARTS, OUT, N, offset, H: tl.constexpr, HQ: tl.constexpr,
                     D: tl.constexpr, BLOCK: tl.constexpr):
    i = (offset + tl.program_id(0)) * BLOCK + tl.arange(0, BLOCK)
    G: tl.constexpr = HQ // H
    start = (i // D * G) * D + i % D
    total = tl.zeros((BLOCK,), tl.float32)
    for h in range(G):
        total += tl.load(PARTS + start + h * D, i < N, 0)
    tl.store(OUT + i, total, i < N)


@triton.jit
def _sink_grad(SINK, STATS, DELTA, OUT, N, HQ: tl.constexpr, BT: tl.constexpr):
    h = tl.program_id(0)
    sink = tl.load(SINK + h).to(tl.float32) * 1.4426950216
    r = tl.arange(0, BT)
    acc = tl.zeros((BT,), tl.float32)
    for block in range(tl.cdiv(N, BT)):
        t = block * BT + r
        row = t * HQ + h
        row_max = tl.load(STATS + 2 * row, t < N, 0)
        log_denom = tl.load(STATS + 2 * row + 1, t < N, 0)
        delta = tl.load(DELTA + t * HQ + h, t < N, 0)
        p = tl.exp2(tl.where(t < N, (sink - row_max) - log_denom, -float("inf")))
        acc -= delta * p
    tl.store(OUT + h, tl.sum(acc, axis=0))


def _launch(kernel, programs, *args, **kwargs):
    for start in range(0, programs, 32768):
        kernel[(min(programs - start, 32768),)](*args, start, **kwargs)


def cast_output(o, dtype):
    if o.dtype == dtype:
        return o
    out = torch.empty_like(o, dtype=dtype)
    _launch(_cast_output, triton.cdiv(o.numel(), 256), o, out, o.numel(), BLOCK=256)
    return out


def _segment_backward(q, k, v, p, c, stats, do, delta, scale, window_size, dq, dk, dv, dg, dg_scalar):
    B, T, HQ, K = q.shape
    H, V = v.shape[2:]
    bt = 8 if K <= 64 else 4
    bk, bv = triton.next_power_of_2(K), 16
    tiles = B * HQ * triton.cdiv(T, bt)
    dq_prefix = torch.empty_like(p)
    dh_key = torch.empty_like(p)
    dh_value = v.new_empty((B, T, HQ, V), dtype=torch.float32)
    dcq = torch.empty_like(c) if c is not None else None
    dck = torch.empty_like(c) if c is not None else None
    w = window_size if window_size is not None else 0
    for mode, out, dc in ((0, dq, dcq), (1, dh_key, dck), (2, dh_value, None)):
        programs = tiles * (triton.cdiv(V, bv) if mode == 2 else 1)
        _launch(
            _wall_grad_tile, programs,
            q, k, v, p, c if c is not None else p, stats, do, delta,
            out, dq_prefix, dc if dc is not None else delta, scale, T, w,
            H=H, HQ=HQ, K=K, V=V, BT=bt, BK=bk, BV=bv, MODE=mode,
            HAS_WINDOW=window_size is not None, HAS_SCALAR=c is not None,
        )

    # Prefix derivatives use natural-log gate units.
    _launch(_gate_grad, triton.cdiv(p.numel(), 256), dq_prefix, dh_key, k, p.numel(),
            HQ=HQ, H=H, K=K, VECTOR=True, BLOCK=256)
    gate_prefix_scan(dq_prefix, 1.0, out=dg, reverse=True)
    if c is not None:
        _launch(_gate_grad, triton.cdiv(c.numel(), 256), dcq, dck, k, c.numel(),
                HQ=HQ, H=H, K=1, VECTOR=False, BLOCK=256)
        gate_prefix_scan(dcq, 1.0, out=dg_scalar, reverse=True)
    for parts, out, dim in ((dh_key, dk, K), (dh_value, dv, V)):
        _launch(_reduce_kv_heads, triton.cdiv(out.numel(), 256), parts, out, out.numel(),
                H=H, HQ=HQ, D=dim, BLOCK=256)


def wall_attn_backward(q, k, v, o, p, c, stats, sink_bias, do, scale, window_size,
                       offsets, gate_dtype, scalar_dtype):
    B, T, HQ, _ = q.shape
    do = do.contiguous()
    delta = stats.new_empty((B, T, HQ))
    _launch(_row_delta, triton.cdiv(delta.numel(), 16), o, do, delta, delta.numel(),
            V=v.shape[-1], BM=16, BV=32)

    dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
    dg = torch.empty_like(p, dtype=gate_dtype)
    dg_scalar = torch.empty_like(c, dtype=scalar_dtype) if c is not None else None
    for start, end in zip(offsets, offsets[1:]):
        if start == end or B == 0:
            continue
        rows = slice(start, end)
        _segment_backward(
            q[:, rows], k[:, rows], v[:, rows], p[:, rows],
            c[:, rows] if c is not None else None, stats[:, rows], do[:, rows], delta[:, rows],
            scale, window_size, dq[:, rows], dk[:, rows], dv[:, rows], dg[:, rows],
            dg_scalar[:, rows] if dg_scalar is not None else None,
        )

    dsink = torch.empty_like(sink_bias) if sink_bias is not None else None
    if dsink is not None:
        _sink_grad[(HQ,)](sink_bias, stats, delta, dsink, B * T, HQ=HQ, BT=128)
    return dq, dk, dv, dg, dsink, dg_scalar
