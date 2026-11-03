"""Ascend Triton backward kernels for Parallax attention."""

from __future__ import annotations

import triton
import triton.language as tl
import torch

from ._utils import backward_streams, supports_staged


@triton.jit
def _preprocess_kernel(o, barv, grad_o, delta_t, delta_b, T: tl.constexpr, HQ: tl.constexpr,
                       D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    tiles_per_batch = tl.cdiv(T, BT)
    tile = tl.program_id(0) % tiles_per_batch
    batch = tl.program_id(0) // tiles_per_batch
    h = tl.program_id(1)
    rows = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    x = batch * T * HQ + rows * HQ + h
    offsets = x[:, None] * D + d[None, :]
    mask = (rows[:, None] < T) & (d[None, :] < D)
    go = tl.load(grad_o + offsets, mask, 0).to(tl.float32)
    out = tl.load(o + offsets, mask, 0).to(tl.float32)
    bv = tl.load(barv + offsets, mask, 0).to(tl.float32)
    tl.store(delta_t + x, tl.sum(go * out, axis=1), rows < T)
    tl.store(delta_b + x, tl.sum(go * bv, axis=1), rows < T)


@triton.jit
def _dqr_kernel(q, r, k, v, o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
                scale, T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                D: tl.constexpr, BD: tl.constexpr, WINDOW: tl.constexpr,
                BT: tl.constexpr, BS: tl.constexpr, INLINE_DELTA: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    b, hq = bh // HQ, bh % HQ
    h = hq // G
    bos_q = b * T * HQ * D + hq * D
    bos_k = b * T * H * D + h * D
    rows = tile * BT + tl.arange(0, BT)
    ds = tl.arange(0, BD)
    rowmask = rows < T
    qoff = bos_q + rows[:, None] * HQ * D + ds[None, :]
    qv = tl.load(q + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
    rv = tl.load(r + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
    go = tl.load(grad_o + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
    if INLINE_DELTA:
        ov = tl.load(o + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
        bv = tl.load(barv + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
        dt = tl.sum(go.to(tl.float32) * ov, axis=1)[:, None]
        db = tl.sum(go.to(tl.float32) * bv, axis=1)[:, None]
    else:
        dt = tl.load(delta_t + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
        db = tl.load(delta_b + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    norm = tl.load(d1 + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    maxv = tl.load(m + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    bt = tl.load(bart + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    gq = tl.zeros((BT, BD), tl.float32)
    gr = tl.zeros((BT, BD), tl.float32)
    safe_norm = tl.where(norm > 0.0, norm, 1.0)
    cols = tl.arange(0, BS)
    first_key = 0
    if WINDOW >= 0:
        first_key = tl.maximum(tile * BT - WINDOW + 1, 0)
    first_key_block = first_key // BS
    last_key = tl.minimum((tile + 1) * BT, T)
    last_key_block = tl.cdiv(last_key, BS)
    for block in range(first_key_block, last_key_block):
        ks = block * BS + cols
        kmask = (ks < T)[:, None] & (ds[None, :] < D)
        koff = bos_k + ks[:, None] * H * D + ds[None, :]
        kv = tl.load(k + koff, kmask, 0)
        vv = tl.load(v + koff, kmask, 0)
        allowed = (ks[None, :] <= rows[:, None]) & (ks[None, :] < T)
        if WINDOW >= 0:
            allowed = allowed & (ks[None, :] >= rows[:, None] - WINDOW + 1)
        scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
        if WINDOW < 0 and block < tile:
            p = tl.math.exp2(scores - maxv) / safe_norm
            p = tl.where(rowmask[:, None], p, 0.0)
        else:
            p = tl.where(allowed & rowmask[:, None], tl.math.exp2(scores - maxv) / safe_norm, 0.0)
        a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
        delta = a - db
        gr += tl.dot((-p * delta).to(kv.dtype), kv, out_dtype=tl.float32)
        rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
        gl = p * (a - dt + (bt - rk) * delta)
        gq += tl.dot(gl.to(kv.dtype), kv, out_dtype=tl.float32)
    gq *= scale
    tl.store(grad_q + qoff, gq, rowmask[:, None] & (ds[None, :] < D))
    tl.store(grad_r + qoff, gr, rowmask[:, None] & (ds[None, :] < D))


@triton.jit
def _dkv_kernel(q, r, k, v, o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_k, grad_v,
                scale, T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                D: tl.constexpr, BD: tl.constexpr, WINDOW: tl.constexpr,
                BT: tl.constexpr, BS: tl.constexpr, INLINE_DELTA: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    b, hq = bh // HQ, bh % HQ
    h = hq // G
    bos_q = b * T * HQ * D + hq * D
    bos_k = b * T * H * D + h * D
    cols = tile * BS + tl.arange(0, BS)
    ds = tl.arange(0, BD)
    colmask = cols < T
    koff = bos_k + cols[:, None] * H * D + ds[None, :]
    kv = tl.load(k + koff, colmask[:, None] & (ds[None, :] < D), 0)
    vv = tl.load(v + koff, colmask[:, None] & (ds[None, :] < D), 0)
    gk = tl.zeros((BS, BD), tl.float32)
    gv = tl.zeros((BS, BD), tl.float32)
    rows = tl.arange(0, BT)
    # Causality excludes queries before this key tile. A finite left window
    # also excludes queries after the last key in the tile leaves the window.
    first_query_block = (tile * BS) // BT
    last_query = T
    if WINDOW >= 0:
        last_query = tl.minimum((tile + 1) * BS + WINDOW - 1, T)
    last_query_block = tl.cdiv(last_query, BT)
    if WINDOW < 0:
        # Only the first query block intersects the causal diagonal. Later blocks
        # see every key in this tile and can skip the pairwise causal mask.
        causal_end = tl.minimum(tl.cdiv((tile + 1) * BS, BT), last_query_block)
        for block in range(first_query_block, causal_end):
            qs = block * BT + rows
            rowmask = qs < T
            qoff = bos_q + qs[:, None] * HQ * D + ds[None, :]
            qv = tl.load(q + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            rv = tl.load(r + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            go = tl.load(grad_o + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            if INLINE_DELTA:
                ov = tl.load(o + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
                bv = tl.load(barv + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
                dt = tl.sum(go.to(tl.float32) * ov, axis=1)[:, None]
                db = tl.sum(go.to(tl.float32) * bv, axis=1)[:, None]
            else:
                dt = tl.load(delta_t + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
                db = tl.load(delta_b + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            norm = tl.load(d1 + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            maxv = tl.load(m + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            bt = tl.load(bart + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            allowed = (qs[:, None] >= cols[None, :]) & rowmask[:, None] & colmask[None, :]
            scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
            scores = tl.where(allowed, scores, float("-inf"))
            safe_norm = tl.where(norm > 0.0, norm, 1.0)
            p = tl.where(allowed, tl.math.exp2(scores - maxv) / safe_norm, 0.0)
            rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
            a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
            rk = tl.where(allowed, rk, 0.0)
            a = tl.where(allowed, a, 0.0)
            delta = a - db
            correction = bt - rk
            gl = p * (a - dt + correction * delta) * scale
            gu = -p * delta
            weights = p * (1.0 + correction)
            gl = tl.where(allowed, gl, 0.0)
            gu = tl.where(allowed, gu, 0.0)
            weights = tl.where(allowed, weights, 0.0)
            gk += tl.dot(tl.trans(gl).to(qv.dtype), qv, out_dtype=tl.float32)
            gk += tl.dot(tl.trans(gu).to(rv.dtype), rv, out_dtype=tl.float32)
            gv += tl.dot(tl.trans(weights).to(go.dtype), go, out_dtype=tl.float32)
        for block in range(causal_end, last_query_block):
            qs = block * BT + rows
            rowmask = qs < T
            qoff = bos_q + qs[:, None] * HQ * D + ds[None, :]
            qv = tl.load(q + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            rv = tl.load(r + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            go = tl.load(grad_o + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            if INLINE_DELTA:
                ov = tl.load(o + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
                bv = tl.load(barv + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
                dt = tl.sum(go.to(tl.float32) * ov, axis=1)[:, None]
                db = tl.sum(go.to(tl.float32) * bv, axis=1)[:, None]
            else:
                dt = tl.load(delta_t + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
                db = tl.load(delta_b + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            norm = tl.load(d1 + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            maxv = tl.load(m + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            bt = tl.load(bart + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
            safe_norm = tl.where(norm > 0.0, norm, 1.0)
            p = tl.math.exp2(scores - maxv) / safe_norm
            p = tl.where(rowmask[:, None], p, 0.0)
            rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
            a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
            delta = a - db
            correction = bt - rk
            gl = p * (a - dt + correction * delta) * scale
            gu = -p * delta
            weights = p * (1.0 + correction)
            gk += tl.dot(tl.trans(gl).to(qv.dtype), qv, out_dtype=tl.float32)
            gk += tl.dot(tl.trans(gu).to(rv.dtype), rv, out_dtype=tl.float32)
            gv += tl.dot(tl.trans(weights).to(go.dtype), go, out_dtype=tl.float32)
    else:
        for block in range(first_query_block, last_query_block):
            qs = block * BT + rows
            rowmask = qs < T
            qoff = bos_q + qs[:, None] * HQ * D + ds[None, :]
            qv = tl.load(q + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            rv = tl.load(r + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            go = tl.load(grad_o + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
            if INLINE_DELTA:
                ov = tl.load(o + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
                bv = tl.load(barv + qoff, rowmask[:, None] & (ds[None, :] < D), 0).to(tl.float32)
                dt = tl.sum(go.to(tl.float32) * ov, axis=1)[:, None]
                db = tl.sum(go.to(tl.float32) * bv, axis=1)[:, None]
            else:
                dt = tl.load(delta_t + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
                db = tl.load(delta_b + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            norm = tl.load(d1 + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            maxv = tl.load(m + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            bt = tl.load(bart + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
            allowed = (qs[:, None] >= cols[None, :]) & rowmask[:, None] & colmask[None, :]
            allowed = allowed & (cols[None, :] >= qs[:, None] - WINDOW + 1)
            scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
            scores = tl.where(allowed, scores, float("-inf"))
            safe_norm = tl.where(norm > 0.0, norm, 1.0)
            p = tl.where(allowed, tl.math.exp2(scores - maxv) / safe_norm, 0.0)
            rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
            a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
            rk = tl.where(allowed, rk, 0.0)
            a = tl.where(allowed, a, 0.0)
            delta = a - db
            correction = bt - rk
            gl = p * (a - dt + correction * delta) * scale
            gu = -p * delta
            weights = p * (1.0 + correction)
            gl = tl.where(allowed, gl, 0.0)
            gu = tl.where(allowed, gu, 0.0)
            weights = tl.where(allowed, weights, 0.0)
            gk += tl.dot(tl.trans(gl).to(qv.dtype), qv, out_dtype=tl.float32)
            gk += tl.dot(tl.trans(gu).to(rv.dtype), rv, out_dtype=tl.float32)
            gv += tl.dot(tl.trans(weights).to(go.dtype), go, out_dtype=tl.float32)
    # Per-query-head scratch has HQ in its token/head stride.
    outoff = (b * T * HQ + cols[:, None] * HQ + hq) * D + ds[None, :]
    tl.store(grad_k + outoff, gk, colmask[:, None] & (ds[None, :] < D))
    tl.store(grad_v + outoff, gv, colmask[:, None] & (ds[None, :] < D))


@triton.jit
def _reduce_gqa_kernel(gk_buf, gv_buf, gk, gv, N: tl.constexpr, HQ: tl.constexpr,
                       H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                       BLOCK: tl.constexpr, NUM_PROGRAMS: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    first = tl.program_id(0) * BLOCK
    # Keep the grid bounded and let each CTA process multiple contiguous tiles.
    for base in tl.range(first, N, NUM_PROGRAMS * BLOCK, loop_unroll_factor=1):
        x = base + offsets
        hdim = x % D
        rowhead = x // D
        h = rowhead % H
        bhrow = rowhead // H
        src = (bhrow * HQ + h * G) * D + hdim
        valid = x < N
        ak = tl.zeros((BLOCK,), tl.float32)
        av = tl.zeros((BLOCK,), tl.float32)
        for g in range(G):
            ak += tl.load(gk_buf + src + g * D, valid, 0).to(tl.float32)
            av += tl.load(gv_buf + src + g * D, valid, 0).to(tl.float32)
        tl.store(gk + x, ak, valid)
        tl.store(gv + x, av, valid)


_REDUCE_GQA_BLOCK = 128
_REDUCE_GQA_MAX_PROGRAMS = 1024


def _launch_reduce_gqa(gk_buf, gv_buf, gk, gv, n, hq, h, group, dim):
    """Reduce per-query-head scratch with a small grid and persistent CTAs."""
    if n <= 0:
        return
    block = _REDUCE_GQA_BLOCK
    programs = min(triton.cdiv(n, block), _REDUCE_GQA_MAX_PROGRAMS)
    _reduce_gqa_kernel[(programs,)](
        gk_buf, gv_buf, gk, gv, n, hq, h, group, dim, block, programs,
        num_warps=4,
    )


def parallax_backward(q, r, k, v, o, barv, d1, bart, m, grad_o, scale,
                      cu_seqlens=None, window_size_left=-1):
    """Compute gradients for Q, R, K and V using saved Parallax forward state."""
    if q.ndim != 4 or r.shape != q.shape or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("q/r/k/v must have compatible [B, T, H, D] shapes")
    bsz, total_t, hq, dim = q.shape
    hkv = k.shape[2]
    if k.shape[:2] != (bsz, total_t) or k.shape[-1] != dim or hq % hkv:
        raise ValueError("q and k/v shapes must agree and HQ must be divisible by H")
    if q.device.type != "npu":
        raise ValueError(f"Ascend Parallax backward requires NPU tensors, got {q.device}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported dtype {q.dtype}; expected float16 or bfloat16")
    if grad_o.shape != q.shape:
        raise ValueError("grad_o must have the same shape as q")

    if (
        cu_seqlens is None
        and window_size_left < 0
        and bsz == 1
        and total_t == 128
        and hq == 1
        and hkv == 1
        and dim == 128
        and q.dtype == torch.bfloat16
        and all(x.is_contiguous() for x in (q, r, k, v, o, barv, d1, bart, m, grad_o))
    ):
        from .short_backward import short_parallax_backward

        return short_parallax_backward(q, r, k, v, o, barv, d1, bart, m, grad_o, scale)

    G = hq // hkv
    bd = triton.next_power_of_2(dim)
    bt = dqr_bt = 64
    dqr_bs = 64
    dkv_bs = 64
    preprocess_bt = 64

    if cu_seqlens is None:
        segments = None
    else:
        if bsz != 1:
            raise ValueError("packed varlen backward requires B == 1")
        offsets = cu_seqlens.detach().cpu().tolist()
        segments = [(int(start), int(end - start), 0) for start, end in zip(offsets[:-1], offsets[1:])]
    inline_delta = segments is None and total_t <= 128
    use_staged = (
        segments is None
        and not inline_delta
        and supports_staged(q, window_size_left)
        and (G <= 8 or G % 8 == 0)
    )
    if use_staged:
        from .staged_backward import staged_parallax_backward

        return staged_parallax_backward(
            q, r, k, v, o, barv, d1, bart, m, grad_o, scale, window_size_left
        )

    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    grad_k_buf = torch.empty((bsz, total_t, hq, dim), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty_like(grad_k_buf)
    delta_t = None
    delta_b = None
    if not inline_delta:
        delta_t = torch.empty((bsz, total_t, hq), device=q.device, dtype=torch.float32)
        delta_b = torch.empty_like(delta_t)

    if not inline_delta:
        _preprocess_kernel[(bsz * triton.cdiv(total_t, preprocess_bt), hq)](
            o, barv, grad_o, delta_t, delta_b, total_t, hq, dim, bd, preprocess_bt,
            num_warps=4,
        )
    if segments is None:
        # Keep the batch dimension in the grid so each derivative kernel launches once.
        dqr_grid = (triton.cdiv(total_t, dqr_bt), bsz * hq)
        dkv_grid = (triton.cdiv(total_t, dkv_bs), bsz * hq)
        with backward_streams(q.device, concurrent=total_t >= 256) as (dqr_context, dkv_context):
            with dqr_context:
                _dqr_kernel[dqr_grid](
                    q, r, k, v, o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
                    scale, total_t, hq, hkv, G, dim, bd, window_size_left, dqr_bt, dqr_bs, inline_delta,
                    num_warps=4,
                )
            with dkv_context:
                _dkv_kernel[dkv_grid](
                    q, r, k, v, o, barv, d1, bart, m, delta_t, delta_b, grad_o, grad_k_buf, grad_v_buf,
                    scale, total_t, hq, hkv, G, dim, bd, window_size_left, bt, dkv_bs, inline_delta,
                    num_warps=4,
                )
    else:
        for bos, t, bi in segments:
            if t == 0:
                continue
            # Packed segments are isolated in their respective tensor views.
            _dqr_kernel[(triton.cdiv(t, dqr_bt), hq)](
                q[bi, bos:], r[bi, bos:], k[bi, bos:], v[bi, bos:], o[bi, bos:], barv[bi, bos:],
                d1[bi, bos:], bart[bi, bos:], m[bi, bos:],
                delta_t[bi, bos:], delta_b[bi, bos:], grad_o[bi, bos:],
                grad_q[bi, bos:], grad_r[bi, bos:], scale, t, hq, hkv, G, dim, bd,
                window_size_left, dqr_bt, dqr_bs, False, num_warps=4,
            )
            _dkv_kernel[(triton.cdiv(t, dkv_bs), hq)](
                q[bi, bos:], r[bi, bos:], k[bi, bos:], v[bi, bos:], o[bi, bos:], barv[bi, bos:],
                d1[bi, bos:], bart[bi, bos:], m[bi, bos:],
                delta_t[bi, bos:], delta_b[bi, bos:], grad_o[bi, bos:],
                grad_k_buf[bi, bos:], grad_v_buf[bi, bos:], scale, t, hq, hkv, G, dim, bd,
                window_size_left, bt, dkv_bs, False, num_warps=4,
            )

    if G == 1:
        # DKV already holds one contribution per query head in token-major storage.
        grad_k = grad_k_buf
        grad_v = grad_v_buf
    else:
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        n = bsz * total_t * hkv * dim
        _launch_reduce_gqa(grad_k_buf, grad_v_buf, grad_k, grad_v, n, hq, hkv, G, dim)
    return grad_q, grad_r, grad_k, grad_v
