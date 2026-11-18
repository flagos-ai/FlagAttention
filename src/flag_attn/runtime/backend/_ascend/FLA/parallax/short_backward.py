"""Small fixed-length Triton backward path for Parallax training."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _short_preprocess_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    gl_ptr,
    gu_ptr,
    weights_ptr,
    scale,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    """Compute row-wise derivatives/weights for the parallel gradient kernels."""
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    ds = tl.arange(0, BD)
    row_mask = rows < 128

    qv = tl.load(q + rows[:, None] * BD + ds[None, :], row_mask[:, None], 0)
    rv = tl.load(r + rows[:, None] * BD + ds[None, :], row_mask[:, None], 0)
    go = tl.load(grad_o + rows[:, None] * BD + ds[None, :], row_mask[:, None], 0)
    ov = tl.load(o + rows[:, None] * BD + ds[None, :], row_mask[:, None], 0).to(tl.float32)
    bv = tl.load(barv + rows[:, None] * BD + ds[None, :], row_mask[:, None], 0).to(tl.float32)
    kv = tl.load(k + cols[:, None] * BD + ds[None, :])
    vv = tl.load(v + cols[:, None] * BD + ds[None, :])

    scale_log2 = scale * 1.4426950216
    scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * scale_log2
    rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
    a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
    allowed = (rows[:, None] >= cols[None, :]) & row_mask[:, None]
    scores = tl.where(allowed, scores, -float("inf"))
    rk = tl.where(allowed, rk, 0.0)
    a = tl.where(allowed, a, 0.0)

    row_m = tl.load(m + rows, row_mask, 0).to(tl.float32)[:, None]
    row_d1 = tl.load(d1 + rows, row_mask, 0).to(tl.float32)[:, None]
    row_bart = tl.load(bart + rows, row_mask, 0).to(tl.float32)[:, None]
    dt = tl.sum(go.to(tl.float32) * ov, axis=1, keep_dims=True)
    db = tl.sum(go.to(tl.float32) * bv, axis=1, keep_dims=True)
    prob = tl.exp2(scores - row_m) / tl.where(row_d1 > 0.0, row_d1, 1.0)
    prob = tl.where(allowed, prob, 0.0)
    delta = a - db
    correction = row_bart - rk
    gl = prob * (a - dt + correction * delta) * scale
    gu = -prob * delta
    weights = prob * (1.0 + correction)

    score_offsets = rows[:, None] * 128 + cols[None, :]
    tl.store(gl_ptr + score_offsets, gl, row_mask[:, None])
    tl.store(gu_ptr + score_offsets, gu, row_mask[:, None])
    tl.store(weights_ptr + score_offsets, weights, row_mask[:, None])


@triton.jit
def _short_dqr_finalize(
    k,
    gl_ptr,
    gu_ptr,
    grad_q,
    grad_r,
    tile_id,
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    """Compute dQ/dR from saved BF16 row derivatives."""
    rows = tile_id * BM + tl.arange(0, BM)
    cols = tl.arange(0, 128)
    ds = tl.arange(0, BD)
    row_mask = rows < 128
    kv = tl.load(k + cols[:, None] * BD + ds[None, :])
    offsets = rows[:, None] * 128 + cols[None, :]
    gl = tl.load(gl_ptr + offsets, row_mask[:, None], 0)
    gu = tl.load(gu_ptr + offsets, row_mask[:, None], 0)
    gq = tl.dot(gl, kv, out_dtype=tl.float32)
    gr = tl.dot(gu, kv, out_dtype=tl.float32)
    out_offsets = rows[:, None] * BD + ds[None, :]
    tl.store(grad_q + out_offsets, gq, row_mask[:, None])
    tl.store(grad_r + out_offsets, gr, row_mask[:, None])


@triton.jit
def _short_dkv_finalize(
    q,
    r,
    grad_o,
    gl_ptr,
    gu_ptr,
    weights_ptr,
    grad_k,
    grad_v,
    tile_id,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    """Reduce saved row derivatives over all queries for one key tile."""
    rows = tl.arange(0, 128)
    cols = tile_id * BK + tl.arange(0, BK)
    ds = tl.arange(0, BD)
    col_mask = cols < 128

    qv = tl.load(q + rows[:, None] * BD + ds[None, :])
    rv = tl.load(r + rows[:, None] * BD + ds[None, :])
    go = tl.load(grad_o + rows[:, None] * BD + ds[None, :])
    score_offsets = rows[:, None] * 128 + cols[None, :]
    gl = tl.load(gl_ptr + score_offsets, col_mask[None, :], 0)
    gu = tl.load(gu_ptr + score_offsets, col_mask[None, :], 0)
    weights = tl.load(weights_ptr + score_offsets, col_mask[None, :], 0)

    gk = tl.dot(tl.trans(gl), qv, out_dtype=tl.float32)
    gk += tl.dot(tl.trans(gu), rv, out_dtype=tl.float32)
    gv = tl.dot(tl.trans(weights), go, out_dtype=tl.float32)
    out_offsets = cols[:, None] * BD + ds[None, :]
    tl.store(grad_k + out_offsets, gk, col_mask[:, None])
    tl.store(grad_v + out_offsets, gv, col_mask[:, None])


@triton.jit
def _short_stage2_kernel(
    q,
    r,
    k,
    grad_o,
    gl_ptr,
    gu_ptr,
    weights_ptr,
    grad_q,
    grad_r,
    grad_k,
    grad_v,
    NDQR: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    """Run dQ/dR and dK/dV CTA groups concurrently after row preprocessing."""
    pid = tl.program_id(0)
    if pid < NDQR:
        _short_dqr_finalize(k, gl_ptr, gu_ptr, grad_q, grad_r, pid, BM, BD)
    else:
        _short_dkv_finalize(
            q, r, grad_o, gl_ptr, gu_ptr, weights_ptr, grad_k, grad_v,
            pid - NDQR, BK, BD,
        )


def short_parallax_backward(
    q, r, k, v, o, barv, d1, bart, m, grad_o, scale,
    block_m=16, block_k=32, num_warps=4,
):
    """Backward for the official B1/Hq1/Hkv1/T128/D128 BF16 case."""
    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    shape = (128, 128)
    gl = torch.empty(shape, device=q.device, dtype=q.dtype)
    gu = torch.empty_like(gl)
    weights = torch.empty_like(gl)
    bm = block_m
    _short_preprocess_kernel[(triton.cdiv(128, bm),)](
        q,
        r,
        k,
        v,
        o,
        barv,
        d1,
        bart,
        m,
        grad_o,
        gl,
        gu,
        weights,
        float(scale),
        BM=bm,
        BD=128,
        num_warps=num_warps,
        num_stages=1,
    )
    bk = block_k
    _short_stage2_kernel[(triton.cdiv(128, bm) + triton.cdiv(128, bk),)](
        q,
        r,
        k,
        grad_o,
        gl,
        gu,
        weights,
        grad_q,
        grad_r,
        grad_k,
        grad_v,
        NDQR=triton.cdiv(128, bm),
        BM=bm,
        BK=bk,
        BD=128,
        num_warps=num_warps,
        num_stages=1,
    )
    return grad_q, grad_r, grad_k, grad_v
