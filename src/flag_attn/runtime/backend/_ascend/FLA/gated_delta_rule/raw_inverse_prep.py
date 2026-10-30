# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-length TLE-Raw inverse preparation for eligible GDN workloads.

This module is deliberately separate from the generic prep implementation.
It materializes a dense contiguous FP32 64x64 chunk-local KKT matrix, applies
the Raw inverse to its two diagonal 32x32 blocks, stitches the full 64x64
inverse, then computes the same two-pass affine outputs as the generic path.
The dense KKT write avoids the severe serialization seen with dynamic
compact-block addressing on CANN 9.1. Dispatch is restricted to the
large-workload class and can be disabled with
``FLAG_ATTN_ASCEND_GDN_RAW_INVERSE=0``.
"""
from __future__ import annotations

import torch
import triton

from .fast_launch import fast_launch
import triton.language as tl

try:
    # Dual-diagonal Raw inverse (a single CTA inverts both 32x32 diagonal
    # blocks of the 64x64 tile).  It needs triton.experimental.tle; fall back
    # to the batch2 kernel where the TLE extension is not available.
    from .raw_inverse_dual import (
        launch_inverse64_rawdiag_dual as launch_inverse64_rawdiag_outer,
    )
except ImportError:  # pragma: no cover - depends on the Triton build
    from .raw_inverse_ascend import (
        launch_inverse64_rawdiag_outer_batch2 as launch_inverse64_rawdiag_outer,
    )


@triton.jit(do_not_specialize=["T"])
def _kkt_blocks_kernel(ke, kr, L_FULL, T,
                       H: tl.constexpr, K: tl.constexpr,
                       NT: tl.constexpr, BT: tl.constexpr):
    """Write dense FP32 `[BT,BT]` strictly-lower `ke @ kr.T` per chunk."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    base = (i_b * T) * H * K + i_h * K
    p_ke = tl.make_block_ptr(
        ke + base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_l = tl.dot(tl.load(p_ke, boundary_check=(0, 1)),
                 tl.load(p_kr, boundary_check=(0, 1)))
    r = tl.arange(0, BT)
    c = tl.arange(0, BT)
    lower = r[:, None] > c[None, :]
    b_l = tl.where(lower, b_l, 0.0).to(tl.float32)

    tile = (i_b * NT + i_t) * H + i_h
    # Keep the KKT tile in its native contiguous 64x64 layout. Dynamic
    # three-block offset selection serialized this producer on CANN 9.1.
    tl.store(L_FULL + tile * (BT * BT) + r[:, None] * BT + c[None, :], b_l)


@triton.jit(do_not_specialize=["T"])
def _kkt_blocks_one_job_batched(ke, kr, L_FULL, T,
                                H: tl.constexpr, K: tl.constexpr,
                                NT: tl.constexpr, BT: tl.constexpr,
                                i_t, i_bh):
    i_b, i_h = i_bh // H, i_bh % H
    base = (i_b * T) * H * K + i_h * K
    p_ke = tl.make_block_ptr(
        ke + base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_l = tl.dot(
        tl.load(p_ke, boundary_check=(0, 1), padding_option="zero"),
        tl.load(p_kr, boundary_check=(0, 1), padding_option="zero"),
    )
    r = tl.arange(0, BT)
    c = tl.arange(0, BT)
    valid_rows = (i_t * BT + r < T)[:, None]
    valid_cols = (i_t * BT + c < T)[None, :]
    lower = r[:, None] > c[None, :]
    b_l = tl.where(lower & valid_rows & valid_cols, b_l, 0.0).to(tl.float32)
    tile = (i_b * NT + i_t) * H + i_h
    tl.store(L_FULL + tile * (BT * BT) + r[:, None] * BT + c[None, :], b_l)


@triton.jit(do_not_specialize=["T"])
def _kkt_blocks_batched_kernel(ke, kr, L_FULL, T,
                               H: tl.constexpr, K: tl.constexpr,
                               NT: tl.constexpr, BT: tl.constexpr,
                               B: tl.constexpr, JOBS: tl.constexpr,
                               BATCH: tl.constexpr):
    first = tl.program_id(0) * BATCH
    for j in tl.static_range(BATCH):
        job = first + j
        if job < JOBS:
            i_t = job // (B * H)
            i_bh = job % (B * H)
            _kkt_blocks_one_job_batched(
                ke, kr, L_FULL, T, H, K, NT, BT, i_t, i_bh
            )


def launch_kkt_blocks_batched(ke: torch.Tensor, kr: torch.Tensor,
                              L_full: torch.Tensor, T: int, H: int, K: int,
                              NT: int, BT: int, batch_size: int = 4) -> None:
    """Produce dense KKT tiles with 2 or 4 independent jobs per CTA."""
    if batch_size < 1 or batch_size > 16:
        raise ValueError("KKT batch_size must be in [1, 16]")
    batch = ke.shape[0]
    jobs = batch * NT * H
    fast_launch(_kkt_blocks_batched_kernel, (triton.cdiv(jobs, batch_size),), 4, 2)(
        ke, kr, L_full, T, H, K, NT, BT, batch, jobs, batch_size,
    )


@triton.jit
def _stitch_inverse_kernel(L_FULL, DIAG, INVERSE):
    tile = tl.program_id(0)
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    a = tl.load(DIAG + tile * 2112 + r[:, None] * 32 + c[None, :])
    d = tl.load(DIAG + tile * 2112 + 1056 + r[:, None] * 32 + c[None, :])
    cb = tl.load(L_FULL + tile * 4096 + (r[:, None] + 32) * 64 + c[None, :])
    lo = -tl.dot(tl.dot(d, cb), a)
    base = INVERSE + tile * 4096 + r[:, None] * 64 + c[None, :]
    tl.store(base, a)
    tl.store(base + 32 * 64, lo)
    tl.store(base + 32, tl.full((32, 32), 0.0, tl.float32))
    tl.store(base + 32 * 64 + 32, d)


@triton.jit
def _stitch_inverse_one_tile(L_FULL, DIAG, INVERSE, tile):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    a = tl.load(DIAG + tile * 2112 + r[:, None] * 32 + c[None, :])
    d = tl.load(DIAG + tile * 2112 + 1056 + r[:, None] * 32 + c[None, :])
    cb = tl.load(L_FULL + tile * 4096 + (r[:, None] + 32) * 64 + c[None, :])
    lo = -tl.dot(tl.dot(d, cb), a)
    base = INVERSE + tile * 4096 + r[:, None] * 64 + c[None, :]
    tl.store(base, a)
    tl.store(base + 32 * 64, lo)
    tl.store(base + 32, tl.full((32, 32), 0.0, tl.float32))
    tl.store(base + 32 * 64 + 32, d)


@triton.jit
def _stitch_inverse_batched_kernel(L_FULL, DIAG, INVERSE,
                                   JOBS: tl.constexpr,
                                   BATCH: tl.constexpr):
    first = tl.program_id(0) * BATCH
    for j in tl.static_range(BATCH):
        tile = first + j
        if tile < JOBS:
            _stitch_inverse_one_tile(L_FULL, DIAG, INVERSE, tile)


def launch_stitch_inverse_batched(L_full: torch.Tensor,
                                  diag: torch.Tensor,
                                  inverse: torch.Tensor,
                                  batch_size: int = 4) -> None:
    """Stitch a small static batch of independent inverse tiles per CTA."""
    if batch_size < 1 or batch_size > 4:
        raise ValueError("stitch batch_size must be in [1, 4]")
    jobs = L_full.shape[0]
    fast_launch(_stitch_inverse_batched_kernel, (triton.cdiv(jobs, batch_size),), 4, 1)(
        L_full, diag, inverse, jobs, batch_size,
    )


def launch_stitch_inverse_batch4(L_full: torch.Tensor,
                                 diag: torch.Tensor,
                                 inverse: torch.Tensor) -> None:
    """Default four-tile stitch launch for the eligible Raw prep path."""
    launch_stitch_inverse_batched(L_full, diag, inverse, batch_size=4)


@triton.jit
def _affine_from_inverse_body(
    qe, ke, kr, vb, INVERSE,
    qe2, au, cu, mw, a_out,
    i_t, tile, base_k, base_v, base_bt, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NT: tl.constexpr, BT: tl.constexpr,
):
    r = tl.arange(0, BT)
    p_ai = tl.make_block_ptr(INVERSE + tile * BT * BT,
                             (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    b_ai = tl.load(p_ai)

    p_ke = tl.make_block_ptr(ke + base_k, (T, K), (H * K, 1),
                             (i_t * BT, 0), (BT, K), (1, 0))
    p_kr = tl.make_block_ptr(kr + base_k, (K, T), (1, H * K),
                             (0, i_t * BT), (K, BT), (0, 1))
    b_ke = tl.load(p_ke, boundary_check=(0, 1))
    b_kr = tl.load(p_kr, boundary_check=(0, 1))
    b_w = tl.dot(b_ai.to(b_ke.dtype), b_ke)
    b_u = tl.dot(b_ai.to(b_ke.dtype), tl.load(tl.make_block_ptr(
        vb + base_v, (T, V), (H * V, 1), (i_t * BT, 0),
        (BT, V), (1, 0)), boundary_check=(0, 1)))
    b_w = b_w.to(a_out.dtype.element_ty)
    b_u = b_u.to(a_out.dtype.element_ty)

    p_qe = tl.make_block_ptr(qe + base_k, (T, K), (H * K, 1),
                             (i_t * BT, 0), (BT, K), (1, 0))
    b_qe = tl.load(p_qe, boundary_check=(0, 1))
    o_t = i_t * BT + r
    b_a = tl.dot(b_qe, b_kr)
    b_a = tl.where(o_t[:, None] >= o_t[None, :], b_a, 0.0)
    b_a = b_a.to(a_out.dtype.element_ty)
    tl.store(tl.make_block_ptr(a_out + base_bt, (T, BT), (H * BT, 1),
                              (i_t * BT, 0), (BT, BT), (1, 0)),
             b_a, boundary_check=(0, 1))

    b_qw = tl.dot(b_a, b_w.to(b_a.dtype))
    tl.store(tl.make_block_ptr(qe2 + base_k, (T, K), (H * K, 1),
                              (i_t * BT, 0), (BT, K), (1, 0)),
             (b_qe.to(tl.float32) - b_qw).to(qe2.dtype.element_ty),
             boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(au + base_v, (T, V), (H * V, 1),
                              (i_t * BT, 0), (BT, V), (1, 0)),
             tl.dot(b_a, b_u).to(au.dtype.element_ty), boundary_check=(0, 1))

    cbase_v = tile * K * V
    p_cu = tl.make_block_ptr(cu + cbase_v, (K, V), (V, 1),
                             (0, 0), (K, V), (1, 0))
    tl.store(p_cu, tl.dot(b_kr, b_u.to(b_kr.dtype)).to(p_cu.dtype.element_ty))
    cbase_m = tile * K * K
    p_mw = tl.make_block_ptr(mw + cbase_m, (K, K), (K, 1),
                             (0, 0), (K, K), (1, 0))
    tl.store(p_mw, tl.dot(b_kr, b_w.to(b_kr.dtype)).to(p_mw.dtype.element_ty))


@triton.jit
def _prep_affine_from_inverse_kernel(
    qe, ke, kr, vb, INVERSE,
    qe2, au, cu, mw, a_out,
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NT: tl.constexpr, BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    base = i_b * T * H + i_h
    tile = (i_b * NT + i_t) * H + i_h
    _affine_from_inverse_body(
        qe, ke, kr, vb, INVERSE, qe2, au, cu, mw, a_out,
        i_t, tile, base * K, base * V, base * BT, T, H, K, V, NT, BT,
    )


@triton.jit
def _prep_affine_from_inverse_batched_kernel(
    qe, ke, kr, vb, INVERSE,
    qe2, au, cu, mw, a_out,
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NT: tl.constexpr, BT: tl.constexpr,
    BATCH: tl.constexpr,
):
    i_g, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    base = i_b * T * H + i_h
    for k in tl.static_range(0, BATCH):
        i_t = i_g * BATCH + k
        tile = (i_b * NT + i_t) * H + i_h
        _affine_from_inverse_body(
            qe, ke, kr, vb, INVERSE, qe2, au, cu, mw, a_out,
            i_t, tile, base * K, base * V, base * BT, T, H, K, V, NT, BT,
        )


# The affine kernel is limited by per-program scalar/scheduling overhead, not
# by MACs or HBM.  Folding four consecutive chunks into one program amortises
# that overhead, and BATCH=4 is the empirical value for this backend; larger
# batches do not reduce the program count enough to pay for the longer body.
_AFFINE_BATCH = 4


def _launch_prep_affine_from_inverse(
    qe, ke, kr, vb, inverse, qe2, au, cu, mw, a_out,
    T, H, K, V, NT, BT,
):
    B = qe.shape[0]
    batch = _AFFINE_BATCH if NT % _AFFINE_BATCH == 0 else 1
    if batch > 1:
        fast_launch(_prep_affine_from_inverse_batched_kernel, (NT // batch, B * H), 4, 2)(
            qe, ke, kr, vb, inverse, qe2, au, cu, mw, a_out,
            T, H, K, V, NT, BT, batch,
        )
    else:
        fast_launch(_prep_affine_from_inverse_kernel, (NT, B * H), 4, 2)(
            qe, ke, kr, vb, inverse, qe2, au, cu, mw, a_out,
            T, H, K, V, NT, BT,
        )





def raw_inverse_prep_affine_ascend(qe, ke, kr, vb, chunk_size, output_dtype=None):
    """Replace affine prep with dense KKT, Raw inverse, stitch, and affine prep."""
    B, T, H, K = ke.shape
    V = vb.shape[-1]
    BT = int(chunk_size)
    if BT != 64 or V != K or K > 128:
        raise ValueError("Raw prep supports only fixed BT=64 and K=V<=128")
    NT = triton.cdiv(T, BT)
    output_dtype = output_dtype or ke.dtype
    L_full = torch.empty((B, NT, H, BT, BT),
                         dtype=torch.float32, device=ke.device)
    diag = torch.empty((B, NT, H, 2, 33, 32),
                       dtype=torch.float32, device=ke.device)
    inverse = torch.empty((B, NT, H, BT, BT),
                          dtype=torch.float32, device=ke.device)
    qe2 = qe.new_empty((B, T, H, K))
    au = torch.empty((B, T, H, V), dtype=output_dtype, device=vb.device)
    cu = qe.new_empty((B, NT, H, K, V))
    mw = qe.new_empty((B, NT, H, K, K))
    a_out = torch.empty((B, T, H, BT), dtype=output_dtype, device=ke.device)
    # Batch-size thresholds for the chunk-parallel KKT/stitch kernels; the Raw
    # diagonal inverse stays as is (it is bound inside the CCEC inverse).
    from .raw_inverse_batch import (
        launch_kkt_blocks_dispatch,
        launch_stitch_inverse_dispatch,
    )

    jobs = B * NT * H
    launch_kkt_blocks_dispatch(ke, kr, L_full, T, H, K, NT, BT, jobs)
    L_flat = L_full.view(-1, BT, BT)
    diag_flat = diag.view(-1, 2, 33, 32)
    launch_inverse64_rawdiag_outer(L_flat, diag_flat)
    launch_stitch_inverse_dispatch(L_flat, diag_flat, inverse, H, NT, jobs)
    _launch_prep_affine_from_inverse(
        qe, ke, kr, vb, inverse, qe2, au, cu, mw, a_out,
        T, H, K, V, NT, BT,
    )
    return qe2, au, cu, mw, a_out


@triton.jit
def _affine_body_trim(
    q, v, E1, B1, ke, kr, INVERSE,
    qe2, au, cu, mw, a_out,
    i_t, tile, base_k, base_v, base_bt, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NT: tl.constexpr, BT: tl.constexpr, REP: tl.constexpr,
    FACTOR_T: tl.constexpr,
    SQB: tl.constexpr, SQT: tl.constexpr, SQH: tl.constexpr, SQK: tl.constexpr,
    SVB: tl.constexpr, SVT: tl.constexpr, SVH: tl.constexpr, SVV: tl.constexpr,
):
    """Affine body for the v1 fold trim: qe/vb recomputed from q/v/factors.

    The dot sequence, dtypes and store layout are identical to
    ``_affine_from_inverse_body``; only the qe/vb loads are replaced by the
    elementwise products q*exp(g_cum) and v*beta, which the fold would
    otherwise have materialised into GM.
    """
    r = tl.arange(0, BT)
    p_ai = tl.make_block_ptr(INVERSE + tile * BT * BT,
                             (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    b_ai = tl.load(p_ai)

    p_ke = tl.make_block_ptr(ke + base_k, (T, K), (H * K, 1),
                             (i_t * BT, 0), (BT, K), (1, 0))
    p_kr = tl.make_block_ptr(kr + base_k, (K, T), (1, H * K),
                             (0, i_t * BT), (K, BT), (0, 1))
    b_ke = tl.load(p_ke, boundary_check=(0, 1))
    b_kr = tl.load(p_kr, boundary_check=(0, 1))

    i_b = base_k // (T * H * K)
    i_h = (base_k // K) % H
    fr = i_b * H * FACTOR_T + i_h * FACTOR_T + i_t * BT + r
    b_e = tl.load(E1 + fr).to(tl.float32)
    b_beta = tl.load(B1 + fr).to(tl.float32)

    i_hg = i_h // REP
    p_q = tl.make_block_ptr(q + i_b * SQB + i_hg * SQH, (T, K), (SQT, SQK),
                            (i_t * BT, 0), (BT, K), (1, 0))
    b_qe = (tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
            * b_e[:, None]).to(ke.dtype.element_ty)
    p_v = tl.make_block_ptr(v + i_b * SVB + i_h * SVH, (T, V), (SVT, SVV),
                            (i_t * BT, 0), (BT, V), (1, 0))
    b_vb = (tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
            * b_beta[:, None]).to(ke.dtype.element_ty)

    b_w = tl.dot(b_ai.to(b_ke.dtype), b_ke)
    b_u = tl.dot(b_ai.to(b_ke.dtype), b_vb)
    b_w = b_w.to(a_out.dtype.element_ty)
    b_u = b_u.to(a_out.dtype.element_ty)

    o_t = i_t * BT + r
    b_a = tl.dot(b_qe, b_kr)
    b_a = tl.where(o_t[:, None] >= o_t[None, :], b_a, 0.0)
    b_a = b_a.to(a_out.dtype.element_ty)
    tl.store(tl.make_block_ptr(a_out + base_bt, (T, BT), (H * BT, 1),
                              (i_t * BT, 0), (BT, BT), (1, 0)),
             b_a, boundary_check=(0, 1))

    b_qw = tl.dot(b_a, b_w.to(b_a.dtype))
    tl.store(tl.make_block_ptr(qe2 + base_k, (T, K), (H * K, 1),
                              (i_t * BT, 0), (BT, K), (1, 0)),
             (b_qe.to(tl.float32) - b_qw).to(qe2.dtype.element_ty),
             boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(au + base_v, (T, V), (H * V, 1),
                              (i_t * BT, 0), (BT, V), (1, 0)),
             tl.dot(b_a, b_u).to(au.dtype.element_ty), boundary_check=(0, 1))

    cbase_v = tile * K * V
    p_cu = tl.make_block_ptr(cu + cbase_v, (K, V), (V, 1),
                             (0, 0), (K, V), (1, 0))
    tl.store(p_cu, tl.dot(b_kr, b_u.to(b_kr.dtype)).to(p_cu.dtype.element_ty))
    cbase_m = tile * K * K
    p_mw = tl.make_block_ptr(mw + cbase_m, (K, K), (K, 1),
                             (0, 0), (K, K), (1, 0))
    tl.store(p_mw, tl.dot(b_kr, b_w.to(b_kr.dtype)).to(p_mw.dtype.element_ty))


@triton.jit
def _prep_affine_from_inverse_trim_kernel(
    q, v, E1, B1, ke, kr, INVERSE,
    qe2, au, cu, mw, a_out,
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NT: tl.constexpr, BT: tl.constexpr, REP: tl.constexpr,
    FACTOR_T: tl.constexpr,
    SQB: tl.constexpr, SQT: tl.constexpr, SQH: tl.constexpr, SQK: tl.constexpr,
    SVB: tl.constexpr, SVT: tl.constexpr, SVH: tl.constexpr, SVV: tl.constexpr,
    BATCH: tl.constexpr,
):
    i_g, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    base = i_b * T * H + i_h
    for j in tl.static_range(0, BATCH):
        i_t = i_g * BATCH + j
        tile = (i_b * NT + i_t) * H + i_h
        _affine_body_trim(
            q, v, E1, B1, ke, kr, INVERSE, qe2, au, cu, mw, a_out,
            i_t, tile, base * K, base * V, base * BT, T,
            H, K, V, NT, BT, REP, FACTOR_T,
            SQB, SQT, SQH, SQK, SVB, SVT, SVH, SVV,
        )


def raw_inverse_prep_affine_trim_ascend(
    q, v, e1, b1, factor_t, rep, ke, kr, chunk_size, output_dtype=None,
):
    """Raw inverse prep for the v1 fold trim (no qe/vb in GM).

    ``q``/``v`` are the caller's (possibly head-first strided) source tensors
    and ``e1``/``b1`` the fp32 exp(g_cum)/beta factor buffers produced by the
    fold's factor pass.
    """
    B, T, H, K = ke.shape
    V = kr.shape[-1]
    BT = int(chunk_size)
    if BT != 64 or V != K or K > 128:
        raise ValueError("Raw prep supports only fixed BT=64 and K=V<=128")
    NT = triton.cdiv(T, BT)
    output_dtype = output_dtype or ke.dtype
    L_full = torch.empty((B, NT, H, BT, BT),
                         dtype=torch.float32, device=ke.device)
    diag = torch.empty((B, NT, H, 2, 33, 32),
                       dtype=torch.float32, device=ke.device)
    inverse = torch.empty((B, NT, H, BT, BT),
                          dtype=torch.float32, device=ke.device)
    qe2 = q.new_empty((B, T, H, K))
    au = torch.empty((B, T, H, V), dtype=output_dtype, device=ke.device)
    cu = q.new_empty((B, NT, H, K, V))
    mw = q.new_empty((B, NT, H, K, K))
    a_out = torch.empty((B, T, H, BT), dtype=output_dtype, device=ke.device)
    from .raw_inverse_batch import (
        launch_kkt_blocks_dispatch,
        launch_stitch_inverse_dispatch,
    )

    jobs = B * NT * H
    launch_kkt_blocks_dispatch(ke, kr, L_full, T, H, K, NT, BT, jobs)
    L_flat = L_full.view(-1, BT, BT)
    diag_flat = diag.view(-1, 2, 33, 32)
    launch_inverse64_rawdiag_outer(L_flat, diag_flat)
    launch_stitch_inverse_dispatch(L_flat, diag_flat, inverse, H, NT, jobs)
    batch = 4 if NT % 4 == 0 else 1
    fast_launch(_prep_affine_from_inverse_trim_kernel, (NT // batch, B * H), 4, 2)(
        q, v, e1, b1, ke, kr, inverse,
        qe2, au, cu, mw, a_out,
        T, H, K, V, NT, BT, rep, factor_t,
        *q.stride(), *v.stride(),
        batch,
    )
    return qe2, au, cu, mw, a_out


__all__ = ["raw_inverse_prep_affine_ascend",
           "raw_inverse_prep_affine_trim_ascend"]
