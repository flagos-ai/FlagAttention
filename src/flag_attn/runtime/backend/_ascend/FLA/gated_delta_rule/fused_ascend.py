# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused Ascend stages for the GDN v1 fast path.

The pipeline is three Triton launches:

1. ``fold`` (grid ``(chunks, B*H)``) -- a **vector-only** kernel: chunk-local
   cumsum (fp32), then all per-token decay/beta folds written out in the
   input dtype (FP32 for FP16 inputs): ``q_e = q*exp(g)``,
   ``k_e = k*beta*exp(g)``, ``k_r = k*exp(-g)``,
   ``v_beta = v*beta``, ``k_s = k*exp(G-g)`` (``G`` = chunk-total log decay, so
   the state recurrence becomes ``h <- exp(G) h + k_s @ v_new`` with no
   per-token decay in the sequential loop).  No ``tl.dot`` in this kernel.
2. ``prep_w_u`` (grid ``(chunks, B*H)``) -- a **cube-only** kernel: KKT
   ``L = k_e @ k_r^T``, Newton inverse, ``A = tril(q_e @ k_r^T)``,
   ``w = A_inv @ k_e`` and ``u = A_inv @ v_beta`` from block-pointer loads.
   ``A_inv`` never hits GM.
3. ``fwd_h_o`` (grid ``(V/BV, N*H)``) -- state scan fused with the output
   stage; ``h`` and ``v_new`` never hit GM.

Why fold and prep are separate kernels
--------------------------------------
``prep_w_u`` is cube-only because both of its 2D dot operands (``k_e`` and
``k_r``) come straight from GM as tiles.  A dot operand produced by the vector
pipeline has to be re-laid out before it can feed a cube dot, which is
expensive per program per tile on this backend; feeding the same folded values
into *stores* (vector layout) is cheap.  The folds therefore live in a vector
kernel and the prep kernel stays cube-only.

The fold itself is split into ``factor`` + ``fold`` because several vector
patterns serialize across the grid on this backend instead of running in
parallel: a bf16 2D gather, two or more 2D gathers, and arithmetic on a 1D
vector (e.g. ``exp``) whose result is then broadcast into a 2D tile.
Block-ptr 2D loads/stores, 2D ``cumsum``, multiple stores and plain 1D-load +
``[:, None]`` broadcast multiplies all parallelize fine.

The fold is therefore two kernels:

* ``factor``: 1D only -- ``g_cum``, ``exp(g_cum)``, ``exp(-g_cum)``, fp32
  ``beta`` and ``beta*exp(g_cum)``, each written to a 1D fp32 buffer;
* ``fold``: plain 1D block loads of those buffers + ``[:, None]`` broadcast
  multiplies + 4 stores.  No 1D arithmetic is ever broadcast, no gathers.

Fast-path invariants (see ``wy_fast_ascend.py``): fp32 buffers for
anything gathered as a 2D tile, so 2D gathers read fp32 rather than fp16/bf16;
gathered factors never feed a dot; K must cover the whole head dimension in a
single dot tile; multi-trip dot loops are unreliable on this backend; no
autotune.

Two-pass state scan (``_affine_kernel`` + ``_fwd_h_o_2pass_kernel``, enabled for
``NT >= 32`` / ``FLAG_ATTN_ASCEND_GDN_TWOPASS=1``)
    The delta-rule chunk recurrence is affine *with a matrix transition*:
    ``S' = a S + kr (u - w S) = (a I - a kr w) S + a kr u``.  Pass 1
    (chunk-parallel) precomputes ``qe2 = qe - A w``, ``Au = A u``,
    ``Cu = kr u`` and ``Mw = kr w``; pass 2 then needs only two dots per chunk
    (``o = qe2 S + Au``, ``S' = a (S + Cu - Mw S)``) instead of four, halving
    the serial chain.  Note that dropping the ``Mw S`` term (i.e. using only
    ``C_c = sum_j exp(G-g_j) k_j u_j^T`` and scalar prefix factors) is *not*
    exact; a fully parallel prefix scan would have to compose the K x K
    matrices per chunk and is more expensive than the sequential scan at these
    sizes.  The affine pass costs extra device time plus a host launch, so the
    two-pass path is enabled only once the serial chain is long (NT >= 32);
    below that it is a net loss.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from .fast_launch import fast_launch

from flag_attn.FLA.index import prepare_chunk_indices


# ---------------------------------------------------------------------------
# kernel 1: chunk cumsum + all per-token folds (vector-only, no dots)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _factor_kernel(
    g,
    beta,
    g_cum,
    e1,
    r1,
    b1,
    be1,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """1D-only per-token factors (all parallel-safe).

    Writes the chunk-local cumsum ``g_cum`` plus ``exp(g_cum)``,
    ``exp(-g_cum)`` and the fp32 copy of ``beta`` as 1D fp32 buffers.  Doing the
    exponentials here (1D) is what allows the fold kernel to use plain
    broadcast multiplies instead of ``exp`` on a 1D vector, which would
    serialize this backend's vector pipeline.
    """
    i_t0, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos0 = i_b * T
    for j in tl.static_range(0, NCHUNK):
        i_t = i_t0 * NCHUNK + j
        bos = bos0
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        base = bos * H + i_h
        b_g = tl.load(
            tl.make_block_ptr(g + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_gc = tl.cumsum(b_g, axis=0)
        tl.store(
            tl.make_block_ptr(g_cum + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_gc, boundary_check=(0,),
        )
        b_e = tl.exp(b_gc)
        tl.store(
            tl.make_block_ptr(e1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_e, boundary_check=(0,),
        )
        tl.store(
            tl.make_block_ptr(r1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            tl.exp(tl.minimum(-b_gc, 80.0)), boundary_check=(0,),
        )
        b_beta = tl.load(
            tl.make_block_ptr(beta + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        tl.store(
            tl.make_block_ptr(b1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_beta, boundary_check=(0,),
        )
        tl.store(
            tl.make_block_ptr(be1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_beta * b_e, boundary_check=(0,),
        )


@triton.jit(do_not_specialize=["T"])
def _factor_head_tile_kernel(
    g,
    beta,
    g_cum,
    e1,
    r1,
    b1,
    be1,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    """Factor adjacent heads together to coalesce token/head memory access."""
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1)
    n_h_tiles: tl.constexpr = tl.cdiv(H, BH)
    i_b = i_bh // n_h_tiles
    i_h0 = (i_bh % n_h_tiles) * BH
    p_g = tl.make_block_ptr(
        g + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    p_beta = tl.make_block_ptr(
        beta + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    p_gc = tl.make_block_ptr(
        g_cum + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    p_e = tl.make_block_ptr(
        e1 + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    p_r = tl.make_block_ptr(
        r1 + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    p_b = tl.make_block_ptr(
        b1 + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    p_be = tl.make_block_ptr(
        be1 + i_b * T * H, (T, H), (H, 1), (i_c * BT, i_h0), (BT, BH), (1, 0)
    )
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_gc = tl.cumsum(b_g, axis=0)
    b_e = tl.exp(b_gc)
    b_beta = tl.load(p_beta, boundary_check=(0, 1)).to(tl.float32)
    tl.store(p_gc, b_gc, boundary_check=(0, 1))
    tl.store(p_e, b_e, boundary_check=(0, 1))
    tl.store(p_r, tl.exp(tl.minimum(-b_gc, 80.0)), boundary_check=(0, 1))
    tl.store(p_b, b_beta, boundary_check=(0, 1))
    tl.store(p_be, b_beta * b_e, boundary_check=(0, 1))


def launch_factor_head_tile(
    g,
    beta,
    g_cum,
    e1,
    r1,
    b1,
    be1,
    *,
    chunk_size: int,
    heads_per_program: int = 16,
    num_warps: int = 4,
    num_stages: int = 2,
    buf=None,
):
    """Launch fixed-length coalesced factor tiles (all math/output fp32)."""
    b, t, h = g.shape
    grid = (
        triton.cdiv(t, chunk_size),
        b * triton.cdiv(h, heads_per_program),
    )
    key = ("factor_ht", h, chunk_size, heads_per_program, beta.dtype)
    _lk(buf, key, _factor_head_tile_kernel, grid, num_warps, num_stages)(
        g,
        beta,
        g_cum,
        e1,
        r1,
        b1,
        be1,
        t,
        h,
        chunk_size,
        heads_per_program,
    )


@triton.jit(do_not_specialize=["T"])
def _fold_kernel(
    q,
    k,
    v,
    e1,
    r1,
    b1,
    be1,
    qe,
    ke,
    kr,
    vb,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """Vector-only folds: plain 1D block loads + ``[:, None]`` broadcast muls.

    No 2D gather, no ``exp`` on a 1D vector, no 2D cumsum: each of those
    serializes the vector pipeline on this backend.
    """
    i_t0, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    rep = H // Hg
    i_hg = i_h // rep
    bos0 = i_b * T
    for j in tl.static_range(0, NCHUNK):
        i_t = i_t0 * NCHUNK + j
        bos = bos0
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        rep = H // Hg
        i_hg = i_h // rep
        base = bos * H + i_h
        base_qk = (bos * Hg + i_hg) * K

        b_e = tl.load(
            tl.make_block_ptr(e1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_r = tl.load(
            tl.make_block_ptr(r1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_beta = tl.load(
            tl.make_block_ptr(b1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_be = tl.load(
            tl.make_block_ptr(be1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)

        b_k = tl.load(
            tl.make_block_ptr(
                k + base_qk, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                ke + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            (b_k.to(tl.float32) * b_be[:, None]).to(ke.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                kr + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            (b_k.to(tl.float32) * b_r[:, None]).to(kr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                qe + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            (
                tl.load(
                    tl.make_block_ptr(
                        q + base_qk,
                        (T, K),
                        (Hg * K, 1),
                        (i_t * BT, 0),
                        (BT, K),
                        (1, 0),
                    ),
                    boundary_check=(0, 1),
                ).to(tl.float32)
                * b_e[:, None]
            ).to(qe.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                vb + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0)
            ),
            (
                tl.load(
                    tl.make_block_ptr(
                        v + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V),
                        (1, 0)
                    ),
                    boundary_check=(0, 1),
                ).to(tl.float32)
                * b_beta[:, None]
            ).to(vb.dtype.element_ty),
            boundary_check=(0, 1),
        )


@triton.jit(do_not_specialize=["T"])
def _fold_group_kloop_kernel(
    q, k, v, e1, r1, b1, be1, qe, ke, kr, vb,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    """Fold one GQA group per CTA and reuse its Q/K rows across the group."""
    i_c, i_group = tl.program_id(0), tl.program_id(1)
    n_groups: tl.constexpr = tl.cdiv(H, BH)
    i_b = i_group // n_groups
    h0 = (i_group % n_groups) * BH
    rep: tl.constexpr = H // Hg
    rt = i_c * BT + tl.arange(0, BT)
    rh = h0 + tl.arange(0, BH)
    mt = rt < T
    mh = rh < H

    p_e = tl.make_block_ptr(e1 + i_b * T * H, (T, H), (H, 1),
                            (i_c * BT, h0), (BT, BH), (1, 0))
    p_r = tl.make_block_ptr(r1 + i_b * T * H, (T, H), (H, 1),
                            (i_c * BT, h0), (BT, BH), (1, 0))
    p_b = tl.make_block_ptr(b1 + i_b * T * H, (T, H), (H, 1),
                            (i_c * BT, h0), (BT, BH), (1, 0))
    p_be = tl.make_block_ptr(be1 + i_b * T * H, (T, H), (H, 1),
                             (i_c * BT, h0), (BT, BH), (1, 0))
    e = tl.load(p_e, boundary_check=(0, 1)).to(tl.float32)
    r = tl.load(p_r, boundary_check=(0, 1)).to(tl.float32)
    beta = tl.load(p_b, boundary_check=(0, 1)).to(tl.float32)
    be = tl.load(p_be, boundary_check=(0, 1)).to(tl.float32)

    q_head = h0 // rep
    base_q = (i_b * T + rt[:, None]) * Hg + q_head
    base_v = (i_b * T + rt[:, None, None]) * H * V + rh[None, :, None] * V
    for ik in tl.static_range(0, K // BK):
        rk = ik * BK + tl.arange(0, BK)
        base_qk = base_q * K + rk[None, :]
        qv = tl.load(q + base_qk, mask=mt[:, None] & (rk[None, :] < K), other=0)
        kv = tl.load(k + base_qk, mask=mt[:, None] & (rk[None, :] < K), other=0)
        vv = tl.load(v + base_v + rk[None, :],
                     mask=mt[:, None, None] & mh[None, :, None] & (rk[None, None, :] < V),
                     other=0)
        qe_tile = qv.to(tl.float32)[:, None, :] * e[:, :, None]
        ke_tile = kv.to(tl.float32)[:, None, :] * be[:, :, None]
        kr_tile = kv.to(tl.float32)[:, None, :] * r[:, :, None]
        vb_tile = vv.to(tl.float32) * beta[:, :, None]
        q_offset = (i_b * T + rt[:, None, None]) * H * K + rh[None, :, None] * K + rk[None, None, :]
        v_offset = base_v + rk[None, None, :]
        mask3 = mt[:, None, None] & mh[None, :, None] & (rk[None, None, :] < K)
        tl.store(qe + q_offset, qe_tile.to(qe.dtype.element_ty), mask=mask3)
        tl.store(ke + q_offset, ke_tile.to(ke.dtype.element_ty), mask=mask3)
        tl.store(kr + q_offset, kr_tile.to(kr.dtype.element_ty), mask=mask3)
        tl.store(vb + v_offset, vb_tile.to(vb.dtype.element_ty),
                 mask=mt[:, None, None] & mh[None, :, None] & (rk[None, None, :] < V))


def launch_fold_group_kloop(
    q, k, v, e1, r1, b1, be1, qe, ke, kr, vb, *,
    chunk_size: int = 64, heads_per_program: int = 4, key_tile: int = 32,
    num_warps: int = 4, num_stages: int = 2,
):
    """Launch the long fixed-length fold that reuses Q/K across a GQA group."""
    B, T, Hg, K = k.shape
    H, V = e1.shape[-1], v.shape[-1]
    rep = H // Hg
    if H % Hg or heads_per_program != rep:
        raise ValueError("fold group tile must equal an aligned GQA group")
    if K != V or K % key_tile:
        raise ValueError("fold group kernel requires equal, tile-aligned K and V")
    grid = (triton.cdiv(T, chunk_size), B * triton.cdiv(H, heads_per_program))
    fast_launch(_fold_group_kloop_kernel, grid, num_warps, num_stages)(
        q, k, v, e1, r1, b1, be1, qe, ke, kr, vb, T, H, Hg, K, V,
        chunk_size, heads_per_program, key_tile,
    )


@triton.jit(do_not_specialize=["T"])
def _factor_headfirst_stride_kernel(
    g, beta, g_cum, e1, r1, b1, be1, T,
    H: tl.constexpr, BT: tl.constexpr, BH: tl.constexpr,
    FACTOR_T: tl.constexpr,
    SGB: tl.constexpr, SGT: tl.constexpr, SGH: tl.constexpr,
    SBB: tl.constexpr, SBT: tl.constexpr, SBH: tl.constexpr,
):
    i_c, i_bht = tl.program_id(0), tl.program_id(1)
    n_h_tiles: tl.constexpr = tl.cdiv(H, BH)
    i_b = i_bht // n_h_tiles
    i_h0 = (i_bht % n_h_tiles) * BH
    rt = i_c * BT + tl.arange(0, BT)
    rh = i_h0 + tl.arange(0, BH)
    mask = (rt[None, :] < T) & (rh[:, None] < H)
    g_offset = i_b * SGB + rh[:, None] * SGH + rt[None, :] * SGT
    beta_offset = i_b * SBB + rh[:, None] * SBH + rt[None, :] * SBT
    b_g = tl.load(g + g_offset, mask=mask, other=0.0).to(tl.float32)
    b_gc = tl.cumsum(b_g, axis=1)
    b_e = tl.exp(b_gc)
    b_r = tl.exp(tl.minimum(-b_gc, 80.0))
    b_beta = tl.load(beta + beta_offset, mask=mask, other=0.0).to(tl.float32)
    # Factor scratch uses [B,H,T] physical order only for this immediate
    # head-first fold. g_cum stays in the standard [B,T,H] output layout.
    factor_dst = i_b * H * FACTOR_T + rh[:, None] * FACTOR_T + rt[None, :]
    # Store g_cum in the same head-major orientation as ``b_gc`` instead of
    # ``tl.trans(b_gc)``/``tl.trans(mask)``: a transposed store on this backend
    # can drop or garble a random subset of token rows for some shapes,
    # whereas the non-transposed stores of this kernel are exact.  This offset
    # expression writes element (b, t, h) at b*T*H + t*H + h, i.e. the standard
    # [B,T,H] layout.
    gc_dst = i_b * T * H + rt[None, :] * H + rh[:, None]
    tl.store(g_cum + gc_dst, b_gc, mask=mask)
    tl.store(e1 + factor_dst, b_e, mask=mask)
    tl.store(r1 + factor_dst, b_r, mask=mask)
    tl.store(b1 + factor_dst, b_beta, mask=mask)
    tl.store(be1 + factor_dst, b_beta * b_e, mask=mask)


@triton.jit(do_not_specialize=["T"])
def _fold_headfirst_stride_kernel(
    q, k, v, e1, r1, b1, be1, qe, ke, kr, vb, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, FACTOR_T: tl.constexpr,
    SQB: tl.constexpr, SQT: tl.constexpr, SQH: tl.constexpr, SQK: tl.constexpr,
    SKB: tl.constexpr, SKT: tl.constexpr, SKH: tl.constexpr, SKK: tl.constexpr,
    SVB: tl.constexpr, SVT: tl.constexpr, SVH: tl.constexpr, SVV: tl.constexpr,
):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    rep = H // Hg
    i_hg = i_h // rep
    rt = i_c * BT + tl.arange(0, BT)
    rk = tl.arange(0, K)
    mask_t = rt < T
    factor_offset = i_b * H * FACTOR_T + i_h * FACTOR_T + rt
    b_e = tl.load(e1 + factor_offset, mask=mask_t, other=0.0).to(tl.float32)
    b_r = tl.load(r1 + factor_offset, mask=mask_t, other=0.0).to(tl.float32)
    b_beta = tl.load(b1 + factor_offset, mask=mask_t, other=0.0).to(tl.float32)
    b_be = tl.load(be1 + factor_offset, mask=mask_t, other=0.0).to(tl.float32)
    q_offset = (i_b * SQB + rt[:, None] * SQT + i_hg * SQH
                + rk[None, :] * SQK)
    k_offset = (i_b * SKB + rt[:, None] * SKT + i_hg * SKH
                + rk[None, :] * SKK)
    v_offset = (i_b * SVB + rt[:, None] * SVT + i_h * SVH
                + rk[None, :] * SVV)
    mask = mask_t[:, None] & (rk[None, :] < K)
    b_q = tl.load(q + q_offset, mask=mask, other=0.0)
    b_k = tl.load(k + k_offset, mask=mask, other=0.0)
    b_v = tl.load(v + v_offset,
                  mask=mask_t[:, None] & (rk[None, :] < V), other=0.0)
    base = i_b * T * H + i_h
    out_offset = base * K + rt[:, None] * (H * K) + rk[None, :]
    tl.store(qe + out_offset,
             (b_q.to(tl.float32) * b_e[:, None]).to(qe.dtype.element_ty),
             mask=mask)
    tl.store(ke + out_offset,
             (b_k.to(tl.float32) * b_be[:, None]).to(ke.dtype.element_ty),
             mask=mask)
    tl.store(kr + out_offset,
             (b_k.to(tl.float32) * b_r[:, None]).to(kr.dtype.element_ty),
             mask=mask)
    out_v_offset = base * V + rt[:, None] * (H * V) + rk[None, :]
    tl.store(vb + out_v_offset,
             (b_v.to(tl.float32) * b_beta[:, None]).to(vb.dtype.element_ty),
             mask=mask_t[:, None] & (rk[None, :] < V))


@triton.jit(do_not_specialize=["T"])
def _fold_headfirst_stride_trim_kernel(
    k, r1, be1, ke, kr, T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, FACTOR_T: tl.constexpr,
    SKB: tl.constexpr, SKT: tl.constexpr, SKH: tl.constexpr, SKK: tl.constexpr,
):
    """Strided fold that materialises only k_e/k_r (v1 fold trim).

    qe = q*e1 and vb = v*beta are recomputed inside the affine kernel from the
    source q/v tensors and the small fp32 factor buffers, which removes the
    qe/vb GM round trips.
    """
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    rep = H // Hg
    i_hg = i_h // rep
    rt = i_c * BT + tl.arange(0, BT)
    rk = tl.arange(0, K)
    mask_t = rt < T
    factor_offset = i_b * H * FACTOR_T + i_h * FACTOR_T + rt
    b_r = tl.load(r1 + factor_offset, mask=mask_t, other=0.0).to(tl.float32)
    b_be = tl.load(be1 + factor_offset, mask=mask_t, other=0.0).to(tl.float32)
    k_offset = (i_b * SKB + rt[:, None] * SKT + i_hg * SKH
                + rk[None, :] * SKK)
    mask = mask_t[:, None] & (rk[None, :] < K)
    b_k = tl.load(k + k_offset, mask=mask, other=0.0)
    base = i_b * T * H + i_h
    out_offset = base * K + rt[:, None] * (H * K) + rk[None, :]
    tl.store(ke + out_offset,
             (b_k.to(tl.float32) * b_be[:, None]).to(ke.dtype.element_ty),
             mask=mask)
    tl.store(kr + out_offset,
             (b_k.to(tl.float32) * b_r[:, None]).to(kr.dtype.element_ty),
             mask=mask)


def _fold_headfirst_strided(g, q, k, v, beta, chunk_size):
    B, T, Hg, K = q.shape
    H, V = beta.shape[-1], v.shape[-1]
    BT = int(chunk_size)
    NC = triton.cdiv(T, BT)
    BH = 16
    factor_t = triton.cdiv(T, 32) * 32
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, q.dtype, q.device)
        g_cum = buf["g_cum"]
        qe, ke, kr, vb = buf["qe"], buf["ke"], buf["kr"], buf["vb"]
        factor_shape = (B, H, factor_t)
        if "headfirst_e1" not in buf:
            buf["headfirst_e1"] = torch.empty(
                factor_shape, dtype=torch.float32, device=g.device,
            )
            buf["headfirst_r1"] = torch.empty_like(buf["headfirst_e1"])
            buf["headfirst_b1"] = torch.empty_like(buf["headfirst_e1"])
            buf["headfirst_be1"] = torch.empty_like(buf["headfirst_e1"])
        e1 = buf["headfirst_e1"]
        r1 = buf["headfirst_r1"]
        b1 = buf["headfirst_b1"]
        be1 = buf["headfirst_be1"]
    else:
        g_cum = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        fold_dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
        qe = torch.empty(B, T, H, K, dtype=fold_dtype, device=q.device)
        ke = torch.empty(B, T, H, K, dtype=fold_dtype, device=q.device)
        kr = torch.empty(B, T, H, K, dtype=fold_dtype, device=q.device)
        vb = torch.empty(B, T, H, V, dtype=fold_dtype, device=v.device)
        # Pad each head row so factor stores from adjacent CTAs never share a
        # partially aligned tail cache line when T is not a multiple of 32.
        factor_shape = (B, H, factor_t)
        e1 = torch.empty(factor_shape, dtype=torch.float32, device=g.device)
        r1 = torch.empty_like(e1)
        b1 = torch.empty_like(e1)
        be1 = torch.empty_like(e1)
    warps, stages = _fold_config()
    fast_launch(_factor_headfirst_stride_kernel, (NC, B * triton.cdiv(H, BH)), warps, stages)(
        g, beta, g_cum, e1, r1, b1, be1, T,
        H, BT, BH, factor_t, *g.stride(), *beta.stride(),
    )
    fast_launch(_fold_headfirst_stride_kernel, (NC, B * H), warps, stages)(
        q, k, v, e1, r1, b1, be1, qe, ke, kr, vb, T,
        H, Hg, K, V, BT, factor_t,
        *q.stride(), *k.stride(), *v.stride(),
    )
    return g_cum, qe, ke, kr, vb


@triton.jit(do_not_specialize=["T"])
def _fold_gather_kernel(
    q,
    k,
    v,
    g,
    beta,
    g_cum,
    qe,
    ke,
    kr,
    vb,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Single-kernel fold for *small* workloads (one chunk-head tile per program).

    It does 2D gathers from the bf16 inputs, which serialize this backend's
    vector kernels; with only ``chunks*H <= 64`` programs the serialized cost
    stays small and one launch beats the two-launch ``factor`` + ``fold`` pair
    on host time.
    """
    i_t, i_bh, i_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    rep = H // Hg
    i_hg = i_h // rep
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    base = bos * H + i_h
    base_qk = (bos * Hg + i_hg) * K

    p_g = tl.make_block_ptr(g + base, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g1 = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    p_gc = tl.make_block_ptr(g_cum + base, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_gc, tl.cumsum(b_g1, axis=0), boundary_check=(0,))

    b_gc = tl.cumsum(
        tl.load(
            g + base + o_t[:, None] * H + 0 * o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    b_e = tl.exp(b_gc).to(tl.bfloat16)
    b_beta = tl.load(
        beta + base + o_t[:, None] * H + 0 * o_k[None, :],
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    b_k = tl.load(
        tl.make_block_ptr(
            k + base_qk,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            ke + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        b_k * (b_beta * b_e),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            kr + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        b_k * tl.exp(tl.minimum(-b_gc, 80.0)).to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            qe + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        tl.load(
            tl.make_block_ptr(
                q + base_qk,
                (T, K),
                (Hg * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            ),
            boundary_check=(0, 1),
        ) * b_e,
        boundary_check=(0, 1),
    )
    # The fast path requires K == V, so this K slice is the same V slice.
    tl.store(
        tl.make_block_ptr(
            vb + base * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        tl.load(
            tl.make_block_ptr(
                v + base * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            ),
            boundary_check=(0, 1),
        ) * b_beta,
        boundary_check=(0, 1),
    )


# ---------------------------------------------------------------------------
# kernel 2: cube-only prep (KKT + Newton inverse + w/u + output A)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _prep_wu_kernel(
    qe,
    ke,
    kr,
    vb,
    w,
    u,
    a_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    N_NEWTON: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    PRECOMPUTE_AFFINE: tl.constexpr,
    qe2,
    au,
    cu,
    mw,
    NT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    rep = H // Hg
    i_hg = i_h // rep

    o_t = i_t * BT + tl.arange(0, BT)
    base = bos * H + i_h
    # q/k are shared inside a GQA group but the decay is per head, so the
    # pre-folded q_e/k_e/k_r/k_s tensors are indexed per head; only ``w`` and
    # ``u`` are per H as well.
    base_qk = (bos * Hg + i_hg) * K

    p_ke = tl.make_block_ptr(
        ke + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base * K, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_ke = tl.load(p_ke, boundary_check=(0, 1))
    b_kr = tl.load(p_kr, boundary_check=(0, 1))
    b_L = tl.dot(b_ke, b_kr)
    b_L = tl.where(o_t[:, None] > o_t[None, :], b_L, 0.0)

    # A_inv = (I + L)^-1 by Newton doubling; L is strictly lower triangular
    # (nilpotent, L^BT = 0), so log2(BT) - 1 steps are exact.
    o_r = tl.arange(0, BT)
    m_I = (o_r[:, None] == o_r[None, :]).to(tl.float32)
    b_M = b_L + m_I
    b_Ai = m_I - b_L
    for _ in tl.static_range(0, N_NEWTON):
        b_Y = tl.dot(b_M, b_Ai)
        b_Ai = tl.dot(b_Ai, 2.0 * m_I - b_Y)

    # A = tril(q_e @ k_r^T), the decayed causal q/k product chunk_o needs.
    p_qe = tl.make_block_ptr(
        qe + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    b_qe = tl.load(p_qe, boundary_check=(0, 1))
    b_A = tl.dot(b_qe, b_kr)
    b_A = tl.where(o_t[:, None] >= o_t[None, :], b_A, 0)
    p_A = tl.make_block_ptr(
        a_out + base * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = b_A.to(p_A.dtype.element_ty)
    tl.store(p_A, b_A, boundary_check=(0, 1))

    p_w = tl.make_block_ptr(
        w + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    b_w = tl.dot(b_Ai.to(b_ke.dtype), b_ke)
    b_w = b_w.to(p_w.dtype.element_ty)
    if not PRECOMPUTE_AFFINE:
        tl.store(p_w, b_w, boundary_check=(0, 1))

    p_vb = tl.make_block_ptr(
        vb + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
    )
    p_u = tl.make_block_ptr(
        u + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
    )
    b_vb = tl.load(p_vb, boundary_check=(0, 1))
    b_u = tl.dot(b_Ai.to(b_vb.dtype), b_vb)
    b_u = b_u.to(p_u.dtype.element_ty)
    if not PRECOMPUTE_AFFINE:
        tl.store(p_u, b_u, boundary_check=(0, 1))

    if PRECOMPUTE_AFFINE:
        # Keep A/W/U in the input dtype for these products, matching the
        # precision they have after the GM round trip the two-pass path
        # performs in ``affine_precompute_ascend``.
        b_qw = tl.dot(b_A, b_w.to(b_A.dtype))
        p_qe2 = tl.make_block_ptr(
            qe2 + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        tl.store(
            p_qe2,
            (b_qe.to(tl.float32) - b_qw).to(p_qe2.dtype.element_ty),
            boundary_check=(0, 1),
        )
        p_au = tl.make_block_ptr(
            au + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
        )
        tl.store(
            p_au,
            tl.dot(b_A, b_u).to(p_au.dtype.element_ty),
            boundary_check=(0, 1),
        )
        b_kr = tl.load(p_kr, boundary_check=(0, 1))
        cbase_v = ((i_b * NT + i_t) * H + i_h) * K * V
        p_cu = tl.make_block_ptr(
            cu + cbase_v, (K, V), (V, 1), (0, 0), (K, BV), (1, 0)
        )
        tl.store(
            p_cu, tl.dot(b_kr, b_u.to(b_kr.dtype)).to(p_cu.dtype.element_ty)
        )
        cbase_m = ((i_b * NT + i_t) * H + i_h) * K * K
        p_mw = tl.make_block_ptr(
            mw + cbase_m, (K, K), (K, 1), (0, 0), (K, K), (1, 0)
        )
        tl.store(
            p_mw, tl.dot(b_kr, b_w.to(b_kr.dtype)).to(p_mw.dtype.element_ty)
        )


# ---------------------------------------------------------------------------
# kernel 3: state scan + output stage, state tile stays in registers
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _fwd_h_o_kernel(
    qe,
    kr,
    a_in,
    w,
    u,
    g,
    o,
    h0,
    ht,
    scale,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    b_h = tl.zeros([K, BV], dtype=tl.float32)

    base_h = (bos * H + i_h) * K
    base_a = (bos * H + i_h) * BT
    base_v = (bos * H + i_h) * V
    gb = g + bos * H + i_h
    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
    if STORE_FINAL_STATE:
        ht += i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_qe = tl.make_block_ptr(
            qe + base_h, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        b_qe = tl.load(p_qe, boundary_check=(0, 1))
        p_A = tl.make_block_ptr(
            a_in + base_a, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))
        p_kr = tl.make_block_ptr(
            kr + base_h, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
        )
        b_kr = tl.load(p_kr, boundary_check=(0, 1))
        p_w = tl.make_block_ptr(
            w + base_h, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)
        b_v = b_v - tl.dot(b_w, b_h.to(b_w.dtype))

        b_o = tl.dot(b_qe, b_h.to(b_qe.dtype)) + tl.dot(
            b_A, b_v.to(a_in.dtype.element_ty)
        )
        p_o = tl.make_block_ptr(
            o + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, (b_o * scale).to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # h <- exp(G) h + k_s^T v_new with k_s = k_r exp(G); equivalently
        # h <- exp(G) (h + k_r^T v_new), i.e. one scalar multiply per chunk.
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.exp(tl.load(gb + last_idx * H).to(tl.float32))
        b_h = b_h + tl.dot(b_kr, b_v.to(kr.dtype.element_ty))
        if USE_G:
            b_h = b_h * b_g_last

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# exact two-pass state scan
#
# Per chunk the delta-rule recurrence is affine *with a matrix transition*:
#     v_new = u - w S
#     S'    = a S + kr (u - w S) = (a I - a kr w) S + a kr u
# so a chunk-parallel pass can precompute
#     qe2 = qe - A w        (o = qe2 S + A u)
#     Au  = A u             (o = qe2 S + Au)
#     Cu  = kr u            (S' = a (S + Cu - Mw S))
#     Mw  = kr w
# and the sequential part then needs only two dots per chunk (S' needs one:
# Mw S; the output needs one: qe2 S) with no u/w/A/kr loads at all.
#
# NOTE: the "C_c = sum_j exp(G-g_j) k_j u_j^T, prefix-combine with a_c only"
# recurrence would *not* be exact: the -a (kr w) S term is what Mw @ S below
# evaluates, and dropping it changes the results by O(1) per chunk.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _affine_kernel(
    qe,
    a_in,
    kr,
    w,
    u,
    qe2,
    au,
    cu,
    mw,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    base = bos * H + i_h
    p_A = tl.make_block_ptr(
        a_in + base * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    p_w = tl.make_block_ptr(
        w + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_u = tl.make_block_ptr(
        u + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0)
    )
    p_qe = tl.make_block_ptr(
        qe + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base * K, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0, 1))
    b_qe = tl.load(p_qe, boundary_check=(0, 1))

    b_qw = tl.dot(b_A, b_w.to(b_A.dtype))
    p_qe2 = tl.make_block_ptr(
        qe2 + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    tl.store(
        p_qe2,
        (b_qe.to(tl.float32) - b_qw).to(p_qe2.dtype.element_ty),
        boundary_check=(0, 1),
    )
    p_au = tl.make_block_ptr(
        au + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0)
    )
    tl.store(p_au, tl.dot(b_A, b_u).to(p_au.dtype.element_ty),
             boundary_check=(0, 1))

    b_kr = tl.load(p_kr, boundary_check=(0, 1))
    cbase_v = ((i_b * NT + i_t) * H + i_h) * K * V
    p_cu = tl.make_block_ptr(cu + cbase_v, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    # Decay factors can be kept in fp32 for FP16 inputs to avoid overflow and
    # underflow in the folded operands. Match the dot operand types here; for
    # BF16 inputs this cast is a no-op.
    b_u_state = b_u.to(b_kr.dtype)
    b_w_state = b_w.to(b_kr.dtype)
    tl.store(p_cu, tl.dot(b_kr, b_u_state).to(p_cu.dtype.element_ty))
    cbase_m = ((i_b * NT + i_t) * H + i_h) * K * K
    p_mw = tl.make_block_ptr(mw + cbase_m, (K, K), (K, 1), (0, 0), (K, K), (1, 0))
    tl.store(p_mw, tl.dot(b_kr, b_w_state).to(p_mw.dtype.element_ty))


@triton.jit(do_not_specialize=["T"])
def _fwd_h_o_unroll2_kernel(
    qe, kr, a_in, w, u, g, o, h0, ht, scale, cu_seqlens,
    T,
    H: tl.constexpr, Hg: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    NT: tl.constexpr,
):
    """One-pass state scan with two chunks per iteration.

    Both chunks' tiles are loaded before chunk 0's dots, which lets the
    independent work of the two chunks overlap.  Fixed-length only; NT must be
    even.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    NPAIR: tl.constexpr = NT // 2
    b_h = tl.zeros([K, BV], dtype=tl.float32)
    base_h = (bos * H + i_h) * K
    base_a = (bos * H + i_h) * BT
    base_v = (bos * H + i_h) * V
    gb = g + bos * H + i_h
    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
    if STORE_FINAL_STATE:
        ht += i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_p in range(NPAIR):
        ia = i_p * 2
        ib = ia + 1
        b_qe0 = tl.load(tl.make_block_ptr(qe + base_h, (T, K), (H * K, 1),
                       (ia * BT, 0), (BT, K), (1, 0)), boundary_check=(0, 1))
        b_A0 = tl.load(tl.make_block_ptr(a_in + base_a, (T, BT), (H * BT, 1),
                      (ia * BT, 0), (BT, BT), (1, 0)), boundary_check=(0, 1))
        b_kr0 = tl.load(tl.make_block_ptr(kr + base_h, (K, T), (1, H * K),
                       (0, ia * BT), (K, BT), (0, 1)), boundary_check=(0, 1))
        b_w0 = tl.load(tl.make_block_ptr(w + base_h, (T, K), (H * K, 1),
                      (ia * BT, 0), (BT, K), (1, 0)), boundary_check=(0, 1))
        b_v0 = tl.load(tl.make_block_ptr(u + base_v, (T, V), (H * V, 1),
                      (ia * BT, i_v * BV), (BT, BV), (1, 0)),
                      boundary_check=(0, 1)).to(tl.float32)
        b_qe1 = tl.load(tl.make_block_ptr(qe + base_h, (T, K), (H * K, 1),
                       (ib * BT, 0), (BT, K), (1, 0)), boundary_check=(0, 1))
        b_A1 = tl.load(tl.make_block_ptr(a_in + base_a, (T, BT), (H * BT, 1),
                      (ib * BT, 0), (BT, BT), (1, 0)), boundary_check=(0, 1))
        b_kr1 = tl.load(tl.make_block_ptr(kr + base_h, (K, T), (1, H * K),
                       (0, ib * BT), (K, BT), (0, 1)), boundary_check=(0, 1))
        b_w1 = tl.load(tl.make_block_ptr(w + base_h, (T, K), (H * K, 1),
                      (ib * BT, 0), (BT, K), (1, 0)), boundary_check=(0, 1))
        b_v1 = tl.load(tl.make_block_ptr(u + base_v, (T, V), (H * V, 1),
                      (ib * BT, i_v * BV), (BT, BV), (1, 0)),
                      boundary_check=(0, 1)).to(tl.float32)

        b_v0 = b_v0 - tl.dot(b_w0, b_h.to(b_w0.dtype))
        b_o = tl.dot(b_qe0, b_h.to(b_qe0.dtype)) + tl.dot(
            b_A0, b_v0.to(b_qe0.dtype))
        tl.store(tl.make_block_ptr(o + base_v, (T, V), (H * V, 1),
                 (ia * BT, i_v * BV), (BT, BV), (1, 0)),
                 (b_o * scale).to(o.dtype.element_ty), boundary_check=(0, 1))
        b_h = b_h + tl.dot(b_kr0, b_v0.to(b_kr0.dtype))
        if USE_G:
            last0 = (ia + 1) * BT - 1
            b_h = b_h * tl.exp(tl.load(gb + last0 * H).to(tl.float32))

        b_v1 = b_v1 - tl.dot(b_w1, b_h.to(b_w1.dtype))
        b_o1 = tl.dot(b_qe1, b_h.to(b_qe1.dtype)) + tl.dot(
            b_A1, b_v1.to(b_qe1.dtype))
        tl.store(tl.make_block_ptr(o + base_v, (T, V), (H * V, 1),
                 (ib * BT, i_v * BV), (BT, BV), (1, 0)),
                 (b_o1 * scale).to(o.dtype.element_ty), boundary_check=(0, 1))
        b_h = b_h + tl.dot(b_kr1, b_v1.to(b_kr1.dtype))
        if USE_G:
            last1 = (ib + 1) * BT - 1
            b_h = b_h * tl.exp(tl.load(gb + last1 * H).to(tl.float32))

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _fwd_h_o_2pass_kernel(
    qe2,
    au,
    cu,
    mw,
    g,
    o,
    h0,
    ht,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    b_h = tl.zeros([K, BV], dtype=tl.float32)
    base_v = (bos * H + i_h) * V
    base_k = (bos * H + i_h) * K
    gb = g + bos * H + i_h
    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
    if STORE_FINAL_STATE:
        ht += i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        cb_v = ((i_n * NT + i_t) * H + i_h) * K * V
        cb_m = ((i_n * NT + i_t) * H + i_h) * K * K
        p_qe2 = tl.make_block_ptr(
            qe2 + base_k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_au = tl.make_block_ptr(
            au + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_qe2 = tl.load(p_qe2, boundary_check=(0, 1))
        b_au = tl.load(p_au, boundary_check=(0, 1)).to(tl.float32)
        p_o = tl.make_block_ptr(
            o + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_o = tl.dot(b_qe2, b_h.to(b_qe2.dtype)) + b_au
        tl.store(p_o, (b_o * scale).to(p_o.dtype.element_ty), boundary_check=(0, 1))

        b_cu = tl.load(
            tl.make_block_ptr(
                cu + cb_v, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
            )
        )
        b_mw = tl.load(
            tl.make_block_ptr(mw + cb_m, (K, K), (K, 1), (0, 0), (K, K), (1, 0))
        )
        b_corr = tl.dot(b_mw, b_h.to(b_mw.dtype))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_a = tl.exp(tl.load(gb + last_idx * H).to(tl.float32))
        b_h = (b_h + b_cu.to(tl.float32) - b_corr) * b_a

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# host wrappers
# ---------------------------------------------------------------------------


# Workload size below which the single-kernel (gather) fold wins on host time.
_SMALL_FOLD_PROGRAMS = 64
# The one-launch gather fold only pays off for short sequences: its bf16 2D
# gathers grow with T, while the factor+fold pair is launch-bound and its extra
# launch is cheap.  T=512 is the empirical boundary on this backend; shapes
# above it take factor+fold.
_GATHER_MAX_T = 512

# --- stride-aware fold dispatch -------------------------------------------
# Single source of truth shared by ``api.py`` (which decides the input layout)
# and ``fold_ascend`` (which picks the kernel).  The materialised fold has two
# variants and the stride-aware pair only wins once the grid is big enough,
# but *which* crossover applies depends on the variant it would replace:
#
# * one-launch gather branch (``NC*H <= _SMALL_FOLD_PROGRAMS``): the gather
#   kernel cost grows with batch/head/K while the strided pair is shape-flat,
#   so it loses from ~64 chunk/head jobs upward.
# * two-kernel factor+fold branch: already efficient, the strided pair only
#   wins on large grids (>=512 jobs) or the long-sequence rule below.
_STRIDED_MIN_JOBS = 64
_STRIDED_LARGE_JOBS = 512


def strided_fold_supported(
    B: int,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    dtype,
    cu_seqlens,
    chunk_size: int,
) -> bool:
    """Return True when the stride-aware fold is the preferred choice here."""
    if cu_seqlens is not None or int(chunk_size) != 64:
        return False
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    if T < 64 or K != V or not (0 < K <= 128) or H % Hg != 0:
        return False
    nc = (T + 63) // 64
    jobs = B * nc * H
    if jobs < _STRIDED_MIN_JOBS:
        return False
    if T >= 2048 and jobs > _SMALL_FOLD_PROGRAMS:
        return True
    if nc * H <= _SMALL_FOLD_PROGRAMS:
        return True
    return jobs >= _STRIDED_LARGE_JOBS

# Chunk-batched fold/factor kernels (vector-only loops, no dots): each program
# handles NCHUNK consecutive chunks.  FLAG_ATTN_ASCEND_GDN_NCHUNK overrides the
# automatic choice; 1 disables batching.
_NCHUNK_ENV = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_NCHUNK", "0"))
# The coalesced fixed-length factor kernel is default-on; set this to 0 to fall
# back to the one-head factor kernel.
_FACTOR_HEAD_TILE_ENABLED = os.environ.get(
    "FLAG_ATTN_ASCEND_GDN_FACTOR_HEAD_TILE", "1"
).lower() not in {"0", "false", "off", "no"}
_FOLD_GROUP_KLOOP_ENABLED = os.environ.get(
    "FLAG_ATTN_ASCEND_GDN_FOLD_GROUP_KLOOP", "1"
).lower() not in {"0", "false", "off", "no"}
# Automatic batching: batching several chunks per program is not a win on this
# backend, so it is OFF unless requested explicitly.
_NCHUNK_AUTO = 1


def _fold_nchunk(NC: int, is_varlen: bool) -> int:
    if is_varlen:
        return 1
    if _NCHUNK_ENV > 0:
        return _NCHUNK_ENV if NC % _NCHUNK_ENV == 0 else 1
    return _NCHUNK_AUTO

# Fixed kernel configs, read once (not per call).
_FOLD_WARPS = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_F_WARPS", "4"))
_FOLD_STAGES = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_F_STAGES", "2"))
_PREP_WARPS = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_P_WARPS", "4"))
_PREP_STAGES = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_P_STAGES", "2"))
_HO_WARPS = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_HO_WARPS", "4"))
_HO_STAGES = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_HO_STAGES", "2"))
_HO_BV = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_HO_BV", "0"))
# One-pass state scan with two chunks per iteration.  Only for fixed-length
# BF16 grids where the two-chunk working set still fits UB.
_FWD_UNROLL2_ENV = "FLAG_ATTN_ASCEND_GDN_FWD_UNROLL2"


def fwd_unroll2_enabled() -> bool:
    forced = os.environ.get(_FWD_UNROLL2_ENV, "").lower()
    return forced not in _OFF_VALUES if forced else True


# The fast path is host-bound, so its per-call intermediate tensors are cached
# (keyed by shape/dtype/device) instead of being allocated and freed every
# call.  ``o`` and the final state are always freshly allocated (they are
# returned to the caller); only never-returned scratch is pooled.  The cache is
# bounded, and can be disabled with FLAG_ATTN_ASCEND_GDN_SCRATCH=0.
_SCRATCH_LIMIT = 8
_scratch: "OrderedDict[tuple, dict]" = OrderedDict()


_OFF_VALUES = {"0", "false", "off", "no"}
# v1 fold trim: only materialise k_e/k_r in the strided fold and recompute
# qe/vb inside the affine kernel, which removes the qe/vb GM round trips.
# Only for BF16 fixed-length Raw-prep grids (jobs >= 2048); the FP16 path still
# uses the full 4-output fold.
_FOLD_TRIM_ENV = "FLAG_ATTN_ASCEND_GDN_FOLD_TRIM"


def fold_trim_enabled() -> bool:
    forced = os.environ.get(_FOLD_TRIM_ENV, "").lower()
    return forced not in _OFF_VALUES if forced else True

_SCRATCH_ENABLED = (
    os.environ.get("FLAG_ATTN_ASCEND_GDN_SCRATCH", "1") not in _OFF_VALUES
)


def _scratch_enabled() -> bool:
    return _SCRATCH_ENABLED


def _fast_scratch(B, T, H, K, V, BT, dtype, device) -> dict:
    key = (B, T, H, K, V, BT, dtype, device.type, device.index)
    ent = _scratch.get(key)
    if ent is not None:
        _scratch.move_to_end(key)
        return ent
    if len(_scratch) >= _SCRATCH_LIMIT:
        # Defensive: the evicted buffers may still be referenced by kernels that
        # are queued but not executed yet.  A sync here is cheap (only happens
        # when a new shape is first seen) and keeps eviction safe.
        torch.npu.synchronize()
        _scratch.popitem(last=False)
    fp = torch.float32
    fold_dtype = torch.float32 if dtype == torch.float16 else dtype
    ent = {
        "g_cum": torch.empty(B, T, H, dtype=fp, device=device),
        "e1": torch.empty(B, T, H, dtype=fp, device=device),
        "r1": torch.empty(B, T, H, dtype=fp, device=device),
        "b1": torch.empty(B, T, H, dtype=fp, device=device),
        "be1": torch.empty(B, T, H, dtype=fp, device=device),
        "qe": torch.empty(B, T, H, K, dtype=fold_dtype, device=device),
        "ke": torch.empty(B, T, H, K, dtype=fold_dtype, device=device),
        "vb": torch.empty(B, T, H, V, dtype=fold_dtype, device=device),
        "kr": torch.empty(B, T, H, K, dtype=fold_dtype, device=device),
        "w": torch.empty(B, T, H, K, dtype=dtype, device=device),
        "u": torch.empty(B, T, H, V, dtype=dtype, device=device),
        "a": torch.empty(B, T, H, BT, dtype=dtype, device=device),
    }
    _scratch[key] = ent
    return ent


def clear_scratch_cache() -> None:
    """Drop the scratch pool (e.g. before releasing the device)."""
    _scratch.clear()


def _fold_config() -> tuple[int, int]:
    return _FOLD_WARPS, _FOLD_STAGES


def _prep_wu_config() -> tuple[int, int]:
    return _PREP_WARPS, _PREP_STAGES


def _fwd_h_o_bv(V: int) -> int:
    return _HO_BV or min(V, 64)


def _fwd_h_o_config() -> tuple[int, int]:
    return _HO_WARPS, _HO_STAGES


def _lk(buf, key, fn, grid, warps, stages):
    """Opt-in cached launcher for one GDN kernel site.

    ``fast_launch`` returns a callable that behaves like ``fn[grid]`` but skips
    Triton's per-call launch preparation after the first call.  ``key`` names
    the site inside the per-shape scratch entry (``buf["lk"]``), so a hot launch
    costs one small dictionary lookup; without scratch the launcher falls back
    to its module-level site cache.  Fallback to the regular launch is handled
    by the launcher itself (hooks, disabled via env, incompatible Triton build).
    """
    cache = None
    if buf is not None:
        cache = buf.get("lk")
        if cache is None:
            cache = buf["lk"] = {}
    return fast_launch(fn, grid, warps, stages, key, cache)



def fold_headfirst_strided_trim(g, q, k, v, beta, chunk_size):
    """v1 fold trim: write only k_e/k_r plus the fp32 factor buffers.

    Returns ``(g_cum, ke, kr, ctx)`` where ``ctx = (q, v, e1, b1, factor_t,
    rep)`` is handed to ``raw_inverse_prep_affine_trim_ascend``, which
    recomputes qe/vb while it already reads those sources.
    """
    B, T, Hg, K = q.shape
    H, V = beta.shape[-1], v.shape[-1]
    BT = int(chunk_size)
    NC = triton.cdiv(T, BT)
    BH = 16
    factor_t = triton.cdiv(T, 32) * 32
    factor_shape = (B, H, factor_t)
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, q.dtype, q.device)
        g_cum = buf["g_cum"]
        ke, kr = buf["ke"], buf["kr"]
        if "headfirst_e1" not in buf:
            buf["headfirst_e1"] = torch.empty(
                factor_shape, dtype=torch.float32, device=g.device,
            )
            buf["headfirst_r1"] = torch.empty_like(buf["headfirst_e1"])
            buf["headfirst_b1"] = torch.empty_like(buf["headfirst_e1"])
            buf["headfirst_be1"] = torch.empty_like(buf["headfirst_e1"])
        e1 = buf["headfirst_e1"]
        r1 = buf["headfirst_r1"]
        b1 = buf["headfirst_b1"]
        be1 = buf["headfirst_be1"]
    else:
        g_cum = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        ke = torch.empty(B, T, H, K, dtype=q.dtype, device=q.device)
        kr = torch.empty_like(ke)
        e1 = torch.empty(factor_shape, dtype=torch.float32, device=g.device)
        r1 = torch.empty_like(e1)
        b1 = torch.empty_like(e1)
        be1 = torch.empty_like(e1)
    warps, stages = _fold_config()
    fast_launch(_factor_headfirst_stride_kernel, (NC, B * triton.cdiv(H, BH)), warps, stages)(
        g, beta, g_cum, e1, r1, b1, be1, T,
        H, BT, BH, factor_t, *g.stride(), *beta.stride(),
    )
    fast_launch(_fold_headfirst_stride_trim_kernel, (NC, B * H), warps, stages)(
        k, r1, be1, ke, kr, T,
        H, Hg, K, V, BT, factor_t, *k.stride(),
    )
    ctx = (q, v, e1, b1, factor_t, H // Hg)
    return g_cum, ke, kr, ctx


def fold_ascend(
    g: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    head_first_strided: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Return ``(g_cum, q_e, k_e, k_r, v_beta)``.

    Fixed-length inputs choose between the stride-aware fold, the one-launch
    gather fold for small grids, and the two Triton vector kernels
    (``_factor_kernel`` + ``_fold_kernel``); varlen inputs always take the two
    vector kernels.
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    V = v.shape[-1]
    BT = int(chunk_size)
    is_varlen = cu_seqlens is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if is_varlen else None
    NC = triton.cdiv(T, BT) if not is_varlen else len(chunk_indices)
    if (
        head_first_strided
        and not is_varlen
        and strided_fold_supported(
            B, T, H, Hg, K, V, q.dtype, cu_seqlens, BT
        )
    ):
        return _fold_headfirst_strided(g, q, k, v, beta, BT)
    buf = None
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, k.dtype, k.device)
        g_cum = buf["g_cum"]
        qe, ke, kr, vb = buf["qe"], buf["ke"], buf["kr"], buf["vb"]
    else:
        g_cum = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        fold_dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
        qe = torch.empty(B, T, H, K, dtype=fold_dtype, device=q.device)
        ke = torch.empty(B, T, H, K, dtype=fold_dtype, device=q.device)
        kr = torch.empty(B, T, H, K, dtype=fold_dtype, device=q.device)
        vb = torch.empty(B, T, H, V, dtype=fold_dtype, device=v.device)
    warps, stages = _fold_config()
    if (
        q.dtype != torch.float16
        and k.shape[-1] == v.shape[-1]
        and NC * H <= _SMALL_FOLD_PROGRAMS
        and T <= _GATHER_MAX_T
    ):
        # Small/short workload: one launch (serialized 2D gathers are cheap when
        # the grid is small and the sequence is short).
        BK = 64 if K % 64 == 0 else K
        key = ("gather", T, H, Hg, K, V, BT, BK, is_varlen, qe.dtype)
        grid = (NC, H if is_varlen else B * H, K // BK)
        _lk(buf, key, _fold_gather_kernel, grid, warps, stages)(
            q, k, v, g, beta, g_cum, qe, ke, kr, vb,
            cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, BK, is_varlen,
        )
        return g_cum, qe, ke, kr, vb
    if _scratch_enabled():
        e1, r1, b1, be1 = buf["e1"], buf["r1"], buf["b1"], buf["be1"]
    else:
        e1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        r1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        b1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        be1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
    nchunk = _fold_nchunk(NC, is_varlen)
    use_head_tiled_factor = (
        _FACTOR_HEAD_TILE_ENABLED
        and not is_varlen
        and BT == 64
        and H >= 8
        and NC * H > _SMALL_FOLD_PROGRAMS
        and nchunk == 1
        and q.dtype in (torch.bfloat16, torch.float16)
    )
    if use_head_tiled_factor:
        launch_factor_head_tile(
            g,
            beta,
            g_cum,
            e1,
            r1,
            b1,
            be1,
            chunk_size=BT,
            heads_per_program=16,
            num_warps=warps,
            num_stages=stages,
            buf=buf,
        )
    else:
        key = ("factor", T, H, BT, is_varlen, nchunk, beta.dtype)
        grid = (NC // nchunk, H if is_varlen else B * H)
        _lk(buf, key, _factor_kernel, grid, warps, stages)(
            g, beta, g_cum, e1, r1, b1, be1,
            cu_seqlens, chunk_indices, T, H, BT, is_varlen, nchunk,
        )
    rep = H // Hg
    use_group_kloop = (
        _FOLD_GROUP_KLOOP_ENABLED
        and use_head_tiled_factor
        and not is_varlen
        and BT == 64
        and T >= 2048
        and q.dtype == torch.bfloat16
        and K == V == 128
        and H % Hg == 0
        and rep in (2, 4)
        and nchunk == 1
    )
    if use_group_kloop:
        launch_fold_group_kloop(
            q, k, v, e1, r1, b1, be1, qe, ke, kr, vb,
            chunk_size=BT,
            heads_per_program=rep,
            key_tile=32,
            num_warps=warps,
            num_stages=stages,
        )
    else:
        key = ("fold", T, H, Hg, K, V, BT, is_varlen, nchunk, qe.dtype)
        grid = (NC // nchunk, H if is_varlen else B * H)
        _lk(buf, key, _fold_kernel, grid, warps, stages)(
            q, k, v, e1, r1, b1, be1, qe, ke, kr, vb,
            cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, is_varlen, nchunk,
        )
    return g_cum, qe, ke, kr, vb


def prep_w_u_ascend(
    qe: torch.Tensor,
    ke: torch.Tensor,
    kr: torch.Tensor,
    vb: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(w, u, A)``: KKT/Newton inverse applied to the folded operands."""
    B, T, H, K = ke.shape
    V = vb.shape[-1]
    BT = int(chunk_size)
    output_dtype = output_dtype or ke.dtype
    BV = min(V, 128)  # single V tile in the kernel (V <= 128 on the fast path)
    is_varlen = cu_seqlens is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if is_varlen else None
    NC = triton.cdiv(T, BT) if not is_varlen else len(chunk_indices)
    buf = None
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, output_dtype, ke.device)
        w, u, a_masked = buf["w"], buf["u"], buf["a"]
    else:
        w = torch.empty(B, T, H, K, dtype=output_dtype, device=ke.device)
        u = torch.empty(B, T, H, V, dtype=output_dtype, device=vb.device)
        a_masked = torch.empty(B, T, H, BT, dtype=output_dtype, device=ke.device)
    warps, stages = _prep_wu_config()
    key = (
        "prep", T, H, H, K, V, BT, BV, max(1, BT.bit_length() - 2),
        is_varlen, False, qe.dtype,
    )
    grid = (NC, H if is_varlen else B * H)
    _lk(buf, key, _prep_wu_kernel, grid, warps, stages)(
        qe, ke, kr, vb, w, u, a_masked,
        cu_seqlens, chunk_indices, T, H, H, K, V, BT, BV,
        max(1, BT.bit_length() - 2), is_varlen, False,
        w, u, w, u, NC,
    )
    return w, u, a_masked


def prep_affine_ascend(
    qe: torch.Tensor,
    ke: torch.Tensor,
    kr: torch.Tensor,
    vb: torch.Tensor,
    chunk_size: int,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare A and two-pass affine operands without materializing W/U."""
    B, T, H, K = ke.shape
    V = vb.shape[-1]
    BT = int(chunk_size)
    NT = triton.cdiv(T, BT)
    output_dtype = output_dtype or ke.dtype
    buf = None
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, output_dtype, ke.device)
        w, u, a_masked = buf["w"], buf["u"], buf["a"]
    else:
        w = torch.empty(B, T, H, K, dtype=output_dtype, device=ke.device)
        u = torch.empty(B, T, H, V, dtype=output_dtype, device=vb.device)
        a_masked = torch.empty(B, T, H, BT, dtype=output_dtype, device=ke.device)
    qe2 = qe.new_empty(B, T, H, K)
    au = torch.empty(B, T, H, V, dtype=output_dtype, device=vb.device)
    cu = qe.new_empty(B, NT, H, K, V)
    mw = qe.new_empty(B, NT, H, K, K)
    BV = min(V, 128)
    key = (
        "prep_affine", T, H, H, K, V, BT, BV, max(1, BT.bit_length() - 2),
        False, True, NT, qe.dtype,
    )
    _lk(buf, key, _prep_wu_kernel, (NT, B * H), _PREP_WARPS, _PREP_STAGES)(
        qe, ke, kr, vb, w, u, a_masked, None, None,
        T, H, H, K, V, BT, BV, max(1, BT.bit_length() - 2), False,
        True, qe2, au, cu, mw, NT,
    )
    return qe2, au, cu, mw, a_masked


def fwd_h_o_ascend(
    qe: torch.Tensor,
    a_in: torch.Tensor,
    kr: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    bv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(o, final_state)``: state scan fused with the output stage."""
    B, T, H, _ = u.shape
    V = u.shape[-1]
    Hg, K = qe.shape[-2], qe.shape[-1]
    BT = int(chunk_size)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BV = min(V, int(bv) if bv else _fwd_h_o_bv(V))
    o = torch.empty_like(u)
    final_state = (
        u.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    warps, stages = _fwd_h_o_config()
    buf = (_fast_scratch(B, T, H, K, V, BT, u.dtype, u.device)
           if _scratch_enabled() else None)
    NT = triton.cdiv(T, BT)
    unroll2 = (
        fwd_unroll2_enabled()
        and cu_seqlens is None
        and u.dtype == torch.bfloat16
        and BT == 64
        and NT >= 8
        and NT % 2 == 0
        and 8 <= B * H <= 32
        and K <= 128
        and V <= 128
    )
    if unroll2:
        key = (
            "fwd_h_o_u2", T, H, Hg, K, V, BT, BV, NT,
            g is not None, initial_state is not None,
            final_state is not None, u.dtype,
        )
        grid = (triton.cdiv(V, BV), N * H)
        _lk(buf, key, _fwd_h_o_unroll2_kernel, grid,
            num_warps or warps, num_stages or stages)(
            qe, kr, a_in, w, u, g, o, initial_state, final_state, scale,
            cu_seqlens, T, H, Hg, K, V, BT, BV,
            g is not None,
            initial_state is not None,
            final_state is not None,
            NT,
        )
        return o, final_state
    key = (
        "fwd_h_o", T, H, Hg, K, V, BT, BV, g is not None,
        initial_state is not None, final_state is not None,
        cu_seqlens is not None, u.dtype,
    )
    grid = (triton.cdiv(V, BV), N * H)
    _lk(buf, key, _fwd_h_o_kernel, grid,
        num_warps or warps, num_stages or stages)(
        qe, kr, a_in, w, u, g, o, initial_state, final_state, scale,
        cu_seqlens, T, H, Hg, K, V, BT, BV,
        g is not None,
        initial_state is not None,
        final_state is not None,
        cu_seqlens is not None,
    )
    return o, final_state


def affine_precompute_ascend(
    qe: torch.Tensor,
    a_in: torch.Tensor,
    kr: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass scan pass 1: chunk-parallel ``(qe2, Au, Cu, Mw)`` buffers."""
    B, T, H, K = qe.shape
    V = u.shape[-1]
    BT = int(chunk_size)
    NT = triton.cdiv(T, BT)
    # Not the default path: allocate fresh (these buffers are large: Cu/Mw are
    # K x V / K x K per chunk, so they are not pooled in the scratch cache).
    qe2 = qe.new_empty(B, T, H, K)
    au = u.new_empty(B, T, H, V)
    cu = qe.new_empty(B, NT, H, K, V)
    mw = qe.new_empty(B, NT, H, K, K)
    warps, stages = _prep_wu_config()
    buf = (_fast_scratch(B, T, H, K, V, BT, qe.dtype, qe.device)
           if _scratch_enabled() else None)
    _lk(buf, ("affine", T, H, K, V, BT, NT, qe.dtype), _affine_kernel, (NT, B * H), warps, stages)(
        qe, a_in, kr, w, u, qe2, au, cu, mw,
        T, H, K, V, BT, NT,
    )
    return qe2, au, cu, mw


def fwd_h_o_2pass_ascend(
    qe2: torch.Tensor,
    au: torch.Tensor,
    cu: torch.Tensor,
    mw: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    bv: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Two-pass scan pass 2: chunk-parallel output + 2-dot state advance."""
    B, T, H, _ = au.shape
    V = au.shape[-1]
    K = qe2.shape[-1]
    BT = int(chunk_size)
    NT = triton.cdiv(T, BT)
    forced_bv = os.environ.get("FLAG_ATTN_ASCEND_GDN_2PASS_BV", "")
    if bv:
        BV = min(V, int(bv))
    elif forced_bv and int(forced_bv) > 0:
        BV = min(V, int(forced_bv))
    elif au.dtype == torch.bfloat16 and K == V == 128:
        # Full-width BF16 tile removes the duplicated per-chunk Mw/Qe2 reads,
        # but it also caps the grid at B*H programs.  For tiny grids
        # (B*H <= 8) halving the tile doubles the grid and is the faster
        # choice, so 8 is the empirical crossover on this backend.
        BV = 64 if B * H <= 8 else 128
    else:
        BV = min(V, _fwd_h_o_bv(V))
    o = torch.empty_like(au)
    final_state = (
        au.new_empty(B, H, K, V, dtype=torch.float32)
        if output_final_state
        else None
    )
    warps, stages = _fwd_h_o_config()
    buf = (_fast_scratch(B, T, H, K, V, BT, au.dtype, au.device)
           if _scratch_enabled() else None)
    key = (
        "fwd_h_o_2pass", T, H, K, V, BT, BV, NT,
        initial_state is not None, final_state is not None, au.dtype,
    )
    _lk(buf, key, _fwd_h_o_2pass_kernel, (triton.cdiv(V, BV), B * H),
        warps, stages)(
        qe2, au, cu, mw, g, o, initial_state, final_state, scale,
        T, H, K, V, BT, BV, NT,
        initial_state is not None,
        final_state is not None,
    )
    return o, final_state


__all__ = [
    "fold_ascend",
    "prep_w_u_ascend",
    "prep_affine_ascend",
    "fwd_h_o_ascend",
    "affine_precompute_ascend",
    "fwd_h_o_2pass_ascend",
    "clear_scratch_cache",
]
