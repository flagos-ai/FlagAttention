# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend-only GDN v1 chunk-local cumsum, KKT/solve_tril and ``recompute_w_u``.

This module holds Ascend replacements for
``flag_attn.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril`` (the version in
``kkt.py``) and ``flag_attn.FLA.gated_delta_rule.wy_fast.recompute_w_u_fwd``.
The generic and ``kkt.py`` files are left untouched.

Why a separate copy
-------------------
``kkt.py`` folds the decay into the KKT dot operands and uses Newton doubling,
but that fold uses a 1D->2D broadcast multiply::

    b_L = tl.dot(b_k * (b_beta * b_e)[:, None], ...)

On Ascend any broadcast of a ``[BT]`` vector into a ``[BT, K]`` tile
(``v[:, None] * tile``) is much slower than a plain 2D x 2D multiply, and a
per-program broadcast multiply can dominate the cost of the whole pipeline.

The form used here (and in ``chunk_o_ascend.py`` / ``fwd_h_ascend.py``) is to
materialise the per-token factor as a *2D tile* with a broadcast *address*
computation instead of broadcasting a loaded 1D vector::

    o_t = i_t * BT + tl.arange(0, BT)
    b_g2 = tl.load(g + base + o_t[:, None] * H + 0 * tl.arange(0, K)[None, :],
                   mask=..., other=0.)

The integer address arithmetic is cheap and the resulting ``[BT, K]`` tile
multiplies a loaded tile at full 2D speed.  Same trick for the transposed
``[K, BT]`` factor used by the KKT dot.

Numerics/behaviour are unchanged relative to ``kkt.py``:
``A_inv = (I + L)^-1`` with ``L[i, j] = beta_i (k_i . k_j) exp(g_i - g_j)``
strictly lower triangular, inverted by Newton doubling with
``log2(BT) - 1`` steps.  ``g`` is the chunk-local cumsum in fp32, which is also
returned because ``fwd_h``/``chunk_o`` need it.
"""

from __future__ import annotations

import os

import torch
import triton

from .fast_launch import fast_launch
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit(do_not_specialize=["T"])
def _chunk_cumsum_kernel(
    g,
    g_out,
    beta,
    beta_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BETA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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
    p_g = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_o = tl.make_block_ptr(
        g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    tl.store(p_o, b_g, boundary_check=(0,))
    if HAS_BETA:
        p_b = tl.make_block_ptr(
            beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        p_bo = tl.make_block_ptr(
            beta_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
        )
        tl.store(p_bo, tl.load(p_b, boundary_check=(0,)).to(tl.float32),
                 boundary_check=(0,))


@triton.jit(do_not_specialize=["T"])
def _prep_kernel(
    g_cum,
    k,
    beta,
    A_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    N_NEWTON: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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
    i_hg = i_h // (H // Hg)

    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, K)
    m_t = o_t < T
    base = bos * H + i_h

    # [BT, K] factor: beta_i * exp(g_i) and [K, BT] factor exp(-g_j), both
    # gathered as 2D tiles so the decay is applied with 2D x 2D multiplies.
    b_g2 = tl.load(
        g_cum + base + o_t[:, None] * H + 0 * o_k[None, :],
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.float32)
    # ``beta`` is the fp32 copy materialised by ``_chunk_cumsum_kernel``: a
    # 2D gather from an fp32 buffer is fast on this backend while a bf16
    # gather is much slower.
    b_beta2 = tl.load(
        beta + base + o_t[:, None] * H + 0 * o_k[None, :],
        mask=m_t[:, None],
        other=0.0,
    )
    b_fac = (b_beta2 * tl.exp(b_g2)).to(tl.bfloat16)
    b_gr = tl.load(
        g_cum + base + o_t[None, :] * H + 0 * o_k[:, None],
        mask=m_t[None, :],
        other=0.0,
    ).to(tl.float32)
    b_r = tl.exp(tl.minimum(-b_gr, 80.0)).to(tl.bfloat16)

    p_k = tl.make_block_ptr(
        k + (bos * Hg + i_hg) * K,
        (T, K),
        (Hg * K, 1),
        (i_t * BT, 0),
        (BT, K),
        (1, 0),
    )
    p_kt = tl.make_block_ptr(
        k + (bos * Hg + i_hg) * K,
        (K, T),
        (1, Hg * K),
        (0, i_t * BT),
        (K, BT),
        (0, 1),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kt = tl.load(p_kt, boundary_check=(0, 1))
    b_ke = b_k * b_fac
    b_kr = b_kt * b_r
    b_L = tl.dot(b_ke, b_kr)
    b_L = tl.where(o_t[:, None] > o_t[None, :], b_L, 0.0)

    o_r = tl.arange(0, BT)
    m_I = (o_r[:, None] == o_r[None, :]).to(tl.float32)
    b_M = b_L + m_I
    b_Ai = m_I - b_L
    for _ in tl.static_range(0, N_NEWTON):
        b_Y = tl.dot(b_M, b_Ai)
        b_Ai = tl.dot(b_Ai, 2.0 * m_I - b_Y)

    p_A = tl.make_block_ptr(
        A_inv + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A, b_Ai.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _recompute_w_u_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g_cum,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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
    i_hg = i_h // (H // Hg)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    base = bos * H + i_h

    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v = tl.make_block_ptr(
            v + base * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + base * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_beta2 = tl.load(
            beta + base + o_t[:, None] * H + 0 * o_v[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        b_vb = (b_v.to(tl.float32) * b_beta2).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_hg) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g2 = tl.load(
            g_cum + base + o_t[:, None] * H + 0 * o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        b_beta2 = tl.load(
            beta + base + o_t[:, None] * H + 0 * o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        )
        # beta_i * exp(g_i) as a 2D tile; product of two 2D tiles (fast) rather
        # than a broadcast multiply of 1D vectors (slow on Ascend, see header).
        b_fac = (b_beta2 * tl.exp(b_g2)).to(b_k.dtype)
        b_kb = b_k * b_fac
        b_w = tl.dot(b_A, b_kb)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def _prep_params() -> tuple[int, int]:
    warps = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_P_WARPS", "4"))
    stages = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_P_STAGES", "2"))
    return warps, stages


def chunk_local_cumsum_ascend(
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Chunk-local cumsum in fp32 (and an fp32 copy of ``beta`` if given).

    ``beta`` is widened here only so that downstream kernels can load its
    per-token values with a *2D gather from an fp32 buffer*: bf16 2D gathers
    are much slower than fp32 gathers on this backend.
    """
    B, T, H = g.shape
    BT = int(chunk_size)
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NC = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_out = torch.empty_like(g, dtype=torch.float32)
    beta32 = None
    if beta is not None:
        beta32 = torch.empty(B, T, beta.shape[-1], device=g.device,
                             dtype=torch.float32)
    _chunk_cumsum_kernel[(NC, H if cu_seqlens is not None else B * H)](
        g,
        g_out,
        beta if beta is not None else g,
        beta32 if beta32 is not None else g_out,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        BT=BT,
        HAS_BETA=beta is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    if beta is None:
        return g_out
    return g_out, beta32


def chunk_gated_delta_rule_prep_ascend(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = None,
    g_cum: torch.Tensor | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(chunk_local_cumsum(g), (I - L)^-1)`` for GDN v1 chunking.

    ``beta`` must be fp32 (the ``beta32`` returned by
    ``chunk_local_cumsum_ascend``); a bf16 ``beta`` is widened here for
    convenience, which costs one extra tiny elementwise op.
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    output_dtype = output_dtype or k.dtype
    if beta.dtype != torch.float32:
        beta = beta.float()
    if g_cum is None:
        g_cum, beta = chunk_local_cumsum_ascend(
            g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=BT
        )
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NC = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    n_newton = max(1, BT.bit_length() - 2)
    # ``torch.empty`` rather than ``zeros``: every element that is read back is
    # written below (reads use boundary_check, so the tail-padding rows are
    # never consumed), and this avoids a memset launch on the staged path.
    A_inv = torch.empty(B, T, H, BT, device=g.device, dtype=output_dtype)
    warps, stages = _prep_params()
    fast_launch(_prep_kernel, (NC, H if cu_seqlens is not None else B * H), num_warps or warps, num_stages or stages)(
        g_cum,
        k,
        beta,
        A_inv,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        N_NEWTON=n_newton,
        IS_VARLEN=cu_seqlens is not None,

    )
    return g_cum, A_inv


def _wu_params() -> tuple[int, int]:
    warps = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_WU_WARPS", "4"))
    stages = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_WU_STAGES", "2"))
    return warps, stages


def recompute_w_u_fwd_ascend(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    block_v: int | None = None,
    pipe_stages: int = 2,
    bk: int | None = None,
    bv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del pipe_stages  # signature parity with the generic wrapper
    if beta.dtype != torch.float32:
        beta = beta.float()
    B, T, Hg, K = k.shape
    V = v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NC = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(K, int(bk or 128))
    BV = min(V, int(bv or block_v or 128))
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    warps, stages = _wu_params()
    grid = (NC, H if cu_seqlens is not None else B * H)
    fast_launch(
        _recompute_w_u_kernel, grid, num_warps or warps, num_stages or stages,
    )(
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g_cum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,

    )
    return w, u


__all__ = [
    "chunk_local_cumsum_ascend",
    "chunk_gated_delta_rule_prep_ascend",
    "recompute_w_u_fwd_ascend",
]
