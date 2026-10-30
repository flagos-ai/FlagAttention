# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend prep kernel for GDN v1: chunk-local cumsum + KKT + solve_tril.

The generic ``flag_attn.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril``
fuses cumsum, the KKT matrix and its triangular inverse into a single kernel
whose UB footprint at BT=64 is 4_243_456 bits (530KB) against 192KB, so it
cannot compile for Ascend 910B.  Its inverse is also hard-coded to four 16x16
diagonal blocks, so simply lowering ``chunk_size`` produces wrong results.

This module keeps BT a parameter (default 32) and computes the inverse the
generic kernel expects, ``(I + L)^-1``, by Newton doubling on BT x BT fp32
tiles: ``L`` is strictly lower triangular (nilpotent, ``L^BT = 0``), so starting
from ``X0 = I - L`` (residual ``L^2``) each step
``X_{k+1} = X_k (2I - (I + L) X_k)`` squares the residual and ``log2(BT) - 1``
steps are exact -- 2*4 = 8 small ``tl.dot`` for BT=32 instead of BT-1 sequential
masked reductions.  The input-side decay folding described below removes the
elementwise pass over the BT x BT dot result.

``L`` is the strictly-lower chunk matrix ``beta_i * (k_i . k_j) * exp(g_i - g_j)``
(the same quantity the generic kernel stores in ``A``), so the result matches
the generic ``A_inv`` up to the BT granularity.

Three backend constraints shape the kernel:

* the KKT dot must cover the *whole* head dim in a single ``tl.dot``.  The
  usual ``for i_k in range(cdiv(K, BK)): b_L += tl.dot(...)`` accumulator form
  is miscompiled as soon as the loop has more than one trip; loading the full
  head dim in one block and issuing one dot keeps the accumulator exact.
* the decay ``exp(g_i - g_j)`` is folded into the operands of that single dot
  (``beta_i exp(g_i) k_i`` against ``exp(-g_j) k_j``) instead of being applied
  to the BT x BT dot result.  Elementwise work on the cube output of a dot
  (in particular the broadcast outer product of the gate) forces a layout
  conversion, so the decay is folded into the dot operands and the whole
  L/Newton chain stays in one layout.
* the chunk-local cumsum is written as fp32 (see
  ``chunk_gated_delta_rule_prep``).  bf16 cumsum storage accumulates too much
  error because ``|sum(g)| ~ 0.5 * BT`` makes the bf16 ulp of the exponent
  huge compared with ``exp(g_i - g_j)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit
def _chunk_gated_delta_rule_prep_kernel(
    g_in,
    g_out,
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
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T

    # chunk-local cumulative sum of the log decay
    p_g_in = tl.make_block_ptr(
        g_in + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g_in, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    p_g_out = tl.make_block_ptr(
        g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    tl.store(p_g_out, b_g.to(p_g_out.dtype.element_ty), boundary_check=(0,))

    # KKT matrix L = strict_lower(beta_i * k_i . k_j * exp(g_i - g_j))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    # NOTE: the natural ``for i_k in range(tl.cdiv(K, BK)): b_L += tl.dot(...)``
    # form is miscompiled on this backend: as soon as the loop has more than one
    # trip (K > BK) the fp32 accumulator carried through the loop is
    # unreliable.  Load the full head dim in a single block and use one dot.
    #
    # The decay is folded into the *inputs* of that dot:
    #     L[i, j] = (beta_i exp(g_i) k_i) . (exp(-g_j) k_j)
    # instead of scaling the BT x BT dot result by exp(g_i - g_j).  Elementwise
    # decay on the cube output forces a layout conversion (and a second one per
    # Newton step); the input-side scaling runs on 2*BT values and keeps the
    # whole L/Newton chain in one layout.  ``exp(-g)`` is clamped to e^80:
    # exp(g_i - g_j) <= 1 for the kept strict-lower part, so the clamp only
    # touches contributions that are already below fp32 resolution.
    b_e = tl.exp(b_g)
    b_r = tl.exp(tl.minimum(-b_g, 80.0))
    p_k = tl.make_block_ptr(
        k + (bos * Hg + i_h // (H // Hg)) * K,
        (T, K),
        (Hg * K, 1),
        (i_t * BT, 0),
        (BT, K),
        (1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_L = tl.dot(
        b_k * (b_beta * b_e)[:, None],
        tl.trans(b_k * b_r[:, None]),
    )
    b_L = tl.where((o_t[:, None] > o_t[None, :]) & m_t[:, None], b_L, 0.0)

    # A_inv = (I + L)^-1 by Newton doubling (log2(BT)-1 steps instead of BT-1
    # sequential row eliminations).  L is strictly lower triangular, hence
    # nilpotent: L^BT = 0.  Starting from X0 = I - L makes the initial residual
    # E0 = I - (I+L)X0 = L^2 exact, and one Newton step squares the residual:
    #     E_{k+1} = E_k^2,   X_{k+1} = 2 X_k - X_k (I+L) X_k
    # so log2(BT) - 1 steps are exact in exact arithmetic (K = 4 for BT = 32),
    # i.e. 2*K tl.dot of BT x BT x BT instead of ~BT masked BT x BT reductions.
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


def chunk_gated_delta_rule_prep(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 32,
    output_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(chunk_local_cumsum(g), (I - L)^-1)`` for GDN v1 chunking."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    output_dtype = output_dtype or k.dtype
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    # X0 = I - L leaves residual L^2, so log2(BT)-1 squarings reach L^BT = 0.
    n_newton = max(1, BT.bit_length() - 2)
    # The chunk-local cumsum is kept in fp32 on purpose.  Its magnitude grows
    # with BT (|g| ~ 0.5 * BT for a fast-decay gate), so storing it in bf16
    # loses an absolute ~BT * eps / 2 in the exponent: at BT=32 / bf16 this is
    # ~0.03, i.e. a sizable relative error on exp(g_i - g_j) for *every* pair,
    # which accumulates through the downstream recurrence.  All downstream
    # generic kernels load g with `.to(tl.float32)`, so an fp32 g_cumsum is a
    # drop-in for them.
    g_out = torch.empty_like(g, dtype=torch.float32)
    A_inv = torch.zeros(B, T, H, BT, device=g.device, dtype=output_dtype)
    _chunk_gated_delta_rule_prep_kernel[(NT, H if cu_seqlens is not None else B * H)](
        g_in=g,
        g_out=g_out,
        k=k,
        beta=beta,
        A_inv=A_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        N_NEWTON=n_newton,
        IS_VARLEN=cu_seqlens is not None,
    )
    return g_out, A_inv


__all__ = ["chunk_gated_delta_rule_prep"]
