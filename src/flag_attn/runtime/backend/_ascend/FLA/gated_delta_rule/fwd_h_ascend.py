# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend-only ``chunk_gated_delta_rule_fwd_h`` for GDN v1.

Same recurrence as ``flag_attn.FLA.chunk_delta_h`` (per chunk ``i_t``:
``v_new = u - w h``, ``h <- exp(g_last) h + k^T (v_new * exp(g_last - g))``,
storing the chunk-start state ``h`` for ``chunk_o``), with three Ascend
changes; the generic module is left untouched:

* **No autotune.**  The generic wrapper searched 4 warps * 3 stages * 2 BV = 24
  configs per shape key, which costs a long per-process cold start while the
  kernel only has ``V/BV * N*H`` programs.
* **K is a single tile.**  The generic kernel keeps ``64 x BV`` fp32 state
  tiles (``b_h1..b_h4``) and walks K in 64-wide slabs; for K=128 that is two
  slabs, two cube dots per chunk and two state stores/loads.  Here the state
  tile is ``K x BV`` (single tile for K<=128) so each chunk is one ``BT x K @
  K x BV`` dot and one ``K x BT @ BT x BV`` dot (no loop-carried accumulator
  over K slabs).
* **V-axis splitting** (``bv``): the recurrence is sequential over chunks, so
  the only parallelism is over the ``V/BV`` state column tiles times ``N*H``
  heads.  Fewer, larger programs win on this backend, so ``bv`` defaults to 64
  at BT=64 (an empirical value for this backend).
"""

from __future__ import annotations

import os

import torch
import triton

from .fast_launch import fast_launch
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices, prepare_chunk_offsets


@triton.jit(do_not_specialize=["T"])
def _fwd_h_kernel(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
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
    SAVE_NEW_VALUE: tl.constexpr,
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
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h = tl.zeros([K, BV], dtype=tl.float32)

    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * Hg + i_h // (H // Hg)).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
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
        p_h = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_u = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)
        b_v = b_v - tl.dot(b_w, b_h.to(b_w.dtype))
        if SAVE_NEW_VALUE:
            p_vn = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_vn, b_v.to(p_vn.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            o_t = i_t * BT + tl.arange(0, BT)
            o_v = i_v * BV + tl.arange(0, BV)
            m_t = o_t < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h).to(tl.float32)
            # Gather the per-token factor as a [BT, BV] tile: broadcasting a
            # loaded [BT] vector (``v[:, None] * tile``) is slow on this
            # backend while a 2D x 2D multiply is cheap (see
            # ``wy_fast_ascend.py``).  The row scaling is otherwise identical
            # to the generic kernel.
            gb = g + bos * H + i_h
            b_g2 = tl.load(
                gb + o_t[:, None] * H + 0 * o_v[None, :],
                mask=m_t[:, None],
                other=0.0,
            ).to(tl.float32)
            b_fac = tl.where(m_t[:, None], tl.exp(b_g_last - b_g2), 0.0)
            b_v = b_v * b_fac
            b_h *= tl.exp(b_g_last)

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (K, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def _default_bv(V: int) -> int:
    # BV=64 is the empirical value at BT=64 on this backend.
    return int(os.environ.get("FLAG_ATTN_ASCEND_GDN_H_BV", "0")) or min(V, 64)


def _default_warps() -> int:
    return int(os.environ.get("FLAG_ATTN_ASCEND_GDN_H_WARPS", "4"))


def _default_stages() -> int:
    return int(os.environ.get("FLAG_ATTN_ASCEND_GDN_H_STAGES", "2"))


def chunk_gated_delta_rule_fwd_h_ascend(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    bv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u) if save_new_value else None
    BV = min(V, int(bv) if bv else _default_bv(V))

    grid = (triton.cdiv(V, BV), N * H)
    fast_launch(
        _fwd_h_kernel, grid, num_warps or _default_warps(),
        num_stages or _default_stages(),
    )(
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=g is not None,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        SAVE_NEW_VALUE=save_new_value,
        IS_VARLEN=cu_seqlens is not None,

    )
    return h, v_new, final_state


__all__ = ["chunk_gated_delta_rule_fwd_h_ascend"]
