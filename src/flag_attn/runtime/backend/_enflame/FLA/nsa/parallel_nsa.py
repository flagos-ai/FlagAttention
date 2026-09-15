# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame-specific compatibility entry for parallel NSA."""

import torch
from flag_attn.parallel_nsa.parallel_nsa import (
    parallel_nsa as _generic_parallel_nsa,
)
import os
import triton
import triton.language as tl
from flag_attn.parallel_nsa.triton_ops_helper import (
    HAS_TLE,
    autotune_cache_kwargs,
    exp,
    log,
    tle,
)
from ..index import (
    prepare_lens_enflame,
)
from ..index import (
    prepare_token_indices_enflame as prepare_token_indices,
    prepare_chunk_offsets_enflame as prepare_chunk_offsets,
)


# Selected attention forward

MAX_GRID_X = 65535


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["token_offset"])
def _parallel_nsa_fwd_kernel_enflame(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    token_offset,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t = tl.program_id(0) + token_offset
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [G, BS] — compute QK^T scores
            b_s = tl.dot(b_q, b_k)
            # b_k registers are now free — will be reused for later loads
            b_s = tl.where(
                (i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf")
            )

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BS] — b_s registers reused for softmax output b_p
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            # [BS, BV] — load V tile AFTER score computation
            # so b_k registers are freed and peak register usage is reduced
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


# ===========================================================================
# TLE-optimized forward kernel (Triton Language Extensions)
# ===========================================================================

if HAS_TLE:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
            "USE_BLOCK_COUNTS": lambda args: isinstance(
                args["block_counts"], torch.Tensor
            ),
        }
    )
    @triton.autotune(
        configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
        key=["BS", "BK", "BV"],
        **autotune_cache_kwargs,
    )
    @triton.jit(do_not_specialize=["token_offset"])
    def _parallel_nsa_fwd_kernel_tle_enflame(
        q,
        k,
        v,
        o,
        lse,
        scale,
        block_indices,
        block_counts,
        cu_seqlens,
        token_indices,
        token_offset,
        T,
        H: tl.constexpr,
        HQ: tl.constexpr,
        G: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        S: tl.constexpr,
        BS: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        USE_BLOCK_COUNTS: tl.constexpr,
    ):
        i_t = tl.program_id(0) + token_offset
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
                token_indices + i_t * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        k += (bos * H + i_h) * K
        v += (bos * H + i_h) * V
        block_indices += (bos + i_t) * H * S + i_h * S

        if USE_BLOCK_COUNTS:
            NS = tl.load(block_counts + (bos + i_t) * H + i_h)
        else:
            NS = S

        p_q = tl.make_block_ptr(
            q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_o = tl.make_block_ptr(
            o + (bos + i_t) * HQ * V,
            (HQ, V),
            (V, 1),
            (i_h * G, i_v * BV),
            (G, BV),
            (1, 0),
        )
        p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_o = tl.zeros([G, BV], dtype=tl.float32)

        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        b_acc = tl.zeros([G], dtype=tl.float32)

        # Precompute indices for async K/V loads via tle.load(is_async=True).
        offs_k = tl.arange(0, BK)
        offs_s = tl.arange(0, BS)
        offs_v = tl.arange(0, BV)
        k_base = k  # already offset to (bos*H + i_h)*K
        v_base = v  # already offset to (bos*H + i_h)*V
        v_start = i_v * BV

        for i in range(NS):
            i_s = tl.load(block_indices + i).to(tl.int32) * BS
            if i_s <= i_t and i_s >= 0:
                # Async K load via regular pointer arithmetic
                k_ptrs = k_base + offs_k[:, None] + (i_s + offs_s[None, :]) * H * K
                k_mask = (offs_k[:, None] < K) & ((i_s + offs_s[None, :]) < T)
                b_k = tle.load(k_ptrs, mask=k_mask, other=0.0, is_async=True)

                # Compute QK^T scores
                b_s = tl.dot(b_q, b_k.to(b_q.dtype))
                b_s = tl.where(
                    (i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf")
                )

                # [G]
                b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                b_r = exp(b_mp - b_m)
                # [G, BS] — b_s registers reused for softmax output b_p
                b_p = exp(b_s - b_m[:, None])
                # [G]
                b_acc = b_acc * b_r + tl.sum(b_p, 1)

                # Async V load AFTER score computation
                v_ptrs = (
                    v_base
                    + (i_s + offs_s[:, None]) * H * V
                    + (v_start + offs_v[None, :])
                )
                v_mask = ((i_s + offs_s[:, None]) < T) & (
                    (v_start + offs_v[None, :]) < V
                )
                b_v = tle.load(v_ptrs, mask=v_mask, other=0.0, is_async=True)

                # [G, BV]
                b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

                b_mp = b_m
        b_o = b_o / b_acc[:, None]
        b_m += log(b_acc)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


def parallel_nsa_fwd_enflame(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    # The validated S60 path supports head dimensions
    # up to 128 and uses the GCU300-compatible tile size.
    BK = min(128, triton.next_power_of_2(K))
    BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    # Dispatch to TLE-optimized kernel when available.
    _use_tle = HAS_TLE and os.environ.get("FLA_NSA_TLE", "1") != "0"
    if _use_tle:
        kernel = _parallel_nsa_fwd_kernel_tle_enflame
    else:
        kernel = _parallel_nsa_fwd_kernel_enflame

    for token_offset in range(0, T, MAX_GRID_X):
        grid = (
            min(
                MAX_GRID_X,
                T - token_offset,
            ),
            NV,
            B * H,
        )
        kernel[grid](
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            scale=scale,
            block_indices=block_indices,
            block_counts=block_counts,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            token_offset=token_offset,
            T=T,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            S=S,
            BS=BS,
            BK=BK,
            BV=BV,
        )
    return o, lse


# Selected attention backward

MAX_GRID_X = 65535
MAX_GRID_Y = 255
MAX_GRID_Z = 255


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "token_offset",
        "value_offset",
        "bh_offset",
    ]
)
def _parallel_nsa_bwd_kernel_dq_enflame(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    token_offset,
    value_offset,
    bh_offset,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t = tl.program_id(0) + token_offset
    i_v = tl.program_id(1) + value_offset
    i_bh = tl.program_id(2) + bh_offset
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "value_offset",
        "block_offset",
        "bh_offset",
    ]
)
def _parallel_nsa_bwd_kernel_dkv_enflame(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    cu_seqlens,
    chunk_indices,
    scale,
    value_offset,
    block_offset,
    bh_offset,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v = tl.program_id(0) + value_offset
    i_s = tl.program_id(1) + block_offset
    i_bh = tl.program_id(2) + bh_offset
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(
            chunk_indices + i_s * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * all * H + bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_s * BS, 0),
        (BS, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(
                q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
            )
            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(
                do + (bos + i) * HQ * V,
                (HQ, V),
                (V, 1),
                (i_h * G, i_v * BV),
                (G, BV),
                (1, 0),
            )
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


class _ChunkedKernel3D:
    def __init__(
        self,
        kernel,
        offset_names,
    ):
        self.kernel = kernel
        self.offset_names = offset_names

    def __getitem__(self, grid):
        if len(grid) != 3:
            raise ValueError(
                "expected a three-dimensional grid"
            )

        grid_x, grid_y, grid_z = map(
            int,
            grid,
        )

        def launch(*args, **kwargs):
            for offset_x in range(
                0,
                grid_x,
                MAX_GRID_X,
            ):
                count_x = min(
                    MAX_GRID_X,
                    grid_x - offset_x,
                )

                for offset_y in range(
                    0,
                    grid_y,
                    MAX_GRID_Y,
                ):
                    count_y = min(
                        MAX_GRID_Y,
                        grid_y - offset_y,
                    )

                    for offset_z in range(
                        0,
                        grid_z,
                        MAX_GRID_Z,
                    ):
                        count_z = min(
                            MAX_GRID_Z,
                            grid_z - offset_z,
                        )

                        call_kwargs = dict(kwargs)
                        call_kwargs[
                            self.offset_names[0]
                        ] = offset_x
                        call_kwargs[
                            self.offset_names[1]
                        ] = offset_y
                        call_kwargs[
                            self.offset_names[2]
                        ] = offset_z

                        self.kernel[
                            (
                                count_x,
                                count_y,
                                count_z,
                            )
                        ](
                            *args,
                            **call_kwargs,
                        )

        return launch


parallel_nsa_bwd_kernel_dq_enflame = (
    _ChunkedKernel3D(
        _parallel_nsa_bwd_kernel_dq_enflame,
        (
            "token_offset",
            "value_offset",
            "bh_offset",
        ),
    )
)

parallel_nsa_bwd_kernel_dkv_enflame = (
    _ChunkedKernel3D(
        _parallel_nsa_bwd_kernel_dkv_enflame,
        (
            "value_offset",
            "block_offset",
            "bh_offset",
        ),
    )
)


# Selected block mask

MAX_GRID_X = 65535
MAX_GRID_Y = 255
MAX_GRID_Z = 255


@triton.heuristics(
    {
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.jit(
    do_not_specialize=[
        "T",
        "token_offset",
        "batch_offset",
        "hs_offset",
    ]
)
def _parallel_nsa_kernel_mask_enflame(
    block_indices,
    block_counts,
    block_mask,
    token_offset,
    batch_offset,
    hs_offset,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t = tl.program_id(0) + token_offset
    i_b = tl.program_id(1) + batch_offset
    i_hs = tl.program_id(2) + hs_offset
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(
            block_counts + i_b * T * H + i_t * H + i_h
        )
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(
            block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i,
            b_m.to(block_mask.dtype.element_ty),
        )


def parallel_nsa_block_mask_enflame(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor | None,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size

    if cu_seqlens is not None:
        NS = triton.cdiv(
            prepare_lens_enflame(
                cu_seqlens
            ).max().item(),
            BS,
        )
    else:
        NS = triton.cdiv(T, BS)

    block_mask = torch.zeros(
        B,
        T,
        H,
        NS,
        dtype=torch.bool,
        device=block_indices.device,
    )

    for token_offset in range(
        0,
        T,
        MAX_GRID_X,
    ):
        token_count = min(
            MAX_GRID_X,
            T - token_offset,
        )

        for batch_offset in range(
            0,
            B,
            MAX_GRID_Y,
        ):
            batch_count = min(
                MAX_GRID_Y,
                B - batch_offset,
            )

            for hs_offset in range(
                0,
                H * S,
                MAX_GRID_Z,
            ):
                hs_count = min(
                    MAX_GRID_Z,
                    H * S - hs_offset,
                )

                grid = (
                    token_count,
                    batch_count,
                    hs_count,
                )

                _parallel_nsa_kernel_mask_enflame[
                    grid
                ](
                    block_indices=block_indices,
                    block_counts=block_counts,
                    block_mask=block_mask,
                    token_offset=token_offset,
                    batch_offset=batch_offset,
                    hs_offset=hs_offset,
                    T=T,
                    H=H,
                    S=S,
                    BS=BS,
                    NS=NS,
                )

    return block_mask


# Selected block Top-K

MAX_GRID_X = 65535


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["token_offset"])
def _parallel_nsa_kernel_topk_enflame(
    q,
    k,
    lse,
    scale,
    block_indices,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    token_offset,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t = tl.program_id(0) + token_offset
    i_bh = tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS
    ################################
    # 1. lse computation
    ################################
    if lse is not None:
        b_lse = tl.load(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G))
    else:
        # max scores for the current block
        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        # lse = log(acc) + m
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)

            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
            )
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BC]
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([G], dtype=tl.float32)
        else:
            b_lse = b_m + log(b_acc)

    ################################
    # 2. topk selection
    ################################
    # S60 LLVM cannot compile the reshape/broadcast
    # layouts used by the generic bitonic sort.
    # Maintain the global Top-S using scalar reductions.
    o_top_slot = tl.arange(0, S)
    b_top_scores = tl.full(
        [S],
        float("-inf"),
        dtype=tl.float32,
    )
    b_top = tl.full(
        [S],
        -1,
        dtype=tl.int32,
    )

    IC = i_t // BS
    for i_c in range(
        0,
        tl.cdiv(i_t + 1, BS),
        BC,
    ):
        o_c = i_c + tl.arange(0, BC)

        p_k = tl.make_block_ptr(
            k + (boc * H + i_h) * K,
            (K, TC),
            (1, H * K),
            (0, i_c),
            (BK, BC),
            (0, 1),
        )
        b_k = tl.load(
            p_k,
            boundary_check=(0, 1),
        )

        b_s = tl.dot(b_q, b_k)
        b_s = tl.where(
            o_c < IC,
            b_s,
            float("-inf"),
        )

        b_p = tl.where(
            (o_c == 0)
            | (o_c == IC - 1)
            | (o_c == IC),
            1.0,
            exp(b_s - b_lse[:, None]),
        )

        b_candidates = tl.sum(b_p, 0)
        o_candidates = tl.where(
            o_c <= IC,
            o_c,
            -1,
        )

        for _ in tl.static_range(0, S):
            b_best = tl.max(
                b_candidates,
                0,
            )

            m_best = (
                (b_candidates == b_best)
                & (o_candidates >= 0)
            )
            o_best = tl.min(
                tl.where(
                    m_best,
                    o_candidates,
                    2147483647,
                ),
                0,
            )
            best_valid = (
                o_best != 2147483647
            )

            b_candidates = tl.where(
                o_candidates == o_best,
                float("-inf"),
                b_candidates,
            )

            b_worst = tl.min(
                b_top_scores,
                0,
            )
            o_worst_slot = tl.max(
                tl.where(
                    b_top_scores == b_worst,
                    o_top_slot,
                    -1,
                ),
                0,
            )

            replace = (
                best_valid
                & (b_best > b_worst)
            )
            m_replace = (
                (o_top_slot == o_worst_slot)
                & replace
            )

            b_top_scores = tl.where(
                m_replace,
                b_best,
                b_top_scores,
            )
            b_top = tl.where(
                m_replace,
                o_best,
                b_top,
            )

    p_b = tl.make_block_ptr(
        block_indices + (bos + i_t) * H * S, (H * S,), (1,), (i_h * S,), (S,), (0,)
    )
    tl.store(p_b, b_top.to(p_b.dtype.element_ty))


# ===========================================================================
# Forward kernel
# ===========================================================================


def parallel_nsa_topk_enflame(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.LongTensor:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    # the number of selected blocks for each token
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S)
    # here we set BC = BS, but beware that they can be chosen separately if required
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    assert BC >= 2 * S, f"BC ({BC}) must be greater than or equal to 2 * S ({S})"

    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    token_indices = (
        prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    )
    chunk_offsets = (
        prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None
    )
    # the 1st and the last 2 blocks are always selected
    for token_offset in range(0, T, MAX_GRID_X):
        grid = (
            min(
                MAX_GRID_X,
                T - token_offset,
            ),
            B * H,
        )
        _parallel_nsa_kernel_topk_enflame[grid](
            q=q,
            k=k,
            lse=lse,
            scale=scale,
            block_indices=block_indices,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            token_offset=token_offset,
            T=T,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            S=S,
            BC=BC,
            BS=BS,
            BK=BK,
        )
    return block_indices


# S60 backend dispatch

def _to_gcu_index_dtype(
    value: torch.Tensor | None,
) -> torch.Tensor | None:
    """Convert unsupported int64 index tensors to int32."""
    if value is None:
        return None

    if value.dtype == torch.int64:
        return value.to(dtype=torch.int32)

    return value


def _install_enflame_nsa_compat():
    """Install S60-specific NSA helpers on the active call path."""
    from flag_attn.runtime.backend._enflame.FLA.nsa.parallel_nsa_compression import (
        install_enflame_compression_compat,
    )

    install_enflame_compression_compat()





    import importlib














    from .mean_pooling import (
        mean_pooling as mean_pooling_enflame,
    )

    from .bwd_preprocess import (
        parallel_attn_bwd_preprocess_enflame,
    )
    from ..index import (
        prepare_chunk_indices_enflame,
        prepare_chunk_offsets_enflame,
        prepare_lens_enflame,
        prepare_token_indices_enflame,
    )

    main_module = importlib.import_module(
        "flag_attn.parallel_nsa.parallel_nsa"
    )
    compression_module = importlib.import_module(
        "flag_attn.parallel_nsa."
        "parallel_nsa_compression"
    )

    main_module.parallel_nsa_topk = (
        parallel_nsa_topk_enflame
    )
    main_module.parallel_nsa_block_mask = (
        parallel_nsa_block_mask_enflame
    )
    main_module.parallel_nsa_bwd_kernel_dq = (
        parallel_nsa_bwd_kernel_dq_enflame
    )
    main_module.parallel_nsa_bwd_kernel_dkv = (
        parallel_nsa_bwd_kernel_dkv_enflame
    )
    main_module.mean_pooling = mean_pooling_enflame

    main_module.parallel_nsa_fwd = (
        parallel_nsa_fwd_enflame
    )

    main_module.parallel_attn_bwd_preprocess = (
        parallel_attn_bwd_preprocess_enflame
    )
    compression_module.parallel_attn_bwd_preprocess = (
        parallel_attn_bwd_preprocess_enflame
    )

    index_bindings = {
        "prepare_lens": prepare_lens_enflame,
        "prepare_token_indices": (
            prepare_token_indices_enflame
        ),
        "prepare_chunk_offsets": (
            prepare_chunk_offsets_enflame
        ),
        "prepare_chunk_indices": (
            prepare_chunk_indices_enflame
        ),
    }

    for name, function in index_bindings.items():
        if hasattr(main_module, name):
            setattr(main_module, name, function)

        if hasattr(compression_module, name):
            setattr(
                compression_module,
                name,
                function,
            )


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Run generic NSA with GCU-compatible index dtypes."""
    _install_enflame_nsa_compat()

    block_indices = _to_gcu_index_dtype(
        block_indices
    )
    cu_seqlens = _to_gcu_index_dtype(
        cu_seqlens
    )

    if isinstance(block_counts, torch.Tensor):
        block_counts = _to_gcu_index_dtype(
            block_counts
        )

    return _generic_parallel_nsa(
        q=q,
        k=k,
        v=v,
        g_cmp=g_cmp,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["parallel_nsa"]
