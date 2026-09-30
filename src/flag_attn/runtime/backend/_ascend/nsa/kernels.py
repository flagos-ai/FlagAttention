# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Ascend NSA kernels migrated from FlagGems-vllm commit a96e719b."""

import inspect

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as cann
import triton.experimental.tle.language as tle

autotune_cache_kwargs = (
    {"cache_results": True}
    if "cache_results" in inspect.signature(triton.autotune).parameters
    else {}
)


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
@triton.jit
def parallel_nsa_fwd_kernel_tle(
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
    T,
    t_offset,
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
    PACKED_KV: tl.constexpr,
    FAST_VALID_BLOCKS: tl.constexpr,
    FULL_TILES: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0) + t_offset, tl.program_id(1), tl.program_id(2)
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

    if PACKED_KV:
        k += (i_b * H + i_h) * T * K
        v += (i_b * H + i_h) * T * V
    else:
        k += (bos * H + i_h) * K
        v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = block_counts

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

    # Online softmax carries state between selected blocks; keep this serial.
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if FAST_VALID_BLOCKS or (i_s <= i_t and i_s >= 0):
            if PACKED_KV:
                k_ptrs = k_base + (i_s + offs_s[:, None]) * K + offs_k[None, :]
                if FULL_TILES:
                    b_k = tle.load(k_ptrs, is_async=True).trans()
                else:
                    k_mask = ((i_s + offs_s[:, None]) < T) & (offs_k[None, :] < K)
                    b_k = tle.load(k_ptrs, mask=k_mask, other=0.0, is_async=True).trans()
            else:
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
            b_r = tl.exp(b_mp - b_m)
            # [G, BS] — b_s registers reused for softmax output b_p
            b_p = tl.exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            # Async V load AFTER score computation
            if PACKED_KV:
                v_ptrs = (
                    v_base
                    + (i_s + offs_s[:, None]) * V
                    + (v_start + offs_v[None, :])
                )
                if FULL_TILES:
                    b_v = tle.load(v_ptrs, is_async=True)
                else:
                    v_mask = ((i_s + offs_s[:, None]) < T) & (
                        (v_start + offs_v[None, :]) < V
                    )
                    b_v = tle.load(v_ptrs, mask=v_mask, other=0.0, is_async=True)
            else:
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
    b_o = b_o / tl.where(b_acc > 0, b_acc, 1.0)[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))




@triton.jit
def parallel_nsa_fwd_kernel_tle_grouped(
    q, k, v, o, lse, scale, block_indices, T,
    H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, S: tl.constexpr,
    BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    TOKEN_GROUP: tl.constexpr,
):
    pid_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    offs_k = tl.arange(0, BK)
    offs_s = tl.arange(0, BS)
    offs_v = tl.arange(0, BV)
    k_head = k + (i_b * H + i_h) * T * K
    v_head = v + (i_b * H + i_h) * T * V

    # Query tokens have independent accumulators and output locations.
    for token_offset in cann.parallel(0, TOKEN_GROUP, bind_sub_block=True):
        i_t = pid_t * TOKEN_GROUP + token_offset
        if i_t < T:
            q_ptr = tl.make_block_ptr(
                q + (i_b * T + i_t) * HQ * K,
                (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
            )
            b_q = tl.load(q_ptr, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)
            o_ptr = tl.make_block_ptr(
                o + (i_b * T + i_t) * HQ * V,
                (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
            )
            lse_ptr = lse + (i_b * T + i_t) * HQ + i_h * G + tl.arange(0, G)
            indices = block_indices + (i_b * T + i_t) * H * S + i_h * S
            b_o = tl.zeros([G, BV], dtype=tl.float32)
            b_m = tl.full([G], float("-inf"), dtype=tl.float32)
            b_acc = tl.zeros([G], dtype=tl.float32)

            for i in range(S):
                i_s = tl.load(indices + i).to(tl.int32) * BS
                if i_s <= i_t and i_s >= 0:
                    k_ptrs = k_head + (i_s + offs_s[:, None]) * K + offs_k[None, :]
                    b_k = tle.load(k_ptrs, is_async=True).trans()
                    b_s = tl.dot(b_q, b_k)
                    b_s = tl.where(
                        (i_t >= (i_s + offs_s))[None, :], b_s, float("-inf")
                    )
                    b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                    b_r = tl.exp(b_mp - b_m)
                    b_p = tl.exp(b_s - b_m[:, None])
                    b_acc = b_acc * b_r + tl.sum(b_p, 1)
                    v_ptrs = v_head + (i_s + offs_s[:, None]) * V + i_v * BV + offs_v[None, :]
                    b_v = tle.load(v_ptrs, is_async=True)
                    b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
                    b_mp = b_m

            b_o = b_o / tl.where(b_acc > 0, b_acc, 1.0)[:, None]
            b_m += tl.log(b_acc)
            tl.store(o_ptr, b_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
            tl.store(lse_ptr, b_m.to(lse_ptr.dtype.element_ty))
