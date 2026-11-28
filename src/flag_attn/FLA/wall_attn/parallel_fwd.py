# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable cache preparation and causal attention on pre-rescaled Q/K."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_attn import runtime
from flag_attn.FLA.cumsum import chunk_global_cumsum

RCP_LN2 = 1.4426950216


def allocate_descriptor(size: int, alignment: int, stream: int | None):
    """Compatibility allocator on the active backend; no descriptors are needed."""
    del alignment, stream
    return torch.empty(size, dtype=torch.uint8, device=runtime.device.name)


@triton.jit
def parallel_wall_attn_fwd_kernel(
    q_cache, k_cache, v, output, lse, scale,
    T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, V: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950216
    tile, bh, value_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh // HQ, bh % HQ
    kv_head = head // (HQ // H)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, BK)
    values = value_tile * BV + tl.arange(0, BV)
    query = tl.load(q_cache + (bh * T + rows[:, None]) * D + dims[None, :],
                    (rows[:, None] < T) & (dims[None, :] < D), other=0)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, BV), tl.float32)
    for start in range(0, tl.minimum((tile + 1) * BM, T), BN):
        cols = start + tl.arange(0, BN)
        key = tl.load(k_cache + (bh * T + cols[None, :]) * D + dims[:, None],
                      (cols[None, :] < T) & (dims[:, None] < D), other=0)
        value = tl.load(v + ((batch * T + cols[:, None]) * H + kv_head) * V + values[None, :],
                        (cols[:, None] < T) & (values[None, :] < V), other=0)
        score = tl.dot(query, key, input_precision="ieee") * scale * RCP_LN2
        score = tl.where((rows[:, None] >= cols[None, :]) & (cols[None, :] < T), score, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(score, 1))
        correction = tl.exp2(maximum - next_maximum)
        probability = tl.exp2(score - next_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        accumulator = accumulator * correction[:, None] + tl.dot(probability.to(value.dtype), value, input_precision="ieee")
        maximum = next_maximum
    tl.store(output + ((batch * T + rows[:, None]) * HQ + head) * V + values[None, :],
             accumulator / denominator[:, None], (rows[:, None] < T) & (values[None, :] < V))
    if value_tile == 0:
        tl.store(lse + (batch * T + rows) * HQ + head,
                 maximum + tl.log2(denominator), rows < T)


def parallel_wall_attn_fwd(q_cache, k_cache, v, output, lse, capacity: int, scale=None):
    """Fill output/LSE from [B, HQ, T, D] caches using standard Triton.

    ``capacity`` is retained for compatibility with pipelined implementations.
    Callers must check the unsafe flag from ``prepare_qk_cache`` first.
    """
    del capacity
    b, hq, t, d = q_cache.shape
    value_dim = v.shape[-1]
    return parallel_wall_attn_fwd_kernel[(triton.cdiv(t, 32), b * hq, triton.cdiv(value_dim, 64))](
        q_cache, k_cache, v, output, lse, d**-0.5 if scale is None else scale,
        T=t, HQ=hq, H=v.shape[2], D=d, V=value_dim,
        BM=32, BN=32, BK=max(16, triton.next_power_of_2(d)), BV=64,
        num_warps=4, num_stages=1,
    )


@triton.jit
def parallel_wall_attn_fwd_kernel_preprocess(
    q,
    k,
    g,
    q_cache,
    k_cache,
    anchor,
    unsafe,
    T_VALUE: tl.constexpr,
    HQ_VALUE: tl.constexpr,
    H_VALUE: tl.constexpr,
    G_VALUE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < HQ_VALUE * D_VALUE

    total = tl.zeros((1, BLOCK_C), tl.float32)
    first = tl.zeros((1, BLOCK_C), tl.float32)
    for tile in range(tl.cdiv(T_VALUE, BLOCK_T)):
        tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        qg_offsets = batch * T_VALUE * HQ_VALUE * D_VALUE + tokens[:, None] * HQ_VALUE * D_VALUE + channels[None, :]
        mask = (tokens[:, None] < T_VALUE) & channel_mask[None, :]
        values = tl.load(g + qg_offsets, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(values, axis=0)[None, :]
        if tile == 0:
            first += tl.sum(tl.where(tokens[:, None] == 0, values, 0.0), axis=0)[None, :]
    reference = 0.5 * (first + total) * SCALE
    tl.store(
        anchor + batch * HQ_VALUE * D_VALUE + channels[None, :],
        reference,
        mask=channel_mask[None, :],
    )

    carry = tl.zeros((1, BLOCK_C), tl.float32)
    invalid = tl.full((), False, tl.int1)
    query_heads = channels // D_VALUE
    dims = channels - query_heads * D_VALUE
    kv_heads = query_heads // G_VALUE
    for tile in range(tl.cdiv(T_VALUE, BLOCK_T)):
        tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        qg_offsets = batch * T_VALUE * HQ_VALUE * D_VALUE + tokens[:, None] * HQ_VALUE * D_VALUE + channels[None, :]
        k_offsets = (
            batch * T_VALUE * H_VALUE * D_VALUE
            + tokens[:, None] * H_VALUE * D_VALUE
            + kv_heads[None, :] * D_VALUE
            + dims[None, :]
        )
        mask = (tokens[:, None] < T_VALUE) & channel_mask[None, :]
        values = tl.load(g + qg_offsets, mask=mask, other=0.0).to(tl.float32)
        raw_prefix = tl.cumsum(values, axis=0) + carry
        prefix = raw_prefix * SCALE
        q_value = tl.load(q + qg_offsets, mask=mask, other=0.0).to(tl.float32)
        k_value = tl.load(k + k_offsets, mask=mask, other=0.0).to(tl.float32)
        q_operand = q_value * tl.exp2(prefix - reference)
        k_operand = k_value * tl.exp2(reference - prefix)
        q_stored = q_operand.to(q_cache.dtype.element_ty)
        k_stored = k_operand.to(k_cache.dtype.element_ty)
        # Global-gauge operands use BF16. Bound the FP32 exp2 argument and
        # reject overflow or nonzero values lost to cache underflow. Substitutes
        # and exponent clipping would change the accepted attention operation.
        valid = (
            (values <= 0.0)
            & (tl.abs(prefix - reference) <= 125.0)
            & (tl.abs(q_stored.to(tl.float32)) < float("inf"))
            & (tl.abs(k_stored.to(tl.float32)) < float("inf"))
            & ((q_value == 0.0) | (q_stored != 0.0))
            & ((k_value == 0.0) | (k_stored != 0.0))
        )
        invalid |= tl.sum(tl.sum((mask & ~valid).to(tl.int32), axis=0), axis=0) > 0
        cache_offsets = (
            (batch * HQ_VALUE + query_heads[None, :]) * T_VALUE * D_VALUE + tokens[:, None] * D_VALUE + dims[None, :]
        )
        tl.store(q_cache + cache_offsets, q_stored, mask=mask)
        tl.store(k_cache + cache_offsets, k_stored, mask=mask)
        carry += tl.sum(values, axis=0)[None, :]
    if invalid:
        tl.atomic_or(unsafe, 1, sem="relaxed")


def _prepare_fused_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe, bt=128, bc=8, warps=4):
    b, t, hq, d = q.shape
    h = k.shape[2]
    return parallel_wall_attn_fwd_kernel_preprocess[(b, triton.cdiv(hq * d, bc))](
        q,
        k,
        g,
        q_cache,
        k_cache,
        anchor,
        unsafe,
        T_VALUE=t,
        HQ_VALUE=hq,
        H_VALUE=h,
        G_VALUE=hq // h,
        D_VALUE=d,
        BLOCK_T=bt,
        BLOCK_C=bc,
        SCALE=RCP_LN2,
        num_warps=warps,
        num_stages=1,
    )


@triton.jit
def _wall_attn_anchor_kernel(prefix, anchor, T: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    bhq = tl.program_id(0)
    block_d = tl.program_id(1)
    batch = bhq // HQ
    head = bhq % HQ
    dims = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dims < D
    first = tl.load(
        prefix + ((batch * T) * HQ + head) * D + dims,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    last = tl.load(
        prefix + ((batch * T + T - 1) * HQ + head) * D + dims,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(anchor + bhq * D + dims, 0.5 * (first + last), mask=mask)


@triton.jit
def _wall_attn_cache_kernel(
    q,
    k,
    g,
    prefix,
    anchor,
    q_cache,
    k_cache,
    unsafe,
    T: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_t = tl.program_id(0)
    block_d = tl.program_id(1)
    bhq = tl.program_id(2)
    batch = bhq // HQ
    query_head = bhq % HQ
    kv_head = query_head // G

    tokens = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (tokens[:, None] < T) & (dims[None, :] < D)
    p = tl.load(
        prefix + ((batch * T + tokens[:, None]) * HQ + query_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    reference = tl.load(
        anchor + bhq * D + dims,
        mask=dims < D,
        other=0.0,
    ).to(tl.float32)
    q_value = tl.load(
        q + ((batch * T + tokens[:, None]) * HQ + query_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k_value = tl.load(
        k + ((batch * T + tokens[:, None]) * H + kv_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        g + ((batch * T + tokens[:, None]) * HQ + query_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q_operand = q_value * tl.exp2(p - reference[None, :])
    k_operand = k_value * tl.exp2(reference[None, :] - p)
    q_stored = q_operand.to(q_cache.dtype.element_ty)
    k_stored = k_operand.to(k_cache.dtype.element_ty)
    valid = (
        (gate <= 0.0)
        & (tl.abs(p - reference[None, :]) <= 125.0)
        & (tl.abs(q_stored.to(tl.float32)) < float("inf"))
        & (tl.abs(k_stored.to(tl.float32)) < float("inf"))
        & ((q_value == 0.0) | (q_stored != 0.0))
        & ((k_value == 0.0) | (k_stored != 0.0))
    )
    if tl.sum(tl.sum((mask & ~valid).to(tl.int32), axis=0), axis=0) > 0:
        tl.atomic_or(unsafe, 1, sem="relaxed")
    output_offset = ((bhq * T + tokens[:, None]) * D) + dims[None, :]
    tl.store(q_cache + output_offset, q_stored, mask=mask)
    tl.store(k_cache + output_offset, k_stored, mask=mask)


def _build_qk_cache(q, k, g, prefix, q_cache, k_cache, anchor, unsafe):
    b, t, hq, d = q.shape
    h = k.shape[2]
    _wall_attn_anchor_kernel[(b * hq, triton.cdiv(d, 32))](
        prefix,
        anchor,
        T=t,
        HQ=hq,
        D=d,
        BLOCK_D=32,
        num_warps=1,
        num_stages=1,
    )
    _wall_attn_cache_kernel[(triton.cdiv(t, 64), triton.cdiv(d, 32), b * hq)](
        q,
        k,
        g,
        prefix,
        anchor,
        q_cache,
        k_cache,
        unsafe,
        T=t,
        H=h,
        HQ=hq,
        G=hq // h,
        D=d,
        BLOCK_T=64,
        BLOCK_D=32,
        num_warps=4,
        num_stages=2,
    )


def prepare_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe):
    """Construct gated operands and set the device flag for unsafe data."""
    if q.shape[1] <= 2048:
        _prepare_fused_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe, bt=128, bc=8, warps=4)
    else:
        prefix = chunk_global_cumsum(g, scale=RCP_LN2)
        _build_qk_cache(q, k, g, prefix, q_cache, k_cache, anchor, unsafe)


__all__ = ["allocate_descriptor", "parallel_wall_attn_fwd", "prepare_qk_cache"]
