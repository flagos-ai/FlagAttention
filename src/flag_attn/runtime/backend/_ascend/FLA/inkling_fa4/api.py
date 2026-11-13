# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend implementation of Inkling FA4 paged relative attention.

Semantics follow ``flag_attn.testing.inkling_fa4.ref_rel_attn``: the query rows
of one sequence are the last ``q_len`` positions of its cached key sequence, the
relative logits are added only for ``0 <= q_pos - k_pos < rel_extent``, and the
causal / sliding-window mask is applied on the shifted positions.

This first version is correctness-first: one query row block per (sequence, query
head), no GQA packing and no split-KV (``num_splits`` is accepted but the full KV
range is walked, which is numerically equivalent up to fp32 rounding).
"""

import torch
import triton

from .kernels import _rel_attn_paged_kernel

_BLOCK_M = 64
_BLOCK_N = 64
_MAX_HEAD_DIM = 256


def _check(
    q,
    key_cache,
    value_cache,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    rel_logits,
    rel_extent,
    out,
):
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "key_cache and value_cache must have shape "
            "[num_blocks, block_size, num_kv_heads, head_dim]"
        )
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_num_blocks]")
    if cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
        raise ValueError("cache_seqlens and cu_seqlens_q must be 1-D")
    devices = {
        t.device
        for t in (q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel_logits)
    }
    if len(devices) != 1:
        raise ValueError("all attention inputs must be on the same device")
    if q.shape[1] % key_cache.shape[2] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if rel_logits.shape != (q.shape[0], q.shape[1], rel_extent):
        raise ValueError("rel_logits must have shape [total_q, num_heads, rel_extent]")
    if key_cache.shape[:3] != value_cache.shape[:3]:
        raise ValueError("key_cache and value_cache block/token/head shapes must match")
    if key_cache.shape[3] > _MAX_HEAD_DIM or value_cache.shape[3] > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim must be <= {_MAX_HEAD_DIM} on Ascend")
    if block_table.shape[0] != cache_seqlens.shape[0]:
        raise ValueError("block_table and cache_seqlens batch sizes differ")
    if cu_seqlens_q.shape[0] != cache_seqlens.shape[0] + 1:
        raise ValueError("cu_seqlens_q must contain batch_size + 1 entries")
    if rel_logits.dtype != q.dtype:
        raise TypeError("rel_logits and q must have the same dtype")
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("out must match q in shape and dtype")


def inkling_fa4_rel_attention(
    q,
    key_cache,
    value_cache,
    *,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale,
    causal,
    window_size,
    rel_extent,
    rel_logits,
    num_splits=1,
    out=None,
    backend=None,
):
    # ``backend`` (``triton``/``tle``) selects between the upstream CUDA
    # variants; on Ascend this module is the only implementation, so the
    # argument is accepted and ignored to keep the public signature identical.
    del backend
    _check(q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel_logits, rel_extent, out)

    batch_size = cache_seqlens.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim_q = q.shape[2]
    head_dim_v = value_cache.shape[3]
    block_size = key_cache.shape[1]

    if out is None:
        out = torch.empty(
            (q.shape[0], num_heads, head_dim_v), dtype=q.dtype, device=q.device
        )
    if batch_size == 0 or q.shape[0] == 0:
        return out.zero_()
    if max_seqlen_q <= 0:
        raise ValueError("max_seqlen_q must be positive for non-empty q")

    rel_logits = rel_logits.contiguous()
    window_left, window_right = window_size
    has_left = window_left is not None and window_left >= 0
    has_right = window_right is not None and window_right >= 0

    block_d = max(16, triton.next_power_of_2(max(head_dim_q, head_dim_v)))
    grid = (triton.cdiv(max_seqlen_q, _BLOCK_M), batch_size * num_heads)
    _rel_attn_paged_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        out,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        float(softmax_scale),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim_q,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        REL_EXTENT=rel_extent,
        BLOCK_SIZE=block_size,
        CAUSAL=bool(causal),
        HAS_LEFT=has_left,
        WINDOW_LEFT=window_left if has_left else 0,
        HAS_RIGHT=has_right,
        WINDOW_RIGHT=window_right if has_right else 0,
        num_warps=4,
    )
    return out
