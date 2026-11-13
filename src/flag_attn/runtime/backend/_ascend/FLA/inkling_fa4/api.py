# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend implementation of Inkling FA4 paged relative attention.

Semantics follow ``flag_attn.testing.inkling_fa4.ref_rel_attn``: the query rows
of one sequence are the last ``q_len`` positions of its cached key sequence, the
relative logits are added only for ``0 <= q_pos - k_pos < rel_extent``, and the
causal / sliding-window mask is applied on the shifted positions.

Ascend layout choices:
* the ``group_size`` query heads that share one KV head are packed into the tile
  rows together with a token block, so the paged KV tile is loaded once per
  group instead of once per query head;
* the KV range is clipped by the causal bound and the sliding window (past and
  future keys are never fetched);
* the range is split across ``NUM_SPLITS`` programs and merged by a small fp32
  combine kernel, because decode shapes otherwise have only a handful of
  programs.  ``num_splits`` from the caller wins when it is greater than one.
"""

import os

import torch
import triton

from .kernels import _combine_splits_kernel, _rel_attn_paged_kernel

_BLOCK_M = 64
_BLOCK_N = 64
_COMBINE_ROWS = 16
_MAX_HEAD_DIM = 256
_MAX_SPLITS = 32
#: Programs we want in flight (roughly three waves of the 40 AI cores).
_TARGET_PROGRAMS = 32
#: ``INKLING_FA4_SPLITS`` overrides the automatic split choice for A/B runs.
_SPLITS_ENV = "INKLING_FA4_SPLITS"

#: Split-KV scratch buffers, kept alive between calls.  Allocating them per call
#: lets the caching allocator hand the block to the next allocation while the
#: combine kernel is still reading it: under ``torch.inference_mode()`` the
#: buffers have no autograd reference, so that reuse really happens and the
#: result is silently corrupted.  Keying by shape also removes the per-call
#: allocation from the decode latency.
_PARTIAL_CACHE: dict = {}
_PARTIAL_CACHE_MAX = 16


def _partial_buffers(splits, total_q, num_heads, head_dim_v, device):
    key = (str(device), splits, total_q, num_heads, head_dim_v)
    buffers = _PARTIAL_CACHE.get(key)
    if buffers is None:
        if len(_PARTIAL_CACHE) >= _PARTIAL_CACHE_MAX:
            # Drop the oldest entry; the caller-side kernels are long finished by
            # the time a new shape shows up in practice.
            _PARTIAL_CACHE.pop(next(iter(_PARTIAL_CACHE)))
        buffers = (
            torch.empty((splits, total_q, num_heads, head_dim_v), dtype=torch.float32, device=device),
            torch.empty((splits, total_q, num_heads), dtype=torch.float32, device=device),
        )
        _PARTIAL_CACHE[key] = buffers
    return buffers


def _choose_splits(num_splits, base_programs, kv_upper, block_n):
    max_splits = max(1, min(_MAX_SPLITS, triton.cdiv(max(kv_upper, 1), block_n)))
    override = os.environ.get(_SPLITS_ENV)
    if override:
        return max(1, min(int(override), max_splits))
    if num_splits > 1:
        return max(1, min(num_splits, max_splits))
    want = max(1, _TARGET_PROGRAMS // max(1, base_programs))
    splits = 1
    while splits < want and splits < max_splits:
        splits *= 2
    return splits


def _check(q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel_logits, rel_extent, out):
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
    # ``backend`` (``triton``/``tle``) selects the upstream CUDA variants; on
    # Ascend this module is the only implementation, so it is accepted and
    # ignored to keep the public signature identical.
    del backend
    _check(q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel_logits, rel_extent, out)

    batch_size = cache_seqlens.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim_q = q.shape[2]
    head_dim_v = value_cache.shape[3]
    block_size = key_cache.shape[1]
    total_q = q.shape[0]

    if out is None:
        out = torch.empty((total_q, num_heads, head_dim_v), dtype=q.dtype, device=q.device)
    if batch_size == 0 or total_q == 0:
        return out.zero_()
    if max_seqlen_q <= 0:
        raise ValueError("max_seqlen_q must be positive for non-empty q")

    rel_logits = rel_logits.contiguous()
    # Packing the query heads that share a KV head trades a per-row head index
    # (strided Q / rel-bias row loads) for fewer KV passes.  It only pays off
    # when a token block is full: for decode (q_len of a few tokens) the packed
    # tile would be mostly idle, so fall back to one query head per program.
    group_size = num_heads // num_kv_heads
    pack_gqa = max_seqlen_q >= max(1, _BLOCK_M // group_size)
    group = group_size if pack_gqa else 1
    h_grid = num_kv_heads if pack_gqa else num_heads
    tokens_per_block = max(1, _BLOCK_M // group)
    num_q_blocks = triton.cdiv(max_seqlen_q, tokens_per_block)
    base_programs = num_q_blocks * batch_size * h_grid
    kv_upper = block_table.shape[1] * block_size
    splits = _choose_splits(num_splits, base_programs, kv_upper, _BLOCK_N)
    split_kv = splits > 1

    window_left, window_right = window_size
    has_left = window_left is not None and window_left >= 0
    has_right = window_right is not None and window_right >= 0

    if split_kv:
        partial_out, partial_lse = _partial_buffers(
            splits, total_q, num_heads, head_dim_v, q.device
        )
        po_strides = partial_out.stride()
        pl_strides = partial_lse.stride()
    else:
        partial_out = out
        partial_lse = out
        po_strides = (0, 0, 0, 0)
        pl_strides = (0, 0, 0)

    block_d = max(16, triton.next_power_of_2(max(head_dim_q, head_dim_v)))
    grid = (num_q_blocks, batch_size * h_grid, splits)
    _rel_attn_paged_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel_logits,
        out,
        partial_out,
        partial_lse,
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
        po_strides[0],
        po_strides[1],
        po_strides[2],
        pl_strides[0],
        pl_strides[1],
        pl_strides[2],
        block_table.stride(0),
        block_table.stride(1),
        float(softmax_scale),
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        H_GRID=h_grid,
        GROUP=group,
        KV_GROUP=group_size,
        PACK_GQA=pack_gqa,
        HEAD_DIM=head_dim_q,
        BLOCK_M=_BLOCK_M,
        TOKENS_PER_BLOCK=tokens_per_block,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        REL_EXTENT=rel_extent,
        BLOCK_SIZE=block_size,
        CAUSAL=bool(causal),
        HAS_LEFT=has_left,
        WINDOW_LEFT=window_left if has_left else 0,
        HAS_RIGHT=has_right,
        WINDOW_RIGHT=window_right if has_right else 0,
        NUM_SPLITS=splits,
        SPLIT_KV=split_kv,
        num_warps=4,
    )

    if split_kv:
        total_rows = total_q * num_heads
        combine_grid = (triton.cdiv(total_rows, _COMBINE_ROWS),)
        _combine_splits_kernel[combine_grid](
            partial_out,
            partial_lse,
            out,
            po_strides[0],
            po_strides[1],
            po_strides[2],
            pl_strides[0],
            pl_strides[1],
            pl_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            total_rows,
            NUM_HEADS=num_heads,
            NUM_SPLITS=splits,
            HEAD_DIM=head_dim_v,
            BLOCK_D=block_d,
            ROWS=_COMBINE_ROWS,
            num_warps=4,
        )
    return out
