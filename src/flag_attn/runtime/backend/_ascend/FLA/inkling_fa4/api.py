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

#: Ascend tile shapes.  Two shapes are worth keeping: a 16x128 tile for short-query
#: (decode-like) shapes, where most rows of a 64-row tile are masked and the wider
#: KV tile halves the number of loop iterations (decode_8k 827 -> 586 us), and a
#: 64x64 tile for longer queries, where a 16-row tile would make every program walk
#: the whole KV window (long_prefill 454 -> 739 us).  Wider than 128 keys does not
#: fit the 192 KiB UB together with the fp32 score tile.
_BLOCK_M = 64
_BLOCK_N = 64
_SMALL_Q_BLOCK_M = 16
_SMALL_Q_BLOCK_N = 128
#: Query length up to which the short-query tile is used.
_SMALL_Q = 64
_NUM_WARPS = 4
#: ``INKLING_FA4_SPLITS`` pins the split-KV count for A/B runs; resolved once because
#: reading the environment on every call cost a measurable slice of the budget.
_SPLITS_OVERRIDE = os.environ.get("INKLING_FA4_SPLITS")
_COMBINE_ROWS = 16
_MAX_HEAD_DIM = 256
_MAX_SPLITS = 32
#: Programs we want in flight (roughly three waves of the 40 AI cores).
_TARGET_PROGRAMS = 128
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
    key = (device.index, splits, total_q, num_heads, head_dim_v)
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


#: Per-call launch decisions keyed by the input geometry.  The benchmark cases are
#: ~200-900 us per call, of which ~50-60 us was Python: caching the plan keeps the
#: hot path to a dict lookup plus the stride reads.
_PLAN_CACHE: dict = {}


def _plan(total_q, batch, num_heads, num_kv_heads, head_dim_q, head_dim_v, block_size,
          rel_extent, max_seqlen_q, num_splits, causal, window_left, window_right,
          kv_upper):
    key = (total_q, batch, num_heads, num_kv_heads, head_dim_q, head_dim_v, block_size,
           rel_extent, max_seqlen_q, num_splits, causal, window_left, window_right,
           kv_upper)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan
    if max_seqlen_q <= _SMALL_Q:
        block_m, block_n = _SMALL_Q_BLOCK_M, _SMALL_Q_BLOCK_N
    else:
        block_m, block_n = _BLOCK_M, _BLOCK_N
    num_q_blocks = triton.cdiv(max_seqlen_q, block_m)
    base_programs = num_q_blocks * batch * num_heads
    splits = _choose_splits(num_splits, base_programs, kv_upper, block_n, max_seqlen_q)
    split_kv = splits > 1
    has_left = window_left is not None and window_left >= 0
    has_right = window_right is not None and window_right >= 0
    plan = (
        splits,
        split_kv,
        num_q_blocks,
        block_m,
        block_n,
        max(16, triton.next_power_of_2(max(head_dim_q, head_dim_v))),
        has_left,
        has_right,
        bool(causal) and kv_upper <= rel_extent,
    )
    if len(_PLAN_CACHE) >= 64:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = plan
    return plan


#: Long prefill already gives every program plenty of work, and the combine
#: launch costs more than the extra parallelism buys (measured: long_prefill
#: 0.45 ms with one split vs 0.54 ms with two), so only short-query shapes split.
_MAX_SPLIT_QLEN = 64


def _choose_splits(num_splits, base_programs, kv_upper, block_n, max_seqlen_q):
    max_splits = max(1, min(_MAX_SPLITS, triton.cdiv(max(kv_upper, 1), block_n)))
    if _SPLITS_OVERRIDE is not None:
        return max(1, min(int(_SPLITS_OVERRIDE), max_splits))
    if num_splits > 1:
        return max(1, min(num_splits, max_splits))
    if max_seqlen_q > _MAX_SPLIT_QLEN:
        return 1
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
    device = q.device
    for tensor in (key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel_logits):
        if tensor.device != device:
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

    if not rel_logits.is_contiguous():
        rel_logits = rel_logits.contiguous()
    # One query head per program: packing the heads that share a KV head was
    # measured 4-8x slower here, because the per-row head index turns both the Q
    # tile load and the output store into scalar gathers.
    kv_group = num_heads // num_kv_heads
    h_grid = num_heads
    window_left, window_right = window_size
    kv_upper = block_table.shape[1] * block_size
    (
        splits,
        split_kv,
        num_q_blocks,
        block_m,
        block_n,
        block_d,
        has_left,
        has_right,
        bias_all_in_range,
    ) = _plan(
        total_q, batch_size, num_heads, num_kv_heads, head_dim_q, head_dim_v, block_size,
        rel_extent, max_seqlen_q, num_splits, bool(causal), window_left, window_right,
        kv_upper,
    )

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
        KV_GROUP=kv_group,
        BIAS_ALL_IN_RANGE=bias_all_in_range,
        HEAD_DIM=head_dim_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        REL_EXTENT=rel_extent,
        BLOCK_SIZE=block_size,
        CAUSAL=bool(causal),
        HAS_LEFT=has_left,
        WINDOW_LEFT=window_left if has_left else 0,
        HAS_RIGHT=has_right,
        WINDOW_RIGHT=window_right if has_right else 0,
        num_splits=splits,
        SPLIT_KV=split_kv,
        num_warps=_NUM_WARPS,
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
            splits,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim_v,
            BLOCK_D=block_d,
            ROWS=_COMBINE_ROWS,
            num_warps=4,
        )
    return out
