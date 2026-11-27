# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend paged relative attention with Cube, Vector decode and split-KV.

Queries occupy the last q_len positions of each cached sequence. Relative
logits apply where 0 <= q_pos - k_pos < rel_extent; causal and window masks
use the same shifted positions.
"""

import torch
import triton

from .paged_relative_attention import _combine_splits_kernel, _rel_attn_paged_kernel

# Short queries use fewer rows and wider KV tiles; longer queries use 64x64
# tiles to keep the fp32 scores and working state within the Ascend UB.
_BLOCK_M = 64
_BLOCK_N = 64
_SMALL_Q_BLOCK_M = 16
_SMALL_Q_BLOCK_N = 128
_SMALL_Q = 64
_NUM_WARPS = 4
_COMBINE_ROWS = 16
_MAX_HEAD_DIM = 256
_MAX_SPLITS = 32  # Automatic scheduling limit.
# Target grid size for automatic split-KV scheduling.
_TARGET_PROGRAMS = 128


def _plan(batch, num_heads, head_dim_q, head_dim_v, block_size,
          rel_extent, max_seqlen_q, num_splits, causal, window_left, window_right,
          kv_upper, contiguous_features):
    if max(head_dim_q, head_dim_v) > 128:
        # Bound the UB working set for wide QK or value heads.
        block_m, block_n = _SMALL_Q_BLOCK_M, _BLOCK_N
    elif max_seqlen_q == 1 and contiguous_features:
        block_m, block_n = 1, _SMALL_Q_BLOCK_N
    elif max_seqlen_q <= _SMALL_Q:
        block_m, block_n = _SMALL_Q_BLOCK_M, _SMALL_Q_BLOCK_N
    else:
        block_m, block_n = _BLOCK_M, _BLOCK_N
    num_q_blocks = triton.cdiv(max_seqlen_q, block_m)
    base_programs = num_q_blocks * batch * num_heads
    splits = _choose_splits(num_splits, base_programs, kv_upper, block_n, max_seqlen_q)
    if (block_m == 1 and block_size == 16 and head_dim_q == head_dim_v == 128
            and triton.cdiv(kv_upper, splits) >= 256):
        block_n = 64
    split_kv = splits > 1
    has_left = window_left is not None
    has_right = window_right is not None
    return (
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


# Long queries use query-tile parallelism without an additional combine launch.
_MAX_SPLIT_QLEN = 64


def _choose_splits(num_splits, base_programs, kv_upper, block_n, max_seqlen_q):
    if num_splits > 1:
        return num_splits
    max_splits = max(1, min(_MAX_SPLITS, triton.cdiv(max(kv_upper, 1), block_n)))
    if max_seqlen_q > _MAX_SPLIT_QLEN:
        return 1
    want = max(1, _TARGET_PROGRAMS // max(1, base_programs))
    splits = 1
    while splits < want and splits < max_splits:
        splits *= 2
    return splits


def _resolve_window(causal, window_size):
    """Apply the common FA4 window convention, including mixed -1 bounds."""
    left, right = (None, None) if window_size == (-1, -1) else window_size
    if causal:
        right = 0
    if left is not None and right is not None and left + right < 0:
        left, right = None, None
    if left is None and right == 0:
        return True, None, None
    # A non-positive right bound allows causal KV clipping.
    return bool(causal) or (right is not None and right <= 0), left, right


def _check(q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
           rel_logits, rel_extent, out, max_seqlen_q, window_size, num_splits):
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
    q_shape, k_shape, v_shape = q.shape, key_cache.shape, value_cache.shape
    q_dtype = q.dtype
    device = q.device
    if device.type != "npu":
        raise ValueError("attention inputs must be NPU tensors")
    for tensor in (key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q, rel_logits):
        if tensor.device != device:
            raise ValueError("all attention inputs must be on the same device")
    if q_shape[1] <= 0 or k_shape[2] <= 0:
        raise ValueError("Q and KV head counts must be positive")
    if k_shape[1] <= 0:
        raise ValueError("KV cache block_size must be positive")
    if q_shape[1] % k_shape[2] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if rel_logits.shape != (q_shape[0], q_shape[1], rel_extent):
        raise ValueError("rel_logits must have shape [total_q, num_heads, rel_extent]")
    if k_shape[:3] != v_shape[:3]:
        raise ValueError("key_cache and value_cache block/token/head shapes must match")
    if k_shape[3] != q_shape[2]:
        raise ValueError("Q and K head dimensions must match")
    if not (
        8 <= q_shape[2] <= _MAX_HEAD_DIM
        and 8 <= v_shape[3] <= _MAX_HEAD_DIM
        and q_shape[2] % 8 == 0
        and v_shape[3] % 8 == 0
    ):
        raise ValueError(f"Q/K and V head dimensions must be in [8, {_MAX_HEAD_DIM}] and aligned to 8")
    if block_table.shape[0] != cache_seqlens.shape[0]:
        raise ValueError("block_table and cache_seqlens batch sizes differ")
    if cu_seqlens_q.shape[0] != cache_seqlens.shape[0] + 1:
        raise ValueError("cu_seqlens_q must contain batch_size + 1 entries")
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q must use torch.bfloat16 or torch.float16")
    if key_cache.dtype != q_dtype or value_cache.dtype != q_dtype or rel_logits.dtype != q_dtype:
        raise TypeError("Q, K, V and rel_logits must have the same dtype")
    if block_table.dtype != torch.int32:
        raise TypeError("block_table must use torch.int32")
    if cache_seqlens.dtype != torch.int32 or cu_seqlens_q.dtype != torch.int32:
        raise TypeError("sequence metadata must use torch.int32")
    if cache_seqlens.stride(0) != 1 or cu_seqlens_q.stride(0) != 1:
        raise ValueError("sequence metadata must be contiguous")
    if max_seqlen_q < 0:
        raise ValueError("max_seqlen_q must be non-negative")
    if rel_extent <= 0:
        raise ValueError("rel_extent must be positive")
    if (
        not isinstance(window_size, tuple) or len(window_size) != 2
        or not all(isinstance(bound, int) and bound >= -1 for bound in window_size)
    ):
        raise ValueError("window_size must be a pair of integers >= -1")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    if out is not None:
        expected_shape = (q_shape[0], q_shape[1], v_shape[3])
        if out.shape != expected_shape or out.dtype != q_dtype or out.device != device:
            raise ValueError(
                "out must have shape [total_q, num_heads, value_head_dim] "
                "and match q dtype/device"
            )


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
    """Run paged relative attention with the common argument layout.

    On Ascend, num_splits=1 selects an automatic split count; values greater
    than one select exactly that count. The common implementation uses one
    split for num_splits=1. Q/K and V dimensions are limited to 256 here.
    """
    selected = (backend or "auto").strip().lower()
    if selected not in {"auto", "triton", "base", "tle", "triton_tle", "triton-tle"}:
        raise ValueError(f"unknown Inkling FA4 backend={selected!r}")
    # Accept common backend aliases for call compatibility.
    del backend
    _check(q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
           rel_logits, rel_extent, out, max_seqlen_q, window_size, num_splits)

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
    # Map each query head to the KV head shared by its GQA group.
    kv_group = num_heads // num_kv_heads
    h_grid = num_heads
    causal, window_left, window_right = _resolve_window(causal, window_size)
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
        batch_size, num_heads, head_dim_q, head_dim_v, block_size,
        rel_extent, max_seqlen_q, num_splits, bool(causal), window_left, window_right,
        kv_upper,
        q.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1,
    )

    if split_kv:
        partial_out = torch.empty(
            (splits, total_q, num_heads, head_dim_v), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (splits, total_q, num_heads), dtype=torch.float32, device=q.device
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
        VALUE_HEAD_DIM=head_dim_v,
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
        PAGE_LOAD=block_m == 1 and block_n == 64,
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
