# Enflame FLA shared chunk-state implementation.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import torch
import triton
import triton.language as tl

from .compat import exp2
from .utils import make_grid_3d, requires_grid_remap, tensor_cache

FALLBACK_STATE_BK = 32
FALLBACK_STATE_BV = 32
FALLBACK_STATE_NUM_WARPS = 4
FALLBACK_STATE_NUM_STAGES = 2
MAX_VARLEN_SEQUENCES = 65_536
PARALLEL_FWD_STATE_MAX_BK = 128
PARALLEL_FWD_STATE_MAX_BV = 256
PARALLEL_FWD_STATE_MAX_TILE_ELEMENTS = 32768
PARALLEL_FP32_SCAN_CHUNKS = 16
PARALLEL_FP32_SCAN_MAX_BK = 64
PARALLEL_FP32_SCAN_MAX_BV = 256
PARALLEL_FP32_SCAN_MAX_TILE_ELEMENTS = 16384
PARALLEL_STATE_ENABLED = (
    os.environ.get("FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE", "1") != "0"
)
# Match the generic chunk GLA contract: backward requests FP32 states and the
# backend selects its optimized implementation automatically.  The environment
# variable is retained only as an emergency fallback for older GCU compilers.
PARALLEL_BWD_STATE_ENABLED = (
    os.environ.get("FLAGGEMS_ENFLAME_GLA_PARALLEL_BWD_STATE", "1") != "0"
)


@triton.jit(do_not_specialize=["N"])
def _prepare_chunk_offsets_kernel(
    cu_seqlens,
    split_offsets,
    N,
    BS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    bos = tl.load(cu_seqlens + offsets, mask=mask, other=0).to(tl.int32)
    eos = tl.load(cu_seqlens + offsets + 1, mask=mask, other=0).to(tl.int32)
    chunk_counts = tl.where(mask, tl.cdiv(eos - bos, BS), 0).to(tl.int32)
    cumulative_counts = tl.cumsum(chunk_counts, axis=0).to(tl.int32)

    tl.store(split_offsets, 0)
    tl.store(split_offsets + offsets + 1, cumulative_counts, mask=mask)


@tensor_cache
def _prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    if cu_seqlens.dtype != torch.int32:
        raise TypeError(
            "Enflame chunk GLA requires cu_seqlens to use torch.int32, "
            f"but got {cu_seqlens.dtype}."
        )

    num_sequences = len(cu_seqlens) - 1
    block_size = triton.next_power_of_2(max(num_sequences, 1))
    if block_size > MAX_VARLEN_SEQUENCES:
        raise NotImplementedError(
            "Enflame chunk GLA supports at most "
            f"{MAX_VARLEN_SEQUENCES} variable-length sequences, "
            f"but got {num_sequences}."
        )

    split_offsets = torch.empty_like(cu_seqlens, dtype=torch.int32)
    _prepare_chunk_offsets_kernel[(1,)](
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        N=num_sequences,
        BS=chunk_size,
        BLOCK_N=block_size,
    )
    return split_offsets


def _prepare_state_metadata(
    cu_seqlens: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
    split_size: int,
) -> tuple[int, int, torch.Tensor | None]:
    if cu_seqlens is None:
        return batch_size, triton.cdiv(sequence_length, split_size), None

    split_offsets = _prepare_chunk_offsets(cu_seqlens, split_size)
    num_sequences = len(cu_seqlens) - 1
    num_splits = split_offsets[-1].item()
    return num_sequences, num_splits, split_offsets


def _select_parallel_fwd_state_tiles(key_dim: int, value_dim: int) -> tuple[int, int]:
    """Select a bounded forward tile, preferring wider value blocks.

    Enflame measurements favor expanding V more aggressively than K: BK is
    bounded at 128 while BV may reach 256.  Each tile follows the corresponding
    real dimension, so asymmetric states avoid padding; wide states use at most
    32768 accumulator elements.
    """

    bk = min(
        PARALLEL_FWD_STATE_MAX_BK,
        max(16, triton.next_power_of_2(key_dim)),
    )
    bv = min(
        PARALLEL_FWD_STATE_MAX_BV,
        max(16, triton.next_power_of_2(value_dim)),
    )
    if bk * bv > PARALLEL_FWD_STATE_MAX_TILE_ELEMENTS:
        bk = max(16, PARALLEL_FWD_STATE_MAX_TILE_ELEMENTS // bv)
    return bk, bv


def _select_parallel_bwd_scan_tiles(key_dim: int, value_dim: int) -> tuple[int, int]:
    """Select a bounded FP32 scan tile, favoring contiguous V expansion."""

    bk = min(
        PARALLEL_FP32_SCAN_MAX_BK,
        max(16, triton.next_power_of_2(key_dim)),
    )
    bv = min(
        PARALLEL_FP32_SCAN_MAX_BV,
        max(16, triton.next_power_of_2(value_dim)),
    )
    if bk * bv > PARALLEL_FP32_SCAN_MAX_TILE_ELEMENTS:
        bv = max(16, PARALLEL_FP32_SCAN_MAX_TILE_ELEMENTS // bk)
    return bk, bv


@triton.jit(do_not_specialize=["T", "N_CHUNKS"])
def _chunk_fwd_local_state_kernel(
    k,
    v,
    gk,
    local_state,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N_CHUNKS,
    N_BATCH_HEADS,
    USE_GRID_REMAP: tl.constexpr,
):
    grid_k = tl.cdiv(K, BK)
    grid_v = tl.cdiv(V, BV)
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        i_t = linear_pid % N_CHUNKS
        linear_pid //= N_CHUNKS
        i_kv = linear_pid % (grid_k * grid_v)
        i_bh = linear_pid // (grid_k * grid_v)
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_t, i_kv, i_bh = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
    i_k, i_v = i_kv // grid_v, i_kv % grid_v
    i_b, i_h = i_bh // H, i_bh % H

    p_k = tl.make_block_ptr(
        k + (i_b * T * H + i_h) * K,
        (K, T),
        (1, H * K),
        (i_k * BK, i_t * BT),
        (BK, BT),
        (0, 1),
    )
    p_v = tl.make_block_ptr(
        v + (i_b * T * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_gk = tl.make_block_ptr(
        gk + (i_b * T * H + i_h) * K,
        (K, T),
        (1, H * K),
        (i_k * BK, i_t * BT),
        (BK, BT),
        (0, 1),
    )
    last_idx = min((i_t + 1) * BT, T) - 1
    o_k = i_k * BK + tl.arange(0, BK)
    p_gk_last = gk + ((i_b * T + last_idx) * H + i_h) * K + o_k

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gk_last = tl.load(p_gk_last, mask=o_k < K, other=0.0)
    b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)
    b_local = tl.zeros([BK, BV], dtype=tl.float32)
    b_local = tl.dot(b_k, b_v, acc=b_local, out_dtype=tl.float32)

    state_offset = ((i_b * N_CHUNKS + i_t) * H + i_h).to(tl.int64) * K * V
    p_local = tl.make_block_ptr(
        local_state + state_offset,
        (K, V),
        (V, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(p_local, b_local, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "N_CHUNKS"])
def _chunk_fwd_state_scan_kernel(
    gk,
    local_state,
    h,
    h0,
    ht,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N_CHUNKS,
    N_BATCH_HEADS,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_IN_FP32: tl.constexpr,
    USE_GRID_REMAP: tl.constexpr,
):
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        grid_k = tl.cdiv(K, BK)
        grid_v = tl.cdiv(V, BV)
        i_k = linear_pid % grid_k
        linear_pid //= grid_k
        i_v = linear_pid % grid_v
        i_bh = linear_pid // grid_v
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_k, i_v, i_bh = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
    i_b, i_h = i_bh // H, i_bh % H
    b_state = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0 + i_bh * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_state = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    o_k = i_k * BK + tl.arange(0, BK)
    for i_t in range(N_CHUNKS):
        state_offset = ((i_b * N_CHUNKS + i_t) * H + i_h).to(tl.int64) * K * V
        p_local = tl.make_block_ptr(
            local_state + state_offset,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_local = tl.load(p_local, boundary_check=(0, 1)).to(tl.float32)
        p_h = tl.make_block_ptr(
            h + state_offset,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        if STATE_IN_FP32:
            # Avoid materializing an FP32 -> FP32 no-op conversion.  Enflame's
            # layout-removal pass assumes that conversion still exists and can
            # otherwise dyn_cast a value erased by canonicalization.
            tl.store(p_h, b_state, boundary_check=(0, 1))
        else:
            tl.store(p_h, b_state.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        p_gk_last = gk + ((i_b * T + last_idx) * H + i_h) * K + o_k
        b_gk_last = tl.load(p_gk_last, mask=o_k < K, other=0.0)
        b_state *= exp2(b_gk_last)[:, None]
        b_state += b_local

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht + i_bh * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_ht, b_state, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "N_CHUNKS", "GROUP_START"])
def _chunk_fwd_state_scan_fp32_2d_group_kernel(
    gk,
    local_state,
    h,
    running_state,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNKS_PER_LAUNCH: tl.constexpr,
    N_CHUNKS,
    GROUP_START,
    N_BATCH_HEADS,
    USE_GRID_REMAP: tl.constexpr,
):
    grid_k = tl.cdiv(K, BK)
    grid_v = tl.cdiv(V, BV)
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        i_k = linear_pid % grid_k
        linear_pid //= grid_k
        i_v = linear_pid % grid_v
        i_bh = linear_pid // grid_v
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H
    p_running = tl.make_block_ptr(
        running_state + i_bh * K * V,
        (K, V),
        (V, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    b_state = tl.load(p_running, boundary_check=(0, 1)).to(tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)

    for group_offset in tl.static_range(0, CHUNKS_PER_LAUNCH):
        chunk_index = GROUP_START + group_offset
        state_base = (
            (i_b * N_CHUNKS + chunk_index) * H + i_h
        ).to(tl.int64) * K * V
        p_local = tl.make_block_ptr(
            local_state + state_base,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_h = tl.make_block_ptr(
            h + state_base,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_local = tl.load(p_local, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_h, b_state, boundary_check=(0, 1))

        last_idx = min((chunk_index + 1) * BT, T) - 1
        b_gk_last = tl.load(
            gk + ((i_b * T + last_idx) * H + i_h) * K + o_k,
            mask=o_k < K,
            other=0.0,
        )
        b_state *= exp2(b_gk_last)[:, None]
        b_state += b_local

    tl.store(p_running, b_state, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "N_CHUNKS"])
def _chunk_bwd_local_state_kernel(
    q,
    do,
    gk,
    local_state,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N_CHUNKS,
    N_BATCH_HEADS,
    USE_GRID_REMAP: tl.constexpr,
):
    grid_k = tl.cdiv(K, BK)
    grid_v = tl.cdiv(V, BV)
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        i_t = linear_pid % N_CHUNKS
        linear_pid //= N_CHUNKS
        i_kv = linear_pid % (grid_k * grid_v)
        i_bh = linear_pid // (grid_k * grid_v)
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_t, i_kv, i_bh = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
    i_k, i_v = i_kv // grid_v, i_kv % grid_v
    i_b, i_h = i_bh // H, i_bh % H

    p_q = tl.make_block_ptr(
        q + (i_b * T * H + i_h) * K,
        (K, T),
        (1, H * K),
        (i_k * BK, i_t * BT),
        (BK, BT),
        (0, 1),
    )
    p_do = tl.make_block_ptr(
        do + (i_b * T * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    p_gk = tl.make_block_ptr(
        gk + (i_b * T * H + i_h) * K,
        (K, T),
        (1, H * K),
        (i_k * BK, i_t * BT),
        (BK, BT),
        (0, 1),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_q = (b_q * exp2(b_gk)).to(b_q.dtype)
    b_local = tl.zeros([BK, BV], dtype=tl.float32)
    b_local = tl.dot(
        b_q,
        b_do.to(b_q.dtype),
        acc=b_local,
        out_dtype=tl.float32,
    )

    state_offset = ((i_b * N_CHUNKS + i_t) * H + i_h).to(tl.int64) * K * V
    p_local = tl.make_block_ptr(
        local_state + state_offset,
        (K, V),
        (V, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(p_local, b_local, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "N_CHUNKS"])
def _chunk_bwd_state_scan_kernel(
    gk,
    local_state,
    dh,
    dht,
    dh0,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N_CHUNKS,
    N_BATCH_HEADS,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    STATE_IN_FP32: tl.constexpr,
    USE_GRID_REMAP: tl.constexpr,
):
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        grid_k = tl.cdiv(K, BK)
        grid_v = tl.cdiv(V, BV)
        i_k = linear_pid % grid_k
        linear_pid //= grid_k
        i_v = linear_pid % grid_v
        i_bh = linear_pid // grid_v
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_k, i_v, i_bh = (
            tl.program_id(0),
            tl.program_id(1),
            tl.program_id(2),
        )
    i_b, i_h = i_bh // H, i_bh % H
    b_state = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(
            dht + i_bh * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_state = tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    o_k = i_k * BK + tl.arange(0, BK)
    for i_t in range(N_CHUNKS - 1, -1, -1):
        state_offset = ((i_b * N_CHUNKS + i_t) * H + i_h).to(tl.int64) * K * V
        p_local = tl.make_block_ptr(
            local_state + state_offset,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_local = tl.load(p_local, boundary_check=(0, 1)).to(tl.float32)
        p_dh = tl.make_block_ptr(
            dh + state_offset,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        if STATE_IN_FP32:
            tl.store(p_dh, b_state, boundary_check=(0, 1))
        else:
            tl.store(p_dh, b_state.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        p_gk_last = gk + ((i_b * T + last_idx) * H + i_h) * K + o_k
        b_gk_last = tl.load(p_gk_last, mask=o_k < K, other=0.0)
        b_state *= exp2(b_gk_last)[:, None]
        b_state += b_local

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(
            dh0 + i_bh * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        tl.store(p_dh0, b_state, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T", "N_CHUNKS", "GROUP_START"])
def _chunk_bwd_state_scan_fp32_2d_group_kernel(
    gk,
    local_state,
    dh,
    running_state,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    CHUNKS_PER_LAUNCH: tl.constexpr,
    N_CHUNKS,
    GROUP_START,
    N_BATCH_HEADS,
    USE_GRID_REMAP: tl.constexpr,
):
    grid_k = tl.cdiv(K, BK)
    grid_v = tl.cdiv(V, BV)
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        i_k = linear_pid % grid_k
        linear_pid //= grid_k
        i_v = linear_pid % grid_v
        i_bh = linear_pid // grid_v
        if i_bh >= N_BATCH_HEADS:
            return
    else:
        i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    i_b, i_h = i_bh // H, i_bh % H
    p_running = tl.make_block_ptr(
        running_state + i_bh * K * V,
        (K, V),
        (V, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    b_state = tl.load(p_running, boundary_check=(0, 1)).to(tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)

    for group_offset in tl.static_range(0, CHUNKS_PER_LAUNCH):
        chunk_index = GROUP_START - group_offset
        state_base = (
            (i_b * N_CHUNKS + chunk_index) * H + i_h
        ).to(tl.int64) * K * V
        p_local = tl.make_block_ptr(
            local_state + state_base,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_dh = tl.make_block_ptr(
            dh + state_base,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        b_local = tl.load(p_local, boundary_check=(0, 1)).to(tl.float32)
        tl.store(p_dh, b_state, boundary_check=(0, 1))

        last_idx = min((chunk_index + 1) * BT, T) - 1
        b_gk_last = tl.load(
            gk + ((i_b * T + last_idx) * H + i_h) * K + o_k,
            mask=o_k < K,
            other=0.0,
        )
        b_state *= exp2(b_gk_last)[:, None]
        b_state += b_local

    tl.store(p_running, b_state, boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    g_gamma,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens,
    split_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    N_BATCH_HEADS,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_GRID_REMAP: tl.constexpr,
):
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        grid_x = tl.cdiv(K, BK)
        grid_y = tl.cdiv(V, BV)
        i_k = linear_pid % grid_x
        linear_pid //= grid_x
        i_v = linear_pid % grid_y
        i_nh = linear_pid // grid_y
        if i_nh >= N_BATCH_HEADS:
            return
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT, NS = tl.cdiv(T, BT), tl.cdiv(T, BS)
        boh = i_n * NS
    NTS = BS // BT

    if USE_G_GAMMA:
        # decay rate given the head index
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] accumulator; STATE_V_FIRST only flips the stored state's HBM layout to [V, K]
    # applied at the load/store below.
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0 = tl.make_block_ptr(
                h0 + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            b_h = tl.trans(tl.load(p_h0, boundary_check=(0, 1))).to(tl.float32)
        else:
            p_h0 = tl.make_block_ptr(
                h0 + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        i_s = i_t // NTS
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (K, T),
            (1, H * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )

        o_h = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(
                h + o_h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_h = tl.make_block_ptr(
                h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )

        if i_t % NTS == 0:
            tl.store(
                p_h,
                (tl.trans(b_h) if STATE_V_FIRST else b_h).to(p_h.dtype.element_ty),
                boundary_check=(0, 1),
            )
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1

        # scalar decay
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.0)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_h *= exp2(b_g_last)
            b_v = (b_v * exp2(b_g_last - b_g)[:, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            p_gk = tl.make_block_ptr(
                gk + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            p_gk_last = (
                gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            )

            b_gk_last = tl.load(
                p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.0
            )
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_h *= exp2(b_gk_last)[:, None]
            b_k = (b_k * exp2(b_gk_last[:, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            p_gv = tl.make_block_ptr(
                gv + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_gv_last = (
                gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            )

            b_gv_last = tl.load(
                p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.0
            )
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_h *= exp2(b_gv_last)[None, :]
            b_v = (b_v * exp2(b_gv_last[None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            tl.store(
                p_ht, tl.trans(b_h).to(p_ht.dtype.element_ty), boundary_check=(0, 1)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_kernel_dh(
    q,
    g,
    g_gamma,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    cu_seqlens,
    split_offsets,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    N_BATCH_HEADS,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    USE_GRID_REMAP: tl.constexpr,
):
    if USE_GRID_REMAP:
        linear_pid = tl.program_id(0) + tl.num_programs(0) * (
            tl.program_id(1) + tl.num_programs(1) * tl.program_id(2)
        )
        grid_x = tl.cdiv(K, BK)
        grid_y = tl.cdiv(V, BV)
        i_k = linear_pid % grid_x
        linear_pid //= grid_x
        i_v = linear_pid % grid_y
        i_nh = linear_pid // grid_y
        if i_nh >= N_BATCH_HEADS:
            return
    else:
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // NG
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = tl.load(split_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        NS = tl.cdiv(T, BS)
        boh = i_n * NS

    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)

    # [BK, BV] accumulator; STATE_V_FIRST only flips the stored state's HBM layout to [V, K]
    # applied at the load/store below.
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dht = tl.make_block_ptr(
                dht + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            b_dh += tl.trans(tl.load(p_dht, boundary_check=(0, 1))).to(tl.float32)
        else:
            p_dht = tl.make_block_ptr(
                dht + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            b_dh += tl.load(p_dht, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        i_s = i_t // (BS // BT)
        o_dh = ((boh + i_s) * H + i_h).to(tl.int64) * K * V
        if STATE_V_FIRST:
            p_dh = tl.make_block_ptr(
                dh + o_dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0)
            )
        else:
            p_dh = tl.make_block_ptr(
                dh + o_dh, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0)
            )

        if i_t % (BS // BT) == 0:
            tl.store(
                p_dh,
                (tl.trans(b_dh) if STATE_V_FIRST else b_dh).to(p_dh.dtype.element_ty),
                boundary_check=(0, 1),
            )
        last_idx = min(i_t * BT + BT, T) - 1
        # [BK, BT]
        p_q = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K,
            (K, T),
            (1, HQ * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V,
            (T, V),
            (HQ * V, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        if USE_G:
            p_g = g + (bos + i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(g + (bos + last_idx) * H + i_h)
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.0)
            b_q = (b_q * exp2(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp2(b_g_last)

        if USE_G_GAMMA:
            b_g_last = b_gamma * min(BT, T - i_t * BT)
            b_q = (b_q * exp2(b_g)[None, :]).to(b_q.dtype)
            b_dh *= exp2(b_g_last)

        if USE_GK:
            p_gk = tl.make_block_ptr(
                gk + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            p_gk_last = (
                gk + (bos + last_idx) * H * K + i_h * K + i_k * BK + tl.arange(0, BK)
            )

            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_gk_last = tl.load(
                p_gk_last, mask=(i_k * BK + tl.arange(0, BK) < K), other=0.0
            )
            b_q = (b_q * exp2(b_gk)).to(b_q.dtype)
            b_dh *= exp2(b_gk_last)[:, None]

        if USE_GV:
            p_gv = tl.make_block_ptr(
                gv + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_gv_last = (
                gv + (bos + last_idx) * H * V + i_h * V + i_v * BV + tl.arange(0, BV)
            )

            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_gv_last = tl.load(
                p_gv_last, mask=(i_v * BV + tl.arange(0, BV) < V), other=0.0
            )
            b_do = b_do * exp2(b_gv)
            b_dh *= exp2(b_gv_last)[None, :]

        b_dh += tl.dot(b_q, b_do.to(b_q.dtype))

    if STORE_INITIAL_STATE_GRADIENT:
        if STATE_V_FIRST:
            p_dh0 = tl.make_block_ptr(
                dh0 + i_nh * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, i_k * BK),
                (BV, BK),
                (1, 0),
            )
            tl.store(
                p_dh0, tl.trans(b_dh).to(p_dh0.dtype.element_ty), boundary_check=(0, 1)
            )
        else:
            p_dh0 = tl.make_block_ptr(
                dh0 + i_nh * K * V,
                (K, V),
                (V, 1),
                (i_k * BK, i_v * BV),
                (BK, BV),
                (1, 0),
            )
            tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    if BS % BT != 0:
        raise ValueError(
            f"The split_size (got {BS}) must be a multiple of chunk_size {BT}."
        )
    N, NS, split_offsets = _prepare_state_metadata(cu_seqlens, B, T, BS)

    # `state_v_first` stores the states in V-first `[V, K]` layout instead of `[K, V]`
    state_shape = (V, K) if state_v_first else (K, V)
    h = k.new_empty(
        B, NS, H, *state_shape, dtype=k.dtype if not states_in_fp32 else torch.float
    )
    ht = (
        k.new_empty(N, H, *state_shape, dtype=torch.float)
        if output_final_state
        else None
    )

    use_parallel_state = (
        PARALLEL_STATE_ENABLED
        and cu_seqlens is None
        and BS == BT
        and BT == 64
        # Autograd recomputes h in FP32.  Gate that specialization together
        # with the independently controlled parallel backward-state path.
        and (not states_in_fp32 or PARALLEL_BWD_STATE_ENABLED)
        and not state_v_first
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and g is None
        and g_gamma is None
        and gk is not None
        and gv is None
    )
    if use_parallel_state:
        NT = NS
        # Keep local contributions and scanned outputs in distinct buffers.
        # Besides making the two-stage dataflow explicit, this avoids alias
        # folding in Enflame's ConvertLayout lowering for FP32 output states.
        local_state = k.new_empty(B, NT, H, K, V, dtype=torch.float)
        state_bk, state_bv = _select_parallel_fwd_state_tiles(K, V)
        grid_k = triton.cdiv(K, state_bk)
        grid_v = triton.cdiv(V, state_bv)

        local_grid_remap = requires_grid_remap(NT, grid_k * grid_v, B * H)
        local_grid = make_grid_3d(
            NT,
            grid_k * grid_v,
            B * H,
            remap=local_grid_remap,
        )
        _chunk_fwd_local_state_kernel[local_grid](
            k=k,
            v=v,
            gk=gk,
            local_state=local_state,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=state_bk,
            BV=state_bv,
            N_CHUNKS=NT,
            N_BATCH_HEADS=B * H,
            USE_GRID_REMAP=local_grid_remap,
            num_warps=1,
            num_stages=1,
        )

        if states_in_fp32:
            running_state = (
                h0.to(dtype=torch.float).clone()
                if h0 is not None
                else k.new_zeros(B, H, K, V, dtype=torch.float)
            )
            scan_bk, scan_bv = _select_parallel_bwd_scan_tiles(K, V)
            scan_grid_k = triton.cdiv(K, scan_bk)
            scan_grid_v = triton.cdiv(V, scan_bv)
            scan_2d_remap = requires_grid_remap(scan_grid_k, scan_grid_v, B * H)
            scan_2d_grid = make_grid_3d(
                scan_grid_k,
                scan_grid_v,
                B * H,
                remap=scan_2d_remap,
            )
            n_full_chunks = NT // PARALLEL_FP32_SCAN_CHUNKS * PARALLEL_FP32_SCAN_CHUNKS
            for group_start in range(0, n_full_chunks, PARALLEL_FP32_SCAN_CHUNKS):
                _chunk_fwd_state_scan_fp32_2d_group_kernel[scan_2d_grid](
                    gk=gk,
                    local_state=local_state,
                    h=h,
                    running_state=running_state,
                    T=T,
                    H=H,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=scan_bk,
                    BV=scan_bv,
                    CHUNKS_PER_LAUNCH=PARALLEL_FP32_SCAN_CHUNKS,
                    N_CHUNKS=NT,
                    GROUP_START=group_start,
                    N_BATCH_HEADS=B * H,
                    USE_GRID_REMAP=scan_2d_remap,
                    num_warps=1,
                    num_stages=1,
                )
            n_tail_chunks = NT - n_full_chunks
            if n_tail_chunks > 0:
                _chunk_fwd_state_scan_fp32_2d_group_kernel[scan_2d_grid](
                    gk=gk,
                    local_state=local_state,
                    h=h,
                    running_state=running_state,
                    T=T,
                    H=H,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=scan_bk,
                    BV=scan_bv,
                    CHUNKS_PER_LAUNCH=n_tail_chunks,
                    N_CHUNKS=NT,
                    GROUP_START=n_full_chunks,
                    N_BATCH_HEADS=B * H,
                    USE_GRID_REMAP=scan_2d_remap,
                    num_warps=1,
                    num_stages=1,
                )
            if ht is not None:
                ht.copy_(running_state)
        else:
            scan_grid_remap = requires_grid_remap(grid_k, grid_v, B * H)
            scan_grid = make_grid_3d(
                grid_k,
                grid_v,
                B * H,
                remap=scan_grid_remap,
            )
            _chunk_fwd_state_scan_kernel[scan_grid](
                gk=gk,
                local_state=local_state,
                h=h,
                h0=h0,
                ht=ht,
                T=T,
                H=H,
                K=K,
                V=V,
                BT=BT,
                BK=state_bk,
                BV=state_bv,
                N_CHUNKS=NT,
                N_BATCH_HEADS=B * H,
                USE_INITIAL_STATE=h0 is not None,
                STORE_FINAL_STATE=ht is not None,
                STATE_IN_FP32=False,
                USE_GRID_REMAP=scan_grid_remap,
                num_warps=1,
                num_stages=1,
            )
        return h, ht

    use_grid_remap = requires_grid_remap(
        triton.cdiv(K, FALLBACK_STATE_BK),
        triton.cdiv(V, FALLBACK_STATE_BV),
        N * H,
    )

    def grid(meta):
        grid_x = triton.cdiv(K, meta["BK"])
        grid_y = triton.cdiv(V, meta["BV"])
        return make_grid_3d(
            grid_x,
            grid_y,
            N * H,
            remap=use_grid_remap,
        )

    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        N_BATCH_HEADS=N * H,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        STATE_V_FIRST=state_v_first,
        USE_GRID_REMAP=use_grid_remap,
        BK=FALLBACK_STATE_BK,
        BV=FALLBACK_STATE_BV,
        num_warps=FALLBACK_STATE_NUM_WARPS,
        num_stages=FALLBACK_STATE_NUM_STAGES,
    )
    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    split_size: int | None = None,
    states_in_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    BT = chunk_size
    BS = BT if split_size is None else split_size
    if BS % BT != 0:
        raise ValueError(
            f"The split_size (got {BS}) must be a multiple of chunk_size {BT}."
        )
    N, NS, split_offsets = _prepare_state_metadata(cu_seqlens, B, T, BS)
    NG = HQ // H

    # `state_v_first` stores the states in V-first `[V, K]` layout instead of `[K, V]`
    state_shape = (V, K) if state_v_first else (K, V)
    dh = k.new_empty(
        B, NS, HQ, *state_shape, dtype=k.dtype if not states_in_fp32 else torch.float
    )
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    use_parallel_state = (
        PARALLEL_STATE_ENABLED
        and PARALLEL_BWD_STATE_ENABLED
        and cu_seqlens is None
        and BS == BT
        and BT == 64
        and HQ == H
        and not state_v_first
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and do.dtype == torch.bfloat16
        and g is None
        and g_gamma is None
        and gk is not None
        and gv is None
    )
    if use_parallel_state:
        NT = NS
        local_state = k.new_empty(B, NT, H, K, V, dtype=torch.float)
        state_bk, state_bv = _select_parallel_fwd_state_tiles(K, V)
        grid_k = triton.cdiv(K, state_bk)
        grid_v = triton.cdiv(V, state_bv)

        local_grid_remap = requires_grid_remap(NT, grid_k * grid_v, B * H)
        local_grid = make_grid_3d(
            NT,
            grid_k * grid_v,
            B * H,
            remap=local_grid_remap,
        )
        _chunk_bwd_local_state_kernel[local_grid](
            q=q,
            do=do,
            gk=gk,
            local_state=local_state,
            scale=scale,
            T=T,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=state_bk,
            BV=state_bv,
            N_CHUNKS=NT,
            N_BATCH_HEADS=B * H,
            USE_GRID_REMAP=local_grid_remap,
            num_warps=1,
            num_stages=1,
        )

        if states_in_fp32:
            running_state = (
                dht.to(dtype=torch.float).clone()
                if dht is not None
                else k.new_zeros(B, H, K, V, dtype=torch.float)
            )
            scan_bk, scan_bv = _select_parallel_bwd_scan_tiles(K, V)
            scan_grid_k = triton.cdiv(K, scan_bk)
            scan_grid_v = triton.cdiv(V, scan_bv)
            scan_2d_remap = requires_grid_remap(scan_grid_k, scan_grid_v, B * H)
            scan_2d_grid = make_grid_3d(
                scan_grid_k,
                scan_grid_v,
                B * H,
                remap=scan_2d_remap,
            )
            n_full_groups = NT // PARALLEL_FP32_SCAN_CHUNKS
            for group_index in range(n_full_groups):
                group_start = NT - 1 - group_index * PARALLEL_FP32_SCAN_CHUNKS
                _chunk_bwd_state_scan_fp32_2d_group_kernel[scan_2d_grid](
                    gk=gk,
                    local_state=local_state,
                    dh=dh,
                    running_state=running_state,
                    T=T,
                    H=H,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=scan_bk,
                    BV=scan_bv,
                    CHUNKS_PER_LAUNCH=PARALLEL_FP32_SCAN_CHUNKS,
                    N_CHUNKS=NT,
                    GROUP_START=group_start,
                    N_BATCH_HEADS=B * H,
                    USE_GRID_REMAP=scan_2d_remap,
                    num_warps=1,
                    num_stages=1,
                )
            n_tail_chunks = NT - n_full_groups * PARALLEL_FP32_SCAN_CHUNKS
            if n_tail_chunks > 0:
                _chunk_bwd_state_scan_fp32_2d_group_kernel[scan_2d_grid](
                    gk=gk,
                    local_state=local_state,
                    dh=dh,
                    running_state=running_state,
                    T=T,
                    H=H,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=scan_bk,
                    BV=scan_bv,
                    CHUNKS_PER_LAUNCH=n_tail_chunks,
                    N_CHUNKS=NT,
                    GROUP_START=n_tail_chunks - 1,
                    N_BATCH_HEADS=B * H,
                    USE_GRID_REMAP=scan_2d_remap,
                    num_warps=1,
                    num_stages=1,
                )
            if dh0 is not None:
                dh0.copy_(running_state)
        else:
            scan_grid_remap = requires_grid_remap(grid_k, grid_v, B * H)
            scan_grid = make_grid_3d(
                grid_k,
                grid_v,
                B * H,
                remap=scan_grid_remap,
            )
            _chunk_bwd_state_scan_kernel[scan_grid](
                gk=gk,
                local_state=local_state,
                dh=dh,
                dht=dht,
                dh0=dh0,
                T=T,
                H=H,
                K=K,
                V=V,
                BT=BT,
                BK=state_bk,
                BV=state_bv,
                N_CHUNKS=NT,
                N_BATCH_HEADS=B * H,
                USE_FINAL_STATE_GRADIENT=dht is not None,
                STORE_INITIAL_STATE_GRADIENT=dh0 is not None,
                STATE_IN_FP32=False,
                USE_GRID_REMAP=scan_grid_remap,
                num_warps=1,
                num_stages=1,
            )
        return dh, dh0

    use_grid_remap = requires_grid_remap(
        triton.cdiv(K, FALLBACK_STATE_BK),
        triton.cdiv(V, FALLBACK_STATE_BV),
        N * H,
    )

    def grid(meta):
        grid_x = triton.cdiv(K, meta["BK"])
        grid_y = triton.cdiv(V, meta["BV"])
        return make_grid_3d(
            grid_x,
            grid_y,
            N * H,
            remap=use_grid_remap,
        )

    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        split_offsets=split_offsets,
        scale=scale,
        T=T,
        HQ=HQ,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        NG=NG,
        N_BATCH_HEADS=N * H,
        USE_G=g is not None,
        USE_G_GAMMA=g_gamma is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
        STATE_V_FIRST=state_v_first,
        USE_GRID_REMAP=use_grid_remap,
        BK=FALLBACK_STATE_BK,
        BV=FALLBACK_STATE_BV,
        num_warps=FALLBACK_STATE_NUM_WARPS,
        num_stages=FALLBACK_STATE_NUM_STAGES,
    )
    return dh, dh0
