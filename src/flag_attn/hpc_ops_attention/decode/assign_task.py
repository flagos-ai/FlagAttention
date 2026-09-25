# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU task-map construction shared by dynamic attention-decode kernels."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import triton
import triton.language as tl
from . import tle


# The no-TLE path deliberately does not emulate ``tle.cumsum``.  It uses the
# independent scalar scheduler below, matching the proven v3 pure-Triton
# split-K task-map format.

PURE_TRITON_TASK_STRIDE = 12
PURE_TRITON_TILE_N = 64
_PURE_TRITON_TASK_STRIDE = tl.constexpr(12)
_PURE_TRITON_TILE_N = tl.constexpr(64)


@triton.jit
def assign_pure_triton_task_map_kernel(
    SEQLENS_KV,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MIN_PROCESS_LEN: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    SCHED_INTS: tl.constexpr,
):
    """Build the portable MTP1 split-K schedule without TLE primitives."""
    total_tiles_per_head = 0
    b_scan = 0
    while b_scan < B:
        total_len = tl.load(SEQLENS_KV + b_scan)
        total_tiles_per_head += (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N
        b_scan += 1

    total_tiles = total_tiles_per_head * H_KV
    tiles_per_cta = (total_tiles + NUM_CTAS - 1) // NUM_CTAS
    minimum_tiles = MIN_PROCESS_LEN // _PURE_TRITON_TILE_N
    if minimum_tiles < 1:
        minimum_tiles = 1
    tiles_per_cta = tl.maximum(tiles_per_cta, minimum_tiles)
    if tl.program_id(0) == 0:
        tl.store(TASK_MAP + 0, tiles_per_cta + 1)
        tl.store(TASK_MAP + 1, NUM_CTAS)
        tl.store(TASK_MAP + 2, H_KV)
        tl.store(TASK_MAP + 3, B)
        tl.store(TASK_MAP + 4, SCHED_INTS * 4)

    num_chunks_base = (MAX_TASKS * NUM_CTAS + 1) * _PURE_TRITON_TASK_STRIDE
    hkv = 0
    batch = 0
    chunks = 0
    start_tiles = 0
    total_len = tl.load(SEQLENS_KV)
    tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

    cta = 0
    while cta < NUM_CTAS:
        bucket = tiles_per_cta
        slot = 0
        while (bucket > 0) & (hkv < H_KV) & (slot < MAX_TASKS - 1):
            if tiles_left <= 0:
                batch += 1
                if batch >= B:
                    batch = 0
                    hkv += 1
                chunks = 0
                start_tiles = 0
                if hkv < H_KV:
                    total_len = tl.load(SEQLENS_KV + batch)
                    tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

            if (tiles_left > 0) & (hkv < H_KV):
                cur_hkv = hkv
                cur_batch = batch
                total_len = tl.load(SEQLENS_KV + cur_batch)
                chunk = chunks
                add_tiles = tl.minimum(tiles_left, bucket)
                if chunks == NUM_CTAS - 1:
                    add_tiles = tiles_left
                seq_start = start_tiles * _PURE_TRITON_TILE_N
                seq_len = tl.minimum(
                    add_tiles * _PURE_TRITON_TILE_N,
                    total_len - seq_start,
                )
                start_tiles += add_tiles
                tiles_left -= add_tiles
                bucket -= add_tiles
                chunks += 1

                task_base = (MAX_TASKS * cta + slot + 1) * _PURE_TRITON_TASK_STRIDE
                tl.store(TASK_MAP + task_base + 0, cur_hkv)
                tl.store(TASK_MAP + task_base + 1, cur_batch)
                tl.store(TASK_MAP + task_base + 2, chunk)
                tl.store(TASK_MAP + task_base + 3, seq_start)
                tl.store(TASK_MAP + task_base + 4, seq_len)
                tl.store(TASK_MAP + task_base + 5, seq_len)
                tl.store(
                    TASK_MAP + task_base + 6,
                    (seq_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N,
                )
                tl.store(
                    TASK_MAP + task_base + 7,
                    seq_len // _PURE_TRITON_TILE_N,
                )
                tl.store(TASK_MAP + task_base + 8, tiles_left <= 0)
                tl.store(TASK_MAP + task_base + 9, 0)
                tl.store(TASK_MAP + task_base + 10, 0)
                tl.store(TASK_MAP + task_base + 11, 0)
                slot += 1

                if tiles_left <= 0:
                    tl.store(
                        TASK_MAP + num_chunks_base + cur_hkv * B + cur_batch,
                        chunks,
                    )
                    batch = cur_batch + 1
                    if batch >= B:
                        batch = 0
                        hkv = cur_hkv + 1
                    chunks = 0
                    start_tiles = 0
                    if hkv < H_KV:
                        total_len = tl.load(SEQLENS_KV + batch)
                        tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

        terminator = (MAX_TASKS * cta + slot + 1) * _PURE_TRITON_TASK_STRIDE
        tl.store(TASK_MAP + terminator + 0, -1)
        tl.store(TASK_MAP + terminator + 1, -1)
        cta += 1


def pure_triton_task_map_metadata(
    kv_lens: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
) -> tuple[int, int, int]:
    """Return fixed-capacity metadata for the portable scheduler."""
    lengths = kv_lens.detach().cpu().to(torch.int32).tolist()
    tiles = [(length + PURE_TRITON_TILE_N - 1) // PURE_TRITON_TILE_N for length in lengths]
    total_tiles = sum(tiles) * num_head_kv
    tiles_per_cta = max(
        (total_tiles + num_ctas - 1) // num_ctas,
        max(1, min_process_len // PURE_TRITON_TILE_N),
    )
    max_tasks = tiles_per_cta + 1
    chunk_ints = len(lengths) * num_head_kv
    chunk_pad = (chunk_ints + PURE_TRITON_TASK_STRIDE - 1) // PURE_TRITON_TASK_STRIDE * PURE_TRITON_TASK_STRIDE
    sched_ints = (max_tasks * num_ctas + 1) * PURE_TRITON_TASK_STRIDE + chunk_pad
    return max_tasks, sched_ints, sched_ints


def launch_pure_triton_task_map(
    kv_lens: torch.Tensor,
    task_map: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
    max_tasks: int,
    sched_ints: int,
) -> None:
    assign_pure_triton_task_map_kernel[(1,)](
        kv_lens,
        task_map,
        B=kv_lens.numel(),
        H_KV=num_head_kv,
        NUM_CTAS=num_ctas,
        MIN_PROCESS_LEN=min_process_len,
        MAX_TASKS=max_tasks,
        SCHED_INTS=sched_ints,
        num_warps=1,
        num_stages=1,
    )


_scheduler_mtp1__TASK_STRIDE = 12
_scheduler_mtp1__TASK_SLOTS = 2
_scheduler_mtp1__TILE_N = 64
_scheduler_mtp1__DIRECT_MODE = 0
_scheduler_mtp1__GROUP_MODE = 1
_scheduler_mtp1__DUMMY_MODE = 2
_scheduler_mtp1__META_NUM_CLUSTERS = 0
_scheduler_mtp1__META_PHYSICAL_CTAS = 1
_scheduler_mtp1__META_REDUCTION_CLUSTERS = 2
_scheduler_mtp1__META_DIRECT_TASKS = 3
_scheduler_mtp1__META_FINE_CHUNKS_MAX = 4
_scheduler_mtp1__META_EFFECTIVE_CHUNKS_MAX = 5
_scheduler_mtp1__META_COMPUTE_TASKS = 6
_scheduler_mtp1__META_DUMMY_TASKS = 7
_scheduler_mtp1__META_SCHED_INTS = 8
_scheduler_mtp1__META_INVALID_LENGTHS = 9
_scheduler_mtp1__META_SIZE = 10
_scheduler_mtp1___TASK_STRIDE_JIT = tl.constexpr(_scheduler_mtp1__TASK_STRIDE)
_scheduler_mtp1___TASK_SLOTS_JIT = tl.constexpr(_scheduler_mtp1__TASK_SLOTS)
_scheduler_mtp1___TILE_N_JIT = tl.constexpr(_scheduler_mtp1__TILE_N)
_scheduler_mtp1___DIRECT_MODE_JIT = tl.constexpr(_scheduler_mtp1__DIRECT_MODE)
_scheduler_mtp1___GROUP_MODE_JIT = tl.constexpr(_scheduler_mtp1__GROUP_MODE)
_scheduler_mtp1___DUMMY_MODE_JIT = tl.constexpr(_scheduler_mtp1__DUMMY_MODE)
_scheduler_mtp1___META_NUM_CLUSTERS_JIT = tl.constexpr(_scheduler_mtp1__META_NUM_CLUSTERS)
_scheduler_mtp1___META_PHYSICAL_CTAS_JIT = tl.constexpr(_scheduler_mtp1__META_PHYSICAL_CTAS)
_scheduler_mtp1___META_REDUCTION_CLUSTERS_JIT = tl.constexpr(_scheduler_mtp1__META_REDUCTION_CLUSTERS)
_scheduler_mtp1___META_DIRECT_TASKS_JIT = tl.constexpr(_scheduler_mtp1__META_DIRECT_TASKS)
_scheduler_mtp1___META_FINE_CHUNKS_MAX_JIT = tl.constexpr(_scheduler_mtp1__META_FINE_CHUNKS_MAX)
_scheduler_mtp1___META_EFFECTIVE_CHUNKS_MAX_JIT = tl.constexpr(_scheduler_mtp1__META_EFFECTIVE_CHUNKS_MAX)
_scheduler_mtp1___META_COMPUTE_TASKS_JIT = tl.constexpr(_scheduler_mtp1__META_COMPUTE_TASKS)
_scheduler_mtp1___META_DUMMY_TASKS_JIT = tl.constexpr(_scheduler_mtp1__META_DUMMY_TASKS)
_scheduler_mtp1___META_SCHED_INTS_JIT = tl.constexpr(_scheduler_mtp1__META_SCHED_INTS)
_scheduler_mtp1___META_INVALID_LENGTHS_JIT = tl.constexpr(_scheduler_mtp1__META_INVALID_LENGTHS)


@triton.jit
def _scheduler_mtp1__assign_cluster_task_prefix_kernel(
    SEQLENS_KV,
    OFFSETS,
    META,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Compute per-sequence offsets and exact task-map metadata on device."""
    batch = tl.arange(0, BLOCK_SEQ)
    num_sequences = B * H_KV
    valid = batch < B
    total_len = tl.load(SEQLENS_KV + batch, mask=valid, other=0).to(tl.int32)
    positive = valid & (total_len >= NUM_SEQ_Q)
    num_chunks = tl.where(positive, (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS, 0)
    direct = (positive & (num_chunks == 1)).to(tl.int32)
    reduction_clusters = tl.where(positive & (num_chunks > 1), (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE, 0).to(
        tl.int32
    )
    effective_chunks = tl.where(num_chunks == 1, 1, reduction_clusters)
    reduction_offsets, reduction_per_head = tle.cumsum(reduction_clusters, axis=0, reverse=False)
    direct_offsets, direct_per_head = tle.cumsum(direct, axis=0, reverse=False)
    for hkv in range(H_KV):
        seq = hkv * B + batch
        tl.store(OFFSETS + seq * 2 + 0, hkv * reduction_per_head + reduction_offsets, mask=valid)
        tl.store(OFFSETS + seq * 2 + 1, hkv * direct_per_head + direct_offsets, mask=valid)
    reduction_total = reduction_per_head * H_KV
    direct_total = direct_per_head * H_KV
    direct_clusters = (direct_total + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    num_clusters = reduction_total + direct_clusters
    physical_ctas = num_clusters * CLUSTER_SIZE
    num_chunks_base = (_scheduler_mtp1___TASK_SLOTS_JIT * physical_ctas + 1) * _scheduler_mtp1___TASK_STRIDE_JIT
    chunk_pad_ints = (
        (num_sequences + _scheduler_mtp1___TASK_STRIDE_JIT - 1)
        // _scheduler_mtp1___TASK_STRIDE_JIT
        * _scheduler_mtp1___TASK_STRIDE_JIT
    )
    sched_ints = num_chunks_base + chunk_pad_ints
    fine_chunks_max = tl.max(num_chunks, axis=0)
    effective_chunks_max = tl.max(effective_chunks, axis=0)
    compute_tasks = tl.sum(num_chunks, axis=0) * H_KV
    dummy_tasks = tl.sum(tl.where(num_chunks > 1, reduction_clusters * CLUSTER_SIZE - num_chunks, 0), axis=0) * H_KV
    invalid_lengths = tl.sum((valid & (total_len < NUM_SEQ_Q)).to(tl.int32), axis=0) * H_KV
    tl.store(META + _scheduler_mtp1___META_NUM_CLUSTERS_JIT, num_clusters)
    tl.store(META + _scheduler_mtp1___META_PHYSICAL_CTAS_JIT, physical_ctas)
    tl.store(META + _scheduler_mtp1___META_REDUCTION_CLUSTERS_JIT, reduction_total)
    tl.store(META + _scheduler_mtp1___META_DIRECT_TASKS_JIT, direct_total)
    tl.store(META + _scheduler_mtp1___META_FINE_CHUNKS_MAX_JIT, fine_chunks_max)
    tl.store(META + _scheduler_mtp1___META_EFFECTIVE_CHUNKS_MAX_JIT, effective_chunks_max)
    tl.store(META + _scheduler_mtp1___META_COMPUTE_TASKS_JIT, compute_tasks)
    tl.store(META + _scheduler_mtp1___META_DUMMY_TASKS_JIT, dummy_tasks)
    tl.store(META + _scheduler_mtp1___META_SCHED_INTS_JIT, sched_ints)
    tl.store(META + _scheduler_mtp1___META_INVALID_LENGTHS_JIT, invalid_lengths)
    tl.store(TASK_MAP + 0, CHUNK_TOKENS // _scheduler_mtp1___TILE_N_JIT + 1)
    tl.store(TASK_MAP + 1, physical_ctas)
    tl.store(TASK_MAP + 2, H_KV)
    tl.store(TASK_MAP + 3, B)
    tl.store(TASK_MAP + 4, sched_ints * 4)


@triton.jit
def _scheduler_mtp1___store_task_record(
    TASK_MAP,
    cta,
    hkv,
    batch,
    chunk,
    seq_start,
    seq_len,
    seq_kvcache,
    num_tile_kv,
    num_tile_full,
    is_causal,
    mode,
    group_chunk,
    group_count,
    mask,
):
    task_base = (cta * _scheduler_mtp1___TASK_SLOTS_JIT + 1) * _scheduler_mtp1___TASK_STRIDE_JIT
    tl.store(TASK_MAP + task_base + 0, hkv, mask=mask)
    tl.store(TASK_MAP + task_base + 1, batch, mask=mask)
    tl.store(TASK_MAP + task_base + 2, chunk, mask=mask)
    tl.store(TASK_MAP + task_base + 3, seq_start, mask=mask)
    tl.store(TASK_MAP + task_base + 4, seq_len, mask=mask)
    tl.store(TASK_MAP + task_base + 5, seq_kvcache, mask=mask)
    tl.store(TASK_MAP + task_base + 6, num_tile_kv, mask=mask)
    tl.store(TASK_MAP + task_base + 7, num_tile_full, mask=mask)
    tl.store(TASK_MAP + task_base + 8, is_causal, mask=mask)
    tl.store(TASK_MAP + task_base + 9, mode, mask=mask)
    tl.store(TASK_MAP + task_base + 10, group_chunk, mask=mask)
    tl.store(TASK_MAP + task_base + 11, group_count, mask=mask)
    sentinel_base = (cta * _scheduler_mtp1___TASK_SLOTS_JIT + 2) * _scheduler_mtp1___TASK_STRIDE_JIT
    tl.store(TASK_MAP + sentinel_base + 0, -1, mask=mask)
    tl.store(TASK_MAP + sentinel_base + 1, -1, mask=mask)


@triton.jit
def _scheduler_mtp1__assign_cluster_task_records_compact_kernel(
    SEQLENS_KV,
    OFFSETS,
    META,
    TASK_MAP,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
):
    """Write only this sequence's real chunks and required cluster padding.

    A vectorized records kernel would use ``BLOCK_CHUNKS`` equal to the
    longest sequence's chunk count for every sequence. This compact kernel
    is deliberately scalar-loop based: short sequences in a long-tail batch
    write O(their own chunks) task records rather than O(max chunks in batch).
    """
    seq_id = tl.program_id(0)
    hkv = seq_id // B
    batch = seq_id - hkv * B
    total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
    if total_len < NUM_SEQ_Q:
        return
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    reduction_offset = tl.load(OFFSETS + seq_id * 2 + 0)
    direct_index = tl.load(OFFSETS + seq_id * 2 + 1)
    reduction_total = tl.load(META + _scheduler_mtp1___META_REDUCTION_CLUSTERS_JIT)
    direct_total = tl.load(META + _scheduler_mtp1___META_DIRECT_TASKS_JIT)
    physical_ctas = tl.load(META + _scheduler_mtp1___META_PHYSICAL_CTAS_JIT)
    num_chunks_base = (_scheduler_mtp1___TASK_SLOTS_JIT * physical_ctas + 1) * _scheduler_mtp1___TASK_STRIDE_JIT
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    effective_chunks = tl.where(num_chunks == 1, 1, group_count)
    tl.store(TASK_MAP + num_chunks_base + seq_id, effective_chunks)
    if num_chunks <= 0:
        return
    if num_chunks == 1:
        cluster = reduction_total + direct_index // CLUSTER_SIZE
        rank = direct_index % CLUSTER_SIZE
        cta = cluster * CLUSTER_SIZE + rank
        seq_kvcache = total_len - NUM_SEQ_Q
        _scheduler_mtp1___store_task_record(
            TASK_MAP,
            cta,
            hkv,
            batch,
            0,
            0,
            total_len,
            seq_kvcache,
            (total_len + _scheduler_mtp1___TILE_N_JIT - 1) // _scheduler_mtp1___TILE_N_JIT,
            seq_kvcache // _scheduler_mtp1___TILE_N_JIT,
            1,
            _scheduler_mtp1___DIRECT_MODE_JIT,
            0,
            1,
            True,
        )
        if direct_index == direct_total - 1:
            clear_rank = tl.arange(0, CLUSTER_SIZE)[:, None]
            clear_field = tl.arange(0, 16)[None, :]
            clear_cta = cluster * CLUSTER_SIZE + clear_rank
            clear_task_base = (clear_cta * _scheduler_mtp1___TASK_SLOTS_JIT + 1) * _scheduler_mtp1___TASK_STRIDE_JIT
            clear_mask = (clear_rank > rank) & (clear_field < _scheduler_mtp1___TASK_STRIDE_JIT)
            tl.store(TASK_MAP + clear_task_base + clear_field, -1, mask=clear_mask)
            clear_sentinel_base = (clear_cta * _scheduler_mtp1___TASK_SLOTS_JIT + 2) * _scheduler_mtp1___TASK_STRIDE_JIT
            sentinel_mask = (clear_rank > rank) & (clear_field < 2)
            tl.store(TASK_MAP + clear_sentinel_base + clear_field, -1, mask=sentinel_mask)
        return
    covered_chunks = group_count * CLUSTER_SIZE
    chunk = 0
    while chunk < covered_chunks:
        real = chunk < num_chunks
        group_chunk = chunk // CLUSTER_SIZE
        rank = chunk % CLUSTER_SIZE
        cluster = reduction_offset + group_chunk
        cta = cluster * CLUSTER_SIZE + rank
        logical_start = chunk * CHUNK_TOKENS
        seq_start = tl.where(real, logical_start, 0)
        seq_len = tl.where(real, tl.minimum(CHUNK_TOKENS, total_len - logical_start), 0)
        is_last = real & (chunk == num_chunks - 1)
        seq_kvcache = tl.where(is_last, seq_len - NUM_SEQ_Q, seq_len)
        num_tile_kv = (seq_len + _scheduler_mtp1___TILE_N_JIT - 1) // _scheduler_mtp1___TILE_N_JIT
        num_tile_full = tl.where(
            is_last, tl.maximum(seq_kvcache, 0) // _scheduler_mtp1___TILE_N_JIT, seq_len // _scheduler_mtp1___TILE_N_JIT
        )
        mode = tl.where(real, _scheduler_mtp1___GROUP_MODE_JIT, _scheduler_mtp1___DUMMY_MODE_JIT)
        _scheduler_mtp1___store_task_record(
            TASK_MAP,
            cta,
            hkv,
            batch,
            chunk,
            seq_start,
            seq_len,
            seq_kvcache,
            num_tile_kv,
            num_tile_full,
            is_last.to(tl.int32),
            mode,
            group_chunk,
            group_count,
            True,
        )
        chunk += 1


@triton.jit
def _scheduler_mtp1__refresh_cluster_task_tail_kernel(
    SEQLENS_KV,
    OFFSETS,
    META,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Refresh length-dependent tail fields while cluster topology is stable."""
    seq_id = tl.arange(0, BLOCK_SEQ)
    valid = seq_id < B * H_KV
    hkv = seq_id // B
    batch = seq_id - hkv * B
    total_len = tl.load(SEQLENS_KV + batch, mask=valid, other=1).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    reduction_offset = tl.load(OFFSETS + seq_id * 2 + 0, mask=valid, other=0)
    direct_index = tl.load(OFFSETS + seq_id * 2 + 1, mask=valid, other=0)
    reduction_total = tl.load(META + _scheduler_mtp1___META_REDUCTION_CLUSTERS_JIT)
    direct = num_chunks == 1
    tail_chunk = num_chunks - 1
    direct_cluster = reduction_total + direct_index // CLUSTER_SIZE
    direct_rank = direct_index % CLUSTER_SIZE
    reduction_cluster = reduction_offset + tail_chunk // CLUSTER_SIZE
    reduction_rank = tail_chunk % CLUSTER_SIZE
    cluster = tl.where(direct, direct_cluster, reduction_cluster)
    rank = tl.where(direct, direct_rank, reduction_rank)
    cta = cluster * CLUSTER_SIZE + rank
    task_base = (cta * _scheduler_mtp1___TASK_SLOTS_JIT + 1) * _scheduler_mtp1___TASK_STRIDE_JIT
    seq_start = (num_chunks - 1) * CHUNK_TOKENS
    seq_len = total_len - seq_start
    seq_kvcache = seq_len - NUM_SEQ_Q
    tl.store(TASK_MAP + task_base + 3, seq_start, mask=valid)
    tl.store(TASK_MAP + task_base + 4, seq_len, mask=valid)
    tl.store(TASK_MAP + task_base + 5, seq_kvcache, mask=valid)
    tl.store(
        TASK_MAP + task_base + 6,
        (seq_len + _scheduler_mtp1___TILE_N_JIT - 1) // _scheduler_mtp1___TILE_N_JIT,
        mask=valid,
    )
    tl.store(TASK_MAP + task_base + 7, tl.maximum(seq_kvcache, 0) // _scheduler_mtp1___TILE_N_JIT, mask=valid)
    tl.store(TASK_MAP + task_base + 8, 1, mask=valid)


@dataclass
class _scheduler_mtp1__DecodeTaskSchedule:
    task_workspace: torch.Tensor
    task_map: torch.Tensor
    offsets: torch.Tensor
    meta: torch.Tensor
    cluster_size: int
    chunk_tokens: int
    block_seq: int
    block_chunks: int
    capacity_clusters: int
    capacity_ints: int
    num_clusters: int = 0
    physical_ctas: int = 0
    sched_ints: int = 0
    partial_slots: int = 0
    stats: dict | None = None


def _scheduler_mtp1___capacity(
    *, num_sequences: int, max_seq_kv: int, cluster_size: int, chunk_tokens: int
) -> tuple[int, int, int]:
    max_chunks = max(1, (max_seq_kv + chunk_tokens - 1) // chunk_tokens)
    max_groups = (max_chunks + cluster_size - 1) // cluster_size
    max_reduction_clusters = num_sequences * max_groups if max_chunks > 1 else 0
    max_direct_clusters = (num_sequences + cluster_size - 1) // cluster_size
    capacity_clusters = max_reduction_clusters + max_direct_clusters
    capacity_ctas = max(capacity_clusters * cluster_size, cluster_size)
    chunk_pad_ints = (
        (num_sequences + _scheduler_mtp1__TASK_STRIDE - 1)
        // _scheduler_mtp1__TASK_STRIDE
        * _scheduler_mtp1__TASK_STRIDE
    )
    capacity_ints = (_scheduler_mtp1__TASK_SLOTS * capacity_ctas + 1) * _scheduler_mtp1__TASK_STRIDE + chunk_pad_ints
    block_chunks = triton.next_power_of_2(max(max_chunks, cluster_size))
    return (capacity_clusters, capacity_ints, block_chunks)


def _scheduler_mtp1__allocate_cluster_task_map(
    kv_lens: torch.Tensor, *, num_head_kv: int, max_seq_kv: int, cluster_size: int, chunk_tokens: int
) -> _scheduler_mtp1__DecodeTaskSchedule:
    """Allocate capacity and populate a cluster task map entirely on GPU."""
    if not kv_lens.is_cuda:
        raise ValueError("cluster GPU assignment requires CUDA kv_lens")
    if cluster_size not in (2, 4, 8):
        raise ValueError(f"cluster_size must be 2, 4, or 8, got {cluster_size}")
    if chunk_tokens < _scheduler_mtp1__TILE_N or chunk_tokens % _scheduler_mtp1__TILE_N:
        raise ValueError(f"chunk_tokens must be a positive multiple of {_scheduler_mtp1__TILE_N}, got {chunk_tokens}")
    num_sequences = kv_lens.numel() * num_head_kv
    block_seq = triton.next_power_of_2(kv_lens.numel())
    if block_seq > 1024:
        raise ValueError(f"cluster GPU assign currently supports B <= 1024, got {kv_lens.numel()}")
    capacity_clusters, capacity_ints, block_chunks = _scheduler_mtp1___capacity(
        num_sequences=num_sequences, max_seq_kv=max_seq_kv, cluster_size=cluster_size, chunk_tokens=chunk_tokens
    )
    task_map = torch.full((capacity_ints,), -1, dtype=torch.int32, device=kv_lens.device)
    assignment = _scheduler_mtp1__DecodeTaskSchedule(
        task_workspace=task_map.view(torch.int8),
        task_map=task_map,
        offsets=torch.empty((num_sequences, 2), dtype=torch.int32, device=kv_lens.device),
        meta=torch.empty((_scheduler_mtp1__META_SIZE,), dtype=torch.int32, device=kv_lens.device),
        cluster_size=cluster_size,
        chunk_tokens=chunk_tokens,
        block_seq=block_seq,
        block_chunks=block_chunks,
        capacity_clusters=capacity_clusters,
        capacity_ints=capacity_ints,
    )
    _scheduler_mtp1__launch_cluster_task_map_assign(
        kv_lens, assignment, num_head_kv=num_head_kv, refresh_host_metadata=True
    )
    return assignment


def _scheduler_mtp1__launch_cluster_task_map_assign(
    kv_lens: torch.Tensor,
    assignment: _scheduler_mtp1__DecodeTaskSchedule,
    *,
    num_head_kv: int,
    refresh_host_metadata: bool,
) -> None:
    """Regenerate an allocated map; fixed-topology calls need no host sync."""
    batch = kv_lens.numel()
    num_sequences = batch * num_head_kv
    if assignment.offsets.numel() != num_sequences * 2:
        raise ValueError("kv_lens/H_KV shape differs from the allocated cluster task map")
    prefix_warps = max(1, min(32, assignment.block_seq // 32))
    _scheduler_mtp1__assign_cluster_task_prefix_kernel[1,](
        kv_lens,
        assignment.offsets,
        assignment.meta,
        assignment.task_map,
        B=batch,
        H_KV=num_head_kv,
        NUM_SEQ_Q=1,
        CLUSTER_SIZE=assignment.cluster_size,
        CHUNK_TOKENS=assignment.chunk_tokens,
        BLOCK_SEQ=assignment.block_seq,
        num_warps=prefix_warps,
        num_stages=1,
    )
    record_warps = 1
    records_kernel = _scheduler_mtp1__assign_cluster_task_records_compact_kernel
    record_args = {
        "B": batch,
        "NUM_SEQ_Q": 1,
        "CLUSTER_SIZE": assignment.cluster_size,
        "CHUNK_TOKENS": assignment.chunk_tokens,
        "num_warps": record_warps,
        "num_stages": 1,
    }
    records_kernel[num_sequences,](kv_lens, assignment.offsets, assignment.meta, assignment.task_map, **record_args)
    if not refresh_host_metadata:
        return
    values = assignment.meta.detach().cpu().to(torch.int64).tolist()
    invalid_lengths = values[_scheduler_mtp1__META_INVALID_LENGTHS]
    if invalid_lengths:
        raise ValueError(f"MTP=1 cluster GPU assign found {invalid_lengths} total KV lengths below 1")
    num_clusters = values[_scheduler_mtp1__META_NUM_CLUSTERS]
    sched_ints = values[_scheduler_mtp1__META_SCHED_INTS]
    if num_clusters > assignment.capacity_clusters or sched_ints > assignment.capacity_ints:
        raise RuntimeError(
            f"cluster task-map capacity was underestimated: clusters={num_clusters}/{assignment.capacity_clusters}, ints={sched_ints}/{assignment.capacity_ints}"
        )
    assignment.num_clusters = num_clusters
    assignment.physical_ctas = values[_scheduler_mtp1__META_PHYSICAL_CTAS]
    assignment.sched_ints = sched_ints
    assignment.partial_slots = max(values[_scheduler_mtp1__META_EFFECTIVE_CHUNKS_MAX], 1)
    assignment.stats = {
        "cluster_size": assignment.cluster_size,
        "num_clusters": num_clusters,
        "physical_ctas": values[_scheduler_mtp1__META_PHYSICAL_CTAS],
        "compute_tasks": values[_scheduler_mtp1__META_COMPUTE_TASKS],
        "fine_chunks_max": values[_scheduler_mtp1__META_FINE_CHUNKS_MAX],
        "effective_chunks_max": values[_scheduler_mtp1__META_EFFECTIVE_CHUNKS_MAX],
        "direct_tasks": values[_scheduler_mtp1__META_DIRECT_TASKS],
        "dummy_tasks": values[_scheduler_mtp1__META_DUMMY_TASKS],
    }


def _scheduler_mtp1__launch_cluster_task_tail_refresh(
    kv_lens: torch.Tensor, assignment: _scheduler_mtp1__DecodeTaskSchedule, *, num_head_kv: int
) -> None:
    """Update tail records without rebuilding unchanged 512-token topology.

    The caller must run ``launch_cluster_task_map_assign`` with refreshed host
    metadata whenever any ``ceil(kv_len / chunk_tokens)`` value changes.
    """
    batch = kv_lens.numel()
    num_sequences = batch * num_head_kv
    if assignment.offsets.numel() != num_sequences * 2:
        raise ValueError("kv_lens/H_KV shape differs from the allocated cluster task map")
    block_seq = triton.next_power_of_2(num_sequences)
    num_warps = max(1, min(8, block_seq // 32))
    _scheduler_mtp1__refresh_cluster_task_tail_kernel[1,](
        kv_lens,
        assignment.offsets,
        assignment.meta,
        assignment.task_map,
        B=batch,
        H_KV=num_head_kv,
        NUM_SEQ_Q=1,
        CLUSTER_SIZE=assignment.cluster_size,
        CHUNK_TOKENS=assignment.chunk_tokens,
        BLOCK_SEQ=block_seq,
        num_warps=num_warps,
        num_stages=1,
    )


_scheduler_mtp24__TASK_STRIDE = 12
_scheduler_mtp24__TASK_SLOTS = 2
_scheduler_mtp24__TILE_N = 64
_scheduler_mtp24__DIRECT_MODE = 0
_scheduler_mtp24__GROUP_MODE = 1
_scheduler_mtp24__DUMMY_MODE = 2
_scheduler_mtp24__SUBGROUP2_MODE = 3
_scheduler_mtp24__META_NUM_CLUSTERS = 0
_scheduler_mtp24__META_PHYSICAL_CTAS = 1
_scheduler_mtp24__META_REDUCTION_CLUSTERS = 2
_scheduler_mtp24__META_DIRECT_TASKS = 3
_scheduler_mtp24__META_FINE_CHUNKS_MAX = 4
_scheduler_mtp24__META_EFFECTIVE_CHUNKS_MAX = 5
_scheduler_mtp24__META_COMPUTE_TASKS = 6
_scheduler_mtp24__META_DUMMY_TASKS = 7
_scheduler_mtp24__META_SCHED_INTS = 8
_scheduler_mtp24__META_INVALID_LENGTHS = 9
_scheduler_mtp24__META_LONG_REDUCTION_CLUSTERS = 10
_scheduler_mtp24__META_SUBGROUP_TASKS = 11
_scheduler_mtp24__META_SIZE = 12
_scheduler_mtp24___TASK_STRIDE_JIT = tl.constexpr(_scheduler_mtp24__TASK_STRIDE)
_scheduler_mtp24___TASK_SLOTS_JIT = tl.constexpr(_scheduler_mtp24__TASK_SLOTS)
_scheduler_mtp24___TILE_N_JIT = tl.constexpr(_scheduler_mtp24__TILE_N)
_scheduler_mtp24___DIRECT_MODE_JIT = tl.constexpr(_scheduler_mtp24__DIRECT_MODE)
_scheduler_mtp24___GROUP_MODE_JIT = tl.constexpr(_scheduler_mtp24__GROUP_MODE)
_scheduler_mtp24___DUMMY_MODE_JIT = tl.constexpr(_scheduler_mtp24__DUMMY_MODE)
_scheduler_mtp24___SUBGROUP2_MODE_JIT = tl.constexpr(_scheduler_mtp24__SUBGROUP2_MODE)
_scheduler_mtp24___META_NUM_CLUSTERS_JIT = tl.constexpr(_scheduler_mtp24__META_NUM_CLUSTERS)
_scheduler_mtp24___META_PHYSICAL_CTAS_JIT = tl.constexpr(_scheduler_mtp24__META_PHYSICAL_CTAS)
_scheduler_mtp24___META_REDUCTION_CLUSTERS_JIT = tl.constexpr(_scheduler_mtp24__META_REDUCTION_CLUSTERS)
_scheduler_mtp24___META_DIRECT_TASKS_JIT = tl.constexpr(_scheduler_mtp24__META_DIRECT_TASKS)
_scheduler_mtp24___META_FINE_CHUNKS_MAX_JIT = tl.constexpr(_scheduler_mtp24__META_FINE_CHUNKS_MAX)
_scheduler_mtp24___META_EFFECTIVE_CHUNKS_MAX_JIT = tl.constexpr(_scheduler_mtp24__META_EFFECTIVE_CHUNKS_MAX)
_scheduler_mtp24___META_COMPUTE_TASKS_JIT = tl.constexpr(_scheduler_mtp24__META_COMPUTE_TASKS)
_scheduler_mtp24___META_DUMMY_TASKS_JIT = tl.constexpr(_scheduler_mtp24__META_DUMMY_TASKS)
_scheduler_mtp24___META_SCHED_INTS_JIT = tl.constexpr(_scheduler_mtp24__META_SCHED_INTS)
_scheduler_mtp24___META_INVALID_LENGTHS_JIT = tl.constexpr(_scheduler_mtp24__META_INVALID_LENGTHS)
_scheduler_mtp24___META_LONG_REDUCTION_CLUSTERS_JIT = tl.constexpr(_scheduler_mtp24__META_LONG_REDUCTION_CLUSTERS)
_scheduler_mtp24___META_SUBGROUP_TASKS_JIT = tl.constexpr(_scheduler_mtp24__META_SUBGROUP_TASKS)


@triton.jit
def _scheduler_mtp24__assign_cluster_task_prefix_kernel(
    SEQLENS_KV,
    OFFSETS,
    META,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    DIRECT_THRESHOLD: tl.constexpr,
    SHORT_THRESHOLD: tl.constexpr,
    SHORT_CHUNK_TOKENS: tl.constexpr,
    SUBGROUP2_THRESHOLD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Compute per-sequence offsets and exact task-map metadata on device."""
    batch = tl.arange(0, BLOCK_SEQ)
    num_sequences = B * H_KV
    valid = batch < B
    total_len = tl.load(SEQLENS_KV + batch, mask=valid, other=0).to(tl.int32)
    positive = valid & (total_len >= NUM_SEQ_Q)
    sequence_chunk = tl.where((SHORT_THRESHOLD > 0) & (total_len <= SHORT_THRESHOLD), SHORT_CHUNK_TOKENS, CHUNK_TOKENS)
    num_chunks = tl.where(positive, (total_len + sequence_chunk - 1) // sequence_chunk, 0)
    subgroup_mask = positive & (CLUSTER_SIZE == 4) & (SUBGROUP2_THRESHOLD > 0) & (total_len <= SUBGROUP2_THRESHOLD)
    direct_mask = positive & ~subgroup_mask & ((num_chunks == 1) | (total_len <= DIRECT_THRESHOLD))
    direct = direct_mask.to(tl.int32)
    long_reduction_clusters = tl.where(
        positive & ~direct_mask & ~subgroup_mask, (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE, 0
    ).to(tl.int32)
    subgroup = subgroup_mask.to(tl.int32)
    effective_chunks = tl.where(direct_mask | subgroup_mask, 1, long_reduction_clusters)
    reduction_offsets, reduction_per_head = tle.cumsum(long_reduction_clusters, axis=0, reverse=False)
    subgroup_offsets, subgroup_per_head = tle.cumsum(subgroup, axis=0, reverse=False)
    direct_offsets, direct_per_head = tle.cumsum(direct, axis=0, reverse=False)
    for hkv in range(H_KV):
        seq = hkv * B + batch
        long_offset = hkv * reduction_per_head + reduction_offsets
        subgroup_offset = hkv * subgroup_per_head + subgroup_offsets
        tl.store(OFFSETS + seq * 2 + 0, tl.where(subgroup_mask, subgroup_offset, long_offset), mask=valid)
        tl.store(OFFSETS + seq * 2 + 1, hkv * direct_per_head + direct_offsets, mask=valid)
    long_reduction_total = reduction_per_head * H_KV
    subgroup_total = subgroup_per_head * H_KV
    subgroup_clusters = (subgroup_total + 1) // 2
    reduction_total = long_reduction_total + subgroup_clusters
    direct_total = direct_per_head * H_KV
    direct_clusters = (direct_total + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    num_clusters = reduction_total + direct_clusters
    physical_ctas = num_clusters * CLUSTER_SIZE
    num_chunks_base = (_scheduler_mtp24___TASK_SLOTS_JIT * physical_ctas + 1) * _scheduler_mtp24___TASK_STRIDE_JIT
    chunk_pad_ints = (
        (num_sequences + _scheduler_mtp24___TASK_STRIDE_JIT - 1)
        // _scheduler_mtp24___TASK_STRIDE_JIT
        * _scheduler_mtp24___TASK_STRIDE_JIT
    )
    sched_ints = num_chunks_base + chunk_pad_ints
    fine_chunks_max = tl.max(num_chunks, axis=0)
    effective_chunks_max = tl.max(effective_chunks, axis=0)
    compute_tasks = tl.sum(tl.where(direct_mask, 1, tl.where(subgroup_mask, 2, num_chunks)), axis=0) * H_KV
    long_dummy = (
        tl.sum(
            tl.where(positive & ~direct_mask & ~subgroup_mask, long_reduction_clusters * CLUSTER_SIZE - num_chunks, 0),
            axis=0,
        )
        * H_KV
    )
    dummy_tasks = long_dummy + (subgroup_total & 1) * 2
    invalid_lengths = tl.sum((valid & (total_len < NUM_SEQ_Q)).to(tl.int32), axis=0) * H_KV
    tl.store(META + _scheduler_mtp24___META_NUM_CLUSTERS_JIT, num_clusters)
    tl.store(META + _scheduler_mtp24___META_PHYSICAL_CTAS_JIT, physical_ctas)
    tl.store(META + _scheduler_mtp24___META_REDUCTION_CLUSTERS_JIT, reduction_total)
    tl.store(META + _scheduler_mtp24___META_DIRECT_TASKS_JIT, direct_total)
    tl.store(META + _scheduler_mtp24___META_FINE_CHUNKS_MAX_JIT, fine_chunks_max)
    tl.store(META + _scheduler_mtp24___META_EFFECTIVE_CHUNKS_MAX_JIT, effective_chunks_max)
    tl.store(META + _scheduler_mtp24___META_COMPUTE_TASKS_JIT, compute_tasks)
    tl.store(META + _scheduler_mtp24___META_DUMMY_TASKS_JIT, dummy_tasks)
    tl.store(META + _scheduler_mtp24___META_SCHED_INTS_JIT, sched_ints)
    tl.store(META + _scheduler_mtp24___META_INVALID_LENGTHS_JIT, invalid_lengths)
    tl.store(META + _scheduler_mtp24___META_LONG_REDUCTION_CLUSTERS_JIT, long_reduction_total)
    tl.store(META + _scheduler_mtp24___META_SUBGROUP_TASKS_JIT, subgroup_total)
    tl.store(TASK_MAP + 0, CHUNK_TOKENS // _scheduler_mtp24___TILE_N_JIT + 1)
    tl.store(TASK_MAP + 1, physical_ctas)
    tl.store(TASK_MAP + 2, H_KV)
    tl.store(TASK_MAP + 3, B)
    tl.store(TASK_MAP + 4, sched_ints * 4)


@triton.jit
def _scheduler_mtp24___store_task_record(
    TASK_MAP,
    cta,
    hkv,
    batch,
    chunk,
    seq_start,
    seq_len,
    seq_kvcache,
    num_tile_kv,
    num_tile_full,
    is_causal,
    mode,
    group_chunk,
    group_count,
    mask,
):
    task_base = (cta * _scheduler_mtp24___TASK_SLOTS_JIT + 1) * _scheduler_mtp24___TASK_STRIDE_JIT
    tl.store(TASK_MAP + task_base + 0, hkv, mask=mask)
    tl.store(TASK_MAP + task_base + 1, batch, mask=mask)
    tl.store(TASK_MAP + task_base + 2, chunk, mask=mask)
    tl.store(TASK_MAP + task_base + 3, seq_start, mask=mask)
    tl.store(TASK_MAP + task_base + 4, seq_len, mask=mask)
    tl.store(TASK_MAP + task_base + 5, seq_kvcache, mask=mask)
    tl.store(TASK_MAP + task_base + 6, num_tile_kv, mask=mask)
    tl.store(TASK_MAP + task_base + 7, num_tile_full, mask=mask)
    tl.store(TASK_MAP + task_base + 8, is_causal, mask=mask)
    tl.store(TASK_MAP + task_base + 9, mode, mask=mask)
    tl.store(TASK_MAP + task_base + 10, group_chunk, mask=mask)
    tl.store(TASK_MAP + task_base + 11, group_count, mask=mask)
    sentinel_base = (cta * _scheduler_mtp24___TASK_SLOTS_JIT + 2) * _scheduler_mtp24___TASK_STRIDE_JIT
    tl.store(TASK_MAP + sentinel_base + 0, -1, mask=mask)
    tl.store(TASK_MAP + sentinel_base + 1, -1, mask=mask)


@triton.jit
def _scheduler_mtp24__assign_cluster_task_records_compact_kernel(
    SEQLENS_KV,
    OFFSETS,
    META,
    TASK_MAP,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    DIRECT_THRESHOLD: tl.constexpr,
    SHORT_THRESHOLD: tl.constexpr,
    SHORT_CHUNK_TOKENS: tl.constexpr,
    SUBGROUP2_THRESHOLD: tl.constexpr,
):
    """Write only this sequence's real chunks and required cluster padding.

    A vectorized records kernel would use ``BLOCK_CHUNKS`` equal to the
    longest sequence's chunk count for every sequence. This compact kernel
    is deliberately scalar-loop based: short sequences in a long-tail batch
    write O(their own chunks) task records rather than O(max chunks in batch).
    """
    seq_id = tl.program_id(0)
    hkv = seq_id // B
    batch = seq_id - hkv * B
    total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
    if total_len < NUM_SEQ_Q:
        return
    sequence_chunk = tl.where((SHORT_THRESHOLD > 0) & (total_len <= SHORT_THRESHOLD), SHORT_CHUNK_TOKENS, CHUNK_TOKENS)
    num_chunks = (total_len + sequence_chunk - 1) // sequence_chunk
    subgroup = (CLUSTER_SIZE == 4) & (SUBGROUP2_THRESHOLD > 0) & (total_len <= SUBGROUP2_THRESHOLD)
    direct = ~subgroup & ((num_chunks == 1) | (total_len <= DIRECT_THRESHOLD))
    reduction_offset = tl.load(OFFSETS + seq_id * 2 + 0)
    direct_index = tl.load(OFFSETS + seq_id * 2 + 1)
    reduction_total = tl.load(META + _scheduler_mtp24___META_REDUCTION_CLUSTERS_JIT)
    direct_total = tl.load(META + _scheduler_mtp24___META_DIRECT_TASKS_JIT)
    long_reduction_total = tl.load(META + _scheduler_mtp24___META_LONG_REDUCTION_CLUSTERS_JIT)
    subgroup_total = tl.load(META + _scheduler_mtp24___META_SUBGROUP_TASKS_JIT)
    physical_ctas = tl.load(META + _scheduler_mtp24___META_PHYSICAL_CTAS_JIT)
    num_chunks_base = (_scheduler_mtp24___TASK_SLOTS_JIT * physical_ctas + 1) * _scheduler_mtp24___TASK_STRIDE_JIT
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    effective_chunks = tl.where(direct, 1, group_count)
    tl.store(TASK_MAP + num_chunks_base + seq_id, effective_chunks)
    if num_chunks <= 0:
        return
    if subgroup:
        subgroup_index = reduction_offset
        subgroup_slot = subgroup_index & 1
        cluster = long_reduction_total + subgroup_index // 2
        first_rank = subgroup_slot * 2
        total_tiles = (total_len + _scheduler_mtp24___TILE_N_JIT - 1) // _scheduler_mtp24___TILE_N_JIT
        first_tiles = (total_tiles + 1) // 2
        first_len = tl.minimum(first_tiles * _scheduler_mtp24___TILE_N_JIT, total_len)
        second_len = total_len - first_len
        seq_kvcache = total_len - NUM_SEQ_Q
        _scheduler_mtp24___store_task_record(
            TASK_MAP,
            cluster * CLUSTER_SIZE + first_rank,
            hkv,
            batch,
            0,
            0,
            first_len,
            first_len,
            first_tiles,
            first_tiles,
            0,
            _scheduler_mtp24___SUBGROUP2_MODE_JIT,
            0,
            1,
            True,
        )
        _scheduler_mtp24___store_task_record(
            TASK_MAP,
            cluster * CLUSTER_SIZE + first_rank + 1,
            hkv,
            batch,
            1,
            first_len,
            second_len,
            second_len - NUM_SEQ_Q,
            (second_len + _scheduler_mtp24___TILE_N_JIT - 1) // _scheduler_mtp24___TILE_N_JIT,
            tl.maximum(second_len - NUM_SEQ_Q, 0) // _scheduler_mtp24___TILE_N_JIT,
            1,
            _scheduler_mtp24___SUBGROUP2_MODE_JIT,
            0,
            1,
            True,
        )
        if subgroup_index == subgroup_total - 1 and subgroup_slot == 0:
            _scheduler_mtp24___store_task_record(
                TASK_MAP,
                cluster * CLUSTER_SIZE + 2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                _scheduler_mtp24___SUBGROUP2_MODE_JIT,
                0,
                1,
                True,
            )
            _scheduler_mtp24___store_task_record(
                TASK_MAP,
                cluster * CLUSTER_SIZE + 3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                _scheduler_mtp24___SUBGROUP2_MODE_JIT,
                0,
                1,
                True,
            )
        return
    if direct:
        cluster = reduction_total + direct_index // CLUSTER_SIZE
        rank = direct_index % CLUSTER_SIZE
        cta = cluster * CLUSTER_SIZE + rank
        seq_kvcache = total_len - NUM_SEQ_Q
        _scheduler_mtp24___store_task_record(
            TASK_MAP,
            cta,
            hkv,
            batch,
            0,
            0,
            total_len,
            seq_kvcache,
            (total_len + _scheduler_mtp24___TILE_N_JIT - 1) // _scheduler_mtp24___TILE_N_JIT,
            seq_kvcache // _scheduler_mtp24___TILE_N_JIT,
            1,
            _scheduler_mtp24___DIRECT_MODE_JIT,
            0,
            1,
            True,
        )
        if direct_index == direct_total - 1:
            clear_rank = tl.arange(0, CLUSTER_SIZE)[:, None]
            clear_field = tl.arange(0, 16)[None, :]
            clear_cta = cluster * CLUSTER_SIZE + clear_rank
            clear_task_base = (clear_cta * _scheduler_mtp24___TASK_SLOTS_JIT + 1) * _scheduler_mtp24___TASK_STRIDE_JIT
            clear_mask = (clear_rank > rank) & (clear_field < _scheduler_mtp24___TASK_STRIDE_JIT)
            tl.store(TASK_MAP + clear_task_base + clear_field, -1, mask=clear_mask)
            clear_sentinel_base = (
                clear_cta * _scheduler_mtp24___TASK_SLOTS_JIT + 2
            ) * _scheduler_mtp24___TASK_STRIDE_JIT
            sentinel_mask = (clear_rank > rank) & (clear_field < 2)
            tl.store(TASK_MAP + clear_sentinel_base + clear_field, -1, mask=sentinel_mask)
        return
    covered_chunks = group_count * CLUSTER_SIZE
    chunk = 0
    while chunk < covered_chunks:
        real = chunk < num_chunks
        group_chunk = chunk // CLUSTER_SIZE
        rank = chunk % CLUSTER_SIZE
        cluster = reduction_offset + group_chunk
        cta = cluster * CLUSTER_SIZE + rank
        logical_start = chunk * sequence_chunk
        seq_start = tl.where(real, logical_start, 0)
        seq_len = tl.where(real, tl.minimum(sequence_chunk, total_len - logical_start), 0)
        is_last = real & (chunk == num_chunks - 1)
        seq_kvcache = tl.where(is_last, seq_len - NUM_SEQ_Q, seq_len)
        num_tile_kv = (seq_len + _scheduler_mtp24___TILE_N_JIT - 1) // _scheduler_mtp24___TILE_N_JIT
        num_tile_full = tl.where(
            is_last,
            tl.maximum(seq_kvcache, 0) // _scheduler_mtp24___TILE_N_JIT,
            seq_len // _scheduler_mtp24___TILE_N_JIT,
        )
        mode = tl.where(real, _scheduler_mtp24___GROUP_MODE_JIT, _scheduler_mtp24___DUMMY_MODE_JIT)
        _scheduler_mtp24___store_task_record(
            TASK_MAP,
            cta,
            hkv,
            batch,
            chunk,
            seq_start,
            seq_len,
            seq_kvcache,
            num_tile_kv,
            num_tile_full,
            is_last.to(tl.int32),
            mode,
            group_chunk,
            group_count,
            True,
        )
        chunk += 1


@triton.jit
def _scheduler_mtp24__refresh_cluster_task_tail_kernel(
    SEQLENS_KV,
    OFFSETS,
    META,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    DIRECT_THRESHOLD: tl.constexpr,
    SHORT_THRESHOLD: tl.constexpr,
    SHORT_CHUNK_TOKENS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Refresh length-dependent tail fields while cluster topology is stable."""
    seq_id = tl.arange(0, BLOCK_SEQ)
    valid = seq_id < B * H_KV
    hkv = seq_id // B
    batch = seq_id - hkv * B
    total_len = tl.load(SEQLENS_KV + batch, mask=valid, other=1).to(tl.int32)
    sequence_chunk = tl.where((SHORT_THRESHOLD > 0) & (total_len <= SHORT_THRESHOLD), SHORT_CHUNK_TOKENS, CHUNK_TOKENS)
    num_chunks = (total_len + sequence_chunk - 1) // sequence_chunk
    reduction_offset = tl.load(OFFSETS + seq_id * 2 + 0, mask=valid, other=0)
    direct_index = tl.load(OFFSETS + seq_id * 2 + 1, mask=valid, other=0)
    reduction_total = tl.load(META + _scheduler_mtp24___META_REDUCTION_CLUSTERS_JIT)
    direct = (num_chunks == 1) | (total_len <= DIRECT_THRESHOLD)
    tail_chunk = num_chunks - 1
    direct_cluster = reduction_total + direct_index // CLUSTER_SIZE
    direct_rank = direct_index % CLUSTER_SIZE
    reduction_cluster = reduction_offset + tail_chunk // CLUSTER_SIZE
    reduction_rank = tail_chunk % CLUSTER_SIZE
    cluster = tl.where(direct, direct_cluster, reduction_cluster)
    rank = tl.where(direct, direct_rank, reduction_rank)
    cta = cluster * CLUSTER_SIZE + rank
    task_base = (cta * _scheduler_mtp24___TASK_SLOTS_JIT + 1) * _scheduler_mtp24___TASK_STRIDE_JIT
    seq_start = tl.where(direct, 0, (num_chunks - 1) * sequence_chunk)
    seq_len = tl.where(direct, total_len, total_len - seq_start)
    seq_kvcache = seq_len - NUM_SEQ_Q
    tl.store(TASK_MAP + task_base + 3, seq_start, mask=valid)
    tl.store(TASK_MAP + task_base + 4, seq_len, mask=valid)
    tl.store(TASK_MAP + task_base + 5, seq_kvcache, mask=valid)
    tl.store(
        TASK_MAP + task_base + 6,
        (seq_len + _scheduler_mtp24___TILE_N_JIT - 1) // _scheduler_mtp24___TILE_N_JIT,
        mask=valid,
    )
    tl.store(TASK_MAP + task_base + 7, tl.maximum(seq_kvcache, 0) // _scheduler_mtp24___TILE_N_JIT, mask=valid)
    tl.store(TASK_MAP + task_base + 8, 1, mask=valid)


@dataclass
class _scheduler_mtp24__DecodeTaskSchedule:
    task_workspace: torch.Tensor
    task_map: torch.Tensor
    offsets: torch.Tensor
    meta: torch.Tensor
    cluster_size: int
    chunk_tokens: int
    block_seq: int
    block_chunks: int
    capacity_clusters: int
    capacity_ints: int
    direct_threshold: int = 0
    short_threshold: int = 0
    short_chunk_tokens: int = 0
    subgroup2_threshold: int = 0
    num_seq_q: int = 2
    num_clusters: int = 0
    physical_ctas: int = 0
    sched_ints: int = 0
    partial_slots: int = 0
    stats: dict | None = None


def _scheduler_mtp24___capacity(
    *, num_sequences: int, max_seq_kv: int, cluster_size: int, chunk_tokens: int
) -> tuple[int, int, int]:
    max_chunks = max(1, (max_seq_kv + chunk_tokens - 1) // chunk_tokens)
    max_groups = (max_chunks + cluster_size - 1) // cluster_size
    max_reduction_clusters = num_sequences * max_groups if max_chunks > 1 else 0
    max_direct_clusters = (num_sequences + cluster_size - 1) // cluster_size
    capacity_clusters = max_reduction_clusters + max_direct_clusters
    capacity_ctas = max(capacity_clusters * cluster_size, cluster_size)
    chunk_pad_ints = (
        (num_sequences + _scheduler_mtp24__TASK_STRIDE - 1)
        // _scheduler_mtp24__TASK_STRIDE
        * _scheduler_mtp24__TASK_STRIDE
    )
    capacity_ints = (_scheduler_mtp24__TASK_SLOTS * capacity_ctas + 1) * _scheduler_mtp24__TASK_STRIDE + chunk_pad_ints
    block_chunks = triton.next_power_of_2(max(max_chunks, cluster_size))
    return (capacity_clusters, capacity_ints, block_chunks)


def _scheduler_mtp24__allocate_cluster_task_map(
    kv_lens: torch.Tensor,
    *,
    num_head_kv: int,
    max_seq_kv: int,
    cluster_size: int,
    chunk_tokens: int,
    direct_threshold: int,
    subgroup2_threshold: int,
    num_seq_q: int,
) -> _scheduler_mtp24__DecodeTaskSchedule:
    """Allocate capacity and populate a cluster task map entirely on GPU."""
    if not kv_lens.is_cuda:
        raise ValueError("cluster GPU assignment requires CUDA kv_lens")
    if cluster_size not in (2, 4, 8):
        raise ValueError(f"cluster_size must be 2, 4, or 8, got {cluster_size}")
    if num_seq_q < 1:
        raise ValueError(f"num_seq_q must be positive, got {num_seq_q}")
    if chunk_tokens < _scheduler_mtp24__TILE_N or chunk_tokens % _scheduler_mtp24__TILE_N:
        raise ValueError(f"chunk_tokens must be a positive multiple of {_scheduler_mtp24__TILE_N}, got {chunk_tokens}")
    if direct_threshold < 0 or direct_threshold % _scheduler_mtp24__TILE_N:
        raise ValueError(
            f"direct_threshold must be zero or a multiple of {_scheduler_mtp24__TILE_N}, got {direct_threshold}"
        )
    if subgroup2_threshold < 0 or subgroup2_threshold % _scheduler_mtp24__TILE_N:
        raise ValueError(
            f"subgroup2_threshold must be zero or a multiple of {_scheduler_mtp24__TILE_N}, got {subgroup2_threshold}"
        )
    if subgroup2_threshold and cluster_size != 4:
        raise ValueError("subgroup2_threshold requires cluster_size=4")
    if 0 % _scheduler_mtp24__TILE_N:
        raise ValueError(f"short_threshold must be zero or a multiple of {_scheduler_mtp24__TILE_N}, got {0}")
    num_sequences = kv_lens.numel() * num_head_kv
    block_seq = triton.next_power_of_2(kv_lens.numel())
    if block_seq > 1024:
        raise ValueError(f"cluster GPU assign currently supports B <= 1024, got {kv_lens.numel()}")
    capacity_clusters, capacity_ints, block_chunks = _scheduler_mtp24___capacity(
        num_sequences=num_sequences, max_seq_kv=max_seq_kv, cluster_size=cluster_size, chunk_tokens=chunk_tokens
    )
    task_map = torch.full((capacity_ints,), -1, dtype=torch.int32, device=kv_lens.device)
    assignment = _scheduler_mtp24__DecodeTaskSchedule(
        task_workspace=task_map.view(torch.int8),
        task_map=task_map,
        offsets=torch.empty((num_sequences, 2), dtype=torch.int32, device=kv_lens.device),
        meta=torch.empty((_scheduler_mtp24__META_SIZE,), dtype=torch.int32, device=kv_lens.device),
        cluster_size=cluster_size,
        chunk_tokens=chunk_tokens,
        block_seq=block_seq,
        block_chunks=block_chunks,
        capacity_clusters=capacity_clusters,
        capacity_ints=capacity_ints,
        direct_threshold=direct_threshold,
        short_threshold=0,
        short_chunk_tokens=0,
        subgroup2_threshold=subgroup2_threshold,
        num_seq_q=num_seq_q,
    )
    _scheduler_mtp24__launch_cluster_task_map_assign(
        kv_lens, assignment, num_head_kv=num_head_kv, refresh_host_metadata=True
    )
    return assignment


def _scheduler_mtp24__launch_cluster_task_map_assign(
    kv_lens: torch.Tensor,
    assignment: _scheduler_mtp24__DecodeTaskSchedule,
    *,
    num_head_kv: int,
    refresh_host_metadata: bool,
) -> None:
    """Regenerate an allocated map; fixed-topology calls need no host sync."""
    batch = kv_lens.numel()
    num_sequences = batch * num_head_kv
    if assignment.offsets.numel() != num_sequences * 2:
        raise ValueError("kv_lens/H_KV shape differs from the allocated cluster task map")
    prefix_warps = max(1, min(32, assignment.block_seq // 32))
    _scheduler_mtp24__assign_cluster_task_prefix_kernel[1,](
        kv_lens,
        assignment.offsets,
        assignment.meta,
        assignment.task_map,
        B=batch,
        H_KV=num_head_kv,
        NUM_SEQ_Q=assignment.num_seq_q,
        CLUSTER_SIZE=assignment.cluster_size,
        CHUNK_TOKENS=assignment.chunk_tokens,
        DIRECT_THRESHOLD=assignment.direct_threshold,
        SHORT_THRESHOLD=assignment.short_threshold,
        SHORT_CHUNK_TOKENS=assignment.short_chunk_tokens,
        SUBGROUP2_THRESHOLD=assignment.subgroup2_threshold,
        BLOCK_SEQ=assignment.block_seq,
        num_warps=prefix_warps,
        num_stages=1,
    )
    record_warps = 1
    records_kernel = _scheduler_mtp24__assign_cluster_task_records_compact_kernel
    record_args = {
        "B": batch,
        "NUM_SEQ_Q": assignment.num_seq_q,
        "CLUSTER_SIZE": assignment.cluster_size,
        "CHUNK_TOKENS": assignment.chunk_tokens,
        "DIRECT_THRESHOLD": assignment.direct_threshold,
        "SHORT_THRESHOLD": assignment.short_threshold,
        "SHORT_CHUNK_TOKENS": assignment.short_chunk_tokens,
        "SUBGROUP2_THRESHOLD": assignment.subgroup2_threshold,
        "num_warps": record_warps,
        "num_stages": 1,
    }
    records_kernel[num_sequences,](kv_lens, assignment.offsets, assignment.meta, assignment.task_map, **record_args)
    if not refresh_host_metadata:
        return
    values = assignment.meta.detach().cpu().to(torch.int64).tolist()
    invalid_lengths = values[_scheduler_mtp24__META_INVALID_LENGTHS]
    if invalid_lengths:
        raise ValueError(f"cluster GPU assign found {invalid_lengths} total KV lengths below {assignment.num_seq_q}")
    num_clusters = values[_scheduler_mtp24__META_NUM_CLUSTERS]
    sched_ints = values[_scheduler_mtp24__META_SCHED_INTS]
    if num_clusters > assignment.capacity_clusters or sched_ints > assignment.capacity_ints:
        raise RuntimeError(
            f"cluster task-map capacity was underestimated: clusters={num_clusters}/{assignment.capacity_clusters}, ints={sched_ints}/{assignment.capacity_ints}"
        )
    assignment.num_clusters = num_clusters
    assignment.physical_ctas = values[_scheduler_mtp24__META_PHYSICAL_CTAS]
    assignment.sched_ints = sched_ints
    assignment.partial_slots = max(values[_scheduler_mtp24__META_EFFECTIVE_CHUNKS_MAX], 1)
    assignment.stats = {
        "cluster_size": assignment.cluster_size,
        "num_clusters": num_clusters,
        "physical_ctas": values[_scheduler_mtp24__META_PHYSICAL_CTAS],
        "reduction_clusters": values[_scheduler_mtp24__META_REDUCTION_CLUSTERS],
        "compute_tasks": values[_scheduler_mtp24__META_COMPUTE_TASKS],
        "fine_chunks_max": values[_scheduler_mtp24__META_FINE_CHUNKS_MAX],
        "effective_chunks_max": values[_scheduler_mtp24__META_EFFECTIVE_CHUNKS_MAX],
        "direct_tasks": values[_scheduler_mtp24__META_DIRECT_TASKS],
        "dummy_tasks": values[_scheduler_mtp24__META_DUMMY_TASKS],
        "subgroup2_tasks": values[_scheduler_mtp24__META_SUBGROUP_TASKS],
    }


def _scheduler_mtp24__launch_cluster_task_tail_refresh(
    kv_lens: torch.Tensor, assignment: _scheduler_mtp24__DecodeTaskSchedule, *, num_head_kv: int
) -> None:
    """Update tail records without rebuilding unchanged 512-token topology.

    The caller must run ``launch_cluster_task_map_assign`` with refreshed host
    metadata whenever any ``ceil(kv_len / chunk_tokens)`` value changes.
    """
    batch = kv_lens.numel()
    num_sequences = batch * num_head_kv
    if assignment.offsets.numel() != num_sequences * 2:
        raise ValueError("kv_lens/H_KV shape differs from the allocated cluster task map")
    block_seq = triton.next_power_of_2(num_sequences)
    num_warps = max(1, min(8, block_seq // 32))
    _scheduler_mtp24__refresh_cluster_task_tail_kernel[1,](
        kv_lens,
        assignment.offsets,
        assignment.meta,
        assignment.task_map,
        B=batch,
        H_KV=num_head_kv,
        NUM_SEQ_Q=assignment.num_seq_q,
        CLUSTER_SIZE=assignment.cluster_size,
        CHUNK_TOKENS=assignment.chunk_tokens,
        DIRECT_THRESHOLD=assignment.direct_threshold,
        SHORT_THRESHOLD=assignment.short_threshold,
        SHORT_CHUNK_TOKENS=assignment.short_chunk_tokens,
        BLOCK_SEQ=block_seq,
        num_warps=num_warps,
        num_stages=1,
    )


# BF16 uses a compact 8-int compute-task ABI after assignment.  Keep that ABI
# next to the common 12-int scheduler instead of maintaining a dtype-specific
# assign-task module.  The distinction is an output-record format required by
# the consumer kernel, not a separate BF16 scheduling policy.
_bf16_compact__TASK_STRIDE = 8
_bf16_compact__TASK_STRIDE_JIT = tl.constexpr(_bf16_compact__TASK_STRIDE)


@triton.jit
def _bf16_compact__assign_prefix_kernel(
    KV_LENS,
    OFFSETS,
    META,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Compact variable-length sequences into contiguous cluster slots."""
    batch = tl.arange(0, BLOCK_SEQ)
    valid = batch < B
    total_len = tl.load(KV_LENS + batch, mask=valid, other=0).to(tl.int32)
    positive = valid & (total_len >= NUM_SEQ_Q)
    chunks = tl.where(positive, (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS, 0)
    groups = (chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    is_direct = positive & (chunks == 1) & (CLUSTER_SIZE > 1)
    is_subgroup2 = positive & (chunks == 2) & (CLUSTER_SIZE >= 4)
    is_subgroup4 = positive & (chunks >= 3) & (chunks <= 4) & (CLUSTER_SIZE >= 8)
    is_compact = is_direct | is_subgroup2 | is_subgroup4
    reduction_groups = tl.where(is_compact, 0, groups)
    direct_tasks = is_direct.to(tl.int32)
    subgroup2_groups = is_subgroup2.to(tl.int32)
    subgroup4_groups = is_subgroup4.to(tl.int32)
    reduction_offsets, groups_per_head = tle.cumsum(reduction_groups, axis=0, reverse=False)
    direct_offsets, direct_per_head = tle.cumsum(direct_tasks, axis=0, reverse=False)
    subgroup2_offsets, subgroup2_per_head = tle.cumsum(subgroup2_groups, axis=0, reverse=False)
    subgroup4_offsets, subgroup4_per_head = tle.cumsum(subgroup4_groups, axis=0, reverse=False)
    tl.store(OFFSETS + batch * 2, reduction_offsets, mask=valid)
    tl.store(OFFSETS + batch * 2 + 1, direct_offsets, mask=valid)
    tl.store(OFFSETS + B * 2 + batch * 2, subgroup2_offsets, mask=valid)
    tl.store(OFFSETS + B * 2 + batch * 2 + 1, subgroup4_offsets, mask=valid)
    tl.store(META, groups_per_head * H_KV)
    tl.store(META + 1, groups_per_head)
    tl.store(META + 2, direct_per_head * H_KV)
    tl.store(META + 3, direct_per_head)
    tl.store(META + 4, subgroup2_per_head * H_KV)
    tl.store(META + 5, subgroup2_per_head)
    tl.store(META + 6, subgroup4_per_head * H_KV)
    tl.store(META + 7, subgroup4_per_head)


@triton.jit
def _bf16_compact__assign_records_kernel(
    KV_LENS,
    OFFSETS,
    TASK_MAP,
    DIRECT_TASK_MAP,
    SUBGROUP2_TASK_MAP,
    SUBGROUP4_TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    GROUPS_PER_HEAD: tl.constexpr,
    DIRECT_PER_HEAD: tl.constexpr,
    SUBGROUP2_PER_HEAD: tl.constexpr,
    SUBGROUP4_PER_HEAD: tl.constexpr,
):
    """Emit the compact BF16 consumer records from the shared scheduler."""
    sequence = tl.program_id(0)
    hkv = sequence // B
    batch = sequence - hkv * B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if (CLUSTER_SIZE > 1) & (num_chunks == 1):
        direct = hkv * DIRECT_PER_HEAD + tl.load(OFFSETS + batch * 2 + 1)
        task = direct * _bf16_compact__TASK_STRIDE_JIT
        tl.store(DIRECT_TASK_MAP + task, hkv)
        tl.store(DIRECT_TASK_MAP + task + 1, batch)
        tl.store(DIRECT_TASK_MAP + task + 2, 0)
        tl.store(DIRECT_TASK_MAP + task + 3, total_len)
        tl.store(DIRECT_TASK_MAP + task + 4, 0)
        tl.store(DIRECT_TASK_MAP + task + 5, 1)
        tl.store(DIRECT_TASK_MAP + task + 6, total_len)
        tl.store(DIRECT_TASK_MAP + task + 7, 1)
    elif (CLUSTER_SIZE >= 4) & (num_chunks == 2):
        first = hkv * SUBGROUP2_PER_HEAD + tl.load(OFFSETS + B * 2 + batch * 2)
        for rank in tl.static_range(0, 2):
            task = (first * 2 + rank) * _bf16_compact__TASK_STRIDE_JIT
            chunk_start = rank * CHUNK_TOKENS
            chunk_len = tl.minimum(CHUNK_TOKENS, total_len - chunk_start)
            tl.store(SUBGROUP2_TASK_MAP + task, hkv)
            tl.store(SUBGROUP2_TASK_MAP + task + 1, batch)
            tl.store(SUBGROUP2_TASK_MAP + task + 2, chunk_start)
            tl.store(SUBGROUP2_TASK_MAP + task + 3, chunk_len)
            tl.store(SUBGROUP2_TASK_MAP + task + 4, 0)
            tl.store(SUBGROUP2_TASK_MAP + task + 5, 1)
            tl.store(SUBGROUP2_TASK_MAP + task + 6, total_len)
            tl.store(SUBGROUP2_TASK_MAP + task + 7, 1)
    elif (CLUSTER_SIZE >= 8) & (num_chunks <= 4):
        first = hkv * SUBGROUP4_PER_HEAD + tl.load(OFFSETS + B * 2 + batch * 2 + 1)
        for rank in tl.static_range(0, 4):
            has_work = rank < num_chunks
            task = (first * 4 + rank) * _bf16_compact__TASK_STRIDE_JIT
            chunk_start = tl.where(has_work, rank * CHUNK_TOKENS, 0)
            chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
            tl.store(SUBGROUP4_TASK_MAP + task, hkv)
            tl.store(SUBGROUP4_TASK_MAP + task + 1, batch)
            tl.store(SUBGROUP4_TASK_MAP + task + 2, chunk_start)
            tl.store(SUBGROUP4_TASK_MAP + task + 3, chunk_len)
            tl.store(SUBGROUP4_TASK_MAP + task + 4, 0)
            tl.store(SUBGROUP4_TASK_MAP + task + 5, 1)
            tl.store(SUBGROUP4_TASK_MAP + task + 6, total_len)
            tl.store(SUBGROUP4_TASK_MAP + task + 7, has_work.to(tl.int32))
    else:
        first_group = hkv * GROUPS_PER_HEAD + tl.load(OFFSETS + batch * 2)
        group = 0
        while group < group_count:
            rank = 0
            while rank < CLUSTER_SIZE:
                chunk = group * CLUSTER_SIZE + rank
                has_work = chunk < num_chunks
                chunk_start = tl.where(has_work, chunk * CHUNK_TOKENS, 0)
                chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
                task = ((first_group + group) * CLUSTER_SIZE + rank) * _bf16_compact__TASK_STRIDE_JIT
                tl.store(TASK_MAP + task, hkv)
                tl.store(TASK_MAP + task + 1, batch)
                tl.store(TASK_MAP + task + 2, chunk_start)
                tl.store(TASK_MAP + task + 3, chunk_len)
                tl.store(TASK_MAP + task + 4, group)
                tl.store(TASK_MAP + task + 5, group_count)
                tl.store(TASK_MAP + task + 6, total_len)
                tl.store(TASK_MAP + task + 7, has_work.to(tl.int32))
                rank += 1
            group += 1


__all__ = [
    "PURE_TRITON_TASK_STRIDE",
    "PURE_TRITON_TILE_N",
    "assign_pure_triton_task_map_kernel",
    "pure_triton_task_map_metadata",
    "launch_pure_triton_task_map",
    "_scheduler_mtp1__DecodeTaskSchedule",
    "_scheduler_mtp1__allocate_cluster_task_map",
    "_scheduler_mtp1__launch_cluster_task_map_assign",
    "_scheduler_mtp1__launch_cluster_task_tail_refresh",
    "_scheduler_mtp24__DecodeTaskSchedule",
    "_scheduler_mtp24__allocate_cluster_task_map",
    "_scheduler_mtp24__launch_cluster_task_map_assign",
    "_scheduler_mtp24__launch_cluster_task_tail_refresh",
    "_bf16_compact__assign_prefix_kernel",
    "_bf16_compact__assign_records_kernel",
]
