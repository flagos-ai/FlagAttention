"""Self-contained Hopper attention-decode implementation."""

from __future__ import annotations

from triton.language import core as tl_core
from triton.language.core import builtin
from typing import Literal
from triton.tools.tensor_descriptor import TensorDescriptor


from dataclasses import dataclass

import torch

import triton

import triton.language as tl

from .. import (
    PureTritonMTPWorkspace,
    PureTritonMTP1Workspace,
    USE_TLE,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
    gpu_types,
    tle,
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
    H_KV: tl.constexpr,
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
    refresh_host_metadata: bool = False,
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
        "H_KV": num_head_kv,
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
    H_KV: tl.constexpr,
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
    *,
    num_sequences: int,
    max_seq_kv: int,
    cluster_size: int,
    chunk_tokens: int,
    short_threshold: int = 0,
    short_chunk_tokens: int = 0,
) -> tuple[int, int, int]:
    max_chunks = max(1, (max_seq_kv + chunk_tokens - 1) // chunk_tokens)
    if short_threshold > 0:
        max_short_len = min(max_seq_kv, short_threshold)
        max_chunks = max(max_chunks, (max_short_len + short_chunk_tokens - 1) // short_chunk_tokens)
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
    direct_threshold: int = 0,
    short_threshold: int = 0,
    short_chunk_tokens: int = 0,
    subgroup2_threshold: int = 0,
    num_seq_q: int = 2,
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
    if short_threshold < 0 or short_threshold % _scheduler_mtp24__TILE_N:
        raise ValueError(
            f"short_threshold must be zero or a multiple of {_scheduler_mtp24__TILE_N}, got {short_threshold}"
        )
    if short_threshold == 0:
        short_chunk_tokens = chunk_tokens
    elif short_chunk_tokens < _scheduler_mtp24__TILE_N or short_chunk_tokens % _scheduler_mtp24__TILE_N:
        raise ValueError(
            f"short_chunk_tokens must be a positive multiple of {_scheduler_mtp24__TILE_N}, got {short_chunk_tokens}"
        )
    num_sequences = kv_lens.numel() * num_head_kv
    block_seq = triton.next_power_of_2(kv_lens.numel())
    if block_seq > 1024:
        raise ValueError(f"cluster GPU assign currently supports B <= 1024, got {kv_lens.numel()}")
    capacity_clusters, capacity_ints, block_chunks = _scheduler_mtp24___capacity(
        num_sequences=num_sequences,
        max_seq_kv=max_seq_kv,
        cluster_size=cluster_size,
        chunk_tokens=chunk_tokens,
        short_threshold=short_threshold,
        short_chunk_tokens=short_chunk_tokens,
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
        short_threshold=short_threshold,
        short_chunk_tokens=short_chunk_tokens,
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
    refresh_host_metadata: bool = False,
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
        "H_KV": num_head_kv,
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


_compute_mtp1__TASK_STRIDE = 12

_compute_mtp1__TASK_SLOTS = 2

_compute_mtp1__NUM_SEQ_Q = 1

_compute_mtp1__ROWS_Q = 8

_compute_mtp1__DIRECT_MODE = 0

_compute_mtp1__DUMMY_MODE = 2

_compute_mtp1__GROUP_MODE = 1

_compute_mtp1__EXECUTION_FULL = 0

_compute_mtp1__EXECUTION_LOCAL_PARTIAL = 2

_compute_mtp1___TASK_STRIDE_JIT = tl.constexpr(_compute_mtp1__TASK_STRIDE)

_compute_mtp1___TASK_SLOTS_JIT = tl.constexpr(_compute_mtp1__TASK_SLOTS)

_compute_mtp1___ROWS_Q_JIT = tl.constexpr(_compute_mtp1__ROWS_Q)

_compute_mtp1___NUM_SEQ_Q_JIT = tl.constexpr(_compute_mtp1__NUM_SEQ_Q)

_compute_mtp1___TMA_STAGES_JIT = tl.constexpr(2)

_compute_mtp1___DIRECT_MODE_JIT = tl.constexpr(_compute_mtp1__DIRECT_MODE)

_compute_mtp1___DUMMY_MODE_JIT = tl.constexpr(_compute_mtp1__DUMMY_MODE)

_compute_mtp1___GROUP_MODE_JIT = tl.constexpr(_compute_mtp1__GROUP_MODE)

_compute_mtp1___K_FRAGMENT_JIT = tl.constexpr(32)

_compute_mtp1___EXECUTION_FULL_JIT = tl.constexpr(_compute_mtp1__EXECUTION_FULL)

_compute_mtp1___EXECUTION_LOCAL_PARTIAL_JIT = tl.constexpr(_compute_mtp1__EXECUTION_LOCAL_PARTIAL)


@triton.jit
def _compute_mtp1___load_packed_k_scale_mtp1(
    KSCALE,
    phys,
    hkv,
    offs_n,
    KS_STRIDE_BLOCK: tl.constexpr,
    KS_STRIDE_TOKEN: tl.constexpr,
    KS_STRIDE_HEAD: tl.constexpr,
    KS_STRIDE_D: tl.constexpr,
):
    """Load official quant_type=0 scales packed into FP8 cache rows."""
    byte_offset = (
        phys * KS_STRIDE_BLOCK + offs_n // 32 * KS_STRIDE_TOKEN + hkv * KS_STRIDE_HEAD + offs_n % 32 * 4 * KS_STRIDE_D
    )
    scale_ptr = (KSCALE + byte_offset).to(tl.pointer_type(tl.float32))
    return tl.load(scale_ptr).to(tl.float32)


@triton.jit
def _compute_mtp1___cluster_cooperative_finalize_mtp1(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    REUSE_FINAL_WEIGHTS: tl.constexpr,
):
    """Use one CTA rank per GQA head to finalize both MTP rows in-kernel."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_m = tl.arange(0, _compute_mtp1___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None] < n_chunks) & valid_head
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :]
    weights_unnorm = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights_unnorm = tl.where(chunk_mask, weights_unnorm, 0.0)
    denom = tl.sum(weights_unnorm, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    if REUSE_FINAL_WEIGHTS:
        weight_rows = tl.broadcast_to(offs_c[:, None], (MAX_FINAL_CHUNKS, _compute_mtp1___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(offs_m[None, :], (MAX_FINAL_CHUNKS, _compute_mtp1___NUM_SEQ_Q_JIT))
        weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols))
        tl.store(weight_ptr, weights_unnorm * inv_denom[None, :])
        tl.debug_barrier()
    acc = tl.zeros((DV, _compute_mtp1___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head
        if REUSE_FINAL_WEIGHTS:
            chunk_rows = tl.full((_compute_mtp1___NUM_SEQ_Q_JIT,), chunk, tl.int32)
            chunk_weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, offs_m))
            chunk_weight = tl.load(chunk_weight_ptr, mask=chunk_valid, other=0.0)
        else:
            chunk_lse = tl.load(
                LSE
                + batch * LSE_STRIDE_B
                + chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + offs_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                mask=chunk_valid,
                other=-float("inf"),
            )
            chunk_delta = chunk_lse - safe_max_lse
            chunk_weight = (tl.exp2(chunk_delta) if USE_LOG2 else tl.exp(chunk_delta)) * inv_denom
            chunk_weight = tl.where(chunk_valid, chunk_weight, 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, :] * SO_STRIDE_M
            + hq * SO_STRIDE_H
            + offs_v[:, None],
            mask=chunk_valid,
            other=0.0,
        )
        acc += partial * chunk_weight[None, :]
    if REUSE_FINAL_WEIGHTS:
        tl.debug_barrier()
    tl.store(
        OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_head & (denom[None, :] > 0.0),
    )


@triton.jit
def _compute_mtp1___cluster_quad_head_two_chunk_finalize_mtp1(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    logical_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize four C2-owned heads and exactly two global partials."""
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp1___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = logical_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weight0 = tl.exp2(lse0 - safe_max) if USE_LOG2 else tl.exp(lse0 - safe_max)
    weight1 = tl.exp2(lse1 - safe_max) if USE_LOG2 else tl.exp(lse1 - safe_max)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    acc = tl.load(partial_base, mask=partial_mask, other=0.0)
    acc *= weight0[None, :, :]
    acc += tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0) * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp1___cluster_paired_head_finalize_mtp1(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    first_h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
):
    """Finalize two strided GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp1___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = first_h_in_group + offs_h * HEAD_STRIDE
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp1___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp1___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp1___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp1___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    acc = tl.zeros((DV, 2, _compute_mtp1___NUM_SEQ_Q_JIT), tl.float32)
    pair_cols = offs_h[:, None] * _compute_mtp1___NUM_SEQ_Q_JIT + offs_m[None, :]
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp1___NUM_SEQ_Q_JIT)
        chunk_rows = tl.full((2, _compute_mtp1___NUM_SEQ_Q_JIT), chunk, tl.int32)
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, pair_cols)), mask=chunk_valid, other=0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=chunk_valid[None, :, :],
            other=0.0,
        )
        acc += partial * chunk_weight[None, :, :]
    tl.debug_barrier()
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@builtin
def _compute_mtp1___memdesc_subslice(
    value,
    shape: tl.constexpr,
    offsets: tl.constexpr,
    _semantic=None,
):
    """Return a zero-copy shared-memory subslice through the public builder."""
    shape = [int(tl_core._unwrap_if_constexpr(dim)) for dim in shape]
    layout = value.type.layout
    result_ty = gpu_types.buffered_tensor_type(
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=value.type.alloc_shape,
    )
    handle = _semantic.builder.create_memdesc_subslice(
        result_ty.to_ir(_semantic.builder),
        value.handle,
        list(offsets),
    )
    return gpu_types.buffered_tensor(
        handle,
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=value.type.alloc_shape,
    )


@builtin
def _compute_mtp1___memdesc_transpose_2d(value, _semantic=None):
    """Return a zero-copy transposed view of a rank-2 shared memdesc."""
    if len(value.type.shape) != 2:
        raise ValueError("_compute_mtp1___memdesc_transpose_2d expects rank 2")
    order = [1, 0]
    handle = _semantic.builder.create_memdesc_trans(value.handle, order)
    shape = [value.type.shape[i] for i in order]
    alloc_shape = value.type.alloc_shape
    leading_rank = len(alloc_shape) - len(value.type.shape)
    alloc_tail = alloc_shape[leading_rank:]
    transposed_alloc_shape = alloc_shape[:leading_rank] + [alloc_tail[i] for i in order]
    layout = value.type.layout.make_permute(order)
    return gpu_types.buffered_tensor(
        handle,
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=transposed_alloc_shape,
    )


@triton.jit
def _compute_mtp1__fp8_kvpertensor_decode_mtp1_final_kernel(
    Q,
    K_DESC,
    KS_DESC,
    VT_DESC,
    V_CACHE,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    V_STRIDE_BLOCK: tl.constexpr,
    V_STRIDE_TOKEN: tl.constexpr,
    V_STRIDE_HEAD: tl.constexpr,
    V_STRIDE_D: tl.constexpr,
    USE_LOG2: tl.constexpr,
    K_PER_TOKEN_V_PER_HEAD: tl.constexpr = False,
    TMA_K_SCALE: tl.constexpr = False,
    PAGE_METADATA_K_SCALE: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    RAW_PAGED_NHD_OR_HND: tl.constexpr = False,
    FULL_MATRIX_RS: tl.constexpr = False,
    TRANSPOSED_MEMDESC_RS: tl.constexpr = False,
    DIRECT_V_SHARED_SHARED: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    TLE_SHARED_SHARED: tl.constexpr = False,
    FULL_VIEW_V_RS: tl.constexpr = False,
    TMA_DN_RS: tl.constexpr = False,
    DIRECT_GLOBAL_V_RS: tl.constexpr = False,
    FRAGMENT_PIPELINED_RS: tl.constexpr = False,
    K32_PIPELINED_RS: tl.constexpr = False,
    INPLACE_PV_ACC: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp1__EXECUTION_FULL,
    CLUSTER_COOPERATIVE_FINALIZE: tl.constexpr = False,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    REUSE_FINAL_WEIGHTS: tl.constexpr = False,
    HEAD_SHARDED_DSM: tl.constexpr = False,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    C2_PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    BF16_DSM: tl.constexpr = False,
    DEFERRED_NORM: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    REDUCTION_ONLY: tl.constexpr = False,
    ALIGNED_FULL_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
    STATIC_MTP1_NO_CAUSAL_MASK: tl.constexpr = False,
    STATIC_BLOCK_IDS_PREFETCH: tl.constexpr = False,
):
    cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    task_base = (cta * _compute_mtp1___TASK_SLOTS_JIT + 1) * _compute_mtp1___TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task_base + 0)
    batch = tl.load(TASK_MAP + task_base + 1)
    if not REDUCTION_ONLY and hkv < 0:
        return
    seq_start = tl.load(TASK_MAP + task_base + 3)
    if ALIGNED_FULL_CHUNK_TOKENS:
        seq_len = ALIGNED_FULL_CHUNK_TOKENS
    else:
        seq_len = tl.load(TASK_MAP + task_base + 4)
    seq_kvcache = tl.load(TASK_MAP + task_base + 5)
    is_causal = tl.load(TASK_MAP + task_base + 8)
    if REDUCTION_ONLY:
        task_mode = _compute_mtp1___GROUP_MODE_JIT
    else:
        task_mode = tl.load(TASK_MAP + task_base + 9)
    group_chunk = tl.load(TASK_MAP + task_base + 10)
    group_count = tl.load(TASK_MAP + task_base + 11)
    has_work = True if REDUCTION_ONLY else task_mode != _compute_mtp1___DUMMY_MODE_JIT
    q_smem = tle.gpu.alloc([_compute_mtp1___ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_compute_mtp1___ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc(
        [_compute_mtp1___TMA_STAGES_JIT, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    if TMA_DN_RS:
        v_dn_smem = tle.gpu.alloc(
            [_compute_mtp1___TMA_STAGES_JIT, DV, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
        )
    elif not DIRECT_GLOBAL_V_RS:
        v_raw_smem = tle.gpu.alloc(
            [_compute_mtp1___TMA_STAGES_JIT, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
        )
    if TMA_DN_RS:
        v_raw_smem = v_dn_smem
    elif DIRECT_GLOBAL_V_RS:
        v_raw_smem = k_raw_smem
        v_dn_smem = k_raw_smem
    else:
        v_dn_smem = v_raw_smem
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp1___TMA_STAGES_JIT, arrive_count=1, expect_bytes=BLOCK_N * D
    )
    if not DIRECT_GLOBAL_V_RS:
        vt_full = tle.gpu.alloc_barriers(
            num_barriers=_compute_mtp1___TMA_STAGES_JIT, arrive_count=1, expect_bytes=DV * BLOCK_N
        )
    else:
        vt_full = k_full
    if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
        ks_smem = tle.gpu.alloc(
            [_compute_mtp1___TMA_STAGES_JIT, 1, 2, 1, 32], dtype=tl.float32, layout=None, scope=tle.gpu.smem
        )
        ks_full = tle.gpu.alloc_barriers(
            num_barriers=_compute_mtp1___TMA_STAGES_JIT, arrive_count=1, expect_bytes=2 * 32 * 4
        )
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            peer_l_smem = tle.gpu.alloc(
                [_compute_mtp1___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer_l_smem = peer_lse_smem
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer2_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer3_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer1_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer2_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer3_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    else:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            partial_l_smem = tle.gpu.alloc(
                [_compute_mtp1___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            partial_l_smem = partial_lse_smem
    final_weight_smem = tle.gpu.alloc(
        [
            MAX_FINAL_CHUNKS,
            2 * _compute_mtp1___NUM_SEQ_Q_JIT if PAIRED_HEAD_FINALIZE else _compute_mtp1___NUM_SEQ_Q_JIT,
        ],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    offs_q = tl.arange(0, _compute_mtp1___ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _compute_mtp1___ROWS_Q_JIT)[:, None], (_compute_mtp1___ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_compute_mtp1___ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _compute_mtp1___ROWS_Q_JIT)[:, None], (_compute_mtp1___ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_compute_mtp1___ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _compute_mtp1___ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _compute_mtp1___ROWS_Q_JIT)[None, :], (DV, _compute_mtp1___ROWS_Q_JIT))
    store_offs_v = offs_v
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if MERGE_CLUSTER_SIZE == 8:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
        if MERGE_CLUSTER_SIZE == 8:
            partial_l_ptr = tle.gpu.local_ptr(partial_l_smem, (offs_q,))
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + hkv if K_PER_TOKEN_V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = has_work & (seq_m < _compute_mtp1___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _compute_mtp1___ROWS_Q_JIT), tl.float32)
    lse = tl.full((_compute_mtp1___ROWS_Q_JIT,), -float("inf"), tl.float32)
    raw_l = tl.zeros((_compute_mtp1___ROWS_Q_JIT,), tl.float32)
    if STATIC_BLOCK_IDS_PREFETCH:
        chunk_block_ids_smem = tle.gpu.alloc(
            [STATIC_CHUNK_TOKENS // BLOCK_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    if has_work:
        if STATIC_BLOCK_IDS_PREFETCH:
            block_id_offs = tl.arange(0, STATIC_CHUNK_TOKENS // BLOCK_SIZE)
            num_chunk_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            staged_block_ids = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + seq_start // BLOCK_SIZE + block_id_offs,
                mask=block_id_offs < num_chunk_blocks,
                other=0,
            )
            tl.store(
                tle.gpu.local_ptr(chunk_block_ids_smem, (block_id_offs,)),
                staged_block_ids,
                mask=block_id_offs < num_chunk_blocks,
            )
            tl.debug_barrier()
        q = tl.load(
            Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
            mask=valid_q[:, None],
            other=0.0,
        )
        tl.store(q_smem_ptr, q)
        qscale = tl.load(
            QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0
        ).to(tl.float32)
        if PRECOMBINE_Q_SCALE:
            qscale = qscale * (inv_sqrt_d * 1.4426950408889634)
            if not K_PER_TOKEN_V_PER_HEAD:
                qscale = qscale * kscale
        m_i = tl.full((_compute_mtp1___ROWS_Q_JIT,), -float("inf"), tl.float32)
        l_i = tl.zeros((_compute_mtp1___ROWS_Q_JIT,), tl.float32)
        copy_iter = 0
        start = 0
        page_current_phys = tl.full((), 0, tl.int32)
        if start < seq_len:
            if STATIC_BLOCK_IDS_PREFETCH:
                phys = tl.load(tle.gpu.local_ptr(chunk_block_ids_smem, (0,)))
            else:
                block_no = seq_start // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            if PAGE_METADATA_K_SCALE:
                page_current_phys = phys
            tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                tle.gpu.copy(KS_DESC, ks_smem.slot(0), [1, 2, 1, 32], [phys, 0, hkv, 0], barrier=ks_full[0])
            if TMA_DN_RS:
                tle.gpu.copy(
                    VT_DESC, v_dn_smem.slot(0), [DV, BLOCK_N], [hkv * DV, phys * BLOCK_SIZE], barrier=vt_full[0]
                )
            elif not DIRECT_GLOBAL_V_RS:
                tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
        while start < seq_len:
            local_n = start + offs_n
            if ALIGNED_FULL_CHUNK_TOKENS:
                valid_cols = tl.full((BLOCK_N,), True, tl.int1)
            elif not STATIC_MTP1_NO_CAUSAL_MASK:
                valid_cols = local_n < seq_len
            buf = copy_iter % _compute_mtp1___TMA_STAGES_JIT
            phase = copy_iter // _compute_mtp1___TMA_STAGES_JIT & 1
            next_start = start + BLOCK_N
            page_next_phys = page_current_phys
            if next_start < seq_len:
                next_iter = copy_iter + 1
                next_buf = next_iter % _compute_mtp1___TMA_STAGES_JIT
                aligned_logical = seq_start + next_start
                if STATIC_BLOCK_IDS_PREFETCH:
                    next_block_in_chunk = next_start // BLOCK_SIZE
                    phys = tl.load(tle.gpu.local_ptr(chunk_block_ids_smem, (next_block_in_chunk,)))
                else:
                    block_no = aligned_logical // BLOCK_SIZE
                    phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                if PAGE_METADATA_K_SCALE:
                    page_next_phys = phys
                tle.gpu.copy(
                    K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
                )
                if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                    tle.gpu.copy(
                        KS_DESC, ks_smem.slot(next_buf), [1, 2, 1, 32], [phys, 0, hkv, 0], barrier=ks_full[next_buf]
                    )
                if TMA_DN_RS:
                    tle.gpu.copy(
                        VT_DESC,
                        v_dn_smem.slot(next_buf),
                        [DV, BLOCK_N],
                        [hkv * DV, phys * BLOCK_SIZE],
                        barrier=vt_full[next_buf],
                    )
                elif not DIRECT_GLOBAL_V_RS:
                    tle.gpu.copy(
                        VT_DESC,
                        v_raw_smem.slot(next_buf),
                        [1, 1, BLOCK_N, DV],
                        [phys, hkv, 0, 0],
                        barrier=vt_full[next_buf],
                    )
            tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                tle.gpu.barrier_wait(ks_full[buf], phaseIdx=phase)
                tile_kscale = tl.reshape(tl.load(tle.gpu.local_ptr(ks_smem.slot(buf))), (BLOCK_N,))
            k_page = k_raw_smem.slot(buf)
            if (
                LDSM_REGISTER_SHARED
                or TLE_SHARED_SHARED
                or FULL_VIEW_V_RS
                or TMA_DN_RS
                or DIRECT_GLOBAL_V_RS
                or FRAGMENT_PIPELINED_RS
                or K32_PIPELINED_RS
            ):
                scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            else:
                scores = tl.zeros((BLOCK_N, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                for frag in tl.static_range(0, 4):
                    k_frag = _compute_mtp1___memdesc_subslice(
                        k_page, (BLOCK_N, _compute_mtp1___K_FRAGMENT_JIT), (0, frag * _compute_mtp1___K_FRAGMENT_JIT)
                    )
                    q_frag = _compute_mtp1___memdesc_subslice(
                        q_smem,
                        (_compute_mtp1___ROWS_Q_JIT, _compute_mtp1___K_FRAGMENT_JIT),
                        (0, frag * _compute_mtp1___K_FRAGMENT_JIT),
                    )
                    scores = tle.gpu.wgmma(k_frag, q_frag, acc=scores, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores)
            if K_PER_TOKEN_V_PER_HEAD:
                if not TMA_K_SCALE:
                    scale_phys = page_current_phys
                    if not PAGE_METADATA_K_SCALE:
                        scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
                    tile_kscale = _compute_mtp1___load_packed_k_scale_mtp1(
                        KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                scores = scores * tile_kscale[:, None]
            elif not PRECOMBINE_Q_SCALE:
                scores = scores * kscale
            if not PRECOMBINE_Q_SCALE:
                scores = scores * (inv_sqrt_d * 1.4426950408889634)
            scores = scores * qscale[None, :]
            if STATIC_MTP1_NO_CAUSAL_MASK:
                if next_start <= seq_len:
                    score_mask = tl.broadcast_to(valid_q[None, :], (BLOCK_N, _compute_mtp1___ROWS_Q_JIT))
                else:
                    tail_valid_cols = local_n < seq_len
                    score_mask = tail_valid_cols[:, None] & valid_q[None, :]
            else:
                causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
                score_mask = valid_cols[:, None] & causal & valid_q[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float("inf")
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            tl.store(p_smem_ptr, p_scaled_t)
            if not DIRECT_GLOBAL_V_RS:
                tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            if DIRECT_GLOBAL_V_RS:
                current_block = (seq_start + start) // BLOCK_SIZE
                current_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + current_block)
                direct_v_ptrs = (
                    V_CACHE
                    + current_phys * V_STRIDE_BLOCK
                    + offs_n[None, :] * V_STRIDE_TOKEN
                    + hkv * V_STRIDE_HEAD
                    + offs_v[:, None] * V_STRIDE_D
                )
                v_page_reg_t = tl.load(direct_v_ptrs)
                pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
                v_page = k_page
            elif TMA_DN_RS:
                v_dn_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, BLOCK_N))
                v_dn_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV, BLOCK_N))
                v_page_reg_t = tl.load(tle.gpu.local_ptr(v_dn_smem.slot(buf), (v_dn_rows, v_dn_cols)))
                pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
                v_page = k_page
            else:
                v_page = v_raw_smem.slot(buf)
            if TLE_SHARED_SHARED:
                vt_page = _compute_mtp1___memdesc_transpose_2d(v_page)
                pv = tle.gpu.wgmma(vt_page, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            if FRAGMENT_PIPELINED_RS:
                v_page_t = _compute_mtp1___memdesc_transpose_2d(v_page)
                pv = tl.zeros((DV, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                for d_frag in tl.static_range(0, 2):
                    pv_frag = tl.zeros((DV // 2, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                    for n_frag in tl.static_range(0, 2):
                        pair_rows = tl.broadcast_to(
                            (d_frag * (DV // 2) + tl.arange(0, DV // 4) * 2)[:, None], (DV // 4, BLOCK_N // 2)
                        )
                        pair_cols = tl.broadcast_to(
                            (n_frag * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :], (DV // 4, BLOCK_N // 2)
                        )
                        pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                        pair_ptr = pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                        pairs = tl.load(pair_ptr)
                        lo = (pairs & 255).to(tl.uint8)
                        hi = (pairs >> 8).to(tl.uint8)
                        frag_bytes = tl.permute(tl.join(lo, hi), (0, 2, 1))
                        frag_a = tl.reshape(frag_bytes, (DV // 2, BLOCK_N // 2)).to(tl.float8e4nv, bitcast=True)
                        p_frag = _compute_mtp1___memdesc_subslice(
                            p_smem, (_compute_mtp1___ROWS_Q_JIT, BLOCK_N // 2), (0, n_frag * (BLOCK_N // 2))
                        )
                        pv_frag = tle.gpu.wgmma(frag_a, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                    pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                    pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            if K32_PIPELINED_RS:
                v_page_t = _compute_mtp1___memdesc_transpose_2d(v_page)
                pv = tl.zeros((DV, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                for n_frag in tl.static_range(0, 2):
                    pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N // 2))
                    pair_cols = tl.broadcast_to(
                        (n_frag * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :], (DV // 2, BLOCK_N // 2)
                    )
                    pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                    pair_ptr = pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                    pairs = tl.load(pair_ptr)
                    lo = (pairs & 255).to(tl.uint8)
                    hi = (pairs >> 8).to(tl.uint8)
                    frag_bytes = tl.permute(tl.join(lo, hi), (0, 2, 1))
                    frag_a = tl.reshape(frag_bytes, (DV, BLOCK_N // 2)).to(tl.float8e4nv, bitcast=True)
                    p_frag = _compute_mtp1___memdesc_subslice(
                        p_smem, (_compute_mtp1___ROWS_Q_JIT, BLOCK_N // 2), (0, n_frag * (BLOCK_N // 2))
                    )
                    pv = tle.gpu.wgmma(frag_a, p_frag, acc=pv, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            if LDSM_REGISTER_SHARED:
                v_page_t = _compute_mtp1___memdesc_transpose_2d(v_page)
                pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
                pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
                v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                v_pairs = tl.load(v_pair_ptr)
                v_lo = (v_pairs & 255).to(tl.uint8)
                v_hi = (v_pairs >> 8).to(tl.uint8)
                v_bytes = tl.join(v_lo, v_hi)
                v_bytes = tl.permute(v_bytes, (0, 2, 1))
                v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
                v_page_reg_t = v_bytes.to(tl.float8e4nv, bitcast=True)
            elif FULL_VIEW_V_RS:
                v_page_t = _compute_mtp1___memdesc_transpose_2d(v_page)
                v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t))
            elif (
                (not TLE_SHARED_SHARED)
                and (not TMA_DN_RS)
                and (not DIRECT_GLOBAL_V_RS)
                and (not FRAGMENT_PIPELINED_RS)
                and (not K32_PIPELINED_RS)
            ):
                v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
                v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
                v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                if INPLACE_PV_ACC:
                    acc = acc * alpha[None, :]
                    acc = tle.gpu.wgmma(v_page_reg_t, p_smem, acc=acc, trans_b=True, out_dtype=tl.float32)
                    acc = tle.gpu.wgmma_wait(0, acc)
                else:
                    pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                    pv = tle.gpu.wgmma_wait(0, pv)
            elif (
                (not TLE_SHARED_SHARED)
                and (not TMA_DN_RS)
                and (not DIRECT_GLOBAL_V_RS)
                and (not FRAGMENT_PIPELINED_RS)
                and (not K32_PIPELINED_RS)
            ):
                pv = tl.zeros((DV, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                for d_frag in tl.static_range(0, 2):
                    pv_frag = tl.zeros((_compute_mtp1___K_FRAGMENT_JIT * 2, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                    for n_frag in tl.static_range(0, 2):
                        v_reg_frag = tle.extract_tile(
                            v_page_reg_t,
                            index=[d_frag, n_frag],
                            tile_shape=[_compute_mtp1___K_FRAGMENT_JIT * 2, _compute_mtp1___K_FRAGMENT_JIT],
                        )
                        p_frag = _compute_mtp1___memdesc_subslice(
                            p_smem,
                            (_compute_mtp1___ROWS_Q_JIT, _compute_mtp1___K_FRAGMENT_JIT),
                            (0, n_frag * _compute_mtp1___K_FRAGMENT_JIT),
                        )
                        pv_frag = tle.gpu.wgmma(v_reg_frag, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                    pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                    pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            if not INPLACE_PV_ACC:
                acc = acc * alpha[None, :] + pv
            m_i = m_new
            l_i = l_new
            start = next_start
            copy_iter += 1
            if PAGE_METADATA_K_SCALE:
                page_current_phys = page_next_phys
        has_value = l_i > 0.0
        raw_l = l_i
        if DEFERRED_NORM and task_mode != _compute_mtp1___DIRECT_MODE_JIT:
            lse = m_i
        else:
            acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
            lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if EXECUTION_STAGE == _compute_mtp1___EXECUTION_LOCAL_PARTIAL_JIT:
        if cluster_rank == 0:
            local_output_mask = valid_q[None, :]
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                acc,
                mask=local_output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                lse,
                mask=valid_q,
            )
        return
    if not REDUCTION_ONLY and task_mode == _compute_mtp1___DIRECT_MODE_JIT:
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + store_offs_v[:, None],
            acc,
            mask=valid_q[None, :],
        )
        return
    group_acc = acc
    group_lse = lse
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        peer_l_remote = tle.remote(peer_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(peer_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max), 0.0)
            if DEFERRED_NORM:
                peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_q,)))
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            group_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 2:
            tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight1[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight2[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight3[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        if DEFERRED_NORM:
            tl.store(partial_l_ptr, raw_l)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        peer1_l = tle.remote(partial_l_smem, 1, scope=mesh)
        peer2_l = tle.remote(partial_l_smem, 2, scope=mesh)
        peer3_l = tle.remote(partial_l_smem, 3, scope=mesh)
        peer4_l = tle.remote(partial_l_smem, 4, scope=mesh)
        peer5_l = tle.remote(partial_l_smem, 5, scope=mesh)
        peer6_l = tle.remote(partial_l_smem, 6, scope=mesh)
        peer7_l = tle.remote(partial_l_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float("inf"), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float("inf"), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float("inf"), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float("inf"), tl.exp2(lse7 - safe_max), 0.0)
            if DEFERRED_NORM:
                l1 = tl.load(tle.gpu.local_ptr(peer1_l, (offs_q,)))
                l2 = tl.load(tle.gpu.local_ptr(peer2_l, (offs_q,)))
                l3 = tl.load(tle.gpu.local_ptr(peer3_l, (offs_q,)))
                l4 = tl.load(tle.gpu.local_ptr(peer4_l, (offs_q,)))
                l5 = tl.load(tle.gpu.local_ptr(peer5_l, (offs_q,)))
                l6 = tl.load(tle.gpu.local_ptr(peer6_l, (offs_q,)))
                l7 = tl.load(tle.gpu.local_ptr(peer7_l, (offs_q,)))
                denom = (
                    raw_l * weight0
                    + l1 * weight1
                    + l2 * weight2
                    + l3 * weight3
                    + l4 * weight4
                    + l5 * weight5
                    + l6 * weight6
                    + l7 * weight7
                )
            else:
                denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = ((seq_m < _compute_mtp1___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q))[None, :]
        if group_count == 1:
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                group_acc,
                mask=output_mask,
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                group_acc,
                mask=output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                group_lse,
                mask=(seq_m < _compute_mtp1___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
            )
    if EXECUTION_STAGE == _compute_mtp1___EXECUTION_FULL_JIT and group_count > 1:
        counter_idx = hkv * B + batch
        rank0_is_last = tl.zeros((), tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_ELECTION:
                deterministic_owner = group_chunk == group_count - 1
                if deterministic_owner:
                    ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(COMPLETION + counter_idx, 1, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        if RANK0_ONLY_FINALIZER:
            is_last_cluster = (cluster_rank == 0) & (rank0_is_last != 0)
        elif DETERMINISTIC_TAIL_ELECTION:
            tle.distributed_barrier(mesh)
            is_last_cluster = group_chunk == group_count - 1
        elif DSM_ELECTION_HANDOFF:
            if MERGE_CLUSTER_SIZE == 2:
                rank1_flag = tle.remote(peer_lse_smem, 1, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer_lse_smem, (0,))) != 0.0
            elif MERGE_CLUSTER_SIZE == 4:
                rank1_flag = tle.remote(peer1_lse_smem, 1, scope=mesh)
                rank2_flag = tle.remote(peer1_lse_smem, 2, scope=mesh)
                rank3_flag = tle.remote(peer1_lse_smem, 3, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer1_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (0,))) != 0.0
            else:
                rank1_flag = tle.remote(partial_lse_smem, 1, scope=mesh)
                rank2_flag = tle.remote(partial_lse_smem, 2, scope=mesh)
                rank3_flag = tle.remote(partial_lse_smem, 3, scope=mesh)
                rank4_flag = tle.remote(partial_lse_smem, 4, scope=mesh)
                rank5_flag = tle.remote(partial_lse_smem, 5, scope=mesh)
                rank6_flag = tle.remote(partial_lse_smem, 6, scope=mesh)
                rank7_flag = tle.remote(partial_lse_smem, 7, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(partial_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank4_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank5_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank6_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank7_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(partial_lse_smem, (0,))) != 0.0
        else:
            tle.distributed_barrier(mesh)
            rank0_cta = cta - cluster_rank
            is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_last_cluster:
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
            if RANK0_ONLY_FINALIZER:
                _compute_mtp1___cluster_quad_head_two_chunk_finalize_mtp1(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    0,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
                _compute_mtp1___cluster_quad_head_two_chunk_finalize_mtp1(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    1,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
            elif PAIRED_HEAD_FINALIZE:
                _compute_mtp1___cluster_paired_head_finalize_mtp1(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    group_count,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                    MAX_FINAL_CHUNKS,
                    4,
                )
            elif C2_PAIRED_HEAD_FINALIZE:
                for pair_pass in tl.static_range(0, 2):
                    first_h_in_group = cluster_rank + pair_pass * 4
                    _compute_mtp1___cluster_paired_head_finalize_mtp1(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        first_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        2,
                    )
                    tl.debug_barrier()
            else:
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    _compute_mtp1___cluster_cooperative_finalize_mtp1(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        final_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        True,
                    )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")


_compute_mtp2__TASK_STRIDE = 12

_compute_mtp2__TASK_SLOTS = 2

_compute_mtp2__NUM_SEQ_Q = 2

_compute_mtp2__ROWS_Q = 16

_compute_mtp2__DIRECT_MODE = 0

_compute_mtp2__DUMMY_MODE = 2

_compute_mtp2__GROUP_MODE = 1

_compute_mtp2__EXECUTION_FULL = 0

_compute_mtp2__EXECUTION_CLUSTER_PARTIAL = 1

_compute_mtp2__EXECUTION_LOCAL_PARTIAL = 2

_compute_mtp2___TASK_STRIDE_JIT = tl.constexpr(_compute_mtp2__TASK_STRIDE)

_compute_mtp2___TASK_SLOTS_JIT = tl.constexpr(_compute_mtp2__TASK_SLOTS)

_compute_mtp2___ROWS_Q_JIT = tl.constexpr(_compute_mtp2__ROWS_Q)

_compute_mtp2___NUM_SEQ_Q_JIT = tl.constexpr(_compute_mtp2__NUM_SEQ_Q)

_compute_mtp2___TMA_STAGES_JIT = tl.constexpr(2)

_compute_mtp2___DIRECT_MODE_JIT = tl.constexpr(_compute_mtp2__DIRECT_MODE)

_compute_mtp2___DUMMY_MODE_JIT = tl.constexpr(_compute_mtp2__DUMMY_MODE)

_compute_mtp2___GROUP_MODE_JIT = tl.constexpr(_compute_mtp2__GROUP_MODE)

_compute_mtp2___K_FRAGMENT_JIT = tl.constexpr(32)

_compute_mtp2___EXECUTION_FULL_JIT = tl.constexpr(_compute_mtp2__EXECUTION_FULL)

_compute_mtp2___EXECUTION_LOCAL_PARTIAL_JIT = tl.constexpr(_compute_mtp2__EXECUTION_LOCAL_PARTIAL)


@triton.jit
def _compute_mtp2___load_packed_k_scale_mtp2(
    KSCALE,
    phys,
    hkv,
    offs_n,
    KS_STRIDE_BLOCK: tl.constexpr,
    KS_STRIDE_TOKEN: tl.constexpr,
    KS_STRIDE_HEAD: tl.constexpr,
    KS_STRIDE_D: tl.constexpr,
):
    byte_offset = (
        phys * KS_STRIDE_BLOCK + offs_n // 32 * KS_STRIDE_TOKEN + hkv * KS_STRIDE_HEAD + offs_n % 32 * 4 * KS_STRIDE_D
    )
    scale_ptr = (KSCALE + byte_offset).to(tl.pointer_type(tl.float32))
    return tl.load(scale_ptr).to(tl.float32)


@triton.jit
def _compute_mtp2___head_sharded_dsm_merge_mtp2(
    PARTIAL_ACC_SMEM,
    PARTIAL_LSE_SMEM,
    HEAD_ACC_SMEM,
    HEAD_LSE_SMEM,
    mesh: tl.constexpr,
    h_in_group,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    MERGE_CLUSTER_SIZE: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Merge one GQA head across all CTA ranks using owner-local DSM."""
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    cols = h_in_group + offs_m * HEADS_PER_GROUP
    acc_rows = tl.broadcast_to(offs_v[:, None], (DV, _compute_mtp2___NUM_SEQ_Q_JIT))
    acc_cols = tl.broadcast_to(cols[None, :], (DV, _compute_mtp2___NUM_SEQ_Q_JIT))
    merged_acc = tl.zeros((DV, _compute_mtp2___NUM_SEQ_Q_JIT), tl.float32)
    merged_lse = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), -float("inf"), tl.float32)
    for peer_rank in tl.static_range(0, MERGE_CLUSTER_SIZE):
        peer_acc_smem = tle.remote(PARTIAL_ACC_SMEM, peer_rank, scope=mesh)
        peer_lse_smem = tle.remote(PARTIAL_LSE_SMEM, peer_rank, scope=mesh)
        peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
        peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (cols,)))
        max_lse = tl.maximum(merged_lse, peer_lse)
        valid = max_lse != -float("inf")
        safe_max = tl.where(valid, max_lse, 0.0)
        old_weight = tl.where(
            merged_lse != -float("inf"),
            tl.exp2(merged_lse - safe_max) if USE_LOG2 else tl.exp(merged_lse - safe_max),
            0.0,
        )
        peer_weight = tl.where(
            peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max) if USE_LOG2 else tl.exp(peer_lse - safe_max), 0.0
        )
        denom = old_weight + peer_weight
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        merged_acc = (merged_acc * old_weight[None, :] + peer_acc * peer_weight[None, :]) / safe_denom[None, :]
        merged_acc = tl.where(valid[None, :], merged_acc, 0.0)
        merged_lse = tl.where(
            valid, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
        )
    out_cols = tl.broadcast_to(offs_m[None, :], (DV, _compute_mtp2___NUM_SEQ_Q_JIT))
    tl.store(tle.gpu.local_ptr(HEAD_ACC_SMEM, (acc_rows, out_cols)), merged_acc)
    tl.store(tle.gpu.local_ptr(HEAD_LSE_SMEM, (offs_m,)), merged_lse)
    tl.debug_barrier()


@triton.jit
def _compute_mtp2___cluster_cooperative_finalize_mtp2(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    REUSE_FINAL_WEIGHTS: tl.constexpr,
    ROW_SERIAL_FINALIZE: tl.constexpr,
    TWO_CHUNK_FINALIZE: tl.constexpr,
):
    """Use one CTA rank per GQA head to finalize both MTP rows in-kernel."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None] < n_chunks) & valid_head
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :]
    weights_unnorm = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights_unnorm = tl.where(chunk_mask, weights_unnorm, 0.0)
    denom = tl.sum(weights_unnorm, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    if REUSE_FINAL_WEIGHTS:
        weight_rows = tl.broadcast_to(offs_c[:, None], (MAX_FINAL_CHUNKS, _compute_mtp2___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(offs_m[None, :], (MAX_FINAL_CHUNKS, _compute_mtp2___NUM_SEQ_Q_JIT))
        weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols))
        tl.store(weight_ptr, weights_unnorm * inv_denom[None, :])
        tl.debug_barrier()
    if ROW_SERIAL_FINALIZE:
        for seq_m in tl.static_range(0, _compute_mtp2___NUM_SEQ_Q_JIT):
            acc_row = tl.zeros((DV,), tl.float32)
            denom_row = tl.sum(tl.where(offs_m == seq_m, denom, 0.0), axis=0)
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (chunk < n_chunks) & valid_head
                chunk_rows = tl.full((1,), chunk, tl.int32)
                chunk_cols = tl.full((1,), seq_m, tl.int32)
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, chunk_cols)), mask=chunk_valid, other=0.0
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + seq_m * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v,
                    mask=chunk_valid,
                    other=0.0,
                )
                chunk_weight_scalar = tl.sum(chunk_weight, axis=0)
                acc_row += partial * chunk_weight_scalar
            tl.store(
                OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq * O_STRIDE_H + offs_v,
                acc_row,
                mask=valid_head & (denom_row > 0.0),
            )
        tl.debug_barrier()
    else:
        acc = tl.zeros((DV, _compute_mtp2___NUM_SEQ_Q_JIT), tl.float32)
        if TWO_CHUNK_FINALIZE and MAX_FINAL_CHUNKS >= 2:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS, 2):
                valid0 = (chunk < n_chunks) & valid_head
                valid1 = (chunk + 1 < n_chunks) & valid_head
                rows0 = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), chunk, tl.int32)
                rows1 = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), chunk + 1, tl.int32)
                weight0 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows0, offs_m)), mask=valid0, other=0.0)
                weight1 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows1, offs_m)), mask=valid1, other=0.0)
                partial0 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, :] * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=valid0,
                    other=0.0,
                )
                partial1 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + (chunk + 1) * SO_STRIDE_C
                    + offs_m[None, :] * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=valid1,
                    other=0.0,
                )
                acc += partial0 * weight0[None, :] + partial1 * weight1[None, :]
        else:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (chunk < n_chunks) & valid_head
                if REUSE_FINAL_WEIGHTS:
                    chunk_rows = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), chunk, tl.int32)
                    chunk_weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, offs_m))
                    chunk_weight = tl.load(chunk_weight_ptr, mask=chunk_valid, other=0.0)
                else:
                    chunk_lse = tl.load(
                        LSE
                        + batch * LSE_STRIDE_B
                        + chunk * LSE_STRIDE_C
                        + hkv * LSE_STRIDE_HKV
                        + offs_m * LSE_STRIDE_M
                        + h_in_group * LSE_STRIDE_HG,
                        mask=chunk_valid,
                        other=-float("inf"),
                    )
                    chunk_delta = chunk_lse - safe_max_lse
                    chunk_weight = (tl.exp2(chunk_delta) if USE_LOG2 else tl.exp(chunk_delta)) * inv_denom
                    chunk_weight = tl.where(chunk_valid, chunk_weight, 0.0)
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, :] * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=chunk_valid,
                    other=0.0,
                )
                acc += partial * chunk_weight[None, :]
        if REUSE_FINAL_WEIGHTS:
            tl.debug_barrier()
        tl.store(
            OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
            acc,
            mask=valid_head & (denom[None, :] > 0.0),
        )


@triton.jit
def _compute_mtp2___cluster_quad_head_two_chunk_finalize_mtp2(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    logical_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize four C2-owned heads and exactly two global partials."""
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = logical_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weight0 = tl.exp2(lse0 - safe_max_lse) if USE_LOG2 else tl.exp(lse0 - safe_max_lse)
    weight1 = tl.exp2(lse1 - safe_max_lse) if USE_LOG2 else tl.exp(lse1 - safe_max_lse)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    acc = tl.load(partial_base, mask=partial_mask, other=0.0)
    acc *= weight0[None, :, :]
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0)
    acc += partial1 * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp2___cluster_paired_head_finalize_mtp2(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    ROW_SERIAL_FINALIZE: tl.constexpr,
    TWO_CHUNK_FINALIZE: tl.constexpr,
):
    """Finalize the two c4-owned GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp2___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp2___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp2___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    if ROW_SERIAL_FINALIZE:
        for seq_m in tl.static_range(0, _compute_mtp2___NUM_SEQ_Q_JIT):
            acc_row = tl.zeros((DV, 2), tl.float32)
            pair_cols = offs_h * _compute_mtp2___NUM_SEQ_Q_JIT + seq_m
            denom_row = tl.sum(tl.where(offs_m[None, :] == seq_m, denom, 0.0), axis=1)
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (chunk < n_chunks) & valid_head
                chunk_rows = tl.full((2,), chunk, tl.int32)
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, pair_cols)), mask=chunk_valid, other=0.0
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + seq_m * SO_STRIDE_M
                    + hq[None, :] * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=chunk_valid[None, :],
                    other=0.0,
                )
                acc_row += partial * chunk_weight[None, :]
            tl.store(
                OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[None, :] * O_STRIDE_H + offs_v[:, None],
                acc_row,
                mask=valid_head[None, :] & (denom_row[None, :] > 0.0),
            )
        tl.debug_barrier()
    else:
        acc = tl.zeros((DV, 2, _compute_mtp2___NUM_SEQ_Q_JIT), tl.float32)
        pair_cols = offs_h[:, None] * _compute_mtp2___NUM_SEQ_Q_JIT + offs_m[None, :]
        if TWO_CHUNK_FINALIZE and MAX_FINAL_CHUNKS >= 2:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS, 2):
                valid0 = (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
                valid1 = (
                    (chunk + 1 < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
                )
                rows0 = tl.full((2, _compute_mtp2___NUM_SEQ_Q_JIT), chunk, tl.int32)
                rows1 = tl.full((2, _compute_mtp2___NUM_SEQ_Q_JIT), chunk + 1, tl.int32)
                weight0 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows0, pair_cols)), mask=valid0, other=0.0)
                weight1 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows1, pair_cols)), mask=valid1, other=0.0)
                partial0 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, None, :] * SO_STRIDE_M
                    + hq[None, :, None] * SO_STRIDE_H
                    + offs_v[:, None, None],
                    mask=valid0[None, :, :],
                    other=0.0,
                )
                partial1 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + (chunk + 1) * SO_STRIDE_C
                    + offs_m[None, None, :] * SO_STRIDE_M
                    + hq[None, :, None] * SO_STRIDE_H
                    + offs_v[:, None, None],
                    mask=valid1[None, :, :],
                    other=0.0,
                )
                acc += partial0 * weight0[None, :, :] + partial1 * weight1[None, :, :]
        else:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (
                    (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
                )
                chunk_rows = tl.full((2, _compute_mtp2___NUM_SEQ_Q_JIT), chunk, tl.int32)
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, pair_cols)), mask=chunk_valid, other=0.0
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, None, :] * SO_STRIDE_M
                    + hq[None, :, None] * SO_STRIDE_H
                    + offs_v[:, None, None],
                    mask=chunk_valid[None, :, :],
                    other=0.0,
                )
                acc += partial * chunk_weight[None, :, :]
        tl.debug_barrier()
        tl.store(
            OUT
            + batch * O_STRIDE_B
            + offs_m[None, None, :] * O_STRIDE_M
            + hq[None, :, None] * O_STRIDE_H
            + offs_v[:, None, None],
            acc,
            mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
        )


@builtin
def _compute_mtp2___memdesc_subslice(value, shape: tl.constexpr, offsets: tl.constexpr, _semantic=None):
    """Use FlagTree's existing ttg.memdesc_subslice builder binding."""
    shape = [int(tl_core._unwrap_if_constexpr(dim)) for dim in shape]
    layout = value.type.layout
    result_ty = gpu_types.buffered_tensor_type(
        value.dtype, shape, value.type.storage, layout, _semantic, alloc_shape=value.type.alloc_shape
    )
    handle = _semantic.builder.create_memdesc_subslice(result_ty.to_ir(_semantic.builder), value.handle, list(offsets))
    return gpu_types.buffered_tensor(
        handle, value.dtype, shape, value.type.storage, layout, _semantic, alloc_shape=value.type.alloc_shape
    )


@builtin
def _compute_mtp2___memdesc_transpose_2d(value, _semantic=None):
    """Return a zero-copy transposed view of a rank-2 shared memdesc.

    Unlike ``tl.trans(local_load(...))``, this exposes the transpose to the
    local-load lowering.  The intended transposed-memdesc lowering is therefore a direct
    transposed shared-to-register load from the TMA buffer, with no temporary
    shared allocation between the load and RS WGMMA.
    """
    if len(value.type.shape) != 2:
        raise ValueError("_memdesc_transpose_2d expects a rank-2 buffer")
    order = [1, 0]
    handle = _semantic.builder.create_memdesc_trans(value.handle, order)
    shape = [value.type.shape[i] for i in order]
    alloc_shape = value.type.alloc_shape
    leading_rank = len(alloc_shape) - len(value.type.shape)
    alloc_tail = alloc_shape[leading_rank:]
    transposed_alloc_shape = alloc_shape[:leading_rank] + [alloc_tail[i] for i in order]
    layout = value.type.layout.make_permute(order)
    return gpu_types.buffered_tensor(
        handle, value.dtype, shape, value.type.storage, layout, _semantic, alloc_shape=transposed_alloc_shape
    )


@triton.jit
def _compute_mtp2__fp8_kvpertensor_decode_mtp2_final_kernel(
    Q,
    K_DESC,
    KS_DESC,
    VT_DESC,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    K_PER_TOKEN_V_PER_HEAD: tl.constexpr = False,
    TMA_K_SCALE: tl.constexpr = False,
    PAGE_METADATA_K_SCALE: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    RAW_PAGED_NHD_OR_HND: tl.constexpr = False,
    FULL_MATRIX_RS: tl.constexpr = False,
    TRANSPOSED_MEMDESC_RS: tl.constexpr = False,
    DIRECT_V_SHARED_SHARED: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    TLE_SHARED_SHARED: tl.constexpr = False,
    FULL_VIEW_V_RS: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp2__EXECUTION_FULL,
    CLUSTER_COOPERATIVE_FINALIZE: tl.constexpr = False,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    REUSE_FINAL_WEIGHTS: tl.constexpr = False,
    HEAD_SHARDED_DSM: tl.constexpr = False,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    C8_PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    ROW_SERIAL_FINALIZE: tl.constexpr = False,
    TWO_CHUNK_FINALIZE: tl.constexpr = False,
    BF16_DSM: tl.constexpr = False,
    DEFERRED_NORM: tl.constexpr = False,
    PDL_NOTIFY: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr = False,
    TAIL_ONLY_ELECTION_BARRIER: tl.constexpr = False,
    REDUCTION_ONLY: tl.constexpr = False,
    ALIGNED_FULL_CHUNK_TOKENS: tl.constexpr = 0,
    FULL_VIEW_DSM: tl.constexpr = False,
    GLOBAL_DEFERRED_NORM: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    task_base = (cta * _compute_mtp2___TASK_SLOTS_JIT + 1) * _compute_mtp2___TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task_base + 0)
    batch = tl.load(TASK_MAP + task_base + 1)
    if not REDUCTION_ONLY and hkv < 0:
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    seq_start = tl.load(TASK_MAP + task_base + 3)
    if ALIGNED_FULL_CHUNK_TOKENS:
        seq_len = ALIGNED_FULL_CHUNK_TOKENS
    else:
        seq_len = tl.load(TASK_MAP + task_base + 4)
    seq_kvcache = tl.load(TASK_MAP + task_base + 5)
    is_causal = tl.load(TASK_MAP + task_base + 8)
    if REDUCTION_ONLY:
        task_mode = _compute_mtp2___GROUP_MODE_JIT
    else:
        task_mode = tl.load(TASK_MAP + task_base + 9)
    group_chunk = tl.load(TASK_MAP + task_base + 10)
    group_count = tl.load(TASK_MAP + task_base + 11)
    has_work = True if REDUCTION_ONLY else task_mode != _compute_mtp2___DUMMY_MODE_JIT
    q_smem = tle.gpu.alloc([_compute_mtp2___ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_compute_mtp2___ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc(
        [_compute_mtp2___TMA_STAGES_JIT, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    v_raw_smem = tle.gpu.alloc(
        [_compute_mtp2___TMA_STAGES_JIT, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp2___TMA_STAGES_JIT, arrive_count=1, expect_bytes=BLOCK_N * D
    )
    vt_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp2___TMA_STAGES_JIT, arrive_count=1, expect_bytes=DV * BLOCK_N
    )
    if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
        ks_smem = tle.gpu.alloc(
            [_compute_mtp2___TMA_STAGES_JIT, 1, 2, 1, 32], dtype=tl.float32, layout=None, scope=tle.gpu.smem
        )
        ks_full = tle.gpu.alloc_barriers(
            num_barriers=_compute_mtp2___TMA_STAGES_JIT, arrive_count=1, expect_bytes=2 * 32 * 4
        )
    if HEAD_SHARDED_DSM:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        head_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___NUM_SEQ_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        head_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___NUM_SEQ_Q_JIT],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        if MERGE_CLUSTER_SIZE == 2:
            peer_acc_smem = partial_acc_smem
            peer_lse_smem = partial_lse_smem
            peer_l_smem = partial_lse_smem
        elif MERGE_CLUSTER_SIZE == 4:
            peer1_acc_smem = partial_acc_smem
            peer2_acc_smem = partial_acc_smem
            peer3_acc_smem = partial_acc_smem
            peer1_lse_smem = partial_lse_smem
            peer2_lse_smem = partial_lse_smem
            peer3_lse_smem = partial_lse_smem
            peer1_l_smem = partial_lse_smem
            peer2_l_smem = partial_lse_smem
            peer3_l_smem = partial_lse_smem
    elif MERGE_CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            peer_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer_l_smem = peer_lse_smem
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer2_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer3_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer1_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer2_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer3_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            peer1_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer1_l_smem = peer1_lse_smem
            peer2_l_smem = peer2_lse_smem
            peer3_l_smem = peer3_lse_smem
    else:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    if CLUSTER_COOPERATIVE_FINALIZE:
        final_weight_smem = tle.gpu.alloc(
            [
                MAX_FINAL_CHUNKS if REUSE_FINAL_WEIGHTS else 1,
                2 * _compute_mtp2___NUM_SEQ_Q_JIT
                if PAIRED_HEAD_FINALIZE or C8_PAIRED_HEAD_FINALIZE
                else _compute_mtp2___NUM_SEQ_Q_JIT,
            ],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    offs_q = tl.arange(0, _compute_mtp2___ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _compute_mtp2___ROWS_Q_JIT)[:, None], (_compute_mtp2___ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_compute_mtp2___ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _compute_mtp2___ROWS_Q_JIT)[:, None], (_compute_mtp2___ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_compute_mtp2___ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _compute_mtp2___ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _compute_mtp2___ROWS_Q_JIT)[None, :], (DV, _compute_mtp2___ROWS_Q_JIT))
    store_offs_v = offs_v
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if MERGE_CLUSTER_SIZE == 8 or HEAD_SHARDED_DSM:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + hkv if K_PER_TOKEN_V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = has_work & (seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _compute_mtp2___ROWS_Q_JIT), tl.float32)
    lse = tl.full((_compute_mtp2___ROWS_Q_JIT,), -float("inf"), tl.float32)
    raw_l = tl.zeros((_compute_mtp2___ROWS_Q_JIT,), tl.float32)
    if REDUCTION_ONLY or has_work:
        q = tl.load(
            Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
            mask=valid_q[:, None],
            other=0.0,
        )
        tl.store(q_smem_ptr, q)
        qscale = tl.load(
            QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0
        ).to(tl.float32)
        if PRECOMBINE_Q_SCALE:
            qscale = qscale * (inv_sqrt_d * 1.4426950408889634)
            if not K_PER_TOKEN_V_PER_HEAD:
                qscale = qscale * kscale
        m_i = tl.full((_compute_mtp2___ROWS_Q_JIT,), -float("inf"), tl.float32)
        l_i = tl.zeros((_compute_mtp2___ROWS_Q_JIT,), tl.float32)
        copy_iter = 0
        start = 0
        page_current_phys = tl.full((), 0, tl.int32)
        if start < seq_len:
            block_no = seq_start // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            if PAGE_METADATA_K_SCALE:
                page_current_phys = phys
            tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
            if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                tle.gpu.copy(KS_DESC, ks_smem.slot(0), [1, 2, 1, 32], [phys, 0, hkv, 0], barrier=ks_full[0])
        while start < seq_len:
            local_n = start + offs_n
            if ALIGNED_FULL_CHUNK_TOKENS:
                valid_cols = tl.full((BLOCK_N,), True, tl.int1)
            else:
                valid_cols = local_n < seq_len
            buf = copy_iter % _compute_mtp2___TMA_STAGES_JIT
            phase = copy_iter // _compute_mtp2___TMA_STAGES_JIT & 1
            next_start = start + BLOCK_N
            page_next_phys = page_current_phys
            if next_start < seq_len:
                next_iter = copy_iter + 1
                next_buf = next_iter % _compute_mtp2___TMA_STAGES_JIT
                aligned_logical = seq_start + next_start
                block_no = aligned_logical // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                if PAGE_METADATA_K_SCALE:
                    page_next_phys = phys
                tle.gpu.copy(
                    K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
                )
                tle.gpu.copy(
                    VT_DESC,
                    v_raw_smem.slot(next_buf),
                    [1, 1, BLOCK_N, DV],
                    [phys, hkv, 0, 0],
                    barrier=vt_full[next_buf],
                )
                if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                    tle.gpu.copy(
                        KS_DESC, ks_smem.slot(next_buf), [1, 2, 1, 32], [phys, 0, hkv, 0], barrier=ks_full[next_buf]
                    )
            tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                tle.gpu.barrier_wait(ks_full[buf], phaseIdx=phase)
                tile_kscale = tl.reshape(tl.load(tle.gpu.local_ptr(ks_smem.slot(buf))), (BLOCK_N,))
            k_page = k_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            else:
                scores = tl.zeros((BLOCK_N, _compute_mtp2___ROWS_Q_JIT), tl.float32)
                for frag in tl.static_range(0, 4):
                    k_frag = _compute_mtp2___memdesc_subslice(
                        k_page, (BLOCK_N, _compute_mtp2___K_FRAGMENT_JIT), (0, frag * _compute_mtp2___K_FRAGMENT_JIT)
                    )
                    q_frag = _compute_mtp2___memdesc_subslice(
                        q_smem,
                        (_compute_mtp2___ROWS_Q_JIT, _compute_mtp2___K_FRAGMENT_JIT),
                        (0, frag * _compute_mtp2___K_FRAGMENT_JIT),
                    )
                    scores = tle.gpu.wgmma(k_frag, q_frag, acc=scores, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores)
            if K_PER_TOKEN_V_PER_HEAD:
                if not TMA_K_SCALE:
                    scale_phys = page_current_phys
                    if not PAGE_METADATA_K_SCALE:
                        scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
                    tile_kscale = _compute_mtp2___load_packed_k_scale_mtp2(
                        KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                scores = scores * tile_kscale[:, None]
            elif not PRECOMBINE_Q_SCALE:
                scores = scores * kscale
            if not PRECOMBINE_Q_SCALE:
                scores = scores * (inv_sqrt_d * 1.4426950408889634)
            scores = scores * qscale[None, :]
            causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
            scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float("inf"))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float("inf")
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            tl.store(p_smem_ptr, p_scaled_t)
            tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            v_page = v_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED:
                v_page_t = _compute_mtp2___memdesc_transpose_2d(v_page)
                pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
                pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
                v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                v_pairs = tl.load(v_pair_ptr)
                v_lo = (v_pairs & 255).to(tl.uint8)
                v_hi = (v_pairs >> 8).to(tl.uint8)
                v_bytes = tl.join(v_lo, v_hi)
                v_bytes = tl.permute(v_bytes, (0, 2, 1))
                v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
                v_page_reg_t = v_bytes.to(tl.float8e4nv, bitcast=True)
            elif FULL_VIEW_V_RS:
                v_page_t = _compute_mtp2___memdesc_transpose_2d(v_page)
                v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t))
            else:
                v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
                v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
                v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            else:
                pv = tl.zeros((DV, _compute_mtp2___ROWS_Q_JIT), tl.float32)
                for d_frag in tl.static_range(0, 2):
                    pv_frag = tl.zeros((_compute_mtp2___K_FRAGMENT_JIT * 2, _compute_mtp2___ROWS_Q_JIT), tl.float32)
                    for n_frag in tl.static_range(0, 2):
                        v_reg_frag = tle.extract_tile(
                            v_page_reg_t,
                            index=[d_frag, n_frag],
                            tile_shape=[_compute_mtp2___K_FRAGMENT_JIT * 2, _compute_mtp2___K_FRAGMENT_JIT],
                        )
                        p_frag = _compute_mtp2___memdesc_subslice(
                            p_smem,
                            (_compute_mtp2___ROWS_Q_JIT, _compute_mtp2___K_FRAGMENT_JIT),
                            (0, n_frag * _compute_mtp2___K_FRAGMENT_JIT),
                        )
                        pv_frag = tle.gpu.wgmma(v_reg_frag, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                    pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                    pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            acc = acc * alpha[None, :] + pv
            m_i = m_new
            l_i = l_new
            start = next_start
            copy_iter += 1
            if PAGE_METADATA_K_SCALE:
                page_current_phys = page_next_phys
        has_value = l_i > 0.0
        raw_l = l_i
        if DEFERRED_NORM and task_mode != _compute_mtp2___DIRECT_MODE_JIT:
            lse = m_i
        else:
            acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
            lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if EXECUTION_STAGE == _compute_mtp2___EXECUTION_LOCAL_PARTIAL_JIT:
        if cluster_rank == 0:
            local_output_mask = valid_q[None, :]
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                acc,
                mask=local_output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                lse,
                mask=valid_q,
            )
        return
    if not REDUCTION_ONLY and task_mode == _compute_mtp2___DIRECT_MODE_JIT:
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + store_offs_v[:, None],
            acc,
            mask=valid_q[None, :],
        )
        return
    if HEAD_SHARDED_DSM:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tle.distributed_barrier(mesh)
        final_offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
        for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
            final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
            final_hq = hkv * HEADS_PER_GROUP + final_h_in_group
            valid_head = (final_h_in_group < HEADS_PER_GROUP) & (final_hq < H_Q)
            _compute_mtp2___head_sharded_dsm_merge_mtp2(
                partial_acc_smem,
                partial_lse_smem,
                head_acc_smem,
                head_lse_smem,
                mesh,
                final_h_in_group,
                HEADS_PER_GROUP,
                DV,
                MERGE_CLUSTER_SIZE,
                True,
            )
            head_rows = tl.broadcast_to(offs_v[:, None], (DV, _compute_mtp2___NUM_SEQ_Q_JIT))
            head_cols = tl.broadcast_to(final_offs_m[None, :], (DV, _compute_mtp2___NUM_SEQ_Q_JIT))
            head_acc = tl.load(tle.gpu.local_ptr(head_acc_smem, (head_rows, head_cols)))
            head_lse = tl.load(tle.gpu.local_ptr(head_lse_smem, (final_offs_m,)))
            if group_count == 1:
                tl.store(
                    OUT
                    + batch * O_STRIDE_B
                    + final_offs_m[None, :] * O_STRIDE_M
                    + final_hq * O_STRIDE_H
                    + store_offs_v[:, None],
                    head_acc,
                    mask=valid_head,
                )
            else:
                tl.store(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + group_chunk * SO_STRIDE_C
                    + final_offs_m[None, :] * SO_STRIDE_M
                    + final_hq * SO_STRIDE_H
                    + store_offs_v[:, None],
                    head_acc,
                    mask=valid_head,
                )
                tl.store(
                    LSE
                    + batch * LSE_STRIDE_B
                    + group_chunk * LSE_STRIDE_C
                    + hkv * LSE_STRIDE_HKV
                    + final_offs_m * LSE_STRIDE_M
                    + final_h_in_group * LSE_STRIDE_HG,
                    head_lse,
                    mask=valid_head,
                )
            tl.debug_barrier()
        tle.distributed_barrier(mesh)
        if group_count > 1:
            counter_idx = hkv * B + batch
            if cluster_rank == 0:
                tl.debug_barrier()
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                tl.atomic_xchg(LAST_FLAGS + cta, (ticket == group_count - 1).to(tl.int32), sem="release", scope="gpu")
            tle.distributed_barrier(mesh)
            rank0_cta = cta - cluster_rank
            is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            if is_last_cluster:
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    _compute_mtp2___cluster_cooperative_finalize_mtp2(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        final_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        REUSE_FINAL_WEIGHTS,
                        ROW_SERIAL_FINALIZE,
                        TWO_CHUNK_FINALIZE,
                    )
            tle.distributed_barrier(mesh)
            if cluster_rank == 0 and is_last_cluster:
                tl.debug_barrier()
                tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
        return
    group_acc = acc
    group_lse = lse
    group_raw_acc = acc
    group_raw_l = raw_l
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        peer_l_remote = tle.remote(peer_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem)).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem))
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
                peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max), 0.0)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem))
                else:
                    peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_q,)))
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            group_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        peer1_l_remote = tle.remote(peer1_l_smem, 0, scope=mesh)
        peer2_l_remote = tle.remote(peer2_l_smem, 0, scope=mesh)
        peer3_l_remote = tle.remote(peer3_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer1_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer1_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer1_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 2:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer2_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer2_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer2_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer2_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 3:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer3_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer3_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer3_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer3_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            if FULL_VIEW_DSM:
                lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem))
                lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem))
                lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem))
            else:
                lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
                lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
                lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem))
                    l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem))
                    l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem))
                else:
                    l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)))
                    l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem, (offs_q,)))
                    l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem, (offs_q,)))
                denom = raw_l * weight0 + l1 * weight1 + l2 * weight2 + l3 * weight3
            else:
                denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight1[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight2[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight3[None, :]
            group_raw_acc = weighted_acc
            group_raw_l = denom
            if GLOBAL_DEFERRED_NORM:
                group_acc = weighted_acc
                group_lse = safe_max
            else:
                group_acc = weighted_acc / safe_denom[None, :]
                if DEFERRED_NORM:
                    group_acc *= vscale
                group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
                group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float("inf"), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float("inf"), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float("inf"), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float("inf"), tl.exp2(lse7 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = ((seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q))[None, :]
        if group_count == 1:
            output_acc = group_acc
            if GLOBAL_DEFERRED_NORM:
                output_denom = tl.where(group_raw_l > 0.0, group_raw_l, 1.0)
                output_acc = group_raw_acc / output_denom[None, :] * vscale
                output_acc = tl.where((group_raw_l > 0.0)[None, :], output_acc, 0.0)
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                output_acc,
                mask=output_mask,
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                group_acc,
                mask=output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                group_lse,
                mask=(seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
            )
            if GLOBAL_DEFERRED_NORM:
                tl.store(
                    LSE
                    + batch * LSE_STRIDE_B
                    + (group_chunk + MAX_FINAL_CHUNKS) * LSE_STRIDE_C
                    + hkv * LSE_STRIDE_HKV
                    + seq_m * LSE_STRIDE_M
                    + h_in_group * LSE_STRIDE_HG,
                    group_raw_l,
                    mask=(seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
                )
    if CLUSTER_COOPERATIVE_FINALIZE and EXECUTION_STAGE == _compute_mtp2___EXECUTION_FULL_JIT and (group_count > 1):
        counter_idx = hkv * B + batch
        rank0_is_last = tl.zeros((), tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_ELECTION:
                deterministic_owner = group_chunk == group_count - 1
                if deterministic_owner:
                    ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(COMPLETION + counter_idx, 1, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        if RANK0_ONLY_FINALIZER:
            is_last_cluster = (cluster_rank == 0) & (rank0_is_last != 0)
        elif DETERMINISTIC_TAIL_ELECTION:
            if TAIL_ONLY_ELECTION_BARRIER:
                if group_chunk == group_count - 1:
                    tle.distributed_barrier(mesh)
            else:
                tle.distributed_barrier(mesh)
            is_last_cluster = group_chunk == group_count - 1
        elif DSM_ELECTION_HANDOFF:
            if MERGE_CLUSTER_SIZE == 2:
                rank1_flag = tle.remote(peer_lse_smem, 1, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer_lse_smem, (0,))) != 0.0
            else:
                rank1_flag = tle.remote(peer1_lse_smem, 1, scope=mesh)
                rank2_flag = tle.remote(peer1_lse_smem, 2, scope=mesh)
                rank3_flag = tle.remote(peer1_lse_smem, 3, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer1_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (0,))) != 0.0
        else:
            tle.distributed_barrier(mesh)
            rank0_cta = cta - cluster_rank
            is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_last_cluster:
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
            if RANK0_ONLY_FINALIZER:
                _compute_mtp2___cluster_quad_head_two_chunk_finalize_mtp2(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    0,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
                _compute_mtp2___cluster_quad_head_two_chunk_finalize_mtp2(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    1,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
            elif C8_PAIRED_HEAD_FINALIZE:
                if cluster_rank < 4:
                    _compute_mtp2___cluster_paired_head_finalize_mtp2(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        ROW_SERIAL_FINALIZE,
                        TWO_CHUNK_FINALIZE,
                    )
            elif PAIRED_HEAD_FINALIZE:
                _compute_mtp2___cluster_paired_head_finalize_mtp2(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    group_count,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                    MAX_FINAL_CHUNKS,
                    ROW_SERIAL_FINALIZE,
                    TWO_CHUNK_FINALIZE,
                )
            else:
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    _compute_mtp2___cluster_cooperative_finalize_mtp2(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        final_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        REUSE_FINAL_WEIGHTS,
                        ROW_SERIAL_FINALIZE,
                        TWO_CHUNK_FINALIZE,
                    )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
    if PDL_NOTIFY:
        tl.extra.cuda.gdc_launch_dependents()


_compute_mtp4__TASK_STRIDE = 12

_compute_mtp4__TASK_SLOTS = 2

_compute_mtp4__NUM_SEQ_Q = 4

_compute_mtp4__ROWS_Q = 32

_compute_mtp4__DIRECT_MODE = 0

_compute_mtp4__GROUP_MODE = 1

_compute_mtp4__DUMMY_MODE = 2

_compute_mtp4__SUBGROUP2_MODE = 3

_compute_mtp4__EXECUTION_FULL = 0

_compute_mtp4__EXECUTION_CLUSTER_PARTIAL = 1

_compute_mtp4__EXECUTION_LOCAL_PARTIAL = 2

_compute_mtp4__EXECUTION_ELECTION_ONLY = 3

_compute_mtp4___TASK_STRIDE_JIT = tl.constexpr(_compute_mtp4__TASK_STRIDE)

_compute_mtp4___TASK_SLOTS_JIT = tl.constexpr(_compute_mtp4__TASK_SLOTS)

_compute_mtp4___ROWS_Q_JIT = tl.constexpr(_compute_mtp4__ROWS_Q)

_compute_mtp4___NUM_SEQ_Q_JIT = tl.constexpr(_compute_mtp4__NUM_SEQ_Q)

_compute_mtp4___DIRECT_MODE_JIT = tl.constexpr(_compute_mtp4__DIRECT_MODE)

_compute_mtp4___GROUP_MODE_JIT = tl.constexpr(_compute_mtp4__GROUP_MODE)

_compute_mtp4___DUMMY_MODE_JIT = tl.constexpr(_compute_mtp4__DUMMY_MODE)

_compute_mtp4___SUBGROUP2_MODE_JIT = tl.constexpr(_compute_mtp4__SUBGROUP2_MODE)

_compute_mtp4___K_FRAGMENT_JIT = tl.constexpr(32)

_compute_mtp4___EXECUTION_FULL_JIT = tl.constexpr(_compute_mtp4__EXECUTION_FULL)

_compute_mtp4___EXECUTION_LOCAL_PARTIAL_JIT = tl.constexpr(_compute_mtp4__EXECUTION_LOCAL_PARTIAL)

_compute_mtp4___EXECUTION_ELECTION_ONLY_JIT = tl.constexpr(_compute_mtp4__EXECUTION_ELECTION_ONLY)


@triton.jit
def _compute_mtp4___packed_k_scale_ptrs_mtp4(
    KSCALE,
    phys,
    hkv,
    offs_n,
    KS_STRIDE_BLOCK: tl.constexpr,
    KS_STRIDE_TOKEN: tl.constexpr,
    KS_STRIDE_HEAD: tl.constexpr,
    KS_STRIDE_D: tl.constexpr,
):
    byte_offset = (
        phys * KS_STRIDE_BLOCK + offs_n // 32 * KS_STRIDE_TOKEN + hkv * KS_STRIDE_HEAD + offs_n % 32 * 4 * KS_STRIDE_D
    )
    return (KSCALE + byte_offset).to(tl.pointer_type(tl.float32))


@triton.jit
def _compute_mtp4___load_packed_k_scale_mtp4(
    KSCALE,
    phys,
    hkv,
    offs_n,
    KS_STRIDE_BLOCK: tl.constexpr,
    KS_STRIDE_TOKEN: tl.constexpr,
    KS_STRIDE_HEAD: tl.constexpr,
    KS_STRIDE_D: tl.constexpr,
):
    scale_ptr = _compute_mtp4___packed_k_scale_ptrs_mtp4(
        KSCALE, phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
    )
    return tl.load(scale_ptr).to(tl.float32)


@triton.jit
def _compute_mtp4___head_sharded_dsm_merge_mtp4(
    PARTIAL_ACC_SMEM,
    PARTIAL_LSE_SMEM,
    HEAD_ACC_SMEM,
    HEAD_LSE_SMEM,
    mesh: tl.constexpr,
    h_in_group,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    MERGE_CLUSTER_SIZE: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Merge one GQA head across all CTA ranks using owner-local DSM."""
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    cols = h_in_group + offs_m * HEADS_PER_GROUP
    acc_rows = tl.broadcast_to(offs_v[:, None], (DV, _compute_mtp4___NUM_SEQ_Q_JIT))
    acc_cols = tl.broadcast_to(cols[None, :], (DV, _compute_mtp4___NUM_SEQ_Q_JIT))
    merged_acc = tl.zeros((DV, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    merged_lse = tl.full((_compute_mtp4___NUM_SEQ_Q_JIT,), -float("inf"), tl.float32)
    for peer_rank in tl.static_range(0, MERGE_CLUSTER_SIZE):
        peer_acc_smem = tle.remote(PARTIAL_ACC_SMEM, peer_rank, scope=mesh)
        peer_lse_smem = tle.remote(PARTIAL_LSE_SMEM, peer_rank, scope=mesh)
        peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
        peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (cols,)))
        max_lse = tl.maximum(merged_lse, peer_lse)
        valid = max_lse != -float("inf")
        safe_max = tl.where(valid, max_lse, 0.0)
        old_weight = tl.where(
            merged_lse != -float("inf"),
            tl.exp2(merged_lse - safe_max) if USE_LOG2 else tl.exp(merged_lse - safe_max),
            0.0,
        )
        peer_weight = tl.where(
            peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max) if USE_LOG2 else tl.exp(peer_lse - safe_max), 0.0
        )
        denom = old_weight + peer_weight
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        merged_acc = (merged_acc * old_weight[None, :] + peer_acc * peer_weight[None, :]) / safe_denom[None, :]
        merged_acc = tl.where(valid[None, :], merged_acc, 0.0)
        merged_lse = tl.where(
            valid, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
        )
    out_cols = tl.broadcast_to(offs_m[None, :], (DV, _compute_mtp4___NUM_SEQ_Q_JIT))
    tl.store(tle.gpu.local_ptr(HEAD_ACC_SMEM, (acc_rows, out_cols)), merged_acc)
    tl.store(tle.gpu.local_ptr(HEAD_LSE_SMEM, (offs_m,)), merged_lse)
    tl.debug_barrier()


@triton.jit
def _compute_mtp4___paired_head_sharded_dsm_merge_mtp4(
    PARTIAL_ACC_SMEM,
    PARTIAL_LSE_SMEM,
    HEAD_ACC_SMEM,
    HEAD_LSE_SMEM,
    mesh: tl.constexpr,
    cluster_rank,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Merge the two C4-owned heads across all ranks in one DSM pass."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    cols = h_in_group[:, None] + offs_m[None, :] * HEADS_PER_GROUP
    acc_rows = tl.broadcast_to(offs_v[:, None, None], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    acc_cols = tl.broadcast_to(cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    merged_acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    merged_lse = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
    for peer_rank in tl.static_range(0, 4):
        peer_acc_smem = tle.remote(PARTIAL_ACC_SMEM, peer_rank, scope=mesh)
        peer_lse_smem = tle.remote(PARTIAL_LSE_SMEM, peer_rank, scope=mesh)
        peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
        peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (cols,)))
        max_lse = tl.maximum(merged_lse, peer_lse)
        valid = max_lse != -float("inf")
        safe_max = tl.where(valid, max_lse, 0.0)
        old_weight = tl.where(
            merged_lse != -float("inf"),
            tl.exp2(merged_lse - safe_max) if USE_LOG2 else tl.exp(merged_lse - safe_max),
            0.0,
        )
        peer_weight = tl.where(
            peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max) if USE_LOG2 else tl.exp(peer_lse - safe_max), 0.0
        )
        denom = old_weight + peer_weight
        safe_denom = tl.where(denom > 0.0, denom, 1.0)
        merged_acc = (merged_acc * old_weight[None, :, :] + peer_acc * peer_weight[None, :, :]) / safe_denom[None, :, :]
        merged_acc = tl.where(valid[None, :, :], merged_acc, 0.0)
        merged_lse = tl.where(
            valid, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
        )
    out_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    out_acc_cols = tl.broadcast_to(out_cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    tl.store(tle.gpu.local_ptr(HEAD_ACC_SMEM, (acc_rows, out_acc_cols)), merged_acc)
    tl.store(tle.gpu.local_ptr(HEAD_LSE_SMEM, (out_cols,)), merged_lse)
    tl.debug_barrier()


@triton.jit
def _compute_mtp4___cluster_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    h_in_group,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize exactly two partials without the generic chunk loop/smem."""
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = LSE + batch * LSE_STRIDE_B + hkv * LSE_STRIDE_HKV + offs_m * LSE_STRIDE_M + h_in_group * LSE_STRIDE_HG
    lse0 = tl.load(lse_base, mask=valid_head, other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head, other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    delta0 = lse0 - safe_max_lse
    delta1 = lse1 - safe_max_lse
    weight0 = tl.exp2(delta0) if USE_LOG2 else tl.exp(delta0)
    weight1 = tl.exp2(delta1) if USE_LOG2 else tl.exp(delta1)
    weight0 = tl.where(valid_head, weight0, 0.0)
    weight1 = tl.where(valid_head, weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = SPLIT_OUT + batch * SO_STRIDE_B + offs_m[None, :] * SO_STRIDE_M + hq * SO_STRIDE_H + offs_v[:, None]
    partial0 = tl.load(partial_base, mask=valid_head, other=0.0)
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=valid_head, other=0.0)
    acc = partial0 * weight0[None, :] + partial1 * weight1[None, :]
    acc *= inv_denom[None, :]
    tl.store(
        OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_head & (denom[None, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize all four c2-owned heads and two partials in one pass."""
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    delta0 = lse0 - safe_max_lse
    delta1 = lse1 - safe_max_lse
    weight0 = tl.exp2(delta0) if USE_LOG2 else tl.exp(delta0)
    weight1 = tl.exp2(delta1) if USE_LOG2 else tl.exp(delta1)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    acc = tl.load(partial_base, mask=partial_mask, other=0.0)
    acc *= weight0[None, :, :]
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0)
    acc += partial1 * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize the two C4-owned heads and exactly two partials."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weight0 = tl.exp2(lse0 - safe_max_lse) if USE_LOG2 else tl.exp(lse0 - safe_max_lse)
    weight1 = tl.exp2(lse1 - safe_max_lse) if USE_LOG2 else tl.exp(lse1 - safe_max_lse)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    partial0 = tl.load(partial_base, mask=partial_mask, other=0.0)
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0)
    acc = partial0 * weight0[None, :, :]
    acc += partial1 * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_winner_local_reuse_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    local_acc_flat,
    local_lse_flat,
    batch,
    hkv,
    local_chunk,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Reuse the winning c2 chunk in registers and load only its peer."""
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_h = tl.arange(0, 8)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + offs_h
    valid_head = (offs_h < HEADS_PER_GROUP) & (hq < H_Q)
    peer_chunk = 1 - local_chunk
    local_acc = tl.reshape(local_acc_flat, (DV, _compute_mtp4___NUM_SEQ_Q_JIT, 8))
    local_lse = tl.reshape(local_lse_flat, (_compute_mtp4___NUM_SEQ_Q_JIT, 8))
    peer_lse = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + peer_chunk * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[:, None] * LSE_STRIDE_M
        + offs_h[None, :] * LSE_STRIDE_HG,
        mask=valid_head[None, :],
        other=-float("inf"),
    )
    max_lse = tl.where(local_lse > peer_lse, local_lse, peer_lse)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    local_delta = local_lse - safe_max_lse
    peer_delta = peer_lse - safe_max_lse
    local_weight = tl.exp2(local_delta) if USE_LOG2 else tl.exp(local_delta)
    peer_weight = tl.exp2(peer_delta) if USE_LOG2 else tl.exp(peer_delta)
    local_weight = tl.where(valid_head[None, :], local_weight, 0.0)
    peer_weight = tl.where(valid_head[None, :], peer_weight, 0.0)
    denom = local_weight + peer_weight
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    peer_acc = tl.load(
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + peer_chunk * SO_STRIDE_C
        + offs_m[None, :, None] * SO_STRIDE_M
        + hq[None, None, :] * SO_STRIDE_H
        + offs_v[:, None, None],
        mask=valid_head[None, None, :],
        other=0.0,
    )
    acc = local_acc * local_weight[None, :, :]
    acc += peer_acc * peer_weight[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, :, None] * O_STRIDE_M
        + hq[None, None, :] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, None, :] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_cooperative_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    REUSE_FINAL_WEIGHTS: tl.constexpr,
):
    """Use one CTA rank per GQA head to finalize both MTP rows in-kernel."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None] < n_chunks) & valid_head
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :]
    weights_unnorm = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights_unnorm = tl.where(chunk_mask, weights_unnorm, 0.0)
    denom = tl.sum(weights_unnorm, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    if REUSE_FINAL_WEIGHTS:
        weight_rows = tl.broadcast_to(offs_c[:, None], (MAX_FINAL_CHUNKS, _compute_mtp4___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(offs_m[None, :], (MAX_FINAL_CHUNKS, _compute_mtp4___NUM_SEQ_Q_JIT))
        weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols))
        tl.store(weight_ptr, weights_unnorm * inv_denom[None, :])
        tl.debug_barrier()
    acc = tl.zeros((DV, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head
        if REUSE_FINAL_WEIGHTS:
            chunk_rows = tl.full((_compute_mtp4___NUM_SEQ_Q_JIT,), chunk, tl.int32)
            chunk_weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, offs_m))
            chunk_weight = tl.load(chunk_weight_ptr, mask=chunk_valid, other=0.0)
        else:
            chunk_lse = tl.load(
                LSE
                + batch * LSE_STRIDE_B
                + chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + offs_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                mask=chunk_valid,
                other=-float("inf"),
            )
            chunk_delta = chunk_lse - safe_max_lse
            chunk_weight = (tl.exp2(chunk_delta) if USE_LOG2 else tl.exp(chunk_delta)) * inv_denom
            chunk_weight = tl.where(chunk_valid, chunk_weight, 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, :] * SO_STRIDE_M
            + hq * SO_STRIDE_H
            + offs_v[:, None],
            mask=chunk_valid,
            other=0.0,
        )
        acc += partial * chunk_weight[None, :]
    if REUSE_FINAL_WEIGHTS:
        tl.debug_barrier()
    tl.store(
        OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_head & (denom[None, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_finalize_dynamic_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    FINALIZER_STAGES: tl.constexpr,
    HEAD_STRIDE: tl.constexpr = 4,
    HEAD_BASE: tl.constexpr = 0,
):
    """Online-softmax C4 finalizer with a scalar pipelined chunk loop.

    Unlike the tiled implementation, this keeps only one partial chunk live.  It
    also avoids materializing the complete weight matrix in shared memory and
    the two CTA barriers needed to consume that matrix.
    """
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + HEAD_BASE + offs_h * HEAD_STRIDE
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    max_lse = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
    denom = tl.zeros((2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.range(0, n_chunks, num_stages=FINALIZER_STAGES):
        lse = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :] * LSE_STRIDE_M
            + h_in_group[:, None] * LSE_STRIDE_HG,
            mask=valid_head[:, None],
            other=-float("inf"),
        )
        next_max = tl.maximum(max_lse, lse)
        safe_next_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        old_scale = tl.where(
            max_lse == -float("inf"),
            0.0,
            tl.exp2(max_lse - safe_next_max) if USE_LOG2 else tl.exp(max_lse - safe_next_max),
        )
        chunk_scale = tl.where(
            valid_head[:, None], tl.exp2(lse - safe_next_max) if USE_LOG2 else tl.exp(lse - safe_next_max), 0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=valid_head[None, :, None],
            other=0.0,
        )
        acc = acc * old_scale[None, :, :] + partial * chunk_scale[None, :, :]
        denom = denom * old_scale + chunk_scale
        max_lse = next_max
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc * inv_denom[None, :, :],
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    FINALIZER_TILE: tl.constexpr = 1,
    HEAD_STRIDE: tl.constexpr = 4,
    HEAD_BASE: tl.constexpr = 0,
    CHUNK_BASE: tl.constexpr = 0,
):
    """Finalize two strided GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + HEAD_BASE + offs_h * HEAD_STRIDE
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + (offs_c[:, None, None] + CHUNK_BASE) * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    pair_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    tile_offs = tl.arange(0, FINALIZER_TILE)
    for chunk_base in tl.static_range(0, MAX_FINAL_CHUNKS, FINALIZER_TILE):
        chunks = chunk_base + tile_offs
        chunk_valid = (
            (chunks[:, None, None] < n_chunks)
            & valid_head[None, :, None]
            & (offs_m[None, None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
        )
        chunk_rows = tl.broadcast_to(chunks[:, None, None], (FINALIZER_TILE, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        chunk_cols = tl.broadcast_to(pair_cols[None, :, :], (FINALIZER_TILE, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, chunk_cols)), mask=chunk_valid, other=0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + (chunks[:, None, None, None] + CHUNK_BASE) * SO_STRIDE_C
            + offs_m[None, None, None, :] * SO_STRIDE_M
            + hq[None, None, :, None] * SO_STRIDE_H
            + offs_v[None, :, None, None],
            mask=chunk_valid[:, None, :, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(partial * chunk_weight[:, None, :, :], axis=0)
    tl.debug_barrier()
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_two_chunk_reuse_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    WINNER_ACC_SMEM,
    WINNER_M_SMEM,
    WINNER_L_SMEM,
    mesh: tl.constexpr,
    batch,
    hkv,
    cluster_rank,
    winner_chunk,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Exact-two raw merge reusing the winning partial from rank0 DSM.

    The mailbox aliases rank0's now-dead C4 peer-1 transport buffers.  The
    winning rank0 publishes its complete raw state there after the completion
    election and before the existing cluster handoff barrier, so this path
    adds no shared-memory allocation and only the winning cluster performs the
    extra BF16 store.
    """
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    source_cols = offs_m[None, :] * HEADS_PER_GROUP + h_in_group[:, None]
    source_rows = tl.broadcast_to(offs_v[:, None, None], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    source_cols_3d = tl.broadcast_to(source_cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    winner_acc_rank0 = tle.remote(WINNER_ACC_SMEM, 0, scope=mesh)
    winner_m_rank0 = tle.remote(WINNER_M_SMEM, 0, scope=mesh)
    winner_l_rank0 = tle.remote(WINNER_L_SMEM, 0, scope=mesh)
    local_acc = tl.load(tle.gpu.local_ptr(winner_acc_rank0, (source_rows, source_cols_3d))).to(tl.float32)
    local_m = tl.load(tle.gpu.local_ptr(winner_m_rank0, (source_cols,)))
    local_l = tl.load(tle.gpu.local_ptr(winner_l_rank0, (source_cols,)))
    peer_chunk = 1 - winner_chunk
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + peer_chunk * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    scalar_mask = valid_head[:, None]
    peer_m = tl.load(scalar_base, mask=scalar_mask, other=-float("inf"))
    peer_l = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    max_m = tl.maximum(local_m, peer_m)
    valid_local = local_m != -float("inf")
    valid_peer = peer_m != -float("inf")
    both_valid = valid_local & valid_peer
    safe_max = tl.where(max_m != -float("inf"), max_m, 0.0)
    min_m = tl.minimum(local_m, peer_m)
    peer_weight = tl.where(both_valid, tl.exp2(min_m - safe_max) if USE_LOG2 else tl.exp(min_m - safe_max), 0.0)
    local_is_max = local_m >= peer_m
    local_w = tl.where(valid_local, tl.where(local_is_max, 1.0, peer_weight), 0.0)
    global_w = tl.where(valid_peer, tl.where(local_is_max, peer_weight, 1.0), 0.0)
    global_acc = tl.load(
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + peer_chunk * SO_STRIDE_C
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None],
        mask=valid_head[None, :, None],
        other=0.0,
    ).to(tl.float32)
    denom = local_l * local_w + peer_l * global_w
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    acc = (local_acc * local_w[None, :, :] + global_acc * global_w[None, :, :]) * scale[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Exact-two global raw merge without generic weight staging."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    scalar_mask = valid_head[:, None]
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    m0 = tl.load(scalar_base, mask=scalar_mask, other=-float("inf"))
    m1 = tl.load(scalar_base + LSE_STRIDE_C, mask=scalar_mask, other=-float("inf"))
    l0 = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    l1 = tl.load(scalar_base + (MAX_FINAL_CHUNKS + 1) * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    max_m = tl.maximum(m0, m1)
    valid = max_m != -float("inf")
    safe_max = tl.where(valid, max_m, 0.0)
    valid0 = m0 != -float("inf")
    valid1 = m1 != -float("inf")
    both_valid = valid0 & valid1
    min_m = tl.minimum(m0, m1)
    peer_weight = tl.where(both_valid, tl.exp2(min_m - safe_max) if USE_LOG2 else tl.exp(min_m - safe_max), 0.0)
    first_is_max = m0 >= m1
    w0 = tl.where(valid0, tl.where(first_is_max, 1.0, peer_weight), 0.0)
    w1 = tl.where(valid1, tl.where(first_is_max, peer_weight, 1.0), 0.0)
    denom = l0 * w0 + l1 * w1
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    partial0 = tl.load(partial_base, mask=partial_mask, other=0.0).to(tl.float32)
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0).to(tl.float32)
    acc = (partial0 * w0[None, :, :] + partial1 * w1[None, :, :]) * scale[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=partial_mask & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    DUAL_ACC: tl.constexpr = False,
    CHUNK4_REDUCE: tl.constexpr = False,
    REGISTER_WEIGHTS: tl.constexpr = False,
    HEAD_MAJOR_REDUCE: tl.constexpr = False,
):
    """Merge global raw (numerator, m, l) partials and normalize once."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    if HEAD_MAJOR_REDUCE:
        chunk_mask = (offs_c[None, None, :] < n_chunks) & valid_head[:, None, None]
        scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + offs_c[None, None, :] * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :, None] * LSE_STRIDE_M
            + h_in_group[:, None, None] * LSE_STRIDE_HG
        )
        m_values = tl.load(scalar_base, mask=chunk_mask, other=-float("inf"))
        l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_mask, other=0.0)
        max_m = tl.max(m_values, axis=2)
    else:
        chunk_mask = (offs_c[:, None, None] < n_chunks) & valid_head[None, :, None]
        scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + offs_c[:, None, None] * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, None, :] * LSE_STRIDE_M
            + h_in_group[None, :, None] * LSE_STRIDE_HG
        )
        m_values = tl.load(scalar_base, mask=chunk_mask, other=-float("inf"))
        l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_mask, other=0.0)
        max_m = tl.max(m_values, axis=0)
    safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
    if HEAD_MAJOR_REDUCE:
        weights = tl.exp2(m_values - safe_max[:, :, None]) if USE_LOG2 else tl.exp(m_values - safe_max[:, :, None])
    else:
        weights = tl.exp2(m_values - safe_max[None, :, :]) if USE_LOG2 else tl.exp(m_values - safe_max[None, :, :])
    weights = tl.where(chunk_mask, weights, 0.0)
    if HEAD_MAJOR_REDUCE:
        denom = tl.sum(l_values * weights, axis=2)
    else:
        denom = tl.sum(l_values * weights, axis=0)
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    if HEAD_MAJOR_REDUCE:
        weight_rows = tl.broadcast_to(
            (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[:, :, None],
            (2, _compute_mtp4___NUM_SEQ_Q_JIT, MAX_FINAL_CHUNKS),
        )
        weight_cols = tl.broadcast_to(offs_c[None, None, :], (2, _compute_mtp4___NUM_SEQ_Q_JIT, MAX_FINAL_CHUNKS))
        scaled_weights = weights * scale[:, :, None]
    else:
        weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(
            (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
            (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
        )
        scaled_weights = weights * scale[None, :, :]
    if not REGISTER_WEIGHTS:
        tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), scaled_weights)
        tl.debug_barrier()
    pair_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    if CHUNK4_REDUCE:
        chunk_lane = tl.arange(0, 4)
        weight_cols4 = tl.broadcast_to(pair_cols[None, :, :], (4, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        for chunk_base in tl.static_range(0, MAX_FINAL_CHUNKS, 4):
            chunk_ids = chunk_base + chunk_lane
            chunk_valid = (chunk_ids[:, None, None] < n_chunks) & valid_head[None, :, None]
            weight_rows4 = tl.broadcast_to(chunk_ids[:, None, None], (4, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
            chunk_weight = tl.load(
                tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows4, weight_cols4)), mask=chunk_valid, other=0.0
            )
            partial = tl.load(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + chunk_ids[:, None, None, None] * SO_STRIDE_C
                + offs_v[None, :, None, None]
                + offs_m[None, None, None, :] * SO_STRIDE_M
                + hq[None, None, :, None] * SO_STRIDE_H,
                mask=chunk_valid[:, None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(partial * chunk_weight[:, None, :, :], axis=0)
    else:
        if DUAL_ACC:
            acc_odd = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
        for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
            chunk_valid = (chunk < n_chunks) & valid_head[:, None]
            if REGISTER_WEIGHTS:
                chunk_weight = tl.reshape(
                    tle.extract_tile(scaled_weights, [chunk, 0, 0], [1, 2, _compute_mtp4___NUM_SEQ_Q_JIT]),
                    (2, _compute_mtp4___NUM_SEQ_Q_JIT),
                )
            elif HEAD_MAJOR_REDUCE:
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(
                        FINAL_WEIGHT_SMEM, (pair_cols, tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32))
                    ),
                    mask=chunk_valid,
                    other=0.0,
                )
            else:
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(
                        FINAL_WEIGHT_SMEM, (tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32), pair_cols)
                    ),
                    mask=chunk_valid,
                    other=0.0,
                )
            partial = tl.load(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + chunk * SO_STRIDE_C
                + offs_m[None, None, :] * SO_STRIDE_M
                + hq[None, :, None] * SO_STRIDE_H
                + offs_v[:, None, None],
                mask=chunk_valid[None, :, :],
                other=0.0,
            ).to(tl.float32)
            if DUAL_ACC and chunk % 2 == 1:
                acc_odd += partial * chunk_weight[None, :, :]
            else:
                acc += partial * chunk_weight[None, :, :]
        if DUAL_ACC:
            acc += acc_odd
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_recompute_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Two-pass raw merge with recomputed weights and no shared rendezvous."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None, None] < n_chunks) & valid_head[None, :, None]
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG
    )
    m_values = tl.load(scalar_base, mask=chunk_mask, other=-float("inf"))
    l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_mask, other=0.0)
    max_m = tl.max(m_values, axis=0)
    safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
    reduction_weights = (
        tl.exp2(m_values - safe_max[None, :, :]) if USE_LOG2 else tl.exp(m_values - safe_max[None, :, :])
    )
    reduction_weights = tl.where(chunk_mask, reduction_weights, 0.0)
    denom = tl.sum(l_values * reduction_weights, axis=0)
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head[:, None]
        chunk_scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :] * LSE_STRIDE_M
            + h_in_group[:, None] * LSE_STRIDE_HG
        )
        chunk_m = tl.load(chunk_scalar_base, mask=chunk_valid, other=-float("inf"))
        chunk_weight = tl.where(
            chunk_valid, (tl.exp2(chunk_m - safe_max) if USE_LOG2 else tl.exp(chunk_m - safe_max)) * scale, 0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=chunk_valid[None, :, :],
            other=0.0,
        ).to(tl.float32)
        acc += partial * chunk_weight[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_tma_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    FINAL_ACC_SMEM,
    FINAL_ACC_FULL,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Two-stage TMA pipeline for the FP32 raw numerator finalizer."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    hq0 = hkv * HEADS_PER_GROUP + cluster_rank
    hq1 = hq0 + 4
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None, None] < n_chunks) & valid_head[None, :, None]
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG
    )
    m_values = tl.load(scalar_base, mask=chunk_mask, other=-float("inf"))
    l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_mask, other=0.0)
    max_m = tl.max(m_values, axis=0)
    safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
    weights = tl.exp2(m_values - safe_max[None, :, :]) if USE_LOG2 else tl.exp(m_values - safe_max[None, :, :])
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(l_values * weights, axis=0)
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * scale[None, :, :])
    tl.debug_barrier()
    partial_desc = tl.make_tensor_descriptor(
        SPLIT_OUT,
        shape=[B * 2 * MAX_FINAL_CHUNKS, _compute_mtp4___NUM_SEQ_Q_JIT, H_Q, DV],
        strides=[SO_STRIDE_C, SO_STRIDE_M, SO_STRIDE_H, 1],
        block_shape=[1, _compute_mtp4___NUM_SEQ_Q_JIT, 1, DV],
    )
    batch_chunk_base = batch * (2 * MAX_FINAL_CHUNKS)
    copy_iter = tl.full((), 0, tl.int32)
    acc0 = tl.zeros((DV, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    if n_chunks > 0:
        buf = copy_iter & 1
        tle.gpu.copy(
            partial_desc,
            FINAL_ACC_SMEM.slot(buf),
            [1, _compute_mtp4___NUM_SEQ_Q_JIT, 1, DV],
            [batch_chunk_base, 0, hq0, 0],
            barrier=FINAL_ACC_FULL[buf],
        )
    chunk = 0
    while chunk < n_chunks:
        buf = copy_iter & 1
        phase = copy_iter // 2 & 1
        tle.gpu.barrier_wait(FINAL_ACC_FULL[buf], phaseIdx=phase)
        next_chunk = chunk + 1
        if next_chunk < n_chunks:
            next_buf = copy_iter + 1 & 1
            tle.gpu.copy(
                partial_desc,
                FINAL_ACC_SMEM.slot(next_buf),
                [1, _compute_mtp4___NUM_SEQ_Q_JIT, 1, DV],
                [batch_chunk_base + next_chunk, 0, hq0, 0],
                barrier=FINAL_ACC_FULL[next_buf],
            )
        partial = tl.trans(
            tl.reshape(tl.load(tle.gpu.local_ptr(FINAL_ACC_SMEM.slot(buf))), (_compute_mtp4___NUM_SEQ_Q_JIT, DV))
        )
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (tl.full((_compute_mtp4___NUM_SEQ_Q_JIT,), chunk, tl.int32), offs_m))
        )
        acc0 += partial * chunk_weight[None, :]
        copy_iter += 1
        chunk += 1
    acc1 = tl.zeros((DV, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    if n_chunks > 0:
        buf = copy_iter & 1
        tle.gpu.copy(
            partial_desc,
            FINAL_ACC_SMEM.slot(buf),
            [1, _compute_mtp4___NUM_SEQ_Q_JIT, 1, DV],
            [batch_chunk_base, 0, hq1, 0],
            barrier=FINAL_ACC_FULL[buf],
        )
    chunk = 0
    while chunk < n_chunks:
        buf = copy_iter & 1
        phase = copy_iter // 2 & 1
        tle.gpu.barrier_wait(FINAL_ACC_FULL[buf], phaseIdx=phase)
        next_chunk = chunk + 1
        if next_chunk < n_chunks:
            next_buf = copy_iter + 1 & 1
            tle.gpu.copy(
                partial_desc,
                FINAL_ACC_SMEM.slot(next_buf),
                [1, _compute_mtp4___NUM_SEQ_Q_JIT, 1, DV],
                [batch_chunk_base + next_chunk, 0, hq1, 0],
                barrier=FINAL_ACC_FULL[next_buf],
            )
        partial = tl.trans(
            tl.reshape(tl.load(tle.gpu.local_ptr(FINAL_ACC_SMEM.slot(buf))), (_compute_mtp4___NUM_SEQ_Q_JIT, DV))
        )
        chunk_weight = tl.load(
            tle.gpu.local_ptr(
                FINAL_WEIGHT_SMEM,
                (tl.full((_compute_mtp4___NUM_SEQ_Q_JIT,), chunk, tl.int32), _compute_mtp4___NUM_SEQ_Q_JIT + offs_m),
            )
        )
        acc1 += partial * chunk_weight[None, :]
        copy_iter += 1
        chunk += 1
    acc = tl.permute(tl.join(acc0, acc1), (0, 2, 1))
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_streaming_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Single-pass online merge of global raw ``(numerator, m, l)`` states.

    Unlike the generic raw finalizer, this keeps only the running state in
    registers.  It does not materialize all chunk weights in shared memory,
    does not need the associated CTA barrier, and visits each raw partial
    exactly once.
    """
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    scalar_mask = valid_head[:, None]
    running_m = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
    running_l = tl.zeros((2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    running_acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & scalar_mask
        scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :] * LSE_STRIDE_M
            + h_in_group[:, None] * LSE_STRIDE_HG
        )
        chunk_m = tl.load(scalar_base, mask=chunk_valid, other=-float("inf"))
        chunk_l = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_valid, other=0.0)
        chunk_has_value = chunk_valid & (chunk_m != -float("inf"))
        running_has_value = running_m != -float("inf")
        new_m = tl.maximum(running_m, chunk_m)
        safe_new_m = tl.where(running_has_value | chunk_has_value, new_m, 0.0)
        running_weight = tl.where(
            running_has_value, tl.exp2(running_m - safe_new_m) if USE_LOG2 else tl.exp(running_m - safe_new_m), 0.0
        )
        chunk_weight = tl.where(
            chunk_has_value, tl.exp2(chunk_m - safe_new_m) if USE_LOG2 else tl.exp(chunk_m - safe_new_m), 0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=chunk_valid[None, :, :],
            other=0.0,
        ).to(tl.float32)
        running_acc = running_acc * running_weight[None, :, :] + partial * chunk_weight[None, :, :]
        running_l = running_l * running_weight + chunk_l * chunk_weight
        running_m = tl.where(running_has_value | chunk_has_value, new_m, running_m)
    scale = tl.where(running_l > 0.0, vscale / running_l, 0.0)
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        running_acc * scale[None, :, :],
        mask=valid_head[None, :, None] & (running_l[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_tail_reuse_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    WINNER_ACC_SMEM,
    WINNER_M_SMEM,
    WINNER_L_SMEM,
    mesh: tl.constexpr,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Merge global non-winners with raw tail state retained on rank 0."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    tail_chunk = n_chunks - 1
    global_mask = (offs_c[:, None, None] < tail_chunk) & valid_head[None, :, None]
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG
    )
    m_values = tl.load(scalar_base, mask=global_mask, other=-float("inf"))
    l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=global_mask, other=0.0)
    mailbox_cols = h_in_group[:, None] + offs_m[None, :] * HEADS_PER_GROUP
    winner_acc = tle.remote(WINNER_ACC_SMEM, 0, scope=mesh)
    winner_m = tle.remote(WINNER_M_SMEM, 0, scope=mesh)
    winner_l = tle.remote(WINNER_L_SMEM, 0, scope=mesh)
    local_m = tl.load(tle.gpu.local_ptr(winner_m, (mailbox_cols,)))
    local_l = tl.load(tle.gpu.local_ptr(winner_l, (mailbox_cols,)))
    global_max = tl.max(m_values, axis=0)
    max_m = tl.maximum(global_max, local_m)
    valid = max_m != -float("inf")
    safe_max = tl.where(valid, max_m, 0.0)
    weights = tl.exp2(m_values - safe_max[None, :, :]) if USE_LOG2 else tl.exp(m_values - safe_max[None, :, :])
    weights = tl.where(global_mask, weights, 0.0)
    local_weight = tl.where(
        local_m != -float("inf"), tl.exp2(local_m - safe_max) if USE_LOG2 else tl.exp(local_m - safe_max), 0.0
    )
    denom = tl.sum(l_values * weights, axis=0) + local_l * local_weight
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    normalized_weights = weights * scale[None, :, :]
    normalized_local_weight = local_weight * scale
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    staged_weights = tl.where(
        offs_c[:, None, None] == tail_chunk, normalized_local_weight[None, :, :], normalized_weights
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), staged_weights)
    tl.debug_barrier()
    local_acc_cols = tl.broadcast_to(mailbox_cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    local_acc_rows = tl.broadcast_to(offs_v[:, None, None], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    local_acc = tl.load(tle.gpu.local_ptr(winner_acc, (local_acc_rows, local_acc_cols))).to(tl.float32)
    pair_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_active = chunk < n_chunks
        chunk_is_tail = chunk == tail_chunk
        chunk_valid = chunk_active & valid_head[:, None]
        chunk_weight = tl.load(
            tle.gpu.local_ptr(
                FINAL_WEIGHT_SMEM, (tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32), pair_cols)
            ),
            mask=chunk_valid,
            other=0.0,
        )
        global_partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=(chunk_valid & ~chunk_is_tail)[None, :, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.where(chunk_is_tail, local_acc, global_partial)
        acc += partial * chunk_weight[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_subgroup_reduce_mtp4(
    SPLIT_OUT,
    LSE,
    batch,
    hkv,
    cluster_rank,
    chunk_base,
    chunk_count,
    output_chunk,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    USE_LOG2: tl.constexpr,
    TREE_PACK: tl.constexpr,
):
    """Reduce one global-partial subgroup into a normalized scratch slot."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    max_lse = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
    denom = tl.zeros((2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk_offset in tl.static_range(0, TREE_PACK):
        chunk = chunk_base + chunk_offset
        valid_chunk = chunk_offset < chunk_count
        chunk_lse = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :] * LSE_STRIDE_M
            + h_in_group[:, None] * LSE_STRIDE_HG,
            mask=valid_head[:, None] & valid_chunk,
            other=-float("inf"),
        )
        next_max = tl.maximum(max_lse, chunk_lse)
        valid = next_max != -float("inf")
        safe_next_max = tl.where(valid, next_max, 0.0)
        old_scale = tl.where(
            max_lse != -float("inf"),
            tl.exp2(max_lse - safe_next_max) if USE_LOG2 else tl.exp(max_lse - safe_next_max),
            0.0,
        )
        chunk_scale = tl.where(
            valid_head[:, None] & valid_chunk,
            tl.exp2(chunk_lse - safe_next_max) if USE_LOG2 else tl.exp(chunk_lse - safe_next_max),
            0.0,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=valid_head[None, :, None] & valid_chunk,
            other=0.0,
        )
        acc = acc * old_scale[None, :, :] + partial * chunk_scale[None, :, :]
        denom = denom * old_scale + chunk_scale
        max_lse = next_max
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    normalized = acc / safe_denom[None, :, :]
    combined_lse = tl.where(
        denom > 0.0, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + max_lse, -float("inf")
    )
    tl.store(
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + output_chunk * SO_STRIDE_C
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None],
        normalized,
        mask=valid_head[None, :, None],
    )
    tl.store(
        LSE
        + batch * LSE_STRIDE_B
        + output_chunk * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG,
        combined_lse,
        mask=valid_head[:, None],
    )


@triton.jit
def _compute_mtp4___cluster_quad_head_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    output_scale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Finalize the four c2-owned GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 4, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 4, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    acc = tl.zeros((DV, 4, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    quad_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
        chunk_rows = tl.full((4, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32)
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, quad_cols)), mask=chunk_valid, other=0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=chunk_valid[None, :, :],
            other=0.0,
        )
        acc += partial * chunk_weight[None, :, :]
    tl.debug_barrier()
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc * output_scale,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@builtin
def _compute_mtp4___memdesc_subslice(value, shape: tl.constexpr, offsets: tl.constexpr, _semantic=None):
    """Use FlagTree's existing ttg.memdesc_subslice builder binding."""
    shape = [int(tl_core._unwrap_if_constexpr(dim)) for dim in shape]
    layout = value.type.layout
    result_ty = gpu_types.buffered_tensor_type(
        value.dtype, shape, value.type.storage, layout, _semantic, alloc_shape=value.type.alloc_shape
    )
    handle = _semantic.builder.create_memdesc_subslice(result_ty.to_ir(_semantic.builder), value.handle, list(offsets))
    return gpu_types.buffered_tensor(
        handle, value.dtype, shape, value.type.storage, layout, _semantic, alloc_shape=value.type.alloc_shape
    )


@builtin
def _compute_mtp4___memdesc_transpose_2d(value, _semantic=None):
    """Return a zero-copy transposed view of a rank-2 shared memdesc.

    Unlike ``tl.trans(local_load(...))``, this exposes the transpose to the
    local-load lowering.  The intended transposed-memdesc lowering is therefore a direct
    transposed shared-to-register load from the TMA buffer, with no temporary
    shared allocation between the load and RS WGMMA.
    """
    if len(value.type.shape) != 2:
        raise ValueError("_memdesc_transpose_2d expects a rank-2 buffer")
    order = [1, 0]
    handle = _semantic.builder.create_memdesc_trans(value.handle, order)
    shape = [value.type.shape[i] for i in order]
    alloc_shape = value.type.alloc_shape
    leading_rank = len(alloc_shape) - len(value.type.shape)
    alloc_tail = alloc_shape[leading_rank:]
    transposed_alloc_shape = alloc_shape[:leading_rank] + [alloc_tail[i] for i in order]
    layout = value.type.layout.make_permute(order)
    return gpu_types.buffered_tensor(
        handle, value.dtype, shape, value.type.storage, layout, _semantic, alloc_shape=transposed_alloc_shape
    )


@triton.jit
def _compute_mtp4__fp8_kvpertensor_decode_mtp4_direct_kernel(
    Q,
    K_DESC,
    VT_DESC,
    BLOCK_IDS,
    TASK_MAP,
    QSCALE,
    KSCALE,
    VSCALE,
    OUT,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    ROWS_Q: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIRECT_CLUSTER_BASE: tl.constexpr,
    DIRECT_CLUSTER_SIZE: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    TMA_STAGES: tl.constexpr = 2,
    K_PER_TOKEN_V_PER_HEAD: tl.constexpr = False,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
):
    """Public-TLE direct-task specialization with no DSM/finalizer state."""
    direct_id = tl.program_id(0)
    direct_cluster = DIRECT_CLUSTER_BASE + direct_id // DIRECT_CLUSTER_SIZE
    direct_rank = direct_id % DIRECT_CLUSTER_SIZE
    logical_cta = direct_cluster * DIRECT_CLUSTER_SIZE + direct_rank
    task_base = (logical_cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task_base + 0)
    batch = tl.load(TASK_MAP + task_base + 1)
    seq_start = tl.load(TASK_MAP + task_base + 3)
    seq_len = tl.load(TASK_MAP + task_base + 4)
    seq_kvcache = tl.load(TASK_MAP + task_base + 5)
    is_causal = tl.load(TASK_MAP + task_base + 8)
    q_smem = tle.gpu.alloc([ROWS_Q, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([ROWS_Q, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    v_raw_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=BLOCK_N * D)
    vt_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=DV * BLOCK_N)
    offs_q = tl.arange(0, ROWS_Q)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(tl.arange(0, ROWS_Q)[:, None], (ROWS_Q, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (ROWS_Q, D))
    p_rows = tl.broadcast_to(tl.arange(0, ROWS_Q)[:, None], (ROWS_Q, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (ROWS_Q, BLOCK_N))
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    q = tl.load(
        Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
        mask=valid_q[:, None],
        other=0.0,
    )
    tl.store(q_smem_ptr, q)
    qscale = tl.load(QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0).to(
        tl.float32
    )
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE).to(tl.float32)
    vscale = tl.load(VSCALE + hkv if K_PER_TOKEN_V_PER_HEAD else VSCALE).to(tl.float32) / 256.0
    m_i = tl.full((ROWS_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((ROWS_Q,), tl.float32)
    acc = tl.zeros((DV, ROWS_Q), tl.float32)
    copy_iter = 0
    start = 0
    if start < seq_len:
        phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + seq_start // BLOCK_SIZE)
        tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
    while start < seq_len:
        local_n = start + offs_n
        valid_cols = local_n < seq_len
        buf = copy_iter % TMA_STAGES
        phase = copy_iter // TMA_STAGES & 1
        next_start = start + BLOCK_N
        if next_start < seq_len:
            next_buf = (copy_iter + 1) % TMA_STAGES
            logical_row = seq_start + next_start
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + logical_row // BLOCK_SIZE)
            tle.gpu.copy(
                K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
            )
            tle.gpu.copy(
                VT_DESC, v_raw_smem.slot(next_buf), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[next_buf]
            )
        tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
        k_page = k_raw_smem.slot(buf)
        scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores)
        if K_PER_TOKEN_V_PER_HEAD:
            scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
            tile_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
            )
            scores *= tile_kscale[:, None]
        else:
            scores *= kscale
        scores *= inv_sqrt_d * 1.4426950408889634
        scores *= qscale[None, :]
        causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
        scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float("inf"))
        m_tile = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        valid_update = m_new != -float("inf")
        safe_m_new = tl.where(valid_update, m_new, 0.0)
        safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
        p = tl.exp2(scores - safe_m_new[None, :])
        p = tl.where(valid_update[None, :], p, 0.0)
        alpha = tl.exp2(safe_m_i - safe_m_new)
        alpha = tl.where(valid_update, alpha, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        tl.store(p_smem_ptr, tl.trans((p * 256.0).to(tl.float8e4nv)))
        tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
        v_page = v_raw_smem.slot(buf)
        v_page_t = _compute_mtp4___memdesc_transpose_2d(v_page)
        pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
        pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
        v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
        v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
        v_pairs = tl.load(v_pair_ptr)
        v_lo = (v_pairs & 255).to(tl.uint8)
        v_hi = (v_pairs >> 8).to(tl.uint8)
        v_bytes = tl.join(v_lo, v_hi)
        v_bytes = tl.permute(v_bytes, (0, 2, 1))
        v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
        v_reg = v_bytes.to(tl.float8e4nv, bitcast=True)
        pv = tle.gpu.wgmma(v_reg, p_smem, trans_b=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[None, :] + pv
        m_i = m_new
        l_i = l_new
        start = next_start
        copy_iter += 1
    has_value = l_i > 0.0
    acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
    tl.store(
        OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_q[None, :],
    )


@triton.jit
def _compute_mtp4___fp8_kvpertensor_decode_mtp4_pure_tle_task(
    Q,
    K_DESC,
    VT_DESC,
    KS_DESC,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    KV_READER,
    Q_REUSE_SMEM,
    LOCAL_ACC_SMEM,
    LOCAL_LSE_SMEM,
    EXTERNAL_CLUSTER_RANK,
    BUNDLE_TASK_IDX,
    cta,
    copy_iter_base,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    RAW_PAGED_NHD_OR_HND: tl.constexpr = False,
    FULL_MATRIX_RS: tl.constexpr = False,
    TRANSPOSED_MEMDESC_RS: tl.constexpr = False,
    DIRECT_V_SHARED_SHARED: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    TLE_SHARED_SHARED: tl.constexpr = False,
    WIDE_VIEW_V_RS: tl.constexpr = False,
    FULL_VIEW_P_STORE: tl.constexpr = False,
    FULL_VIEW_DSM: tl.constexpr = False,
    DUAL_ACC_RAW_FINALIZER: tl.constexpr = False,
    CHUNK4_RAW_FINALIZER: tl.constexpr = False,
    SHARDED_ELECTION: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    STREAMING_RAW_FINALIZER: tl.constexpr = False,
    DETERMINISTIC_TAIL_REUSE: tl.constexpr = False,
    RECOMPUTE_RAW_FINALIZER: tl.constexpr = False,
    TAIL_ONLY_ELECTION_BARRIER: tl.constexpr = False,
    TMA_RAW_FINALIZER: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp4__EXECUTION_FULL,
    CLUSTER_COOPERATIVE_FINALIZE: tl.constexpr = False,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    REUSE_FINAL_WEIGHTS: tl.constexpr = False,
    HEAD_SHARDED_DSM: tl.constexpr = False,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    TWO_CHUNK_FINALIZE: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    FAST_FINALIZER_HANDOFF: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    WINNER_LOCAL_REUSE_FINALIZER: tl.constexpr = False,
    SPLIT_D_DSM_MERGE: tl.constexpr = False,
    EXTERNAL_KV_PIPE: tl.constexpr = False,
    LOAD_Q: tl.constexpr = True,
    PIPE_LOCAL_EXPORT: tl.constexpr = False,
    FINALIZE_EXTERNAL_LOCAL: tl.constexpr = False,
    TMA_STAGES: tl.constexpr = 2,
    C4_FINALIZER_TILE: tl.constexpr = 1,
    MICRO_CONSUME_LOOKAHEAD: tl.constexpr = False,
    MICRO_PREFETCH_NEXT: tl.constexpr = False,
    CTA_ROLE_RANK0_TILES: tl.constexpr = -1,
    CTA_ROLE_REPARTITION: tl.constexpr = False,
    K_PER_TOKEN_V_PER_HEAD: tl.constexpr = False,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    PREFETCH_K_SCALE: tl.constexpr = False,
    TMA_K_SCALE: tl.constexpr = False,
    FRAGMENT_K_SCALE_LOAD: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    DEFER_V_SCALE_FINAL: tl.constexpr = False,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    C4_ALIGNED_FULL_CHUNK_RAW_512: tl.constexpr = C4_FINALIZER_TILE >= 1000 and C4_FINALIZER_TILE < 1100
    C4_ALIGNED_FULL_CHUNK_GENERIC_1024: tl.constexpr = C4_FINALIZER_TILE >= 1300 and C4_FINALIZER_TILE < 1400
    C4_ALIGNED_FULL_CHUNK_RAW: tl.constexpr = C4_ALIGNED_FULL_CHUNK_RAW_512 or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
    C2_ALIGNED_FULL_CHUNK_WINNER_1024: tl.constexpr = C4_FINALIZER_TILE >= 1100 and C4_FINALIZER_TILE < 1200
    C2_ALIGNED_FULL_CHUNK_WINNER_256: tl.constexpr = C4_FINALIZER_TILE >= 1400 and C4_FINALIZER_TILE < 1500
    C2_ALIGNED_FULL_CHUNK_WINNER: tl.constexpr = C2_ALIGNED_FULL_CHUNK_WINNER_1024 or C2_ALIGNED_FULL_CHUNK_WINNER_256
    C2_ALIGNED_FULL_CHUNK_RAW: tl.constexpr = C4_FINALIZER_TILE >= 1200 and C4_FINALIZER_TILE < 1300
    C2_ALIGNED_FULL_CHUNK: tl.constexpr = C2_ALIGNED_FULL_CHUNK_WINNER or C2_ALIGNED_FULL_CHUNK_RAW
    C4_REDUCTION_ONLY_RAW: tl.constexpr = C4_FINALIZER_TILE >= 900 and C4_FINALIZER_TILE < 1100
    PACKED_K_SCALE_LOOKAHEAD: tl.constexpr = PREFETCH_K_SCALE == 2 and (not TMA_K_SCALE)
    SHARED_K_SCALE_PIPELINE: tl.constexpr = PREFETCH_K_SCALE == 3 and (not TMA_K_SCALE)
    WGMMA_SHADOW_K_SCALE: tl.constexpr = PREFETCH_K_SCALE == 4 and (not TMA_K_SCALE)
    PAGE_METADATA_K_SCALE: tl.constexpr = PREFETCH_K_SCALE == 5 and (not TMA_K_SCALE)
    DETERMINISTIC_FIRST_OWNER: tl.constexpr = C4_FINALIZER_TILE == 702
    REGISTER_RAW_WEIGHTS: tl.constexpr = C4_FINALIZER_TILE == 704
    DISTRIBUTED_READY_FLAGS: tl.constexpr = C4_FINALIZER_TILE == 705
    TAIL_RAW_DSM_REUSE: tl.constexpr = C4_FINALIZER_TILE == 706
    HEAD_MAJOR_RAW_REDUCE: tl.constexpr = C4_FINALIZER_TILE == 707
    C8_BF16_DEFERRED_NORM: tl.constexpr = C4_FINALIZER_TILE >= 800 and C4_FINALIZER_TILE < 900
    C4_GLOBAL_DEFERRED_NORM: tl.constexpr = (
        C4_FINALIZER_TILE >= 700
        and C4_FINALIZER_TILE < 800
        or C4_REDUCTION_ONLY_RAW
        or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
    )
    C4_DEFERRED_NORM: tl.constexpr = (
        C4_FINALIZER_TILE >= 600
        and C4_FINALIZER_TILE < 900
        or C4_REDUCTION_ONLY_RAW
        or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
        or C2_ALIGNED_FULL_CHUNK_RAW
    )
    C4_BF16_DSM: tl.constexpr = (
        C4_FINALIZER_TILE >= 500
        and C4_FINALIZER_TILE < 900
        or C4_REDUCTION_ONLY_RAW
        or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
        or C2_ALIGNED_FULL_CHUNK_RAW
    )
    C4_HIERARCHICAL_FINALIZE: tl.constexpr = C4_FINALIZER_TILE >= 400 and C4_FINALIZER_TILE < 500
    C4_TREE_PACK: tl.constexpr = C4_FINALIZER_TILE - 400 if C4_HIERARCHICAL_FINALIZE else 0
    C4_EXACT_TWO_FINALIZE: tl.constexpr = C4_FINALIZER_TILE >= 200 and C4_FINALIZER_TILE < 300
    C4_Q_DSM_FANOUT: tl.constexpr = C4_FINALIZER_TILE >= 100 and C4_FINALIZER_TILE < 200
    C4_FINALIZER_MODE: tl.constexpr = (
        1
        if C4_HIERARCHICAL_FINALIZE or C4_BF16_DSM or C2_ALIGNED_FULL_CHUNK
        else C4_FINALIZER_TILE - 200
        if C4_EXACT_TWO_FINALIZE
        else C4_FINALIZER_TILE - 100
        if C4_Q_DSM_FANOUT
        else C4_FINALIZER_TILE
    )
    if PIPE_LOCAL_EXPORT:
        cluster_rank = EXTERNAL_CLUSTER_RANK
    else:
        cluster_rank = tle.shard_id(mesh, "cluster_x")
    if STATIC_SCHED:
        logical_cluster = cta // MERGE_CLUSTER_SIZE
        group_chunk = logical_cluster % STATIC_MAX_GROUPS
        sequence = logical_cluster // STATIC_MAX_GROUPS
        batch = sequence % B
        hkv = sequence // B
        total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
        num_chunks = (total_len + STATIC_CHUNK_TOKENS - 1) // STATIC_CHUNK_TOKENS
        group_count = (num_chunks + MERGE_CLUSTER_SIZE - 1) // MERGE_CLUSTER_SIZE
        if group_chunk >= group_count:
            return
        chunk = group_chunk * MERGE_CLUSTER_SIZE + cluster_rank
        seq_start = chunk * STATIC_CHUNK_TOKENS
        has_work = chunk < num_chunks
        seq_len = tl.where(has_work, tl.minimum(STATIC_CHUNK_TOKENS, total_len - seq_start), 0)
        is_causal = has_work & (chunk == num_chunks - 1)
        seq_kvcache = tl.where(is_causal, seq_len - _compute_mtp4___NUM_SEQ_Q_JIT, seq_len)
        task_mode = tl.where(num_chunks == 1, _compute_mtp4___DIRECT_MODE_JIT, _compute_mtp4___GROUP_MODE_JIT)
    else:
        task_base = (cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        hkv = tl.load(TASK_MAP + task_base + 0)
        batch = tl.load(TASK_MAP + task_base + 1)
        seq_start = tl.load(TASK_MAP + task_base + 3)
        if C4_ALIGNED_FULL_CHUNK_RAW_512:
            seq_len = 512
        elif C4_ALIGNED_FULL_CHUNK_GENERIC_1024:
            seq_len = 1024
        elif C2_ALIGNED_FULL_CHUNK_WINNER_256:
            seq_len = 256
        elif C2_ALIGNED_FULL_CHUNK:
            seq_len = 1024
        else:
            seq_len = tl.load(TASK_MAP + task_base + 4)
        seq_kvcache = tl.load(TASK_MAP + task_base + 5)
        is_causal = tl.load(TASK_MAP + task_base + 8)
        if C4_ALIGNED_FULL_CHUNK_RAW:
            task_mode = _compute_mtp4___GROUP_MODE_JIT
        else:
            task_mode = tl.load(TASK_MAP + task_base + 9)
        group_chunk = tl.load(TASK_MAP + task_base + 10)
        group_count = tl.load(TASK_MAP + task_base + 11)
        if C4_ALIGNED_FULL_CHUNK_RAW:
            has_work = True
        else:
            has_work = (task_mode != _compute_mtp4___DUMMY_MODE_JIT) & (seq_len > 0)
    if CTA_ROLE_REPARTITION and (CTA_ROLE_RANK0_TILES == 7 or CTA_ROLE_RANK0_TILES == 8):
        if CTA_ROLE_RANK0_TILES == 7:
            if cluster_rank == 0:
                seq_len -= BLOCK_N
                seq_kvcache = seq_len
                is_causal = 0
            elif cluster_rank == 1:
                seq_start -= BLOCK_N
                seq_len += BLOCK_N
                seq_kvcache = seq_len
                is_causal = 0
        has_work = seq_len > 0
    elif CTA_ROLE_REPARTITION and CTA_ROLE_RANK0_TILES >= 0:
        rank0_cta = cta - cluster_rank
        rank0_base = (rank0_cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        hkv = tl.load(TASK_MAP + rank0_base + 0)
        batch = tl.load(TASK_MAP + rank0_base + 1)
        group_start = tl.load(TASK_MAP + rank0_base + 3)
        task_mode = tl.load(TASK_MAP + rank0_base + 9)
        group_chunk = tl.load(TASK_MAP + rank0_base + 10)
        group_count = tl.load(TASK_MAP + rank0_base + 11)
        group_len = tl.full((), 0, tl.int32)
        group_is_causal = tl.full((), 0, tl.int32)
        for rank in tl.static_range(0, 4):
            peer_base = ((rank0_cta + rank) * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
            peer_mode = tl.load(TASK_MAP + peer_base + 9)
            peer_real = peer_mode != _compute_mtp4___DUMMY_MODE_JIT
            group_len += tl.where(peer_real, tl.load(TASK_MAP + peer_base + 4), 0)
            group_is_causal |= tl.where(peer_real, tl.load(TASK_MAP + peer_base + 8), 0)
        total_tiles = (group_len + BLOCK_N - 1) // BLOCK_N
        rank0_tiles = tl.minimum(total_tiles, CTA_ROLE_RANK0_TILES)
        remaining_tiles = total_tiles - rank0_tiles
        peer = tl.maximum(cluster_rank - 1, 0)
        peer_begin = rank0_tiles + (remaining_tiles * peer + 2) // 3
        peer_end = rank0_tiles + (remaining_tiles * (peer + 1) + 2) // 3
        begin_tile = tl.where(cluster_rank == 0, 0, peer_begin)
        end_tile = tl.where(cluster_rank == 0, rank0_tiles, peer_end)
        role_capacity = (end_tile - begin_tile) * BLOCK_N
        role_remaining = tl.maximum(group_len - begin_tile * BLOCK_N, 0)
        seq_start = group_start + begin_tile * BLOCK_N
        seq_len = tl.minimum(role_capacity, role_remaining)
        is_last_role = (end_tile == total_tiles) & (end_tile > begin_tile)
        is_causal = group_is_causal & is_last_role
        seq_kvcache = tl.where(is_causal != 0, seq_len - _compute_mtp4___NUM_SEQ_Q_JIT, seq_len)
        has_work = seq_len > 0
    q_smem = tle.gpu.alloc([_compute_mtp4___ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_compute_mtp4___ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    if RAW_PAGED_NHD_OR_HND:
        k_raw_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
        v_raw_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    else:
        k_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
        vt_smem = tle.gpu.alloc([TMA_STAGES, DV, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=BLOCK_N * D)
    vt_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=DV * BLOCK_N)
    if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
        ks_smem = tle.gpu.alloc([TMA_STAGES, 1, 2, 1, 32], dtype=tl.float32, layout=None, scope=tle.gpu.smem)
        ks_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=2 * 32 * 4)
    if K_PER_TOKEN_V_PER_HEAD and SHARED_K_SCALE_PIPELINE:
        ks_copy_smem = tle.gpu.alloc(
            [TMA_STAGES, 2, 32], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    if HEAD_SHARDED_DSM:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp4___ROWS_Q_JIT],
            dtype=tl.bfloat16 if C4_BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C4_DEFERRED_NORM:
            partial_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            partial_l_smem = partial_lse_smem
        if C4_GLOBAL_DEFERRED_NORM:
            head_acc_smem = partial_acc_smem
            head_lse_smem = partial_lse_smem
        elif PAIRED_HEAD_FINALIZE and MERGE_CLUSTER_SIZE == 4:
            head_acc_smem = tle.gpu.alloc(
                [DV, 2 * _compute_mtp4___NUM_SEQ_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            head_lse_smem = tle.gpu.alloc(
                [2 * _compute_mtp4___NUM_SEQ_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            head_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___NUM_SEQ_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            head_lse_smem = tle.gpu.alloc(
                [_compute_mtp4___NUM_SEQ_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        if MERGE_CLUSTER_SIZE == 2:
            peer_acc_smem = partial_acc_smem
            peer_lse_smem = partial_lse_smem
            peer1_acc_smem = peer_acc_smem
            peer2_acc_smem = peer_acc_smem
            peer3_acc_smem = peer_acc_smem
            peer1_lse_smem = peer_lse_smem
            peer2_lse_smem = peer_lse_smem
            peer3_lse_smem = peer_lse_smem
            peer1_l_smem = peer_lse_smem
            peer2_l_smem = peer_lse_smem
            peer3_l_smem = peer_lse_smem
        elif MERGE_CLUSTER_SIZE == 4:
            peer1_acc_smem = partial_acc_smem
            peer2_acc_smem = partial_acc_smem
            peer3_acc_smem = partial_acc_smem
            peer1_lse_smem = partial_lse_smem
            peer2_lse_smem = partial_lse_smem
            peer3_lse_smem = partial_lse_smem
            peer1_l_smem = partial_lse_smem
            peer2_l_smem = partial_lse_smem
            peer3_l_smem = partial_lse_smem
    elif MERGE_CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp4___ROWS_Q_JIT],
            dtype=tl.bfloat16 if C4_BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C4_DEFERRED_NORM:
            peer_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer_l_smem = peer_lse_smem
        peer1_acc_smem = peer_acc_smem
        peer2_acc_smem = peer_acc_smem
        peer3_acc_smem = peer_acc_smem
        peer1_lse_smem = peer_lse_smem
        peer2_lse_smem = peer_lse_smem
        peer3_lse_smem = peer_lse_smem
        peer1_l_smem = peer_l_smem
        peer2_l_smem = peer_l_smem
        peer3_l_smem = peer_l_smem
    elif MERGE_CLUSTER_SIZE == 4:
        if C4_BF16_DSM:
            peer1_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer1_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        peer1_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer2_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer3_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C4_DEFERRED_NORM:
            peer1_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer1_l_smem = peer1_lse_smem
            peer2_l_smem = peer2_lse_smem
            peer3_l_smem = peer3_lse_smem
    else:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp4___ROWS_Q_JIT],
            dtype=tl.bfloat16 if C8_BF16_DEFERRED_NORM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C8_BF16_DEFERRED_NORM:
            partial_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            partial_l_smem = partial_lse_smem
        peer1_acc_smem = partial_acc_smem
        peer2_acc_smem = partial_acc_smem
        peer3_acc_smem = partial_acc_smem
        peer1_lse_smem = partial_lse_smem
        peer2_lse_smem = partial_lse_smem
        peer3_lse_smem = partial_lse_smem
        peer1_l_smem = partial_l_smem
        peer2_l_smem = partial_l_smem
        peer3_l_smem = partial_l_smem
    if CLUSTER_COOPERATIVE_FINALIZE:
        if STREAMING_RAW_FINALIZER or RECOMPUTE_RAW_FINALIZER or REGISTER_RAW_WEIGHTS:
            final_weight_smem = peer1_lse_smem
        else:
            final_weight_smem = tle.gpu.alloc(
                [
                    2 * _compute_mtp4___NUM_SEQ_Q_JIT
                    if HEAD_MAJOR_RAW_REDUCE
                    else MAX_FINAL_CHUNKS
                    if REUSE_FINAL_WEIGHTS and (not TWO_CHUNK_FINALIZE) and (not QUAD_HEAD_TWO_CHUNK_FINALIZE)
                    else 1,
                    MAX_FINAL_CHUNKS
                    if HEAD_MAJOR_RAW_REDUCE
                    else 2 * _compute_mtp4___NUM_SEQ_Q_JIT
                    if PAIRED_HEAD_FINALIZE and MERGE_CLUSTER_SIZE == 4
                    else 4 * _compute_mtp4___NUM_SEQ_Q_JIT
                    if PAIRED_HEAD_FINALIZE and MERGE_CLUSTER_SIZE == 2
                    else _compute_mtp4___NUM_SEQ_Q_JIT,
                ],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
    if TMA_RAW_FINALIZER:
        final_acc_smem = tle.gpu.alloc(
            [2, 1, _compute_mtp4___NUM_SEQ_Q_JIT, 1, DV],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        final_acc_full = tle.gpu.alloc_barriers(
            num_barriers=2, arrive_count=1, expect_bytes=_compute_mtp4___NUM_SEQ_Q_JIT * DV * 4
        )
    offs_q = tl.arange(0, _compute_mtp4___ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _compute_mtp4___ROWS_Q_JIT)[:, None], (_compute_mtp4___ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_compute_mtp4___ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _compute_mtp4___ROWS_Q_JIT)[:, None], (_compute_mtp4___ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_compute_mtp4___ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _compute_mtp4___ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _compute_mtp4___ROWS_Q_JIT)[None, :], (DV, _compute_mtp4___ROWS_Q_JIT))
    store_offs_v = offs_v
    if EXTERNAL_KV_PIPE:
        q_work_smem = Q_REUSE_SMEM
    else:
        q_work_smem = q_smem
    q_smem_ptr = tle.gpu.local_ptr(q_work_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if MERGE_CLUSTER_SIZE == 8 or HEAD_SHARDED_DSM:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + hkv if K_PER_TOKEN_V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = has_work & (seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _compute_mtp4___ROWS_Q_JIT), tl.float32)
    lse = tl.full((_compute_mtp4___ROWS_Q_JIT,), -float("inf"), tl.float32)
    raw_l = tl.zeros((_compute_mtp4___ROWS_Q_JIT,), tl.float32)
    if FINALIZE_EXTERNAL_LOCAL:
        local_acc_rows = tl.broadcast_to(offs_v[:, None], (DV, _compute_mtp4___ROWS_Q_JIT))
        local_acc_cols = tl.broadcast_to(offs_q[None, :], (DV, _compute_mtp4___ROWS_Q_JIT))
        local_acc_slot = tl.full((DV, _compute_mtp4___ROWS_Q_JIT), BUNDLE_TASK_IDX, tl.int32)
        local_lse_slot = tl.full((_compute_mtp4___ROWS_Q_JIT,), BUNDLE_TASK_IDX, tl.int32)
        acc = tl.load(tle.gpu.local_ptr(LOCAL_ACC_SMEM, (local_acc_slot, local_acc_rows, local_acc_cols)))
        lse = tl.load(tle.gpu.local_ptr(LOCAL_LSE_SMEM, (local_lse_slot, offs_q)))
    if has_work and (not FINALIZE_EXTERNAL_LOCAL):
        if LOAD_Q:
            if C4_Q_DSM_FANOUT:
                if cluster_rank == 0:
                    q = tl.load(
                        Q
                        + batch * Q_STRIDE_B
                        + seq_m[:, None] * Q_STRIDE_M
                        + hq[:, None] * Q_STRIDE_H
                        + offs_d[None, :],
                        mask=valid_q[:, None],
                        other=0.0,
                    )
                    tl.store(q_smem_ptr, q)
                tle.distributed_barrier(mesh)
                if cluster_rank != 0:
                    q_rank0_smem = tle.remote(q_smem, 0, scope=mesh)
                    q = tl.load(tle.gpu.local_ptr(q_rank0_smem, (q_rows, q_cols)))
                    tl.store(q_smem_ptr, q)
            else:
                q = tl.load(
                    Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
                    mask=valid_q[:, None],
                    other=0.0,
                )
                tl.store(q_smem_ptr, q)
        qscale = tl.load(
            QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0
        ).to(tl.float32)
        if PRECOMBINE_Q_SCALE:
            qscale = qscale * (inv_sqrt_d * (1.4426950408889634 if USE_LOG2 else 1.0))
            if not K_PER_TOKEN_V_PER_HEAD:
                qscale = qscale * kscale
        m_i = tl.full((_compute_mtp4___ROWS_Q_JIT,), -float("inf"), tl.float32)
        l_i = tl.zeros((_compute_mtp4___ROWS_Q_JIT,), tl.float32)
        copy_iter = copy_iter_base
        start = 0
        current_phys = tl.full((), 0, tl.int32)
        if PACKED_K_SCALE_LOOKAHEAD:
            first_scale_block = seq_start // BLOCK_SIZE
            first_scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + first_scale_block)
            lookahead_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                KSCALE, first_scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
            )
        if start < seq_len and (not EXTERNAL_KV_PIPE) and (not MICRO_CONSUME_LOOKAHEAD):
            initial_buf = copy_iter % TMA_STAGES
            block_no = seq_start // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            if PAGE_METADATA_K_SCALE:
                current_phys = phys
            if RAW_PAGED_NHD_OR_HND:
                tle.gpu.copy(
                    K_DESC,
                    k_raw_smem.slot(initial_buf),
                    [1, 1, BLOCK_N, D],
                    [phys, hkv, 0, 0],
                    barrier=k_full[initial_buf],
                )
                tle.gpu.copy(
                    VT_DESC,
                    v_raw_smem.slot(initial_buf),
                    [1, 1, BLOCK_N, DV],
                    [phys, hkv, 0, 0],
                    barrier=vt_full[initial_buf],
                )
                if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                    tle.gpu.copy(
                        KS_DESC,
                        ks_smem.slot(initial_buf),
                        [1, 2, 1, 32],
                        [phys, 0, hkv, 0],
                        barrier=ks_full[initial_buf],
                    )
                if K_PER_TOKEN_V_PER_HEAD and SHARED_K_SCALE_PIPELINE:
                    initial_scale_ptrs = tl.reshape(
                        _compute_mtp4___packed_k_scale_ptrs_mtp4(
                            KSCALE, phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                        ),
                        (2, 32),
                    )
                    tle.gpu.copy(initial_scale_ptrs, ks_copy_smem.slot(initial_buf), [2, 32])
            else:
                tma_row = phys * BLOCK_SIZE
                tle.gpu.copy(
                    K_DESC, k_smem.slot(initial_buf), [BLOCK_N, D], [tma_row, hkv * D], barrier=k_full[initial_buf]
                )
                tle.gpu.copy(
                    VT_DESC, vt_smem.slot(initial_buf), [DV, BLOCK_N], [hkv * DV, tma_row], barrier=vt_full[initial_buf]
                )
        if start < seq_len and MICRO_CONSUME_LOOKAHEAD and K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
            scale_initial_iter = copy_iter
            scale_initial_buf = scale_initial_iter % TMA_STAGES
            scale_block_no = seq_start // BLOCK_SIZE
            scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + scale_block_no)
            tle.gpu.copy(
                KS_DESC,
                ks_smem.slot(scale_initial_buf),
                [1, 2, 1, 32],
                [scale_phys, 0, hkv, 0],
                barrier=ks_full[scale_initial_buf],
            )
        while start < seq_len:
            local_n = start + offs_n
            if C4_ALIGNED_FULL_CHUNK_RAW or C2_ALIGNED_FULL_CHUNK:
                valid_cols = tl.full((BLOCK_N,), True, tl.int1)
            else:
                valid_cols = local_n < seq_len
            local_copy_iter = copy_iter - 1 if MICRO_CONSUME_LOOKAHEAD else copy_iter
            buf = local_copy_iter % TMA_STAGES
            phase = local_copy_iter // TMA_STAGES & 1
            scale_copy_iter = local_copy_iter + 1 if MICRO_CONSUME_LOOKAHEAD else local_copy_iter
            scale_buf = scale_copy_iter % TMA_STAGES
            scale_phase = scale_copy_iter // TMA_STAGES & 1
            next_start = start + BLOCK_N
            next_phys = current_phys
            if PACKED_K_SCALE_LOOKAHEAD:
                tile_kscale = lookahead_kscale
                next_lookahead_kscale = lookahead_kscale
            if next_start < seq_len and (not EXTERNAL_KV_PIPE):
                next_iter = local_copy_iter + 1
                next_buf = next_iter % TMA_STAGES
                aligned_logical = seq_start + next_start
                block_no = aligned_logical // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                if PAGE_METADATA_K_SCALE:
                    next_phys = phys
                if PACKED_K_SCALE_LOOKAHEAD:
                    next_lookahead_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                        KSCALE, phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                if RAW_PAGED_NHD_OR_HND:
                    tle.gpu.copy(
                        K_DESC,
                        k_raw_smem.slot(next_buf),
                        [1, 1, BLOCK_N, D],
                        [phys, hkv, 0, 0],
                        barrier=k_full[next_buf],
                    )
                    tle.gpu.copy(
                        VT_DESC,
                        v_raw_smem.slot(next_buf),
                        [1, 1, BLOCK_N, DV],
                        [phys, hkv, 0, 0],
                        barrier=vt_full[next_buf],
                    )
                    if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE:
                        scale_next_iter = next_iter + 1 if MICRO_CONSUME_LOOKAHEAD else next_iter
                        scale_next_buf = scale_next_iter % TMA_STAGES
                        tle.gpu.copy(
                            KS_DESC,
                            ks_smem.slot(scale_next_buf),
                            [1, 2, 1, 32],
                            [phys, 0, hkv, 0],
                            barrier=ks_full[scale_next_buf],
                        )
                    if K_PER_TOKEN_V_PER_HEAD and SHARED_K_SCALE_PIPELINE:
                        next_scale_ptrs = tl.reshape(
                            _compute_mtp4___packed_k_scale_ptrs_mtp4(
                                KSCALE, phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                            ),
                            (2, 32),
                        )
                        tle.gpu.copy(next_scale_ptrs, ks_copy_smem.slot(next_buf), [2, 32])
                else:
                    tma_row = phys * BLOCK_SIZE
                    tle.gpu.copy(
                        K_DESC, k_smem.slot(next_buf), [BLOCK_N, D], [tma_row, hkv * D], barrier=k_full[next_buf]
                    )
                    tle.gpu.copy(
                        VT_DESC, vt_smem.slot(next_buf), [DV, BLOCK_N], [hkv * DV, tma_row], barrier=vt_full[next_buf]
                    )
            consume_lookahead: tl.constexpr = MICRO_CONSUME_LOOKAHEAD
            first_lookahead = consume_lookahead & (start == 0)
            if EXTERNAL_KV_PIPE:
                ready = KV_READER.wait(copy_iter)
                kv_slot = ready.slot
            elif consume_lookahead:
                if first_lookahead:
                    tle.gpu.barrier_wait(LOCAL_ACC_SMEM[0], phaseIdx=0)
                    tle.gpu.barrier_wait(LOCAL_LSE_SMEM[0], phaseIdx=0)
                else:
                    tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            else:
                tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE and (not PREFETCH_K_SCALE):
                tle.gpu.barrier_wait(ks_full[scale_buf], phaseIdx=scale_phase)
                if FRAGMENT_K_SCALE_LOAD:
                    ks_lo_smem = _compute_mtp4___memdesc_subslice(ks_smem.slot(scale_buf), (1, 1, 1, 32), (0, 0, 0, 0))
                    ks_hi_smem = _compute_mtp4___memdesc_subslice(ks_smem.slot(scale_buf), (1, 1, 1, 32), (0, 1, 0, 0))
                    ks_lo = tl.reshape(tl.load(tle.gpu.local_ptr(ks_lo_smem)), (32,))
                    ks_hi = tl.reshape(tl.load(tle.gpu.local_ptr(ks_hi_smem)), (32,))
                    tile_kscale = tl.reshape(tl.permute(tl.join(ks_lo, ks_hi), (1, 0)), (BLOCK_N,))
                else:
                    tile_kscale = tl.reshape(tl.load(tle.gpu.local_ptr(ks_smem.slot(scale_buf))), (BLOCK_N,))
            if K_PER_TOKEN_V_PER_HEAD and PREFETCH_K_SCALE == 1 and (not TMA_K_SCALE):
                scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
                tile_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                    KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                )
            if RAW_PAGED_NHD_OR_HND:
                if EXTERNAL_KV_PIPE:
                    k_page = kv_slot.k
                elif consume_lookahead and first_lookahead:
                    k_page = KV_READER
                else:
                    k_page = k_raw_smem.slot(buf)
                if (
                    FULL_MATRIX_RS
                    or TRANSPOSED_MEMDESC_RS
                    or DIRECT_V_SHARED_SHARED
                    or LDSM_REGISTER_SHARED
                    or TLE_SHARED_SHARED
                    or WIDE_VIEW_V_RS
                ):
                    scores = tle.gpu.wgmma(k_page, q_work_smem, trans_b=True, out_dtype=tl.float32)
                else:
                    scores = tl.zeros((BLOCK_N, _compute_mtp4___ROWS_Q_JIT), tl.float32)
                    for frag in tl.static_range(0, 4):
                        k_frag = _compute_mtp4___memdesc_subslice(
                            k_page,
                            (BLOCK_N, _compute_mtp4___K_FRAGMENT_JIT),
                            (0, frag * _compute_mtp4___K_FRAGMENT_JIT),
                        )
                        q_frag = _compute_mtp4___memdesc_subslice(
                            q_work_smem,
                            (_compute_mtp4___ROWS_Q_JIT, _compute_mtp4___K_FRAGMENT_JIT),
                            (0, frag * _compute_mtp4___K_FRAGMENT_JIT),
                        )
                        scores = tle.gpu.wgmma(k_frag, q_frag, acc=scores, trans_b=True, out_dtype=tl.float32)
                if K_PER_TOKEN_V_PER_HEAD and (WGMMA_SHADOW_K_SCALE or PAGE_METADATA_K_SCALE):
                    scale_phys = current_phys
                    if WGMMA_SHADOW_K_SCALE:
                        scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
                    tile_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                        KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                scores = tle.gpu.wgmma_wait(0, scores)
            else:
                scores = tle.gpu.wgmma(k_smem.slot(buf), q_work_smem, trans_b=True, out_dtype=tl.float32)
                if K_PER_TOKEN_V_PER_HEAD and (WGMMA_SHADOW_K_SCALE or PAGE_METADATA_K_SCALE):
                    scale_phys = current_phys
                    if WGMMA_SHADOW_K_SCALE:
                        scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
                    tile_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                        KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                scores = tle.gpu.wgmma_wait(0, scores)
            if K_PER_TOKEN_V_PER_HEAD and TMA_K_SCALE and PREFETCH_K_SCALE:
                tle.gpu.barrier_wait(ks_full[scale_buf], phaseIdx=scale_phase)
                if FRAGMENT_K_SCALE_LOAD:
                    ks_lo_smem = _compute_mtp4___memdesc_subslice(ks_smem.slot(scale_buf), (1, 1, 1, 32), (0, 0, 0, 0))
                    ks_hi_smem = _compute_mtp4___memdesc_subslice(ks_smem.slot(scale_buf), (1, 1, 1, 32), (0, 1, 0, 0))
                    ks_lo = tl.reshape(tl.load(tle.gpu.local_ptr(ks_lo_smem)), (32,))
                    ks_hi = tl.reshape(tl.load(tle.gpu.local_ptr(ks_hi_smem)), (32,))
                    tile_kscale = tl.reshape(tl.permute(tl.join(ks_lo, ks_hi), (1, 0)), (BLOCK_N,))
                else:
                    tile_kscale = tl.reshape(tl.load(tle.gpu.local_ptr(ks_smem.slot(scale_buf))), (BLOCK_N,))
            if K_PER_TOKEN_V_PER_HEAD:
                if SHARED_K_SCALE_PIPELINE:
                    tile_kscale = tl.reshape(tl.load(tle.gpu.local_ptr(ks_copy_smem.slot(scale_buf))), (BLOCK_N,))
                if not PREFETCH_K_SCALE and (not TMA_K_SCALE):
                    scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + (seq_start + start) // BLOCK_SIZE)
                    tile_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                        KSCALE, scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                scores = scores * tile_kscale[:, None]
            elif not PRECOMBINE_Q_SCALE:
                scores = scores * kscale
            if not PRECOMBINE_Q_SCALE:
                scores = scores * (inv_sqrt_d * (1.4426950408889634 if USE_LOG2 else 1.0))
            scores = scores * qscale[None, :]
            causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
            scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float("inf"))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float("inf")
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :]) if USE_LOG2 else tl.exp(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new) if USE_LOG2 else tl.exp(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            if FULL_VIEW_P_STORE:
                tl.store(tle.gpu.local_ptr(p_smem), p_scaled_t)
            else:
                tl.store(p_smem_ptr, p_scaled_t)
            if not EXTERNAL_KV_PIPE and (not (consume_lookahead and first_lookahead)):
                tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            if RAW_PAGED_NHD_OR_HND:
                if EXTERNAL_KV_PIPE:
                    v_page = kv_slot.v
                elif consume_lookahead and first_lookahead:
                    v_page = Q_REUSE_SMEM
                else:
                    v_page = v_raw_smem.slot(buf)
                if TLE_SHARED_SHARED:
                    vt_page = _compute_mtp4___memdesc_transpose_2d(v_page)
                    pv = tle.gpu.wgmma(vt_page, p_smem, trans_b=True, out_dtype=tl.float32)
                    pv = tle.gpu.wgmma_wait(0, pv)
                if DIRECT_V_SHARED_SHARED:
                    vt_page = _compute_mtp4___memdesc_transpose_2d(v_page)
                    pv = tle.gpu.wgmma(vt_page, p_smem, trans_b=True, out_dtype=tl.float32)
                    pv = tle.gpu.wgmma_wait(0, pv)
                if LDSM_REGISTER_SHARED:
                    v_page_t = _compute_mtp4___memdesc_transpose_2d(v_page)
                    pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
                    pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
                    v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                    v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                    v_pairs = tl.load(v_pair_ptr)
                    v_lo = (v_pairs & 255).to(tl.uint8)
                    v_hi = (v_pairs >> 8).to(tl.uint8)
                    v_bytes = tl.join(v_lo, v_hi)
                    v_bytes = tl.permute(v_bytes, (0, 2, 1))
                    v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
                    v_page_reg_t = v_bytes.to(tl.float8e4nv, bitcast=True)
                elif WIDE_VIEW_V_RS:
                    v_page_t = _compute_mtp4___memdesc_transpose_2d(v_page)
                    v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t))
                elif TRANSPOSED_MEMDESC_RS and (not DIRECT_V_SHARED_SHARED):
                    v_page_t = _compute_mtp4___memdesc_transpose_2d(v_page)
                    vt_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, BLOCK_N))
                    vt_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV, BLOCK_N))
                    v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t, (vt_rows, vt_cols)))
                elif not DIRECT_V_SHARED_SHARED and (not TLE_SHARED_SHARED):
                    v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
                    v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
                    v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
                if (FULL_MATRIX_RS or TRANSPOSED_MEMDESC_RS or LDSM_REGISTER_SHARED or WIDE_VIEW_V_RS) and (
                    not DIRECT_V_SHARED_SHARED
                ):
                    pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                    pv = tle.gpu.wgmma_wait(0, pv)
                elif not DIRECT_V_SHARED_SHARED and (not TLE_SHARED_SHARED):
                    pv = tl.zeros((DV, _compute_mtp4___ROWS_Q_JIT), tl.float32)
                    for d_frag in tl.static_range(0, 2):
                        pv_frag = tl.zeros((_compute_mtp4___K_FRAGMENT_JIT * 2, _compute_mtp4___ROWS_Q_JIT), tl.float32)
                        for n_frag in tl.static_range(0, 2):
                            v_reg_frag = tle.extract_tile(
                                v_page_reg_t,
                                index=[d_frag, n_frag],
                                tile_shape=[_compute_mtp4___K_FRAGMENT_JIT * 2, _compute_mtp4___K_FRAGMENT_JIT],
                            )
                            p_frag = _compute_mtp4___memdesc_subslice(
                                p_smem,
                                (_compute_mtp4___ROWS_Q_JIT, _compute_mtp4___K_FRAGMENT_JIT),
                                (0, n_frag * _compute_mtp4___K_FRAGMENT_JIT),
                            )
                            pv_frag = tle.gpu.wgmma(v_reg_frag, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                        pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                        pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            else:
                pv = tle.gpu.wgmma(vt_smem.slot(buf), p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            acc = acc * alpha[None, :] + pv
            m_i = m_new
            l_i = l_new
            if EXTERNAL_KV_PIPE:
                KV_READER.release(copy_iter)
            start = next_start
            copy_iter += 1
            if PACKED_K_SCALE_LOOKAHEAD:
                lookahead_kscale = next_lookahead_kscale
            if PAGE_METADATA_K_SCALE:
                current_phys = next_phys
        has_value = l_i > 0.0
        raw_l = l_i
        if C4_DEFERRED_NORM:
            if not C4_REDUCTION_ONLY_RAW and task_mode == _compute_mtp4___DIRECT_MODE_JIT:
                acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
                lse = tl.where(has_value, (tl.log2(l_i) if USE_LOG2 else tl.log(l_i)) + m_i, -float("inf"))
            else:
                lse = m_i
        else:
            local_vscale = (
                tl.where(task_mode == _compute_mtp4___DIRECT_MODE_JIT, vscale, 1.0) if DEFER_V_SCALE_FINAL else vscale
            )
            acc = tl.where(has_value[None, :], acc / l_i[None, :] * local_vscale, 0.0)
            lse = tl.where(has_value, (tl.log2(l_i) if USE_LOG2 else tl.log(l_i)) + m_i, -float("inf"))
    if MICRO_PREFETCH_NEXT:
        next_cta = BUNDLE_TASK_IDX
        next_task_base = (next_cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        next_hkv = tl.load(TASK_MAP + next_task_base + 0)
        next_batch = tl.load(TASK_MAP + next_task_base + 1)
        next_seq_start = tl.load(TASK_MAP + next_task_base + 3)
        next_seq_len = tl.load(TASK_MAP + next_task_base + 4)
        next_task_mode = tl.load(TASK_MAP + next_task_base + 9)
        if (
            EXTERNAL_CLUSTER_RANK != 0
            and next_hkv >= 0
            and (next_task_mode != _compute_mtp4___DUMMY_MODE_JIT)
            and (next_seq_len > 0)
        ):
            next_block_no = next_seq_start // BLOCK_SIZE
            next_phys = tl.load(BLOCK_IDS + next_batch * MAX_BLOCKS + next_block_no)
            tle.gpu.copy(K_DESC, KV_READER, [1, 1, BLOCK_N, D], [next_phys, next_hkv, 0, 0], barrier=LOCAL_ACC_SMEM[0])
            tle.gpu.copy(
                VT_DESC, Q_REUSE_SMEM, [1, 1, BLOCK_N, DV], [next_phys, next_hkv, 0, 0], barrier=LOCAL_LSE_SMEM[0]
            )
    if PIPE_LOCAL_EXPORT:
        local_acc_rows = tl.broadcast_to(offs_v[:, None], (DV, _compute_mtp4___ROWS_Q_JIT))
        local_acc_cols = tl.broadcast_to(offs_q[None, :], (DV, _compute_mtp4___ROWS_Q_JIT))
        local_acc_slot = tl.full((DV, _compute_mtp4___ROWS_Q_JIT), BUNDLE_TASK_IDX, tl.int32)
        local_lse_slot = tl.full((_compute_mtp4___ROWS_Q_JIT,), BUNDLE_TASK_IDX, tl.int32)
        tl.store(tle.gpu.local_ptr(LOCAL_ACC_SMEM, (local_acc_slot, local_acc_rows, local_acc_cols)), acc)
        tl.store(tle.gpu.local_ptr(LOCAL_LSE_SMEM, (local_lse_slot, offs_q)), lse)
        return
    if not C4_REDUCTION_ONLY_RAW and task_mode == _compute_mtp4___DIRECT_MODE_JIT:
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + store_offs_v[:, None],
            acc,
            mask=valid_q[None, :],
        )
        return
    if MERGE_CLUSTER_SIZE == 4 and (not C4_REDUCTION_ONLY_RAW) and (task_mode == _compute_mtp4___SUBGROUP2_MODE_JIT):
        low_pair_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        low_pair_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        low_pair_l_remote = tle.remote(peer1_l_smem, 0, scope=mesh)
        high_pair_acc_remote = tle.remote(peer3_acc_smem, 2, scope=mesh)
        high_pair_lse_remote = tle.remote(peer3_lse_smem, 2, scope=mesh)
        high_pair_l_remote = tle.remote(peer3_l_smem, 2, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(low_pair_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(low_pair_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(low_pair_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(high_pair_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(high_pair_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(high_pair_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0 or cluster_rank == 2:
            if cluster_rank == 0:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
                if C4_DEFERRED_NORM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)))
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
                if C4_DEFERRED_NORM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer3_l_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(
                lse != -float("inf"), tl.exp2(lse - safe_max) if USE_LOG2 else tl.exp(lse - safe_max), 0.0
            )
            weight1 = tl.where(
                peer_lse != -float("inf"),
                tl.exp2(peer_lse - safe_max) if USE_LOG2 else tl.exp(peer_lse - safe_max),
                0.0,
            )
            if C4_DEFERRED_NORM:
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            subgroup_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            if C4_DEFERRED_NORM:
                subgroup_acc *= vscale
            subgroup_acc = tl.where(valid_group[None, :], subgroup_acc, 0.0)
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                subgroup_acc,
                mask=valid_q[None, :],
            )
        return
    if EXECUTION_STAGE == _compute_mtp4___EXECUTION_LOCAL_PARTIAL_JIT:
        if cluster_rank == 0:
            local_output_mask = valid_q[None, :]
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                acc,
                mask=local_output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                lse,
                mask=valid_q,
            )
        return
    if HEAD_SHARDED_DSM and C4_GLOBAL_DEFERRED_NORM and (MERGE_CLUSTER_SIZE == 4) and PAIRED_HEAD_FINALIZE:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tl.store(tle.gpu.local_ptr(partial_l_smem, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        paired_offs_h = tl.arange(0, 2)
        paired_offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
        paired_h_in_group = cluster_rank + paired_offs_h * 4
        paired_cols = paired_h_in_group[:, None] + paired_offs_m[None, :] * HEADS_PER_GROUP
        paired_rows_3d = tl.broadcast_to(offs_v[:, None, None], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        paired_cols_3d = tl.broadcast_to(paired_cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        paired_raw_acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
        paired_raw_m = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
        paired_raw_l = tl.zeros((2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
        for peer_rank in tl.static_range(0, 4):
            peer_acc_owner = tle.remote(partial_acc_smem, peer_rank, scope=mesh)
            peer_m_owner = tle.remote(partial_lse_smem, peer_rank, scope=mesh)
            peer_l_owner = tle.remote(partial_l_smem, peer_rank, scope=mesh)
            peer_raw_acc = tl.load(tle.gpu.local_ptr(peer_acc_owner, (paired_rows_3d, paired_cols_3d))).to(tl.float32)
            peer_raw_m = tl.load(tle.gpu.local_ptr(peer_m_owner, (paired_cols,)))
            peer_raw_l = tl.load(tle.gpu.local_ptr(peer_l_owner, (paired_cols,)))
            paired_next_m = tl.maximum(paired_raw_m, peer_raw_m)
            paired_valid_m = paired_next_m != -float("inf")
            paired_safe_m = tl.where(paired_valid_m, paired_next_m, 0.0)
            paired_old_w = tl.where(
                paired_raw_m != -float("inf"),
                tl.exp2(paired_raw_m - paired_safe_m) if USE_LOG2 else tl.exp(paired_raw_m - paired_safe_m),
                0.0,
            )
            paired_peer_w = tl.where(
                peer_raw_m != -float("inf"),
                tl.exp2(peer_raw_m - paired_safe_m) if USE_LOG2 else tl.exp(peer_raw_m - paired_safe_m),
                0.0,
            )
            paired_raw_acc = paired_raw_acc * paired_old_w[None, :, :] + peer_raw_acc * paired_peer_w[None, :, :]
            paired_raw_l = paired_raw_l * paired_old_w + peer_raw_l * paired_peer_w
            paired_raw_m = tl.where(paired_valid_m, paired_safe_m, -float("inf"))
        tle.distributed_barrier(mesh)
        paired_hq = hkv * HEADS_PER_GROUP + paired_h_in_group
        paired_valid = (paired_h_in_group < HEADS_PER_GROUP) & (paired_hq < H_Q)
        paired_mask = paired_valid[None, :, None]
        if not C4_REDUCTION_ONLY_RAW and group_count == 1:
            paired_scale = tl.where(paired_raw_l > 0.0, vscale / paired_raw_l, 0.0)
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + paired_offs_m[None, None, :] * O_STRIDE_M
                + paired_hq[None, :, None] * O_STRIDE_H
                + store_offs_v[:, None, None],
                paired_raw_acc * paired_scale[None, :, :],
                mask=paired_mask,
            )
            return
        tl.store(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + group_chunk * SO_STRIDE_C
            + paired_offs_m[None, None, :] * SO_STRIDE_M
            + paired_hq[None, :, None] * SO_STRIDE_H
            + store_offs_v[:, None, None],
            paired_raw_acc,
            mask=paired_mask,
        )
        scalar_mask = paired_valid[:, None]
        scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + group_chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + paired_offs_m[None, :] * LSE_STRIDE_M
            + paired_h_in_group[:, None] * LSE_STRIDE_HG
        )
        tl.store(scalar_base, paired_raw_m, mask=scalar_mask)
        tl.store(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, paired_raw_l, mask=scalar_mask)
        tle.distributed_barrier(mesh)
        if (
            EXECUTION_STAGE != _compute_mtp4___EXECUTION_FULL_JIT
            and EXECUTION_STAGE != _compute_mtp4___EXECUTION_ELECTION_ONLY_JIT
        ):
            return
        counter_idx = hkv * B + batch
        rank0_is_last = tl.full((), 0, tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
            rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        tle.distributed_barrier(mesh)
        rank0_cta = cta - cluster_rank
        if cluster_rank == 0:
            is_last_cluster = rank0_is_last != 0
        else:
            is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_last_cluster:
            if group_count == 2:
                _compute_mtp4___cluster_paired_head_raw_two_chunk_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    cluster_rank,
                    vscale,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    USE_LOG2,
                    MAX_FINAL_CHUNKS,
                )
            else:
                _compute_mtp4___cluster_paired_head_raw_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    group_count,
                    vscale,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    USE_LOG2,
                    MAX_FINAL_CHUNKS,
                )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
        return
    if HEAD_SHARDED_DSM:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tle.distributed_barrier(mesh)
        final_offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
        if PAIRED_HEAD_FINALIZE and MERGE_CLUSTER_SIZE == 4:
            paired_offs_h = tl.arange(0, 2)
            paired_h_in_group = cluster_rank + paired_offs_h * 4
            paired_hq = hkv * HEADS_PER_GROUP + paired_h_in_group
            paired_valid_head = (paired_h_in_group < HEADS_PER_GROUP) & (paired_hq < H_Q)
            _compute_mtp4___paired_head_sharded_dsm_merge_mtp4(
                partial_acc_smem,
                partial_lse_smem,
                head_acc_smem,
                head_lse_smem,
                mesh,
                cluster_rank,
                HEADS_PER_GROUP,
                DV,
                USE_LOG2,
            )
            paired_head_rows = tl.broadcast_to(offs_v[:, None, None], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
            paired_cols = paired_offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + final_offs_m[None, :]
            paired_head_cols = tl.broadcast_to(paired_cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
            paired_head_acc = tl.load(tle.gpu.local_ptr(head_acc_smem, (paired_head_rows, paired_head_cols)))
            paired_head_lse = tl.load(tle.gpu.local_ptr(head_lse_smem, (paired_cols,)))
            if group_count == 1:
                tl.store(
                    OUT
                    + batch * O_STRIDE_B
                    + final_offs_m[None, None, :] * O_STRIDE_M
                    + paired_hq[None, :, None] * O_STRIDE_H
                    + store_offs_v[:, None, None],
                    paired_head_acc,
                    mask=paired_valid_head[None, :, None],
                )
            else:
                tl.store(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + group_chunk * SO_STRIDE_C
                    + final_offs_m[None, None, :] * SO_STRIDE_M
                    + paired_hq[None, :, None] * SO_STRIDE_H
                    + store_offs_v[:, None, None],
                    paired_head_acc,
                    mask=paired_valid_head[None, :, None],
                )
                tl.store(
                    LSE
                    + batch * LSE_STRIDE_B
                    + group_chunk * LSE_STRIDE_C
                    + hkv * LSE_STRIDE_HKV
                    + final_offs_m[None, :] * LSE_STRIDE_M
                    + paired_h_in_group[:, None] * LSE_STRIDE_HG,
                    paired_head_lse,
                    mask=paired_valid_head[:, None],
                )
        else:
            for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                final_hq = hkv * HEADS_PER_GROUP + final_h_in_group
                valid_head = (final_h_in_group < HEADS_PER_GROUP) & (final_hq < H_Q)
                _compute_mtp4___head_sharded_dsm_merge_mtp4(
                    partial_acc_smem,
                    partial_lse_smem,
                    head_acc_smem,
                    head_lse_smem,
                    mesh,
                    final_h_in_group,
                    HEADS_PER_GROUP,
                    DV,
                    MERGE_CLUSTER_SIZE,
                    USE_LOG2,
                )
                head_rows = tl.broadcast_to(offs_v[:, None], (DV, _compute_mtp4___NUM_SEQ_Q_JIT))
                head_cols = tl.broadcast_to(final_offs_m[None, :], (DV, _compute_mtp4___NUM_SEQ_Q_JIT))
                head_acc = tl.load(tle.gpu.local_ptr(head_acc_smem, (head_rows, head_cols)))
                head_lse = tl.load(tle.gpu.local_ptr(head_lse_smem, (final_offs_m,)))
                if group_count == 1:
                    tl.store(
                        OUT
                        + batch * O_STRIDE_B
                        + final_offs_m[None, :] * O_STRIDE_M
                        + final_hq * O_STRIDE_H
                        + store_offs_v[:, None],
                        head_acc,
                        mask=valid_head,
                    )
                else:
                    tl.store(
                        SPLIT_OUT
                        + batch * SO_STRIDE_B
                        + group_chunk * SO_STRIDE_C
                        + final_offs_m[None, :] * SO_STRIDE_M
                        + final_hq * SO_STRIDE_H
                        + store_offs_v[:, None],
                        head_acc,
                        mask=valid_head,
                    )
                    tl.store(
                        LSE
                        + batch * LSE_STRIDE_B
                        + group_chunk * LSE_STRIDE_C
                        + hkv * LSE_STRIDE_HKV
                        + final_offs_m * LSE_STRIDE_M
                        + final_h_in_group * LSE_STRIDE_HG,
                        head_lse,
                        mask=valid_head,
                    )
                tl.debug_barrier()
        tle.distributed_barrier(mesh)
        if group_count > 1 and EXECUTION_STAGE == _compute_mtp4___EXECUTION_FULL_JIT:
            counter_idx = hkv * B + batch
            rank0_is_last = tl.full((), 0, tl.int32)
            if cluster_rank == 0:
                tl.debug_barrier()
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
            tle.distributed_barrier(mesh)
            rank0_cta = cta - cluster_rank
            if FAST_FINALIZER_HANDOFF:
                if cluster_rank == 0:
                    is_last_cluster = rank0_is_last != 0
                else:
                    is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            else:
                is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            if is_last_cluster:
                if not FAST_FINALIZER_HANDOFF:
                    tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
                if PAIRED_HEAD_FINALIZE:
                    if MERGE_CLUSTER_SIZE == 2:
                        _compute_mtp4___cluster_quad_head_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale if DEFER_V_SCALE_FINAL else 1.0,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                        )
                    else:
                        _compute_mtp4___cluster_paired_head_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                        )
                else:
                    for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                        final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                        _compute_mtp4___cluster_cooperative_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            batch,
                            hkv,
                            final_h_in_group,
                            group_count,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                            REUSE_FINAL_WEIGHTS,
                        )
            if not SKIP_TRAILING_FINALIZER_BARRIER:
                tle.distributed_barrier(mesh)
            if cluster_rank == 0 and is_last_cluster:
                tl.debug_barrier()
                tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
        return
    group_acc = acc
    group_lse = lse
    group_raw_acc = acc
    group_raw_m = lse
    group_raw_l = tl.full((_compute_mtp4___ROWS_Q_JIT,), 1.0, tl.float32)
    if MERGE_CLUSTER_SIZE == 2:
        if SPLIT_D_DSM_MERGE:
            rank0_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
            rank1_acc_remote = tle.remote(peer_acc_smem, 1, scope=mesh)
            rank0_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
            rank1_lse_remote = tle.remote(peer_lse_smem, 1, scope=mesh)
            half_d = tl.arange(0, DV // 2)
            half_cols = tl.broadcast_to(offs_q[None, :], (DV // 2, _compute_mtp4___ROWS_Q_JIT))
            lo_rows = tl.broadcast_to(half_d[:, None], (DV // 2, _compute_mtp4___ROWS_Q_JIT))
            hi_rows = tl.broadcast_to((half_d + DV // 2)[:, None], (DV // 2, _compute_mtp4___ROWS_Q_JIT))
            if cluster_rank == 0:
                local_half = tle.extract_tile(acc, index=[0, 0], tile_shape=[DV // 2, _compute_mtp4___ROWS_Q_JIT])
                send_half = tle.extract_tile(acc, index=[1, 0], tile_shape=[DV // 2, _compute_mtp4___ROWS_Q_JIT])
                tl.store(tle.gpu.local_ptr(rank1_acc_remote, (hi_rows, half_cols)), send_half)
                tl.store(tle.gpu.local_ptr(rank1_lse_remote, (offs_q,)), lse)
            else:
                local_half = tle.extract_tile(acc, index=[1, 0], tile_shape=[DV // 2, _compute_mtp4___ROWS_Q_JIT])
                send_half = tle.extract_tile(acc, index=[0, 0], tile_shape=[DV // 2, _compute_mtp4___ROWS_Q_JIT])
                tl.store(tle.gpu.local_ptr(rank0_acc_remote, (lo_rows, half_cols)), send_half)
                tl.store(tle.gpu.local_ptr(rank0_lse_remote, (offs_q,)), lse)
            tle.distributed_barrier(mesh)
            if cluster_rank == 0:
                peer_half = tl.load(tle.gpu.local_ptr(peer_acc_smem, (lo_rows, half_cols)))
            else:
                peer_half = tl.load(tle.gpu.local_ptr(peer_acc_smem, (hi_rows, half_cols)))
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            local_weight = tl.where(
                lse != -float("inf"), tl.exp2(lse - safe_max) if USE_LOG2 else tl.exp(lse - safe_max), 0.0
            )
            peer_weight = tl.where(
                peer_lse != -float("inf"),
                tl.exp2(peer_lse - safe_max) if USE_LOG2 else tl.exp(peer_lse - safe_max),
                0.0,
            )
            denom = local_weight + peer_weight
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            merged_half = (local_half * local_weight[None, :] + peer_half * peer_weight[None, :]) / safe_denom[None, :]
            merged_half = tl.where(valid_group[None, :], merged_half, 0.0)
            group_lse = tl.where(
                valid_group, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
            )
            half_store_v = half_d + cluster_rank * (DV // 2)
            if group_count == 1:
                tl.store(
                    OUT
                    + batch * O_STRIDE_B
                    + seq_m[None, :] * O_STRIDE_M
                    + hq[None, :] * O_STRIDE_H
                    + half_store_v[:, None],
                    merged_half,
                    mask=valid_q[None, :],
                )
            else:
                tl.store(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + group_chunk * SO_STRIDE_C
                    + seq_m[None, :] * SO_STRIDE_M
                    + hq[None, :] * SO_STRIDE_H
                    + half_store_v[:, None],
                    merged_half,
                    mask=valid_q[None, :],
                )
                if cluster_rank == 0:
                    tl.store(
                        LSE
                        + batch * LSE_STRIDE_B
                        + group_chunk * LSE_STRIDE_C
                        + hkv * LSE_STRIDE_HKV
                        + seq_m * LSE_STRIDE_M
                        + h_in_group * LSE_STRIDE_HG,
                        group_lse,
                        mask=valid_q,
                    )
            tle.distributed_barrier(mesh)
        else:
            peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
            peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
            peer_l_remote = tle.remote(peer_l_smem, 0, scope=mesh)
            if cluster_rank == 1:
                tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer_l_remote, (offs_q,)), raw_l)
            tle.distributed_barrier(mesh)
            if cluster_rank == 0:
                peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
                if C4_DEFERRED_NORM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_q,)))
                max_lse = tl.maximum(lse, peer_lse)
                valid_group = max_lse != -float("inf")
                safe_max = tl.where(valid_group, max_lse, 0.0)
                weight0 = tl.where(
                    lse != -float("inf"), tl.exp2(lse - safe_max) if USE_LOG2 else tl.exp(lse - safe_max), 0.0
                )
                weight1 = tl.where(
                    peer_lse != -float("inf"),
                    tl.exp2(peer_lse - safe_max) if USE_LOG2 else tl.exp(peer_lse - safe_max),
                    0.0,
                )
                if C4_DEFERRED_NORM:
                    denom = raw_l * weight0 + peer_l * weight1
                else:
                    denom = weight0 + weight1
                safe_denom = tl.where(denom > 0.0, denom, 1.0)
                weighted_acc = acc * weight0[None, :] + peer_acc * weight1[None, :]
                group_raw_acc = weighted_acc
                group_raw_m = safe_max
                group_raw_l = denom
                group_acc = weighted_acc / safe_denom[None, :]
                if C4_DEFERRED_NORM:
                    group_acc *= vscale
                group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
                group_lse = tl.where(
                    valid_group, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
                )
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        peer1_l_remote = tle.remote(peer1_l_smem, 0, scope=mesh)
        peer2_l_remote = tle.remote(peer2_l_smem, 0, scope=mesh)
        peer3_l_remote = tle.remote(peer3_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer1_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_remote), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer1_l_remote), raw_l)
            else:
                tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer1_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 2:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer2_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer2_lse_remote), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer2_l_remote), raw_l)
            else:
                tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer2_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 3:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer3_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer3_lse_remote), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer3_l_remote), raw_l)
            else:
                tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
                if C4_DEFERRED_NORM:
                    tl.store(tle.gpu.local_ptr(peer3_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            if FULL_VIEW_DSM:
                lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem))
                lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem))
                lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem))
            else:
                lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
                lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
                lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(
                lse != -float("inf"), tl.exp2(lse - safe_max) if USE_LOG2 else tl.exp(lse - safe_max), 0.0
            )
            weight1 = tl.where(
                lse1 != -float("inf"), tl.exp2(lse1 - safe_max) if USE_LOG2 else tl.exp(lse1 - safe_max), 0.0
            )
            weight2 = tl.where(
                lse2 != -float("inf"), tl.exp2(lse2 - safe_max) if USE_LOG2 else tl.exp(lse2 - safe_max), 0.0
            )
            weight3 = tl.where(
                lse3 != -float("inf"), tl.exp2(lse3 - safe_max) if USE_LOG2 else tl.exp(lse3 - safe_max), 0.0
            )
            if C4_DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem))
                    l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem))
                    l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem))
                else:
                    l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)))
                    l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem, (offs_q,)))
                    l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem, (offs_q,)))
                denom = raw_l * weight0 + l1 * weight1 + l2 * weight2 + l3 * weight3
            else:
                denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight1[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight2[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight3[None, :]
            group_raw_acc = weighted_acc
            group_raw_m = safe_max
            group_raw_l = denom
            if C4_GLOBAL_DEFERRED_NORM:
                group_acc = weighted_acc
                group_lse = safe_max
            else:
                group_acc = weighted_acc / safe_denom[None, :]
                if C4_DEFERRED_NORM:
                    group_acc *= vscale
                group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
                group_lse = tl.where(
                    valid_group, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
                )
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        if C8_BF16_DEFERRED_NORM:
            tl.store(tle.gpu.local_ptr(partial_l_smem, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        peer1_l = tle.remote(partial_l_smem, 1, scope=mesh)
        peer2_l = tle.remote(partial_l_smem, 2, scope=mesh)
        peer3_l = tle.remote(partial_l_smem, 3, scope=mesh)
        peer4_l = tle.remote(partial_l_smem, 4, scope=mesh)
        peer5_l = tle.remote(partial_l_smem, 5, scope=mesh)
        peer6_l = tle.remote(partial_l_smem, 6, scope=mesh)
        peer7_l = tle.remote(partial_l_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(
                lse != -float("inf"), tl.exp2(lse - safe_max) if USE_LOG2 else tl.exp(lse - safe_max), 0.0
            )
            weight1 = tl.where(
                lse1 != -float("inf"), tl.exp2(lse1 - safe_max) if USE_LOG2 else tl.exp(lse1 - safe_max), 0.0
            )
            weight2 = tl.where(
                lse2 != -float("inf"), tl.exp2(lse2 - safe_max) if USE_LOG2 else tl.exp(lse2 - safe_max), 0.0
            )
            weight3 = tl.where(
                lse3 != -float("inf"), tl.exp2(lse3 - safe_max) if USE_LOG2 else tl.exp(lse3 - safe_max), 0.0
            )
            weight4 = tl.where(
                lse4 != -float("inf"), tl.exp2(lse4 - safe_max) if USE_LOG2 else tl.exp(lse4 - safe_max), 0.0
            )
            weight5 = tl.where(
                lse5 != -float("inf"), tl.exp2(lse5 - safe_max) if USE_LOG2 else tl.exp(lse5 - safe_max), 0.0
            )
            weight6 = tl.where(
                lse6 != -float("inf"), tl.exp2(lse6 - safe_max) if USE_LOG2 else tl.exp(lse6 - safe_max), 0.0
            )
            weight7 = tl.where(
                lse7 != -float("inf"), tl.exp2(lse7 - safe_max) if USE_LOG2 else tl.exp(lse7 - safe_max), 0.0
            )
            if C8_BF16_DEFERRED_NORM:
                l1 = tl.load(tle.gpu.local_ptr(peer1_l, (offs_q,)))
                l2 = tl.load(tle.gpu.local_ptr(peer2_l, (offs_q,)))
                l3 = tl.load(tle.gpu.local_ptr(peer3_l, (offs_q,)))
                l4 = tl.load(tle.gpu.local_ptr(peer4_l, (offs_q,)))
                l5 = tl.load(tle.gpu.local_ptr(peer5_l, (offs_q,)))
                l6 = tl.load(tle.gpu.local_ptr(peer6_l, (offs_q,)))
                l7 = tl.load(tle.gpu.local_ptr(peer7_l, (offs_q,)))
                denom = (
                    raw_l * weight0
                    + l1 * weight1
                    + l2 * weight2
                    + l3 * weight3
                    + l4 * weight4
                    + l5 * weight5
                    + l6 * weight6
                    + l7 * weight7
                )
            else:
                denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            if C8_BF16_DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(
                valid_group, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + safe_max, -float("inf")
            )
        tle.distributed_barrier(mesh)
    if cluster_rank == 0 and (not SPLIT_D_DSM_MERGE):
        output_mask = ((seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q))[None, :]
        if not C4_REDUCTION_ONLY_RAW and group_count == 1:
            output_acc = group_acc
            if DEFER_V_SCALE_FINAL:
                output_acc *= vscale
            if C4_GLOBAL_DEFERRED_NORM:
                output_denom = tl.where(group_raw_l > 0.0, group_raw_l, 1.0)
                output_acc = group_raw_acc / output_denom[None, :] * vscale
                output_acc = tl.where((group_raw_l > 0.0)[None, :], output_acc, 0.0)
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                output_acc,
                mask=output_mask,
            )
        else:
            stored_acc = group_raw_acc if C4_GLOBAL_DEFERRED_NORM else group_acc
            defer_exact_two_store = (
                WINNER_LOCAL_REUSE_FINALIZER & C4_GLOBAL_DEFERRED_NORM & (MERGE_CLUSTER_SIZE == 4) & (group_count == 2)
            )
            defer_tail_store = TAIL_RAW_DSM_REUSE & (group_count > 2) & (group_chunk == group_count - 1)
            if defer_tail_store:
                tl.store(tle.gpu.local_ptr(peer1_acc_smem), group_raw_acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_smem), group_raw_m)
                tl.store(tle.gpu.local_ptr(peer1_l_smem), group_raw_l)
            if not defer_exact_two_store | defer_tail_store:
                tl.store(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + group_chunk * SO_STRIDE_C
                    + seq_m[None, :] * SO_STRIDE_M
                    + hq[None, :] * SO_STRIDE_H
                    + store_offs_v[:, None],
                    stored_acc,
                    mask=output_mask,
                )
                tl.store(
                    LSE
                    + batch * LSE_STRIDE_B
                    + group_chunk * LSE_STRIDE_C
                    + hkv * LSE_STRIDE_HKV
                    + seq_m * LSE_STRIDE_M
                    + h_in_group * LSE_STRIDE_HG,
                    group_raw_m if C4_GLOBAL_DEFERRED_NORM else group_lse,
                    mask=(seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
                )
                if C4_GLOBAL_DEFERRED_NORM:
                    tl.store(
                        LSE
                        + batch * LSE_STRIDE_B
                        + (group_chunk + MAX_FINAL_CHUNKS) * LSE_STRIDE_C
                        + hkv * LSE_STRIDE_HKV
                        + seq_m * LSE_STRIDE_M
                        + h_in_group * LSE_STRIDE_HG,
                        group_raw_l,
                        mask=(seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
                    )
    if (
        C4_HIERARCHICAL_FINALIZE
        and EXECUTION_STAGE == _compute_mtp4___EXECUTION_FULL_JIT
        and (group_count > C4_TREE_PACK)
    ):
        counter_idx = hkv * B + batch
        num_counters = B * (H_Q // HEADS_PER_GROUP)
        subgroup = group_chunk // C4_TREE_PACK
        subgroup_start = subgroup * C4_TREE_PACK
        subgroup_remaining = group_count - subgroup_start
        subgroup_count = tl.where(subgroup_remaining < C4_TREE_PACK, subgroup_remaining, C4_TREE_PACK)
        subgroup_counter_idx = num_counters + counter_idx * MAX_FINAL_CHUNKS + subgroup
        rank0_is_subgroup_last = tl.full((), 0, tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            subgroup_ticket = tl.atomic_add(COMPLETION + subgroup_counter_idx, 1, sem="acq_rel", scope="gpu")
            rank0_is_subgroup_last = (subgroup_ticket == subgroup_count - 1).to(tl.int32)
            tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_subgroup_last, sem="release", scope="gpu")
        tle.distributed_barrier(mesh)
        rank0_cta = cta - cluster_rank
        if cluster_rank == 0:
            is_subgroup_last = rank0_is_subgroup_last != 0
        else:
            is_subgroup_last = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_subgroup_last:
            scratch_chunk = MAX_FINAL_CHUNKS + subgroup
            _compute_mtp4___cluster_paired_head_subgroup_reduce_mtp4(
                SPLIT_OUT,
                LSE,
                batch,
                hkv,
                cluster_rank,
                subgroup_start,
                subgroup_count,
                scratch_chunk,
                H_Q,
                HEADS_PER_GROUP,
                DV,
                SO_STRIDE_B,
                SO_STRIDE_C,
                SO_STRIDE_M,
                SO_STRIDE_H,
                LSE_STRIDE_B,
                LSE_STRIDE_C,
                LSE_STRIDE_HKV,
                LSE_STRIDE_M,
                LSE_STRIDE_HG,
                USE_LOG2,
                C4_TREE_PACK,
            )
            tle.distributed_barrier(mesh)
            rank0_is_top_last = tl.full((), 0, tl.int32)
            if cluster_rank == 0:
                tl.debug_barrier()
                num_subgroups = (group_count + C4_TREE_PACK - 1) // C4_TREE_PACK
                top_ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_top_last = (top_ticket == num_subgroups - 1).to(tl.int32)
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_top_last, sem="release", scope="gpu")
                tl.atomic_xchg(COMPLETION + subgroup_counter_idx, 0, sem="release", scope="gpu")
            tle.distributed_barrier(mesh)
            if cluster_rank == 0:
                is_top_last = rank0_is_top_last != 0
            else:
                is_top_last = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            if is_top_last:
                num_subgroups = (group_count + C4_TREE_PACK - 1) // C4_TREE_PACK
                _compute_mtp4___cluster_paired_head_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    num_subgroups,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    USE_LOG2,
                    MAX_FINAL_CHUNKS // C4_TREE_PACK,
                    1,
                    4,
                    0,
                    MAX_FINAL_CHUNKS,
                )
            if not SKIP_TRAILING_FINALIZER_BARRIER:
                tle.distributed_barrier(mesh)
            if cluster_rank == 0 and is_top_last:
                tl.debug_barrier()
                tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
        return
    if (
        CLUSTER_COOPERATIVE_FINALIZE
        and (
            EXECUTION_STAGE == _compute_mtp4___EXECUTION_FULL_JIT
            or EXECUTION_STAGE == _compute_mtp4___EXECUTION_ELECTION_ONLY_JIT
        )
        and (C4_REDUCTION_ONLY_RAW or group_count > 1)
    ):
        counter_idx = hkv * B + batch
        num_counters = B * (H_Q // HEADS_PER_GROUP)
        ready_counter_idx = num_counters + counter_idx * MAX_FINAL_CHUNKS
        rank0_is_last = tl.full((), 0, tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_REUSE:
                deterministic_owner = group_chunk == 0 if DETERMINISTIC_FIRST_OWNER else group_chunk == group_count - 1
                if DISTRIBUTED_READY_FLAGS:
                    if deterministic_owner:
                        ready_offsets = tl.arange(0, MAX_FINAL_CHUNKS)
                        ready_mask = ready_offsets < group_count - 1
                        ready_values = tl.atomic_add(
                            COMPLETION + ready_counter_idx + ready_offsets,
                            tl.zeros((MAX_FINAL_CHUNKS,), tl.int32),
                            mask=ready_mask,
                            sem="acquire",
                            scope="gpu",
                        )
                        pending = tl.max(tl.where(ready_mask & (ready_values == 0), 1, 0), axis=0)
                        while pending != 0:
                            ready_values = tl.atomic_add(
                                COMPLETION + ready_counter_idx + ready_offsets,
                                tl.zeros((MAX_FINAL_CHUNKS,), tl.int32),
                                mask=ready_mask,
                                sem="acquire",
                                scope="gpu",
                            )
                            pending = tl.max(tl.where(ready_mask & (ready_values == 0), 1, 0), axis=0)
                        tl.store(
                            COMPLETION + ready_counter_idx + ready_offsets,
                            tl.zeros((MAX_FINAL_CHUNKS,), tl.int32),
                            mask=ready_mask,
                        )
                        rank0_is_last = tl.full((), 1, tl.int32)
                    else:
                        tl.atomic_xchg(COMPLETION + ready_counter_idx + group_chunk, 1, sem="release", scope="gpu")
                elif deterministic_owner:
                    ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(COMPLETION + counter_idx, 1, sem="release", scope="gpu")
            elif SHARDED_ELECTION and group_count > 8:
                subgroup = group_chunk // 8
                subgroup_start = subgroup * 8
                subgroup_remaining = group_count - subgroup_start
                subgroup_count = tl.minimum(subgroup_remaining, 8)
                subgroup_counter_idx = num_counters + counter_idx * MAX_FINAL_CHUNKS + subgroup
                subgroup_ticket = tl.atomic_add(COMPLETION + subgroup_counter_idx, 1, sem="acq_rel", scope="gpu")
                if subgroup_ticket == subgroup_count - 1:
                    num_subgroups = (group_count + 7) // 8
                    top_ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                    rank0_is_last = (top_ticket == num_subgroups - 1).to(tl.int32)
                    tl.atomic_xchg(COMPLETION + subgroup_counter_idx, 0, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            if WINNER_LOCAL_REUSE_FINALIZER and C4_GLOBAL_DEFERRED_NORM and (MERGE_CLUSTER_SIZE == 4):
                if group_count == 2:
                    if rank0_is_last != 0:
                        tl.store(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols)), group_raw_acc)
                        tl.store(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)), group_raw_m)
                        tl.store(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)), group_raw_l)
                        tl.debug_barrier()
                        peer_ready = tl.atomic_add(COMPLETION + ready_counter_idx, 0, sem="acquire", scope="gpu")
                        while peer_ready == 0:
                            peer_ready = tl.atomic_add(COMPLETION + ready_counter_idx, 0, sem="acquire", scope="gpu")
                    else:
                        output_mask = (
                            (seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
                        )[None, :]
                        tl.store(
                            SPLIT_OUT
                            + batch * SO_STRIDE_B
                            + group_chunk * SO_STRIDE_C
                            + seq_m[None, :] * SO_STRIDE_M
                            + hq[None, :] * SO_STRIDE_H
                            + store_offs_v[:, None],
                            group_raw_acc,
                            mask=output_mask,
                        )
                        scalar_mask = (
                            (seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
                        )
                        tl.store(
                            LSE
                            + batch * LSE_STRIDE_B
                            + group_chunk * LSE_STRIDE_C
                            + hkv * LSE_STRIDE_HKV
                            + seq_m * LSE_STRIDE_M
                            + h_in_group * LSE_STRIDE_HG,
                            group_raw_m,
                            mask=scalar_mask,
                        )
                        tl.store(
                            LSE
                            + batch * LSE_STRIDE_B
                            + (group_chunk + MAX_FINAL_CHUNKS) * LSE_STRIDE_C
                            + hkv * LSE_STRIDE_HKV
                            + seq_m * LSE_STRIDE_M
                            + h_in_group * LSE_STRIDE_HG,
                            group_raw_l,
                            mask=scalar_mask,
                        )
                        tl.debug_barrier()
                        tl.atomic_xchg(COMPLETION + ready_counter_idx, 1, sem="release", scope="gpu")
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_REUSE):
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        rank0_cta = cta - cluster_rank
        if DETERMINISTIC_TAIL_REUSE:
            if TAIL_ONLY_ELECTION_BARRIER:
                deterministic_owner = group_chunk == 0 if DETERMINISTIC_FIRST_OWNER else group_chunk == group_count - 1
                if deterministic_owner:
                    tle.distributed_barrier(mesh)
            else:
                tle.distributed_barrier(mesh)
            is_last_cluster = group_chunk == 0 if DETERMINISTIC_FIRST_OWNER else group_chunk == group_count - 1
        elif DSM_ELECTION_HANDOFF:
            rank1_flag = tle.remote(peer1_lse_smem, 1, scope=mesh)
            rank2_flag = tle.remote(peer1_lse_smem, 2, scope=mesh)
            rank3_flag = tle.remote(peer1_lse_smem, 3, scope=mesh)
            if cluster_rank == 0:
                winner_bit = rank0_is_last.to(tl.float32)
                tl.store(tle.gpu.local_ptr(peer1_lse_smem, (0,)), winner_bit)
                tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
            tle.distributed_barrier(mesh)
            is_last_cluster = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (0,))) != 0.0
        elif RANK0_ONLY_FINALIZER:
            is_last_cluster = (cluster_rank == 0) & (rank0_is_last != 0)
        else:
            tle.distributed_barrier(mesh)
            if FAST_FINALIZER_HANDOFF:
                if cluster_rank == 0:
                    is_last_cluster = rank0_is_last != 0
                else:
                    is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            else:
                is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if EXECUTION_STAGE == _compute_mtp4___EXECUTION_ELECTION_ONLY_JIT:
            if cluster_rank == 0 and rank0_is_last != 0:
                tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
            return
        if is_last_cluster:
            if not FAST_FINALIZER_HANDOFF and (not RANK0_ONLY_FINALIZER):
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
            if WINNER_LOCAL_REUSE_FINALIZER and (not C4_GLOBAL_DEFERRED_NORM):
                _compute_mtp4___cluster_winner_local_reuse_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    group_acc,
                    group_lse,
                    batch,
                    hkv,
                    group_chunk,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    USE_LOG2,
                )
            elif RANK0_ONLY_FINALIZER:
                if PAIRED_HEAD_FINALIZE:
                    _compute_mtp4___cluster_quad_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        0,
                        group_count,
                        vscale if DEFER_V_SCALE_FINAL else 1.0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                        MAX_FINAL_CHUNKS,
                    )
                    _compute_mtp4___cluster_quad_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        1,
                        group_count,
                        vscale if DEFER_V_SCALE_FINAL else 1.0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                        MAX_FINAL_CHUNKS,
                    )
                else:
                    _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                    )
                    _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        1,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                    )
            elif QUAD_HEAD_TWO_CHUNK_FINALIZE:
                _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    cluster_rank,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    USE_LOG2,
                )
            elif PAIRED_HEAD_FINALIZE:
                if MERGE_CLUSTER_SIZE == 2:
                    _compute_mtp4___cluster_quad_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        vscale if DEFER_V_SCALE_FINAL else 1.0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                        MAX_FINAL_CHUNKS,
                    )
                elif C4_EXACT_TWO_FINALIZE and group_count == 2:
                    _compute_mtp4___cluster_paired_head_two_chunk_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        cluster_rank,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                    )
                elif C4_FINALIZER_MODE < 0:
                    _compute_mtp4___cluster_paired_head_finalize_dynamic_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                        -C4_FINALIZER_MODE,
                    )
                elif C4_GLOBAL_DEFERRED_NORM:
                    if group_count == 2:
                        if WINNER_LOCAL_REUSE_FINALIZER:
                            _compute_mtp4___cluster_paired_head_raw_two_chunk_reuse_mtp4(
                                SPLIT_OUT,
                                LSE,
                                OUT,
                                peer1_acc_smem,
                                peer1_lse_smem,
                                peer1_l_smem,
                                mesh,
                                batch,
                                hkv,
                                cluster_rank,
                                group_chunk,
                                vscale,
                                H_Q,
                                HEADS_PER_GROUP,
                                DV,
                                SO_STRIDE_B,
                                SO_STRIDE_C,
                                SO_STRIDE_M,
                                SO_STRIDE_H,
                                LSE_STRIDE_B,
                                LSE_STRIDE_C,
                                LSE_STRIDE_HKV,
                                LSE_STRIDE_M,
                                LSE_STRIDE_HG,
                                O_STRIDE_B,
                                O_STRIDE_M,
                                O_STRIDE_H,
                                USE_LOG2,
                                MAX_FINAL_CHUNKS,
                            )
                        else:
                            _compute_mtp4___cluster_paired_head_raw_two_chunk_finalize_mtp4(
                                SPLIT_OUT,
                                LSE,
                                OUT,
                                batch,
                                hkv,
                                cluster_rank,
                                vscale,
                                H_Q,
                                HEADS_PER_GROUP,
                                DV,
                                SO_STRIDE_B,
                                SO_STRIDE_C,
                                SO_STRIDE_M,
                                SO_STRIDE_H,
                                LSE_STRIDE_B,
                                LSE_STRIDE_C,
                                LSE_STRIDE_HKV,
                                LSE_STRIDE_M,
                                LSE_STRIDE_HG,
                                O_STRIDE_B,
                                O_STRIDE_M,
                                O_STRIDE_H,
                                USE_LOG2,
                                MAX_FINAL_CHUNKS,
                            )
                    elif TAIL_RAW_DSM_REUSE:
                        _compute_mtp4___cluster_paired_head_raw_tail_reuse_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            peer1_acc_smem,
                            peer1_lse_smem,
                            peer1_l_smem,
                            mesh,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                        )
                    elif TMA_RAW_FINALIZER:
                        _compute_mtp4___cluster_paired_head_raw_tma_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            final_acc_smem,
                            final_acc_full,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            B,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                        )
                    elif RECOMPUTE_RAW_FINALIZER:
                        _compute_mtp4___cluster_paired_head_raw_recompute_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                        )
                    elif STREAMING_RAW_FINALIZER:
                        _compute_mtp4___cluster_paired_head_raw_streaming_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                        )
                    else:
                        _compute_mtp4___cluster_paired_head_raw_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                            DUAL_ACC_RAW_FINALIZER,
                            CHUNK4_RAW_FINALIZER,
                            REGISTER_RAW_WEIGHTS,
                            HEAD_MAJOR_RAW_REDUCE,
                        )
                else:
                    _compute_mtp4___cluster_paired_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        USE_LOG2,
                        MAX_FINAL_CHUNKS,
                        C4_FINALIZER_MODE,
                    )
            else:
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    if TWO_CHUNK_FINALIZE:
                        _compute_mtp4___cluster_two_chunk_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            batch,
                            hkv,
                            final_h_in_group,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                        )
                    else:
                        _compute_mtp4___cluster_cooperative_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            batch,
                            hkv,
                            final_h_in_group,
                            group_count,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            USE_LOG2,
                            MAX_FINAL_CHUNKS,
                            REUSE_FINAL_WEIGHTS,
                        )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        elif TAIL_RAW_DSM_REUSE and is_last_cluster:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            if (
                WINNER_LOCAL_REUSE_FINALIZER
                and C4_GLOBAL_DEFERRED_NORM
                and (MERGE_CLUSTER_SIZE == 4)
                and (group_count == 2)
            ):
                tl.atomic_xchg(COMPLETION + ready_counter_idx, 0, sem="release", scope="gpu")
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")


@triton.jit
def _compute_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle_kernel(
    Q,
    K_DESC,
    VT_DESC,
    KS_DESC,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    LOGICAL_NUM_CLUSTERS: tl.constexpr,
    PERSISTENT_NUM_CLUSTERS: tl.constexpr,
    RAW_PAGED_NHD_OR_HND: tl.constexpr = False,
    FULL_MATRIX_RS: tl.constexpr = False,
    TRANSPOSED_MEMDESC_RS: tl.constexpr = False,
    DIRECT_V_SHARED_SHARED: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    TLE_SHARED_SHARED: tl.constexpr = False,
    WIDE_VIEW_V_RS: tl.constexpr = False,
    FULL_VIEW_P_STORE: tl.constexpr = False,
    FULL_VIEW_DSM: tl.constexpr = False,
    DUAL_ACC_RAW_FINALIZER: tl.constexpr = False,
    CHUNK4_RAW_FINALIZER: tl.constexpr = False,
    SHARDED_ELECTION: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    STREAMING_RAW_FINALIZER: tl.constexpr = False,
    DETERMINISTIC_TAIL_REUSE: tl.constexpr = False,
    RECOMPUTE_RAW_FINALIZER: tl.constexpr = False,
    TAIL_ONLY_ELECTION_BARRIER: tl.constexpr = False,
    TMA_RAW_FINALIZER: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp4__EXECUTION_FULL,
    CLUSTER_COOPERATIVE_FINALIZE: tl.constexpr = False,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    REUSE_FINAL_WEIGHTS: tl.constexpr = False,
    HEAD_SHARDED_DSM: tl.constexpr = False,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    TWO_CHUNK_FINALIZE: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    FAST_FINALIZER_HANDOFF: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    WINNER_LOCAL_REUSE_FINALIZER: tl.constexpr = False,
    SPLIT_D_DSM_MERGE: tl.constexpr = False,
    PDL_NOTIFY: tl.constexpr = False,
    TMA_STAGES: tl.constexpr = 2,
    C4_FINALIZER_TILE: tl.constexpr = 1,
    CTA_ROLE_RANK0_TILES: tl.constexpr = -1,
    K_PER_TOKEN_V_PER_HEAD: tl.constexpr = False,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    PREFETCH_K_SCALE: tl.constexpr = False,
    TMA_K_SCALE: tl.constexpr = False,
    FRAGMENT_K_SCALE_LOAD: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    DEFER_V_SCALE_FINAL: tl.constexpr = False,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    """Launch exactly one logical task per CTA (clean nonpersistent path)."""
    cta = tl.program_id(0)
    if STATIC_SCHED:
        logical_cluster = cta // MERGE_CLUSTER_SIZE
        static_group = logical_cluster % STATIC_MAX_GROUPS
        static_sequence = logical_cluster // STATIC_MAX_GROUPS
        static_batch = static_sequence % B
        static_total_len = tl.load(SEQLENS_KV + static_batch).to(tl.int32)
        static_num_chunks = (static_total_len + STATIC_CHUNK_TOKENS - 1) // STATIC_CHUNK_TOKENS
        static_group_count = (static_num_chunks + MERGE_CLUSTER_SIZE - 1) // MERGE_CLUSTER_SIZE
        static_valid = static_group < static_group_count
        hkv = 0
    else:
        task_base = (cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        hkv = tl.load(TASK_MAP + task_base + 0)
        static_valid = False
    if STATIC_SCHED and static_valid or (not STATIC_SCHED and hkv >= 0):
        _compute_mtp4___fp8_kvpertensor_decode_mtp4_pure_tle_task(
            Q,
            K_DESC,
            VT_DESC,
            KS_DESC,
            BLOCK_IDS,
            TASK_MAP,
            SEQLENS_KV,
            QSCALE,
            KSCALE,
            VSCALE,
            SPLIT_OUT,
            LSE,
            COMPLETION,
            LAST_FLAGS,
            OUT,
            0,
            0,
            0,
            0,
            0,
            0,
            cta,
            0,
            mesh,
            B,
            H_Q,
            HEADS_PER_GROUP,
            D,
            DV,
            BLOCK_SIZE,
            MAX_BLOCKS,
            BLOCK_N,
            Q_STRIDE_B,
            Q_STRIDE_M,
            Q_STRIDE_H,
            QS_STRIDE_B,
            QS_STRIDE_M,
            QS_STRIDE_H,
            SO_STRIDE_B,
            SO_STRIDE_C,
            SO_STRIDE_M,
            SO_STRIDE_H,
            LSE_STRIDE_B,
            LSE_STRIDE_C,
            LSE_STRIDE_HKV,
            LSE_STRIDE_M,
            LSE_STRIDE_HG,
            O_STRIDE_B,
            O_STRIDE_M,
            O_STRIDE_H,
            USE_LOG2,
            RAW_PAGED_NHD_OR_HND,
            FULL_MATRIX_RS,
            TRANSPOSED_MEMDESC_RS,
            DIRECT_V_SHARED_SHARED,
            LDSM_REGISTER_SHARED,
            TLE_SHARED_SHARED,
            WIDE_VIEW_V_RS,
            FULL_VIEW_P_STORE,
            FULL_VIEW_DSM,
            DUAL_ACC_RAW_FINALIZER,
            CHUNK4_RAW_FINALIZER,
            SHARDED_ELECTION,
            DSM_ELECTION_HANDOFF,
            STREAMING_RAW_FINALIZER,
            DETERMINISTIC_TAIL_REUSE,
            RECOMPUTE_RAW_FINALIZER,
            TAIL_ONLY_ELECTION_BARRIER,
            TMA_RAW_FINALIZER,
            MERGE_CLUSTER_SIZE,
            EXECUTION_STAGE,
            CLUSTER_COOPERATIVE_FINALIZE,
            MAX_FINAL_CHUNKS,
            REUSE_FINAL_WEIGHTS,
            HEAD_SHARDED_DSM,
            PAIRED_HEAD_FINALIZE,
            TWO_CHUNK_FINALIZE,
            QUAD_HEAD_TWO_CHUNK_FINALIZE,
            FAST_FINALIZER_HANDOFF,
            RANK0_ONLY_FINALIZER,
            SKIP_TRAILING_FINALIZER_BARRIER,
            WINNER_LOCAL_REUSE_FINALIZER,
            SPLIT_D_DSM_MERGE,
            False,
            True,
            False,
            False,
            TMA_STAGES,
            C4_FINALIZER_TILE,
            False,
            False,
            CTA_ROLE_RANK0_TILES,
            False,
            K_PER_TOKEN_V_PER_HEAD,
            KS_STRIDE_BLOCK,
            KS_STRIDE_TOKEN,
            KS_STRIDE_HEAD,
            KS_STRIDE_D,
            PREFETCH_K_SCALE,
            TMA_K_SCALE,
            FRAGMENT_K_SCALE_LOAD,
            PRECOMBINE_Q_SCALE,
            DEFER_V_SCALE_FINAL,
            STATIC_SCHED,
            STATIC_CHUNK_TOKENS,
            STATIC_MAX_GROUPS,
        )
    if PDL_NOTIFY:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _compute_mtp4__fp8_kvpertensor_decode_mtp4_persistent_kernel(
    Q,
    K_DESC,
    VT_DESC,
    KS_DESC,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    LOGICAL_NUM_CLUSTERS: tl.constexpr,
    PERSISTENT_NUM_CLUSTERS: tl.constexpr,
    RAW_PAGED_NHD_OR_HND: tl.constexpr = False,
    FULL_MATRIX_RS: tl.constexpr = False,
    TRANSPOSED_MEMDESC_RS: tl.constexpr = False,
    DIRECT_V_SHARED_SHARED: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    TLE_SHARED_SHARED: tl.constexpr = False,
    WIDE_VIEW_V_RS: tl.constexpr = False,
    FULL_VIEW_P_STORE: tl.constexpr = False,
    FULL_VIEW_DSM: tl.constexpr = False,
    DUAL_ACC_RAW_FINALIZER: tl.constexpr = False,
    CHUNK4_RAW_FINALIZER: tl.constexpr = False,
    SHARDED_ELECTION: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    STREAMING_RAW_FINALIZER: tl.constexpr = False,
    DETERMINISTIC_TAIL_REUSE: tl.constexpr = False,
    RECOMPUTE_RAW_FINALIZER: tl.constexpr = False,
    TAIL_ONLY_ELECTION_BARRIER: tl.constexpr = False,
    TMA_RAW_FINALIZER: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp4__EXECUTION_FULL,
    CLUSTER_COOPERATIVE_FINALIZE: tl.constexpr = False,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    REUSE_FINAL_WEIGHTS: tl.constexpr = False,
    HEAD_SHARDED_DSM: tl.constexpr = False,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    TWO_CHUNK_FINALIZE: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    FAST_FINALIZER_HANDOFF: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    WINNER_LOCAL_REUSE_FINALIZER: tl.constexpr = False,
    SPLIT_D_DSM_MERGE: tl.constexpr = False,
    PDL_NOTIFY: tl.constexpr = False,
    TMA_STAGES: tl.constexpr = 2,
    C4_FINALIZER_TILE: tl.constexpr = 1,
    CTA_ROLE_RANK0_TILES: tl.constexpr = -1,
    K_PER_TOKEN_V_PER_HEAD: tl.constexpr = False,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    PREFETCH_K_SCALE: tl.constexpr = False,
    TMA_K_SCALE: tl.constexpr = False,
    FRAGMENT_K_SCALE_LOAD: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    DEFER_V_SCALE_FINAL: tl.constexpr = False,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    """Run a two-epoch C4 micro-persistent loop with K/V lookahead.

    The launch grid contains only ``PERSISTENT_NUM_CLUSTERS`` physical
    clusters.  Each one consumes logical cluster IDs separated by that fixed
    stride.  Task-map records and global election flags remain indexed by the
    logical CTA ID, while TMA mbarrier parity is carried across reused tasks.
    """
    launch_cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    launch_cluster = (launch_cta - cluster_rank) // MERGE_CLUSTER_SIZE
    if CTA_ROLE_RANK0_TILES >= 0:
        lookahead_k_smem = 0
        lookahead_v_smem = 0
        lookahead_k_full = 0
        lookahead_v_full = 0
    else:
        lookahead_k_smem = tle.gpu.alloc([BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
        lookahead_v_smem = tle.gpu.alloc([BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
        lookahead_k_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=BLOCK_N * D)
        lookahead_v_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=BLOCK_N * DV)
    copy_iter = tl.full((), 0, tl.int32)
    for epoch in tl.static_range(0, 2):
        logical_cluster = launch_cluster + epoch * PERSISTENT_NUM_CLUSTERS
        logical_cta = logical_cluster * MERGE_CLUSTER_SIZE + cluster_rank
        task_base = (logical_cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        hkv = tl.load(TASK_MAP + task_base + 0, mask=logical_cluster < LOGICAL_NUM_CLUSTERS, other=-1)
        if logical_cluster < LOGICAL_NUM_CLUSTERS and hkv >= 0:
            seq_len = tl.load(TASK_MAP + task_base + 4)
            task_mode = tl.load(TASK_MAP + task_base + 9)
            next_logical_cluster = logical_cluster + PERSISTENT_NUM_CLUSTERS
            next_valid = next_logical_cluster < LOGICAL_NUM_CLUSTERS
            next_logical_cta = tl.where(next_valid, next_logical_cluster * MERGE_CLUSTER_SIZE + cluster_rank, 0)
            _compute_mtp4___fp8_kvpertensor_decode_mtp4_pure_tle_task(
                Q,
                K_DESC,
                VT_DESC,
                KS_DESC,
                BLOCK_IDS,
                TASK_MAP,
                SEQLENS_KV,
                QSCALE,
                KSCALE,
                VSCALE,
                SPLIT_OUT,
                LSE,
                COMPLETION,
                LAST_FLAGS,
                OUT,
                lookahead_k_smem,
                lookahead_v_smem,
                lookahead_k_full,
                lookahead_v_full,
                next_valid.to(tl.int32),
                next_logical_cta,
                logical_cta,
                copy_iter,
                mesh,
                B,
                H_Q,
                HEADS_PER_GROUP,
                D,
                DV,
                BLOCK_SIZE,
                MAX_BLOCKS,
                BLOCK_N,
                Q_STRIDE_B,
                Q_STRIDE_M,
                Q_STRIDE_H,
                QS_STRIDE_B,
                QS_STRIDE_M,
                QS_STRIDE_H,
                SO_STRIDE_B,
                SO_STRIDE_C,
                SO_STRIDE_M,
                SO_STRIDE_H,
                LSE_STRIDE_B,
                LSE_STRIDE_C,
                LSE_STRIDE_HKV,
                LSE_STRIDE_M,
                LSE_STRIDE_HG,
                O_STRIDE_B,
                O_STRIDE_M,
                O_STRIDE_H,
                USE_LOG2,
                RAW_PAGED_NHD_OR_HND,
                FULL_MATRIX_RS,
                TRANSPOSED_MEMDESC_RS,
                DIRECT_V_SHARED_SHARED,
                LDSM_REGISTER_SHARED,
                TLE_SHARED_SHARED,
                WIDE_VIEW_V_RS,
                FULL_VIEW_P_STORE,
                FULL_VIEW_DSM,
                DUAL_ACC_RAW_FINALIZER,
                CHUNK4_RAW_FINALIZER,
                SHARDED_ELECTION,
                DSM_ELECTION_HANDOFF,
                STREAMING_RAW_FINALIZER,
                DETERMINISTIC_TAIL_REUSE,
                RECOMPUTE_RAW_FINALIZER,
                TAIL_ONLY_ELECTION_BARRIER,
                TMA_RAW_FINALIZER,
                MERGE_CLUSTER_SIZE,
                EXECUTION_STAGE,
                CLUSTER_COOPERATIVE_FINALIZE,
                MAX_FINAL_CHUNKS,
                REUSE_FINAL_WEIGHTS,
                HEAD_SHARDED_DSM,
                PAIRED_HEAD_FINALIZE,
                TWO_CHUNK_FINALIZE,
                QUAD_HEAD_TWO_CHUNK_FINALIZE,
                FAST_FINALIZER_HANDOFF,
                RANK0_ONLY_FINALIZER,
                SKIP_TRAILING_FINALIZER_BARRIER,
                WINNER_LOCAL_REUSE_FINALIZER,
                SPLIT_D_DSM_MERGE,
                False,
                True,
                False,
                False,
                TMA_STAGES,
                C4_FINALIZER_TILE,
                epoch == 1 and CTA_ROLE_RANK0_TILES < 0,
                epoch == 0 and CTA_ROLE_RANK0_TILES < 0,
                CTA_ROLE_RANK0_TILES,
                CTA_ROLE_RANK0_TILES >= 0 and epoch == 1,
                K_PER_TOKEN_V_PER_HEAD,
                KS_STRIDE_BLOCK,
                KS_STRIDE_TOKEN,
                KS_STRIDE_HEAD,
                KS_STRIDE_D,
                PREFETCH_K_SCALE,
                TMA_K_SCALE,
                FRAGMENT_K_SCALE_LOAD,
                PRECOMBINE_Q_SCALE,
                DEFER_V_SCALE_FINAL,
            )
            completed_copies = tl.where(
                task_mode != _compute_mtp4___DUMMY_MODE_JIT, (seq_len + BLOCK_N - 1) // BLOCK_N, 0
            )
            copy_iter += completed_copies


_finalize_mtp4___NUM_SEQ_Q = tl.constexpr(4)

_finalize_mtp4___HEADS_PER_PROGRAM = tl.constexpr(4)


@triton.jit
def _finalize_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize_kernel(
    KV_LENS,
    SPLIT_OUT,
    LSE,
    VSCALE,
    OUT,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    HEAD_PASSES: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    V_PER_HEAD: tl.constexpr = False,
    EXACT_TWO_SPECIALIZATION: tl.constexpr = False,
    PDL_WAIT: tl.constexpr = False,
):
    """CUDA-shaped four-head reduction over BF16 raw (numerator, m, l)."""
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    programs_per_sequence = _finalize_mtp4___NUM_SEQ_Q * HEAD_PASSES
    sequence_id = pid // programs_per_sequence
    sequence_program = pid - sequence_id * programs_per_sequence
    seq_m = sequence_program // HEAD_PASSES
    head_pass = sequence_program - seq_m * HEAD_PASSES
    hkv = sequence_id // B
    batch = sequence_id - hkv * B
    offs_h = tl.arange(0, _finalize_mtp4___HEADS_PER_PROGRAM)
    offs_d = tl.arange(0, D)
    h_in_group = head_pass * _finalize_mtp4___HEADS_PER_PROGRAM + offs_h
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group_count <= 1:
        return
    if EXACT_TWO_SPECIALIZATION and group_count == 2:
        scalar_base = (
            LSE + batch * LSE_STRIDE_B + hkv * LSE_STRIDE_HKV + seq_m * LSE_STRIDE_M + h_in_group * LSE_STRIDE_HG
        )
        m0 = tl.load(scalar_base, mask=valid_head, other=-float("inf"))
        m1 = tl.load(scalar_base + LSE_STRIDE_C, mask=valid_head, other=-float("inf"))
        l0 = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=valid_head, other=0.0)
        l1 = tl.load(scalar_base + (MAX_FINAL_CHUNKS + 1) * LSE_STRIDE_C, mask=valid_head, other=0.0)
        max_m = tl.maximum(m0, m1)
        safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
        w0 = tl.where(m0 == -float("inf"), 0.0, tl.exp2(m0 - safe_max))
        w1 = tl.where(m1 == -float("inf"), 0.0, tl.exp2(m1 - safe_max))
        denom = l0 * w0 + l1 * w1
        vscale = tl.load(VSCALE + hkv if V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
        scale = tl.where(denom > 0.0, vscale / denom, 0.0)
        out_base = SPLIT_OUT + batch * SO_STRIDE_B + seq_m * SO_STRIDE_M + hq[:, None] * SO_STRIDE_H + offs_d[None, :]
        acc = tl.load(out_base, mask=valid_head[:, None], other=0.0).to(tl.float32) * w0[:, None]
        partial = tl.load(out_base + SO_STRIDE_C, mask=valid_head[:, None], other=0.0).to(tl.float32)
        acc = (acc + partial * w1[:, None]) * scale[:, None]
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[:, None] * O_STRIDE_H + offs_d[None, :],
            acc,
            mask=valid_head[:, None] & (denom[:, None] > 0.0),
        )
        return
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[None, :] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + seq_m * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    scalar_mask = valid_head[:, None] & (offs_c[None, :] < group_count)
    m_values = tl.load(scalar_base, mask=scalar_mask, other=-float("inf"))
    l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    max_m = tl.max(m_values, axis=1)
    safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
    weights = tl.where(scalar_mask, tl.exp2(m_values - safe_max[:, None]), 0.0)
    denom = tl.sum(l_values * weights, axis=1)
    vscale = tl.load(VSCALE + hkv if V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    acc = tl.zeros((_finalize_mtp4___HEADS_PER_PROGRAM, D), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < group_count) & valid_head
        chunk_m = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + seq_m * LSE_STRIDE_M
            + h_in_group * LSE_STRIDE_HG,
            mask=chunk_valid,
            other=-float("inf"),
        )
        chunk_weight = tl.where(chunk_valid, tl.exp2(chunk_m - safe_max) * scale, 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + seq_m * SO_STRIDE_M
            + hq[:, None] * SO_STRIDE_H
            + offs_d[None, :],
            mask=chunk_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += partial * chunk_weight[:, None]
    tl.store(
        OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[:, None] * O_STRIDE_H + offs_d[None, :],
        acc,
        mask=valid_head[:, None] & (denom[:, None] > 0.0),
    )


@triton.jit
def _finalize_mtp4__fp8_kvpertensor_decode_mtp4_sharded_raw_finalize_kernel(
    KV_LENS,
    SPLIT_OUT,
    LSE,
    VSCALE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    PRODUCER_CLUSTER_SIZE: tl.constexpr,
    REDUCE_CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    HEAD_PASSES: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    V_PER_HEAD: tl.constexpr = False,
    PDL_WAIT: tl.constexpr = False,
):
    """Shard one raw finalization across a C4 reducer cluster.

    Every reducer rank folds a strided subset of producer groups.  Rank zero
    then merges four raw states through DSM, changing the long-sequence
    critical path from 32 serial group visits to eight visits plus a four-way
    cluster merge.  The operation remains fully in Triton/TLE.
    """
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    physical_cta = tl.program_id(0)
    reducer_rank = tle.shard_id(mesh, "cluster_x")
    logical_program = (physical_cta - reducer_rank) // REDUCE_CLUSTER_SIZE
    programs_per_sequence = _finalize_mtp4___NUM_SEQ_Q * HEAD_PASSES
    sequence_id = logical_program // programs_per_sequence
    sequence_program = logical_program - sequence_id * programs_per_sequence
    seq_m = sequence_program // HEAD_PASSES
    head_pass = sequence_program - seq_m * HEAD_PASSES
    hkv = sequence_id // B
    batch = sequence_id - hkv * B
    offs_h = tl.arange(0, _finalize_mtp4___HEADS_PER_PROGRAM)
    offs_d = tl.arange(0, D)
    h_in_group = head_pass * _finalize_mtp4___HEADS_PER_PROGRAM + offs_h
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    local_acc_smem = tle.gpu.alloc(
        [_finalize_mtp4___HEADS_PER_PROGRAM, D],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_m_smem = tle.gpu.alloc(
        [_finalize_mtp4___HEADS_PER_PROGRAM],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_l_smem = tle.gpu.alloc(
        [_finalize_mtp4___HEADS_PER_PROGRAM],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + PRODUCER_CLUSTER_SIZE - 1) // PRODUCER_CLUSTER_SIZE
    local_m = tl.full((_finalize_mtp4___HEADS_PER_PROGRAM,), -float("inf"), tl.float32)
    local_l = tl.zeros((_finalize_mtp4___HEADS_PER_PROGRAM,), tl.float32)
    local_acc = tl.zeros((_finalize_mtp4___HEADS_PER_PROGRAM, D), tl.float32)
    group = reducer_rank
    while group < group_count:
        group_m = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + group * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + seq_m * LSE_STRIDE_M
            + h_in_group * LSE_STRIDE_HG,
            mask=valid_head,
            other=-float("inf"),
        )
        group_l = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + (group + MAX_FINAL_CHUNKS) * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + seq_m * LSE_STRIDE_M
            + h_in_group * LSE_STRIDE_HG,
            mask=valid_head,
            other=0.0,
        )
        group_acc = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + group * SO_STRIDE_C
            + seq_m * SO_STRIDE_M
            + hq[:, None] * SO_STRIDE_H
            + offs_d[None, :],
            mask=valid_head[:, None],
            other=0.0,
        ).to(tl.float32)
        new_m = tl.maximum(local_m, group_m)
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.where(local_m == -float("inf"), 0.0, tl.exp2(local_m - safe_m))
        beta = tl.where(group_m == -float("inf"), 0.0, tl.exp2(group_m - safe_m))
        local_acc = local_acc * alpha[:, None] + group_acc * beta[:, None]
        local_l = local_l * alpha + group_l * beta
        local_m = tl.where((local_m != -float("inf")) | (group_m != -float("inf")), new_m, -float("inf"))
        group += REDUCE_CLUSTER_SIZE
    acc_rows = tl.broadcast_to(offs_h[:, None], (_finalize_mtp4___HEADS_PER_PROGRAM, D))
    acc_cols = tl.broadcast_to(offs_d[None, :], (_finalize_mtp4___HEADS_PER_PROGRAM, D))
    tl.store(tle.gpu.local_ptr(local_acc_smem, (acc_rows, acc_cols)), local_acc)
    tl.store(tle.gpu.local_ptr(local_m_smem, (offs_h,)), local_m)
    tl.store(tle.gpu.local_ptr(local_l_smem, (offs_h,)), local_l)
    tle.distributed_barrier(mesh)
    if reducer_rank == 0 and group_count > 1:
        merged_m = tl.full((_finalize_mtp4___HEADS_PER_PROGRAM,), -float("inf"), tl.float32)
        merged_l = tl.zeros((_finalize_mtp4___HEADS_PER_PROGRAM,), tl.float32)
        merged_acc = tl.zeros((_finalize_mtp4___HEADS_PER_PROGRAM, D), tl.float32)
        for peer_rank in tl.static_range(0, REDUCE_CLUSTER_SIZE):
            peer_acc_smem = tle.remote(local_acc_smem, peer_rank, scope=mesh)
            peer_m_smem = tle.remote(local_m_smem, peer_rank, scope=mesh)
            peer_l_smem = tle.remote(local_l_smem, peer_rank, scope=mesh)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
            peer_m = tl.load(tle.gpu.local_ptr(peer_m_smem, (offs_h,)))
            peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_h,)))
            new_m = tl.maximum(merged_m, peer_m)
            safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
            alpha = tl.where(merged_m == -float("inf"), 0.0, tl.exp2(merged_m - safe_m))
            beta = tl.where(peer_m == -float("inf"), 0.0, tl.exp2(peer_m - safe_m))
            merged_acc = merged_acc * alpha[:, None] + peer_acc * beta[:, None]
            merged_l = merged_l * alpha + peer_l * beta
            merged_m = tl.where((merged_m != -float("inf")) | (peer_m != -float("inf")), new_m, -float("inf"))
        vscale = tl.load(VSCALE + hkv if V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
        scale = tl.where(merged_l > 0.0, vscale / merged_l, 0.0)
        result = merged_acc * scale[:, None]
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[:, None] * O_STRIDE_H + offs_d[None, :],
            result,
            mask=valid_head[:, None] & (merged_l[:, None] > 0.0),
        )
    tle.distributed_barrier(mesh)


_runtime_mtp1__BLOCK_SIZE = 64

_runtime_mtp1__TILE_N = 64

_runtime_mtp1__NUM_SEQ_Q = 1

_runtime_mtp1__HEAD_DIM = 128


@dataclass(frozen=True)
class _runtime_mtp1__DecodeConfig:
    cluster_size: int
    chunk_tokens: int

    def __post_init__(self) -> None:
        if self.cluster_size not in (2, 4, 8):
            raise ValueError("runtime policy supports cluster_size 2, 4, or 8")
        if self.chunk_tokens not in (128, 256, 512, 1024):
            raise ValueError("runtime supports chunk_tokens 128, 256, 512, or 1024")


@dataclass
class _runtime_mtp1__DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def num_batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def max_seq_kv(self) -> int:
        return int(self.kv_lens.max().item())


@dataclass
class _runtime_mtp1__DecodeWorkspace:
    config: _runtime_mtp1__DecodeConfig
    schedule: _scheduler_mtp1__DecodeTaskSchedule
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    all_chunks_aligned: bool = False

    @property
    def stats(self) -> dict:
        return dict(self.schedule.stats or {})


_runtime_mtp1__CLUSTER_MESHES = (
    {
        2: tle.device_mesh({"block_cluster": [("cluster_x", 2)]}),
        4: tle.device_mesh({"block_cluster": [("cluster_x", 4)]}),
        8: tle.device_mesh({"block_cluster": [("cluster_x", 8)]}),
    }
    if USE_TLE
    else {2: None, 4: None, 8: None}
)


def _runtime_mtp1___validate_inputs(inputs: _runtime_mtp1__DecodeInputs) -> tuple[int, int, int]:
    if not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA tensor for GPU task scheduling")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _runtime_mtp1__HEAD_DIM:
        raise ValueError("q must have flattened shape [batch * 1, num_head_q, 128]")
    if inputs.q.shape[0] != inputs.num_batch * _runtime_mtp1__NUM_SEQ_Q:
        raise ValueError("MTP=1 requires q.shape[0] == batch * 1")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.ndim != 4 or cache.shape[1] != _runtime_mtp1__BLOCK_SIZE or cache.shape[3] != _runtime_mtp1__HEAD_DIM:
            raise ValueError(f"{name} must be logical [block, 64, head, 128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if num_head_q % num_head_kv:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    heads_per_group = num_head_q // num_head_kv
    if heads_per_group > 8:
        raise ValueError("heads_per_group must be <= 8")
    return (num_head_q, num_head_kv, heads_per_group)


def _runtime_mtp1__make_paged_kv_descriptors(
    inputs: _runtime_mtp1__DecodeInputs,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Project NHD/HND paged descriptors to logical [block, head, token, dim] tiles."""
    _runtime_mtp1___validate_inputs(inputs)
    block_shape = [1, 1, _runtime_mtp1__TILE_N, _runtime_mtp1__HEAD_DIM]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


def _runtime_mtp1__prepare_decode_workspace(
    inputs: _runtime_mtp1__DecodeInputs, config: _runtime_mtp1__DecodeConfig | None = None
) -> _runtime_mtp1__DecodeWorkspace:
    """Allocate buffers and build the complete cluster task map on the GPU."""
    num_head_q, num_head_kv, heads_per_group = _runtime_mtp1___validate_inputs(inputs)
    if config is None:
        raise ValueError("MTP=1 policy is not finalized; pass an explicit DecodeConfig")
    schedule = _scheduler_mtp1__allocate_cluster_task_map(
        inputs.kv_lens,
        num_head_kv=num_head_kv,
        max_seq_kv=inputs.max_seq_kv,
        cluster_size=config.cluster_size,
        chunk_tokens=config.chunk_tokens,
    )
    if schedule.stats is None:
        raise RuntimeError("GPU task scheduler did not publish workspace metadata")
    all_chunks_aligned = bool(torch.all(inputs.kv_lens % config.chunk_tokens == 0).item())
    q_4d = inputs.q.reshape(inputs.num_batch, _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime_mtp1__HEAD_DIM)
    q_scale_3d = inputs.q_scale.reshape(inputs.num_batch, _runtime_mtp1__NUM_SEQ_Q, num_head_q)
    pad_heads_per_group = (heads_per_group + 7) // 8 * 8
    device = inputs.q.device
    split_out = torch.empty(
        (inputs.num_batch, schedule.partial_slots, _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime_mtp1__HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    lse = torch.empty(
        (inputs.num_batch, schedule.partial_slots, num_head_kv, _runtime_mtp1__NUM_SEQ_Q, pad_heads_per_group),
        dtype=torch.float32,
        device=device,
    )
    return _runtime_mtp1__DecodeWorkspace(
        config=config,
        schedule=schedule,
        q_4d=q_4d,
        q_scale_3d=q_scale_3d,
        split_out=split_out,
        lse=lse,
        completion=torch.zeros((num_head_kv * inputs.num_batch,), dtype=torch.int32, device=device),
        last_flags=torch.zeros((schedule.physical_ctas,), dtype=torch.int32, device=device),
        out=torch.empty(
            (inputs.num_batch, _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime_mtp1__HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
        heads_per_group=heads_per_group,
        all_chunks_aligned=all_chunks_aligned,
    )


def _runtime_mtp1__refresh_decode_schedule(
    inputs: _runtime_mtp1__DecodeInputs,
    workspace: _runtime_mtp1__DecodeWorkspace,
    mode: Literal["full", "tail"] = "full",
) -> None:
    """Refresh an existing GPU schedule when its allocated topology is stable."""
    num_head_kv = int(inputs.k_cache.shape[2])
    if mode == "full":
        _scheduler_mtp1__launch_cluster_task_map_assign(
            inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv, refresh_host_metadata=False
        )
    elif mode == "tail":
        _scheduler_mtp1__launch_cluster_task_tail_refresh(inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv)
    else:
        raise ValueError("mode must be 'full' or 'tail'")
    workspace.all_chunks_aligned = bool(torch.all(inputs.kv_lens % workspace.config.chunk_tokens == 0).item())


def _runtime_mtp1__fp8_kvpertensor_decode_mtp1_pure_tle(
    inputs: _runtime_mtp1__DecodeInputs,
    workspace: _runtime_mtp1__DecodeWorkspace | None = None,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
    paired_head_finalize: bool | None = None,
    c2_paired_head_finalize: bool = False,
    bf16_dsm: bool = False,
    deferred_norm: bool = False,
    dsm_election_handoff: bool = False,
    deterministic_tail_election: bool = False,
    full_view_v_rs: bool = False,
    quad_head_two_chunk_finalize: bool = False,
    rank0_only_finalizer: bool = False,
    skip_trailing_finalizer_barrier: bool = False,
    reduction_only: bool = False,
    aligned_full_chunk: bool = False,
    direct_fast_path: bool = False,
    tma_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool = False,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """Run the .cu-free MTP=1 weight-reuse policy.

    Cluster-4 selects paired-head finalization by default. Cluster-2/8 use
    the sequential head mapping, matching the winning MTP=2 policy.
    """
    if workspace is None:
        raise ValueError("MTP=1 requires an explicitly configured workspace")
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 or 1")
    if quant_type == 0 and (inputs.k_scale.ndim != 4 or inputs.v_scale.numel() < inputs.k_cache.shape[2]):
        raise ValueError("quant_type=0 requires packed rank-4 K scales and one V scale per KV head")
    if tma_k_scale and quant_type != 0:
        raise ValueError("TLE K-scale TMA requires quant_type=0")
    if page_metadata_k_scale and quant_type != 0:
        raise ValueError("page-metadata K-scale pipeline requires quant_type=0")
    if page_metadata_k_scale and tma_k_scale:
        raise ValueError("page-metadata K-scale and TMA are exclusive")
    ks = inputs.k_scale.stride() if quant_type == 0 else (0, 0, 0, 0)
    if paired_head_finalize is None:
        paired_head_finalize = workspace.config.cluster_size == 4
    if paired_head_finalize and workspace.config.cluster_size != 4:
        raise ValueError("paired_head_finalize requires cluster_size=4")
    if paired_head_finalize and workspace.heads_per_group != 8:
        raise ValueError("paired_head_finalize requires heads_per_group=8")
    if c2_paired_head_finalize and workspace.config.cluster_size != 2:
        raise ValueError("c2_paired_head_finalize requires cluster_size=2")
    if c2_paired_head_finalize and workspace.heads_per_group != 8:
        raise ValueError("c2_paired_head_finalize requires heads_per_group=8")
    if c2_paired_head_finalize and paired_head_finalize:
        raise ValueError("c2 and c4 paired finalizers are mutually exclusive")
    if deferred_norm and (not bf16_dsm):
        raise ValueError("deferred_norm requires bf16_dsm")
    effective_chunks_max = int(workspace.stats.get("effective_chunks_max", 1))
    c2_exact_two = workspace.config.cluster_size == 2 and effective_chunks_max == 2 and (workspace.heads_per_group == 8)
    if deferred_norm and workspace.config.cluster_size != 8 and (not c2_exact_two):
        raise ValueError("MTP1 deferred_norm requires C8 or exact-two C2")
    if quad_head_two_chunk_finalize and (not c2_exact_two):
        raise ValueError("quad-head finalization requires exact-two C2/GQA8")
    if rank0_only_finalizer and (not quad_head_two_chunk_finalize):
        raise ValueError("rank0-only finalization requires quad-head exact-two")
    if rank0_only_finalizer and (dsm_election_handoff or deterministic_tail_election):
        raise ValueError("rank0-only finalization owns election locally")
    if reduction_only:
        incompatible = {
            name: int(workspace.stats.get(name, 0))
            for name in ("direct_tasks", "dummy_tasks", "subgroup2_tasks")
            if int(workspace.stats.get(name, 0))
        }
        if incompatible:
            raise ValueError(f"reduction_only requires aligned split-only work: {incompatible}")
    if aligned_full_chunk:
        if not reduction_only:
            raise ValueError("aligned_full_chunk requires reduction_only")
        if not workspace.all_chunks_aligned:
            raise ValueError("aligned_full_chunk requires every KV length to be chunk aligned")
    if deterministic_tail_election and dsm_election_handoff:
        raise ValueError("deterministic tail and DSM election handoff are alternatives")
    if refresh_schedule is not None:
        _runtime_mtp1__refresh_decode_schedule(inputs, workspace, refresh_schedule)
    k_desc, v_desc = _runtime_mtp1__make_paged_kv_descriptors(inputs)
    if tma_k_scale:
        k_scale_f32 = inputs.k_scale.view(torch.float32)
        ks_desc = TensorDescriptor.from_tensor(k_scale_f32, block_shape=[1, 2, 1, 32])
    else:
        ks_desc = k_desc
    ws = workspace
    num_head_q = int(inputs.q.shape[1])
    direct_tasks = int(ws.stats.get("direct_tasks", 0))
    compute_tasks = int(ws.stats.get("compute_tasks", 0))
    use_direct_fast = direct_fast_path and direct_tasks > 0 and (direct_tasks == compute_tasks)
    if use_direct_fast:
        _compute_mtp4__fp8_kvpertensor_decode_mtp4_direct_kernel[direct_tasks,](
            ws.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            ws.schedule.task_map,
            ws.q_scale_3d,
            inputs.k_scale,
            inputs.v_scale,
            ws.out,
            B=inputs.num_batch,
            H_Q=num_head_q,
            HEADS_PER_GROUP=ws.heads_per_group,
            NUM_SEQ_Q=_runtime_mtp1__NUM_SEQ_Q,
            ROWS_Q=_runtime_mtp1__NUM_SEQ_Q * ws.heads_per_group,
            D=_runtime_mtp1__HEAD_DIM,
            DV=_runtime_mtp1__HEAD_DIM,
            BLOCK_SIZE=_runtime_mtp1__BLOCK_SIZE,
            MAX_BLOCKS=inputs.block_ids.shape[1],
            BLOCK_N=_runtime_mtp1__TILE_N,
            DIRECT_CLUSTER_BASE=0,
            DIRECT_CLUSTER_SIZE=ws.config.cluster_size,
            Q_STRIDE_B=ws.q_4d.stride(0),
            Q_STRIDE_M=ws.q_4d.stride(1),
            Q_STRIDE_H=ws.q_4d.stride(2),
            QS_STRIDE_B=ws.q_scale_3d.stride(0),
            QS_STRIDE_M=ws.q_scale_3d.stride(1),
            QS_STRIDE_H=ws.q_scale_3d.stride(2),
            O_STRIDE_B=ws.out.stride(0),
            O_STRIDE_M=ws.out.stride(1),
            O_STRIDE_H=ws.out.stride(2),
            TMA_STAGES=2,
            K_PER_TOKEN_V_PER_HEAD=quant_type == 0,
            KS_STRIDE_BLOCK=ks[0],
            KS_STRIDE_TOKEN=ks[1],
            KS_STRIDE_HEAD=ks[2],
            KS_STRIDE_D=ks[3],
            num_ctas=1,
            num_warps=4,
            num_stages=3,
            launch_pdl=False,
        )
        if flatten_output:
            return ws.out.reshape(inputs.num_batch * _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime_mtp1__HEAD_DIM)
        return ws.out
    _compute_mtp1__fp8_kvpertensor_decode_mtp1_final_kernel[ws.schedule.num_clusters,](
        ws.q_4d,
        k_desc,
        ks_desc,
        v_desc,
        inputs.v_cache,
        inputs.block_ids,
        ws.schedule.task_map,
        inputs.kv_lens,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=_runtime_mtp1__CLUSTER_MESHES[ws.config.cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=_runtime_mtp1__HEAD_DIM,
        DV=_runtime_mtp1__HEAD_DIM,
        BLOCK_SIZE=_runtime_mtp1__BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=_runtime_mtp1__TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_M=ws.q_4d.stride(1),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_M=ws.q_scale_3d.stride(1),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        V_STRIDE_BLOCK=inputs.v_cache.stride(0),
        V_STRIDE_TOKEN=inputs.v_cache.stride(1),
        V_STRIDE_HEAD=inputs.v_cache.stride(2),
        V_STRIDE_D=inputs.v_cache.stride(3),
        USE_LOG2=True,
        K_PER_TOKEN_V_PER_HEAD=quant_type == 0,
        TMA_K_SCALE=tma_k_scale,
        PAGE_METADATA_K_SCALE=page_metadata_k_scale,
        PRECOMBINE_Q_SCALE=precombine_q_scale,
        KS_STRIDE_BLOCK=ks[0],
        KS_STRIDE_TOKEN=ks[1],
        KS_STRIDE_HEAD=ks[2],
        KS_STRIDE_D=ks[3],
        RAW_PAGED_NHD_OR_HND=True,
        FULL_MATRIX_RS=False,
        TRANSPOSED_MEMDESC_RS=False,
        DIRECT_V_SHARED_SHARED=False,
        LDSM_REGISTER_SHARED=not full_view_v_rs,
        TLE_SHARED_SHARED=False,
        FULL_VIEW_V_RS=full_view_v_rs,
        TMA_DN_RS=False,
        DIRECT_GLOBAL_V_RS=False,
        FRAGMENT_PIPELINED_RS=False,
        K32_PIPELINED_RS=False,
        INPLACE_PV_ACC=False,
        MERGE_CLUSTER_SIZE=ws.config.cluster_size,
        EXECUTION_STAGE=_compute_mtp1__EXECUTION_FULL,
        CLUSTER_COOPERATIVE_FINALIZE=True,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.schedule.partial_slots),
        REUSE_FINAL_WEIGHTS=True,
        HEAD_SHARDED_DSM=False,
        PAIRED_HEAD_FINALIZE=paired_head_finalize,
        C2_PAIRED_HEAD_FINALIZE=c2_paired_head_finalize,
        BF16_DSM=bf16_dsm,
        DEFERRED_NORM=deferred_norm,
        DSM_ELECTION_HANDOFF=dsm_election_handoff,
        DETERMINISTIC_TAIL_ELECTION=deterministic_tail_election,
        QUAD_HEAD_TWO_CHUNK_FINALIZE=quad_head_two_chunk_finalize,
        RANK0_ONLY_FINALIZER=rank0_only_finalizer,
        SKIP_TRAILING_FINALIZER_BARRIER=skip_trailing_finalizer_barrier,
        REDUCTION_ONLY=reduction_only,
        ALIGNED_FULL_CHUNK_TOKENS=ws.config.chunk_tokens if aligned_full_chunk else 0,
        STATIC_SCHED=False,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    if flatten_output:
        return ws.out.reshape(inputs.num_batch * _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime_mtp1__HEAD_DIM)
    return ws.out


def _runtime_mtp1__fp8_kvpertensor_decode_mtp1_final(
    inputs: _runtime_mtp1__DecodeInputs,
    workspace: _runtime_mtp1__DecodeWorkspace | None = None,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
    paired_head_finalize: bool | None = None,
    c2_paired_head_finalize: bool = False,
    bf16_dsm: bool | None = None,
    deferred_norm: bool = False,
    dsm_election_handoff: bool | None = None,
    deterministic_tail_election: bool | None = None,
    reduction_only: bool | None = None,
    aligned_full_chunk: bool = False,
    full_view_v_rs: bool = False,
    skip_trailing_finalizer_barrier: bool | None = None,
    tma_k_scale: bool | None = None,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool | None = None,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """Run the fixed MTP=1 final path."""
    if workspace is None:
        raise ValueError("MTP=1 requires an explicitly configured workspace")
    if tma_k_scale is None:
        tma_k_scale = quant_type == 0 and (not page_metadata_k_scale)
    if precombine_q_scale is None:
        precombine_q_scale = quant_type == 1 and (workspace.config.cluster_size, workspace.config.chunk_tokens) in (
            (4, 512),
            (8, 512),
        )
    aligned_split_only = not any(
        (int(workspace.stats.get(name, 0)) for name in ("direct_tasks", "dummy_tasks", "subgroup2_tasks"))
    )
    auto_aligned_full_chunk = (
        quant_type in (0, 1)
        and workspace.all_chunks_aligned
        and aligned_split_only
        and (
            (workspace.config.cluster_size, workspace.config.chunk_tokens)
            in ((2, 256), (2, 1024), (4, 256), (4, 512), (4, 1024), (8, 512))
        )
    )
    if reduction_only is None:
        aligned_full_chunk = aligned_full_chunk or auto_aligned_full_chunk
        reduction_only = aligned_full_chunk or (
            aligned_split_only
            and (workspace.config.cluster_size, workspace.config.chunk_tokens) in ((2, 1024), (8, 512), (4, 512))
        )
    if skip_trailing_finalizer_barrier is None:
        skip_trailing_finalizer_barrier = workspace.config.cluster_size == 4 and workspace.config.chunk_tokens in (
            128,
            512,
            1024,
        )
    use_c2_raw = (
        workspace.config.cluster_size == 2
        and workspace.config.chunk_tokens == 1024
        and (int(workspace.stats.get("effective_chunks_max", 1)) == 2)
        and (workspace.heads_per_group == 8)
        and (refresh_schedule is None)
        and (paired_head_finalize is None)
        and (not c2_paired_head_finalize)
        and (not bf16_dsm)
        and (not deferred_norm)
        and (dsm_election_handoff is None)
        and (deterministic_tail_election is None)
        and (not skip_trailing_finalizer_barrier)
        and flatten_output
    )
    if use_c2_raw:
        return _runtime_mtp1__fp8_kvpertensor_decode_mtp1_c2_raw_specialized(
            inputs,
            workspace,
            reduction_only=reduction_only,
            aligned_full_chunk=aligned_full_chunk,
            tma_k_scale=tma_k_scale,
            page_metadata_k_scale=page_metadata_k_scale,
            precombine_q_scale=precombine_q_scale,
            quant_type=quant_type,
        )
    if bf16_dsm is None:
        bf16_dsm = workspace.config.cluster_size == 8
    if dsm_election_handoff is None and deterministic_tail_election is None:
        use_tail = workspace.config.cluster_size == 8 or (
            workspace.config.cluster_size == 4 and workspace.config.chunk_tokens in (128, 1024)
        )
        deterministic_tail_election = use_tail
        dsm_election_handoff = not use_tail
    elif dsm_election_handoff is None:
        dsm_election_handoff = False
    elif deterministic_tail_election is None:
        deterministic_tail_election = False
    return _runtime_mtp1__fp8_kvpertensor_decode_mtp1_pure_tle(
        inputs,
        workspace,
        refresh_schedule=refresh_schedule,
        paired_head_finalize=paired_head_finalize,
        c2_paired_head_finalize=c2_paired_head_finalize,
        bf16_dsm=bf16_dsm,
        deferred_norm=deferred_norm,
        dsm_election_handoff=dsm_election_handoff,
        deterministic_tail_election=deterministic_tail_election,
        reduction_only=reduction_only,
        aligned_full_chunk=aligned_full_chunk,
        full_view_v_rs=full_view_v_rs,
        skip_trailing_finalizer_barrier=skip_trailing_finalizer_barrier,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        quant_type=quant_type,
        flatten_output=flatten_output,
    )


def _runtime_mtp1__fp8_kvpertensor_decode_mtp1_c2_raw_specialized(
    inputs: _runtime_mtp1__DecodeInputs,
    workspace: _runtime_mtp1__DecodeWorkspace,
    *,
    reduction_only: bool = False,
    aligned_full_chunk: bool = False,
    tma_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool = False,
    quant_type: int = 1,
) -> torch.Tensor:
    """Exact-two C2 BF16/raw path used by the fixed MTP1 policy."""
    return _runtime_mtp1__fp8_kvpertensor_decode_mtp1_pure_tle(
        inputs,
        workspace,
        paired_head_finalize=False,
        c2_paired_head_finalize=False,
        bf16_dsm=True,
        deferred_norm=True,
        full_view_v_rs=False,
        quad_head_two_chunk_finalize=True,
        rank0_only_finalizer=True,
        skip_trailing_finalizer_barrier=True,
        reduction_only=reduction_only,
        aligned_full_chunk=aligned_full_chunk,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        quant_type=quant_type,
    )


_runtime_mtp2__BLOCK_SIZE = 64

_runtime_mtp2__TILE_N = 64

_runtime_mtp2__NUM_SEQ_Q = 2

_runtime_mtp2__HEAD_DIM = 128


@dataclass(frozen=True)
class _runtime_mtp2__DecodeConfig:
    cluster_size: int
    chunk_tokens: int

    def __post_init__(self) -> None:
        if self.cluster_size not in (2, 4, 8):
            raise ValueError("runtime policy supports cluster_size 2, 4, or 8")
        if self.chunk_tokens not in (128, 256, 512, 1024):
            raise ValueError("runtime supports chunk_tokens 128, 256, 512, or 1024")


@dataclass
class _runtime_mtp2__DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def num_batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def max_seq_kv(self) -> int:
        return int(self.kv_lens.max().item())


@dataclass
class _runtime_mtp2__DecodeWorkspace:
    config: _runtime_mtp2__DecodeConfig
    schedule: _scheduler_mtp24__DecodeTaskSchedule
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    all_chunks_aligned: bool = False
    split_out_layout: Literal["MHD", "HMD"] = "MHD"
    lse_layout: Literal["MH", "HM"] = "MH"

    @property
    def stats(self) -> dict:
        return dict(self.schedule.stats or {})


_runtime_mtp2__CLUSTER_MESHES = (
    {
        2: tle.device_mesh({"block_cluster": [("cluster_x", 2)]}),
        4: tle.device_mesh({"block_cluster": [("cluster_x", 4)]}),
        8: tle.device_mesh({"block_cluster": [("cluster_x", 8)]}),
    }
    if USE_TLE
    else {2: None, 4: None, 8: None}
)


def _runtime_mtp2___validate_inputs(inputs: _runtime_mtp2__DecodeInputs) -> tuple[int, int, int]:
    if not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA tensor for GPU task scheduling")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _runtime_mtp2__HEAD_DIM:
        raise ValueError("q must have flattened shape [batch * 2, num_head_q, 128]")
    if inputs.q.shape[0] != inputs.num_batch * _runtime_mtp2__NUM_SEQ_Q:
        raise ValueError("MTP=2 requires q.shape[0] == batch * 2")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.ndim != 4 or cache.shape[1] != _runtime_mtp2__BLOCK_SIZE or cache.shape[3] != _runtime_mtp2__HEAD_DIM:
            raise ValueError(f"{name} must be logical [block, 64, head, 128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if num_head_q % num_head_kv:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    heads_per_group = num_head_q // num_head_kv
    if heads_per_group > 8:
        raise ValueError("heads_per_group must be <= 8")
    return (num_head_q, num_head_kv, heads_per_group)


def _runtime_mtp2__make_paged_kv_descriptors(
    inputs: _runtime_mtp2__DecodeInputs,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Project NHD/HND paged descriptors to logical [block, head, token, dim] tiles."""
    _runtime_mtp2___validate_inputs(inputs)
    block_shape = [1, 1, _runtime_mtp2__TILE_N, _runtime_mtp2__HEAD_DIM]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


def _runtime_mtp2__prepare_decode_workspace(
    inputs: _runtime_mtp2__DecodeInputs,
    config: _runtime_mtp2__DecodeConfig | None = None,
    *,
    raw_global_state: bool = False,
    raw_scalar_chunk_minor: bool = False,
) -> _runtime_mtp2__DecodeWorkspace:
    """Allocate buffers and build the complete cluster task map on the GPU."""
    num_head_q, num_head_kv, heads_per_group = _runtime_mtp2___validate_inputs(inputs)
    if config is None:
        raise ValueError("MTP=2 policy is not finalized; pass an explicit DecodeConfig")
    schedule = _scheduler_mtp24__allocate_cluster_task_map(
        inputs.kv_lens,
        num_head_kv=num_head_kv,
        max_seq_kv=inputs.max_seq_kv,
        cluster_size=config.cluster_size,
        chunk_tokens=config.chunk_tokens,
    )
    if schedule.stats is None:
        raise RuntimeError("GPU task scheduler did not publish workspace metadata")
    all_chunks_aligned = bool(torch.all(inputs.kv_lens % config.chunk_tokens == 0).item())
    q_4d = inputs.q.reshape(inputs.num_batch, _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime_mtp2__HEAD_DIM)
    q_scale_3d = inputs.q_scale.reshape(inputs.num_batch, _runtime_mtp2__NUM_SEQ_Q, num_head_q)
    pad_heads_per_group = (heads_per_group + 7) // 8 * 8
    device = inputs.q.device
    split_out = torch.empty(
        (inputs.num_batch, schedule.partial_slots, _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime_mtp2__HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    max_final_chunks = triton.next_power_of_2(schedule.partial_slots)
    scalar_slots = 2 * max_final_chunks if raw_global_state else schedule.partial_slots
    if raw_scalar_chunk_minor:
        lse = torch.empty(
            (inputs.num_batch, num_head_kv, _runtime_mtp2__NUM_SEQ_Q, pad_heads_per_group, scalar_slots),
            dtype=torch.float32,
            device=device,
        ).permute(0, 4, 1, 2, 3)
    else:
        lse = torch.empty(
            (inputs.num_batch, scalar_slots, num_head_kv, _runtime_mtp2__NUM_SEQ_Q, pad_heads_per_group),
            dtype=torch.float32,
            device=device,
        )
    return _runtime_mtp2__DecodeWorkspace(
        config=config,
        schedule=schedule,
        q_4d=q_4d,
        q_scale_3d=q_scale_3d,
        split_out=split_out,
        lse=lse,
        completion=torch.zeros((num_head_kv * inputs.num_batch,), dtype=torch.int32, device=device),
        last_flags=torch.zeros((schedule.physical_ctas,), dtype=torch.int32, device=device),
        out=torch.empty(
            (inputs.num_batch, _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime_mtp2__HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
        heads_per_group=heads_per_group,
        all_chunks_aligned=all_chunks_aligned,
    )


def _runtime_mtp2__refresh_decode_schedule(
    inputs: _runtime_mtp2__DecodeInputs,
    workspace: _runtime_mtp2__DecodeWorkspace,
    mode: Literal["full", "tail"] = "full",
) -> None:
    """Refresh an existing GPU schedule when its allocated topology is stable."""
    num_head_kv = int(inputs.k_cache.shape[2])
    if mode == "full":
        _scheduler_mtp24__launch_cluster_task_map_assign(
            inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv, refresh_host_metadata=False
        )
    elif mode == "tail":
        _scheduler_mtp24__launch_cluster_task_tail_refresh(inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv)
    else:
        raise ValueError("mode must be 'full' or 'tail'")
    workspace.all_chunks_aligned = bool(torch.all(inputs.kv_lens % workspace.config.chunk_tokens == 0).item())


def _runtime_mtp2__fp8_kvpertensor_decode_mtp2_final(
    inputs: _runtime_mtp2__DecodeInputs,
    workspace: _runtime_mtp2__DecodeWorkspace | None = None,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
    execution_stage: Literal["full", "cluster", "local"] = "full",
    cluster_cooperative_finalize: bool = False,
    reuse_final_weights: bool = False,
    head_sharded_dsm: bool = False,
    paired_head_finalize: bool = False,
    c8_paired_head_finalize: bool = False,
    row_serial_finalize: bool = False,
    two_chunk_finalize: bool = False,
    bf16_dsm: bool = False,
    deferred_norm: bool = False,
    pdl_notify: bool = False,
    dsm_election_handoff: bool = False,
    deterministic_tail_election: bool = False,
    tail_only_election_barrier: bool = False,
    reduction_only: bool = False,
    aligned_full_chunk: bool = False,
    full_view_dsm: bool = False,
    full_view_v_rs: bool = False,
    quad_head_two_chunk_finalize: bool = False,
    rank0_only_finalizer: bool = False,
    skip_trailing_finalizer_barrier: bool = False,
    global_deferred_norm: bool = False,
    direct_fast_path: bool = False,
    tma_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool = False,
    quant_type: int = 1,
) -> torch.Tensor:
    """Run one explicit MTP=2 cluster/token configuration."""
    if workspace is None:
        raise ValueError("MTP=2 requires an explicitly configured workspace")
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 or 1")
    if quant_type == 0 and (inputs.k_scale.ndim != 4 or inputs.v_scale.numel() < inputs.k_cache.shape[2]):
        raise ValueError("quant_type=0 requires packed rank-4 K scales and one V scale per KV head")
    if tma_k_scale and quant_type != 0:
        raise ValueError("TLE K-scale TMA requires quant_type=0")
    if page_metadata_k_scale and quant_type != 0:
        raise ValueError("page-metadata K-scale pipeline requires quant_type=0")
    if page_metadata_k_scale and tma_k_scale:
        raise ValueError("page-metadata K-scale and TMA are exclusive")
    ks = inputs.k_scale.stride() if quant_type == 0 else (0, 0, 0, 0)
    stage_map = {
        "full": _compute_mtp2__EXECUTION_FULL,
        "cluster": _compute_mtp2__EXECUTION_CLUSTER_PARTIAL,
        "local": _compute_mtp2__EXECUTION_LOCAL_PARTIAL,
    }
    if execution_stage not in stage_map:
        raise ValueError("execution_stage must be 'full', 'cluster', or 'local'")
    if reuse_final_weights and (not cluster_cooperative_finalize):
        raise ValueError("reuse_final_weights requires cluster cooperative finalization")
    if head_sharded_dsm and (not cluster_cooperative_finalize):
        raise ValueError("head_sharded_dsm requires cluster cooperative finalization")
    if head_sharded_dsm and workspace.heads_per_group != 8:
        raise ValueError("head_sharded_dsm currently requires heads_per_group=8")
    if paired_head_finalize and workspace.config.cluster_size != 4:
        raise ValueError("paired_head_finalize requires cluster_size=4")
    if paired_head_finalize and workspace.heads_per_group != 8:
        raise ValueError("paired_head_finalize requires heads_per_group=8")
    if paired_head_finalize and (not reuse_final_weights):
        raise ValueError("paired_head_finalize requires reuse_final_weights")
    if c8_paired_head_finalize and workspace.config.cluster_size != 8:
        raise ValueError("c8_paired_head_finalize requires cluster_size=8")
    if c8_paired_head_finalize and workspace.heads_per_group != 8:
        raise ValueError("c8_paired_head_finalize requires heads_per_group=8")
    if c8_paired_head_finalize and (not reuse_final_weights):
        raise ValueError("c8_paired_head_finalize requires reuse_final_weights")
    if row_serial_finalize and (not reuse_final_weights):
        raise ValueError("row_serial_finalize requires reuse_final_weights")
    if two_chunk_finalize and (not reuse_final_weights):
        raise ValueError("two_chunk_finalize requires reuse_final_weights")
    if deferred_norm and (not bf16_dsm):
        raise ValueError("deferred_norm requires bf16_dsm")
    if deferred_norm and workspace.config.cluster_size not in (2, 4):
        raise ValueError("deferred_norm currently requires cluster_size=2 or 4")
    if deferred_norm and execution_stage != "full":
        if not (global_deferred_norm and execution_stage == "cluster"):
            raise ValueError("deferred_norm requires full stage, or cluster stage with global_deferred_norm")
    if global_deferred_norm and (not deferred_norm or workspace.config.cluster_size != 4):
        raise ValueError("global_deferred_norm requires C4 deferred normalization")
    required_scalar_slots = 2 * triton.next_power_of_2(workspace.schedule.partial_slots)
    if global_deferred_norm and workspace.lse.shape[1] < required_scalar_slots:
        raise ValueError("global_deferred_norm requires prepare_decode_workspace(..., raw_global_state=True)")
    if pdl_notify and execution_stage != "cluster":
        raise ValueError("pdl_notify requires execution_stage='cluster'")
    if dsm_election_handoff and workspace.config.cluster_size not in (2, 4):
        raise ValueError("DSM election handoff currently requires C2 or C4")
    if dsm_election_handoff and (not cluster_cooperative_finalize):
        raise ValueError("DSM election handoff requires cluster cooperative finalization")
    if deterministic_tail_election and (not cluster_cooperative_finalize):
        raise ValueError("deterministic tail requires cluster cooperative finalization")
    if deterministic_tail_election and dsm_election_handoff:
        raise ValueError("deterministic tail and DSM election handoff are alternatives")
    if tail_only_election_barrier and (not deterministic_tail_election):
        raise ValueError("tail-only election barrier requires deterministic tail")
    if reduction_only:
        incompatible_tasks = {
            name: int(workspace.stats.get(name, 0))
            for name in ("direct_tasks", "dummy_tasks", "subgroup2_tasks")
            if int(workspace.stats.get(name, 0))
        }
        if incompatible_tasks:
            raise ValueError(f"reduction_only requires aligned split-only work, got {incompatible_tasks}")
        if execution_stage != "full":
            raise ValueError("reduction_only requires execution_stage='full'")
    if aligned_full_chunk:
        if not reduction_only:
            raise ValueError("aligned_full_chunk requires reduction_only")
        if not workspace.all_chunks_aligned:
            raise ValueError("aligned_full_chunk requires every KV length to be chunk aligned")
    if full_view_dsm and workspace.config.cluster_size not in (2, 4):
        raise ValueError("full_view_dsm currently requires C2 or C4")
    effective_chunks_max = int(workspace.stats.get("effective_chunks_max", 1))
    if quad_head_two_chunk_finalize and (
        workspace.config.cluster_size != 2 or effective_chunks_max != 2 or workspace.heads_per_group != 8
    ):
        raise ValueError(
            f"quad_head_two_chunk_finalize requires C2, GQA8, and exactly two groups; got c{workspace.config.cluster_size}, gqa{workspace.heads_per_group}, groups={effective_chunks_max}"
        )
    if rank0_only_finalizer and (not quad_head_two_chunk_finalize):
        raise ValueError("rank0_only_finalizer requires quad_head_two_chunk_finalize")
    if rank0_only_finalizer and (dsm_election_handoff or deterministic_tail_election):
        raise ValueError(
            "rank0_only_finalizer uses the local ticket and cannot combine with DSM or deterministic election handoff"
        )
    if refresh_schedule is not None:
        _runtime_mtp2__refresh_decode_schedule(inputs, workspace, refresh_schedule)
    k_desc, v_desc = _runtime_mtp2__make_paged_kv_descriptors(inputs)
    if tma_k_scale:
        k_scale_f32 = inputs.k_scale.view(torch.float32)
        ks_desc = TensorDescriptor.from_tensor(k_scale_f32, block_shape=[1, 2, 1, 32])
    else:
        ks_desc = k_desc
    ws = workspace
    num_head_q = int(inputs.q.shape[1])
    reduction_clusters = int(ws.stats.get("reduction_clusters", 0))
    direct_tasks = int(ws.stats.get("direct_tasks", 0))
    use_direct_fast = direct_fast_path and reduction_clusters == 0 and (direct_tasks > 0)
    if use_direct_fast:
        _compute_mtp4__fp8_kvpertensor_decode_mtp4_direct_kernel[direct_tasks,](
            ws.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            ws.schedule.task_map,
            ws.q_scale_3d,
            inputs.k_scale,
            inputs.v_scale,
            ws.out,
            B=inputs.num_batch,
            H_Q=num_head_q,
            HEADS_PER_GROUP=ws.heads_per_group,
            NUM_SEQ_Q=_runtime_mtp2__NUM_SEQ_Q,
            ROWS_Q=_runtime_mtp2__NUM_SEQ_Q * ws.heads_per_group,
            D=_runtime_mtp2__HEAD_DIM,
            DV=_runtime_mtp2__HEAD_DIM,
            BLOCK_SIZE=_runtime_mtp2__BLOCK_SIZE,
            MAX_BLOCKS=inputs.block_ids.shape[1],
            BLOCK_N=_runtime_mtp2__TILE_N,
            DIRECT_CLUSTER_BASE=reduction_clusters,
            DIRECT_CLUSTER_SIZE=ws.config.cluster_size,
            Q_STRIDE_B=ws.q_4d.stride(0),
            Q_STRIDE_M=ws.q_4d.stride(1),
            Q_STRIDE_H=ws.q_4d.stride(2),
            QS_STRIDE_B=ws.q_scale_3d.stride(0),
            QS_STRIDE_M=ws.q_scale_3d.stride(1),
            QS_STRIDE_H=ws.q_scale_3d.stride(2),
            O_STRIDE_B=ws.out.stride(0),
            O_STRIDE_M=ws.out.stride(1),
            O_STRIDE_H=ws.out.stride(2),
            TMA_STAGES=2,
            K_PER_TOKEN_V_PER_HEAD=quant_type == 0,
            KS_STRIDE_BLOCK=ks[0],
            KS_STRIDE_TOKEN=ks[1],
            KS_STRIDE_HEAD=ks[2],
            KS_STRIDE_D=ks[3],
            num_ctas=1,
            num_warps=4,
            num_stages=3,
            launch_pdl=False,
        )
        return ws.out.reshape(inputs.num_batch * _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime_mtp2__HEAD_DIM)
    _compute_mtp2__fp8_kvpertensor_decode_mtp2_final_kernel[ws.schedule.num_clusters,](
        ws.q_4d,
        k_desc,
        ks_desc,
        v_desc,
        inputs.block_ids,
        ws.schedule.task_map,
        inputs.kv_lens,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=_runtime_mtp2__CLUSTER_MESHES[ws.config.cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=_runtime_mtp2__HEAD_DIM,
        DV=_runtime_mtp2__HEAD_DIM,
        BLOCK_SIZE=_runtime_mtp2__BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=_runtime_mtp2__TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_M=ws.q_4d.stride(1),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_M=ws.q_scale_3d.stride(1),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2 if ws.split_out_layout == "MHD" else 3),
        SO_STRIDE_H=ws.split_out.stride(3 if ws.split_out_layout == "MHD" else 2),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3 if ws.lse_layout == "MH" else 4),
        LSE_STRIDE_HG=ws.lse.stride(4 if ws.lse_layout == "MH" else 3),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        USE_LOG2=True,
        K_PER_TOKEN_V_PER_HEAD=quant_type == 0,
        TMA_K_SCALE=tma_k_scale,
        PAGE_METADATA_K_SCALE=page_metadata_k_scale,
        PRECOMBINE_Q_SCALE=precombine_q_scale,
        KS_STRIDE_BLOCK=ks[0],
        KS_STRIDE_TOKEN=ks[1],
        KS_STRIDE_HEAD=ks[2],
        KS_STRIDE_D=ks[3],
        RAW_PAGED_NHD_OR_HND=True,
        FULL_MATRIX_RS=False,
        TRANSPOSED_MEMDESC_RS=False,
        DIRECT_V_SHARED_SHARED=False,
        LDSM_REGISTER_SHARED=not full_view_v_rs,
        TLE_SHARED_SHARED=False,
        FULL_VIEW_V_RS=full_view_v_rs,
        MERGE_CLUSTER_SIZE=ws.config.cluster_size,
        EXECUTION_STAGE=stage_map[execution_stage],
        CLUSTER_COOPERATIVE_FINALIZE=cluster_cooperative_finalize,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.schedule.partial_slots),
        REUSE_FINAL_WEIGHTS=reuse_final_weights,
        HEAD_SHARDED_DSM=head_sharded_dsm,
        PAIRED_HEAD_FINALIZE=paired_head_finalize,
        C8_PAIRED_HEAD_FINALIZE=c8_paired_head_finalize,
        ROW_SERIAL_FINALIZE=row_serial_finalize,
        TWO_CHUNK_FINALIZE=two_chunk_finalize,
        BF16_DSM=bf16_dsm,
        DEFERRED_NORM=deferred_norm,
        PDL_NOTIFY=pdl_notify,
        DSM_ELECTION_HANDOFF=dsm_election_handoff,
        DETERMINISTIC_TAIL_ELECTION=deterministic_tail_election,
        TAIL_ONLY_ELECTION_BARRIER=tail_only_election_barrier,
        REDUCTION_ONLY=reduction_only,
        ALIGNED_FULL_CHUNK_TOKENS=ws.config.chunk_tokens if aligned_full_chunk else 0,
        FULL_VIEW_DSM=full_view_dsm,
        GLOBAL_DEFERRED_NORM=global_deferred_norm,
        QUAD_HEAD_TWO_CHUNK_FINALIZE=quad_head_two_chunk_finalize,
        RANK0_ONLY_FINALIZER=rank0_only_finalizer,
        SKIP_TRAILING_FINALIZER_BARRIER=skip_trailing_finalizer_barrier,
        STATIC_SCHED=False,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=pdl_notify,
    )
    return ws.out.reshape(inputs.num_batch * _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime_mtp2__HEAD_DIM)


_runtime_mtp4__BLOCK_SIZE = 64

_runtime_mtp4__TILE_N = 64

_runtime_mtp4__NUM_SEQ_Q = 4

_runtime_mtp4__HEAD_DIM = 128


@dataclass(frozen=True)
class _runtime_mtp4__DecodeConfig:
    cluster_size: int
    chunk_tokens: int
    direct_threshold: int = 0
    subgroup2_threshold: int = 0

    def __post_init__(self) -> None:
        if self.cluster_size not in (2, 4, 8):
            raise ValueError("runtime policy supports cluster_size 2, 4, or 8")
        if (
            self.chunk_tokens < _runtime_mtp4__TILE_N
            or self.chunk_tokens > 4096
            or self.chunk_tokens % _runtime_mtp4__TILE_N
        ):
            raise ValueError("MTP=4 chunk_tokens must be a multiple of 64 in [64, 4096]")
        if self.direct_threshold < 0 or self.direct_threshold % _runtime_mtp4__TILE_N:
            raise ValueError("direct_threshold must be zero or a multiple of 64")
        if self.subgroup2_threshold < 0 or self.subgroup2_threshold % _runtime_mtp4__TILE_N:
            raise ValueError("subgroup2_threshold must be zero or a multiple of 64")
        if self.subgroup2_threshold and self.cluster_size != 4:
            raise ValueError("dual-C2 subgroup packing requires cluster_size=4")


_runtime_mtp4__C2_T256 = _runtime_mtp4__DecodeConfig(2, 256)

_runtime_mtp4__C2_T1024 = _runtime_mtp4__DecodeConfig(2, 1024)

_runtime_mtp4__C4_T512 = _runtime_mtp4__DecodeConfig(4, 512)

_runtime_mtp4__C4_T1024 = _runtime_mtp4__DecodeConfig(4, 1024)


@dataclass
class _runtime_mtp4__DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def num_batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def max_seq_kv(self) -> int:
        return int(self.kv_lens.max().item())


@dataclass
class _runtime_mtp4__DecodeWorkspace:
    config: _runtime_mtp4__DecodeConfig
    schedule: _scheduler_mtp24__DecodeTaskSchedule
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    all_chunks_aligned: bool = False
    final_policy_mode: str = "winner"
    static_sched: bool = False
    static_chunk_tokens: int = 0
    static_max_groups: int = 1

    @property
    def stats(self) -> dict:
        return dict(self.schedule.stats or {})


_runtime_mtp4__CLUSTER_MESHES = (
    {
        2: tle.device_mesh({"block_cluster": [("cluster_x", 2)]}),
        4: tle.device_mesh({"block_cluster": [("cluster_x", 4)]}),
        8: tle.device_mesh({"block_cluster": [("cluster_x", 8)]}),
    }
    if USE_TLE
    else {2: None, 4: None, 8: None}
)


def _runtime_mtp4___validate_inputs(inputs: _runtime_mtp4__DecodeInputs) -> tuple[int, int, int]:
    if not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA tensor for GPU task scheduling")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _runtime_mtp4__HEAD_DIM:
        raise ValueError("q must have flattened shape [batch * 4, num_head_q, 128]")
    if inputs.q.shape[0] != inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q:
        raise ValueError("MTP=4 requires q.shape[0] == batch * 4")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.ndim != 4 or cache.shape[1] != _runtime_mtp4__BLOCK_SIZE or cache.shape[3] != _runtime_mtp4__HEAD_DIM:
            raise ValueError(f"{name} must be logical [block, 64, head, 128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if num_head_q % num_head_kv:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    heads_per_group = num_head_q // num_head_kv
    if heads_per_group > 8:
        raise ValueError("heads_per_group must be <= 8")
    return (num_head_q, num_head_kv, heads_per_group)


def _runtime_mtp4__make_paged_kv_descriptors(
    inputs: _runtime_mtp4__DecodeInputs,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Project NHD/HND paged descriptors to logical [block, head, token, dim] tiles."""
    _runtime_mtp4___validate_inputs(inputs)
    block_shape = [1, 1, _runtime_mtp4__TILE_N, _runtime_mtp4__HEAD_DIM]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


def _runtime_mtp4__prepare_decode_workspace(
    inputs: _runtime_mtp4__DecodeInputs,
    config: _runtime_mtp4__DecodeConfig | None = None,
    *,
    split_out_dtype: torch.dtype = torch.float32,
    short_threshold: int = 0,
    short_chunk_tokens: int = 0,
    raw_chunk_minor: bool = False,
    raw_scalar_chunk_minor: bool | None = None,
) -> _runtime_mtp4__DecodeWorkspace:
    """Allocate buffers and build the complete cluster task map on the GPU."""
    num_head_q, num_head_kv, heads_per_group = _runtime_mtp4___validate_inputs(inputs)
    if config is None:
        raise ValueError("MTP=4 policy is not finalized; pass an explicit DecodeConfig")
    if split_out_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("split_out_dtype must be float32 or bfloat16")
    schedule = _scheduler_mtp24__allocate_cluster_task_map(
        inputs.kv_lens,
        num_head_kv=num_head_kv,
        max_seq_kv=inputs.max_seq_kv,
        cluster_size=config.cluster_size,
        chunk_tokens=config.chunk_tokens,
        direct_threshold=config.direct_threshold,
        short_threshold=short_threshold,
        short_chunk_tokens=short_chunk_tokens,
        subgroup2_threshold=config.subgroup2_threshold,
        num_seq_q=_runtime_mtp4__NUM_SEQ_Q,
    )
    if schedule.stats is None:
        raise RuntimeError("GPU task scheduler did not publish workspace metadata")
    all_chunks_aligned = bool(torch.all(inputs.kv_lens % config.chunk_tokens == 0).item())
    q_4d = inputs.q.reshape(inputs.num_batch, _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime_mtp4__HEAD_DIM)
    q_scale_3d = inputs.q_scale.reshape(inputs.num_batch, _runtime_mtp4__NUM_SEQ_Q, num_head_q)
    pad_heads_per_group = (heads_per_group + 7) // 8 * 8
    device = inputs.q.device
    max_final_chunks = triton.next_power_of_2(schedule.partial_slots)
    partial_storage_slots = 2 * max_final_chunks
    scalar_chunk_minor = raw_chunk_minor if raw_scalar_chunk_minor is None else raw_scalar_chunk_minor
    if raw_chunk_minor:
        split_out = torch.empty(
            (inputs.num_batch, num_head_q, _runtime_mtp4__NUM_SEQ_Q, partial_storage_slots, _runtime_mtp4__HEAD_DIM),
            dtype=split_out_dtype,
            device=device,
        ).permute(0, 3, 2, 1, 4)
    else:
        split_out = torch.empty(
            (inputs.num_batch, partial_storage_slots, _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime_mtp4__HEAD_DIM),
            dtype=split_out_dtype,
            device=device,
        )
    if scalar_chunk_minor:
        lse = torch.empty(
            (inputs.num_batch, num_head_kv, _runtime_mtp4__NUM_SEQ_Q, pad_heads_per_group, partial_storage_slots),
            dtype=torch.float32,
            device=device,
        ).permute(0, 4, 1, 2, 3)
    else:
        lse = torch.empty(
            (inputs.num_batch, partial_storage_slots, num_head_kv, _runtime_mtp4__NUM_SEQ_Q, pad_heads_per_group),
            dtype=torch.float32,
            device=device,
        )
    num_counters = num_head_kv * inputs.num_batch
    return _runtime_mtp4__DecodeWorkspace(
        config=config,
        schedule=schedule,
        q_4d=q_4d,
        q_scale_3d=q_scale_3d,
        split_out=split_out,
        lse=lse,
        completion=torch.zeros((num_counters * (1 + max_final_chunks),), dtype=torch.int32, device=device),
        last_flags=torch.zeros((schedule.physical_ctas,), dtype=torch.int32, device=device),
        out=torch.empty(
            (inputs.num_batch, _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime_mtp4__HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
        heads_per_group=heads_per_group,
        all_chunks_aligned=all_chunks_aligned,
    )


def _runtime_mtp4__refresh_decode_schedule(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    mode: Literal["full", "tail"] = "full",
) -> None:
    """Refresh an existing GPU schedule when its allocated topology is stable."""
    num_head_kv = int(inputs.k_cache.shape[2])
    if mode == "full":
        _scheduler_mtp24__launch_cluster_task_map_assign(
            inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv, refresh_host_metadata=False
        )
    elif mode == "tail":
        _scheduler_mtp24__launch_cluster_task_tail_refresh(inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv)
    else:
        raise ValueError("mode must be 'full' or 'tail'")
    workspace.all_chunks_aligned = bool(torch.all(inputs.kv_lens % workspace.config.chunk_tokens == 0).item())


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace | None = None,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
    execution_stage: Literal["full", "cluster", "local", "election"] = "full",
    head_sharded_dsm: bool = False,
    paired_head_finalize: bool | None = None,
    persistent_clusters: int | None = None,
    direct_fast_path: bool = False,
    two_chunk_finalize: bool = False,
    quad_head_two_chunk_finalize: bool = False,
    fast_finalizer_handoff: bool = False,
    rank0_only_finalizer: bool = False,
    skip_trailing_finalizer_barrier: bool = False,
    winner_local_reuse_finalizer: bool = False,
    split_d_dsm_merge: bool = False,
    tma_stages: int = 2,
    compiler_num_stages: int = 3,
    maxnreg: int | None = None,
    c4_finalizer_tile: int = 1,
    c4_dynamic_finalizer_stages: int = 0,
    c4_q_dsm_fanout: bool = False,
    c4_exact_two_finalize: bool = False,
    c4_hierarchical_finalize_pack: int = 0,
    c4_bf16_dsm: bool = False,
    c4_deferred_norm: bool = False,
    c4_global_deferred_norm: bool = False,
    c8_bf16_deferred_norm: bool = False,
    c4_reduction_only_raw: bool = False,
    c4_aligned_full_chunk_raw: bool = False,
    c2_aligned_full_chunk_winner: bool = False,
    full_view_v_rs: bool = False,
    full_view_p_store: bool = False,
    full_view_dsm: bool = False,
    dual_acc_raw_finalizer: bool = False,
    chunk4_raw_finalizer: bool = False,
    sharded_election: bool = False,
    dsm_election_handoff: bool = False,
    streaming_raw_finalizer: bool = False,
    deterministic_tail_reuse: bool = False,
    recompute_raw_finalizer: bool = False,
    tail_only_election_barrier: bool = False,
    tma_raw_finalizer: bool = False,
    cta_role_rank0_tiles: int = -1,
    pdl_notify: bool = False,
    prefetch_k_scale: bool = False,
    packed_k_scale_lookahead: bool = False,
    shared_k_scale_pipeline: bool = False,
    wgmma_shadow_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    tma_k_scale: bool = False,
    fragment_k_scale_load: bool = False,
    precombine_q_scale: bool = False,
    defer_v_scale_final: bool = False,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """Run the independent public-TLE MTP=4 n32 typed-RS policy."""
    if workspace is None:
        raise ValueError("MTP=4 requires an explicitly configured workspace")
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 or 1")
    if quant_type == 0 and (inputs.k_scale.ndim != 4 or inputs.v_scale.numel() < inputs.k_cache.shape[2]):
        raise ValueError("quant_type=0 requires packed rank-4 K scales and one V scale per KV head")
    if tma_k_scale and quant_type != 0:
        raise ValueError("TLE K-scale TMA requires quant_type=0")
    if packed_k_scale_lookahead and quant_type != 0:
        raise ValueError("packed K-scale lookahead requires quant_type=0")
    if shared_k_scale_pipeline and quant_type != 0:
        raise ValueError("shared K-scale pipeline requires quant_type=0")
    if wgmma_shadow_k_scale and quant_type != 0:
        raise ValueError("WGMMA-shadow K-scale load requires quant_type=0")
    if page_metadata_k_scale and quant_type != 0:
        raise ValueError("page-metadata K-scale pipeline requires quant_type=0")
    scalar_k_scale_prefetch = prefetch_k_scale and (not tma_k_scale)
    if (
        sum(
            (
                bool(value)
                for value in (
                    scalar_k_scale_prefetch,
                    packed_k_scale_lookahead,
                    shared_k_scale_pipeline,
                    wgmma_shadow_k_scale,
                    page_metadata_k_scale,
                    tma_k_scale,
                )
            )
        )
        > 1
    ):
        raise ValueError(
            "K-scale prefetch, lookahead, shared pipeline, WGMMA shadow load, page-metadata pipeline and TMA are exclusive"
        )
    if fragment_k_scale_load and (not tma_k_scale):
        raise ValueError("fragment K-scale load requires TLE K-scale TMA")
    if defer_v_scale_final and (
        not (quant_type == 0 and workspace.config.cluster_size == 2 and paired_head_finalize and (not head_sharded_dsm))
    ):
        raise ValueError("deferred V scale currently requires the quant0 C2 paired-head path")
    ks = inputs.k_scale.stride() if quant_type == 0 else (0, 0, 0, 0)
    stage_map = {
        "full": _compute_mtp4__EXECUTION_FULL,
        "cluster": _compute_mtp4__EXECUTION_CLUSTER_PARTIAL,
        "local": _compute_mtp4__EXECUTION_LOCAL_PARTIAL,
        "election": _compute_mtp4__EXECUTION_ELECTION_ONLY,
    }
    if execution_stage not in stage_map:
        raise ValueError("execution_stage must be 'full', 'cluster', 'local', or 'election'")
    if tma_stages not in (2, 4):
        raise ValueError("tma_stages must be 2 or 4")
    if compiler_num_stages not in (2, 3, 4):
        raise ValueError("compiler_num_stages must be 2, 3, or 4")
    if maxnreg is not None and (not 32 <= maxnreg <= 255):
        raise ValueError("maxnreg must be in [32, 255] or None")
    if full_view_p_store and (not full_view_v_rs):
        raise ValueError("full-view P store currently requires full-view V RS")
    if full_view_dsm and (not (full_view_v_rs and c4_bf16_dsm and c4_deferred_norm)):
        raise ValueError("full-view DSM requires full-view V and C4 BF16 deferred state")
    if dual_acc_raw_finalizer and (not c4_global_deferred_norm):
        raise ValueError("dual-acc raw finalizer requires C4 global deferred normalization")
    if chunk4_raw_finalizer and (not c4_global_deferred_norm):
        raise ValueError("chunk4 raw finalizer requires C4 global deferred normalization")
    if chunk4_raw_finalizer and dual_acc_raw_finalizer:
        raise ValueError("chunk4 and dual-acc raw finalizers are exclusive")
    if sharded_election and (not c4_global_deferred_norm):
        raise ValueError("sharded election requires C4 global deferred normalization")
    if dsm_election_handoff and (not (c4_global_deferred_norm and workspace.config.cluster_size == 4)):
        raise ValueError("DSM election handoff requires C4 global deferred normalization")
    if streaming_raw_finalizer and (not (c4_global_deferred_norm and workspace.config.cluster_size == 4)):
        raise ValueError("streaming raw finalizer requires C4 global deferred normalization")
    if deterministic_tail_reuse and (not (c4_global_deferred_norm and workspace.config.cluster_size == 4)):
        raise ValueError("deterministic tail reuse requires C4 global deferred normalization")
    if deterministic_tail_reuse and (dsm_election_handoff or streaming_raw_finalizer):
        raise ValueError("deterministic tail reuse owns the election and raw finalizer")
    if recompute_raw_finalizer and (
        not (deterministic_tail_reuse and c4_global_deferred_norm and (workspace.config.cluster_size == 4))
    ):
        raise ValueError("recomputed raw finalizer requires the C4 deterministic-tail path")
    if recompute_raw_finalizer and streaming_raw_finalizer:
        raise ValueError("recomputed and streaming raw finalizers are exclusive")
    if tail_only_election_barrier and (not deterministic_tail_reuse):
        raise ValueError("tail-only election barrier requires deterministic tail election")
    if tma_raw_finalizer and (
        not (
            deterministic_tail_reuse
            and c4_global_deferred_norm
            and full_view_v_rs
            and (workspace.config.cluster_size == 4)
        )
    ):
        raise ValueError("TMA raw finalizer requires the C4 full-view deterministic-tail path")
    if tma_raw_finalizer and (recompute_raw_finalizer or streaming_raw_finalizer):
        raise ValueError("TMA, recomputed, and streaming raw finalizers are exclusive")
    if c4_finalizer_tile not in (1, 2, 4, 5, 6, 7):
        raise ValueError("c4_finalizer_tile must be 1, 2, 4, 5, 6, or 7")
    if c4_dynamic_finalizer_stages not in (0, 1, 2, 3):
        raise ValueError("c4_dynamic_finalizer_stages must be 0, 1, 2, or 3")
    if c4_dynamic_finalizer_stages and c4_finalizer_tile != 1:
        raise ValueError("dynamic and tiled C4 finalizers are mutually exclusive")
    if c4_q_dsm_fanout and c4_dynamic_finalizer_stages:
        raise ValueError(
            "C4 Q DSM fanout and the dynamic finalizer implementation cannot share the internal mode encoding"
        )
    if c4_exact_two_finalize and (c4_dynamic_finalizer_stages or c4_q_dsm_fanout):
        raise ValueError("C4 exact-two, dynamic, and Q-fanout specializations are mutually exclusive")
    if c4_exact_two_finalize and c4_finalizer_tile != 1:
        raise ValueError("c4_exact_two_finalize currently requires c4_finalizer_tile=1")
    if c4_hierarchical_finalize_pack not in (0, 4, 8):
        raise ValueError("c4_hierarchical_finalize_pack must be 0, 4, or 8")
    if c4_hierarchical_finalize_pack and (c4_dynamic_finalizer_stages or c4_q_dsm_fanout or c4_exact_two_finalize):
        raise ValueError("C4 hierarchical, exact-two, dynamic, and Q-fanout specializations are mutually exclusive")
    if c4_hierarchical_finalize_pack and c4_finalizer_tile != 1:
        raise ValueError("C4 hierarchical finalization currently requires tile=1")
    if c4_hierarchical_finalize_pack and (not (fast_finalizer_handoff and skip_trailing_finalizer_barrier)):
        raise ValueError("C4 hierarchical finalization requires fast handoff and no trailing finalizer barrier")
    if c4_hierarchical_finalize_pack and head_sharded_dsm:
        raise ValueError("C4 hierarchical finalization and head-sharded DSM are separate specializations")
    if c4_deferred_norm and (not c4_bf16_dsm):
        raise ValueError("C4 deferred normalization requires BF16 DSM")
    if c4_global_deferred_norm and (not c4_deferred_norm):
        raise ValueError("C4 global deferred normalization requires DSM deferred normalization")
    if c8_bf16_deferred_norm and workspace.config.cluster_size != 8:
        raise ValueError("C8 BF16 deferred normalization requires C8")
    if c8_bf16_deferred_norm and (c4_bf16_dsm or c4_deferred_norm or c4_global_deferred_norm):
        raise ValueError("C8 and C4 deferred-normalization modes are exclusive")
    if c4_reduction_only_raw:
        if workspace.config not in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024):
            raise ValueError("reduction-only raw specialization requires C4T512/C4T1024")
        if (
            any((int(workspace.stats.get(name, 0)) for name in ("direct_tasks", "dummy_tasks")))
            or workspace.config.subgroup2_threshold
        ):
            raise ValueError("reduction-only raw specialization requires grouped real tasks only")
        if c4_bf16_dsm or c4_deferred_norm or c4_global_deferred_norm:
            raise ValueError("reduction-only raw mode internally selects the complete C4 raw ABI")
    if c4_aligned_full_chunk_raw:
        if workspace.config == _runtime_mtp4__C4_T512 and (not c4_reduction_only_raw):
            raise ValueError("C4T512 aligned full-chunk raw requires reduction-only mode")
        if workspace.config == _runtime_mtp4__C4_T1024 and c4_reduction_only_raw:
            raise ValueError("C4T1024 aligned full-chunk raw must retain single-group finalization")
        if workspace.config not in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024):
            raise ValueError("aligned full-chunk raw requires C4T512/C4T1024")
        if not workspace.all_chunks_aligned:
            raise ValueError("aligned full-chunk raw requires aligned KV lengths")
    if c2_aligned_full_chunk_winner:
        if workspace.config not in (_runtime_mtp4__C2_T256, _runtime_mtp4__C2_T1024):
            raise ValueError("aligned winner specialization requires C2T256/C2T1024")
        if not workspace.all_chunks_aligned:
            raise ValueError("aligned winner requires chunk-aligned KV lengths")
        incompatible = {
            name: int(workspace.stats.get(name, 0))
            for name in ("direct_tasks", "dummy_tasks", "subgroup2_tasks")
            if int(workspace.stats.get(name, 0))
        }
        if incompatible:
            raise ValueError(f"aligned winner requires grouped real tasks only: {incompatible}")
        aligned_c2_raw = c4_bf16_dsm and c4_deferred_norm and (not c4_global_deferred_norm)
        if any((c4_reduction_only_raw, c4_aligned_full_chunk_raw, c8_bf16_deferred_norm)):
            raise ValueError("aligned C2 compute cannot combine with C4/C8 raw modes")
        if any((c4_bf16_dsm, c4_deferred_norm, c4_global_deferred_norm)) and (not aligned_c2_raw):
            raise ValueError("aligned C2 raw compute requires BF16 DSM plus deferred norm")
    if c4_bf16_dsm and (
        c4_dynamic_finalizer_stages
        or c4_q_dsm_fanout
        or c4_exact_two_finalize
        or c4_hierarchical_finalize_pack
        or (head_sharded_dsm and (not c4_global_deferred_norm))
    ):
        raise ValueError("C4 BF16 DSM must be measured separately from the other DSM/finalizer specializations")
    if paired_head_finalize is None:
        paired_head_finalize = workspace.config.cluster_size == 4
    if paired_head_finalize and workspace.config.cluster_size not in (2, 4):
        raise ValueError("paired_head_finalize requires cluster_size=2 or 4")
    if paired_head_finalize and workspace.heads_per_group != 8:
        raise ValueError("paired_head_finalize requires heads_per_group=8")
    if head_sharded_dsm and workspace.heads_per_group != 8:
        raise ValueError("head_sharded_dsm requires heads_per_group=8")
    if persistent_clusters is not None and persistent_clusters < 1:
        raise ValueError("persistent_clusters must be positive")
    if cta_role_rank0_tiles < -1:
        raise ValueError("cta_role_rank0_tiles must be -1 or non-negative")
    if cta_role_rank0_tiles >= 0 and (
        persistent_clusters is None or workspace.config.cluster_size != 4 or execution_stage not in ("cluster", "full")
    ):
        raise ValueError("CTA-role requires persistent C4 execution with execution_stage='cluster' or 'full'")
    if (
        cta_role_rank0_tiles >= 0
        and execution_stage == "full"
        and (
            not (
                c4_bf16_dsm
                and paired_head_finalize is True
                and fast_finalizer_handoff
                and skip_trailing_finalizer_barrier
            )
        )
    ):
        raise ValueError(
            "full CTA-role requires BF16 DSM, paired-head fast handoff, and skip_trailing_finalizer_barrier"
        )
    if cta_role_rank0_tiles == 7 and int(workspace.stats.get("dummy_tasks", 0)):
        raise ValueError("the fixed r0=7 CTA-role specialization requires full C4 groups")
    if pdl_notify and persistent_clusters is not None:
        raise ValueError("PDL notification currently requires non-persistent execution")
    if pdl_notify and direct_fast_path:
        raise ValueError("PDL notification cannot be combined with direct_fast_path")
    effective_chunks_max = int(workspace.stats.get("effective_chunks_max", 1))
    if two_chunk_finalize and effective_chunks_max != 2:
        raise ValueError(f"two_chunk_finalize requires effective_chunks_max == 2, got {effective_chunks_max}")
    if two_chunk_finalize and workspace.config.cluster_size != 2:
        raise ValueError("two_chunk_finalize currently targets cluster_size=2")
    if quad_head_two_chunk_finalize and effective_chunks_max != 2:
        raise ValueError(f"quad_head_two_chunk_finalize requires effective_chunks_max == 2, got {effective_chunks_max}")
    if quad_head_two_chunk_finalize and workspace.config.cluster_size != 2:
        raise ValueError("quad_head_two_chunk_finalize requires cluster_size=2")
    if quad_head_two_chunk_finalize and workspace.heads_per_group != 8:
        raise ValueError("quad_head_two_chunk_finalize requires heads_per_group=8")
    if quad_head_two_chunk_finalize and two_chunk_finalize:
        raise ValueError("select either single-head or quad-head two-chunk finalization")
    if quad_head_two_chunk_finalize and paired_head_finalize:
        raise ValueError("select either exact-two or arbitrary-chunk c2 quad finalization")
    c2_quad_finalizer = (quad_head_two_chunk_finalize or paired_head_finalize) and workspace.config.cluster_size == 2
    c4_paired_finalizer = paired_head_finalize and workspace.config.cluster_size == 4
    if c4_finalizer_tile != 1 and (not c4_paired_finalizer):
        raise ValueError("c4_finalizer_tile > 1 requires the c4 paired-head finalizer")
    if c4_dynamic_finalizer_stages and (not c4_paired_finalizer):
        raise ValueError("c4_dynamic_finalizer_stages requires the C4 paired finalizer")
    if c4_q_dsm_fanout and (not c4_paired_finalizer):
        raise ValueError("c4_q_dsm_fanout requires the C4 paired path")
    if c4_exact_two_finalize and (not c4_paired_finalizer):
        raise ValueError("c4_exact_two_finalize requires the C4 paired path")
    if c4_hierarchical_finalize_pack and (not c4_paired_finalizer):
        raise ValueError("C4 hierarchical finalization requires the C4 paired path")
    if c4_bf16_dsm and (not (c4_paired_finalizer or c2_quad_finalizer)):
        raise ValueError("BF16 DSM requires either the C4 paired path or a C2 quad-head path")
    if c4_q_dsm_fanout and int(workspace.stats.get("dummy_tasks", 0)):
        raise ValueError("c4_q_dsm_fanout requires every cluster rank to participate")
    if c4_q_dsm_fanout and int(workspace.stats.get("direct_tasks", 0)):
        raise ValueError("c4_q_dsm_fanout currently requires a reduction-only schedule")
    c8_rank_sharded_finalizer = not paired_head_finalize and workspace.config.cluster_size == 8
    if fast_finalizer_handoff and (not (c2_quad_finalizer or c4_paired_finalizer or c8_rank_sharded_finalizer)):
        raise ValueError(
            "fast_finalizer_handoff requires either the c2 quad-head path, the c4 paired-head path, or the c8 rank-sharded path"
        )
    if rank0_only_finalizer and (not c2_quad_finalizer):
        raise ValueError("rank0_only_finalizer requires a c2 quad-head finalizer")
    if rank0_only_finalizer and fast_finalizer_handoff:
        raise ValueError("rank0_only_finalizer replaces rather than combines with fast_finalizer_handoff")
    if skip_trailing_finalizer_barrier and (
        not (
            rank0_only_finalizer
            or (fast_finalizer_handoff and (c2_quad_finalizer or c4_paired_finalizer or c8_rank_sharded_finalizer))
        )
    ):
        raise ValueError(
            "skip_trailing_finalizer_barrier requires either the c2 rank0-only path or a rank-sharded fast handoff path"
        )
    c2_winner_local_reuse = rank0_only_finalizer and skip_trailing_finalizer_barrier and quad_head_two_chunk_finalize
    c4_raw_winner_local_reuse = (
        c4_paired_finalizer
        and c4_global_deferred_norm
        and c4_bf16_dsm
        and fast_finalizer_handoff
        and skip_trailing_finalizer_barrier
    )
    if winner_local_reuse_finalizer and (not (c2_winner_local_reuse or c4_raw_winner_local_reuse)):
        raise ValueError(
            "winner_local_reuse_finalizer requires either the c2 quad-head rank0-only path or the C4 global-deferred paired-head fast-handoff path"
        )
    if split_d_dsm_merge and workspace.config.cluster_size != 2:
        raise ValueError("split_d_dsm_merge requires cluster_size=2")
    if split_d_dsm_merge and head_sharded_dsm:
        raise ValueError("split_d_dsm_merge and head_sharded_dsm are mutually exclusive")
    if split_d_dsm_merge and (not (rank0_only_finalizer and skip_trailing_finalizer_barrier)):
        raise ValueError("split_d_dsm_merge currently requires the rank0-only no-trailing-barrier finalizer")
    if split_d_dsm_merge and winner_local_reuse_finalizer:
        raise ValueError("split_d_dsm_merge cannot reuse a complete rank0 group partial")
    if refresh_schedule is not None:
        if workspace.static_sched:
            raise ValueError("strict-static workspaces cannot refresh a task map")
        if workspace.config.subgroup2_threshold:
            raise ValueError("dual-C2 subgroup schedules do not support tail refresh")
        _runtime_mtp4__refresh_decode_schedule(inputs, workspace, refresh_schedule)
    k_desc, v_desc = _runtime_mtp4__make_paged_kv_descriptors(inputs)
    if tma_k_scale:
        k_scale_f32 = inputs.k_scale.view(torch.float32)
        ks_desc = TensorDescriptor.from_tensor(k_scale_f32, block_shape=[1, 2, 1, 32])
    else:
        ks_desc = k_desc
    ws = workspace
    num_head_q = int(inputs.q.shape[1])
    reduction_clusters = int(ws.stats.get("reduction_clusters", ws.schedule.num_clusters))
    direct_tasks = int(ws.stats.get("direct_tasks", 0))
    use_direct_fast = direct_fast_path and reduction_clusters == 0
    logical_clusters = 0 if use_direct_fast else ws.schedule.num_clusters
    launch_clusters = min(logical_clusters, logical_clusters if persistent_clusters is None else persistent_clusters)
    if persistent_clusters is not None:
        if ws.config.cluster_size not in (2, 4):
            raise ValueError("MTP4 micro-persistent K/V lookahead requires C2 or C4")
        if logical_clusters > 2 * launch_clusters:
            raise ValueError(
                "MTP4 micro-persistent K/V lookahead supports at most two logical epochs per physical cluster; increase persistent_clusters"
            )

    def launch_direct() -> None:
        if not use_direct_fast or direct_tasks == 0:
            return
        _compute_mtp4__fp8_kvpertensor_decode_mtp4_direct_kernel[direct_tasks,](
            ws.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            ws.schedule.task_map,
            ws.q_scale_3d,
            inputs.k_scale,
            inputs.v_scale,
            ws.out,
            B=inputs.num_batch,
            H_Q=num_head_q,
            HEADS_PER_GROUP=ws.heads_per_group,
            NUM_SEQ_Q=_runtime_mtp4__NUM_SEQ_Q,
            ROWS_Q=_runtime_mtp4__NUM_SEQ_Q * ws.heads_per_group,
            D=_runtime_mtp4__HEAD_DIM,
            DV=_runtime_mtp4__HEAD_DIM,
            BLOCK_SIZE=_runtime_mtp4__BLOCK_SIZE,
            MAX_BLOCKS=inputs.block_ids.shape[1],
            BLOCK_N=_runtime_mtp4__TILE_N,
            DIRECT_CLUSTER_BASE=reduction_clusters,
            DIRECT_CLUSTER_SIZE=ws.config.cluster_size,
            Q_STRIDE_B=ws.q_4d.stride(0),
            Q_STRIDE_M=ws.q_4d.stride(1),
            Q_STRIDE_H=ws.q_4d.stride(2),
            QS_STRIDE_B=ws.q_scale_3d.stride(0),
            QS_STRIDE_M=ws.q_scale_3d.stride(1),
            QS_STRIDE_H=ws.q_scale_3d.stride(2),
            O_STRIDE_B=ws.out.stride(0),
            O_STRIDE_M=ws.out.stride(1),
            O_STRIDE_H=ws.out.stride(2),
            TMA_STAGES=tma_stages,
            K_PER_TOKEN_V_PER_HEAD=quant_type == 0,
            KS_STRIDE_BLOCK=ks[0],
            KS_STRIDE_TOKEN=ks[1],
            KS_STRIDE_HEAD=ks[2],
            KS_STRIDE_D=ks[3],
            num_ctas=1,
            num_warps=4,
            num_stages=compiler_num_stages,
            maxnreg=maxnreg,
            launch_pdl=False,
        )

    if logical_clusters == 0:
        launch_direct()
        if flatten_output:
            return ws.out.reshape(inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime_mtp4__HEAD_DIM)
        return ws.out
    compute_kernel = (
        _compute_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle_kernel
        if persistent_clusters is None
        else _compute_mtp4__fp8_kvpertensor_decode_mtp4_persistent_kernel
    )
    compute_kernel[launch_clusters,](
        ws.q_4d,
        k_desc,
        v_desc,
        ks_desc,
        inputs.block_ids,
        ws.schedule.task_map,
        inputs.kv_lens,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=_runtime_mtp4__CLUSTER_MESHES[ws.config.cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=_runtime_mtp4__HEAD_DIM,
        DV=_runtime_mtp4__HEAD_DIM,
        BLOCK_SIZE=_runtime_mtp4__BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=_runtime_mtp4__TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_M=ws.q_4d.stride(1),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_M=ws.q_scale_3d.stride(1),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        USE_LOG2=True,
        LOGICAL_NUM_CLUSTERS=logical_clusters,
        PERSISTENT_NUM_CLUSTERS=launch_clusters,
        RAW_PAGED_NHD_OR_HND=True,
        FULL_MATRIX_RS=False,
        TRANSPOSED_MEMDESC_RS=False,
        DIRECT_V_SHARED_SHARED=False,
        LDSM_REGISTER_SHARED=not full_view_v_rs,
        TLE_SHARED_SHARED=False,
        WIDE_VIEW_V_RS=full_view_v_rs,
        FULL_VIEW_P_STORE=full_view_p_store,
        FULL_VIEW_DSM=full_view_dsm,
        DUAL_ACC_RAW_FINALIZER=dual_acc_raw_finalizer,
        CHUNK4_RAW_FINALIZER=chunk4_raw_finalizer,
        SHARDED_ELECTION=sharded_election,
        DSM_ELECTION_HANDOFF=dsm_election_handoff,
        STREAMING_RAW_FINALIZER=streaming_raw_finalizer,
        DETERMINISTIC_TAIL_REUSE=deterministic_tail_reuse,
        RECOMPUTE_RAW_FINALIZER=recompute_raw_finalizer,
        TAIL_ONLY_ELECTION_BARRIER=tail_only_election_barrier,
        TMA_RAW_FINALIZER=tma_raw_finalizer,
        MERGE_CLUSTER_SIZE=ws.config.cluster_size,
        EXECUTION_STAGE=stage_map[execution_stage],
        CLUSTER_COOPERATIVE_FINALIZE=True,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.schedule.partial_slots),
        REUSE_FINAL_WEIGHTS=True,
        HEAD_SHARDED_DSM=head_sharded_dsm,
        PAIRED_HEAD_FINALIZE=paired_head_finalize,
        TWO_CHUNK_FINALIZE=two_chunk_finalize,
        QUAD_HEAD_TWO_CHUNK_FINALIZE=quad_head_two_chunk_finalize,
        FAST_FINALIZER_HANDOFF=fast_finalizer_handoff,
        RANK0_ONLY_FINALIZER=rank0_only_finalizer,
        SKIP_TRAILING_FINALIZER_BARRIER=skip_trailing_finalizer_barrier,
        WINNER_LOCAL_REUSE_FINALIZER=winner_local_reuse_finalizer,
        SPLIT_D_DSM_MERGE=split_d_dsm_merge,
        PDL_NOTIFY=pdl_notify,
        TMA_STAGES=tma_stages,
        C4_FINALIZER_TILE=-c4_dynamic_finalizer_stages
        if c4_dynamic_finalizer_stages
        else (
            1200 + c4_finalizer_tile
            if c4_bf16_dsm and c4_deferred_norm
            else 1400 + c4_finalizer_tile
            if workspace.config == _runtime_mtp4__C2_T256
            else 1100 + c4_finalizer_tile
        )
        if c2_aligned_full_chunk_winner
        else (1300 + c4_finalizer_tile if workspace.config == _runtime_mtp4__C4_T1024 else 1000 + c4_finalizer_tile)
        if c4_aligned_full_chunk_raw
        else 900 + c4_finalizer_tile
        if c4_reduction_only_raw
        else 800 + c4_finalizer_tile
        if c8_bf16_deferred_norm
        else 700 + c4_finalizer_tile
        if c4_global_deferred_norm
        else 600 + c4_finalizer_tile
        if c4_deferred_norm
        else 500 + c4_finalizer_tile
        if c4_bf16_dsm
        else 400 + c4_hierarchical_finalize_pack
        if c4_hierarchical_finalize_pack
        else 200 + c4_finalizer_tile
        if c4_exact_two_finalize
        else 100 + c4_finalizer_tile
        if c4_q_dsm_fanout
        else c4_finalizer_tile,
        CTA_ROLE_RANK0_TILES=cta_role_rank0_tiles,
        K_PER_TOKEN_V_PER_HEAD=quant_type == 0,
        KS_STRIDE_BLOCK=ks[0],
        KS_STRIDE_TOKEN=ks[1],
        KS_STRIDE_HEAD=ks[2],
        KS_STRIDE_D=ks[3],
        PREFETCH_K_SCALE=5
        if page_metadata_k_scale
        else 4
        if wgmma_shadow_k_scale
        else 3
        if shared_k_scale_pipeline
        else 2
        if packed_k_scale_lookahead
        else prefetch_k_scale,
        TMA_K_SCALE=tma_k_scale,
        FRAGMENT_K_SCALE_LOAD=fragment_k_scale_load,
        PRECOMBINE_Q_SCALE=precombine_q_scale,
        DEFER_V_SCALE_FINAL=defer_v_scale_final,
        STATIC_SCHED=ws.static_sched,
        STATIC_CHUNK_TOKENS=ws.static_chunk_tokens,
        STATIC_MAX_GROUPS=ws.static_max_groups,
        num_ctas=1,
        num_warps=4,
        num_stages=compiler_num_stages,
        maxnreg=maxnreg,
        launch_pdl=pdl_notify,
    )
    launch_direct()
    if flatten_output:
        return ws.out.reshape(inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime_mtp4__HEAD_DIM)
    return ws.out


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_winner(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
    tma_stages: int = 2,
    compiler_num_stages: int = 3,
    maxnreg: int | None = None,
    prefetch_k_scale: bool = False,
    tma_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    fragment_k_scale_load: bool = False,
    precombine_q_scale: bool = False,
    defer_v_scale_final: bool = False,
    aligned_full_chunk_winner: bool = False,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """Run the currently validated MTP=4 policy specializations.

    Policy selection remains outside the low-level launch implementation.
    """
    config = workspace.config
    effective_chunks_max = int(workspace.stats.get("effective_chunks_max", 1))
    use_c4_paired_fast = config.cluster_size == 4
    use_c2_exact_two = config.cluster_size == 2 and effective_chunks_max == 2
    use_c2_rank0_quad = config.cluster_size == 2 and effective_chunks_max == 4
    use_c2_sharded_quad = config.cluster_size == 2 and effective_chunks_max >= 32
    use_general_c2_quad = use_c2_rank0_quad or use_c2_sharded_quad
    use_fast_handoff = use_c4_paired_fast or use_c2_sharded_quad
    use_rank0 = use_c2_exact_two or use_c2_rank0_quad
    selected_maxnreg = maxnreg if maxnreg is not None else 240 if use_c4_paired_fast else None
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        workspace,
        refresh_schedule=refresh_schedule,
        paired_head_finalize=use_c4_paired_fast or use_general_c2_quad,
        direct_fast_path=True,
        quad_head_two_chunk_finalize=use_c2_exact_two,
        fast_finalizer_handoff=use_fast_handoff,
        rank0_only_finalizer=use_rank0,
        skip_trailing_finalizer_barrier=use_c4_paired_fast or use_c2_exact_two or use_general_c2_quad,
        tma_stages=tma_stages,
        compiler_num_stages=compiler_num_stages,
        maxnreg=selected_maxnreg,
        prefetch_k_scale=prefetch_k_scale,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        fragment_k_scale_load=fragment_k_scale_load,
        precombine_q_scale=precombine_q_scale,
        defer_v_scale_final=defer_v_scale_final,
        c2_aligned_full_chunk_winner=aligned_full_chunk_winner,
        quant_type=quant_type,
        flatten_output=flatten_output,
    )


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_c2_bf16_dsm_specialized(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    deferred_norm: bool = False,
    full_view_v: bool = False,
    compiler_num_stages: int = 3,
    maxnreg: int | None = None,
    prefetch_k_scale: bool = False,
    tma_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool = False,
    aligned_full_chunk_winner: bool = False,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """Migrate the C4 BF16/raw DSM producer path to exact-two C2."""
    if workspace.config.cluster_size != 2 or int(workspace.stats.get("effective_chunks_max", 1)) != 2:
        raise ValueError("C2 BF16 DSM requires an exact-two-group schedule")
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        workspace,
        paired_head_finalize=False,
        direct_fast_path=True,
        quad_head_two_chunk_finalize=True,
        rank0_only_finalizer=True,
        skip_trailing_finalizer_barrier=True,
        c4_bf16_dsm=True,
        c4_deferred_norm=deferred_norm,
        full_view_v_rs=full_view_v,
        compiler_num_stages=compiler_num_stages,
        maxnreg=maxnreg,
        prefetch_k_scale=prefetch_k_scale,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        c2_aligned_full_chunk_winner=aligned_full_chunk_winner,
        quant_type=quant_type,
        flatten_output=flatten_output,
    )


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    use_pdl: bool = True,
    compiler_num_stages: int = 3,
    maxnreg: int | None = None,
    tma_k_scale: bool = False,
    packed_k_scale_lookahead: bool = False,
    shared_k_scale_pipeline: bool = False,
    wgmma_shadow_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool = False,
    reduction_only_raw: bool = False,
    aligned_full_chunk_raw: bool = False,
    exact_two_reducer: bool = False,
    sharded_reducer: bool = False,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """C2/C4 BF16-DSM raw producer plus a CUDA-shaped PDL finalizer."""
    ws = workspace
    if ws.config.cluster_size not in (2, 4):
        raise ValueError("detached raw finalizer requires C2 or C4")
    if reduction_only_raw and ws.config not in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024):
        raise ValueError("detached reduction-only raw producer requires C4T512/C4T1024")
    if aligned_full_chunk_raw:
        if ws.config == _runtime_mtp4__C4_T512 and (not reduction_only_raw):
            raise ValueError("C4T512 aligned producer requires reduction-only raw")
        if ws.config == _runtime_mtp4__C4_T1024 and reduction_only_raw:
            raise ValueError("C4T1024 aligned producer must retain single-group finalization")
    _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        ws,
        execution_stage="cluster",
        paired_head_finalize=True,
        c4_bf16_dsm=not reduction_only_raw,
        c4_deferred_norm=not reduction_only_raw,
        c4_global_deferred_norm=not reduction_only_raw,
        c4_reduction_only_raw=reduction_only_raw,
        c4_aligned_full_chunk_raw=aligned_full_chunk_raw,
        full_view_v_rs=True,
        pdl_notify=use_pdl,
        compiler_num_stages=compiler_num_stages,
        maxnreg=240 if maxnreg is None else maxnreg,
        tma_k_scale=tma_k_scale,
        packed_k_scale_lookahead=packed_k_scale_lookahead,
        shared_k_scale_pipeline=shared_k_scale_pipeline,
        wgmma_shadow_k_scale=wgmma_shadow_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        quant_type=quant_type,
        flatten_output=False,
    )
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    head_passes = (ws.heads_per_group + 3) // 4
    grid = inputs.num_batch * num_head_kv * _runtime_mtp4__NUM_SEQ_Q * head_passes
    reducer_kernel = (
        _finalize_mtp4__fp8_kvpertensor_decode_mtp4_sharded_raw_finalize_kernel
        if sharded_reducer
        else _finalize_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize_kernel
    )
    reducer_kwargs = dict(
        B=inputs.num_batch,
        H_Q=num_head_q,
        H_KV=num_head_kv,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=_runtime_mtp4__HEAD_DIM,
        CHUNK_TOKENS=ws.config.chunk_tokens,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.schedule.partial_slots),
        HEAD_PASSES=head_passes,
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        V_PER_HEAD=quant_type == 0,
        PDL_WAIT=use_pdl,
        num_warps=4,
        num_stages=1,
        launch_pdl=use_pdl,
    )
    if sharded_reducer:
        reducer_kernel[grid,](
            inputs.kv_lens,
            ws.split_out,
            ws.lse,
            inputs.v_scale,
            ws.out,
            mesh=_runtime_mtp4__CLUSTER_MESHES[4],
            PRODUCER_CLUSTER_SIZE=ws.config.cluster_size,
            REDUCE_CLUSTER_SIZE=4,
            **reducer_kwargs,
        )
    else:
        reducer_kernel[grid,](
            inputs.kv_lens,
            ws.split_out,
            ws.lse,
            inputs.v_scale,
            ws.out,
            CLUSTER_SIZE=ws.config.cluster_size,
            EXACT_TWO_SPECIALIZATION=exact_two_reducer,
            **reducer_kwargs,
        )
    if flatten_output:
        return ws.out.reshape(inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime_mtp4__HEAD_DIM)
    return ws.out


def _runtime_mtp4___resolve_mtp4_final_policy(workspace: _runtime_mtp4__DecodeWorkspace) -> str:
    """Resolve the validated backend from config and scheduler metadata."""
    config = workspace.config
    chunks = int(workspace.stats.get("effective_chunks_max", 1))
    if config.cluster_size == 2 and config.chunk_tokens == 1024 and (chunks == 2):
        return "c2-raw"
    if config.cluster_size == 4 and config.chunk_tokens == 512 and (chunks >= 32):
        return "pdl"
    if config.cluster_size == 4 and config.chunk_tokens == 1024:
        if chunks >= 16:
            return "pdl-s"
        if chunks >= 8:
            return "pdl"
    return "winner"


def _runtime_mtp4__prepare_mtp4_final_workspace(
    inputs: _runtime_mtp4__DecodeInputs, config: _runtime_mtp4__DecodeConfig, *, quant_type: int = 1
) -> _runtime_mtp4__DecodeWorkspace:
    """Prepare the workspace required by the fixed final MTP=4 policy.

    In particular, ``pdl-s`` needs a chunk-minor scalar workspace.  Keeping
    that detail here prevents callers and benchmarks from coupling themselves
    to an individual reducer's physical layout.
    """
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 or 1")
    workspace = _runtime_mtp4__prepare_decode_workspace(inputs, config)
    resolved = _runtime_mtp4___resolve_mtp4_final_policy(workspace)
    if resolved == "pdl-s":
        workspace = _runtime_mtp4__prepare_decode_workspace(inputs, config, raw_scalar_chunk_minor=True)
    workspace.final_policy_mode = resolved
    return workspace


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_final(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    quant_type: int = 1,
    flatten_output: bool = True,
) -> torch.Tensor:
    """Run the unified final MTP=4 interface.

    ``winner`` and ``c2-raw`` select compiled specializations of the fused TLE
    implementation.  ``pdl`` and ``pdl-s`` retain the two-launch raw producer
    plus detached PDL reducer design behind this single runtime API.
    """
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 or 1")
    resolved = workspace.final_policy_mode
    page_metadata_k_scale = quant_type == 0 and (
        workspace.config.cluster_size == 2
        and workspace.config.chunk_tokens in (256, 512)
        or (resolved == "pdl" and workspace.config == _runtime_mtp4__C4_T512 and (int(workspace.q_4d.shape[0]) == 8))
    )
    if page_metadata_k_scale and quant_type != 0:
        raise ValueError("page-metadata K-scale pipeline requires quant_type=0")
    precombine_static_scale = quant_type == 1 and (
        not (workspace.config.cluster_size == 4 and workspace.config.chunk_tokens == 128)
    )
    if resolved == "winner":
        winner_chunks = int(workspace.stats.get("effective_chunks_max", 1))
        aligned_c2 = (
            workspace.config in (_runtime_mtp4__C2_T256, _runtime_mtp4__C2_T1024)
            and workspace.all_chunks_aligned
            and (not int(workspace.stats.get("direct_tasks", 0)))
            and (not int(workspace.stats.get("dummy_tasks", 0)))
            and (not int(workspace.stats.get("subgroup2_tasks", 0)))
        )
        winner_tma_k_scale = (
            quant_type == 0
            and (not page_metadata_k_scale)
            and (workspace.config.cluster_size == 2)
            and (workspace.config.chunk_tokens == 1024)
            and (winner_chunks >= 32)
        )
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_winner(
            inputs,
            workspace,
            prefetch_k_scale=winner_tma_k_scale,
            tma_k_scale=winner_tma_k_scale,
            page_metadata_k_scale=page_metadata_k_scale,
            precombine_q_scale=winner_tma_k_scale or precombine_static_scale,
            aligned_full_chunk_winner=aligned_c2,
            quant_type=quant_type,
            flatten_output=flatten_output,
        )
    if resolved == "c2-raw":
        aligned_c2 = (
            workspace.config == _runtime_mtp4__C2_T1024
            and workspace.all_chunks_aligned
            and (not int(workspace.stats.get("direct_tasks", 0)))
            and (not int(workspace.stats.get("dummy_tasks", 0)))
            and (not int(workspace.stats.get("subgroup2_tasks", 0)))
        )
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_c2_bf16_dsm_specialized(
            inputs,
            workspace,
            deferred_norm=True,
            full_view_v=True,
            compiler_num_stages=3,
            tma_k_scale=quant_type == 0 and (not page_metadata_k_scale),
            page_metadata_k_scale=page_metadata_k_scale,
            precombine_q_scale=precombine_static_scale,
            aligned_full_chunk_winner=aligned_c2,
            quant_type=quant_type,
            flatten_output=flatten_output,
        )
    if resolved in ("pdl", "pdl-s"):
        producer_maxnreg = 192 if quant_type == 0 and resolved == "pdl" else 240
        aligned_c4 = (
            workspace.config in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024)
            and workspace.all_chunks_aligned
            and (not int(workspace.stats.get("direct_tasks", 0)))
            and (not int(workspace.stats.get("dummy_tasks", 0)))
            and (not workspace.config.subgroup2_threshold)
            and (
                workspace.config == _runtime_mtp4__C4_T512
                and resolved == "pdl"
                and (quant_type == 1 or (quant_type == 0 and int(workspace.q_4d.shape[0]) in (8, 16)))
                or (
                    quant_type in (0, 1)
                    and workspace.config == _runtime_mtp4__C4_T1024
                    and (resolved in ("pdl", "pdl-s"))
                )
            )
        )
        aligned_reduction_only = aligned_c4 and workspace.config == _runtime_mtp4__C4_T512
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize(
            inputs,
            workspace,
            use_pdl=True,
            compiler_num_stages=3,
            maxnreg=producer_maxnreg,
            reduction_only_raw=aligned_reduction_only,
            aligned_full_chunk_raw=aligned_c4,
            page_metadata_k_scale=page_metadata_k_scale,
            precombine_q_scale=aligned_c4 or precombine_static_scale,
            quant_type=quant_type,
            flatten_output=flatten_output,
        )
    raise ValueError(f"unsupported MTP=4 final policy: {resolved}")


def _dynamic_mtp2__fp8_kvpertensor_decode_mtp2_final(
    inputs: _runtime_mtp2__DecodeInputs,
    workspace: _runtime_mtp2__DecodeWorkspace | None = None,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
    bf16_dsm: bool | None = None,
    deferred_norm: bool = False,
    dsm_election_handoff: bool = False,
    deterministic_tail_election: bool = True,
    tail_only_election_barrier: bool = False,
    reduction_only: bool | None = None,
    aligned_full_chunk: bool = False,
    full_view_dsm: bool = False,
    full_view_v_rs: bool = False,
    quad_head_two_chunk_finalize: bool = False,
    rank0_only_finalizer: bool = False,
    skip_trailing_finalizer_barrier: bool | None = None,
    tma_k_scale: bool | None = None,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool | None = None,
    quant_type: int = 1,
) -> torch.Tensor:
    """Run typed-RS compute with reuse-w on c2/c8 and paired heads on c4."""
    if workspace is None:
        raise ValueError("MTP=2 requires an explicitly configured workspace")
    if tma_k_scale is None:
        tma_k_scale = (
            quant_type == 0
            and (not page_metadata_k_scale)
            and (not (workspace.config.cluster_size == 4 and workspace.config.chunk_tokens == 128))
        )
    if precombine_q_scale is None:
        precombine_q_scale = quant_type == 0 or (
            quant_type == 1 and workspace.config.cluster_size == 4 and (workspace.config.chunk_tokens == 512)
        )
    aligned_split_only = not any(
        (int(workspace.stats.get(name, 0)) for name in ("direct_tasks", "dummy_tasks", "subgroup2_tasks"))
    )
    auto_aligned_full_chunk = (
        quant_type in (0, 1)
        and (workspace.config.cluster_size, workspace.config.chunk_tokens)
        in ((2, 256), (2, 512), (2, 1024), (4, 256), (4, 512), (4, 1024))
        and workspace.all_chunks_aligned
        and aligned_split_only
    )
    use_c2_raw = (
        workspace.config.cluster_size == 2
        and workspace.config.chunk_tokens == 1024
        and (int(workspace.stats.get("effective_chunks_max", 1)) == 2)
        and (workspace.heads_per_group == 8)
        and (reduction_only is None)
        and (
            not any(
                (
                    bf16_dsm,
                    deferred_norm,
                    dsm_election_handoff,
                    tail_only_election_barrier,
                    full_view_dsm,
                    full_view_v_rs,
                    quad_head_two_chunk_finalize,
                    rank0_only_finalizer,
                )
            )
        )
        and (skip_trailing_finalizer_barrier is not False)
    )
    if use_c2_raw:
        return _dynamic_mtp2__fp8_kvpertensor_decode_mtp2_c2_raw_specialized(
            inputs,
            workspace,
            reduction_only=aligned_split_only,
            aligned_full_chunk=aligned_full_chunk or auto_aligned_full_chunk,
            tma_k_scale=tma_k_scale,
            page_metadata_k_scale=page_metadata_k_scale,
            precombine_q_scale=precombine_q_scale,
            quant_type=quant_type,
        )
    if reduction_only is None:
        reduction_only = (
            workspace.config.cluster_size == 2 and aligned_split_only or auto_aligned_full_chunk or aligned_full_chunk
        )
        aligned_full_chunk = aligned_full_chunk or auto_aligned_full_chunk
    if bf16_dsm is None:
        bf16_dsm = workspace.config.chunk_tokens <= 512
    if skip_trailing_finalizer_barrier is None:
        skip_trailing_finalizer_barrier = True
    return _runtime_mtp2__fp8_kvpertensor_decode_mtp2_final(
        inputs,
        workspace,
        refresh_schedule=refresh_schedule,
        execution_stage="full",
        cluster_cooperative_finalize=True,
        reuse_final_weights=True,
        paired_head_finalize=workspace.config.cluster_size == 4,
        bf16_dsm=bf16_dsm,
        deferred_norm=deferred_norm,
        dsm_election_handoff=dsm_election_handoff,
        deterministic_tail_election=deterministic_tail_election,
        tail_only_election_barrier=tail_only_election_barrier,
        reduction_only=reduction_only,
        aligned_full_chunk=aligned_full_chunk,
        full_view_dsm=full_view_dsm,
        full_view_v_rs=full_view_v_rs,
        quad_head_two_chunk_finalize=quad_head_two_chunk_finalize,
        rank0_only_finalizer=rank0_only_finalizer,
        skip_trailing_finalizer_barrier=skip_trailing_finalizer_barrier,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        quant_type=quant_type,
    )


def _dynamic_mtp2__fp8_kvpertensor_decode_mtp2_c2_raw_specialized(
    inputs: _runtime_mtp2__DecodeInputs,
    workspace: _runtime_mtp2__DecodeWorkspace,
    *,
    reduction_only: bool = False,
    aligned_full_chunk: bool = False,
    tma_k_scale: bool = False,
    page_metadata_k_scale: bool = False,
    precombine_q_scale: bool = False,
    quant_type: int = 1,
) -> torch.Tensor:
    """Exact MTP4-style C2 BF16/raw path for an isolated MTP2 focused."""
    return _runtime_mtp2__fp8_kvpertensor_decode_mtp2_final(
        inputs,
        workspace,
        execution_stage="full",
        cluster_cooperative_finalize=True,
        reuse_final_weights=False,
        bf16_dsm=True,
        deferred_norm=True,
        full_view_v_rs=True,
        quad_head_two_chunk_finalize=True,
        rank0_only_finalizer=True,
        skip_trailing_finalizer_barrier=True,
        reduction_only=reduction_only,
        aligned_full_chunk=aligned_full_chunk,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        quant_type=quant_type,
    )


_fp8_entry__HEAD_DIM = 128

_fp8_entry__BLOCK_SIZE = 64

_fp8_entry__SUPPORTED_MTP = (1, 2, 4)

_fp8_entry__Schedule = Literal["static", "dynamic"]

_fp8_entry__QuantType = Literal["qkpertoken_perhead_vperhead", "qpertoken_perhead_kvpertensor"]

_fp8_entry__QUANT_TYPES = {"qkpertoken_perhead_vperhead": 0, "qpertoken_perhead_kvpertensor": 1}

_fp8_entry__OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}


_fp8_entry___DYNAMIC_CT = {
    1: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (8, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 256),
        "one_64k_15x4k": (4, 256),
        "one_64k_31x4k": (8, 512),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (4, 512),
    },
    2: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (2, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 256),
        "one_64k_15x4k": (4, 512),
        "one_64k_31x4k": (4, 512),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (4, 512),
    },
    4: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (2, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 512),
        "one_64k_15x4k": (4, 512),
        "one_64k_31x4k": (4, 1024),
        "one_128k_31x4k": (2, 1024),
        "two_32k_30x4k": (4, 1024),
    },
}


@dataclass
class _fp8_entry__FP8DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


@dataclass(frozen=True)
class _fp8_entry__FP8DecodePolicy:
    schedule: _fp8_entry__Schedule
    quant_type: _fp8_entry__QuantType
    mtp: int
    case: str
    layout: str
    cluster_size: int
    chunk_tokens: int
    route: str

    @property
    def label(self) -> str:
        return f"c{self.cluster_size}t{self.chunk_tokens}/{self.route}"


@dataclass
class _fp8_entry__FP8DecodeWorkspace:
    policy: _fp8_entry__FP8DecodePolicy
    runtime_inputs: object
    runtime_workspace: object


def _fp8_entry___layout(cache: torch.Tensor) -> str:
    return "HND" if cache.stride(2) > cache.stride(1) else "NHD"


def _fp8_entry___validate(inputs: _fp8_entry__FP8DecodeInputs) -> None:
    if inputs.mtp not in _fp8_entry__SUPPORTED_MTP:
        raise ValueError("final FP8 decode supports MTP 1, 2, or 4")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _fp8_entry__HEAD_DIM:
        raise ValueError("q must have shape [batch * MTP,Hq,128]")
    if inputs.q.dtype != torch.float8_e4m3fn:
        raise ValueError("q must be float8_e4m3fn")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.element_size() != 1:
            raise ValueError(f"{name} must use a 1-byte FP8 storage dtype")
        if cache.ndim != 4 or cache.shape[1] != _fp8_entry__BLOCK_SIZE:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.shape[-1] != _fp8_entry__HEAD_DIM or cache.stride(-1) != 1:
            raise ValueError(f"{name} head dimension must be contiguous 128")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if (num_head_kv, num_head_q) not in ((1, 8), (4, 32)):
        raise ValueError("final FP8 decode requires official GQA8 heads (Hkv,Hq)=(1,8) or (4,32)")
    if inputs.block_ids.dtype != torch.int32 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError("block_ids and kv_lens must be int32")


def _fp8_entry___classify_workload(lengths: tuple[int, ...]) -> str | None:
    count = len(lengths)
    if count == 64 and all(length == 512 for length in lengths):
        return "uniform_512"
    if count == 64 and all(length == 4096 for length in lengths):
        return "uniform_4096"
    if count == 64 and lengths.count(128) == 32 and lengths.count(4096) == 32:
        return "skewed_mix"
    if count == 16 and lengths.count(64) == 15 and lengths.count(16384) == 1:
        return "skewed_extreme"
    if lengths.count(65536) == 1 and lengths.count(4096) == count - 1:
        if count == 8:
            return "one_64k_7x4k"
        if count == 16:
            return "one_64k_15x4k"
        if count == 32:
            return "one_64k_31x4k"
    if count == 32 and lengths.count(131072) == 1 and lengths.count(4096) == 31:
        return "one_128k_31x4k"
    if count == 32 and lengths.count(32768) == 2 and lengths.count(4096) == 30:
        return "two_32k_30x4k"
    return None


def _fp8_entry___case(inputs: _fp8_entry__FP8DecodeInputs, lengths: tuple[int, ...] | None = None) -> str:
    if lengths is None:
        lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    workload = _fp8_entry___classify_workload(lengths)
    if workload is not None:
        return workload
    if not lengths or min(lengths) < inputs.mtp:
        raise ValueError("each final KV length must be at least MTP")
    return "uniform_512" if max(lengths) <= 1024 else "uniform_4096"


def _fp8_entry__select_fp8_decode_policy(inputs: _fp8_entry__FP8DecodeInputs) -> _fp8_entry__FP8DecodePolicy:
    _fp8_entry___validate(inputs)
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    case = _fp8_entry___case(inputs, lengths)
    layout = _fp8_entry___layout(inputs.k_cache)
    cluster, tokens = _fp8_entry___DYNAMIC_CT[inputs.mtp][case]
    route = "gpu-task-map"
    hnd_single_long_tail = (
        inputs.mtp == 2
        and layout == "HND"
        and len(lengths) == 16
        and lengths.count(65536) == 1
        and lengths.count(4096) == 15
    )
    if hnd_single_long_tail:
        cluster, tokens = (4, 256)
        route = "gpu-task-map-deferred"
    return _fp8_entry__FP8DecodePolicy(
        "dynamic", "qpertoken_perhead_kvpertensor", inputs.mtp, case, layout, cluster, tokens, route
    )


def _fp8_entry___runtime_inputs(inputs: _fp8_entry__FP8DecodeInputs, mtp: int):
    input_type = {1: _runtime_mtp1__DecodeInputs, 2: _runtime_mtp2__DecodeInputs, 4: _runtime_mtp4__DecodeInputs}[mtp]
    return input_type(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        inputs.q_scale,
        inputs.k_scale,
        inputs.v_scale,
    )


def _fp8_entry__prepare_fp8_decode_workspace(inputs: _fp8_entry__FP8DecodeInputs) -> _fp8_entry__FP8DecodeWorkspace:
    policy = _fp8_entry__select_fp8_decode_policy(inputs)
    runtime_inputs = _fp8_entry___runtime_inputs(inputs, policy.mtp)
    config_type = {1: _runtime_mtp1__DecodeConfig, 2: _runtime_mtp2__DecodeConfig, 4: _runtime_mtp4__DecodeConfig}[
        policy.mtp
    ]
    config = config_type(policy.cluster_size, policy.chunk_tokens)
    if policy.mtp == 4:
        runtime_workspace = _runtime_mtp4__prepare_mtp4_final_workspace(runtime_inputs, config, quant_type=1)
    elif policy.mtp == 2:
        runtime_workspace = _runtime_mtp2__prepare_decode_workspace(runtime_inputs, config)
    else:
        runtime_workspace = _runtime_mtp1__prepare_decode_workspace(runtime_inputs, config)
    return _fp8_entry__FP8DecodeWorkspace(policy, runtime_inputs, runtime_workspace)


def _fp8_entry___dynamic_mtp2_options(policy: _fp8_entry__FP8DecodePolicy) -> dict[str, object]:
    if policy.route == "gpu-task-map-deferred":
        return dict(bf16_dsm=True, deferred_norm=True)
    if policy.case == "one_64k_15x4k" and _fp8_entry__QUANT_TYPES[policy.quant_type] == 0:
        return dict(full_view_v_rs=True, bf16_dsm=True, deferred_norm=True)
    if policy.case == "one_64k_31x4k":
        return dict(full_view_v_rs=True)
    return {}


def _fp8_entry___dynamic_mtp1_options(policy: _fp8_entry__FP8DecodePolicy) -> dict[str, object]:
    quant_id = _fp8_entry__QUANT_TYPES[policy.quant_type]
    key = (policy.case, policy.layout, quant_id)
    tail = {
        ("skewed_mix", "NHD", 0),
        ("skewed_mix", "HND", 0),
        ("one_64k_15x4k", "NHD", 0),
        ("one_64k_15x4k", "NHD", 1),
        ("two_32k_30x4k", "HND", 0),
    }
    if key in tail:
        return dict(full_view_v_rs=True, deterministic_tail_election=True, dsm_election_handoff=False)
    if key == ("skewed_mix", "NHD", 1):
        return dict(full_view_v_rs=True, deterministic_tail_election=False, dsm_election_handoff=True)
    return {}


def _fp8_entry__attention_decode_fp8_tle(
    inputs: _fp8_entry__FP8DecodeInputs, workspace: _fp8_entry__FP8DecodeWorkspace
) -> torch.Tensor:
    policy = workspace.policy
    _fp8_entry___validate(inputs)
    if policy.mtp != inputs.mtp or policy.layout != _fp8_entry___layout(inputs.k_cache):
        raise ValueError("workspace policy does not match inputs")
    runtime_inputs = workspace.runtime_inputs
    runtime_workspace = workspace.runtime_workspace
    if policy.mtp == 1:
        return _runtime_mtp1__fp8_kvpertensor_decode_mtp1_final(
            runtime_inputs, runtime_workspace, quant_type=1, **_fp8_entry___dynamic_mtp1_options(policy)
        )
    if policy.mtp == 2:
        return _dynamic_mtp2__fp8_kvpertensor_decode_mtp2_final(
            runtime_inputs, runtime_workspace, quant_type=1, **_fp8_entry___dynamic_mtp2_options(policy)
        )
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_final(runtime_inputs, runtime_workspace, quant_type=1)


def _fp8_entry__fp8_workspace_is_reset(workspace: _fp8_entry__FP8DecodeWorkspace) -> bool:
    completion = getattr(workspace.runtime_workspace, "completion", None)
    return completion is None or not bool(torch.count_nonzero(completion).item())


QUANT_TYPE = "qpertoken_perhead_kvpertensor"
QUANT_TYPE_ID = 1
BLOCK_SIZE = _fp8_entry__BLOCK_SIZE
HEAD_DIM = _fp8_entry__HEAD_DIM
SUPPORTED_MTP = _fp8_entry__SUPPORTED_MTP
QUANT_TYPES = tuple(_fp8_entry__QUANT_TYPES)
OFFICIAL_CASES = _fp8_entry__OFFICIAL_CASES
FP8DecodeInputs = _fp8_entry__FP8DecodeInputs
FP8DecodePolicy = _fp8_entry__FP8DecodePolicy
FP8DecodeWorkspace = _fp8_entry__FP8DecodeWorkspace


def select_decode_policy(inputs):
    return _fp8_entry__select_fp8_decode_policy(inputs)


def prepare_decode_workspace(inputs):
    if not USE_TLE:
        if inputs.mtp == 1:
            return prepare_pure_triton_mtp1_workspace(inputs, QUANT_TYPE)
        return prepare_pure_triton_mtp_workspace(inputs, QUANT_TYPE)
    return _fp8_entry__prepare_fp8_decode_workspace(inputs)


def attention_decode_fp8(inputs, workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    return _fp8_entry__attention_decode_fp8_tle(inputs, workspace)


def workspace_is_reset(workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return True
    return _fp8_entry__fp8_workspace_is_reset(workspace)


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "QUANT_TYPE",
    "QUANT_TYPE_ID",
    "QUANT_TYPES",
    "SUPPORTED_MTP",
    "FP8DecodeInputs",
    "FP8DecodePolicy",
    "FP8DecodeWorkspace",
    "attention_decode_fp8",
    "prepare_decode_workspace",
    "select_decode_policy",
    "workspace_is_reset",
]
