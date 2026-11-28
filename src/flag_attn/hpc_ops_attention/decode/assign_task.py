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

import torch
import triton
import triton.language as tl

# Portable split-K task-map format.

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
