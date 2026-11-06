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

"""Ascend Triton implementation of the two HY3 FP8 decode formats.

Ascend 910B does not expose E4M3 tensors through torch-npu.  Q/K/V therefore
use uint8 (or int8) tensors containing the IEEE/OCP E4M3FN bit patterns.  The
producer decodes only the tiles it consumes; it never expands the full paged
cache to BF16.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
import triton.experimental.tle as tle  # noqa: F401
from triton.experimental.tle.language.dsa import (
    tile_alloc,
    tile_copy,
    tile_to_tensor,
)

from .launch import launch_kernel

from .mtp_reduce import (
    _bf16_mtp_finalize_groups_kernel,
    _bf16_mtp_finalize_kernel,
    _bf16_mtp_reduce_splits_kernel,
)

from ..e4m3 import (
    e4m3_bits16,
    e4m3_exact,
)


BLOCK_SIZE = 64
HEAD_DIM = 128
SUPPORTED_MTP = (1, 2, 4)
# Pages of 64 KV tokens one producer program consumes, and the split sizes the
# workspace builder may pick between.
PAGES_PER_SPLIT = 8
PAGES_PER_SPLIT_CHOICES = (1, 2, 4, 8)
# FP8-side reduction group bound (BF16-shared SPLITS_PER_REDUCTION=16 untouched): one
# group = one reduce kernel call; beyond this count a finalize kernel is added. Taking
# 32 gives shapes with splits<=32 one fewer kernel launch -- per-launch host/launch cost
# ~0.1ms (eager wall, true after replay), big for 0.4ms shapes.
FP8_SPLITS_PER_REDUCTION = 64
COMPACT_WASTE_RATIO = 1.25
# A dense batch/head/split grid is only worth widening when it can keep the AIV
# cores busy; two waves is where the measurement flattens.
PRODUCER_WAVES_TARGET = 2
# Fallback when the runtime cannot report the vector core count.
PRODUCER_VECTOR_CORES_FALLBACK = 40
QPERTOKEN_KVPERTENSOR = 1
QKPERTOKEN_VPERHEAD = 0
_BLOCK_SIZE_JIT = tl.constexpr(BLOCK_SIZE)
_SUB_N = 32
_SUB_N_JIT = tl.constexpr(_SUB_N)
# producer side: bf16 scratch frees UB from arithmetic-decode temporaries: a page per
# iteration; the fp8 path falls back to 32 ((64,128) overflows UB, see scratch64).
_SCRATCH_SUB_N = 64
# Max Q rows the wide producer can hold (measured UB: Q_ROWS=32 needs 377KB/192KB).
_WIDE_MAX_Q_ROWS = 8
# Standalone decode wins only when the producer's real page-iteration count is big: it
# swaps vec-pipe-serialized arithmetic decode for one streaming read/write, at one fixed
# kernel launch (~0.12ms) + 4B/element extra bf16 write/read.  Use **padded page slot
# count** (grid programs x pages_per_split), not referenced pages: fused programs loop
# PAGES_PER_SPLIT times (non-full split clamps and repeats last page), so cost tracks
# programs x pages_per_split; scratch decodes each page once.  Equal-element verdicts
# invert (same process/card, forced, B=64/8x1/NHD/mtp1/qk):
#   ref M / padded M / fused / scratch / verdict
#   4.194 / 4.194  512 slots   0.413   0.496   fused  +20%
#   4.456 / 4.456  544 slots   0.421   0.480   fused  +14%
#   4.719 / 4.719  576 slots   0.457   0.488   fused   +6.7%
#   5.243 / 5.243  640 slots   0.452   0.463   fused   +2.3%   <- break-even
#   6.291 / 6.291  768 slots   0.527   0.453   scratch -14.0%
#   7.340 / 7.340  896 slots   0.618   0.542   scratch -12.3%
#   8.389 / 8.389 1024 slots   0.695   0.490   scratch -29.6%
#   4.719 / 8.389 1024 slots   0.664   0.484   scratch -27.1% <- equal elems, diff pad
#   3.080 / 3.080  376 slots   0.395   0.529   fused  +34%    (skewed_extreme, compact)
#  18.9   / 18.9  2304 slots   1.301   0.793   scratch -39%   (skewed_mix)
# Padded-element crossover 5.24M..6.29M; take the 5.5Mi midpoint.  Rule: "padded page
# iters from page table/schedule": one expression, any shape, no shape special case.
_SCRATCH_MIN_K_ELEMS = 5_767_168  # 5.5 * 1024 * 1024, padded K elements
# Standalone decode: (page,hkv) slots per program.  More slots = fewer programs, more
# independent DMA, amortised setup; measured best 8 (16 -> +3.0%, 32 -> +9.8%).
_DECODE_SLOTS = 8
_HEAD_DIM_JIT = tl.constexpr(HEAD_DIM)
_QPERTOKEN_KVPERTENSOR_JIT = tl.constexpr(QPERTOKEN_KVPERTENSOR)
_QKPERTOKEN_VPERHEAD_JIT = tl.constexpr(QKPERTOKEN_VPERHEAD)


@dataclass
class FP8DecodeInputs:
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

    @property
    def layout(self) -> str:
        if self.k_cache.shape[2] == 1:
            return "NHD"
        return "HND" if self.k_cache.stride(2) > self.k_cache.stride(1) else "NHD"


@dataclass
class FP8DecodeWorkspace:
    producer_task_map: torch.Tensor
    reduce_task_map: torch.Tensor
    final_task_map: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    reduced_out: torch.Tensor
    reduced_lse: torch.Tensor
    out: torch.Tensor
    fp8_lut: torch.Tensor
    k_bf16: torch.Tensor
    v_bf16: torch.Tensor
    phys_slots: torch.Tensor
    # Compact slot table: page_base[b] is the slot index of batch b's first referenced
    # page in k_bf16/v_bf16 (exclusive prefix sum); padding entries are INT_MAX so one
    # in-kernel vector compare resolves a slot's batch.  total_pages (total referenced
    # pages) is the decode kernel grid; pool is the upper bound (block_ids range).
    page_base: torch.Tensor
    total_pages: int
    pool_pages: int
    nb_p2: int
    use_scratch: bool
    max_splits: int
    max_reduction_groups: int
    num_producer_tasks: int
    num_reduce_tasks: int
    num_final_tasks: int
    compact_producer: bool
    hierarchical_reduction: bool
    full_producer_splits: bool
    mtp: int
    pages_per_split: int
    quant_type: int

    def is_reset(self) -> bool:
        """No completion counters persist between the separate kernel launches."""
        return True


def _validate_fp8_layout(inputs: FP8DecodeInputs, quant_type: int) -> tuple[int, int, int]:
    """Host-only structural checks for one FP8 decode call.

    Deliberately free of any device synchronisation: the per-call producer and
    finalize launches must stay capturable (an ``.item()`` in the hot path
    drains the device queue and forbids NPUGraph capture).  The one check that
    needs device data, ``kv_lens.min() >= mtp``, lives in
    ``validate_fp8_inputs`` and is therefore evaluated exactly where the
    schedule is built from ``kv_lens`` (``prepare_fp8_workspace`` and
    ``refresh_fp8_task_map``), not on every producer launch.
    """
    if quant_type not in (QPERTOKEN_KVPERTENSOR, QKPERTOKEN_VPERHEAD):
        raise ValueError("unknown HY3 FP8 decode quantization type")
    if inputs.mtp not in SUPPORTED_MTP:
        raise NotImplementedError("Ascend FP8 decode supports MTP=1, MTP=2, or MTP=4")
    if inputs.q.device.type != "npu":
        raise ValueError("Ascend FP8 decode requires NPU tensors")
    tensors = (
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        inputs.q_scale,
        inputs.k_scale,
        inputs.v_scale,
    )
    if any(t.device != inputs.q.device for t in tensors):
        raise ValueError("all inputs must be on the same NPU device")
    if inputs.q.dtype not in (torch.uint8, torch.int8):
        raise ValueError("q must contain E4M3FN bits in uint8/int8 storage")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM or inputs.q.stride(-1) != 1:
        raise ValueError("q must have contiguous shape [batch * MTP,Hq,128]")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.dtype not in (torch.uint8, torch.int8):
            raise ValueError(f"{name} must contain E4M3FN bits in uint8/int8 storage")
        if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE or cache.shape[-1] != HEAD_DIM:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(-1) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.k_cache.shape != inputs.v_cache.shape:
        raise ValueError("K and V cache logical shapes must match")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32 or inputs.kv_lens.ndim != 1:
        raise ValueError("kv_lens must be rank-1 int32")
    if inputs.q_scale.dtype != torch.float32 or inputs.q_scale.numel() != inputs.q.shape[0] * inputs.q.shape[1]:
        raise ValueError("q_scale must be FP32 with one value per query token/head")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("Ascend FP8 decode requires official GQA8 heads")
    if quant_type == QPERTOKEN_KVPERTENSOR:
        if inputs.k_scale.dtype != torch.float32 or inputs.k_scale.numel() != 1:
            raise ValueError("KV-per-tensor k_scale must be one FP32 value")
        if inputs.v_scale.dtype != torch.float32 or inputs.v_scale.numel() != 1:
            raise ValueError("KV-per-tensor v_scale must be one FP32 value")
    else:
        if inputs.k_scale.dtype not in (torch.uint8, torch.int8):
            raise ValueError("per-token k_scale must contain packed FP32 bytes")
        if inputs.k_scale.ndim != 4 or inputs.k_scale.shape[1] != 2:
            raise ValueError("packed k_scale must have logical shape [block,2,Hkv,128]")
        if inputs.v_scale.dtype != torch.float32 or inputs.v_scale.numel() != hkv:
            raise ValueError("per-head v_scale must contain one FP32 value per KV head")
    return inputs.mtp, hq, hkv


def validate_fp8_inputs(inputs: FP8DecodeInputs, quant_type: int) -> tuple[int, int, int]:
    """Full validation, including the device-side ``kv_lens`` domain check."""
    mtp, hq, hkv = _validate_fp8_layout(inputs, quant_type)
    if int(inputs.kv_lens.min().item()) < mtp:
        raise ValueError("each final KV length must be at least MTP")
    return mtp, hq, hkv


def _records_tensor(records: list[list[int]], width: int, device: torch.device) -> torch.Tensor:
    if not records:
        records = [[0] * width]
    return torch.tensor(records, dtype=torch.int32, device=device)


def _build_schedule(inputs: FP8DecodeInputs, pages_per_split: int) -> dict[str, object]:
    lengths = inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    tokens_per_split = pages_per_split * BLOCK_SIZE
    split_counts = [triton.cdiv(length, tokens_per_split) for length in lengths]
    max_splits = max(split_counts)
    group_counts = [triton.cdiv(count, FP8_SPLITS_PER_REDUCTION) for count in split_counts]
    producer = [
        [batch, h, split]
        for batch, count in enumerate(split_counts)
        for split in range(count)
        for h in range(hkv)
    ]
    reduce = [
        [batch, h, group]
        for batch, count in enumerate(group_counts)
        for group in range(count)
        for h in range(hq)
    ]
    final = [
        [batch, h]
        for batch, count in enumerate(group_counts)
        if count > 1
        for h in range(hq)
    ]
    device = inputs.q.device
    return {
        "producer_task_map": _records_tensor(producer, 3, device),
        "reduce_task_map": _records_tensor(reduce, 3, device),
        "final_task_map": _records_tensor(final, 2, device),
        "max_splits": max_splits,
        "max_reduction_groups": max(group_counts),
        "num_producer_tasks": len(producer),
        "num_reduce_tasks": len(reduce),
        "num_final_tasks": len(final),
        "compact_producer": any(count != max_splits for count in split_counts),
        "hierarchical_reduction": max_splits > FP8_SPLITS_PER_REDUCTION,
        "full_producer_splits": all(length % tokens_per_split == 0 for length in lengths),
    }


def _vector_cores(device: torch.device) -> int:
    try:
        return int(torch.npu.get_device_properties(device).vector_core_num)
    except Exception:  # pragma: no cover - older runtime without the property
        return PRODUCER_VECTOR_CORES_FALLBACK


def _split_programs(batch: int, hkv: int, lengths: list[int], pages_per_split: int) -> int:
    """Programs the dense ``(batch, hkv, max_splits)`` producer grid launches."""
    tokens_per_split = pages_per_split * BLOCK_SIZE
    return batch * hkv * max(
        triton.cdiv(length, tokens_per_split) for length in lengths
    )


def _choose_pages_per_split(inputs: FP8DecodeInputs) -> int:
    """Pick the producer's pages per split from the parallelism it can reach.

    The widest split is the fastest as soon as its grid can fill the AIV cores:
    finer splits only add a per-program Q-tile decode and epilogue, which on
    Ascend910B4 (8q/1kv, NHD, static) costs +28% on uniform_4096 and +173% on
    one_64k_7x4k when going from eight pages to one, for *both* FP8 formats.
    A workload whose whole grid fits in a single wave is starved instead, and
    one step finer is worth ~8% (uniform_512: four pages 500us against eight
    pages 542us).  The choice therefore only depends on the available
    parallelism, and it is fixed when the workspace is built so that a growing
    KV length cannot re-shard an existing workspace.
    """
    lengths = inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    if not lengths:
        return PAGES_PER_SPLIT
    batch = len(lengths)
    hkv = int(inputs.k_cache.shape[2])
    hq = int(inputs.q.shape[1])
    q_rows = inputs.mtp * (hq // hkv)
    # A sixteen-page wide split only fits in UB when the Q tile is small
    # (Q_ROWS <= 8) and it needs the whole split to be full, because the
    # BF16-isomorphic producer reads V as one affine block of consecutive
    # compact slots.  Outside that regime the choice stays at eight.
    pages = [(length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in lengths]
    scratch_eligible = (
        sum(pages) <= int(inputs.k_cache.shape[0])
        and sum(pages) * BLOCK_SIZE * HEAD_DIM * hkv >= _SCRATCH_MIN_K_ELEMS
    )
    choices = PAGES_PER_SPLIT_CHOICES
    if (
        q_rows <= 8
        and scratch_eligible
        and all(length % (16 * BLOCK_SIZE) == 0 for length in lengths)
    ):
        choices = choices + (16,)
    target = PRODUCER_WAVES_TARGET * _vector_cores(inputs.q.device)
    for candidate in reversed(choices):
        if _split_programs(batch, hkv, lengths, candidate) >= target:
            return candidate
    return PAGES_PER_SPLIT_CHOICES[0]


def _scratch_shape(cache: torch.Tensor, pages_per_split: int) -> torch.Size:
    """Pool shape plus one split of read-ahead padding.

    The wide producer consumes a whole ``pages_per_split``-page block of the
    compact scratch per chunk, so a tail split reads up to ``pages_per_split``
    pages past its last referenced slot.  The decoded region ends at
    ``total_pages``, which can sit arbitrarily close to the pool bound, so the
    padding is what keeps that tile read inside the allocation.  The lanes that
    fall on padding are masked out of ``p`` and of the V load, so their content
    never reaches the result.
    """
    shape = list(cache.shape)
    shape[0] = int(shape[0]) + pages_per_split
    return torch.Size(shape)


def _e4m3fn_lut(device: torch.device) -> torch.Tensor:
    # torch CPU understands E4M3FN even though torch-npu cannot store that
    # dtype.  Preserve the exact byte interpretation and upload BF16 values.
    bits = torch.arange(256, dtype=torch.uint8)
    return bits.view(torch.float8_e4m3fn).float().to(torch.bfloat16).to(device)


def _page_plan(
    inputs: FP8DecodeInputs, hkv: int, padded_kv_elems: int
) -> tuple[torch.Tensor, int, int, int, bool]:
    """Referenced-page plan for the standalone decode pass.

    ``block_ids`` is a page table: batch ``b`` references exactly
    ``ceil(kv_lens[b] / 64)`` physical pages, and the physical pool
    (``k_cache.shape[0]``) is over-allocated relative to that (1.2x + batch + 8
    in the official benchmark).  Walking the page table therefore decodes
    strictly fewer pages than walking the pool, and the scratch buffer is
    indexed by a compact slot ``page_base[b] + logical_page`` so that the
    producer address is affine (no per-page gather).

    ``padded_kv_elems`` is the K element count the *fused* producer would
    actually walk (producer grid programs x pages_per_split x page x hkv), not
    the referenced-page count; the scratch path decodes each referenced page
    once, so only the fused side pays for the padding a clamped split repeats.

    Returns ``(page_base, phys_slots, total_pages, pool_pages, nb_p2, use_scratch)``.

    ``phys_slots`` is the compact slot -> physical page table, built here so the
    decode kernel no longer inverts the slot mapping per page (a vector compare
    over ``NB_P2`` lanes plus two dependent scalar loads, measured at 0.077ms of
    the 0.32ms pass).  It is only built for the scratch path.
    """
    device = inputs.q.device
    lengths = inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    batch = len(lengths)
    pages = [(length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in lengths]
    pool_pages = int(inputs.k_cache.shape[0])
    total_pages = int(sum(pages))
    nb_p2 = max(triton.next_power_of_2(max(batch, 1)), 1)
    base = [0] * nb_p2
    running = 0
    for index in range(batch):
        base[index] = running
        running += pages[index]
    for index in range(batch, nb_p2):
        base[index] = 0x7FFFFFFF
    page_base = torch.tensor(base, dtype=torch.int32, device=device)
    # A page beyond the pool would mean an invalid page table; fall back to the
    # fused producer rather than reading out of bounds.
    use_scratch = (
        total_pages <= pool_pages and padded_kv_elems >= _SCRATCH_MIN_K_ELEMS
    )
    phys_slots = torch.empty(0, dtype=torch.int32, device=device)
    if use_scratch and total_pages:
        slot_batch = torch.tensor(
            [b for b, count in enumerate(pages) for _ in range(count)],
            dtype=torch.int64,
            device=device,
        )
        logical = (
            torch.arange(total_pages, dtype=torch.int64, device=device)
            - page_base[slot_batch].to(torch.int64)
        )
        phys_slots = inputs.block_ids[slot_batch, logical].to(torch.int32)
    return page_base, phys_slots, total_pages, pool_pages, nb_p2, use_scratch


def prepare_fp8_workspace(
    inputs: FP8DecodeInputs,
    quant_type: int,
    *,
    static_sched: bool = False,
) -> FP8DecodeWorkspace:
    mtp, hq, hkv = validate_fp8_inputs(inputs, quant_type)
    # The BF16 MTP1 path may select sixteen pages, but an FP8 producer also
    # carries the decode LUT and scale state.  Sixteen pages used to hit an
    # AICore timeout; after the online-softmax state chain was dropped it
    # compiles and is no longer a hard bound, but the measurement is mixed
    # (64x4096 -2.6%, one_128k -2.5%, one_64k_7x4k -2.6%, two_32k +1.4%,
    # skewed_mix +5.4%), so the search still stops at eight and the split size
    # below that bound follows the producer grid's occupancy.
    pages_per_split = _choose_pages_per_split(inputs)
    schedule = _build_schedule(inputs, pages_per_split)
    batch = inputs.batch
    if static_sched:
        lengths = inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
        dense = _split_programs(batch, int(inputs.k_cache.shape[2]), lengths, pages_per_split)
        valid = int(schedule["num_producer_tasks"])
        schedule["compact_producer"] = bool(valid > 0 and dense >= COMPACT_WASTE_RATIO * valid)
    rows = mtp * hq
    max_splits = int(schedule["max_splits"])
    max_groups = int(schedule["max_reduction_groups"])
    grid_programs = (
        int(schedule["num_producer_tasks"])
        if schedule["compact_producer"]
        else batch * hkv * max_splits
    )
    padded_kv_elems = grid_programs * pages_per_split * BLOCK_SIZE * HEAD_DIM * hkv
    page_base, phys_slots, total_pages, pool_pages, nb_p2, use_scratch = _page_plan(
        inputs, hkv, padded_kv_elems
    )
    scratch_slots = torch.empty(
        pool_pages, dtype=torch.int32, device=inputs.q.device
    )
    if use_scratch and total_pages:
        scratch_slots[:total_pages].copy_(phys_slots)
    return FP8DecodeWorkspace(
        producer_task_map=schedule["producer_task_map"],
        reduce_task_map=schedule["reduce_task_map"],
        final_task_map=schedule["final_task_map"],
        split_out=torch.empty((batch, max_splits, rows, HEAD_DIM), dtype=torch.bfloat16, device=inputs.q.device),
        split_lse=torch.empty((batch, max_splits, rows), dtype=torch.float32, device=inputs.q.device),
        reduced_out=torch.empty((batch, max_groups, rows, HEAD_DIM), dtype=torch.bfloat16, device=inputs.q.device),
        reduced_lse=torch.empty((batch, max_groups, rows), dtype=torch.float32, device=inputs.q.device),
        out=torch.empty(inputs.q.shape, dtype=torch.bfloat16, device=inputs.q.device),
        fp8_lut=_e4m3fn_lut(inputs.q.device),
        # Per-element decode target: pool-shaped, NHD contiguous, read by the producer.
        # Sized to pool bound (referenced pages <= pool); only [0, total_pages) used.
        k_bf16=torch.empty(
            _scratch_shape(inputs.k_cache, pages_per_split),
            dtype=torch.bfloat16,
            device=inputs.q.device,
        ),
        v_bf16=torch.empty(
            _scratch_shape(inputs.v_cache, pages_per_split),
            dtype=torch.bfloat16,
            device=inputs.q.device,
        ),
        # Stable buffer: dynamic schedule rewrites in place, captured graph ptrs hold.
        phys_slots=scratch_slots,
        page_base=page_base,
        total_pages=total_pages,
        pool_pages=pool_pages,
        nb_p2=nb_p2,
        use_scratch=use_scratch,
        max_splits=max_splits,
        max_reduction_groups=max_groups,
        num_producer_tasks=int(schedule["num_producer_tasks"]),
        num_reduce_tasks=int(schedule["num_reduce_tasks"]),
        num_final_tasks=int(schedule["num_final_tasks"]),
        compact_producer=bool(schedule["compact_producer"]),
        hierarchical_reduction=bool(schedule["hierarchical_reduction"]),
        full_producer_splits=bool(schedule["full_producer_splits"]),
        mtp=mtp,
        pages_per_split=pages_per_split,
        quant_type=quant_type,
    )


@triton.jit
def _fp8_decode_kv_kernel(
    K,
    V,
    K_SCALE,
    V_SCALE,
    BLOCK_IDS,
    PHYS_SLOTS,
    PAGE_BASE,
    K_BF16,
    V_BF16,
    H_KV: tl.constexpr,
    NB_P2: tl.constexpr,
    SLOTS: tl.constexpr,
    TOTAL_SLOTS: tl.constexpr,
    BID_SB: tl.constexpr,
    K_SPAGE: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    V_SPAGE: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    KS_SPAGE: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
):
    """Decode one *referenced* physical KV page to BF16 in one streaming pass.

    The per-element arithmetic decode is pure elementwise work with no
    dependencies between elements, so it belongs in a wide grid of streaming
    programs.  Inside the producer it competes with the QK/PV dots and the
    softmax update for the same vector pipe while being wrapped in sixteen
    serial steps; here every element is decoded exactly once, with full
    overlap against the memory traffic.

    The grid is ``total_pages * H_KV``: one program per page-table entry, i.e.
    exactly the pages the page table references -- not the over-allocated
    physical pool.  ``slot`` is the compact index ``page_base[b] + logical_page``
    that the producer also uses, so the slot -> (batch, logical page) inverse is
    one vector compare against ``page_base``.
    """
    pid = tl.program_id(0)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    for s in tl.range(0, SLOTS):
        flat = pid * SLOTS + s
        if flat < TOTAL_SLOTS:
            slot = flat // H_KV
            hkv = flat - slot * H_KV
            # slot -> physical page is host-precomputed (_page_plan's phys_slots):
            # previous page-table inversion did a vector compare over NB_P2 lanes plus
            # two dependent scalar loads, measured at 0.077ms of this step.
            physical = tl.load(PHYS_SLOTS + slot).to(tl.int64)
            if QUANT_TYPE == _QKPERTOKEN_VPERHEAD_JIT:
                value_scale = tl.load(V_SCALE + hkv)
            else:
                value_scale = tl.load(V_SCALE)
            # Program constants hoisted from the slot loop: scalar base, one add/slot.
            k_phys = K + physical * K_SPAGE + hkv * K_SHEAD
            v_phys = V + physical * V_SPAGE + hkv * V_SHEAD
            for sub in tl.static_range(0, _BLOCK_SIZE_JIT // _SUB_N_JIT):
                sub_n = sub * _SUB_N_JIT + tl.arange(0, _SUB_N_JIT)
                off = sub_n[:, None] * KB_STOKEN + offs_d[None, :]
                k = e4m3_bits16(k_phys + sub_n[:, None] * K_STOKEN + offs_d[None, :])
                if QUANT_TYPE == _QKPERTOKEN_VPERHEAD_JIT:
                    scale_base = (
                        K_SCALE
                        + physical * KS_SPAGE
                        + ((sub * _SUB_N_JIT) // 32) * KS_STOKEN
                        + hkv * KS_SHEAD
                    )
                    token_scale = tl.load(scale_base + tl.arange(0, _SUB_N_JIT))
                    k = (k * token_scale[:, None]).to(tl.bfloat16)
                else:
                    k = (k * tl.load(K_SCALE)).to(tl.bfloat16)
                v = e4m3_bits16(v_phys + sub_n[:, None] * V_STOKEN + offs_d[None, :])
                v = (v * value_scale).to(tl.bfloat16)
                tl.store(K_BF16 + slot * KB_SPAGE + hkv * KB_SHEAD + off, k)
                tl.store(V_BF16 + slot * VB_SPAGE + hkv * VB_SHEAD + off, v)


@triton.jit
def _fp8_decode_producer_fused(
    TASK_MAP,
    Q,
    K,
    V,
    BLOCK_IDS,
    KV_LENS,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    FP8_LUT,
    SPLIT_OUT,
    SPLIT_LSE,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    BID_SB: tl.constexpr,
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
    FULL_SPLITS: tl.constexpr,
):
    """Decode one split and reduce it to a partial output.

    One program per ``(batch, kv_head, split)`` task: it E4M3-decodes the Q rows
    and the ``PAGES_PER_SPLIT`` KV pages itself and runs the QK and PV stages with
    the same online-softmax as the BF16 producer.  Decoding only the pages a
    program is about to consume is what keeps the whole FP8 cache from being
    materialised in BF16.

    A page is decoded in ``BLOCK_SIZE // _SUB_N`` sub-blocks because the
    arithmetic decoder needs about 13 fp32 temporaries for a ``(64, 128)`` tile
    (442KB), far beyond the 192KB UB; ``(32, 128)`` halves that and fits.  The
    decoded values are bit-identical either way -- only the online-softmax merge
    granularity changes.
    """
    if USE_TASK_MAP:
        task = tl.program_id(0) * 3
        batch = tl.load(TASK_MAP + task)
        hkv = tl.load(TASK_MAP + task + 1)
        split_id = tl.load(TASK_MAP + task + 2)
    else:
        batch = tl.program_id(0)
        hkv = tl.program_id(1)
        split_id = tl.program_id(2)
    first_page = split_id * PAGES_PER_SPLIT
    seq_len = tl.load(KV_LENS + batch)
    if first_page * _BLOCK_SIZE_JIT >= seq_len:
        return

    offs_r = tl.arange(0, Q_ROWS)
    offs_n = tl.arange(0, _BLOCK_SIZE_JIT)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (hq < H_Q)
    q_ptr = Q + (batch * NUM_SEQ_Q + seq_m[:, None]) * Q_SB + hq[:, None] * Q_SH + offs_d[None, :]
    q = e4m3_exact(q_ptr)
    qs = tl.load(
        Q_SCALE + (batch * NUM_SEQ_Q + seq_m) * QS_SB + hq * QS_SH,
        mask=valid_row,
        other=0.0,
    )
    log2e_over_sqrt_d: tl.constexpr = 0.127499612793
    if QUANT_TYPE == _QPERTOKEN_KVPERTENSOR_JIT:
        # K's tensor scale is constant: folded into q's scale, the per-element multiply
        # vanishes (dot(q*s,k) == s*dot(q,k)).  V's tensor scale hits the epilogue.
        value_scale = tl.load(V_SCALE)
        qs = qs * tl.load(K_SCALE)
    else:
        value_scale = tl.load(V_SCALE + hkv)
    q = (q * qs[:, None]).to(tl.bfloat16)
    running_max = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    running_sum = tl.zeros((Q_ROWS,), tl.float32)
    accumulator = tl.zeros((Q_ROWS, _HEAD_DIM_JIT), tl.float32)
    remaining = seq_len - first_page * _BLOCK_SIZE_JIT
    num_pages = tl.minimum((remaining + _BLOCK_SIZE_JIT - 1) // _BLOCK_SIZE_JIT, PAGES_PER_SPLIT)
    last_page = first_page + tl.maximum(num_pages - 1, 0)
    query_pos = seq_len - NUM_SEQ_Q + seq_m

    # Program constants (hkv offset) are page-independent: scalar base, one add/page.
    k_head = K + hkv * K_SHEAD
    v_head = V + hkv * V_SHEAD
    for page_offset in tl.range(0, PAGES_PER_SPLIT):
        logical_page = first_page + page_offset
        page = logical_page if FULL_SPLITS else tl.minimum(logical_page, last_page)
        physical = tl.load(BLOCK_IDS + batch * BID_SB + page).to(tl.int64)
        k_page = k_head + physical * K_SBLOCK
        v_page = v_head + physical * V_SBLOCK
        # Split a page into BLOCK_SIZE // _SUB_N sub-blocks: (64,128) arithmetic decode
        # needs ~13 fp32 temporaries (442KB), beyond 192KB UB; (32,128) halves and fits.
        # Decoded values bit-identical; only online-softmax merge granularity changes.
        for sub in tl.static_range(0, _BLOCK_SIZE_JIT // _SUB_N_JIT):
            sub_n = sub * _SUB_N_JIT + tl.arange(0, _SUB_N_JIT)
            global_n = logical_page * _BLOCK_SIZE_JIT + sub_n
            valid_n = global_n < seq_len
            k_ptr = k_page + sub_n[:, None] * K_STOKEN + offs_d[None, :]
            k = e4m3_exact(k_ptr)
            if QUANT_TYPE == _QKPERTOKEN_VPERHEAD_JIT:
                # per-token K scale to score domain: dot(q,k) fp32-accumulates then
                # multiplies (1,32) token_scale -- per-element (32,128) multiply/convert
                # becomes (8,32) per-row; K unscaled then bf16-rounded, more accurate.
                scale_base = (
                    K_SCALE
                    + physical * KS_SBLOCK
                    + ((sub * _SUB_N_JIT) // 32) * KS_STOKEN
                    + hkv * KS_SHEAD
                )
                token_scale = tl.load(scale_base + tl.arange(0, _SUB_N_JIT))
                scores = tl.dot(q, tl.trans(k)) * (token_scale * log2e_over_sqrt_d)[None, :]
            else:
                scores = tl.dot(q, tl.trans(k)) * log2e_over_sqrt_d
            score_mask = (
                valid_row[:, None] & valid_n[None, :] & (global_n[None, :] <= query_pos[:, None])
            )
            p = tl.where(score_mask, tl.exp2(tl.minimum(scores, 100.0)), 0.0)
            pb = p.to(tl.bfloat16)
            running_sum += tl.sum(p, axis=1)
            v_ptr = v_page + sub_n[:, None] * V_STOKEN + offs_d[None, :]
            v = e4m3_exact(v_ptr)
            accumulator += tl.dot(pb, v)

    # Tensor-wide V scale is one scalar per program: move it to the epilogue, where one
    # (Q_ROWS,128) multiply replaces 16 (32,128) elementwise multiplies+bf16 converts.
    accumulator = accumulator * value_scale
    has_value = running_sum > 0.0
    partial = tl.where(has_value[:, None], accumulator / tl.where(has_value[:, None], running_sum[:, None], 1.0), 0.0)
    row = seq_m * H_Q + hq
    tl.store(SPLIT_OUT + batch * SO_SB + split_id * SO_SP + row[:, None] * SO_SR + offs_d[None, :], partial)
    tl.store(
        SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        tl.where(has_value, tl.log2(tl.where(has_value, running_sum, 1.0)), -float("inf")),
        mask=valid_row,
    )


@triton.jit
def _fp8_decode_producer_scratch(
    TASK_MAP,
    Q,
    K,
    V,
    BLOCK_IDS,
    PAGE_BASE,
    KV_LENS,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    FP8_LUT,
    K_BF16,
    V_BF16,
    SPLIT_OUT,
    SPLIT_LSE,
    USE_SCRATCH: tl.constexpr,
    PROD_SUB_N: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    BID_SB: tl.constexpr,
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
    FULL_SPLITS: tl.constexpr,
):
    """Reduce one split to a partial output, from the bf16 scratch or inline.

    Same task grid and same QK/PV pipeline as ``_fp8_decode_producer_fused``, but
    the KV side is read from the compact per-page bf16 scratch that the standalone
    decoding pass filled when ``USE_SCRATCH``, and decoded here when the scratch
    is not worth its extra write and read.  The compact slot layout keeps the K/V
    addresses affine, so this form needs no per-page ``block_ids`` gather.
    """
    if USE_TASK_MAP:
        task = tl.program_id(0) * 3
        batch = tl.load(TASK_MAP + task)
        hkv = tl.load(TASK_MAP + task + 1)
        split_id = tl.load(TASK_MAP + task + 2)
    else:
        batch = tl.program_id(0)
        hkv = tl.program_id(1)
        split_id = tl.program_id(2)
    first_page = split_id * PAGES_PER_SPLIT
    seq_len = tl.load(KV_LENS + batch)
    if first_page * _BLOCK_SIZE_JIT >= seq_len:
        return

    offs_r = tl.arange(0, Q_ROWS)
    offs_n = tl.arange(0, _BLOCK_SIZE_JIT)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (hq < H_Q)
    q_ptr = Q + (batch * NUM_SEQ_Q + seq_m[:, None]) * Q_SB + hq[:, None] * Q_SH + offs_d[None, :]
    q = e4m3_exact(q_ptr)
    qs = tl.load(
        Q_SCALE + (batch * NUM_SEQ_Q + seq_m) * QS_SB + hq * QS_SH,
        mask=valid_row,
        other=0.0,
    )
    q = (q * qs[:, None]).to(tl.bfloat16)

    log2e_over_sqrt_d: tl.constexpr = 0.127499612793
    if QUANT_TYPE == _QPERTOKEN_KVPERTENSOR_JIT:
        tensor_k_scale = tl.load(K_SCALE)
        value_scale = tl.load(V_SCALE)
    else:
        tensor_k_scale = 1.0
        value_scale = tl.load(V_SCALE + hkv)
    running_sum = tl.zeros((Q_ROWS,), tl.float32)
    accumulator = tl.zeros((Q_ROWS, _HEAD_DIM_JIT), tl.float32)
    remaining = seq_len - first_page * _BLOCK_SIZE_JIT
    num_pages = tl.minimum((remaining + _BLOCK_SIZE_JIT - 1) // _BLOCK_SIZE_JIT, PAGES_PER_SPLIT)
    last_page = first_page + tl.maximum(num_pages - 1, 0)
    query_pos = seq_len - NUM_SEQ_Q + seq_m

    for page_offset in tl.range(0, PAGES_PER_SPLIT):
        logical_page = first_page + page_offset
        page = logical_page if FULL_SPLITS else tl.minimum(logical_page, last_page)
        if USE_SCRATCH:
            # Compact slot: same slot as the decode kernel, producer pays one extra
            # scalar load and K/V addresses affine, no per-page block_ids gather.
            page_slot = tl.load(PAGE_BASE + batch) + page
        else:
            physical = tl.load(BLOCK_IDS + batch * BID_SB + page).to(tl.int64)
        # Split a page into BLOCK_SIZE // _SUB_N sub-blocks: (64,128) arithmetic decode
        # needs ~13 fp32 temporaries (442KB), beyond 192KB UB; (32,128) halves and fits.
        # Decoded values bit-identical; only online-softmax merge granularity changes.
        for sub in tl.static_range(0, _BLOCK_SIZE_JIT // PROD_SUB_N):
            sub_n = sub * PROD_SUB_N + tl.arange(0, PROD_SUB_N)
            global_n = logical_page * _BLOCK_SIZE_JIT + sub_n
            valid_n = global_n < seq_len
            if USE_SCRATCH:
                k = tl.load(
                    K_BF16 + page_slot * KB_SPAGE + sub_n[:, None] * KB_STOKEN
                    + hkv * KB_SHEAD + offs_d[None, :]
                )
            else:
                k = e4m3_exact(
                    K + physical * K_SBLOCK + sub_n[:, None] * K_STOKEN
                    + hkv * K_SHEAD + offs_d[None, :]
                )
            if not USE_SCRATCH and QUANT_TYPE == _QKPERTOKEN_VPERHEAD_JIT:
                # k_scale's 128-byte group is 32 little-endian fp32 values (host float32
                # view), so this is **one plain contiguous float load** -- no reshape,
                # no weighted sum, fully affine addresses.  Old form: 4 per-lane strided
                # loads + per-lane shifts + div/mod addresses ("vector-indexed access
                # falls out of DMA", prefill item-15), B=64 ablation ~4.2ms.
                scale_base = (
                    K_SCALE
                    + physical * KS_SBLOCK
                    + ((sub * PROD_SUB_N) // 32) * KS_STOKEN
                    + hkv * KS_SHEAD
                )
                token_scale = tl.load(scale_base + tl.arange(0, PROD_SUB_N))
                k = (k * token_scale[:, None]).to(tl.bfloat16)
            elif not USE_SCRATCH:
                k = (k * tensor_k_scale).to(tl.bfloat16)
            scores = tl.dot(q, tl.trans(k)) * log2e_over_sqrt_d
            score_mask = (
                valid_row[:, None] & valid_n[None, :] & (global_n[None, :] <= query_pos[:, None])
            )
            scores = tl.where(score_mask, scores, -float("inf"))
            # Isomorphic to the BF16 producer: no running max, no accumulator rescaling,
            # only a clip against exp2 overflow; normalisation deferred to split end.
            # The reduce kernel merges "normalised" partials via LSE, so the end divides
            # by this split's l (equivalent to per-step max shift up to FP rounding).
            p = tl.where(score_mask, tl.exp2(tl.minimum(scores, 100.0)), 0.0)
            running_sum += tl.sum(p, axis=1)
            if USE_SCRATCH:
                v = tl.load(
                    V_BF16 + page_slot * VB_SPAGE + sub_n[:, None] * VB_STOKEN
                    + hkv * VB_SHEAD + offs_d[None, :]
                )
            else:
                v = (
                    e4m3_exact(
                        V + physical * V_SBLOCK + sub_n[:, None] * V_STOKEN
                        + hkv * V_SHEAD + offs_d[None, :]
                    )
                    * value_scale
                ).to(tl.bfloat16)
            accumulator += tl.dot(p.to(tl.bfloat16), v)

    has_value = running_sum > 0.0
    partial = tl.where(has_value[:, None], accumulator / tl.where(has_value[:, None], running_sum[:, None], 1.0), 0.0)
    row = seq_m * H_Q + hq
    tl.store(SPLIT_OUT + batch * SO_SB + split_id * SO_SP + row[:, None] * SO_SR + offs_d[None, :], partial)
    tl.store(
        SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        tl.where(has_value, tl.log2(tl.where(has_value, running_sum, 1.0)), -float("inf")),
        mask=valid_row,
    )


@triton.jit
def _fp8_decode_producer_scratch_wide(
    TASK_MAP,
    Q,
    PAGE_BASE,
    KV_LENS,
    Q_SCALE,
    K_BF16,
    V_BF16,
    SPLIT_OUT,
    SPLIT_LSE,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
    FULL_SPLITS: tl.constexpr,
):
    """BF16-isomorphic scratch producer: page-block QK, one wide PV per block.

    The whole split occupies ``PAGES_PER_SPLIT`` consecutive compact scratch
    slots, so K and V for the split are affine blocks of the page-major scratch.
    Both scratch layouts satisfy ``stride(page) == BLOCK_SIZE * stride(token)``,
    so a flat 2-D row index ``r`` addresses ``(page = r // 64, token = r % 64)``
    for any number of KV heads: K and V needs no per-page walk at all.

    The split is consumed in blocks of at most eight pages.  QK is then a
    ``(Q_ROWS, D) x (D, CHUNK*64)`` dot whose result already has exactly the
    shape of the PV left operand, so the per-page probabilities are never
    materialised as a one-hot tensor -- that, and the sixteen narrow dots it
    forced, is what kept the previous producer at 2.8x the BF16 one.

    A tail (non-full) split is handled by the same wide path.  The causal mask
    ``global_n <= query_pos`` already zeroes every lane past the sequence end
    (``query_pos < seq_len``), so the scores of the invalid pages need no
    separate clamp; the only extra work is that the V load is masked, because
    ``0 * NaN`` from an undecoded padding lane would poison the PV dot.  K is
    read as an unmasked tile whose tail rows land in the extra
    ``PAGES_PER_SPLIT`` pages allocated by ``_scratch_shape`` and are discarded
    by the score mask.
    """
    if USE_TASK_MAP:
        task = tl.program_id(0) * 3
        batch = tl.load(TASK_MAP + task)
        hkv = tl.load(TASK_MAP + task + 1)
        split_id = tl.load(TASK_MAP + task + 2)
    else:
        batch = tl.program_id(0)
        hkv = tl.program_id(1)
        split_id = tl.program_id(2)
    first_page = split_id * PAGES_PER_SPLIT
    seq_len = tl.load(KV_LENS + batch)
    if first_page * _BLOCK_SIZE_JIT >= seq_len:
        return

    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (hq < H_Q)
    q_ptr = Q + (batch * NUM_SEQ_Q + seq_m[:, None]) * Q_SB + hq[:, None] * Q_SH + offs_d[None, :]
    q = e4m3_exact(q_ptr)
    qs = tl.load(
        Q_SCALE + (batch * NUM_SEQ_Q + seq_m) * QS_SB + hq * QS_SH,
        mask=valid_row,
        other=0.0,
    )
    q = (q * qs[:, None]).to(tl.bfloat16)
    log2e_over_sqrt_d: tl.constexpr = 0.127499612793
    query_pos = seq_len - NUM_SEQ_Q + seq_m
    slot0 = tl.load(PAGE_BASE + batch) + first_page

    if PAGES_PER_SPLIT > 8:
        CHUNK: tl.constexpr = 8
    else:
        CHUNK: tl.constexpr = PAGES_PER_SPLIT
    chunk_lanes: tl.constexpr = CHUNK * _BLOCK_SIZE_JIT
    n_chunks: tl.constexpr = PAGES_PER_SPLIT // CHUNK
    offs_c = tl.arange(0, chunk_lanes)

    running_sum = tl.zeros((Q_ROWS,), tl.float32)
    split_acc = tl.zeros((Q_ROWS, _HEAD_DIM_JIT), tl.float32)
    for chunk in tl.static_range(0, n_chunks):
        base_slot = slot0 + chunk * CHUNK
        # One L1 tile per block, as in the BF16 producer: the cube reads the
        # transpose straight out of L1 instead of needing a UB layout change.
        k_l1 = tile_alloc(
            [chunk_lanes, _HEAD_DIM_JIT],
            K_BF16.dtype.element_ty,
            tle.language.dsa.ascend.L1,
        )
        k_ptr = tl.make_block_ptr(
            base=K_BF16 + base_slot * KB_SPAGE + hkv * KB_SHEAD,
            shape=(chunk_lanes, _HEAD_DIM_JIT),
            strides=(KB_STOKEN, 1),
            offsets=(0, 0),
            block_shape=(chunk_lanes, _HEAD_DIM_JIT),
            order=(1, 0),
        )
        tile_copy(k_ptr, k_l1, [chunk_lanes, _HEAD_DIM_JIT], inter_no_alias=True)
        k = tile_to_tensor(k_l1, writable=False)
        scores = tl.dot(q, tl.trans(k)) * log2e_over_sqrt_d
        global_n = (first_page + chunk * CHUNK) * _BLOCK_SIZE_JIT + offs_c
        score_mask = valid_row[:, None] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))
        p = tl.exp2(tl.minimum(scores, 100.0))
        running_sum += tl.sum(p, axis=1)
        v_ptrs = (
            V_BF16
            + base_slot * VB_SPAGE
            + hkv * VB_SHEAD
            + offs_c[:, None] * VB_STOKEN
            + offs_d[None, :]
        )
        if FULL_SPLITS:
            v = tl.load(v_ptrs)
        else:
            # Tail split: every lane past the sequence end carries p == 0, so
            # the load only has to avoid turning an undecoded padding element
            # into 0 * NaN.
            v = tl.load(v_ptrs, mask=(global_n < seq_len)[:, None], other=0.0)
        split_acc += tl.dot(p.to(tl.bfloat16), v)

    has_value = running_sum > 0.0
    partial = tl.where(
        has_value[:, None],
        split_acc / tl.where(has_value[:, None], running_sum[:, None], 1.0),
        0.0,
    )
    row = seq_m * H_Q + hq
    tl.store(
        SPLIT_OUT + batch * SO_SB + split_id * SO_SP + row[:, None] * SO_SR + offs_d[None, :],
        partial,
    )
    tl.store(
        SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        tl.where(has_value, tl.log2(tl.where(has_value, running_sum, 1.0)), -float("inf")),
        mask=valid_row,
    )


# ---------------------------------------------------------------------------
# Launch fast path
# ---------------------------------------------------------------------------
# The launches below call ``launch.launch_kernel``, which caches the specialized
# expert per kernel and argument specialization (structural key: tensors by dtype,
# 16B alignment and shape, everything else by value) and falls back to
# ``JITFunction.run`` for a key it has not seen.  Measured host cost, the Triton
# private API it depends on and the fallback path are documented in ``launch.py``.
def attention_decode_fp8(
    inputs: FP8DecodeInputs,
    workspace: FP8DecodeWorkspace,
    quant_type: int,
) -> torch.Tensor:
    mtp, hq, hkv = _validate_fp8_layout(inputs, quant_type)
    if workspace.mtp != mtp or workspace.quant_type != quant_type:
        raise ValueError("workspace does not match FP8 inputs")
    heads_per_group = hq // hkv
    q_rows = mtp * heads_per_group
    producer_grid = (
        (workspace.num_producer_tasks,)
        if workspace.compact_producer
        else (inputs.batch, hkv, workspace.max_splits)
    )
    if workspace.use_scratch:
        _decode_slots = _DECODE_SLOTS
        _total_slots = workspace.total_pages * hkv
        launch_kernel(
            _fp8_decode_kv_kernel,
            (triton.cdiv(_total_slots, _decode_slots),),
            inputs.k_cache,
            inputs.v_cache,
            inputs.k_scale.view(torch.float32),
            inputs.v_scale,
            inputs.block_ids,
            workspace.phys_slots,
            workspace.page_base,
            workspace.k_bf16,
            workspace.v_bf16,
            H_KV=hkv,
            NB_P2=workspace.nb_p2,
            SLOTS=_decode_slots,
            TOTAL_SLOTS=_total_slots,
            BID_SB=inputs.block_ids.stride(0),
            K_SPAGE=inputs.k_cache.stride(0),
            K_STOKEN=inputs.k_cache.stride(1),
            K_SHEAD=inputs.k_cache.stride(2),
            V_SPAGE=inputs.v_cache.stride(0),
            V_STOKEN=inputs.v_cache.stride(1),
            V_SHEAD=inputs.v_cache.stride(2),
            KS_SPAGE=inputs.k_scale.stride(0) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_STOKEN=inputs.k_scale.stride(1) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_SHEAD=inputs.k_scale.stride(2) // 4 if inputs.k_scale.ndim == 4 else 0,
            KB_SPAGE=workspace.k_bf16.stride(0),
            KB_STOKEN=workspace.k_bf16.stride(1),
            KB_SHEAD=workspace.k_bf16.stride(2),
            VB_SPAGE=workspace.v_bf16.stride(0),
            VB_STOKEN=workspace.v_bf16.stride(1),
            VB_SHEAD=workspace.v_bf16.stride(2),
            QUANT_TYPE=quant_type,
            num_warps=4,
        )
    # The wide producer keeps a (Q_ROWS, page-block * 64) score tile and its
    # fp32 temporaries in UB: at Q_ROWS=32 the compiler asks for 3016192 bits
    # against 1572864 available (Q_ROWS=16 is already within 2% of the bound),
    # so it is only used for the small MTP=1 tile the measurements cover.  The
    # wider Q tiles stay on the per-page scratch producer below.
    #
    # No ``full_producer_splits`` gate: the causal mask already zeroes the
    # lanes past the sequence end of a tail split, and the V load is masked so
    # the undecoded padding cannot leak through the PV dot.  A shape whose
    # *every* split is full still takes the identical (unmasked) fast path
    # through the FULL_SPLITS constexpr.
    if workspace.use_scratch and q_rows <= _WIDE_MAX_Q_ROWS:
        launch_kernel(
            _fp8_decode_producer_scratch_wide,
            producer_grid,
            workspace.producer_task_map,
            inputs.q,
            workspace.page_base,
            inputs.kv_lens,
            inputs.q_scale,
            workspace.k_bf16,
            workspace.v_bf16,
            workspace.split_out,
            workspace.split_lse,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            Q_SB=inputs.q.stride(0),
            Q_SH=inputs.q.stride(1),
            QS_SB=inputs.q_scale.stride(0),
            QS_SH=inputs.q_scale.stride(1),
            KB_SPAGE=workspace.k_bf16.stride(0),
            KB_STOKEN=workspace.k_bf16.stride(1),
            KB_SHEAD=workspace.k_bf16.stride(2),
            VB_SPAGE=workspace.v_bf16.stride(0),
            VB_STOKEN=workspace.v_bf16.stride(1),
            VB_SHEAD=workspace.v_bf16.stride(2),
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SR=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            USE_TASK_MAP=workspace.compact_producer,
            FULL_SPLITS=workspace.full_producer_splits,
            num_warps=8,
            num_stages=1,
        )
    elif workspace.use_scratch:
        launch_kernel(
            _fp8_decode_producer_scratch,
            producer_grid,
            workspace.producer_task_map,
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.page_base,
            inputs.kv_lens,
            inputs.q_scale,
            inputs.k_scale.view(torch.float32),
            inputs.v_scale,
            workspace.fp8_lut,
            workspace.k_bf16,
            workspace.v_bf16,
            workspace.split_out,
            workspace.split_lse,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            Q_SB=inputs.q.stride(0),
            Q_SH=inputs.q.stride(1),
            K_SBLOCK=inputs.k_cache.stride(0),
            K_STOKEN=inputs.k_cache.stride(1),
            K_SHEAD=inputs.k_cache.stride(2),
            V_SBLOCK=inputs.v_cache.stride(0),
            V_STOKEN=inputs.v_cache.stride(1),
            V_SHEAD=inputs.v_cache.stride(2),
            BID_SB=inputs.block_ids.stride(0),
            QS_SB=inputs.q_scale.stride(0),
            QS_SH=inputs.q_scale.stride(1),
            # qk-format per-token scale packs as (page, 32-token group, hkv, 128B),
            # where 128 bytes = 32 little-endian fp32.  A host-side float32 view gives
            # one plain float load in-kernel; strides are fp32 units (byte stride // 4).
            KS_SBLOCK=inputs.k_scale.stride(0) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_STOKEN=inputs.k_scale.stride(1) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_SHEAD=inputs.k_scale.stride(2) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_SD=inputs.k_scale.stride(3) if inputs.k_scale.ndim == 4 else 0,
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SR=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            QUANT_TYPE=quant_type,
            USE_SCRATCH=True,
            PROD_SUB_N=_SCRATCH_SUB_N,
            KB_SPAGE=workspace.k_bf16.stride(0),
            KB_STOKEN=workspace.k_bf16.stride(1),
            KB_SHEAD=workspace.k_bf16.stride(2),
            VB_SPAGE=workspace.v_bf16.stride(0),
            VB_STOKEN=workspace.v_bf16.stride(1),
            VB_SHEAD=workspace.v_bf16.stride(2),
            USE_TASK_MAP=workspace.compact_producer,
            FULL_SPLITS=workspace.full_producer_splits,
            num_warps=8,
            num_stages=1,
        )
    else:
        launch_kernel(
            _fp8_decode_producer_fused,
            producer_grid,
            workspace.producer_task_map,
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            inputs.kv_lens,
            inputs.q_scale,
            inputs.k_scale.view(torch.float32),
            inputs.v_scale,
            workspace.fp8_lut,
            workspace.split_out,
            workspace.split_lse,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            Q_SB=inputs.q.stride(0),
            Q_SH=inputs.q.stride(1),
            K_SBLOCK=inputs.k_cache.stride(0),
            K_STOKEN=inputs.k_cache.stride(1),
            K_SHEAD=inputs.k_cache.stride(2),
            V_SBLOCK=inputs.v_cache.stride(0),
            V_STOKEN=inputs.v_cache.stride(1),
            V_SHEAD=inputs.v_cache.stride(2),
            BID_SB=inputs.block_ids.stride(0),
            QS_SB=inputs.q_scale.stride(0),
            QS_SH=inputs.q_scale.stride(1),
            # qk-format per-token scale packs as (page, 32-token group, hkv, 128B),
            # where 128 bytes = 32 little-endian fp32.  A host-side float32 view gives
            # one plain float load in-kernel; strides are fp32 units (byte stride // 4).
            KS_SBLOCK=inputs.k_scale.stride(0) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_STOKEN=inputs.k_scale.stride(1) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_SHEAD=inputs.k_scale.stride(2) // 4 if inputs.k_scale.ndim == 4 else 0,
            KS_SD=inputs.k_scale.stride(3) if inputs.k_scale.ndim == 4 else 0,
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SR=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            QUANT_TYPE=quant_type,
            USE_TASK_MAP=workspace.compact_producer,
            FULL_SPLITS=workspace.full_producer_splits,
            num_warps=8,
            num_stages=1,
        )

    common = dict(
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        D=HEAD_DIM,
        SO_SB=workspace.split_out.stride(0),
        SO_SP=workspace.split_out.stride(1),
        SO_SR=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SP=workspace.split_lse.stride(1),
        O_SB=workspace.out.stride(0),
        O_SH=workspace.out.stride(1),
    )
    if workspace.hierarchical_reduction:
        launch_kernel(
            _bf16_mtp_reduce_splits_kernel,
            (workspace.num_reduce_tasks * mtp,),
            workspace.reduce_task_map,
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.reduced_out,
            workspace.reduced_lse,
            workspace.out,
            TOKENS_PER_SPLIT=BLOCK_SIZE * workspace.pages_per_split,
            SPLITS_PER_REDUCTION=FP8_SPLITS_PER_REDUCTION,
            RO_SB=workspace.reduced_out.stride(0),
            RO_SG=workspace.reduced_out.stride(1),
            RO_SR=workspace.reduced_out.stride(2),
            RL_SB=workspace.reduced_lse.stride(0),
            RL_SG=workspace.reduced_lse.stride(1),
            num_warps=4,
            num_stages=1,
            **common,
        )
        launch_kernel(
            _bf16_mtp_finalize_groups_kernel,
            (workspace.num_final_tasks * mtp,),
            workspace.final_task_map,
            workspace.reduced_out,
            workspace.reduced_lse,
            inputs.kv_lens,
            workspace.out,
            NUM_SEQ_Q=mtp,
            H_Q=hq,
            TOKENS_PER_GROUP=BLOCK_SIZE * workspace.pages_per_split * FP8_SPLITS_PER_REDUCTION,
            D=HEAD_DIM,
            RO_SB=workspace.reduced_out.stride(0),
            RO_SG=workspace.reduced_out.stride(1),
            RO_SR=workspace.reduced_out.stride(2),
            RL_SB=workspace.reduced_lse.stride(0),
            RL_SG=workspace.reduced_lse.stride(1),
            O_SB=workspace.out.stride(0),
            O_SH=workspace.out.stride(1),
            num_warps=4,
            num_stages=1,
        )
    else:
        launch_kernel(
            _bf16_mtp_finalize_kernel,
            (inputs.batch, mtp * hq),
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.out,
            TOKENS_PER_SPLIT=BLOCK_SIZE * workspace.pages_per_split,
            MAX_SPLITS=FP8_SPLITS_PER_REDUCTION,
            num_warps=4,
            num_stages=1,
            **common,
        )
    return workspace.out


def refresh_fp8_task_map(inputs: FP8DecodeInputs, workspace: FP8DecodeWorkspace) -> None:
    validate_fp8_inputs(inputs, workspace.quant_type)
    schedule = _build_schedule(inputs, workspace.pages_per_split)
    expected = (
        workspace.max_splits,
        workspace.max_reduction_groups,
        workspace.num_producer_tasks,
        workspace.num_reduce_tasks,
        workspace.num_final_tasks,
        workspace.compact_producer,
        workspace.hierarchical_reduction,
    )
    actual = (
        int(schedule["max_splits"]),
        int(schedule["max_reduction_groups"]),
        int(schedule["num_producer_tasks"]),
        int(schedule["num_reduce_tasks"]),
        int(schedule["num_final_tasks"]),
        bool(schedule["compact_producer"]),
        bool(schedule["hierarchical_reduction"]),
    )
    if actual != expected:
        raise ValueError("dynamic schedule topology changed; rebuild the workspace")
    for destination, name in (
        (workspace.producer_task_map, "producer_task_map"),
        (workspace.reduce_task_map, "reduce_task_map"),
        (workspace.final_task_map, "final_task_map"),
    ):
        destination.copy_(schedule[name])
    workspace.full_producer_splits = bool(schedule["full_producer_splits"])
    # Page table and per-batch referenced page counts change per step; slot table and
    # grid refresh with them, dense batch shape fixed (topology check fixes max_splits).
    hkv = int(inputs.k_cache.shape[2])
    grid_programs = (
        workspace.num_producer_tasks
        if workspace.compact_producer
        else inputs.batch * hkv * workspace.max_splits
    )
    padded_kv_elems = (
        grid_programs * workspace.pages_per_split * BLOCK_SIZE * HEAD_DIM * hkv
    )
    page_base, phys_slots, total_pages, pool_pages, nb_p2, use_scratch = _page_plan(
        inputs, hkv, padded_kv_elems
    )
    if nb_p2 != workspace.nb_p2:
        raise ValueError("dynamic page-column count changed; rebuild the workspace")
    workspace.page_base.copy_(page_base)
    if total_pages > workspace.phys_slots.numel():
        raise ValueError("dynamic page count exceeded the scratch slot buffer")
    if total_pages:
        workspace.phys_slots[:total_pages].copy_(phys_slots)
    workspace.total_pages = total_pages
    workspace.pool_pages = pool_pages
    workspace.use_scratch = use_scratch


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "SUPPORTED_MTP",
    "QPERTOKEN_KVPERTENSOR",
    "QKPERTOKEN_VPERHEAD",
    "FP8DecodeInputs",
    "FP8DecodeWorkspace",
    "attention_decode_fp8",
    "prepare_fp8_workspace",
    "refresh_fp8_task_map",
    "validate_fp8_inputs",
]
