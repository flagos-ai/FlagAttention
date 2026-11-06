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

"""Ascend Triton implementation of the HY3 BF16 decode formats.

The kernels cover both MTP=1 and MTP>=2: the page gather, the QK and PV
producers, and the finalize/reduce tail.  The part of that tail which the FP8
implementation reuses lives in ``decode/mtp_reduce.py``.  The host side builds
the task maps, prepares the workspace and launches the phases.

``decode/__init__.py`` re-exports the public entries, and the ``static`` and
``dynamic`` wrappers import them from the decode package.
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

from .mtp_reduce import (
    _bf16_mtp_finalize_groups_kernel,
    _bf16_mtp_finalize_kernel,
    _bf16_mtp_reduce_splits_kernel,
)


HAS_TLE = False
USE_TLE = False
# 64 tokens per KV page, so a split covers PAGES_PER_SPLIT * 64 tokens.
PAGES_PER_SPLIT = 8
# Candidate split sizes for the MTP1 producer, 64 tokens per page.  The producer
# walks its pages serially inside one program, so its cost is a fixed
# per-program part (Q tile load, epilogue stores, the program's serial latency)
# plus a per-page part; longer splits amortise the fixed part but burn page
# slots on the tail split of short sequences.  PAGES_PER_SPLIT=32 does not
# compile: "cbuf overflow, requires 4718592 bits while 4194304 bits available"
# (L1 is 512KB).
PAGES_PER_SPLIT_CHOICES = (8, 16)
# Per-program and per-page-slot producer costs in microseconds, fitted on
# Ascend910B4 at 8q/1kv, BLOCK_SIZE=64, D=128 (see
# tools/ascend_profile/probe_pps.py).  Only their ratio drives the choice, so
# the units cancel; the values reproduce the measured winner on every case.
PRODUCER_PROGRAM_COST_US = 0.133
PRODUCER_PAGE_COST_US = 0.0522
# Pages merged into a single QK/PV dot by the MTP1 producer.  The pages are made
# contiguous by _bf16_mtp_gather_v_kernel.  Measured on Ascend910B4 (8q/1kv,
# same-window A/B vs 98a67b5): 8 pages/dot is best end to end
# (uniform_512 91.2->81.0us, uniform_4096 576.1->489.3us,
# one_64k_7x4k 222.1->157.8us); 4 pages/dot is worse than the per-page
# baseline on uniform_512 (94.6us) because the gather is not amortised.
PAGES_PER_DOT = 8
SPLITS_PER_REDUCTION = 16


def _choose_pages_per_split(lengths, mtp: int) -> int:
    """Pick the MTP1 producer's pages per split from the KV length histogram.

    A split covers ``PAGES_PER_SPLIT * 64`` tokens, so ``length`` needs
    ``ceil(length / (PAGES_PER_SPLIT * 64))`` producer programs and the last one
    only pays for the pages it has.  The model therefore scores each candidate
    as::

        sum_b ceil(length_b / (PPS * 64)) * (program_cost + PPS * page_cost)

    which prefers longer splits on long sequences (fewer programs for the same
    page work) and shorter splits as soon as the tail splits stop being full.
    The choice is fixed when the workspace is built and reused afterwards, so a
    growing KV length cannot re-shard an existing workspace.
    """
    if mtp != 1:
        # The model was fitted against the MTP1 producer; MTP2/3/4 keep the
        # stable eight-page split size.
        return PAGES_PER_SPLIT
    if not lengths:
        return PAGES_PER_SPLIT
    best = PAGES_PER_SPLIT
    best_cost = None
    for candidate in PAGES_PER_SPLIT_CHOICES:
        programs = sum(
            triton.cdiv(length, candidate * 64) for length in lengths
        )
        cost = programs * (
            PRODUCER_PROGRAM_COST_US + candidate * PRODUCER_PAGE_COST_US
        )
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    return best


@dataclass
class AscendBF16MTP1Workspace:
    """Split-K buffers for the Ascend BF16 MTP1 decode kernel."""

    producer_task_map: torch.Tensor
    reduce_task_map: torch.Tensor
    final_task_map: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    reduced_out: torch.Tensor
    reduced_lse: torch.Tensor
    # Contiguous per-split copies of V.  K remains paged and is consumed one
    # page at a time; contiguous V lets the producer issue one PV dot per split.
    gather_v: torch.Tensor
    # MTP3/4 materialise probabilities between the combined QK/V-gather stage
    # and the contiguous PV stage.  MTP1 keeps its lower-overhead fused path.
    score_p: torch.Tensor
    score_sum: torch.Tensor
    out: torch.Tensor
    max_splits: int
    max_reduction_groups: int
    num_producer_tasks: int
    num_reduce_tasks: int
    num_final_tasks: int
    compact_producer: bool
    hierarchical_reduction: bool
    full_producer_splits: bool
    mtp: int
    # Pages per split chosen from the KV lengths when the workspace was built.
    # Every kernel that re-shards by split must use this, not the module
    # default, so that a schedule refresh (KV lengths grow during decoding)
    # cannot change the sharding of an existing workspace.
    pages_per_split: int

    def is_reset(self) -> bool:
        """No completion counters persist between the separate kernel launches."""
        return True

    @property
    def route(self) -> str:
        """Describe the actual producer and reduction choices for callers."""
        producer = "staged-qk-pv" if self.mtp >= 3 else "fused-pv"
        schedule = "compact" if self.compact_producer else "rectangular"
        reduction = "hierarchical" if self.hierarchical_reduction else "direct"
        return f"ascend/mtp{self.mtp}/{producer}/{schedule}/pps{self.pages_per_split}/{reduction}"


@triton.jit
def _bf16_mtp1_update_page(
    scores,
    valid_h,
    valid_n,
    v,
    unit,
    split_sum,
    split_acc,
):
    """Fold one page into a fixed-shift softmax accumulation.

    ``unit`` is exactly 1.0 but comes from a loaded value, so the compiler
    cannot fold the rescale away.  Keeping that vector op is required by the
    BiSheng store lowering (an untouched Cube accumulator stays in L0C and
    cannot be stored to GM), and it lets us drop the per-page ``tl.max`` /
    ``tl.maximum`` running max in favour of one clamp.
    """
    scores = tl.where(
        valid_h[:, None] & valid_n[None, :],
        scores,
        -float("inf"),
    )
    p = tl.exp2(tl.minimum(scores, 100.0))
    split_sum = split_sum * unit + tl.sum(p, axis=1)
    split_acc = split_acc * unit[:, None] + tl.dot(
        p.to(tl.bfloat16), v
    )
    return split_sum, split_acc


@triton.jit
def _bf16_mtp_gather_v_kernel(
    TASK_MAP,
    V,
    BLOCK_IDS,
    KV_LENS,
    GATHER_V,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    BID_SB: tl.constexpr,
    GK_SB: tl.constexpr,
    GK_SH: tl.constexpr,
    GK_SP: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
):
    """Copy one split's V pages into a contiguous scratch buffer."""
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
    if first_page * BLOCK_SIZE >= seq_len:
        return
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)
    num_pages = tl.minimum(
        (seq_len - first_page * BLOCK_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE,
        PAGES_PER_SPLIT,
    )
    base = batch * GK_SB + hkv * GK_SH + split_id * GK_SP
    for slot in tl.static_range(0, PAGES_PER_SPLIT):
        off = slot * BLOCK_SIZE * D + offs_n[:, None] * D + offs_d[None, :]
        if slot < num_pages:
            physical = tl.load(
                BLOCK_IDS + batch * BID_SB + first_page + slot
            ).to(tl.int64)
            tl.store(
                GATHER_V + base + off,
                tl.load(
                    V
                    + physical * V_SBLOCK
                    + hkv * V_SHEAD
                    + offs_n[:, None] * V_STOKEN
                    + offs_d[None, :] * V_SD
                ),
            )
        else:
            tl.store(GATHER_V + base + off, tl.zeros([BLOCK_SIZE, D], tl.bfloat16))



@triton.jit
def _bf16_mtp1_page_kernel(
    TASK_MAP,
    Q,
    K,
    GATHER_V,
    BLOCK_IDS,
    KV_LENS,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    PAGES_PER_DOT: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    BID_SB: tl.constexpr,
    GK_SB: tl.constexpr,
    GK_SH: tl.constexpr,
    GK_SP: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
    DIRECT_OUTPUT: tl.constexpr,
    FULL_SPLITS: tl.constexpr,
):
    """Produce one partial attention result for a fixed group of KV pages.

    QK reads paged K one page at a time. V was gathered by
    ``_bf16_mtp_gather_v_kernel``, so PV consumes one contiguous split operand.
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
    if first_page * BLOCK_SIZE >= seq_len:
        return

    block_h: tl.constexpr = HEADS_PER_GROUP
    block_n: tl.constexpr = BLOCK_SIZE
    offs_h = tl.arange(0, block_h)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, D)
    valid_h = offs_h < HEADS_PER_GROUP
    q_ptrs = tl.make_block_ptr(
        base=Q + batch * Q_SB + hkv * HEADS_PER_GROUP * Q_SH,
        shape=(HEADS_PER_GROUP, D),
        strides=(Q_SH, 1),
        offsets=(0, 0),
        block_shape=(block_h, D),
        order=(1, 0),
    )
    q_l1 = tile_alloc(
        [block_h, D],
        Q.dtype.element_ty,
        tle.language.dsa.ascend.L1,
    )
    tile_copy(q_ptrs, q_l1, [block_h, D], inter_no_alias=True)
    q = tile_to_tensor(q_l1, writable=False)
    scale = tl.rsqrt(tl.full((), D, tl.float32)) * 1.4426950408889634
    # Loaded (not computed) unit rescale: see _bf16_mtp1_update_page.
    unit = tl.load(
        KV_LENS + batch + offs_h * 0
    ).to(tl.float32) * 0.0 + 1.0
    split_sum = tl.zeros((block_h,), tl.float32)
    split_acc = tl.zeros((block_h, D), tl.float32)
    lanes: tl.constexpr = PAGES_PER_SPLIT * BLOCK_SIZE
    offs_l = tl.arange(0, lanes)
    offs_p = tl.arange(0, PAGES_PER_SPLIT)
    offs_n = tl.arange(0, BLOCK_SIZE)
    gather_base = batch * GK_SB + hkv * GK_SH + split_id * GK_SP
    num_pages = tl.minimum(
        (seq_len - first_page * BLOCK_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE,
        PAGES_PER_SPLIT,
    )
    last_page = first_page + tl.maximum(num_pages - 1, 0)
    # QK stays per page (K is read straight from the cache through L1); the
    # per-page probabilities are accumulated into a [HEADS, PAGES, BLOCK]
    # tensor so that PV becomes a single dot over the gathered V.
    if PAGES_PER_SPLIT == 16:
        half_pages: tl.constexpr = PAGES_PER_SPLIT // 2
        half_lanes: tl.constexpr = half_pages * BLOCK_SIZE
        offs_half_p = tl.arange(0, half_pages)
        p3_lo = tl.zeros((block_h, half_pages, BLOCK_SIZE), tl.bfloat16)
        p3_hi = tl.zeros((block_h, half_pages, BLOCK_SIZE), tl.bfloat16)
    else:
        p3 = tl.zeros((block_h, PAGES_PER_SPLIT, BLOCK_SIZE), tl.bfloat16)
    for slot in tl.static_range(0, PAGES_PER_SPLIT):
        # One L1 tile per page: reusing a single tile across the eight
        # iterations leaves a write-after-read dependency between the tile_copy
        # and the previous read that the backend does not always fence.
        k_l1 = tile_alloc(
            [BLOCK_SIZE, D], Q.dtype.element_ty, tle.language.dsa.ascend.L1,
        )
        # Full splits need neither the tail-page clamp nor its lane mask.  Keep
        # the clamped re-read only for a workload that has at least one tail
        # split; its invalid lanes become -inf below and contribute zero.
        if FULL_SPLITS:
            page = first_page + slot
        else:
            page = tl.minimum(first_page + slot, last_page)
        physical = tl.load(BLOCK_IDS + batch * BID_SB + page).to(tl.int64)
        k_ptr = tl.make_block_ptr(
            base=K + physical * K_SBLOCK + hkv * K_SHEAD,
            shape=(BLOCK_SIZE, D),
            strides=(K_STOKEN, K_SD),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE, D),
            order=(1, 0),
        )
        tile_copy(k_ptr, k_l1, [BLOCK_SIZE, D], inter_no_alias=True)
        k = tile_to_tensor(k_l1, writable=False)
        scores = tl.dot(q, tl.trans(k)) * scale
        if not FULL_SPLITS:
            valid_n = (first_page + slot) * BLOCK_SIZE + offs_n < seq_len
            scores = tl.where(valid_n[None, :], scores, -float("inf"))
        p = tl.exp2(tl.minimum(scores, 100.0))
        split_sum = split_sum * unit + tl.sum(p, axis=1)
        if PAGES_PER_SPLIT == 16:
            if slot < half_pages:
                p3_lo = tl.where(
                    offs_half_p[None, :, None] == slot,
                    tl.broadcast_to(
                        p.to(tl.bfloat16)[:, None, :],
                        (block_h, half_pages, BLOCK_SIZE),
                    ),
                    p3_lo,
                )
            else:
                p3_hi = tl.where(
                    offs_half_p[None, :, None] == slot - half_pages,
                    tl.broadcast_to(
                        p.to(tl.bfloat16)[:, None, :],
                        (block_h, half_pages, BLOCK_SIZE),
                    ),
                    p3_hi,
                )
        else:
            p3 = tl.where(
                offs_p[None, :, None] == slot,
                tl.broadcast_to(
                    p.to(tl.bfloat16)[:, None, :],
                    (block_h, PAGES_PER_SPLIT, BLOCK_SIZE),
                ),
                p3,
            )
    if PAGES_PER_SPLIT == 16:
        offs_half_l = tl.arange(0, half_lanes)
        pm_lo = tl.reshape(p3_lo, (block_h, half_lanes))
        pm_hi = tl.reshape(p3_hi, (block_h, half_lanes))
        vm_lo = tl.load(
            GATHER_V
            + gather_base
            + offs_half_l[:, None] * D
            + offs_d[None, :]
        )
        vm_hi = tl.load(
            GATHER_V
            + gather_base
            + (offs_half_l[:, None] + half_lanes) * D
            + offs_d[None, :]
        )
        split_acc = (
            split_acc * unit[:, None]
            + tl.dot(pm_lo, vm_lo)
            + tl.dot(pm_hi, vm_hi)
        )
    else:
        pm = tl.reshape(p3, (block_h, lanes))
        vm = tl.load(
            GATHER_V + gather_base + offs_l[:, None] * D + offs_d[None, :]
        )
        split_acc = split_acc * unit[:, None] + tl.dot(pm, vm)
    partial = split_acc / split_sum[:, None]
    hq = hkv * HEADS_PER_GROUP + offs_h
    if DIRECT_OUTPUT:
        tl.store(
            OUT + batch * O_SB + hq[:, None] * O_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
    else:
        tl.store(
            SPLIT_OUT
            + batch * SO_SB
            + split_id * SO_SP
            + hq[:, None] * SO_SH
            + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + split_id * SL_SP + hq,
            tl.log2(split_sum),
            mask=valid_h,
        )


@triton.jit
def _bf16_mtp_page_kernel(
    TASK_MAP,
    Q,
    K,
    GATHER_V,
    BLOCK_IDS,
    KV_LENS,
    SPLIT_OUT,
    SPLIT_LSE,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    BID_SB: tl.constexpr,
    GV_SB: tl.constexpr,
    GV_SH: tl.constexpr,
    GV_SP: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
    FULL_SPLITS: tl.constexpr,
):
    """Produce fused MTP partials for one KV head and split."""
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
    if first_page * BLOCK_SIZE >= seq_len:
        return

    offs_r = tl.arange(0, Q_ROWS)
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (
        (seq_m < NUM_SEQ_Q)
        & (h_in_group < HEADS_PER_GROUP)
        & (hq < H_Q)
    )
    q = tl.load(
        Q
        + (batch * NUM_SEQ_Q + seq_m[:, None]) * Q_SB
        + hq[:, None] * Q_SH
        + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    scale = tl.rsqrt(tl.full((), D, tl.float32)) * 1.4426950408889634
    # Loaded (not computed) unit rescale: see _bf16_mtp1_update_page.  The
    # rescale has to stay so the Cube accumulator passes through UB for the
    # BiSheng store lowering.
    unit = tl.load(
        KV_LENS + batch + offs_r * 0
    ).to(tl.float32) * 0.0 + 1.0
    split_sum = tl.zeros((Q_ROWS,), tl.float32)
    split_acc = tl.zeros((Q_ROWS, D), tl.float32)
    lanes: tl.constexpr = PAGES_PER_SPLIT * BLOCK_SIZE
    offs_l = tl.arange(0, lanes)
    offs_p = tl.arange(0, PAGES_PER_SPLIT)
    p3 = tl.zeros((Q_ROWS, PAGES_PER_SPLIT, BLOCK_SIZE), tl.bfloat16)
    remaining_tokens = seq_len - first_page * BLOCK_SIZE
    num_pages = tl.minimum(
        (remaining_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
        PAGES_PER_SPLIT,
    )
    last_page = first_page + tl.maximum(num_pages - 1, 0)
    query_pos = seq_len - NUM_SEQ_Q + seq_m
    for page_offset in tl.static_range(0, PAGES_PER_SPLIT):
        if FULL_SPLITS:
            page_id = first_page + page_offset
        else:
            page_id = tl.minimum(first_page + page_offset, last_page)
        global_n = (first_page + page_offset) * BLOCK_SIZE + offs_n
        physical = tl.load(
            BLOCK_IDS + batch * BID_SB + page_id,
        ).to(tl.int64)
        k_ptrs = (
            K
            + physical * K_SBLOCK
            + offs_n[:, None] * K_STOKEN
            + hkv * K_SHEAD
            + offs_d[None, :] * K_SD
        )
        if FULL_SPLITS:
            k = tl.load(k_ptrs)
        else:
            valid_n = global_n < seq_len
            k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * scale
        if FULL_SPLITS:
            score_mask = (
                valid_row[:, None]
                & (global_n[None, :] <= query_pos[:, None])
            )
        else:
            score_mask = (
                valid_row[:, None]
                & valid_n[None, :]
                & (global_n[None, :] <= query_pos[:, None])
            )
        scores = tl.where(score_mask, scores, -float("inf"))
        p = tl.exp2(tl.minimum(scores, 100.0))
        split_sum = split_sum * unit + tl.sum(p, axis=1)
        p3 = tl.where(
            offs_p[None, :, None] == page_offset,
            tl.broadcast_to(
                p.to(tl.bfloat16)[:, None, :],
                (Q_ROWS, PAGES_PER_SPLIT, BLOCK_SIZE),
            ),
            p3,
        )
    gather_base = batch * GV_SB + hkv * GV_SH + split_id * GV_SP
    pm = tl.reshape(p3, (Q_ROWS, lanes))
    vm = tl.load(
        GATHER_V + gather_base + offs_l[:, None] * D + offs_d[None, :]
    )
    split_acc = split_acc * unit[:, None] + tl.dot(pm, vm)
    has_value = split_sum > 0.0
    partial = tl.where(
        has_value[:, None],
        split_acc / tl.where(split_sum[:, None] > 0.0, split_sum[:, None], 1.0),
        0.0,
    )
    row = seq_m * H_Q + hq
    # Unmasked: the split buffer is padded to Q_ROWS, and invalid rows land in
    # padding that finalisation never reads.
    tl.store(
        SPLIT_OUT
        + batch * SO_SB
        + split_id * SO_SP
        + row[:, None] * SO_SR
        + offs_d[None, :],
        partial,
    )
    tl.store(
        SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        tl.where(
            has_value,
            tl.log2(tl.where(has_value, split_sum, 1.0)),
            -float("inf"),
        ),
        mask=valid_row,
    )


@triton.jit
def _bf16_mtp_qk_gather_v_kernel(
    TASK_MAP,
    Q,
    K,
    V,
    BLOCK_IDS,
    KV_LENS,
    GATHER_V,
    SCORE_P,
    SCORE_SUM,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    BID_SB: tl.constexpr,
    GV_SB: tl.constexpr,
    GV_SH: tl.constexpr,
    GV_SP: tl.constexpr,
    SP_SB: tl.constexpr,
    SP_SH: tl.constexpr,
    SP_SP: tl.constexpr,
    SP_SR: tl.constexpr,
    SS_SB: tl.constexpr,
    SS_SH: tl.constexpr,
    SS_SP: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
    FULL_SPLITS: tl.constexpr,
):
    """Overlap paged QK work with the V copy needed by the following PV."""
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
    if first_page * BLOCK_SIZE >= seq_len:
        return

    offs_r = tl.arange(0, Q_ROWS)
    offs_n = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (
        (seq_m < NUM_SEQ_Q)
        & (h_in_group < HEADS_PER_GROUP)
        & (hq < H_Q)
    )
    q = tl.load(
        Q
        + (batch * NUM_SEQ_Q + seq_m[:, None]) * Q_SB
        + hq[:, None] * Q_SH
        + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    scale = tl.rsqrt(tl.full((), D, tl.float32)) * 1.4426950408889634
    unit = tl.load(KV_LENS + batch + offs_r * 0).to(tl.float32) * 0.0 + 1.0
    split_sum = tl.zeros((Q_ROWS,), tl.float32)
    remaining_tokens = seq_len - first_page * BLOCK_SIZE
    num_pages = tl.minimum(
        (remaining_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
        PAGES_PER_SPLIT,
    )
    last_page = first_page + tl.maximum(num_pages - 1, 0)
    query_pos = seq_len - NUM_SEQ_Q + seq_m
    gv_base = batch * GV_SB + hkv * GV_SH + split_id * GV_SP
    p_base = batch * SP_SB + hkv * SP_SH + split_id * SP_SP
    for page_offset in tl.static_range(0, PAGES_PER_SPLIT):
        if FULL_SPLITS:
            page_id = first_page + page_offset
        else:
            page_id = tl.minimum(first_page + page_offset, last_page)
        global_n = (first_page + page_offset) * BLOCK_SIZE + offs_n
        valid_n = global_n < seq_len
        physical = tl.load(
            BLOCK_IDS + batch * BID_SB + page_id,
        ).to(tl.int64)
        k_ptrs = (
            K
            + physical * K_SBLOCK
            + offs_n[:, None] * K_STOKEN
            + hkv * K_SHEAD
            + offs_d[None, :] * K_SD
        )
        v_ptrs = (
            V
            + physical * V_SBLOCK
            + offs_n[:, None] * V_STOKEN
            + hkv * V_SHEAD
            + offs_d[None, :] * V_SD
        )
        if FULL_SPLITS:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=valid_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=valid_n[:, None], other=0.0)
        tl.store(
            GATHER_V
            + gv_base
            + (page_offset * BLOCK_SIZE + offs_n[:, None]) * D
            + offs_d[None, :],
            v,
        )
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(
            valid_row[:, None]
            & valid_n[None, :]
            & (global_n[None, :] <= query_pos[:, None]),
            scores,
            -float("inf"),
        )
        p = tl.exp2(tl.minimum(scores, 100.0))
        split_sum = split_sum * unit + tl.sum(p, axis=1)
        tl.store(
            SCORE_P
            + p_base
            + offs_r[:, None] * SP_SR
            + page_offset * BLOCK_SIZE
            + offs_n[None, :],
            p.to(tl.bfloat16),
        )
    tl.store(
        SCORE_SUM
        + batch * SS_SB
        + hkv * SS_SH
        + split_id * SS_SP
        + offs_r,
        split_sum,
    )


@triton.jit
def _bf16_mtp_pv_kernel(
    TASK_MAP,
    SCORE_P,
    SCORE_SUM,
    GATHER_V,
    KV_LENS,
    SPLIT_OUT,
    SPLIT_LSE,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    SP_SB: tl.constexpr,
    SP_SH: tl.constexpr,
    SP_SP: tl.constexpr,
    SP_SR: tl.constexpr,
    SS_SB: tl.constexpr,
    SS_SH: tl.constexpr,
    SS_SP: tl.constexpr,
    GV_SB: tl.constexpr,
    GV_SH: tl.constexpr,
    GV_SP: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    USE_TASK_MAP: tl.constexpr,
):
    """Finish MTP3/4 with one contiguous PV dot per split."""
    if USE_TASK_MAP:
        task = tl.program_id(0) * 3
        batch = tl.load(TASK_MAP + task)
        hkv = tl.load(TASK_MAP + task + 1)
        split_id = tl.load(TASK_MAP + task + 2)
    else:
        batch = tl.program_id(0)
        hkv = tl.program_id(1)
        split_id = tl.program_id(2)
    offs_r = tl.arange(0, Q_ROWS)
    offs_l = tl.arange(0, PAGES_PER_SPLIT * BLOCK_SIZE)
    offs_d = tl.arange(0, D)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (
        (seq_m < NUM_SEQ_Q)
        & (h_in_group < HEADS_PER_GROUP)
        & (hq < H_Q)
    )
    p = tl.load(
        SCORE_P
        + batch * SP_SB
        + hkv * SP_SH
        + split_id * SP_SP
        + offs_r[:, None] * SP_SR
        + offs_l[None, :]
    )
    vm = tl.load(
        GATHER_V
        + batch * GV_SB
        + hkv * GV_SH
        + split_id * GV_SP
        + offs_l[:, None] * D
        + offs_d[None, :]
    )
    unit = tl.load(KV_LENS + batch + offs_r * 0).to(tl.float32) * 0.0 + 1.0
    split_acc = tl.dot(p, vm) * unit[:, None]
    split_sum = tl.load(
        SCORE_SUM
        + batch * SS_SB
        + hkv * SS_SH
        + split_id * SS_SP
        + offs_r
    )
    has_value = split_sum > 0.0
    partial = tl.where(
        has_value[:, None],
        split_acc / tl.where(split_sum[:, None] > 0.0, split_sum[:, None], 1.0),
        0.0,
    )
    row = seq_m * H_Q + hq
    tl.store(
        SPLIT_OUT
        + batch * SO_SB
        + split_id * SO_SP
        + row[:, None] * SO_SR
        + offs_d[None, :],
        partial,
    )
    tl.store(
        SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        tl.where(
            has_value,
            tl.log2(tl.where(has_value, split_sum, 1.0)),
            -float("inf"),
        ),
        mask=valid_row,
    )


@triton.jit
def _bf16_mtp1_finalize_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    KV_LENS,
    OUT,
    HEADS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PAGES_PER_SPLIT: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Merge all split partials for one batch/GQA group."""
    batch = tl.program_id(0)
    hkv = tl.program_id(1)
    offs_h = tl.arange(0, HEADS_PER_GROUP)
    offs_d = tl.arange(0, D)
    hq = hkv * HEADS_PER_GROUP + offs_h
    num_splits = (
        tl.load(KV_LENS + batch) + BLOCK_SIZE * PAGES_PER_SPLIT - 1
    ) // (BLOCK_SIZE * PAGES_PER_SPLIT)
    running_max = tl.full((HEADS_PER_GROUP,), -float("inf"), tl.float32)
    running_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    accumulator = tl.zeros((HEADS_PER_GROUP, D), tl.float32)
    split_id = 0
    while split_id < num_splits:
        lse = tl.load(
            SPLIT_LSE + batch * SL_SB + split_id * SL_SP + hq,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + split_id * SO_SP
            + hq[:, None] * SO_SH
            + offs_d[None, :],
        )
        new_max = tl.maximum(running_max, lse)
        old_weight = tl.exp2(running_max - new_max)
        new_weight = tl.exp2(lse - new_max)
        accumulator = (
            accumulator * old_weight[:, None]
            + partial * new_weight[:, None]
        )
        running_sum = running_sum * old_weight + new_weight
        running_max = new_max
        split_id += 1
    tl.store(
        OUT + batch * O_SB + hq[:, None] * O_SH + offs_d[None, :],
        accumulator / running_sum[:, None],
    )


@triton.jit
def _bf16_mtp1_reduce_splits_kernel(
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    KV_LENS,
    REDUCED_OUT,
    REDUCED_LSE,
    OUT,
    HEADS_PER_GROUP: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    SPLITS_PER_REDUCTION: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    RO_SB: tl.constexpr,
    RO_SG: tl.constexpr,
    RO_SH: tl.constexpr,
    RL_SB: tl.constexpr,
    RL_SG: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Reduce a compact task's consecutive split range."""
    task = tl.program_id(0) * 3
    batch = tl.load(TASK_MAP + task)
    hkv = tl.load(TASK_MAP + task + 1)
    group = tl.load(TASK_MAP + task + 2)
    offs_h = tl.arange(0, HEADS_PER_GROUP)
    hq = hkv * HEADS_PER_GROUP + offs_h
    num_splits = (
        tl.load(KV_LENS + batch) + TOKENS_PER_SPLIT - 1
    ) // TOKENS_PER_SPLIT
    num_groups = (
        num_splits + SPLITS_PER_REDUCTION - 1
    ) // SPLITS_PER_REDUCTION
    first_split = group * SPLITS_PER_REDUCTION
    end_split = tl.minimum(first_split + SPLITS_PER_REDUCTION, num_splits)
    offs_d = tl.arange(0, D)
    running_max = tl.full((HEADS_PER_GROUP,), -float("inf"), tl.float32)
    running_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    accumulator = tl.zeros((HEADS_PER_GROUP, D), tl.float32)
    split_id = first_split
    while split_id < end_split:
        lse = tl.load(
            SPLIT_LSE + batch * SL_SB + split_id * SL_SP + hq,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + split_id * SO_SP
            + hq[:, None] * SO_SH
            + offs_d[None, :],
        )
        new_max = tl.maximum(running_max, lse)
        old_weight = tl.exp2(running_max - new_max)
        new_weight = tl.exp2(lse - new_max)
        accumulator = (
            accumulator * old_weight[:, None]
            + partial * new_weight[:, None]
        )
        running_sum = running_sum * old_weight + new_weight
        running_max = new_max
        split_id += 1
    combined = accumulator / running_sum[:, None]
    combined_lse = running_max + tl.log2(running_sum)
    if num_groups == 1:
        tl.store(
            OUT + batch * O_SB + hq[:, None] * O_SH + offs_d[None, :],
            combined,
        )
    else:
        tl.store(
            REDUCED_OUT
            + batch * RO_SB
            + group * RO_SG
            + hq[:, None] * RO_SH
            + offs_d[None, :],
            combined,
        )
        tl.store(
            REDUCED_LSE + batch * RL_SB + group * RL_SG + hq,
            combined_lse,
        )


@triton.jit
def _bf16_mtp1_finalize_groups_kernel(
    TASK_MAP,
    REDUCED_OUT,
    REDUCED_LSE,
    KV_LENS,
    OUT,
    HEADS_PER_GROUP: tl.constexpr,
    TOKENS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    RO_SB: tl.constexpr,
    RO_SG: tl.constexpr,
    RO_SH: tl.constexpr,
    RL_SB: tl.constexpr,
    RL_SG: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Finalize only heads whose first reduction produced multiple groups."""
    task = tl.program_id(0) * 2
    batch = tl.load(TASK_MAP + task)
    hkv = tl.load(TASK_MAP + task + 1)
    offs_h = tl.arange(0, HEADS_PER_GROUP)
    hq = hkv * HEADS_PER_GROUP + offs_h
    num_groups = (
        tl.load(KV_LENS + batch) + TOKENS_PER_GROUP - 1
    ) // TOKENS_PER_GROUP
    offs_d = tl.arange(0, D)
    running_max = tl.full((HEADS_PER_GROUP,), -float("inf"), tl.float32)
    running_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    accumulator = tl.zeros((HEADS_PER_GROUP, D), tl.float32)
    group = 0
    while group < num_groups:
        lse = tl.load(
            REDUCED_LSE + batch * RL_SB + group * RL_SG + hq,
        )
        partial = tl.load(
            REDUCED_OUT
            + batch * RO_SB
            + group * RO_SG
            + hq[:, None] * RO_SH
            + offs_d[None, :],
        )
        new_max = tl.maximum(running_max, lse)
        old_weight = tl.exp2(running_max - new_max)
        new_weight = tl.exp2(lse - new_max)
        accumulator = (
            accumulator * old_weight[:, None]
            + partial * new_weight[:, None]
        )
        running_sum = running_sum * old_weight + new_weight
        running_max = new_max
        group += 1
    tl.store(
        OUT + batch * O_SB + hq[:, None] * O_SH + offs_d[None, :],
        accumulator / running_sum[:, None],
    )


def _build_ascend_bf16_mtp1_schedule(
    inputs,
    pages_per_split: int = PAGES_PER_SPLIT,
) -> dict[str, object]:
    lengths = inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("each KV length must be positive")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    tokens_per_split = 64 * pages_per_split
    split_counts = [triton.cdiv(length, tokens_per_split) for length in lengths]
    max_splits = max(split_counts)
    compact_producer = any(count != max_splits for count in split_counts)
    full_producer_splits = all(
        length % tokens_per_split == 0 for length in lengths
    )
    producer_records = [
        [batch, kv_head, split]
        for batch, count in enumerate(split_counts)
        for split in range(count)
        for kv_head in range(hkv)
    ]
    group_counts = [
        triton.cdiv(count, SPLITS_PER_REDUCTION) for count in split_counts
    ]
    hierarchical_reduction = max_splits > SPLITS_PER_REDUCTION
    reduction_heads = hkv if inputs.mtp == 1 else hq
    reduce_records = [
        [batch, reduction_head, group]
        for batch, count in enumerate(group_counts)
        for group in range(count)
        for reduction_head in range(reduction_heads)
    ]
    final_records = [
        [batch, reduction_head]
        for batch, count in enumerate(group_counts)
        if count > 1
        for reduction_head in range(reduction_heads)
    ]
    device = inputs.q.device

    def records_tensor(records: list[list[int]], width: int) -> torch.Tensor:
        if not records:
            records = [[0] * width]
        return torch.tensor(records, dtype=torch.int32, device=device)

    return {
        "producer_task_map": records_tensor(producer_records, 3),
        "reduce_task_map": records_tensor(reduce_records, 3),
        "final_task_map": records_tensor(final_records, 2),
        "max_splits": max_splits,
        "max_reduction_groups": max(group_counts),
        "num_producer_tasks": len(producer_records),
        "num_reduce_tasks": len(reduce_records),
        "num_final_tasks": len(final_records),
        "compact_producer": compact_producer,
        "hierarchical_reduction": hierarchical_reduction,
        "full_producer_splits": full_producer_splits,
        "pages_per_split": pages_per_split,
    }


def _prepare_ascend_bf16_workspace(
    inputs,
    mtp: int,
    *,
    static_sched: bool = False,
) -> AscendBF16MTP1Workspace:
    batch = inputs.batch
    hq = int(inputs.q.shape[1])
    if inputs.mtp != mtp:
        raise ValueError(f"expected MTP={mtp}, got MTP={inputs.mtp}")
    if int(inputs.kv_lens.min().item()) < mtp:
        raise ValueError("each final KV length must be at least MTP")
    pages_per_split = _choose_pages_per_split(
        inputs.kv_lens.detach().cpu().to(torch.int64).tolist(), mtp
    )
    schedule = _build_ascend_bf16_mtp1_schedule(inputs, pages_per_split)
    if static_sched:
        # A static launch covers the rectangular batch/head/split grid.  Empty
        # tail programs leave through the producer's first-page guard instead
        # of consulting the compact dynamic task map.
        schedule["compact_producer"] = False
    max_splits = int(schedule["max_splits"])
    max_reduction_groups = int(schedule["max_reduction_groups"])
    # Pad the split buffers up to the highest row the producer can write so it
    # can store its partial tile unmasked (the masked store path is ~8x slower
    # on this backend).  The producer indexes rows as ``seq_m * hq + head``, so
    # with Q_ROWS wider than ``mtp * heads_per_group`` the invalid MTP rows also
    # land inside the buffer; finalisation only ever reads rows below ``mtp * hq``.
    hkv = int(inputs.k_cache.shape[2])
    heads_per_group = hq // hkv
    q_rows = triton.next_power_of_2(mtp * heads_per_group)
    padded_rows = (q_rows // heads_per_group) * hq
    rows = max(mtp * hq, padded_rows)
    partial_dtype = torch.bfloat16 if mtp == 1 else torch.float32
    return AscendBF16MTP1Workspace(
        producer_task_map=schedule["producer_task_map"],
        reduce_task_map=schedule["reduce_task_map"],
        final_task_map=schedule["final_task_map"],
        split_out=torch.empty(
            (batch, max_splits, rows, 128),
            dtype=partial_dtype,
            device=inputs.q.device,
        ),
        split_lse=torch.empty(
            (batch, max_splits, rows),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        reduced_out=torch.empty(
            (batch, max_reduction_groups, rows, 128),
            dtype=partial_dtype,
            device=inputs.q.device,
        ),
        reduced_lse=torch.empty(
            (batch, max_reduction_groups, rows),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        gather_v=torch.empty(
            (batch, hkv, max_splits, pages_per_split * 64, 128),
            dtype=inputs.v_cache.dtype,
            device=inputs.q.device,
        ),
        score_p=torch.empty(
            (
                (batch, hkv, max_splits, q_rows, pages_per_split * 64)
                if mtp >= 3 else (1,)
            ),
            dtype=inputs.q.dtype,
            device=inputs.q.device,
        ),
        score_sum=torch.empty(
            ((batch, hkv, max_splits, q_rows) if mtp >= 3 else (1,)),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        out=torch.empty(
            inputs.q.shape,
            dtype=inputs.q.dtype,
            device=inputs.q.device,
        ),
        max_splits=max_splits,
        max_reduction_groups=max_reduction_groups,
        num_producer_tasks=int(schedule["num_producer_tasks"]),
        num_reduce_tasks=int(schedule["num_reduce_tasks"]),
        num_final_tasks=int(schedule["num_final_tasks"]),
        compact_producer=bool(schedule["compact_producer"]),
        hierarchical_reduction=bool(schedule["hierarchical_reduction"]),
        full_producer_splits=bool(schedule["full_producer_splits"]),
        mtp=mtp,
        pages_per_split=pages_per_split,
    )


def prepare_ascend_bf16_mtp1_workspace(
    inputs, *, static_sched: bool = False,
) -> AscendBF16MTP1Workspace:
    return _prepare_ascend_bf16_workspace(inputs, 1, static_sched=static_sched)


def prepare_ascend_bf16_mtp2_workspace(
    inputs, *, static_sched: bool = False,
) -> AscendBF16MTP1Workspace:
    return _prepare_ascend_bf16_workspace(inputs, 2, static_sched=static_sched)


def prepare_ascend_bf16_mtp3_workspace(
    inputs, *, static_sched: bool = False,
) -> AscendBF16MTP1Workspace:
    return _prepare_ascend_bf16_workspace(inputs, 3, static_sched=static_sched)


def prepare_ascend_bf16_mtp4_workspace(
    inputs, *, static_sched: bool = False,
) -> AscendBF16MTP1Workspace:
    return _prepare_ascend_bf16_workspace(inputs, 4, static_sched=static_sched)


def refresh_ascend_bf16_mtp1_task_map(
    inputs,
    workspace: AscendBF16MTP1Workspace,
) -> None:
    schedule = _build_ascend_bf16_mtp1_schedule(inputs, workspace.pages_per_split)
    topology = (
        int(schedule["max_splits"]),
        int(schedule["max_reduction_groups"]),
        int(schedule["num_producer_tasks"]),
        int(schedule["num_reduce_tasks"]),
        int(schedule["num_final_tasks"]),
        bool(schedule["compact_producer"]),
        bool(schedule["hierarchical_reduction"]),
    )
    expected = (
        workspace.max_splits,
        workspace.max_reduction_groups,
        workspace.num_producer_tasks,
        workspace.num_reduce_tasks,
        workspace.num_final_tasks,
        workspace.compact_producer,
        workspace.hierarchical_reduction,
    )
    if topology != expected:
        raise ValueError("dynamic schedule topology changed; rebuild the workspace")
    for destination, name in (
        (workspace.producer_task_map, "producer_task_map"),
        (workspace.reduce_task_map, "reduce_task_map"),
        (workspace.final_task_map, "final_task_map"),
    ):
        source = schedule[name]
        if destination.shape != source.shape:
            raise ValueError("dynamic schedule capacity changed; rebuild the workspace")
        destination.copy_(source)
    workspace.full_producer_splits = bool(schedule["full_producer_splits"])


def attention_decode_ascend_bf16_mtp1(
    inputs,
    workspace: AscendBF16MTP1Workspace,
) -> torch.Tensor:
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    heads_per_group = hq // hkv
    producer_grid = (
        (workspace.num_producer_tasks,)
        if workspace.compact_producer
        else (inputs.batch, hkv, workspace.max_splits)
    )
    _bf16_mtp_gather_v_kernel[producer_grid](
        workspace.producer_task_map,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        workspace.gather_v,
        BLOCK_SIZE=64,
        D=128,
        PAGES_PER_SPLIT=workspace.pages_per_split,
        V_SBLOCK=inputs.v_cache.stride(0),
        V_STOKEN=inputs.v_cache.stride(1),
        V_SHEAD=inputs.v_cache.stride(2),
        V_SD=inputs.v_cache.stride(3),
        BID_SB=inputs.block_ids.stride(0),
        GK_SB=workspace.gather_v.stride(0),
        GK_SH=workspace.gather_v.stride(1),
        GK_SP=workspace.gather_v.stride(2),
        USE_TASK_MAP=workspace.compact_producer,
        num_warps=4,
        num_stages=1,
    )
    _bf16_mtp1_page_kernel[producer_grid](
        workspace.producer_task_map,
        inputs.q,
        inputs.k_cache,
        workspace.gather_v,
        inputs.block_ids,
        inputs.kv_lens,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        HEADS_PER_GROUP=heads_per_group,
        BLOCK_SIZE=64,
        D=128,
        PAGES_PER_SPLIT=workspace.pages_per_split,
        PAGES_PER_DOT=PAGES_PER_DOT,
        Q_SB=inputs.q.stride(0),
        Q_SH=inputs.q.stride(1),
        K_SBLOCK=inputs.k_cache.stride(0),
        K_STOKEN=inputs.k_cache.stride(1),
        K_SHEAD=inputs.k_cache.stride(2),
        K_SD=inputs.k_cache.stride(3),
        BID_SB=inputs.block_ids.stride(0),
        GK_SB=workspace.gather_v.stride(0),
        GK_SH=workspace.gather_v.stride(1),
        GK_SP=workspace.gather_v.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SP=workspace.split_out.stride(1),
        SO_SH=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SP=workspace.split_lse.stride(1),
        O_SB=workspace.out.stride(0),
        O_SH=workspace.out.stride(1),
        USE_TASK_MAP=workspace.compact_producer,
        DIRECT_OUTPUT=workspace.max_splits == 1,
        FULL_SPLITS=workspace.full_producer_splits,
        # Interleaved same-process A/B: 4 warps wins for the single-split
        # (direct-output) workload; 8 stays best for the split paths.
        num_warps=4 if workspace.max_splits == 1 else 8,
        num_stages=1,
    )
    if workspace.max_splits == 1:
        pass
    elif workspace.hierarchical_reduction:
        _bf16_mtp1_reduce_splits_kernel[(workspace.num_reduce_tasks,)](
            workspace.reduce_task_map,
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.reduced_out,
            workspace.reduced_lse,
            workspace.out,
            HEADS_PER_GROUP=heads_per_group,
            TOKENS_PER_SPLIT=64 * workspace.pages_per_split,
            SPLITS_PER_REDUCTION=SPLITS_PER_REDUCTION,
            D=128,
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SH=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            RO_SB=workspace.reduced_out.stride(0),
            RO_SG=workspace.reduced_out.stride(1),
            RO_SH=workspace.reduced_out.stride(2),
            RL_SB=workspace.reduced_lse.stride(0),
            RL_SG=workspace.reduced_lse.stride(1),
            O_SB=workspace.out.stride(0),
            O_SH=workspace.out.stride(1),
            num_warps=4,
            num_stages=1,
        )
        _bf16_mtp1_finalize_groups_kernel[(workspace.num_final_tasks,)](
            workspace.final_task_map,
            workspace.reduced_out,
            workspace.reduced_lse,
            inputs.kv_lens,
            workspace.out,
            HEADS_PER_GROUP=heads_per_group,
            TOKENS_PER_GROUP=(
                64 * workspace.pages_per_split * SPLITS_PER_REDUCTION
            ),
            D=128,
            RO_SB=workspace.reduced_out.stride(0),
            RO_SG=workspace.reduced_out.stride(1),
            RO_SH=workspace.reduced_out.stride(2),
            RL_SB=workspace.reduced_lse.stride(0),
            RL_SG=workspace.reduced_lse.stride(1),
            O_SB=workspace.out.stride(0),
            O_SH=workspace.out.stride(1),
            num_warps=4,
            num_stages=1,
        )
    else:
        _bf16_mtp1_finalize_kernel[(inputs.batch, hkv)](
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.out,
            HEADS_PER_GROUP=heads_per_group,
            BLOCK_SIZE=64,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            D=128,
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SH=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            O_SB=workspace.out.stride(0),
            O_SH=workspace.out.stride(1),
            num_warps=4,
            num_stages=1,
        )
    return workspace.out


def _attention_decode_ascend_bf16_mtp(
    inputs,
    workspace: AscendBF16MTP1Workspace,
    mtp: int,
    q_rows: int,
) -> torch.Tensor:
    if workspace.mtp != mtp or inputs.mtp != mtp:
        raise ValueError(f"MTP{mtp} attention requires an MTP{mtp} workspace")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    heads_per_group = hq // hkv
    producer_grid = (
        (workspace.num_producer_tasks,)
        if workspace.compact_producer
        else (inputs.batch, hkv, workspace.max_splits)
    )
    if mtp == 2:
        _bf16_mtp_gather_v_kernel[producer_grid](
            workspace.producer_task_map,
            inputs.v_cache,
            inputs.block_ids,
            inputs.kv_lens,
            workspace.gather_v,
            BLOCK_SIZE=64,
            D=128,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            V_SBLOCK=inputs.v_cache.stride(0),
            V_STOKEN=inputs.v_cache.stride(1),
            V_SHEAD=inputs.v_cache.stride(2),
            V_SD=inputs.v_cache.stride(3),
            BID_SB=inputs.block_ids.stride(0),
            GK_SB=workspace.gather_v.stride(0),
            GK_SH=workspace.gather_v.stride(1),
            GK_SP=workspace.gather_v.stride(2),
            USE_TASK_MAP=workspace.compact_producer,
            num_warps=4,
            num_stages=1,
        )
        _bf16_mtp_page_kernel[producer_grid](
            workspace.producer_task_map,
            inputs.q,
            inputs.k_cache,
            workspace.gather_v,
            inputs.block_ids,
            inputs.kv_lens,
            workspace.split_out,
            workspace.split_lse,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            BLOCK_SIZE=64,
            D=128,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            Q_SB=inputs.q.stride(0),
            Q_SH=inputs.q.stride(1),
            K_SBLOCK=inputs.k_cache.stride(0),
            K_STOKEN=inputs.k_cache.stride(1),
            K_SHEAD=inputs.k_cache.stride(2),
            K_SD=inputs.k_cache.stride(3),
            BID_SB=inputs.block_ids.stride(0),
            GV_SB=workspace.gather_v.stride(0),
            GV_SH=workspace.gather_v.stride(1),
            GV_SP=workspace.gather_v.stride(2),
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SR=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            USE_TASK_MAP=workspace.compact_producer,
            FULL_SPLITS=workspace.full_producer_splits,
            num_warps=4 if workspace.max_splits == 1 else 8,
            num_stages=1,
        )
    else:
        _bf16_mtp_qk_gather_v_kernel[producer_grid](
            workspace.producer_task_map,
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            inputs.kv_lens,
            workspace.gather_v,
            workspace.score_p,
            workspace.score_sum,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            BLOCK_SIZE=64,
            D=128,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            Q_SB=inputs.q.stride(0),
            Q_SH=inputs.q.stride(1),
            K_SBLOCK=inputs.k_cache.stride(0),
            K_STOKEN=inputs.k_cache.stride(1),
            K_SHEAD=inputs.k_cache.stride(2),
            K_SD=inputs.k_cache.stride(3),
            V_SBLOCK=inputs.v_cache.stride(0),
            V_STOKEN=inputs.v_cache.stride(1),
            V_SHEAD=inputs.v_cache.stride(2),
            V_SD=inputs.v_cache.stride(3),
            BID_SB=inputs.block_ids.stride(0),
            GV_SB=workspace.gather_v.stride(0),
            GV_SH=workspace.gather_v.stride(1),
            GV_SP=workspace.gather_v.stride(2),
            SP_SB=workspace.score_p.stride(0),
            SP_SH=workspace.score_p.stride(1),
            SP_SP=workspace.score_p.stride(2),
            SP_SR=workspace.score_p.stride(3),
            SS_SB=workspace.score_sum.stride(0),
            SS_SH=workspace.score_sum.stride(1),
            SS_SP=workspace.score_sum.stride(2),
            USE_TASK_MAP=workspace.compact_producer,
            FULL_SPLITS=workspace.full_producer_splits,
            num_warps=4 if workspace.max_splits == 1 else 8,
            num_stages=1,
        )
        _bf16_mtp_pv_kernel[producer_grid](
            workspace.producer_task_map,
            workspace.score_p,
            workspace.score_sum,
            workspace.gather_v,
            inputs.kv_lens,
            workspace.split_out,
            workspace.split_lse,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            BLOCK_SIZE=64,
            D=128,
            PAGES_PER_SPLIT=workspace.pages_per_split,
            SP_SB=workspace.score_p.stride(0),
            SP_SH=workspace.score_p.stride(1),
            SP_SP=workspace.score_p.stride(2),
            SP_SR=workspace.score_p.stride(3),
            SS_SB=workspace.score_sum.stride(0),
            SS_SH=workspace.score_sum.stride(1),
            SS_SP=workspace.score_sum.stride(2),
            GV_SB=workspace.gather_v.stride(0),
            GV_SH=workspace.gather_v.stride(1),
            GV_SP=workspace.gather_v.stride(2),
            SO_SB=workspace.split_out.stride(0),
            SO_SP=workspace.split_out.stride(1),
            SO_SR=workspace.split_out.stride(2),
            SL_SB=workspace.split_lse.stride(0),
            SL_SP=workspace.split_lse.stride(1),
            USE_TASK_MAP=workspace.compact_producer,
            num_warps=4 if workspace.max_splits == 1 else 8,
            num_stages=1,
        )
    common = dict(
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        D=128,
        SO_SB=workspace.split_out.stride(0),
        SO_SP=workspace.split_out.stride(1),
        SO_SR=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SP=workspace.split_lse.stride(1),
        O_SB=workspace.out.stride(0),
        O_SH=workspace.out.stride(1),
    )
    if workspace.hierarchical_reduction:
        _bf16_mtp_reduce_splits_kernel[(workspace.num_reduce_tasks * mtp,)](
            workspace.reduce_task_map,
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.reduced_out,
            workspace.reduced_lse,
            workspace.out,
            TOKENS_PER_SPLIT=64 * workspace.pages_per_split,
            SPLITS_PER_REDUCTION=SPLITS_PER_REDUCTION,
            RO_SB=workspace.reduced_out.stride(0),
            RO_SG=workspace.reduced_out.stride(1),
            RO_SR=workspace.reduced_out.stride(2),
            RL_SB=workspace.reduced_lse.stride(0),
            RL_SG=workspace.reduced_lse.stride(1),
            num_warps=4,
            num_stages=1,
            **common,
        )
        _bf16_mtp_finalize_groups_kernel[(workspace.num_final_tasks * mtp,)](
            workspace.final_task_map,
            workspace.reduced_out,
            workspace.reduced_lse,
            inputs.kv_lens,
            workspace.out,
            TOKENS_PER_GROUP=(
                64 * workspace.pages_per_split * SPLITS_PER_REDUCTION
            ),
            RO_SB=workspace.reduced_out.stride(0),
            RO_SG=workspace.reduced_out.stride(1),
            RO_SR=workspace.reduced_out.stride(2),
            RL_SB=workspace.reduced_lse.stride(0),
            RL_SG=workspace.reduced_lse.stride(1),
            NUM_SEQ_Q=common["NUM_SEQ_Q"],
            H_Q=common["H_Q"],
            D=common["D"],
            O_SB=common["O_SB"],
            O_SH=common["O_SH"],
            num_warps=4,
            num_stages=1,
        )
    else:
        _bf16_mtp_finalize_kernel[(inputs.batch, mtp * hq)](
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.out,
            TOKENS_PER_SPLIT=64 * workspace.pages_per_split,
            MAX_SPLITS=SPLITS_PER_REDUCTION,
            num_warps=4,
            num_stages=1,
            **common,
        )
    return workspace.out


def attention_decode_ascend_bf16_mtp2(
    inputs,
    workspace: AscendBF16MTP1Workspace,
) -> torch.Tensor:
    return _attention_decode_ascend_bf16_mtp(inputs, workspace, 2, 16)


def attention_decode_ascend_bf16_mtp3(
    inputs,
    workspace: AscendBF16MTP1Workspace,
) -> torch.Tensor:
    return _attention_decode_ascend_bf16_mtp(inputs, workspace, 3, 32)


def attention_decode_ascend_bf16_mtp4(
    inputs,
    workspace: AscendBF16MTP1Workspace,
) -> torch.Tensor:
    return _attention_decode_ascend_bf16_mtp(inputs, workspace, 4, 32)


__all__ = [
    "HAS_TLE",
    "USE_TLE",
    "AscendBF16MTP1Workspace",
    "attention_decode_ascend_bf16_mtp1",
    "attention_decode_ascend_bf16_mtp2",
    "attention_decode_ascend_bf16_mtp3",
    "attention_decode_ascend_bf16_mtp4",
    "prepare_ascend_bf16_mtp1_workspace",
    "prepare_ascend_bf16_mtp2_workspace",
    "prepare_ascend_bf16_mtp3_workspace",
    "prepare_ascend_bf16_mtp4_workspace",
    "refresh_ascend_bf16_mtp1_task_map",
]
