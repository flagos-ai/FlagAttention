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

"""MTP split reduction chain shared by the BF16 and the FP8 decode kernels.

Both decode implementations split a long KV into partial outputs plus a
per-split log-sum-exp and then merge them.  The merge chain is three kernels: one
reduces a group of splits into one partial, one merges the group partials, and
one merges everything for a single MTP query and Q head.

They are format-agnostic -- they only read and write the BF16 split buffers --
so the FP8 path launches them unchanged, and the group width is a ``constexpr``
argument rather than a module constant: the BF16 path passes
``SPLITS_PER_REDUCTION`` (16), the FP8 path passes ``FP8_SPLITS_PER_REDUCTION``
(64) because its own split count is much larger.

The kernels used to live inside the BF16 decode module, which forced
``decode/fp8.py`` to import three private kernels out of ``decode/__init__.py``.
They keep their ``_bf16_`` names and their source text so that both their
provenance and their Triton cache keys stay unchanged.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _bf16_mtp_finalize_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    KV_LENS,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Merge all split partials for one MTP query and Q head."""
    batch = tl.program_id(0)
    query_head = tl.program_id(1)
    seq_m = query_head // H_Q
    hq = query_head - seq_m * H_Q
    offs_d = tl.arange(0, D)
    row = seq_m * H_Q + hq
    num_splits = (
        tl.load(KV_LENS + batch) + TOKENS_PER_SPLIT - 1
    ) // TOKENS_PER_SPLIT
    if (num_splits > 1) & (num_splits <= MAX_SPLITS):
        # Vectorised merge: one masked vector load of the split LSEs plus one
        # masked tile load of the partials, instead of a serial scalar load per
        # split (which costs a GM scalar round trip per iteration).
        offs_s = tl.arange(0, MAX_SPLITS)
        s_valid = offs_s < num_splits
        lse_vec = tl.load(
            SPLIT_LSE + batch * SL_SB + offs_s * SL_SP + row,
            mask=s_valid,
            other=-float("inf"),
        )
        part = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + offs_s[:, None] * SO_SP
            + row * SO_SR
            + offs_d[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        merged_max = tl.max(lse_vec, axis=0)
        weight = tl.where(
            lse_vec != -float("inf"),
            tl.exp2(lse_vec - tl.where(merged_max != -float("inf"), merged_max, 0.0)),
            0.0,
        )
        den = tl.sum(weight, axis=0)
        num = tl.sum(part * weight[:, None], axis=0)
        tl.store(
            OUT
            + (batch * NUM_SEQ_Q + seq_m) * O_SB
            + hq * O_SH
            + offs_d,
            tl.where(den > 0.0, num / tl.where(den > 0.0, den, 1.0), 0.0),
        )
        return
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((D,), tl.float32)
    split_id = 0
    while split_id < num_splits:
        lse = tl.load(
            SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + split_id * SO_SP
            + row * SO_SR
            + offs_d,
        )
        new_max = tl.maximum(running_max, lse)
        has_value = new_max != -float("inf")
        safe_max = tl.where(has_value, new_max, 0.0)
        old_weight = tl.where(
            running_max != -float("inf"),
            tl.exp2(running_max - safe_max),
            0.0,
        )
        new_weight = tl.where(
            lse != -float("inf"),
            tl.exp2(lse - safe_max),
            0.0,
        )
        accumulator = (
            accumulator * old_weight
            + partial * new_weight
        )
        running_sum = running_sum * old_weight + new_weight
        running_max = new_max
        split_id += 1
    tl.store(
        OUT
        + (batch * NUM_SEQ_Q + seq_m) * O_SB
        + hq * O_SH
        + offs_d,
        accumulator / running_sum,
    )


@triton.jit
def _bf16_mtp_reduce_splits_kernel(
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    KV_LENS,
    REDUCED_OUT,
    REDUCED_LSE,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    SPLITS_PER_REDUCTION: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SP: tl.constexpr,
    SO_SR: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SP: tl.constexpr,
    RO_SB: tl.constexpr,
    RO_SG: tl.constexpr,
    RO_SR: tl.constexpr,
    RL_SB: tl.constexpr,
    RL_SG: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Reduce up to sixteen split partials for one MTP query."""
    program = tl.program_id(0)
    seq_m = program % NUM_SEQ_Q
    task = (program // NUM_SEQ_Q) * 3
    batch = tl.load(TASK_MAP + task)
    hq = tl.load(TASK_MAP + task + 1)
    group = tl.load(TASK_MAP + task + 2)
    num_splits = (
        tl.load(KV_LENS + batch) + TOKENS_PER_SPLIT - 1
    ) // TOKENS_PER_SPLIT
    num_groups = (
        num_splits + SPLITS_PER_REDUCTION - 1
    ) // SPLITS_PER_REDUCTION
    first_split = group * SPLITS_PER_REDUCTION
    end_split = tl.minimum(first_split + SPLITS_PER_REDUCTION, num_splits)
    offs_d = tl.arange(0, D)
    row = seq_m * H_Q + hq
    group_len = end_split - first_split
    if (group_len > 1) & (group_len <= SPLITS_PER_REDUCTION):
        # Vectorised group merge: one masked LSE vector load plus one masked
        # [SPLITS_PER_REDUCTION, D] partial load instead of a serial scalar
        # load per split.
        offs_s = tl.arange(0, SPLITS_PER_REDUCTION)
        slot = first_split + offs_s
        s_valid = slot < end_split
        lse_vec = tl.load(
            SPLIT_LSE + batch * SL_SB + slot * SL_SP + row,
            mask=s_valid,
            other=-float("inf"),
        )
        part = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + slot[:, None] * SO_SP
            + row * SO_SR
            + offs_d[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        merged_max = tl.max(lse_vec, axis=0)
        safe_max = tl.where(merged_max != -float("inf"), merged_max, 0.0)
        weight = tl.where(
            lse_vec != -float("inf"), tl.exp2(lse_vec - safe_max), 0.0
        )
        den = tl.sum(weight, axis=0)
        num = tl.sum(part * weight[:, None], axis=0)
        has_value = den > 0.0
        combined = tl.where(
            has_value, num / tl.where(has_value, den, 1.0), 0.0
        )
        combined_lse = tl.where(
            has_value,
            safe_max + tl.log2(tl.where(has_value, den, 1.0)),
            -float("inf"),
        )
        if num_groups == 1:
            tl.store(
                OUT
                + (batch * NUM_SEQ_Q + seq_m) * O_SB
                + hq * O_SH
                + offs_d,
                combined,
            )
        else:
            tl.store(
                REDUCED_OUT
                + batch * RO_SB
                + group * RO_SG
                + row * RO_SR
                + offs_d,
                combined,
            )
            tl.store(
                REDUCED_LSE + batch * RL_SB + group * RL_SG + row,
                combined_lse,
            )
        return
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((D,), tl.float32)
    split_id = first_split
    while split_id < end_split:
        lse = tl.load(
            SPLIT_LSE + batch * SL_SB + split_id * SL_SP + row,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + split_id * SO_SP
            + row * SO_SR
            + offs_d,
        )
        new_max = tl.maximum(running_max, lse)
        has_value = new_max != -float("inf")
        safe_max = tl.where(has_value, new_max, 0.0)
        old_weight = tl.where(
            running_max != -float("inf"),
            tl.exp2(running_max - safe_max),
            0.0,
        )
        new_weight = tl.where(
            lse != -float("inf"),
            tl.exp2(lse - safe_max),
            0.0,
        )
        accumulator = (
            accumulator * old_weight
            + partial * new_weight
        )
        running_sum = running_sum * old_weight + new_weight
        running_max = new_max
        split_id += 1
    has_value = running_sum > 0.0
    combined = tl.where(
        has_value,
        accumulator / tl.where(has_value, running_sum, 1.0),
        0.0,
    )
    combined_lse = tl.where(
        has_value,
        running_max + tl.log2(tl.where(has_value, running_sum, 1.0)),
        -float("inf"),
    )
    if num_groups == 1:
        tl.store(
            OUT
            + (batch * NUM_SEQ_Q + seq_m) * O_SB
            + hq * O_SH
            + offs_d,
            combined,
        )
    else:
        tl.store(
            REDUCED_OUT
            + batch * RO_SB
            + group * RO_SG
            + row * RO_SR
            + offs_d,
            combined,
        )
        tl.store(
            REDUCED_LSE + batch * RL_SB + group * RL_SG + row,
            combined_lse,
        )


@triton.jit
def _bf16_mtp_finalize_groups_kernel(
    TASK_MAP,
    REDUCED_OUT,
    REDUCED_LSE,
    KV_LENS,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    TOKENS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    RO_SB: tl.constexpr,
    RO_SG: tl.constexpr,
    RO_SR: tl.constexpr,
    RL_SB: tl.constexpr,
    RL_SG: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Finalize one query whose first reduction made many groups."""
    program = tl.program_id(0)
    seq_m = program % NUM_SEQ_Q
    task = (program // NUM_SEQ_Q) * 2
    batch = tl.load(TASK_MAP + task)
    hq = tl.load(TASK_MAP + task + 1)
    num_groups = (
        tl.load(KV_LENS + batch) + TOKENS_PER_GROUP - 1
    ) // TOKENS_PER_GROUP
    offs_d = tl.arange(0, D)
    row = seq_m * H_Q + hq
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.zeros((), tl.float32)
    accumulator = tl.zeros((D,), tl.float32)
    group = 0
    while group < num_groups:
        lse = tl.load(
            REDUCED_LSE + batch * RL_SB + group * RL_SG + row,
        )
        partial = tl.load(
            REDUCED_OUT
            + batch * RO_SB
            + group * RO_SG
            + row * RO_SR
            + offs_d,
        )
        new_max = tl.maximum(running_max, lse)
        has_value = new_max != -float("inf")
        safe_max = tl.where(has_value, new_max, 0.0)
        old_weight = tl.where(
            running_max != -float("inf"),
            tl.exp2(running_max - safe_max),
            0.0,
        )
        new_weight = tl.where(
            lse != -float("inf"),
            tl.exp2(lse - safe_max),
            0.0,
        )
        accumulator = (
            accumulator * old_weight
            + partial * new_weight
        )
        running_sum = running_sum * old_weight + new_weight
        running_max = new_max
        group += 1
    tl.store(
        OUT
        + (batch * NUM_SEQ_Q + seq_m) * O_SB
        + hq * O_SH
        + offs_d,
        accumulator / running_sum,
    )


__all__ = [
    "_bf16_mtp_finalize_groups_kernel",
    "_bf16_mtp_finalize_kernel",
    "_bf16_mtp_reduce_splits_kernel",
]
