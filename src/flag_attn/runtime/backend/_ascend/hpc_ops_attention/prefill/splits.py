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

"""Split and finalize kernels that merge the per-split prefill partials."""

import triton
import triton.language as tl

from .constants import (
    _HEAD_DIM_JIT,
)


@triton.jit
def _bf16_prefill_finalize_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    CU_SEQLENS_Q,
    KV_LENS,
    OUT,
    Q_BASE,
    SO_SBATCH: tl.constexpr,
    SO_SSPLIT: tl.constexpr,
    SO_SQ: tl.constexpr,
    SO_SHEAD: tl.constexpr,
    SL_SBATCH: tl.constexpr,
    SL_SSPLIT: tl.constexpr,
    SL_SQ: tl.constexpr,
    O_STOKEN: tl.constexpr,
    O_SHEAD: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Merge every split partial of one query head into the output.

    One program handles ``ROWS`` query rows: it loads the
    ``(MAX_SPLITS, ROWS, D)`` tile, reweights each split by its log-sum-exp and
    reduces over the split axis.  Several independent rows in flight let the
    compiler overlap the loads with the reduction and hide their fixed latency.
    """
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    num_splits = (
        tl.load(KV_LENS + batch) + TOKENS_PER_SPLIT - 1
    ) // TOKENS_PER_SPLIT
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    if ROWS > 1:
        # One program handles ROWS rows: tile (MAX_SPLITS, ROWS, D), independent
        # chains the compiler overlaps loads/reductions on, hiding fixed latency.
        offs_r = tl.arange(0, ROWS)
        q_locals = Q_BASE + tl.program_id(0) * ROWS + offs_r
        valid_m = q_locals < q_len
        offs_s = tl.arange(0, MAX_SPLITS)
        valid_s = offs_s < num_splits
        lse = tl.load(
            SPLIT_LSE
            + batch * SL_SBATCH
            + offs_s[:, None] * SL_SSPLIT
            + q_locals[None, :] * SL_SQ
            + q_head,
            mask=valid_s[:, None] & valid_m[None, :],
            other=-float("inf"),
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SBATCH
            + offs_s[:, None, None] * SO_SSPLIT
            + q_locals[None, :, None] * SO_SQ
            + q_head * SO_SHEAD
            + offs_d[None, None, :],
            mask=valid_s[:, None, None] & valid_m[None, :, None],
            other=0.0,
        )
        merged_max = tl.max(lse, axis=0)
        safe_max = tl.where(merged_max == -float("inf"), 0.0, merged_max)
        weights = tl.where(
            valid_s[:, None] & (lse != -float("inf")),
            tl.exp2(lse - safe_max[None, :]),
            0.0,
        )
        denominator = tl.sum(weights, axis=0)
        numerator = tl.sum(partial * weights[:, :, None], axis=0)
        result = tl.where(
            denominator[:, None] > 0.0,
            numerator / tl.where(denominator[:, None] > 0.0, denominator[:, None], 1.0),
            0.0,
        )
        tl.store(
            OUT
            + (q_begin + q_locals[:, None]) * O_STOKEN
            + q_head * O_SHEAD
            + offs_d[None, :],
            result.to(OUT.dtype.element_ty),
            mask=valid_m[:, None],
        )
        return
    q_local = Q_BASE + tl.program_id(0)
    if q_local >= q_len:
        return
    if MAX_SPLITS <= 16:
        offs_s = tl.arange(0, MAX_SPLITS)
        valid_s = offs_s < num_splits
        lse = tl.load(
            SPLIT_LSE + batch * SL_SBATCH + offs_s * SL_SSPLIT + q_local * SL_SQ + q_head,
            mask=valid_s,
            other=-float("inf"),
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SBATCH
            + offs_s[:, None] * SO_SSPLIT
            + q_local * SO_SQ
            + q_head * SO_SHEAD
            + offs_d[None, :],
            mask=valid_s[:, None],
            other=0.0,
        )
        merged_max = tl.max(lse, axis=0)
        safe_max = tl.where(merged_max == -float("inf"), 0.0, merged_max)
        weights = tl.where(valid_s & (lse != -float("inf")), tl.exp2(lse - safe_max), 0.0)
        denominator = tl.sum(weights, axis=0)
        numerator = tl.sum(partial * weights[:, None], axis=0)
        result = tl.where(
            denominator > 0.0,
            numerator / tl.where(denominator > 0.0, denominator, 1.0),
            0.0,
        )
    else:
        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.zeros((), tl.float32)
        accumulator = tl.zeros((_HEAD_DIM_JIT,), tl.float32)
        for split_id in tl.static_range(0, MAX_SPLITS):
            if split_id < num_splits:
                lse = tl.load(
                    SPLIT_LSE
                    + batch * SL_SBATCH
                    + split_id * SL_SSPLIT
                    + q_local * SL_SQ
                    + q_head
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_SBATCH
                    + split_id * SO_SSPLIT
                    + q_local * SO_SQ
                    + q_head * SO_SHEAD
                    + offs_d
                )
                new_max = tl.maximum(running_max, lse)
                safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
                old_weight = tl.where(
                    running_max == -float("inf"),
                    0.0,
                    tl.exp2(running_max - safe_max),
                )
                new_weight = tl.where(
                    lse == -float("inf"), 0.0, tl.exp2(lse - safe_max)
                )
                accumulator = accumulator * old_weight + partial * new_weight
                running_sum = running_sum * old_weight + new_weight
                running_max = new_max
        result = tl.where(
            running_sum > 0.0,
            accumulator / tl.where(running_sum > 0.0, running_sum, 1.0),
            0.0,
        )
    tl.store(OUT + (q_begin + q_local) * O_STOKEN + q_head * O_SHEAD + offs_d, result)
