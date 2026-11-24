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

"""Online-softmax kernel used when the prefill is split over KV."""

import triton
import triton.language as tl

from .constants import (
    _PREFILL_VEC_ROWS_JIT,
)


@triton.jit
def _bf16_prefill_softmax_splits_kernel(
    SCORES,
    PROBABILITIES,
    CU_SEQLENS_Q,
    KV_LENS,
    SPLIT_LSE,
    SPLIT_SUM,
    SPLIT_BASE,
    Q_BASE,
    S_SBATCH: tl.constexpr,
    S_SSPLIT: tl.constexpr,
    S_SQ: tl.constexpr,
    S_SHEAD: tl.constexpr,
    P_SBATCH: tl.constexpr,
    P_SSPLIT: tl.constexpr,
    P_SQ: tl.constexpr,
    P_SHEAD: tl.constexpr,
    SL_SBATCH: tl.constexpr,
    SL_SSPLIT: tl.constexpr,
    SL_SQ: tl.constexpr,
    SS_SBATCH: tl.constexpr,
    SS_SSPLIT: tl.constexpr,
    SS_SQ: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    VEC: tl.constexpr,
    EXACT: tl.constexpr,
    UNMASKED: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    HPP_UNROLL: tl.constexpr,
):
    """Compute each row/head group's softmax statistics over the stage's splits.

    Each split has independent normalization state and a separate output slot.
    """
    q_local = Q_BASE + tl.program_id(0) * _PREFILL_VEC_ROWS_JIT + tl.arange(
        0, _PREFILL_VEC_ROWS_JIT
    )
    q_head_base = tl.program_id(1) * HEADS_PER_PROG
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    kv_len = tl.load(KV_LENS + batch)
    valid_m = q_local < q_end - q_begin
    if (Q_BASE + tl.program_id(0) * _PREFILL_VEC_ROWS_JIT >= q_end - q_begin):
        return
    offs_n = tl.arange(0, VEC)
    # HPP_UNROLL controls head-loop unrolling; a runtime loop bounds code size
    # when a program processes several heads.
    for q_head_off in tl.range(
        0, HEADS_PER_PROG, loop_unroll_factor=HPP_UNROLL
    ):
      q_head = q_head_base + q_head_off
      # Iterate splits dynamically to avoid multiplying the head-loop code size.
      # A final stage can contain fewer splits than the scores buffer has slots.
      # Do not store LSE/sums past the allocated split buffers.
      num_steps = tl.minimum(
          MAX_SPLITS,
          tl.maximum(0, (kv_len + TOKENS_PER_SPLIT - 1) // TOKENS_PER_SPLIT - SPLIT_BASE),
      )
      for step in tl.range(0, num_steps):
        split_id = SPLIT_BASE + step
        split_begin = split_id * TOKENS_PER_SPLIT
        if UNMASKED:
            # Preparation guarantees valid rows and full splits for this path.
            scores = tl.load(
                SCORES + batch * S_SBATCH + step * S_SSPLIT
                + q_local[:, None] * S_SQ + q_head * S_SHEAD + offs_n[None, :]
            ).to(tl.float32)
            maximum = tl.max(scores, axis=1)
            # A fully block-masked row has max -inf; clamp it (else exp2(nan))
            safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
            probabilities = tl.exp2(scores - safe_max[:, None])
            denominator = tl.sum(probabilities, axis=1)
            tl.store(
                PROBABILITIES + batch * P_SBATCH + step * P_SSPLIT
                + q_local[:, None] * P_SQ + q_head * P_SHEAD + offs_n[None, :],
                probabilities.to(tl.bfloat16),
            )
            tl.store(
                SPLIT_LSE + batch * SL_SBATCH + split_id * SL_SSPLIT
                + q_local * SL_SQ + q_head,
                tl.where(
                    denominator > 0.0,
                    maximum + tl.log2(denominator),
                    -float("inf"),
                ),
            )
            tl.store(
                SPLIT_SUM + batch * SS_SBATCH + split_id * SS_SSPLIT
                + q_local * SS_SQ + q_head,
                denominator,
            )
        else:
            if EXACT:
                valid_n = split_begin + offs_n < kv_len
            else:
                valid_n = (split_begin + offs_n < kv_len) & (
                    offs_n < TOKENS_PER_SPLIT
                )
            scores = tl.load(
                SCORES + batch * S_SBATCH + step * S_SSPLIT
                + q_local[:, None] * S_SQ + q_head * S_SHEAD + offs_n[None, :],
                mask=valid_m[:, None] & valid_n[None, :],
                other=-float("inf"),
            ).to(tl.float32)
            maximum = tl.max(scores, axis=1)
            safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
            # Invalid scores load as -inf and exponentiate to zero. Store masks
            # exclude invalid rows and vector-overhang lanes.
            probabilities = tl.exp2(scores - safe_max[:, None])
            denominator = tl.sum(probabilities, axis=1)
            tl.store(
                PROBABILITIES + batch * P_SBATCH + step * P_SSPLIT
                + q_local[:, None] * P_SQ + q_head * P_SHEAD + offs_n[None, :],
                probabilities.to(tl.bfloat16),
                mask=valid_m[:, None] & valid_n[None, :],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SBATCH + split_id * SL_SSPLIT
                + q_local * SL_SQ + q_head,
                tl.where(
                    denominator > 0.0,
                    maximum + tl.log2(denominator),
                    -float("inf"),
                ),
                mask=valid_m,
            )
            tl.store(
                SPLIT_SUM + batch * SS_SBATCH + split_id * SS_SSPLIT
                + q_local * SS_SQ + q_head,
                denominator,
                mask=valid_m,
            )
