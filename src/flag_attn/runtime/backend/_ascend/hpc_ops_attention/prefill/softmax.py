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
    _FP8_P_SCALE_JIT,
    _PREFILL_VEC_ROWS_JIT,
)

from .quant import (
    _positive_e4m3_round,
)


@triton.jit
def _fp8_prefill_softmax_splits_kernel(
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
    """One program does (row, HEADS_PER_PROG heads) for all splits of this stage.

    Bit-identical to the per-split version (each split's softmax is independent),
    with MAX_SPLITS fewer instances -- mostly program count x fixed overhead.
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
    # HEADS_PER_PROG == 1 unrolls this loop into the original code, byte-identical.
    # HPP_UNROLL=HEADS_PER_PROG is the old static_range unroll; =1 is a runtime
    # loop. Same-process A/B rotation (NPUGraph replay, median, bit-identical):
    #   q=128/kv=1152 (VEC 2048) -10.6%    q=9  /kv=129  (VEC  256)  -6.7%
    #   q=129/kv=257  (VEC  512)  -5.1%    q=127/kv=127  (VEC  128)  -5.2%
    #   q=512/kv=2048 (VEC 2048)  -2.0%    q=17 /kv=8193 (VEC 4096)  -0.6%
    #   other VEC=4096 shapes (q=512/kv=4096, 8192, 16385, 32768, masked and
    #   unmasked): -0.1% ~ +0.4%, inconsistent direction, i.e. noise.
    # Narrow tiles pay unroll size/slot overhead; a 4096-wide tile unrolls so the
    # compiler schedules across iterations, so pick by VEC: >=4096 fully unrolled.
    for q_head_off in tl.range(
        0, HEADS_PER_PROG, loop_unroll_factor=HPP_UNROLL
    ):
      q_head = q_head_base + q_head_off
      # Split loop is runtime too: MAX_SPLITS now reaches 8~16 and the head loop
      # is already runtime, so their product bloats the tile body many-fold.
      # Same-process A/B (bit-identical): 512/32768 masked -1.00% / unmasked -1.39%,
      # 512/4096 -0.40%/-0.84%; <0.3ms shapes are capture noise (+2.7% = 0.09ms).
      for step in tl.range(0, MAX_SPLITS):
        split_id = SPLIT_BASE + step
        split_begin = split_id * TOKENS_PER_SPLIT
        if UNMASKED:
            # Whole split, stage divides splits: no invalid lanes, so no mask is
            # needed. Masked stores are clearly costlier here (see bf16 decode), and
            # VEC_ROWS==1 early return already guarantees valid rows; valid_m is dead.
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
                _positive_e4m3_round(probabilities * _FP8_P_SCALE_JIT),
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
            # Invalid lanes have scores = load other=-inf, exp2(-inf)=0; invalid
            # rows/lanes are store-masked by valid_m/valid_n anyway, so this whole-
            # tile where is redundant: removing it at q=512/kv=16385 measured -3.9%
            # (softmax 4.317 -> 4.147ms); output/probs/sums all bit-identical.
            probabilities = tl.exp2(scores - safe_max[:, None])
            denominator = tl.sum(probabilities, axis=1)
            tl.store(
                PROBABILITIES + batch * P_SBATCH + step * P_SSPLIT
                + q_local[:, None] * P_SQ + q_head * P_SHEAD + offs_n[None, :],
                _positive_e4m3_round(probabilities * _FP8_P_SCALE_JIT),
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
