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
    _BLOCK_N_JIT,
    _FP8_P_SCALE_JIT,
    _HEAD_DIM_JIT,
    _LOGICAL_BLOCK_M_JIT,
    _PREFILL_TOKENS_PER_SPLIT_JIT,
)

from .quant import (
    _positive_e4m3_round,
)


@triton.jit
def _fp8_prefill_split_kernel(
    Q_BF16,
    K_BF16,
    V_BF16,
    CU_SEQLENS_Q,
    BLOCK_IDS,
    KV_LENS,
    BLOCK_MASK,
    SPLIT_OUT,
    SPLIT_LSE,
    QB_STOKEN: tl.constexpr,
    QB_SHEAD: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
    MASK_SBATCH: tl.constexpr,
    MASK_SHEAD: tl.constexpr,
    MASK_SQTILE: tl.constexpr,
    MASK_SKVTILE: tl.constexpr,
    SO_SBATCH: tl.constexpr,
    SO_SSPLIT: tl.constexpr,
    SO_SQ: tl.constexpr,
    SO_SHEAD: tl.constexpr,
    SL_SBATCH: tl.constexpr,
    SL_SSPLIT: tl.constexpr,
    SL_SQ: tl.constexpr,
    NUM_HEAD_Q: tl.constexpr,
    NUM_HEAD_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_Q_TILES: tl.constexpr,
    FLAT_Q_TILE_OFFSET: tl.constexpr,
    NUM_MASK_KV_TILES: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    Q_TOKENS: tl.constexpr,
):
    """Disabled (no launch site). Note: indexes the old q-major scores layout."""
    batch_q_tile = tl.program_id(0) + FLAT_Q_TILE_OFFSET
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    batch = batch_q_tile // MAX_Q_TILES
    q_tile = batch_q_tile - batch * MAX_Q_TILES
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    q_local = q_tile * Q_TOKENS
    kv_len = tl.load(KV_LENS + batch)
    split_begin = split_id * _PREFILL_TOKENS_PER_SPLIT_JIT
    if (q_local >= q_len) | (split_begin >= kv_len):
        return

    group_size: tl.constexpr = NUM_HEAD_Q // NUM_HEAD_KV
    q_rows: tl.constexpr = Q_TOKENS * group_size
    offs_r = tl.arange(0, q_rows)
    seq_m = offs_r // group_size
    head_in_group = offs_r - seq_m * group_size
    q_head = kv_head * group_size + head_in_group
    q_pos_local = q_local + seq_m
    valid_r = (q_pos_local < q_len) & (q_head < NUM_HEAD_Q)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    q = tl.load(
        Q_BF16 + (q_begin + q_pos_local[:, None]) * QB_STOKEN + q_head[:, None] * QB_SHEAD + offs_d[None, :],
        mask=valid_r[:, None],
        other=0.0,
    )
    running_max = tl.full((q_rows,), -float("inf"), tl.float32)
    running_sum = tl.zeros((q_rows,), tl.float32)
    accumulator = tl.zeros((q_rows, _HEAD_DIM_JIT), tl.float32)
    offs_n = tl.arange(0, _BLOCK_N_JIT)
    q_abs_pos = kv_len - q_len + q_pos_local
    logical_q_tile = q_pos_local // _LOGICAL_BLOCK_M_JIT
    score_scale: tl.constexpr = 0.127499612793

    for tile_in_split in tl.static_range(0, _PREFILL_TOKENS_PER_SPLIT_JIT // _BLOCK_N_JIT):
        kv_tile = split_id * (_PREFILL_TOKENS_PER_SPLIT_JIT // _BLOCK_N_JIT) + tile_in_split
        kv_tokens = kv_tile * _BLOCK_N_JIT + offs_n
        valid_n = kv_tokens < kv_len
        logical_page = kv_tokens // PAGE_SIZE
        token_in_page = kv_tokens % PAGE_SIZE
        physical = tl.load(
            BLOCK_IDS + batch * BID_SBATCH + logical_page * BID_SPAGE,
            mask=valid_n,
            other=0,
        ).to(tl.int64)
        k = tl.load(
            K_BF16 + physical[:, None] * KB_SPAGE + token_in_page[:, None] * KB_STOKEN + kv_head * KB_SHEAD + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        )
        v = tl.load(
            V_BF16 + physical[:, None] * VB_SPAGE + token_in_page[:, None] * VB_STOKEN + kv_head * VB_SHEAD + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * score_scale
        selected = valid_r
        if HAS_BLOCK_MASK:
            in_range = kv_tile < NUM_MASK_KV_TILES
            selected = selected & (
                tl.load(
                    BLOCK_MASK
                    + batch * MASK_SBATCH
                    + q_head * MASK_SHEAD
                    + logical_q_tile * MASK_SQTILE
                    + kv_tile * MASK_SKVTILE,
                    mask=valid_r & in_range,
                    other=0,
                ) != 0
            )
        score_valid = selected[:, None] & valid_n[None, :] & (kv_tokens[None, :] <= q_abs_pos[:, None])
        scores = tl.where(score_valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.where(running_max == -float("inf"), 0.0, tl.exp2(running_max - safe_max))
        p = tl.where(score_valid, tl.exp2(scores - safe_max[:, None]), 0.0)
        running_sum = running_sum * alpha + tl.sum(p, axis=1)
        accumulator *= alpha[:, None]
        accumulator += tl.dot(_positive_e4m3_round(p * _FP8_P_SCALE_JIT), v)
        running_max = new_max

    has_value = running_sum > 0.0
    partial = tl.where(
        has_value[:, None],
        accumulator / tl.where(has_value[:, None], running_sum[:, None], 1.0),
        0.0,
    )
    tl.store(
        SPLIT_OUT
        + batch * SO_SBATCH
        + split_id * SO_SSPLIT
        + q_pos_local[:, None] * SO_SQ
        + q_head[:, None] * SO_SHEAD
        + offs_d[None, :],
        partial,
        mask=valid_r[:, None],
    )
    tl.store(
        SPLIT_LSE
        + batch * SL_SBATCH
        + split_id * SL_SSPLIT
        + q_pos_local * SL_SQ
        + q_head,
        tl.where(
            has_value,
            running_max + tl.log2(tl.where(has_value, running_sum, 1.0)),
            -float("inf"),
        ),
        mask=valid_r,
    )


@triton.jit
def _fp8_prefill_finalize_kernel(
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
