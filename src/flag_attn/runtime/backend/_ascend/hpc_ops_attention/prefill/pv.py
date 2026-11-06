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

"""PV kernels of the Ascend HY3 FP8 block-sparse prefill."""

import triton
import triton.language as tl

from .constants import (
    _BLOCK_N_JIT,
    _FP8_P_SCALE_JIT,
    _HEAD_DIM_JIT,
    _LOGICAL_BLOCK_M_JIT,
    _PREFILL_BLOCK_M_JIT,
    _PREFILL_TOKENS_PER_SPLIT_JIT,
    _PREFILL_VEC_ROWS_JIT,
)

from .quant import (
    _positive_e4m3_round,
)


@triton.jit
def _fp8_prefill_pv_kernel(
    V_BF16,
    CU_SEQLENS_Q,
    BLOCK_IDS,
    KV_LENS,
    SCORES,
    SPLIT_OUT,
    SPLIT_LSE,
    SPLIT_ID,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
    S_SBATCH: tl.constexpr,
    S_SQ: tl.constexpr,
    S_SHEAD: tl.constexpr,
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
):
    """Disabled (no launch site). Note: indexes the old q-major scores layout."""
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(KV_LENS + batch)
    q_local_start = q_tile * _PREFILL_BLOCK_M_JIT
    split_begin = SPLIT_ID * _PREFILL_TOKENS_PER_SPLIT_JIT
    if (q_local_start >= q_len) | (split_begin >= kv_len):
        return

    group_size: tl.constexpr = NUM_HEAD_Q // NUM_HEAD_KV
    kv_head = q_head // group_size
    offs_m = q_local_start + tl.arange(0, _PREFILL_BLOCK_M_JIT)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    valid_m = offs_m < q_len
    offs_n = tl.arange(0, _BLOCK_N_JIT)
    running_max = tl.full((_PREFILL_BLOCK_M_JIT,), -float("inf"), tl.float32)
    running_sum = tl.zeros((_PREFILL_BLOCK_M_JIT,), tl.float32)
    accumulator = tl.zeros(
        (_PREFILL_BLOCK_M_JIT, _HEAD_DIM_JIT), tl.float32
    )

    for tile_in_split in tl.static_range(
        0, _PREFILL_TOKENS_PER_SPLIT_JIT // _BLOCK_N_JIT
    ):
        kv_tile = SPLIT_ID * (
            _PREFILL_TOKENS_PER_SPLIT_JIT // _BLOCK_N_JIT
        ) + tile_in_split
        kv_tokens = kv_tile * _BLOCK_N_JIT + offs_n
        valid_n = kv_tokens < kv_len
        scores = tl.load(
            SCORES
            + batch * S_SBATCH
            + offs_m[:, None] * S_SQ
            + q_head * S_SHEAD
            + tile_in_split * _BLOCK_N_JIT
            + offs_n[None, :],
            mask=valid_m[:, None],
            other=-float("inf"),
        ).to(tl.float32)
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.where(
            running_max == -float("inf"),
            0.0,
            tl.exp2(running_max - safe_max),
        )
        p = tl.where(
            valid_m[:, None] & valid_n[None, :],
            tl.exp2(scores - safe_max[:, None]),
            0.0,
        )
        running_sum = running_sum * alpha + tl.sum(p, axis=1)
        accumulator *= alpha[:, None]
        logical_page = kv_tokens // PAGE_SIZE
        token_in_page = kv_tokens % PAGE_SIZE
        physical = tl.load(
            BLOCK_IDS + batch * BID_SBATCH + logical_page * BID_SPAGE,
            mask=valid_n,
            other=0,
        ).to(tl.int64)
        v = tl.load(
            V_BF16
            + physical[:, None] * VB_SPAGE
            + token_in_page[:, None] * VB_STOKEN
            + kv_head * VB_SHEAD
            + offs_d[None, :],
            mask=valid_n[:, None],
            other=0.0,
        )
        accumulator += tl.dot(
            _positive_e4m3_round(p * _FP8_P_SCALE_JIT), v
        )
        running_max = new_max

    has_value = running_sum > 0.0
    partial = tl.where(
        has_value[:, None],
        accumulator
        / tl.where(has_value[:, None], running_sum[:, None], 1.0),
        0.0,
    )
    tl.store(
        SPLIT_OUT
        + batch * SO_SBATCH
        + SPLIT_ID * SO_SSPLIT
        + offs_m[:, None] * SO_SQ
        + q_head * SO_SHEAD
        + offs_d[None, :],
        partial,
        mask=valid_m[:, None],
    )
    tl.store(
        SPLIT_LSE
        + batch * SL_SBATCH
        + SPLIT_ID * SL_SSPLIT
        + offs_m * SL_SQ
        + q_head,
        tl.where(
            has_value,
            running_max
            + tl.log2(tl.where(has_value, running_sum, 1.0)),
            -float("inf"),
        ),
        mask=valid_m,
    )


@triton.jit
def _fp8_prefill_pv_tile_kernel(
    V_BF16,
    PROBABILITIES,
    CU_SEQLENS_Q,
    BLOCK_IDS,
    KV_LENS,
    PV_TILES,
    SPLIT_BASE,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
    P_SBATCH: tl.constexpr,
    P_SSPLIT: tl.constexpr,
    P_SQ: tl.constexpr,
    P_SHEAD: tl.constexpr,
    PT_SBATCH: tl.constexpr,
    PT_SSPLIT: tl.constexpr,
    PT_STILE: tl.constexpr,
    PT_SQ: tl.constexpr,
    PT_SHEAD: tl.constexpr,
    NUM_HEAD_Q: tl.constexpr,
    NUM_HEAD_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    """Disabled (no launch site). Note: indexes the old q-major scores layout."""
    flat_tile = tl.program_id(0)
    tile_in_split = flat_tile % 4
    flat_split = flat_tile // 4
    split_id = flat_split % MAX_SPLITS
    q_tile = flat_split // MAX_SPLITS
    global_split_id = split_id + SPLIT_BASE
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(KV_LENS + batch)
    q_local_start = q_tile * _PREFILL_BLOCK_M_JIT
    split_begin = global_split_id * _PREFILL_TOKENS_PER_SPLIT_JIT
    if (q_local_start >= q_len) | (split_begin >= kv_len):
        return
    group_size: tl.constexpr = NUM_HEAD_Q // NUM_HEAD_KV
    kv_head = q_head // group_size
    offs_m = q_local_start + tl.arange(0, _PREFILL_BLOCK_M_JIT)
    offs_n = tl.arange(0, _BLOCK_N_JIT)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    valid_m = offs_m < q_len
    kv_tokens = (
        split_begin + tile_in_split * _BLOCK_N_JIT + offs_n
    )
    valid_n = kv_tokens < kv_len
    p = tl.load(
        PROBABILITIES
        + batch * P_SBATCH
        + split_id * P_SSPLIT
        + offs_m[:, None] * P_SQ
        + q_head * P_SHEAD
        + tile_in_split * _BLOCK_N_JIT
        + offs_n[None, :],
        mask=valid_m[:, None] & valid_n[None, :],
        other=0.0,
    )
    logical_page = kv_tokens // PAGE_SIZE
    token_in_page = kv_tokens % PAGE_SIZE
    physical = tl.load(
        BLOCK_IDS + batch * BID_SBATCH + logical_page * BID_SPAGE,
        mask=valid_n,
        other=0,
    ).to(tl.int64)
    v = tl.load(
        V_BF16
        + physical[:, None] * VB_SPAGE
        + token_in_page[:, None] * VB_STOKEN
        + kv_head * VB_SHEAD
        + offs_d[None, :],
        mask=valid_n[:, None],
        other=0.0,
    )
    partial = tl.dot(p, v)
    tl.store(
        PV_TILES
        + batch * PT_SBATCH
        + split_id * PT_SSPLIT
        + tile_in_split * PT_STILE
        + offs_m[:, None] * PT_SQ
        + q_head * PT_SHEAD
        + offs_d[None, :],
        partial,
        mask=valid_m[:, None],
    )


@triton.jit
def _fp8_prefill_pv_accum_kernel(
    V_BF16,
    PROBABILITIES,
    CU_SEQLENS_Q,
    BLOCK_IDS,
    KV_LENS,
    SPLIT_SUM,
    SPLIT_OUT,
    OUT,
    ACTIVE,
    COUNTS,
    SPLIT_BASE,
    Q_BASE,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
    P_SBATCH: tl.constexpr,
    P_SSPLIT: tl.constexpr,
    P_SQ: tl.constexpr,
    P_SHEAD: tl.constexpr,
    SS_SBATCH: tl.constexpr,
    SS_SSPLIT: tl.constexpr,
    SS_SQ: tl.constexpr,
    SO_SBATCH: tl.constexpr,
    SO_SSPLIT: tl.constexpr,
    SO_SQ: tl.constexpr,
    SO_SHEAD: tl.constexpr,
    NUM_HEAD_Q: tl.constexpr,
    NUM_HEAD_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    PAGE_TILE: tl.constexpr,
    PAGE_UNROLL: tl.constexpr,
    FUSE_FINALIZE: tl.constexpr,
    O_STOKEN: tl.constexpr,
    O_SHEAD: tl.constexpr,
    LISTED: tl.constexpr,
    NUM_Q_TILES: tl.constexpr,
    LIST_SPLITS: tl.constexpr,
):
    """Accumulate the PV partial over the active 128-token blocks of a split.

    One V tile is one whole page, so the page number is a scalar and the V address
    stays affine; the older 128-token tile spanned two pages and its per-row
    gather took 68% of pv.  Only the blocks the QK kernel marked active are
    visited, and a skipped block contributes ``dot(0, V)`` -- exactly the value
    the softmax wrote for it.
    """
    flat_tile = tl.program_id(0)
    q_tile = flat_tile // MAX_SPLITS
    split_id = flat_tile - q_tile * MAX_SPLITS
    global_split_id = split_id + SPLIT_BASE
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(KV_LENS + batch)
    q_local_start = Q_BASE + q_tile * _PREFILL_BLOCK_M_JIT
    split_begin = global_split_id * TOKENS_PER_SPLIT
    if (q_local_start >= q_len) | (split_begin >= kv_len):
        return
    group_size: tl.constexpr = NUM_HEAD_Q // NUM_HEAD_KV
    kv_head = q_head // group_size
    offs_m = q_local_start + tl.arange(0, _PREFILL_BLOCK_M_JIT)
    offs_n = tl.arange(0, _BLOCK_N_JIT)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    valid_m = offs_m < q_len
    accumulator = tl.zeros(
        (_PREFILL_BLOCK_M_JIT, _HEAD_DIM_JIT), tl.float32
    )
    blocks_per_split: tl.constexpr = TOKENS_PER_SPLIT // _BLOCK_N_JIT
    # Shares qk's active list: visit only active 128-token blocks. Softmax writes
    # whole probability rows (dead block = 128 zeros); pv skip = exactly dot(0,V).
    list_row = ACTIVE
    n_iter = blocks_per_split
    if LISTED:
        lq_tile = q_local_start // _LOGICAL_BLOCK_M_JIT
        list_row = (
            ACTIVE
            + ((batch * NUM_HEAD_Q + q_head) * NUM_Q_TILES + lq_tile)
            * (LIST_SPLITS * blocks_per_split)
            + global_split_id * blocks_per_split
        )
        n_iter = tl.load(
            COUNTS
            + ((batch * NUM_HEAD_Q + q_head) * NUM_Q_TILES + lq_tile)
            * LIST_SPLITS
            + global_split_id
        )
    if PAGE_TILE > 0:
        for it in tl.range(0, n_iter):
            tile_in_split = it
            if LISTED:
                tile_in_split = tl.load(list_row + it)
        # V tile = one whole page: scalar page number, V address is "scalar base +
        # arange*const" -> affine full-block DMA. The old 128-token tile spans 2
        # pages (per-row gather vector, non-affine) -> on head-major this V load
        # took 68% of pv (5.73ms -> 1.83ms is the no-mask bound; removing the mask
        # computes wrong silently). Page table stays, scalar. pv 5.73ms -> 2.24ms.
            offs_p = tl.arange(0, PAGE_TILE)
            for page_in_tile in tl.static_range(0, PAGE_UNROLL):
                chunk = tile_in_split * PAGE_UNROLL + page_in_tile
                tok0 = split_begin + chunk * PAGE_TILE
                # A tail chunk wholly past kv_len puts the page number outside
                # block_ids; scalar, unmaskable -> clamp to 0, valid_n zeroes rows.
                physical = tl.load(
                    BLOCK_IDS
                    + batch * BID_SBATCH
                    + tl.where(tok0 < kv_len, tok0 // PAGE_SIZE, 0) * BID_SPAGE
                ).to(tl.int64)
                kv_tokens = tok0 + offs_p
                valid_n = kv_tokens < kv_len
                p = tl.load(
                    PROBABILITIES
                    + batch * P_SBATCH
                    + split_id * P_SSPLIT
                    + offs_m[:, None] * P_SQ
                    + q_head * P_SHEAD
                    + chunk * PAGE_TILE
                    + offs_p[None, :],
                    mask=valid_m[:, None] & valid_n[None, :],
                    other=0.0,
                )
                v = tl.load(
                    V_BF16
                    + physical * VB_SPAGE
                    + offs_p[:, None] * VB_STOKEN
                    + kv_head * VB_SHEAD
                    + offs_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                accumulator += tl.dot(p, v)
    else:
        for it in tl.range(0, n_iter):
            tile_in_split = it
            if LISTED:
                tile_in_split = tl.load(list_row + it)
                kv_tokens = split_begin + tile_in_split * _BLOCK_N_JIT + offs_n
                valid_n = kv_tokens < kv_len
                logical_page = kv_tokens // PAGE_SIZE
                token_in_page = kv_tokens % PAGE_SIZE
                p = tl.load(
                    PROBABILITIES
                    + batch * P_SBATCH
                    + split_id * P_SSPLIT
                    + offs_m[:, None] * P_SQ
                    + q_head * P_SHEAD
                    + tile_in_split * _BLOCK_N_JIT
                    + offs_n[None, :],
                    mask=valid_m[:, None] & valid_n[None, :],
                    other=0.0,
                )
                physical = tl.load(
                    BLOCK_IDS + batch * BID_SBATCH + logical_page * BID_SPAGE,
                    mask=valid_n,
                    other=0,
                ).to(tl.int64)
                v = tl.load(
                    V_BF16
                    + physical[:, None] * VB_SPAGE
                    + token_in_page[:, None] * VB_STOKEN
                    + kv_head * VB_SHEAD
                    + offs_d[None, :],
                    mask=valid_n[:, None],
                    other=0.0,
                )
                accumulator += tl.dot(p, v)
    denominator = tl.load(
        SPLIT_SUM
        + batch * SS_SBATCH
        + global_split_id * SS_SSPLIT
        + offs_m * SS_SQ
        + q_head,
        mask=valid_m,
        other=1.0,
    )
    normalized = accumulator / tl.where(
        denominator[:, None] > 0.0, denominator[:, None], 1.0
    )
    if FUSE_FINALIZE:
        # With one split, finalize just copies split_out into output: weights =
        # exp2(lse - lse) = 1 and denominator = 1, so writing the normalized result
        # to OUT is bit-identical, saving the finalize kernel and split_out trip.
        tl.store(
            OUT
            + (q_begin + offs_m[:, None]) * O_STOKEN
            + q_head * O_SHEAD
            + offs_d[None, :],
            normalized.to(OUT.dtype.element_ty),
            mask=valid_m[:, None],
        )
    else:
        tl.store(
            SPLIT_OUT
            + batch * SO_SBATCH
            + global_split_id * SO_SSPLIT
            + offs_m[:, None] * SO_SQ
            + q_head * SO_SHEAD
            + offs_d[None, :],
            normalized,
            mask=valid_m[:, None],
        )


@triton.jit
def _fp8_prefill_pv_reduce_kernel(
    PV_TILES,
    CU_SEQLENS_Q,
    KV_LENS,
    SPLIT_SUM,
    SPLIT_OUT,
    SPLIT_BASE,
    PT_SBATCH: tl.constexpr,
    PT_SSPLIT: tl.constexpr,
    PT_STILE: tl.constexpr,
    PT_SQ: tl.constexpr,
    PT_SHEAD: tl.constexpr,
    SS_SBATCH: tl.constexpr,
    SS_SSPLIT: tl.constexpr,
    SS_SQ: tl.constexpr,
    SO_SBATCH: tl.constexpr,
    SO_SSPLIT: tl.constexpr,
    SO_SQ: tl.constexpr,
    SO_SHEAD: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    """Disabled (no launch site). Note: indexes the old q-major scores layout."""
    flat_q = tl.program_id(0)
    q_block = flat_q // MAX_SPLITS
    split_id = flat_q - q_block * MAX_SPLITS
    global_split_id = split_id + SPLIT_BASE
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    split_begin = global_split_id * _PREFILL_TOKENS_PER_SPLIT_JIT
    kv_len = tl.load(KV_LENS + batch)
    q_local = q_block * _PREFILL_VEC_ROWS_JIT + tl.arange(
        0, _PREFILL_VEC_ROWS_JIT
    )
    valid_m = q_local < q_end - q_begin
    if (q_block * _PREFILL_VEC_ROWS_JIT >= q_end - q_begin) | (
        split_begin >= kv_len
    ):
        return
    offs_t = tl.arange(0, 4)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    valid_t = split_begin + offs_t * _BLOCK_N_JIT < kv_len
    partials = tl.load(
        PV_TILES
        + batch * PT_SBATCH
        + split_id * PT_SSPLIT
        + offs_t[:, None, None] * PT_STILE
        + q_local[None, :, None] * PT_SQ
        + q_head * PT_SHEAD
        + offs_d[None, None, :],
        mask=valid_t[:, None, None] & valid_m[None, :, None],
        other=0.0,
    )
    denominator = tl.load(
        SPLIT_SUM
        + batch * SS_SBATCH
        + global_split_id * SS_SSPLIT
        + q_local * SS_SQ
        + q_head
    )
    tl.store(
        SPLIT_OUT
        + batch * SO_SBATCH
        + global_split_id * SO_SSPLIT
        + q_local[:, None] * SO_SQ
        + q_head * SO_SHEAD
        + offs_d[None, :],
        tl.sum(partials, axis=0)
        / tl.where(denominator[:, None] > 0.0, denominator[:, None], 1.0),
        mask=valid_m[:, None],
    )
