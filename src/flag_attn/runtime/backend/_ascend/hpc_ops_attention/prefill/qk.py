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

"""QK kernels of the Ascend HY3 FP8 block-sparse prefill."""

import triton
import triton.language as tl

from .constants import (
    _HEAD_DIM_JIT,
    _LOGICAL_BLOCK_M_JIT,
)


@triton.jit
def _fp8_prefill_qk_compact_kernel(
    BLOCK_MASK,
    ACTIVE,
    COUNTS,
    MASK_SBATCH: tl.constexpr,
    MASK_SHEAD: tl.constexpr,
    MASK_SQTILE: tl.constexpr,
    MASK_SKVTILE: tl.constexpr,
    NUM_HEAD_Q: tl.constexpr,
    NUM_MASK_KV_TILES: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    TILES_PER_SPLIT: tl.constexpr,
    NUM_Q_TILES: tl.constexpr,
):
    """Compaction: list the kv 128-token blocks the sparse mask keeps.

    One program per (q tile, split, head, batch).  ``ACTIVE`` holds, for each
    (batch, head, q tile, split), the within-split indices of the 128-token
    blocks that are *not* fully masked out (``COUNTS`` the segment length).
    ``_fp8_prefill_qk_kernel`` then walks only those, so the masked blocks are
    never stored -- their -inf comes from the host ``fill_`` -- and its store
    stays unconditional (a data-dependent store predicate costs +38..117% on
    this backend, see the note in the qk kernel).
    """
    flat = tl.program_id(0)
    q_tile = flat // MAX_SPLITS
    split_id = flat - q_tile * MAX_SPLITS
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    mask_row = (
        BLOCK_MASK
        + batch * MASK_SBATCH
        + q_head * MASK_SHEAD
        + q_tile * MASK_SQTILE
    )
    list_row = (
        ACTIVE
        + ((batch * NUM_HEAD_Q + q_head) * NUM_Q_TILES + q_tile)
        * (MAX_SPLITS * TILES_PER_SPLIT)
        + split_id * TILES_PER_SPLIT
    )
    count_ptr = (
        COUNTS
        + ((batch * NUM_HEAD_Q + q_head) * NUM_Q_TILES + q_tile) * MAX_SPLITS
        + split_id
    )
    base_tile = split_id * TILES_PER_SPLIT
    cnt = 0
    for t in tl.static_range(0, TILES_PER_SPLIT):
        kv_tile = base_tile + t
        # clamped (unmasked) scalar load: a masked scalar load serialises here
        m = tl.load(
            mask_row + tl.minimum(kv_tile, NUM_MASK_KV_TILES - 1) * MASK_SKVTILE
        )
        keep = (m != 0) & (kv_tile < NUM_MASK_KV_TILES)
        if keep:
            tl.store(list_row + cnt, t)
        cnt += tl.where(keep, 1, 0)
    tl.store(count_ptr, cnt)


@triton.jit
def _fp8_prefill_qk_kernel(
    Q_BF16,
    K_BF16,
    CU_SEQLENS_Q,
    BLOCK_IDS,
    KV_LENS,
    BLOCK_MASK,
    SCORES,
    COUNTER,
    ACTIVE,
    COUNTS,
    SPLIT_BASE,
    Q_BASE,
    QB_STOKEN: tl.constexpr,
    QB_SHEAD: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
    MASK_SBATCH: tl.constexpr,
    MASK_SHEAD: tl.constexpr,
    MASK_SQTILE: tl.constexpr,
    MASK_SKVTILE: tl.constexpr,
    S_SBATCH: tl.constexpr,
    S_SSPLIT: tl.constexpr,
    S_SQ: tl.constexpr,
    S_SHEAD: tl.constexpr,
    NUM_HEAD_Q: tl.constexpr,
    NUM_HEAD_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_MASK_KV_TILES: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    QK_BLOCK_M: tl.constexpr,
    SKIP: tl.constexpr,
    COUNT: tl.constexpr,
    LISTED: tl.constexpr,
    NUM_Q_TILES: tl.constexpr,
    LIST_SPLITS: tl.constexpr,
):
    """q @ k^T for one (q tile, kv split, q head).

    The inner KV tile is one full page (PAGE_SIZE tokens) instead of the
    128-token block that softmax/pv use.  That is what makes this kernel fast:
    a 128-token tile straddles two pages, so its page table entry varies per
    row and the K address becomes a per-row gather (int64 page base + row
    stride); a page-aligned tile has a *scalar* page number, so the page-table
    lookup becomes a single scalar load and the K address collapses to
    `physical * KB_SPAGE + arange(PAGE_SIZE) * KB_STOKEN + arange(128)`, an
    affine load the backend turns into a plain DMA.  The page table is still
    applied (BLOCK_IDS may be any permutation), it is just no longer per lane.
    Measured on q=512/kv=32768/32q4kv in one process: 13.16ms -> 6.21ms for
    this change alone (bit-exact scores), and the (128-token) block mask is
    simply read once per 128 KV tokens, i.e. `_LOGICAL_BLOCK_M // PAGE_SIZE`
    inner tiles, so the mask traffic is unchanged.  `_validate` only admits
    PAGE_SIZE 32 or 64, both of which divide _LOGICAL_BLOCK_M, so the tiling is
    always exact.

    The store keeps the softmax layout (column offset tile_in_split*PAGE_SIZE),
    so nothing outside this kernel changes.  No stride is hard-coded: S_SQ /
    S_SHEAD / S_* come from the caller, so a head-major SCORES layout works
    unchanged.
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
    q_local_start = Q_BASE + q_tile * QK_BLOCK_M
    split_begin = global_split_id * TOKENS_PER_SPLIT
    if (q_local_start >= q_len) | (split_begin >= kv_len):
        return

    group_size: tl.constexpr = NUM_HEAD_Q // NUM_HEAD_KV
    kv_head = q_head // group_size
    offs_m = q_local_start + tl.arange(0, QK_BLOCK_M)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    valid_m = offs_m < q_len
    q = tl.load(
        Q_BF16
        + (q_begin + offs_m[:, None]) * QB_STOKEN
        + q_head * QB_SHEAD
        + offs_d[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )
    offs_n = tl.arange(0, PAGE_SIZE)
    # A page-aligned tile starts exactly at a page boundary, so the in-page
    # offset is just offs_n.
    token_in_page = offs_n % PAGE_SIZE
    q_abs_pos = kv_len - q_len + offs_m
    logical_q_tile = offs_m // _LOGICAL_BLOCK_M_JIT
    score_scale: tl.constexpr = 0.127499612793
    mask_tiles: tl.constexpr = _LOGICAL_BLOCK_M_JIT // PAGE_SIZE
    k_base = (
        K_BF16
        + kv_head * KB_SHEAD
        + token_in_page[:, None] * KB_STOKEN
        + offs_d[None, :]
    )
    s_base = (
        SCORES
        + batch * S_SBATCH
        + split_id * S_SSPLIT
        + offs_m[:, None] * S_SQ
        + q_head * S_SHEAD
        + offs_n[None, :]
    )
    max_page = (kv_len - 1) // PAGE_SIZE

    n_skip = 0
    n_block = 0
    # List path: iterate only active blocks listed by compaction. Masked blocks
    # write nothing; their -inf comes from host scores.fill_(-inf), so this store
    # stays **unconditional** (a data-dependent predicate drops out of DMA
    # lowering: per-row +38%, scalar branch +117%, reduce-only +159% (in-process)).
    lq_tile = q_local_start // _LOGICAL_BLOCK_M_JIT
    if LISTED:
        list_row = (
            ACTIVE
            + ((batch * NUM_HEAD_Q + q_head) * NUM_Q_TILES + lq_tile)
            * (LIST_SPLITS * (TOKENS_PER_SPLIT // _LOGICAL_BLOCK_M_JIT))
            + global_split_id * (TOKENS_PER_SPLIT // _LOGICAL_BLOCK_M_JIT)
        )
        n_iter = tl.load(
            COUNTS
            + ((batch * NUM_HEAD_Q + q_head) * NUM_Q_TILES + lq_tile)
            * LIST_SPLITS
            + global_split_id
        )
    else:
        n_iter = TOKENS_PER_SPLIT // _LOGICAL_BLOCK_M_JIT
    for it in tl.range(0, n_iter):
        if LISTED:
            big = tl.load(list_row + it)
        else:
            big = it
        kv_tile = split_begin // _LOGICAL_BLOCK_M_JIT + big
        selected = valid_m
        m0 = 0
        m1 = 0
        if HAS_BLOCK_MASK:
            # One block-mask entry covers mask_tiles inner tiles.  With
            # QK_BLOCK_M rows the M tile spans QK_BLOCK_M // 128 logical q
            # tiles, so exactly QK_BLOCK_M // 128 entries matter -- load them as
            # *scalars* and broadcast.  (The obvious per-row gather
            # `tl.load(... logical_q_tile * MASK_SQTILE ...)` is a 256-lane
            # masked vector load for two distinct addresses, and it sits on the
            # critical path in the skip path where there is no K DMA left to
            # overlap it with.)  The kv index is clamped rather than masked: a
            # masked scalar load serialises against the next DMA on this
            # backend, and an out-of-range kv tile is dead anyway (`in_range`).
            kv_safe = tl.minimum(kv_tile, NUM_MASK_KV_TILES - 1)
            mask_base = (
                BLOCK_MASK
                + batch * MASK_SBATCH
                + q_head * MASK_SHEAD
                + kv_safe * MASK_SKVTILE
                + (q_local_start // _LOGICAL_BLOCK_M_JIT) * MASK_SQTILE
            )
            m0 = tl.load(mask_base)
            in_range = kv_tile < NUM_MASK_KV_TILES
            if QK_BLOCK_M // _LOGICAL_BLOCK_M_JIT > 1:
                m1 = tl.load(mask_base + MASK_SQTILE)
                selected = selected & in_range & (
                    tl.where(logical_q_tile == q_local_start // _LOGICAL_BLOCK_M_JIT,
                             m0, m1)
                    != 0
                )
            else:
                m1 = m0
                selected = selected & in_range & (m0 != 0)
        # Block sparsity: the benchmark/real masks kill ~75% of the
        # (128-row, kv-tile) cells, so a whole (QK_BLOCK_M x 128) block is very
        # often dead.  A dead block's scores are -inf in every lane, so the K
        # load + dot can be skipped and an -inf tile stored instead -- softmax
        # (whose UNMASKED path has no valid_n and relies on qk writing -inf)
        # and pv see byte-identical input.
        #
        # The test is deliberately a *scalar*: OR the block-mask entries of the
        # QK_BLOCK_M // 128 logical q tiles covered by this program, and clamp
        # the kv index (a masked scalar load serialises against the next K DMA,
        # see the page-table note below).  An out-of-range kv tile is dead by
        # construction (the mask load below returns other=0 for it).
        do_skip = False
        if SKIP == 1:
            do_skip = True
        elif SKIP == 2 and HAS_BLOCK_MASK:
            # conservative: the whole (QK_BLOCK_M x 128) block is dead when
            # neither logical q tile has an entry set (or the kv tile is past
            # the mask)
            do_skip = ((m0 | m1) == 0) | (kv_tile >= NUM_MASK_KV_TILES)
        if COUNT:
            n_block += 1
            n_skip += tl.where(do_skip, 1, 0)
        if do_skip:
            # store the -inf tile without touching K or the cube
            for sub in tl.static_range(0, mask_tiles):
                tl.store(
                    s_base + (big * mask_tiles + sub) * PAGE_SIZE,
                    tl.full((QK_BLOCK_M, PAGE_SIZE), -float("inf"), tl.float32),
                    mask=valid_m[:, None],
                )
        else:
            for sub in tl.static_range(0, mask_tiles):
                tile_in_split = big * mask_tiles + sub
                kv_tokens = split_begin + tile_in_split * PAGE_SIZE + offs_n
                valid_n = kv_tokens < kv_len
                # BLOCK_IDS is a logical -> physical page table, so the
                # translation must be applied here (decode_kv writes K in place,
                # keyed by physical page).  One inner tile lives entirely inside
                # one page, so the logical page is a *scalar* and this is a
                # single scalar load that leaves the K address affine.
                #
                # It must be UNMASKED: a masked scalar load costs ~2x on this
                # backend (measured 9.06ms vs 4.57ms on q=512/kv=32768/32q4kv)
                # -- it serialises against the following K DMA.  The tail page
                # is instead clamped so the page-table address stays in bounds,
                # and its lanes are dropped by valid_n / the causal mask below.
                logical_page = tl.minimum(
                    (split_begin + tile_in_split * PAGE_SIZE) // PAGE_SIZE,
                    max_page,
                )
                page = tl.load(
                    BLOCK_IDS + batch * BID_SBATCH + logical_page * BID_SPAGE
                ).to(tl.int64)
                k = tl.load(
                    k_base + page * KB_SPAGE,
                    mask=valid_n[:, None],
                    other=0.0,
                )
                scores = tl.dot(q, tl.trans(k)) * score_scale
                score_valid = (
                    selected[:, None]
                    & valid_n[None, :]
                    & (kv_tokens[None, :] <= q_abs_pos[:, None])
                )
                scores = tl.where(score_valid, scores, -float("inf"))
                tl.store(
                    s_base + tile_in_split * PAGE_SIZE,
                    scores,
                    mask=valid_m[:, None],
                )
    if COUNT:
        pid = (
            tl.program_id(0)
            + (tl.program_id(1) << 10)
            + (tl.program_id(2) << 16)
        )
        tl.store(COUNTER + pid, n_skip)
        tl.store(COUNTER + (1 << 19) + pid, n_block)
