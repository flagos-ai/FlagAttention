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

"""Ascend routing kernels for MoBA.

``tl.topk`` with K > 1 and ``tl.bitonic_merge`` are not usable on Triton-Ascend
3.5: the first fails to lower, the second produces a non-sorted tile.  Routing
selection is therefore an exact rank-by-rank argmax over FP32 scores with the
same tie-break as the CUDA implementation (equal scores choose the smaller
chunk id).  Invalid candidates use a finite sentinel because masked lanes must
never feed ``-inf`` into an FP32 maximum.
"""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.moba.routing import (
    _build_dense_chunk_metadata_triton,
    _chunk_mean_kernel,
    _dense_chunk_mean_kernel,
    _expand_kv_heads_kernel,
    _gather_moba_backward_inputs_kernel,
    _gather_moba_kv_kernel,
    _linear_offsets_triton,
    _reduce_expanded_kv_grads_kernel,
    _scatter_moba_kv_grads_kernel,
    _target_chunk_bounds_triton,
    _validated_chunk_metadata,
    cdiv,
)

# Candidate ids travel through FP32 rank tiles, so every id written by the
# selector must be exactly representable there.  The selector enforces the
# ``< 2**24`` bound explicitly; do not assume it from the shape.

# Ascend rejects any launch whose grid product exceeds this value
# (``Invalid_Argument EE1003 ... value N for parameter coreDim is invalid``).
# The limit applies to the product of every grid dimension, so a 1-D grid may
# have at most ``_MAX_LAUNCH_GRID`` programs.
_MAX_LAUNCH_GRID = 65535

# Triton-Ascend also reserves a per-program workspace when a kernel launches,
# so a legal grid can still exhaust device memory.  The costs below were
# measured on this 910B by launching each kernel with grids of 1024..8192 and
# dividing the ``torch.npu.max_memory_allocated()`` delta by the grid
# (BLOCK_M=64, BLOCK_N=64, BLOCK_D=128, CHUNK_SIZE=128; the deltas were exactly
# linear in the grid):
#     _varlen_attn_fwd_kernel          56 KiB
#     _varlen_attn_bwd_dq_kernel      192 KiB
#     _varlen_attn_bwd_dkdv_kernel     80 KiB
#     _routed_gqa_attn_fwd_kernel     240 KiB (64-row); ~490 KiB with the
#                                      shipped 128-row tile
#     _chunk_topk_*_streaming_kernel    8 KiB
#     all load/store-only kernels       0 KiB
# Slices are therefore capped by a workspace budget as well as by legality.
_WORKSPACE_BUDGET_BYTES = 2 * 1024**3


def _workspace_programs(per_program_bytes: int) -> int:
    """Largest slice that keeps one launch's reserved workspace in budget."""
    return max(1, _WORKSPACE_BUDGET_BYTES // max(1, per_program_bytes))


# 2048 descriptors x ~490 KiB (the 128-row routed tile) ~= 1.0 GB of per-launch
# workspace; at the original 240 KiB tile it was ~480 MiB.  Before slicing, a
# 16384-token call (30720 descriptors) reserved ~8 GB for this kernel alone; the
# 8192-program slice still reserved 2.0 GB, which made routed attention the
# peak-memory owner at short sequences (2.40 GB of the 2.40 GB total at 8192
# tokens).  The launcher additionally caps a launch at an eighth of the pass, so
# short calls use 1024 programs.  The slice only bounds one launch, so it does
# not change results.
_ROUTED_DESC_SLICE = 2048
# 4096 programs: 224 MiB (fwd, 56 KiB/program), 768 MiB (dq, 192 KiB/program),
# 320 MiB (dkdv, 80 KiB/program).
_VARLEN_PROGRAMS_PER_SLICE = 4096
# 8 KiB/program (measured at 1024 and 2048 chunk tiles, so the static chunk
# scan does not grow the workspace), but one launch is also bounded by the
# AICore watchdog: measured 0.92 s at 8192 programs, 1.84 s at 16384 and a
# timeout at 65520 (D=128, 1024 chunks), so slice well below the legality cap.
_TOPK_PROGRAMS_PER_SLICE = 8192
# Parked candidates are 8 bytes per (tile, rank, token) plus a 4-byte tile
# count per program.  The whole stage-1 scratch is launch-local, so the launch
# is shrunk until this one buffer fits the budget below; the gather for that
# token range consumes the same buffer before the next park launch reuses it.
_TOPK_PARK_BUDGET_BYTES = 256 * 1024**2
# Stage 2 ranks the candidate axis in groups of at most this many lanes.  128
# lanes x up to 32 tokens keeps the score/id tiles and the unrolled rank chain
# inside the 192 KiB unified buffer; 512 lanes (the 524288-token case) needs
# 225 KiB and fails to compile.  Shapes with fewer lanes take the single-group
# path unchanged.
_GATHER_MAX_CAND_LANES = 128
# Load/store-only kernels reserve no workspace: legality-bound.
_GATHER_ROWS_PER_SLICE = 8192
_CHUNK_MEAN_PROGRAMS_PER_SLICE = min(_MAX_LAUNCH_GRID, _workspace_programs(0))
_GATHER_KV_ROWS_PER_SLICE = 8192
_EXPAND_PROGRAMS_PER_SLICE = min(_MAX_LAUNCH_GRID, _workspace_programs(0))


def _streaming_tile_shape(history_chunks: int, block_topk: int):
    """Tile policy for the streaming routing scans: (BLOCK_TOK, BLOCK_CHUNK).

    Measured on the 910B. A 128-token tile is best for the common rank widths:
    the scan is bound by the dependent rank chain, not by the cube dot, so the
    win comes from fewer programs each covering more tokens. A 64-wide chunk
    tile only pays once the history spans several of them; below that the wider
    tile is mostly masked and its extra rank lanes are pure cost.

    Wide rank tiles are different: at BLOCK_TOPK=64 the keep lists and the
    merged candidate tile alone need more than the 192 KiB unified buffer at 128
    tokens, so the narrow 16x32 tile is the widest that still compiles there.
    """
    if block_topk > 16:
        return 16, 32
    return 128, (32 if history_chunks <= 32 else 64)


def _training_tile_shape(num_chunks: int, block_d: int, block_topk: int):
    """Tile policy for the compacted training scan: (BLOCK_TOK, BLOCK_CHUNK).

    Triton-Ascend lowers a chunk loop with a single compile-time trip count into
    a body that is ~20x slower per program than the same body inside a loop with
    two or more trips, so the chunk tile is chosen to keep at least two trips
    whenever that does not cost more than the narrower tile. 128x32 does not fit
    the unified buffer at ``head_dim == 128``, and a BLOCK_TOPK of 32/64 keeps
    only the baseline 16x32 tile inside it.
    """
    if block_topk > 16:
        return 16, 32
    if num_chunks <= 32:
        return 128, 16
    if num_chunks <= 64:
        return 64, 32
    if block_d <= 64:
        return 128, 64
    return 64, 64


@triton.jit
def _rank_candidates(scores, lane, id_base, keep_scores, keep_ids, BLOCK_K: tl.constexpr):
    """Keep the best BLOCK_K candidates of one chunk tile, best rank first.

    Validity travels inside ``scores``: masked lanes hold the negative sentinel,
    so a single maximum finds the winner and no id tile is needed. Chunk ids in
    a tile are ``id_base + lane`` and therefore increase with the lane index, so
    the reference tie-break (equal scores pick the smallest chunk id) is the
    smallest lane and reduces to an FP32 minimum over the lane index.

    Every reduction here stays in FP32 on purpose: INT32 row reductions on the
    Ascend vector unit are substantially slower, and the batched routing scan is
    a dependent rank chain, so they dominate its runtime. Lanes are dropped by
    identity rather than by score so that exact score ties fill consecutive
    ranks exactly like the reference.
    """
    slots = tl.arange(0, BLOCK_K)
    for rank in tl.static_range(0, BLOCK_K):
        best = tl.max(scores, axis=1)
        pick = tl.min(tl.where(scores == best[:, None], lane, 1.0e9), axis=1)
        valid = best > -3.0e38
        keep_scores = tl.where(
            slots[None, :] == rank,
            tl.where(valid, best, -3.0e38)[:, None],
            keep_scores,
        )
        keep_ids = tl.where(
            slots[None, :] == rank,
            tl.where(valid, id_base + pick, -1.0)[:, None],
            keep_ids,
        )
        scores = tl.where(lane == pick[:, None], -3.0e38, scores)
    return keep_scores, keep_ids


@triton.jit
def _rank_candidates_by_id(scores, ids, BLOCK_TOK: tl.constexpr, BLOCK_K: tl.constexpr):
    """Top-BLOCK_K of a candidate tile, ordered by (score desc, id asc).

    Same rank-by-rank chain as ``_rank_candidates`` but the tie-break key is the
    candidate chunk id itself, because a gathered candidate array is no longer
    ordered by chunk id (each tile contributes its own score-ranked list).
    """
    slots = tl.arange(0, BLOCK_K)
    keep_scores = tl.full([BLOCK_TOK, BLOCK_K], -3.0e38, dtype=tl.float32)
    keep_ids = tl.full([BLOCK_TOK, BLOCK_K], -1.0, dtype=tl.float32)
    for rank in tl.static_range(0, BLOCK_K):
        best = tl.max(scores, axis=1)
        valid = best > -3.0e38
        pick = tl.min(tl.where(scores == best[:, None], ids, 1.0e9), axis=1)
        keep_scores = tl.where(
            slots[None, :] == rank,
            tl.where(valid, best, -3.0e38)[:, None],
            keep_scores,
        )
        keep_ids = tl.where(
            slots[None, :] == rank,
            tl.where(valid, pick, -1.0)[:, None],
            keep_ids,
        )
        scores = tl.where(ids == pick[:, None], -3.0e38, scores)
    return keep_scores, keep_ids


@triton.jit
def _chunk_topk_park_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    cu_seqlens_ptr,
    park_score_ptr,
    park_id_ptr,
    park_count_ptr,
    num_heads,
    park_batch_extent,
    park_program_base,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_chunk_mean_chunk,
    stride_chunk_mean_head,
    stride_chunk_mean_d,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MAX_CHUNKS_PER_SEQUENCE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAND_TILES: tl.constexpr,
    pid_tok_offset,
    batch_offset,
):
    """Stage 1 of the split scan: park each chunk tile's best candidates.

    The reference kernel merges the carried best list with every tile's list.
    That merge is 64% of the runtime because its 8 dependent reductions run on a
    2-element (per token) wide slice -- a rank-16 tile that is nearly all
    overhead.  Here each tile's best BLOCK_TOPK candidates are parked in a
    per-program scratch slot (addressable by the tile counter) and the whole
    cross-tile selection becomes a single rank chain in stage 2.
    """
    pid_tok = tl.program_id(0) + pid_tok_offset
    pid_head = tl.program_id(1)
    batch = tl.program_id(2) + batch_offset
    if pid_head >= num_heads:
        return

    kv_head = pid_head // GROUP_SIZE
    sequence_start = tl.load(cu_seqlens_ptr + batch)
    sequence_end = tl.load(cu_seqlens_ptr + batch + 1)

    local_tokens = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    offs_token = sequence_start + local_tokens
    token_mask = offs_token < sequence_end
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    chunk_rows = tl.arange(0, BLOCK_CHUNK)
    lane = chunk_rows.to(tl.float32)[None, :]

    q_values = tl.load(
        q_ptr + offs_token[:, None] * stride_q_seq + pid_head * stride_q_head + offs_d[None, :] * stride_q_d,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    )

    chunk_base = batch * MAX_CHUNKS_PER_SEQUENCE
    program_slot = (
        (park_program_base + tl.program_id(0) * num_heads + pid_head) * park_batch_extent
        + tl.program_id(2)
    )
    park_stride = CAND_TILES * BLOCK_TOPK * BLOCK_TOK
    park_base = program_slot * park_stride
    token_lane = tl.arange(0, BLOCK_TOK)

    if CHUNK_SIZE > 0:
        local_last = tl.minimum(
            pid_tok * BLOCK_TOK + BLOCK_TOK - 1,
            sequence_end - sequence_start - 1,
        )
        history_chunks = tl.maximum(local_last // CHUNK_SIZE, 0)
        history_chunks = tl.minimum(history_chunks, MAX_CHUNKS_PER_SEQUENCE)
        num_chunk_tiles = (history_chunks + BLOCK_CHUNK - 1) // BLOCK_CHUNK
    else:
        num_chunk_tiles = (MAX_CHUNKS_PER_SEQUENCE + BLOCK_CHUNK - 1) // BLOCK_CHUNK

    for chunk_tile in range(0, num_chunk_tiles):
        local_start = chunk_tile * BLOCK_CHUNK
        local_chunks = local_start + chunk_rows
        chunk_mask = local_chunks < MAX_CHUNKS_PER_SEQUENCE
        chunk_offsets = chunk_base + local_chunks
        safe_offsets = tl.where(chunk_mask, chunk_offsets, chunk_base)

        chunk_values = tl.load(
            chunk_means_ptr
            + safe_offsets[:, None] * stride_chunk_mean_chunk
            + kv_head * stride_chunk_mean_head
            + offs_d[None, :] * stride_chunk_mean_d,
            mask=chunk_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        gating = tl.dot(q_values, tl.trans(chunk_values), out_dtype=tl.float32)

        chunk_end = tl.load(chunk_end_ptr + safe_offsets, mask=chunk_mask, other=0)
        valid = (
            chunk_mask[None, :]
            & token_mask[:, None]
            & (chunk_end[None, :] > sequence_start)
            & (offs_token[:, None] >= chunk_end[None, :])
        )
        scores = tl.where(valid, gating, -3.0e38)

        tile_scores = tl.full([BLOCK_TOK, BLOCK_TOPK], -3.0e38, dtype=tl.float32)
        tile_ids = tl.full([BLOCK_TOK, BLOCK_TOPK], -1.0, dtype=tl.float32)
        tile_scores, tile_ids = _rank_candidates(
            scores,
            lane,
            (chunk_base + local_start).to(tl.float32),
            tile_scores,
            tile_ids,
            BLOCK_TOPK,
        )
        slot = park_base + chunk_tile * BLOCK_TOPK * BLOCK_TOK
        tl.store(
            park_score_ptr + slot + tl.arange(0, BLOCK_TOPK)[:, None] * BLOCK_TOK + token_lane[None, :],
            tl.trans(tile_scores),
            mask=token_mask[None, :],
        )
        tl.store(
            park_id_ptr + slot + tl.arange(0, BLOCK_TOPK)[:, None] * BLOCK_TOK + token_lane[None, :],
            tl.trans(tile_ids),
            mask=token_mask[None, :],
        )

    tl.store(park_count_ptr + program_slot, num_chunk_tiles)


@triton.jit
def _chunk_topk_gather_kernel(
    park_score_ptr,
    park_id_ptr,
    park_count_ptr,
    cu_seqlens_ptr,
    selected_chunks_ptr,
    num_heads,
    park_batch_extent,
    park_program_base,
    stride_selected_rank,
    stride_selected_head,
    stride_selected_token,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_TOK_PARK: tl.constexpr,
    CAND_TILES: tl.constexpr,
    GROUP_CAND: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    pid_tok_offset,
    batch_offset,
):
    """Stage 2: one exact rank chain over every parked candidate of a program.

    ``BLOCK_TOK`` (stage-2 token tile) may be narrower than
    ``BLOCK_TOK_PARK``; each stage-2 program then reads a token slice of its
    stage-1 program.  Lane order is tile-major, i.e. still increasing in chunk
    id within a tile, but the ranks inside a tile are score-ordered, so the
    chain breaks ties on the parked id rather than the lane index.

    The candidate axis is ranked in groups of at most ``GROUP_CAND`` lanes:
    materializing all ``CAND_TILES * BLOCK_TOPK`` lanes at once (512 at the
    524288-token case) needs 225 KiB of the 192 KiB unified buffer for the
    score and id tiles alone.  ``NUM_GROUPS`` is a ceiling, so the last group
    may be partially filled; its live mask keeps the unused lanes out of the
    ranking.  Each group is ranked exactly by (score desc, chunk id asc), then
    merged with the running top list by ranking the ``2 * BLOCK_TOPK`` join,
    which preserves that order because the group lists and the running list are
    disjoint in id.
    """
    pid_tok = tl.program_id(0) + pid_tok_offset
    pid_head = tl.program_id(1)
    batch = tl.program_id(2) + batch_offset
    if pid_head >= num_heads:
        return

    tokens_per_park = BLOCK_TOK_PARK // BLOCK_TOK
    # ``park_program_base`` is the first park token-block of this launch: the
    # scratch is launch-local, so the slot must subtract it.
    park_token_block = pid_tok // tokens_per_park - park_program_base
    token_shift = (pid_tok % tokens_per_park) * BLOCK_TOK
    program_slot = (
        (park_token_block * num_heads + pid_head) * park_batch_extent
        + tl.program_id(2)
    )
    park_stride = CAND_TILES * BLOCK_TOPK * BLOCK_TOK_PARK
    park_base = program_slot * park_stride
    num_tiles = tl.load(park_count_ptr + program_slot)

    token_lane = tl.arange(0, BLOCK_TOK)
    group_lane = tl.arange(0, GROUP_CAND)
    if NUM_GROUPS == 1:
        cand_lane = group_lane
        gather = (
            park_base
            + (cand_lane // BLOCK_TOPK)[:, None] * BLOCK_TOPK * BLOCK_TOK_PARK
            + (cand_lane % BLOCK_TOPK)[:, None] * BLOCK_TOK_PARK
            + token_shift
            + token_lane[None, :]
        )
        live = (cand_lane < num_tiles * BLOCK_TOPK)[:, None]
        scores = tl.load(park_score_ptr + gather, mask=live, other=-3.0e38)
        ids = tl.load(park_id_ptr + gather, mask=live, other=-1.0)
        _, ids = _rank_candidates_axis0(scores, ids, BLOCK_TOK, BLOCK_TOPK)
    else:
        best_scores = tl.full([BLOCK_TOK, BLOCK_TOPK], -3.0e38, dtype=tl.float32)
        best_ids = tl.full([BLOCK_TOK, BLOCK_TOPK], -1.0, dtype=tl.float32)
        for group in range(0, NUM_GROUPS):
            cand_lane = group * GROUP_CAND + group_lane
            # The last group may be partially filled (NUM_GROUPS is a ceil), so
            # clamp the tile index: masked-off lanes then still address inside
            # this program's scratch slot instead of the next one.
            cand_tile = tl.minimum(cand_lane // BLOCK_TOPK, CAND_TILES - 1)
            gather = (
                park_base
                + cand_tile[:, None] * BLOCK_TOPK * BLOCK_TOK_PARK
                + (cand_lane % BLOCK_TOPK)[:, None] * BLOCK_TOK_PARK
                + token_shift
                + token_lane[None, :]
            )
            live = (cand_lane < num_tiles * BLOCK_TOPK)[:, None]
            scores = tl.load(park_score_ptr + gather, mask=live, other=-3.0e38)
            ids = tl.load(park_id_ptr + gather, mask=live, other=-1.0)
            group_scores, group_ids = _rank_candidates_axis0(scores, ids, BLOCK_TOK, BLOCK_TOPK)
            join_scores = tl.reshape(
                tl.join(best_scores, tl.trans(group_scores)),
                (BLOCK_TOK, 2 * BLOCK_TOPK),
            )
            join_ids = tl.reshape(
                tl.join(best_ids, tl.trans(group_ids)),
                (BLOCK_TOK, 2 * BLOCK_TOPK),
            )
            best_scores, best_ids = _rank_candidates_by_id(
                join_scores, join_ids, BLOCK_TOK, BLOCK_TOPK
            )
        ids = tl.trans(best_ids)

    sequence_start = tl.load(cu_seqlens_ptr + batch)
    sequence_end = tl.load(cu_seqlens_ptr + batch + 1)
    offs_token = sequence_start + pid_tok * BLOCK_TOK + token_lane
    token_mask = offs_token < sequence_end
    rank_offsets = tl.arange(0, BLOCK_TOPK)
    selected = ids.to(tl.int32)
    tl.store(
        selected_chunks_ptr
        + rank_offsets[:, None] * stride_selected_rank
        + pid_head * stride_selected_head
        + offs_token[None, :] * stride_selected_token,
        selected,
        mask=(rank_offsets[:, None] < TOPK) & token_mask[None, :] & (selected >= 0),
    )


@triton.jit
def _rank_candidates_axis0(scores, ids, BLOCK_TOK: tl.constexpr, BLOCK_K: tl.constexpr):
    """Same chain as ``_rank_candidates_by_id`` over a [BLOCK_CAND, BLOCK_TOK] tile.

    The gather loads its candidates transposed (candidate rows are contiguous in
    the scratch), so reducing axis 0 avoids an explicit transpose per program;
    the keep list is already in the [rank, token] layout the epilogue stores.
    """
    slots = tl.arange(0, BLOCK_K)
    keep_scores = tl.full([BLOCK_K, BLOCK_TOK], -3.0e38, dtype=tl.float32)
    keep_ids = tl.full([BLOCK_K, BLOCK_TOK], -1.0, dtype=tl.float32)
    for rank in tl.static_range(0, BLOCK_K):
        best = tl.max(scores, axis=0)
        valid = best > -3.0e38
        pick = tl.min(tl.where(scores == best[None, :], ids, 1.0e9), axis=0)
        keep_scores = tl.where(
            slots[:, None] == rank,
            tl.where(valid, best, -3.0e38)[None, :],
            keep_scores,
        )
        keep_ids = tl.where(
            slots[:, None] == rank,
            tl.where(valid, pick, -1.0)[None, :],
            keep_ids,
        )
        scores = tl.where(ids == pick[None, :], -3.0e38, scores)
    return keep_scores, keep_ids


@triton.jit
def _merge_candidates(best_scores, best_ids, tile_scores, tile_ids, BLOCK_TOK: tl.constexpr, BLOCK_K: tl.constexpr):
    """Merge two candidate lists keeping the global best BLOCK_K.

    Ids arrive already ordered by chunk, so the smallest id wins an exact score
    tie; dropping by id (not by score) keeps tied candidates on consecutive
    ranks. Ids are FP32 so both reductions stay on the fast path.
    """
    candidates_scores = tl.reshape(tl.join(best_scores, tile_scores), (BLOCK_TOK, 2 * BLOCK_K))
    candidates_ids = tl.reshape(tl.join(best_ids, tile_ids), (BLOCK_TOK, 2 * BLOCK_K))
    merged_scores = tl.full([BLOCK_TOK, BLOCK_K], -3.0e38, dtype=tl.float32)
    merged_ids = tl.full([BLOCK_TOK, BLOCK_K], -1.0, dtype=tl.float32)
    slots = tl.arange(0, BLOCK_K)
    for rank in tl.static_range(0, BLOCK_K):
        best = tl.max(candidates_scores, axis=1)
        valid = best > -3.0e38
        best_id = tl.min(
            tl.where(candidates_scores == best[:, None], candidates_ids, 1.0e9),
            axis=1,
        )
        merged_scores = tl.where(
            slots[None, :] == rank,
            tl.where(valid, best, -3.0e38)[:, None],
            merged_scores,
        )
        merged_ids = tl.where(
            slots[None, :] == rank,
            tl.where(valid, best_id, -1.0)[:, None],
            merged_ids,
        )
        candidates_scores = tl.where(candidates_ids == best_id[:, None], -3.0e38, candidates_scores)
    return merged_scores, merged_ids


def _topk_park_plan(
    max_chunks_per_sequence: int,
    block_chunk: int,
    block_topk: int,
    block_tok: int,
    num_heads: int,
    batch_size: int,
) -> dict:
    """Plan candidate coverage, token tiling and scratch for the two stages.

    Callers split the head axis when ``num_heads > head_limit``. One token
    block and one batch must fit both stages for each resulting head slice.
    """
    cand_tiles = max(1, cdiv(max_chunks_per_sequence, block_chunk))
    if cand_tiles > 8:
        # 16 live tile slots plus the parking index tiles push the 128-token
        # park kernel past the unified buffer; narrow the chunk tile instead.
        block_chunk = 32
        cand_tiles = max(1, cdiv(max_chunks_per_sequence, block_chunk))
    park_program_bytes = cand_tiles * block_topk * block_tok * 8 + 4
    park_programs = _TOPK_PARK_BUDGET_BYTES // park_program_bytes
    if not park_programs:
        raise ValueError(
            f"one MoBA top-k park program needs {park_program_bytes} bytes, exceeding the "
            f"{_TOPK_PARK_BUDGET_BYTES}-byte scratch budget; reduce max_seqlen or top-k"
        )
    budget = min(_TOPK_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID, park_programs)
    gather_budget = min(_TOPK_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID)
    total_cand_lanes = cand_tiles * block_topk
    group_cand = min(total_cand_lanes, _GATHER_MAX_CAND_LANES)
    num_groups = cdiv(total_cand_lanes, group_cand)  # Include the partial tail group.
    group_tok = min(block_tok, 64 if cand_tiles <= 8 else 32)
    if num_groups > 1 and block_topk == 2:  # Fit the Ascend unified buffer.
        group_tok = min(group_tok, 8)
    tokens_per_park = block_tok // group_tok
    # A single parked token block expands to tokens_per_park gather programs
    # per head and batch. Both stages must fit before slicing the token axis.
    head_limit = min(budget, gather_budget // tokens_per_park)
    batches_per_launch = max(1, head_limit // num_heads)
    blocks_per_launch = max(1, budget // (num_heads * min(batch_size, batches_per_launch)))
    return {
        "cand_tiles": cand_tiles,
        "block_chunk": block_chunk,
        "park_program_bytes": park_program_bytes,
        "park_programs": park_programs,
        "budget": budget,
        "gather_budget": gather_budget,
        "head_limit": head_limit,
        "batches_per_launch": batches_per_launch,
        "blocks_per_launch": blocks_per_launch,
        "group_tok": group_tok,
        "tokens_per_park": tokens_per_park,
        "total_cand_lanes": total_cand_lanes,
        "group_cand": group_cand,
        "num_groups": num_groups,
        "num_heads": num_heads,
    }


def _gather_blocks_per_launch(plan: dict, batch_extent: int) -> int:
    """Token blocks one stage-2 launch may consume.

    The park scratch is launch-local, so a gather launch may never cover more
    token blocks than its own stage-1 launch parked; the budget is a second,
    independent bound on the program count.
    """
    per_block = plan["num_heads"] * batch_extent * plan["tokens_per_park"]
    return min(
        plan["blocks_per_launch"],
        max(1, plan["gather_budget"] // per_block),
    )


def _compute_selected_chunks_batched_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    max_chunks_per_sequence: int,
    topk: int,
    chunk_size: int | None = None,
    *,
    _out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select historical IDs in rank-major order for a packed batch.

    An existing output view supplies its own strides for a head slice.
    The caller initializes its missing-route slots to -1.
    """
    num_chunk_slots, num_kv_heads, head_dim = chunk_means.shape
    num_heads = q.shape[1]
    batch_size = cu_seqlens.numel() - 1
    expected_slots = batch_size * max_chunks_per_sequence
    if num_chunk_slots != expected_slots:
        raise ValueError(
            "dense chunk metadata must contain batch_size * "
            "max_chunks_per_sequence slots, got "
            f"{num_chunk_slots} and expected {expected_slots}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    # Chunk ids are FP32 in this kernel (lane-based rank chain) and are written
    # out by casting ``id_base + lane``, so every id must be exact in FP32.
    # Ids are dense slots ``batch * capacity + chunk``: a single batch of
    # 65536 chunks needs only 65536 slots, but 256 such batches need 16.8M,
    # which is where the 2**24 exactness bound starts to matter.  Refuse
    # rather than store a rounded id.
    if num_chunk_slots >= (1 << 24):
        raise ValueError(
            f"routing top-k needs {num_chunk_slots} dense chunk slots, but the "
            f"FP32 chunk-id path is exact only below {1 << 24}; pass the "
            "sequences in smaller groups"
        )
    if topk <= 0 or num_chunk_slots == 0:
        return torch.empty(
            (0, num_heads, q.shape[0]),
            device=q.device,
            dtype=torch.int32,
        )
    if topk >= max_chunks_per_sequence:
        raise ValueError(
            f"historical topk ({topk}) must be smaller than the per-sequence chunk capacity ({max_chunks_per_sequence})"
        )

    # One-element rank tiles cannot be legalized by Triton-Ascend, so rank 0 of
    # a Top-1 request runs with a masked second lane.
    block_topk = max(2, triton.next_power_of_2(topk))
    if block_topk > 64:
        raise ValueError(f"streaming Top-K currently supports topk <= 64, got topk={topk}")
    block_d = max(16, triton.next_power_of_2(head_dim))
    block_tok, block_chunk = _streaming_tile_shape(max_chunks_per_sequence, block_topk)
    plan = _topk_park_plan(
        max_chunks_per_sequence, block_chunk, block_topk, block_tok, num_heads, batch_size
    )
    selected_chunks = _out
    if selected_chunks is None:
        selected_chunks = torch.full(
            (topk, num_heads, q.shape[0]), -1, device=q.device, dtype=torch.int32
        )
    if num_heads > plan["head_limit"]:
        _select_topk_head_slices(
            q, chunk_means, chunk_end, cu_seqlens, max_seqlen, max_chunks_per_sequence,
            topk, chunk_size, selected_chunks, plan["head_limit"],
        )
        return selected_chunks

    num_token_blocks = cdiv(max_seqlen, block_tok)
    park_scores = None
    park_ids = None
    park_counts = None
    cand_tiles = plan["cand_tiles"]
    block_chunk = plan["block_chunk"]
    batches_per_launch = plan["batches_per_launch"]
    blocks_per_launch = plan["blocks_per_launch"]
    group_tok = plan["group_tok"]
    tokens_per_park = plan["tokens_per_park"]
    group_cand = plan["group_cand"]
    num_groups = plan["num_groups"]
    for batch_start in range(0, batch_size, batches_per_launch):
        batch_end = min(batch_start + batches_per_launch, batch_size)
        batch_extent = batch_end - batch_start
        for block_start in range(0, num_token_blocks, blocks_per_launch):
            block_end = min(block_start + blocks_per_launch, num_token_blocks)
            blocks = block_end - block_start
            programs = blocks * num_heads * batch_extent
            elems = programs * cand_tiles * block_topk * block_tok
            if (
                park_scores is None
                or park_scores.numel() < elems
            ):
                park_scores = torch.empty(elems, device=q.device, dtype=torch.float32)
                park_ids = torch.empty(elems, device=q.device, dtype=torch.float32)
                park_counts = torch.empty(programs, device=q.device, dtype=torch.int32)
            grid = (blocks, num_heads, batch_extent)
            _chunk_topk_park_kernel[grid](
                q,
                chunk_means,
                chunk_end,
                cu_seqlens,
                park_scores,
                park_ids,
                park_counts,
                num_heads,
                batch_extent,
                0,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                chunk_means.stride(0),
                chunk_means.stride(1),
                chunk_means.stride(2),
                HEAD_DIM=head_dim,
                GROUP_SIZE=num_heads // num_kv_heads,
                MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
                CHUNK_SIZE=chunk_size if chunk_size is not None else 0,
                BLOCK_TOPK=block_topk,
                BLOCK_TOK=block_tok,
                BLOCK_CHUNK=block_chunk,
                BLOCK_D=block_d,
                CAND_TILES=cand_tiles,
                pid_tok_offset=block_start,
                batch_offset=batch_start,
                num_warps=4,
                num_stages=1,
            )
            # Stage 2 must keep stage 1's batch slicing (the scratch is indexed
            # by the launch-local batch id) and may only slice the token axis.
            # ``tokens_per_park >= 1`` is guaranteed by the plan.
            assert tokens_per_park >= 1
            # The park scratch is launch-local: a gather launch may consume at
            # most the token blocks parked by its own stage-1 launch.  Crossing
            # that boundary reads slots that were never written.
            gblocks_per_launch = _gather_blocks_per_launch(plan, batch_extent)
            for gblock_start in range(block_start, block_end, gblocks_per_launch):
                gblock_end = min(gblock_start + gblocks_per_launch, block_end)
                _chunk_topk_gather_kernel[
                    ((gblock_end - gblock_start) * tokens_per_park, num_heads, batch_extent)
                ](
                    park_scores,
                    park_ids,
                    park_counts,
                    cu_seqlens,
                    selected_chunks,
                    num_heads,
                    batch_extent,
                    block_start,
                    selected_chunks.stride(0),
                    selected_chunks.stride(1),
                    selected_chunks.stride(2),
                    TOPK=topk,
                    BLOCK_TOPK=block_topk,
                    BLOCK_TOK=group_tok,
                    BLOCK_TOK_PARK=block_tok,
                    CAND_TILES=cand_tiles,
                    GROUP_CAND=group_cand,
                    NUM_GROUPS=num_groups,
                    pid_tok_offset=gblock_start * tokens_per_park,
                    batch_offset=batch_start,
                    num_warps=4,
                    num_stages=1,
                )
    return selected_chunks


@triton.jit
def _chunk_topk_streaming_kernel(
    q_ptr,
    chunk_means_ptr,
    chunk_end_ptr,
    batch_end_ptr,
    selected_chunks_ptr,
    num_heads,
    seqlen,
    stride_q_seq,
    stride_q_head,
    stride_q_d,
    stride_chunk_mean_chunk,
    stride_chunk_mean_head,
    stride_chunk_mean_d,
    stride_selected_rank,
    stride_selected_head,
    stride_selected_token,
    HEAD_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    pid_tok_offset,
):
    """Streaming top-k over compacted training chunks."""
    pid_tok = tl.program_id(0) + pid_tok_offset
    pid_head = tl.program_id(1)
    if pid_head >= num_heads:
        return

    kv_head = pid_head // GROUP_SIZE
    offs_token = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    token_mask = offs_token < seqlen
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    chunk_rows = tl.arange(0, BLOCK_CHUNK)
    lane = chunk_rows.to(tl.float32)[None, :]

    q_values = tl.load(
        q_ptr + offs_token[:, None] * stride_q_seq + pid_head * stride_q_head + offs_d[None, :] * stride_q_d,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    )

    best_scores = tl.full([BLOCK_TOK, BLOCK_TOPK], -3.0e38, dtype=tl.float32)
    best_ids = tl.full([BLOCK_TOK, BLOCK_TOPK], -1.0, dtype=tl.float32)

    for chunk_start in range(0, NUM_CHUNKS, BLOCK_CHUNK):
        chunk_offsets = chunk_start + chunk_rows
        chunk_mask = chunk_offsets < NUM_CHUNKS
        safe_offsets = tl.where(chunk_mask, chunk_offsets, 0)

        chunk_values = tl.load(
            chunk_means_ptr
            + safe_offsets[:, None] * stride_chunk_mean_chunk
            + kv_head * stride_chunk_mean_head
            + offs_d[None, :] * stride_chunk_mean_d,
            mask=chunk_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        gating = tl.dot(q_values, tl.trans(chunk_values), out_dtype=tl.float32)

        chunk_end = tl.load(chunk_end_ptr + safe_offsets, mask=chunk_mask, other=0)
        batch_end = tl.load(batch_end_ptr + safe_offsets, mask=chunk_mask, other=0)
        valid = (
            chunk_mask[None, :]
            & token_mask[:, None]
            & (offs_token[:, None] >= chunk_end[None, :])
            & (offs_token[:, None] < batch_end[None, :])
        )
        scores = tl.where(valid, gating, -3.0e38)

        tile_scores = tl.full([BLOCK_TOK, BLOCK_TOPK], -3.0e38, dtype=tl.float32)
        tile_ids = tl.full([BLOCK_TOK, BLOCK_TOPK], -1.0, dtype=tl.float32)
        tile_scores, tile_ids = _rank_candidates(
            scores,
            lane,
            chunk_start.to(tl.float32),
            tile_scores,
            tile_ids,
            BLOCK_TOPK,
        )
        best_scores, best_ids = _merge_candidates(
            best_scores,
            best_ids,
            tile_scores,
            tile_ids,
            BLOCK_TOK,
            BLOCK_TOPK,
        )

    rank_offsets = tl.arange(0, BLOCK_TOPK)
    selected = tl.trans(best_ids).to(tl.int32)
    tl.store(
        selected_chunks_ptr
        + rank_offsets[:, None] * stride_selected_rank
        + pid_head * stride_selected_head
        + offs_token[None, :] * stride_selected_token,
        selected,
        mask=(rank_offsets[:, None] < TOPK) & token_mask[None, :] & (selected >= 0),
    )


@triton.jit
def _single_chunk_selection_kernel(
    chunk_end_ptr,
    batch_end_ptr,
    selected_chunks_ptr,
    seqlen,
    stride_selected_head,
    stride_selected_token,
    NUM_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Top-1 of the only historical chunk.

    Triton-Ascend crashes lowering the streaming kernel when a single chunk is
    live; the degenerate case is a masked fill instead.
    """
    tokens = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = tokens < seqlen
    chunk_end = tl.load(chunk_end_ptr)
    batch_end = tl.load(batch_end_ptr)
    valid = mask & (tokens >= chunk_end) & (tokens < batch_end)
    for head in range(0, NUM_HEADS):
        tl.store(
            selected_chunks_ptr + head * stride_selected_head + tokens * stride_selected_token,
            0,
            mask=valid,
        )


def _compute_selected_chunks_streaming_triton(
    q: torch.Tensor,
    chunk_means: torch.Tensor,
    chunk_end: torch.Tensor,
    batch_end: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    num_chunks, num_kv_heads, head_dim = chunk_means.shape
    num_heads = q.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    # The training scan writes ``chunk_start + lane`` through FP32 rank tiles,
    # exactly like the inference selector, so it needs the same exactness guard.
    if num_chunks >= (1 << 24):
        raise ValueError(
            f"training top-k needs {num_chunks} compacted chunks, but FP32 "
            f"chunk ids are exact only below {1 << 24}; split the batch"
        )
    if topk <= 0 or num_chunks == 0:
        return torch.empty(
            (0, num_heads, q.shape[0]),
            device=q.device,
            dtype=torch.int32,
        )
    if topk > num_chunks:
        raise ValueError(f"topk ({topk}) must not exceed the available chunks ({num_chunks})")
    if num_chunks == 1:
        # Triton-Ascend fails to lower the streaming kernel's single-iteration
        # chunk loop; the only reachable top-k is rank 0.
        selected_chunks = torch.full(
            (topk, num_heads, q.shape[0]),
            -1,
            device=q.device,
            dtype=torch.int32,
        )
        block = 512
        _single_chunk_selection_kernel[(cdiv(q.shape[0], block),)](
            chunk_end,
            batch_end,
            selected_chunks,
            q.shape[0],
            selected_chunks.stride(1),
            selected_chunks.stride(2),
            NUM_HEADS=num_heads,
            BLOCK=block,
        )
        return selected_chunks

    # See `_compute_selected_chunks_batched_streaming_triton`: rank 0 of a
    # Top-1 request keeps a masked second lane so no tile is one element wide.
    block_topk = max(2, triton.next_power_of_2(topk))
    if block_topk > 64:
        raise ValueError(f"streaming Top-K currently supports topk <= 64, got topk={topk}")
    selected_chunks = torch.full(
        (topk, num_heads, q.shape[0]),
        -1,
        device=q.device,
        dtype=torch.int32,
    )

    block_d = max(16, triton.next_power_of_2(head_dim))
    block_tok, block_chunk = _training_tile_shape(num_chunks, block_d, block_topk)
    num_token_blocks = cdiv(q.shape[0], block_tok)
    # Token blocks are independent, so the launch axis can be partitioned.
    blocks_per_launch = max(1, min(_TOPK_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID) // num_heads)
    for block_start in range(0, num_token_blocks, blocks_per_launch):
        block_end = min(block_start + blocks_per_launch, num_token_blocks)
        grid = (block_end - block_start, num_heads)
        _chunk_topk_streaming_kernel[grid](
            q,
            chunk_means,
            chunk_end,
            batch_end,
            selected_chunks,
            num_heads,
            q.shape[0],
            q.stride(0),
            q.stride(1),
            q.stride(2),
            chunk_means.stride(0),
            chunk_means.stride(1),
            chunk_means.stride(2),
            selected_chunks.stride(0),
            selected_chunks.stride(1),
            selected_chunks.stride(2),
            HEAD_DIM=head_dim,
            GROUP_SIZE=num_heads // num_kv_heads,
            NUM_CHUNKS=num_chunks,
            TOPK=topk,
            BLOCK_TOPK=block_topk,
            BLOCK_TOK=block_tok,
            BLOCK_CHUNK=block_chunk,
            BLOCK_D=block_d,
            pid_tok_offset=block_start,
            num_warps=4,
            num_stages=1,
        )
    return selected_chunks


def gather_moba_q(q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather flattened query rows, accepting NPU tensors."""
    if q.ndim != 3:
        raise ValueError(f"q must have shape [seqlen, heads, head_dim], got {q.shape}")
    if indices.ndim != 1:
        raise ValueError(f"indices must be one-dimensional, got {indices.shape}")
    if q.device != indices.device:
        raise ValueError("q and indices must be on the same device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must use int32 or int64, got {indices.dtype}")

    if torch.is_grad_enabled() and q.requires_grad:
        return _GatherMobaQNpu.apply(q, indices)
    return _gather_moba_q_forward(q, indices)


def _gather_moba_q_forward(q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Row gather without the CUDA-tuned autotune kernel (whose configs use
    ``cache_modifier=".cg"`` and L1-bypass hints that do not exist on 910B)."""
    head_dim = q.shape[-1]
    if indices.numel() == 0:
        return torch.empty((0, 1, head_dim), device=q.device, dtype=q.dtype)
    gathered = q.reshape(-1, head_dim).index_select(0, indices.to(torch.int64))
    return gathered.view(indices.numel(), 1, head_dim)


class _GatherMobaQNpu(torch.autograd.Function):
    """Gather MoBA query rows and scatter-accumulate their gradients.

    Triton-Ascend returns non-unique values from ``tl.atomic_add`` when several
    lanes hit the same address, so the CUDA backward kernel cannot be used.
    ``index_add_`` performs the same segmented reduction on the vendor side.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.q_shape = q.shape
        return _gather_moba_q_forward(q, indices)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indices,) = ctx.saved_tensors
        flat = grad_output.reshape(indices.numel(), -1).to(torch.float32)
        grad_q = torch.zeros(
            (ctx.q_shape[0] * ctx.q_shape[1], ctx.q_shape[2]),
            device=grad_output.device,
            dtype=torch.float32,
        )
        grad_q.index_add_(0, indices.to(torch.int64), flat)
        return grad_q.to(grad_output.dtype).view(ctx.q_shape), None


_ROUTE_SCATTER_BLOCK = 4096


@triton.jit
def _scatter_route_metadata_kernel(
    order_ptr,
    route_positions_ptr,
    num_slots,
    num_routes_ptr,
    start_offset,
    BLOCK: tl.constexpr,
):
    """Invert the expert-sorted permutation into compacted route positions.

    ``order[j]`` is the slot of the j-th compacted route, so ``route_positions``
    receives the global compacted position ``j`` at slot ``order[j]``.  Every
    address is written exactly once (``order`` is a permutation), which is what
    ``tl.atomic_add`` cannot guarantee on this hardware.  Invalid routes sort
    past ``num_routes`` and get -1.

    Only the scatter is done here: Triton-Ascend's integer ``//``/``%`` are
    approximate for values near 2**24 and would mangle the tail routes of every
    rank, so ``query_indices`` is derived with an exact torch remainder instead.
    """
    offsets = start_offset + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_slots
    slot = tl.load(order_ptr + offsets, mask=mask, other=0)
    num_routes = tl.load(num_routes_ptr)
    tl.store(
        route_positions_ptr + slot,
        tl.where(offsets < num_routes, offsets, -1),
        mask=mask,
    )


def _compact_routes_int64(selected_chunks: torch.Tensor, num_chunks: int, num_heads: int, seqlen: int):
    """Fallback for expert spaces that float32 cannot represent exactly.

    ``num_chunks * num_heads`` reaching 2**24 needs more than ``2**24 /
    num_heads`` chunks (1M chunks at 16 heads), which no supported sequence
    length produces; keep the exact int64 path anyway.
    """
    topk = selected_chunks.shape[0]
    device = selected_chunks.device
    num_slots = selected_chunks.numel()
    num_experts = num_chunks * num_heads

    flat = selected_chunks.permute(0, 2, 1).reshape(-1)
    slots = torch.arange(num_slots, device=device, dtype=torch.int64)
    heads = slots.remainder(num_heads)
    tokens = slots.div(num_heads, rounding_mode="floor").remainder(seqlen)
    valid = (flat >= 0) & (flat < num_chunks)
    experts = torch.where(valid, flat.to(torch.int64) * num_heads + heads, num_experts)

    order = torch.argsort(experts * num_slots + slots)
    positions_of_slot = torch.empty_like(slots)
    positions_of_slot.scatter_(0, order, slots)

    query_indices = (tokens.index_select(0, order) * num_heads + heads.index_select(0, order)).to(torch.int32)

    # Boolean-mask indexing (``t[pos] = ...``, ``t[mask]``) lowers to
    # aclnnNonzeroV2, which times out on device for the multi-million-element
    # route tensors of long sequences; select elementwise instead.
    route_positions = torch.where(valid, positions_of_slot, torch.full_like(slots, -1)).to(torch.int32)
    route_positions = route_positions.view(topk, seqlen, num_heads).reshape(topk, seqlen * num_heads)

    # Invalid routes were assigned expert id ``num_experts`` above, so counting
    # over one extra bucket and dropping it equals gathering the valid ids,
    # without a nonzero.
    expert_counts = torch.bincount(experts, minlength=num_experts + 1)[:num_experts].to(torch.int32)
    expert_offsets = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.int32), expert_counts.cumsum(0, dtype=torch.int32))
    )
    return query_indices, route_positions, expert_counts, expert_offsets


def _compact_routes(selected_chunks: torch.Tensor, num_chunks: int, num_heads: int, seqlen: int):
    """Exact, atomic-free compaction of (rank, head, token) routes.

    ``tl.atomic_add`` cannot allocate unique slots on Ascend, so routes are
    ordered by a sort on the expert id.  The sort key is FP32 because
    ``torch.sort`` runs int64 on AiCPU (6.5 s of a 7.6 s stage at 29.4M slots)
    while FP32 sort runs on AiCore (52 ms); every expert id is below 2**24, so
    the key is exact.  The resulting expert slices and per-route positions are
    equivalent to the CUDA counter-based compaction (any within-expert order is
    valid).

    Invalid routes (chunk id out of range) sort last under the sentinel
    ``num_experts``; they are not counted and their route position stays -1.

    The inverse permutation is obtained by sorting ``order`` itself: because
    ``order`` is a permutation of ``0 .. num_slots - 1``, ``torch.sort`` returns
    the identity as values and the inverse permutation as indices, and that is
    exactly ``positions_of_slot``.  An FP32 key represents every slot id exactly
    while ``num_slots < 2**24``; above that (and for the expert-space fallback
    below) the shipped scatter kernel is kept.
    """
    topk = selected_chunks.shape[0]
    device = selected_chunks.device
    num_slots = selected_chunks.numel()
    num_experts = num_chunks * num_heads
    # Slots, query indices and route positions are int32 throughout this path.
    # Reject the shape explicitly instead of wrapping the index arithmetic.
    if num_slots >= (1 << 31):
        raise ValueError(
            f"routing compaction needs {num_slots} route slots, but route and "
            "query indices are int32; split the batch into smaller calls"
        )
    if seqlen * num_heads >= (1 << 31):
        raise ValueError(
            f"routing compaction needs a {seqlen * num_heads}-element query "
            "index space, which exceeds int32; split the batch"
        )
    if num_experts >= (1 << 24):
        return _compact_routes_int64(selected_chunks, num_chunks, num_heads, seqlen)
    if num_slots == 0:
        route_positions = torch.empty(
            (topk, seqlen * num_heads), device=device, dtype=torch.int32
        )
        return (
            torch.empty((0,), device=device, dtype=torch.int32),
            route_positions,
            torch.zeros((num_experts,), device=device, dtype=torch.int32),
            torch.zeros((num_experts + 1,), device=device, dtype=torch.int32),
        )

    # [rank, token, head] order makes expert = chunk * heads + head.
    flat = selected_chunks.permute(0, 2, 1).reshape(-1)
    heads_of_slot = torch.arange(num_slots, device=device, dtype=torch.int32) % num_heads
    valid = (flat >= 0) & (flat < num_chunks)
    experts = torch.where(valid, flat * num_heads + heads_of_slot, num_experts)

    sorted_experts, order = torch.sort(experts.to(torch.float32))
    # Offsets come from a binary search over the sorted key: no bincount (which
    # is also AiCPU) and no cumsum over the route axis are needed.
    limits = torch.arange(num_experts + 1, device=device, dtype=torch.float32)
    expert_offsets = torch.searchsorted(sorted_experts, limits).to(torch.int32)
    expert_counts = expert_offsets[1:] - expert_offsets[:-1]

    route_positions = torch.empty((num_slots,), device=device, dtype=torch.int32)
    # slots are laid out as [rank, token, head], so the compacted query index of
    # a slot is exactly slot % (seqlen * num_heads).
    order = order.to(torch.int32)
    query_indices = order % (seqlen * num_heads)
    if num_slots < (1 << 24):
        # ``order`` is a permutation, so sorting it returns (arange, order**-1)
        # and the indices are the compacted position of every slot.  Two FP32
        # sorts (expert key + inverse key) are still cheaper than the scatter
        # kernel: the Ascend FP32 sort is AiCore-resident, the scatter is not.
        _, positions_of_slot = torch.sort(order.to(torch.float32))
        route_positions = torch.where(
            valid,
            positions_of_slot.to(torch.int32),
            torch.full_like(order, -1),
        )
    else:
        # FP32 keys would round slot ids >= 2**24 and break the inverse.  The
        # grid is legality-bound, so walk the slots in 65535-program chunks
        # instead of emitting one oversized launch.
        num_blocks = cdiv(num_slots, _ROUTE_SCATTER_BLOCK)
        for block_start in range(0, num_blocks, _MAX_LAUNCH_GRID):
            block_end = min(block_start + _MAX_LAUNCH_GRID, num_blocks)
            _scatter_route_metadata_kernel[(block_end - block_start,)](
                order,
                route_positions,
                num_slots,
                expert_offsets[num_experts:],
                block_start * _ROUTE_SCATTER_BLOCK,
                BLOCK=_ROUTE_SCATTER_BLOCK,
                num_warps=4,
                num_stages=1,
            )
    route_positions = route_positions.view(topk, seqlen, num_heads).reshape(topk, seqlen * num_heads)
    return query_indices, route_positions, expert_counts, expert_offsets


def _build_route_descriptors(expert_counts, expert_offsets, num_experts, num_slots, block_m, device):
    """Deterministic (expert, tile start) list sized by the CUDA capacity bound.

    Descriptors are expert-major, one per ``block_m`` routes of an expert.  They
    are derived with a vectorized binary search over the inclusive tile prefix
    sum instead of ``repeat_interleave``, which ran on AiCPU (167 ms at 491K
    descriptors).  No route counts are read back to Python.
    """
    capacity = cdiv(num_slots, block_m) + num_experts
    if num_experts == 0:
        empty = torch.full((capacity,), -1, device=device, dtype=torch.int32)
        return empty, torch.full_like(empty, -1)

    tile_counts = (expert_counts + (block_m - 1)) // block_m
    tile_ends = tile_counts.cumsum(0, dtype=torch.int32)
    tile_offsets = tile_ends - tile_counts
    total_tiles = tile_ends[-1]
    positions = torch.arange(capacity, device=device, dtype=torch.int32)
    # First expert whose inclusive tile end exceeds the descriptor index. Empty
    # experts share their end with the previous expert and are skipped.
    expert_of_tile = torch.searchsorted(tile_ends, positions, right=True).to(torch.int32)
    live = positions < total_tiles
    safe_experts = torch.where(live, expert_of_tile, 0).clamp_(0, num_experts - 1).to(torch.int64)
    descriptor_experts = torch.where(live, expert_of_tile, -1)
    descriptor_starts = torch.where(
        live,
        expert_offsets.index_select(0, safe_experts)
        + (positions - tile_offsets.index_select(0, safe_experts)) * block_m,
        -1,
    )
    return descriptor_experts, descriptor_starts


def _build_sparse_routes_device_triton(
    selected_chunks: torch.Tensor,
    num_chunks: int,
    num_heads: int,
    seqlen: int,
    block_m: int,
):
    """Build route metadata without reading route counts back to Python."""
    num_slots = selected_chunks.numel()
    (
        query_indices,
        route_positions,
        _expert_counts,
        expert_offsets,
    ) = _compact_routes(selected_chunks, num_chunks, num_heads, seqlen)
    num_experts = num_chunks * num_heads
    descriptor_experts, descriptor_starts = _build_route_descriptors(
        _expert_counts,
        expert_offsets,
        num_experts,
        num_slots,
        block_m,
        selected_chunks.device,
    )
    return query_indices, route_positions, expert_offsets, descriptor_experts, descriptor_starts


def _build_sparse_routes_triton(
    selected_chunks: torch.Tensor,
    num_chunks: int,
    num_heads: int,
    seqlen: int,
):
    """Group selected queries by (target chunk, head) for the autograd path."""
    (
        query_indices,
        route_positions,
        expert_counts,
        expert_offsets,
    ) = _compact_routes(selected_chunks, num_chunks, num_heads, seqlen)
    num_routes = int(expert_offsets[-1].item())
    max_count = int(expert_counts.max().item()) if expert_counts.numel() else 0
    return (
        query_indices[:num_routes],
        route_positions,
        expert_counts,
        expert_offsets,
        max_count,
    )


def _compute_chunk_means_triton(
    k: torch.Tensor,
    cu_chunk: torch.Tensor,
    filtered_chunk_indices: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Ascend launch policy for the shared per-chunk key-mean kernel.

    The generic launch requests two pipeline stages, which the BiSheng compiler
    turns into an over-sized unified buffer for strided inputs. A single stage
    keeps the same arithmetic with a discrete-program footprint.
    """
    num_chunks = filtered_chunk_indices.numel()
    _, num_heads, head_dim = k.shape
    if num_chunks == 0:
        return torch.empty((0, num_heads, head_dim), device=k.device, dtype=k.dtype)
    chunk_means = torch.empty((num_chunks, num_heads, head_dim), device=k.device, dtype=k.dtype)
    block_tok = min(64, triton.next_power_of_2(chunk_size))
    block_d = max(16, triton.next_power_of_2(head_dim))
    # The kernel indexes the source chunk and the output by its local program
    # id, so both are sliced together.
    chunks_per_launch = max(
        1, min(_CHUNK_MEAN_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID) // num_heads
    )
    for chunk_start in range(0, num_chunks, chunks_per_launch):
        chunk_end = min(chunk_start + chunks_per_launch, num_chunks)
        _chunk_mean_kernel[(chunk_end - chunk_start, num_heads)](
            k,
            cu_chunk,
            filtered_chunk_indices[chunk_start:chunk_end],
            chunk_means[chunk_start:chunk_end],
            k.stride(0),
            k.stride(1),
            k.stride(2),
            chunk_means.stride(0),
            chunk_means.stride(1),
            chunk_means.stride(2),
            chunk_end - chunk_start,
            num_heads,
            CHUNK_SIZE=chunk_size,
            HEAD_DIM=head_dim,
            BLOCK_TOK=block_tok,
            BLOCK_D=block_d,
            num_warps=4 if block_d >= 64 else 2,
            num_stages=1,
        )
    return chunk_means


def _compute_dense_chunk_means_triton(
    k: torch.Tensor,
    cu_chunk: torch.Tensor,
    valid_historical: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Ascend launch policy for the dense-slot key-mean kernel."""
    num_chunk_slots = valid_historical.numel()
    _, num_kv_heads, head_dim = k.shape
    chunk_means = torch.empty(
        (num_chunk_slots, num_kv_heads, head_dim), device=k.device, dtype=k.dtype
    )
    block_tok = min(64, triton.next_power_of_2(chunk_size))
    block_d = max(16, triton.next_power_of_2(head_dim))
    slots_per_launch = max(
        1, min(_CHUNK_MEAN_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID) // num_kv_heads
    )
    for slot_start in range(0, num_chunk_slots, slots_per_launch):
        slot_end = min(slot_start + slots_per_launch, num_chunk_slots)
        _dense_chunk_mean_kernel[(slot_end - slot_start, num_kv_heads)](
            k,
            cu_chunk[slot_start:slot_end],
            valid_historical[slot_start:slot_end],
            chunk_means[slot_start:slot_end],
            k.stride(0),
            k.stride(1),
            k.stride(2),
            chunk_means.stride(0),
            chunk_means.stride(1),
            chunk_means.stride(2),
            slot_end - slot_start,
            num_kv_heads,
            CHUNK_SIZE=chunk_size,
            HEAD_DIM=head_dim,
            BLOCK_TOK=block_tok,
            BLOCK_D=block_d,
            num_warps=4 if block_d >= 64 else 2,
            num_stages=1,
        )
    return chunk_means


def _gather_moba_backward_inputs_triton(
    d_output: torch.Tensor,
    output: torch.Tensor,
    mixed_attn_vlse_flat: torch.Tensor,
    moba_indices: torch.Tensor,
    gathered_d_output: torch.Tensor,
    gathered_output: torch.Tensor,
    gathered_lse: torch.Tensor,
) -> None:
    """Row-chunked launch of the shared backward-input gather kernel.

    The kernel reads ``indices[pid]`` and writes row ``pid`` of every gathered
    tensor, so contiguous row slices are independent launches.
    """
    num_indices = moba_indices.numel()
    if num_indices == 0:
        return
    head_dim = d_output.shape[2]
    block_d = min(128, triton.next_power_of_2(head_dim))
    rows_per_launch = min(_GATHER_ROWS_PER_SLICE, _MAX_LAUNCH_GRID)
    for row_start in range(0, num_indices, rows_per_launch):
        row_end = min(row_start + rows_per_launch, num_indices)
        _gather_moba_backward_inputs_kernel[(row_end - row_start,)](
            d_output,
            output,
            mixed_attn_vlse_flat,
            moba_indices[row_start:row_end],
            gathered_d_output[row_start:row_end],
            gathered_output[row_start:row_end],
            gathered_lse[row_start:row_end],
            row_end - row_start,
            head_dim,
            d_output.stride(0),
            d_output.stride(1),
            d_output.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            gathered_d_output.stride(0),
            gathered_d_output.stride(1),
            gathered_output.stride(0),
            gathered_output.stride(1),
            mixed_attn_vlse_flat.stride(0),
            mixed_attn_vlse_flat.stride(1),
            num_heads=mixed_attn_vlse_flat.shape[0],
            BLOCK_D=block_d,
        )


class _ExpandKVHeadsNpu(torch.autograd.Function):
    """GQA head expansion with a legal token-chunked launch grid.

    The shared kernels map ``program_id(0)`` to a token, so token slices of the
    packed tensors are independent launches.
    """

    @staticmethod
    def forward(ctx, k, v, num_q_heads):
        total_tokens, num_kv_heads, head_dim = k.shape
        group_size = num_q_heads // num_kv_heads
        expanded_k = torch.empty((total_tokens, num_q_heads, head_dim), device=k.device, dtype=k.dtype)
        expanded_v = torch.empty_like(expanded_k)
        block_d = max(16, triton.next_power_of_2(head_dim))
        tokens_per_launch = max(
            1, min(_EXPAND_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID) // num_q_heads
        )
        for token_start in range(0, total_tokens, tokens_per_launch):
            token_end = min(token_start + tokens_per_launch, total_tokens)
            k_slice, v_slice = k[token_start:token_end], v[token_start:token_end]
            ek_slice = expanded_k[token_start:token_end]
            ev_slice = expanded_v[token_start:token_end]
            _expand_kv_heads_kernel[(token_end - token_start, num_q_heads)](
                k_slice,
                v_slice,
                ek_slice,
                ev_slice,
                token_end - token_start,
                k_slice.stride(0),
                k_slice.stride(1),
                k_slice.stride(2),
                v_slice.stride(0),
                v_slice.stride(1),
                v_slice.stride(2),
                ek_slice.stride(0),
                ek_slice.stride(1),
                ek_slice.stride(2),
                ev_slice.stride(0),
                ev_slice.stride(1),
                ev_slice.stride(2),
                NUM_Q_HEADS=num_q_heads,
                GROUP_SIZE=group_size,
                HEAD_DIM=head_dim,
                BLOCK_D=block_d,
            )
        ctx.input_shape = k.shape
        ctx.num_q_heads = num_q_heads
        return expanded_k, expanded_v

    @staticmethod
    def backward(ctx, d_expanded_k, d_expanded_v):
        total_tokens, num_kv_heads, head_dim = ctx.input_shape
        dk = torch.zeros(ctx.input_shape, device=d_expanded_k.device, dtype=d_expanded_k.dtype)
        dv = torch.zeros(ctx.input_shape, device=d_expanded_v.device, dtype=d_expanded_v.dtype)
        block_d = max(16, triton.next_power_of_2(head_dim))
        tokens_per_launch = max(
            1, min(_EXPAND_PROGRAMS_PER_SLICE, _MAX_LAUNCH_GRID) // num_kv_heads
        )
        for token_start in range(0, total_tokens, tokens_per_launch):
            token_end = min(token_start + tokens_per_launch, total_tokens)
            dek_slice = d_expanded_k[token_start:token_end]
            dev_slice = d_expanded_v[token_start:token_end]
            dk_slice, dv_slice = dk[token_start:token_end], dv[token_start:token_end]
            _reduce_expanded_kv_grads_kernel[(token_end - token_start, num_kv_heads)](
                dek_slice,
                dev_slice,
                dk_slice,
                dv_slice,
                token_end - token_start,
                dek_slice.stride(0),
                dek_slice.stride(1),
                dek_slice.stride(2),
                dev_slice.stride(0),
                dev_slice.stride(1),
                dev_slice.stride(2),
                dk_slice.stride(0),
                dk_slice.stride(1),
                dk_slice.stride(2),
                dv_slice.stride(0),
                dv_slice.stride(1),
                dv_slice.stride(2),
                NUM_KV_HEADS=num_kv_heads,
                GROUP_SIZE=ctx.num_q_heads // num_kv_heads,
                HEAD_DIM=head_dim,
                BLOCK_D=block_d,
            )
        return dk, dv, None


def _expand_kv_heads_triton(k, v, num_q_heads):
    if k.shape[1] == num_q_heads:
        return k, v
    return _ExpandKVHeadsNpu.apply(k, v, num_q_heads)


class _GatherMobaKVNpu(torch.autograd.Function):
    """Gather per-target-chunk K/V with a legal launch grid.

    Rows are laid out as ``(target_chunk, head, chunk_token)``, so slices of
    whole target chunks are contiguous in both the source and destination
    tensors and can be launched independently.
    """

    @staticmethod
    def _chunks_per_launch(num_heads: int, chunk_size: int) -> int:
        rows_per_chunk = num_heads * chunk_size
        # A single target chunk cannot be split (its rows are one contiguous
        # block in source and destination), so only the legality limit applies.
        if rows_per_chunk > _MAX_LAUNCH_GRID:
            raise RuntimeError(
                f"a single MoBA target chunk spans {rows_per_chunk} rows "
                f"({num_heads} heads x {chunk_size} tokens), which exceeds the "
                f"Ascend grid limit of {_MAX_LAUNCH_GRID}"
            )
        return max(1, min(_GATHER_KV_ROWS_PER_SLICE, _MAX_LAUNCH_GRID) // rows_per_chunk)

    @staticmethod
    def forward(ctx, k, v, cu_chunk, filtered_chunk_indices, chunk_size):
        total_tokens, num_heads, head_dim = k.shape
        num_chunks = filtered_chunk_indices.numel()
        num_rows = num_chunks * num_heads * chunk_size
        moba_kv = torch.empty((num_rows, 2, 1, head_dim), device=k.device, dtype=k.dtype)
        block_d = max(16, triton.next_power_of_2(head_dim))
        rows_per_chunk = num_heads * chunk_size
        chunks_per_launch = _GatherMobaKVNpu._chunks_per_launch(num_heads, chunk_size)
        for chunk_start in range(0, num_chunks, chunks_per_launch):
            chunk_end = min(chunk_start + chunks_per_launch, num_chunks)
            row_start = chunk_start * rows_per_chunk
            row_end = chunk_end * rows_per_chunk
            moba_kv_slice = moba_kv[row_start:row_end]
            _gather_moba_kv_kernel[(row_end - row_start,)](
                k,
                v,
                cu_chunk,
                filtered_chunk_indices[chunk_start:chunk_end],
                moba_kv_slice,
                row_end - row_start,
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                moba_kv_slice.stride(0),
                moba_kv_slice.stride(1),
                moba_kv_slice.stride(3),
                NUM_HEADS=num_heads,
                CHUNK_SIZE=chunk_size,
                HEAD_DIM=head_dim,
                BLOCK_D=block_d,
            )
        ctx.save_for_backward(cu_chunk, filtered_chunk_indices)
        ctx.input_shape = (total_tokens, num_heads, head_dim)
        ctx.chunk_size = chunk_size
        return moba_kv

    @staticmethod
    def backward(ctx, dmoba_kv):
        cu_chunk, filtered_chunk_indices = ctx.saved_tensors
        total_tokens, num_heads, head_dim = ctx.input_shape
        dk = torch.zeros(ctx.input_shape, device=dmoba_kv.device, dtype=dmoba_kv.dtype)
        dv = torch.zeros(ctx.input_shape, device=dmoba_kv.device, dtype=dmoba_kv.dtype)
        num_rows = dmoba_kv.shape[0]
        block_d = max(16, triton.next_power_of_2(head_dim))
        rows_per_chunk = num_heads * ctx.chunk_size
        chunks_per_launch = _GatherMobaKVNpu._chunks_per_launch(num_heads, ctx.chunk_size)
        for chunk_start in range(0, filtered_chunk_indices.numel(), chunks_per_launch):
            chunk_end = min(chunk_start + chunks_per_launch, filtered_chunk_indices.numel())
            row_start = chunk_start * rows_per_chunk
            row_end = min(chunk_end * rows_per_chunk, num_rows)
            dmoba_slice = dmoba_kv[row_start:row_end]
            _scatter_moba_kv_grads_kernel[(row_end - row_start,)](
                dmoba_slice,
                cu_chunk,
                filtered_chunk_indices[chunk_start:chunk_end],
                dk,
                dv,
                row_end - row_start,
                dmoba_slice.stride(0),
                dmoba_slice.stride(1),
                dmoba_slice.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                NUM_HEADS=num_heads,
                CHUNK_SIZE=ctx.chunk_size,
                HEAD_DIM=head_dim,
                BLOCK_D=block_d,
            )
        return dk, dv, None, None, None


def _gather_moba_kv_triton(k, v, cu_chunk, filtered_chunk_indices, chunk_size):
    return _GatherMobaKVNpu.apply(k, v, cu_chunk, filtered_chunk_indices, chunk_size)


def _select_topk_head_slices(
    q, chunk_means, chunk_end, cu_seqlens, max_seqlen, max_chunks, topk,
    chunk_size, selected_chunks, head_limit,
):
    """Fit head slices into both launch budgets while preserving GQA mapping."""
    kv_heads = chunk_means.shape[1]
    group_size = q.shape[1] // kv_heads
    kv_per_slice = max(1, head_limit // group_size)
    for kv_start in range(0, kv_heads, kv_per_slice):
        kv_end = min(kv_start + kv_per_slice, kv_heads)
        group_end = kv_end * group_size
        # A slice can contain several full KV groups or part of one KV group.
        for start in range(kv_start * group_size, group_end, head_limit):
            end = min(start + head_limit, group_end)
            _compute_selected_chunks_batched_streaming_triton(
                q[:, start:end], chunk_means[:, kv_start:kv_end], chunk_end,
                cu_seqlens, max_seqlen, max_chunks, topk, chunk_size,
                _out=selected_chunks[:, start:end],
            )


__all__ = [
    "_build_dense_chunk_metadata_triton",
    "_build_sparse_routes_device_triton",
    "_build_sparse_routes_triton",
    "_compute_chunk_means_triton",
    "_compute_dense_chunk_means_triton",
    "_compute_selected_chunks_batched_streaming_triton",
    "_compute_selected_chunks_streaming_triton",
    "_expand_kv_heads_triton",
    "_gather_moba_backward_inputs_triton",
    "_gather_moba_kv_triton",
    "_linear_offsets_triton",
    "_target_chunk_bounds_triton",
    "_validated_chunk_metadata",
    "gather_moba_q",
]
