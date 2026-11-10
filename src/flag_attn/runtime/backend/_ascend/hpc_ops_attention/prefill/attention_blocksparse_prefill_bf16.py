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

"""Ascend HY3 BF16 block-sparse prefill.

Q/K/V are BF16 tensors. The staged QK, softmax, PV and split reduction
kernels consume them directly, without quantization or conversion buffers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import triton

from .constants import (
    _BLOCK_N,
    _HEAD_DIM,
    _LOGICAL_BLOCK_M,
    _PREFILL_BLOCK_M,
    _PREFILL_HEADS_PER_PROG_MAX,
    _PREFILL_LIST_MIN_KV_TILES,
    _PREFILL_QK_BLOCK_M,
    _PREFILL_QK_BLOCK_M_MASKED,
    _PREFILL_STAGE_CAP,
    _PREFILL_TOKENS_PER_SPLIT_MAX,
    _PREFILL_UNMASK_SM,
    _PREFILL_VEC_ROWS,
    _SCORES_ROW_PAD,
    _TILE_STEP,
)
from .pv import (
    _bf16_prefill_pv_accum_kernel,
)
from .qk import (
    _bf16_prefill_qk_compact_kernel,
    _bf16_prefill_qk_kernel,
)
from .softmax import (
    _bf16_prefill_softmax_splits_kernel,
)
from .splits import (
    _bf16_prefill_finalize_kernel,
)


# Scheduling controls are read when attention is launched; None selects the policy.
_PREFILL_SOFTMAX_HPP_OVERRIDE = None
# QK skip modes: None follows the mask policy, 0 visits every block,
# 1 skips every block for diagnostics, and 2 follows the block mask.
_QK_SKIP_OVERRIDE = None
_QK_SKIP_COUNT = False
_SKIP_COUNTERS: dict = {}
# None selects compaction from the sparsity declaration; a bool forces the choice.
_QK_LIST_OVERRIDE = None
_QK_LIST_CACHE: dict = {}


def _qk_active_lists(
    device: torch.device,
    batch: int,
    num_head_q: int,
    num_q_tiles: int,
    max_splits: int,
    tiles_per_split: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached (active-block list, per-segment counts) for the compaction path."""
    key = (str(device), batch, num_head_q, num_q_tiles, max_splits, tiles_per_split)
    bufs = _QK_LIST_CACHE.get(key)
    if bufs is None:
        cells = batch * num_head_q * num_q_tiles * max_splits
        active = torch.empty(
            cells * tiles_per_split, dtype=torch.int32, device=device
        )
        counts = torch.empty(cells, dtype=torch.int32, device=device)
        bufs = (active, counts)
        _QK_LIST_CACHE[key] = bufs
    return bufs


def _skip_counter(device: torch.device, slot: int = 0) -> torch.Tensor:
    key = (device, slot)
    counter = _SKIP_COUNTERS.get(key)
    if counter is None:
        counter = torch.zeros(1 << 20, dtype=torch.int32, device=device)
        _SKIP_COUNTERS[key] = counter
    return counter


@dataclass
class AscendBF16PrefillWorkspace:
    q_shape: tuple[int, ...]
    cache_shape: tuple[int, ...]
    device: torch.device
    scores: torch.Tensor
    probabilities: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    split_sum: torch.Tensor
    max_splits: int
    padded_q: int
    q_tokens: int
    tokens_per_split: int
    # Whether the three small kernels can use fully unmasked load/store (whole
    # split, stage divides splits, and q is also a multiple of q_tokens)
    unmasked: bool


def _prefill_q_tokens(num_head_q: int, num_head_kv: int) -> int:
    del num_head_q, num_head_kv
    return _PREFILL_BLOCK_M


def _prefill_tokens_per_split(max_tokens: int) -> int:
    """Choose a bounded KV split size aligned to the QK/PV inner tile.

    Use the fewest splits allowed by the split-size cap, then round the split
    width to a multiple of _TILE_STEP. Softmax masks the extra lanes of its
    power-of-two vector width, so the split width need not be a power of two.
    """
    splits = max(1, -(-max_tokens // _PREFILL_TOKENS_PER_SPLIT_MAX))
    tile = -(-max_tokens // splits)
    return max(_TILE_STEP, -(-tile // _TILE_STEP) * _TILE_STEP)


def _prefill_scores_vec(tokens_per_split: int) -> int:
    """Softmax vector width: the tile is rounded up to a power of two."""
    return triton.next_power_of_2(tokens_per_split)


def _prefill_scores_exact(tokens_per_split: int) -> bool:
    """Whether the tile already is a power of two, i.e. has no overhang lanes."""
    return tokens_per_split == triton.next_power_of_2(tokens_per_split)


def _validate(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    block_mask: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> None:
    tensors = (
        q,
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
    )
    if q.device.type != "npu":
        raise ValueError("Ascend BF16 prefill requires NPU tensors")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("all inputs must be on the same NPU device")
    if q.dtype != torch.bfloat16:
        raise TypeError("q must use torch.bfloat16")
    if q.ndim != 3 or q.shape[2] != _HEAD_DIM or q.stride(2) != 1:
        raise ValueError("q must have shape [total_q,num_head_q,128]")
    for name, cache in (("kcache", kcache), ("vcache", vcache)):
        if cache.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16")
        if cache.ndim != 4 or cache.shape[1] not in (32, 64):
            raise ValueError(f"{name} must have shape [pages,32|64,num_head_kv,128]")
        if cache.shape[3] != _HEAD_DIM or cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous 128")
    if kcache.shape != vcache.shape:
        raise ValueError("K and V cache shapes must match")
    if q.shape[1] <= 0 or kcache.shape[2] <= 0 or q.shape[1] % kcache.shape[2]:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    batch = int(seqlens_kvcache.numel())
    if batch <= 0 or max_seqlens_q <= 0:
        raise ValueError("batch and max_seqlens_q must be positive")
    if block_ids.ndim == 2 and block_ids.shape[1] <= 0:
        raise ValueError("block_ids must contain at least one logical KV page")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("cu_seqlens_q must be int32 [batch+1]")
    if block_ids.dtype != torch.int32 or block_ids.ndim != 2 or block_ids.shape[0] != batch:
        raise ValueError("block_ids must be int32 [batch,max_pages]")
    if seqlens_kvcache.dtype != torch.int32 or seqlens_kvcache.ndim != 1:
        raise ValueError("seqlens_kvcache must be int32 [batch]")
    if block_mask is not None:
        if block_mask.device != q.device or block_mask.dtype != torch.uint8:
            raise TypeError("block_mask must be uint8 on the Q device")
        if not block_mask.is_contiguous() or block_mask.ndim != 4:
            raise ValueError("block_mask must be contiguous rank-4")
        if block_mask.shape[:2] != (batch, q.shape[1]):
            raise ValueError("block_mask batch/head dimensions do not match Q")
        expected_q_tiles = triton.cdiv(max_seqlens_q, _LOGICAL_BLOCK_M)
        if block_mask.shape[2] != expected_q_tiles or block_mask.shape[3] <= 0:
            raise ValueError(
                "block_mask must have shape [batch,num_head_q,ceil(max_q/128),Kb]"
            )
    if output is not None:
        if (
            output.device != q.device
            or output.dtype != torch.bfloat16
            or output.shape != q.shape
            or output.stride(2) != 1
        ):
            raise ValueError("output must be BF16 on the Q device with Q shape")


def prepare_attention_blocksparse_prefill_bf16_workspace(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
) -> AscendBF16PrefillWorkspace:
    """Allocate compute buffers for the given BF16 shapes and sequence metadata.

    Prepare before graph capture and rebuild when the sequence metadata changes.
    Q/K/V are read from each attention call rather than copied into the workspace.
    """
    _validate(
        q,
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        None,
        None,
    )
    batch = int(seqlens_kvcache.numel())
    hq = int(q.shape[1])
    page_size = int(kcache.shape[1])
    max_tokens = int(block_ids.shape[1]) * page_size
    tokens_per_split = _prefill_tokens_per_split(max_tokens)
    max_splits = max(1, triton.cdiv(max_tokens, tokens_per_split))
    # Bound the number of split slots held in the intermediate scores buffer.
    # Launch-time row chunking also respects the flattened grid limit.
    stage_capacity = max(1, min(_PREFILL_STAGE_CAP, max_splits))
    q_tokens = _prefill_q_tokens(hq, int(kcache.shape[2]))
    padded_q = triton.cdiv(max_seqlens_q, q_tokens) * q_tokens
    # Resolve the unmasked path during preparation to avoid host synchronization
    # inside graph capture. It requires full query tiles and KV splits, a stage
    # depth that divides the split count, and no softmax vector overhang.
    stage_slots_eff = min(stage_capacity, max_splits)
    kv_lengths = [int(x) for x in seqlens_kvcache.reshape(-1).tolist()]
    q_lengths = [
        int(cu_seqlens_q[i + 1]) - int(cu_seqlens_q[i])
        for i in range(int(cu_seqlens_q.numel()) - 1)
    ]
    unmasked = bool(
        kv_lengths
        and all(length % tokens_per_split == 0 for length in kv_lengths)
        and max_splits % stage_slots_eff == 0
        and all(length % q_tokens == 0 for length in q_lengths)
        and _prefill_scores_exact(tokens_per_split)
    )
    return AscendBF16PrefillWorkspace(
        q_shape=tuple(q.shape),
        cache_shape=tuple(kcache.shape),
        device=q.device,
        # Head-major [batch, stage, head, query, token] buffers keep each
        # head's probability rows adjacent for PV tile loads.
        scores=torch.empty(
            (
                batch,
                stage_capacity,
                hq,
                padded_q,
                tokens_per_split
                + (_SCORES_ROW_PAD if (tokens_per_split * 2) % 4096 == 0 else 0),
            ),
            dtype=torch.bfloat16,
            device=q.device,
        ),
        probabilities=torch.empty(
            (batch, stage_capacity, hq, padded_q, tokens_per_split),
            dtype=torch.bfloat16,
            device=q.device,
        ),
        split_out=torch.empty(
            (batch, max_splits, padded_q, hq, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        ),
        split_lse=torch.empty(
            (batch, max_splits, padded_q, hq),
            dtype=torch.float32,
            device=q.device,
        ),
        split_sum=torch.empty(
            (batch, max_splits, padded_q, hq),
            dtype=torch.float32,
            device=q.device,
        ),
        max_splits=max_splits,
        padded_q=padded_q,
        q_tokens=q_tokens,
        tokens_per_split=tokens_per_split,
        unmasked=unmasked,
    )


def attention_with_kvcache_blocksparse_prefill_bf16(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    sparsity_bucket: Optional[int] = None,
    workspace: Optional[AscendBF16PrefillWorkspace] = None,
) -> torch.Tensor:
    """Run causal paged prefill with BF16 Q/K/V and FP32 accumulation/softmax.

    Q has shape [total_q, Hq, 128]; K/V have logical shape
    [pages, 32|64, Hkv, 128], supporting NHD and HND strides. Queries align
    with the final query-length tokens of each KV sequence. An optional uint8
    block mask selects 128-by-128 tiles; fully masked rows produce zero.
    """
    # Compaction requires both an explicit sparse bucket and enough KV tiles
    # to amortize the active-list construction and score-buffer initialization.
    list_allowed = (
        sparsity_bucket is not None
        and int(sparsity_bucket) in (2, 3)
        and (
            block_mask is None
            or int(block_mask.shape[3]) >= _PREFILL_LIST_MIN_KV_TILES
        )
    )

    _validate(
        q,
        kcache,
        vcache,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        block_mask,
        output,
    )
    if output is None:
        output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    batch = int(seqlens_kvcache.numel())
    num_head_q = int(q.shape[1])
    num_head_kv = int(kcache.shape[2])
    page_size = int(kcache.shape[1])
    if workspace is None:
        workspace = prepare_attention_blocksparse_prefill_bf16_workspace(
            q,
            kcache,
            vcache,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
        )
    max_tokens = int(block_ids.shape[1]) * page_size
    expected_tokens_per_split = _prefill_tokens_per_split(max_tokens)
    expected_splits = max(1, triton.cdiv(max_tokens, expected_tokens_per_split))
    expected_q_tokens = _prefill_q_tokens(num_head_q, num_head_kv)
    if (
        workspace.max_splits != expected_splits
        or workspace.tokens_per_split != expected_tokens_per_split
        or workspace.q_tokens != expected_q_tokens
        or workspace.padded_q < max_seqlens_q
        or workspace.q_shape != tuple(q.shape)
        or workspace.cache_shape != tuple(kcache.shape)
        or workspace.device != q.device
        or workspace.scores.shape[0] != batch
    ):
        raise ValueError("workspace does not match BF16 prefill inputs")
    mask_arg = q if block_mask is None else block_mask
    mask_strides = (0, 0, 0, 0) if block_mask is None else block_mask.stride()
    stage_slots = min(int(workspace.scores.shape[1]), workspace.max_splits)
    # Keep both the softmax row grid and the QK/PV tile grids within the
    # backend's flattened launch limit of 65535 programs.
    _per_row = num_head_q * batch
    _rows_budget = min(
        65535 // max(1, _per_row),
        (65535 // max(1, stage_slots * num_head_q * batch)) * workspace.q_tokens,
    )
    rows_per_chunk = max(
        workspace.q_tokens,
        (_rows_budget // workspace.q_tokens) * workspace.q_tokens,
    )
    rows_per_chunk = min(rows_per_chunk, workspace.padded_q)
    # QK uses independent row chunks because scores use absolute query indices.
    # With a mask, align its tile to one logical mask row for block skipping;
    # without a mask, use a larger tile to reduce the number of programs.
    qk_tile = (
        _PREFILL_QK_BLOCK_M_MASKED if block_mask is not None else _PREFILL_QK_BLOCK_M
    )
    # Each QK tile occupies stage_slots * num_head_q * batch program slots.
    # Bound the row chunk so the flattened grid stays within 65535 programs.
    _qk_rows_budget = (65535 * qk_tile) // max(1, stage_slots * num_head_q * batch)
    qk_rows_per_chunk = max(
        qk_tile,
        (_qk_rows_budget // qk_tile) * qk_tile,
    )
    qk_rows_per_chunk = min(qk_rows_per_chunk, workspace.padded_q)
    # Compaction list path: with a mask, scan BLOCK_MASK first and for each
    # (batch, head, q tile, split) list the active 128-token blocks qk then visits.
    use_list = bool(
        block_mask is not None
        and qk_tile == _LOGICAL_BLOCK_M
        and list_allowed
        and (_QK_LIST_OVERRIDE is None or _QK_LIST_OVERRIDE)
    )
    act_lists = act_counts = None
    num_q_tiles = triton.cdiv(workspace.padded_q, _LOGICAL_BLOCK_M)
    tiles_per_split = workspace.tokens_per_split // _LOGICAL_BLOCK_M
    if use_list:
        act_lists, act_counts = _qk_active_lists(
            workspace.scores.device, batch, num_head_q, num_q_tiles,
            workspace.max_splits, tiles_per_split,
        )
        _bf16_prefill_qk_compact_kernel[
            (num_q_tiles * workspace.max_splits, num_head_q, batch)
        ](
            mask_arg, act_lists, act_counts,
            MASK_SBATCH=mask_strides[0], MASK_SHEAD=mask_strides[1],
            MASK_SQTILE=mask_strides[2], MASK_SKVTILE=mask_strides[3],
            NUM_HEAD_Q=num_head_q,
            NUM_MASK_KV_TILES=int(block_mask.shape[3]),
            MAX_SPLITS=workspace.max_splits, TILES_PER_SPLIT=tiles_per_split,
            NUM_Q_TILES=num_q_tiles,
            num_warps=4, num_stages=1,
        )
    for split_base in range(0, workspace.max_splits, stage_slots):
      if use_list:
        # Listed QK writes only active blocks. Initialize the current score
        # slots to -inf before QK and before softmax/PV consume them.
        workspace.scores[:, :stage_slots].fill_(float("-inf"))
      for q_base in range(0, workspace.padded_q, qk_rows_per_chunk):
        qk_rows = min(qk_rows_per_chunk, workspace.padded_q - q_base)
        # A last chunk of only 128 rows: a 256 tile pushes offs_m past padded_q
        # (valid_m masks them, but OOB addresses are needless) -- take per-chunk tile.
        qk_block = min(qk_tile, qk_rows)
        stage_grid = (
            triton.cdiv(qk_rows, qk_block) * stage_slots,
            num_head_q,
            batch,
        )
        _bf16_prefill_qk_kernel[stage_grid](
            q, kcache, cu_seqlens_q, block_ids,
            seqlens_kvcache, mask_arg, workspace.scores,
            # COUNT=False never writes the counter pointer; reuse an existing buffer.
            (
                _skip_counter(workspace.scores.device)
                if _QK_SKIP_COUNT
                else workspace.split_sum
            ),
            act_lists if use_list else workspace.split_sum,
            act_counts if use_list else workspace.split_sum,
            split_base, q_base,
            QB_STOKEN=q.stride(0), QB_SHEAD=q.stride(1),
            KB_SPAGE=kcache.stride(0), KB_STOKEN=kcache.stride(1),
            KB_SHEAD=kcache.stride(2), BID_SBATCH=block_ids.stride(0),
            BID_SPAGE=block_ids.stride(1), MASK_SBATCH=mask_strides[0],
            MASK_SHEAD=mask_strides[1], MASK_SQTILE=mask_strides[2],
            MASK_SKVTILE=mask_strides[3], S_SBATCH=workspace.scores.stride(0),
            S_SSPLIT=workspace.scores.stride(1),
            # head-major: dim2 is head, dim3 is q row
            S_SQ=workspace.scores.stride(3), S_SHEAD=workspace.scores.stride(2),
            NUM_HEAD_Q=num_head_q,
            NUM_HEAD_KV=num_head_kv, PAGE_SIZE=page_size,
            NUM_MASK_KV_TILES=0 if block_mask is None else int(block_mask.shape[3]),
            HAS_BLOCK_MASK=block_mask is not None, MAX_SPLITS=stage_slots,
            TOKENS_PER_SPLIT=workspace.tokens_per_split,
            QK_BLOCK_M=qk_block,
            # List-driven: no do_skip branch, listed blocks active, store unconditional.
            SKIP=(
                0
                if use_list
                else (
                    (2 if block_mask is not None else 0)
                    if _QK_SKIP_OVERRIDE is None
                    else int(_QK_SKIP_OVERRIDE)
                )
            ),
            COUNT=bool(_QK_SKIP_COUNT),
            LISTED=bool(use_list),
            NUM_Q_TILES=num_q_tiles,
            LIST_SPLITS=workspace.max_splits,
            num_warps=8, num_stages=1,
        )
      for q_base in range(0, workspace.padded_q, rows_per_chunk):
        chunk_rows = min(rows_per_chunk, workspace.padded_q - q_base)
        # pv still chunks by q_tokens, independent of the qk grid above
        pv_grid = (
            triton.cdiv(chunk_rows, workspace.q_tokens) * stage_slots,
            num_head_q,
            batch,
        )
        # Serialise max heads per program (largest divisor of num_head_q <= cap).
        hpp = 1
        for cand in range(_PREFILL_HEADS_PER_PROG_MAX, 1, -1):
            if num_head_q % cand == 0:
                hpp = cand
                break
        if _PREFILL_SOFTMAX_HPP_OVERRIDE is not None:
            hpp = min(int(_PREFILL_SOFTMAX_HPP_OVERRIDE), num_head_q)
        vector_grid = (
            triton.cdiv(chunk_rows, _PREFILL_VEC_ROWS),
            triton.cdiv(num_head_q, hpp),
            batch,
        )
        # Keep the head loop dynamic to bound generated code size as head count grows.
        hpp_unroll = 1
        _bf16_prefill_softmax_splits_kernel[vector_grid](
            workspace.scores, workspace.probabilities, cu_seqlens_q,
            seqlens_kvcache, workspace.split_lse, workspace.split_sum,
            split_base, q_base,
            S_SBATCH=workspace.scores.stride(0),
            S_SSPLIT=workspace.scores.stride(1),
            S_SQ=workspace.scores.stride(3),
            S_SHEAD=workspace.scores.stride(2),
            P_SBATCH=workspace.probabilities.stride(0),
            P_SSPLIT=workspace.probabilities.stride(1),
            P_SQ=workspace.probabilities.stride(3),
            P_SHEAD=workspace.probabilities.stride(2),
            SL_SBATCH=workspace.split_lse.stride(0),
            SL_SSPLIT=workspace.split_lse.stride(1),
            SL_SQ=workspace.split_lse.stride(2),
            SS_SBATCH=workspace.split_sum.stride(0),
            SS_SSPLIT=workspace.split_sum.stride(1),
            SS_SQ=workspace.split_sum.stride(2),
            MAX_SPLITS=stage_slots,
            TOKENS_PER_SPLIT=workspace.tokens_per_split,
            VEC=_prefill_scores_vec(workspace.tokens_per_split),
            EXACT=_prefill_scores_exact(workspace.tokens_per_split),
            UNMASKED=workspace.unmasked and _PREFILL_UNMASK_SM,
            HEADS_PER_PROG=hpp,
            HPP_UNROLL=hpp_unroll,
            num_warps=4, num_stages=1,
        )
        _bf16_prefill_pv_accum_kernel[pv_grid](
            vcache, workspace.probabilities, cu_seqlens_q, block_ids,
            seqlens_kvcache, workspace.split_sum, workspace.split_out, output,
            act_lists if use_list else workspace.split_sum,
            act_counts if use_list else workspace.split_sum,
            split_base, q_base,
            VB_SPAGE=vcache.stride(0), VB_STOKEN=vcache.stride(1),
            VB_SHEAD=vcache.stride(2), BID_SBATCH=block_ids.stride(0),
            BID_SPAGE=block_ids.stride(1), P_SBATCH=workspace.probabilities.stride(0),
            P_SSPLIT=workspace.probabilities.stride(1),
            P_SQ=workspace.probabilities.stride(3), P_SHEAD=workspace.probabilities.stride(2),
            SS_SBATCH=workspace.split_sum.stride(0),
            SS_SSPLIT=workspace.split_sum.stride(1), SS_SQ=workspace.split_sum.stride(2),
            SO_SBATCH=workspace.split_out.stride(0), SO_SSPLIT=workspace.split_out.stride(1),
            SO_SQ=workspace.split_out.stride(2), SO_SHEAD=workspace.split_out.stride(3),
            NUM_HEAD_Q=num_head_q, NUM_HEAD_KV=num_head_kv, PAGE_SIZE=page_size,
            MAX_SPLITS=stage_slots, TOKENS_PER_SPLIT=workspace.tokens_per_split,
            PAGE_TILE=(
                page_size
                if page_size <= _BLOCK_N and _BLOCK_N % page_size == 0
                else 0
            ),
            PAGE_UNROLL=(
                _BLOCK_N // page_size
                if page_size <= _BLOCK_N and _BLOCK_N % page_size == 0
                else 1
            ),
            FUSE_FINALIZE=workspace.max_splits == 1,
            O_STOKEN=output.stride(0),
            O_SHEAD=output.stride(1),
            LISTED=bool(use_list),
            NUM_Q_TILES=num_q_tiles,
            LIST_SPLITS=workspace.max_splits,
            num_warps=8, num_stages=1,
        )
    # Process multiple query rows per finalize program while keeping the
    # split-by-row-by-head-dimension FP32 tile within the scratch-memory budget.
    finalize_rows = 1
    for _cand in (32, 16, 8, 4, 2):
        if triton.next_power_of_2(workspace.max_splits) * _cand * _HEAD_DIM * 4 <= 96 * 1024:
            finalize_rows = _cand
            break
    # With max_splits == 1 pv already writes the normalized result straight to output
    # (FUSE_FINALIZE), so a finalize pass would only copy; this loop stays empty.
    for q_base in range(
        0, 0 if workspace.max_splits == 1 else workspace.padded_q, rows_per_chunk
    ):
        chunk_rows = min(rows_per_chunk, workspace.padded_q - q_base)
        _bf16_prefill_finalize_kernel[
            (triton.cdiv(chunk_rows, finalize_rows), num_head_q, batch)
        ](
        workspace.split_out,
        workspace.split_lse,
        cu_seqlens_q,
        seqlens_kvcache,
        output,
        q_base,
        SO_SBATCH=workspace.split_out.stride(0),
        SO_SSPLIT=workspace.split_out.stride(1),
        SO_SQ=workspace.split_out.stride(2),
        SO_SHEAD=workspace.split_out.stride(3),
        SL_SBATCH=workspace.split_lse.stride(0),
        SL_SSPLIT=workspace.split_lse.stride(1),
        SL_SQ=workspace.split_lse.stride(2),
        O_STOKEN=output.stride(0),
        O_SHEAD=output.stride(1),
        MAX_SPLITS=triton.next_power_of_2(workspace.max_splits),
        TOKENS_PER_SPLIT=workspace.tokens_per_split,
        ROWS=finalize_rows,
        num_warps=4,
        num_stages=1,
    )
    return output
