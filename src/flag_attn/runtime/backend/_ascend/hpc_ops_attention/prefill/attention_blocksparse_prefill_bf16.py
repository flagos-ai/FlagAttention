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


# Override switches for host scheduling / same-process A/B; None = policy above.
# Module globals read at call time; benchmarks set them before capture, so one
# process/input can capture different schedules without cold-start/environment noise.
_PREFILL_SOFTMAX_HPP_OVERRIDE = None
# Benchmark/ablation only: qk block-sparse skip. None = default (real mask with a
# block mask, never skip without one, keeping the old path). 0 = never skip
# (baseline), 1 = always skip (upper bound), 2 = real mask. _QK_SKIP_COUNT writes
# (skips, blocks) per program to counters read by benchmarks only; prod = False.
_QK_SKIP_OVERRIDE = None
_QK_SKIP_COUNT = False
_SKIP_COUNTERS: dict = {}
# Benchmark/ablation only: compaction list path. None = default (used only when the
# caller declares sparsity, i.e. sparsity_bucket 2/3); True/False = force on/off (A/B).
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
    """Split the KV into few, long, and (almost) completely full tiles.

    The pipeline launches qk / softmax / pv_accum once per split, and the
    softmax kernel is per-program overhead bound, so fewer splits is better.
    Measured on q=512/kv=4096 (8 heads, page 64):

      tokens/split   512    1024    2048    4096
      end-to-end    3506    2159    1733    1511us
      softmax       2189    1063     617     417us
      qk+pv          916     748     766     749us

    A tile that is mostly padding is wasted work in every kernel though (at
    q=17/kv=2305 a 2048 tile is only 12% full and costs +13.6%), and powers of
    two are far too coarse for that: 2305 tokens is 2x1152 but 3x768 or 2x2048
    with a nearly empty tail.  So instead of a bucket ladder, pick the fewest
    splits that keep the tile within _PREFILL_TOKENS_PER_SPLIT_MAX and round the
    tile up to a multiple of _TILE_STEP (the qk/pv inner block); the softmax
    kernel takes a power-of-two vector width and masks the overhang, so the tile
    no longer has to be a power of two itself.

    Examples: 1024 -> 1024 (1 split), 1088 -> 1152 (1), 2305 -> 1152 (2),
    4096 -> 4096 (1), 32768 -> 4096 (8).

    The cap was 2048 until the softmax ablation showed its math is only ~14% of
    the kernel and the rest is a fixed ~60-75ns per program: doubling the tile
    halves the program count and measured -7.9% (q=512/kv=4096) and -15.4%
    (q=512/kv=32768) end-to-end.
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
    # Flattened grid (coreDim) caps at 65535: softmax is per q row, tightest term
    # rows * stage_slots * num_head_q * batch, so stage depth is bound by it too.
    # The softmax grid is only rows*heads now, no coreDim bound; qk/pv still use
    # q_tiles*stage*heads, held by rows_per_chunk below.
    stage_capacity = max(1, min(_PREFILL_STAGE_CAP, max_splits))
    q_tokens = _prefill_q_tokens(hq, int(kcache.shape[2]))
    padded_q = triton.cdiv(max_seqlens_q, q_tokens) * q_tokens
    # The softmax "unmasked" condition is computed here: dispatch cannot do a
    # device->host sync (it errors out during NPUGraph capture).
    #
    # Unmasked has no valid_n, so every lane must be a real token: no row padding
    # (q_len % 128 == 0), no split tail (kv_len % tile == 0), no tile overhang.
    # tokens_per_split is a 128 multiple but **not necessarily a power of two**
    # (see _prefill_tokens_per_split), so softmax VEC = next_pow2(tile) > tile and
    # the extra columns read as **next q row** scores, polluting row max and row
    # sum: measured q=128/kv=384 error 1.95e-03 (masked) -> 8.89e-02 (unmasked),
    # q=128/kv=1152 4.88e-04 -> 6.89e-02, and test atol=0.1 exactly covers it.
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
        # head-major: scores/probabilities use (batch, stage, hq, q, tps).
        # Under q-major a (128,128) tile had q row stride hq*tps (32*4096 = 131072
        # elements), a 64MB address span that Triton-Ascend lowered to element-wise
        # access: pv's p load took 69% of the whole kernel (same 1.07GB, only the
        # row stride changed to tps=4096, 18.66ms -> 5.745ms). head-major cuts the
        # span to 1MB; softmax touches one contiguous token row at a time, unaffected.
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
    # The list path first fills scores with -inf; under a dense mask that fill is
    # pure overhead (dense-causal mask +5~13%), so **only an explicit sparse caller
    # declaration** (sparsity_bucket 2/3, active <= 50%) uses it: the official
    # benchmark's 75% sparse mask reports bucket 2. bucket 0/1 (active >= 50%) and
    # **no bucket** keep the old path (no regressions); _QK_LIST_OVERRIDE forces it.
    #
    # Few kv tiles make the list unprofitable: compaction + fill is fixed cost and
    # few blocks skip. Forced both in-process (75% sparse, bucket=2, bit-identical):
    #   kv_tiles=3  -> list slower 7.9%  kv_tiles=16 -> list faster 4.7%
    #   kv_tiles=9  -> list slower 0.7%  kv_tiles=32 -> faster 12.0%
    #   kv_tiles=12 -> list faster 4.1%  kv_tiles>=64 -> faster 11.0%
    # The crossover is between 9 and 12 (another agent independently measured
    # kv=1536/12 tiles: list faster by 3.65%, consistent), so below 12 use dense.
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
    # Max 65535 programs per screen (flattened coreDim cap). Softmax is per q row,
    # so the limit comes from rows * stage_slots * num_head_q * batch; stage depth
    # was already cut in the workspace, and q-row chunking here covers any q_len.
    # Both constraints together: softmax per q row (split-loop rows*heads, per-split
    # rows*stage*heads), qk/pv per q tile (q_tiles*stage*heads). Take the tighter.
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
    # qk's M tile is larger than softmax/pv, so it has its own row chunking: SCORES
    # indexes absolute q rows, so the chunkings are independent -- a stage's qk must
    # finish before softmax/pv read complete scores (same stream order anyway).
    # Block-sparse skip needs a block mask; the rate depends on how many 128-row
    # mask rows the M tile spans: M=256 keeps the whole block if either row is
    # active (measured skippable 54~56%), M=128 decides per 128 rows (74~75%).
    # With a mask at q=512/kv=32768, M=128 is 5.5 points faster end to end; without
    # a mask nothing skips, so larger M=256 wins (exp_qk: qk -8%); pick by mask.
    qk_tile = (
        _PREFILL_QK_BLOCK_M_MASKED if block_mask is not None else _PREFILL_QK_BLOCK_M
    )
    # One q tile costs stage_slots*num_head_q*batch program slots, so the row cap is
    # 65535*qk_tile/(stage*hq*batch). The old form treated rows as tiles and squeezed
    # chunks to 384 rows (masked)/256 (unmasked), so 512 rows launched qk twice with
    # half the programs each, not filling 20 AICs -- unmasked 512/32768 lost 2.2%.
    # Grid cap (flattened) still holds: cdiv(qk_rows,qk_block)*stage*hq*batch<=65535.
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
        # qk no longer writes -inf for masked blocks; this fills them at once (the
        # scores (batch, stage) slice is contiguous, only this pass's stage slots),
        # ordered before this pass's qk and after the previous pass's softmax/pv reads.
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
            # With COUNT=False this pointer is never written, so pass an existing
            # tensor instead of allocating a 4MB counter for the production path.
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
        # The head loop is always a runtime loop: hpp can now equal num_head_q, and
        # full unrolling would bloat the tile body many-fold (see the kernel table).
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
    # finalize moves only ~1.5KB per program; the bottleneck is per-program fixed
    # latency. One program on 8 rows (independent, mutually hiding latency) measured
    # 32q/4kv -20% (short)/-3.5% (long), small shapes -32%. A 3-D tile takes
    # MAX_SPLITS*ROWS*D*4 bytes, so fall back to 1 row for large split counts.
    # Same per-program fixed latency bound; multiple rows per program hide each
    # other (ROWS=8 measured -3.5% long). Rows can go higher: ROWS 8 -> 32 gained
    # another -2.2~-3.0% on long cases in same-process A/B (bit-identical). Cap: UB
    # 192KB vs MAX_SPLITS*ROWS*HEAD_DIM*4 bytes, inverse to splits (rule, not table).
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
