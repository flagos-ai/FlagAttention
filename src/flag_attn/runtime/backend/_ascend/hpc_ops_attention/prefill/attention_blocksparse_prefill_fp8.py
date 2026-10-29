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

"""Ascend Triton implementation of HY3 FP8 block-sparse prefill.

This module owns the block-sparse entry kernel, the run-time switches the tests
flip, and the host API (validation, workspace preparation and the attention
entry point).  The phase kernels it launches live in ``decode``, ``qk``, ``pv``,
``softmax`` and ``splits``, their shared tunables in ``constants`` and the E4M3
rounding helper in ``quant``.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import triton
import triton.language as tl
from ..e4m3 import (
    e4m3_exact_masked,
    e4m3_lut,
)

from .constants import (
    _ARITHMETIC_KV_D_BLOCK,
    _ARITHMETIC_KV_PAGES_PER_PROGRAM,
    _BLOCK_M_JIT,
    _BLOCK_N,
    _BLOCK_N_JIT,
    _FP8_P_SCALE_JIT,
    _HEAD_DIM,
    _HEAD_DIM_JIT,
    _LOGICAL_BLOCK_M,
    _LOGICAL_BLOCK_M_JIT,
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
from .decode import (
    _fp8_prefill_decode_kv_arithmetic_kernel,
    _fp8_prefill_decode_kv_kernel,
    _fp8_prefill_decode_q_arithmetic_kernel,
    _fp8_prefill_decode_q_kernel,
)
from .pv import (
    _fp8_prefill_pv_accum_kernel,
)
from .qk import (
    _fp8_prefill_qk_compact_kernel,
    _fp8_prefill_qk_kernel,
)
from .quant import (
    _e4m3fn_lut,
    _positive_e4m3_round,
)
from .softmax import (
    _fp8_prefill_softmax_splits_kernel,
)
from .splits import (
    _fp8_prefill_finalize_kernel,
)


# Override switches for host scheduling / same-process A/B; None = policy above.
# Module globals read at call time; benchmarks set them before capture, so one
# process/input can capture different schedules without cold-start/environment noise.
_KV_ARITHMETIC_OVERRIDE = None
_KV_PAGES_PER_PROGRAM_OVERRIDE = None
_Q_ARITHMETIC_OVERRIDE = None
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
class AscendFP8PrefillWorkspace:
    q_bf16: torch.Tensor
    k_bf16: torch.Tensor
    v_bf16: torch.Tensor
    scores: torch.Tensor
    probabilities: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    split_sum: torch.Tensor
    fp8_lut: torch.Tensor
    max_splits: int
    padded_q: int
    q_tokens: int
    tokens_per_split: int
    quant_type: int
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


def _prefill_launch_chunks(
    flat_q_tiles: int, num_head_kv: int, max_splits: int
) -> tuple[tuple[int, int], ...]:
    programs_per_q_tile = num_head_kv * max_splits
    chunk_size = max(1, 8192 // programs_per_q_tile)
    return tuple(
        (offset, min(chunk_size, flat_q_tiles - offset))
        for offset in range(0, flat_q_tiles, chunk_size)
    )


@triton.jit
def _fp8_blocksparse_prefill_kernel(
    Q,
    K,
    V,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    CU_SEQLENS_Q,
    BLOCK_IDS,
    KV_LENS,
    BLOCK_MASK,
    FP8_LUT,
    OUT,
    Q_STOKEN: tl.constexpr,
    Q_SHEAD: tl.constexpr,
    K_SPAGE: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    V_SPAGE: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    QS_SBATCH: tl.constexpr,
    QS_SHEAD: tl.constexpr,
    QS_STOKEN: tl.constexpr,
    KS_SPAGE: tl.constexpr,
    KS_SGROUP: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SBYTE: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
    MASK_SBATCH: tl.constexpr,
    MASK_SHEAD: tl.constexpr,
    MASK_SQTILE: tl.constexpr,
    MASK_SKVTILE: tl.constexpr,
    O_STOKEN: tl.constexpr,
    O_SHEAD: tl.constexpr,
    NUM_HEAD_Q: tl.constexpr,
    NUM_HEAD_KV: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_KV_TILES: tl.constexpr,
    NUM_MASK_KV_TILES: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
):
    """Disabled (no launch site)."""
    q_half = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(KV_LENS + batch)
    q_local_start = q_half * _BLOCK_M_JIT
    if q_local_start >= q_len:
        return

    group_size: tl.constexpr = NUM_HEAD_Q // NUM_HEAD_KV
    kv_head = q_head // group_size
    q_start_in_kv = kv_len - q_len
    offs_m = q_local_start + tl.arange(0, _BLOCK_M_JIT)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    q_valid = offs_m < q_len
    q_ptrs = (
        Q
        + (q_begin + offs_m[:, None]) * Q_STOKEN
        + q_head * Q_SHEAD
        + offs_d[None, :]
    )
    q = e4m3_lut(FP8_LUT, q_ptrs, q_valid[:, None])
    q_scale = tl.load(
        Q_SCALE
        + batch * QS_SBATCH
        + q_head * QS_SHEAD
        + offs_m * QS_STOKEN,
        mask=q_valid,
        other=0.0,
    )
    q = q.to(tl.bfloat16)

    m_i = tl.full((_BLOCK_M_JIT,), -float("inf"), tl.float32)
    l_i = tl.zeros((_BLOCK_M_JIT,), tl.float32)
    acc = tl.zeros((_BLOCK_M_JIT, _HEAD_DIM_JIT), tl.float32)
    q_half_end = tl.minimum(q_local_start + _BLOCK_M_JIT, q_len)
    causal_kv_end = q_start_in_kv + q_half_end
    num_causal_kv_tiles = tl.maximum(
        0, (causal_kv_end + _BLOCK_N_JIT - 1) // _BLOCK_N_JIT
    )
    logical_q_tile = q_local_start // _LOGICAL_BLOCK_M_JIT
    offs_n = tl.arange(0, _BLOCK_N_JIT)
    score_scale_const: tl.constexpr = 0.127499612793

    for kv_tile in range(MAX_KV_TILES):
        tile_active = kv_tile < num_causal_kv_tiles
        if HAS_BLOCK_MASK:
            mask_in_range = kv_tile < NUM_MASK_KV_TILES
            selected = tl.load(
                BLOCK_MASK
                + batch * MASK_SBATCH
                + q_head * MASK_SHEAD
                + logical_q_tile * MASK_SQTILE
                + kv_tile * MASK_SKVTILE,
                mask=mask_in_range,
                other=0,
            ) != 0
            selected = selected | (kv_tile == NUM_MASK_KV_TILES)
            tile_active = tile_active & selected
        if tile_active:
            kv_tokens = kv_tile * _BLOCK_N_JIT + offs_n
            kv_valid = kv_tokens < kv_len
            logical_pages = kv_tokens // PAGE_SIZE
            token_in_page = kv_tokens % PAGE_SIZE
            physical_pages = tl.load(
                BLOCK_IDS
                + batch * BID_SBATCH
                + logical_pages * BID_SPAGE,
                mask=kv_valid,
                other=0,
            ).to(tl.int64)
            k_ptrs = (
                K
                + physical_pages[:, None] * K_SPAGE
                + token_in_page[:, None] * K_STOKEN
                + kv_head * K_SHEAD
                + offs_d[None, :]
            )
            k = e4m3_lut(FP8_LUT, k_ptrs, kv_valid[:, None])
            scores = tl.dot(q, tl.trans(k.to(tl.bfloat16)))
            if K_SCALE_PER_TOKEN:
                scale_ptr = (
                    K_SCALE
                    + physical_pages * KS_SPAGE
                    + (token_in_page // 32) * KS_SGROUP
                    + kv_head * KS_SHEAD
                    + (token_in_page % 32) * 4 * KS_SBYTE
                )
                scale_word = tl.zeros((_BLOCK_N_JIT,), tl.int32)
                for byte_id in range(4):
                    byte = tl.load(
                        scale_ptr + byte_id * KS_SBYTE,
                        mask=kv_valid,
                        other=0,
                    ).to(tl.int32) & 255
                    scale_word |= byte << (byte_id * 8)
                k_scale = scale_word.to(tl.uint32).to(
                    tl.float32, bitcast=True
                )
                score_scale = q_scale[:, None] * k_scale[None, :]
            else:
                score_scale = q_scale[:, None] * tl.load(K_SCALE)
            q_positions = q_start_in_kv + offs_m
            score_valid = (
                q_valid[:, None]
                & kv_valid[None, :]
                & (kv_tokens[None, :] <= q_positions[:, None])
            )
            scores = tl.where(
                score_valid,
                scores * score_scale * score_scale_const,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m_i, tile_max)
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            alpha = tl.where(
                m_i == -float("inf"), 0.0, tl.exp2(m_i - safe_max)
            )
            p = tl.where(
                score_valid,
                tl.exp2(scores - safe_max[:, None]),
                0.0,
            )
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc *= alpha[:, None]
            p_fp8 = _positive_e4m3_round(p * _FP8_P_SCALE_JIT)
            v_ptrs = (
                V
                + physical_pages[:, None] * V_SPAGE
                + token_in_page[:, None] * V_STOKEN
                + kv_head * V_SHEAD
                + offs_d[None, :]
            )
            v = e4m3_lut(FP8_LUT, v_ptrs, kv_valid[:, None])
            acc += tl.dot(p_fp8, v.to(tl.bfloat16))
            m_i = new_max

    value_scale = (
        tl.load(V_SCALE + kv_head)
        if K_SCALE_PER_TOKEN
        else tl.load(V_SCALE)
    )
    output = acc * (value_scale / _FP8_P_SCALE_JIT) / l_i[:, None]
    out_ptrs = (
        OUT
        + (q_begin + offs_m[:, None]) * O_STOKEN
        + q_head * O_SHEAD
        + offs_d[None, :]
    )
    tl.store(out_ptrs, output, mask=q_valid[:, None])


def _validate(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type: int,
    block_mask: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> None:
    tensors = (
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
    )
    if q.device.type != "npu":
        raise ValueError("Ascend FP8 prefill requires NPU tensors")
    if any(t.device != q.device for t in tensors[1:]):
        raise ValueError("all inputs must be on the same NPU device")
    if q.dtype not in (torch.uint8, torch.int8):
        raise TypeError("q must contain E4M3FN bits in uint8/int8 storage")
    if q.ndim != 3 or q.shape[2] != _HEAD_DIM or q.stride(2) != 1:
        raise ValueError("q must have shape [total_q,num_head_q,128]")
    for name, cache in (("kcache", kcache), ("vcache", vcache)):
        if cache.dtype not in (torch.uint8, torch.int8):
            raise TypeError(f"{name} must contain E4M3FN bits in uint8/int8 storage")
        if cache.ndim != 4 or cache.shape[1] not in (32, 64):
            raise ValueError(f"{name} must have shape [pages,32|64,num_head_kv,128]")
        if cache.shape[3] != _HEAD_DIM or cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous 128")
    if kcache.shape != vcache.shape:
        raise ValueError("K and V cache shapes must match")
    if q.shape[1] % kcache.shape[2]:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    if qscale.dtype != torch.float32 or qscale.ndim != 3:
        raise TypeError("qscale must be float32 [batch,num_head_q,max_q]")
    batch = int(seqlens_kvcache.numel())
    if qscale.shape[0] != batch or qscale.shape[1] != q.shape[1]:
        raise ValueError("qscale batch/head dimensions do not match Q")
    if qscale.shape[2] < max_seqlens_q:
        raise ValueError("qscale token capacity is smaller than max_seqlens_q")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.shape != (batch + 1,):
        raise ValueError("cu_seqlens_q must be int32 [batch+1]")
    if block_ids.dtype != torch.int32 or block_ids.ndim != 2 or block_ids.shape[0] != batch:
        raise ValueError("block_ids must be int32 [batch,max_pages]")
    if seqlens_kvcache.dtype != torch.int32 or seqlens_kvcache.ndim != 1:
        raise ValueError("seqlens_kvcache must be int32 [batch]")
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 or 1")
    if vscale.dtype != torch.float32:
        raise TypeError("vscale must be float32")
    if quant_type == 1:
        if kscale.dtype != torch.float32 or kscale.numel() < 1 or vscale.numel() < 1:
            raise TypeError("per-tensor mode requires scalar FP32 K/V scales")
    else:
        if kscale.dtype not in (torch.uint8, torch.int8) or kscale.ndim != 4:
            raise TypeError("per-token K scale must use packed uint8/int8 FP32 bytes")
        expected_kscale_shape = (
            kcache.shape[0],
            kcache.shape[1] // 32,
            kcache.shape[2],
            128,
        )
        if tuple(kscale.shape) != expected_kscale_shape or kscale.stride(3) != 1:
            raise ValueError(
                "packed K scale must have shape [pages,page_size//32,num_head_kv,128]"
            )
        if vscale.numel() != kcache.shape[2]:
            raise ValueError("per-head vscale must contain num_head_kv values")
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


def prepare_attention_blocksparse_prefill_fp8_workspace(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type=1,
) -> AscendFP8PrefillWorkspace:
    quant_type = int(getattr(quant_type, "value", quant_type))
    _validate(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type,
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
    return AscendFP8PrefillWorkspace(
        q_bf16=torch.empty(q.shape, dtype=torch.bfloat16, device=q.device),
        k_bf16=torch.empty(kcache.shape, dtype=torch.bfloat16, device=q.device),
        v_bf16=torch.empty(vcache.shape, dtype=torch.bfloat16, device=q.device),
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
        fp8_lut=_e4m3fn_lut(q.device),
        max_splits=max_splits,
        padded_q=padded_q,
        q_tokens=q_tokens,
        tokens_per_split=tokens_per_split,
        quant_type=quant_type,
        unmasked=unmasked,
    )


def attention_with_kvcache_blocksparse_prefill_fp8(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type=1,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    sparsity_bucket: Optional[int] = None,
    workspace: Optional[AscendFP8PrefillWorkspace] = None,
) -> torch.Tensor:
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
    quant_type = int(getattr(quant_type, "value", quant_type))
    _validate(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type,
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
        workspace = prepare_attention_blocksparse_prefill_fp8_workspace(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q,
            quant_type,
        )
    max_tokens = int(block_ids.shape[1]) * page_size
    expected_tokens_per_split = _prefill_tokens_per_split(max_tokens)
    expected_splits = max(1, triton.cdiv(max_tokens, expected_tokens_per_split))
    expected_q_tokens = _prefill_q_tokens(num_head_q, num_head_kv)
    if (
        workspace.quant_type != quant_type
        or workspace.max_splits != expected_splits
        or workspace.tokens_per_split != expected_tokens_per_split
        or workspace.q_tokens != expected_q_tokens
        or workspace.padded_q < max_seqlens_q
        or workspace.q_bf16.shape != q.shape
        or workspace.k_bf16.shape != kcache.shape
    ):
        raise ValueError("workspace does not match FP8 prefill inputs")
    mask_arg = q if block_mask is None else block_mask
    mask_strides = (0, 0, 0, 0) if block_mask is None else block_mask.stride()
    ks_strides = (0, 0, 0, 0) if quant_type == 1 else kscale.stride()

    max_q_tiles = triton.cdiv(max_seqlens_q, workspace.q_tokens)
    decode_d_block = 32
    q_arithmetic = (
        True if _Q_ARITHMETIC_OVERRIDE is None else bool(_Q_ARITHMETIC_OVERRIDE)
    )
    q_kernel = (
        _fp8_prefill_decode_q_arithmetic_kernel
        if q_arithmetic else _fp8_prefill_decode_q_kernel
    )
    q_grid = (max_q_tiles * (_HEAD_DIM // decode_d_block), num_head_q, batch) if q_arithmetic else (max_q_tiles, num_head_q, batch)
    q_kwargs = dict(
        D_BLOCK=decode_d_block,
        D_BLOCKS=_HEAD_DIM // decode_d_block,
        USE_ARITHMETIC=True,
    ) if q_arithmetic else {}
    q_kernel[q_grid](
        q,
        qscale,
        cu_seqlens_q,
        workspace.fp8_lut,
        workspace.q_bf16,
        Q_STOKEN=q.stride(0),
        Q_SHEAD=q.stride(1),
        QS_SBATCH=qscale.stride(0),
        QS_SHEAD=qscale.stride(1),
        QS_STOKEN=qscale.stride(2),
        QB_STOKEN=workspace.q_bf16.stride(0),
        QB_SHEAD=workspace.q_bf16.stride(1),
        Q_TOKENS=workspace.q_tokens,
        num_warps=4,
        num_stages=1,
        **q_kwargs,
    )
    num_kv_pages = int(kcache.shape[0])
    # See top comments for measured arithmetic-vs-LUT decode gain; overrides bench-only.
    kv_arithmetic = (
        True if _KV_ARITHMETIC_OVERRIDE is None else bool(_KV_ARITHMETIC_OVERRIDE)
    )
    kv_d_block = _ARITHMETIC_KV_D_BLOCK
    # On a small grid (not a full wave) halving programs with a 64-wide tile costs
    # more than the row-width gain, so fall back to 32 (ppp=1 = pre-change default).
    if (
        triton.cdiv(num_kv_pages, 1) * (_HEAD_DIM // kv_d_block) * num_head_kv
        < 32
    ):
        kv_d_block = 32
    if _KV_PAGES_PER_PROGRAM_OVERRIDE is not None:
        pages_per_program = _KV_PAGES_PER_PROGRAM_OVERRIDE
    else:
        # Flattened coreDim cap 65535: programs = ceil(pages/ppp) * D_BLOCKS * hkv.
        # Start at ppp=1 (measured fastest) and double only when the grid would exceed.
        pages_per_program = _ARITHMETIC_KV_PAGES_PER_PROGRAM
        while (
            triton.cdiv(
                batch * int(block_ids.shape[1])
                if batch * int(block_ids.shape[1]) < num_kv_pages
                else num_kv_pages,
                pages_per_program,
            )
            * (_HEAD_DIM // kv_d_block)
            * num_head_kv
            > 65535
        ):
            pages_per_program *= 2
    kv_kernel = (
        _fp8_prefill_decode_kv_arithmetic_kernel
        if kv_arithmetic else _fp8_prefill_decode_kv_kernel
    )
    # Page table = list of pages read; walking it decodes only referenced pages.
    # Official inputs allocate 2*requested physical pages and reference half, so the
    # walk halves the grid too (identity for full inputs; repeated/padded tables larger
    # than cache fall back to physical pages to avoid polynomial duplicate decoding).
    table_entries = batch * int(block_ids.shape[1])
    # Table walk only when smaller than cache (strictly less work); table == cache
    # (test block_ids=arange) keeps the physical-page walk, other constexpr branch,
    # **byte-identical** pre-change (else one extra scalar load/program, +2% small).
    page_from_table = table_entries < num_kv_pages
    decode_units = table_entries if page_from_table else num_kv_pages
    kv_grid = (
        triton.cdiv(decode_units, pages_per_program) * (_HEAD_DIM // kv_d_block),
        num_head_kv,
    ) if kv_arithmetic else (
        triton.cdiv(decode_units, pages_per_program), num_head_kv
    )
    table_kwargs = dict(
        BLOCK_IDS=block_ids,
        PAGE_FROM_TABLE=page_from_table,
        TABLE_TOTAL=table_entries,
        TABLE_MAX_PAGES=int(block_ids.shape[1]),
        BID_SBATCH=block_ids.stride(0),
        BID_SPAGE=block_ids.stride(1),
    )
    kv_kwargs = dict(
        NUM_PAGES=num_kv_pages,
        PAGES_PER_PROGRAM=pages_per_program,
        D_BLOCK=kv_d_block,
        D_BLOCKS=_HEAD_DIM // kv_d_block,
        USE_ARITHMETIC=True,
        **table_kwargs,
    ) if kv_arithmetic else dict(NUM_PAGES=num_kv_pages, **table_kwargs)
    kv_kernel[kv_grid](
        kcache,
        vcache,
        kscale,
        vscale,
        workspace.fp8_lut,
        workspace.k_bf16,
        workspace.v_bf16,
        PAGE_SIZE=page_size,
        K_SPAGE=kcache.stride(0),
        K_STOKEN=kcache.stride(1),
        K_SHEAD=kcache.stride(2),
        V_SPAGE=vcache.stride(0),
        V_STOKEN=vcache.stride(1),
        V_SHEAD=vcache.stride(2),
        KS_SPAGE=ks_strides[0],
        KS_SGROUP=ks_strides[1],
        KS_SHEAD=ks_strides[2],
        KS_SBYTE=ks_strides[3],
        KB_SPAGE=workspace.k_bf16.stride(0),
        KB_STOKEN=workspace.k_bf16.stride(1),
        KB_SHEAD=workspace.k_bf16.stride(2),
        VB_SPAGE=workspace.v_bf16.stride(0),
        VB_STOKEN=workspace.v_bf16.stride(1),
        VB_SHEAD=workspace.v_bf16.stride(2),
        K_SCALE_PER_TOKEN=quant_type == 0,
        num_warps=4,
        num_stages=1,
        **kv_kwargs,
    )
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
        _fp8_prefill_qk_compact_kernel[
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
        _fp8_prefill_qk_kernel[stage_grid](
            workspace.q_bf16, workspace.k_bf16, cu_seqlens_q, block_ids,
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
            QB_STOKEN=workspace.q_bf16.stride(0), QB_SHEAD=workspace.q_bf16.stride(1),
            KB_SPAGE=workspace.k_bf16.stride(0), KB_STOKEN=workspace.k_bf16.stride(1),
            KB_SHEAD=workspace.k_bf16.stride(2), BID_SBATCH=block_ids.stride(0),
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
        _fp8_prefill_softmax_splits_kernel[vector_grid](
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
        _fp8_prefill_pv_accum_kernel[pv_grid](
            workspace.v_bf16, workspace.probabilities, cu_seqlens_q, block_ids,
            seqlens_kvcache, workspace.split_sum, workspace.split_out, output,
            act_lists if use_list else workspace.split_sum,
            act_counts if use_list else workspace.split_sum,
            split_base, q_base,
            VB_SPAGE=workspace.v_bf16.stride(0), VB_STOKEN=workspace.v_bf16.stride(1),
            VB_SHEAD=workspace.v_bf16.stride(2), BID_SBATCH=block_ids.stride(0),
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
        if workspace.max_splits * _cand * _HEAD_DIM * 4 <= 96 * 1024:
            finalize_rows = _cand
            break
    # With max_splits == 1 pv already writes the normalized result straight to output
    # (FUSE_FINALIZE), so a finalize pass would only copy; this loop stays empty.
    for q_base in range(
        0, 0 if workspace.max_splits == 1 else workspace.padded_q, rows_per_chunk
    ):
        chunk_rows = min(rows_per_chunk, workspace.padded_q - q_base)
        _fp8_prefill_finalize_kernel[
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
        MAX_SPLITS=workspace.max_splits,
        TOKENS_PER_SPLIT=workspace.tokens_per_split,
        ROWS=finalize_rows,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = [
    "AscendFP8PrefillWorkspace",
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "prepare_attention_blocksparse_prefill_fp8_workspace",
]
