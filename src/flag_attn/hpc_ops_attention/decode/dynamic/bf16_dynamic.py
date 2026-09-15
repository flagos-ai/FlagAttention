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

"""Hopper BF16 dynamic attention decode with compact GPU scheduling."""

from __future__ import annotations
from dataclasses import dataclass, replace
from threading import RLock
from typing import Any, TypeAlias
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ..assign_task import (
    _bf16_compact__assign_prefix_kernel as _dynamic__assign_prefix_kernel,
    _bf16_compact__assign_records_kernel as _dynamic__assign_records_kernel,
)
from .. import (
    DecodeWorkload,
    PureTritonMTP1Workspace,
    USE_TLE,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp1_workspace,
    tle,
)

_FINAL_KERNEL_LOCK = RLock()


_bf16_entry__OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}

# Production dispatch keys.  These are input-distribution features, not
# benchmark labels; OFFICIAL_CASES remains only as a public benchmark fixture.
_F_UNIFORM_512 = DecodeWorkload.from_lengths((512,) * 64)
_F_UNIFORM_4096 = DecodeWorkload.from_lengths((4096,) * 64)
_F_MIX_128_4096 = DecodeWorkload.from_lengths((128,) * 32 + (4096,) * 32)
_F_ONE_16K_MANY_64 = DecodeWorkload.from_lengths((64,) * 15 + (16 * 1024,))
_F_ONE_64K_7_SHORT = DecodeWorkload.from_lengths((64 * 1024,) + (4096,) * 7)
_F_ONE_64K_15_SHORT = DecodeWorkload.from_lengths((64 * 1024,) + (4096,) * 15)
_F_ONE_64K_31_SHORT = DecodeWorkload.from_lengths((64 * 1024,) + (4096,) * 31)
_F_ONE_128K_31_SHORT = DecodeWorkload.from_lengths((128 * 1024,) + (4096,) * 31)
_F_TWO_32K_30_SHORT = DecodeWorkload.from_lengths((32 * 1024,) * 2 + (4096,) * 30)
_KNOWN_FEATURES = frozenset(
    {
        _F_UNIFORM_512,
        _F_UNIFORM_4096,
        _F_MIX_128_4096,
        _F_ONE_16K_MANY_64,
        _F_ONE_64K_7_SHORT,
        _F_ONE_64K_15_SHORT,
        _F_ONE_64K_31_SHORT,
        _F_ONE_128K_31_SHORT,
        _F_TWO_32K_30_SHORT,
    }
)
_FINAL_CT_BY_FEATURE = {
    1: {
        _F_UNIFORM_512.signature: {"NHD": (1, 1024), "HND": (1, 1024)},
        _F_UNIFORM_4096.signature: {"NHD": (1, 1024), "HND": (1, 1024)},
        _F_MIX_128_4096.signature: {"NHD": (4, 1024), "HND": (8, 1024)},
        _F_ONE_16K_MANY_64.signature: {"NHD": (8, 256), "HND": (8, 128)},
        _F_ONE_64K_7_SHORT.signature: {"NHD": (8, 512), "HND": (8, 512)},
        _F_ONE_64K_15_SHORT.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
        _F_ONE_64K_31_SHORT.signature: {"NHD": (8, 512), "HND": (8, 512)},
        _F_ONE_128K_31_SHORT.signature: {"NHD": (1, 2048), "HND": (1, 2048)},
        _F_TWO_32K_30_SHORT.signature: {"NHD": (1, 2048), "HND": (1, 2048)},
    },
    2: {
        _F_UNIFORM_512.signature: {"NHD": (1, 1024), "HND": (1, 1024)},
        _F_UNIFORM_4096.signature: {"NHD": (1, 1024), "HND": (1, 1024)},
        _F_MIX_128_4096.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
        _F_ONE_16K_MANY_64.signature: {"NHD": (8, 128), "HND": (8, 128)},
        _F_ONE_64K_7_SHORT.signature: {"NHD": (8, 512), "HND": (8, 512)},
        _F_ONE_64K_15_SHORT.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
        _F_ONE_64K_31_SHORT.signature: {"NHD": (8, 512), "HND": (8, 512)},
        _F_ONE_128K_31_SHORT.signature: {"NHD": (1, 2048), "HND": (1, 2048)},
        _F_TWO_32K_30_SHORT.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
    },
    3: {
        _F_UNIFORM_512.signature: {"NHD": (1, 1024), "HND": (1, 1024)},
        _F_UNIFORM_4096.signature: {"NHD": (1, 1024), "HND": (1, 1024)},
        _F_MIX_128_4096.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
        _F_ONE_16K_MANY_64.signature: {"NHD": (8, 256), "HND": (8, 128)},
        _F_ONE_64K_7_SHORT.signature: {"NHD": (8, 512), "HND": (8, 512)},
        _F_ONE_64K_15_SHORT.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
        _F_ONE_64K_31_SHORT.signature: {"NHD": (8, 512), "HND": (8, 512)},
        _F_ONE_128K_31_SHORT.signature: {"NHD": (1, 2048), "HND": (1, 2048)},
        _F_TWO_32K_30_SHORT.signature: {"NHD": (4, 1024), "HND": (4, 1024)},
    },
}
_FULLTAIL_FEATURES = {
    1: frozenset({_F_UNIFORM_4096, _F_MIX_128_4096, _F_ONE_64K_15_SHORT, _F_ONE_64K_31_SHORT, _F_ONE_128K_31_SHORT}),
    2: _KNOWN_FEATURES - {_F_UNIFORM_512, _F_ONE_16K_MANY_64},
    3: _KNOWN_FEATURES - {_F_UNIFORM_512},
}
_DEFERRED_FEATURES = {
    1: frozenset(
        {
            _F_MIX_128_4096,
            _F_ONE_16K_MANY_64,
            _F_ONE_64K_7_SHORT,
            _F_ONE_64K_15_SHORT,
            _F_ONE_64K_31_SHORT,
            _F_ONE_128K_31_SHORT,
        }
    ),
    2: frozenset(
        {
            _F_MIX_128_4096,
            _F_ONE_64K_7_SHORT,
            _F_ONE_64K_15_SHORT,
            _F_ONE_64K_31_SHORT,
            _F_ONE_128K_31_SHORT,
            _F_TWO_32K_30_SHORT,
        }
    ),
    3: frozenset(
        {
            _F_MIX_128_4096,
            _F_ONE_64K_7_SHORT,
            _F_ONE_64K_15_SHORT,
            _F_ONE_64K_31_SHORT,
            _F_ONE_128K_31_SHORT,
            _F_TWO_32K_30_SHORT,
        }
    ),
}
_PIPE_FEATURES = _KNOWN_FEATURES - {
    _F_UNIFORM_512,
    _F_UNIFORM_4096,
    _F_ONE_16K_MANY_64,
}
_bf16_entry___CLUSTER_MESHES = (
    {size: tle.device_mesh({"block_cluster": [("cluster_x", size)]}) for size in (2, 4, 8)}
    if USE_TLE
    else {2: None, 4: None, 8: None}
)

OFFICIAL_CASES = _bf16_entry__OFFICIAL_CASES

_dynamic__UNIFORM512_TILE_N = tl.constexpr(64)
_dynamic__UNIFORM512_STATIC_PAGES = tl.constexpr(8)
_dynamic__UNIFORM512_TMA_STAGES = tl.constexpr(2)


@triton.jit
def _dynamic__bf16_decode_uniform512_delayed_v_kernel(
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    OUT,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    WIDE_BASELINE: tl.constexpr,
):
    """Exact BF16 static winner: tail-first QK with delayed current-V wait."""
    task = tl.program_id(0)
    batch = task % B
    hkv = task // B
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc(
        [_dynamic__UNIFORM512_TILE_N, Q_ROWS],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_raw = tle.gpu.alloc(
        [_dynamic__UNIFORM512_TMA_STAGES, _dynamic__UNIFORM512_TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_raw = tle.gpu.alloc(
        [_dynamic__UNIFORM512_TMA_STAGES, _dynamic__UNIFORM512_TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_dynamic__UNIFORM512_TMA_STAGES,
        arrive_count=1,
        expect_bytes=_dynamic__UNIFORM512_TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_dynamic__UNIFORM512_TMA_STAGES,
        arrive_count=1,
        expect_bytes=_dynamic__UNIFORM512_TILE_N * D * 2,
    )
    block_ids_smem = tle.gpu.alloc(
        [16],
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _dynamic__UNIFORM512_TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_dynamic__UNIFORM512_TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_dynamic__UNIFORM512_TILE_N, Q_ROWS))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, 16)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + bid_offs,
        mask=bid_offs < 8,
        other=0,
    )
    tl.store(
        tle.gpu.local_ptr(block_ids_smem, (bid_offs,)),
        bids,
        mask=bid_offs < 8,
    )
    tl.debug_barrier()
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    if WIDE_BASELINE:
        acc = tl.zeros((Q_ROWS, D), tl.float32)
    else:
        acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    first_phys = tl.load(tle.gpu.local_ptr(block_ids_smem, (7,)))
    tle.gpu.copy(
        K_DESC,
        k_raw.slot(0),
        [1, 1, _dynamic__UNIFORM512_TILE_N, D],
        [first_phys, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_raw.slot(0),
        [1, 1, _dynamic__UNIFORM512_TILE_N, D],
        [first_phys, hkv, 0, 0],
        barrier=v_full[0],
    )
    page_iter = 0
    while page_iter < _dynamic__UNIFORM512_STATIC_PAGES:
        slot = page_iter % _dynamic__UNIFORM512_TMA_STAGES
        phase = page_iter // _dynamic__UNIFORM512_TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        next_iter = page_iter + 1
        if next_iter < _dynamic__UNIFORM512_STATIC_PAGES:
            next_slot = next_iter % _dynamic__UNIFORM512_TMA_STAGES
            next_phys = tl.load(tle.gpu.local_ptr(block_ids_smem, (next_iter - 1,)))
            tle.gpu.copy(
                K_DESC,
                k_raw.slot(next_slot),
                [1, 1, _dynamic__UNIFORM512_TILE_N, D],
                [next_phys, hkv, 0, 0],
                barrier=k_full[next_slot],
            )
            tle.gpu.copy(
                V_DESC,
                v_raw.slot(next_slot),
                [1, 1, _dynamic__UNIFORM512_TILE_N, D],
                [next_phys, hkv, 0, 0],
                barrier=v_full[next_slot],
            )
        k_page = k_raw.slot(slot)
        if WIDE_BASELINE:
            scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
            tail_limit = _dynamic__UNIFORM512_TILE_N - NUM_SEQ_Q + seq_m
            score_mask = valid_row[:, None] & ((page_iter != 0) | (offs_n[None, :] <= tail_limit[:, None]))
            scores = tl.where(score_mask, scores, -float("inf"))
            page_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, page_max)
            safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
            alpha = tl.exp2(m_i - safe_new)
            p = tl.where(score_mask, tl.exp2(scores - safe_new[:, None]), 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)
            wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _dynamic__UNIFORM512_TILE_N))
            wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _dynamic__UNIFORM512_TILE_N))
            tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        else:
            scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
            tail_limit = _dynamic__UNIFORM512_TILE_N - NUM_SEQ_Q + seq_m
            score_mask = valid_row[None, :] & ((page_iter != 0) | (offs_n[:, None] <= tail_limit[None, :]))
            scores = tl.where(score_mask, scores, -float("inf"))
            page_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, page_max)
            safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
            alpha = tl.exp2(m_i - safe_new)
            p = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            tl.store(p_ptr, p)
        tle.gpu.barrier_wait(v_full[slot], phaseIdx=phase)
        v_page = v_raw.slot(slot)
        if WIDE_BASELINE:
            pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
            pv = tle.gpu.wgmma_wait(0, pv)
            acc = acc * alpha[:, None] + pv
        else:
            pv = tle.gpu.wgmma(v_page, p_smem, trans_a=True, out_dtype=tl.float32)
            pv = tle.gpu.wgmma_wait(0, pv)
            acc = acc * alpha[None, :] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    if WIDE_BASELINE:
        result = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    else:
        acc = tl.where(has_value[None, :], acc / l_i[None, :], 0.0)
        result = tl.trans(acc)
    tl.store(
        OUT + batch * O_SB + seq_m[:, None] * O_SM + hq[:, None] * O_SH + offs_d[None, :],
        result,
        mask=valid_row[:, None],
    )


@triton.jit
def _base__bf16_decode_finalize_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    KV_LENS,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PDL_WAIT: tl.constexpr = False,
):
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, MAX_GROUPS)
    total_len = tl.load(KV_LENS + batch)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    groups = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    needs_finalize = groups > 1
    valid_g = (offs_g < groups) & needs_finalize
    lse = tl.load(
        SPLIT_LSE + batch * SL_SB + offs_g[:, None] * SL_SG + offs_m[None, :] * SL_SM + hq * SL_SH,
        mask=valid_g[:, None] & valid_m[None, :],
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=0)
    safe = tl.where(max_lse != -float("inf"), max_lse, 0.0)
    weights = tl.where(valid_g[:, None], tl.exp2(lse - safe[None, :]), 0.0)
    denom = tl.sum(weights, axis=0)
    acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
    for group in tl.static_range(0, MAX_GROUPS):
        group_valid = group < groups
        group_lse = tl.load(
            SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + hq * SL_SH,
            mask=group_valid & valid_m,
            other=-float("inf"),
        )
        group_weight = tl.where(group_valid & valid_m, tl.exp2(group_lse - safe), 0.0)
        partial = tl.load(
            SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + hq * SO_SH + offs_d[None, :],
            mask=group_valid & valid_m[:, None],
            other=0.0,
        )
        acc += partial * group_weight[:, None]
    acc /= tl.where(denom[:, None] > 0.0, denom[:, None], 1.0)
    tl.store(
        OUT + batch * O_SB + offs_m[:, None] * O_SM + hq * O_SH + offs_d[None, :],
        acc,
        mask=needs_finalize & valid_m[:, None],
    )


@triton.jit
def _finalizer__bf16_decode_finalize_multihead_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    KV_LENS,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    D: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PDL_WAIT: tl.constexpr = False,
):
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    batch = tl.program_id(0)
    head_pass = tl.program_id(1)
    offs_h = head_pass * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, MAX_GROUPS)
    valid_h = offs_h < H_Q
    valid_m = offs_m < NUM_SEQ_Q
    total_len = tl.load(KV_LENS + batch)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    groups = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    needs_finalize = groups > 1
    valid_g = (offs_g < groups) & needs_finalize
    lse = tl.load(
        SPLIT_LSE
        + batch * SL_SB
        + offs_g[:, None, None] * SL_SG
        + offs_m[None, None, :] * SL_SM
        + offs_h[None, :, None] * SL_SH,
        mask=valid_g[:, None, None] & valid_h[None, :, None] & valid_m[None, None, :],
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=0)
    safe_lse = tl.where(max_lse != -float("inf"), max_lse, 0.0)
    weights = tl.where(valid_g[:, None, None], tl.exp2(lse - safe_lse[None, :, :]), 0.0)
    denominator = tl.sum(weights, axis=0)
    acc = tl.zeros((HEADS_PER_PROGRAM, NUM_SEQ_Q_PAD, D), tl.float32)
    for group in tl.static_range(0, MAX_GROUPS):
        group_valid = group < groups
        group_lse = tl.load(
            SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m[None, :] * SL_SM + offs_h[:, None] * SL_SH,
            mask=group_valid & valid_h[:, None] & valid_m[None, :],
            other=-float("inf"),
        )
        group_weight = tl.where(group_valid & valid_h[:, None] & valid_m[None, :], tl.exp2(group_lse - safe_lse), 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + group * SO_SG
            + offs_m[None, :, None] * SO_SM
            + offs_h[:, None, None] * SO_SH
            + offs_d[None, None, :],
            mask=group_valid & valid_h[:, None, None] & valid_m[None, :, None],
            other=0.0,
        )
        acc += partial * group_weight[:, :, None]
    safe_denom = tl.where(denominator > 0.0, denominator, 1.0)
    acc /= safe_denom[:, :, None]
    tl.store(
        OUT + batch * O_SB + offs_m[None, :, None] * O_SM + offs_h[:, None, None] * O_SH + offs_d[None, None, :],
        acc,
        mask=needs_finalize & valid_h[:, None, None] & valid_m[None, :, None],
    )


_dynamic__TASK_STRIDE = 8
_dynamic__TASK_STRIDE_JIT = tl.constexpr(_dynamic__TASK_STRIDE)
HEAD_DIM = 128
BLOCK_SIZE = 64

# Every finalized BF16 decode implementation uses the official 64-token KV tile.
# These names are retained because the clean combined module preserves each
# kernel's generated namespace.  The original split sources defined the
# aliases independently; define them together here for stock Triton JIT.
_base__TILE_N = tl.constexpr(64)
_cluster_deferred__TILE_N = tl.constexpr(64)
_cluster_fulltail__TILE_N = tl.constexpr(64)
_cluster_pipeline__TILE_N = tl.constexpr(64)
_cluster_pipeline_deferred__TILE_N = tl.constexpr(64)
_cluster_pipeline__TMA_STAGES = tl.constexpr(2)
_cluster_pipeline_deferred__TMA_STAGES = tl.constexpr(2)


@triton.jit
def _dynamic__cooperative_group_finalize(
    SPLIT_OUT,
    SPLIT_LSE,
    COMPLETION,
    OUT,
    WINNER_SMEM,
    mesh: tl.constexpr,
    rank,
    batch,
    hkv,
    group,
    group_count,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Elect one cluster and perform the cross-group reduction in-kernel.

    This retains both handoff modes used by the final FP8 dynamic kernels.
    Policy-selected topologies use a deterministic final group; the others
    use atomic last-arrival election followed by DSM winner-bit broadcast.
    """
    if group_count > 1:
        counter = COMPLETION + hkv * B + batch
        rank0_is_last = tl.zeros((), tl.int32)
        if rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_ELECTION:
                deterministic_owner = group == group_count - 1
                if deterministic_owner:
                    ready = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(counter, 1, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            winner = rank0_is_last.to(tl.float32)
            tl.store(tle.gpu.local_ptr(WINNER_SMEM, (0,)), winner)
            for peer in tl.static_range(1, CLUSTER_SIZE):
                peer_flag = tle.remote(WINNER_SMEM, peer, scope=mesh)
                tl.store(tle.gpu.local_ptr(peer_flag, (0,)), winner)
        tle.distributed_barrier(mesh)
        is_last = tl.load(tle.gpu.local_ptr(WINNER_SMEM, (0,))) != 0.0
        if is_last:
            tl.atomic_add(counter, 0, sem="acq_rel", scope="gpu")
            offs_g = tl.arange(0, MAX_GROUPS)
            offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
            offs_d = tl.arange(0, D)
            valid_m = offs_m < NUM_SEQ_Q
            for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
                owned_h = rank + head_pass * CLUSTER_SIZE
                hq = hkv * HEADS_PER_GROUP + owned_h
                valid_h = (owned_h < HEADS_PER_GROUP) & (hq < H_Q)
                valid_g = offs_g < group_count
                lse = tl.load(
                    SPLIT_LSE + batch * SL_SB + offs_g[:, None] * SL_SG + offs_m[None, :] * SL_SM + hq * SL_SH,
                    mask=valid_g[:, None] & valid_m[None, :] & valid_h,
                    other=-float("inf"),
                )
                max_lse = tl.max(lse, axis=0)
                safe = tl.where(max_lse != -float("inf"), max_lse, 0.0)
                weights = tl.where(valid_g[:, None], tl.exp2(lse - safe[None, :]), 0.0)
                denom = tl.sum(weights, axis=0)
                acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
                for finalize_group in tl.static_range(0, MAX_GROUPS):
                    group_valid = (finalize_group < group_count) & valid_h
                    partial = tl.load(
                        SPLIT_OUT
                        + batch * SO_SB
                        + finalize_group * SO_SG
                        + offs_m[:, None] * SO_SM
                        + hq * SO_SH
                        + offs_d[None, :],
                        mask=group_valid & valid_m[:, None],
                        other=0.0,
                    )
                    # Stock Triton does not support indexing a tensor with a
                    # tl.static_range constexpr (weights[group, :]).  Reload
                    # this group's tiny LSE row and reconstruct the same
                    # normalized numerator without changing the reduction.
                    group_lse = tl.load(
                        SPLIT_LSE + batch * SL_SB + finalize_group * SL_SG + offs_m * SL_SM + hq * SL_SH,
                        mask=group_valid & valid_m,
                        other=-float("inf"),
                    )
                    group_weight = tl.where(
                        group_valid & valid_m,
                        tl.exp2(group_lse - safe),
                        0.0,
                    )
                    acc += partial * group_weight[:, None]
                acc /= tl.where(denom[:, None] > 0.0, denom[:, None], 1.0)
                tl.store(
                    OUT + batch * O_SB + offs_m[:, None] * O_SM + hq * O_SH + offs_d[None, :],
                    acc,
                    mask=valid_h & valid_m[:, None],
                )
        tle.distributed_barrier(mesh)
        if (rank == 0) & is_last:
            tl.debug_barrier()
            tl.atomic_xchg(counter, 0, sem="release", scope="gpu")


@triton.jit
def _dynamic__bf16_decode_cluster_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    COOPERATIVE_FINALIZE: tl.constexpr,
):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = True if REDUCTION_ONLY else chunk_len > 0
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _base__TILE_N
    else:
        num_pages = (chunk_len + _base__TILE_N - 1) // _base__TILE_N
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_base__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([_base__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([_base__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_base__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_base__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc(
        [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    partial_lse = tle.gpu.alloc([Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_lse = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    if PREFETCH_BLOCK_IDS:
        chunk_block_ids = tle.gpu.alloc(
            [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _base__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = has_work & (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(q_ptr, q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        if ALIGNED_FULL_CHUNK:
            num_pages = CHUNK_TOKENS // BLOCK_SIZE
        else:
            num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        bids = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
        )
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
        tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    while page_iter < num_pages:
        page_index = page_iter
        start = page_index * _base__TILE_N
        if PREFETCH_BLOCK_IDS:
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        else:
            block_no = (chunk_start + start) // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _base__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _base__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_page = k_raw
        v_page = v_raw
        global_n = chunk_start + start + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_base__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float("inf")
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _base__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _base__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    acc_rows = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), acc_rows)
    tl.store(tle.gpu.local_ptr(partial_lse, (offs_r,)), lse)
    tle.distributed_barrier(mesh)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
        owned_h = rank + head_pass * CLUSTER_SIZE
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_head = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
        valid_owned = valid_head & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        combined = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_lse = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_lse_md = tle.remote(partial_lse, peer, scope=mesh)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D)), out_cols)),
                mask=valid_owned[:, None],
                other=0.0,
            ).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            new_lse = tl.maximum(combined_lse, peer_lse)
            valid_merge = new_lse != -float("inf")
            safe = tl.where(valid_merge, new_lse, 0.0)
            w0 = tl.where(combined_lse != -float("inf"), tl.exp2(combined_lse - safe), 0.0)
            w1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe), 0.0)
            denom = w0 + w1
            combined = tl.where(
                valid_merge[:, None],
                (combined * w0[:, None] + peer_acc * w1[:, None]) / tl.where(denom[:, None] > 0.0, denom[:, None], 1.0),
                0.0,
            )
            combined_lse = tl.where(valid_merge, tl.log2(tl.where(denom > 0.0, denom, 1.0)) + safe, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()
    if not COOPERATIVE_FINALIZE:
        tl.extra.cuda.gdc_launch_dependents()
    if COOPERATIVE_FINALIZE:
        tle.distributed_barrier(mesh)  # publish group partials before election
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            CLUSTER_SIZE,
            MAX_GROUPS,
            DETERMINISTIC_TAIL_ELECTION,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


@triton.jit
def _dynamic__bf16_decode_cluster_deferred_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    COOPERATIVE_FINALIZE: tl.constexpr,
):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = True if REDUCTION_ONLY else chunk_len > 0
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _cluster_deferred__TILE_N
    else:
        num_pages = (chunk_len + _cluster_deferred__TILE_N - 1) // _cluster_deferred__TILE_N
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_cluster_deferred__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([_cluster_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([_cluster_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_cluster_deferred__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_cluster_deferred__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc(
        [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    partial_stats = tle.gpu.alloc(
        [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_acc = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_lse = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    if PREFETCH_BLOCK_IDS:
        chunk_block_ids = tle.gpu.alloc(
            [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _cluster_deferred__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = has_work & (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(q_ptr, q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        if ALIGNED_FULL_CHUNK:
            num_pages = CHUNK_TOKENS // BLOCK_SIZE
        else:
            num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        bids = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
        )
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
        tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    while page_iter < num_pages:
        page_index = page_iter
        start = page_index * _cluster_deferred__TILE_N
        if PREFETCH_BLOCK_IDS:
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        else:
            block_no = (chunk_start + start) // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _cluster_deferred__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _cluster_deferred__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_page = k_raw
        v_page = v_raw
        global_n = chunk_start + start + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_cluster_deferred__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float("inf")
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _cluster_deferred__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _cluster_deferred__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    valid_local = l_i > 0.0
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc, 0.0))
    tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
    tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
    tle.distributed_barrier(mesh)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
        owned_h = rank + head_pass * CLUSTER_SIZE
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_head = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
        valid_owned = valid_head & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        source_rows_2d = tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D))
        combined_acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0
            ).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            valid_merge = new_m != -float("inf")
            safe_m = tl.where(valid_merge, new_m, 0.0)
            old_weight = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe_m), 0.0)
            peer_weight = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe_m), 0.0)
            combined_acc = combined_acc * old_weight[:, None] + peer_acc * peer_weight[:, None]
            combined_l = combined_l * old_weight + peer_l * peer_weight
            combined_m = new_m
        safe_denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / safe_denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(safe_denom) + combined_m, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()
    if not COOPERATIVE_FINALIZE:
        tl.extra.cuda.gdc_launch_dependents()
    if COOPERATIVE_FINALIZE:
        tle.distributed_barrier(mesh)  # publish group partials before election
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            CLUSTER_SIZE,
            MAX_GROUPS,
            DETERMINISTIC_TAIL_ELECTION,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


@triton.jit
def _dynamic__bf16_decode_cluster_fulltail_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    COOPERATIVE_FINALIZE: tl.constexpr,
):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = True if REDUCTION_ONLY else chunk_len > 0
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _cluster_fulltail__TILE_N
    else:
        num_pages = (chunk_len + _cluster_fulltail__TILE_N - 1) // _cluster_fulltail__TILE_N
    is_last_chunk = has_work & (chunk_start + chunk_len >= total_len)
    causal_start_page = (total_len - NUM_SEQ_Q - chunk_start) // _cluster_fulltail__TILE_N
    tail_start_page = tl.where(is_last_chunk, tl.maximum(causal_start_page, 0), num_pages)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_cluster_fulltail__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([_cluster_fulltail__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([_cluster_fulltail__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_cluster_fulltail__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_cluster_fulltail__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc(
        [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    partial_lse = tle.gpu.alloc([Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_lse = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    if PREFETCH_BLOCK_IDS:
        chunk_block_ids = tle.gpu.alloc(
            [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _cluster_fulltail__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = has_work & (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(q_ptr, q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        if ALIGNED_FULL_CHUNK:
            num_pages = CHUNK_TOKENS // BLOCK_SIZE
        else:
            num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        bids = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
        )
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
        tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    while page_iter < tail_start_page:
        page_index = page_iter
        start = page_index * _cluster_fulltail__TILE_N
        if PREFETCH_BLOCK_IDS:
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        else:
            block_no = (chunk_start + start) // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_page = k_raw
        v_page = v_raw
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_fulltail__TILE_N))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float("inf")
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _cluster_fulltail__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _cluster_fulltail__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    while page_iter < num_pages:
        page_index = page_iter
        start = page_index * _cluster_fulltail__TILE_N
        if PREFETCH_BLOCK_IDS:
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        else:
            block_no = (chunk_start + start) // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_page = k_raw
        v_page = v_raw
        global_n = chunk_start + start + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_cluster_fulltail__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float("inf")
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _cluster_fulltail__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _cluster_fulltail__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    acc_rows = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), acc_rows)
    tl.store(tle.gpu.local_ptr(partial_lse, (offs_r,)), lse)
    tle.distributed_barrier(mesh)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
        owned_h = rank + head_pass * CLUSTER_SIZE
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_head = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
        valid_owned = valid_head & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        combined = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_lse = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_lse_md = tle.remote(partial_lse, peer, scope=mesh)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D)), out_cols)),
                mask=valid_owned[:, None],
                other=0.0,
            ).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            new_lse = tl.maximum(combined_lse, peer_lse)
            valid_merge = new_lse != -float("inf")
            safe = tl.where(valid_merge, new_lse, 0.0)
            w0 = tl.where(combined_lse != -float("inf"), tl.exp2(combined_lse - safe), 0.0)
            w1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe), 0.0)
            denom = w0 + w1
            combined = tl.where(
                valid_merge[:, None],
                (combined * w0[:, None] + peer_acc * w1[:, None]) / tl.where(denom[:, None] > 0.0, denom[:, None], 1.0),
                0.0,
            )
            combined_lse = tl.where(valid_merge, tl.log2(tl.where(denom > 0.0, denom, 1.0)) + safe, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()
    if not COOPERATIVE_FINALIZE:
        tl.extra.cuda.gdc_launch_dependents()
    if COOPERATIVE_FINALIZE:
        tle.distributed_barrier(mesh)  # publish group partials before election
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            CLUSTER_SIZE,
            MAX_GROUPS,
            DETERMINISTIC_TAIL_ELECTION,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


@triton.jit
def _dynamic__bf16_decode_cluster_pipeline_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    COOPERATIVE_FINALIZE: tl.constexpr,
):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = True if REDUCTION_ONLY else chunk_len > 0
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _cluster_pipeline__TILE_N
    else:
        num_pages = (chunk_len + _cluster_pipeline__TILE_N - 1) // _cluster_pipeline__TILE_N
    pipeline_pages = tl.maximum(num_pages, 1)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_cluster_pipeline__TMA_STAGES, _cluster_pipeline__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_cluster_pipeline__TMA_STAGES, _cluster_pipeline__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc([Q_ROWS, _cluster_pipeline__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_cluster_pipeline__TMA_STAGES, expect_bytes=_cluster_pipeline__TILE_N * D * 2
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_cluster_pipeline__TMA_STAGES, expect_bytes=_cluster_pipeline__TILE_N * D * 2
    )
    partial_acc = tle.gpu.alloc(
        [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    partial_lse = tle.gpu.alloc([Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_lse = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _cluster_pipeline__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = has_work & (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _cluster_pipeline__TILE_N))
    p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _cluster_pipeline__TILE_N))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        safe_start = tl.where(has_work, chunk_start // BLOCK_SIZE, 0)
        bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + safe_start + bid_offs, mask=bid_offs < pipeline_pages, other=0)
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < pipeline_pages)
        tl.debug_barrier()
    pipeline_start = tl.where(has_work, chunk_start, 0)
    if PREFETCH_BLOCK_IDS:
        phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    else:
        phys0 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE)
    tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0])
    tle.gpu.copy(V_DESC, v_smem.slot(0), [1, 1, _cluster_pipeline__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0])
    if pipeline_pages > 1:
        if PREFETCH_BLOCK_IDS:
            phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
        else:
            phys1 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + 1)
        tle.gpu.copy(
            K_DESC, k_smem.slot(1), [1, 1, _cluster_pipeline__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1]
        )
        tle.gpu.copy(
            V_DESC, v_smem.slot(1), [1, 1, _cluster_pipeline__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1]
        )
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    k_page = k_smem.slot(0)
    qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
    qk = tle.gpu.wgmma_wait(0, qk) * scale
    global_n = pipeline_start + offs_n
    if ALIGNED_FULL_CHUNK:
        token_valid = tl.full((_cluster_pipeline__TILE_N,), True, tl.int1)
    else:
        token_valid = global_n < total_len
    query_pos = total_len - NUM_SEQ_Q + seq_m
    mask0 = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
    qk = tl.where(mask0, qk, -float("inf"))
    m_i = tl.max(qk, axis=1)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(mask0, tl.exp2(qk - safe_m[:, None]), 0.0)
    l_i = tl.sum(p0, axis=1)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if pipeline_pages > 2:
        if PREFETCH_BLOCK_IDS:
            phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        else:
            phys2 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + 2)
        tle.gpu.copy(
            K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0]
        )
    page = 1
    while page < pipeline_pages:
        slot = page % _cluster_pipeline__TMA_STAGES
        phase = page // _cluster_pipeline__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_page = k_smem.slot(slot)
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        global_n = pipeline_start + page * _cluster_pipeline__TILE_N + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_cluster_pipeline__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        qk = tl.where(score_mask, qk, -float("inf"))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline__TMA_STAGES
        if next_k < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            else:
                next_k_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_k)
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _cluster_pipeline__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _cluster_pipeline__TMA_STAGES
        if next_v < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            else:
                next_v_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_v)
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _cluster_pipeline__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    last_page = pipeline_pages - 1
    last_slot = last_page % _cluster_pipeline__TMA_STAGES
    last_phase = last_page // _cluster_pipeline__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(p_smem, v_last, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    has_value = l_i > 0.0
    acc = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), acc)
    tl.store(tle.gpu.local_ptr(partial_lse, (offs_r,)), lse)
    tle.distributed_barrier(mesh)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
        owned_h = rank + head_pass * CLUSTER_SIZE
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_head = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
        valid_owned = valid_head & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        combined = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_lse = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_lse_md = tle.remote(partial_lse, peer, scope=mesh)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D)), out_cols)),
                mask=valid_owned[:, None],
                other=0.0,
            ).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            new_lse = tl.maximum(combined_lse, peer_lse)
            valid_merge = new_lse != -float("inf")
            safe = tl.where(valid_merge, new_lse, 0.0)
            w0 = tl.where(combined_lse != -float("inf"), tl.exp2(combined_lse - safe), 0.0)
            w1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe), 0.0)
            denom = w0 + w1
            combined = tl.where(
                valid_merge[:, None],
                (combined * w0[:, None] + peer_acc * w1[:, None]) / tl.where(denom[:, None] > 0.0, denom[:, None], 1.0),
                0.0,
            )
            combined_lse = tl.where(valid_merge, tl.log2(tl.where(denom > 0.0, denom, 1.0)) + safe, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()
    if not COOPERATIVE_FINALIZE:
        tl.extra.cuda.gdc_launch_dependents()
    if COOPERATIVE_FINALIZE:
        tle.distributed_barrier(mesh)  # publish group partials before election
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            CLUSTER_SIZE,
            MAX_GROUPS,
            DETERMINISTIC_TAIL_ELECTION,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


@triton.jit
def _dynamic__bf16_decode_cluster_pipeline_deferred_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    COOPERATIVE_FINALIZE: tl.constexpr,
):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = True if REDUCTION_ONLY else chunk_len > 0
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _cluster_pipeline_deferred__TILE_N
    else:
        num_pages = (chunk_len + _cluster_pipeline_deferred__TILE_N - 1) // _cluster_pipeline_deferred__TILE_N
    pipeline_pages = tl.maximum(num_pages, 1)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_cluster_pipeline_deferred__TMA_STAGES, _cluster_pipeline_deferred__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_cluster_pipeline_deferred__TMA_STAGES, _cluster_pipeline_deferred__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [Q_ROWS, _cluster_pipeline_deferred__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_cluster_pipeline_deferred__TMA_STAGES, expect_bytes=_cluster_pipeline_deferred__TILE_N * D * 2
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_cluster_pipeline_deferred__TMA_STAGES, expect_bytes=_cluster_pipeline_deferred__TILE_N * D * 2
    )
    partial_acc = tle.gpu.alloc(
        [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    partial_stats = tle.gpu.alloc(
        [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_acc = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_lse = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _cluster_pipeline_deferred__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = has_work & (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _cluster_pipeline_deferred__TILE_N))
    p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _cluster_pipeline_deferred__TILE_N))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        safe_start = tl.where(has_work, chunk_start // BLOCK_SIZE, 0)
        bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + safe_start + bid_offs, mask=bid_offs < pipeline_pages, other=0)
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < pipeline_pages)
        tl.debug_barrier()
    pipeline_start = tl.where(has_work, chunk_start, 0)
    if PREFETCH_BLOCK_IDS:
        phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    else:
        phys0 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE)
    tle.gpu.copy(
        K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0]
    )
    tle.gpu.copy(
        V_DESC, v_smem.slot(0), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0]
    )
    if pipeline_pages > 1:
        if PREFETCH_BLOCK_IDS:
            phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
        else:
            phys1 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + 1)
        tle.gpu.copy(
            K_DESC, k_smem.slot(1), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1]
        )
        tle.gpu.copy(
            V_DESC, v_smem.slot(1), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1]
        )
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    k_page = k_smem.slot(0)
    qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
    qk = tle.gpu.wgmma_wait(0, qk) * scale
    global_n = pipeline_start + offs_n
    if ALIGNED_FULL_CHUNK:
        token_valid = tl.full((_cluster_pipeline_deferred__TILE_N,), True, tl.int1)
    else:
        token_valid = global_n < total_len
    query_pos = total_len - NUM_SEQ_Q + seq_m
    mask0 = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
    qk = tl.where(mask0, qk, -float("inf"))
    m_i = tl.max(qk, axis=1)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(mask0, tl.exp2(qk - safe_m[:, None]), 0.0)
    l_i = tl.sum(p0, axis=1)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if pipeline_pages > 2:
        if PREFETCH_BLOCK_IDS:
            phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        else:
            phys2 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + 2)
        tle.gpu.copy(
            K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0]
        )
    page = 1
    while page < pipeline_pages:
        slot = page % _cluster_pipeline_deferred__TMA_STAGES
        phase = page // _cluster_pipeline_deferred__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline_deferred__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline_deferred__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_page = k_smem.slot(slot)
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        global_n = pipeline_start + page * _cluster_pipeline_deferred__TILE_N + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_cluster_pipeline_deferred__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        qk = tl.where(score_mask, qk, -float("inf"))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline_deferred__TMA_STAGES
        if next_k < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            else:
                next_k_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_k)
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _cluster_pipeline_deferred__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _cluster_pipeline_deferred__TMA_STAGES
        if next_v < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            else:
                next_v_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_v)
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _cluster_pipeline_deferred__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    last_page = pipeline_pages - 1
    last_slot = last_page % _cluster_pipeline_deferred__TMA_STAGES
    last_phase = last_page // _cluster_pipeline_deferred__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(p_smem, v_last, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    valid_local = l_i > 0.0
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc, 0.0))
    tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
    tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
    tle.distributed_barrier(mesh)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
        owned_h = rank + head_pass * CLUSTER_SIZE
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_head = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
        valid_owned = valid_head & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        source_rows_2d = tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D))
        combined_acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0
            ).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            valid_merge = new_m != -float("inf")
            safe_m = tl.where(valid_merge, new_m, 0.0)
            old_weight = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe_m), 0.0)
            peer_weight = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe_m), 0.0)
            combined_acc = combined_acc * old_weight[:, None] + peer_acc * peer_weight[:, None]
            combined_l = combined_l * old_weight + peer_l * peer_weight
            combined_m = new_m
        safe_denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / safe_denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(safe_denom) + combined_m, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()
    if not COOPERATIVE_FINALIZE:
        tl.extra.cuda.gdc_launch_dependents()
    if COOPERATIVE_FINALIZE:
        tle.distributed_barrier(mesh)  # publish group partials before election
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            CLUSTER_SIZE,
            MAX_GROUPS,
            DETERMINISTIC_TAIL_ELECTION,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


_cluster_pipeline_fulltail_deferred__TILE_N = tl.constexpr(64)
_cluster_pipeline_fulltail_deferred__TMA_STAGES = tl.constexpr(2)


@triton.jit
def _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    COOPERATIVE_FINALIZE: tl.constexpr,
):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = True if REDUCTION_ONLY else chunk_len > 0
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _cluster_pipeline_fulltail_deferred__TILE_N
    else:
        num_pages = (
            chunk_len + _cluster_pipeline_fulltail_deferred__TILE_N - 1
        ) // _cluster_pipeline_fulltail_deferred__TILE_N
    pipeline_pages = tl.maximum(num_pages, 1)
    is_last_chunk = has_work & (chunk_start + chunk_len >= total_len)
    last_page = pipeline_pages - 1
    causal_start_page = (total_len - NUM_SEQ_Q - chunk_start) // _cluster_pipeline_fulltail_deferred__TILE_N
    tail_start_page = tl.where(is_last_chunk, tl.maximum(causal_start_page, 0), last_page)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_cluster_pipeline_fulltail_deferred__TMA_STAGES, _cluster_pipeline_fulltail_deferred__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_cluster_pipeline_fulltail_deferred__TMA_STAGES, _cluster_pipeline_fulltail_deferred__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_cluster_pipeline_fulltail_deferred__TMA_STAGES,
        expect_bytes=_cluster_pipeline_fulltail_deferred__TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_cluster_pipeline_fulltail_deferred__TMA_STAGES,
        expect_bytes=_cluster_pipeline_fulltail_deferred__TILE_N * D * 2,
    )
    partial_acc = tle.gpu.alloc(
        [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    partial_stats = tle.gpu.alloc(
        [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_acc = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    merged_lse = tle.gpu.alloc(
        [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _cluster_pipeline_fulltail_deferred__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = has_work & (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
    p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        safe_start = tl.where(has_work, chunk_start // BLOCK_SIZE, 0)
        bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + safe_start + bid_offs, mask=bid_offs < pipeline_pages, other=0)
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < pipeline_pages)
        tl.debug_barrier()
    pipeline_start = tl.where(has_work, chunk_start, 0)
    if PREFETCH_BLOCK_IDS:
        phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    else:
        phys0 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE)
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(0),
        [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=v_full[0],
    )
    if pipeline_pages > 1:
        if PREFETCH_BLOCK_IDS:
            phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
        else:
            phys1 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + 1)
        tle.gpu.copy(
            K_DESC,
            k_smem.slot(1),
            [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
            [phys1, hkv, 0, 0],
            barrier=k_full[1],
        )
        tle.gpu.copy(
            V_DESC,
            v_smem.slot(1),
            [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
            [phys1, hkv, 0, 0],
            barrier=v_full[1],
        )
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    k_page = k_smem.slot(0)
    qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
    qk = tle.gpu.wgmma_wait(0, qk) * scale
    query_pos = total_len - NUM_SEQ_Q + seq_m
    page0_is_tail = is_last_chunk & (tail_start_page == 0)
    if page0_is_tail:
        global_n = pipeline_start + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_cluster_pipeline_fulltail_deferred__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        mask0 = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
    else:
        mask0 = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
    qk = tl.where(mask0, qk, -float("inf"))
    m_i = tl.max(qk, axis=1)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(mask0, tl.exp2(qk - safe_m[:, None]), 0.0)
    l_i = tl.sum(p0, axis=1)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if pipeline_pages > 2:
        if PREFETCH_BLOCK_IDS:
            phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        else:
            phys2 = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + 2)
        tle.gpu.copy(
            K_DESC,
            k_smem.slot(0),
            [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
            [phys2, hkv, 0, 0],
            barrier=k_full[0],
        )
    page = 1
    while page < tail_start_page:
        slot = page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        phase = page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_page = k_smem.slot(slot)
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
        qk = tl.where(score_mask, qk, -float("inf"))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_k < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            else:
                next_k_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_k)
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_v < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            else:
                next_v_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_v)
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    while page < pipeline_pages:
        slot = page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        phase = page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_page = k_smem.slot(slot)
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        if is_last_chunk:
            global_n = pipeline_start + page * _cluster_pipeline_fulltail_deferred__TILE_N + offs_n
            if ALIGNED_FULL_CHUNK:
                token_valid = tl.full((_cluster_pipeline_fulltail_deferred__TILE_N,), True, tl.int1)
            else:
                token_valid = global_n < total_len
            score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        else:
            score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
        qk = tl.where(score_mask, qk, -float("inf"))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_k < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            else:
                next_k_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_k)
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_v < pipeline_pages:
            if PREFETCH_BLOCK_IDS:
                next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            else:
                next_v_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + pipeline_start // BLOCK_SIZE + next_v)
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    last_slot = last_page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
    last_phase = last_page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(p_smem, v_last, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    valid_local = l_i > 0.0
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc, 0.0))
    tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
    tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
    tle.distributed_barrier(mesh)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 8 // CLUSTER_SIZE):
        owned_h = rank + head_pass * CLUSTER_SIZE
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_head = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
        valid_owned = valid_head & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        source_rows_2d = tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D))
        combined_acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0
            ).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            valid_merge = new_m != -float("inf")
            safe_m = tl.where(valid_merge, new_m, 0.0)
            old_weight = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe_m), 0.0)
            peer_weight = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe_m), 0.0)
            combined_acc = combined_acc * old_weight[:, None] + peer_acc * peer_weight[:, None]
            combined_l = combined_l * old_weight + peer_l * peer_weight
            combined_m = new_m
        safe_denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / safe_denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(safe_denom) + combined_m, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()
    if not COOPERATIVE_FINALIZE:
        tl.extra.cuda.gdc_launch_dependents()
    if COOPERATIVE_FINALIZE:
        tle.distributed_barrier(mesh)  # publish group partials before election
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            CLUSTER_SIZE,
            MAX_GROUPS,
            DETERMINISTIC_TAIL_ELECTION,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


_direct_fulltail__TILE_N = tl.constexpr(64)


@triton.jit
def _dynamic__bf16_decode_direct_fulltail_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
    TMA_STAGES: tl.constexpr = 1,
):
    cta = tl.program_id(0)
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _direct_fulltail__TILE_N
    else:
        num_pages = (chunk_len + _direct_fulltail__TILE_N - 1) // _direct_fulltail__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    causal_start_page = (total_len - NUM_SEQ_Q - chunk_start) // _direct_fulltail__TILE_N
    tail_start_page = tl.where(is_last_chunk, tl.maximum(causal_start_page, 0), num_pages)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_direct_fulltail__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([TMA_STAGES, _direct_fulltail__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([TMA_STAGES, _direct_fulltail__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(
        num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_fulltail__TILE_N * D * 2
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_fulltail__TILE_N * D * 2
    )
    if PREFETCH_BLOCK_IDS:
        chunk_block_ids = tle.gpu.alloc(
            [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _direct_fulltail__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(q_ptr, q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        if ALIGNED_FULL_CHUNK:
            num_pages = CHUNK_TOKENS // BLOCK_SIZE
        else:
            num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        bids = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
        )
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
        tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    if TMA_STAGES == 2:
        first_page = 0
        if PREFETCH_BLOCK_IDS:
            first_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (first_page,)))
        else:
            first_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + first_page)
        tle.gpu.copy(
            K_DESC, k_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [first_phys, hkv, 0, 0], barrier=k_full[0]
        )
        tle.gpu.copy(
            V_DESC, v_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [first_phys, hkv, 0, 0], barrier=v_full[0]
        )
    while page_iter < tail_start_page:
        page_index = page_iter
        start = page_index * _direct_fulltail__TILE_N
        if TMA_STAGES == 1:
            if PREFETCH_BLOCK_IDS:
                phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
            else:
                block_no = (chunk_start + start) // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            tle.gpu.copy(
                K_DESC, k_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0]
            )
            tle.gpu.copy(
                V_DESC, v_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0]
            )
            tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
            tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
            slot = 0
        else:
            slot = page_iter % 2
            phase = page_iter // 2
            tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
            tle.gpu.barrier_wait(v_full[slot], phaseIdx=phase)
            next_iter = page_iter + 1
            if next_iter < num_pages:
                next_slot = next_iter % 2
                next_page = next_iter
                if PREFETCH_BLOCK_IDS:
                    next_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_page,)))
                else:
                    next_block = chunk_start // BLOCK_SIZE + next_page
                    next_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + next_block)
                tle.gpu.copy(
                    K_DESC,
                    k_raw.slot(next_slot),
                    [1, 1, _direct_fulltail__TILE_N, D],
                    [next_phys, hkv, 0, 0],
                    barrier=k_full[next_slot],
                )
                tle.gpu.copy(
                    V_DESC,
                    v_raw.slot(next_slot),
                    [1, 1, _direct_fulltail__TILE_N, D],
                    [next_phys, hkv, 0, 0],
                    barrier=v_full[next_slot],
                )
        k_page = k_raw.slot(slot)
        v_page = v_raw.slot(slot)
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _direct_fulltail__TILE_N))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float("inf")
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _direct_fulltail__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _direct_fulltail__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    while page_iter < num_pages:
        page_index = page_iter
        start = page_index * _direct_fulltail__TILE_N
        if TMA_STAGES == 1:
            if PREFETCH_BLOCK_IDS:
                phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
            else:
                block_no = (chunk_start + start) // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            tle.gpu.copy(
                K_DESC, k_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0]
            )
            tle.gpu.copy(
                V_DESC, v_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0]
            )
            tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
            tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
            slot = 0
        else:
            slot = page_iter % 2
            phase = page_iter // 2
            tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
            tle.gpu.barrier_wait(v_full[slot], phaseIdx=phase)
            next_iter = page_iter + 1
            if next_iter < num_pages:
                next_slot = next_iter % 2
                next_page = next_iter
                if PREFETCH_BLOCK_IDS:
                    next_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_page,)))
                else:
                    next_block = chunk_start // BLOCK_SIZE + next_page
                    next_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + next_block)
                tle.gpu.copy(
                    K_DESC,
                    k_raw.slot(next_slot),
                    [1, 1, _direct_fulltail__TILE_N, D],
                    [next_phys, hkv, 0, 0],
                    barrier=k_full[next_slot],
                )
                tle.gpu.copy(
                    V_DESC,
                    v_raw.slot(next_slot),
                    [1, 1, _direct_fulltail__TILE_N, D],
                    [next_phys, hkv, 0, 0],
                    barrier=v_full[next_slot],
                )
        k_page = k_raw.slot(slot)
        v_page = v_raw.slot(slot)
        global_n = chunk_start + start + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_direct_fulltail__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float("inf")
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _direct_fulltail__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _direct_fulltail__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    acc_rows = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if group_count == 1:
        tl.store(
            OUT + batch * O_SB + seq_m[:, None] * O_SM + hq[:, None] * O_SH + offs_d[None, :],
            acc_rows,
            mask=valid_row[:, None],
        )
    else:
        tl.store(
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :],
            acc_rows,
            mask=valid_row[:, None],
        )
        tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH, lse, mask=valid_row)
    if PDL_NOTIFY:
        tl.extra.cuda.gdc_launch_dependents()


_direct_narrow__TILE_N = tl.constexpr(64)


@triton.jit
def _dynamic__bf16_decode_direct_narrow_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PREFETCH_BLOCK_IDS: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    WIDE_BASELINE: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
    TMA_STAGES: tl.constexpr = 1,
):
    cta = tl.program_id(0)
    task = cta * _dynamic__TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    chunk_len = tl.load(TASK_MAP + task + 3)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    if ALIGNED_FULL_CHUNK:
        num_pages = CHUNK_TOKENS // _direct_narrow__TILE_N
    else:
        num_pages = (chunk_len + _direct_narrow__TILE_N - 1) // _direct_narrow__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_direct_narrow__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([TMA_STAGES, _direct_narrow__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([TMA_STAGES, _direct_narrow__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(
        num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_narrow__TILE_N * D * 2
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_narrow__TILE_N * D * 2
    )
    if PREFETCH_BLOCK_IDS:
        chunk_block_ids = tle.gpu.alloc(
            [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _direct_narrow__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_direct_narrow__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_direct_narrow__TILE_N, Q_ROWS))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(q_ptr, q)
    if PREFETCH_BLOCK_IDS:
        bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
        if ALIGNED_FULL_CHUNK:
            num_pages = CHUNK_TOKENS // BLOCK_SIZE
        else:
            num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        bids = tl.load(
            BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
        )
        tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
        tl.debug_barrier()
    if WIDE_BASELINE:
        acc = tl.zeros((Q_ROWS, D), tl.float32)
    else:
        acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    if TMA_STAGES == 2:
        first_page = tl.where(is_last_chunk, num_pages - 1, 0)
        if PREFETCH_BLOCK_IDS:
            first_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (first_page,)))
        else:
            first_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + first_page)
        tle.gpu.copy(
            K_DESC, k_raw.slot(0), [1, 1, _direct_narrow__TILE_N, D], [first_phys, hkv, 0, 0], barrier=k_full[0]
        )
        tle.gpu.copy(
            V_DESC, v_raw.slot(0), [1, 1, _direct_narrow__TILE_N, D], [first_phys, hkv, 0, 0], barrier=v_full[0]
        )
    while page_iter < num_pages:
        page_index = tl.where(is_last_chunk, tl.where(page_iter == 0, num_pages - 1, page_iter - 1), page_iter)
        start = page_index * _direct_narrow__TILE_N
        if TMA_STAGES == 1:
            if PREFETCH_BLOCK_IDS:
                phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
            else:
                block_no = (chunk_start + start) // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            tle.gpu.copy(K_DESC, k_raw.slot(0), [1, 1, _direct_narrow__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            tle.gpu.copy(V_DESC, v_raw.slot(0), [1, 1, _direct_narrow__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
            tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
            tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
            slot = 0
        else:
            slot = page_iter % 2
            phase = page_iter // 2
            tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
            tle.gpu.barrier_wait(v_full[slot], phaseIdx=phase)
            next_iter = page_iter + 1
            if next_iter < num_pages:
                next_slot = next_iter % 2
                next_page = tl.where(is_last_chunk, tl.where(next_iter == 0, num_pages - 1, next_iter - 1), next_iter)
                if PREFETCH_BLOCK_IDS:
                    next_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_page,)))
                else:
                    next_block = chunk_start // BLOCK_SIZE + next_page
                    next_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + next_block)
                tle.gpu.copy(
                    K_DESC,
                    k_raw.slot(next_slot),
                    [1, 1, _direct_narrow__TILE_N, D],
                    [next_phys, hkv, 0, 0],
                    barrier=k_full[next_slot],
                )
                tle.gpu.copy(
                    V_DESC,
                    v_raw.slot(next_slot),
                    [1, 1, _direct_narrow__TILE_N, D],
                    [next_phys, hkv, 0, 0],
                    barrier=v_full[next_slot],
                )
        k_page = k_raw.slot(slot)
        v_page = v_raw.slot(slot)
        global_n = chunk_start + start + offs_n
        if ALIGNED_FULL_CHUNK:
            token_valid = tl.full((_direct_narrow__TILE_N,), True, tl.int1)
        else:
            token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        if WIDE_BASELINE:
            scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
            score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
            scores = tl.where(score_mask, scores, -float("inf"))
            page_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, page_max)
            active = m_new != -float("inf")
            safe_m = tl.where(active, m_new, 0.0)
            alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
            p = tl.exp2(scores - safe_m[:, None])
            p = tl.where(score_mask, p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)
            wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _direct_narrow__TILE_N))
            wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _direct_narrow__TILE_N))
            tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
            pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
            pv = tle.gpu.wgmma_wait(0, pv)
            acc = acc * alpha[:, None] + pv
        else:
            scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
            score_mask = token_valid[:, None] & valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
            scores = tl.where(score_mask, scores, -float("inf"))
            page_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, page_max)
            active = m_new != -float("inf")
            safe_m = tl.where(active, m_new, 0.0)
            alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_m), 0.0)
            p = tl.exp2(scores - safe_m[None, :])
            p = tl.where(score_mask, p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            tl.store(p_ptr, p)
            v_rows = tl.broadcast_to(offs_n[:, None], (_direct_narrow__TILE_N, D))
            v_cols = tl.broadcast_to(offs_d[None, :], (_direct_narrow__TILE_N, D))
            v_regs = tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols)))
            v_regs_t = tl.trans(v_regs)
            v_wgmma_view = tle.gpu.alloc(
                [D, _direct_narrow__TILE_N],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                init_value=v_regs_t,
            )
            pv = tle.gpu.wgmma(v_wgmma_view, p_smem, out_dtype=tl.float32)
            pv = tle.gpu.wgmma_wait(0, pv)
            acc = acc * alpha[None, :] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    if WIDE_BASELINE:
        acc_rows = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    else:
        acc = tl.where(has_value[None, :], acc / l_i[None, :], 0.0)
        acc_rows = tl.trans(acc)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if group_count == 1:
        tl.store(
            OUT + batch * O_SB + seq_m[:, None] * O_SM + hq[:, None] * O_SH + offs_d[None, :],
            acc_rows,
            mask=valid_row[:, None],
        )
    else:
        tl.store(
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :],
            acc_rows,
            mask=valid_row[:, None],
        )
        tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH, lse, mask=valid_row)
    if PDL_NOTIFY:
        tl.extra.cuda.gdc_launch_dependents()


@dataclass(frozen=True)
class DynamicBF16Policy:
    layout: str
    mtp: int
    workload: DecodeWorkload
    cluster_size: int
    chunk_tokens: int
    kernel: str
    narrow: bool
    compact_direct_narrow: bool = False
    finalizer_heads: int = 1
    pdl_finalizer: bool = False
    deterministic_tail_election: bool = False
    reduction_only: bool = False
    aligned_full_chunk: bool = False

    @property
    def label(self) -> str:
        label = f"c{self.cluster_size}t{self.chunk_tokens}/{self.kernel}"
        if self.aligned_full_chunk:
            label += "-aligned"
        elif self.reduction_only:
            label += "-reduce"
        if self.deterministic_tail_election and self.cluster_size > 1:
            label += "-det"
        if self.finalizer_heads > 1:
            label += f"-h{self.finalizer_heads}-pdl"
        elif self.pdl_finalizer:
            label += "-pdl"
        return label


@dataclass
class DynamicBF16Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    layout: str

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


@dataclass
class DynamicBF16Workspace:
    policy: DynamicBF16Policy
    q_4d: torch.Tensor
    task_map: torch.Tensor
    direct_task_map: torch.Tensor
    subgroup2_task_map: torch.Tensor
    subgroup4_task_map: torch.Tensor
    offsets: torch.Tensor
    meta: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    completion: torch.Tensor
    out: torch.Tensor
    max_groups: int
    num_clusters: int
    num_direct_tasks: int
    num_subgroup2_clusters: int
    num_subgroup4_clusters: int
    groups_per_head: int
    direct_per_head: int
    subgroup2_per_head: int
    subgroup4_per_head: int


def _dynamic__build_schedule(
    inputs: DynamicBF16Inputs,
    policy: DynamicBF16Policy,
    hkv_count: int,
) -> dict[str, object]:
    """Allocate compact records and populate them with the shared GPU assigner.

    Host-visible lengths are used only to size fixed-capacity buffers and to
    validate topology.  The records consumed by BF16 compute kernels are
    emitted by :mod:`decode.assign_task`, like the FP8 dynamic paths.
    """
    lengths = inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    cluster_size = policy.cluster_size
    chunk_tokens = policy.chunk_tokens

    regular_prefix: list[int] = []
    direct_prefix: list[int] = []
    subgroup2_prefix: list[int] = []
    subgroup4_prefix: list[int] = []
    regular_per_head = direct_per_head = 0
    subgroup2_per_head = subgroup4_per_head = 0
    kinds: list[tuple[str, int, int]] = []

    for total_len in lengths:
        chunks = (total_len + chunk_tokens - 1) // chunk_tokens
        if total_len < inputs.mtp:
            kind = "invalid"
            groups = 0
        elif cluster_size > 1 and chunks == 1:
            kind = "direct"
            groups = 1
        elif cluster_size >= 4 and chunks == 2:
            kind = "subgroup2"
            groups = 1
        elif cluster_size >= 8 and 3 <= chunks <= 4:
            kind = "subgroup4"
            groups = 1
        else:
            kind = "regular"
            groups = (chunks + cluster_size - 1) // cluster_size

        regular_prefix.append(regular_per_head)
        direct_prefix.append(direct_per_head)
        subgroup2_prefix.append(subgroup2_per_head)
        subgroup4_prefix.append(subgroup4_per_head)
        if kind == "regular":
            regular_per_head += groups
        elif kind == "direct":
            direct_per_head += 1
        elif kind == "subgroup2":
            subgroup2_per_head += 1
        elif kind == "subgroup4":
            subgroup4_per_head += 1
        kinds.append((kind, chunks, groups))

    regular_records: list[list[int]] = []
    direct_records: list[list[int]] = []
    subgroup2_records: list[list[int]] = []
    subgroup4_records: list[list[int]] = []
    for hkv in range(hkv_count):
        for batch, total_len in enumerate(lengths):
            kind, chunks, group_count = kinds[batch]
            if kind == "invalid":
                continue
            if kind == "direct":
                direct_records.append([hkv, batch, 0, total_len, 0, 1, total_len, 1])
                continue
            if kind in ("subgroup2", "subgroup4"):
                width = 2 if kind == "subgroup2" else 4
                destination = subgroup2_records if kind == "subgroup2" else subgroup4_records
                for rank in range(width):
                    has_work = int(rank < chunks)
                    chunk_start = rank * chunk_tokens if has_work else 0
                    chunk_len = min(chunk_tokens, total_len - chunk_start) if has_work else 0
                    destination.append(
                        [
                            hkv,
                            batch,
                            chunk_start,
                            chunk_len,
                            0,
                            1,
                            total_len,
                            has_work,
                        ]
                    )
                continue
            for group in range(group_count):
                for rank in range(cluster_size):
                    chunk = group * cluster_size + rank
                    has_work = int(chunk < chunks)
                    chunk_start = chunk * chunk_tokens if has_work else 0
                    chunk_len = min(chunk_tokens, total_len - chunk_start) if has_work else 0
                    regular_records.append(
                        [
                            hkv,
                            batch,
                            chunk_start,
                            chunk_len,
                            group,
                            group_count,
                            total_len,
                            has_work,
                        ]
                    )

    device = inputs.q.device

    def record_tensor(records: list[list[int]]) -> torch.Tensor:
        if not records:
            records = [[0] * _dynamic__TASK_STRIDE]
        return torch.tensor(records, dtype=torch.int32, device=device)

    # The decode scheduler treats OFFSETS as two consecutive [B, 2]
    # planes: (regular, direct), then (subgroup2, subgroup4).  Preserve
    # that flattened layout even though the owning tensor is shaped [B, 4].
    offset_values = list(zip(regular_prefix, direct_prefix))
    offset_values += list(zip(subgroup2_prefix, subgroup4_prefix))
    offsets = torch.tensor(
        offset_values,
        dtype=torch.int32,
        device=device,
    ).reshape(inputs.batch, 4)
    meta_values = [
        regular_per_head * hkv_count,
        regular_per_head,
        direct_per_head * hkv_count,
        direct_per_head,
        subgroup2_per_head * hkv_count,
        subgroup2_per_head,
        subgroup4_per_head * hkv_count,
        subgroup4_per_head,
    ]
    schedule = {
        "offsets": offsets,
        "meta": torch.tensor(meta_values, dtype=torch.int32, device=device),
        "task_map": record_tensor(regular_records),
        "direct_task_map": record_tensor(direct_records),
        "subgroup2_task_map": record_tensor(subgroup2_records),
        "subgroup4_task_map": record_tensor(subgroup4_records),
        "num_clusters": meta_values[0],
        "groups_per_head": meta_values[1],
        "num_direct_tasks": meta_values[2],
        "direct_per_head": meta_values[3],
        "num_subgroup2_clusters": meta_values[4],
        "subgroup2_per_head": meta_values[5],
        "num_subgroup4_clusters": meta_values[6],
        "subgroup4_per_head": meta_values[7],
    }
    block_seq = triton.next_power_of_2(inputs.batch)
    _dynamic__assign_prefix_kernel[(1,)](
        inputs.kv_lens,
        schedule["offsets"],
        schedule["meta"],
        B=inputs.batch,
        H_KV=hkv_count,
        NUM_SEQ_Q=inputs.mtp,
        CLUSTER_SIZE=cluster_size,
        CHUNK_TOKENS=chunk_tokens,
        BLOCK_SEQ=block_seq,
        num_warps=max(1, min(32, block_seq // 32)),
        num_stages=1,
    )
    _dynamic__assign_records_kernel[(inputs.batch * hkv_count,)](
        inputs.kv_lens,
        schedule["offsets"],
        schedule["task_map"],
        schedule["direct_task_map"],
        schedule["subgroup2_task_map"],
        schedule["subgroup4_task_map"],
        B=inputs.batch,
        H_KV=hkv_count,
        CLUSTER_SIZE=cluster_size,
        CHUNK_TOKENS=chunk_tokens,
        GROUPS_PER_HEAD=regular_per_head,
        DIRECT_PER_HEAD=direct_per_head,
        SUBGROUP2_PER_HEAD=subgroup2_per_head,
        SUBGROUP4_PER_HEAD=subgroup4_per_head,
        num_warps=1,
        num_stages=1,
    )
    return schedule


def _dynamic__validate(inputs: DynamicBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16 or inputs.q.ndim != 3 or inputs.q.shape[-1] != 128:
        raise ValueError("q must be BF16 [batch * MTP, Hq, 128]")
    if inputs.mtp not in (1, 2, 3):
        raise ValueError("BF16 dynamic decode supports MTP 1, 2, or 3")
    for name, cache in (
        ("k_cache", inputs.k_cache),
        ("v_cache", inputs.v_cache),
    ):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[3] != 128:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32 or not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA int32 tensor")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("BF16 dynamic decode requires official GQA8 heads")
    if hkv > 1:
        inferred_k = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
        inferred_v = "HND" if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else "NHD"
        if inferred_k != inputs.layout or inferred_v != inputs.layout:
            raise ValueError("explicit layout does not match cache strides")
    return inputs.mtp, hq, hkv


def select_dynamic_bf16_policy(
    inputs: DynamicBF16Inputs,
) -> DynamicBF16Policy:
    """Apply measured static winners, with a safe variable-length fallback."""
    mtp, _hq, _hkv = _dynamic__validate(inputs)
    layout = inputs.layout
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    workload = DecodeWorkload.from_lengths(lengths)
    feature_panel = _FINAL_CT_BY_FEATURE[mtp]
    if workload.signature in feature_panel:
        cluster_size, chunk_tokens = feature_panel[workload.signature][layout]
        narrow = workload == _F_UNIFORM_512 and mtp == 2
        # The standalone H2 finalizer relies on PDL handoff.  Stock Triton
        # executes it correctly once, but repeated CUDA Graph replay is not
        # stable for either C4 or C8.  Keep finalization cooperative and
        # in-kernel for the finalized dynamic policy.
        finalizer_heads = 1
    else:
        max_len = max(lengths, default=0)
        if max_len <= 512:
            # The narrow/delayed-V winners were validated for the exact
            # uniform-512 panel.  A variable tail below 512 is not equivalent:
            # retain the static implementation's correctness-safe full-tail
            # route instead of extending that specialization's policy domain.
            cluster_size, chunk_tokens, narrow = 1, 1024, False
        elif max_len <= 4096:
            cluster_size, chunk_tokens, narrow = 1, 1024, False
        elif max_len <= 32768:
            cluster_size = 2 if mtp < 3 else 4
            chunk_tokens, narrow = 1024, False
        elif max_len <= 65536:
            cluster_size = 4 if mtp < 3 else 8
            chunk_tokens, narrow = 1024, False
        else:
            cluster_size = 4 if mtp < 3 else 8
            chunk_tokens, narrow = 1024, False
        finalizer_heads = 1
    if workload == _F_UNIFORM_512 and mtp in (1, 3):
        kernel = "uniform512-delayed-v"
        # The removed custom memdesc reshape carried a narrow WGMMA operand-A
        # layout that stock TLE's explicit rank-2 restaging cannot reproduce.
        # Preserve delayed-V scheduling while using its correct wide variant.
        narrow = False
    elif workload == _F_UNIFORM_512:
        kernel = "direct-fulltail"
        narrow = False
    elif cluster_size == 1:
        kernel = "direct-narrow" if narrow else "direct-fulltail"
    elif workload not in _KNOWN_FEATURES:
        kernel = "pipeline-fulltail-deferred"
        narrow = False
    elif workload in _PIPE_FEATURES:
        if workload in _FULLTAIL_FEATURES[mtp]:
            kernel = "pipeline-fulltail-deferred"
        elif workload in _DEFERRED_FEATURES[mtp]:
            kernel = "pipeline-deferred"
        else:
            kernel = "pipeline"
        narrow = False
    elif workload in _FULLTAIL_FEATURES[mtp]:
        kernel = "cluster-fulltail"
        narrow = False
    elif workload in _DEFERRED_FEATURES[mtp]:
        kernel = "cluster-deferred"
        narrow = False
    else:
        kernel = "cluster"
        narrow = False
    final_lengths = inputs.kv_lens.detach().to(torch.int64)
    chunks = torch.div(
        final_lengths + chunk_tokens - 1,
        chunk_tokens,
        rounding_mode="floor",
    )
    reduction_only = cluster_size > 1 and bool(torch.all((chunks > 1) & (chunks.remainder(cluster_size) == 0)).item())
    aligned_full_chunk = bool(
        torch.all(
            (final_lengths.remainder(chunk_tokens) == 0) & (chunks > 1) & (chunks.remainder(cluster_size) == 0)
        ).item()
    )
    deterministic_tail_election = mtp == 2 or (
        mtp == 1 and (cluster_size == 8 or (cluster_size == 4 and chunk_tokens in (128, 1024)))
    )
    pdl_finalizer = finalizer_heads > 1 or (cluster_size == 1 and int(chunks.max().item()) > 1)
    compact_direct_narrow = False
    return DynamicBF16Policy(
        layout=layout,
        mtp=mtp,
        workload=workload,
        cluster_size=cluster_size,
        chunk_tokens=chunk_tokens,
        kernel=kernel,
        narrow=narrow,
        compact_direct_narrow=compact_direct_narrow,
        finalizer_heads=finalizer_heads,
        pdl_finalizer=pdl_finalizer,
        deterministic_tail_election=deterministic_tail_election,
        reduction_only=reduction_only,
        aligned_full_chunk=aligned_full_chunk,
    )


def _prepare_dynamic_bf16_workspace_base(
    inputs: DynamicBF16Inputs,
) -> DynamicBF16Workspace:
    """Build the compact schedule outside the measured decode launch."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = select_dynamic_bf16_policy(inputs)
    max_len = int(inputs.kv_lens.max().item())
    max_chunks = triton.cdiv(max_len, policy.chunk_tokens)
    max_groups = max(1, triton.cdiv(max_chunks, policy.cluster_size))
    schedule = _dynamic__build_schedule(inputs, policy, hkv)
    offsets = schedule["offsets"]
    meta = schedule["meta"]
    task_map = schedule["task_map"]
    direct_task_map = schedule["direct_task_map"]
    subgroup2_task_map = schedule["subgroup2_task_map"]
    subgroup4_task_map = schedule["subgroup4_task_map"]
    num_clusters = int(schedule["num_clusters"])
    groups_per_head = int(schedule["groups_per_head"])
    num_direct_tasks = int(schedule["num_direct_tasks"])
    direct_per_head = int(schedule["direct_per_head"])
    num_subgroup2_clusters = int(schedule["num_subgroup2_clusters"])
    subgroup2_per_head = int(schedule["subgroup2_per_head"])
    num_subgroup4_clusters = int(schedule["num_subgroup4_clusters"])
    subgroup4_per_head = int(schedule["subgroup4_per_head"])
    if not any(
        (
            num_clusters,
            num_direct_tasks,
            num_subgroup2_clusters,
            num_subgroup4_clusters,
        )
    ):
        raise ValueError("each KV length must be at least MTP")
    storage_groups = triton.next_power_of_2(max_groups)
    device = inputs.q.device
    return DynamicBF16Workspace(
        policy=policy,
        q_4d=inputs.q.reshape(inputs.batch, mtp, hq, 128),
        task_map=task_map,
        direct_task_map=direct_task_map,
        subgroup2_task_map=subgroup2_task_map,
        subgroup4_task_map=subgroup4_task_map,
        offsets=offsets,
        meta=meta,
        split_out=torch.empty(
            (inputs.batch, storage_groups, mtp, hq, 128),
            dtype=torch.float32,
            device=device,
        ),
        split_lse=torch.empty(
            (inputs.batch, storage_groups, mtp, hq),
            dtype=torch.float32,
            device=device,
        ),
        completion=torch.zeros((inputs.batch * hkv,), dtype=torch.int32, device=device),
        out=torch.empty(
            (inputs.batch, mtp, hq, 128),
            dtype=torch.bfloat16,
            device=device,
        ),
        max_groups=max_groups,
        num_clusters=num_clusters,
        num_direct_tasks=num_direct_tasks,
        num_subgroup2_clusters=num_subgroup2_clusters,
        num_subgroup4_clusters=num_subgroup4_clusters,
        groups_per_head=groups_per_head,
        direct_per_head=direct_per_head,
        subgroup2_per_head=subgroup2_per_head,
        subgroup4_per_head=subgroup4_per_head,
    )


def _refresh_dynamic_bf16_task_map_base(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
) -> None:
    """Refresh records after KV tails move without changing schedule topology."""
    mtp, _hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if policy.mtp != mtp:
        raise ValueError("workspace policy does not match inputs")
    if policy.kernel == "uniform512-delayed-v":
        final_workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
        if final_workload != _F_UNIFORM_512:
            raise ValueError("uniform512 delayed-V workspace must be rebuilt after a tail move")
    if policy.aligned_full_chunk and not bool(
        torch.all(inputs.kv_lens.to(torch.int64).remainder(policy.chunk_tokens) == 0).item()
    ):
        raise ValueError("aligned-full-chunk workspace must be rebuilt after a tail move")
    schedule = _dynamic__build_schedule(inputs, policy, hkv)
    topology = (
        int(schedule["num_clusters"]),
        int(schedule["num_direct_tasks"]),
        int(schedule["num_subgroup2_clusters"]),
        int(schedule["num_subgroup4_clusters"]),
        int(schedule["groups_per_head"]),
        int(schedule["direct_per_head"]),
        int(schedule["subgroup2_per_head"]),
        int(schedule["subgroup4_per_head"]),
    )
    expected = (
        workspace.num_clusters,
        workspace.num_direct_tasks,
        workspace.num_subgroup2_clusters,
        workspace.num_subgroup4_clusters,
        workspace.groups_per_head,
        workspace.direct_per_head,
        workspace.subgroup2_per_head,
        workspace.subgroup4_per_head,
    )
    if topology != expected:
        raise ValueError("dynamic schedule topology changed; rebuild the workspace")
    for destination, name in (
        (workspace.offsets, "offsets"),
        (workspace.meta, "meta"),
        (workspace.task_map, "task_map"),
        (workspace.direct_task_map, "direct_task_map"),
        (workspace.subgroup2_task_map, "subgroup2_task_map"),
        (workspace.subgroup4_task_map, "subgroup4_task_map"),
    ):
        source = schedule[name]
        if destination.shape != source.shape:
            raise ValueError("dynamic schedule capacity changed; rebuild the workspace")
        destination.copy_(source)


def _attention_decode_bf16_dynamic_base(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
) -> torch.Tensor:
    """Run the fixed BF16 dynamic policy selected during workspace setup."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if policy.mtp != mtp or policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    heads_per_group = hq // hkv
    q_rows = max(8, triton.next_power_of_2(mtp * heads_per_group)) if policy.narrow else 64
    # Project the paged cache descriptor to [block, head, token, dim].  TMA
    # can then land each page directly in a rank-2 [64, 128] WGMMA operand,
    # avoiding the old rank-4 shared-memory restage.
    k_desc = TensorDescriptor.from_tensor(
        inputs.k_cache.permute(0, 2, 1, 3),
        block_shape=[1, 1, 64, 128],
    )
    v_desc = TensorDescriptor.from_tensor(
        inputs.v_cache.permute(0, 2, 1, 3),
        block_shape=[1, 1, 64, 128],
    )
    if policy.kernel == "uniform512-delayed-v":
        _dynamic__bf16_decode_uniform512_delayed_v_kernel[(inputs.batch * hkv,)](
            workspace.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            workspace.out,
            B=inputs.batch,
            NUM_SEQ_Q=mtp,
            Q_ROWS=q_rows,
            H_Q=hq,
            HEADS_PER_GROUP=heads_per_group,
            D=128,
            MAX_BLOCKS=inputs.block_ids.shape[1],
            Q_SB=workspace.q_4d.stride(0),
            Q_SM=workspace.q_4d.stride(1),
            Q_SH=workspace.q_4d.stride(2),
            O_SB=workspace.out.stride(0),
            O_SM=workspace.out.stride(1),
            O_SH=workspace.out.stride(2),
            WIDE_BASELINE=not policy.narrow,
            num_warps=4,
            num_stages=3,
            launch_pdl=False,
        )
        return workspace.out.reshape_as(inputs.q)
    common = dict(
        B=inputs.batch,
        NUM_SEQ_Q=mtp,
        Q_ROWS=q_rows,
        H_Q=hq,
        HEADS_PER_GROUP=heads_per_group,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=policy.chunk_tokens,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        PREFETCH_BLOCK_IDS=True,
        ALIGNED_FULL_CHUNK=policy.aligned_full_chunk,
    )
    c1_pdl = policy.pdl_finalizer and policy.finalizer_heads == 1

    def cluster_args(task_map: torch.Tensor) -> tuple[object, ...]:
        return (
            task_map,
            workspace.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            workspace.completion,
            workspace.split_out,
            workspace.split_lse,
            workspace.out,
        )

    def direct_args(task_map: torch.Tensor) -> tuple[object, ...]:
        return (
            task_map,
            workspace.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            workspace.split_out,
            workspace.split_lse,
            workspace.out,
        )

    cluster_kernels = {
        "cluster": _dynamic__bf16_decode_cluster_kernel,
        "cluster-deferred": _dynamic__bf16_decode_cluster_deferred_kernel,
        "cluster-fulltail": _dynamic__bf16_decode_cluster_fulltail_kernel,
        "pipeline": _dynamic__bf16_decode_cluster_pipeline_kernel,
        "pipeline-deferred": _dynamic__bf16_decode_cluster_pipeline_deferred_kernel,
        "pipeline-fulltail-deferred": _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel,
    }
    # FP8 dynamic's direct-task packing is retained for BF16.  The compact
    # path also keeps the BF16 static short-sequence winner: narrow WGMMA for
    # every validated MTP/layout pair except MTP1 HND.
    if workspace.num_direct_tasks:
        direct_common = dict(common)
        use_direct_narrow = policy.compact_direct_narrow
        if use_direct_narrow:
            direct_common["Q_ROWS"] = max(8, triton.next_power_of_2(mtp * heads_per_group))
            _dynamic__bf16_decode_direct_narrow_kernel[(workspace.num_direct_tasks,)](
                *direct_args(workspace.direct_task_map),
                **direct_common,
                WIDE_BASELINE=False,
                PDL_NOTIFY=False,
                TMA_STAGES=2,
                num_warps=4,
                num_stages=3,
                launch_pdl=False,
            )
        else:
            direct_common["Q_ROWS"] = 64
            _dynamic__bf16_decode_direct_fulltail_kernel[(workspace.num_direct_tasks,)](
                *direct_args(workspace.direct_task_map),
                **direct_common,
                PDL_NOTIFY=False,
                TMA_STAGES=1,
                num_warps=4,
                num_stages=3,
                launch_pdl=False,
            )
    if policy.kernel in cluster_kernels:
        for subgroup_size, subgroup_clusters, subgroup_map in (
            (2, workspace.num_subgroup2_clusters, workspace.subgroup2_task_map),
            (4, workspace.num_subgroup4_clusters, workspace.subgroup4_task_map),
        ):
            if subgroup_clusters:
                cluster_kernels[policy.kernel][(subgroup_clusters,)](
                    *cluster_args(subgroup_map),
                    mesh=_bf16_entry___CLUSTER_MESHES[subgroup_size],
                    NUM_SEQ_Q_PAD=triton.next_power_of_2(mtp),
                    CLUSTER_SIZE=subgroup_size,
                    REDUCTION_ONLY=policy.reduction_only,
                    DETERMINISTIC_TAIL_ELECTION=(policy.deterministic_tail_election),
                    COOPERATIVE_FINALIZE=policy.finalizer_heads == 1,
                    **common,
                    num_ctas=1,
                    num_warps=4,
                    num_stages=3,
                    launch_pdl=policy.finalizer_heads > 1,
                )
    # A TLE device-mesh grid is expressed in logical clusters.  program_id(0)
    # inside the kernel is expanded to the physical CTA id used by TASK_MAP.
    launch_clusters = workspace.num_clusters
    if workspace.num_clusters and policy.kernel == "direct-narrow":
        _dynamic__bf16_decode_direct_narrow_kernel[(launch_clusters,)](
            *direct_args(workspace.task_map),
            **common,
            WIDE_BASELINE=False,
            PDL_NOTIFY=c1_pdl,
            TMA_STAGES=2,
            num_warps=4,
            num_stages=3,
            launch_pdl=c1_pdl,
        )
    elif workspace.num_clusters and policy.kernel == "direct-fulltail":
        _dynamic__bf16_decode_direct_fulltail_kernel[(launch_clusters,)](
            *direct_args(workspace.task_map),
            **common,
            PDL_NOTIFY=c1_pdl,
            TMA_STAGES=1,
            num_warps=4,
            num_stages=3,
            launch_pdl=c1_pdl,
        )
    elif workspace.num_clusters:
        cluster_kernels[policy.kernel][(launch_clusters,)](
            *cluster_args(workspace.task_map),
            mesh=_bf16_entry___CLUSTER_MESHES[policy.cluster_size],
            NUM_SEQ_Q_PAD=triton.next_power_of_2(mtp),
            CLUSTER_SIZE=policy.cluster_size,
            REDUCTION_ONLY=policy.reduction_only,
            DETERMINISTIC_TAIL_ELECTION=(policy.deterministic_tail_election),
            COOPERATIVE_FINALIZE=policy.finalizer_heads == 1,
            **common,
            num_ctas=1,
            num_warps=4,
            num_stages=3,
            launch_pdl=policy.finalizer_heads > 1,
        )
    if workspace.max_groups > 1 and (policy.cluster_size == 1 or policy.finalizer_heads > 1):
        finalize_args = (
            workspace.split_out,
            workspace.split_lse,
            inputs.kv_lens,
            workspace.out,
        )
        finalize_kwargs = dict(
            NUM_SEQ_Q=mtp,
            NUM_SEQ_Q_PAD=triton.next_power_of_2(mtp),
            H_Q=hq,
            D=128,
            CLUSTER_SIZE=policy.cluster_size,
            CHUNK_TOKENS=policy.chunk_tokens,
            MAX_GROUPS=triton.next_power_of_2(workspace.max_groups),
            SO_SB=workspace.split_out.stride(0),
            SO_SG=workspace.split_out.stride(1),
            SO_SM=workspace.split_out.stride(2),
            SO_SH=workspace.split_out.stride(3),
            SL_SB=workspace.split_lse.stride(0),
            SL_SG=workspace.split_lse.stride(1),
            SL_SM=workspace.split_lse.stride(2),
            SL_SH=workspace.split_lse.stride(3),
            O_SB=workspace.out.stride(0),
            O_SM=workspace.out.stride(1),
            O_SH=workspace.out.stride(2),
        )
        if policy.finalizer_heads == 1:
            _base__bf16_decode_finalize_kernel[(inputs.batch, hq)](
                *finalize_args,
                **finalize_kwargs,
                PDL_WAIT=c1_pdl,
                num_warps=4,
                launch_pdl=c1_pdl,
            )
        else:
            _finalizer__bf16_decode_finalize_multihead_kernel[(inputs.batch, triton.cdiv(hq, policy.finalizer_heads))](
                *finalize_args,
                **finalize_kwargs,
                HEADS_PER_PROGRAM=policy.finalizer_heads,
                PDL_WAIT=True,
                num_warps=4,
                launch_pdl=True,
            )
    return workspace.out.reshape_as(inputs.q)


# ---- MTP1/MTP2 measured final paths ----
def _final_mtp1_full_final__workload(inputs: DynamicBF16Inputs) -> DecodeWorkload:
    if inputs.mtp != 1:
        raise ValueError("MTP1 full-final requires MTP=1")
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    if workload not in _KNOWN_FEATURES:
        raise ValueError("MTP1 full-final requires a tuned workload shape")
    return workload


@dataclass
class _final_mtp1_raw__MTP1RawWorkspace:
    base: DynamicBF16Workspace
    raw_m: torch.Tensor
    raw_l: torch.Tensor
    chunk_tokens: int


@dataclass
class _final_mtp1_full_final__MTP1FullFinalWorkspace:
    base: DynamicBF16Workspace
    raw: _final_mtp1_raw__MTP1RawWorkspace | None
    route: str


def _final_mtp1_full_final__mtp1_full_final_base_workspace(
    workspace: _final_mtp1_full_final__MTP1FullFinalWorkspace,
) -> DynamicBF16Workspace:
    return workspace.base


def _final_mtp1_full_final__mtp1_full_final_route_label(
    workspace: _final_mtp1_full_final__MTP1FullFinalWorkspace,
) -> str:
    return workspace.route


def _final_mtp1_raw__prepare_ct(inputs: DynamicBF16Inputs, cluster: int, tokens: int) -> DynamicBF16Workspace:
    _final_mtp1_full_final__workload(inputs)
    workspace = _prepare_dynamic_bf16_workspace_base(inputs)
    if (workspace.policy.cluster_size, workspace.policy.chunk_tokens) != (cluster, tokens):
        raise AssertionError("requested MTP1 C/T policy was not selected")
    return workspace


def _final_mtp1_full_final__narrow_policy_workspace(inputs: DynamicBF16Inputs) -> DynamicBF16Workspace:
    workload = _final_mtp1_full_final__workload(inputs)
    cluster, tokens = _FINAL_CT_BY_FEATURE[1][workload.signature][inputs.layout]
    ws = _final_mtp1_raw__prepare_ct(inputs, cluster, tokens)
    kernel = ws.policy.kernel
    if workload == _F_UNIFORM_4096:
        kernel = "direct-narrow"
    ws.policy = replace(ws.policy, kernel=kernel, narrow=True, compact_direct_narrow=True)
    return ws


def _final_mtp1_raw__prepare_dynamic_bf16_mtp1_raw_workspace(
    inputs: DynamicBF16Inputs, *, chunk_tokens: int
) -> _final_mtp1_raw__MTP1RawWorkspace:
    if chunk_tokens not in (1024, 2048):
        raise ValueError("MTP1 raw final supports T1024 or T2048")
    base = _final_mtp1_raw__prepare_ct(inputs, 1, chunk_tokens)
    if any((base.num_direct_tasks, base.num_subgroup2_clusters, base.num_subgroup4_clusters)):
        raise ValueError("raw final requires regular tasks only")
    batch, groups, mtp, heads = base.split_lse.shape
    device = base.split_lse.device

    def allocate() -> torch.Tensor:
        return torch.empty((batch, mtp, heads, groups), dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    return _final_mtp1_raw__MTP1RawWorkspace(base=base, raw_m=allocate(), raw_l=allocate(), chunk_tokens=chunk_tokens)


def _final_mtp1_full_final__prepare_dynamic_bf16_mtp1_full_final_workspace(
    inputs: DynamicBF16Inputs,
) -> _final_mtp1_full_final__MTP1FullFinalWorkspace:
    workload = _final_mtp1_full_final__workload(inputs)
    layout = inputs.layout
    if workload == _F_ONE_128K_31_SHORT or (workload == _F_TWO_32K_30_SHORT and layout == "HND"):
        raw = _final_mtp1_raw__prepare_dynamic_bf16_mtp1_raw_workspace(inputs, chunk_tokens=1024)
        raw.base.policy = replace(raw.base.policy, narrow=True)
        return _final_mtp1_full_final__MTP1FullFinalWorkspace(
            raw.base, raw, "c1t1024/narrow16-raw/chunk-minor/pdl/hpp2/r240"
        )
    ws = _final_mtp1_full_final__narrow_policy_workspace(inputs)
    if workload == _F_UNIFORM_512:
        route = "c1t1024/narrow16/tma2/delayed-v"
    elif workload == _F_UNIFORM_4096:
        route = "c1t1024/narrow16/direct/tma2"
    elif workload == _F_MIX_128_4096:
        route = f"c{ws.policy.cluster_size}t1024/narrow16/direct-v-ss/unified-direct-pack/paired-finalize"
    elif workload == _F_ONE_16K_MANY_64:
        route = "c8t256/narrow16/direct-v-ss/unified-direct-pack/bf16-dsm"
    elif workload == _F_ONE_64K_7_SHORT:
        route = "c8t512/narrow16/direct-v-ss/paired-producer/pdl-detached-hpp2-reducer"
    elif workload == _F_ONE_64K_15_SHORT:
        topology = "local-paired" if layout == "NHD" else "serial"
        route = f"c4t1024/narrow16/direct-v-ss/{topology}"
    elif workload == _F_ONE_64K_31_SHORT:
        route = "c8t512/narrow16/direct-v-ss/paired-bf16-dsm"
    else:
        route = "c4t1024/narrow16/direct-v-ss/local-paired"
    return _final_mtp1_full_final__MTP1FullFinalWorkspace(ws, None, route)


@dataclass
class _final_mtp1_merged_final__MTP1MergedFinalWorkspace:
    base: DynamicBF16Workspace
    kind: str
    inner: Any
    route: str


@dataclass(frozen=True)
class _final_mtp1_final_routes__MTP1C4Config:
    local_paired: bool
    causal_free: bool
    producer_maxnreg: int | None
    producer_num_stages: int

    @property
    def label(self) -> str:
        topology = "paired" if self.local_paired else "serial"
        tail = "causal-free" if self.causal_free else "causal"
        reg = "rdef" if self.producer_maxnreg is None else f"r{self.producer_maxnreg}"
        return f"c4t1024/narrow16/direct-v-ss/{topology}/{tail}/prefetch/pdl-detached-hpp2/{reg}/s{self.producer_num_stages}"


def _final_mtp1_merged_final__one64_winner(layout: str) -> _final_mtp1_final_routes__MTP1C4Config:
    if layout == "NHD":
        return _final_mtp1_final_routes__MTP1C4Config(True, False, None, 3)
    return _final_mtp1_final_routes__MTP1C4Config(False, True, 224, 2)


def _final_mtp1_final_routes__prepare_dynamic_bf16_mtp1_c4_config_workspace(
    inputs: DynamicBF16Inputs, config: _final_mtp1_final_routes__MTP1C4Config
) -> DynamicBF16Workspace:
    if _final_mtp1_full_final__workload(inputs) != _F_ONE_64K_15_SHORT:
        raise ValueError("C4 config is restricted to one_64k_15x4k")
    ws = _final_mtp1_raw__prepare_ct(inputs, 4, 1024)
    if any((ws.num_direct_tasks, ws.num_subgroup2_clusters, ws.num_subgroup4_clusters)):
        raise ValueError("C4 config requires regular tasks only")
    ws.policy = replace(ws.policy, narrow=True)
    return ws


def _final_mtp1_tail_reducers__prepare_dynamic_bf16_mtp1_one128_hpp1_workspace(
    inputs: DynamicBF16Inputs,
) -> _final_mtp1_raw__MTP1RawWorkspace:
    """Prepare the fixed C1T2048 chunk-minor producer workspace."""
    if _final_mtp1_full_final__workload(inputs) != _F_ONE_128K_31_SHORT:
        raise ValueError("compact reducer is restricted to one_128k_31x4k")
    base = _final_mtp1_raw__prepare_ct(inputs, 1, 2048)
    if any((base.num_direct_tasks, base.num_subgroup2_clusters, base.num_subgroup4_clusters)):
        raise ValueError("raw config requires regular tasks only")
    batch, groups, mtp, heads = base.split_lse.shape
    device = base.split_lse.device

    def allocate():
        return torch.empty((batch, mtp, heads, groups), dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    base.policy = replace(base.policy, narrow=True)
    return _final_mtp1_raw__MTP1RawWorkspace(base, allocate(), allocate(), 2048)


def _final_mtp1_one128_compact__prepare_dynamic_bf16_mtp1_one128_compact_workspace(
    inputs: DynamicBF16Inputs,
) -> _final_mtp1_raw__MTP1RawWorkspace:
    if DecodeWorkload.from_lengths(inputs.kv_lens.tolist()) != _F_ONE_128K_31_SHORT:
        raise ValueError("compact reducer is specialized for one_128k_31x4k")
    return _final_mtp1_tail_reducers__prepare_dynamic_bf16_mtp1_one128_hpp1_workspace(inputs)


_final_mtp1_tail_reducers__ONE64_HND_COMPACT = _final_mtp1_final_routes__MTP1C4Config(False, True, 224, 2)


def _final_mtp1_tail_reducers__prepare_dynamic_bf16_mtp1_one64_hnd_compact_workspace(
    inputs: DynamicBF16Inputs,
) -> DynamicBF16Workspace:
    if inputs.layout != "HND":
        raise ValueError("compact one64 route requires HND")
    return _final_mtp1_final_routes__prepare_dynamic_bf16_mtp1_c4_config_workspace(
        inputs, _final_mtp1_tail_reducers__ONE64_HND_COMPACT
    )


def _final_mtp1_merged_final__prepare_dynamic_bf16_mtp1_merged_final_workspace(
    inputs: DynamicBF16Inputs,
) -> _final_mtp1_merged_final__MTP1MergedFinalWorkspace:
    workload = _final_mtp1_full_final__workload(inputs)
    if workload == _F_UNIFORM_512:
        workspace = _prepare_dynamic_bf16_workspace_base(inputs)
        return _final_mtp1_merged_final__MTP1MergedFinalWorkspace(
            workspace, "current", workspace, "c1t1024/wide64/tma1/delayed-v"
        )
    if workload == _F_ONE_64K_15_SHORT:
        if inputs.layout == "HND":
            workspace = _final_mtp1_tail_reducers__prepare_dynamic_bf16_mtp1_one64_hnd_compact_workspace(inputs)
            return _final_mtp1_merged_final__MTP1MergedFinalWorkspace(
                workspace,
                "one64-compact",
                workspace,
                "c4t1024/narrow16/direct-v-ss/serial/causal-free/prefetch/compact-pdl-detached-hpp2/r224/s2",
            )
        config = _final_mtp1_merged_final__one64_winner(inputs.layout)
        workspace = _final_mtp1_final_routes__prepare_dynamic_bf16_mtp1_c4_config_workspace(inputs, config)
        return _final_mtp1_merged_final__MTP1MergedFinalWorkspace(workspace, "c4", (workspace, config), config.label)
    if workload == _F_ONE_128K_31_SHORT:
        workspace = _final_mtp1_one128_compact__prepare_dynamic_bf16_mtp1_one128_compact_workspace(inputs)
        return _final_mtp1_merged_final__MTP1MergedFinalWorkspace(
            workspace.base,
            "one128-compact",
            workspace,
            "c1t2048/narrow16-raw/chunk-minor/causal-free/pdl/heterogeneous-compact-39/r240/s3",
        )
    if workload == _F_TWO_32K_30_SHORT:
        workspace = _final_mtp1_raw__prepare_dynamic_bf16_mtp1_raw_workspace(inputs, chunk_tokens=2048)
        return _final_mtp1_merged_final__MTP1MergedFinalWorkspace(
            workspace.base, "raw-transfer", workspace, "c1t2048/narrow16-raw/chunk-minor/pdl/hpp2/r240"
        )
    workspace = _final_mtp1_full_final__prepare_dynamic_bf16_mtp1_full_final_workspace(inputs)
    return _final_mtp1_merged_final__MTP1MergedFinalWorkspace(
        _final_mtp1_full_final__mtp1_full_final_base_workspace(workspace),
        "base",
        workspace,
        _final_mtp1_full_final__mtp1_full_final_route_label(workspace),
    )


_final_mtp1_full_final__C4_MESH = _bf16_entry___CLUSTER_MESHES[4]

_final_mtp1_full_final__C8_MESH = _bf16_entry___CLUSTER_MESHES[8]


def _final_mtp1_full_final__common(inputs: DynamicBF16Inputs, ws: DynamicBF16Workspace):
    return dict(
        B=inputs.batch,
        NUM_SEQ_Q=1,
        Q_ROWS=16,
        H_Q=inputs.q.shape[1],
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=ws.policy.chunk_tokens,
        MAX_GROUPS=ws.max_groups,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=ws.split_lse.stride(0),
        SL_SG=ws.split_lse.stride(1),
        SL_SM=ws.split_lse.stride(2),
        SL_SH=ws.split_lse.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
    )


_final_mtp2_c4_narrow16_direct_v_ss__TASK_STRIDE = tl.constexpr(8)

_final_mtp2_c4_narrow16_direct_v_ss__TILE_N = tl.constexpr(64)

_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES = tl.constexpr(2)
_final_mtp2_c4_narrow16_direct_v_ss__MESH = _bf16_entry___CLUSTER_MESHES[4]


@triton.jit
def _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_paired_head_merge(
    partial_acc,
    partial_stats,
    paired_acc,
    paired_lse,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    rank,
    batch,
    hkv,
    group,
    group_count,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    Q_ROWS: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Merge rank-owned heads ``rank`` and ``rank + 4`` in one peer loop."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    offs_d = tl.arange(0, D)
    owned_h = rank + offs_h * 4
    owned_hq = hkv * HEADS_PER_GROUP + owned_h
    valid_h = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
    valid_m = offs_m < NUM_SEQ_Q
    valid = valid_h[:, None] & valid_m[None, :]
    source_rows = owned_h[:, None] + offs_m[None, :] * HEADS_PER_GROUP
    rows_3d = tl.broadcast_to(source_rows[:, :, None], (2, NUM_SEQ_Q_PAD, D))
    cols_3d = tl.broadcast_to(offs_d[None, None, :], (2, NUM_SEQ_Q_PAD, D))
    combined_acc = tl.zeros((2, NUM_SEQ_Q_PAD, D), tl.float32)
    combined_m = tl.full((2, NUM_SEQ_Q_PAD), -float("inf"), tl.float32)
    combined_l = tl.zeros((2, NUM_SEQ_Q_PAD), tl.float32)
    for peer in tl.static_range(0, 4):
        peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
        peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
        peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid, other=-float("inf"))
        peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid, other=0.0)
        peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (rows_3d, cols_3d)), mask=valid[:, :, None], other=0.0).to(
            tl.float32
        )
        new_m = tl.maximum(combined_m, peer_m)
        safe = tl.where(new_m != -float("inf"), new_m, 0.0)
        old_w = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe), 0.0)
        peer_w = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe), 0.0)
        combined_acc = combined_acc * old_w[:, :, None] + peer_acc * peer_w[:, :, None]
        combined_l = combined_l * old_w + peer_l * peer_w
        combined_m = new_m
    denom = tl.where(combined_l > 0.0, combined_l, 1.0)
    combined = tl.where(valid[:, :, None] & (combined_l[:, :, None] > 0.0), combined_acc / denom[:, :, None], 0.0)
    combined_lse = tl.where(valid & (combined_l > 0.0), tl.log2(denom) + combined_m, -float("inf"))
    stage_h = tl.broadcast_to(offs_h[:, None, None], (2, NUM_SEQ_Q_PAD, D))
    stage_m = tl.broadcast_to(offs_m[None, :, None], (2, NUM_SEQ_Q_PAD, D))
    tl.store(tle.gpu.local_ptr(paired_acc, (stage_h, stage_m, cols_3d)), combined)
    lse_h = tl.broadcast_to(offs_h[:, None], (2, NUM_SEQ_Q_PAD))
    lse_m = tl.broadcast_to(offs_m[None, :], (2, NUM_SEQ_Q_PAD))
    tl.store(tle.gpu.local_ptr(paired_lse, (lse_h, lse_m)), combined_lse)
    tl.debug_barrier()
    combined = tl.load(tle.gpu.local_ptr(paired_acc, (stage_h, stage_m, cols_3d)))
    combined_lse = tl.load(tle.gpu.local_ptr(paired_lse, (lse_h, lse_m)))
    out_ptr = OUT + batch * O_SB + offs_m[None, :, None] * O_SM + owned_hq[:, None, None] * O_SH + offs_d[None, None, :]
    split_ptr = (
        SPLIT_OUT
        + batch * SO_SB
        + group * SO_SG
        + offs_m[None, :, None] * SO_SM
        + owned_hq[:, None, None] * SO_SH
        + offs_d[None, None, :]
    )
    split_lse_ptr = SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m[None, :] * SL_SM + owned_hq[:, None] * SL_SH
    if group_count == 1:
        tl.store(out_ptr, combined, mask=valid[:, :, None])
    else:
        tl.store(split_ptr, combined, mask=valid[:, :, None])
        tl.store(split_lse_ptr, combined_lse, mask=valid)
    tl.debug_barrier()


@triton.jit
def _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_serial_head_merge(
    partial_acc,
    partial_stats,
    merged_acc,
    merged_lse,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    rank,
    batch,
    hkv,
    group,
    group_count,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    Q_ROWS: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Merge rank-owned heads serially."""
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    offs_d = tl.arange(0, D)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 2):
        owned_h = rank + head_pass * 4
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_owned = (owned_hq < H_Q) & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        source_rows_2d = tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D))
        combined_acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, 4):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0
            ).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            safe = tl.where(new_m != -float("inf"), new_m, 0.0)
            old_w = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe), 0.0)
            peer_w = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe), 0.0)
            combined_acc = combined_acc * old_w[:, None] + peer_acc * peer_w[:, None]
            combined_l = combined_l * old_w + peer_l * peer_w
            combined_m = new_m
        denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(denom) + combined_m, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()


@triton.jit
def _final_mtp2_c4_paired_finalize__kernel__c4_paired_head_group_finalize(
    SPLIT_OUT,
    SPLIT_LSE,
    COMPLETION,
    OUT,
    WINNER_SMEM,
    mesh: tl.constexpr,
    rank,
    batch,
    hkv,
    group,
    group_count,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    if group_count > 1:
        counter = COMPLETION + hkv * B + batch
        rank0_is_last = tl.zeros((), tl.int32)
        if rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_ELECTION:
                deterministic_owner = group == group_count - 1
                if deterministic_owner:
                    ready = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(counter, 1, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(counter, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            winner = rank0_is_last.to(tl.float32)
            tl.store(tle.gpu.local_ptr(WINNER_SMEM, (0,)), winner)
            for peer in tl.static_range(1, CLUSTER_SIZE):
                peer_flag = tle.remote(WINNER_SMEM, peer, scope=mesh)
                tl.store(tle.gpu.local_ptr(peer_flag, (0,)), winner)
        tle.distributed_barrier(mesh)
        is_last = tl.load(tle.gpu.local_ptr(WINNER_SMEM, (0,))) != 0.0
        if is_last:
            tl.atomic_add(counter, 0, sem="acq_rel", scope="gpu")
            offs_g = tl.arange(0, MAX_GROUPS)
            offs_h = tl.arange(0, 2)
            offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
            offs_d = tl.arange(0, D)
            owned_h = rank + offs_h * 4
            hq = hkv * HEADS_PER_GROUP + owned_h
            valid_h = (owned_h < HEADS_PER_GROUP) & (hq < H_Q)
            valid_m = offs_m < NUM_SEQ_Q
            valid_g = offs_g < group_count
            lse = tl.load(
                SPLIT_LSE
                + batch * SL_SB
                + offs_g[:, None, None] * SL_SG
                + offs_m[None, None, :] * SL_SM
                + hq[None, :, None] * SL_SH,
                mask=valid_g[:, None, None] & valid_h[None, :, None] & valid_m[None, None, :],
                other=-float("inf"),
            )
            max_lse = tl.max(lse, axis=0)
            safe = tl.where(max_lse != -float("inf"), max_lse, 0.0)
            weights = tl.where(valid_g[:, None, None], tl.exp2(lse - safe[None, :, :]), 0.0)
            denom = tl.sum(weights, axis=0)
            acc = tl.zeros((D, 2, NUM_SEQ_Q_PAD), tl.float32)
            for finalize_group in tl.static_range(0, MAX_GROUPS):
                group_valid = (finalize_group < group_count) & valid_h[:, None]
                group_lse = tl.load(
                    SPLIT_LSE + batch * SL_SB + finalize_group * SL_SG + offs_m[None, :] * SL_SM + hq[:, None] * SL_SH,
                    mask=group_valid & valid_m[None, :],
                    other=-float("inf"),
                )
                group_weight = tl.where(group_valid & valid_m[None, :], tl.exp2(group_lse - safe), 0.0)
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_SB
                    + finalize_group * SO_SG
                    + offs_m[None, None, :] * SO_SM
                    + hq[None, :, None] * SO_SH
                    + offs_d[:, None, None],
                    mask=group_valid[None, :, :] & valid_m[None, None, :],
                    other=0.0,
                )
                acc += partial * group_weight[None, :, :]
            acc /= tl.where(denom[None, :, :] > 0.0, denom[None, :, :], 1.0)
            tl.store(
                OUT + batch * O_SB + offs_m[None, None, :] * O_SM + hq[None, :, None] * O_SH + offs_d[:, None, None],
                acc,
                mask=valid_h[None, :, None] & valid_m[None, None, :],
            )
        tle.distributed_barrier(mesh)
        if (rank == 0) & is_last:
            tl.debug_barrier()
            tl.atomic_xchg(counter, 0, sem="release", scope="gpu")


@triton.jit
def _final_mtp2_c4_narrow16_direct_v_ss__kernel__mtp2_c4t1024_narrow16_direct_v_ss_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c4_narrow16_direct_v_ss__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c4_narrow16_direct_v_ss__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES,
        expect_bytes=_final_mtp2_c4_narrow16_direct_v_ss__TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES,
        expect_bytes=_final_mtp2_c4_narrow16_direct_v_ss__TILE_N * D * 2,
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=v_full[0],
    )
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(1),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=k_full[1],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(1),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=v_full[1],
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    query_pos = total_len - NUM_SEQ_Q + seq_m
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if CHUNK_TOKENS > 128:
        phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        tle.gpu.copy(
            K_DESC,
            k_smem.slot(0),
            [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
            [phys2, hkv, 0, 0],
            barrier=k_full[0],
        )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        phase = page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    phase = page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    global_n = chunk_start + page * _final_mtp2_c4_narrow16_direct_v_ss__TILE_N + offs_n
    if is_last_chunk:
        score_mask = valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
    else:
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(v_last, p_smem, trans_a=True, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_paired_head_merge(
                partial_acc,
                partial_stats,
                paired_acc,
                paired_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        else:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        _final_mtp2_c4_paired_finalize__kernel__c4_paired_head_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            4,
            MAX_GROUPS,
            True,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


_final_mtp2_c8_narrow16_direct_v_ss_detached__TASK_STRIDE = tl.constexpr(8)

_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N = tl.constexpr(64)

_final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES = tl.constexpr(2)


@triton.jit
def _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__c4_local_serial_head_merge(
    partial_acc,
    partial_stats,
    merged_acc,
    merged_lse,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    rank,
    batch,
    hkv,
    group,
    group_count,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    Q_ROWS: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Merge rank-owned heads serially."""
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    offs_d = tl.arange(0, D)
    valid_m = offs_m < NUM_SEQ_Q
    out_rows = tl.broadcast_to(offs_m[:, None], (NUM_SEQ_Q_PAD, D))
    out_cols = tl.broadcast_to(offs_d[None, :], (NUM_SEQ_Q_PAD, D))
    for head_pass in tl.static_range(0, 1):
        owned_h = rank + head_pass * 8
        owned_hq = hkv * HEADS_PER_GROUP + owned_h
        valid_owned = (owned_hq < H_Q) & valid_m
        source_rows = offs_m * HEADS_PER_GROUP + owned_h
        source_rows_2d = tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D))
        combined_acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float("inf"), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, 8):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float("inf"))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(
                tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0
            ).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            safe = tl.where(new_m != -float("inf"), new_m, 0.0)
            old_w = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe), 0.0)
            peer_w = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe), 0.0)
            combined_acc = combined_acc * old_w[:, None] + peer_acc * peer_w[:, None]
            combined_l = combined_l * old_w + peer_l * peer_w
            combined_m = new_m
        denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(denom) + combined_m, -float("inf"))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if group_count == 1:
            tl.store(
                OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + group * SO_SG
                + offs_m[:, None] * SO_SM
                + owned_hq * SO_SH
                + offs_d[None, :],
                combined,
                mask=valid_owned[:, None],
            )
            tl.store(
                SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH,
                combined_lse,
                mask=valid_owned,
            )
        tl.debug_barrier()


@triton.jit
def _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__c8_local_paired_head_merge(
    partial_acc,
    partial_stats,
    paired_acc,
    paired_lse,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    rank,
    batch,
    hkv,
    group,
    group_count,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    Q_ROWS: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Let C8 ranks 0--3 merge heads ``rank`` and ``rank + 4`` together."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    offs_d = tl.arange(0, D)
    owned_h = rank + offs_h * 4
    owned_hq = hkv * HEADS_PER_GROUP + owned_h
    valid_h = (owned_h < HEADS_PER_GROUP) & (owned_hq < H_Q)
    valid_m = offs_m < NUM_SEQ_Q
    valid = valid_h[:, None] & valid_m[None, :]
    source_rows = owned_h[:, None] + offs_m[None, :] * HEADS_PER_GROUP
    rows_3d = tl.broadcast_to(source_rows[:, :, None], (2, NUM_SEQ_Q_PAD, D))
    cols_3d = tl.broadcast_to(offs_d[None, None, :], (2, NUM_SEQ_Q_PAD, D))
    combined_acc = tl.zeros((2, NUM_SEQ_Q_PAD, D), tl.float32)
    combined_m = tl.full((2, NUM_SEQ_Q_PAD), -float("inf"), tl.float32)
    combined_l = tl.zeros((2, NUM_SEQ_Q_PAD), tl.float32)
    for peer in tl.static_range(0, 8):
        peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
        peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
        peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid, other=-float("inf"))
        peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid, other=0.0)
        peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (rows_3d, cols_3d)), mask=valid[:, :, None], other=0.0).to(
            tl.float32
        )
        new_m = tl.maximum(combined_m, peer_m)
        safe = tl.where(new_m != -float("inf"), new_m, 0.0)
        old_w = tl.where(combined_m != -float("inf"), tl.exp2(combined_m - safe), 0.0)
        peer_w = tl.where(peer_m != -float("inf"), tl.exp2(peer_m - safe), 0.0)
        combined_acc = combined_acc * old_w[:, :, None] + peer_acc * peer_w[:, :, None]
        combined_l = combined_l * old_w + peer_l * peer_w
        combined_m = new_m
    denom = tl.where(combined_l > 0.0, combined_l, 1.0)
    combined = tl.where(valid[:, :, None] & (combined_l[:, :, None] > 0.0), combined_acc / denom[:, :, None], 0.0)
    combined_lse = tl.where(valid & (combined_l > 0.0), tl.log2(denom) + combined_m, -float("inf"))
    stage_h = tl.broadcast_to(offs_h[:, None, None], (2, NUM_SEQ_Q_PAD, D))
    stage_m = tl.broadcast_to(offs_m[None, :, None], (2, NUM_SEQ_Q_PAD, D))
    tl.store(tle.gpu.local_ptr(paired_acc, (stage_h, stage_m, cols_3d)), combined)
    lse_h = tl.broadcast_to(offs_h[:, None], (2, NUM_SEQ_Q_PAD))
    lse_m = tl.broadcast_to(offs_m[None, :], (2, NUM_SEQ_Q_PAD))
    tl.store(tle.gpu.local_ptr(paired_lse, (lse_h, lse_m)), combined_lse)
    tl.debug_barrier()
    combined = tl.load(tle.gpu.local_ptr(paired_acc, (stage_h, stage_m, cols_3d)))
    combined_lse = tl.load(tle.gpu.local_ptr(paired_lse, (lse_h, lse_m)))
    out_ptr = OUT + batch * O_SB + offs_m[None, :, None] * O_SM + owned_hq[:, None, None] * O_SH + offs_d[None, None, :]
    split_ptr = (
        SPLIT_OUT
        + batch * SO_SB
        + group * SO_SG
        + offs_m[None, :, None] * SO_SM
        + owned_hq[:, None, None] * SO_SH
        + offs_d[None, None, :]
    )
    split_lse_ptr = SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m[None, :] * SL_SM + owned_hq[:, None] * SL_SH
    if group_count == 1:
        tl.store(out_ptr, combined, mask=valid[:, :, None])
    else:
        tl.store(split_ptr, combined, mask=valid[:, :, None])
        tl.store(split_lse_ptr, combined_lse, mask=valid)
    tl.debug_barrier()


@triton.jit
def _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
    DETACHED_GROUP_REDUCER: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c8_narrow16_direct_v_ss_detached__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [
            _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES,
            _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N,
            D,
        ],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [
            _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES,
            _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N,
            D,
        ],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, Q_ROWS],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES,
        expect_bytes=_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES,
        expect_bytes=_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N * D * 2,
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(0),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=v_full[0],
    )
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(1),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=k_full[1],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(1),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=v_full[1],
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    query_pos = total_len - NUM_SEQ_Q + seq_m
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    # T128 has exactly two pages.  The original C8 producer was written for
    # T512 and unconditionally prefetched page 2 after consuming page 0,
    # which made the otherwise valid T128 specialization read past the
    # two-entry block-id buffer.  Keep the same schedule for T256/T512 while
    # allowing the two-page pipeline to fall straight through to its final
    # page.
    if CHUNK_TOKENS > 128:
        phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        tle.gpu.copy(
            K_DESC,
            k_smem.slot(0),
            [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
            [phys2, hkv, 0, 0],
            barrier=k_full[0],
        )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
        phase = page // _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, Q_ROWS))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
    phase = page // _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    global_n = chunk_start + page * _final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N + offs_n
    if is_last_chunk:
        score_mask = valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
    else:
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_detached__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c8_narrow16_direct_v_ss_detached__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(v_last, p_smem, trans_a=True, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            if rank < 4:
                _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__c8_local_paired_head_merge(
                    partial_acc,
                    partial_stats,
                    paired_acc,
                    paired_lse,
                    SPLIT_OUT,
                    SPLIT_LSE,
                    OUT,
                    mesh,
                    rank,
                    batch,
                    hkv,
                    group,
                    group_count,
                    NUM_SEQ_Q,
                    NUM_SEQ_Q_PAD,
                    H_Q,
                    HEADS_PER_GROUP,
                    Q_ROWS,
                    D,
                    SO_SB,
                    SO_SG,
                    SO_SM,
                    SO_SH,
                    SL_SB,
                    SL_SG,
                    SL_SM,
                    SL_SH,
                    O_SB,
                    O_SM,
                    O_SH,
                )
        else:
            _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        if DETACHED_GROUP_REDUCER:
            tl.extra.cuda.gdc_launch_dependents()
        else:
            _dynamic__cooperative_group_finalize(
                SPLIT_OUT,
                SPLIT_LSE,
                COMPLETION,
                OUT,
                merged_lse,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                B,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                D,
                8,
                MAX_GROUPS,
                True,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )


def _final_mtp1_full_final__launch_cluster(
    inputs: DynamicBF16Inputs,
    ws: DynamicBF16Workspace,
    task_map: torch.Tensor,
    count: int,
    cluster: int,
    *,
    local_paired: bool,
    detached: bool = False,
    producer_maxnreg: int | None = None,
    producer_num_stages: int = 3,
) -> None:
    if not count:
        return
    k_desc, v_desc = _tensor_descriptors(inputs)
    common = _final_mtp1_full_final__common(inputs, ws)
    producer = (
        _final_mtp2_c4_narrow16_direct_v_ss__kernel__mtp2_c4t1024_narrow16_direct_v_ss_kernel
        if cluster == 4
        else _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel
    )
    kwargs = dict(
        mesh=_final_mtp1_full_final__C4_MESH if cluster == 4 else _final_mtp1_full_final__C8_MESH,
        NUM_SEQ_Q_PAD=1,
        LOCAL_PAIRED_HEADS=local_paired,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=producer_num_stages,
        maxnreg=producer_maxnreg,
        launch_pdl=detached,
    )
    if producer is _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel:
        kwargs["DETACHED_GROUP_REDUCER"] = detached
    producer[count,](
        task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        ws.split_lse,
        ws.split_lse,
        **common,
        **kwargs,
    )


def _final_mtp1_full_final__launch_direct(
    inputs: DynamicBF16Inputs, ws: DynamicBF16Workspace, task_map: torch.Tensor, count: int
) -> None:
    if not count:
        return
    k_desc, v_desc = _tensor_descriptors(inputs)
    common = _final_mtp1_full_final__common(inputs, ws)
    _dynamic__bf16_decode_direct_narrow_kernel[count,](
        task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.split_out,
        ws.split_lse,
        ws.out,
        **common,
        PREFETCH_BLOCK_IDS=True,
        ALIGNED_FULL_CHUNK=ws.policy.aligned_full_chunk,
        WIDE_BASELINE=False,
        PDL_NOTIFY=False,
        TMA_STAGES=2,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )


def _final_mtp1_full_final__run_c1_direct_narrow16(inputs: DynamicBF16Inputs, ws: DynamicBF16Workspace) -> torch.Tensor:
    _final_mtp1_full_final__launch_direct(inputs, ws, ws.task_map, ws.num_clusters)
    if ws.max_groups > 1:
        _base__bf16_decode_finalize_kernel[inputs.batch, inputs.q.shape[1]](
            ws.split_out,
            ws.split_lse,
            inputs.kv_lens,
            ws.out,
            NUM_SEQ_Q=1,
            NUM_SEQ_Q_PAD=1,
            H_Q=inputs.q.shape[1],
            D=128,
            CLUSTER_SIZE=1,
            CHUNK_TOKENS=ws.policy.chunk_tokens,
            MAX_GROUPS=triton.next_power_of_2(ws.max_groups),
            SO_SB=ws.split_out.stride(0),
            SO_SG=ws.split_out.stride(1),
            SO_SM=ws.split_out.stride(2),
            SO_SH=ws.split_out.stride(3),
            SL_SB=ws.split_lse.stride(0),
            SL_SG=ws.split_lse.stride(1),
            SL_SM=ws.split_lse.stride(2),
            SL_SH=ws.split_lse.stride(3),
            O_SB=ws.out.stride(0),
            O_SM=ws.out.stride(1),
            O_SH=ws.out.stride(2),
            PDL_WAIT=False,
            num_warps=4,
            launch_pdl=False,
        )
    return ws.out.reshape_as(inputs.q)


@triton.jit
def _final_mtp2_one64_7_c8_detached_reducer__kernel__mtp2_one64_c8_detached_hpp2_reducer(
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Finalize the sole 64K batch after the complete C8 producer grid."""
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    head_passes = tl.cdiv(HEADS_PER_GROUP, HEADS_PER_PROGRAM)
    seq_m = pid // head_passes
    head_pass = pid - seq_m * head_passes
    offs_h = tl.arange(0, HEADS_PER_PROGRAM)
    offs_g = tl.arange(0, GROUPS)
    offs_d = tl.arange(0, D)
    hq = head_pass * HEADS_PER_PROGRAM + offs_h
    valid_h = hq < HEADS_PER_GROUP
    lse = tl.load(
        SPLIT_LSE + offs_g[None, :] * SL_SG + seq_m * SL_SM + hq[:, None] * SL_SH,
        mask=valid_h[:, None],
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=1)
    safe_lse = tl.where(max_lse != -float("inf"), max_lse, 0.0)
    weights = tl.where(valid_h[:, None], tl.exp2(lse - safe_lse[:, None]), 0.0)
    denom = tl.sum(weights, axis=1)
    acc = tl.zeros((HEADS_PER_PROGRAM, D), tl.float32)
    for group in tl.static_range(0, GROUPS):
        group_lse = tl.load(SPLIT_LSE + group * SL_SG + seq_m * SL_SM + hq * SL_SH, mask=valid_h, other=-float("inf"))
        weight = tl.where(valid_h, tl.exp2(group_lse - safe_lse), 0.0)
        partial = tl.load(
            SPLIT_OUT + group * SO_SG + seq_m * SO_SM + hq[:, None] * SO_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += partial * weight[:, None]
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    result = acc / safe_denom[:, None]
    tl.store(
        OUT + seq_m * O_SM + hq[:, None] * O_SH + offs_d[None, :],
        result,
        mask=valid_h[:, None] & (denom[:, None] > 0.0),
    )


def _final_mtp1_full_final__run_c8_detached(inputs: DynamicBF16Inputs, ws: DynamicBF16Workspace) -> torch.Tensor:
    _final_mtp1_full_final__launch_cluster(
        inputs,
        ws,
        ws.task_map,
        ws.num_clusters,
        8,
        local_paired=True,
        detached=True,
        producer_maxnreg=None if inputs.layout == "NHD" else 192,
        producer_num_stages=3 if inputs.layout == "NHD" else 2,
    )
    heads_per_program = 2
    _final_mtp2_one64_7_c8_detached_reducer__kernel__mtp2_one64_c8_detached_hpp2_reducer[
        triton.cdiv(8, heads_per_program),
    ](
        ws.split_out,
        ws.split_lse,
        ws.out,
        NUM_SEQ_Q=1,
        HEADS_PER_GROUP=8,
        D=128,
        GROUPS=16,
        HEADS_PER_PROGRAM=heads_per_program,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=ws.split_lse.stride(0),
        SL_SG=ws.split_lse.stride(1),
        SL_SM=ws.split_lse.stride(2),
        SL_SH=ws.split_lse.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


_final_mtp1_full_final__TASK_STRIDE = tl.constexpr(8)

_final_mtp1_full_final__direct_narrow_producer = _dynamic__bf16_decode_direct_narrow_kernel


@triton.jit
def _final_mtp1_full_final__kernel__mtp1_narrow16_unified_cluster_direct_kernel(
    REGULAR_TASK_MAP,
    DIRECT_TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    NUM_REGULAR_CLUSTERS: tl.constexpr,
    NUM_DIRECT_TASKS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """One grid for the retained narrow cluster and compact-direct paths."""
    cta = tl.program_id(0)
    regular_ctas = NUM_REGULAR_CLUSTERS * CLUSTER_SIZE
    if cta < regular_ctas:
        if CLUSTER_SIZE == 4:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__mtp2_c4t1024_narrow16_direct_v_ss_kernel(
                REGULAR_TASK_MAP,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                COMPLETION,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                SPLIT_LSE,
                SPLIT_LSE,
                mesh=mesh,
                B=B,
                NUM_SEQ_Q=1,
                Q_ROWS=16,
                NUM_SEQ_Q_PAD=1,
                H_Q=H_Q,
                HEADS_PER_GROUP=8,
                D=D,
                BLOCK_SIZE=64,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                LOCAL_PAIRED_HEADS=True,
                RAW_DETACHED=False,
                PDL_NOTIFY=False,
            )
        else:
            _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel(
                REGULAR_TASK_MAP,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                COMPLETION,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                SPLIT_LSE,
                SPLIT_LSE,
                mesh=mesh,
                B=B,
                NUM_SEQ_Q=1,
                Q_ROWS=16,
                NUM_SEQ_Q_PAD=1,
                H_Q=H_Q,
                HEADS_PER_GROUP=8,
                D=D,
                BLOCK_SIZE=64,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                LOCAL_PAIRED_HEADS=True,
                RAW_DETACHED=False,
                PDL_NOTIFY=False,
                DETACHED_GROUP_REDUCER=False,
            )
    else:
        direct_index = cta - regular_ctas
        if direct_index < NUM_DIRECT_TASKS:
            shifted_map = DIRECT_TASK_MAP - regular_ctas * _final_mtp1_full_final__TASK_STRIDE
            _final_mtp1_full_final__direct_narrow_producer(
                shifted_map,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                B=B,
                NUM_SEQ_Q=1,
                Q_ROWS=16,
                H_Q=H_Q,
                HEADS_PER_GROUP=8,
                D=D,
                BLOCK_SIZE=64,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                PREFETCH_BLOCK_IDS=True,
                ALIGNED_FULL_CHUNK=False,
                WIDE_BASELINE=False,
                PDL_NOTIFY=False,
                TMA_STAGES=2,
            )


def _final_mtp1_full_final__run_compact_mixed(inputs: DynamicBF16Inputs, ws: DynamicBF16Workspace) -> torch.Tensor:
    if ws.num_subgroup2_clusters:
        raise ValueError("official MTP1 full-final unexpectedly selected C2")
    if ws.num_clusters and ws.num_subgroup4_clusters:
        raise ValueError("mixed MTP1 route must have one regular topology")
    if ws.num_subgroup4_clusters:
        cluster = 4
        regular_count = ws.num_subgroup4_clusters
        regular_map = ws.subgroup4_task_map
    else:
        cluster = ws.policy.cluster_size
        regular_count = ws.num_clusters
        regular_map = ws.task_map
    k_desc, v_desc = _tensor_descriptors(inputs)
    physical_direct_clusters = triton.cdiv(ws.num_direct_tasks, cluster)
    grid = regular_count + physical_direct_clusters
    _final_mtp1_full_final__kernel__mtp1_narrow16_unified_cluster_direct_kernel[grid,](
        regular_map,
        ws.direct_task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        mesh=_final_mtp1_full_final__C4_MESH if cluster == 4 else _final_mtp1_full_final__C8_MESH,
        B=inputs.batch,
        H_Q=inputs.q.shape[1],
        D=128,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=ws.policy.chunk_tokens,
        MAX_GROUPS=ws.max_groups,
        CLUSTER_SIZE=cluster,
        NUM_REGULAR_CLUSTERS=regular_count,
        NUM_DIRECT_TASKS=ws.num_direct_tasks,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=ws.split_lse.stride(0),
        SL_SG=ws.split_lse.stride(1),
        SL_SM=ws.split_lse.stride(2),
        SL_SH=ws.split_lse.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return ws.out.reshape_as(inputs.q)


def _final_mtp1_full_final__run_uniform512_narrow16(
    inputs: DynamicBF16Inputs, ws: DynamicBF16Workspace
) -> torch.Tensor:
    k_desc, v_desc = _tensor_descriptors(inputs)
    _dynamic__bf16_decode_uniform512_delayed_v_kernel[inputs.batch,](
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.out,
        B=inputs.batch,
        NUM_SEQ_Q=1,
        Q_ROWS=16,
        H_Q=inputs.q.shape[1],
        HEADS_PER_GROUP=8,
        D=128,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        WIDE_BASELINE=False,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return ws.out.reshape_as(inputs.q)


@triton.jit
def _final_mtp2_c1_raw_detached__kernel__mtp2_c1_raw_detached_reducer(
    KV_LENS,
    SPLIT_OUT,
    RAW_M,
    RAW_L,
    OUT,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    RS_SB: tl.constexpr,
    RS_SG: tl.constexpr,
    RS_SM: tl.constexpr,
    RS_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    PDL_WAIT: tl.constexpr,
    CUDA_SPLIT16: tl.constexpr,
):
    """Merge four heads of C1 raw ``(numerator, m, l)`` state."""
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    head_passes = tl.cdiv(HEADS_PER_GROUP, HEADS_PER_PROGRAM)
    programs_per_sequence = NUM_SEQ_Q * head_passes
    sequence_id = pid // programs_per_sequence
    sequence_program = pid - sequence_id * programs_per_sequence
    seq_m = sequence_program // head_passes
    head_pass = sequence_program - seq_m * head_passes
    hkv = sequence_id // B
    batch = sequence_id - hkv * B
    offs_h = tl.arange(0, HEADS_PER_PROGRAM)
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, MAX_GROUPS)
    h_in_group = head_pass * HEADS_PER_PROGRAM + offs_h
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_h = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    if CUDA_SPLIT16:
        split_tokens = (total_len + 15) // 16
        split_tokens = (split_tokens + 63) // 64 * 64
        split_tokens = tl.maximum(split_tokens, 512)
        groups = (total_len + split_tokens - 1) // split_tokens
    else:
        groups = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    valid_g = offs_g[None, :] < groups
    scalar_ptr = batch * RS_SB + offs_g[None, :] * RS_SG + seq_m * RS_SM + hq[:, None] * RS_SH
    scalar_mask = valid_h[:, None] & valid_g
    m_values = tl.load(RAW_M + scalar_ptr, mask=scalar_mask, other=-float("inf"))
    l_values = tl.load(RAW_L + scalar_ptr, mask=scalar_mask, other=0.0)
    max_m = tl.max(m_values, axis=1)
    safe_m = tl.where(max_m != -float("inf"), max_m, 0.0)
    weights = tl.where(scalar_mask, tl.exp2(m_values - safe_m[:, None]), 0.0)
    denom = tl.sum(l_values * weights, axis=1)
    acc = tl.zeros((HEADS_PER_PROGRAM, D), tl.float32)
    for group in tl.static_range(0, MAX_GROUPS):
        group_valid = (group < groups) & valid_h
        group_m = tl.load(
            RAW_M + batch * RS_SB + group * RS_SG + seq_m * RS_SM + hq * RS_SH, mask=group_valid, other=-float("inf")
        )
        group_weight = tl.where(group_valid, tl.exp2(group_m - safe_m), 0.0)
        partial = tl.load(
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m * SO_SM + hq[:, None] * SO_SH + offs_d[None, :],
            mask=group_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += partial * group_weight[:, None]
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    result = acc / safe_denom[:, None]
    tl.store(
        OUT + batch * O_SB + seq_m * O_SM + hq[:, None] * O_SH + offs_d[None, :],
        result,
        mask=valid_h[:, None] & (denom[:, None] > 0.0),
    )


_final_mtp2_c4_narrow16__MESH = _bf16_entry___CLUSTER_MESHES[4]

_final_mtp2_c4_narrow16__MESH_C1 = _final_mtp2_c4_narrow16__MESH

_final_mtp2_c4_narrow16__TASK_STRIDE = tl.constexpr(8)

_final_mtp2_c4_narrow16__TILE_N = tl.constexpr(64)

_final_mtp2_c4_narrow16__TMA_STAGES = tl.constexpr(2)


@triton.jit
def _final_mtp2_c4_narrow16__kernel__mtp2_c4t1024_narrow16_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c4_narrow16__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c4_narrow16__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16__TMA_STAGES, _final_mtp2_c4_narrow16__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16__TMA_STAGES, _final_mtp2_c4_narrow16__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16__TMA_STAGES, expect_bytes=_final_mtp2_c4_narrow16__TILE_N * D * 2
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16__TMA_STAGES, expect_bytes=_final_mtp2_c4_narrow16__TILE_N * D * 2
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c4_narrow16__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC, k_smem.slot(0), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0]
    )
    tle.gpu.copy(
        V_DESC, v_smem.slot(0), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0]
    )
    tle.gpu.copy(
        K_DESC, k_smem.slot(1), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1]
    )
    tle.gpu.copy(
        V_DESC, v_smem.slot(1), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1]
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    query_pos = total_len - NUM_SEQ_Q + seq_m
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
    tle.gpu.copy(
        K_DESC, k_smem.slot(0), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0]
    )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c4_narrow16__TMA_STAGES
        phase = page // _final_mtp2_c4_narrow16__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c4_narrow16__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c4_narrow16__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        v_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, D))
        v_cols = tl.broadcast_to(offs_d[None, :], (_final_mtp2_c4_narrow16__TILE_N, D))
        v_regs = tl.load(tle.gpu.local_ptr(v_prev, (v_rows, v_cols)))
        v_regs_t = tl.trans(v_regs)
        v_wgmma = tle.gpu.alloc(
            [D, _final_mtp2_c4_narrow16__TILE_N],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            init_value=v_regs_t,
        )
        pv = tle.gpu.wgmma(v_wgmma, p_smem, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c4_narrow16__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c4_narrow16__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c4_narrow16__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c4_narrow16__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c4_narrow16__TMA_STAGES
    phase = page // _final_mtp2_c4_narrow16__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c4_narrow16__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c4_narrow16__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    global_n = chunk_start + page * _final_mtp2_c4_narrow16__TILE_N + offs_n
    if is_last_chunk:
        score_mask = valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
    else:
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    v_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_cols = tl.broadcast_to(offs_d[None, :], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_regs = tl.load(tle.gpu.local_ptr(v_prev, (v_rows, v_cols)))
    v_regs_t = tl.trans(v_regs)
    v_wgmma = tle.gpu.alloc(
        [D, _final_mtp2_c4_narrow16__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, init_value=v_regs_t
    )
    pv = tle.gpu.wgmma(v_wgmma, p_smem, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c4_narrow16__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c4_narrow16__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    v_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_cols = tl.broadcast_to(offs_d[None, :], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_regs = tl.load(tle.gpu.local_ptr(v_last, (v_rows, v_cols)))
    v_regs_t = tl.trans(v_regs)
    v_wgmma = tle.gpu.alloc(
        [D, _final_mtp2_c4_narrow16__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, init_value=v_regs_t
    )
    pv_last = tle.gpu.wgmma(v_wgmma, p_smem, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_paired_head_merge(
                partial_acc,
                partial_stats,
                paired_acc,
                paired_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        else:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        _final_mtp2_c4_paired_finalize__kernel__c4_paired_head_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            4,
            MAX_GROUPS,
            True,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


def _final_mtp1_raw__attention_decode_bf16_dynamic_mtp1_raw(
    inputs: DynamicBF16Inputs, workspace: _final_mtp1_raw__MTP1RawWorkspace
):
    ws = workspace.base
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 1 or hq // hkv != 8:
        raise ValueError("raw final requires MTP1 GQA8")
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c4_narrow16__kernel__mtp2_c4t1024_narrow16_kernel[ws.num_clusters,](
        ws.task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        workspace.raw_m,
        workspace.raw_l,
        mesh=_final_mtp2_c4_narrow16__MESH_C1,
        B=inputs.batch,
        NUM_SEQ_Q=1,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=1,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=workspace.chunk_tokens,
        MAX_GROUPS=ws.max_groups,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=workspace.raw_m.stride(0),
        SL_SG=workspace.raw_m.stride(1),
        SL_SM=workspace.raw_m.stride(2),
        SL_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=True,
        PDL_NOTIFY=True,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        maxnreg=240,
        launch_pdl=True,
    )
    heads_per_program = 2
    grid = inputs.batch * hkv * triton.cdiv(8, heads_per_program)
    _final_mtp2_c1_raw_detached__kernel__mtp2_c1_raw_detached_reducer[grid,](
        inputs.kv_lens,
        ws.split_out,
        workspace.raw_m,
        workspace.raw_l,
        ws.out,
        B=inputs.batch,
        NUM_SEQ_Q=1,
        NUM_SEQ_Q_PAD=1,
        H_Q=hq,
        H_KV=hkv,
        HEADS_PER_GROUP=8,
        D=128,
        CHUNK_TOKENS=workspace.chunk_tokens,
        MAX_GROUPS=triton.next_power_of_2(ws.max_groups),
        HEADS_PER_PROGRAM=heads_per_program,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        RS_SB=workspace.raw_m.stride(0),
        RS_SG=workspace.raw_m.stride(1),
        RS_SM=workspace.raw_m.stride(2),
        RS_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        PDL_WAIT=True,
        CUDA_SPLIT16=False,
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


def _final_mtp1_full_final__attention_decode_bf16_dynamic_mtp1_full_final(
    inputs: DynamicBF16Inputs, workspace: _final_mtp1_full_final__MTP1FullFinalWorkspace
):
    ws = workspace.base
    workload = ws.policy.workload
    if workspace.raw is not None:
        return _final_mtp1_raw__attention_decode_bf16_dynamic_mtp1_raw(inputs, workspace.raw)
    if workload == _F_UNIFORM_512:
        return _final_mtp1_full_final__run_uniform512_narrow16(inputs, ws)
    if workload == _F_UNIFORM_4096:
        return _final_mtp1_full_final__run_c1_direct_narrow16(inputs, ws)
    if workload in (_F_MIX_128_4096, _F_ONE_16K_MANY_64):
        return _final_mtp1_full_final__run_compact_mixed(inputs, ws)
    if workload == _F_ONE_64K_7_SHORT:
        return _final_mtp1_full_final__run_c8_detached(inputs, ws)
    if workload == _F_ONE_64K_15_SHORT:
        _final_mtp1_full_final__launch_cluster(
            inputs,
            ws,
            ws.task_map,
            ws.num_clusters,
            4,
            local_paired=inputs.layout == "NHD",
            producer_maxnreg=None if inputs.layout == "NHD" else 224,
            producer_num_stages=3 if inputs.layout == "NHD" else 2,
        )
        return ws.out.reshape_as(inputs.q)
    if workload == _F_ONE_64K_31_SHORT:
        _final_mtp1_full_final__launch_cluster(
            inputs,
            ws,
            ws.task_map,
            ws.num_clusters,
            8,
            local_paired=True,
            producer_maxnreg=None if inputs.layout == "NHD" else 192,
            producer_num_stages=3 if inputs.layout == "NHD" else 2,
        )
        return ws.out.reshape_as(inputs.q)
    _final_mtp1_full_final__launch_cluster(inputs, ws, ws.task_map, ws.num_clusters, 4, local_paired=True)
    return ws.out.reshape_as(inputs.q)


_final_mtp1_final_routes__C4_MESH = _bf16_entry___CLUSTER_MESHES[4]


@triton.jit
def _final_mtp1_final_routes__kernel__mtp1_c4_detached_hpp_reducer(
    KV_LENS,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    B: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """PDL reducer for every batch; C4T512 short batches have two groups."""
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    head_passes = tl.cdiv(HEADS_PER_GROUP, HEADS_PER_PROGRAM)
    batch = pid // head_passes
    head_pass = pid - batch * head_passes
    total_len = tl.load(KV_LENS + batch)
    group_count = tl.cdiv(total_len, 4 * CHUNK_TOKENS)
    needs_reduce = group_count > 1
    offs_h = tl.arange(0, HEADS_PER_PROGRAM)
    offs_g = tl.arange(0, MAX_GROUPS)
    offs_d = tl.arange(0, D)
    hq = head_pass * HEADS_PER_PROGRAM + offs_h
    valid_h = hq < HEADS_PER_GROUP
    valid_g = (offs_g < group_count) & needs_reduce
    lse = tl.load(
        SPLIT_LSE + batch * SL_SB + offs_g[None, :] * SL_SG + hq[:, None] * SL_SH,
        mask=valid_h[:, None] & valid_g[None, :],
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=1)
    safe_lse = tl.where(max_lse != -float("inf"), max_lse, 0.0)
    weights = tl.where(valid_h[:, None] & valid_g[None, :], tl.exp2(lse - safe_lse[:, None]), 0.0)
    denom = tl.sum(weights, axis=1)
    acc = tl.zeros((HEADS_PER_PROGRAM, D), tl.float32)
    for group in tl.static_range(0, MAX_GROUPS):
        valid_group = (group < group_count) & needs_reduce
        group_lse = tl.load(
            SPLIT_LSE + batch * SL_SB + group * SL_SG + hq * SL_SH, mask=valid_h & valid_group, other=-float("inf")
        )
        weight = tl.where(valid_h & valid_group, tl.exp2(group_lse - safe_lse), 0.0)
        partial = tl.load(
            SPLIT_OUT + batch * SO_SB + group * SO_SG + hq[:, None] * SO_SH + offs_d[None, :],
            mask=valid_h[:, None] & valid_group,
            other=0.0,
        ).to(tl.float32)
        acc += partial * weight[:, None]
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + batch * O_SB + hq[:, None] * O_SH + offs_d[None, :],
        acc / safe_denom[:, None],
        mask=valid_h[:, None] & needs_reduce & (denom[:, None] > 0.0),
    )


def _final_mtp1_final_routes__attention_decode_bf16_dynamic_mtp1_c4_config(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace, config: _final_mtp1_final_routes__MTP1C4Config
):
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 1 or (hq, hkv) != (8, 1):
        raise ValueError("C4 config requires MTP1/HQ8/HKV1")
    if (workspace.policy.cluster_size, workspace.policy.chunk_tokens) != (4, 1024):
        raise ValueError("workspace does not match C4 config")
    k_desc, v_desc = _tensor_descriptors(inputs)
    kernel = _mtp1_one64_hnd_producer if config.causal_free else _mtp1_one64_nhd_producer
    kernel[workspace.num_clusters,](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_final_mtp1_final_routes__C4_MESH,
        B=inputs.batch,
        NUM_SEQ_Q=1,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=1,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=1024,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=config.local_paired,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=config.producer_num_stages,
        maxnreg=config.producer_maxnreg,
        launch_pdl=True,
    )
    grid = inputs.batch * triton.cdiv(8, 2)
    _final_mtp1_final_routes__kernel__mtp1_c4_detached_hpp_reducer[grid,](
        inputs.kv_lens,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        B=inputs.batch,
        CHUNK_TOKENS=1024,
        MAX_GROUPS=triton.next_power_of_2(workspace.max_groups),
        HEADS_PER_GROUP=8,
        D=128,
        HEADS_PER_PROGRAM=2,
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return workspace.out.reshape_as(inputs.q)


@triton.jit
def _final_mtp1_one128_compact__kernel__mtp1_one128_heterogeneous_compact_reducer(
    SPLIT_OUT,
    RAW_M,
    RAW_L,
    OUT,
    LONG_GROUPS: tl.constexpr,
    SHORT_GROUPS: tl.constexpr,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    RS_SB: tl.constexpr,
    RS_SG: tl.constexpr,
    RS_SM: tl.constexpr,
    RS_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Use HPP1 for batch 0 and HPP8 for each exact-two-group batch."""
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    if pid < H_Q:
        hq = pid
        long_offs_g = tl.arange(0, LONG_GROUPS)
        long_m = tl.load(RAW_M + long_offs_g * RS_SG + hq * RS_SH)
        long_l = tl.load(RAW_L + long_offs_g * RS_SG + hq * RS_SH)
        long_max_m = tl.max(long_m, axis=0)
        long_weights = tl.exp2(long_m - long_max_m)
        long_denom = tl.sum(long_l * long_weights, axis=0)
        long_acc = tl.zeros((D,), tl.float32)
        for long_group in tl.static_range(0, LONG_GROUPS):
            long_group_m = tl.load(RAW_M + long_group * RS_SG + hq * RS_SH)
            long_partial = tl.load(SPLIT_OUT + long_group * SO_SG + hq * SO_SH + offs_d).to(tl.float32)
            long_acc += long_partial * tl.exp2(long_group_m - long_max_m)
        tl.store(OUT + hq * O_SH + offs_d, long_acc / long_denom)
    else:
        batch = pid - H_Q + 1
        short_offs_h = tl.arange(0, H_Q)
        short_offs_g = tl.arange(0, SHORT_GROUPS)
        short_scalar = batch * RS_SB + short_offs_g[None, :] * RS_SG + short_offs_h[:, None] * RS_SH
        short_m = tl.load(RAW_M + short_scalar)
        short_l = tl.load(RAW_L + short_scalar)
        short_max_m = tl.max(short_m, axis=1)
        short_weights = tl.exp2(short_m - short_max_m[:, None])
        short_denom = tl.sum(short_l * short_weights, axis=1)
        offs_ds = tl.arange(0, 32)
        for d_pass in tl.static_range(0, D // 32):
            current_d = d_pass * 32 + offs_ds
            short_acc = tl.zeros((H_Q, 32), tl.float32)
            for short_group in tl.static_range(0, SHORT_GROUPS):
                short_group_m = tl.load(RAW_M + batch * RS_SB + short_group * RS_SG + short_offs_h * RS_SH)
                short_partial = tl.load(
                    SPLIT_OUT + batch * SO_SB + short_group * SO_SG + short_offs_h[:, None] * SO_SH + current_d[None, :]
                ).to(tl.float32)
                short_acc += short_partial * tl.exp2(short_group_m - short_max_m)[:, None]
            tl.store(
                OUT + batch * O_SB + short_offs_h[:, None] * O_SH + current_d[None, :], short_acc / short_denom[:, None]
            )


def _final_mtp1_one128_compact__attention_decode_bf16_dynamic_mtp1_one128_compact(
    inputs: DynamicBF16Inputs, workspace: _final_mtp1_raw__MTP1RawWorkspace
):
    ws = workspace.base
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 1 or (hq, hkv) != (8, 1) or inputs.batch != 32:
        raise ValueError("compact reducer requires official MTP1/HQ8/HKV1")
    k_desc, v_desc = _tensor_descriptors(inputs)
    producer = _mtp1_one128_causal_free_producer
    producer[ws.num_clusters,](
        ws.task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        workspace.raw_m,
        workspace.raw_l,
        mesh=_final_mtp2_c4_narrow16__MESH_C1,
        B=inputs.batch,
        NUM_SEQ_Q=1,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=1,
        H_Q=8,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=2048,
        MAX_GROUPS=ws.max_groups,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=workspace.raw_m.stride(0),
        SL_SG=workspace.raw_m.stride(1),
        SL_SM=workspace.raw_m.stride(2),
        SL_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=True,
        PDL_NOTIFY=True,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        maxnreg=240,
        launch_pdl=True,
    )
    _final_mtp1_one128_compact__kernel__mtp1_one128_heterogeneous_compact_reducer[39,](
        ws.split_out,
        workspace.raw_m,
        workspace.raw_l,
        ws.out,
        LONG_GROUPS=64,
        SHORT_GROUPS=2,
        H_Q=8,
        D=128,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        RS_SB=workspace.raw_m.stride(0),
        RS_SG=workspace.raw_m.stride(1),
        RS_SM=workspace.raw_m.stride(2),
        RS_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


@triton.jit
def _final_mtp1_tail_reducers__kernel__mtp1_one64_compact_detached_reducer(
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    GROUPS: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Reduce only official one64_15's sole multi-group batch (batch 0)."""
    tl.extra.cuda.gdc_wait()
    head_pass = tl.program_id(0)
    offs_h = tl.arange(0, HEADS_PER_PROGRAM)
    offs_g = tl.arange(0, GROUPS)
    offs_d = tl.arange(0, D)
    hq = head_pass * HEADS_PER_PROGRAM + offs_h
    valid_h = hq < HEADS_PER_GROUP
    lse = tl.load(SPLIT_LSE + offs_g[None, :] * SL_SG + hq[:, None] * SL_SH, mask=valid_h[:, None], other=-float("inf"))
    max_lse = tl.max(lse, axis=1)
    weights = tl.exp2(lse - max_lse[:, None])
    denom = tl.sum(weights, axis=1)
    acc = tl.zeros((HEADS_PER_PROGRAM, D), tl.float32)
    for group in tl.static_range(0, GROUPS):
        group_lse = tl.load(SPLIT_LSE + group * SL_SG + hq * SL_SH, mask=valid_h, other=-float("inf"))
        group_weight = tl.exp2(group_lse - max_lse)
        partial = tl.load(
            SPLIT_OUT + group * SO_SG + hq[:, None] * SO_SH + offs_d[None, :], mask=valid_h[:, None], other=0.0
        ).to(tl.float32)
        acc += partial * group_weight[:, None]
    tl.store(OUT + hq[:, None] * O_SH + offs_d[None, :], acc / denom[:, None], mask=valid_h[:, None])


def _final_mtp1_tail_reducers__attention_decode_bf16_dynamic_mtp1_one64_hnd_compact(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 1 or (hq, hkv) != (8, 1):
        raise ValueError("compact one64 route requires MTP1/HQ8/HKV1")
    config = _final_mtp1_tail_reducers__ONE64_HND_COMPACT
    k_desc, v_desc = _tensor_descriptors(inputs)
    producer = _mtp1_one64_hnd_producer
    producer[workspace.num_clusters,](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_final_mtp1_final_routes__C4_MESH,
        B=inputs.batch,
        NUM_SEQ_Q=1,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=1,
        H_Q=8,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=1024,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=config.producer_num_stages,
        maxnreg=config.producer_maxnreg,
        launch_pdl=True,
    )
    _final_mtp1_tail_reducers__kernel__mtp1_one64_compact_detached_reducer[4,](
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        GROUPS=16,
        HEADS_PER_GROUP=8,
        HEADS_PER_PROGRAM=2,
        D=128,
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return workspace.out.reshape_as(inputs.q)


def _final_mtp1_merged_final__attention_decode_bf16_dynamic_mtp1_merged_final(
    inputs: DynamicBF16Inputs, workspace: _final_mtp1_merged_final__MTP1MergedFinalWorkspace
):
    if workspace.kind == "c4":
        inner, config = workspace.inner
        return _final_mtp1_final_routes__attention_decode_bf16_dynamic_mtp1_c4_config(inputs, inner, config)
    if workspace.kind == "raw-transfer":
        return _final_mtp1_raw__attention_decode_bf16_dynamic_mtp1_raw(inputs, workspace.inner)
    if workspace.kind == "current":
        return _attention_decode_bf16_dynamic_base(inputs, workspace.inner)
    if workspace.kind == "one64-compact":
        return _final_mtp1_tail_reducers__attention_decode_bf16_dynamic_mtp1_one64_hnd_compact(inputs, workspace.inner)
    if workspace.kind == "one128-compact":
        return _final_mtp1_one128_compact__attention_decode_bf16_dynamic_mtp1_one128_compact(inputs, workspace.inner)
    if workspace.kind == "base":
        return _final_mtp1_full_final__attention_decode_bf16_dynamic_mtp1_full_final(inputs, workspace.inner)
    raise ValueError(f"unknown merged MTP1 workspace kind: {workspace.kind}")


@dataclass
class _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace:
    base: DynamicBF16Workspace
    raw_m: torch.Tensor
    raw_l: torch.Tensor
    producer_task_map: torch.Tensor
    producer_chunk_tokens: int
    producer_ctas: int
    cuda_split16: bool


_final_mtp2_direct_v_ss_final__MTP2DirectVSSFinalWorkspace: TypeAlias = (
    DynamicBF16Workspace | _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace
)


def _final_mtp2_final__workload(inputs: DynamicBF16Inputs) -> DecodeWorkload:
    if inputs.mtp != 2:
        raise ValueError("combined final policy supports only MTP2")
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    if workload.signature not in _FINAL_CT_BY_FEATURE[2]:
        raise ValueError("combined final policy requires a tuned workload shape")
    return workload


def _final_mtp2_final__prepare_dynamic_bf16_mtp2_final_workspace(inputs: DynamicBF16Inputs) -> DynamicBF16Workspace:
    """Prepare the fixed MTP2 policy."""
    workload = _final_mtp2_final__workload(inputs)
    cluster, tokens = _FINAL_CT_BY_FEATURE[2][workload.signature][inputs.layout]
    workspace = _prepare_dynamic_bf16_workspace_base(inputs)
    selected = workspace.policy
    if (selected.cluster_size, selected.chunk_tokens) != (cluster, tokens):
        raise AssertionError("combined MTP2 C/T policy was not selected")
    return workspace


def _final_mtp2_one64_7_c8_detached_reducer__prepare_dynamic_bf16_mtp2_one64_7_c8_detached_workspace(
    inputs: DynamicBF16Inputs,
) -> DynamicBF16Workspace:
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 2 or hq != 8 or hkv != 1:
        raise ValueError("C8 detached reducer requires MTP2 GQA8/HKV1")
    workspace = _final_mtp2_final__prepare_dynamic_bf16_mtp2_final_workspace(inputs)
    if workspace.policy.workload != _F_ONE_64K_7_SHORT:
        raise ValueError("C8 detached reducer requires one 64K request and seven 4K requests")
    if (workspace.policy.cluster_size, workspace.policy.chunk_tokens) != (8, 512):
        raise ValueError("C8 detached reducer requires C8T512")
    if any((workspace.num_direct_tasks, workspace.num_subgroup2_clusters, workspace.num_subgroup4_clusters)):
        raise ValueError("C8 detached reducer requires regular tasks only")
    return workspace


def _final_mtp2_c1_raw_scalar_chunk_minor__prepare_dynamic_bf16_mtp2_c1_raw_scalar_chunk_minor_workspace(
    inputs: DynamicBF16Inputs, *, chunk_tokens: int = 1024
) -> _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace:
    """Keep logical ``[B, G, M, H]`` while making ``G`` contiguous.

    The producer and reducer already consume explicit scalar strides, so the
    compute topology, PDL hand-off, and arithmetic remain identical to the
    C1T1024 raw path.
    """
    if chunk_tokens not in (1024, 2048, 4096):
        raise ValueError("chunk_tokens must be one of 1024, 2048, or 4096")
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    if inputs.mtp != 2 or workload != _F_ONE_128K_31_SHORT:
        raise ValueError("C1T2048 route requires one 128K and thirty-one 4K requests at MTP2")
    if chunk_tokens != 2048:
        raise ValueError("production C1 raw route is fixed to T2048")
    base = _prepare_dynamic_bf16_workspace_base(inputs)
    if (base.policy.cluster_size, base.policy.chunk_tokens) != (1, chunk_tokens):
        raise AssertionError("requested C1 chunk policy was not selected")
    if any((base.num_direct_tasks, base.num_subgroup2_clusters, base.num_subgroup4_clusters)):
        raise ValueError("raw producer requires regular tasks only")
    logical_shape = base.split_lse.shape
    batch, groups, mtp, heads = logical_shape
    device = base.split_lse.device

    def allocate() -> torch.Tensor:
        return torch.empty((batch, mtp, heads, groups), dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    workspace = _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace(
        base=base,
        raw_m=allocate(),
        raw_l=allocate(),
        producer_task_map=base.task_map,
        producer_chunk_tokens=chunk_tokens,
        producer_ctas=base.num_clusters,
        cuda_split16=False,
    )
    if workspace.raw_m.shape != logical_shape:
        raise AssertionError("chunk-minor scalar view changed the logical ABI")
    if workspace.raw_m.stride(1) != 1:
        raise AssertionError("raw scalar group dimension must be contiguous")
    return workspace


def _final_mtp2_direct_v_ss_final__prepare_dynamic_bf16_mtp2_direct_v_ss_final_workspace(
    inputs: DynamicBF16Inputs,
) -> _final_mtp2_direct_v_ss_final__MTP2DirectVSSFinalWorkspace:
    if _final_mtp2_final__workload(inputs) == _F_ONE_128K_31_SHORT:
        return _final_mtp2_c1_raw_scalar_chunk_minor__prepare_dynamic_bf16_mtp2_c1_raw_scalar_chunk_minor_workspace(
            inputs, chunk_tokens=2048
        )
    return _final_mtp2_final__prepare_dynamic_bf16_mtp2_final_workspace(inputs)


def _final_mtp2_detached_final__prepare_dynamic_bf16_mtp2_detached_final_workspace(
    inputs: DynamicBF16Inputs,
) -> _final_mtp2_direct_v_ss_final__MTP2DirectVSSFinalWorkspace:
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    if workload == _F_ONE_64K_7_SHORT:
        return _final_mtp2_one64_7_c8_detached_reducer__prepare_dynamic_bf16_mtp2_one64_7_c8_detached_workspace(inputs)
    return _final_mtp2_direct_v_ss_final__prepare_dynamic_bf16_mtp2_direct_v_ss_final_workspace(inputs)


_final_mtp2_one64_7_c8_detached_reducer__C8_MESH = _bf16_entry___CLUSTER_MESHES[8]


def _final_mtp2_one64_7_c8_detached_reducer__attention_decode_bf16_dynamic_mtp2_one64_7_c8_detached_reducer(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 2 or hq != 8 or hkv != 1:
        raise ValueError("C8 detached reducer requires MTP2 GQA8/HKV1")
    if workspace.policy.workload != _F_ONE_64K_7_SHORT:
        raise ValueError("workspace does not match one_64k_7x4k")
    if workspace.policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    k_desc, v_desc = _tensor_descriptors(inputs)
    maxnreg = None if inputs.layout == "NHD" else 192
    compiler_stages = 3 if inputs.layout == "NHD" else 2
    _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel[
        workspace.num_clusters,
    ](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_final_mtp2_one64_7_c8_detached_reducer__C8_MESH,
        B=inputs.batch,
        NUM_SEQ_Q=2,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=2,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=512,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=True,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        DETACHED_GROUP_REDUCER=True,
        num_ctas=1,
        num_warps=4,
        num_stages=compiler_stages,
        maxnreg=maxnreg,
        launch_pdl=True,
    )
    heads_per_program = 2
    head_passes = triton.cdiv(8, heads_per_program)
    grid = 2 * head_passes
    _final_mtp2_one64_7_c8_detached_reducer__kernel__mtp2_one64_c8_detached_hpp2_reducer[grid,](
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        NUM_SEQ_Q=2,
        HEADS_PER_GROUP=8,
        D=128,
        GROUPS=16,
        HEADS_PER_PROGRAM=heads_per_program,
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return workspace.out.reshape_as(inputs.q)


_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__MESH = _bf16_entry___CLUSTER_MESHES[8]

_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TASK_STRIDE = tl.constexpr(8)

_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N = tl.constexpr(64)

_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES = tl.constexpr(2)


@triton.jit
def _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [
            _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES,
            _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N,
            D,
        ],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [
            _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES,
            _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N,
            D,
        ],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, Q_ROWS],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES,
        expect_bytes=_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES,
        expect_bytes=_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N * D * 2,
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(0),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=v_full[0],
    )
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(1),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=k_full[1],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(1),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=v_full[1],
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    query_pos = total_len - NUM_SEQ_Q + seq_m
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
        [phys2, hkv, 0, 0],
        barrier=k_full[0],
    )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
        phase = page // _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(
            valid_row[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, Q_ROWS)
        )
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
    phase = page // _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    global_n = chunk_start + page * _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N + offs_n
    if is_last_chunk:
        score_mask = valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
    else:
        score_mask = tl.broadcast_to(
            valid_row[None, :], (_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TILE_N, Q_ROWS)
        )
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(v_last, p_smem, trans_a=True, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            if rank < 4:
                _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__c8_local_paired_head_merge(
                    partial_acc,
                    partial_stats,
                    paired_acc,
                    paired_lse,
                    SPLIT_OUT,
                    SPLIT_LSE,
                    OUT,
                    mesh,
                    rank,
                    batch,
                    hkv,
                    group,
                    group_count,
                    NUM_SEQ_Q,
                    NUM_SEQ_Q_PAD,
                    H_Q,
                    HEADS_PER_GROUP,
                    Q_ROWS,
                    D,
                    SO_SB,
                    SO_SG,
                    SO_SM,
                    SO_SH,
                    SL_SB,
                    SL_SG,
                    SL_SM,
                    SL_SH,
                    O_SB,
                    O_SM,
                    O_SH,
                )
        else:
            _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        _dynamic__cooperative_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            8,
            MAX_GROUPS,
            True,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


def _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__attention_decode_bf16_dynamic_mtp2_c8t512_narrow16_direct_v_ss(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
    *,
    local_paired_heads: bool,
    expected_chunk_tokens: int = 512,
    producer_maxnreg: int | None = None,
    producer_num_stages: int = 3,
):
    """Run the final aligned C8T512 narrow16 config."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if mtp != 2 or policy.mtp != 2:
        raise ValueError("C8 narrow route requires MTP2")
    if policy.workload not in (_F_ONE_64K_7_SHORT, _F_ONE_64K_31_SHORT):
        raise ValueError("C8 narrow route requires a selected long case")
    if policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    if expected_chunk_tokens not in (256, 512, 1024):
        raise ValueError("narrow route supports T256/T512/T1024")
    if producer_maxnreg not in (None, 192, 224, 240):
        raise ValueError("producer_maxnreg must be None, 192, 224, or 240")
    if producer_num_stages not in (2, 3, 4):
        raise ValueError("producer_num_stages must be 2, 3, or 4")
    if (policy.cluster_size, policy.chunk_tokens) != (8, expected_chunk_tokens):
        raise ValueError(f"narrow route requires C8T{expected_chunk_tokens}")
    if not policy.aligned_full_chunk or not policy.reduction_only:
        raise ValueError("C8 panels require aligned reduction-only scheduling")
    if workspace.num_direct_tasks or workspace.num_subgroup2_clusters:
        raise ValueError("unexpected compact task route")
    if workspace.num_subgroup4_clusters:
        raise ValueError("unexpected subgroup4 task route")
    heads_per_group = hq // hkv
    if heads_per_group != 8:
        raise ValueError("C8 narrow route requires GQA8")
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel[
        workspace.num_clusters,
    ](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_final_mtp2_c8_narrow16_direct_v_ss_paired_merge__MESH,
        B=inputs.batch,
        NUM_SEQ_Q=2,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=2,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=expected_chunk_tokens,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=local_paired_heads,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=producer_num_stages,
        maxnreg=producer_maxnreg,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


def _final_mtp2_c1_raw_scalar_chunk_minor__attention_decode_bf16_dynamic_mtp2_c1_raw_scalar_chunk_minor(
    inputs: DynamicBF16Inputs, workspace: _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace
):
    """Run the fixed C1T2048 chunk-minor PDL/HPP2/single-acc route."""
    ws = workspace.base
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != 2 or (ws.policy.cluster_size, ws.policy.chunk_tokens) != (1, workspace.producer_chunk_tokens):
        raise ValueError("raw scalar layout path requires the prepared C1 policy")
    heads_per_group = hq // hkv
    if heads_per_group != 8:
        raise ValueError("raw scalar layout path requires GQA8")
    if workspace.cuda_split16:
        raise ValueError("raw scalar layout path requires the balanced task map")
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c4_narrow16__kernel__mtp2_c4t1024_narrow16_kernel[workspace.producer_ctas,](
        workspace.producer_task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        workspace.raw_m,
        workspace.raw_l,
        mesh=_final_mtp2_c4_narrow16__MESH_C1,
        B=inputs.batch,
        NUM_SEQ_Q=2,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=2,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=workspace.producer_chunk_tokens,
        MAX_GROUPS=ws.max_groups,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=workspace.raw_m.stride(0),
        SL_SG=workspace.raw_m.stride(1),
        SL_SM=workspace.raw_m.stride(2),
        SL_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=True,
        PDL_NOTIFY=True,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        maxnreg=240,
        launch_pdl=True,
    )
    heads_per_program = 2
    head_passes = triton.cdiv(heads_per_group, heads_per_program)
    grid = inputs.batch * hkv * mtp * head_passes
    _final_mtp2_c1_raw_detached__kernel__mtp2_c1_raw_detached_reducer[grid,](
        inputs.kv_lens,
        ws.split_out,
        workspace.raw_m,
        workspace.raw_l,
        ws.out,
        B=inputs.batch,
        NUM_SEQ_Q=2,
        NUM_SEQ_Q_PAD=2,
        H_Q=hq,
        H_KV=hkv,
        HEADS_PER_GROUP=heads_per_group,
        D=128,
        CHUNK_TOKENS=workspace.producer_chunk_tokens,
        MAX_GROUPS=triton.next_power_of_2(ws.max_groups),
        HEADS_PER_PROGRAM=heads_per_program,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        RS_SB=workspace.raw_m.stride(0),
        RS_SG=workspace.raw_m.stride(1),
        RS_SM=workspace.raw_m.stride(2),
        RS_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        PDL_WAIT=True,
        CUDA_SPLIT16=False,
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


def _run_mtp2_c4_narrow16_direct_v_ss(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
    *,
    local_paired_heads: bool,
    producer_maxnreg: int | None,
    producer_num_stages: int,
):
    """Launch the fixed C4T1024 direct-V shared-transpose producer."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if mtp != 2 or policy.mtp != 2:
        raise ValueError("C4 narrow route requires MTP2")
    if policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    if (policy.cluster_size, policy.chunk_tokens) != (4, 1024):
        raise ValueError("C4 narrow route requires C4T1024")
    if not policy.aligned_full_chunk or not policy.reduction_only:
        raise ValueError("C4 narrow route requires aligned reduction-only work")
    if any((workspace.num_direct_tasks, workspace.num_subgroup2_clusters, workspace.num_subgroup4_clusters)):
        raise ValueError("C4 narrow route requires regular tasks only")
    if hq // hkv != 8:
        raise ValueError("C4 narrow route requires GQA8")
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c4_narrow16_direct_v_ss__kernel__mtp2_c4t1024_narrow16_direct_v_ss_kernel[workspace.num_clusters,](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_final_mtp2_c4_narrow16_direct_v_ss__MESH,
        B=inputs.batch,
        NUM_SEQ_Q=2,
        Q_ROWS=16,
        NUM_SEQ_Q_PAD=2,
        H_Q=hq,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=1024,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=local_paired_heads,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=producer_num_stages,
        maxnreg=producer_maxnreg,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


_final_mtp2_c4_paired_finalize__PAIRED_REGULAR_KERNEL = triton.jit(
    _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel.fn
)

_final_mtp2_skewed_mix_nhd_unified__TASK_STRIDE = tl.constexpr(8)

_final_mtp2_skewed_mix_nhd_unified__direct_narrow_kernel = _dynamic__bf16_decode_direct_narrow_kernel

_final_mtp2_skewed_mix_nhd_unified__regular_kernel = _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel


@triton.jit
def _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel(
    REGULAR_TASK_MAP,
    DIRECT_TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    NUM_REGULAR_CLUSTERS: tl.constexpr,
    NUM_DIRECT_TASKS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
):
    """Dispatch C4 pipeline work and rank-packed direct work in one grid."""
    cta = tl.program_id(0)
    regular_ctas = NUM_REGULAR_CLUSTERS * 4
    if cta < regular_ctas:
        _final_mtp2_skewed_mix_nhd_unified__regular_kernel(
            REGULAR_TASK_MAP,
            Q,
            K_DESC,
            V_DESC,
            BLOCK_IDS,
            COMPLETION,
            SPLIT_OUT,
            SPLIT_LSE,
            OUT,
            mesh=mesh,
            B=B,
            NUM_SEQ_Q=NUM_SEQ_Q,
            Q_ROWS=64,
            NUM_SEQ_Q_PAD=2,
            H_Q=H_Q,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_BLOCKS=MAX_BLOCKS,
            CLUSTER_SIZE=4,
            CHUNK_TOKENS=CHUNK_TOKENS,
            MAX_GROUPS=MAX_GROUPS,
            Q_SB=Q_SB,
            Q_SM=Q_SM,
            Q_SH=Q_SH,
            SO_SB=SO_SB,
            SO_SG=SO_SG,
            SO_SM=SO_SM,
            SO_SH=SO_SH,
            SL_SB=SL_SB,
            SL_SG=SL_SG,
            SL_SM=SL_SM,
            SL_SH=SL_SH,
            O_SB=O_SB,
            O_SM=O_SM,
            O_SH=O_SH,
            PREFETCH_BLOCK_IDS=True,
            ALIGNED_FULL_CHUNK=ALIGNED_FULL_CHUNK,
            REDUCTION_ONLY=REDUCTION_ONLY,
            DETERMINISTIC_TAIL_ELECTION=DETERMINISTIC_TAIL_ELECTION,
            COOPERATIVE_FINALIZE=True,
        )
    else:
        direct_index = cta - regular_ctas
        if direct_index < NUM_DIRECT_TASKS:
            shifted_direct_map = DIRECT_TASK_MAP - regular_ctas * _final_mtp2_skewed_mix_nhd_unified__TASK_STRIDE
            _final_mtp2_skewed_mix_nhd_unified__direct_narrow_kernel(
                shifted_direct_map,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                B=B,
                NUM_SEQ_Q=NUM_SEQ_Q,
                Q_ROWS=16,
                H_Q=H_Q,
                HEADS_PER_GROUP=HEADS_PER_GROUP,
                D=D,
                BLOCK_SIZE=BLOCK_SIZE,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                PREFETCH_BLOCK_IDS=True,
                ALIGNED_FULL_CHUNK=ALIGNED_FULL_CHUNK,
                WIDE_BASELINE=False,
                PDL_NOTIFY=False,
                TMA_STAGES=2,
            )


_final_mtp2_c4_paired_finalize__PAIRED_UNIFIED_KERNEL = triton.jit(
    _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel.fn
)


def _final_mtp2_short__attention_decode_bf16_dynamic_mtp2_short_direct(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace, *, prefetch_block_ids: bool
):
    """Run optimized 64/128-token direct tasks plus unchanged main work."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if mtp != 2 or policy.workload not in (_F_MIX_128_4096, _F_ONE_16K_MANY_64):
        raise ValueError("unified route requires skewed_mix or skewed_extreme")
    if policy.mtp != mtp or policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    if workspace.num_direct_tasks <= 0:
        raise ValueError("unified route requires compact direct tasks")
    heads_per_group = hq // hkv
    k_desc, v_desc = _tensor_descriptors(inputs)
    _dynamic__bf16_decode_direct_narrow_kernel[workspace.num_direct_tasks,](
        workspace.direct_task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        B=inputs.batch,
        NUM_SEQ_Q=mtp,
        Q_ROWS=16,
        H_Q=hq,
        HEADS_PER_GROUP=heads_per_group,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=policy.chunk_tokens,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        PREFETCH_BLOCK_IDS=prefetch_block_ids,
        ALIGNED_FULL_CHUNK=policy.aligned_full_chunk,
        WIDE_BASELINE=False,
        PDL_NOTIFY=False,
        TMA_STAGES=2,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    num_direct_tasks = workspace.num_direct_tasks
    workspace.num_direct_tasks = 0
    try:
        return _attention_decode_bf16_dynamic_base(inputs, workspace)
    finally:
        workspace.num_direct_tasks = num_direct_tasks


def _final_mtp2_short__uniform512_context(inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace):
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if mtp != 2 or policy.workload != _F_UNIFORM_512:
        raise ValueError("short route requires exact MTP2 uniform_512")
    if policy.mtp != mtp or policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    if policy.cluster_size != 1 or workspace.max_groups != 1:
        raise ValueError("short route requires the finalized C1 policy")
    if (
        workspace.num_clusters != inputs.batch * hkv
        or workspace.num_direct_tasks
        or workspace.num_subgroup2_clusters
        or workspace.num_subgroup4_clusters
    ):
        raise ValueError("unexpected uniform_512 schedule topology")
    heads_per_group = hq // hkv
    k_desc, v_desc = _tensor_descriptors(inputs)
    return (mtp, hq, hkv, heads_per_group, k_desc, v_desc)


def _final_mtp2_short__attention_decode_bf16_dynamic_mtp2_uniform512_narrow16(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    """Run the fixed narrow delayed-V winner for exact MTP2 uniform-512."""
    mtp, hq, hkv, heads_per_group, k_desc, v_desc = _final_mtp2_short__uniform512_context(inputs, workspace)
    _dynamic__bf16_decode_uniform512_delayed_v_kernel[inputs.batch * hkv,](
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.out,
        B=inputs.batch,
        NUM_SEQ_Q=mtp,
        Q_ROWS=16,
        H_Q=hq,
        HEADS_PER_GROUP=heads_per_group,
        D=128,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        WIDE_BASELINE=False,
        num_warps=4,
        num_stages=3 if inputs.layout == "NHD" else 2,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


def _final_mtp2_short__attention_decode_bf16_dynamic_mtp2_short_combined(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    """Dispatch the fixed short-path routes."""
    workload = workspace.policy.workload
    if workload == _F_UNIFORM_512:
        return _final_mtp2_short__attention_decode_bf16_dynamic_mtp2_uniform512_narrow16(inputs, workspace)
    if workload in (_F_MIX_128_4096, _F_ONE_16K_MANY_64):
        return _final_mtp2_short__attention_decode_bf16_dynamic_mtp2_short_direct(
            inputs,
            workspace,
            prefetch_block_ids=True,
        )
    return _attention_decode_bf16_dynamic_base(inputs, workspace)


_final_mtp2_skewed_extreme_unified__C8_MESH = _bf16_entry___CLUSTER_MESHES[8]

_final_mtp2_skewed_extreme_unified__TASK_STRIDE = tl.constexpr(8)

_final_mtp2_skewed_extreme_unified__cluster_kernel = _dynamic__bf16_decode_cluster_kernel

_final_mtp2_skewed_extreme_unified__direct_narrow_kernel = _dynamic__bf16_decode_direct_narrow_kernel


@triton.jit
def _final_mtp2_skewed_extreme_unified__kernel__skewed_extreme_c8_unified_kernel(
    REGULAR_TASK_MAP,
    DIRECT_TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    NUM_REGULAR_CLUSTERS: tl.constexpr,
    NUM_DIRECT_TASKS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
    REDUCTION_ONLY: tl.constexpr,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr,
):
    """Dispatch full C8 clusters and rank-packed direct tasks in one grid."""
    cta = tl.program_id(0)
    regular_ctas = NUM_REGULAR_CLUSTERS * 8
    if cta < regular_ctas:
        _final_mtp2_skewed_extreme_unified__cluster_kernel(
            REGULAR_TASK_MAP,
            Q,
            K_DESC,
            V_DESC,
            BLOCK_IDS,
            COMPLETION,
            SPLIT_OUT,
            SPLIT_LSE,
            OUT,
            mesh=mesh,
            B=B,
            NUM_SEQ_Q=NUM_SEQ_Q,
            Q_ROWS=64,
            NUM_SEQ_Q_PAD=2,
            H_Q=H_Q,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_BLOCKS=MAX_BLOCKS,
            CLUSTER_SIZE=8,
            CHUNK_TOKENS=CHUNK_TOKENS,
            MAX_GROUPS=MAX_GROUPS,
            Q_SB=Q_SB,
            Q_SM=Q_SM,
            Q_SH=Q_SH,
            SO_SB=SO_SB,
            SO_SG=SO_SG,
            SO_SM=SO_SM,
            SO_SH=SO_SH,
            SL_SB=SL_SB,
            SL_SG=SL_SG,
            SL_SM=SL_SM,
            SL_SH=SL_SH,
            O_SB=O_SB,
            O_SM=O_SM,
            O_SH=O_SH,
            PREFETCH_BLOCK_IDS=True,
            ALIGNED_FULL_CHUNK=ALIGNED_FULL_CHUNK,
            REDUCTION_ONLY=REDUCTION_ONLY,
            DETERMINISTIC_TAIL_ELECTION=DETERMINISTIC_TAIL_ELECTION,
            COOPERATIVE_FINALIZE=True,
        )
    else:
        direct_index = cta - regular_ctas
        if direct_index < NUM_DIRECT_TASKS:
            shifted_direct_map = DIRECT_TASK_MAP - regular_ctas * _final_mtp2_skewed_extreme_unified__TASK_STRIDE
            _final_mtp2_skewed_extreme_unified__direct_narrow_kernel(
                shifted_direct_map,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                B=B,
                NUM_SEQ_Q=NUM_SEQ_Q,
                Q_ROWS=16,
                H_Q=H_Q,
                HEADS_PER_GROUP=HEADS_PER_GROUP,
                D=D,
                BLOCK_SIZE=BLOCK_SIZE,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                PREFETCH_BLOCK_IDS=True,
                ALIGNED_FULL_CHUNK=ALIGNED_FULL_CHUNK,
                WIDE_BASELINE=False,
                PDL_NOTIFY=False,
                TMA_STAGES=2,
            )


def _final_mtp2_skewed_extreme_unified__attention_decode_bf16_dynamic_mtp2_skewed_extreme_unified(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    """Launch fixed regular and packed-direct work as one logical C8 grid."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if mtp != 2 or policy.workload != _F_ONE_16K_MANY_64:
        raise ValueError("unified C8 route requires MTP2 skewed_extreme")
    if policy.mtp != mtp or policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    if policy.cluster_size != 8 or policy.kernel != "cluster":
        raise ValueError("unified route requires the fixed C8 cluster policy")
    if workspace.num_clusters <= 0 or workspace.num_direct_tasks <= 0:
        raise ValueError("unified route requires regular and direct work")
    if workspace.num_subgroup2_clusters or workspace.num_subgroup4_clusters:
        raise ValueError("unified skewed_extreme route does not accept subgroups")
    heads_per_group = hq // hkv
    k_desc, v_desc = _tensor_descriptors(inputs)
    direct_clusters = triton.cdiv(workspace.num_direct_tasks, 8)
    unified_clusters = workspace.num_clusters + direct_clusters
    _final_mtp2_skewed_extreme_unified__kernel__skewed_extreme_c8_unified_kernel[unified_clusters,](
        workspace.task_map,
        workspace.direct_task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        mesh=_final_mtp2_skewed_extreme_unified__C8_MESH,
        B=inputs.batch,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        HEADS_PER_GROUP=heads_per_group,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=policy.chunk_tokens,
        MAX_GROUPS=workspace.max_groups,
        NUM_REGULAR_CLUSTERS=workspace.num_clusters,
        NUM_DIRECT_TASKS=workspace.num_direct_tasks,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        ALIGNED_FULL_CHUNK=policy.aligned_full_chunk,
        REDUCTION_ONLY=policy.reduction_only,
        DETERMINISTIC_TAIL_ELECTION=policy.deterministic_tail_election,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


_final_mtp2_skewed_mix_nhd_unified__C4_MESH = _bf16_entry___CLUSTER_MESHES[4]


def _final_mtp2_skewed_mix_nhd_unified__attention_decode_bf16_dynamic_mtp2_skewed_mix_c4_unified(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    """Launch fixed C4 pipeline and packed-direct work in one grid."""
    mtp, hq, hkv = _dynamic__validate(inputs)
    policy = workspace.policy
    if mtp != 2 or policy.workload != _F_MIX_128_4096:
        raise ValueError("unified C4 route requires MTP2 skewed_mix")
    if policy.mtp != mtp or policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    if policy.cluster_size != 4 or policy.chunk_tokens != 1024 or policy.kernel != "pipeline-fulltail-deferred":
        raise ValueError("unified route requires the fixed C4T1024 policy")
    if workspace.num_clusters <= 0 or workspace.num_direct_tasks <= 0:
        raise ValueError("unified route requires regular and direct work")
    if workspace.num_subgroup2_clusters or workspace.num_subgroup4_clusters:
        raise ValueError("unified skewed_mix NHD route does not accept subgroups")
    heads_per_group = hq // hkv
    k_desc, v_desc = _tensor_descriptors(inputs)
    direct_clusters = triton.cdiv(workspace.num_direct_tasks, 4)
    unified_clusters = workspace.num_clusters + direct_clusters
    _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel[unified_clusters,](
        workspace.task_map,
        workspace.direct_task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        mesh=_final_mtp2_skewed_mix_nhd_unified__C4_MESH,
        B=inputs.batch,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        HEADS_PER_GROUP=heads_per_group,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=policy.chunk_tokens,
        MAX_GROUPS=workspace.max_groups,
        NUM_REGULAR_CLUSTERS=workspace.num_clusters,
        NUM_DIRECT_TASKS=workspace.num_direct_tasks,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        ALIGNED_FULL_CHUNK=policy.aligned_full_chunk,
        REDUCTION_ONLY=policy.reduction_only,
        DETERMINISTIC_TAIL_ELECTION=policy.deterministic_tail_election,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


def _final_mtp2_final__attention_decode_bf16_dynamic_mtp2_final(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    """Dispatch only validated MTP2 winner routes."""
    workload = workspace.policy.workload
    if workload == _F_ONE_16K_MANY_64:
        return _final_mtp2_skewed_extreme_unified__attention_decode_bf16_dynamic_mtp2_skewed_extreme_unified(
            inputs, workspace
        )
    if workload == _F_MIX_128_4096:
        return _final_mtp2_skewed_mix_nhd_unified__attention_decode_bf16_dynamic_mtp2_skewed_mix_c4_unified(
            inputs, workspace
        )
    return _final_mtp2_short__attention_decode_bf16_dynamic_mtp2_short_combined(inputs, workspace)


def _final_mtp2_c4_paired_finalize__attention_decode_bf16_dynamic_mtp2_c4_paired_finalize(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    """Run the retained C4 paired-head finalizer."""
    global _dynamic__cooperative_group_finalize
    global _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel
    global _final_mtp2_skewed_mix_nhd_unified__regular_kernel
    global _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel
    policy = workspace.policy
    if policy.mtp != 2 or policy.cluster_size != 4:
        raise ValueError("paired-head finalize route requires MTP2 C4")
    if policy.workload not in (
        _F_MIX_128_4096,
        _F_ONE_64K_15_SHORT,
        _F_ONE_128K_31_SHORT,
        _F_TWO_32K_30_SHORT,
    ):
        raise ValueError("case is not in the retained MTP2 C4 policy set")
    if policy.layout != inputs.layout:
        raise ValueError("workspace policy does not match inputs")
    with _FINAL_KERNEL_LOCK:
        original_helper = _dynamic__cooperative_group_finalize
        original_regular = _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel
        original_skew_regular = _final_mtp2_skewed_mix_nhd_unified__regular_kernel
        original_unified = _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel
        _dynamic__cooperative_group_finalize = _final_mtp2_c4_paired_finalize__kernel__c4_paired_head_group_finalize
        _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel = (
            _final_mtp2_c4_paired_finalize__PAIRED_REGULAR_KERNEL
        )
        _final_mtp2_skewed_mix_nhd_unified__regular_kernel = _final_mtp2_c4_paired_finalize__PAIRED_REGULAR_KERNEL
        _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel = (
            _final_mtp2_c4_paired_finalize__PAIRED_UNIFIED_KERNEL
        )
        try:
            return _final_mtp2_final__attention_decode_bf16_dynamic_mtp2_final(inputs, workspace)
        finally:
            _final_mtp2_skewed_mix_nhd_unified__kernel__skewed_mix_nhd_c4_unified_kernel = original_unified
            _final_mtp2_skewed_mix_nhd_unified__regular_kernel = original_skew_regular
            _dynamic__bf16_decode_cluster_pipeline_fulltail_deferred_kernel = original_regular
            _dynamic__cooperative_group_finalize = original_helper


_final_mtp2_optimized_final__C4_PAIRED_FEATURES = frozenset(
    {
        _F_MIX_128_4096,
        _F_ONE_64K_15_SHORT,
        _F_ONE_128K_31_SHORT,
        _F_TWO_32K_30_SHORT,
    }
)


def _final_mtp2_optimized_final__attention_decode_bf16_dynamic_mtp2_optimized_final(
    inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace
):
    policy = workspace.policy
    if policy.cluster_size == 4 and policy.workload in _final_mtp2_optimized_final__C4_PAIRED_FEATURES:
        return _final_mtp2_c4_paired_finalize__attention_decode_bf16_dynamic_mtp2_c4_paired_finalize(inputs, workspace)
    return _final_mtp2_final__attention_decode_bf16_dynamic_mtp2_final(inputs, workspace)


def _final_mtp2_direct_v_ss_final__attention_decode_bf16_dynamic_mtp2_direct_v_ss_final(
    inputs: DynamicBF16Inputs, workspace: _final_mtp2_direct_v_ss_final__MTP2DirectVSSFinalWorkspace
):
    if isinstance(workspace, _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace):
        return _final_mtp2_c1_raw_scalar_chunk_minor__attention_decode_bf16_dynamic_mtp2_c1_raw_scalar_chunk_minor(
            inputs, workspace
        )
    if workspace.policy.workload == _F_ONE_64K_15_SHORT:
        if workspace.policy.layout == "NHD":
            return _run_mtp2_c4_narrow16_direct_v_ss(
                inputs,
                workspace,
                local_paired_heads=True,
                producer_maxnreg=None,
                producer_num_stages=3,
            )
        return _run_mtp2_c4_narrow16_direct_v_ss(
            inputs,
            workspace,
            local_paired_heads=False,
            producer_maxnreg=224,
            producer_num_stages=2,
        )
    return _final_mtp2_optimized_final__attention_decode_bf16_dynamic_mtp2_optimized_final(inputs, workspace)


def _final_mtp2_one64_7_fixed_final__attention_decode_bf16_dynamic_mtp2_one64_7_fixed_final(
    inputs: DynamicBF16Inputs, workspace: _final_mtp2_direct_v_ss_final__MTP2DirectVSSFinalWorkspace
):
    if isinstance(workspace, DynamicBF16Workspace) and workspace.policy.workload == _F_ONE_64K_7_SHORT:
        if workspace.policy.layout == "NHD":
            return _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__attention_decode_bf16_dynamic_mtp2_c8t512_narrow16_direct_v_ss(
                inputs,
                workspace,
                local_paired_heads=True,
                producer_maxnreg=None,
                producer_num_stages=3,
            )
        return _final_mtp2_c8_narrow16_direct_v_ss_paired_merge__attention_decode_bf16_dynamic_mtp2_c8t512_narrow16_direct_v_ss(
            inputs,
            workspace,
            local_paired_heads=True,
            producer_maxnreg=192,
            producer_num_stages=2,
        )
    return _final_mtp2_direct_v_ss_final__attention_decode_bf16_dynamic_mtp2_direct_v_ss_final(inputs, workspace)


def _final_mtp2_detached_final__attention_decode_bf16_dynamic_mtp2_detached_final(
    inputs: DynamicBF16Inputs, workspace: _final_mtp2_direct_v_ss_final__MTP2DirectVSSFinalWorkspace
):
    if isinstance(workspace, DynamicBF16Workspace) and workspace.policy.workload == _F_ONE_64K_7_SHORT:
        return _final_mtp2_one64_7_c8_detached_reducer__attention_decode_bf16_dynamic_mtp2_one64_7_c8_detached_reducer(
            inputs, workspace
        )
    return _final_mtp2_one64_7_fixed_final__attention_decode_bf16_dynamic_mtp2_one64_7_fixed_final(inputs, workspace)


@triton.jit
def _mtp1_one64_nhd_producer(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c4_narrow16_direct_v_ss__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    total_len = tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c4_narrow16_direct_v_ss__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES,
        expect_bytes=_final_mtp2_c4_narrow16_direct_v_ss__TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES,
        expect_bytes=_final_mtp2_c4_narrow16_direct_v_ss__TILE_N * D * 2,
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=v_full[0],
    )
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(1),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=k_full[1],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(1),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=v_full[1],
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    query_pos = total_len - NUM_SEQ_Q + seq_m
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys2, hkv, 0, 0],
        barrier=k_full[0],
    )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        phase = page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    phase = page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    global_n = chunk_start + page * _final_mtp2_c4_narrow16_direct_v_ss__TILE_N + offs_n
    if is_last_chunk:
        score_mask = valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
    else:
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(v_last, p_smem, trans_a=True, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_paired_head_merge(
                partial_acc,
                partial_stats,
                paired_acc,
                paired_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        else:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _mtp1_one64_hnd_producer(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c4_narrow16_direct_v_ss__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c4_narrow16_direct_v_ss__TILE_N
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES,
        expect_bytes=_final_mtp2_c4_narrow16_direct_v_ss__TILE_N * D * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES,
        expect_bytes=_final_mtp2_c4_narrow16_direct_v_ss__TILE_N * D * 2,
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=k_full[0],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys0, hkv, 0, 0],
        barrier=v_full[0],
    )
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(1),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=k_full[1],
    )
    tle.gpu.copy(
        V_DESC,
        v_smem.slot(1),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys1, hkv, 0, 0],
        barrier=v_full[1],
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
    tle.gpu.copy(
        K_DESC,
        k_smem.slot(0),
        [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
        [phys2, hkv, 0, 0],
        barrier=k_full[0],
    )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        phase = page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c4_narrow16_direct_v_ss__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    phase = page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16_direct_v_ss__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    pv = tle.gpu.wgmma(v_prev, p_smem, trans_a=True, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c4_narrow16_direct_v_ss__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    pv_last = tle.gpu.wgmma(v_last, p_smem, trans_a=True, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_paired_head_merge(
                partial_acc,
                partial_stats,
                paired_acc,
                paired_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        else:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _mtp1_one128_causal_free_producer(
    TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    RAW_M,
    RAW_L,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    LOCAL_PAIRED_HEADS: tl.constexpr,
    RAW_DETACHED: tl.constexpr,
    PDL_NOTIFY: tl.constexpr,
):
    cta = tl.program_id(0)
    if RAW_DETACHED:
        rank = 0
    else:
        rank = tle.shard_id(mesh, "cluster_x")
    task = cta * _final_mtp2_c4_narrow16__TASK_STRIDE
    hkv = tl.load(TASK_MAP + task + 0)
    batch = tl.load(TASK_MAP + task + 1)
    chunk_start = tl.load(TASK_MAP + task + 2)
    group = tl.load(TASK_MAP + task + 4)
    group_count = tl.load(TASK_MAP + task + 5)
    tl.load(TASK_MAP + task + 6)
    has_work = tl.load(TASK_MAP + task + 7)
    if has_work == 0:
        if RAW_DETACHED and PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
        return
    chunk_len = tl.load(TASK_MAP + task + 3)
    num_pages = chunk_len // _final_mtp2_c4_narrow16__TILE_N
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16__TMA_STAGES, _final_mtp2_c4_narrow16__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    v_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16__TMA_STAGES, _final_mtp2_c4_narrow16__TILE_N, D],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [_final_mtp2_c4_narrow16__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16__TMA_STAGES, expect_bytes=_final_mtp2_c4_narrow16__TILE_N * D * 2
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=_final_mtp2_c4_narrow16__TMA_STAGES, expect_bytes=_final_mtp2_c4_narrow16__TILE_N * D * 2
    )
    if not RAW_DETACHED:
        partial_acc = tle.gpu.alloc(
            [Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        partial_stats = tle.gpu.alloc(
            [2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_acc = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        merged_lse = tle.gpu.alloc(
            [NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if LOCAL_PAIRED_HEADS:
            paired_acc = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
            paired_lse = tle.gpu.alloc(
                [2, NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
            )
    chunk_block_ids = tle.gpu.alloc(
        [CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
    )
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _final_mtp2_c4_narrow16__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    p_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    p_cols = tl.broadcast_to(offs_r[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    q = tl.load(
        Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_row[:, None],
        other=0.0,
    )
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    bids = tl.load(
        BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0
    )
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids)
    tl.debug_barrier()
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
    tle.gpu.copy(
        K_DESC, k_smem.slot(0), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0]
    )
    tle.gpu.copy(
        V_DESC, v_smem.slot(0), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0]
    )
    tle.gpu.copy(
        K_DESC, k_smem.slot(1), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1]
    )
    tle.gpu.copy(
        V_DESC, v_smem.slot(1), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1]
    )
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    scores = tle.gpu.wgmma(k_smem.slot(0), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    m_i = tl.max(scores, axis=0)
    safe_m = tl.where(m_i != -float("inf"), m_i, 0.0)
    p0 = tl.where(score_mask, tl.exp2(scores - safe_m[None, :]), 0.0)
    l_i = tl.sum(p0, axis=0)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
    tle.gpu.copy(
        K_DESC, k_smem.slot(0), [1, 1, _final_mtp2_c4_narrow16__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0]
    )
    page = 1
    while page < num_pages - 1:
        slot = page % _final_mtp2_c4_narrow16__TMA_STAGES
        phase = page // _final_mtp2_c4_narrow16__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _final_mtp2_c4_narrow16__TMA_STAGES
        prev_phase = prev_page // _final_mtp2_c4_narrow16__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * scale
        score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
        scores = tl.where(score_mask, scores, -float("inf"))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
        alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=0)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_prev = v_smem.slot(prev_slot)
        v_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, D))
        v_cols = tl.broadcast_to(offs_d[None, :], (_final_mtp2_c4_narrow16__TILE_N, D))
        v_regs = tl.load(tle.gpu.local_ptr(v_prev, (v_rows, v_cols)))
        v_regs_t = tl.trans(v_regs)
        v_wgmma = tle.gpu.alloc(
            [D, _final_mtp2_c4_narrow16__TILE_N],
            dtype=tl.bfloat16,
            layout=None,
            scope=tle.gpu.smem,
            init_value=v_regs_t,
        )
        pv = tle.gpu.wgmma(v_wgmma, p_smem, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[None, :]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _final_mtp2_c4_narrow16__TMA_STAGES
        if next_k < num_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(
                K_DESC,
                k_smem.slot(slot),
                [1, 1, _final_mtp2_c4_narrow16__TILE_N, D],
                [next_k_phys, hkv, 0, 0],
                barrier=k_full[slot],
            )
        next_v = prev_page + _final_mtp2_c4_narrow16__TMA_STAGES
        if next_v < num_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(
                V_DESC,
                v_smem.slot(prev_slot),
                [1, 1, _final_mtp2_c4_narrow16__TILE_N, D],
                [next_v_phys, hkv, 0, 0],
                barrier=v_full[prev_slot],
            )
        page += 1
    slot = page % _final_mtp2_c4_narrow16__TMA_STAGES
    phase = page // _final_mtp2_c4_narrow16__TMA_STAGES
    prev_page = page - 1
    prev_slot = prev_page % _final_mtp2_c4_narrow16__TMA_STAGES
    prev_phase = prev_page // _final_mtp2_c4_narrow16__TMA_STAGES
    tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
    scores = tle.gpu.wgmma(k_smem.slot(slot), q_smem, trans_b=True, out_dtype=tl.float32)
    scores = tle.gpu.wgmma_wait(0, scores) * scale
    score_mask = tl.broadcast_to(valid_row[None, :], (_final_mtp2_c4_narrow16__TILE_N, Q_ROWS))
    scores = tl.where(score_mask, scores, -float("inf"))
    page_max = tl.max(scores, axis=0)
    m_new = tl.maximum(m_i, page_max)
    safe_new = tl.where(m_new != -float("inf"), m_new, 0.0)
    alpha = tl.where(m_i != -float("inf"), tl.exp2(m_i - safe_new), 0.0)
    p_curr = tl.where(score_mask, tl.exp2(scores - safe_new[None, :]), 0.0)
    l_i = l_i * alpha + tl.sum(p_curr, axis=0)
    m_i = m_new
    tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
    v_prev = v_smem.slot(prev_slot)
    v_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_cols = tl.broadcast_to(offs_d[None, :], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_regs = tl.load(tle.gpu.local_ptr(v_prev, (v_rows, v_cols)))
    v_regs_t = tl.trans(v_regs)
    v_wgmma = tle.gpu.alloc(
        [D, _final_mtp2_c4_narrow16__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, init_value=v_regs_t
    )
    pv = tle.gpu.wgmma(v_wgmma, p_smem, out_dtype=tl.float32)
    pv = tle.gpu.wgmma_wait(0, pv)
    acc = (acc + pv) * alpha[None, :]
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
    last_slot = (num_pages - 1) % _final_mtp2_c4_narrow16__TMA_STAGES
    last_phase = (num_pages - 1) // _final_mtp2_c4_narrow16__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_last = v_smem.slot(last_slot)
    v_rows = tl.broadcast_to(offs_n[:, None], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_cols = tl.broadcast_to(offs_d[None, :], (_final_mtp2_c4_narrow16__TILE_N, D))
    v_regs = tl.load(tle.gpu.local_ptr(v_last, (v_rows, v_cols)))
    v_regs_t = tl.trans(v_regs)
    v_wgmma = tle.gpu.alloc(
        [D, _final_mtp2_c4_narrow16__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, init_value=v_regs_t
    )
    pv_last = tle.gpu.wgmma(v_wgmma, p_smem, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    acc_rows = tl.trans(acc)
    valid_local = l_i > 0.0
    if RAW_DETACHED:
        raw_ptr = (
            SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :]
        )
        scalar_ptr = batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH
        tl.store(raw_ptr, tl.where(valid_local[:, None], acc_rows, 0.0), mask=valid_row[:, None])
        tl.store(RAW_M + scalar_ptr, tl.where(valid_local, m_i, -float("inf")), mask=valid_row)
        tl.store(RAW_L + scalar_ptr, tl.where(valid_local, l_i, 0.0), mask=valid_row)
        if PDL_NOTIFY:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc_rows, 0.0))
        tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float("inf")))
        tl.store(tle.gpu.local_ptr(partial_stats, (Q_ROWS + offs_r,)), tl.where(valid_local, l_i, 0.0))
        tle.distributed_barrier(mesh)
        if LOCAL_PAIRED_HEADS:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_paired_head_merge(
                partial_acc,
                partial_stats,
                paired_acc,
                paired_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        else:
            _final_mtp2_c4_narrow16_direct_v_ss__kernel__c4_local_serial_head_merge(
                partial_acc,
                partial_stats,
                merged_acc,
                merged_lse,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                mesh,
                rank,
                batch,
                hkv,
                group,
                group_count,
                NUM_SEQ_Q,
                NUM_SEQ_Q_PAD,
                H_Q,
                HEADS_PER_GROUP,
                Q_ROWS,
                D,
                SO_SB,
                SO_SG,
                SO_SM,
                SO_SH,
                SL_SB,
                SL_SG,
                SL_SM,
                SL_SH,
                O_SB,
                O_SM,
                O_SH,
            )
        tle.distributed_barrier(mesh)
        _final_mtp2_c4_paired_finalize__kernel__c4_paired_head_group_finalize(
            SPLIT_OUT,
            SPLIT_LSE,
            COMPLETION,
            OUT,
            merged_lse,
            mesh,
            rank,
            batch,
            hkv,
            group,
            group_count,
            B,
            NUM_SEQ_Q,
            NUM_SEQ_Q_PAD,
            H_Q,
            HEADS_PER_GROUP,
            D,
            4,
            MAX_GROUPS,
            True,
            SO_SB,
            SO_SG,
            SO_SM,
            SO_SH,
            SL_SB,
            SL_SG,
            SL_SM,
            SL_SH,
            O_SB,
            O_SM,
            O_SH,
        )


@triton.jit
def _compact_one64_reducer(
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    GROUPS: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    D: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Reduce batch zero; the remaining one-group batches are already final."""
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    head_passes = tl.cdiv(HEADS_PER_GROUP, HEADS_PER_PROGRAM)
    seq_m = pid // head_passes
    head_pass = pid - seq_m * head_passes
    offs_h = tl.arange(0, HEADS_PER_PROGRAM)
    offs_g = tl.arange(0, GROUPS)
    offs_d = tl.arange(0, D)
    hq = head_pass * HEADS_PER_PROGRAM + offs_h
    valid_h = hq < HEADS_PER_GROUP
    lse = tl.load(
        SPLIT_LSE + offs_g[None, :] * SL_SG + seq_m * SL_SM + hq[:, None] * SL_SH,
        mask=valid_h[:, None],
        other=-float("inf"),
    )
    max_lse = tl.max(lse, axis=1)
    weights = tl.where(valid_h[:, None], tl.exp2(lse - max_lse[:, None]), 0.0)
    denom = tl.sum(weights, axis=1)
    acc = tl.zeros((HEADS_PER_PROGRAM, D), tl.float32)
    for group in tl.static_range(0, GROUPS):
        group_lse = tl.load(
            SPLIT_LSE + group * SL_SG + seq_m * SL_SM + hq * SL_SH,
            mask=valid_h,
            other=-float("inf"),
        )
        partial = tl.load(
            SPLIT_OUT + group * SO_SG + seq_m * SO_SM + hq[:, None] * SO_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += partial * tl.exp2(group_lse - max_lse)[:, None]
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + seq_m * O_SM + hq[:, None] * O_SH + offs_d[None, :],
        acc / safe_denom[:, None],
        mask=valid_h[:, None] & (denom[:, None] > 0.0),
    )


@triton.jit
def _compact_one128_reducer(
    SPLIT_OUT,
    RAW_M,
    RAW_L,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    LONG_GROUPS: tl.constexpr,
    SHORT_GROUPS: tl.constexpr,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    RS_SB: tl.constexpr,
    RS_SG: tl.constexpr,
    RS_SM: tl.constexpr,
    RS_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """HPP1 for the long batch and HPP8 for exact-two-group batches."""
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    long_programs = NUM_SEQ_Q * H_Q
    offs_d = tl.arange(0, D)
    if pid < long_programs:
        seq_m = pid // H_Q
        hq = pid - seq_m * H_Q
        long_offs_g = tl.arange(0, LONG_GROUPS)
        long_scalar = long_offs_g * RS_SG + seq_m * RS_SM + hq * RS_SH
        long_m = tl.load(RAW_M + long_scalar)
        long_l = tl.load(RAW_L + long_scalar)
        long_max_m = tl.max(long_m, axis=0)
        long_weights = tl.exp2(long_m - long_max_m)
        long_denom = tl.sum(long_l * long_weights, axis=0)
        long_acc = tl.zeros((D,), tl.float32)
        for long_group in tl.static_range(0, LONG_GROUPS):
            long_group_m = tl.load(RAW_M + long_group * RS_SG + seq_m * RS_SM + hq * RS_SH)
            long_partial = tl.load(SPLIT_OUT + long_group * SO_SG + seq_m * SO_SM + hq * SO_SH + offs_d).to(tl.float32)
            long_acc += long_partial * tl.exp2(long_group_m - long_max_m)
        tl.store(
            OUT + seq_m * O_SM + hq * O_SH + offs_d,
            long_acc / long_denom,
        )
    else:
        short_pid = pid - long_programs
        batch = short_pid // NUM_SEQ_Q + 1
        seq_m = short_pid - (batch - 1) * NUM_SEQ_Q
        short_h = tl.arange(0, H_Q)
        short_g = tl.arange(0, SHORT_GROUPS)
        short_scalar = batch * RS_SB + short_g[None, :] * RS_SG + seq_m * RS_SM + short_h[:, None] * RS_SH
        short_m = tl.load(RAW_M + short_scalar)
        short_l = tl.load(RAW_L + short_scalar)
        short_max_m = tl.max(short_m, axis=1)
        short_weights = tl.exp2(short_m - short_max_m[:, None])
        short_denom = tl.sum(short_l * short_weights, axis=1)
        short_d = tl.arange(0, 32)
        for d_pass in tl.static_range(0, D // 32):
            current_d = d_pass * 32 + short_d
            short_acc = tl.zeros((H_Q, 32), tl.float32)
            for short_group in tl.static_range(0, SHORT_GROUPS):
                short_group_m = tl.load(RAW_M + batch * RS_SB + short_group * RS_SG + seq_m * RS_SM + short_h * RS_SH)
                short_partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_SB
                    + short_group * SO_SG
                    + seq_m * SO_SM
                    + short_h[:, None] * SO_SH
                    + current_d[None, :]
                ).to(tl.float32)
                short_acc += short_partial * tl.exp2(short_group_m - short_max_m)[:, None]
            tl.store(
                OUT + batch * O_SB + seq_m * O_SM + short_h[:, None] * O_SH + current_d[None, :],
                short_acc / short_denom[:, None],
            )


_MESH_C1 = _final_mtp2_c4_narrow16__MESH_C1
_MESH_C4 = _bf16_entry___CLUSTER_MESHES[4]
_MESH_C8 = _bf16_entry___CLUSTER_MESHES[8]


def _tensor_descriptors(inputs: DynamicBF16Inputs):
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=[1, 1, 64, 128]),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=[1, 1, 64, 128]),
    )


def _run_c4_compact(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
    *,
    num_seq_q: int,
    q_rows: int,
    num_seq_q_pad: int,
    producer_maxnreg: int | None,
    producer_num_stages: int,
):
    _mtp, hq, hkv = _dynamic__validate(inputs)
    if hq != 8 or hkv != 1:
        raise ValueError("compact C4 route requires HQ8/HKV1")
    k_desc, v_desc = _tensor_descriptors(inputs)
    _mtp1_one64_nhd_producer[workspace.num_clusters,](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_MESH_C4,
        B=inputs.batch,
        NUM_SEQ_Q=num_seq_q,
        Q_ROWS=q_rows,
        NUM_SEQ_Q_PAD=num_seq_q_pad,
        H_Q=8,
        HEADS_PER_GROUP=8,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=1024,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=producer_num_stages,
        maxnreg=producer_maxnreg,
        launch_pdl=True,
    )
    _compact_one64_reducer[(num_seq_q * 4,)](
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        NUM_SEQ_Q=num_seq_q,
        GROUPS=16,
        HEADS_PER_GROUP=8,
        HEADS_PER_PROGRAM=2,
        D=128,
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return workspace.out.reshape_as(inputs.q)


def _run_raw_producer(
    inputs: DynamicBF16Inputs,
    workspace: _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace,
    *,
    num_seq_q: int,
    q_rows: int,
    num_seq_q_pad: int,
):
    ws = workspace.base
    _mtp, hq, hkv = _dynamic__validate(inputs)
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c4_narrow16__kernel__mtp2_c4t1024_narrow16_kernel[ws.num_clusters,](
        ws.task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        workspace.raw_m,
        workspace.raw_l,
        mesh=_MESH_C1,
        B=inputs.batch,
        NUM_SEQ_Q=num_seq_q,
        Q_ROWS=q_rows,
        NUM_SEQ_Q_PAD=num_seq_q_pad,
        H_Q=hq,
        HEADS_PER_GROUP=hq // hkv,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=2048,
        MAX_GROUPS=ws.max_groups,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=workspace.raw_m.stride(0),
        SL_SG=workspace.raw_m.stride(1),
        SL_SM=workspace.raw_m.stride(2),
        SL_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=True,
        PDL_NOTIFY=True,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        maxnreg=240,
        launch_pdl=True,
    )


def _run_compact_one128(
    inputs: DynamicBF16Inputs,
    workspace: _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace,
    *,
    num_seq_q: int,
    q_rows: int,
    num_seq_q_pad: int,
):
    _run_raw_producer(
        inputs,
        workspace,
        num_seq_q=num_seq_q,
        q_rows=q_rows,
        num_seq_q_pad=num_seq_q_pad,
    )
    ws = workspace.base
    grid = num_seq_q * 8 + (inputs.batch - 1) * num_seq_q
    _compact_one128_reducer[(grid,)](
        ws.split_out,
        workspace.raw_m,
        workspace.raw_l,
        ws.out,
        NUM_SEQ_Q=num_seq_q,
        LONG_GROUPS=64,
        SHORT_GROUPS=2,
        H_Q=8,
        D=128,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        RS_SB=workspace.raw_m.stride(0),
        RS_SG=workspace.raw_m.stride(1),
        RS_SM=workspace.raw_m.stride(2),
        RS_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


@dataclass
class _MTP3Workspace:
    base: DynamicBF16Workspace
    payload: DynamicBF16Workspace | _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace
    workload: DecodeWorkload
    route: str


_MTP3_Q_ROWS = tl.constexpr(32)
_MTP3_Q_PAD = tl.constexpr(4)
_TASK_RECORD_STRIDE = tl.constexpr(8)


@triton.jit
def _mtp3_skewed_mix_kernel(
    REGULAR_TASK_MAP,
    DIRECT_TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    NUM_REGULAR_CLUSTERS: tl.constexpr,
    NUM_DIRECT_TASKS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
    ALIGNED_FULL_CHUNK: tl.constexpr,
):
    cta = tl.program_id(0)
    regular_ctas = NUM_REGULAR_CLUSTERS * 4
    if cta < regular_ctas:
        _final_mtp2_c4_narrow16_direct_v_ss__kernel__mtp2_c4t1024_narrow16_direct_v_ss_kernel(
            REGULAR_TASK_MAP,
            Q,
            K_DESC,
            V_DESC,
            BLOCK_IDS,
            COMPLETION,
            SPLIT_OUT,
            SPLIT_LSE,
            OUT,
            SPLIT_LSE,
            SPLIT_LSE,
            mesh=mesh,
            B=B,
            NUM_SEQ_Q=NUM_SEQ_Q,
            Q_ROWS=_MTP3_Q_ROWS,
            NUM_SEQ_Q_PAD=_MTP3_Q_PAD,
            H_Q=H_Q,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_BLOCKS=MAX_BLOCKS,
            CHUNK_TOKENS=CHUNK_TOKENS,
            MAX_GROUPS=MAX_GROUPS,
            Q_SB=Q_SB,
            Q_SM=Q_SM,
            Q_SH=Q_SH,
            SO_SB=SO_SB,
            SO_SG=SO_SG,
            SO_SM=SO_SM,
            SO_SH=SO_SH,
            SL_SB=SL_SB,
            SL_SG=SL_SG,
            SL_SM=SL_SM,
            SL_SH=SL_SH,
            O_SB=O_SB,
            O_SM=O_SM,
            O_SH=O_SH,
            LOCAL_PAIRED_HEADS=False,
            RAW_DETACHED=False,
            PDL_NOTIFY=False,
        )
    else:
        direct_index = cta - regular_ctas
        if direct_index < NUM_DIRECT_TASKS:
            shifted_map = DIRECT_TASK_MAP - regular_ctas * _TASK_RECORD_STRIDE
            _dynamic__bf16_decode_direct_narrow_kernel(
                shifted_map,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                B=B,
                NUM_SEQ_Q=NUM_SEQ_Q,
                Q_ROWS=_MTP3_Q_ROWS,
                H_Q=H_Q,
                HEADS_PER_GROUP=HEADS_PER_GROUP,
                D=D,
                BLOCK_SIZE=BLOCK_SIZE,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                PREFETCH_BLOCK_IDS=True,
                ALIGNED_FULL_CHUNK=ALIGNED_FULL_CHUNK,
                WIDE_BASELINE=False,
                PDL_NOTIFY=False,
                TMA_STAGES=2,
            )


def _prepare_mtp3_workspace(inputs: DynamicBF16Inputs) -> _MTP3Workspace:
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    if inputs.mtp != 3 or workload not in _KNOWN_FEATURES:
        raise ValueError("fixed MTP3 routing requires a tuned workload shape")
    base = _prepare_dynamic_bf16_workspace_base(inputs)
    expected = _FINAL_CT_BY_FEATURE[3][workload.signature][inputs.layout]
    selected = (base.policy.cluster_size, base.policy.chunk_tokens)
    if selected != expected:
        raise AssertionError(f"MTP3 policy mismatch: {selected} != {expected}")
    if workload == _F_ONE_128K_31_SHORT:
        batch, groups, mtp, heads = base.split_lse.shape

        def allocate():
            return torch.empty(
                (batch, mtp, heads, groups),
                dtype=torch.float32,
                device=inputs.q.device,
            ).permute(0, 3, 1, 2)

        payload = _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace(
            base=base,
            raw_m=allocate(),
            raw_l=allocate(),
            producer_task_map=base.task_map,
            producer_chunk_tokens=2048,
            producer_ctas=base.num_clusters,
            cuda_split16=False,
        )
        route = "c1t2048/padded32-raw/chunk-minor/pdl/hpp2/r240-s3"
    else:
        payload = base
        routes = {
            _F_MIX_128_4096.signature: "c4t1024/padded32/unified-direct-pack",
            _F_ONE_64K_7_SHORT.signature: "c8t512/padded32/direct-v-ss/detached-hpp2/r192-s2",
            _F_ONE_64K_15_SHORT.signature: (
                "c4t1024/padded32/direct-v-ss/compact-hpp2/" + ("r192-s2" if inputs.layout == "NHD" else "r240-s2")
            ),
            _F_TWO_32K_30_SHORT.signature: "c4t1024/padded32",
        }
        route = routes.get(workload.signature, f"c{expected[0]}t{expected[1]}/wide64")
    return _MTP3Workspace(base=base, payload=payload, workload=workload, route=route)


def _run_mtp3_padded32(inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace):
    _mtp, hq, hkv = _dynamic__validate(inputs)
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c4_narrow16__kernel__mtp2_c4t1024_narrow16_kernel[workspace.num_clusters,](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_MESH_C4,
        B=inputs.batch,
        NUM_SEQ_Q=3,
        Q_ROWS=32,
        NUM_SEQ_Q_PAD=4,
        H_Q=hq,
        HEADS_PER_GROUP=hq // hkv,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=1024,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=False,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


def _run_mtp3_skewed_mix(inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace):
    _mtp, hq, hkv = _dynamic__validate(inputs)
    k_desc, v_desc = _tensor_descriptors(inputs)
    direct_clusters = triton.cdiv(workspace.num_direct_tasks, 4)
    grid = workspace.num_clusters + direct_clusters
    _mtp3_skewed_mix_kernel[(grid,)](
        workspace.task_map,
        workspace.direct_task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        mesh=_MESH_C4,
        B=inputs.batch,
        NUM_SEQ_Q=3,
        H_Q=hq,
        HEADS_PER_GROUP=hq // hkv,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=1024,
        MAX_GROUPS=workspace.max_groups,
        NUM_REGULAR_CLUSTERS=workspace.num_clusters,
        NUM_DIRECT_TASKS=workspace.num_direct_tasks,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        ALIGNED_FULL_CHUNK=workspace.policy.aligned_full_chunk,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return workspace.out.reshape_as(inputs.q)


def _run_mtp3_one64_7(inputs: DynamicBF16Inputs, workspace: DynamicBF16Workspace):
    _mtp, hq, hkv = _dynamic__validate(inputs)
    k_desc, v_desc = _tensor_descriptors(inputs)
    _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel[
        workspace.num_clusters,
    ](
        workspace.task_map,
        workspace.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        workspace.completion,
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        workspace.split_lse,
        workspace.split_lse,
        mesh=_MESH_C8,
        B=inputs.batch,
        NUM_SEQ_Q=3,
        Q_ROWS=32,
        NUM_SEQ_Q_PAD=4,
        H_Q=hq,
        HEADS_PER_GROUP=hq // hkv,
        D=128,
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=512,
        MAX_GROUPS=workspace.max_groups,
        Q_SB=workspace.q_4d.stride(0),
        Q_SM=workspace.q_4d.stride(1),
        Q_SH=workspace.q_4d.stride(2),
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        LOCAL_PAIRED_HEADS=True,
        RAW_DETACHED=False,
        PDL_NOTIFY=False,
        DETACHED_GROUP_REDUCER=True,
        num_ctas=1,
        num_warps=4,
        num_stages=2,
        maxnreg=192,
        launch_pdl=True,
    )
    _final_mtp2_one64_7_c8_detached_reducer__kernel__mtp2_one64_c8_detached_hpp2_reducer[(12,)](
        workspace.split_out,
        workspace.split_lse,
        workspace.out,
        NUM_SEQ_Q=3,
        HEADS_PER_GROUP=8,
        D=128,
        GROUPS=16,
        HEADS_PER_PROGRAM=2,
        SO_SB=workspace.split_out.stride(0),
        SO_SG=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SG=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return workspace.out.reshape_as(inputs.q)


def _run_mtp3(inputs: DynamicBF16Inputs, workspace: _MTP3Workspace):
    if workspace.workload == _F_MIX_128_4096:
        return _run_mtp3_skewed_mix(inputs, workspace.base)
    if workspace.workload == _F_ONE_64K_7_SHORT:
        return _run_mtp3_one64_7(inputs, workspace.base)
    if workspace.workload == _F_ONE_64K_15_SHORT:
        reg = 192 if inputs.layout == "NHD" else 240
        return _run_c4_compact(
            inputs,
            workspace.base,
            num_seq_q=3,
            q_rows=32,
            num_seq_q_pad=4,
            producer_maxnreg=reg,
            producer_num_stages=2,
        )
    if workspace.workload == _F_ONE_128K_31_SHORT:
        raw = workspace.payload
        return _run_raw_producer_and_reduce_mtp3(inputs, raw)
    if workspace.workload == _F_TWO_32K_30_SHORT:
        return _run_mtp3_padded32(inputs, workspace.base)
    return _attention_decode_bf16_dynamic_base(inputs, workspace.base)


def _run_raw_producer_and_reduce_mtp3(
    inputs: DynamicBF16Inputs,
    workspace: _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace,
):
    _run_raw_producer(inputs, workspace, num_seq_q=3, q_rows=32, num_seq_q_pad=4)
    ws = workspace.base
    grid = inputs.batch * 3 * triton.cdiv(8, 2)
    _final_mtp2_c1_raw_detached__kernel__mtp2_c1_raw_detached_reducer[(grid,)](
        inputs.kv_lens,
        ws.split_out,
        workspace.raw_m,
        workspace.raw_l,
        ws.out,
        B=inputs.batch,
        NUM_SEQ_Q=3,
        NUM_SEQ_Q_PAD=4,
        H_Q=8,
        H_KV=1,
        HEADS_PER_GROUP=8,
        D=128,
        CHUNK_TOKENS=2048,
        MAX_GROUPS=triton.next_power_of_2(ws.max_groups),
        HEADS_PER_PROGRAM=2,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        RS_SB=workspace.raw_m.stride(0),
        RS_SG=workspace.raw_m.stride(1),
        RS_SM=workspace.raw_m.stride(2),
        RS_SH=workspace.raw_m.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        PDL_WAIT=True,
        CUDA_SPLIT16=False,
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


_SKEWED_DETACHED_TASK_STRIDE = tl.constexpr(8)
_SKEWED_DETACHED_MESH = _bf16_entry___CLUSTER_MESHES[8]
_SKEWED_DETACHED_PRODUCER = (
    _final_mtp2_c8_narrow16_direct_v_ss_detached__kernel__mtp2_c8t512_narrow16_direct_v_ss_kernel
)
_SKEWED_DETACHED_DIRECT = _dynamic__bf16_decode_direct_narrow_kernel


@triton.jit
def _skewed_detached_unified_kernel(
    REGULAR_TASK_MAP,
    DIRECT_TASK_MAP,
    Q,
    K_DESC,
    V_DESC,
    BLOCK_IDS,
    COMPLETION,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    Q_ROWS: tl.constexpr,
    NUM_SEQ_Q_PAD: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    NUM_REGULAR_CLUSTERS: tl.constexpr,
    NUM_DIRECT_TASKS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    cta = tl.program_id(0)
    regular_ctas = NUM_REGULAR_CLUSTERS * 8
    if cta < regular_ctas:
        _SKEWED_DETACHED_PRODUCER(
            REGULAR_TASK_MAP,
            Q,
            K_DESC,
            V_DESC,
            BLOCK_IDS,
            COMPLETION,
            SPLIT_OUT,
            SPLIT_LSE,
            OUT,
            SPLIT_LSE,
            SPLIT_LSE,
            mesh=mesh,
            B=B,
            NUM_SEQ_Q=NUM_SEQ_Q,
            Q_ROWS=Q_ROWS,
            NUM_SEQ_Q_PAD=NUM_SEQ_Q_PAD,
            H_Q=H_Q,
            HEADS_PER_GROUP=HEADS_PER_GROUP,
            D=D,
            BLOCK_SIZE=64,
            MAX_BLOCKS=MAX_BLOCKS,
            CHUNK_TOKENS=CHUNK_TOKENS,
            MAX_GROUPS=MAX_GROUPS,
            Q_SB=Q_SB,
            Q_SM=Q_SM,
            Q_SH=Q_SH,
            SO_SB=SO_SB,
            SO_SG=SO_SG,
            SO_SM=SO_SM,
            SO_SH=SO_SH,
            SL_SB=SL_SB,
            SL_SG=SL_SG,
            SL_SM=SL_SM,
            SL_SH=SL_SH,
            O_SB=O_SB,
            O_SM=O_SM,
            O_SH=O_SH,
            LOCAL_PAIRED_HEADS=True,
            RAW_DETACHED=False,
            PDL_NOTIFY=True,
            DETACHED_GROUP_REDUCER=True,
        )
    else:
        direct_index = cta - regular_ctas
        if direct_index < NUM_DIRECT_TASKS:
            shifted_map = DIRECT_TASK_MAP - regular_ctas * _SKEWED_DETACHED_TASK_STRIDE
            _SKEWED_DETACHED_DIRECT(
                shifted_map,
                Q,
                K_DESC,
                V_DESC,
                BLOCK_IDS,
                SPLIT_OUT,
                SPLIT_LSE,
                OUT,
                B=B,
                NUM_SEQ_Q=NUM_SEQ_Q,
                Q_ROWS=Q_ROWS,
                H_Q=H_Q,
                HEADS_PER_GROUP=HEADS_PER_GROUP,
                D=D,
                BLOCK_SIZE=64,
                MAX_BLOCKS=MAX_BLOCKS,
                CHUNK_TOKENS=CHUNK_TOKENS,
                MAX_GROUPS=MAX_GROUPS,
                Q_SB=Q_SB,
                Q_SM=Q_SM,
                Q_SH=Q_SH,
                SO_SB=SO_SB,
                SO_SG=SO_SG,
                SO_SM=SO_SM,
                SO_SH=SO_SH,
                SL_SB=SL_SB,
                SL_SG=SL_SG,
                SL_SM=SL_SM,
                SL_SH=SL_SH,
                O_SB=O_SB,
                O_SM=O_SM,
                O_SH=O_SH,
                PREFETCH_BLOCK_IDS=True,
                ALIGNED_FULL_CHUNK=False,
                WIDE_BASELINE=False,
                PDL_NOTIFY=True,
                TMA_STAGES=2,
            )


@triton.jit
def _skewed_detached_hpp1_reducer(
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    LONG_BATCH: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SG: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SG: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    seq_m = pid // H_Q
    hq = pid - seq_m * H_Q
    offs_g = tl.arange(0, GROUPS)
    offs_d = tl.arange(0, D)
    lse = tl.load(
        SPLIT_LSE + LONG_BATCH * SL_SB + offs_g * SL_SG + seq_m * SL_SM + hq * SL_SH,
    )
    max_lse = tl.max(lse, axis=0)
    weights = tl.exp2(lse - max_lse)
    denom = tl.sum(weights, axis=0)
    acc = tl.zeros((D,), tl.float32)
    for group in tl.static_range(0, GROUPS):
        group_lse = tl.load(
            SPLIT_LSE + LONG_BATCH * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH,
        )
        partial = tl.load(
            SPLIT_OUT + LONG_BATCH * SO_SB + group * SO_SG + seq_m * SO_SM + hq * SO_SH + offs_d,
        ).to(tl.float32)
        acc += partial * tl.exp2(group_lse - max_lse)
    tl.store(
        OUT + LONG_BATCH * O_SB + seq_m * O_SM + hq * O_SH + offs_d,
        acc / denom,
    )


@dataclass
class _SkewedDetachedWorkspace:
    base: DynamicBF16Workspace
    long_batch: int
    groups: int
    chunk_tokens: int
    producer_maxnreg: int | None
    producer_num_stages: int


def _is_one_16k_many_64(lengths: tuple[int, ...]) -> bool:
    return len(lengths) == 16 and lengths.count(64) == 15 and lengths.count(16 * 1024) == 1


def _prepare_skewed_detached_workspace(
    inputs: DynamicBF16Inputs,
    base: DynamicBF16Workspace,
    lengths: tuple[int, ...],
) -> _SkewedDetachedWorkspace:
    mtp = inputs.mtp
    token = {
        (1, "NHD"): 256,
        (1, "HND"): 128,
        (2, "NHD"): 128,
        (2, "HND"): 128,
        (3, "NHD"): 256,
        (3, "HND"): 128,
    }[(mtp, inputs.layout)]
    expected = (8, token)
    selected = (base.policy.cluster_size, base.policy.chunk_tokens)
    if selected != expected:
        raise AssertionError(f"skewed detached policy mismatch: {selected} != {expected}")
    maxnreg, stages = {
        (1, "NHD"): (None, 2),
        (1, "HND"): (None, 2),
        (2, "NHD"): (None, 2),
        (2, "HND"): (None, 3),
        (3, "NHD"): (224, 2),
        (3, "HND"): (192, 2),
    }[(mtp, inputs.layout)]
    long_batch = lengths.index(max(lengths))
    groups = (lengths[long_batch] + 8 * token - 1) // (8 * token)
    return _SkewedDetachedWorkspace(
        base,
        long_batch,
        groups,
        token,
        maxnreg,
        stages,
    )


def _run_skewed_detached(
    inputs: DynamicBF16Inputs,
    workspace: _SkewedDetachedWorkspace,
) -> torch.Tensor:
    ws = workspace.base
    mtp, hq, hkv = _dynamic__validate(inputs)
    q_rows, q_pad = {1: (16, 1), 2: (16, 2), 3: (32, 4)}[mtp]
    k_desc, v_desc = _tensor_descriptors(inputs)
    grid = ws.num_clusters + triton.cdiv(ws.num_direct_tasks, 8)
    _skewed_detached_unified_kernel[(grid,)](
        ws.task_map,
        ws.direct_task_map,
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.completion,
        ws.split_out,
        ws.split_lse,
        ws.out,
        mesh=_SKEWED_DETACHED_MESH,
        B=inputs.batch,
        NUM_SEQ_Q=mtp,
        Q_ROWS=q_rows,
        NUM_SEQ_Q_PAD=q_pad,
        H_Q=hq,
        HEADS_PER_GROUP=hq // hkv,
        D=128,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        CHUNK_TOKENS=workspace.chunk_tokens,
        MAX_GROUPS=ws.max_groups,
        NUM_REGULAR_CLUSTERS=ws.num_clusters,
        NUM_DIRECT_TASKS=ws.num_direct_tasks,
        Q_SB=ws.q_4d.stride(0),
        Q_SM=ws.q_4d.stride(1),
        Q_SH=ws.q_4d.stride(2),
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=ws.split_lse.stride(0),
        SL_SG=ws.split_lse.stride(1),
        SL_SM=ws.split_lse.stride(2),
        SL_SH=ws.split_lse.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        num_ctas=1,
        num_warps=4,
        num_stages=workspace.producer_num_stages,
        maxnreg=workspace.producer_maxnreg,
        launch_pdl=True,
    )
    _skewed_detached_hpp1_reducer[(mtp * hq,)](
        ws.split_out,
        ws.split_lse,
        ws.out,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        D=128,
        GROUPS=workspace.groups,
        LONG_BATCH=workspace.long_batch,
        SO_SB=ws.split_out.stride(0),
        SO_SG=ws.split_out.stride(1),
        SO_SM=ws.split_out.stride(2),
        SO_SH=ws.split_out.stride(3),
        SL_SB=ws.split_lse.stride(0),
        SL_SG=ws.split_lse.stride(1),
        SL_SM=ws.split_lse.stride(2),
        SL_SH=ws.split_lse.stride(3),
        O_SB=ws.out.stride(0),
        O_SM=ws.out.stride(1),
        O_SH=ws.out.stride(2),
        num_warps=4,
        num_stages=1,
        launch_pdl=True,
    )
    return ws.out.reshape_as(inputs.q)


@dataclass
class _FinalDynamicBF16Workspace:
    """Own the selected production implementation and its base buffers."""

    base: DynamicBF16Workspace
    implementation: object
    mtp: int
    route: str
    workload: DecodeWorkload

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def _base_workspace(workspace) -> DynamicBF16Workspace:
    if isinstance(workspace, _FinalDynamicBF16Workspace):
        return workspace.base
    if isinstance(workspace, DynamicBF16Workspace):
        return workspace
    raise TypeError(f"unsupported BF16 dynamic workspace: {type(workspace)!r}")


def prepare_dynamic_bf16_workspace(inputs: DynamicBF16Inputs):
    """Prepare the fixed MTP1/MTP2/MTP3 winner, with a generic fallback."""
    if not USE_TLE:
        return prepare_pure_triton_mtp1_workspace(inputs, "bf16")
    mtp, _, _ = _dynamic__validate(inputs)
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    workload = DecodeWorkload.from_lengths(lengths)
    tuned = workload in _KNOWN_FEATURES
    if mtp == 1 and tuned:
        implementation = _final_mtp1_merged_final__prepare_dynamic_bf16_mtp1_merged_final_workspace(inputs)
        if _is_one_16k_many_64(lengths):
            detached = _prepare_skewed_detached_workspace(
                inputs,
                implementation.base,
                lengths,
            )
            route = (
                f"c8t{detached.chunk_tokens}/narrow16/unified-direct-pack/"
                f"pdl-detached-hpp1/r{detached.producer_maxnreg or 'default'}"
                f"-s{detached.producer_num_stages}"
            )
            return _FinalDynamicBF16Workspace(
                detached.base,
                detached,
                1,
                route,
                workload,
            )
        return _FinalDynamicBF16Workspace(implementation.base, implementation, 1, implementation.route, workload)
    if mtp == 2 and tuned:
        implementation = _final_mtp2_detached_final__prepare_dynamic_bf16_mtp2_detached_final_workspace(inputs)
        base = (
            implementation.base
            if isinstance(implementation, _final_mtp2_c1_raw_detached__MTP2C1RawDetachedWorkspace)
            else implementation
        )
        if _is_one_16k_many_64(lengths):
            detached = _prepare_skewed_detached_workspace(inputs, base, lengths)
            route = (
                f"c8t{detached.chunk_tokens}/narrow16/unified-direct-pack/"
                f"pdl-detached-hpp1/r{detached.producer_maxnreg or 'default'}"
                f"-s{detached.producer_num_stages}"
            )
            return _FinalDynamicBF16Workspace(
                detached.base,
                detached,
                2,
                route,
                workload,
            )
        if workload == _F_ONE_64K_15_SHORT:
            reg = "default" if inputs.layout == "NHD" else "240"
            route = f"c4t1024/direct-v-ss/compact-hpp2/r{reg}-s{3 if inputs.layout == 'NHD' else 2}"
        elif workload == _F_ONE_128K_31_SHORT:
            route = "c1t2048/raw-chunk-minor/pdl/heterogeneous-compact-78/r240-s3"
        else:
            route = f"{base.policy.label}/final"
        return _FinalDynamicBF16Workspace(base, implementation, 2, route, workload)
    if mtp == 3 and tuned:
        implementation = _prepare_mtp3_workspace(inputs)
        if _is_one_16k_many_64(lengths):
            detached = _prepare_skewed_detached_workspace(
                inputs,
                implementation.base,
                lengths,
            )
            route = (
                f"c8t{detached.chunk_tokens}/padded32/unified-direct-pack/"
                f"pdl-detached-hpp1/r{detached.producer_maxnreg or 'default'}"
                f"-s{detached.producer_num_stages}"
            )
            return _FinalDynamicBF16Workspace(
                detached.base,
                detached,
                3,
                route,
                workload,
            )
        return _FinalDynamicBF16Workspace(implementation.base, implementation, 3, implementation.route, workload)
    return _prepare_dynamic_bf16_workspace_base(inputs)


def attention_decode_bf16_dynamic(inputs: DynamicBF16Inputs, workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    """Run the fixed production route selected during workspace preparation."""
    if isinstance(workspace, _FinalDynamicBF16Workspace):
        if isinstance(workspace.implementation, _SkewedDetachedWorkspace):
            return _run_skewed_detached(inputs, workspace.implementation)
        if workspace.mtp == 1:
            return _final_mtp1_merged_final__attention_decode_bf16_dynamic_mtp1_merged_final(
                inputs, workspace.implementation
            )
        if workspace.mtp == 2:
            if workspace.workload == _F_ONE_64K_15_SHORT:
                return _run_c4_compact(
                    inputs,
                    workspace.base,
                    num_seq_q=2,
                    q_rows=16,
                    num_seq_q_pad=2,
                    producer_maxnreg=None if inputs.layout == "NHD" else 240,
                    producer_num_stages=3 if inputs.layout == "NHD" else 2,
                )
            if workspace.workload == _F_ONE_128K_31_SHORT:
                return _run_compact_one128(
                    inputs,
                    workspace.implementation,
                    num_seq_q=2,
                    q_rows=16,
                    num_seq_q_pad=2,
                )
            return _final_mtp2_detached_final__attention_decode_bf16_dynamic_mtp2_detached_final(
                inputs, workspace.implementation
            )
        if workspace.mtp == 3:
            return _run_mtp3(inputs, workspace.implementation)
        raise ValueError(f"unsupported finalized MTP: {workspace.mtp}")
    return _attention_decode_bf16_dynamic_base(inputs, workspace)


def refresh_dynamic_bf16_task_map(inputs: DynamicBF16Inputs, workspace) -> None:
    """Refresh a base/final workspace without changing its selected route."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        if inputs.mtp != 1:
            raise NotImplementedError("the no-TLE fallback supports MTP=1 only")
        return
    if isinstance(workspace, _FinalDynamicBF16Workspace):
        lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
        if DecodeWorkload.from_lengths(lengths) != workspace.workload:
            raise ValueError("specialized MTP1/MTP2/MTP3 workspace must be rebuilt after a tail move")
    _refresh_dynamic_bf16_task_map_base(inputs, _base_workspace(workspace))


def bf16_dynamic_workspace_is_reset(workspace) -> bool:
    """Return whether cooperative-finalization counters are replay-ready."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return True
    base = _base_workspace(workspace)
    return not bool(torch.count_nonzero(base.completion).item())


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "DynamicBF16Inputs",
    "DynamicBF16Policy",
    "DynamicBF16Workspace",
    "attention_decode_bf16_dynamic",
    "bf16_dynamic_workspace_is_reset",
    "prepare_dynamic_bf16_workspace",
    "refresh_dynamic_bf16_task_map",
    "select_dynamic_bf16_policy",
]
