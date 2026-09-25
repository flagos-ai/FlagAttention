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

"""Final Hopper BF16 static attention-decode implementation.

The public cache stays logically ``[block, token, head, dim]``. Descriptor
coordinates are reordered to ``[block, head, token, dim]`` so the two leading
unit TMA tile dimensions project directly into rank-2 WGMMA shared operands.
The MTP2 narrow path expresses V transpose as the exact
``local_load -> trans -> local_alloc`` pattern recognized by
``tritongpu-optimize-dot-operands``. The rewrite replaces the apparent
restage with a zero-copy ``tle.memdesc_wgmma_view`` while the validated wide,
pipeline, full-tail, deferred-DSM and specialized-finalizer policies remain
fixed for their selected panels.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from triton.tools.tensor_descriptor import TensorDescriptor


import triton
import triton.language as tl
from .. import (
    DecodeWorkload,
    PureTritonMTPWorkspace,
    PureTritonMTP1Workspace,
    USE_TLE,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
    tle,
)

_base__TILE_N = tl.constexpr(64)


@triton.jit
def _base__bf16_decode_static_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, mesh: tl.constexpr, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, 'cluster_x')
    logical_cluster = cta // CLUSTER_SIZE
    group = logical_cluster % MAX_GROUPS
    sequence = logical_cluster // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group >= group_count:
        return
    chunk = group * CLUSTER_SIZE + rank
    chunk_start = chunk * CHUNK_TOKENS
    has_work = chunk < num_chunks
    chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
    num_pages = (chunk_len + _base__TILE_N - 1) // _base__TILE_N
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_base__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([_base__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([_base__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_base__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_base__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    partial_lse = tle.gpu.alloc([Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc([NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_lse = tle.gpu.alloc([NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
    tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    while page_iter < num_pages:
        page_index = page_iter
        start = page_index * _base__TILE_N
        phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _base__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _base__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_wgmma = k_raw
        v_wgmma = v_raw
        k_page = k_wgmma
        v_page = v_wgmma
        global_n = chunk_start + start + offs_n
        token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
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
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
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
        combined_lse = tl.full((NUM_SEQ_Q_PAD,), -float('inf'), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_lse_md = tle.remote(partial_lse, peer, scope=mesh)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D)), out_cols)), mask=valid_owned[:, None], other=0.0).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_md, (source_rows,)), mask=valid_owned, other=-float('inf'))
            new_lse = tl.maximum(combined_lse, peer_lse)
            valid_merge = new_lse != -float('inf')
            safe = tl.where(valid_merge, new_lse, 0.0)
            w0 = tl.where(combined_lse != -float('inf'), tl.exp2(combined_lse - safe), 0.0)
            w1 = tl.where(peer_lse != -float('inf'), tl.exp2(peer_lse - safe), 0.0)
            denom = w0 + w1
            combined = tl.where(valid_merge[:, None], (combined * w0[:, None] + peer_acc * w1[:, None]) / tl.where(denom[:, None] > 0.0, denom[:, None], 1.0), 0.0)
            combined_lse = tl.where(valid_merge, tl.log2(tl.where(denom > 0.0, denom, 1.0)) + safe, -float('inf'))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if MAX_GROUPS == 1:
            tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
        else:
            tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + owned_hq * SO_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
            tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH, combined_lse, mask=valid_owned)
        tl.debug_barrier()
    tle.distributed_barrier(mesh)

@triton.jit
def _base__bf16_decode_finalize_kernel(SPLIT_OUT, SPLIT_LSE, KV_LENS, OUT, NUM_SEQ_Q: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, D: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    offs_m = tl.arange(0, NUM_SEQ_Q_PAD)
    valid_m = offs_m < NUM_SEQ_Q
    offs_d = tl.arange(0, D)
    offs_g = tl.arange(0, MAX_GROUPS)
    total_len = tl.load(KV_LENS + batch)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    groups = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    valid_g = offs_g < groups
    lse = tl.load(SPLIT_LSE + batch * SL_SB + offs_g[:, None] * SL_SG + offs_m[None, :] * SL_SM + hq * SL_SH, mask=valid_g[:, None] & valid_m[None, :], other=-float('inf'))
    max_lse = tl.max(lse, axis=0)
    safe = tl.where(max_lse != -float('inf'), max_lse, 0.0)
    weights = tl.where(valid_g[:, None], tl.exp2(lse - safe[None, :]), 0.0)
    denom = tl.sum(weights, axis=0)
    acc = tl.zeros((NUM_SEQ_Q_PAD, D), tl.float32)
    for group in tl.static_range(0, MAX_GROUPS):
        group_valid = group < groups
        group_lse = tl.load(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + hq * SL_SH, mask=group_valid & valid_m, other=-float('inf'))
        group_weight = tl.where(group_valid & valid_m, tl.exp2(group_lse - safe), 0.0)
        partial = tl.load(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + hq * SO_SH + offs_d[None, :], mask=group_valid & valid_m[:, None], other=0.0)
        acc += partial * group_weight[:, None]
    acc /= tl.where(denom[:, None] > 0.0, denom[:, None], 1.0)
    tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + hq * O_SH + offs_d[None, :], acc, mask=valid_m[:, None])
_cluster_deferred__TILE_N = tl.constexpr(64)

@triton.jit
def _cluster_deferred__bf16_decode_static_dsm_deferred_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, mesh: tl.constexpr, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, 'cluster_x')
    logical_cluster = cta // CLUSTER_SIZE
    group = logical_cluster % MAX_GROUPS
    sequence = logical_cluster // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group >= group_count:
        return
    chunk = group * CLUSTER_SIZE + rank
    chunk_start = chunk * CHUNK_TOKENS
    has_work = chunk < num_chunks
    chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
    num_pages = (chunk_len + _cluster_deferred__TILE_N - 1) // _cluster_deferred__TILE_N
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_cluster_deferred__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([_cluster_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([_cluster_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_cluster_deferred__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, expect_bytes=_cluster_deferred__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    partial_stats = tle.gpu.alloc([2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc([NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_lse = tle.gpu.alloc([NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
    tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    while page_iter < num_pages:
        page_index = page_iter
        start = page_index * _cluster_deferred__TILE_N
        phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _cluster_deferred__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _cluster_deferred__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_wgmma = k_raw
        v_wgmma = v_raw
        k_page = k_wgmma
        v_page = v_wgmma
        global_n = chunk_start + start + offs_n
        token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
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
    tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float('inf')))
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
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float('inf'), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float('inf'))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            valid_merge = new_m != -float('inf')
            safe_m = tl.where(valid_merge, new_m, 0.0)
            old_weight = tl.where(combined_m != -float('inf'), tl.exp2(combined_m - safe_m), 0.0)
            peer_weight = tl.where(peer_m != -float('inf'), tl.exp2(peer_m - safe_m), 0.0)
            combined_acc = combined_acc * old_weight[:, None] + peer_acc * peer_weight[:, None]
            combined_l = combined_l * old_weight + peer_l * peer_weight
            combined_m = new_m
        safe_denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / safe_denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(safe_denom) + combined_m, -float('inf'))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if MAX_GROUPS == 1:
            tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
        else:
            tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + owned_hq * SO_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
            tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH, combined_lse, mask=valid_owned)
        tl.debug_barrier()
    tle.distributed_barrier(mesh)
_cluster_fulltail__TILE_N = tl.constexpr(64)

@triton.jit
def _cluster_fulltail__bf16_decode_static_fulltail_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, mesh: tl.constexpr, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, 'cluster_x')
    logical_cluster = cta // CLUSTER_SIZE
    group = logical_cluster % MAX_GROUPS
    sequence = logical_cluster // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group >= group_count:
        return
    chunk = group * CLUSTER_SIZE + rank
    chunk_start = chunk * CHUNK_TOKENS
    has_work = chunk < num_chunks
    chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
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
    partial_acc = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    partial_lse = tle.gpu.alloc([Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc([NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_lse = tle.gpu.alloc([NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
    tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    while page_iter < tail_start_page:
        page_index = page_iter
        start = page_index * _cluster_fulltail__TILE_N
        phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_wgmma = k_raw
        v_wgmma = v_raw
        k_page = k_wgmma
        v_page = v_wgmma
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_fulltail__TILE_N))
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
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
        phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
        tle.gpu.copy(K_DESC, k_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw, [1, 1, _cluster_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
        tle.gpu.barrier_wait(k_full[0], phaseIdx=page_iter)
        tle.gpu.barrier_wait(v_full[0], phaseIdx=page_iter)
        k_wgmma = k_raw
        v_wgmma = v_raw
        k_page = k_wgmma
        v_page = v_wgmma
        global_n = chunk_start + start + offs_n
        token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
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
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
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
        combined_lse = tl.full((NUM_SEQ_Q_PAD,), -float('inf'), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_lse_md = tle.remote(partial_lse, peer, scope=mesh)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D)), out_cols)), mask=valid_owned[:, None], other=0.0).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_md, (source_rows,)), mask=valid_owned, other=-float('inf'))
            new_lse = tl.maximum(combined_lse, peer_lse)
            valid_merge = new_lse != -float('inf')
            safe = tl.where(valid_merge, new_lse, 0.0)
            w0 = tl.where(combined_lse != -float('inf'), tl.exp2(combined_lse - safe), 0.0)
            w1 = tl.where(peer_lse != -float('inf'), tl.exp2(peer_lse - safe), 0.0)
            denom = w0 + w1
            combined = tl.where(valid_merge[:, None], (combined * w0[:, None] + peer_acc * w1[:, None]) / tl.where(denom[:, None] > 0.0, denom[:, None], 1.0), 0.0)
            combined_lse = tl.where(valid_merge, tl.log2(tl.where(denom > 0.0, denom, 1.0)) + safe, -float('inf'))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if MAX_GROUPS == 1:
            tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
        else:
            tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + owned_hq * SO_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
            tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH, combined_lse, mask=valid_owned)
        tl.debug_barrier()
    tle.distributed_barrier(mesh)
_cluster_pipeline__TILE_N = tl.constexpr(64)
_cluster_pipeline__TMA_STAGES = tl.constexpr(2)

@triton.jit
def _cluster_pipeline__bf16_decode_cluster_wide_pipeline_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, mesh: tl.constexpr, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, 'cluster_x')
    logical_cluster = cta // CLUSTER_SIZE
    group = logical_cluster % MAX_GROUPS
    sequence = logical_cluster // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group >= group_count:
        return
    chunk = group * CLUSTER_SIZE + rank
    chunk_start = chunk * CHUNK_TOKENS
    has_work = chunk < num_chunks
    chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
    num_pages = (chunk_len + _cluster_pipeline__TILE_N - 1) // _cluster_pipeline__TILE_N
    pipeline_pages = tl.maximum(num_pages, 1)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc([_cluster_pipeline__TMA_STAGES, _cluster_pipeline__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_smem = tle.gpu.alloc([_cluster_pipeline__TMA_STAGES, _cluster_pipeline__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([Q_ROWS, _cluster_pipeline__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=_cluster_pipeline__TMA_STAGES, expect_bytes=_cluster_pipeline__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=_cluster_pipeline__TMA_STAGES, expect_bytes=_cluster_pipeline__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    partial_lse = tle.gpu.alloc([Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc([NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_lse = tle.gpu.alloc([NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    safe_start = tl.where(has_work, chunk_start // BLOCK_SIZE, 0)
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + safe_start + bid_offs, mask=bid_offs < pipeline_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < pipeline_pages)
    tl.debug_barrier()
    pipeline_start = tl.where(has_work, chunk_start, 0)
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0])
    tle.gpu.copy(V_DESC, v_smem.slot(0), [1, 1, _cluster_pipeline__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0])
    if pipeline_pages > 1:
        phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
        tle.gpu.copy(K_DESC, k_smem.slot(1), [1, 1, _cluster_pipeline__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1])
        tle.gpu.copy(V_DESC, v_smem.slot(1), [1, 1, _cluster_pipeline__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1])
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    k_wgmma = k_smem.slot(0)
    k_page = k_wgmma
    qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
    qk = tle.gpu.wgmma_wait(0, qk) * scale
    global_n = pipeline_start + offs_n
    token_valid = global_n < total_len
    query_pos = total_len - NUM_SEQ_Q + seq_m
    mask0 = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
    qk = tl.where(mask0, qk, -float('inf'))
    m_i = tl.max(qk, axis=1)
    safe_m = tl.where(m_i != -float('inf'), m_i, 0.0)
    p0 = tl.where(mask0, tl.exp2(qk - safe_m[:, None]), 0.0)
    l_i = tl.sum(p0, axis=1)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if pipeline_pages > 2:
        phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0])
    page = 1
    while page < pipeline_pages:
        slot = page % _cluster_pipeline__TMA_STAGES
        phase = page // _cluster_pipeline__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_wgmma = k_smem.slot(slot)
        k_page = k_wgmma
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        global_n = pipeline_start + page * _cluster_pipeline__TILE_N + offs_n
        token_valid = global_n < total_len
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        qk = tl.where(score_mask, qk, -float('inf'))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float('inf'), m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_wgmma = v_smem.slot(prev_slot)
        v_prev = v_wgmma
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline__TMA_STAGES
        if next_k < pipeline_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(K_DESC, k_smem.slot(slot), [1, 1, _cluster_pipeline__TILE_N, D], [next_k_phys, hkv, 0, 0], barrier=k_full[slot])
        next_v = prev_page + _cluster_pipeline__TMA_STAGES
        if next_v < pipeline_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(V_DESC, v_smem.slot(prev_slot), [1, 1, _cluster_pipeline__TILE_N, D], [next_v_phys, hkv, 0, 0], barrier=v_full[prev_slot])
        page += 1
    last_page = pipeline_pages - 1
    last_slot = last_page % _cluster_pipeline__TMA_STAGES
    last_phase = last_page // _cluster_pipeline__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_wgmma = v_smem.slot(last_slot)
    v_last = v_wgmma
    pv_last = tle.gpu.wgmma(p_smem, v_last, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    has_value = l_i > 0.0
    acc = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
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
        combined_lse = tl.full((NUM_SEQ_Q_PAD,), -float('inf'), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_lse_md = tle.remote(partial_lse, peer, scope=mesh)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (tl.broadcast_to(source_rows[:, None], (NUM_SEQ_Q_PAD, D)), out_cols)), mask=valid_owned[:, None], other=0.0).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_md, (source_rows,)), mask=valid_owned, other=-float('inf'))
            new_lse = tl.maximum(combined_lse, peer_lse)
            valid_merge = new_lse != -float('inf')
            safe = tl.where(valid_merge, new_lse, 0.0)
            w0 = tl.where(combined_lse != -float('inf'), tl.exp2(combined_lse - safe), 0.0)
            w1 = tl.where(peer_lse != -float('inf'), tl.exp2(peer_lse - safe), 0.0)
            denom = w0 + w1
            combined = tl.where(valid_merge[:, None], (combined * w0[:, None] + peer_acc * w1[:, None]) / tl.where(denom[:, None] > 0.0, denom[:, None], 1.0), 0.0)
            combined_lse = tl.where(valid_merge, tl.log2(tl.where(denom > 0.0, denom, 1.0)) + safe, -float('inf'))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if MAX_GROUPS == 1:
            tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
        else:
            tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + owned_hq * SO_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
            tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH, combined_lse, mask=valid_owned)
        tl.debug_barrier()
    tle.distributed_barrier(mesh)
_cluster_pipeline_deferred__TILE_N = tl.constexpr(64)
_cluster_pipeline_deferred__TMA_STAGES = tl.constexpr(2)

@triton.jit
def _cluster_pipeline_deferred__bf16_decode_cluster_wide_pipeline_dsm_deferred_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, mesh: tl.constexpr, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, 'cluster_x')
    logical_cluster = cta // CLUSTER_SIZE
    group = logical_cluster % MAX_GROUPS
    sequence = logical_cluster // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group >= group_count:
        return
    chunk = group * CLUSTER_SIZE + rank
    chunk_start = chunk * CHUNK_TOKENS
    has_work = chunk < num_chunks
    chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
    num_pages = (chunk_len + _cluster_pipeline_deferred__TILE_N - 1) // _cluster_pipeline_deferred__TILE_N
    pipeline_pages = tl.maximum(num_pages, 1)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc([_cluster_pipeline_deferred__TMA_STAGES, _cluster_pipeline_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_smem = tle.gpu.alloc([_cluster_pipeline_deferred__TMA_STAGES, _cluster_pipeline_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([Q_ROWS, _cluster_pipeline_deferred__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=_cluster_pipeline_deferred__TMA_STAGES, expect_bytes=_cluster_pipeline_deferred__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=_cluster_pipeline_deferred__TMA_STAGES, expect_bytes=_cluster_pipeline_deferred__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    partial_stats = tle.gpu.alloc([2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc([NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_lse = tle.gpu.alloc([NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    safe_start = tl.where(has_work, chunk_start // BLOCK_SIZE, 0)
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + safe_start + bid_offs, mask=bid_offs < pipeline_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < pipeline_pages)
    tl.debug_barrier()
    pipeline_start = tl.where(has_work, chunk_start, 0)
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0])
    tle.gpu.copy(V_DESC, v_smem.slot(0), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0])
    if pipeline_pages > 1:
        phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
        tle.gpu.copy(K_DESC, k_smem.slot(1), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1])
        tle.gpu.copy(V_DESC, v_smem.slot(1), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1])
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    k_wgmma = k_smem.slot(0)
    k_page = k_wgmma
    qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
    qk = tle.gpu.wgmma_wait(0, qk) * scale
    global_n = pipeline_start + offs_n
    token_valid = global_n < total_len
    query_pos = total_len - NUM_SEQ_Q + seq_m
    mask0 = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
    qk = tl.where(mask0, qk, -float('inf'))
    m_i = tl.max(qk, axis=1)
    safe_m = tl.where(m_i != -float('inf'), m_i, 0.0)
    p0 = tl.where(mask0, tl.exp2(qk - safe_m[:, None]), 0.0)
    l_i = tl.sum(p0, axis=1)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if pipeline_pages > 2:
        phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0])
    page = 1
    while page < pipeline_pages:
        slot = page % _cluster_pipeline_deferred__TMA_STAGES
        phase = page // _cluster_pipeline_deferred__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline_deferred__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline_deferred__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_wgmma = k_smem.slot(slot)
        k_page = k_wgmma
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        global_n = pipeline_start + page * _cluster_pipeline_deferred__TILE_N + offs_n
        token_valid = global_n < total_len
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        qk = tl.where(score_mask, qk, -float('inf'))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float('inf'), m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_wgmma = v_smem.slot(prev_slot)
        v_prev = v_wgmma
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline_deferred__TMA_STAGES
        if next_k < pipeline_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(K_DESC, k_smem.slot(slot), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [next_k_phys, hkv, 0, 0], barrier=k_full[slot])
        next_v = prev_page + _cluster_pipeline_deferred__TMA_STAGES
        if next_v < pipeline_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(V_DESC, v_smem.slot(prev_slot), [1, 1, _cluster_pipeline_deferred__TILE_N, D], [next_v_phys, hkv, 0, 0], barrier=v_full[prev_slot])
        page += 1
    last_page = pipeline_pages - 1
    last_slot = last_page % _cluster_pipeline_deferred__TMA_STAGES
    last_phase = last_page // _cluster_pipeline_deferred__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_wgmma = v_smem.slot(last_slot)
    v_last = v_wgmma
    pv_last = tle.gpu.wgmma(p_smem, v_last, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    valid_local = l_i > 0.0
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc, 0.0))
    tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float('inf')))
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
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float('inf'), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float('inf'))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            valid_merge = new_m != -float('inf')
            safe_m = tl.where(valid_merge, new_m, 0.0)
            old_weight = tl.where(combined_m != -float('inf'), tl.exp2(combined_m - safe_m), 0.0)
            peer_weight = tl.where(peer_m != -float('inf'), tl.exp2(peer_m - safe_m), 0.0)
            combined_acc = combined_acc * old_weight[:, None] + peer_acc * peer_weight[:, None]
            combined_l = combined_l * old_weight + peer_l * peer_weight
            combined_m = new_m
        safe_denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / safe_denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(safe_denom) + combined_m, -float('inf'))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if MAX_GROUPS == 1:
            tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
        else:
            tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + owned_hq * SO_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
            tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH, combined_lse, mask=valid_owned)
        tl.debug_barrier()
    tle.distributed_barrier(mesh)
_cluster_pipeline_fulltail_deferred__TILE_N = tl.constexpr(64)
_cluster_pipeline_fulltail_deferred__TMA_STAGES = tl.constexpr(2)

@triton.jit
def _cluster_pipeline_fulltail_deferred__bf16_decode_cluster_wide_pipeline_fulltail_dsm_deferred_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, mesh: tl.constexpr, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    cta = tl.program_id(0)
    rank = tle.shard_id(mesh, 'cluster_x')
    logical_cluster = cta // CLUSTER_SIZE
    group = logical_cluster % MAX_GROUPS
    sequence = logical_cluster // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group >= group_count:
        return
    chunk = group * CLUSTER_SIZE + rank
    chunk_start = chunk * CHUNK_TOKENS
    has_work = chunk < num_chunks
    chunk_len = tl.where(has_work, tl.minimum(CHUNK_TOKENS, total_len - chunk_start), 0)
    num_pages = (chunk_len + _cluster_pipeline_fulltail_deferred__TILE_N - 1) // _cluster_pipeline_fulltail_deferred__TILE_N
    pipeline_pages = tl.maximum(num_pages, 1)
    is_last_chunk = has_work & (chunk_start + chunk_len >= total_len)
    last_page = pipeline_pages - 1
    causal_start_page = (total_len - NUM_SEQ_Q - chunk_start) // _cluster_pipeline_fulltail_deferred__TILE_N
    tail_start_page = tl.where(is_last_chunk, tl.maximum(causal_start_page, 0), last_page)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_smem = tle.gpu.alloc([_cluster_pipeline_fulltail_deferred__TMA_STAGES, _cluster_pipeline_fulltail_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_smem = tle.gpu.alloc([_cluster_pipeline_fulltail_deferred__TMA_STAGES, _cluster_pipeline_fulltail_deferred__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=_cluster_pipeline_fulltail_deferred__TMA_STAGES, expect_bytes=_cluster_pipeline_fulltail_deferred__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=_cluster_pipeline_fulltail_deferred__TMA_STAGES, expect_bytes=_cluster_pipeline_fulltail_deferred__TILE_N * D * 2)
    partial_acc = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    partial_stats = tle.gpu.alloc([2 * Q_ROWS], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_acc = tle.gpu.alloc([NUM_SEQ_Q_PAD, D], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    merged_lse = tle.gpu.alloc([NUM_SEQ_Q_PAD], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(tle.gpu.local_ptr(q_smem, (q_rows, q_cols)), q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    safe_start = tl.where(has_work, chunk_start // BLOCK_SIZE, 0)
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + safe_start + bid_offs, mask=bid_offs < pipeline_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < pipeline_pages)
    tl.debug_barrier()
    pipeline_start = tl.where(has_work, chunk_start, 0)
    phys0 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (0,)))
    tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [phys0, hkv, 0, 0], barrier=k_full[0])
    tle.gpu.copy(V_DESC, v_smem.slot(0), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [phys0, hkv, 0, 0], barrier=v_full[0])
    if pipeline_pages > 1:
        phys1 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (1,)))
        tle.gpu.copy(K_DESC, k_smem.slot(1), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [phys1, hkv, 0, 0], barrier=k_full[1])
        tle.gpu.copy(V_DESC, v_smem.slot(1), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [phys1, hkv, 0, 0], barrier=v_full[1])
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    scale = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    tle.gpu.barrier_wait(k_full[0], phaseIdx=0)
    k_wgmma = k_smem.slot(0)
    k_page = k_wgmma
    qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
    qk = tle.gpu.wgmma_wait(0, qk) * scale
    query_pos = total_len - NUM_SEQ_Q + seq_m
    page0_is_tail = is_last_chunk & (tail_start_page == 0)
    if page0_is_tail:
        global_n = pipeline_start + offs_n
        token_valid = global_n < total_len
        mask0 = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
    else:
        mask0 = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
    qk = tl.where(mask0, qk, -float('inf'))
    m_i = tl.max(qk, axis=1)
    safe_m = tl.where(m_i != -float('inf'), m_i, 0.0)
    p0 = tl.where(mask0, tl.exp2(qk - safe_m[:, None]), 0.0)
    l_i = tl.sum(p0, axis=1)
    tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p0.to(tl.bfloat16))
    if pipeline_pages > 2:
        phys2 = tl.load(tle.gpu.local_ptr(chunk_block_ids, (2,)))
        tle.gpu.copy(K_DESC, k_smem.slot(0), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [phys2, hkv, 0, 0], barrier=k_full[0])
    page = 1
    while page < tail_start_page:
        slot = page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        phase = page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_wgmma = k_smem.slot(slot)
        k_page = k_wgmma
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
        qk = tl.where(score_mask, qk, -float('inf'))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float('inf'), m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_wgmma = v_smem.slot(prev_slot)
        v_prev = v_wgmma
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_k < pipeline_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(K_DESC, k_smem.slot(slot), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [next_k_phys, hkv, 0, 0], barrier=k_full[slot])
        next_v = prev_page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_v < pipeline_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(V_DESC, v_smem.slot(prev_slot), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [next_v_phys, hkv, 0, 0], barrier=v_full[prev_slot])
        page += 1
    while page < pipeline_pages:
        slot = page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        phase = page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_page = page - 1
        prev_slot = prev_page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
        prev_phase = prev_page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        k_wgmma = k_smem.slot(slot)
        k_page = k_wgmma
        qk = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        qk = tle.gpu.wgmma_wait(0, qk) * scale
        if is_last_chunk:
            global_n = pipeline_start + page * _cluster_pipeline_fulltail_deferred__TILE_N + offs_n
            token_valid = global_n < total_len
            score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        else:
            score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _cluster_pipeline_fulltail_deferred__TILE_N))
        qk = tl.where(score_mask, qk, -float('inf'))
        page_max = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float('inf'), m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_new), 0.0)
        p_curr = tl.where(score_mask, tl.exp2(qk - safe_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p_curr, axis=1)
        m_i = m_new
        tle.gpu.barrier_wait(v_full[prev_slot], phaseIdx=prev_phase)
        v_wgmma = v_smem.slot(prev_slot)
        v_prev = v_wgmma
        pv = tle.gpu.wgmma(p_smem, v_prev, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = (acc + pv) * alpha[:, None]
        tl.store(tle.gpu.local_ptr(p_smem, (p_rows, p_cols)), p_curr.to(tl.bfloat16))
        next_k = page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_k < pipeline_pages:
            next_k_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_k,)))
            tle.gpu.copy(K_DESC, k_smem.slot(slot), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [next_k_phys, hkv, 0, 0], barrier=k_full[slot])
        next_v = prev_page + _cluster_pipeline_fulltail_deferred__TMA_STAGES
        if next_v < pipeline_pages:
            next_v_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_v,)))
            tle.gpu.copy(V_DESC, v_smem.slot(prev_slot), [1, 1, _cluster_pipeline_fulltail_deferred__TILE_N, D], [next_v_phys, hkv, 0, 0], barrier=v_full[prev_slot])
        page += 1
    last_slot = last_page % _cluster_pipeline_fulltail_deferred__TMA_STAGES
    last_phase = last_page // _cluster_pipeline_fulltail_deferred__TMA_STAGES
    tle.gpu.barrier_wait(v_full[last_slot], phaseIdx=last_phase)
    v_wgmma = v_smem.slot(last_slot)
    v_last = v_wgmma
    pv_last = tle.gpu.wgmma(p_smem, v_last, out_dtype=tl.float32)
    pv_last = tle.gpu.wgmma_wait(0, pv_last)
    acc += pv_last
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    valid_local = l_i > 0.0
    tl.store(tle.gpu.local_ptr(partial_acc, (q_rows, q_cols)), tl.where(valid_local[:, None], acc, 0.0))
    tl.store(tle.gpu.local_ptr(partial_stats, (offs_r,)), tl.where(valid_local, m_i, -float('inf')))
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
        combined_m = tl.full((NUM_SEQ_Q_PAD,), -float('inf'), tl.float32)
        combined_l = tl.zeros((NUM_SEQ_Q_PAD,), tl.float32)
        for peer in tl.static_range(0, CLUSTER_SIZE):
            peer_acc_md = tle.remote(partial_acc, peer, scope=mesh)
            peer_stats_md = tle.remote(partial_stats, peer, scope=mesh)
            peer_m = tl.load(tle.gpu.local_ptr(peer_stats_md, (source_rows,)), mask=valid_owned, other=-float('inf'))
            peer_l = tl.load(tle.gpu.local_ptr(peer_stats_md, (Q_ROWS + source_rows,)), mask=valid_owned, other=0.0)
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_md, (source_rows_2d, out_cols)), mask=valid_owned[:, None], other=0.0).to(tl.float32)
            new_m = tl.maximum(combined_m, peer_m)
            valid_merge = new_m != -float('inf')
            safe_m = tl.where(valid_merge, new_m, 0.0)
            old_weight = tl.where(combined_m != -float('inf'), tl.exp2(combined_m - safe_m), 0.0)
            peer_weight = tl.where(peer_m != -float('inf'), tl.exp2(peer_m - safe_m), 0.0)
            combined_acc = combined_acc * old_weight[:, None] + peer_acc * peer_weight[:, None]
            combined_l = combined_l * old_weight + peer_l * peer_weight
            combined_m = new_m
        safe_denom = tl.where(combined_l > 0.0, combined_l, 1.0)
        combined = tl.where(valid_owned[:, None] & (combined_l[:, None] > 0.0), combined_acc / safe_denom[:, None], 0.0)
        combined_lse = tl.where(valid_owned & (combined_l > 0.0), tl.log2(safe_denom) + combined_m, -float('inf'))
        tl.store(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)), combined)
        tl.store(tle.gpu.local_ptr(merged_lse, (offs_m,)), combined_lse)
        tl.debug_barrier()
        combined = tl.load(tle.gpu.local_ptr(merged_acc, (out_rows, out_cols)))
        combined_lse = tl.load(tle.gpu.local_ptr(merged_lse, (offs_m,)))
        if MAX_GROUPS == 1:
            tl.store(OUT + batch * O_SB + offs_m[:, None] * O_SM + owned_hq * O_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
        else:
            tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[:, None] * SO_SM + owned_hq * SO_SH + offs_d[None, :], combined, mask=valid_owned[:, None])
            tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m * SL_SM + owned_hq * SL_SH, combined_lse, mask=valid_owned)
        tl.debug_barrier()
    tle.distributed_barrier(mesh)
_direct_fulltail__TILE_N = tl.constexpr(64)

@triton.jit
def _direct_fulltail__bf16_decode_direct_fulltail_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr, TMA_STAGES: tl.constexpr=1):
    cta = tl.program_id(0)
    group = cta % MAX_GROUPS
    sequence = cta // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    if group >= num_chunks:
        return
    chunk_start = group * CHUNK_TOKENS
    chunk_len = tl.minimum(CHUNK_TOKENS, total_len - chunk_start)
    num_pages = (chunk_len + _direct_fulltail__TILE_N - 1) // _direct_fulltail__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    causal_start_page = (total_len - NUM_SEQ_Q - chunk_start) // _direct_fulltail__TILE_N
    tail_start_page = tl.where(is_last_chunk, tl.maximum(causal_start_page, 0), num_pages)
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_direct_fulltail__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([TMA_STAGES, _direct_fulltail__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([TMA_STAGES, _direct_fulltail__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_fulltail__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_fulltail__TILE_N * D * 2)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
    tl.debug_barrier()
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    if TMA_STAGES == 2:
        first_page = 0
        first_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (first_page,)))
        tle.gpu.copy(K_DESC, k_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [first_phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [first_phys, hkv, 0, 0], barrier=v_full[0])
    while page_iter < tail_start_page:
        page_index = page_iter
        start = page_index * _direct_fulltail__TILE_N
        if TMA_STAGES == 1:
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
            tle.gpu.copy(K_DESC, k_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            tle.gpu.copy(V_DESC, v_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
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
                next_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_page,)))
                tle.gpu.copy(K_DESC, k_raw.slot(next_slot), [1, 1, _direct_fulltail__TILE_N, D], [next_phys, hkv, 0, 0], barrier=k_full[next_slot])
                tle.gpu.copy(V_DESC, v_raw.slot(next_slot), [1, 1, _direct_fulltail__TILE_N, D], [next_phys, hkv, 0, 0], barrier=v_full[next_slot])
        k_wgmma = k_raw.slot(slot)
        v_wgmma = v_raw.slot(slot)
        k_page = k_wgmma
        v_page = v_wgmma
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = tl.broadcast_to(valid_row[:, None], (Q_ROWS, _direct_fulltail__TILE_N))
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
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
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
            tle.gpu.copy(K_DESC, k_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            tle.gpu.copy(V_DESC, v_raw.slot(0), [1, 1, _direct_fulltail__TILE_N, D], [phys, hkv, 0, 0], barrier=v_full[0])
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
                next_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_page,)))
                tle.gpu.copy(K_DESC, k_raw.slot(next_slot), [1, 1, _direct_fulltail__TILE_N, D], [next_phys, hkv, 0, 0], barrier=k_full[next_slot])
                tle.gpu.copy(V_DESC, v_raw.slot(next_slot), [1, 1, _direct_fulltail__TILE_N, D], [next_phys, hkv, 0, 0], barrier=v_full[next_slot])
        k_wgmma = k_raw.slot(slot)
        v_wgmma = v_raw.slot(slot)
        k_page = k_wgmma
        v_page = v_wgmma
        global_n = chunk_start + start + offs_n
        token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = valid_row[:, None] & token_valid[None, :] & (global_n[None, :] <= query_pos[:, None])
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
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
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
    if MAX_GROUPS == 1:
        tl.store(OUT + batch * O_SB + seq_m[:, None] * O_SM + hq[:, None] * O_SH + offs_d[None, :], acc_rows, mask=valid_row[:, None])
    else:
        tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :], acc_rows, mask=valid_row[:, None])
        tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH, lse, mask=valid_row)
_direct_narrow__TILE_N = tl.constexpr(64)

@triton.jit
def _direct_narrow__bf16_decode_direct_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, KV_LENS, SPLIT_OUT, SPLIT_LSE, OUT, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr, TMA_STAGES: tl.constexpr=1):
    cta = tl.program_id(0)
    group = cta % MAX_GROUPS
    sequence = cta // MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    if group >= num_chunks:
        return
    chunk_start = group * CHUNK_TOKENS
    chunk_len = tl.minimum(CHUNK_TOKENS, total_len - chunk_start)
    num_pages = (chunk_len + _direct_narrow__TILE_N - 1) // _direct_narrow__TILE_N
    is_last_chunk = chunk_start + chunk_len >= total_len
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_direct_narrow__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([TMA_STAGES, _direct_narrow__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([TMA_STAGES, _direct_narrow__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_narrow__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=_direct_narrow__TILE_N * D * 2)
    chunk_block_ids = tle.gpu.alloc([CHUNK_TOKENS // BLOCK_SIZE], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
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
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, CHUNK_TOKENS // BLOCK_SIZE)
    num_pages = (chunk_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + chunk_start // BLOCK_SIZE + bid_offs, mask=bid_offs < num_pages, other=0)
    tl.store(tle.gpu.local_ptr(chunk_block_ids, (bid_offs,)), bids, mask=bid_offs < num_pages)
    tl.debug_barrier()
    acc = tl.zeros((D, Q_ROWS), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    page_iter = 0
    if TMA_STAGES == 2:
        first_page = tl.where(is_last_chunk, num_pages - 1, 0)
        first_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (first_page,)))
        tle.gpu.copy(K_DESC, k_raw.slot(0), [1, 1, _direct_narrow__TILE_N, D], [first_phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(V_DESC, v_raw.slot(0), [1, 1, _direct_narrow__TILE_N, D], [first_phys, hkv, 0, 0], barrier=v_full[0])
    while page_iter < num_pages:
        page_index = tl.where(is_last_chunk, tl.where(page_iter == 0, num_pages - 1, page_iter - 1), page_iter)
        start = page_index * _direct_narrow__TILE_N
        if TMA_STAGES == 1:
            phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (page_index,)))
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
                next_phys = tl.load(tle.gpu.local_ptr(chunk_block_ids, (next_page,)))
                tle.gpu.copy(K_DESC, k_raw.slot(next_slot), [1, 1, _direct_narrow__TILE_N, D], [next_phys, hkv, 0, 0], barrier=k_full[next_slot])
                tle.gpu.copy(V_DESC, v_raw.slot(next_slot), [1, 1, _direct_narrow__TILE_N, D], [next_phys, hkv, 0, 0], barrier=v_full[next_slot])
        k_wgmma = k_raw.slot(slot)
        v_wgmma = v_raw.slot(slot)
        k_page = k_wgmma
        v_page = v_wgmma
        global_n = chunk_start + start + offs_n
        token_valid = global_n < total_len
        query_pos = total_len - NUM_SEQ_Q + seq_m
        scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        score_mask = token_valid[:, None] & valid_row[None, :] & (global_n[:, None] <= query_pos[None, :])
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, page_max)
        active = m_new != -float('inf')
        safe_m = tl.where(active, m_new, 0.0)
        alpha = tl.where(m_i != -float('inf'), tl.exp2(m_i - safe_m), 0.0)
        p = tl.exp2(scores - safe_m[None, :])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        tl.store(p_ptr, p)
        v_rows = tl.broadcast_to(offs_n[:, None], (_direct_narrow__TILE_N, D))
        v_cols = tl.broadcast_to(offs_d[None, :], (_direct_narrow__TILE_N, D))
        v_regs = tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols)))
        v_regs_t = tl.trans(v_regs)
        v_wgmma_view = tle.gpu.alloc([D, _direct_narrow__TILE_N], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, init_value=v_regs_t)
        pv = tle.gpu.wgmma(v_wgmma_view, p_smem, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[None, :] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    acc = tl.where(has_value[None, :], acc / l_i[None, :], 0.0)
    acc_rows = tl.trans(acc)
    lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
    if MAX_GROUPS == 1:
        tl.store(OUT + batch * O_SB + seq_m[:, None] * O_SM + hq[:, None] * O_SH + offs_d[None, :], acc_rows, mask=valid_row[:, None])
    else:
        tl.store(SPLIT_OUT + batch * SO_SB + group * SO_SG + seq_m[:, None] * SO_SM + hq[:, None] * SO_SH + offs_d[None, :], acc_rows, mask=valid_row[:, None])
        tl.store(SPLIT_LSE + batch * SL_SB + group * SL_SG + seq_m * SL_SM + hq * SL_SH, lse, mask=valid_row)

@triton.jit
def _finalizer__bf16_decode_finalize_multihead_kernel(SPLIT_OUT, SPLIT_LSE, KV_LENS, OUT, NUM_SEQ_Q: tl.constexpr, NUM_SEQ_Q_PAD: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_PROGRAM: tl.constexpr, D: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, MAX_GROUPS: tl.constexpr, SO_SB: tl.constexpr, SO_SG: tl.constexpr, SO_SM: tl.constexpr, SO_SH: tl.constexpr, SL_SB: tl.constexpr, SL_SG: tl.constexpr, SL_SM: tl.constexpr, SL_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
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
    valid_g = offs_g < groups
    lse = tl.load(SPLIT_LSE + batch * SL_SB + offs_g[:, None, None] * SL_SG + offs_m[None, None, :] * SL_SM + offs_h[None, :, None] * SL_SH, mask=valid_g[:, None, None] & valid_h[None, :, None] & valid_m[None, None, :], other=-float('inf'))
    max_lse = tl.max(lse, axis=0)
    safe_lse = tl.where(max_lse != -float('inf'), max_lse, 0.0)
    weights = tl.where(valid_g[:, None, None], tl.exp2(lse - safe_lse[None, :, :]), 0.0)
    denominator = tl.sum(weights, axis=0)
    acc = tl.zeros((HEADS_PER_PROGRAM, NUM_SEQ_Q_PAD, D), tl.float32)
    for group in tl.static_range(0, MAX_GROUPS):
        group_valid = group < groups
        group_lse = tl.load(SPLIT_LSE + batch * SL_SB + group * SL_SG + offs_m[None, :] * SL_SM + offs_h[:, None] * SL_SH, mask=group_valid & valid_h[:, None] & valid_m[None, :], other=-float('inf'))
        group_weight = tl.where(group_valid & valid_h[:, None] & valid_m[None, :], tl.exp2(group_lse - safe_lse), 0.0)
        partial = tl.load(SPLIT_OUT + batch * SO_SB + group * SO_SG + offs_m[None, :, None] * SO_SM + offs_h[:, None, None] * SO_SH + offs_d[None, None, :], mask=group_valid & valid_h[:, None, None] & valid_m[None, :, None], other=0.0)
        acc += partial * group_weight[:, :, None]
    safe_denom = tl.where(denominator > 0.0, denominator, 1.0)
    acc /= safe_denom[:, :, None]
    tl.store(OUT + batch * O_SB + offs_m[None, :, None] * O_SM + offs_h[:, None, None] * O_SH + offs_d[None, None, :], acc, mask=valid_h[:, None, None] & valid_m[None, :, None])
_uniform512__TILE_N = tl.constexpr(64)
_uniform512__STATIC_PAGES = tl.constexpr(8)
_uniform512__TMA_STAGES = tl.constexpr(2)

@triton.jit
def _uniform512__bf16_decode_uniform512_delayed_v_kernel(Q, K_DESC, V_DESC, BLOCK_IDS, OUT, B: tl.constexpr, NUM_SEQ_Q: tl.constexpr, Q_ROWS: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, MAX_BLOCKS: tl.constexpr, Q_SB: tl.constexpr, Q_SM: tl.constexpr, Q_SH: tl.constexpr, O_SB: tl.constexpr, O_SM: tl.constexpr, O_SH: tl.constexpr):
    task = tl.program_id(0)
    batch = task % B
    hkv = task // B
    q_smem = tle.gpu.alloc([Q_ROWS, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_uniform512__TILE_N, Q_ROWS], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_raw = tle.gpu.alloc([_uniform512__TMA_STAGES, _uniform512__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    v_raw = tle.gpu.alloc([_uniform512__TMA_STAGES, _uniform512__TILE_N, D], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=_uniform512__TMA_STAGES, arrive_count=1, expect_bytes=_uniform512__TILE_N * D * 2)
    v_full = tle.gpu.alloc_barriers(num_barriers=_uniform512__TMA_STAGES, arrive_count=1, expect_bytes=_uniform512__TILE_N * D * 2)
    block_ids_smem = tle.gpu.alloc([16], dtype=tl.int32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_r = tl.arange(0, Q_ROWS)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, _uniform512__TILE_N)
    seq_m = offs_r // HEADS_PER_GROUP
    h_in_group = offs_r - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_row = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, D))
    q_cols = tl.broadcast_to(offs_d[None, :], (Q_ROWS, D))
    q_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    q = tl.load(Q + batch * Q_SB + seq_m[:, None] * Q_SM + hq[:, None] * Q_SH + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    tl.store(q_ptr, q)
    bid_offs = tl.arange(0, 16)
    bids = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + bid_offs, mask=bid_offs < 8, other=0)
    tl.store(tle.gpu.local_ptr(block_ids_smem, (bid_offs,)), bids, mask=bid_offs < 8)
    tl.debug_barrier()
    qk_scale_log2 = 1.4426950408889634 * tl.rsqrt(tl.full((), D, tl.float32))
    acc = tl.zeros((Q_ROWS, D), tl.float32)
    m_i = tl.full((Q_ROWS,), -float('inf'), tl.float32)
    l_i = tl.zeros((Q_ROWS,), tl.float32)
    first_phys = tl.load(tle.gpu.local_ptr(block_ids_smem, (7,)))
    tle.gpu.copy(K_DESC, k_raw.slot(0), [1, 1, _uniform512__TILE_N, D], [first_phys, hkv, 0, 0], barrier=k_full[0])
    tle.gpu.copy(V_DESC, v_raw.slot(0), [1, 1, _uniform512__TILE_N, D], [first_phys, hkv, 0, 0], barrier=v_full[0])
    page_iter = 0
    while page_iter < _uniform512__STATIC_PAGES:
        slot = page_iter % _uniform512__TMA_STAGES
        phase = page_iter // _uniform512__TMA_STAGES
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=phase)
        next_iter = page_iter + 1
        if next_iter < _uniform512__STATIC_PAGES:
            next_slot = next_iter % _uniform512__TMA_STAGES
            next_phys = tl.load(tle.gpu.local_ptr(block_ids_smem, (next_iter - 1,)))
            tle.gpu.copy(K_DESC, k_raw.slot(next_slot), [1, 1, _uniform512__TILE_N, D], [next_phys, hkv, 0, 0], barrier=k_full[next_slot])
            tle.gpu.copy(V_DESC, v_raw.slot(next_slot), [1, 1, _uniform512__TILE_N, D], [next_phys, hkv, 0, 0], barrier=v_full[next_slot])
        k_wgmma = k_raw.slot(slot)
        k_page = k_wgmma
        scores = tle.gpu.wgmma(q_smem, k_page, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores) * qk_scale_log2
        tail_limit = _uniform512__TILE_N - NUM_SEQ_Q + seq_m
        score_mask = valid_row[:, None] & ((page_iter != 0) | (offs_n[None, :] <= tail_limit[:, None]))
        scores = tl.where(score_mask, scores, -float('inf'))
        page_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, page_max)
        safe_new = tl.where(m_new != -float('inf'), m_new, 0.0)
        alpha = tl.exp2(m_i - safe_new)
        p = tl.exp2(scores - safe_new[:, None])
        p = tl.where(score_mask, p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        wide_p_rows = tl.broadcast_to(offs_r[:, None], (Q_ROWS, _uniform512__TILE_N))
        wide_p_cols = tl.broadcast_to(offs_n[None, :], (Q_ROWS, _uniform512__TILE_N))
        tl.store(tle.gpu.local_ptr(p_smem, (wide_p_rows, wide_p_cols)), p)
        tle.gpu.barrier_wait(v_full[slot], phaseIdx=phase)
        v_wgmma = v_raw.slot(slot)
        v_page = v_wgmma
        pv = tle.gpu.wgmma(p_smem, v_page, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        l_i = l_new
        page_iter += 1
    has_value = l_i > 0.0
    result = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
    tl.store(OUT + batch * O_SB + seq_m[:, None] * O_SM + hq[:, None] * O_SH + offs_d[None, :], result, mask=valid_row[:, None])
_bf16_entry__HEAD_DIM = 128
_bf16_entry__BLOCK_SIZE = 64
_bf16_entry__TILE_N = 64
_bf16_entry__OFFICIAL_CASES = {'uniform_512': (512,) * 64, 'uniform_4096': (4096,) * 64, 'skewed_mix': (128,) * 32 + (4096,) * 32, 'skewed_extreme': (64,) * 15 + (16 * 1024,), 'one_64k_7x4k': (64 * 1024,) + (4096,) * 7, 'one_64k_15x4k': (64 * 1024,) + (4096,) * 15, 'one_64k_31x4k': (64 * 1024,) + (4096,) * 31, 'one_128k_31x4k': (128 * 1024,) + (4096,) * 31, 'two_32k_30x4k': (32 * 1024,) * 2 + (4096,) * 30}
_bf16_entry___FINAL_CT = {1: {'uniform_512': {'NHD': (1, 4096), 'HND': (1, 1024)}, 'uniform_4096': {'NHD': (1, 1024), 'HND': (1, 1024)}, 'skewed_mix': {'NHD': (2, 1024), 'HND': (2, 1024)}, 'skewed_extreme': {'NHD': (4, 256), 'HND': (4, 256)}, 'one_64k_7x4k': {'NHD': (4, 512), 'HND': (4, 512)}, 'one_64k_15x4k': {'NHD': (8, 1024), 'HND': (8, 1024)}, 'one_64k_31x4k': {'NHD': (2, 1024), 'HND': (2, 1024)}, 'one_128k_31x4k': {'NHD': (4, 1024), 'HND': (4, 1024)}, 'two_32k_30x4k': {'NHD': (2, 1024), 'HND': (2, 1024)}}, 2: {'uniform_512': {'NHD': (1, 1024), 'HND': (1, 4096)}, 'uniform_4096': {'NHD': (1, 1024), 'HND': (1, 1024)}, 'skewed_mix': {'NHD': (2, 512), 'HND': (2, 512)}, 'skewed_extreme': {'NHD': (4, 256), 'HND': (2, 256)}, 'one_64k_7x4k': {'NHD': (4, 512), 'HND': (4, 512)}, 'one_64k_15x4k': {'NHD': (8, 1024), 'HND': (8, 1024)}, 'one_64k_31x4k': {'NHD': (2, 1024), 'HND': (2, 1024)}, 'one_128k_31x4k': {'NHD': (4, 1024), 'HND': (4, 1024)}, 'two_32k_30x4k': {'NHD': (2, 1024), 'HND': (2, 1024)}}, 3: {'uniform_512': {'NHD': (1, 1024), 'HND': (1, 1024)}, 'uniform_4096': {'NHD': (1, 1024), 'HND': (1, 1024)}, 'skewed_mix': {'NHD': (2, 1024), 'HND': (2, 1024)}, 'skewed_extreme': {'NHD': (8, 256), 'HND': (8, 256)}, 'one_64k_7x4k': {'NHD': (8, 512), 'HND': (8, 512)}, 'one_64k_15x4k': {'NHD': (8, 1024), 'HND': (4, 1024)}, 'one_64k_31x4k': {'NHD': (4, 1024), 'HND': (4, 1024)}, 'one_128k_31x4k': {'NHD': (8, 1024), 'HND': (8, 1024)}, 'two_32k_30x4k': {'NHD': (2, 1024), 'HND': (2, 1024)}}}
_bf16_entry___NARROW = {(1, 'uniform_512', 'NHD'), (2, 'uniform_512', 'NHD'), (2, 'uniform_512', 'HND'), (3, 'uniform_512', 'NHD'), (3, 'uniform_512', 'HND')}
_bf16_entry___FULLTAIL = {1: {'uniform_4096', 'skewed_mix', 'one_64k_15x4k', 'one_64k_31x4k', 'one_128k_31x4k'}, 2: set(_bf16_entry__OFFICIAL_CASES) - {'uniform_512', 'skewed_extreme'}, 3: set(_bf16_entry__OFFICIAL_CASES) - {'uniform_512'}}
_bf16_entry___DEFERRED = {1: {'skewed_mix', 'skewed_extreme', 'one_64k_7x4k', 'one_64k_15x4k', 'one_64k_31x4k', 'one_128k_31x4k'}, 2: {'skewed_mix', 'one_64k_7x4k', 'one_64k_15x4k', 'one_64k_31x4k', 'one_128k_31x4k', 'two_32k_30x4k'}, 3: {'skewed_mix', 'one_64k_7x4k', 'one_64k_15x4k', 'one_64k_31x4k', 'one_128k_31x4k', 'two_32k_30x4k'}}
_bf16_entry___PIPE = set(_bf16_entry__OFFICIAL_CASES) - {'uniform_512', 'uniform_4096', 'skewed_extreme'}
_bf16_entry___CLUSTER_MESHES = (
    {size: tle.device_mesh({'block_cluster': [('cluster_x', size)]})
     for size in (2, 4, 8)}
    if USE_TLE else {2: None, 4: None, 8: None}
)

@dataclass(frozen=True)
class _bf16_entry__StaticBF16Policy:
    workload: DecodeWorkload
    layout: str
    mtp: int
    cluster_size: int
    chunk_tokens: int
    kernel: str
    narrow: bool
    finalizer_heads: int

    @property
    def label(self) -> str:
        return f'c{self.cluster_size}t{self.chunk_tokens}/{self.kernel}'

@dataclass
class _bf16_entry__StaticBF16Inputs:
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
            raise ValueError('q leading dimension must equal batch * MTP')
        return int(self.q.shape[0] // self.batch)

@dataclass
class _bf16_entry__StaticBF16Workspace:
    policy: _bf16_entry__StaticBF16Policy
    q_4d: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    out: torch.Tensor
    effective_kv_lens: torch.Tensor
    max_groups: int

def _bf16_entry___validate(inputs: _bf16_entry__StaticBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ('NHD', 'HND'):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16:
        raise ValueError('q must be bfloat16')
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _bf16_entry__HEAD_DIM:
        raise ValueError('q must have shape [batch * MTP, Hq, 128]')
    if inputs.mtp not in (1, 2, 3):
        raise ValueError('final BF16 static decode supports MTP 1, 2, or 3')
    for name, cache in (('k_cache', inputs.k_cache), ('v_cache', inputs.v_cache)):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f'{name} must be bfloat16')
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[3] != 128:
            raise ValueError(f'{name} must have logical shape [block,64,Hkv,128]')
        if cache.stride(3) != 1:
            raise ValueError(f'{name} head dimension must be contiguous')
    if inputs.block_ids.ndim != 2 or inputs.block_ids.dtype != torch.int32:
        raise ValueError('block_ids must be rank-2 int32')
    if inputs.kv_lens.dtype != torch.int32:
        raise ValueError('kv_lens must be int32')
    hq, hkv = (int(inputs.q.shape[1]), int(inputs.k_cache.shape[2]))
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError('final BF16 decode requires official GQA8 heads (Hkv,Hq)=(1,8) or (4,32)')
    if hkv > 1:
        inferred_layout = 'HND' if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else 'NHD'
        if inferred_layout != inputs.layout:
            raise ValueError('explicit layout does not match K-cache strides')
        inferred_v_layout = 'HND' if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else 'NHD'
        if inferred_v_layout != inputs.layout:
            raise ValueError('explicit layout does not match V-cache strides')
    return (inputs.mtp, hq, hkv)

def _bf16_entry___classify_workload(lengths: tuple[int, ...]) -> str | None:
    count = len(lengths)
    if count == 64 and all((length == 512 for length in lengths)):
        return 'uniform_512'
    if count == 64 and all((length == 4096 for length in lengths)):
        return 'uniform_4096'
    if count == 64 and lengths.count(128) == 32 and (lengths.count(4096) == 32):
        return 'skewed_mix'
    if count == 16 and lengths.count(64) == 15 and (lengths.count(16384) == 1):
        return 'skewed_extreme'
    if lengths.count(65536) == 1 and lengths.count(4096) == count - 1:
        if count == 8:
            return 'one_64k_7x4k'
        if count == 16:
            return 'one_64k_15x4k'
        if count == 32:
            return 'one_64k_31x4k'
    if count == 32 and lengths.count(131072) == 1 and (lengths.count(4096) == 31):
        return 'one_128k_31x4k'
    if count == 32 and lengths.count(32768) == 2 and (lengths.count(4096) == 30):
        return 'two_32k_30x4k'
    return None

def _bf16_entry__select_static_bf16_policy(inputs: _bf16_entry__StaticBF16Inputs) -> _bf16_entry__StaticBF16Policy:
    """Select one validated launch using only host-visible input properties."""
    mtp, _hq, _hkv = _bf16_entry___validate(inputs)
    effective = inputs.kv_lens
    lengths = tuple(effective.detach().cpu().to(torch.int64).tolist())
    case = _bf16_entry___classify_workload(lengths)
    if case is None:
        if not lengths or min(lengths) < mtp:
            raise ValueError('each final KV length must be at least MTP')
        case = 'uniform_4096'
    layout = inputs.layout
    cluster_size, chunk_tokens = _bf16_entry___FINAL_CT[mtp][case][layout]
    key = (mtp, case, layout)
    narrow = key in _bf16_entry___NARROW
    fulltail = case in _bf16_entry___FULLTAIL[mtp]
    deferred = case in _bf16_entry___DEFERRED[mtp]
    if case == 'uniform_512':
        if mtp == 2:
            kernel = 'direct-narrow'
        else:
            kernel = 'delayed-v'
            narrow = False
    elif cluster_size == 1:
        kernel = 'direct-fulltail'
        narrow = False
    elif case in _bf16_entry___PIPE:
        kernel = 'pipeline-fulltail-deferred' if fulltail else 'pipeline-deferred' if deferred else 'pipeline'
        narrow = False
    elif fulltail:
        kernel = 'cluster-fulltail'
        narrow = False
    elif deferred:
        kernel = 'cluster-deferred'
        narrow = False
    else:
        kernel = 'cluster'
        narrow = False
    finalizer_heads = 2 if mtp == 3 and case in {'one_64k_15x4k', 'one_128k_31x4k'} else 1
    workload = DecodeWorkload.from_lengths(lengths)
    return _bf16_entry__StaticBF16Policy(workload, layout, mtp, cluster_size, chunk_tokens, kernel, narrow, finalizer_heads)

def _bf16_entry__prepare_static_bf16_workspace(inputs: _bf16_entry__StaticBF16Inputs) -> _bf16_entry__StaticBF16Workspace:
    mtp, hq, _hkv = _bf16_entry___validate(inputs)
    policy = _bf16_entry__select_static_bf16_policy(inputs)
    max_len = int(inputs.kv_lens.max().item())
    max_chunks = triton.cdiv(max_len, policy.chunk_tokens)
    max_groups = triton.cdiv(max_chunks, policy.cluster_size)
    storage_groups = triton.next_power_of_2(max_groups)
    device = inputs.q.device
    return _bf16_entry__StaticBF16Workspace(policy=policy, q_4d=inputs.q.reshape(inputs.batch, mtp, hq, _bf16_entry__HEAD_DIM), split_out=torch.empty((inputs.batch, storage_groups, mtp, hq, _bf16_entry__HEAD_DIM), dtype=torch.float32, device=device), split_lse=torch.empty((inputs.batch, storage_groups, mtp, hq), dtype=torch.float32, device=device), out=torch.empty((inputs.batch, mtp, hq, _bf16_entry__HEAD_DIM), dtype=torch.bfloat16, device=device), effective_kv_lens=inputs.kv_lens, max_groups=max_groups)

def _bf16_entry__attention_decode_bf16_tle(inputs: _bf16_entry__StaticBF16Inputs, workspace: _bf16_entry__StaticBF16Workspace) -> torch.Tensor:
    """Launch the fixed final BF16 policy selected during workspace setup."""
    mtp, hq, hkv = _bf16_entry___validate(inputs)
    policy = workspace.policy
    if policy.mtp != mtp or policy.layout != inputs.layout:
        raise ValueError('workspace policy does not match inputs')
    heads_per_group = hq // hkv
    max_groups = workspace.max_groups
    logical_clusters = inputs.batch * hkv * max_groups
    q_rows = max(8, triton.next_power_of_2(mtp * heads_per_group)) if policy.narrow else 64
    k_desc = TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=[1, 1, _bf16_entry__TILE_N, _bf16_entry__HEAD_DIM])
    v_desc = TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=[1, 1, _bf16_entry__TILE_N, _bf16_entry__HEAD_DIM])
    common = dict(B=inputs.batch, NUM_SEQ_Q=mtp, Q_ROWS=q_rows, H_Q=hq, HEADS_PER_GROUP=heads_per_group, D=_bf16_entry__HEAD_DIM, BLOCK_SIZE=_bf16_entry__BLOCK_SIZE, MAX_BLOCKS=inputs.block_ids.shape[1], CHUNK_TOKENS=policy.chunk_tokens, MAX_GROUPS=max_groups, Q_SB=workspace.q_4d.stride(0), Q_SM=workspace.q_4d.stride(1), Q_SH=workspace.q_4d.stride(2), SO_SB=workspace.split_out.stride(0), SO_SG=workspace.split_out.stride(1), SO_SM=workspace.split_out.stride(2), SO_SH=workspace.split_out.stride(3), SL_SB=workspace.split_lse.stride(0), SL_SG=workspace.split_lse.stride(1), SL_SM=workspace.split_lse.stride(2), SL_SH=workspace.split_lse.stride(3), O_SB=workspace.out.stride(0), O_SM=workspace.out.stride(1), O_SH=workspace.out.stride(2))
    args = (workspace.q_4d, k_desc, v_desc, inputs.block_ids, workspace.effective_kv_lens, workspace.split_out, workspace.split_lse, workspace.out)
    if policy.kernel == 'delayed-v':
        _uniform512__bf16_decode_uniform512_delayed_v_kernel[logical_clusters,](workspace.q_4d, k_desc, v_desc, inputs.block_ids, workspace.out, B=inputs.batch, NUM_SEQ_Q=mtp, Q_ROWS=q_rows, H_Q=hq, HEADS_PER_GROUP=heads_per_group, D=_bf16_entry__HEAD_DIM, MAX_BLOCKS=inputs.block_ids.shape[1], Q_SB=workspace.q_4d.stride(0), Q_SM=workspace.q_4d.stride(1), Q_SH=workspace.q_4d.stride(2), O_SB=workspace.out.stride(0), O_SM=workspace.out.stride(1), O_SH=workspace.out.stride(2), num_warps=4, num_stages=3)
    elif policy.kernel == 'direct-narrow':
        _direct_narrow__bf16_decode_direct_kernel[logical_clusters,](*args, **common, TMA_STAGES=2, num_warps=4, num_stages=3)
    elif policy.kernel == 'direct-fulltail':
        _direct_fulltail__bf16_decode_direct_fulltail_kernel[logical_clusters,](*args, **common, TMA_STAGES=1, num_warps=4, num_stages=3)
    else:
        cluster_common = dict(**common, mesh=_bf16_entry___CLUSTER_MESHES[policy.cluster_size], NUM_SEQ_Q_PAD=triton.next_power_of_2(mtp), CLUSTER_SIZE=policy.cluster_size)
        kernels = {'cluster': _base__bf16_decode_static_kernel, 'cluster-deferred': _cluster_deferred__bf16_decode_static_dsm_deferred_kernel, 'cluster-fulltail': _cluster_fulltail__bf16_decode_static_fulltail_kernel, 'pipeline': _cluster_pipeline__bf16_decode_cluster_wide_pipeline_kernel, 'pipeline-deferred': _cluster_pipeline_deferred__bf16_decode_cluster_wide_pipeline_dsm_deferred_kernel, 'pipeline-fulltail-deferred': _cluster_pipeline_fulltail_deferred__bf16_decode_cluster_wide_pipeline_fulltail_dsm_deferred_kernel}
        kernels[policy.kernel][logical_clusters,](*args, **cluster_common, num_ctas=1, num_warps=4, num_stages=3, launch_pdl=False)
    if max_groups > 1:
        finalize_args = (workspace.split_out, workspace.split_lse, workspace.effective_kv_lens, workspace.out)
        finalize_kwargs = dict(NUM_SEQ_Q=mtp, NUM_SEQ_Q_PAD=triton.next_power_of_2(mtp), H_Q=hq, D=_bf16_entry__HEAD_DIM, CLUSTER_SIZE=policy.cluster_size, CHUNK_TOKENS=policy.chunk_tokens, MAX_GROUPS=triton.next_power_of_2(max_groups), SO_SB=workspace.split_out.stride(0), SO_SG=workspace.split_out.stride(1), SO_SM=workspace.split_out.stride(2), SO_SH=workspace.split_out.stride(3), SL_SB=workspace.split_lse.stride(0), SL_SG=workspace.split_lse.stride(1), SL_SM=workspace.split_lse.stride(2), SL_SH=workspace.split_lse.stride(3), O_SB=workspace.out.stride(0), O_SM=workspace.out.stride(1), O_SH=workspace.out.stride(2))
        if policy.finalizer_heads == 1:
            finalize_kwargs.pop('H_Q')
            _base__bf16_decode_finalize_kernel[inputs.batch, hq](*finalize_args, **finalize_kwargs, num_warps=4)
        else:
            _finalizer__bf16_decode_finalize_multihead_kernel[inputs.batch, triton.cdiv(hq, policy.finalizer_heads)](*finalize_args, **finalize_kwargs, HEADS_PER_PROGRAM=policy.finalizer_heads, num_warps=4)
    return workspace.out.reshape_as(inputs.q)
BLOCK_SIZE = _bf16_entry__BLOCK_SIZE
HEAD_DIM = _bf16_entry__HEAD_DIM
OFFICIAL_CASES = _bf16_entry__OFFICIAL_CASES
StaticBF16Inputs = _bf16_entry__StaticBF16Inputs
StaticBF16Policy = _bf16_entry__StaticBF16Policy
StaticBF16Workspace = _bf16_entry__StaticBF16Workspace
def prepare_static_bf16_workspace(inputs):
    if not USE_TLE:
        if inputs.mtp == 1:
            return prepare_pure_triton_mtp1_workspace(inputs, "bf16")
        return prepare_pure_triton_mtp_workspace(inputs, "bf16")
    return _bf16_entry__prepare_static_bf16_workspace(inputs)


def attention_decode_bf16_tle(inputs, workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    return _bf16_entry__attention_decode_bf16_tle(inputs, workspace)


select_static_bf16_policy = _bf16_entry__select_static_bf16_policy
__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "StaticBF16Inputs",
    "StaticBF16Policy",
    "StaticBF16Workspace",
    "attention_decode_bf16_tle",
    "prepare_static_bf16_workspace",
    "select_static_bf16_policy",
]
