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

"""Self-contained Hopper attention-decode implementation."""

from __future__ import annotations

from triton.language import core as tl_core
from triton.language.core import builtin
from typing import Literal
from triton.tools.tensor_descriptor import TensorDescriptor


from .. import (
    DecodeWorkload,
    PureTritonMTPWorkspace,
    PureTritonMTP1Workspace,
    USE_TLE,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
    gpu_types,
    tle,
)
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class _scheduler_mtp24__DecodeTaskSchedule:
    task_workspace: torch.Tensor
    task_map: torch.Tensor
    offsets: torch.Tensor
    meta: torch.Tensor
    cluster_size: int
    chunk_tokens: int
    block_seq: int
    block_chunks: int
    capacity_clusters: int
    capacity_ints: int
    direct_threshold: int = 0
    short_threshold: int = 0
    short_chunk_tokens: int = 0
    subgroup2_threshold: int = 0
    num_seq_q: int = 2
    num_clusters: int = 0
    physical_ctas: int = 0
    sched_ints: int = 0
    partial_slots: int = 0
    stats: dict | None = None


_compute_mtp1__NUM_SEQ_Q = 1
_compute_mtp1__ROWS_Q = 8
_compute_mtp1__DIRECT_MODE = 0
_compute_mtp1__GROUP_MODE = 1
_compute_mtp1__EXECUTION_FULL = 0
_compute_mtp1__EXECUTION_LOCAL_PARTIAL = 2
_compute_mtp1___ROWS_Q_JIT = tl.constexpr(_compute_mtp1__ROWS_Q)
_compute_mtp1___NUM_SEQ_Q_JIT = tl.constexpr(_compute_mtp1__NUM_SEQ_Q)
_compute_mtp1___TMA_STAGES_JIT = tl.constexpr(2)
_compute_mtp1___DIRECT_MODE_JIT = tl.constexpr(_compute_mtp1__DIRECT_MODE)
_compute_mtp1___GROUP_MODE_JIT = tl.constexpr(_compute_mtp1__GROUP_MODE)
_compute_mtp1___K_FRAGMENT_JIT = tl.constexpr(32)
_compute_mtp1___EXECUTION_FULL_JIT = tl.constexpr(_compute_mtp1__EXECUTION_FULL)
_compute_mtp1___EXECUTION_LOCAL_PARTIAL_JIT = tl.constexpr(_compute_mtp1__EXECUTION_LOCAL_PARTIAL)


@triton.jit
def _compute_mtp1___cluster_cooperative_finalize_mtp1(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    REUSE_FINAL_WEIGHTS: tl.constexpr,
):
    """Use one CTA rank per GQA head to finalize both MTP rows in-kernel."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_m = tl.arange(0, _compute_mtp1___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None] < n_chunks) & valid_head
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :]
    weights_unnorm = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights_unnorm = tl.where(chunk_mask, weights_unnorm, 0.0)
    denom = tl.sum(weights_unnorm, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    if REUSE_FINAL_WEIGHTS:
        weight_rows = tl.broadcast_to(offs_c[:, None], (MAX_FINAL_CHUNKS, _compute_mtp1___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(offs_m[None, :], (MAX_FINAL_CHUNKS, _compute_mtp1___NUM_SEQ_Q_JIT))
        weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols))
        tl.store(weight_ptr, weights_unnorm * inv_denom[None, :])
        tl.debug_barrier()
    acc = tl.zeros((DV, _compute_mtp1___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head
        if REUSE_FINAL_WEIGHTS:
            chunk_rows = tl.full((_compute_mtp1___NUM_SEQ_Q_JIT,), chunk, tl.int32)
            chunk_weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, offs_m))
            chunk_weight = tl.load(chunk_weight_ptr, mask=chunk_valid, other=0.0)
        else:
            chunk_lse = tl.load(
                LSE
                + batch * LSE_STRIDE_B
                + chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + offs_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                mask=chunk_valid,
                other=-float("inf"),
            )
            chunk_delta = chunk_lse - safe_max_lse
            chunk_weight = (tl.exp2(chunk_delta) if USE_LOG2 else tl.exp(chunk_delta)) * inv_denom
            chunk_weight = tl.where(chunk_valid, chunk_weight, 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, :] * SO_STRIDE_M
            + hq * SO_STRIDE_H
            + offs_v[:, None],
            mask=chunk_valid,
            other=0.0,
        )
        acc += partial * chunk_weight[None, :]
    if REUSE_FINAL_WEIGHTS:
        tl.debug_barrier()
    tl.store(
        OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_head & (denom[None, :] > 0.0),
    )


@triton.jit
def _compute_mtp1___cluster_quad_head_two_chunk_finalize_mtp1(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    logical_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize four C2-owned heads and exactly two global partials."""
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp1___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = logical_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weight0 = tl.exp2(lse0 - safe_max) if USE_LOG2 else tl.exp(lse0 - safe_max)
    weight1 = tl.exp2(lse1 - safe_max) if USE_LOG2 else tl.exp(lse1 - safe_max)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    acc = tl.load(partial_base, mask=partial_mask, other=0.0)
    acc *= weight0[None, :, :]
    acc += tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0) * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp1___cluster_paired_head_finalize_mtp1(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    first_h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
):
    """Finalize two strided GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp1___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = first_h_in_group + offs_h * HEAD_STRIDE
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp1___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp1___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp1___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp1___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    acc = tl.zeros((DV, 2, _compute_mtp1___NUM_SEQ_Q_JIT), tl.float32)
    pair_cols = offs_h[:, None] * _compute_mtp1___NUM_SEQ_Q_JIT + offs_m[None, :]
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp1___NUM_SEQ_Q_JIT)
        chunk_rows = tl.full((2, _compute_mtp1___NUM_SEQ_Q_JIT), chunk, tl.int32)
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, pair_cols)), mask=chunk_valid, other=0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=chunk_valid[None, :, :],
            other=0.0,
        )
        acc += partial * chunk_weight[None, :, :]
    tl.debug_barrier()
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@builtin
def _memdesc_subslice(
    value,
    shape: tl.constexpr,
    offsets: tl.constexpr,
    _semantic=None,
):
    """Use FlagTree's existing ttg.memdesc_subslice builder binding."""
    shape = [int(tl_core._unwrap_if_constexpr(dim)) for dim in shape]
    layout = value.type.layout
    result_ty = gpu_types.buffered_tensor_type(
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=value.type.alloc_shape,
    )
    handle = _semantic.builder.create_memdesc_subslice(
        result_ty.to_ir(_semantic.builder),
        value.handle,
        list(offsets),
    )
    return gpu_types.buffered_tensor(
        handle,
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=value.type.alloc_shape,
    )


@builtin
def _memdesc_transpose_2d(value, _semantic=None):
    """Return a zero-copy transposed view of a rank-2 shared memdesc.

    Unlike ``tl.trans(local_load(...))``, this exposes the transpose to the
    local-load lowering.  The intended transposed-memdesc lowering is therefore a direct
    transposed shared-to-register load from the TMA buffer, with no temporary
    shared allocation between the load and RS WGMMA.
    """
    if len(value.type.shape) != 2:
        raise ValueError("_memdesc_transpose_2d expects a rank-2 buffer")
    order = [1, 0]
    handle = _semantic.builder.create_memdesc_trans(value.handle, order)
    shape = [value.type.shape[i] for i in order]
    alloc_shape = value.type.alloc_shape
    leading_rank = len(alloc_shape) - len(value.type.shape)
    alloc_tail = alloc_shape[leading_rank:]
    transposed_alloc_shape = alloc_shape[:leading_rank] + [alloc_tail[i] for i in order]
    layout = value.type.layout.make_permute(order)
    return gpu_types.buffered_tensor(
        handle,
        value.dtype,
        shape,
        value.type.storage,
        layout,
        _semantic,
        alloc_shape=transposed_alloc_shape,
    )


@triton.jit
def _compute_mtp1__fp8_kvpertensor_decode_mtp1_final_kernel(
    Q,
    K_DESC,
    VT_DESC,
    BLOCK_IDS,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    PAGE_METADATA_K_SCALE: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    FULL_VIEW_V_RS: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp1__EXECUTION_FULL,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    BF16_DSM: tl.constexpr = False,
    DEFERRED_NORM: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    REDUCTION_ONLY: tl.constexpr = False,
    ALIGNED_FULL_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
    STATIC_MTP1_NO_CAUSAL_MASK: tl.constexpr = False,
    STATIC_BLOCK_IDS_PREFETCH: tl.constexpr = False,
):
    cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    logical_cluster = cta // MERGE_CLUSTER_SIZE
    group_chunk = logical_cluster % STATIC_MAX_GROUPS
    sequence = logical_cluster // STATIC_MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
    num_chunks = (total_len + STATIC_CHUNK_TOKENS - 1) // STATIC_CHUNK_TOKENS
    group_count = (num_chunks + MERGE_CLUSTER_SIZE - 1) // MERGE_CLUSTER_SIZE
    if group_chunk >= group_count:
        return
    chunk = group_chunk * MERGE_CLUSTER_SIZE + cluster_rank
    seq_start = chunk * STATIC_CHUNK_TOKENS
    has_work = chunk < num_chunks
    seq_len = tl.where(has_work, tl.minimum(STATIC_CHUNK_TOKENS, total_len - seq_start), 0)
    is_causal = has_work & (chunk == num_chunks - 1)
    seq_kvcache = tl.where(is_causal, seq_len - _compute_mtp1___NUM_SEQ_Q_JIT, seq_len)
    task_mode = tl.where(num_chunks == 1, _compute_mtp1___DIRECT_MODE_JIT, _compute_mtp1___GROUP_MODE_JIT)
    q_smem = tle.gpu.alloc([_compute_mtp1___ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_compute_mtp1___ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc(
        [_compute_mtp1___TMA_STAGES_JIT, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    v_raw_smem = tle.gpu.alloc(
        [_compute_mtp1___TMA_STAGES_JIT, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp1___TMA_STAGES_JIT, arrive_count=1, expect_bytes=BLOCK_N * D
    )
    vt_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp1___TMA_STAGES_JIT, arrive_count=1, expect_bytes=DV * BLOCK_N
    )
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            peer_l_smem = tle.gpu.alloc(
                [_compute_mtp1___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer_l_smem = peer_lse_smem
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer2_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer3_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer1_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer2_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer3_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    else:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp1___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp1___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            partial_l_smem = tle.gpu.alloc(
                [_compute_mtp1___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            partial_l_smem = partial_lse_smem
    final_weight_smem = tle.gpu.alloc(
        [
            MAX_FINAL_CHUNKS,
            2 * _compute_mtp1___NUM_SEQ_Q_JIT if PAIRED_HEAD_FINALIZE else _compute_mtp1___NUM_SEQ_Q_JIT,
        ],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    offs_q = tl.arange(0, _compute_mtp1___ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _compute_mtp1___ROWS_Q_JIT)[:, None], (_compute_mtp1___ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_compute_mtp1___ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _compute_mtp1___ROWS_Q_JIT)[:, None], (_compute_mtp1___ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_compute_mtp1___ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _compute_mtp1___ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _compute_mtp1___ROWS_Q_JIT)[None, :], (DV, _compute_mtp1___ROWS_Q_JIT))
    store_offs_v = offs_v
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if MERGE_CLUSTER_SIZE == 8:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
        if MERGE_CLUSTER_SIZE == 8:
            partial_l_ptr = tle.gpu.local_ptr(partial_l_smem, (offs_q,))
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = has_work & (seq_m < _compute_mtp1___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _compute_mtp1___ROWS_Q_JIT), tl.float32)
    lse = tl.full((_compute_mtp1___ROWS_Q_JIT,), -float("inf"), tl.float32)
    raw_l = tl.zeros((_compute_mtp1___ROWS_Q_JIT,), tl.float32)
    if STATIC_BLOCK_IDS_PREFETCH:
        chunk_block_ids_smem = tle.gpu.alloc(
            [STATIC_CHUNK_TOKENS // BLOCK_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    if has_work:
        if STATIC_BLOCK_IDS_PREFETCH:
            block_id_offs = tl.arange(0, STATIC_CHUNK_TOKENS // BLOCK_SIZE)
            num_chunk_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            staged_block_ids = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + seq_start // BLOCK_SIZE + block_id_offs,
                mask=block_id_offs < num_chunk_blocks,
                other=0,
            )
            tl.store(
                tle.gpu.local_ptr(chunk_block_ids_smem, (block_id_offs,)),
                staged_block_ids,
                mask=block_id_offs < num_chunk_blocks,
            )
            tl.debug_barrier()
        q = tl.load(
            Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
            mask=valid_q[:, None],
            other=0.0,
        )
        tl.store(q_smem_ptr, q)
        qscale = tl.load(
            QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0
        ).to(tl.float32)
        if PRECOMBINE_Q_SCALE:
            qscale = qscale * (inv_sqrt_d * 1.4426950408889634)
            qscale = qscale * kscale
        m_i = tl.full((_compute_mtp1___ROWS_Q_JIT,), -float("inf"), tl.float32)
        l_i = tl.zeros((_compute_mtp1___ROWS_Q_JIT,), tl.float32)
        copy_iter = 0
        start = 0
        page_current_phys = tl.full((), 0, tl.int32)
        if start < seq_len:
            if STATIC_BLOCK_IDS_PREFETCH:
                phys = tl.load(tle.gpu.local_ptr(chunk_block_ids_smem, (0,)))
            else:
                block_no = seq_start // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            if PAGE_METADATA_K_SCALE:
                page_current_phys = phys
            tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
        while start < seq_len:
            local_n = start + offs_n
            if ALIGNED_FULL_CHUNK_TOKENS:
                valid_cols = tl.full((BLOCK_N,), True, tl.int1)
            elif not STATIC_MTP1_NO_CAUSAL_MASK:
                valid_cols = local_n < seq_len
            buf = copy_iter % _compute_mtp1___TMA_STAGES_JIT
            phase = copy_iter // _compute_mtp1___TMA_STAGES_JIT & 1
            next_start = start + BLOCK_N
            page_next_phys = page_current_phys
            if next_start < seq_len:
                next_iter = copy_iter + 1
                next_buf = next_iter % _compute_mtp1___TMA_STAGES_JIT
                aligned_logical = seq_start + next_start
                if STATIC_BLOCK_IDS_PREFETCH:
                    next_block_in_chunk = next_start // BLOCK_SIZE
                    phys = tl.load(tle.gpu.local_ptr(chunk_block_ids_smem, (next_block_in_chunk,)))
                else:
                    block_no = aligned_logical // BLOCK_SIZE
                    phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                if PAGE_METADATA_K_SCALE:
                    page_next_phys = phys
                tle.gpu.copy(
                    K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
                )
                tle.gpu.copy(
                    VT_DESC,
                    v_raw_smem.slot(next_buf),
                    [1, 1, BLOCK_N, DV],
                    [phys, hkv, 0, 0],
                    barrier=vt_full[next_buf],
                )
            tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            k_page = k_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            else:
                scores = tl.zeros((BLOCK_N, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                for frag in tl.static_range(0, 4):
                    k_frag = _memdesc_subslice(
                        k_page, (BLOCK_N, _compute_mtp1___K_FRAGMENT_JIT), (0, frag * _compute_mtp1___K_FRAGMENT_JIT)
                    )
                    q_frag = _memdesc_subslice(
                        q_smem,
                        (_compute_mtp1___ROWS_Q_JIT, _compute_mtp1___K_FRAGMENT_JIT),
                        (0, frag * _compute_mtp1___K_FRAGMENT_JIT),
                    )
                    scores = tle.gpu.wgmma(k_frag, q_frag, acc=scores, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores)
            if not PRECOMBINE_Q_SCALE:
                scores = scores * kscale
            if not PRECOMBINE_Q_SCALE:
                scores = scores * (inv_sqrt_d * 1.4426950408889634)
            scores = scores * qscale[None, :]
            if STATIC_MTP1_NO_CAUSAL_MASK:
                if next_start <= seq_len:
                    score_mask = tl.broadcast_to(valid_q[None, :], (BLOCK_N, _compute_mtp1___ROWS_Q_JIT))
                else:
                    tail_valid_cols = local_n < seq_len
                    score_mask = tail_valid_cols[:, None] & valid_q[None, :]
            else:
                causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
                score_mask = valid_cols[:, None] & causal & valid_q[None, :]
            scores = tl.where(score_mask, scores, -float("inf"))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float("inf")
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            tl.store(p_smem_ptr, p_scaled_t)
            tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            v_page = v_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED:
                v_page_t = _memdesc_transpose_2d(v_page)
                pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
                pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
                v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                v_pairs = tl.load(v_pair_ptr)
                v_lo = (v_pairs & 255).to(tl.uint8)
                v_hi = (v_pairs >> 8).to(tl.uint8)
                v_bytes = tl.join(v_lo, v_hi)
                v_bytes = tl.permute(v_bytes, (0, 2, 1))
                v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
                v_page_reg_t = v_bytes.to(tl.float8e4nv, bitcast=True)
            elif FULL_VIEW_V_RS:
                v_page_t = _memdesc_transpose_2d(v_page)
                v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t))
            else:
                v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
                v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
                v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            else:
                pv = tl.zeros((DV, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                for d_frag in tl.static_range(0, 2):
                    pv_frag = tl.zeros((_compute_mtp1___K_FRAGMENT_JIT * 2, _compute_mtp1___ROWS_Q_JIT), tl.float32)
                    for n_frag in tl.static_range(0, 2):
                        v_reg_frag = tle.extract_tile(
                            v_page_reg_t,
                            index=[d_frag, n_frag],
                            tile_shape=[_compute_mtp1___K_FRAGMENT_JIT * 2, _compute_mtp1___K_FRAGMENT_JIT],
                        )
                        p_frag = _memdesc_subslice(
                            p_smem,
                            (_compute_mtp1___ROWS_Q_JIT, _compute_mtp1___K_FRAGMENT_JIT),
                            (0, n_frag * _compute_mtp1___K_FRAGMENT_JIT),
                        )
                        pv_frag = tle.gpu.wgmma(v_reg_frag, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                    pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                    pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            acc = acc * alpha[None, :] + pv
            m_i = m_new
            l_i = l_new
            start = next_start
            copy_iter += 1
            if PAGE_METADATA_K_SCALE:
                page_current_phys = page_next_phys
        has_value = l_i > 0.0
        raw_l = l_i
        if DEFERRED_NORM and task_mode != _compute_mtp1___DIRECT_MODE_JIT:
            lse = m_i
        else:
            acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
            lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if EXECUTION_STAGE == _compute_mtp1___EXECUTION_LOCAL_PARTIAL_JIT:
        if cluster_rank == 0:
            local_output_mask = valid_q[None, :]
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                acc,
                mask=local_output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                lse,
                mask=valid_q,
            )
        return
    if not REDUCTION_ONLY and task_mode == _compute_mtp1___DIRECT_MODE_JIT:
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + store_offs_v[:, None],
            acc,
            mask=valid_q[None, :],
        )
        return
    group_acc = acc
    group_lse = lse
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        peer_l_remote = tle.remote(peer_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(peer_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max), 0.0)
            if DEFERRED_NORM:
                peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_q,)))
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            group_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 2:
            tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight1[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight2[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight3[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        if DEFERRED_NORM:
            tl.store(partial_l_ptr, raw_l)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        peer1_l = tle.remote(partial_l_smem, 1, scope=mesh)
        peer2_l = tle.remote(partial_l_smem, 2, scope=mesh)
        peer3_l = tle.remote(partial_l_smem, 3, scope=mesh)
        peer4_l = tle.remote(partial_l_smem, 4, scope=mesh)
        peer5_l = tle.remote(partial_l_smem, 5, scope=mesh)
        peer6_l = tle.remote(partial_l_smem, 6, scope=mesh)
        peer7_l = tle.remote(partial_l_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float("inf"), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float("inf"), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float("inf"), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float("inf"), tl.exp2(lse7 - safe_max), 0.0)
            if DEFERRED_NORM:
                l1 = tl.load(tle.gpu.local_ptr(peer1_l, (offs_q,)))
                l2 = tl.load(tle.gpu.local_ptr(peer2_l, (offs_q,)))
                l3 = tl.load(tle.gpu.local_ptr(peer3_l, (offs_q,)))
                l4 = tl.load(tle.gpu.local_ptr(peer4_l, (offs_q,)))
                l5 = tl.load(tle.gpu.local_ptr(peer5_l, (offs_q,)))
                l6 = tl.load(tle.gpu.local_ptr(peer6_l, (offs_q,)))
                l7 = tl.load(tle.gpu.local_ptr(peer7_l, (offs_q,)))
                denom = (
                    raw_l * weight0
                    + l1 * weight1
                    + l2 * weight2
                    + l3 * weight3
                    + l4 * weight4
                    + l5 * weight5
                    + l6 * weight6
                    + l7 * weight7
                )
            else:
                denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = ((seq_m < _compute_mtp1___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q))[None, :]
        if group_count == 1:
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                group_acc,
                mask=output_mask,
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                group_acc,
                mask=output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                group_lse,
                mask=(seq_m < _compute_mtp1___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
            )
    if EXECUTION_STAGE == _compute_mtp1___EXECUTION_FULL_JIT and group_count > 1:
        counter_idx = hkv * B + batch
        rank0_is_last = tl.zeros((), tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_ELECTION:
                deterministic_owner = group_chunk == group_count - 1
                if deterministic_owner:
                    ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(COMPLETION + counter_idx, 1, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        if RANK0_ONLY_FINALIZER:
            is_last_cluster = (cluster_rank == 0) & (rank0_is_last != 0)
        elif DETERMINISTIC_TAIL_ELECTION:
            tle.distributed_barrier(mesh)
            is_last_cluster = group_chunk == group_count - 1
        elif DSM_ELECTION_HANDOFF:
            if MERGE_CLUSTER_SIZE == 2:
                rank1_flag = tle.remote(peer_lse_smem, 1, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer_lse_smem, (0,))) != 0.0
            elif MERGE_CLUSTER_SIZE == 4:
                rank1_flag = tle.remote(peer1_lse_smem, 1, scope=mesh)
                rank2_flag = tle.remote(peer1_lse_smem, 2, scope=mesh)
                rank3_flag = tle.remote(peer1_lse_smem, 3, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer1_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (0,))) != 0.0
            else:
                rank1_flag = tle.remote(partial_lse_smem, 1, scope=mesh)
                rank2_flag = tle.remote(partial_lse_smem, 2, scope=mesh)
                rank3_flag = tle.remote(partial_lse_smem, 3, scope=mesh)
                rank4_flag = tle.remote(partial_lse_smem, 4, scope=mesh)
                rank5_flag = tle.remote(partial_lse_smem, 5, scope=mesh)
                rank6_flag = tle.remote(partial_lse_smem, 6, scope=mesh)
                rank7_flag = tle.remote(partial_lse_smem, 7, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(partial_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank4_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank5_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank6_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank7_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(partial_lse_smem, (0,))) != 0.0
        else:
            tle.distributed_barrier(mesh)
            rank0_cta = cta - cluster_rank
            is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_last_cluster:
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
            if RANK0_ONLY_FINALIZER:
                _compute_mtp1___cluster_quad_head_two_chunk_finalize_mtp1(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    0,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
                _compute_mtp1___cluster_quad_head_two_chunk_finalize_mtp1(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    1,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
            elif PAIRED_HEAD_FINALIZE:
                _compute_mtp1___cluster_paired_head_finalize_mtp1(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    group_count,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                    MAX_FINAL_CHUNKS,
                    4,
                )
            else:
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    _compute_mtp1___cluster_cooperative_finalize_mtp1(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        final_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        True,
                    )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")


_compute_mtp2__NUM_SEQ_Q = 2
_compute_mtp2__ROWS_Q = 16
_compute_mtp2__DIRECT_MODE = 0
_compute_mtp2__GROUP_MODE = 1
_compute_mtp2__EXECUTION_FULL = 0
_compute_mtp2__EXECUTION_LOCAL_PARTIAL = 2
_compute_mtp2___ROWS_Q_JIT = tl.constexpr(_compute_mtp2__ROWS_Q)
_compute_mtp2___NUM_SEQ_Q_JIT = tl.constexpr(_compute_mtp2__NUM_SEQ_Q)
_compute_mtp2___TMA_STAGES_JIT = tl.constexpr(2)
_compute_mtp2___DIRECT_MODE_JIT = tl.constexpr(_compute_mtp2__DIRECT_MODE)
_compute_mtp2___GROUP_MODE_JIT = tl.constexpr(_compute_mtp2__GROUP_MODE)
_compute_mtp2___K_FRAGMENT_JIT = tl.constexpr(32)
_compute_mtp2___EXECUTION_FULL_JIT = tl.constexpr(_compute_mtp2__EXECUTION_FULL)
_compute_mtp2___EXECUTION_LOCAL_PARTIAL_JIT = tl.constexpr(_compute_mtp2__EXECUTION_LOCAL_PARTIAL)


@triton.jit
def _compute_mtp2___cluster_cooperative_finalize_mtp2(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    REUSE_FINAL_WEIGHTS: tl.constexpr,
    ROW_SERIAL_FINALIZE: tl.constexpr,
    TWO_CHUNK_FINALIZE: tl.constexpr,
):
    """Use one CTA rank per GQA head to finalize both MTP rows in-kernel."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None] < n_chunks) & valid_head
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :]
    weights_unnorm = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights_unnorm = tl.where(chunk_mask, weights_unnorm, 0.0)
    denom = tl.sum(weights_unnorm, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    if REUSE_FINAL_WEIGHTS:
        weight_rows = tl.broadcast_to(offs_c[:, None], (MAX_FINAL_CHUNKS, _compute_mtp2___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(offs_m[None, :], (MAX_FINAL_CHUNKS, _compute_mtp2___NUM_SEQ_Q_JIT))
        weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols))
        tl.store(weight_ptr, weights_unnorm * inv_denom[None, :])
        tl.debug_barrier()
    if ROW_SERIAL_FINALIZE:
        for seq_m in tl.static_range(0, _compute_mtp2___NUM_SEQ_Q_JIT):
            acc_row = tl.zeros((DV,), tl.float32)
            denom_row = tl.sum(tl.where(offs_m == seq_m, denom, 0.0), axis=0)
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (chunk < n_chunks) & valid_head
                chunk_rows = tl.full((1,), chunk, tl.int32)
                chunk_cols = tl.full((1,), seq_m, tl.int32)
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, chunk_cols)), mask=chunk_valid, other=0.0
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + seq_m * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v,
                    mask=chunk_valid,
                    other=0.0,
                )
                chunk_weight_scalar = tl.sum(chunk_weight, axis=0)
                acc_row += partial * chunk_weight_scalar
            tl.store(
                OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq * O_STRIDE_H + offs_v,
                acc_row,
                mask=valid_head & (denom_row > 0.0),
            )
        tl.debug_barrier()
    else:
        acc = tl.zeros((DV, _compute_mtp2___NUM_SEQ_Q_JIT), tl.float32)
        if TWO_CHUNK_FINALIZE and MAX_FINAL_CHUNKS >= 2:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS, 2):
                valid0 = (chunk < n_chunks) & valid_head
                valid1 = (chunk + 1 < n_chunks) & valid_head
                rows0 = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), chunk, tl.int32)
                rows1 = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), chunk + 1, tl.int32)
                weight0 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows0, offs_m)), mask=valid0, other=0.0)
                weight1 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows1, offs_m)), mask=valid1, other=0.0)
                partial0 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, :] * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=valid0,
                    other=0.0,
                )
                partial1 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + (chunk + 1) * SO_STRIDE_C
                    + offs_m[None, :] * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=valid1,
                    other=0.0,
                )
                acc += partial0 * weight0[None, :] + partial1 * weight1[None, :]
        else:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (chunk < n_chunks) & valid_head
                if REUSE_FINAL_WEIGHTS:
                    chunk_rows = tl.full((_compute_mtp2___NUM_SEQ_Q_JIT,), chunk, tl.int32)
                    chunk_weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, offs_m))
                    chunk_weight = tl.load(chunk_weight_ptr, mask=chunk_valid, other=0.0)
                else:
                    chunk_lse = tl.load(
                        LSE
                        + batch * LSE_STRIDE_B
                        + chunk * LSE_STRIDE_C
                        + hkv * LSE_STRIDE_HKV
                        + offs_m * LSE_STRIDE_M
                        + h_in_group * LSE_STRIDE_HG,
                        mask=chunk_valid,
                        other=-float("inf"),
                    )
                    chunk_delta = chunk_lse - safe_max_lse
                    chunk_weight = (tl.exp2(chunk_delta) if USE_LOG2 else tl.exp(chunk_delta)) * inv_denom
                    chunk_weight = tl.where(chunk_valid, chunk_weight, 0.0)
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, :] * SO_STRIDE_M
                    + hq * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=chunk_valid,
                    other=0.0,
                )
                acc += partial * chunk_weight[None, :]
        if REUSE_FINAL_WEIGHTS:
            tl.debug_barrier()
        tl.store(
            OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
            acc,
            mask=valid_head & (denom[None, :] > 0.0),
        )


@triton.jit
def _compute_mtp2___cluster_quad_head_two_chunk_finalize_mtp2(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    logical_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize four C2-owned heads and exactly two global partials."""
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = logical_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weight0 = tl.exp2(lse0 - safe_max_lse) if USE_LOG2 else tl.exp(lse0 - safe_max_lse)
    weight1 = tl.exp2(lse1 - safe_max_lse) if USE_LOG2 else tl.exp(lse1 - safe_max_lse)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    acc = tl.load(partial_base, mask=partial_mask, other=0.0)
    acc *= weight0[None, :, :]
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0)
    acc += partial1 * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp2___cluster_paired_head_finalize_mtp2(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    ROW_SERIAL_FINALIZE: tl.constexpr,
    TWO_CHUNK_FINALIZE: tl.constexpr,
):
    """Finalize the two c4-owned GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp2___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp2___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp2___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp2___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    if ROW_SERIAL_FINALIZE:
        for seq_m in tl.static_range(0, _compute_mtp2___NUM_SEQ_Q_JIT):
            acc_row = tl.zeros((DV, 2), tl.float32)
            pair_cols = offs_h * _compute_mtp2___NUM_SEQ_Q_JIT + seq_m
            denom_row = tl.sum(tl.where(offs_m[None, :] == seq_m, denom, 0.0), axis=1)
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (chunk < n_chunks) & valid_head
                chunk_rows = tl.full((2,), chunk, tl.int32)
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, pair_cols)), mask=chunk_valid, other=0.0
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + seq_m * SO_STRIDE_M
                    + hq[None, :] * SO_STRIDE_H
                    + offs_v[:, None],
                    mask=chunk_valid[None, :],
                    other=0.0,
                )
                acc_row += partial * chunk_weight[None, :]
            tl.store(
                OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[None, :] * O_STRIDE_H + offs_v[:, None],
                acc_row,
                mask=valid_head[None, :] & (denom_row[None, :] > 0.0),
            )
        tl.debug_barrier()
    else:
        acc = tl.zeros((DV, 2, _compute_mtp2___NUM_SEQ_Q_JIT), tl.float32)
        pair_cols = offs_h[:, None] * _compute_mtp2___NUM_SEQ_Q_JIT + offs_m[None, :]
        if TWO_CHUNK_FINALIZE and MAX_FINAL_CHUNKS >= 2:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS, 2):
                valid0 = (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
                valid1 = (
                    (chunk + 1 < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
                )
                rows0 = tl.full((2, _compute_mtp2___NUM_SEQ_Q_JIT), chunk, tl.int32)
                rows1 = tl.full((2, _compute_mtp2___NUM_SEQ_Q_JIT), chunk + 1, tl.int32)
                weight0 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows0, pair_cols)), mask=valid0, other=0.0)
                weight1 = tl.load(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (rows1, pair_cols)), mask=valid1, other=0.0)
                partial0 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, None, :] * SO_STRIDE_M
                    + hq[None, :, None] * SO_STRIDE_H
                    + offs_v[:, None, None],
                    mask=valid0[None, :, :],
                    other=0.0,
                )
                partial1 = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + (chunk + 1) * SO_STRIDE_C
                    + offs_m[None, None, :] * SO_STRIDE_M
                    + hq[None, :, None] * SO_STRIDE_H
                    + offs_v[:, None, None],
                    mask=valid1[None, :, :],
                    other=0.0,
                )
                acc += partial0 * weight0[None, :, :] + partial1 * weight1[None, :, :]
        else:
            for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
                chunk_valid = (
                    (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp2___NUM_SEQ_Q_JIT)
                )
                chunk_rows = tl.full((2, _compute_mtp2___NUM_SEQ_Q_JIT), chunk, tl.int32)
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, pair_cols)), mask=chunk_valid, other=0.0
                )
                partial = tl.load(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + chunk * SO_STRIDE_C
                    + offs_m[None, None, :] * SO_STRIDE_M
                    + hq[None, :, None] * SO_STRIDE_H
                    + offs_v[:, None, None],
                    mask=chunk_valid[None, :, :],
                    other=0.0,
                )
                acc += partial * chunk_weight[None, :, :]
        tl.debug_barrier()
        tl.store(
            OUT
            + batch * O_STRIDE_B
            + offs_m[None, None, :] * O_STRIDE_M
            + hq[None, :, None] * O_STRIDE_H
            + offs_v[:, None, None],
            acc,
            mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
        )


@triton.jit
def _compute_mtp2__fp8_kvpertensor_decode_mtp2_final_kernel(
    Q,
    K_DESC,
    VT_DESC,
    BLOCK_IDS,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    PAGE_METADATA_K_SCALE: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    FULL_VIEW_V_RS: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp2__EXECUTION_FULL,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    REUSE_FINAL_WEIGHTS: tl.constexpr = False,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    BF16_DSM: tl.constexpr = False,
    DEFERRED_NORM: tl.constexpr = False,
    DSM_ELECTION_HANDOFF: tl.constexpr = False,
    DETERMINISTIC_TAIL_ELECTION: tl.constexpr = False,
    TAIL_ONLY_ELECTION_BARRIER: tl.constexpr = False,
    REDUCTION_ONLY: tl.constexpr = False,
    ALIGNED_FULL_CHUNK_TOKENS: tl.constexpr = 0,
    FULL_VIEW_DSM: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    logical_cluster = cta // MERGE_CLUSTER_SIZE
    group_chunk = logical_cluster % STATIC_MAX_GROUPS
    sequence = logical_cluster // STATIC_MAX_GROUPS
    batch = sequence % B
    hkv = sequence // B
    total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
    num_chunks = (total_len + STATIC_CHUNK_TOKENS - 1) // STATIC_CHUNK_TOKENS
    group_count = (num_chunks + MERGE_CLUSTER_SIZE - 1) // MERGE_CLUSTER_SIZE
    if group_chunk >= group_count:
        return
    chunk = group_chunk * MERGE_CLUSTER_SIZE + cluster_rank
    seq_start = chunk * STATIC_CHUNK_TOKENS
    has_work = chunk < num_chunks
    seq_len = tl.where(has_work, tl.minimum(STATIC_CHUNK_TOKENS, total_len - seq_start), 0)
    is_causal = has_work & (chunk == num_chunks - 1)
    seq_kvcache = tl.where(is_causal, seq_len - _compute_mtp2___NUM_SEQ_Q_JIT, seq_len)
    task_mode = tl.where(num_chunks == 1, _compute_mtp2___DIRECT_MODE_JIT, _compute_mtp2___GROUP_MODE_JIT)
    q_smem = tle.gpu.alloc([_compute_mtp2___ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_compute_mtp2___ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc(
        [_compute_mtp2___TMA_STAGES_JIT, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    v_raw_smem = tle.gpu.alloc(
        [_compute_mtp2___TMA_STAGES_JIT, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem
    )
    k_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp2___TMA_STAGES_JIT, arrive_count=1, expect_bytes=BLOCK_N * D
    )
    vt_full = tle.gpu.alloc_barriers(
        num_barriers=_compute_mtp2___TMA_STAGES_JIT, arrive_count=1, expect_bytes=DV * BLOCK_N
    )
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            peer_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer_l_smem = peer_lse_smem
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer2_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer3_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer1_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer2_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer3_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if DEFERRED_NORM:
            peer1_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_l_smem = tle.gpu.alloc(
                [_compute_mtp2___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer1_l_smem = peer1_lse_smem
            peer2_l_smem = peer2_lse_smem
            peer3_l_smem = peer3_lse_smem
    else:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp2___ROWS_Q_JIT],
            dtype=tl.bfloat16 if BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp2___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
    final_weight_smem = tle.gpu.alloc(
        [
            MAX_FINAL_CHUNKS if REUSE_FINAL_WEIGHTS else 1,
            2 * _compute_mtp2___NUM_SEQ_Q_JIT if PAIRED_HEAD_FINALIZE else _compute_mtp2___NUM_SEQ_Q_JIT,
        ],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    offs_q = tl.arange(0, _compute_mtp2___ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _compute_mtp2___ROWS_Q_JIT)[:, None], (_compute_mtp2___ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_compute_mtp2___ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _compute_mtp2___ROWS_Q_JIT)[:, None], (_compute_mtp2___ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_compute_mtp2___ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _compute_mtp2___ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _compute_mtp2___ROWS_Q_JIT)[None, :], (DV, _compute_mtp2___ROWS_Q_JIT))
    store_offs_v = offs_v
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if MERGE_CLUSTER_SIZE == 8:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = has_work & (seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _compute_mtp2___ROWS_Q_JIT), tl.float32)
    lse = tl.full((_compute_mtp2___ROWS_Q_JIT,), -float("inf"), tl.float32)
    raw_l = tl.zeros((_compute_mtp2___ROWS_Q_JIT,), tl.float32)
    if REDUCTION_ONLY or has_work:
        q = tl.load(
            Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
            mask=valid_q[:, None],
            other=0.0,
        )
        tl.store(q_smem_ptr, q)
        qscale = tl.load(
            QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0
        ).to(tl.float32)
        if PRECOMBINE_Q_SCALE:
            qscale = qscale * (inv_sqrt_d * 1.4426950408889634)
            qscale = qscale * kscale
        m_i = tl.full((_compute_mtp2___ROWS_Q_JIT,), -float("inf"), tl.float32)
        l_i = tl.zeros((_compute_mtp2___ROWS_Q_JIT,), tl.float32)
        copy_iter = 0
        start = 0
        page_current_phys = tl.full((), 0, tl.int32)
        if start < seq_len:
            block_no = seq_start // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            if PAGE_METADATA_K_SCALE:
                page_current_phys = phys
            tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
            tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
        while start < seq_len:
            local_n = start + offs_n
            if ALIGNED_FULL_CHUNK_TOKENS:
                valid_cols = tl.full((BLOCK_N,), True, tl.int1)
            else:
                valid_cols = local_n < seq_len
            buf = copy_iter % _compute_mtp2___TMA_STAGES_JIT
            phase = copy_iter // _compute_mtp2___TMA_STAGES_JIT & 1
            next_start = start + BLOCK_N
            page_next_phys = page_current_phys
            if next_start < seq_len:
                next_iter = copy_iter + 1
                next_buf = next_iter % _compute_mtp2___TMA_STAGES_JIT
                aligned_logical = seq_start + next_start
                block_no = aligned_logical // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                if PAGE_METADATA_K_SCALE:
                    page_next_phys = phys
                tle.gpu.copy(
                    K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
                )
                tle.gpu.copy(
                    VT_DESC,
                    v_raw_smem.slot(next_buf),
                    [1, 1, BLOCK_N, DV],
                    [phys, hkv, 0, 0],
                    barrier=vt_full[next_buf],
                )
            tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            k_page = k_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            else:
                scores = tl.zeros((BLOCK_N, _compute_mtp2___ROWS_Q_JIT), tl.float32)
                for frag in tl.static_range(0, 4):
                    k_frag = _memdesc_subslice(
                        k_page, (BLOCK_N, _compute_mtp2___K_FRAGMENT_JIT), (0, frag * _compute_mtp2___K_FRAGMENT_JIT)
                    )
                    q_frag = _memdesc_subslice(
                        q_smem,
                        (_compute_mtp2___ROWS_Q_JIT, _compute_mtp2___K_FRAGMENT_JIT),
                        (0, frag * _compute_mtp2___K_FRAGMENT_JIT),
                    )
                    scores = tle.gpu.wgmma(k_frag, q_frag, acc=scores, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores)
            if not PRECOMBINE_Q_SCALE:
                scores = scores * kscale
            if not PRECOMBINE_Q_SCALE:
                scores = scores * (inv_sqrt_d * 1.4426950408889634)
            scores = scores * qscale[None, :]
            causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
            scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float("inf"))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float("inf")
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            tl.store(p_smem_ptr, p_scaled_t)
            tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            v_page = v_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED:
                v_page_t = _memdesc_transpose_2d(v_page)
                pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
                pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
                v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                v_pairs = tl.load(v_pair_ptr)
                v_lo = (v_pairs & 255).to(tl.uint8)
                v_hi = (v_pairs >> 8).to(tl.uint8)
                v_bytes = tl.join(v_lo, v_hi)
                v_bytes = tl.permute(v_bytes, (0, 2, 1))
                v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
                v_page_reg_t = v_bytes.to(tl.float8e4nv, bitcast=True)
            elif FULL_VIEW_V_RS:
                v_page_t = _memdesc_transpose_2d(v_page)
                v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t))
            else:
                v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
                v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
                v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
            if LDSM_REGISTER_SHARED or FULL_VIEW_V_RS:
                pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            else:
                pv = tl.zeros((DV, _compute_mtp2___ROWS_Q_JIT), tl.float32)
                for d_frag in tl.static_range(0, 2):
                    pv_frag = tl.zeros((_compute_mtp2___K_FRAGMENT_JIT * 2, _compute_mtp2___ROWS_Q_JIT), tl.float32)
                    for n_frag in tl.static_range(0, 2):
                        v_reg_frag = tle.extract_tile(
                            v_page_reg_t,
                            index=[d_frag, n_frag],
                            tile_shape=[_compute_mtp2___K_FRAGMENT_JIT * 2, _compute_mtp2___K_FRAGMENT_JIT],
                        )
                        p_frag = _memdesc_subslice(
                            p_smem,
                            (_compute_mtp2___ROWS_Q_JIT, _compute_mtp2___K_FRAGMENT_JIT),
                            (0, n_frag * _compute_mtp2___K_FRAGMENT_JIT),
                        )
                        pv_frag = tle.gpu.wgmma(v_reg_frag, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                    pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                    pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            acc = acc * alpha[None, :] + pv
            m_i = m_new
            l_i = l_new
            start = next_start
            copy_iter += 1
            if PAGE_METADATA_K_SCALE:
                page_current_phys = page_next_phys
        has_value = l_i > 0.0
        raw_l = l_i
        if DEFERRED_NORM and task_mode != _compute_mtp2___DIRECT_MODE_JIT:
            lse = m_i
        else:
            acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
            lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if EXECUTION_STAGE == _compute_mtp2___EXECUTION_LOCAL_PARTIAL_JIT:
        if cluster_rank == 0:
            local_output_mask = valid_q[None, :]
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                acc,
                mask=local_output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                lse,
                mask=valid_q,
            )
        return
    if not REDUCTION_ONLY and task_mode == _compute_mtp2___DIRECT_MODE_JIT:
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + store_offs_v[:, None],
            acc,
            mask=valid_q[None, :],
        )
        return
    group_acc = acc
    group_lse = lse
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        peer_l_remote = tle.remote(peer_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem)).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem))
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
                peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max), 0.0)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem))
                else:
                    peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_q,)))
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            group_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        peer1_l_remote = tle.remote(peer1_l_smem, 0, scope=mesh)
        peer2_l_remote = tle.remote(peer2_l_smem, 0, scope=mesh)
        peer3_l_remote = tle.remote(peer3_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer1_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer1_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer1_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 2:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer2_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer2_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer2_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer2_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 3:
            if FULL_VIEW_DSM:
                tl.store(tle.gpu.local_ptr(peer3_acc_remote), acc)
                tl.store(tle.gpu.local_ptr(peer3_lse_remote), lse)
            else:
                tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
                tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    tl.store(tle.gpu.local_ptr(peer3_l_remote), raw_l)
                else:
                    tl.store(tle.gpu.local_ptr(peer3_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            if FULL_VIEW_DSM:
                lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem))
                lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem))
                lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem))
            else:
                lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
                lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
                lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            if DEFERRED_NORM:
                if FULL_VIEW_DSM:
                    l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem))
                    l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem))
                    l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem))
                else:
                    l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)))
                    l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem, (offs_q,)))
                    l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem, (offs_q,)))
                denom = raw_l * weight0 + l1 * weight1 + l2 * weight2 + l3 * weight3
            else:
                denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight1[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight2[None, :]
            if FULL_VIEW_DSM:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem)).to(tl.float32)
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight3[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            if DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float("inf"), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float("inf"), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float("inf"), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float("inf"), tl.exp2(lse7 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = ((seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q))[None, :]
        if group_count == 1:
            output_acc = group_acc
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                output_acc,
                mask=output_mask,
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                group_acc,
                mask=output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                group_lse,
                mask=(seq_m < _compute_mtp2___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
            )
    if EXECUTION_STAGE == _compute_mtp2___EXECUTION_FULL_JIT and group_count > 1:
        counter_idx = hkv * B + batch
        rank0_is_last = tl.zeros((), tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            if DETERMINISTIC_TAIL_ELECTION:
                deterministic_owner = group_chunk == group_count - 1
                if deterministic_owner:
                    ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    while ready != group_count - 1:
                        ready = tl.atomic_add(COMPLETION + counter_idx, 0, sem="acquire", scope="gpu")
                    rank0_is_last = tl.full((), 1, tl.int32)
                else:
                    tl.atomic_add(COMPLETION + counter_idx, 1, sem="release", scope="gpu")
            else:
                ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        if RANK0_ONLY_FINALIZER:
            is_last_cluster = (cluster_rank == 0) & (rank0_is_last != 0)
        elif DETERMINISTIC_TAIL_ELECTION:
            if TAIL_ONLY_ELECTION_BARRIER:
                if group_chunk == group_count - 1:
                    tle.distributed_barrier(mesh)
            else:
                tle.distributed_barrier(mesh)
            is_last_cluster = group_chunk == group_count - 1
        elif DSM_ELECTION_HANDOFF:
            if MERGE_CLUSTER_SIZE == 2:
                rank1_flag = tle.remote(peer_lse_smem, 1, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer_lse_smem, (0,))) != 0.0
            else:
                rank1_flag = tle.remote(peer1_lse_smem, 1, scope=mesh)
                rank2_flag = tle.remote(peer1_lse_smem, 2, scope=mesh)
                rank3_flag = tle.remote(peer1_lse_smem, 3, scope=mesh)
                if cluster_rank == 0:
                    winner_bit = rank0_is_last.to(tl.float32)
                    tl.store(tle.gpu.local_ptr(peer1_lse_smem, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank1_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank2_flag, (0,)), winner_bit)
                    tl.store(tle.gpu.local_ptr(rank3_flag, (0,)), winner_bit)
                tle.distributed_barrier(mesh)
                is_last_cluster = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (0,))) != 0.0
        else:
            tle.distributed_barrier(mesh)
            rank0_cta = cta - cluster_rank
            is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_last_cluster:
            if not RANK0_ONLY_FINALIZER and (not DSM_ELECTION_HANDOFF) and (not DETERMINISTIC_TAIL_ELECTION):
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
            if RANK0_ONLY_FINALIZER:
                _compute_mtp2___cluster_quad_head_two_chunk_finalize_mtp2(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    0,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
                _compute_mtp2___cluster_quad_head_two_chunk_finalize_mtp2(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    1,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
            elif PAIRED_HEAD_FINALIZE:
                _compute_mtp2___cluster_paired_head_finalize_mtp2(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    group_count,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                    MAX_FINAL_CHUNKS,
                    False,
                    False,
                )
            else:
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    _compute_mtp2___cluster_cooperative_finalize_mtp2(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        final_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        REUSE_FINAL_WEIGHTS,
                        False,
                        False,
                    )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")


_compute_mtp4__TASK_STRIDE = 12
_compute_mtp4__TASK_SLOTS = 2
_compute_mtp4__NUM_SEQ_Q = 4
_compute_mtp4__ROWS_Q = 32
_compute_mtp4__DIRECT_MODE = 0
_compute_mtp4__GROUP_MODE = 1
_compute_mtp4__DUMMY_MODE = 2
_compute_mtp4__SUBGROUP2_MODE = 3
_compute_mtp4__EXECUTION_FULL = 0
_compute_mtp4__EXECUTION_CLUSTER_PARTIAL = 1
_compute_mtp4__EXECUTION_LOCAL_PARTIAL = 2
_compute_mtp4__EXECUTION_ELECTION_ONLY = 3
_compute_mtp4___TASK_STRIDE_JIT = tl.constexpr(_compute_mtp4__TASK_STRIDE)
_compute_mtp4___TASK_SLOTS_JIT = tl.constexpr(_compute_mtp4__TASK_SLOTS)
_compute_mtp4___ROWS_Q_JIT = tl.constexpr(_compute_mtp4__ROWS_Q)
_compute_mtp4___NUM_SEQ_Q_JIT = tl.constexpr(_compute_mtp4__NUM_SEQ_Q)
_compute_mtp4___DIRECT_MODE_JIT = tl.constexpr(_compute_mtp4__DIRECT_MODE)
_compute_mtp4___GROUP_MODE_JIT = tl.constexpr(_compute_mtp4__GROUP_MODE)
_compute_mtp4___DUMMY_MODE_JIT = tl.constexpr(_compute_mtp4__DUMMY_MODE)
_compute_mtp4___SUBGROUP2_MODE_JIT = tl.constexpr(_compute_mtp4__SUBGROUP2_MODE)
_compute_mtp4___K_FRAGMENT_JIT = tl.constexpr(32)
_compute_mtp4___EXECUTION_FULL_JIT = tl.constexpr(_compute_mtp4__EXECUTION_FULL)
_compute_mtp4___EXECUTION_LOCAL_PARTIAL_JIT = tl.constexpr(_compute_mtp4__EXECUTION_LOCAL_PARTIAL)
_compute_mtp4___EXECUTION_ELECTION_ONLY_JIT = tl.constexpr(_compute_mtp4__EXECUTION_ELECTION_ONLY)


@triton.jit
def _compute_mtp4___packed_k_scale_ptrs_mtp4(
    KSCALE,
    phys,
    hkv,
    offs_n,
    KS_STRIDE_BLOCK: tl.constexpr,
    KS_STRIDE_TOKEN: tl.constexpr,
    KS_STRIDE_HEAD: tl.constexpr,
    KS_STRIDE_D: tl.constexpr,
):
    byte_offset = (
        phys * KS_STRIDE_BLOCK + offs_n // 32 * KS_STRIDE_TOKEN + hkv * KS_STRIDE_HEAD + offs_n % 32 * 4 * KS_STRIDE_D
    )
    return (KSCALE + byte_offset).to(tl.pointer_type(tl.float32))


@triton.jit
def _compute_mtp4___load_packed_k_scale_mtp4(
    KSCALE,
    phys,
    hkv,
    offs_n,
    KS_STRIDE_BLOCK: tl.constexpr,
    KS_STRIDE_TOKEN: tl.constexpr,
    KS_STRIDE_HEAD: tl.constexpr,
    KS_STRIDE_D: tl.constexpr,
):
    scale_ptr = _compute_mtp4___packed_k_scale_ptrs_mtp4(
        KSCALE, phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
    )
    return tl.load(scale_ptr).to(tl.float32)


@triton.jit
def _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize all four c2-owned heads and two partials in one pass."""
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    delta0 = lse0 - safe_max_lse
    delta1 = lse1 - safe_max_lse
    weight0 = tl.exp2(delta0) if USE_LOG2 else tl.exp(delta0)
    weight1 = tl.exp2(delta1) if USE_LOG2 else tl.exp(delta1)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    acc = tl.load(partial_base, mask=partial_mask, other=0.0)
    acc *= weight0[None, :, :]
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0)
    acc += partial1 * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
):
    """Finalize the two C4-owned heads and exactly two partials."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    lse_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    lse0 = tl.load(lse_base, mask=valid_head[:, None], other=-float("inf"))
    lse1 = tl.load(lse_base + LSE_STRIDE_C, mask=valid_head[:, None], other=-float("inf"))
    max_lse = tl.where(lse0 > lse1, lse0, lse1)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    weight0 = tl.exp2(lse0 - safe_max_lse) if USE_LOG2 else tl.exp(lse0 - safe_max_lse)
    weight1 = tl.exp2(lse1 - safe_max_lse) if USE_LOG2 else tl.exp(lse1 - safe_max_lse)
    weight0 = tl.where(valid_head[:, None], weight0, 0.0)
    weight1 = tl.where(valid_head[:, None], weight1, 0.0)
    denom = weight0 + weight1
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    partial0 = tl.load(partial_base, mask=partial_mask, other=0.0)
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0)
    acc = partial0 * weight0[None, :, :]
    acc += partial1 * weight1[None, :, :]
    acc *= inv_denom[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_cooperative_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    h_in_group,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    REUSE_FINAL_WEIGHTS: tl.constexpr,
):
    """Use one CTA rank per GQA head to finalize both MTP rows in-kernel."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (offs_c[:, None] < n_chunks) & valid_head
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :]
    weights_unnorm = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights_unnorm = tl.where(chunk_mask, weights_unnorm, 0.0)
    denom = tl.sum(weights_unnorm, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    if REUSE_FINAL_WEIGHTS:
        weight_rows = tl.broadcast_to(offs_c[:, None], (MAX_FINAL_CHUNKS, _compute_mtp4___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(offs_m[None, :], (MAX_FINAL_CHUNKS, _compute_mtp4___NUM_SEQ_Q_JIT))
        weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols))
        tl.store(weight_ptr, weights_unnorm * inv_denom[None, :])
        tl.debug_barrier()
    acc = tl.zeros((DV, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head
        if REUSE_FINAL_WEIGHTS:
            chunk_rows = tl.full((_compute_mtp4___NUM_SEQ_Q_JIT,), chunk, tl.int32)
            chunk_weight_ptr = tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, offs_m))
            chunk_weight = tl.load(chunk_weight_ptr, mask=chunk_valid, other=0.0)
        else:
            chunk_lse = tl.load(
                LSE
                + batch * LSE_STRIDE_B
                + chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + offs_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                mask=chunk_valid,
                other=-float("inf"),
            )
            chunk_delta = chunk_lse - safe_max_lse
            chunk_weight = (tl.exp2(chunk_delta) if USE_LOG2 else tl.exp(chunk_delta)) * inv_denom
            chunk_weight = tl.where(chunk_valid, chunk_weight, 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, :] * SO_STRIDE_M
            + hq * SO_STRIDE_H
            + offs_v[:, None],
            mask=chunk_valid,
            other=0.0,
        )
        acc += partial * chunk_weight[None, :]
    if REUSE_FINAL_WEIGHTS:
        tl.debug_barrier()
    tl.store(
        OUT + batch * O_STRIDE_B + offs_m[None, :] * O_STRIDE_M + hq * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_head & (denom[None, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_finalize_dynamic_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    FINALIZER_STAGES: tl.constexpr,
    HEAD_STRIDE: tl.constexpr = 4,
    HEAD_BASE: tl.constexpr = 0,
):
    """Online-softmax C4 finalizer with a scalar pipelined chunk loop.

    Unlike the tiled implementation, this keeps only one partial chunk live.  It
    also avoids materializing the complete weight matrix in shared memory and
    the two CTA barriers needed to consume that matrix.
    """
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + HEAD_BASE + offs_h * HEAD_STRIDE
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    max_lse = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
    denom = tl.zeros((2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.range(0, n_chunks, num_stages=FINALIZER_STAGES):
        lse = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :] * LSE_STRIDE_M
            + h_in_group[:, None] * LSE_STRIDE_HG,
            mask=valid_head[:, None],
            other=-float("inf"),
        )
        next_max = tl.maximum(max_lse, lse)
        safe_next_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        old_scale = tl.where(
            max_lse == -float("inf"),
            0.0,
            tl.exp2(max_lse - safe_next_max) if USE_LOG2 else tl.exp(max_lse - safe_next_max),
        )
        chunk_scale = tl.where(
            valid_head[:, None], tl.exp2(lse - safe_next_max) if USE_LOG2 else tl.exp(lse - safe_next_max), 0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=valid_head[None, :, None],
            other=0.0,
        )
        acc = acc * old_scale[None, :, :] + partial * chunk_scale[None, :, :]
        denom = denom * old_scale + chunk_scale
        max_lse = next_max
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc * inv_denom[None, :, :],
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    FINALIZER_TILE: tl.constexpr = 1,
    HEAD_STRIDE: tl.constexpr = 4,
    HEAD_BASE: tl.constexpr = 0,
    CHUNK_BASE: tl.constexpr = 0,
):
    """Finalize two strided GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + HEAD_BASE + offs_h * HEAD_STRIDE
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + (offs_c[:, None, None] + CHUNK_BASE) * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    pair_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    tile_offs = tl.arange(0, FINALIZER_TILE)
    for chunk_base in tl.static_range(0, MAX_FINAL_CHUNKS, FINALIZER_TILE):
        chunks = chunk_base + tile_offs
        chunk_valid = (
            (chunks[:, None, None] < n_chunks)
            & valid_head[None, :, None]
            & (offs_m[None, None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
        )
        chunk_rows = tl.broadcast_to(chunks[:, None, None], (FINALIZER_TILE, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        chunk_cols = tl.broadcast_to(pair_cols[None, :, :], (FINALIZER_TILE, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, chunk_cols)), mask=chunk_valid, other=0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + (chunks[:, None, None, None] + CHUNK_BASE) * SO_STRIDE_C
            + offs_m[None, None, None, :] * SO_STRIDE_M
            + hq[None, None, :, None] * SO_STRIDE_H
            + offs_v[None, :, None, None],
            mask=chunk_valid[:, None, :, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(partial * chunk_weight[:, None, :, :], axis=0)
    tl.debug_barrier()
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_two_chunk_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    batch,
    hkv,
    cluster_rank,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Exact-two global raw merge without generic weight staging."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    scalar_mask = valid_head[:, None]
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    m0 = tl.load(scalar_base, mask=scalar_mask, other=-float("inf"))
    m1 = tl.load(scalar_base + LSE_STRIDE_C, mask=scalar_mask, other=-float("inf"))
    l0 = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    l1 = tl.load(scalar_base + (MAX_FINAL_CHUNKS + 1) * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    max_m = tl.maximum(m0, m1)
    valid = max_m != -float("inf")
    safe_max = tl.where(valid, max_m, 0.0)
    valid0 = m0 != -float("inf")
    valid1 = m1 != -float("inf")
    both_valid = valid0 & valid1
    min_m = tl.minimum(m0, m1)
    peer_weight = tl.where(both_valid, tl.exp2(min_m - safe_max) if USE_LOG2 else tl.exp(min_m - safe_max), 0.0)
    first_is_max = m0 >= m1
    w0 = tl.where(valid0, tl.where(first_is_max, 1.0, peer_weight), 0.0)
    w1 = tl.where(valid1, tl.where(first_is_max, peer_weight, 1.0), 0.0)
    denom = l0 * w0 + l1 * w1
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    partial_base = (
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None]
    )
    partial_mask = valid_head[None, :, None]
    partial0 = tl.load(partial_base, mask=partial_mask, other=0.0).to(tl.float32)
    partial1 = tl.load(partial_base + SO_STRIDE_C, mask=partial_mask, other=0.0).to(tl.float32)
    acc = (partial0 * w0[None, :, :] + partial1 * w1[None, :, :]) * scale[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=partial_mask & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    DUAL_ACC: tl.constexpr = False,
    CHUNK4_REDUCE: tl.constexpr = False,
    REGISTER_WEIGHTS: tl.constexpr = False,
    HEAD_MAJOR_REDUCE: tl.constexpr = False,
):
    """Merge global raw (numerator, m, l) partials and normalize once."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    if HEAD_MAJOR_REDUCE:
        chunk_mask = (offs_c[None, None, :] < n_chunks) & valid_head[:, None, None]
        scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + offs_c[None, None, :] * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :, None] * LSE_STRIDE_M
            + h_in_group[:, None, None] * LSE_STRIDE_HG
        )
        m_values = tl.load(scalar_base, mask=chunk_mask, other=-float("inf"))
        l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_mask, other=0.0)
        max_m = tl.max(m_values, axis=2)
    else:
        chunk_mask = (offs_c[:, None, None] < n_chunks) & valid_head[None, :, None]
        scalar_base = (
            LSE
            + batch * LSE_STRIDE_B
            + offs_c[:, None, None] * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, None, :] * LSE_STRIDE_M
            + h_in_group[None, :, None] * LSE_STRIDE_HG
        )
        m_values = tl.load(scalar_base, mask=chunk_mask, other=-float("inf"))
        l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=chunk_mask, other=0.0)
        max_m = tl.max(m_values, axis=0)
    safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
    if HEAD_MAJOR_REDUCE:
        weights = tl.exp2(m_values - safe_max[:, :, None]) if USE_LOG2 else tl.exp(m_values - safe_max[:, :, None])
    else:
        weights = tl.exp2(m_values - safe_max[None, :, :]) if USE_LOG2 else tl.exp(m_values - safe_max[None, :, :])
    weights = tl.where(chunk_mask, weights, 0.0)
    if HEAD_MAJOR_REDUCE:
        denom = tl.sum(l_values * weights, axis=2)
    else:
        denom = tl.sum(l_values * weights, axis=0)
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    if HEAD_MAJOR_REDUCE:
        weight_rows = tl.broadcast_to(
            (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[:, :, None],
            (2, _compute_mtp4___NUM_SEQ_Q_JIT, MAX_FINAL_CHUNKS),
        )
        weight_cols = tl.broadcast_to(offs_c[None, None, :], (2, _compute_mtp4___NUM_SEQ_Q_JIT, MAX_FINAL_CHUNKS))
        scaled_weights = weights * scale[:, :, None]
    else:
        weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        weight_cols = tl.broadcast_to(
            (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
            (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
        )
        scaled_weights = weights * scale[None, :, :]
    if not REGISTER_WEIGHTS:
        tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), scaled_weights)
        tl.debug_barrier()
    pair_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    if CHUNK4_REDUCE:
        chunk_lane = tl.arange(0, 4)
        weight_cols4 = tl.broadcast_to(pair_cols[None, :, :], (4, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
        for chunk_base in tl.static_range(0, MAX_FINAL_CHUNKS, 4):
            chunk_ids = chunk_base + chunk_lane
            chunk_valid = (chunk_ids[:, None, None] < n_chunks) & valid_head[None, :, None]
            weight_rows4 = tl.broadcast_to(chunk_ids[:, None, None], (4, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
            chunk_weight = tl.load(
                tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows4, weight_cols4)), mask=chunk_valid, other=0.0
            )
            partial = tl.load(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + chunk_ids[:, None, None, None] * SO_STRIDE_C
                + offs_v[None, :, None, None]
                + offs_m[None, None, None, :] * SO_STRIDE_M
                + hq[None, None, :, None] * SO_STRIDE_H,
                mask=chunk_valid[:, None, :, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(partial * chunk_weight[:, None, :, :], axis=0)
    else:
        if DUAL_ACC:
            acc_odd = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
        for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
            chunk_valid = (chunk < n_chunks) & valid_head[:, None]
            if REGISTER_WEIGHTS:
                chunk_weight = tl.reshape(
                    tle.extract_tile(scaled_weights, [chunk, 0, 0], [1, 2, _compute_mtp4___NUM_SEQ_Q_JIT]),
                    (2, _compute_mtp4___NUM_SEQ_Q_JIT),
                )
            elif HEAD_MAJOR_REDUCE:
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(
                        FINAL_WEIGHT_SMEM, (pair_cols, tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32))
                    ),
                    mask=chunk_valid,
                    other=0.0,
                )
            else:
                chunk_weight = tl.load(
                    tle.gpu.local_ptr(
                        FINAL_WEIGHT_SMEM, (tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32), pair_cols)
                    ),
                    mask=chunk_valid,
                    other=0.0,
                )
            partial = tl.load(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + chunk * SO_STRIDE_C
                + offs_m[None, None, :] * SO_STRIDE_M
                + hq[None, :, None] * SO_STRIDE_H
                + offs_v[:, None, None],
                mask=chunk_valid[None, :, :],
                other=0.0,
            ).to(tl.float32)
            if DUAL_ACC and chunk % 2 == 1:
                acc_odd += partial * chunk_weight[None, :, :]
            else:
                acc += partial * chunk_weight[None, :, :]
        if DUAL_ACC:
            acc += acc_odd
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_raw_tail_reuse_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    WINNER_ACC_SMEM,
    WINNER_M_SMEM,
    WINNER_L_SMEM,
    mesh: tl.constexpr,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    vscale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Merge global non-winners with raw tail state retained on rank 0."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    tail_chunk = n_chunks - 1
    global_mask = (offs_c[:, None, None] < tail_chunk) & valid_head[None, :, None]
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG
    )
    m_values = tl.load(scalar_base, mask=global_mask, other=-float("inf"))
    l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=global_mask, other=0.0)
    mailbox_cols = h_in_group[:, None] + offs_m[None, :] * HEADS_PER_GROUP
    winner_acc = tle.remote(WINNER_ACC_SMEM, 0, scope=mesh)
    winner_m = tle.remote(WINNER_M_SMEM, 0, scope=mesh)
    winner_l = tle.remote(WINNER_L_SMEM, 0, scope=mesh)
    local_m = tl.load(tle.gpu.local_ptr(winner_m, (mailbox_cols,)))
    local_l = tl.load(tle.gpu.local_ptr(winner_l, (mailbox_cols,)))
    global_max = tl.max(m_values, axis=0)
    max_m = tl.maximum(global_max, local_m)
    valid = max_m != -float("inf")
    safe_max = tl.where(valid, max_m, 0.0)
    weights = tl.exp2(m_values - safe_max[None, :, :]) if USE_LOG2 else tl.exp(m_values - safe_max[None, :, :])
    weights = tl.where(global_mask, weights, 0.0)
    local_weight = tl.where(
        local_m != -float("inf"), tl.exp2(local_m - safe_max) if USE_LOG2 else tl.exp(local_m - safe_max), 0.0
    )
    denom = tl.sum(l_values * weights, axis=0) + local_l * local_weight
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    normalized_weights = weights * scale[None, :, :]
    normalized_local_weight = local_weight * scale
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 2, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    staged_weights = tl.where(
        offs_c[:, None, None] == tail_chunk, normalized_local_weight[None, :, :], normalized_weights
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), staged_weights)
    tl.debug_barrier()
    local_acc_cols = tl.broadcast_to(mailbox_cols[None, :, :], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    local_acc_rows = tl.broadcast_to(offs_v[:, None, None], (DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT))
    local_acc = tl.load(tle.gpu.local_ptr(winner_acc, (local_acc_rows, local_acc_cols))).to(tl.float32)
    pair_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_active = chunk < n_chunks
        chunk_is_tail = chunk == tail_chunk
        chunk_valid = chunk_active & valid_head[:, None]
        chunk_weight = tl.load(
            tle.gpu.local_ptr(
                FINAL_WEIGHT_SMEM, (tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32), pair_cols)
            ),
            mask=chunk_valid,
            other=0.0,
        )
        global_partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=(chunk_valid & ~chunk_is_tail)[None, :, :],
            other=0.0,
        ).to(tl.float32)
        partial = tl.where(chunk_is_tail, local_acc, global_partial)
        acc += partial * chunk_weight[None, :, :]
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4___cluster_paired_head_subgroup_reduce_mtp4(
    SPLIT_OUT,
    LSE,
    batch,
    hkv,
    cluster_rank,
    chunk_base,
    chunk_count,
    output_chunk,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    USE_LOG2: tl.constexpr,
    TREE_PACK: tl.constexpr,
):
    """Reduce one global-partial subgroup into a normalized scratch slot."""
    offs_h = tl.arange(0, 2)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 4
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    max_lse = tl.full((2, _compute_mtp4___NUM_SEQ_Q_JIT), -float("inf"), tl.float32)
    denom = tl.zeros((2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    acc = tl.zeros((DV, 2, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    for chunk_offset in tl.static_range(0, TREE_PACK):
        chunk = chunk_base + chunk_offset
        valid_chunk = chunk_offset < chunk_count
        chunk_lse = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + offs_m[None, :] * LSE_STRIDE_M
            + h_in_group[:, None] * LSE_STRIDE_HG,
            mask=valid_head[:, None] & valid_chunk,
            other=-float("inf"),
        )
        next_max = tl.maximum(max_lse, chunk_lse)
        valid = next_max != -float("inf")
        safe_next_max = tl.where(valid, next_max, 0.0)
        old_scale = tl.where(
            max_lse != -float("inf"),
            tl.exp2(max_lse - safe_next_max) if USE_LOG2 else tl.exp(max_lse - safe_next_max),
            0.0,
        )
        chunk_scale = tl.where(
            valid_head[:, None] & valid_chunk,
            tl.exp2(chunk_lse - safe_next_max) if USE_LOG2 else tl.exp(chunk_lse - safe_next_max),
            0.0,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=valid_head[None, :, None] & valid_chunk,
            other=0.0,
        )
        acc = acc * old_scale[None, :, :] + partial * chunk_scale[None, :, :]
        denom = denom * old_scale + chunk_scale
        max_lse = next_max
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    normalized = acc / safe_denom[None, :, :]
    combined_lse = tl.where(
        denom > 0.0, (tl.log2(safe_denom) if USE_LOG2 else tl.log(safe_denom)) + max_lse, -float("inf")
    )
    tl.store(
        SPLIT_OUT
        + batch * SO_STRIDE_B
        + output_chunk * SO_STRIDE_C
        + offs_m[None, None, :] * SO_STRIDE_M
        + hq[None, :, None] * SO_STRIDE_H
        + offs_v[:, None, None],
        normalized,
        mask=valid_head[None, :, None],
    )
    tl.store(
        LSE
        + batch * LSE_STRIDE_B
        + output_chunk * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, :] * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG,
        combined_lse,
        mask=valid_head[:, None],
    )


@triton.jit
def _compute_mtp4___cluster_quad_head_finalize_mtp4(
    SPLIT_OUT,
    LSE,
    OUT,
    FINAL_WEIGHT_SMEM,
    batch,
    hkv,
    cluster_rank,
    n_chunks,
    output_scale,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    DV: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    USE_LOG2: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
):
    """Finalize the four c2-owned GQA heads in one shared chunk loop."""
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    offs_h = tl.arange(0, 4)
    offs_m = tl.arange(0, _compute_mtp4___NUM_SEQ_Q_JIT)
    offs_v = tl.arange(0, DV)
    h_in_group = cluster_rank + offs_h * 2
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    chunk_mask = (
        (offs_c[:, None, None] < n_chunks)
        & valid_head[None, :, None]
        & (offs_m[None, None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
    )
    lse_values = tl.load(
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[:, None, None] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + offs_m[None, None, :] * LSE_STRIDE_M
        + h_in_group[None, :, None] * LSE_STRIDE_HG,
        mask=chunk_mask,
        other=-float("inf"),
    )
    max_lse = tl.max(lse_values, axis=0)
    safe_max_lse = tl.where(max_lse == -float("inf"), 0.0, max_lse)
    deltas = lse_values - safe_max_lse[None, :, :]
    weights = tl.exp2(deltas) if USE_LOG2 else tl.exp(deltas)
    weights = tl.where(chunk_mask, weights, 0.0)
    denom = tl.sum(weights, axis=0)
    inv_denom = tl.where(denom > 0.0, 1.0 / denom, 0.0)
    weight_rows = tl.broadcast_to(offs_c[:, None, None], (MAX_FINAL_CHUNKS, 4, _compute_mtp4___NUM_SEQ_Q_JIT))
    weight_cols = tl.broadcast_to(
        (offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :])[None, :, :],
        (MAX_FINAL_CHUNKS, 4, _compute_mtp4___NUM_SEQ_Q_JIT),
    )
    tl.store(tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (weight_rows, weight_cols)), weights * inv_denom[None, :, :])
    tl.debug_barrier()
    acc = tl.zeros((DV, 4, _compute_mtp4___NUM_SEQ_Q_JIT), tl.float32)
    quad_cols = offs_h[:, None] * _compute_mtp4___NUM_SEQ_Q_JIT + offs_m[None, :]
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < n_chunks) & valid_head[:, None] & (offs_m[None, :] < _compute_mtp4___NUM_SEQ_Q_JIT)
        chunk_rows = tl.full((4, _compute_mtp4___NUM_SEQ_Q_JIT), chunk, tl.int32)
        chunk_weight = tl.load(
            tle.gpu.local_ptr(FINAL_WEIGHT_SMEM, (chunk_rows, quad_cols)), mask=chunk_valid, other=0.0
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + offs_m[None, None, :] * SO_STRIDE_M
            + hq[None, :, None] * SO_STRIDE_H
            + offs_v[:, None, None],
            mask=chunk_valid[None, :, :],
            other=0.0,
        )
        acc += partial * chunk_weight[None, :, :]
    tl.debug_barrier()
    tl.store(
        OUT
        + batch * O_STRIDE_B
        + offs_m[None, None, :] * O_STRIDE_M
        + hq[None, :, None] * O_STRIDE_H
        + offs_v[:, None, None],
        acc * output_scale,
        mask=valid_head[None, :, None] & (denom[None, :, :] > 0.0),
    )


@triton.jit
def _compute_mtp4__fp8_kvpertensor_decode_mtp4_direct_kernel(
    Q,
    K_DESC,
    VT_DESC,
    BLOCK_IDS,
    TASK_MAP,
    QSCALE,
    KSCALE,
    VSCALE,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    NUM_SEQ_Q: tl.constexpr,
    ROWS_Q: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DIRECT_CLUSTER_BASE: tl.constexpr,
    DIRECT_CLUSTER_SIZE: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    TMA_STAGES: tl.constexpr = 2,
):
    """Public-TLE direct-task specialization with no DSM/finalizer state."""
    direct_id = tl.program_id(0)
    direct_cluster = DIRECT_CLUSTER_BASE + direct_id // DIRECT_CLUSTER_SIZE
    direct_rank = direct_id % DIRECT_CLUSTER_SIZE
    logical_cta = direct_cluster * DIRECT_CLUSTER_SIZE + direct_rank
    task_base = (logical_cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task_base + 0)
    batch = tl.load(TASK_MAP + task_base + 1)
    seq_start = tl.load(TASK_MAP + task_base + 3)
    seq_len = tl.load(TASK_MAP + task_base + 4)
    seq_kvcache = tl.load(TASK_MAP + task_base + 5)
    is_causal = tl.load(TASK_MAP + task_base + 8)
    q_smem = tle.gpu.alloc([ROWS_Q, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([ROWS_Q, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    v_raw_smem = tle.gpu.alloc([TMA_STAGES, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=BLOCK_N * D)
    vt_full = tle.gpu.alloc_barriers(num_barriers=TMA_STAGES, arrive_count=1, expect_bytes=DV * BLOCK_N)
    offs_q = tl.arange(0, ROWS_Q)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = (seq_m < NUM_SEQ_Q) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    q_rows = tl.broadcast_to(tl.arange(0, ROWS_Q)[:, None], (ROWS_Q, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (ROWS_Q, D))
    p_rows = tl.broadcast_to(tl.arange(0, ROWS_Q)[:, None], (ROWS_Q, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (ROWS_Q, BLOCK_N))
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    q = tl.load(
        Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
        mask=valid_q[:, None],
        other=0.0,
    )
    tl.store(q_smem_ptr, q)
    qscale = tl.load(QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0).to(
        tl.float32
    )
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE).to(tl.float32)
    vscale = tl.load(VSCALE).to(tl.float32) / 256.0
    m_i = tl.full((ROWS_Q,), -float("inf"), tl.float32)
    l_i = tl.zeros((ROWS_Q,), tl.float32)
    acc = tl.zeros((DV, ROWS_Q), tl.float32)
    copy_iter = 0
    start = 0
    if start < seq_len:
        phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + seq_start // BLOCK_SIZE)
        tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[0])
        tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[0])
    while start < seq_len:
        local_n = start + offs_n
        valid_cols = local_n < seq_len
        buf = copy_iter % TMA_STAGES
        phase = copy_iter // TMA_STAGES & 1
        next_start = start + BLOCK_N
        if next_start < seq_len:
            next_buf = (copy_iter + 1) % TMA_STAGES
            logical_row = seq_start + next_start
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + logical_row // BLOCK_SIZE)
            tle.gpu.copy(
                K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
            )
            tle.gpu.copy(
                VT_DESC, v_raw_smem.slot(next_buf), [1, 1, BLOCK_N, DV], [phys, hkv, 0, 0], barrier=vt_full[next_buf]
            )
        tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
        k_page = k_raw_smem.slot(buf)
        scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
        scores = tle.gpu.wgmma_wait(0, scores)
        scores *= kscale
        scores *= inv_sqrt_d * 1.4426950408889634
        scores *= qscale[None, :]
        causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
        scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float("inf"))
        m_tile = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_tile)
        valid_update = m_new != -float("inf")
        safe_m_new = tl.where(valid_update, m_new, 0.0)
        safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
        p = tl.exp2(scores - safe_m_new[None, :])
        p = tl.where(valid_update[None, :], p, 0.0)
        alpha = tl.exp2(safe_m_i - safe_m_new)
        alpha = tl.where(valid_update, alpha, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=0)
        tl.store(p_smem_ptr, tl.trans((p * 256.0).to(tl.float8e4nv)))
        tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
        v_page = v_raw_smem.slot(buf)
        v_page_t = _memdesc_transpose_2d(v_page)
        pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
        pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
        v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
        v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
        v_pairs = tl.load(v_pair_ptr)
        v_lo = (v_pairs & 255).to(tl.uint8)
        v_hi = (v_pairs >> 8).to(tl.uint8)
        v_bytes = tl.join(v_lo, v_hi)
        v_bytes = tl.permute(v_bytes, (0, 2, 1))
        v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
        v_reg = v_bytes.to(tl.float8e4nv, bitcast=True)
        pv = tle.gpu.wgmma(v_reg, p_smem, trans_b=True, out_dtype=tl.float32)
        pv = tle.gpu.wgmma_wait(0, pv)
        acc = acc * alpha[None, :] + pv
        m_i = m_new
        l_i = l_new
        start = next_start
        copy_iter += 1
    has_value = l_i > 0.0
    acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
    tl.store(
        OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + offs_v[:, None],
        acc,
        mask=valid_q[None, :],
    )


@triton.jit
def _compute_mtp4___fp8_kvpertensor_decode_mtp4_pure_tle_task(
    Q,
    K_DESC,
    VT_DESC,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    KV_READER,
    Q_REUSE_SMEM,
    LOCAL_ACC_SMEM,
    LOCAL_LSE_SMEM,
    cta,
    copy_iter_base,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    WIDE_VIEW_V_RS: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp4__EXECUTION_FULL,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    FAST_FINALIZER_HANDOFF: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    C4_FINALIZER_TILE: tl.constexpr = 1,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    PREFETCH_K_SCALE: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    C4_ALIGNED_FULL_CHUNK_RAW_512: tl.constexpr = C4_FINALIZER_TILE >= 1000 and C4_FINALIZER_TILE < 1100
    C4_ALIGNED_FULL_CHUNK_GENERIC_1024: tl.constexpr = C4_FINALIZER_TILE >= 1300 and C4_FINALIZER_TILE < 1400
    C4_ALIGNED_FULL_CHUNK_RAW: tl.constexpr = C4_ALIGNED_FULL_CHUNK_RAW_512 or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
    C2_ALIGNED_FULL_CHUNK_WINNER_1024: tl.constexpr = C4_FINALIZER_TILE >= 1100 and C4_FINALIZER_TILE < 1200
    C2_ALIGNED_FULL_CHUNK_WINNER_256: tl.constexpr = C4_FINALIZER_TILE >= 1400 and C4_FINALIZER_TILE < 1500
    C2_ALIGNED_FULL_CHUNK_WINNER: tl.constexpr = C2_ALIGNED_FULL_CHUNK_WINNER_1024 or C2_ALIGNED_FULL_CHUNK_WINNER_256
    C2_ALIGNED_FULL_CHUNK_RAW: tl.constexpr = C4_FINALIZER_TILE >= 1200 and C4_FINALIZER_TILE < 1300
    C2_ALIGNED_FULL_CHUNK: tl.constexpr = C2_ALIGNED_FULL_CHUNK_WINNER or C2_ALIGNED_FULL_CHUNK_RAW
    C4_REDUCTION_ONLY_RAW: tl.constexpr = C4_FINALIZER_TILE >= 900 and C4_FINALIZER_TILE < 1100
    PACKED_K_SCALE_LOOKAHEAD: tl.constexpr = PREFETCH_K_SCALE == 2
    PAGE_METADATA_K_SCALE: tl.constexpr = PREFETCH_K_SCALE == 5
    REGISTER_RAW_WEIGHTS: tl.constexpr = C4_FINALIZER_TILE == 704
    TAIL_RAW_DSM_REUSE: tl.constexpr = C4_FINALIZER_TILE == 706
    HEAD_MAJOR_RAW_REDUCE: tl.constexpr = C4_FINALIZER_TILE == 707
    C8_BF16_DEFERRED_NORM: tl.constexpr = C4_FINALIZER_TILE >= 800 and C4_FINALIZER_TILE < 900
    C4_GLOBAL_DEFERRED_NORM: tl.constexpr = (
        C4_FINALIZER_TILE >= 700
        and C4_FINALIZER_TILE < 800
        or C4_REDUCTION_ONLY_RAW
        or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
    )
    C4_DEFERRED_NORM: tl.constexpr = (
        C4_FINALIZER_TILE >= 600
        and C4_FINALIZER_TILE < 900
        or C4_REDUCTION_ONLY_RAW
        or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
        or C2_ALIGNED_FULL_CHUNK_RAW
    )
    C4_BF16_DSM: tl.constexpr = (
        C4_FINALIZER_TILE >= 500
        and C4_FINALIZER_TILE < 900
        or C4_REDUCTION_ONLY_RAW
        or C4_ALIGNED_FULL_CHUNK_GENERIC_1024
        or C2_ALIGNED_FULL_CHUNK_RAW
    )
    C4_HIERARCHICAL_FINALIZE: tl.constexpr = C4_FINALIZER_TILE >= 400 and C4_FINALIZER_TILE < 500
    C4_TREE_PACK: tl.constexpr = C4_FINALIZER_TILE - 400 if C4_HIERARCHICAL_FINALIZE else 0
    C4_EXACT_TWO_FINALIZE: tl.constexpr = C4_FINALIZER_TILE >= 200 and C4_FINALIZER_TILE < 300
    C4_Q_DSM_FANOUT: tl.constexpr = C4_FINALIZER_TILE >= 100 and C4_FINALIZER_TILE < 200
    C4_FINALIZER_MODE: tl.constexpr = (
        1
        if C4_HIERARCHICAL_FINALIZE or C4_BF16_DSM or C2_ALIGNED_FULL_CHUNK
        else C4_FINALIZER_TILE - 200
        if C4_EXACT_TWO_FINALIZE
        else C4_FINALIZER_TILE - 100
        if C4_Q_DSM_FANOUT
        else C4_FINALIZER_TILE
    )
    cluster_rank = tle.shard_id(mesh, "cluster_x")
    if STATIC_SCHED:
        logical_cluster = cta // MERGE_CLUSTER_SIZE
        group_chunk = logical_cluster % STATIC_MAX_GROUPS
        sequence = logical_cluster // STATIC_MAX_GROUPS
        batch = sequence % B
        hkv = sequence // B
        total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
        num_chunks = (total_len + STATIC_CHUNK_TOKENS - 1) // STATIC_CHUNK_TOKENS
        group_count = (num_chunks + MERGE_CLUSTER_SIZE - 1) // MERGE_CLUSTER_SIZE
        if group_chunk >= group_count:
            return
        chunk = group_chunk * MERGE_CLUSTER_SIZE + cluster_rank
        seq_start = chunk * STATIC_CHUNK_TOKENS
        has_work = chunk < num_chunks
        seq_len = tl.where(has_work, tl.minimum(STATIC_CHUNK_TOKENS, total_len - seq_start), 0)
        is_causal = has_work & (chunk == num_chunks - 1)
        seq_kvcache = tl.where(is_causal, seq_len - _compute_mtp4___NUM_SEQ_Q_JIT, seq_len)
        task_mode = tl.where(num_chunks == 1, _compute_mtp4___DIRECT_MODE_JIT, _compute_mtp4___GROUP_MODE_JIT)
    else:
        task_base = (cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        hkv = tl.load(TASK_MAP + task_base + 0)
        batch = tl.load(TASK_MAP + task_base + 1)
        seq_start = tl.load(TASK_MAP + task_base + 3)
        if C4_ALIGNED_FULL_CHUNK_RAW_512:
            seq_len = 512
        elif C4_ALIGNED_FULL_CHUNK_GENERIC_1024:
            seq_len = 1024
        elif C2_ALIGNED_FULL_CHUNK_WINNER_256:
            seq_len = 256
        elif C2_ALIGNED_FULL_CHUNK:
            seq_len = 1024
        else:
            seq_len = tl.load(TASK_MAP + task_base + 4)
        seq_kvcache = tl.load(TASK_MAP + task_base + 5)
        is_causal = tl.load(TASK_MAP + task_base + 8)
        if C4_ALIGNED_FULL_CHUNK_RAW:
            task_mode = _compute_mtp4___GROUP_MODE_JIT
        else:
            task_mode = tl.load(TASK_MAP + task_base + 9)
        group_chunk = tl.load(TASK_MAP + task_base + 10)
        group_count = tl.load(TASK_MAP + task_base + 11)
        if C4_ALIGNED_FULL_CHUNK_RAW:
            has_work = True
        else:
            has_work = (task_mode != _compute_mtp4___DUMMY_MODE_JIT) & (seq_len > 0)
    q_smem = tle.gpu.alloc([_compute_mtp4___ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_compute_mtp4___ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc([2, BLOCK_N, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    v_raw_smem = tle.gpu.alloc([2, BLOCK_N, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=2, arrive_count=1, expect_bytes=BLOCK_N * D)
    vt_full = tle.gpu.alloc_barriers(num_barriers=2, arrive_count=1, expect_bytes=DV * BLOCK_N)
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp4___ROWS_Q_JIT],
            dtype=tl.bfloat16 if C4_BF16_DSM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        peer_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C4_DEFERRED_NORM:
            peer_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer_l_smem = peer_lse_smem
        peer1_acc_smem = peer_acc_smem
        peer2_acc_smem = peer_acc_smem
        peer3_acc_smem = peer_acc_smem
        peer1_lse_smem = peer_lse_smem
        peer2_lse_smem = peer_lse_smem
        peer3_lse_smem = peer_lse_smem
        peer1_l_smem = peer_l_smem
        peer2_l_smem = peer_l_smem
        peer3_l_smem = peer_l_smem
    elif MERGE_CLUSTER_SIZE == 4:
        if C4_BF16_DSM:
            peer1_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.bfloat16,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer1_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_acc_smem = tle.gpu.alloc(
                [DV, _compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        peer1_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer2_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        peer3_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C4_DEFERRED_NORM:
            peer1_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer2_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
            peer3_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            peer1_l_smem = peer1_lse_smem
            peer2_l_smem = peer2_lse_smem
            peer3_l_smem = peer3_lse_smem
    else:
        partial_acc_smem = tle.gpu.alloc(
            [DV, _compute_mtp4___ROWS_Q_JIT],
            dtype=tl.bfloat16 if C8_BF16_DEFERRED_NORM else tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        partial_lse_smem = tle.gpu.alloc(
            [_compute_mtp4___ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False
        )
        if C8_BF16_DEFERRED_NORM:
            partial_l_smem = tle.gpu.alloc(
                [_compute_mtp4___ROWS_Q_JIT],
                dtype=tl.float32,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )
        else:
            partial_l_smem = partial_lse_smem
        peer1_acc_smem = partial_acc_smem
        peer2_acc_smem = partial_acc_smem
        peer3_acc_smem = partial_acc_smem
        peer1_lse_smem = partial_lse_smem
        peer2_lse_smem = partial_lse_smem
        peer3_lse_smem = partial_lse_smem
        peer1_l_smem = partial_l_smem
        peer2_l_smem = partial_l_smem
        peer3_l_smem = partial_l_smem
    if REGISTER_RAW_WEIGHTS:
        final_weight_smem = peer1_lse_smem
    else:
        final_weight_smem = tle.gpu.alloc(
            [
                2 * _compute_mtp4___NUM_SEQ_Q_JIT
                if HEAD_MAJOR_RAW_REDUCE
                else MAX_FINAL_CHUNKS
                if not QUAD_HEAD_TWO_CHUNK_FINALIZE
                else 1,
                MAX_FINAL_CHUNKS
                if HEAD_MAJOR_RAW_REDUCE
                else 2 * _compute_mtp4___NUM_SEQ_Q_JIT
                if PAIRED_HEAD_FINALIZE and MERGE_CLUSTER_SIZE == 4
                else 4 * _compute_mtp4___NUM_SEQ_Q_JIT
                if PAIRED_HEAD_FINALIZE and MERGE_CLUSTER_SIZE == 2
                else _compute_mtp4___NUM_SEQ_Q_JIT,
            ],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    offs_q = tl.arange(0, _compute_mtp4___ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _compute_mtp4___ROWS_Q_JIT)[:, None], (_compute_mtp4___ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_compute_mtp4___ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _compute_mtp4___ROWS_Q_JIT)[:, None], (_compute_mtp4___ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_compute_mtp4___ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _compute_mtp4___ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _compute_mtp4___ROWS_Q_JIT)[None, :], (DV, _compute_mtp4___ROWS_Q_JIT))
    store_offs_v = offs_v
    q_work_smem = q_smem
    q_smem_ptr = tle.gpu.local_ptr(q_work_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if MERGE_CLUSTER_SIZE == 8:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
    seq_m = offs_q // HEADS_PER_GROUP
    h_in_group = offs_q - seq_m * HEADS_PER_GROUP
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_q = has_work & (seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _compute_mtp4___ROWS_Q_JIT), tl.float32)
    lse = tl.full((_compute_mtp4___ROWS_Q_JIT,), -float("inf"), tl.float32)
    raw_l = tl.zeros((_compute_mtp4___ROWS_Q_JIT,), tl.float32)
    if has_work:
        if C4_Q_DSM_FANOUT:
            if cluster_rank == 0:
                q = tl.load(
                    Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
                    mask=valid_q[:, None],
                    other=0.0,
                )
                tl.store(q_smem_ptr, q)
            tle.distributed_barrier(mesh)
            if cluster_rank != 0:
                q_rank0_smem = tle.remote(q_smem, 0, scope=mesh)
                q = tl.load(tle.gpu.local_ptr(q_rank0_smem, (q_rows, q_cols)))
                tl.store(q_smem_ptr, q)
        else:
            q = tl.load(
                Q + batch * Q_STRIDE_B + seq_m[:, None] * Q_STRIDE_M + hq[:, None] * Q_STRIDE_H + offs_d[None, :],
                mask=valid_q[:, None],
                other=0.0,
            )
            tl.store(q_smem_ptr, q)
        qscale = tl.load(
            QSCALE + batch * QS_STRIDE_B + seq_m * QS_STRIDE_M + hq * QS_STRIDE_H, mask=valid_q, other=1.0
        ).to(tl.float32)
        if PRECOMBINE_Q_SCALE:
            qscale = qscale * (inv_sqrt_d * 1.4426950408889634)
            qscale = qscale * kscale
        m_i = tl.full((_compute_mtp4___ROWS_Q_JIT,), -float("inf"), tl.float32)
        l_i = tl.zeros((_compute_mtp4___ROWS_Q_JIT,), tl.float32)
        copy_iter = copy_iter_base
        start = 0
        current_phys = tl.full((), 0, tl.int32)
        if PACKED_K_SCALE_LOOKAHEAD:
            first_scale_block = seq_start // BLOCK_SIZE
            first_scale_phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + first_scale_block)
            lookahead_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                KSCALE, first_scale_phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
            )
        if start < seq_len:
            initial_buf = copy_iter % 2
            block_no = seq_start // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            if PAGE_METADATA_K_SCALE:
                current_phys = phys
            tle.gpu.copy(
                K_DESC, k_raw_smem.slot(initial_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[initial_buf]
            )
            tle.gpu.copy(
                VT_DESC,
                v_raw_smem.slot(initial_buf),
                [1, 1, BLOCK_N, DV],
                [phys, hkv, 0, 0],
                barrier=vt_full[initial_buf],
            )
        while start < seq_len:
            local_n = start + offs_n
            if C4_ALIGNED_FULL_CHUNK_RAW or C2_ALIGNED_FULL_CHUNK:
                valid_cols = tl.full((BLOCK_N,), True, tl.int1)
            else:
                valid_cols = local_n < seq_len
            local_copy_iter = copy_iter
            buf = local_copy_iter % 2
            phase = local_copy_iter // 2 & 1
            next_start = start + BLOCK_N
            next_phys = current_phys
            if PACKED_K_SCALE_LOOKAHEAD:
                next_lookahead_kscale = lookahead_kscale
            if next_start < seq_len:
                next_iter = local_copy_iter + 1
                next_buf = next_iter % 2
                aligned_logical = seq_start + next_start
                block_no = aligned_logical // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                if PAGE_METADATA_K_SCALE:
                    next_phys = phys
                if PACKED_K_SCALE_LOOKAHEAD:
                    next_lookahead_kscale = _compute_mtp4___load_packed_k_scale_mtp4(
                        KSCALE, phys, hkv, offs_n, KS_STRIDE_BLOCK, KS_STRIDE_TOKEN, KS_STRIDE_HEAD, KS_STRIDE_D
                    )
                tle.gpu.copy(
                    K_DESC, k_raw_smem.slot(next_buf), [1, 1, BLOCK_N, D], [phys, hkv, 0, 0], barrier=k_full[next_buf]
                )
                tle.gpu.copy(
                    VT_DESC,
                    v_raw_smem.slot(next_buf),
                    [1, 1, BLOCK_N, DV],
                    [phys, hkv, 0, 0],
                    barrier=vt_full[next_buf],
                )
            consume_lookahead: tl.constexpr = False
            first_lookahead = consume_lookahead & (start == 0)
            if consume_lookahead:
                if first_lookahead:
                    tle.gpu.barrier_wait(LOCAL_ACC_SMEM[0], phaseIdx=0)
                    tle.gpu.barrier_wait(LOCAL_LSE_SMEM[0], phaseIdx=0)
                else:
                    tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            else:
                tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            if consume_lookahead and first_lookahead:
                k_page = KV_READER
            else:
                k_page = k_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED or WIDE_VIEW_V_RS:
                scores = tle.gpu.wgmma(k_page, q_work_smem, trans_b=True, out_dtype=tl.float32)
            else:
                scores = tl.zeros((BLOCK_N, _compute_mtp4___ROWS_Q_JIT), tl.float32)
                for frag in tl.static_range(0, 4):
                    k_frag = _memdesc_subslice(
                        k_page, (BLOCK_N, _compute_mtp4___K_FRAGMENT_JIT), (0, frag * _compute_mtp4___K_FRAGMENT_JIT)
                    )
                    q_frag = _memdesc_subslice(
                        q_work_smem,
                        (_compute_mtp4___ROWS_Q_JIT, _compute_mtp4___K_FRAGMENT_JIT),
                        (0, frag * _compute_mtp4___K_FRAGMENT_JIT),
                    )
                    scores = tle.gpu.wgmma(k_frag, q_frag, acc=scores, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores)
            if not PRECOMBINE_Q_SCALE:
                scores = scores * kscale
            if not PRECOMBINE_Q_SCALE:
                scores = scores * (inv_sqrt_d * 1.4426950408889634)
            scores = scores * qscale[None, :]
            causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + seq_m[None, :] + 1)
            scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float("inf"))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float("inf")
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float("inf"), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            tl.store(p_smem_ptr, p_scaled_t)
            if not (consume_lookahead and first_lookahead):
                tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            if consume_lookahead and first_lookahead:
                v_page = Q_REUSE_SMEM
            else:
                v_page = v_raw_smem.slot(buf)
            if LDSM_REGISTER_SHARED:
                v_page_t = _memdesc_transpose_2d(v_page)
                pair_rows = tl.broadcast_to((tl.arange(0, DV // 2) * 2)[:, None], (DV // 2, BLOCK_N))
                pair_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (DV // 2, BLOCK_N))
                v_pair_ptr = tle.gpu.local_ptr(v_page_t, (pair_rows, pair_cols))
                v_pair_ptr = v_pair_ptr.to(tl.pointer_type(tl.uint16, address_space=3))
                v_pairs = tl.load(v_pair_ptr)
                v_lo = (v_pairs & 255).to(tl.uint8)
                v_hi = (v_pairs >> 8).to(tl.uint8)
                v_bytes = tl.join(v_lo, v_hi)
                v_bytes = tl.permute(v_bytes, (0, 2, 1))
                v_bytes = tl.reshape(v_bytes, (DV, BLOCK_N))
                v_page_reg_t = v_bytes.to(tl.float8e4nv, bitcast=True)
            elif WIDE_VIEW_V_RS:
                v_page_t = _memdesc_transpose_2d(v_page)
                v_page_reg_t = tl.load(tle.gpu.local_ptr(v_page_t))
            else:
                v_rows = tl.broadcast_to(tl.arange(0, BLOCK_N)[:, None], (BLOCK_N, DV))
                v_cols = tl.broadcast_to(tl.arange(0, DV)[None, :], (BLOCK_N, DV))
                v_page_reg_t = tl.trans(tl.load(tle.gpu.local_ptr(v_page, (v_rows, v_cols))))
            if LDSM_REGISTER_SHARED or WIDE_VIEW_V_RS:
                pv = tle.gpu.wgmma(v_page_reg_t, p_smem, trans_b=True, out_dtype=tl.float32)
                pv = tle.gpu.wgmma_wait(0, pv)
            else:
                pv = tl.zeros((DV, _compute_mtp4___ROWS_Q_JIT), tl.float32)
                for d_frag in tl.static_range(0, 2):
                    pv_frag = tl.zeros((_compute_mtp4___K_FRAGMENT_JIT * 2, _compute_mtp4___ROWS_Q_JIT), tl.float32)
                    for n_frag in tl.static_range(0, 2):
                        v_reg_frag = tle.extract_tile(
                            v_page_reg_t,
                            index=[d_frag, n_frag],
                            tile_shape=[_compute_mtp4___K_FRAGMENT_JIT * 2, _compute_mtp4___K_FRAGMENT_JIT],
                        )
                        p_frag = _memdesc_subslice(
                            p_smem,
                            (_compute_mtp4___ROWS_Q_JIT, _compute_mtp4___K_FRAGMENT_JIT),
                            (0, n_frag * _compute_mtp4___K_FRAGMENT_JIT),
                        )
                        pv_frag = tle.gpu.wgmma(v_reg_frag, p_frag, acc=pv_frag, trans_b=True, out_dtype=tl.float32)
                    pv_frag = tle.gpu.wgmma_wait(0, pv_frag)
                    pv = tle.insert_tile(pv, pv_frag, index=[d_frag, 0])
            acc = acc * alpha[None, :] + pv
            m_i = m_new
            l_i = l_new
            start = next_start
            copy_iter += 1
            if PACKED_K_SCALE_LOOKAHEAD:
                lookahead_kscale = next_lookahead_kscale
            if PAGE_METADATA_K_SCALE:
                current_phys = next_phys
        has_value = l_i > 0.0
        raw_l = l_i
        if C4_DEFERRED_NORM:
            if not C4_REDUCTION_ONLY_RAW and task_mode == _compute_mtp4___DIRECT_MODE_JIT:
                acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
                lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
            else:
                lse = m_i
        else:
            local_vscale = vscale
            acc = tl.where(has_value[None, :], acc / l_i[None, :] * local_vscale, 0.0)
            lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
    if not C4_REDUCTION_ONLY_RAW and task_mode == _compute_mtp4___DIRECT_MODE_JIT:
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m[None, :] * O_STRIDE_M + hq[None, :] * O_STRIDE_H + store_offs_v[:, None],
            acc,
            mask=valid_q[None, :],
        )
        return
    if MERGE_CLUSTER_SIZE == 4 and (not C4_REDUCTION_ONLY_RAW) and (task_mode == _compute_mtp4___SUBGROUP2_MODE_JIT):
        low_pair_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        low_pair_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        low_pair_l_remote = tle.remote(peer1_l_smem, 0, scope=mesh)
        high_pair_acc_remote = tle.remote(peer3_acc_smem, 2, scope=mesh)
        high_pair_lse_remote = tle.remote(peer3_lse_smem, 2, scope=mesh)
        high_pair_l_remote = tle.remote(peer3_l_smem, 2, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(low_pair_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(low_pair_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(low_pair_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(high_pair_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(high_pair_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(high_pair_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0 or cluster_rank == 2:
            if cluster_rank == 0:
                peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
                if C4_DEFERRED_NORM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)))
            else:
                peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
                peer_lse = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
                if C4_DEFERRED_NORM:
                    peer_l = tl.load(tle.gpu.local_ptr(peer3_l_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max), 0.0)
            if C4_DEFERRED_NORM:
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            subgroup_acc = (acc * weight0[None, :] + peer_acc * weight1[None, :]) / safe_denom[None, :]
            if C4_DEFERRED_NORM:
                subgroup_acc *= vscale
            subgroup_acc = tl.where(valid_group[None, :], subgroup_acc, 0.0)
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                subgroup_acc,
                mask=valid_q[None, :],
            )
        return
    if EXECUTION_STAGE == _compute_mtp4___EXECUTION_LOCAL_PARTIAL_JIT:
        if cluster_rank == 0:
            local_output_mask = valid_q[None, :]
            tl.store(
                SPLIT_OUT
                + batch * SO_STRIDE_B
                + group_chunk * SO_STRIDE_C
                + seq_m[None, :] * SO_STRIDE_M
                + hq[None, :] * SO_STRIDE_H
                + store_offs_v[:, None],
                acc,
                mask=local_output_mask,
            )
            tl.store(
                LSE
                + batch * LSE_STRIDE_B
                + group_chunk * LSE_STRIDE_C
                + hkv * LSE_STRIDE_HKV
                + seq_m * LSE_STRIDE_M
                + h_in_group * LSE_STRIDE_HG,
                lse,
                mask=valid_q,
            )
        return
    group_acc = acc
    group_lse = lse
    group_raw_acc = acc
    group_raw_m = lse
    group_raw_l = tl.full((_compute_mtp4___ROWS_Q_JIT,), 1.0, tl.float32)
    if MERGE_CLUSTER_SIZE == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        peer_l_remote = tle.remote(peer_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(peer_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            if C4_DEFERRED_NORM:
                peer_l = tl.load(tle.gpu.local_ptr(peer_l_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(peer_lse != -float("inf"), tl.exp2(peer_lse - safe_max), 0.0)
            if C4_DEFERRED_NORM:
                denom = raw_l * weight0 + peer_l * weight1
            else:
                denom = weight0 + weight1
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :] + peer_acc * weight1[None, :]
            group_raw_acc = weighted_acc
            group_raw_m = safe_max
            group_raw_l = denom
            group_acc = weighted_acc / safe_denom[None, :]
            if C4_DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    elif MERGE_CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        peer1_l_remote = tle.remote(peer1_l_smem, 0, scope=mesh)
        peer2_l_remote = tle.remote(peer2_l_smem, 0, scope=mesh)
        peer3_l_remote = tle.remote(peer3_l_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(peer1_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 2:
            tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(peer2_l_remote, (offs_q,)), raw_l)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
            if C4_DEFERRED_NORM:
                tl.store(tle.gpu.local_ptr(peer3_l_remote, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            if C4_DEFERRED_NORM:
                l1 = tl.load(tle.gpu.local_ptr(peer1_l_smem, (offs_q,)))
                l2 = tl.load(tle.gpu.local_ptr(peer2_l_smem, (offs_q,)))
                l3 = tl.load(tle.gpu.local_ptr(peer3_l_smem, (offs_q,)))
                denom = raw_l * weight0 + l1 * weight1 + l2 * weight2 + l3 * weight3
            else:
                denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight1[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight2[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc * weight3[None, :]
            group_raw_acc = weighted_acc
            group_raw_m = safe_max
            group_raw_l = denom
            if C4_GLOBAL_DEFERRED_NORM:
                group_acc = weighted_acc
                group_lse = safe_max
            else:
                group_acc = weighted_acc / safe_denom[None, :]
                if C4_DEFERRED_NORM:
                    group_acc *= vscale
                group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
                group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        if C8_BF16_DEFERRED_NORM:
            tl.store(tle.gpu.local_ptr(partial_l_smem, (offs_q,)), raw_l)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        peer1_l = tle.remote(partial_l_smem, 1, scope=mesh)
        peer2_l = tle.remote(partial_l_smem, 2, scope=mesh)
        peer3_l = tle.remote(partial_l_smem, 3, scope=mesh)
        peer4_l = tle.remote(partial_l_smem, 4, scope=mesh)
        peer5_l = tle.remote(partial_l_smem, 5, scope=mesh)
        peer6_l = tle.remote(partial_l_smem, 6, scope=mesh)
        peer7_l = tle.remote(partial_l_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float("inf")
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float("inf"), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float("inf"), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float("inf"), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float("inf"), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float("inf"), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float("inf"), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float("inf"), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float("inf"), tl.exp2(lse7 - safe_max), 0.0)
            if C8_BF16_DEFERRED_NORM:
                l1 = tl.load(tle.gpu.local_ptr(peer1_l, (offs_q,)))
                l2 = tl.load(tle.gpu.local_ptr(peer2_l, (offs_q,)))
                l3 = tl.load(tle.gpu.local_ptr(peer3_l, (offs_q,)))
                l4 = tl.load(tle.gpu.local_ptr(peer4_l, (offs_q,)))
                l5 = tl.load(tle.gpu.local_ptr(peer5_l, (offs_q,)))
                l6 = tl.load(tle.gpu.local_ptr(peer6_l, (offs_q,)))
                l7 = tl.load(tle.gpu.local_ptr(peer7_l, (offs_q,)))
                denom = (
                    raw_l * weight0
                    + l1 * weight1
                    + l2 * weight2
                    + l3 * weight3
                    + l4 * weight4
                    + l5 * weight5
                    + l6 * weight6
                    + l7 * weight7
                )
            else:
                denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols))).to(tl.float32)
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            if C8_BF16_DEFERRED_NORM:
                group_acc *= vscale
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float("inf"))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = ((seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q))[None, :]
        if not C4_REDUCTION_ONLY_RAW and group_count == 1:
            output_acc = group_acc
            if C4_GLOBAL_DEFERRED_NORM:
                output_denom = tl.where(group_raw_l > 0.0, group_raw_l, 1.0)
                output_acc = group_raw_acc / output_denom[None, :] * vscale
                output_acc = tl.where((group_raw_l > 0.0)[None, :], output_acc, 0.0)
            tl.store(
                OUT
                + batch * O_STRIDE_B
                + seq_m[None, :] * O_STRIDE_M
                + hq[None, :] * O_STRIDE_H
                + store_offs_v[:, None],
                output_acc,
                mask=output_mask,
            )
        else:
            stored_acc = group_raw_acc if C4_GLOBAL_DEFERRED_NORM else group_acc
            defer_exact_two_store = False & C4_GLOBAL_DEFERRED_NORM & (MERGE_CLUSTER_SIZE == 4) & (group_count == 2)
            defer_tail_store = TAIL_RAW_DSM_REUSE & (group_count > 2) & (group_chunk == group_count - 1)
            if defer_tail_store:
                tl.store(tle.gpu.local_ptr(peer1_acc_smem), group_raw_acc)
                tl.store(tle.gpu.local_ptr(peer1_lse_smem), group_raw_m)
                tl.store(tle.gpu.local_ptr(peer1_l_smem), group_raw_l)
            if not defer_exact_two_store | defer_tail_store:
                tl.store(
                    SPLIT_OUT
                    + batch * SO_STRIDE_B
                    + group_chunk * SO_STRIDE_C
                    + seq_m[None, :] * SO_STRIDE_M
                    + hq[None, :] * SO_STRIDE_H
                    + store_offs_v[:, None],
                    stored_acc,
                    mask=output_mask,
                )
                tl.store(
                    LSE
                    + batch * LSE_STRIDE_B
                    + group_chunk * LSE_STRIDE_C
                    + hkv * LSE_STRIDE_HKV
                    + seq_m * LSE_STRIDE_M
                    + h_in_group * LSE_STRIDE_HG,
                    group_raw_m if C4_GLOBAL_DEFERRED_NORM else group_lse,
                    mask=(seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
                )
                if C4_GLOBAL_DEFERRED_NORM:
                    tl.store(
                        LSE
                        + batch * LSE_STRIDE_B
                        + (group_chunk + MAX_FINAL_CHUNKS) * LSE_STRIDE_C
                        + hkv * LSE_STRIDE_HKV
                        + seq_m * LSE_STRIDE_M
                        + h_in_group * LSE_STRIDE_HG,
                        group_raw_l,
                        mask=(seq_m < _compute_mtp4___NUM_SEQ_Q_JIT) & (h_in_group < HEADS_PER_GROUP) & (hq < H_Q),
                    )
    if (
        C4_HIERARCHICAL_FINALIZE
        and EXECUTION_STAGE == _compute_mtp4___EXECUTION_FULL_JIT
        and (group_count > C4_TREE_PACK)
    ):
        counter_idx = hkv * B + batch
        num_counters = B * (H_Q // HEADS_PER_GROUP)
        subgroup = group_chunk // C4_TREE_PACK
        subgroup_start = subgroup * C4_TREE_PACK
        subgroup_remaining = group_count - subgroup_start
        subgroup_count = tl.where(subgroup_remaining < C4_TREE_PACK, subgroup_remaining, C4_TREE_PACK)
        subgroup_counter_idx = num_counters + counter_idx * MAX_FINAL_CHUNKS + subgroup
        rank0_is_subgroup_last = tl.full((), 0, tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            subgroup_ticket = tl.atomic_add(COMPLETION + subgroup_counter_idx, 1, sem="acq_rel", scope="gpu")
            rank0_is_subgroup_last = (subgroup_ticket == subgroup_count - 1).to(tl.int32)
            tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_subgroup_last, sem="release", scope="gpu")
        tle.distributed_barrier(mesh)
        rank0_cta = cta - cluster_rank
        if cluster_rank == 0:
            is_subgroup_last = rank0_is_subgroup_last != 0
        else:
            is_subgroup_last = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if is_subgroup_last:
            scratch_chunk = MAX_FINAL_CHUNKS + subgroup
            _compute_mtp4___cluster_paired_head_subgroup_reduce_mtp4(
                SPLIT_OUT,
                LSE,
                batch,
                hkv,
                cluster_rank,
                subgroup_start,
                subgroup_count,
                scratch_chunk,
                H_Q,
                HEADS_PER_GROUP,
                DV,
                SO_STRIDE_B,
                SO_STRIDE_C,
                SO_STRIDE_M,
                SO_STRIDE_H,
                LSE_STRIDE_B,
                LSE_STRIDE_C,
                LSE_STRIDE_HKV,
                LSE_STRIDE_M,
                LSE_STRIDE_HG,
                True,
                C4_TREE_PACK,
            )
            tle.distributed_barrier(mesh)
            rank0_is_top_last = tl.full((), 0, tl.int32)
            if cluster_rank == 0:
                tl.debug_barrier()
                num_subgroups = (group_count + C4_TREE_PACK - 1) // C4_TREE_PACK
                top_ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
                rank0_is_top_last = (top_ticket == num_subgroups - 1).to(tl.int32)
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_top_last, sem="release", scope="gpu")
                tl.atomic_xchg(COMPLETION + subgroup_counter_idx, 0, sem="release", scope="gpu")
            tle.distributed_barrier(mesh)
            if cluster_rank == 0:
                is_top_last = rank0_is_top_last != 0
            else:
                is_top_last = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            if is_top_last:
                num_subgroups = (group_count + C4_TREE_PACK - 1) // C4_TREE_PACK
                _compute_mtp4___cluster_paired_head_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    final_weight_smem,
                    batch,
                    hkv,
                    cluster_rank,
                    num_subgroups,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                    MAX_FINAL_CHUNKS // C4_TREE_PACK,
                    1,
                    4,
                    0,
                    MAX_FINAL_CHUNKS,
                )
            if not SKIP_TRAILING_FINALIZER_BARRIER:
                tle.distributed_barrier(mesh)
            if cluster_rank == 0 and is_top_last:
                tl.debug_barrier()
                tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
        return
    if (
        EXECUTION_STAGE == _compute_mtp4___EXECUTION_FULL_JIT
        or EXECUTION_STAGE == _compute_mtp4___EXECUTION_ELECTION_ONLY_JIT
    ) and (C4_REDUCTION_ONLY_RAW or group_count > 1):
        counter_idx = hkv * B + batch
        num_counters = B * (H_Q // HEADS_PER_GROUP)
        rank0_is_last = tl.full((), 0, tl.int32)
        if cluster_rank == 0:
            tl.debug_barrier()
            ticket = tl.atomic_add(COMPLETION + counter_idx, 1, sem="acq_rel", scope="gpu")
            rank0_is_last = (ticket == group_count - 1).to(tl.int32)
            if not RANK0_ONLY_FINALIZER:
                tl.atomic_xchg(LAST_FLAGS + cta, rank0_is_last, sem="release", scope="gpu")
        rank0_cta = cta - cluster_rank
        if RANK0_ONLY_FINALIZER:
            is_last_cluster = (cluster_rank == 0) & (rank0_is_last != 0)
        else:
            tle.distributed_barrier(mesh)
            if FAST_FINALIZER_HANDOFF:
                if cluster_rank == 0:
                    is_last_cluster = rank0_is_last != 0
                else:
                    is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
            else:
                is_last_cluster = tl.atomic_add(LAST_FLAGS + rank0_cta, 0, sem="acquire", scope="gpu") != 0
        if EXECUTION_STAGE == _compute_mtp4___EXECUTION_ELECTION_ONLY_JIT:
            if cluster_rank == 0 and rank0_is_last != 0:
                tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")
            return
        if is_last_cluster:
            if not FAST_FINALIZER_HANDOFF and (not RANK0_ONLY_FINALIZER):
                tl.atomic_add(COMPLETION + counter_idx, 0, sem="acq_rel", scope="gpu")
            if RANK0_ONLY_FINALIZER:
                if PAIRED_HEAD_FINALIZE:
                    _compute_mtp4___cluster_quad_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        0,
                        group_count,
                        1.0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                    )
                    _compute_mtp4___cluster_quad_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        1,
                        group_count,
                        1.0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                    )
                else:
                    _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                    )
                    _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        1,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                    )
            elif QUAD_HEAD_TWO_CHUNK_FINALIZE:
                _compute_mtp4___cluster_quad_head_two_chunk_finalize_mtp4(
                    SPLIT_OUT,
                    LSE,
                    OUT,
                    batch,
                    hkv,
                    cluster_rank,
                    H_Q,
                    HEADS_PER_GROUP,
                    DV,
                    SO_STRIDE_B,
                    SO_STRIDE_C,
                    SO_STRIDE_M,
                    SO_STRIDE_H,
                    LSE_STRIDE_B,
                    LSE_STRIDE_C,
                    LSE_STRIDE_HKV,
                    LSE_STRIDE_M,
                    LSE_STRIDE_HG,
                    O_STRIDE_B,
                    O_STRIDE_M,
                    O_STRIDE_H,
                    True,
                )
            elif PAIRED_HEAD_FINALIZE:
                if MERGE_CLUSTER_SIZE == 2:
                    _compute_mtp4___cluster_quad_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        1.0,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                    )
                elif C4_EXACT_TWO_FINALIZE and group_count == 2:
                    _compute_mtp4___cluster_paired_head_two_chunk_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        cluster_rank,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                    )
                elif C4_FINALIZER_MODE < 0:
                    _compute_mtp4___cluster_paired_head_finalize_dynamic_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        -C4_FINALIZER_MODE,
                    )
                elif C4_GLOBAL_DEFERRED_NORM:
                    if group_count == 2:
                        _compute_mtp4___cluster_paired_head_raw_two_chunk_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            batch,
                            hkv,
                            cluster_rank,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            True,
                            MAX_FINAL_CHUNKS,
                        )
                    elif TAIL_RAW_DSM_REUSE:
                        _compute_mtp4___cluster_paired_head_raw_tail_reuse_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            peer1_acc_smem,
                            peer1_lse_smem,
                            peer1_l_smem,
                            mesh,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            True,
                            MAX_FINAL_CHUNKS,
                        )
                    else:
                        _compute_mtp4___cluster_paired_head_raw_finalize_mtp4(
                            SPLIT_OUT,
                            LSE,
                            OUT,
                            final_weight_smem,
                            batch,
                            hkv,
                            cluster_rank,
                            group_count,
                            vscale,
                            H_Q,
                            HEADS_PER_GROUP,
                            DV,
                            SO_STRIDE_B,
                            SO_STRIDE_C,
                            SO_STRIDE_M,
                            SO_STRIDE_H,
                            LSE_STRIDE_B,
                            LSE_STRIDE_C,
                            LSE_STRIDE_HKV,
                            LSE_STRIDE_M,
                            LSE_STRIDE_HG,
                            O_STRIDE_B,
                            O_STRIDE_M,
                            O_STRIDE_H,
                            True,
                            MAX_FINAL_CHUNKS,
                            False,
                            False,
                            REGISTER_RAW_WEIGHTS,
                            HEAD_MAJOR_RAW_REDUCE,
                        )
                else:
                    _compute_mtp4___cluster_paired_head_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        cluster_rank,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        C4_FINALIZER_MODE,
                    )
            else:
                for head_pass in tl.static_range(0, 8 // MERGE_CLUSTER_SIZE):
                    final_h_in_group = cluster_rank + head_pass * MERGE_CLUSTER_SIZE
                    _compute_mtp4___cluster_cooperative_finalize_mtp4(
                        SPLIT_OUT,
                        LSE,
                        OUT,
                        final_weight_smem,
                        batch,
                        hkv,
                        final_h_in_group,
                        group_count,
                        H_Q,
                        HEADS_PER_GROUP,
                        DV,
                        SO_STRIDE_B,
                        SO_STRIDE_C,
                        SO_STRIDE_M,
                        SO_STRIDE_H,
                        LSE_STRIDE_B,
                        LSE_STRIDE_C,
                        LSE_STRIDE_HKV,
                        LSE_STRIDE_M,
                        LSE_STRIDE_HG,
                        O_STRIDE_B,
                        O_STRIDE_M,
                        O_STRIDE_H,
                        True,
                        MAX_FINAL_CHUNKS,
                        True,
                    )
        if not SKIP_TRAILING_FINALIZER_BARRIER:
            tle.distributed_barrier(mesh)
        elif TAIL_RAW_DSM_REUSE and is_last_cluster:
            tle.distributed_barrier(mesh)
        if cluster_rank == 0 and is_last_cluster:
            tl.debug_barrier()
            tl.atomic_xchg(COMPLETION + counter_idx, 0, sem="release", scope="gpu")


@triton.jit
def _compute_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle_kernel(
    Q,
    K_DESC,
    VT_DESC,
    BLOCK_IDS,
    TASK_MAP,
    SEQLENS_KV,
    QSCALE,
    KSCALE,
    VSCALE,
    SPLIT_OUT,
    LSE,
    COMPLETION,
    LAST_FLAGS,
    OUT,
    mesh: tl.constexpr,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_STRIDE_B: tl.constexpr,
    Q_STRIDE_M: tl.constexpr,
    Q_STRIDE_H: tl.constexpr,
    QS_STRIDE_B: tl.constexpr,
    QS_STRIDE_M: tl.constexpr,
    QS_STRIDE_H: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    LDSM_REGISTER_SHARED: tl.constexpr = False,
    WIDE_VIEW_V_RS: tl.constexpr = False,
    MERGE_CLUSTER_SIZE: tl.constexpr = 8,
    EXECUTION_STAGE: tl.constexpr = _compute_mtp4__EXECUTION_FULL,
    MAX_FINAL_CHUNKS: tl.constexpr = 16,
    PAIRED_HEAD_FINALIZE: tl.constexpr = False,
    QUAD_HEAD_TWO_CHUNK_FINALIZE: tl.constexpr = False,
    FAST_FINALIZER_HANDOFF: tl.constexpr = False,
    RANK0_ONLY_FINALIZER: tl.constexpr = False,
    SKIP_TRAILING_FINALIZER_BARRIER: tl.constexpr = False,
    PDL_NOTIFY: tl.constexpr = False,
    C4_FINALIZER_TILE: tl.constexpr = 1,
    KS_STRIDE_BLOCK: tl.constexpr = 0,
    KS_STRIDE_TOKEN: tl.constexpr = 0,
    KS_STRIDE_HEAD: tl.constexpr = 0,
    KS_STRIDE_D: tl.constexpr = 0,
    PREFETCH_K_SCALE: tl.constexpr = False,
    PRECOMBINE_Q_SCALE: tl.constexpr = False,
    STATIC_SCHED: tl.constexpr = False,
    STATIC_CHUNK_TOKENS: tl.constexpr = 0,
    STATIC_MAX_GROUPS: tl.constexpr = 1,
):
    """Launch exactly one logical task per CTA (clean nonpersistent path)."""
    cta = tl.program_id(0)
    if STATIC_SCHED:
        logical_cluster = cta // MERGE_CLUSTER_SIZE
        static_group = logical_cluster % STATIC_MAX_GROUPS
        static_sequence = logical_cluster // STATIC_MAX_GROUPS
        static_batch = static_sequence % B
        static_total_len = tl.load(SEQLENS_KV + static_batch).to(tl.int32)
        static_num_chunks = (static_total_len + STATIC_CHUNK_TOKENS - 1) // STATIC_CHUNK_TOKENS
        static_group_count = (static_num_chunks + MERGE_CLUSTER_SIZE - 1) // MERGE_CLUSTER_SIZE
        static_valid = static_group < static_group_count
        hkv = 0
    else:
        task_base = (cta * _compute_mtp4___TASK_SLOTS_JIT + 1) * _compute_mtp4___TASK_STRIDE_JIT
        hkv = tl.load(TASK_MAP + task_base + 0)
        static_valid = False
    if STATIC_SCHED and static_valid or (not STATIC_SCHED and hkv >= 0):
        _compute_mtp4___fp8_kvpertensor_decode_mtp4_pure_tle_task(
            Q,
            K_DESC,
            VT_DESC,
            BLOCK_IDS,
            TASK_MAP,
            SEQLENS_KV,
            QSCALE,
            KSCALE,
            VSCALE,
            SPLIT_OUT,
            LSE,
            COMPLETION,
            LAST_FLAGS,
            OUT,
            0,
            0,
            0,
            0,
            cta,
            0,
            mesh,
            B,
            H_Q,
            HEADS_PER_GROUP,
            D,
            DV,
            BLOCK_SIZE,
            MAX_BLOCKS,
            BLOCK_N,
            Q_STRIDE_B,
            Q_STRIDE_M,
            Q_STRIDE_H,
            QS_STRIDE_B,
            QS_STRIDE_M,
            QS_STRIDE_H,
            SO_STRIDE_B,
            SO_STRIDE_C,
            SO_STRIDE_M,
            SO_STRIDE_H,
            LSE_STRIDE_B,
            LSE_STRIDE_C,
            LSE_STRIDE_HKV,
            LSE_STRIDE_M,
            LSE_STRIDE_HG,
            O_STRIDE_B,
            O_STRIDE_M,
            O_STRIDE_H,
            LDSM_REGISTER_SHARED,
            WIDE_VIEW_V_RS,
            MERGE_CLUSTER_SIZE,
            EXECUTION_STAGE,
            MAX_FINAL_CHUNKS,
            PAIRED_HEAD_FINALIZE,
            QUAD_HEAD_TWO_CHUNK_FINALIZE,
            FAST_FINALIZER_HANDOFF,
            RANK0_ONLY_FINALIZER,
            SKIP_TRAILING_FINALIZER_BARRIER,
            C4_FINALIZER_TILE,
            KS_STRIDE_BLOCK,
            KS_STRIDE_TOKEN,
            KS_STRIDE_HEAD,
            KS_STRIDE_D,
            PREFETCH_K_SCALE,
            PRECOMBINE_Q_SCALE,
            STATIC_SCHED,
            STATIC_CHUNK_TOKENS,
            STATIC_MAX_GROUPS,
        )
    if PDL_NOTIFY:
        tl.extra.cuda.gdc_launch_dependents()


_finalize_mtp4___NUM_SEQ_Q = tl.constexpr(4)
_finalize_mtp4___HEADS_PER_PROGRAM = tl.constexpr(4)


@triton.jit
def _finalize_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize_kernel(
    KV_LENS,
    SPLIT_OUT,
    LSE,
    VSCALE,
    OUT,
    B: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    MAX_FINAL_CHUNKS: tl.constexpr,
    HEAD_PASSES: tl.constexpr,
    SO_STRIDE_B: tl.constexpr,
    SO_STRIDE_C: tl.constexpr,
    SO_STRIDE_M: tl.constexpr,
    SO_STRIDE_H: tl.constexpr,
    LSE_STRIDE_B: tl.constexpr,
    LSE_STRIDE_C: tl.constexpr,
    LSE_STRIDE_HKV: tl.constexpr,
    LSE_STRIDE_M: tl.constexpr,
    LSE_STRIDE_HG: tl.constexpr,
    O_STRIDE_B: tl.constexpr,
    O_STRIDE_M: tl.constexpr,
    O_STRIDE_H: tl.constexpr,
    V_PER_HEAD: tl.constexpr = False,
    EXACT_TWO_SPECIALIZATION: tl.constexpr = False,
    PDL_WAIT: tl.constexpr = False,
):
    """CUDA-shaped four-head reduction over BF16 raw (numerator, m, l)."""
    if PDL_WAIT:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    programs_per_sequence = _finalize_mtp4___NUM_SEQ_Q * HEAD_PASSES
    sequence_id = pid // programs_per_sequence
    sequence_program = pid - sequence_id * programs_per_sequence
    seq_m = sequence_program // HEAD_PASSES
    head_pass = sequence_program - seq_m * HEAD_PASSES
    hkv = sequence_id // B
    batch = sequence_id - hkv * B
    offs_h = tl.arange(0, _finalize_mtp4___HEADS_PER_PROGRAM)
    offs_d = tl.arange(0, D)
    h_in_group = head_pass * _finalize_mtp4___HEADS_PER_PROGRAM + offs_h
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_head = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    total_len = tl.load(KV_LENS + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    if group_count <= 1:
        return
    if EXACT_TWO_SPECIALIZATION and group_count == 2:
        scalar_base = (
            LSE + batch * LSE_STRIDE_B + hkv * LSE_STRIDE_HKV + seq_m * LSE_STRIDE_M + h_in_group * LSE_STRIDE_HG
        )
        m0 = tl.load(scalar_base, mask=valid_head, other=-float("inf"))
        m1 = tl.load(scalar_base + LSE_STRIDE_C, mask=valid_head, other=-float("inf"))
        l0 = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=valid_head, other=0.0)
        l1 = tl.load(scalar_base + (MAX_FINAL_CHUNKS + 1) * LSE_STRIDE_C, mask=valid_head, other=0.0)
        max_m = tl.maximum(m0, m1)
        safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
        w0 = tl.where(m0 == -float("inf"), 0.0, tl.exp2(m0 - safe_max))
        w1 = tl.where(m1 == -float("inf"), 0.0, tl.exp2(m1 - safe_max))
        denom = l0 * w0 + l1 * w1
        vscale = tl.load(VSCALE + hkv if V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
        scale = tl.where(denom > 0.0, vscale / denom, 0.0)
        out_base = SPLIT_OUT + batch * SO_STRIDE_B + seq_m * SO_STRIDE_M + hq[:, None] * SO_STRIDE_H + offs_d[None, :]
        acc = tl.load(out_base, mask=valid_head[:, None], other=0.0).to(tl.float32) * w0[:, None]
        partial = tl.load(out_base + SO_STRIDE_C, mask=valid_head[:, None], other=0.0).to(tl.float32)
        acc = (acc + partial * w1[:, None]) * scale[:, None]
        tl.store(
            OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[:, None] * O_STRIDE_H + offs_d[None, :],
            acc,
            mask=valid_head[:, None] & (denom[:, None] > 0.0),
        )
        return
    offs_c = tl.arange(0, MAX_FINAL_CHUNKS)
    scalar_base = (
        LSE
        + batch * LSE_STRIDE_B
        + offs_c[None, :] * LSE_STRIDE_C
        + hkv * LSE_STRIDE_HKV
        + seq_m * LSE_STRIDE_M
        + h_in_group[:, None] * LSE_STRIDE_HG
    )
    scalar_mask = valid_head[:, None] & (offs_c[None, :] < group_count)
    m_values = tl.load(scalar_base, mask=scalar_mask, other=-float("inf"))
    l_values = tl.load(scalar_base + MAX_FINAL_CHUNKS * LSE_STRIDE_C, mask=scalar_mask, other=0.0)
    max_m = tl.max(m_values, axis=1)
    safe_max = tl.where(max_m == -float("inf"), 0.0, max_m)
    weights = tl.where(scalar_mask, tl.exp2(m_values - safe_max[:, None]), 0.0)
    denom = tl.sum(l_values * weights, axis=1)
    vscale = tl.load(VSCALE + hkv if V_PER_HEAD else VSCALE + 0).to(tl.float32) / 256.0
    scale = tl.where(denom > 0.0, vscale / denom, 0.0)
    acc = tl.zeros((_finalize_mtp4___HEADS_PER_PROGRAM, D), tl.float32)
    for chunk in tl.static_range(0, MAX_FINAL_CHUNKS):
        chunk_valid = (chunk < group_count) & valid_head
        chunk_m = tl.load(
            LSE
            + batch * LSE_STRIDE_B
            + chunk * LSE_STRIDE_C
            + hkv * LSE_STRIDE_HKV
            + seq_m * LSE_STRIDE_M
            + h_in_group * LSE_STRIDE_HG,
            mask=chunk_valid,
            other=-float("inf"),
        )
        chunk_weight = tl.where(chunk_valid, tl.exp2(chunk_m - safe_max) * scale, 0.0)
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_STRIDE_B
            + chunk * SO_STRIDE_C
            + seq_m * SO_STRIDE_M
            + hq[:, None] * SO_STRIDE_H
            + offs_d[None, :],
            mask=chunk_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += partial * chunk_weight[:, None]
    tl.store(
        OUT + batch * O_STRIDE_B + seq_m * O_STRIDE_M + hq[:, None] * O_STRIDE_H + offs_d[None, :],
        acc,
        mask=valid_head[:, None] & (denom[:, None] > 0.0),
    )


_runtime__BLOCK_SIZE = 64
_runtime__TILE_N = 64
_runtime_mtp1__NUM_SEQ_Q = 1
_runtime__HEAD_DIM = 128


@dataclass(frozen=True)
class _runtime_mtp1__DecodeConfig:
    cluster_size: int
    chunk_tokens: int

    def __post_init__(self) -> None:
        if self.cluster_size not in (2, 4, 8):
            raise ValueError("runtime policy supports cluster_size 2, 4, or 8")
        if self.chunk_tokens not in (128, 256, 512, 1024):
            raise ValueError("runtime supports chunk_tokens 128, 256, 512, or 1024")


@dataclass
class _runtime_mtp1__DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def num_batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def max_seq_kv(self) -> int:
        return int(self.kv_lens.max().item())


_runtime__CLUSTER_MESHES = (
    {
        2: tle.device_mesh({"block_cluster": [("cluster_x", 2)]}),
        4: tle.device_mesh({"block_cluster": [("cluster_x", 4)]}),
        8: tle.device_mesh({"block_cluster": [("cluster_x", 8)]}),
    }
    if USE_TLE
    else {2: None, 4: None, 8: None}
)


def _runtime__validate_inputs(inputs, num_seq_q: int) -> tuple[int, int, int]:
    if not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA tensor for GPU task scheduling")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _runtime__HEAD_DIM:
        raise ValueError("q must have flattened shape [batch * MTP, num_head_q, 128]")
    if inputs.q.shape[0] != inputs.num_batch * num_seq_q:
        raise ValueError(f"MTP={num_seq_q} requires q.shape[0] == batch * {num_seq_q}")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.ndim != 4 or cache.shape[1] != _runtime__BLOCK_SIZE or cache.shape[3] != _runtime__HEAD_DIM:
            raise ValueError(f"{name} must be logical [block, 64, head, 128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if num_head_q % num_head_kv:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    heads_per_group = num_head_q // num_head_kv
    if heads_per_group > 8:
        raise ValueError("heads_per_group must be <= 8")
    return (num_head_q, num_head_kv, heads_per_group)


def _runtime__make_paged_kv_descriptors(inputs, num_seq_q: int) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Project NHD/HND paged descriptors to logical [block, head, token, dim] tiles."""
    _runtime__validate_inputs(inputs, num_seq_q)
    block_shape = [1, 1, _runtime__TILE_N, _runtime__HEAD_DIM]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache.permute(0, 2, 1, 3), block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache.permute(0, 2, 1, 3), block_shape=block_shape),
    )


_runtime_mtp2__NUM_SEQ_Q = 2

_runtime_mtp2__DecodeConfig = _runtime_mtp1__DecodeConfig

_runtime_mtp2__DecodeInputs = _runtime_mtp1__DecodeInputs


_runtime_mtp4__NUM_SEQ_Q = 4


@dataclass(frozen=True)
class _runtime_mtp4__DecodeConfig:
    cluster_size: int
    chunk_tokens: int
    direct_threshold: int = 0
    subgroup2_threshold: int = 0

    def __post_init__(self) -> None:
        if self.cluster_size not in (2, 4, 8):
            raise ValueError("runtime policy supports cluster_size 2, 4, or 8")
        if self.chunk_tokens < _runtime__TILE_N or self.chunk_tokens > 4096 or self.chunk_tokens % _runtime__TILE_N:
            raise ValueError("MTP=4 chunk_tokens must be a multiple of 64 in [64, 4096]")
        if self.direct_threshold < 0 or self.direct_threshold % _runtime__TILE_N:
            raise ValueError("direct_threshold must be zero or a multiple of 64")
        if self.subgroup2_threshold < 0 or self.subgroup2_threshold % _runtime__TILE_N:
            raise ValueError("subgroup2_threshold must be zero or a multiple of 64")
        if self.subgroup2_threshold and self.cluster_size != 4:
            raise ValueError("dual-C2 subgroup packing requires cluster_size=4")


_runtime_mtp4__C2_T256 = _runtime_mtp4__DecodeConfig(2, 256)
_runtime_mtp4__C2_T1024 = _runtime_mtp4__DecodeConfig(2, 1024)
_runtime_mtp4__C4_T512 = _runtime_mtp4__DecodeConfig(4, 512)
_runtime_mtp4__C4_T1024 = _runtime_mtp4__DecodeConfig(4, 1024)

_runtime_mtp4__DecodeInputs = _runtime_mtp1__DecodeInputs


@dataclass
class _runtime_mtp4__DecodeWorkspace:
    config: _runtime_mtp4__DecodeConfig
    schedule: _scheduler_mtp24__DecodeTaskSchedule
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    all_chunks_aligned: bool = False
    final_policy_mode: str = "winner"
    static_sched: bool = False
    static_chunk_tokens: int = 0
    static_max_groups: int = 1

    @property
    def stats(self) -> dict:
        return dict(self.schedule.stats)


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace | None,
    *,
    execution_stage: Literal["full", "cluster", "local", "election"],
    paired_head_finalize: bool | None,
    direct_fast_path: bool,
    quad_head_two_chunk_finalize: bool,
    fast_finalizer_handoff: bool,
    rank0_only_finalizer: bool,
    skip_trailing_finalizer_barrier: bool,
    maxnreg: int | None,
    c4_bf16_dsm: bool,
    c4_deferred_norm: bool,
    c4_global_deferred_norm: bool,
    c4_reduction_only_raw: bool,
    c4_aligned_full_chunk_raw: bool,
    c2_aligned_full_chunk_winner: bool,
    full_view_v_rs: bool,
    pdl_notify: bool,
    prefetch_k_scale: bool,
    page_metadata_k_scale: bool,
    tma_k_scale: bool,
    precombine_q_scale: bool,
    flatten_output: bool,
) -> torch.Tensor:
    """Run the independent public-TLE MTP=4 n32 typed-RS policy."""
    if workspace is None:
        raise ValueError("MTP=4 requires an explicitly configured workspace")
    if tma_k_scale:
        raise ValueError("TLE K-scale TMA requires quant_type=0")
    if page_metadata_k_scale:
        raise ValueError("page-metadata K-scale pipeline requires quant_type=0")
    scalar_k_scale_prefetch = prefetch_k_scale and (not tma_k_scale)
    if (
        sum(
            (
                bool(value)
                for value in (scalar_k_scale_prefetch, False, False, False, page_metadata_k_scale, tma_k_scale)
            )
        )
        > 1
    ):
        raise ValueError(
            "K-scale prefetch, lookahead, shared pipeline, WGMMA shadow load, page-metadata pipeline and TMA are exclusive"
        )
    ks = (0, 0, 0, 0)
    stage_map = {
        "full": _compute_mtp4__EXECUTION_FULL,
        "cluster": _compute_mtp4__EXECUTION_CLUSTER_PARTIAL,
        "local": _compute_mtp4__EXECUTION_LOCAL_PARTIAL,
        "election": _compute_mtp4__EXECUTION_ELECTION_ONLY,
    }
    if execution_stage not in stage_map:
        raise ValueError("execution_stage must be 'full', 'cluster', 'local', or 'election'")
    if maxnreg is not None and (not 32 <= maxnreg <= 255):
        raise ValueError("maxnreg must be in [32, 255] or None")
    if c4_deferred_norm and (not c4_bf16_dsm):
        raise ValueError("C4 deferred normalization requires BF16 DSM")
    if c4_global_deferred_norm and (not c4_deferred_norm):
        raise ValueError("C4 global deferred normalization requires DSM deferred normalization")
    if c4_reduction_only_raw:
        if workspace.config not in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024):
            raise ValueError("reduction-only raw specialization requires C4T512/C4T1024")
        if (
            any((int(workspace.stats.get(name, 0)) for name in ("direct_tasks", "dummy_tasks")))
            or workspace.config.subgroup2_threshold
        ):
            raise ValueError("reduction-only raw specialization requires grouped real tasks only")
        if c4_bf16_dsm or c4_deferred_norm or c4_global_deferred_norm:
            raise ValueError("reduction-only raw mode internally selects the complete C4 raw ABI")
    if c4_aligned_full_chunk_raw:
        if workspace.config == _runtime_mtp4__C4_T512 and (not c4_reduction_only_raw):
            raise ValueError("C4T512 aligned full-chunk raw requires reduction-only mode")
        if workspace.config == _runtime_mtp4__C4_T1024 and c4_reduction_only_raw:
            raise ValueError("C4T1024 aligned full-chunk raw must retain single-group finalization")
        if workspace.config not in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024):
            raise ValueError("aligned full-chunk raw requires C4T512/C4T1024")
        if not workspace.all_chunks_aligned:
            raise ValueError("aligned full-chunk raw requires aligned KV lengths")
    if c2_aligned_full_chunk_winner:
        if workspace.config not in (_runtime_mtp4__C2_T256, _runtime_mtp4__C2_T1024):
            raise ValueError("aligned winner specialization requires C2T256/C2T1024")
        if not workspace.all_chunks_aligned:
            raise ValueError("aligned winner requires chunk-aligned KV lengths")
        incompatible = {
            name: int(workspace.stats.get(name, 0))
            for name in ("direct_tasks", "dummy_tasks", "subgroup2_tasks")
            if int(workspace.stats.get(name, 0))
        }
        if incompatible:
            raise ValueError(f"aligned winner requires grouped real tasks only: {incompatible}")
        aligned_c2_raw = c4_bf16_dsm and c4_deferred_norm and (not c4_global_deferred_norm)
        if any((c4_reduction_only_raw, c4_aligned_full_chunk_raw, False)):
            raise ValueError("aligned C2 compute cannot combine with C4/C8 raw modes")
        if any((c4_bf16_dsm, c4_deferred_norm, c4_global_deferred_norm)) and (not aligned_c2_raw):
            raise ValueError("aligned C2 raw compute requires BF16 DSM plus deferred norm")
    if paired_head_finalize is None:
        paired_head_finalize = workspace.config.cluster_size == 4
    if paired_head_finalize and workspace.config.cluster_size not in (2, 4):
        raise ValueError("paired_head_finalize requires cluster_size=2 or 4")
    if paired_head_finalize and workspace.heads_per_group != 8:
        raise ValueError("paired_head_finalize requires heads_per_group=8")
    if pdl_notify and direct_fast_path:
        raise ValueError("PDL notification cannot be combined with direct_fast_path")
    effective_chunks_max = int(workspace.stats.get("effective_chunks_max", 1))
    if quad_head_two_chunk_finalize and effective_chunks_max != 2:
        raise ValueError(f"quad_head_two_chunk_finalize requires effective_chunks_max == 2, got {effective_chunks_max}")
    if quad_head_two_chunk_finalize and workspace.config.cluster_size != 2:
        raise ValueError("quad_head_two_chunk_finalize requires cluster_size=2")
    if quad_head_two_chunk_finalize and workspace.heads_per_group != 8:
        raise ValueError("quad_head_two_chunk_finalize requires heads_per_group=8")
    if quad_head_two_chunk_finalize and paired_head_finalize:
        raise ValueError("select either exact-two or arbitrary-chunk c2 quad finalization")
    c2_quad_finalizer = (quad_head_two_chunk_finalize or paired_head_finalize) and workspace.config.cluster_size == 2
    c4_paired_finalizer = paired_head_finalize and workspace.config.cluster_size == 4
    if c4_bf16_dsm and (not (c4_paired_finalizer or c2_quad_finalizer)):
        raise ValueError("BF16 DSM requires either the C4 paired path or a C2 quad-head path")
    c8_rank_sharded_finalizer = not paired_head_finalize and workspace.config.cluster_size == 8
    if fast_finalizer_handoff and (not (c2_quad_finalizer or c4_paired_finalizer or c8_rank_sharded_finalizer)):
        raise ValueError(
            "fast_finalizer_handoff requires either the c2 quad-head path, the c4 paired-head path, or the c8 rank-sharded path"
        )
    if rank0_only_finalizer and (not c2_quad_finalizer):
        raise ValueError("rank0_only_finalizer requires a c2 quad-head finalizer")
    if rank0_only_finalizer and fast_finalizer_handoff:
        raise ValueError("rank0_only_finalizer replaces rather than combines with fast_finalizer_handoff")
    if skip_trailing_finalizer_barrier and (
        not (
            rank0_only_finalizer
            or (fast_finalizer_handoff and (c2_quad_finalizer or c4_paired_finalizer or c8_rank_sharded_finalizer))
        )
    ):
        raise ValueError(
            "skip_trailing_finalizer_barrier requires either the c2 rank0-only path or a rank-sharded fast handoff path"
        )
    k_desc, v_desc = _runtime__make_paged_kv_descriptors(inputs, _runtime_mtp4__NUM_SEQ_Q)
    ws = workspace
    num_head_q = int(inputs.q.shape[1])
    reduction_clusters = int(ws.stats.get("reduction_clusters", ws.schedule.num_clusters))
    direct_tasks = int(ws.stats.get("direct_tasks", 0))
    use_direct_fast = direct_fast_path and reduction_clusters == 0
    logical_clusters = 0 if use_direct_fast else ws.schedule.num_clusters
    launch_clusters = min(logical_clusters, logical_clusters)

    def launch_direct() -> None:
        if not use_direct_fast or direct_tasks == 0:
            return
        _compute_mtp4__fp8_kvpertensor_decode_mtp4_direct_kernel[direct_tasks,](
            ws.q_4d,
            k_desc,
            v_desc,
            inputs.block_ids,
            ws.schedule.task_map,
            ws.q_scale_3d,
            inputs.k_scale,
            inputs.v_scale,
            ws.out,
            H_Q=num_head_q,
            HEADS_PER_GROUP=ws.heads_per_group,
            NUM_SEQ_Q=_runtime_mtp4__NUM_SEQ_Q,
            ROWS_Q=_runtime_mtp4__NUM_SEQ_Q * ws.heads_per_group,
            D=_runtime__HEAD_DIM,
            DV=_runtime__HEAD_DIM,
            BLOCK_SIZE=_runtime__BLOCK_SIZE,
            MAX_BLOCKS=inputs.block_ids.shape[1],
            BLOCK_N=_runtime__TILE_N,
            DIRECT_CLUSTER_BASE=reduction_clusters,
            DIRECT_CLUSTER_SIZE=ws.config.cluster_size,
            Q_STRIDE_B=ws.q_4d.stride(0),
            Q_STRIDE_M=ws.q_4d.stride(1),
            Q_STRIDE_H=ws.q_4d.stride(2),
            QS_STRIDE_B=ws.q_scale_3d.stride(0),
            QS_STRIDE_M=ws.q_scale_3d.stride(1),
            QS_STRIDE_H=ws.q_scale_3d.stride(2),
            O_STRIDE_B=ws.out.stride(0),
            O_STRIDE_M=ws.out.stride(1),
            O_STRIDE_H=ws.out.stride(2),
            TMA_STAGES=2,
            num_ctas=1,
            num_warps=4,
            num_stages=3,
            maxnreg=maxnreg,
            launch_pdl=False,
        )

    if logical_clusters == 0:
        launch_direct()
        if flatten_output:
            return ws.out.reshape(inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM)
        return ws.out
    compute_kernel = _compute_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle_kernel
    compute_kernel[launch_clusters,](
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.schedule.task_map,
        inputs.kv_lens,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=_runtime__CLUSTER_MESHES[ws.config.cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=_runtime__HEAD_DIM,
        DV=_runtime__HEAD_DIM,
        BLOCK_SIZE=_runtime__BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=_runtime__TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_M=ws.q_4d.stride(1),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_M=ws.q_scale_3d.stride(1),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        LDSM_REGISTER_SHARED=not full_view_v_rs,
        WIDE_VIEW_V_RS=full_view_v_rs,
        MERGE_CLUSTER_SIZE=ws.config.cluster_size,
        EXECUTION_STAGE=stage_map[execution_stage],
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.schedule.partial_slots),
        PAIRED_HEAD_FINALIZE=paired_head_finalize,
        QUAD_HEAD_TWO_CHUNK_FINALIZE=quad_head_two_chunk_finalize,
        FAST_FINALIZER_HANDOFF=fast_finalizer_handoff,
        RANK0_ONLY_FINALIZER=rank0_only_finalizer,
        SKIP_TRAILING_FINALIZER_BARRIER=skip_trailing_finalizer_barrier,
        PDL_NOTIFY=pdl_notify,
        C4_FINALIZER_TILE=(
            1200 + 1
            if c4_bf16_dsm and c4_deferred_norm
            else 1400 + 1
            if workspace.config == _runtime_mtp4__C2_T256
            else 1100 + 1
        )
        if c2_aligned_full_chunk_winner
        else (1300 + 1 if workspace.config == _runtime_mtp4__C4_T1024 else 1000 + 1)
        if c4_aligned_full_chunk_raw
        else 900 + 1
        if c4_reduction_only_raw
        else 700 + 1
        if c4_global_deferred_norm
        else 600 + 1
        if c4_deferred_norm
        else 500 + 1
        if c4_bf16_dsm
        else 1,
        KS_STRIDE_BLOCK=ks[0],
        KS_STRIDE_TOKEN=ks[1],
        KS_STRIDE_HEAD=ks[2],
        KS_STRIDE_D=ks[3],
        PREFETCH_K_SCALE=5 if page_metadata_k_scale else prefetch_k_scale,
        PRECOMBINE_Q_SCALE=precombine_q_scale,
        STATIC_SCHED=ws.static_sched,
        STATIC_CHUNK_TOKENS=ws.static_chunk_tokens,
        STATIC_MAX_GROUPS=ws.static_max_groups,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        maxnreg=maxnreg,
        launch_pdl=pdl_notify,
    )
    launch_direct()
    if flatten_output:
        return ws.out.reshape(inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM)
    return ws.out


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_winner(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    prefetch_k_scale: bool,
    tma_k_scale: bool,
    precombine_q_scale: bool,
    aligned_full_chunk_winner: bool,
) -> torch.Tensor:
    """Run the currently validated MTP=4 policy specializations.

    Policy selection remains outside the low-level launch implementation.
    """
    config = workspace.config
    effective_chunks_max = int(workspace.stats.get("effective_chunks_max", 1))
    use_c4_paired_fast = config.cluster_size == 4
    use_c2_exact_two = config.cluster_size == 2 and effective_chunks_max == 2
    use_c2_rank0_quad = config.cluster_size == 2 and effective_chunks_max == 4
    use_c2_sharded_quad = config.cluster_size == 2 and effective_chunks_max >= 32
    use_general_c2_quad = use_c2_rank0_quad or use_c2_sharded_quad
    use_fast_handoff = use_c4_paired_fast or use_c2_sharded_quad
    use_rank0 = use_c2_exact_two or use_c2_rank0_quad
    selected_maxnreg = 240 if use_c4_paired_fast else None
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        workspace,
        paired_head_finalize=use_c4_paired_fast or use_general_c2_quad,
        direct_fast_path=True,
        quad_head_two_chunk_finalize=use_c2_exact_two,
        fast_finalizer_handoff=use_fast_handoff,
        rank0_only_finalizer=use_rank0,
        skip_trailing_finalizer_barrier=use_c4_paired_fast or use_c2_exact_two or use_general_c2_quad,
        maxnreg=selected_maxnreg,
        prefetch_k_scale=prefetch_k_scale,
        tma_k_scale=tma_k_scale,
        page_metadata_k_scale=False,
        precombine_q_scale=precombine_q_scale,
        c2_aligned_full_chunk_winner=aligned_full_chunk_winner,
        flatten_output=True,
        execution_stage="full",
        c4_bf16_dsm=False,
        c4_deferred_norm=False,
        c4_global_deferred_norm=False,
        c4_reduction_only_raw=False,
        c4_aligned_full_chunk_raw=False,
        full_view_v_rs=False,
        pdl_notify=False,
    )


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_c2_bf16_dsm_specialized(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    page_metadata_k_scale: bool,
    precombine_q_scale: bool,
    aligned_full_chunk_winner: bool,
) -> torch.Tensor:
    """Migrate the C4 BF16/raw DSM producer path to exact-two C2."""
    if workspace.config.cluster_size != 2 or int(workspace.stats.get("effective_chunks_max", 1)) != 2:
        raise ValueError("C2 BF16 DSM requires an exact-two-group schedule")
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        workspace,
        paired_head_finalize=False,
        direct_fast_path=True,
        quad_head_two_chunk_finalize=True,
        rank0_only_finalizer=True,
        skip_trailing_finalizer_barrier=True,
        c4_bf16_dsm=True,
        c4_deferred_norm=True,
        full_view_v_rs=True,
        maxnreg=None,
        prefetch_k_scale=False,
        tma_k_scale=False,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        c2_aligned_full_chunk_winner=aligned_full_chunk_winner,
        flatten_output=True,
        execution_stage="full",
        fast_finalizer_handoff=False,
        c4_global_deferred_norm=False,
        c4_reduction_only_raw=False,
        c4_aligned_full_chunk_raw=False,
        pdl_notify=False,
    )


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize(
    inputs: _runtime_mtp4__DecodeInputs,
    workspace: _runtime_mtp4__DecodeWorkspace,
    *,
    use_pdl: bool,
    maxnreg: int | None,
    page_metadata_k_scale: bool,
    precombine_q_scale: bool,
    reduction_only_raw: bool,
    aligned_full_chunk_raw: bool,
) -> torch.Tensor:
    """C2/C4 BF16-DSM raw producer plus a CUDA-shaped PDL finalizer."""
    ws = workspace
    if ws.config.cluster_size not in (2, 4):
        raise ValueError("detached raw finalizer requires C2 or C4")
    if reduction_only_raw and ws.config not in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024):
        raise ValueError("detached reduction-only raw producer requires C4T512/C4T1024")
    if aligned_full_chunk_raw:
        if ws.config == _runtime_mtp4__C4_T512 and (not reduction_only_raw):
            raise ValueError("C4T512 aligned producer requires reduction-only raw")
        if ws.config == _runtime_mtp4__C4_T1024 and reduction_only_raw:
            raise ValueError("C4T1024 aligned producer must retain single-group finalization")
    _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        ws,
        execution_stage="cluster",
        paired_head_finalize=True,
        c4_bf16_dsm=not reduction_only_raw,
        c4_deferred_norm=not reduction_only_raw,
        c4_global_deferred_norm=not reduction_only_raw,
        c4_reduction_only_raw=reduction_only_raw,
        c4_aligned_full_chunk_raw=aligned_full_chunk_raw,
        full_view_v_rs=True,
        pdl_notify=use_pdl,
        maxnreg=240 if maxnreg is None else maxnreg,
        tma_k_scale=False,
        page_metadata_k_scale=page_metadata_k_scale,
        precombine_q_scale=precombine_q_scale,
        flatten_output=False,
        direct_fast_path=False,
        quad_head_two_chunk_finalize=False,
        fast_finalizer_handoff=False,
        rank0_only_finalizer=False,
        skip_trailing_finalizer_barrier=False,
        c2_aligned_full_chunk_winner=False,
        prefetch_k_scale=False,
    )
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    head_passes = (ws.heads_per_group + 3) // 4
    grid = inputs.num_batch * num_head_kv * _runtime_mtp4__NUM_SEQ_Q * head_passes
    reducer_kernel = _finalize_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize_kernel
    reducer_kwargs = dict(
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=_runtime__HEAD_DIM,
        CHUNK_TOKENS=ws.config.chunk_tokens,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.schedule.partial_slots),
        HEAD_PASSES=head_passes,
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        V_PER_HEAD=False,
        PDL_WAIT=use_pdl,
        num_warps=4,
        num_stages=1,
        launch_pdl=use_pdl,
    )
    reducer_kernel[grid,](
        inputs.kv_lens,
        ws.split_out,
        ws.lse,
        inputs.v_scale,
        ws.out,
        CLUSTER_SIZE=ws.config.cluster_size,
        EXACT_TWO_SPECIALIZATION=False,
        **reducer_kwargs,
    )
    return ws.out.reshape(inputs.num_batch * _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM)


def _runtime_mtp4___resolve_mtp4_final_policy(workspace: _runtime_mtp4__DecodeWorkspace) -> str:
    """Resolve the validated backend from config and scheduler metadata."""
    config = workspace.config
    chunks = int(workspace.stats.get("effective_chunks_max", 1))
    if config.cluster_size == 2 and config.chunk_tokens == 1024 and (chunks == 2):
        return "c2-raw"
    if config.cluster_size == 4 and config.chunk_tokens == 512 and (chunks >= 32):
        return "pdl"
    if config.cluster_size == 4 and config.chunk_tokens == 1024:
        if chunks >= 16:
            return "pdl-s"
        if chunks >= 8:
            return "pdl"
    return "winner"


def _runtime_mtp4__fp8_kvpertensor_decode_mtp4_final(
    inputs: _runtime_mtp4__DecodeInputs, workspace: _runtime_mtp4__DecodeWorkspace
) -> torch.Tensor:
    """Run the unified final MTP=4 interface.

    ``winner`` and ``c2-raw`` select compiled specializations of the fused TLE
    implementation.  ``pdl`` and ``pdl-s`` retain the two-launch raw producer
    plus detached PDL reducer design behind this single runtime API.
    """
    resolved = workspace.final_policy_mode
    if resolved == "winner":
        aligned_c2 = (
            workspace.config in (_runtime_mtp4__C2_T256, _runtime_mtp4__C2_T1024)
            and workspace.all_chunks_aligned
            and (not int(workspace.stats.get("direct_tasks", 0)))
            and (not int(workspace.stats.get("dummy_tasks", 0)))
            and (not int(workspace.stats.get("subgroup2_tasks", 0)))
        )
        winner_tma_k_scale = False
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_winner(
            inputs,
            workspace,
            prefetch_k_scale=winner_tma_k_scale,
            tma_k_scale=winner_tma_k_scale,
            precombine_q_scale=winner_tma_k_scale,
            aligned_full_chunk_winner=aligned_c2,
        )
    if resolved == "c2-raw":
        aligned_c2 = (
            workspace.config == _runtime_mtp4__C2_T1024
            and workspace.all_chunks_aligned
            and (not int(workspace.stats.get("direct_tasks", 0)))
            and (not int(workspace.stats.get("dummy_tasks", 0)))
            and (not int(workspace.stats.get("subgroup2_tasks", 0)))
        )
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_c2_bf16_dsm_specialized(
            inputs, workspace, page_metadata_k_scale=None, precombine_q_scale=None, aligned_full_chunk_winner=aligned_c2
        )
    if resolved in ("pdl", "pdl-s"):
        producer_maxnreg = 240
        aligned_c4 = (
            workspace.config in (_runtime_mtp4__C4_T512, _runtime_mtp4__C4_T1024)
            and workspace.all_chunks_aligned
            and (not int(workspace.stats.get("direct_tasks", 0)))
            and (not int(workspace.stats.get("dummy_tasks", 0)))
            and (not workspace.config.subgroup2_threshold)
            and (
                workspace.config == _runtime_mtp4__C4_T512
                and resolved == "pdl"
                or (workspace.config == _runtime_mtp4__C4_T1024 and resolved in ("pdl", "pdl-s"))
            )
        )
        aligned_reduction_only = aligned_c4 and workspace.config == _runtime_mtp4__C4_T512
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize(
            inputs,
            workspace,
            use_pdl=True,
            maxnreg=producer_maxnreg,
            reduction_only_raw=aligned_reduction_only,
            aligned_full_chunk_raw=aligned_c4,
            page_metadata_k_scale=None,
            precombine_q_scale=aligned_c4,
        )
    raise ValueError(f"unsupported MTP=4 final policy: {resolved}")


@dataclass
class _static_mtp1__StaticDecodeWorkspace:
    """Buffers for static scheduling; no task map is allocated or refreshed."""

    config: _runtime_mtp1__DecodeConfig
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    num_head_kv: int
    max_groups: int
    aligned_chunk_tokens: int
    reduction_only: bool


def _static_mtp1__prepare_static_decode_workspace(
    inputs: _runtime_mtp1__DecodeInputs, config: _runtime_mtp1__DecodeConfig
) -> _static_mtp1__StaticDecodeWorkspace:
    """Allocate static buffers while retaining the dynamic winner's config."""
    num_head_q, num_head_kv, heads_per_group = _runtime__validate_inputs(inputs, _runtime_mtp1__NUM_SEQ_Q)
    if heads_per_group != 8:
        raise ValueError("the MTP=1 static reference requires GQA8")
    batch = inputs.num_batch
    cluster_size = config.cluster_size
    chunk_tokens = config.chunk_tokens
    max_chunks = (inputs.max_seq_kv + chunk_tokens - 1) // chunk_tokens
    max_groups = (max_chunks + cluster_size - 1) // cluster_size
    num_chunks_gpu = (inputs.kv_lens + chunk_tokens - 1) // chunk_tokens
    split_only = bool(torch.all((num_chunks_gpu > 1) & (num_chunks_gpu % cluster_size == 0)).item())
    dynamic_aligned_configs = {(2, 256), (2, 1024), (4, 256), (4, 512), (4, 1024), (8, 512)}
    aligned_full_chunk = (
        split_only
        and (cluster_size, chunk_tokens) in dynamic_aligned_configs
        and bool(torch.all(inputs.kv_lens % chunk_tokens == 0).item())
    )
    aligned_chunk_tokens = chunk_tokens if aligned_full_chunk else 0
    reduction_only = aligned_full_chunk or (
        split_only and (cluster_size, chunk_tokens) in ((2, 1024), (8, 512), (4, 512))
    )
    physical_ctas = batch * num_head_kv * max_groups * cluster_size
    pad_heads = (heads_per_group + 7) // 8 * 8
    device = inputs.q.device
    return _static_mtp1__StaticDecodeWorkspace(
        config=config,
        q_4d=inputs.q.reshape(batch, _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM),
        q_scale_3d=inputs.q_scale.reshape(batch, _runtime_mtp1__NUM_SEQ_Q, num_head_q),
        split_out=torch.empty(
            (batch, max_groups, _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
        lse=torch.empty(
            (batch, max_groups, num_head_kv, _runtime_mtp1__NUM_SEQ_Q, pad_heads), dtype=torch.float32, device=device
        ),
        completion=torch.zeros((num_head_kv * batch,), dtype=torch.int32, device=device),
        last_flags=torch.zeros((physical_ctas,), dtype=torch.int32, device=device),
        out=torch.empty(
            (batch, _runtime_mtp1__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM), dtype=torch.bfloat16, device=device
        ),
        heads_per_group=heads_per_group,
        num_head_kv=num_head_kv,
        max_groups=max_groups,
        aligned_chunk_tokens=aligned_chunk_tokens,
        reduction_only=reduction_only,
    )


def _static_mtp1__fp8_kvpertensor_decode_mtp1_static(
    inputs: _runtime_mtp1__DecodeInputs,
    workspace: _static_mtp1__StaticDecodeWorkspace,
    *,
    no_causal_mask: bool,
    block_ids_prefetch: bool,
    paired_head_finalize: bool | None,
    bf16_dsm: bool | None,
    deferred_norm: bool | None,
    election_mode: str | None,
    precombine_q_scale: bool | None,
    skip_trailing_finalizer_barrier: bool | None,
    c2_raw: bool | None,
    tma_k_scale: bool | None,
    page_metadata_k_scale: bool,
    flatten_output: bool,
) -> torch.Tensor:
    """Run static scheduling through the unchanged optimized dynamic kernel."""
    num_head_q, num_head_kv, heads_per_group = _runtime__validate_inputs(inputs, _runtime_mtp1__NUM_SEQ_Q)
    ws = workspace
    if num_head_kv != ws.num_head_kv or heads_per_group != ws.heads_per_group:
        raise ValueError("inputs do not match the prepared static workspace")
    if inputs.k_scale.numel() != 1 or inputs.v_scale.numel() != 1:
        raise ValueError("quant_type=1 requires scalar K/V scales")
    if tma_k_scale is None:
        tma_k_scale = False
    if tma_k_scale:
        raise ValueError("TLE K-scale TMA requires quant_type=0")
    if page_metadata_k_scale:
        raise ValueError("page-metadata K-scale requires quant_type=0")
    if page_metadata_k_scale and tma_k_scale:
        raise ValueError("page-metadata K-scale and TMA are exclusive")
    config = ws.config
    cluster_size = config.cluster_size
    k_desc, v_desc = _runtime__make_paged_kv_descriptors(inputs, _runtime_mtp1__NUM_SEQ_Q)
    paired_finalize = cluster_size == 4 if paired_head_finalize is None else paired_head_finalize
    if paired_finalize and cluster_size != 4:
        raise ValueError("paired-head finalization requires cluster_size=4")
    deterministic_tail = cluster_size == 8 or (cluster_size == 4 and config.chunk_tokens in (128, 1024))
    if election_mode is None:
        dsm_handoff = not deterministic_tail
    elif election_mode == "tail":
        deterministic_tail = True
        dsm_handoff = False
    elif election_mode == "handoff":
        deterministic_tail = False
        dsm_handoff = True
    else:
        raise ValueError("election_mode must be None, 'tail', or 'handoff'")
    if precombine_q_scale is None:
        precombine_q_scale = (cluster_size, config.chunk_tokens) in ((4, 512), (8, 512))
    eligible_c2_raw = cluster_size == 2 and config.chunk_tokens == 1024 and (ws.max_groups == 2) and ws.reduction_only
    use_c2_raw = eligible_c2_raw if c2_raw is None else c2_raw
    if use_c2_raw and (not eligible_c2_raw):
        raise ValueError("c2_raw requires exact-two C2/T1024 reduction")
    if bf16_dsm is None:
        bf16_dsm = cluster_size == 8 or use_c2_raw
    if deferred_norm is None:
        deferred_norm = use_c2_raw
    if deferred_norm and (not bf16_dsm):
        raise ValueError("deferred_norm requires bf16_dsm")
    if deferred_norm and cluster_size == 2 and (not eligible_c2_raw):
        raise ValueError("C2 deferred_norm requires exact-two C2/T1024")
    if deferred_norm and cluster_size not in (2, 8):
        raise ValueError("deferred_norm is supported only for C2/C8")
    if use_c2_raw and (not bf16_dsm or not deferred_norm):
        raise ValueError("c2_raw requires bf16_dsm and deferred_norm")
    if skip_trailing_finalizer_barrier is None:
        skip_trailing_finalizer_barrier = use_c2_raw or (cluster_size == 4 and config.chunk_tokens in (128, 512, 1024))
    logical_clusters = inputs.num_batch * num_head_kv * ws.max_groups
    _compute_mtp1__fp8_kvpertensor_decode_mtp1_final_kernel[logical_clusters,](
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        inputs.kv_lens,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=_runtime__CLUSTER_MESHES[cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=heads_per_group,
        D=_runtime__HEAD_DIM,
        DV=_runtime__HEAD_DIM,
        BLOCK_SIZE=_runtime__BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=_runtime__TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_M=ws.q_4d.stride(1),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_M=ws.q_scale_3d.stride(1),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        PAGE_METADATA_K_SCALE=page_metadata_k_scale,
        PRECOMBINE_Q_SCALE=precombine_q_scale,
        LDSM_REGISTER_SHARED=not use_c2_raw,
        FULL_VIEW_V_RS=use_c2_raw,
        MERGE_CLUSTER_SIZE=cluster_size,
        EXECUTION_STAGE=_compute_mtp1__EXECUTION_FULL,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.max_groups),
        PAIRED_HEAD_FINALIZE=paired_finalize,
        BF16_DSM=bf16_dsm,
        DEFERRED_NORM=deferred_norm,
        DSM_ELECTION_HANDOFF=dsm_handoff and (not use_c2_raw),
        DETERMINISTIC_TAIL_ELECTION=deterministic_tail and (not use_c2_raw),
        RANK0_ONLY_FINALIZER=use_c2_raw,
        REDUCTION_ONLY=ws.reduction_only,
        SKIP_TRAILING_FINALIZER_BARRIER=skip_trailing_finalizer_barrier,
        STATIC_CHUNK_TOKENS=config.chunk_tokens,
        STATIC_MAX_GROUPS=ws.max_groups,
        STATIC_MTP1_NO_CAUSAL_MASK=no_causal_mask,
        STATIC_BLOCK_IDS_PREFETCH=block_ids_prefetch,
        ALIGNED_FULL_CHUNK_TOKENS=ws.aligned_chunk_tokens,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    if flatten_output:
        return ws.out.reshape(inputs.num_batch, num_head_q, _runtime__HEAD_DIM)
    return ws.out


@dataclass
class _static_mtp2__StaticDecodeWorkspace:
    config: _runtime_mtp2__DecodeConfig
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int
    num_head_kv: int
    max_groups: int
    aligned_chunk_tokens: int
    reduction_only: bool


def _static_mtp2__prepare_static_decode_workspace(
    inputs: _runtime_mtp2__DecodeInputs, config: _runtime_mtp2__DecodeConfig
) -> _static_mtp2__StaticDecodeWorkspace:
    num_head_q, num_head_kv, heads_per_group = _runtime__validate_inputs(inputs, _runtime_mtp2__NUM_SEQ_Q)
    if heads_per_group != 8:
        raise ValueError("the MTP=2 static reference requires GQA8")
    batch = inputs.num_batch
    cluster_size = config.cluster_size
    chunk_tokens = config.chunk_tokens
    max_chunks = (inputs.max_seq_kv + chunk_tokens - 1) // chunk_tokens
    max_groups = (max_chunks + cluster_size - 1) // cluster_size
    aligned_chunk_tokens = chunk_tokens if bool(torch.all(inputs.kv_lens % chunk_tokens == 0).item()) else 0
    num_chunks_gpu = (inputs.kv_lens + chunk_tokens - 1) // chunk_tokens
    reduction_only = bool(torch.all((num_chunks_gpu > 1) & (num_chunks_gpu % cluster_size == 0)).item())
    pad_heads = (heads_per_group + 7) // 8 * 8
    physical_ctas = batch * num_head_kv * max_groups * cluster_size
    device = inputs.q.device
    return _static_mtp2__StaticDecodeWorkspace(
        config=config,
        q_4d=inputs.q.reshape(batch, _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM),
        q_scale_3d=inputs.q_scale.reshape(batch, _runtime_mtp2__NUM_SEQ_Q, num_head_q),
        split_out=torch.empty(
            (batch, max_groups, _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
        lse=torch.empty(
            (batch, max_groups, num_head_kv, _runtime_mtp2__NUM_SEQ_Q, pad_heads), dtype=torch.float32, device=device
        ),
        completion=torch.zeros((num_head_kv * batch,), dtype=torch.int32, device=device),
        last_flags=torch.zeros((physical_ctas,), dtype=torch.int32, device=device),
        out=torch.empty(
            (batch, _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM), dtype=torch.bfloat16, device=device
        ),
        heads_per_group=heads_per_group,
        num_head_kv=num_head_kv,
        max_groups=max_groups,
        aligned_chunk_tokens=aligned_chunk_tokens,
        reduction_only=reduction_only,
    )


def _static_mtp2__fp8_kvpertensor_decode_mtp2_static(
    inputs: _runtime_mtp2__DecodeInputs,
    workspace: _static_mtp2__StaticDecodeWorkspace,
    *,
    paired_head_finalize: bool | None,
    bf16_dsm: bool | None,
    deferred_norm: bool | None,
    election_mode: str | None,
    tail_only_election_barrier: bool,
    precombine_q_scale: bool | None,
    skip_trailing_finalizer_barrier: bool | None,
    full_view_dsm: bool,
    full_view_v_rs: bool | None,
    c2_raw: bool | None,
    tma_k_scale: bool | None,
    page_metadata_k_scale: bool,
    flatten_output: bool,
) -> torch.Tensor:
    num_head_q, num_head_kv, heads_per_group = _runtime__validate_inputs(inputs, _runtime_mtp2__NUM_SEQ_Q)
    ws = workspace
    if num_head_kv != ws.num_head_kv or heads_per_group != ws.heads_per_group:
        raise ValueError("inputs do not match the MTP=2 static workspace")
    if inputs.k_scale.numel() != 1 or inputs.v_scale.numel() != 1:
        raise ValueError("quant_type=1 requires scalar K/V scales")
    config = ws.config
    cluster_size = config.cluster_size
    chunk_tokens = config.chunk_tokens
    if tma_k_scale is None:
        tma_k_scale = False
    if tma_k_scale:
        raise ValueError("TLE K-scale TMA requires quant_type=0")
    if page_metadata_k_scale:
        raise ValueError("page-metadata K-scale requires quant_type=0")
    if page_metadata_k_scale and tma_k_scale:
        raise ValueError("page-metadata K-scale and TMA are exclusive")
    k_desc, v_desc = _runtime__make_paged_kv_descriptors(inputs, _runtime_mtp2__NUM_SEQ_Q)
    eligible_c2_raw = cluster_size == 2 and chunk_tokens == 1024 and (ws.max_groups == 2) and ws.reduction_only
    auto_aligned = (
        ws.aligned_chunk_tokens != 0
        and ws.reduction_only
        and ((cluster_size, chunk_tokens) in ((2, 256), (2, 512), (2, 1024), (4, 256), (4, 512), (4, 1024)))
    )
    reduction_only = ws.reduction_only and (cluster_size == 2 or auto_aligned)
    if precombine_q_scale is None:
        precombine_q_scale = cluster_size == 4 and chunk_tokens == 512
    use_c2_raw = eligible_c2_raw if c2_raw is None else c2_raw
    if use_c2_raw and (not eligible_c2_raw):
        raise ValueError("c2_raw requires exact-two C2/T1024 reduction")
    paired_finalize = cluster_size == 4 and (not use_c2_raw) if paired_head_finalize is None else paired_head_finalize
    if paired_finalize and cluster_size != 4:
        raise ValueError("paired-head finalization requires cluster_size=4")
    if bf16_dsm is None:
        bf16_dsm = chunk_tokens <= 512 or use_c2_raw
    if deferred_norm is None:
        deferred_norm = use_c2_raw
    if deferred_norm and (not bf16_dsm):
        raise ValueError("deferred_norm requires bf16_dsm")
    if deferred_norm and cluster_size not in (2, 4):
        raise ValueError("MTP2 deferred_norm requires C2 or C4")
    if full_view_dsm and cluster_size not in (2, 4):
        raise ValueError("full_view_dsm requires C2 or C4")
    if full_view_v_rs is None:
        full_view_v_rs = use_c2_raw
    if skip_trailing_finalizer_barrier is None:
        skip_trailing_finalizer_barrier = True
    if election_mode is None:
        deterministic_tail = not use_c2_raw
        dsm_handoff = False
    elif election_mode == "tail":
        deterministic_tail = True
        dsm_handoff = False
    elif election_mode == "handoff":
        if cluster_size not in (2, 4):
            raise ValueError("DSM handoff requires C2 or C4")
        deterministic_tail = False
        dsm_handoff = True
    else:
        raise ValueError("election_mode must be None, 'tail', or 'handoff'")
    if use_c2_raw and (
        paired_finalize
        or not bf16_dsm
        or (not deferred_norm)
        or deterministic_tail
        or dsm_handoff
        or (not full_view_v_rs)
    ):
        raise ValueError("c2_raw prerequisites were overridden")
    if tail_only_election_barrier and (not deterministic_tail):
        raise ValueError("tail-only barrier requires deterministic election")
    logical_clusters = inputs.num_batch * num_head_kv * ws.max_groups
    _compute_mtp2__fp8_kvpertensor_decode_mtp2_final_kernel[logical_clusters,](
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        inputs.kv_lens,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=_runtime__CLUSTER_MESHES[cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=heads_per_group,
        D=_runtime__HEAD_DIM,
        DV=_runtime__HEAD_DIM,
        BLOCK_SIZE=_runtime__BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=_runtime__TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_M=ws.q_4d.stride(1),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_M=ws.q_scale_3d.stride(1),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        PAGE_METADATA_K_SCALE=page_metadata_k_scale,
        PRECOMBINE_Q_SCALE=precombine_q_scale,
        LDSM_REGISTER_SHARED=not full_view_v_rs,
        FULL_VIEW_V_RS=full_view_v_rs,
        MERGE_CLUSTER_SIZE=cluster_size,
        EXECUTION_STAGE=_compute_mtp2__EXECUTION_FULL,
        MAX_FINAL_CHUNKS=triton.next_power_of_2(ws.max_groups),
        REUSE_FINAL_WEIGHTS=not use_c2_raw,
        PAIRED_HEAD_FINALIZE=paired_finalize,
        BF16_DSM=bf16_dsm,
        DEFERRED_NORM=deferred_norm,
        DSM_ELECTION_HANDOFF=dsm_handoff,
        DETERMINISTIC_TAIL_ELECTION=deterministic_tail,
        TAIL_ONLY_ELECTION_BARRIER=tail_only_election_barrier,
        REDUCTION_ONLY=reduction_only,
        ALIGNED_FULL_CHUNK_TOKENS=chunk_tokens if auto_aligned else 0,
        FULL_VIEW_DSM=full_view_dsm,
        RANK0_ONLY_FINALIZER=use_c2_raw,
        SKIP_TRAILING_FINALIZER_BARRIER=skip_trailing_finalizer_barrier,
        STATIC_CHUNK_TOKENS=chunk_tokens,
        STATIC_MAX_GROUPS=ws.max_groups,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    if flatten_output:
        return ws.out.reshape(inputs.num_batch * _runtime_mtp2__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM)
    return ws.out


def _static_mtp4__prepare_static_decode_workspace(
    inputs: _runtime_mtp4__DecodeInputs, config: _runtime_mtp4__DecodeConfig, *, policy_mode: str
) -> _runtime_mtp4__DecodeWorkspace:
    """Allocate strict-static buffers without constructing a task map."""
    num_head_q, num_head_kv, heads_per_group = _runtime__validate_inputs(inputs, _runtime_mtp4__NUM_SEQ_Q)
    if heads_per_group != 8:
        raise ValueError("the MTP=4 static specialization requires GQA8")
    if config.direct_threshold or config.subgroup2_threshold:
        raise ValueError("strict-static MTP=4 requires plain C/T configs")
    batch = inputs.num_batch
    max_chunks = (inputs.max_seq_kv + config.chunk_tokens - 1) // config.chunk_tokens
    max_groups = max(1, (max_chunks + config.cluster_size - 1) // config.cluster_size)
    logical_clusters = batch * num_head_kv * max_groups
    physical_ctas = logical_clusters * config.cluster_size
    max_final_chunks = triton.next_power_of_2(max_groups)
    partial_storage_slots = 2 * max_final_chunks
    device = inputs.q.device
    placeholder = torch.empty((1,), dtype=torch.int32, device=device)
    stats = {
        "cluster_size": config.cluster_size,
        "num_clusters": logical_clusters,
        "physical_ctas": physical_ctas,
        "reduction_clusters": logical_clusters,
        "compute_tasks": logical_clusters * config.cluster_size,
        "fine_chunks_max": max_chunks,
        "effective_chunks_max": max_groups,
        "direct_tasks": 0,
        "dummy_tasks": 0,
        "subgroup2_tasks": 0,
    }
    schedule = _scheduler_mtp24__DecodeTaskSchedule(
        task_workspace=placeholder.view(torch.int8),
        task_map=placeholder,
        offsets=placeholder,
        meta=placeholder,
        cluster_size=config.cluster_size,
        chunk_tokens=config.chunk_tokens,
        block_seq=1,
        block_chunks=triton.next_power_of_2(max_chunks),
        capacity_clusters=logical_clusters,
        capacity_ints=1,
        num_seq_q=_runtime_mtp4__NUM_SEQ_Q,
        num_clusters=logical_clusters,
        physical_ctas=physical_ctas,
        sched_ints=0,
        partial_slots=max_groups,
        stats=stats,
    )
    q_4d = inputs.q.reshape(batch, _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM)
    q_scale_3d = inputs.q_scale.reshape(batch, _runtime_mtp4__NUM_SEQ_Q, num_head_q)
    pad_heads = (heads_per_group + 7) // 8 * 8
    provisional = _runtime_mtp4__DecodeWorkspace(
        config=config,
        schedule=schedule,
        q_4d=q_4d,
        q_scale_3d=q_scale_3d,
        split_out=torch.empty((1,), dtype=torch.float32, device=device),
        lse=torch.empty((1,), dtype=torch.float32, device=device),
        completion=torch.empty((1,), dtype=torch.int32, device=device),
        last_flags=torch.empty((1,), dtype=torch.int32, device=device),
        out=torch.empty((1,), dtype=torch.bfloat16, device=device),
        heads_per_group=heads_per_group,
        all_chunks_aligned=bool(torch.all(inputs.kv_lens % config.chunk_tokens == 0).item()),
        static_sched=True,
        static_chunk_tokens=config.chunk_tokens,
        static_max_groups=max_groups,
    )
    resolved = _runtime_mtp4___resolve_mtp4_final_policy(provisional) if policy_mode == "auto" else policy_mode
    if resolved not in ("winner", "c2-raw", "pdl", "pdl-s"):
        raise ValueError(f"unsupported MTP=4 static policy: {resolved}")
    split_out = torch.empty(
        (batch, partial_storage_slots, _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    if resolved == "pdl-s":
        lse = torch.empty(
            (batch, num_head_kv, _runtime_mtp4__NUM_SEQ_Q, pad_heads, partial_storage_slots),
            dtype=torch.float32,
            device=device,
        ).permute(0, 4, 1, 2, 3)
    else:
        lse = torch.empty(
            (batch, partial_storage_slots, num_head_kv, _runtime_mtp4__NUM_SEQ_Q, pad_heads),
            dtype=torch.float32,
            device=device,
        )
    provisional.split_out = split_out
    provisional.lse = lse
    provisional.completion = torch.zeros(
        (batch * num_head_kv * (1 + max_final_chunks),), dtype=torch.int32, device=device
    )
    provisional.last_flags = torch.zeros((physical_ctas,), dtype=torch.int32, device=device)
    provisional.out = torch.empty(
        (batch, _runtime_mtp4__NUM_SEQ_Q, num_head_q, _runtime__HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    provisional.final_policy_mode = resolved
    return provisional


def _static_mtp4__fp8_kvpertensor_decode_mtp4_static(
    inputs: _runtime_mtp4__DecodeInputs, workspace: _runtime_mtp4__DecodeWorkspace
) -> torch.Tensor:
    """Run the dynamic winner stack with analytic strict-static scheduling."""
    if not workspace.static_sched:
        raise ValueError("workspace was not prepared for strict-static MTP=4")
    if inputs.k_scale.numel() != 1 or inputs.v_scale.numel() != 1:
        raise ValueError("quant_type=1 requires scalar K/V scales")
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_final(inputs, workspace)


def _static_mtp4___run_static_winner_route(
    inputs: _runtime_mtp4__DecodeInputs, workspace: _runtime_mtp4__DecodeWorkspace, *, full_view_v: bool
) -> torch.Tensor:
    config = workspace.config
    groups = workspace.static_max_groups
    use_c4 = config.cluster_size == 4
    use_c2_exact_two = config.cluster_size == 2 and groups == 2
    use_c2_rank0_quad = config.cluster_size == 2 and groups == 4
    use_c2_sharded_quad = config.cluster_size == 2 and groups >= 32
    use_general_c2_quad = use_c2_rank0_quad or use_c2_sharded_quad
    aligned_c2 = config.cluster_size == 2 and config.chunk_tokens in (256, 1024) and workspace.all_chunks_aligned
    precombine = not (config.cluster_size == 4 and config.chunk_tokens == 128)
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_pure_tle(
        inputs,
        workspace,
        paired_head_finalize=use_c4 or use_general_c2_quad,
        direct_fast_path=True,
        quad_head_two_chunk_finalize=use_c2_exact_two,
        fast_finalizer_handoff=use_c4 or use_c2_sharded_quad,
        rank0_only_finalizer=use_c2_exact_two or use_c2_rank0_quad,
        skip_trailing_finalizer_barrier=use_c4 or use_c2_exact_two or use_general_c2_quad,
        maxnreg=240 if use_c4 else None,
        precombine_q_scale=precombine,
        c2_aligned_full_chunk_winner=aligned_c2,
        full_view_v_rs=full_view_v,
        execution_stage="full",
        c4_bf16_dsm=False,
        c4_deferred_norm=False,
        c4_global_deferred_norm=False,
        c4_reduction_only_raw=False,
        c4_aligned_full_chunk_raw=False,
        pdl_notify=False,
        prefetch_k_scale=False,
        page_metadata_k_scale=False,
        tma_k_scale=False,
        flatten_output=True,
    )


def _static_mtp4___run_static_pdl_route(
    inputs: _runtime_mtp4__DecodeInputs, workspace: _runtime_mtp4__DecodeWorkspace, *, use_pdl: bool, maxnreg: int
) -> torch.Tensor:
    aligned = (
        workspace.config.cluster_size == 4
        and workspace.config.chunk_tokens in (512, 1024)
        and workspace.all_chunks_aligned
    )
    reduction_only = aligned and workspace.config.chunk_tokens == 512
    return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_detached_raw_finalize(
        inputs,
        workspace,
        use_pdl=use_pdl,
        maxnreg=maxnreg,
        precombine_q_scale=True,
        reduction_only_raw=reduction_only,
        aligned_full_chunk_raw=aligned,
        page_metadata_k_scale=False,
    )


def _static_mtp4__fp8_kvpertensor_decode_mtp4_static_tuned(
    inputs: _runtime_mtp4__DecodeInputs, workspace: _runtime_mtp4__DecodeWorkspace, *, route: str
) -> torch.Tensor:
    """Run a validated layout-aware winner from the fixed static policy."""
    if route == "default":
        return _static_mtp4__fp8_kvpertensor_decode_mtp4_static(inputs, workspace)
    if route == "full_v":
        if workspace.final_policy_mode != "winner":
            raise ValueError("full_v winner requires the fused winner backend")
        return _static_mtp4___run_static_winner_route(inputs, workspace, full_view_v=True)
    if route == "noalign":
        if workspace.final_policy_mode == "winner":
            return _static_mtp4___run_static_winner_route(inputs, workspace, full_view_v=False)
        if workspace.final_policy_mode != "c2-raw":
            raise ValueError("noalign winner requires the C2 raw backend")
        return _runtime_mtp4__fp8_kvpertensor_decode_mtp4_c2_bf16_dsm_specialized(
            inputs, workspace, precombine_q_scale=True, aligned_full_chunk_winner=False, page_metadata_k_scale=False
        )
    if route == "nopdl":
        if workspace.final_policy_mode not in ("pdl", "pdl-s"):
            raise ValueError("nopdl winner requires a detached raw backend")
        return _static_mtp4___run_static_pdl_route(inputs, workspace, use_pdl=False, maxnreg=240)
    if route == "r192":
        if workspace.final_policy_mode not in ("pdl", "pdl-s"):
            raise ValueError("r192 winner requires a detached raw backend")
        return _static_mtp4___run_static_pdl_route(inputs, workspace, maxnreg=192, use_pdl=True)
    if route == "scalar_major":
        if workspace.final_policy_mode != "pdl":
            raise ValueError("scalar_major requires prepare_static_decode_workspace(..., policy_mode='pdl')")
        return _static_mtp4___run_static_pdl_route(inputs, workspace, use_pdl=True, maxnreg=240)
    raise ValueError(f"unsupported MTP=4 static tuning: {route}")


_fp8_entry__HEAD_DIM = 128
_fp8_entry__BLOCK_SIZE = 64
_fp8_entry__SUPPORTED_MTP = (1, 2, 4)
_fp8_entry__Schedule = Literal["static", "dynamic"]
_fp8_entry__OFFICIAL_CASES = {
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
_fp8_entry___STATIC_Q1_CT = {
    1: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (8, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 256),
        "one_64k_15x4k": (8, 512),
        "one_64k_31x4k": (8, 512),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (8, 512),
    },
    2: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (2, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 256),
        "one_64k_15x4k": (8, 512),
        "one_64k_31x4k": (4, 512),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (4, 512),
    },
    4: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (2, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 512),
        "one_64k_15x4k": (4, 512),
        "one_64k_31x4k": (4, 1024),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (4, 1024),
    },
}


@dataclass
class _fp8_entry__FP8DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


@dataclass(frozen=True)
class _fp8_entry__FP8DecodePolicy:
    schedule: _fp8_entry__Schedule
    mtp: int
    workload: DecodeWorkload
    layout: str
    cluster_size: int
    chunk_tokens: int
    route: str

    @property
    def label(self) -> str:
        return f"c{self.cluster_size}t{self.chunk_tokens}/{self.route}"


@dataclass
class _fp8_entry__FP8DecodeWorkspace:
    policy: _fp8_entry__FP8DecodePolicy
    runtime_inputs: object
    runtime_workspace: object


def _fp8_entry___layout(cache: torch.Tensor) -> str:
    return "HND" if cache.stride(2) > cache.stride(1) else "NHD"


def _fp8_entry___validate(inputs: _fp8_entry__FP8DecodeInputs) -> None:
    if inputs.mtp not in _fp8_entry__SUPPORTED_MTP:
        raise ValueError("final FP8 decode supports MTP 1, 2, or 4")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != _fp8_entry__HEAD_DIM:
        raise ValueError("q must have shape [batch * MTP,Hq,128]")
    if inputs.q.dtype != torch.float8_e4m3fn:
        raise ValueError("q must be float8_e4m3fn")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.element_size() != 1:
            raise ValueError(f"{name} must use a 1-byte FP8 storage dtype")
        if cache.ndim != 4 or cache.shape[1] != _fp8_entry__BLOCK_SIZE:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.shape[-1] != _fp8_entry__HEAD_DIM or cache.stride(-1) != 1:
            raise ValueError(f"{name} head dimension must be contiguous 128")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if (num_head_kv, num_head_q) not in ((1, 8), (4, 32)):
        raise ValueError("final FP8 decode requires official GQA8 heads (Hkv,Hq)=(1,8) or (4,32)")
    if inputs.block_ids.dtype != torch.int32 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError("block_ids and kv_lens must be int32")


def _fp8_entry___classify_workload(lengths: tuple[int, ...]) -> str | None:
    count = len(lengths)
    if count == 64 and all((length == 512 for length in lengths)):
        return "uniform_512"
    if count == 64 and all((length == 4096 for length in lengths)):
        return "uniform_4096"
    if count == 64 and lengths.count(128) == 32 and (lengths.count(4096) == 32):
        return "skewed_mix"
    if count == 16 and lengths.count(64) == 15 and (lengths.count(16384) == 1):
        return "skewed_extreme"
    if lengths.count(65536) == 1 and lengths.count(4096) == count - 1:
        if count == 8:
            return "one_64k_7x4k"
        if count == 16:
            return "one_64k_15x4k"
        if count == 32:
            return "one_64k_31x4k"
    if count == 32 and lengths.count(131072) == 1 and (lengths.count(4096) == 31):
        return "one_128k_31x4k"
    if count == 32 and lengths.count(32768) == 2 and (lengths.count(4096) == 30):
        return "two_32k_30x4k"
    return None


def _fp8_entry___classify_workload_from_features(workload: DecodeWorkload) -> str | None:
    return _fp8_entry___classify_workload(tuple(value for value, count in workload.histogram for _ in range(count)))


def _fp8_entry___case(inputs: _fp8_entry__FP8DecodeInputs) -> str:
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    workload = _fp8_entry___classify_workload(lengths)
    if workload is not None:
        return workload
    if not lengths or min(lengths) < inputs.mtp:
        raise ValueError("each final KV length must be at least MTP")
    return "uniform_512" if max(lengths) <= 1024 else "uniform_4096"


def _fp8_entry___static_ct(mtp: int, case: str) -> tuple[int, int]:
    return _fp8_entry___STATIC_Q1_CT[mtp][case]


def _fp8_entry___route(mtp: int, case: str, layout: str) -> str:
    if mtp == 1:
        return {
            "uniform_512": "prefetch-qscale",
            "uniform_4096": "barrier-sink",
            "skewed_mix": "default",
            "skewed_extreme": "prefetch",
            "one_64k_7x4k": "barrier-sink",
            "one_64k_15x4k": "default",
            "one_64k_31x4k": "prefetch",
            "one_128k_31x4k": "prefetch",
            "two_32k_30x4k": "prefetch",
        }[case]
    if mtp == 2:
        if case == "uniform_512" and layout == "NHD":
            return "deferred-normalization"
        if case == "uniform_4096":
            return "default"
        if case == "one_64k_7x4k" and layout == "NHD":
            return "single-head-finalize"
        return "full-v-reduction" if not (case == "uniform_512" and layout == "HND") else "default"
    backend = _fp8_entry___MTP4_Q1_ROUTES[case, layout]
    return {
        "default": "fused-default",
        "full_v": "full-v-reduction",
        "noalign": "tail-masked",
        "nopdl": "synchronous-finalize",
        "r192": "reduced-register-finalize",
        "scalar_major": "scalar-major-finalize",
        "scale_flip": "separated-q-scale",
        "scale_on": "precombined-q-scale",
        "tma": "tma-k-scale",
        "dynamic": "dynamic-core",
        "page": "page-k-scale",
    }[backend]


def _fp8_entry__select_fp8_decode_policy(inputs: _fp8_entry__FP8DecodeInputs) -> _fp8_entry__FP8DecodePolicy:
    _fp8_entry___validate(inputs)
    case = _fp8_entry___case(inputs)
    layout = _fp8_entry___layout(inputs.k_cache)
    cluster, tokens = _fp8_entry___static_ct(inputs.mtp, case)
    route = _fp8_entry___route(inputs.mtp, case, layout)
    workload = DecodeWorkload.from_lengths(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return _fp8_entry__FP8DecodePolicy("static", inputs.mtp, workload, layout, cluster, tokens, route)


def _fp8_entry___runtime_inputs(inputs: _fp8_entry__FP8DecodeInputs, mtp: int):
    input_type = {1: _runtime_mtp1__DecodeInputs, 2: _runtime_mtp2__DecodeInputs, 4: _runtime_mtp4__DecodeInputs}[mtp]
    return input_type(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        inputs.q_scale,
        inputs.k_scale,
        inputs.v_scale,
    )


def _fp8_entry__prepare_fp8_decode_workspace(inputs: _fp8_entry__FP8DecodeInputs) -> _fp8_entry__FP8DecodeWorkspace:
    policy = _fp8_entry__select_fp8_decode_policy(inputs)
    runtime_inputs = _fp8_entry___runtime_inputs(inputs, policy.mtp)
    config_type = {1: _runtime_mtp1__DecodeConfig, 2: _runtime_mtp2__DecodeConfig, 4: _runtime_mtp4__DecodeConfig}[
        policy.mtp
    ]
    config = config_type(policy.cluster_size, policy.chunk_tokens)
    if policy.mtp == 1:
        runtime_workspace = _static_mtp1__prepare_static_decode_workspace(runtime_inputs, config)
    elif policy.mtp == 2:
        runtime_workspace = _static_mtp2__prepare_static_decode_workspace(runtime_inputs, config)
    else:
        mode = "pdl" if _fp8_entry___mtp4_static_route(policy) == "scalar_major" else "auto"
        runtime_workspace = _static_mtp4__prepare_static_decode_workspace(runtime_inputs, config, policy_mode=mode)
    return _fp8_entry__FP8DecodeWorkspace(policy, runtime_inputs, runtime_workspace)


def _fp8_entry___mtp1_options(policy: _fp8_entry__FP8DecodePolicy) -> dict[str, object]:
    case = _fp8_entry___classify_workload_from_features(policy.workload) or (
        "uniform_512" if policy.workload.max_length <= 1024 else "uniform_4096"
    )
    return {
        "uniform_512": dict(no_causal_mask=True, block_ids_prefetch=True, precombine_q_scale=True),
        "uniform_4096": dict(no_causal_mask=True, block_ids_prefetch=True, skip_trailing_finalizer_barrier=True),
        "skewed_mix": dict(no_causal_mask=False, block_ids_prefetch=False),
        "skewed_extreme": dict(no_causal_mask=True, block_ids_prefetch=True),
        "one_64k_7x4k": dict(no_causal_mask=True, block_ids_prefetch=False, skip_trailing_finalizer_barrier=True),
        "one_64k_15x4k": dict(no_causal_mask=False, block_ids_prefetch=False),
        "one_64k_31x4k": dict(no_causal_mask=True, block_ids_prefetch=True, precombine_q_scale=False),
        "one_128k_31x4k": dict(no_causal_mask=True, block_ids_prefetch=True),
        "two_32k_30x4k": dict(no_causal_mask=True, block_ids_prefetch=True, precombine_q_scale=False),
    }[case]


def _fp8_entry___mtp2_options(policy: _fp8_entry__FP8DecodePolicy) -> dict[str, object]:
    case = _fp8_entry___classify_workload_from_features(policy.workload) or (
        "uniform_512" if policy.workload.max_length <= 1024 else "uniform_4096"
    )
    layout = policy.layout
    if case == "uniform_512" and layout == "NHD":
        return dict(bf16_dsm=True, deferred_norm=True)
    if case == "uniform_512" and layout == "HND":
        return dict(skip_trailing_finalizer_barrier=False)
    if case == "uniform_4096":
        return {}
    if case == "one_64k_7x4k" and layout == "NHD":
        return dict(paired_head_finalize=False)
    return dict(full_view_v_rs=True)


_fp8_entry___MTP4_Q1_ROUTES = {
    ("uniform_512", "NHD"): "full_v",
    ("uniform_512", "HND"): "full_v",
    ("uniform_4096", "NHD"): "noalign",
    ("uniform_4096", "HND"): "noalign",
    ("skewed_mix", "NHD"): "default",
    ("skewed_mix", "HND"): "full_v",
    ("skewed_extreme", "NHD"): "full_v",
    ("skewed_extreme", "HND"): "full_v",
    ("one_64k_7x4k", "NHD"): "default",
    ("one_64k_7x4k", "HND"): "default",
    ("one_64k_15x4k", "NHD"): "nopdl",
    ("one_64k_15x4k", "HND"): "nopdl",
    ("one_64k_31x4k", "NHD"): "default",
    ("one_64k_31x4k", "HND"): "default",
    ("one_128k_31x4k", "NHD"): "scalar_major",
    ("one_128k_31x4k", "HND"): "scalar_major",
    ("two_32k_30x4k", "NHD"): "r192",
    ("two_32k_30x4k", "HND"): "default",
}


def _fp8_entry___mtp4_static_route(policy: _fp8_entry__FP8DecodePolicy) -> str:
    table = _fp8_entry___MTP4_Q1_ROUTES
    case = _fp8_entry___classify_workload_from_features(policy.workload) or (
        "uniform_512" if policy.workload.max_length <= 1024 else "uniform_4096"
    )
    return table[case, policy.layout]


def _fp8_entry__attention_decode_fp8_tle(
    inputs: _fp8_entry__FP8DecodeInputs, workspace: _fp8_entry__FP8DecodeWorkspace
) -> torch.Tensor:
    policy = workspace.policy
    _fp8_entry___validate(inputs)
    if policy.mtp != inputs.mtp or policy.layout != _fp8_entry___layout(inputs.k_cache):
        raise ValueError("workspace policy does not match inputs")
    runtime_inputs = workspace.runtime_inputs
    runtime_workspace = workspace.runtime_workspace
    if policy.mtp == 1:
        return _static_mtp1__fp8_kvpertensor_decode_mtp1_static(
            runtime_inputs,
            runtime_workspace,
            **{
                "no_causal_mask": True,
                "block_ids_prefetch": False,
                "paired_head_finalize": None,
                "bf16_dsm": None,
                "deferred_norm": None,
                "election_mode": None,
                "precombine_q_scale": None,
                "skip_trailing_finalizer_barrier": None,
                "c2_raw": None,
                "tma_k_scale": None,
                "page_metadata_k_scale": False,
                "flatten_output": True,
                **_fp8_entry___mtp1_options(policy),
            },
        )
    if policy.mtp == 2:
        return _static_mtp2__fp8_kvpertensor_decode_mtp2_static(
            runtime_inputs,
            runtime_workspace,
            **{
                "paired_head_finalize": None,
                "bf16_dsm": None,
                "deferred_norm": None,
                "election_mode": None,
                "tail_only_election_barrier": False,
                "precombine_q_scale": None,
                "skip_trailing_finalizer_barrier": None,
                "full_view_dsm": False,
                "full_view_v_rs": None,
                "c2_raw": None,
                "tma_k_scale": None,
                "page_metadata_k_scale": False,
                "flatten_output": True,
                **_fp8_entry___mtp2_options(policy),
            },
        )
    route = _fp8_entry___mtp4_static_route(policy)
    return _static_mtp4__fp8_kvpertensor_decode_mtp4_static_tuned(runtime_inputs, runtime_workspace, route=route)


def _fp8_entry__fp8_workspace_is_reset(workspace: _fp8_entry__FP8DecodeWorkspace) -> bool:
    completion = getattr(workspace.runtime_workspace, "completion", None)
    return completion is None or not bool(torch.count_nonzero(completion).item())


QUANT_TYPE = "qpertoken_perhead_kvpertensor"
QUANT_TYPE_ID = 1
BLOCK_SIZE = _fp8_entry__BLOCK_SIZE
HEAD_DIM = _fp8_entry__HEAD_DIM
SUPPORTED_MTP = _fp8_entry__SUPPORTED_MTP
QUANT_TYPES = ("qpertoken_perhead_kvpertensor",)
OFFICIAL_CASES = _fp8_entry__OFFICIAL_CASES
FP8DecodeInputs = _fp8_entry__FP8DecodeInputs
FP8DecodePolicy = _fp8_entry__FP8DecodePolicy
FP8DecodeWorkspace = _fp8_entry__FP8DecodeWorkspace

select_decode_policy = _fp8_entry__select_fp8_decode_policy


def prepare_decode_workspace(inputs):
    if not USE_TLE:
        if inputs.mtp == 1:
            return prepare_pure_triton_mtp1_workspace(inputs, QUANT_TYPE)
        return prepare_pure_triton_mtp_workspace(inputs, QUANT_TYPE)
    return _fp8_entry__prepare_fp8_decode_workspace(inputs)


def attention_decode_fp8(inputs, workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    return _fp8_entry__attention_decode_fp8_tle(inputs, workspace)


def workspace_is_reset(workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return True
    return _fp8_entry__fp8_workspace_is_reset(workspace)
