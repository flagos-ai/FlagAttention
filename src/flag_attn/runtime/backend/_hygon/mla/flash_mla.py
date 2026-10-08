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

import logging
import math
import os
from collections import OrderedDict
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from flag_attn.runtime.backend._hygon import device, error, torch_device_fn  # noqa: E402
from flag_attn.runtime.backend._hygon.device_info import get_device_capability  # noqa: E402
from flag_attn.gated_delta_rule.compat import has_triton_tle  # noqa: E402

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_FLASH_MLA = True
    except ImportError:
        tle = None
        HAS_TLE_FLASH_MLA = False
else:
    tle = None
    HAS_TLE_FLASH_MLA = False

vendor_name = device.vendor_name
device = device.name
logger = logging.getLogger(__name__)

FLASH_MLA_META_FIELDS = 8
FLASH_MLA_BLOCK_M = 64
FLASH_MLA_BLOCK_N = 64
FLASH_MLA_FIXED_OVERHEAD_BLOCKS = 5
FLASH_MLA_COMBINE_BLOCK_H = 8
FLASH_MLA_COMBINE_BLOCK_D = 256

_TENSOR_DESCRIPTOR_CLS = None
_CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE: int | None = None
_NUM_SMS_CACHE: dict[int, int] = {}
_FLASH_MLA_TLE_PLAN_CACHE: OrderedDict[
    "FlashMLATLEDecodePlanKey", "FlashMLATLEDecodePlan"
] = OrderedDict()


@dataclass(frozen=True)
class FlashMLATLEDecodePlanKey:
    device: int
    dtype: torch.dtype
    b: int
    s_q: int
    h_q: int
    h_kv: int
    d: int
    dv: int
    block_size: int
    causal: bool
    reuse_output: bool


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "off", "no"}


def _contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _get_cuda_device_index(cuda_device: torch.device) -> int:
    dev_idx = cuda_device.index
    if dev_idx is None:
        dev_idx = torch.cuda.current_device()
    return int(dev_idx)


def _get_tensor_descriptor_cls():
    global _TENSOR_DESCRIPTOR_CLS
    if _TENSOR_DESCRIPTOR_CLS is None:
        from triton.tools.tensor_descriptor import TensorDescriptor

        _TENSOR_DESCRIPTOR_CLS = TensorDescriptor
    return _TENSOR_DESCRIPTOR_CLS


def _ensure_triton_descriptor_allocator(cuda_device: torch.device) -> None:
    global _CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE
    dev_idx = _get_cuda_device_index(cuda_device)
    if _CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE == dev_idx:
        return
    _set_triton_descriptor_allocator(cuda_device)
    _CURRENT_DESCRIPTOR_ALLOCATOR_DEVICE = dev_idx


def _get_num_sms(cuda_device: torch.device) -> int:
    dev_idx = _get_cuda_device_index(cuda_device)
    cached = _NUM_SMS_CACHE.get(dev_idx)
    if cached is not None:
        return cached
    num_sms = torch.cuda.get_device_properties(cuda_device).multi_processor_count
    _NUM_SMS_CACHE[dev_idx] = int(num_sms)
    return int(num_sms)


def _get_plan_cache_size() -> int:
    try:
        return max(
            int(os.environ.get("FLAGGEMS_VLLM_FLASH_MLA_TLE_PLAN_CACHE_SIZE", "32")),
            0,
        )
    except ValueError:
        logger.warning("Invalid FLAGGEMS_VLLM_FLASH_MLA_TLE_PLAN_CACHE_SIZE")
        return 32


def _reuse_tle_output() -> bool:
    return _env_flag("FLAGGEMS_VLLM_FLASH_MLA_TLE_REUSE_OUTPUT", False)


def _force_triton_flash_mla() -> bool:
    return _env_flag("FLAGGEMS_VLLM_FLASH_MLA_FORCE_TRITON", False)


def _same_cuda_device(lhs: torch.device, rhs: torch.device) -> bool:
    return lhs.type == rhs.type == "cuda" and _get_cuda_device_index(
        lhs
    ) == _get_cuda_device_index(rhs)


def _tensor_desc_light_key(t: torch.Tensor, block_shape: tuple[int, int]) -> tuple:
    if t.ndim != 2:
        raise ValueError("TensorDescriptor cache key expects a 2D tensor/view")
    return (
        int(t.data_ptr()),
        int(t.shape[0]),
        int(t.shape[1]),
        int(t.stride(0)),
        int(t.stride(1)),
        t.dtype,
        _get_cuda_device_index(t.device),
        tuple(block_shape),
    )


def _set_triton_descriptor_allocator(cuda_device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=cuda_device)

    triton.set_allocator(alloc_fn)


@triton.jit
def flash_mla_sched_meta_kernel_v3(
    B_seq_len,
    Sched_meta,
    Num_splits,
    CombineReqIds,
    NumCombineReqs,
    BLOCK_B: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    FIXED_OVERHEAD_NUM_BLOCKS: tl.constexpr,
    NUM_SM_PARTS: tl.constexpr,
    META_FIELDS: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask_b = offs_b < BATCH_SIZE
    seqlens = tl.load(B_seq_len + offs_b, mask=mask_b, other=0)
    num_blocks_vec = tl.cdiv(tl.maximum(seqlens, 1), BLOCK_SIZE_N)
    total_num_blocks = tl.sum(
        tl.where(mask_b, num_blocks_vec + FIXED_OVERHEAD_NUM_BLOCKS, 0), axis=0
    )
    payload = tl.maximum(
        tl.cdiv(total_num_blocks, NUM_SM_PARTS) + FIXED_OVERHEAD_NUM_BLOCKS,
        FIXED_OVERHEAD_NUM_BLOCKS + 2,
    )

    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    combine_req_count = 0
    tl.store(Num_splits, 0)

    for part in tl.range(0, NUM_SM_PARTS, 1):
        begin_req_idx = now_req_idx
        begin_block_idx = now_block
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = now_block != 0
        remain_payload = payload

        while (now_req_idx < BATCH_SIZE) & (remain_payload > 0):
            cur_seq_len = tl.load(B_seq_len + now_req_idx)
            cur_num_blocks = tl.cdiv(tl.maximum(cur_seq_len, 1), BLOCK_SIZE_N)
            now_remain_blocks = cur_num_blocks - now_block
            if remain_payload + 1 >= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS:
                req_num_splits = now_n_split_idx + 1
                if req_num_splits != 1:
                    tl.store(CombineReqIds + combine_req_count, now_req_idx)
                    combine_req_count += 1
                cum_num_splits += req_num_splits
                tl.store(Num_splits + now_req_idx + 1, cum_num_splits)
                remain_payload -= now_remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - FIXED_OVERHEAD_NUM_BLOCKS > 0:
                    split_blocks = remain_payload - FIXED_OVERHEAD_NUM_BLOCKS
                    # The WS TLE kernel cannot safely handle a one-block
                    # partial split in this schedule.
                    split_blocks = tl.where(
                        (split_blocks > 1) & (now_remain_blocks - split_blocks == 1),
                        split_blocks - 1,
                        split_blocks,
                    )
                    now_block += split_blocks
                    now_n_split_idx += 1
                remain_payload = 0

        if now_block > 0:
            end_req_idx = now_req_idx
            end_block_idx = now_block
        else:
            end_req_idx = now_req_idx - 1
            if end_req_idx >= 0:
                end_seq_len = tl.load(B_seq_len + end_req_idx)
                end_block_idx = tl.where(
                    end_seq_len == 0, 0, tl.cdiv(end_seq_len, BLOCK_SIZE_N)
                )
            else:
                end_block_idx = 0

        meta = Sched_meta + part * META_FIELDS
        if begin_req_idx >= BATCH_SIZE:
            tl.store(meta + 0, BATCH_SIZE)
            tl.store(meta + 1, BATCH_SIZE - 1)
            tl.store(meta + 2, 0)
            tl.store(meta + 3, 0)
            tl.store(meta + 4, 0)
            tl.store(meta + 5, 0)
            tl.store(meta + 6, 0)
            tl.store(meta + 7, 0)
        else:
            end_seq_len = tl.load(B_seq_len + end_req_idx)
            last_block_exclusive = tl.where(
                end_seq_len == 0, 0, tl.cdiv(end_seq_len, BLOCK_SIZE_N)
            )
            is_last_req_splitted = (end_block_idx != last_block_exclusive) & (
                end_seq_len != 0
            )
            if begin_req_idx == end_req_idx:
                same_req_split = is_first_req_splitted | is_last_req_splitted
                is_first_req_splitted = same_req_split
                is_last_req_splitted = same_req_split

            tl.store(meta + 0, begin_req_idx)
            tl.store(meta + 1, end_req_idx)
            tl.store(meta + 2, begin_block_idx)
            tl.store(meta + 3, end_block_idx)
            tl.store(meta + 4, begin_split_idx)
            tl.store(meta + 5, is_first_req_splitted.to(tl.int32))
            tl.store(meta + 6, is_last_req_splitted.to(tl.int32))
            tl.store(meta + 7, 0)

    tl.store(NumCombineReqs, combine_req_count)


@triton.jit
def _flash_mla_ws_kv_producer(
    q_writer,
    k0_l_writer,
    k0_r_writer,
    k1_l_writer,
    k1_r_writer,
    Q_desc,
    Q_tail_desc,
    Kv,
    Kv_desc,
    Kv_tail_desc,
    Block_table,
    B_seq_len,
    begin_req_idx,
    end_req_idx,
    begin_block_idx_meta,
    end_block_idx_meta,
    head_base,
    head_num,
    stride_kv_token,
    stride_block_table_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    pipe_base = 0
    k1_pipe_base = 0
    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        q_stage = batch_idx - begin_req_idx
        q_row = batch_idx * head_num + head_base
        q_slot = q_writer.acquire(q_stage)
        tle.gpu.copy(Q_desc, q_slot.sQ_l, [BLOCK_M, HEAD_DIM_V // 2], [q_row, 0])
        tle.gpu.copy(
            Q_desc, q_slot.sQ_r, [BLOCK_M, HEAD_DIM_V // 2], [q_row, HEAD_DIM_V // 2]
        )
        if HAVE_TAIL:
            tle.gpu.copy(
                Q_tail_desc, q_slot.sQ_tail, [BLOCK_M, D_CHUNK], [q_row, HEAD_DIM_V]
            )
        q_writer.commit(q_stage)

        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        block_table_base = batch_idx * stride_block_table_b
        n_blocks = end_block_idx - start_block_idx
        n_full_pairs = n_blocks // 2
        has_tail = (n_blocks - n_full_pairs * 2) > 0
        n_pair_slots = n_full_pairs + has_tail.to(tl.int32)

        for pair in tl.range(0, n_full_pairs):
            pipe_idx = pipe_base + pair
            k1_pipe_idx = k1_pipe_base + pair
            block0 = start_block_idx + pair * 2
            block1 = block0 + 1
            page0 = tle.load(Block_table + block_table_base + block0)
            page1 = tle.load(Block_table + block_table_base + block1)
            kv_row0 = page0 * PAGE_SIZE
            kv_row1 = page1 * PAGE_SIZE

            k0_l_slot = k0_l_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc, k0_l_slot.sK, [BLOCK_N, HEAD_DIM_V // 2], [kv_row0, 0]
            )
            k0_l_writer.commit(pipe_idx)

            k0_r_slot = k0_r_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc,
                k0_r_slot.sK,
                [BLOCK_N, HEAD_DIM_V // 2],
                [kv_row0, HEAD_DIM_V // 2],
            )
            if HAVE_TAIL:
                tle.gpu.copy(
                    Kv_tail_desc,
                    k0_r_slot.sK_tail,
                    [BLOCK_N, D_CHUNK],
                    [kv_row0, HEAD_DIM_V],
                )
            k0_r_writer.commit(pipe_idx)

            k1_l_slot = k1_l_writer.acquire(k1_pipe_idx)
            tle.gpu.copy(
                Kv_desc, k1_l_slot.sK, [BLOCK_N, HEAD_DIM_V // 2], [kv_row1, 0]
            )
            k1_l_writer.commit(k1_pipe_idx)

            k1_r_slot = k1_r_writer.acquire(k1_pipe_idx)
            tle.gpu.copy(
                Kv_desc,
                k1_r_slot.sK,
                [BLOCK_N, HEAD_DIM_V // 2],
                [kv_row1, HEAD_DIM_V // 2],
            )
            if HAVE_TAIL:
                tle.gpu.copy(
                    Kv_tail_desc,
                    k1_r_slot.sK_tail,
                    [BLOCK_N, D_CHUNK],
                    [kv_row1, HEAD_DIM_V],
                )
            k1_r_writer.commit(k1_pipe_idx)

        if has_tail:
            pipe_idx = pipe_base + n_full_pairs
            block0 = start_block_idx + n_full_pairs * 2
            page0 = tle.load(Block_table + block_table_base + block0)
            kv_row0 = page0 * PAGE_SIZE

            k0_l_slot = k0_l_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc, k0_l_slot.sK, [BLOCK_N, HEAD_DIM_V // 2], [kv_row0, 0]
            )
            k0_l_writer.commit(pipe_idx)

            k0_r_slot = k0_r_writer.acquire(pipe_idx)
            tle.gpu.copy(
                Kv_desc,
                k0_r_slot.sK,
                [BLOCK_N, HEAD_DIM_V // 2],
                [kv_row0, HEAD_DIM_V // 2],
            )
            if HAVE_TAIL:
                tle.gpu.copy(
                    Kv_tail_desc,
                    k0_r_slot.sK_tail,
                    [BLOCK_N, D_CHUNK],
                    [kv_row0, HEAD_DIM_V],
                )
            k0_r_writer.commit(pipe_idx)

        pipe_base += n_pair_slots
        k1_pipe_base += n_full_pairs


@triton.jit
def _flash_mla_ws_consumer0(
    k0_l_reader,
    k0_r_qk_reader,
    k1_l_remote_reader,
    sM_wg0_writer,
    sM_wg1_reader,
    sP0_writer,
    sP1_reader,
    sL0_writer,
    sL1_reader,
    q_reader,
    sO_stage,
    Output_desc,
    OAccum_desc,
    O,
    sm_scale,
    B_seq_len,
    Num_splits,
    begin_req_idx,
    end_req_idx,
    begin_block_idx_meta,
    end_block_idx_meta,
    begin_split_idx,
    is_first_req_splitted,
    is_last_req_splitted,
    head_base,
    head_num,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    mask_h = head_offsets < head_num

    offs_n = tl.arange(0, BLOCK_N)
    kv_rows = tl.broadcast_to(offs_n[:, None], (BLOCK_N, HALF_DIM_V))
    kv_cols = tl.broadcast_to(tl.arange(0, HALF_DIM_V)[None, :], (BLOCK_N, HALF_DIM_V))
    if HAVE_TAIL:
        kv_rows_tail = tl.broadcast_to(offs_n[:, None], (BLOCK_N, D_CHUNK))
        kv_cols_tail = tl.broadcast_to(
            tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK)
        )

    pipe_base = 0
    k1_pipe_base = 0
    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        q_stage = batch_idx - begin_req_idx
        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        n_split_idx = tl.where(batch_idx == begin_req_idx, begin_split_idx, 0)
        no_split_middle = (batch_idx != begin_req_idx) & (batch_idx != end_req_idx)
        no_split_first = (batch_idx == begin_req_idx) & (~is_first_req_splitted)
        no_split_last = (batch_idx == end_req_idx) & (~is_last_req_splitted)
        is_no_split = no_split_middle | no_split_first | no_split_last
        if begin_req_idx == end_req_idx:
            is_no_split = ~is_first_req_splitted
        split_idx = tl.load(Num_splits + batch_idx) + n_split_idx

        q_wait = q_reader.wait(q_stage)
        q_slot = q_wait.slot

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HALF_DIM_V], dtype=tl.float32)

        n_blocks = end_block_idx - start_block_idx
        n_full_pairs = n_blocks // 2
        has_tail = (n_blocks - n_full_pairs * 2) > 0
        n_pair_slots = n_full_pairs + has_tail.to(tl.int32)

        for pair in tl.range(0, n_full_pairs):
            pipe_idx = pipe_base + pair
            k1_pipe_idx = k1_pipe_base + pair
            block0 = start_block_idx + pair * 2
            k0_l_wait = k0_l_reader.wait(pipe_idx)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            k0_l_slot = k0_l_wait.slot
            k0_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            q_l = tl.load(tle.gpu.local_ptr(q_slot.sQ_l))
            qk = tl.dot(q_l, tl.trans(k0_l), qk, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pipe_idx)
            k0_r_slot = k0_r_wait.slot
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols)))
            q_r = tl.load(tle.gpu.local_ptr(q_slot.sQ_r))
            qk = tl.dot(q_r, tl.trans(k0_r), qk, out_dtype=tl.float32)
            if HAVE_TAIL:
                k_tail = tl.load(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail, (kv_rows_tail, kv_cols_tail))
                )
                q_tail = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                qk = tl.dot(q_tail, tl.trans(k_tail), qk, out_dtype=tl.float32)

            valid_n = block0 * BLOCK_N + offs_n < seq_len
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))

            local_max = tl.maximum(tl.max(qk, axis=1), e_max)
            sM_slot = sM_wg0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sM_slot.sM), local_max)
            sM_wg0_writer.commit(pipe_idx)

            peer_wait = sM_wg1_reader.wait(k1_pipe_idx)
            merged_max = tl.load(tle.gpu.local_ptr(peer_wait.slot.sM))
            sM_wg1_reader.release(k1_pipe_idx)

            re_scale = tl.exp(e_max - merged_max)
            p = tl.exp(qk - merged_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            acc = acc * re_scale[:, None]

            p_save = p.to(k0_l.dtype)
            k0_r_qk_reader.release(pipe_idx)

            v_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p_save, v_l, acc, out_dtype=tl.float32)

            sP_slot = sP0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_save)
            sP0_writer.commit(pipe_idx)

            k0_l_reader.release(pipe_idx)

            peer_p_wait = sP1_reader.wait(k1_pipe_idx)
            peer_p = tl.load(tle.gpu.local_ptr(peer_p_wait.slot.sP))
            k1_l_wait = k1_l_remote_reader.wait(k1_pipe_idx)
            k1_l = tl.load(tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(peer_p, k1_l, acc, out_dtype=tl.float32)
            sP1_reader.release(k1_pipe_idx)
            k1_l_remote_reader.release(k1_pipe_idx)
            e_max = merged_max

        if has_tail:
            pipe_idx = pipe_base + n_full_pairs
            block0 = start_block_idx + n_full_pairs * 2
            k0_l_wait = k0_l_reader.wait(pipe_idx)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            k0_l_slot = k0_l_wait.slot
            k0_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            q_l = tl.load(tle.gpu.local_ptr(q_slot.sQ_l))
            qk = tl.dot(q_l, tl.trans(k0_l), qk, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pipe_idx)
            k0_r_slot = k0_r_wait.slot
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols)))
            q_r = tl.load(tle.gpu.local_ptr(q_slot.sQ_r))
            qk = tl.dot(q_r, tl.trans(k0_r), qk, out_dtype=tl.float32)
            if HAVE_TAIL:
                k_tail = tl.load(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail, (kv_rows_tail, kv_cols_tail))
                )
                q_tail = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                qk = tl.dot(q_tail, tl.trans(k_tail), qk, out_dtype=tl.float32)

            valid_n = block0 * BLOCK_N + offs_n < seq_len
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))

            new_max = tl.maximum(tl.max(qk, axis=1), e_max)
            sM_slot = sM_wg0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sM_slot.sM), new_max)
            sM_wg0_writer.commit(pipe_idx)

            re_scale = tl.exp(e_max - new_max)
            p = tl.exp(qk - new_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            acc = acc * re_scale[:, None]

            p_save = p.to(k0_l.dtype)
            k0_r_qk_reader.release(pipe_idx)

            v_l = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p_save, v_l, acc, out_dtype=tl.float32)

            sP_slot = sP0_writer.acquire(pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_save)
            sP0_writer.commit(pipe_idx)

            e_max = new_max
            k0_l_reader.release(pipe_idx)

        l_stage0 = q_stage * 2
        l_stage1 = l_stage0 + 1

        sL_slot = sL0_writer.acquire(l_stage0)
        tl.store(tle.gpu.local_ptr(sL_slot.sL), e_sum)
        sL0_writer.commit(l_stage0)

        peer_l_wait = sL1_reader.wait(l_stage1)
        total_sum = e_sum + tl.load(tle.gpu.local_ptr(peer_l_wait.slot.sL))
        sL1_reader.release(l_stage1)

        valid = total_sum > 0.0
        safe_total_sum = tl.where(valid, total_sum, 1.0)
        inv_total_sum = tl.fdiv(1.0, safe_total_sum)

        output_row = batch_idx * head_num + head_base
        if is_no_split:
            out_vals = acc * inv_total_sum[:, None]
            out_vals = tl.where(valid[:, None], out_vals, 0.0)
            out_vals_bf16 = out_vals.to(O.dtype.element_ty)
            tl.store(
                tle.gpu.local_ptr(q_slot.sQ_l), out_vals_bf16, mask=mask_h[:, None]
            )
            tle.gpu.copy(
                q_slot.sQ_l, Output_desc, [BLOCK_M, HALF_DIM_V], [output_row, 0]
            )
        else:
            out_vals = acc * inv_total_sum[:, None]
            out_vals = tl.where(valid[:, None], out_vals, 0.0)
            out_vals_q = out_vals.to(O.dtype.element_ty)
            tl.store(tle.gpu.local_ptr(q_slot.sQ_l), out_vals_q, mask=mask_h[:, None])
            oaccum_row = split_idx * head_num + head_base
            tle.gpu.copy(
                q_slot.sQ_l, OAccum_desc, [BLOCK_M, HALF_DIM_V], [oaccum_row, 0]
            )

        q_reader.release(q_stage)
        pipe_base += n_pair_slots
        k1_pipe_base += n_full_pairs


@triton.jit
def _flash_mla_ws_consumer1(
    k1_l_qk_reader,
    k1_r_reader,
    k0_r_remote_reader,
    sM_wg1_writer,
    sM_wg0_reader,
    sP1_writer,
    sP0_reader,
    sL1_writer,
    sL0_reader,
    q_reader,
    sO_stage,
    Output_desc,
    OAccum_desc,
    O,
    LSE_accum,
    sm_scale,
    B_seq_len,
    Num_splits,
    begin_req_idx,
    end_req_idx,
    begin_block_idx_meta,
    end_block_idx_meta,
    begin_split_idx,
    is_first_req_splitted,
    is_last_req_splitted,
    head_base,
    head_num,
    stride_lseaccum_split,
    stride_lseaccum_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    D_CHUNK: tl.constexpr,
    HAVE_TAIL: tl.constexpr,
):
    offs_h = tl.arange(0, BLOCK_M)
    head_offsets = head_base + offs_h
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    mask_h = head_offsets < head_num

    offs_n = tl.arange(0, BLOCK_N)
    kv_rows = tl.broadcast_to(offs_n[:, None], (BLOCK_N, HALF_DIM_V))
    kv_cols = tl.broadcast_to(tl.arange(0, HALF_DIM_V)[None, :], (BLOCK_N, HALF_DIM_V))
    if HAVE_TAIL:
        kv_rows_tail = tl.broadcast_to(offs_n[:, None], (BLOCK_N, D_CHUNK))
        kv_cols_tail = tl.broadcast_to(
            tl.arange(0, D_CHUNK)[None, :], (BLOCK_N, D_CHUNK)
        )

    pipe_base = 0
    k1_pipe_base = 0
    for batch_idx in tl.range(begin_req_idx, end_req_idx + 1):
        q_stage = batch_idx - begin_req_idx
        seq_len = tl.load(B_seq_len + batch_idx)
        start_block_idx = tl.where(batch_idx == begin_req_idx, begin_block_idx_meta, 0)
        full_end_block_idx = tl.cdiv(seq_len, BLOCK_N)
        end_block_idx = tl.where(
            batch_idx == end_req_idx, end_block_idx_meta, full_end_block_idx
        )
        n_split_idx = tl.where(batch_idx == begin_req_idx, begin_split_idx, 0)
        no_split_middle = (batch_idx != begin_req_idx) & (batch_idx != end_req_idx)
        no_split_first = (batch_idx == begin_req_idx) & (~is_first_req_splitted)
        no_split_last = (batch_idx == end_req_idx) & (~is_last_req_splitted)
        is_no_split = no_split_middle | no_split_first | no_split_last
        if begin_req_idx == end_req_idx:
            is_no_split = ~is_first_req_splitted
        split_idx = tl.load(Num_splits + batch_idx) + n_split_idx

        q_wait = q_reader.wait(q_stage)
        q_slot = q_wait.slot

        e_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HALF_DIM_V], dtype=tl.float32)

        n_blocks = end_block_idx - start_block_idx
        n_full_pairs = n_blocks // 2
        has_tail = (n_blocks - n_full_pairs * 2) > 0
        n_pair_slots = n_full_pairs + has_tail.to(tl.int32)

        for pair in tl.range(0, n_full_pairs):
            pipe_idx = pipe_base + pair
            k1_pipe_idx = k1_pipe_base + pair
            block1 = start_block_idx + pair * 2 + 1
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            k1_l_wait = k1_l_qk_reader.wait(k1_pipe_idx)
            k1_l_slot = k1_l_wait.slot
            k1_l = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols)))
            q_l = tl.load(tle.gpu.local_ptr(q_slot.sQ_l))
            qk = tl.dot(q_l, tl.trans(k1_l), qk, out_dtype=tl.float32)

            k1_r_wait = k1_r_reader.wait(k1_pipe_idx)
            k1_r_slot = k1_r_wait.slot
            k1_r = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols)))
            q_r = tl.load(tle.gpu.local_ptr(q_slot.sQ_r))
            qk = tl.dot(q_r, tl.trans(k1_r), qk, out_dtype=tl.float32)
            if HAVE_TAIL:
                k1_tail = tl.load(
                    tle.gpu.local_ptr(k1_r_slot.sK_tail, (kv_rows_tail, kv_cols_tail))
                )
                q_tail = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                qk = tl.dot(q_tail, tl.trans(k1_tail), qk, out_dtype=tl.float32)

            valid_n = block1 * BLOCK_N + offs_n < seq_len
            qk *= sm_scale
            qk = tl.where(valid_n[None, :], qk, float("-inf"))

            peer_wait = sM_wg0_reader.wait(pipe_idx)
            peer_max = tl.load(tle.gpu.local_ptr(peer_wait.slot.sM))
            sM_wg0_reader.release(pipe_idx)
            local_max = tl.maximum(e_max, tl.max(qk, axis=1))
            merged_max = tl.maximum(local_max, peer_max)
            sM_slot = sM_wg1_writer.acquire(k1_pipe_idx)
            tl.store(tle.gpu.local_ptr(sM_slot.sM), merged_max)
            sM_wg1_writer.commit(k1_pipe_idx)

            re_scale = tl.exp(e_max - merged_max)
            p = tl.exp(qk - merged_max[:, None])
            e_sum = e_sum * re_scale + tl.sum(p, axis=1)
            acc = acc * re_scale[:, None]
            p_b = p.to(k1_r.dtype)

            k1_l_qk_reader.release(k1_pipe_idx)

            v_r = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p_b, v_r, acc, out_dtype=tl.float32)

            sP_slot = sP1_writer.acquire(k1_pipe_idx)
            tl.store(tle.gpu.local_ptr(sP_slot.sP), p_b)
            sP1_writer.commit(k1_pipe_idx)

            sP0_wait = sP0_reader.wait(pipe_idx)
            p0 = tl.load(tle.gpu.local_ptr(sP0_wait.slot.sP))
            k0_r_wait = k0_r_remote_reader.wait(pipe_idx)
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p0, k0_r, acc, out_dtype=tl.float32)
            sP0_reader.release(pipe_idx)
            k0_r_remote_reader.release(pipe_idx)
            k1_r_reader.release(k1_pipe_idx)
            e_max = merged_max

        if has_tail:
            pipe_idx = pipe_base + n_full_pairs
            peer_wait = sM_wg0_reader.wait(pipe_idx)
            new_max = tl.load(tle.gpu.local_ptr(peer_wait.slot.sM))
            sM_wg0_reader.release(pipe_idx)

            re_scale = tl.exp(e_max - new_max)
            e_sum = e_sum * re_scale
            acc = acc * re_scale[:, None]
            e_max = new_max

            sP0_wait = sP0_reader.wait(pipe_idx)
            p0 = tl.load(tle.gpu.local_ptr(sP0_wait.slot.sP))
            k0_r_wait = k0_r_remote_reader.wait(pipe_idx)
            k0_r = tl.load(tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols)))
            acc = tl.dot(p0, k0_r, acc, out_dtype=tl.float32)
            sP0_reader.release(pipe_idx)
            k0_r_remote_reader.release(pipe_idx)

        l_stage0 = q_stage * 2
        l_stage1 = l_stage0 + 1
        sL_slot = sL1_writer.acquire(l_stage1)
        tl.store(tle.gpu.local_ptr(sL_slot.sL), e_sum)
        sL1_writer.commit(l_stage1)
        peer_l_wait = sL0_reader.wait(l_stage0)
        total_sum = e_sum + tl.load(tle.gpu.local_ptr(peer_l_wait.slot.sL))
        sL0_reader.release(l_stage0)
        valid = total_sum > 0.0
        safe_total_sum = tl.where(valid, total_sum, 1.0)
        lse_vals = tl.where(valid, tl.log(safe_total_sum) + e_max, float("-inf"))
        inv_total_sum = tl.fdiv(1.0, safe_total_sum)
        output_row = batch_idx * head_num + head_base
        if is_no_split:
            out_vals = acc * inv_total_sum[:, None]
            out_vals = tl.where(valid[:, None], out_vals, 0.0)
            out_vals_bf16 = out_vals.to(O.dtype.element_ty)
            tl.store(
                tle.gpu.local_ptr(q_slot.sQ_r), out_vals_bf16, mask=mask_h[:, None]
            )
            tle.gpu.copy(
                q_slot.sQ_r,
                Output_desc,
                [BLOCK_M, HALF_DIM_V],
                [output_row, HALF_DIM_V],
            )
        else:
            out_vals = acc * inv_total_sum[:, None]
            out_vals = tl.where(valid[:, None], out_vals, 0.0)
            out_vals_q = out_vals.to(O.dtype.element_ty)
            tl.store(tle.gpu.local_ptr(q_slot.sQ_r), out_vals_q, mask=mask_h[:, None])
            oaccum_row = split_idx * head_num + head_base
            tle.gpu.copy(
                q_slot.sQ_r,
                OAccum_desc,
                [BLOCK_M, HALF_DIM_V],
                [oaccum_row, HALF_DIM_V],
            )
            tl.store(
                LSE_accum
                + split_idx * stride_lseaccum_split
                + head_offsets * stride_lseaccum_h,
                lse_vals,
                mask=mask_h,
            )
        q_reader.release(q_stage)
        pipe_base += n_pair_slots
        k1_pipe_base += n_full_pairs


@triton.jit
def flash_mla_splitkv_ws_tle_kernel(
    Q_desc,
    Q_tail_desc,
    Output_desc,
    OAccum_desc,
    Kv_desc,
    Kv_tail_desc,
    Kv,
    Block_table,
    B_seq_len,
    Sched_meta,
    Num_splits,
    O,
    O_accum,
    LSE_accum,
    sm_scale,
    head_num,
    stride_kv_token,
    stride_block_table_b,
    stride_lseaccum_split,
    stride_lseaccum_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    D_CHUNK: tl.constexpr,
    META_FIELDS: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    partition_idx = tl.program_id(1)
    meta_base = Sched_meta + partition_idx * META_FIELDS
    begin_req_idx = tl.load(meta_base + 0)
    end_req_idx = tl.load(meta_base + 1)
    begin_block_idx_meta = tl.load(meta_base + 2)
    end_block_idx_meta = tl.load(meta_base + 3)
    begin_split_idx = tl.load(meta_base + 4)
    is_first_req_splitted = tl.load(meta_base + 5) != 0
    is_last_req_splitted = tl.load(meta_base + 6) != 0
    head_base = m_block_idx * BLOCK_M
    HALF_DIM_V: tl.constexpr = HEAD_DIM_V // 2
    HAVE_TAIL: tl.constexpr = HEAD_DIM > HEAD_DIM_V

    sQ_l_smem = tle.gpu.alloc(
        [1, BLOCK_M, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sQ_r_smem = tle.gpu.alloc(
        [1, BLOCK_M, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    if HAVE_TAIL:
        sQ_tail_smem = tle.gpu.alloc(
            [1, BLOCK_M, D_CHUNK],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        q_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_q",
            readers=("wg0", "wg1"),
            sQ_l=sQ_l_smem,
            sQ_r=sQ_r_smem,
            sQ_tail=sQ_tail_smem,
        )
    else:
        q_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_q",
            readers=("wg0", "wg1"),
            sQ_l=sQ_l_smem,
            sQ_r=sQ_r_smem,
        )
    sK0_l = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK0_r = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK1_l = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    sK1_r = tle.gpu.alloc(
        [1, BLOCK_N, HALF_DIM_V],
        dtype=Kv.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    if HAVE_TAIL:
        sK0_tail = tle.gpu.alloc(
            [1, BLOCK_N, D_CHUNK],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sK1_tail = tle.gpu.alloc(
            [1, BLOCK_N, D_CHUNK],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sP0_smem = sK0_tail
        sP1_smem = sK1_tail
    else:
        sP0_smem = tle.gpu.alloc(
            [1, BLOCK_M, BLOCK_N],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sP1_smem = tle.gpu.alloc(
            [1, BLOCK_M, BLOCK_N],
            dtype=Kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
    sM_smem = tle.gpu.alloc(
        [1, BLOCK_M],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    sL_smem = tle.gpu.alloc(
        [2, BLOCK_M],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    k0_l_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="flash_mla_ws_k0_l",
        sK=sK0_l,
    )
    if HAVE_TAIL:
        k0_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k0_r",
            readers=("qk", "remote"),
            sK=sK0_r,
            sK_tail=sK0_tail,
        )
    else:
        k0_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k0_r",
            readers=("qk", "remote"),
            sK=sK0_r,
        )
    k1_l_pipe = tle.pipe(
        capacity=1,
        scope="cta",
        name="flash_mla_ws_k1_l",
        readers=("qk", "remote"),
        sK=sK1_l,
    )
    if HAVE_TAIL:
        k1_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k1_r",
            sK=sK1_r,
            sK_tail=sK1_tail,
        )
    else:
        k1_r_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flash_mla_ws_k1_r",
            sK=sK1_r,
        )
    sM_wg0_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_m0", sM=sM_smem)
    sM_wg1_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_m1", sM=sM_smem)
    sP0_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_p0", sP=sP0_smem)
    sP1_pipe = tle.pipe(capacity=1, scope="cta", name="flash_mla_ws_p1", sP=sP1_smem)
    sL0_pipe = tle.pipe(capacity=2, scope="cta", name="flash_mla_ws_l0", sL=sL_smem)
    sL1_pipe = tle.pipe(capacity=2, scope="cta", name="flash_mla_ws_l1", sL=sL_smem)

    tle.gpu.warp_specialize(
        [
            (
                _flash_mla_ws_consumer0,
                (
                    k0_l_pipe.reader(),
                    k0_r_pipe.reader("qk"),
                    k1_l_pipe.reader("remote", fields=("sK",)),
                    sM_wg0_pipe.writer(),
                    sM_wg1_pipe.reader(),
                    sP0_pipe.writer(),
                    sP1_pipe.reader(),
                    sL0_pipe.writer(),
                    sL1_pipe.reader(),
                    q_pipe.reader("wg0"),
                    sK0_l,
                    Output_desc,
                    OAccum_desc,
                    O,
                    sm_scale,
                    B_seq_len,
                    Num_splits,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    begin_split_idx,
                    is_first_req_splitted,
                    is_last_req_splitted,
                    head_base,
                    head_num,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM_V,
                    D_CHUNK,
                    HAVE_TAIL,
                ),
            ),
            (
                _flash_mla_ws_consumer1,
                (
                    k1_l_pipe.reader("qk"),
                    k1_r_pipe.reader(),
                    k0_r_pipe.reader("remote", fields=("sK",)),
                    sM_wg1_pipe.writer(),
                    sM_wg0_pipe.reader(),
                    sP1_pipe.writer(),
                    sP0_pipe.reader(),
                    sL1_pipe.writer(),
                    sL0_pipe.reader(),
                    q_pipe.reader("wg1"),
                    sK1_r,
                    Output_desc,
                    OAccum_desc,
                    O,
                    LSE_accum,
                    sm_scale,
                    B_seq_len,
                    Num_splits,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    begin_split_idx,
                    is_first_req_splitted,
                    is_last_req_splitted,
                    head_base,
                    head_num,
                    stride_lseaccum_split,
                    stride_lseaccum_h,
                    BLOCK_M,
                    BLOCK_N,
                    HEAD_DIM_V,
                    D_CHUNK,
                    HAVE_TAIL,
                ),
            ),
            (
                _flash_mla_ws_kv_producer,
                (
                    q_pipe.writer(),
                    k0_l_pipe.writer(),
                    k0_r_pipe.writer(),
                    k1_l_pipe.writer(),
                    k1_r_pipe.writer(),
                    Q_desc,
                    Q_tail_desc,
                    Kv,
                    Kv_desc,
                    Kv_tail_desc,
                    Block_table,
                    B_seq_len,
                    begin_req_idx,
                    end_req_idx,
                    begin_block_idx_meta,
                    end_block_idx_meta,
                    head_base,
                    head_num,
                    stride_kv_token,
                    stride_block_table_b,
                    BLOCK_M,
                    BLOCK_N,
                    PAGE_SIZE,
                    HEAD_DIM_V,
                    D_CHUNK,
                    HAVE_TAIL,
                ),
            ),
        ],
        [4, 4],
        [216, 72],
    )


@triton.jit
def flash_mla_combine_kernel_compact(
    O_accum,
    LSE_accum,
    Num_splits,
    CombineReqIds,
    NumCombineReqs,
    O,
    head_num,
    stride_oaccum_split,
    stride_oaccum_h,
    stride_lseaccum_split,
    stride_lseaccum_h,
    stride_o_b,
    stride_o_h,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
):
    task_idx = tl.program_id(0)
    h_block_idx = tl.program_id(1)
    d_block_idx = tl.program_id(2)

    num_tasks = tl.load(NumCombineReqs)
    if task_idx < num_tasks:
        batch_idx = tl.load(CombineReqIds + task_idx)

        offs_h = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_d = d_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_h = offs_h < head_num
        mask_d = offs_d < HEAD_DIM_V

        start_split = tl.load(Num_splits + batch_idx)
        end_split = tl.load(Num_splits + batch_idx + 1)
        my_num_splits = end_split - start_split

        if my_num_splits > 1:
            max_lse = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
            for s in tl.range(0, my_num_splits):
                lse_s = tl.load(
                    LSE_accum
                    + (start_split + s) * stride_lseaccum_split
                    + offs_h * stride_lseaccum_h,
                    mask=mask_h,
                    other=float("-inf"),
                )
                max_lse = tl.maximum(max_lse, lse_s)

            acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
            sum_w = tl.zeros([BLOCK_H], dtype=tl.float32)
            valid_row = max_lse != float("-inf")
            for s in tl.range(0, my_num_splits):
                lse_s = tl.load(
                    LSE_accum
                    + (start_split + s) * stride_lseaccum_split
                    + offs_h * stride_lseaccum_h,
                    mask=mask_h,
                    other=float("-inf"),
                )
                w = tl.where(valid_row, tl.exp(lse_s - max_lse), 0.0)
                sum_w += w

                o_s = tl.load(
                    O_accum
                    + (start_split + s) * stride_oaccum_split
                    + offs_h[:, None] * stride_oaccum_h
                    + offs_d[None, :],
                    mask=mask_h[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += w[:, None] * o_s

            inv_sum = tl.where(sum_w > 0.0, tl.fdiv(1.0, sum_w), 0.0)
            acc = acc * inv_sum[:, None]

            tl.store(
                O
                + batch_idx * stride_o_b
                + offs_h[:, None] * stride_o_h
                + offs_d[None, :],
                acc.to(O.dtype.element_ty),
                mask=mask_h[:, None] & mask_d[None, :],
            )


class FlashMLATLEDecodePlan:
    def __init__(
        self,
        *,
        b: int,
        s_q: int,
        h_q: int,
        h_kv: int,
        d: int,
        dv: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
        reuse_output: bool = False,
    ) -> None:
        self.b = b
        self.s_q = s_q
        self.h_q = h_q
        self.h_kv = h_kv
        self.d = d
        self.dv = dv
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.causal = causal
        self.reuse_output = reuse_output
        self._cache_run_desc = _env_flag(
            "FLAGGEMS_VLLM_FLASH_MLA_TLE_CACHE_RUN_DESCRIPTORS", False
        )
        self._cache_q_desc = _env_flag(
            "FLAGGEMS_VLLM_FLASH_MLA_TLE_CACHE_Q_DESC", False
        )

        self.d_chunk = 64
        self.sm_scale = 1 / math.sqrt(d)
        self.num_m_blocks = triton.cdiv(s_q * h_q // h_kv, FLASH_MLA_BLOCK_M)
        self.num_sms = _get_num_sms(device)
        self.num_sm_parts = max(self.num_sms // h_kv // self.num_m_blocks, 1)
        self.total_num_splits = b + self.num_sm_parts
        self.max_combine_reqs = min(b, self.num_sm_parts)
        self.block_b = triton.next_power_of_2(b)

        self.sched_meta = torch.empty(
            (self.num_sm_parts, FLASH_MLA_META_FIELDS),
            dtype=torch.int32,
            device=device,
        )
        self.num_splits = torch.empty((b + 1,), dtype=torch.int32, device=device)
        self.combine_req_ids = torch.empty(
            (self.max_combine_reqs,), dtype=torch.int32, device=device
        )
        self.num_combine_reqs = torch.empty((1,), dtype=torch.int32, device=device)
        self.out_accum = torch.empty(
            (self.total_num_splits, h_q, dv), dtype=dtype, device=device
        )
        self.lse_accum = torch.empty(
            (self.total_num_splits, h_q), dtype=torch.float32, device=device
        )
        self.out = (
            torch.empty((b * s_q, h_q, dv), dtype=dtype, device=device)
            if reuse_output
            else None
        )

        self.metadata_valid = False
        self._last_cache_seqlens_ref: torch.Tensor | None = None
        self._last_launch_refs = ()
        self._out_accum_flat = None
        self._oaccum_desc = None
        self._oaccum_desc_refs = None
        self._kv_desc_key = None
        self._kv_desc = None
        self._kv_tail_desc = None
        self._kv_desc_refs = None
        self._out_desc_key = None
        self._output_desc = None
        self._out_desc_refs = None
        self._q_desc_key = None
        self._q_desc = None
        self._q_tail_desc = None
        self._q_desc_refs = None
        self._build_oaccum_desc()

    def invalidate_metadata(self) -> None:
        self.metadata_valid = False
        self._last_cache_seqlens_ref = None

    def _build_oaccum_desc(self) -> None:
        TensorDescriptor = _get_tensor_descriptor_cls()
        _ensure_triton_descriptor_allocator(self.device)

        self._out_accum_flat = self.out_accum.view(
            self.total_num_splits * self.h_q,
            self.dv,
        )
        self._oaccum_desc = TensorDescriptor(
            self._out_accum_flat,
            shape=[self.total_num_splits * self.h_q, self.dv],
            strides=[self.dv, 1],
            block_shape=[FLASH_MLA_BLOCK_M, self.dv // 2],
        )
        self._oaccum_desc_refs = (
            self._out_accum_flat,
            self._oaccum_desc,
        )

    def _get_oaccum_desc(self):
        if self._oaccum_desc is None:
            self._build_oaccum_desc()
        return self._oaccum_desc

    def clear_descriptor_cache(self) -> None:
        self._kv_desc_key = None
        self._kv_desc = None
        self._kv_tail_desc = None
        self._kv_desc_refs = None
        self._out_desc_key = None
        self._output_desc = None
        self._out_desc_refs = None
        self._q_desc_key = None
        self._q_desc = None
        self._q_tail_desc = None
        self._q_desc_refs = None

    def _get_kv_descs(self, TensorDescriptor, kv_flat, can_cache: bool):
        kv_block = (FLASH_MLA_BLOCK_N, self.dv // 2)
        kv_tail_block = (FLASH_MLA_BLOCK_N, self.d_chunk)

        if not self._cache_run_desc or not can_cache:
            kv_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_block),
            )
            kv_tail_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_tail_block),
            )
            return kv_desc, kv_tail_desc

        key = (
            _tensor_desc_light_key(kv_flat, kv_block),
            _tensor_desc_light_key(kv_flat, kv_tail_block),
        )
        if key != self._kv_desc_key:
            self._kv_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_block),
            )
            self._kv_tail_desc = TensorDescriptor(
                kv_flat,
                shape=[kv_flat.shape[0], self.d],
                strides=[self.d, 1],
                block_shape=list(kv_tail_block),
            )
            self._kv_desc_key = key
            self._kv_desc_refs = (
                kv_flat,
                self._kv_desc,
                self._kv_tail_desc,
            )
        return self._kv_desc, self._kv_tail_desc

    def _get_output_desc(self, TensorDescriptor, out_flat, can_cache: bool):
        block = (FLASH_MLA_BLOCK_M, self.dv // 2)

        if not self._cache_run_desc or not can_cache:
            return TensorDescriptor(
                out_flat,
                shape=[self.b * self.s_q * self.h_q, self.dv],
                strides=[self.dv, 1],
                block_shape=list(block),
            )

        key = _tensor_desc_light_key(out_flat, block)
        if key != self._out_desc_key:
            self._output_desc = TensorDescriptor(
                out_flat,
                shape=[self.b * self.s_q * self.h_q, self.dv],
                strides=[self.dv, 1],
                block_shape=list(block),
            )
            self._out_desc_key = key
            self._out_desc_refs = (
                out_flat,
                self._output_desc,
            )
        return self._output_desc

    def _get_q_descs(self, TensorDescriptor, q_flat, can_cache: bool):
        q_block = (FLASH_MLA_BLOCK_M, self.dv // 2)
        q_tail_block = (FLASH_MLA_BLOCK_M, self.d_chunk)

        if not (self._cache_run_desc and self._cache_q_desc and can_cache):
            q_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_block),
            )
            q_tail_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_tail_block),
            )
            return q_desc, q_tail_desc

        key = (
            _tensor_desc_light_key(q_flat, q_block),
            _tensor_desc_light_key(q_flat, q_tail_block),
        )
        if key != self._q_desc_key:
            self._q_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_block),
            )
            self._q_tail_desc = TensorDescriptor(
                q_flat,
                shape=[self.b * self.s_q * self.h_q, self.d],
                strides=[self.d, 1],
                block_shape=list(q_tail_block),
            )
            self._q_desc_key = key
            self._q_desc_refs = (
                q_flat,
                self._q_desc,
                self._q_tail_desc,
            )
        return self._q_desc, self._q_tail_desc

    def plan(self, cache_seqlens: torch.Tensor) -> None:
        if cache_seqlens.dtype != torch.int32:
            raise TypeError("cache_seqlens must be int32")
        if not _same_cuda_device(cache_seqlens.device, self.device):
            raise ValueError("cache_seqlens device mismatch")
        if cache_seqlens.ndim != 1 or cache_seqlens.shape[0] != self.b:
            raise ValueError("cache_seqlens shape mismatch")

        cache_seqlens_tle = _contiguous_if_needed(cache_seqlens)
        flash_mla_sched_meta_kernel_v3[(1,)](
            cache_seqlens_tle,
            self.sched_meta,
            self.num_splits,
            self.combine_req_ids,
            self.num_combine_reqs,
            BLOCK_B=self.block_b,
            BATCH_SIZE=self.b,
            BLOCK_SIZE_N=FLASH_MLA_BLOCK_N,
            FIXED_OVERHEAD_NUM_BLOCKS=FLASH_MLA_FIXED_OVERHEAD_BLOCKS,
            NUM_SM_PARTS=self.num_sm_parts,
            META_FIELDS=FLASH_MLA_META_FIELDS,
            num_warps=1,
            num_stages=1,
        )

        self.metadata_valid = True
        self._last_cache_seqlens_ref = cache_seqlens_tle

    def _check_run_inputs(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        if not (
            _same_cuda_device(q.device, self.device)
            and _same_cuda_device(blocked_k.device, self.device)
            and _same_cuda_device(block_table.device, self.device)
        ):
            raise ValueError("device mismatch")
        if q.dtype != self.dtype or blocked_k.dtype != self.dtype:
            raise TypeError("dtype mismatch")
        if block_table.dtype != torch.int32:
            raise TypeError("block_table must be int32")
        if q.ndim != 4 or tuple(q.shape) != (
            self.b,
            self.s_q,
            self.h_q,
            self.d,
        ):
            raise ValueError("q shape mismatch")
        if (
            blocked_k.ndim != 4
            or blocked_k.shape[1] != self.block_size
            or blocked_k.shape[2] != self.h_kv
            or blocked_k.shape[3] != self.d
        ):
            raise ValueError("blocked_k shape mismatch")
        if block_table.ndim != 2 or block_table.shape[0] != self.b:
            raise ValueError("block_table shape mismatch")

    def _get_out_tensor(self, out: torch.Tensor | None) -> torch.Tensor:
        if out is not None:
            if not _same_cuda_device(out.device, self.device):
                raise ValueError("out device mismatch")
            if out.dtype != self.dtype:
                raise TypeError("out dtype mismatch")
            if tuple(out.shape) != (self.b * self.s_q, self.h_q, self.dv):
                raise ValueError("out shape must be (b * s_q, h_q, dv)")
            return out
        if self.reuse_output:
            if self.out is None:
                self.out = torch.empty(
                    (self.b * self.s_q, self.h_q, self.dv),
                    dtype=self.dtype,
                    device=self.device,
                )
            return self.out
        return torch.empty(
            (self.b * self.s_q, self.h_q, self.dv),
            dtype=self.dtype,
            device=self.device,
        )

    def run(
        self,
        q: torch.Tensor,
        blocked_k: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor | None = None,
        *,
        update_metadata: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._check_run_inputs(q, blocked_k, block_table)
        if update_metadata:
            if cache_seqlens is None:
                raise ValueError("cache_seqlens is required when update_metadata=True")
            self.plan(cache_seqlens)
        elif not self.metadata_valid or self._last_cache_seqlens_ref is None:
            raise RuntimeError("metadata is not valid; call plan(cache_seqlens) first")

        TensorDescriptor = _get_tensor_descriptor_cls()
        _ensure_triton_descriptor_allocator(self.device)

        q_tle = _contiguous_if_needed(q)
        blocked_k_tle = _contiguous_if_needed(blocked_k)
        kv_flat = blocked_k_tle.view(-1, self.d)
        block_table_tle = _contiguous_if_needed(block_table)
        q_flat = q_tle.view(self.b * self.s_q * self.h_q, self.d)
        out_tle = self._get_out_tensor(out)
        out_flat = out_tle.view(self.b * self.s_q * self.h_q, self.dv)
        oaccum_desc = self._get_oaccum_desc()

        q_desc, q_tail_desc = self._get_q_descs(
            TensorDescriptor,
            q_flat,
            can_cache=(q_tle is q),
        )
        output_desc = self._get_output_desc(
            TensorDescriptor,
            out_flat,
            can_cache=(self.reuse_output and out is None),
        )
        kv_desc, kv_tail_desc = self._get_kv_descs(
            TensorDescriptor,
            kv_flat,
            can_cache=(blocked_k_tle is blocked_k),
        )

        flash_mla_splitkv_ws_tle_kernel[(self.num_m_blocks, self.num_sm_parts)](
            q_desc,
            q_tail_desc,
            output_desc,
            oaccum_desc,
            kv_desc,
            kv_tail_desc,
            kv_flat,
            block_table_tle,
            self._last_cache_seqlens_ref,
            self.sched_meta,
            self.num_splits,
            out_tle,
            self.out_accum,
            self.lse_accum,
            self.sm_scale,
            self.h_q,
            kv_flat.stride(0),
            block_table_tle.stride(0),
            self.lse_accum.stride(0),
            self.lse_accum.stride(1),
            BLOCK_M=FLASH_MLA_BLOCK_M,
            BLOCK_N=FLASH_MLA_BLOCK_N,
            PAGE_SIZE=self.block_size,
            HEAD_DIM_V=self.dv,
            HEAD_DIM=self.d,
            D_CHUNK=self.d_chunk,
            META_FIELDS=FLASH_MLA_META_FIELDS,
            num_warps=4,
            num_stages=1,
        )

        flash_mla_combine_kernel_compact[
            (
                self.max_combine_reqs,
                triton.cdiv(self.h_q, FLASH_MLA_COMBINE_BLOCK_H),
                triton.cdiv(self.dv, FLASH_MLA_COMBINE_BLOCK_D),
            )
        ](
            self.out_accum,
            self.lse_accum,
            self.num_splits,
            self.combine_req_ids,
            self.num_combine_reqs,
            out_tle,
            self.h_q,
            self.out_accum.stride(0),
            self.out_accum.stride(1),
            self.lse_accum.stride(0),
            self.lse_accum.stride(1),
            out_tle.stride(0),
            out_tle.stride(1),
            BLOCK_H=FLASH_MLA_COMBINE_BLOCK_H,
            BLOCK_D=FLASH_MLA_COMBINE_BLOCK_D,
            HEAD_DIM_V=self.dv,
            num_warps=4,
            num_stages=1,
        )

        self._last_launch_refs = (
            q_tle,
            blocked_k_tle,
            kv_flat,
            block_table_tle,
            out_tle,
            out_flat,
            self._out_accum_flat,
            q_desc,
            q_tail_desc,
            output_desc,
            self._oaccum_desc,
            kv_desc,
            kv_tail_desc,
        )
        return out_tle.view(self.b, self.s_q, self.h_q, self.dv)


def _get_flash_mla_tle_decode_plan(
    *,
    b: int,
    s_q: int,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool = True,
    reuse_output: bool = False,
) -> FlashMLATLEDecodePlan:
    key = FlashMLATLEDecodePlanKey(
        device=_get_cuda_device_index(device),
        dtype=dtype,
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        causal=causal,
        reuse_output=reuse_output,
    )
    plan = _FLASH_MLA_TLE_PLAN_CACHE.get(key)
    if plan is not None:
        _FLASH_MLA_TLE_PLAN_CACHE.move_to_end(key)
        return plan

    plan = FlashMLATLEDecodePlan(
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        dtype=dtype,
        device=device,
        causal=causal,
        reuse_output=reuse_output,
    )
    max_size = _get_plan_cache_size()
    if max_size == 0:
        return plan
    _FLASH_MLA_TLE_PLAN_CACHE[key] = plan
    while len(_FLASH_MLA_TLE_PLAN_CACHE) > max_size:
        _FLASH_MLA_TLE_PLAN_CACHE.popitem(last=False)
    return plan


def get_flash_mla_tle_decode_plan(
    *,
    b: int,
    s_q: int,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool = True,
    reuse_output: bool = False,
) -> FlashMLATLEDecodePlan:
    return _get_flash_mla_tle_decode_plan(
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        dtype=dtype,
        device=device,
        causal=causal,
        reuse_output=reuse_output,
    )


def _try_flash_mla_tle(
    q: torch.Tensor,
    block_table: torch.Tensor,
    blocked_k: torch.Tensor,
    block_size: int,
    b: int,
    s_q: int,
    cache_seqlens: torch.Tensor,
    h_q: int,
    h_kv: int,
    d: int,
    dv: int,
    causal: bool,
) -> torch.Tensor | None:
    if _force_triton_flash_mla() or not HAS_TLE_FLASH_MLA:
        return None

    plan = _get_flash_mla_tle_decode_plan(
        b=b,
        s_q=s_q,
        h_q=h_q,
        h_kv=h_kv,
        d=d,
        dv=dv,
        block_size=block_size,
        dtype=q.dtype,
        device=q.device,
        causal=causal,
        reuse_output=_reuse_tle_output(),
    )
    return plan.run(
        q=q,
        blocked_k=blocked_k,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        update_metadata=True,
    )


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_H": h, "BLOCK_N": n}, num_warps=w, num_stages=s)
#         for h in [32, 64, 128]
#         for n in [32, 64, 128]
#         for w in [4, 8]
#         for s in [1, 2]
#     ],
#     key=["head_num"]
# )
@triton.heuristics(
    values={
        "EVEN_H": lambda META: META["head_num"] % META["BLOCK_H"] == 0,
    }
)
@triton.jit
def flash_mla_attn_kernel(
    Q_ptr,
    Kv_cache,
    Req_to_tokens,
    B_seq_len,
    O,
    sm_scale,
    head_num,
    stride_q_bs,
    stride_q_h,
    stride_kv_bs,
    stride_req_to_tokens_bs,
    stride_o_b,
    stride_o_h,
    stride_o_s,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_H: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_head_id = tl.program_id(0).to(tl.int64)
    cur_batch_id = tl.program_id(1).to(tl.int64)
    Req_to_tokens += stride_req_to_tokens_bs * cur_batch_id

    cur_head = cur_head_id * BLOCK_H + tl.arange(0, BLOCK_H)

    offs_d_ckv = tl.arange(0, HEAD_DIM_V)
    offs_q_nope = (
        cur_batch_id * stride_q_bs
        + cur_head[:, None] * stride_q_h
        + offs_d_ckv[None, :]
    )

    offs_d_kpe = tl.arange(HEAD_DIM_V, HEAD_DIM)
    offs_q_pe = (
        cur_batch_id * stride_q_bs
        + cur_head[:, None] * stride_q_h
        + offs_d_kpe[None, :]
    )

    if EVEN_H:
        q_nope = tl.load(Q_ptr + offs_q_nope)
        q_pe = tl.load(Q_ptr + offs_q_pe)
    else:
        mask_head = cur_head < head_num
        q_nope = tl.load(Q_ptr + offs_q_nope, mask=mask_head[:, None])
        q_pe = tl.load(Q_ptr + offs_q_pe, mask=mask_head[:, None])

    e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM_V], dtype=tl.float32)

    cur_batch_seq_len = tl.load(B_seq_len + cur_batch_id)
    loop_time = cur_batch_seq_len // BLOCK_N
    remainder = cur_batch_seq_len % BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    for i in range(0, loop_time):
        kv_page_number = tl.load(Req_to_tokens + offs_n // PAGE_SIZE)
        kv_loc = kv_page_number.to(tl.int64) * PAGE_SIZE + (offs_n % PAGE_SIZE).to(
            tl.int64
        )
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
        v_c = tl.load(Kv_cache + offs_v_c)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)  # qk_nope

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
        k_pe = tl.load(Kv_cache + offs_k_pe)

        qk = tl.dot(q_pe, k_pe, acc=qk)  # qk_rope
        qk *= sm_scale

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        offs_n += BLOCK_N

    if remainder:
        mask_kvsplit = offs_n < cur_batch_seq_len
        kv_page_number = tl.load(
            Req_to_tokens + offs_n // PAGE_SIZE,
            mask=mask_kvsplit,
            other=0,
        )
        kv_loc = kv_page_number.to(tl.int64) * PAGE_SIZE + (offs_n % PAGE_SIZE).to(
            tl.int64
        )
        offs_v_c = kv_loc[:, None] * stride_kv_bs + offs_d_ckv[None, :]
        v_c = tl.load(Kv_cache + offs_v_c, mask=mask_kvsplit[:, None], other=0.0)
        k_c = tl.trans(v_c)

        qk = tl.dot(q_nope, k_c)  # qk_nope

        offs_k_pe = kv_loc[None, :] * stride_kv_bs + offs_d_kpe[:, None]
        k_pe = tl.load(Kv_cache + offs_k_pe, mask=mask_kvsplit[None, :], other=0.0)

        qk = tl.dot(q_pe, k_pe, acc=qk)  # qk_rope
        qk *= sm_scale

        qk = tl.where(mask_kvsplit[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc = tl.dot(p.to(v_c.dtype), v_c, acc=acc)

        e_sum = e_sum * re_scale + tl.sum(p, 1)

    offs_o = (
        cur_batch_id * stride_o_b + cur_head[:, None] * stride_o_h + offs_d_ckv[None, :]
    )
    if EVEN_H:
        tl.store(
            O + offs_o,
            acc / e_sum[:, None],
        )
    else:
        tl.store(O + offs_o, acc / e_sum[:, None], mask=mask_head[:, None])


def flash_mla(
    q,
    block_table,
    blocked_k,
    max_seqlen_pad,
    block_size,
    b,
    s_q,
    cache_seqlens,
    h_q,
    h_kv,
    d,
    dv,
    causal,
):
    logger.debug("GEMS FLASH MLA")
    assert causal, "causal False not supported"
    assert d > dv, "mla with rope dim should be larger than no rope dim"

    tle_out = _try_flash_mla_tle(
        q,
        block_table,
        blocked_k,
        block_size,
        b,
        s_q,
        cache_seqlens,
        h_q,
        h_kv,
        d,
        dv,
        causal,
    )
    if tle_out is not None:
        return tle_out

    batch_size, s_q, head_num, d = list(q.shape)
    q = q.view([-1, head_num, d]).contiguous()
    blocked_k = blocked_k.view([-1, d]).contiguous()
    block_table = block_table.contiguous()
    cache_seqlens = cache_seqlens.contiguous()

    sm_scale = 1 / math.sqrt(d)

    o = torch.empty([b * s_q, h_q, dv], dtype=q.dtype, device=device)

    major, _ = get_device_capability()
    if major == 9:
        BLOCK_H = 64
        num_stages = 3
    elif major == 8:
        BLOCK_H = 32
        num_stages = 2
    elif major == 7 and vendor_name == "iluvatar":
        BLOCK_H = 32
        num_stages = 1
    elif major == 3 and vendor_name == "mthreads":
        BLOCK_H = 32
        num_stages = 1
    else:
        error.backend_not_support(device)
    BLOCK_N = 64
    grid = (
        triton.cdiv(head_num, BLOCK_H),
        batch_size,
    )
    with torch_device_fn.device(device):
        flash_mla_attn_kernel[grid](
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            o,
            sm_scale,
            head_num,
            # stride
            q.stride(0),
            q.stride(1),
            blocked_k.stride(-2),
            block_table.stride(0),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            BLOCK_H=BLOCK_H,
            BLOCK_N=BLOCK_N,
            PAGE_SIZE=block_size,
            HEAD_DIM_V=dv,
            HEAD_DIM=d,
            num_warps=8,
            num_stages=num_stages,
        )

    return o.view([b, s_q, h_q, dv])
