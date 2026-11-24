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

"""Paged dynamic split-K decode for Hygon gfx936."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from . import tle
from ..prefill.attention_blocksparse_prefill_fp8 import (
    _gfx936_load_e4m3_as_half,
    _gfx936_round_probability_e4m3,
)


@triton.jit(do_not_specialize=["batch_size", "chunk_pages", "max_splits"])
def _splitk(
    Q, K, V, PAGES, LENGTHS, QS, KS, VS, PARTIAL, LSE, OUT,
    batch_size, chunk_pages, max_splits,
    Q_ST: tl.constexpr, Q_SH: tl.constexpr, Q_SD: tl.constexpr,
    K_SP: tl.constexpr, K_ST: tl.constexpr, K_SH: tl.constexpr, K_SD: tl.constexpr,
    V_SP: tl.constexpr, V_ST: tl.constexpr, V_SH: tl.constexpr, V_SD: tl.constexpr,
    PAGE_SB: tl.constexpr, PAGE_SN: tl.constexpr,
    QS_ST: tl.constexpr, QS_SH: tl.constexpr,
    KS_SP: tl.constexpr, KS_ST: tl.constexpr, KS_SH: tl.constexpr, KS_SD: tl.constexpr,
    VS_SH: tl.constexpr,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, HEADS_PER_GROUP: tl.constexpr,
    HEAD_TILE: tl.constexpr, MTP: tl.constexpr, ROWS: tl.constexpr,
    BATCH_PAD: tl.constexpr, QUANT: tl.constexpr, DIRECT: tl.constexpr,
):
    worker = tl.program_id(0)
    workers = tl.num_programs(0)
    head_group = tl.program_id(1)
    row = tl.arange(0, ROWS)
    dims = tl.arange(0, 128)
    tokens = tl.arange(0, 64)
    row_token = row // HEAD_TILE
    group_head = head_group * HEAD_TILE + row % HEAD_TILE
    row_ok = (row_token < MTP) & (group_head < HEADS_PER_GROUP)
    if DIRECT:
        total = batch_size
    else:
        batch_ids = tl.arange(0, BATCH_PAD)
        batch_lens = tl.load(LENGTHS + batch_ids, batch_ids < batch_size, 0)
        counts = tl.where(batch_ids < batch_size, tl.maximum(1, tl.cdiv(batch_lens, chunk_pages * 64)), 0)
        starts, total = tle.cumsum(counts, axis=0)

    for task in range(worker, total * KV_HEADS, workers):
        hkv = task % KV_HEADS
        slot = task // KV_HEADS
        if DIRECT:
            batch = slot
            chunk = 0
            count = 1
        else:
            owner = (slot >= starts) & (slot < starts + counts)
            batch = tl.sum(tl.where(owner, batch_ids, 0), axis=0)
            chunk = slot - tl.sum(tl.where(owner, starts, 0), axis=0)
            count = tl.sum(tl.where(owner, counts, 0), axis=0)
        length = tl.load(LENGTHS + batch)
        first_page = chunk * chunk_pages
        last_page = tl.minimum(first_page + chunk_pages, tl.cdiv(length, 64))
        hq = hkv * HEADS_PER_GROUP + group_head
        query_token = batch * MTP + row_token
        q_ptrs = Q + query_token[:, None] * Q_ST + hq[:, None] * Q_SH + dims[None, :] * Q_SD
        if QUANT < 0:
            q = tl.load(q_ptrs, row_ok[:, None], 0)
        else:
            q = _gfx936_load_e4m3_as_half(q_ptrs, row_ok[:, None])
            q_scale = tl.load(QS + query_token * QS_ST + hq * QS_SH, row_ok, 1.0)
            if QUANT == 1:
                q_scale *= tl.load(KS)
            v_scale = tl.load(VS + (hkv * VS_SH if QUANT == 0 else 0)) * (1.0 / 256.0)

        maximum = tl.full((ROWS,), -float("inf"), tl.float32)
        total_weight = tl.zeros((ROWS,), tl.float32)
        output = tl.zeros((128, ROWS), tl.float32)
        for page in tl.range(first_page, last_page, num_stages=3):
            page_id = tl.load(PAGES + batch * PAGE_SB + page * PAGE_SN)
            position = page * 64 + tokens
            valid_kv = position < length
            k_ptrs = K + page_id * K_SP + tokens[:, None] * K_ST + hkv * K_SH + dims[None, :] * K_SD
            # Load V through its zero-copy transposed view [dim, token].  The
            # lane mapping follows the contiguous axis, so the tile arrives in
            # the MMA A-operand layout without an explicit tl.trans.  On gfx936
            # an in-register trans(v) is lowered to narrow (u16) shared-memory
            # round-trips that dominate the inner loop otherwise.
            v_ptrs_t = V + page_id * V_SP + hkv * V_SH + dims[:, None] * V_SD + tokens[None, :] * V_ST
            if QUANT < 0:
                k = tl.load(k_ptrs, valid_kv[:, None], 0)
                v = tl.load(v_ptrs_t, valid_kv[None, :], 0)
            else:
                k = _gfx936_load_e4m3_as_half(k_ptrs, valid_kv[:, None])
                v = _gfx936_load_e4m3_as_half(v_ptrs_t, valid_kv[None, :])
            # KV tokens occupy the larger matrix dimension, and all MTP rows
            # in this head group reuse the same paged K and V loads.
            score = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * 0.12751743082459868
            if QUANT >= 0:
                score *= q_scale[None, :]
                if QUANT == 0:
                    byte_offset = (
                        page_id * KS_SP + (tokens // 32) * KS_ST
                        + hkv * KS_SH + (tokens % 32) * 4 * KS_SD
                    )
                    k_scale = tl.load((KS + byte_offset).to(tl.pointer_type(tl.float32)), valid_kv, 0.0)
                    score *= k_scale[:, None]
            causal = position[:, None] <= length - MTP + row_token[None, :]
            score = tl.where(row_ok[None, :] & valid_kv[:, None] & causal, score, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(score, axis=0))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            retain = tl.exp2(maximum - safe_max)
            p = tl.exp2(score - safe_max[None, :])
            total_weight = total_weight * retain + tl.sum(p, axis=0)
            if QUANT < 0:
                operand = p.to(tl.bfloat16)
            else:
                operand = _gfx936_round_probability_e4m3(p * 256.0)
            # v is already the transposed [dim, token] view of the page.
            output = tl.dot(v, operand, output * retain[None, :], out_dtype=tl.float32)
            maximum = next_max

        reciprocal = 1.0 / tl.where(total_weight > 0, total_weight, 1.0)
        if QUANT >= 0:
            reciprocal *= v_scale
        output *= reciprocal[None, :]
        flat_row = row_token * Q_HEADS + hq
        if DIRECT or count == 1:
            offsets = (batch * MTP * Q_HEADS + flat_row[None, :]) * 128 + dims[:, None]
            tl.store(OUT + offsets, output, row_ok[None, :])
        else:
            lse = tl.where(total_weight > 0, maximum + tl.log2(total_weight), -float("inf"))
            offset = (batch * max_splits + chunk) * MTP * Q_HEADS + flat_row
            tl.store(LSE + offset, lse, row_ok)
            tl.store(PARTIAL + offset[None, :] * 128 + dims[:, None], output, row_ok[None, :])


@triton.jit(do_not_specialize=["chunk_pages", "max_splits"])
def _combine(PARTIAL, LSE, LENGTHS, OUT, chunk_pages, max_splits, Q_ROWS: tl.constexpr, SPLITS: tl.constexpr):
    batch = tl.program_id(0)
    row = tl.program_id(1)
    length = tl.load(LENGTHS + batch)
    count = tl.cdiv(length, chunk_pages * 64)
    if count > 1:
        dims = tl.arange(0, 128)
        offsets = tl.arange(0, SPLITS)
        maximum = tl.full((), -float("inf"), tl.float32)
        total_weight = tl.full((), 0.0, tl.float32)
        output = tl.zeros((128,), tl.float32)
        for start in range(0, count, SPLITS):
            split = start + offsets
            index = (batch * max_splits + split) * Q_ROWS + row
            lse = tl.load(LSE + index, split < count, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(lse, axis=0))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            weights = tl.exp2(lse - safe_max)
            retain = tl.exp2(maximum - safe_max)
            values = tl.load(PARTIAL + index[:, None] * 128 + dims[None, :], split[:, None] < count, 0)
            output = output * retain + tl.sum(weights[:, None] * values, axis=0)
            total_weight = total_weight * retain + tl.sum(weights, axis=0)
            maximum = next_max
        output /= tl.where(total_weight > 0, total_weight, 1.0)
        tl.store(OUT + (batch * Q_ROWS + row) * 128 + dims, output)


@dataclass
class HygonDecodeWorkspace:
    mtp: int
    quant: int
    chunk_pages: int
    max_splits: int
    workers: int
    partial: torch.Tensor
    lse: torch.Tensor
    out: torch.Tensor
    route: str = "hygon_split_k"


def is_hygon_decode_device(device: torch.device) -> bool:
    if device.type != "cuda" or torch.version.hip is None or tle is None or not hasattr(tle, "cumsum"):
        return False
    properties = torch.cuda.get_device_properties(device)
    return getattr(properties, "gcnArchName", "").split(":", 1)[0] == "gfx936"


def prepare_hygon_decode(inputs, quant: int) -> HygonDecodeWorkspace:
    q, k, v = inputs.q, inputs.k_cache, inputs.v_cache
    mtp = inputs.mtp
    batch = inputs.batch
    if q.ndim != 3 or q.shape[-1] != 128 or k.ndim != 4 or k.shape != v.shape or k.shape[1] != 64:
        raise ValueError("Hygon decode requires rank-3 Q and matching 64-token paged K/V with head dimension 128")
    if k.shape[-1] != 128 or not k.shape[2] or not q.shape[1] or q.shape[1] % k.shape[2]:
        raise ValueError("Q/K head dimensions or grouped-query head counts are incompatible")
    dtype = torch.bfloat16 if quant < 0 else torch.float8_e4m3fn
    if any(t.dtype != dtype or t.device != q.device for t in (q, k, v)):
        raise ValueError("Q/K/V must have the expected precision and be on the same device")
    if inputs.block_ids.ndim != 2 or inputs.block_ids.shape[0] != batch or inputs.block_ids.dtype != torch.int32:
        raise ValueError("block_ids must be a batch-major int32 page table")
    if inputs.kv_lens.ndim != 1 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError("kv_lens must be rank-1 int32")
    if any(t.device != q.device for t in (inputs.block_ids, inputs.kv_lens)):
        raise ValueError("Decode metadata must be on the Q device")
    if quant >= 0:
        if inputs.q_scale.shape != q.shape[:2] or inputs.q_scale.dtype != torch.float32:
            raise ValueError("FP8 Q scales must be FP32 per query and head")
        if any(t.device != q.device for t in (inputs.q_scale, inputs.k_scale, inputs.v_scale)):
            raise ValueError("FP8 scales must be on the Q device")
        if quant == 0:
            if inputs.k_scale.shape != (k.shape[0], 2, k.shape[2], 128) or inputs.k_scale.element_size() != 1:
                raise ValueError("Per-token FP8 K scales must use the packed cache layout")
            if inputs.v_scale.shape != (k.shape[2],) or inputs.v_scale.dtype != torch.float32:
                raise ValueError("FP8 V scales must be FP32 per KV head")
        elif any(t.numel() != 1 or t.dtype != torch.float32 for t in (inputs.k_scale, inputs.v_scale)):
            raise ValueError("Per-tensor FP8 K/V scales must be FP32 scalars")
    lengths = inputs.kv_lens.tolist()
    if min(lengths) < 0 or max(lengths) > inputs.block_ids.shape[1] * 64:
        raise ValueError("KV lengths must fit in the page table")
    units = torch.cuda.get_device_properties(q.device).multi_processor_count
    tokens = sum(lengths)
    uneven = batch * max(lengths) > 2 * tokens
    min_chunk_pages = 4 if uneven else 8
    target_pages = max(1, triton.cdiv(tokens * k.shape[2], 2 * units * 64))
    chunk_pages = max(min_chunk_pages, triton.next_power_of_2(target_pages))
    max_splits = max(1, triton.cdiv(inputs.block_ids.shape[1], chunk_pages))
    workers = min(2 * units, batch * max_splits * k.shape[2])
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    # Direct-only launches never touch the partial buffers.
    rows = mtp * q.shape[1]
    lse = torch.empty((batch, max_splits, rows) if max_splits > 1 else (0,), dtype=torch.float32, device=q.device)
    partial = torch.empty((*lse.shape, 128) if max_splits > 1 else (0,), dtype=torch.float32, device=q.device)
    return HygonDecodeWorkspace(mtp, quant, chunk_pages, max_splits, workers, partial, lse, out)


def attention_decode_hygon(inputs, workspace: HygonDecodeWorkspace):
    q, k, v = inputs.q, inputs.k_cache, inputs.v_cache
    mtp, quant = workspace.mtp, workspace.quant
    if q.shape != workspace.out.shape or q.device != workspace.out.device or inputs.mtp != mtp:
        raise ValueError("Q no longer matches the prepared Hygon decode workspace")
    if inputs.block_ids.shape[1] > workspace.max_splits * workspace.chunk_pages:
        raise ValueError("Page table capacity exceeds the prepared Hygon decode workspace")
    hq, hkv = q.shape[1], k.shape[2]
    head_group = hq // hkv
    head_tile = min(8, triton.next_power_of_2(head_group))
    qs = inputs.q_scale if quant >= 0 else q
    ks = inputs.k_scale if quant >= 0 else q
    vs = inputs.v_scale if quant >= 0 else q
    ks_strides = ks.stride() if quant == 0 else (0, 0, 0, 0)
    direct = workspace.max_splits == 1
    _splitk[(workspace.workers, triton.cdiv(head_group, head_tile))](
        q, k, v, inputs.block_ids, inputs.kv_lens, qs, ks, vs, workspace.partial, workspace.lse, workspace.out,
        inputs.batch, workspace.chunk_pages, workspace.max_splits,
        *q.stride(),
        k.stride(0), k.stride(1), k.stride(2) if hkv > 1 else 0, k.stride(3),
        v.stride(0), v.stride(1), v.stride(2) if hkv > 1 else 0, v.stride(3),
        *inputs.block_ids.stride(), *(qs.stride() if quant >= 0 else (0, 0)),
        *ks_strides, vs.stride(0) if quant == 0 and hkv > 1 else 0,
        hq, hkv, head_group, head_tile, mtp,
        max(16, triton.next_power_of_2(mtp * head_tile)), triton.next_power_of_2(inputs.batch), quant, direct,
        num_warps=4,
    )
    if not direct:
        _combine[(inputs.batch, mtp * hq)](
            workspace.partial, workspace.lse, inputs.kv_lens, workspace.out,
            workspace.chunk_pages, workspace.max_splits, mtp * hq, 16, num_warps=4,
        )
    return workspace.out
