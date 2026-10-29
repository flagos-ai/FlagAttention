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

"""E4M3 decode kernels that expand the paged FP8 cache tiles the producers read.

Each of the four has an arithmetic and a LUT form; the caller picks one per
shape, and both must stay bit-identical.
"""

import triton
import triton.language as tl

from ..e4m3 import (
    e4m3_exact_masked,
    e4m3_lut,
)

from .constants import (
    _FP8_P_SCALE_JIT,
    _HEAD_DIM_JIT,
)


@triton.jit
def _fp8_prefill_decode_q_kernel(
    Q,
    Q_SCALE,
    CU_SEQLENS_Q,
    FP8_LUT,
    Q_BF16,
    Q_STOKEN: tl.constexpr,
    Q_SHEAD: tl.constexpr,
    QS_SBATCH: tl.constexpr,
    QS_SHEAD: tl.constexpr,
    QS_STOKEN: tl.constexpr,
    QB_STOKEN: tl.constexpr,
    QB_SHEAD: tl.constexpr,
    Q_TOKENS: tl.constexpr,
):
    """Decode one tile of the packed prefill queries through the LUT tier.

    One program per ``(q_tile, q_head, batch)``; the tile is ``Q_TOKENS`` rows of
    the full head dimension, masked by the batch's ``cu_seqlens_q`` span.
    """
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    offs_m = q_tile * Q_TOKENS + tl.arange(0, Q_TOKENS)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    valid = offs_m < q_len
    ptr = Q + (q_begin + offs_m[:, None]) * Q_STOKEN + q_head * Q_SHEAD + offs_d[None, :]
    q = e4m3_lut(FP8_LUT, ptr, valid[:, None])
    scale = tl.load(
        Q_SCALE + batch * QS_SBATCH + q_head * QS_SHEAD + offs_m * QS_STOKEN,
        mask=valid,
        other=0.0,
    )
    tl.store(
        Q_BF16 + (q_begin + offs_m[:, None]) * QB_STOKEN + q_head * QB_SHEAD + offs_d[None, :],
        q * scale[:, None],
        mask=valid[:, None],
    )


@triton.jit
def _fp8_prefill_decode_kv_kernel(
    K,
    V,
    K_SCALE,
    V_SCALE,
    FP8_LUT,
    K_BF16,
    V_BF16,
    PAGE_SIZE: tl.constexpr,
    K_SPAGE: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    V_SPAGE: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    KS_SPAGE: tl.constexpr,
    KS_SGROUP: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SBYTE: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    BLOCK_IDS,
    NUM_PAGES: tl.constexpr,
    PAGE_FROM_TABLE: tl.constexpr,
    TABLE_TOTAL: tl.constexpr,
    TABLE_MAX_PAGES: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
):
    """Decode the referenced KV pages through the LUT tier.

    Walks the page table, so exactly the referenced pages are decoded instead of
    the whole cache (the official inputs allocate twice the physical pages they
    reference).  ``K_SCALE_PER_TOKEN`` selects the per-token or per-tensor key
    scale.
    """
    slot = tl.program_id(0)
    kv_head = tl.program_id(1)
    if PAGE_FROM_TABLE:
        t_batch = slot // TABLE_MAX_PAGES
        t_slot = slot - t_batch * TABLE_MAX_PAGES
        page = tl.minimum(
            tl.load(
                BLOCK_IDS + t_batch * BID_SBATCH + t_slot * BID_SPAGE
            ).to(tl.int64),
            NUM_PAGES - 1,
        )
        page_valid = slot < TABLE_TOTAL
    else:
        page = slot
        page_valid = page < NUM_PAGES
    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, _HEAD_DIM_JIT)
    k_ptr = K + page * K_SPAGE + offs_n[:, None] * K_STOKEN + kv_head * K_SHEAD + offs_d[None, :]
    v_ptr = V + page * V_SPAGE + offs_n[:, None] * V_STOKEN + kv_head * V_SHEAD + offs_d[None, :]
    tile_mask = tl.full((PAGE_SIZE, _HEAD_DIM_JIT), page_valid, tl.int1)
    k = e4m3_lut(FP8_LUT, k_ptr, tile_mask)
    v = e4m3_lut(FP8_LUT, v_ptr, tile_mask)
    if K_SCALE_PER_TOKEN:
        scale_ptr = (
            K_SCALE
            + page * KS_SPAGE
            + (offs_n // 32) * KS_SGROUP
            + kv_head * KS_SHEAD
            + (offs_n % 32) * 4 * KS_SBYTE
        )
        scale_word = tl.zeros((PAGE_SIZE,), tl.int32)
        for byte_id in tl.static_range(0, 4):
            raw = tl.load(scale_ptr + byte_id * KS_SBYTE, mask=page_valid, other=0).to(tl.int32) & 255
            scale_word |= raw << (byte_id * 8)
        k_scale = scale_word.to(tl.uint32).to(tl.float32, bitcast=True)
        value_scale = tl.load(V_SCALE + kv_head)
    else:
        k_scale = tl.load(K_SCALE)
        value_scale = tl.load(V_SCALE)
    tl.store(
        K_BF16 + page * KB_SPAGE + offs_n[:, None] * KB_STOKEN + kv_head * KB_SHEAD + offs_d[None, :],
        k * k_scale[:, None] if K_SCALE_PER_TOKEN else k * k_scale,
        mask=tile_mask,
    )
    tl.store(
        V_BF16 + page * VB_SPAGE + offs_n[:, None] * VB_STOKEN + kv_head * VB_SHEAD + offs_d[None, :],
        v * (value_scale / _FP8_P_SCALE_JIT),
        mask=tile_mask,
    )


@triton.jit
def _fp8_prefill_decode_q_arithmetic_kernel(
    Q,
    Q_SCALE,
    CU_SEQLENS_Q,
    FP8_LUT,
    Q_BF16,
    Q_STOKEN: tl.constexpr,
    Q_SHEAD: tl.constexpr,
    QS_SBATCH: tl.constexpr,
    QS_SHEAD: tl.constexpr,
    QS_STOKEN: tl.constexpr,
    QB_STOKEN: tl.constexpr,
    QB_SHEAD: tl.constexpr,
    Q_TOKENS: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_BLOCKS: tl.constexpr,
    USE_ARITHMETIC: tl.constexpr,
):
    """Arithmetic-tier Q decode, blocked over the head dimension.

    Same contract as ``_fp8_prefill_decode_q_kernel``, but the head dimension is
    split into ``D_BLOCKS`` tiles so that the arithmetic decoder's temporaries fit
    in UB.  ``USE_ARITHMETIC`` picks the tier at compile time, and the two tiers
    are bit-identical.
    """
    flat_tile = tl.program_id(0)
    q_tile = flat_tile // D_BLOCKS
    d_tile = flat_tile % D_BLOCKS
    q_head = tl.program_id(1)
    batch = tl.program_id(2)
    q_begin = tl.load(CU_SEQLENS_Q + batch)
    q_end = tl.load(CU_SEQLENS_Q + batch + 1)
    q_len = q_end - q_begin
    offs_m = q_tile * Q_TOKENS + tl.arange(0, Q_TOKENS)
    offs_d = d_tile * D_BLOCK + tl.arange(0, D_BLOCK)
    valid = offs_m < q_len
    ptr = Q + (q_begin + offs_m[:, None]) * Q_STOKEN + q_head * Q_SHEAD + offs_d[None, :]
    if USE_ARITHMETIC:
        q = e4m3_exact_masked(ptr, valid[:, None])
    else:
        q = e4m3_lut(FP8_LUT, ptr, valid[:, None])
    scale = tl.load(
        Q_SCALE + batch * QS_SBATCH + q_head * QS_SHEAD + offs_m * QS_STOKEN,
        mask=valid,
        other=0.0,
    )
    tl.store(
        Q_BF16 + (q_begin + offs_m[:, None]) * QB_STOKEN + q_head * QB_SHEAD + offs_d[None, :],
        q * scale[:, None],
        mask=valid[:, None],
    )


@triton.jit
def _fp8_prefill_decode_kv_arithmetic_kernel(
    K,
    V,
    K_SCALE,
    V_SCALE,
    FP8_LUT,
    K_BF16,
    V_BF16,
    PAGE_SIZE: tl.constexpr,
    K_SPAGE: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    V_SPAGE: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    KS_SPAGE: tl.constexpr,
    KS_SGROUP: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SBYTE: tl.constexpr,
    KB_SPAGE: tl.constexpr,
    KB_STOKEN: tl.constexpr,
    KB_SHEAD: tl.constexpr,
    VB_SPAGE: tl.constexpr,
    VB_STOKEN: tl.constexpr,
    VB_SHEAD: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    PAGES_PER_PROGRAM: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_BLOCKS: tl.constexpr,
    USE_ARITHMETIC: tl.constexpr,
    BLOCK_IDS,
    PAGE_FROM_TABLE: tl.constexpr,
    TABLE_TOTAL: tl.constexpr,
    TABLE_MAX_PAGES: tl.constexpr,
    BID_SBATCH: tl.constexpr,
    BID_SPAGE: tl.constexpr,
):
    """Arithmetic-tier KV decode over the referenced pages.

    Same contract as ``_fp8_prefill_decode_kv_kernel`` with the arithmetic
    decoder.  The page table is walked (so dead pages cost nothing) and each
    program covers ``PAGES_PER_PROGRAM`` pages of ``D_BLOCK`` head-dimension
    columns at a time.
    """
    flat_group = tl.program_id(0)
    page_group = flat_group // D_BLOCKS
    d_tile = flat_group % D_BLOCKS
    kv_head = tl.program_id(1)
    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = d_tile * D_BLOCK + tl.arange(0, D_BLOCK)
    for page_offset in tl.static_range(0, PAGES_PER_PROGRAM):
        if PAGE_FROM_TABLE:
            # The page table is the list of pages that will ever be read, so
            # walking it decodes exactly the referenced pages (official inputs
            # allocate 2*requested physical pages and reference half of them).
            # The grid is then sized by referenced entries, not by cache size,
            # which also removes the per-program cost of the dead ones -- an
            # early-return over the full grid only recovered 28% of the 50%.
            table_idx = page_group * PAGES_PER_PROGRAM + page_offset
            t_batch = table_idx // TABLE_MAX_PAGES
            t_slot = table_idx - t_batch * TABLE_MAX_PAGES
            page = tl.minimum(
                tl.load(
                    BLOCK_IDS + t_batch * BID_SBATCH + t_slot * BID_SPAGE
                ).to(tl.int64),
                NUM_PAGES - 1,
            )
            page_valid = table_idx < TABLE_TOTAL
        else:
            page = page_group * PAGES_PER_PROGRAM + page_offset
            page_valid = page < NUM_PAGES
        tile_mask = tl.full((PAGE_SIZE, D_BLOCK), page_valid, tl.int1)
        k_ptr = K + page * K_SPAGE + offs_n[:, None] * K_STOKEN + kv_head * K_SHEAD + offs_d[None, :]
        v_ptr = V + page * V_SPAGE + offs_n[:, None] * V_STOKEN + kv_head * V_SHEAD + offs_d[None, :]
        if USE_ARITHMETIC:
            k = e4m3_exact_masked(k_ptr, tile_mask)
            v = e4m3_exact_masked(v_ptr, tile_mask)
        else:
            k = e4m3_lut(FP8_LUT, k_ptr, tile_mask)
            v = e4m3_lut(FP8_LUT, v_ptr, tile_mask)
        if K_SCALE_PER_TOKEN:
            scale_ptr = (
                K_SCALE
                + page * KS_SPAGE
                + (offs_n // 32) * KS_SGROUP
                + kv_head * KS_SHEAD
                + (offs_n % 32) * 4 * KS_SBYTE
            )
            scale_word = tl.zeros((PAGE_SIZE,), tl.int32)
            for byte_id in tl.static_range(0, 4):
                raw = tl.load(
                    scale_ptr + byte_id * KS_SBYTE,
                    mask=page_valid,
                    other=0,
                ).to(tl.int32) & 255
                scale_word |= raw << (byte_id * 8)
            k_scale = scale_word.to(tl.uint32).to(tl.float32, bitcast=True)
            value_scale = tl.load(V_SCALE + kv_head)
        else:
            k_scale = tl.load(K_SCALE)
            value_scale = tl.load(V_SCALE)
        tl.store(
            K_BF16 + page * KB_SPAGE + offs_n[:, None] * KB_STOKEN + kv_head * KB_SHEAD + offs_d[None, :],
            k * k_scale[:, None] if K_SCALE_PER_TOKEN else k * k_scale,
            mask=tile_mask,
        )
        tl.store(
            V_BF16 + page * VB_SPAGE + offs_n[:, None] * VB_STOKEN + kv_head * VB_SHEAD + offs_d[None, :],
            v * (value_scale / _FP8_P_SCALE_JIT),
            mask=tile_mask,
        )
