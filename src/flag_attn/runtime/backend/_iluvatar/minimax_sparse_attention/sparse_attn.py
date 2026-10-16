# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Triton kernels for MiniMax M3 block-sparse GQA attention.

The main heads attend only to the blocks selected by the lightning indexer (see
``index_topk``). Adapted to vLLM's paged KV cache: the KV page size is forced to
equal the sparse block size (128), so one selected block maps to exactly one
page.

Main K/V cache layout (vLLM):
  ``(num_blocks, num_kv_heads, 128, 2 * head_dim)``
  K=[..., :head_dim] V=[..., head_dim:]

Only the paths MiniMax M3 uses are implemented: no attention sink, base-2
(exp2/log2) softmax. The decode kernels use split-K (flash-decoding) over the
selected blocks with a separate merge step, since one query token per request
leaves the prefill kernels (which parallelize over the query dim) idle.
"""

import torch
import triton
import triton.language as tl

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

# Keep sparse selection/page semantics at 128 tokens, but compute direct
# prefill attention in smaller tiles to reduce CoreX shared-memory and register
# pressure. On the 2-warp + V-prefetch stack, 32 beats 64, especially large GQA.
_PREFILL_DIRECT_BLOCK_SIZE_K = 32

# Decode keeps logical page 128. Align the compute stack with current prefill:
# tile 32 + paired softmax, 2 warps, GQA padded only to next_power_of_2.
_DECODE_DIRECT_BLOCK_SIZE_K = 32
_DECODE_NUM_WARPS = 2
# Split-K CTA budget over (token, kv-head). 128 beats 256 on the tile32 /
# 2-warp stack; a single chunk already writes the final output, so skip merge.
_DECODE_TARGET_GRID = 128

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. PAGE_SIZE_K == 128 so each selected block is one page; BLOCK_SIZE_K
# is the smaller page-internal compute tile.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit
def _qk_d128_split(
    q_lo,
    q_hi,
    k_base_ptr,
    tile_offset,
    stride_kv_d,
    stride_kv_pos,
    PAGE_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    k_lo_ptrs = tl.make_block_ptr(
        base=k_base_ptr,
        shape=(128, PAGE_SIZE_K),
        strides=(stride_kv_d, stride_kv_pos),
        offsets=(0, tile_offset),
        block_shape=(64, BLOCK_SIZE_K),
        order=(0, 1),
    )
    k_lo = tl.load(k_lo_ptrs, boundary_check=(0, 1), padding_option="zero")
    scores = tl.dot(q_lo, k_lo)
    k_hi_ptrs = tl.make_block_ptr(
        base=k_base_ptr,
        shape=(128, PAGE_SIZE_K),
        strides=(stride_kv_d, stride_kv_pos),
        offsets=(64, tile_offset),
        block_shape=(64, BLOCK_SIZE_K),
        order=(0, 1),
    )
    k_hi = tl.load(k_hi_ptrs, boundary_check=(0, 1), padding_option="zero")
    return tl.dot(q_hi, k_hi, scores)


@triton.jit
def _pv_d128_split(
    p,
    acc_lo,
    acc_hi,
    v_base_ptr,
    tile_offset,
    stride_kv_pos,
    stride_kv_d,
    PAGE_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    v_lo_ptrs = tl.make_block_ptr(
        base=v_base_ptr,
        shape=(PAGE_SIZE_K, 128),
        strides=(stride_kv_pos, stride_kv_d),
        offsets=(tile_offset, 0),
        block_shape=(BLOCK_SIZE_K, 64),
        order=(1, 0),
    )
    v_lo = tl.load(v_lo_ptrs, boundary_check=(0, 1), padding_option="zero")
    acc_lo += tl.dot(p.to(v_lo.dtype), v_lo)
    v_hi_ptrs = tl.make_block_ptr(
        base=v_base_ptr,
        shape=(PAGE_SIZE_K, 128),
        strides=(stride_kv_pos, stride_kv_d),
        offsets=(tile_offset, 64),
        block_shape=(BLOCK_SIZE_K, 64),
        order=(1, 0),
    )
    v_hi = tl.load(v_hi_ptrs, boundary_check=(0, 1), padding_option="zero")
    acc_hi += tl.dot(p.to(v_hi.dtype), v_hi)
    return acc_lo, acc_hi



@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_direct(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    gqa_group_size,
    head_dim,
    max_topk,
    num_q_loop,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    PAGE_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 2: [kv_head, token]; scalar scales are applied before launch
    SPLIT_D: tl.constexpr = False,
):
    if SPLIT_D:
        sm_scale_log2e = sm_scale * 1.4426950409
        pid_q = tl.program_id(0)
        pid_kh = tl.program_id(1)
        pid_b = tl.program_id(2)
        q_start = tl.load(cu_seqlens_q + pid_b)
        q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
        q_block_start = tl.load(cu_seqblocks_q + pid_b)
        q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
        seq_len = tl.load(seq_lens + pid_b)
        prefix_len = tl.load(prefix_lens + pid_b)
        if pid_q * num_q_loop >= q_block_len:
            return
        real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
        bt_row = block_table_ptr + pid_b * stride_bt_b
        off_n = tl.arange(0, BLOCK_SIZE_K)

        for j in range(real_q_loop):
            pid_q_j = pid_q * num_q_loop + j
            t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
            q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
            valid_blocks = (q_abs + PAGE_SIZE_K) // PAGE_SIZE_K
            real_topk = tl.minimum(max_topk, valid_blocks)
            q_base = q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh
            q_lo_ptrs = tl.make_block_ptr(
                base=q_base,
                shape=(q_len, gqa_group_size, 128),
                strides=(stride_qn, stride_qh, stride_qd),
                offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, 64),
                order=(2, 1, 0),
            )
            q_hi_ptrs = tl.make_block_ptr(
                base=q_base,
                shape=(q_len, gqa_group_size, 128),
                strides=(stride_qn, stride_qh, stride_qd),
                offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 64),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, 64),
                order=(2, 1, 0),
            )
            q_lo = tl.reshape(
                tl.load(q_lo_ptrs, boundary_check=(0, 1, 2), padding_option="zero"),
                BLOCK_SIZE_QH, 64,
            )
            q_hi = tl.reshape(
                tl.load(q_hi_ptrs, boundary_check=(0, 1, 2), padding_option="zero"),
                BLOCK_SIZE_QH, 64,
            )
            q_pos = tl.arange(0, BLOCK_SIZE_Q) + pid_q_j * BLOCK_SIZE_Q + prefix_len
            m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
            acc_lo = tl.zeros((BLOCK_SIZE_QH, 64), dtype=tl.float32)
            acc_hi = tl.zeros((BLOCK_SIZE_QH, 64), dtype=tl.float32)

            for _ in range(real_topk):
                blk = tl.load(t_ptr_j).to(tl.int32)
                t_ptr_j += stride_tk
                page_start = blk * PAGE_SIZE_K
                page = tl.load(bt_row + blk).to(tl.int64)
                k_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
                v_base = k_base + head_dim * stride_kv_d

                # The page contains four tile32 chunks. Merge all four logits
                # under one online-softmax update, halving row reductions and
                # accumulator rescaling versus the pair-at-a-time version.
                tile_0 = 0
                tile_1 = BLOCK_SIZE_K
                tile_2 = 2 * BLOCK_SIZE_K
                tile_3 = 3 * BLOCK_SIZE_K

                pos_0 = page_start + tile_0 + off_n
                pos_1 = page_start + tile_1 + off_n
                pos_2 = page_start + tile_2 + off_n
                pos_3 = page_start + tile_3 + off_n
                mask_0 = pos_0 < seq_len
                mask_1 = pos_1 < seq_len
                mask_2 = pos_2 < seq_len
                mask_3 = pos_3 < seq_len

                qk_0 = tl.zeros(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), tl.float32
                )
                if not ((page_start + tile_0 + BLOCK_SIZE_K) <= q_abs):
                    qk_0 += tl.where(
                        q_pos[:, None, None] >= pos_0[None, None, :],
                        0, float("-inf"),
                    )
                qk_0 = tl.reshape(qk_0, BLOCK_SIZE_QH, BLOCK_SIZE_K)
                qk_0 += _qk_d128_split(
                    q_lo, q_hi, k_base, tile_0, stride_kv_d, stride_kv_pos,
                    PAGE_SIZE_K, BLOCK_SIZE_K,
                ) * sm_scale_log2e
                if not ((page_start + tile_0 + BLOCK_SIZE_K) <= seq_len):
                    qk_0 += tl.where(mask_0[None, :], 0, float("-inf"))

                qk_1 = tl.zeros(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), tl.float32
                )
                if not ((page_start + tile_1 + BLOCK_SIZE_K) <= q_abs):
                    qk_1 += tl.where(
                        q_pos[:, None, None] >= pos_1[None, None, :],
                        0, float("-inf"),
                    )
                qk_1 = tl.reshape(qk_1, BLOCK_SIZE_QH, BLOCK_SIZE_K)
                qk_1 += _qk_d128_split(
                    q_lo, q_hi, k_base, tile_1, stride_kv_d, stride_kv_pos,
                    PAGE_SIZE_K, BLOCK_SIZE_K,
                ) * sm_scale_log2e
                if not ((page_start + tile_1 + BLOCK_SIZE_K) <= seq_len):
                    qk_1 += tl.where(mask_1[None, :], 0, float("-inf"))

                qk_2 = tl.zeros(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), tl.float32
                )
                if not ((page_start + tile_2 + BLOCK_SIZE_K) <= q_abs):
                    qk_2 += tl.where(
                        q_pos[:, None, None] >= pos_2[None, None, :],
                        0, float("-inf"),
                    )
                qk_2 = tl.reshape(qk_2, BLOCK_SIZE_QH, BLOCK_SIZE_K)
                qk_2 += _qk_d128_split(
                    q_lo, q_hi, k_base, tile_2, stride_kv_d, stride_kv_pos,
                    PAGE_SIZE_K, BLOCK_SIZE_K,
                ) * sm_scale_log2e
                if not ((page_start + tile_2 + BLOCK_SIZE_K) <= seq_len):
                    qk_2 += tl.where(mask_2[None, :], 0, float("-inf"))

                qk_3 = tl.zeros(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), tl.float32
                )
                if not ((page_start + tile_3 + BLOCK_SIZE_K) <= q_abs):
                    qk_3 += tl.where(
                        q_pos[:, None, None] >= pos_3[None, None, :],
                        0, float("-inf"),
                    )
                qk_3 = tl.reshape(qk_3, BLOCK_SIZE_QH, BLOCK_SIZE_K)
                qk_3 += _qk_d128_split(
                    q_lo, q_hi, k_base, tile_3, stride_kv_d, stride_kv_pos,
                    PAGE_SIZE_K, BLOCK_SIZE_K,
                ) * sm_scale_log2e
                if not ((page_start + tile_3 + BLOCK_SIZE_K) <= seq_len):
                    qk_3 += tl.where(mask_3[None, :], 0, float("-inf"))

                page_max = tl.max(
                    tl.maximum(
                        tl.maximum(qk_0, qk_1),
                        tl.maximum(qk_2, qk_3),
                    ),
                    1,
                )
                m_new = tl.maximum(m_i, page_max)
                p_0 = tl.exp2(qk_0 - m_new[:, None])
                p_1 = tl.exp2(qk_1 - m_new[:, None])
                p_2 = tl.exp2(qk_2 - m_new[:, None])
                p_3 = tl.exp2(qk_3 - m_new[:, None])
                alpha = tl.exp2(m_i - m_new)
                acc_lo *= alpha[:, None]
                acc_hi *= alpha[:, None]
                l_i = l_i * alpha + tl.sum(p_0 + p_1 + p_2 + p_3, 1)
                acc_lo, acc_hi = _pv_d128_split(
                    p_0, acc_lo, acc_hi, v_base, tile_0,
                    stride_kv_pos, stride_kv_d, PAGE_SIZE_K, BLOCK_SIZE_K,
                )
                acc_lo, acc_hi = _pv_d128_split(
                    p_1, acc_lo, acc_hi, v_base, tile_1,
                    stride_kv_pos, stride_kv_d, PAGE_SIZE_K, BLOCK_SIZE_K,
                )
                acc_lo, acc_hi = _pv_d128_split(
                    p_2, acc_lo, acc_hi, v_base, tile_2,
                    stride_kv_pos, stride_kv_d, PAGE_SIZE_K, BLOCK_SIZE_K,
                )
                acc_lo, acc_hi = _pv_d128_split(
                    p_3, acc_lo, acc_hi, v_base, tile_3,
                    stride_kv_pos, stride_kv_d, PAGE_SIZE_K, BLOCK_SIZE_K,
                )
                m_i = m_new

            inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
            o_base = o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh
            o_lo_ptrs = tl.make_block_ptr(
                base=o_base,
                shape=(q_len, gqa_group_size, 128),
                strides=(stride_on, stride_oh, stride_od),
                offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, 64),
                order=(2, 1, 0),
            )
            o_hi_ptrs = tl.make_block_ptr(
                base=o_base,
                shape=(q_len, gqa_group_size, 128),
                strides=(stride_on, stride_oh, stride_od),
                offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 64),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, 64),
                order=(2, 1, 0),
            )
            out_lo = tl.reshape(acc_lo * inv_l[:, None], BLOCK_SIZE_Q, BLOCK_SIZE_H, 64)
            out_hi = tl.reshape(acc_hi * inv_l[:, None], BLOCK_SIZE_Q, BLOCK_SIZE_H, 64)
            tl.store(o_lo_ptrs, out_lo.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))
            tl.store(o_hi_ptrs, out_hi.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))
    else:
        sm_scale_log2e = sm_scale * 1.4426950409
        pid_q = tl.program_id(0)
        pid_kh = tl.program_id(1)
        pid_b = tl.program_id(2)
        q_start = tl.load(cu_seqlens_q + pid_b)
        q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
        q_block_start = tl.load(cu_seqblocks_q + pid_b)
        q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
        seq_len = tl.load(seq_lens + pid_b)
        prefix_len = tl.load(prefix_lens + pid_b)
        if pid_q * num_q_loop >= q_block_len:
            return
        real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
        bt_row = block_table_ptr + pid_b * stride_bt_b
        off_n = tl.arange(0, BLOCK_SIZE_K)
        # Iluvatar: do not lower QK/PV as FP8 MMA. NVIDIA's q.to(float8e4nv)
        # path hits the 256-register cap and spills (~295 regs), so a bench-scale
        # grid (seq=8192, kv_heads=16) reserves >32GiB local memory at launch.
        # Dequantize K/V to q.dtype like decode, then use BF16 tl.dot.
        for j in range(real_q_loop):
            pid_q_j = pid_q * num_q_loop + j
            t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
            # Valid block count from seq position (no sentinel): block_size_q == 1.
            q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
            valid_blocks = (q_abs + PAGE_SIZE_K) // PAGE_SIZE_K
            real_topk = tl.minimum(max_topk, valid_blocks)
            q_ptrs = tl.make_block_ptr(
                base=q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh,
                shape=(q_len, gqa_group_size, head_dim),
                strides=(stride_qn, stride_qh, stride_qd),
                offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(2, 1, 0),
            )
            q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
            q_pos = (
                tl.arange(0, BLOCK_SIZE_Q)
                + pid_q_j * BLOCK_SIZE_Q
                + prefix_len
            )
            m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

            l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
            acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
            q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
            for _ in range(real_topk):
                blk = tl.load(t_ptr_j).to(tl.int32)
                t_ptr_j = t_ptr_j + stride_tk

                page_start = blk * PAGE_SIZE_K
                page = tl.load(bt_row + blk).to(tl.int64)

                k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
                v_base_ptr = k_base_ptr + head_dim * stride_kv_d

                # Pair two tile32 QK/PV operations under one online-softmax update.
                # MMA tile size and count remain unchanged.
                for tile_pair_idx in tl.static_range(
                    0, PAGE_SIZE_K // (2 * BLOCK_SIZE_K)
                ):
                    tile_offset_0 = tile_pair_idx * (2 * BLOCK_SIZE_K)
                    tile_offset_1 = tile_offset_0 + BLOCK_SIZE_K

                    c_0 = page_start + tile_offset_0
                    c_1 = page_start + tile_offset_1
                    pos_0 = c_0 + off_n
                    pos_1 = c_1 + off_n
                    pos_mask_0 = pos_0 < seq_len
                    pos_mask_1 = pos_1 < seq_len

                    # QK tile 0
                    k_ptrs_0 = tl.make_block_ptr(
                        base=k_base_ptr,
                        shape=(head_dim, PAGE_SIZE_K),
                        strides=(stride_kv_d, stride_kv_pos),
                        offsets=(0, tile_offset_0),
                        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                        order=(0, 1),
                    )
                    k_0 = tl.load(
                        k_ptrs_0,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    if USE_FP8:
                        k_0 = k_0.to(q.dtype)
                        if KV_SCALE_MODE == 2:
                            k_scale_0 = tl.load(
                                k_scale_ptr
                                + pid_kh * stride_ks_h
                                + (
                                    page * PAGE_SIZE_K
                                    + tile_offset_0
                                    + off_n
                                )
                                * stride_ks_t,
                                mask=pos_mask_0,
                                other=1.0,
                            )
                            k_0 = (k_0 * k_scale_0[None, :]).to(q.dtype)
                    qk_dot_0 = tl.dot(q, k_0)

                    # QK tile 1
                    k_ptrs_1 = tl.make_block_ptr(
                        base=k_base_ptr,
                        shape=(head_dim, PAGE_SIZE_K),
                        strides=(stride_kv_d, stride_kv_pos),
                        offsets=(0, tile_offset_1),
                        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                        order=(0, 1),
                    )
                    k_1 = tl.load(
                        k_ptrs_1,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    if USE_FP8:
                        k_1 = k_1.to(q.dtype)
                        if KV_SCALE_MODE == 2:
                            k_scale_1 = tl.load(
                                k_scale_ptr
                                + pid_kh * stride_ks_h
                                + (
                                    page * PAGE_SIZE_K
                                    + tile_offset_1
                                    + off_n
                                )
                                * stride_ks_t,
                                mask=pos_mask_1,
                                other=1.0,
                            )
                            k_1 = (k_1 * k_scale_1[None, :]).to(q.dtype)
                    qk_dot_1 = tl.dot(q, k_1)

                    # Load V0 after K1 has been consumed. This avoids keeping V0
                    # alive at the same time as the next K staging operation.
                    v_ptrs_0 = tl.make_block_ptr(
                        base=v_base_ptr,
                        shape=(PAGE_SIZE_K, head_dim),
                        strides=(stride_kv_pos, stride_kv_d),
                        offsets=(tile_offset_0, 0),
                        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                        order=(1, 0),
                    )
                    v_0 = tl.load(
                        v_ptrs_0,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    if USE_FP8:
                        v_0 = v_0.to(q.dtype)
                        if KV_SCALE_MODE == 2:
                            v_scale_0 = tl.load(
                                v_scale_ptr
                                + pid_kh * stride_vs_h
                                + (
                                    page * PAGE_SIZE_K
                                    + tile_offset_0
                                    + off_n
                                )
                                * stride_vs_t,
                                mask=pos_mask_0,
                                other=1.0,
                            )
                            v_0 = (v_0 * v_scale_0[:, None]).to(q.dtype)

                    # Apply causal and sequence masks independently to both tiles.
                    qk_0 = tl.zeros(
                        (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K),
                        dtype=tl.float32,
                    )
                    if not ((c_0 + BLOCK_SIZE_K) <= q_abs):
                        qk_0 += tl.where(
                            q_pos[:, None, None] >= pos_0[None, None, :],
                            0,
                            float("-inf"),
                        )
                    qk_0 = tl.reshape(
                        qk_0, BLOCK_SIZE_QH, BLOCK_SIZE_K
                    )
                    qk_0 += qk_dot_0 * sm_scale_log2e
                    if not ((c_0 + BLOCK_SIZE_K) <= seq_len):
                        qk_0 += tl.where(
                            pos_mask_0[None, :], 0, float("-inf")
                        )

                    qk_1 = tl.zeros(
                        (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K),
                        dtype=tl.float32,
                    )
                    if not ((c_1 + BLOCK_SIZE_K) <= q_abs):
                        qk_1 += tl.where(
                            q_pos[:, None, None] >= pos_1[None, None, :],
                            0,
                            float("-inf"),
                        )
                    qk_1 = tl.reshape(
                        qk_1, BLOCK_SIZE_QH, BLOCK_SIZE_K
                    )
                    qk_1 += qk_dot_1 * sm_scale_log2e
                    if not ((c_1 + BLOCK_SIZE_K) <= seq_len):
                        qk_1 += tl.where(
                            pos_mask_1[None, :], 0, float("-inf")
                        )

                    # One online-softmax state update for the two tile32 logits.
                    pair_max = tl.maximum(
                        tl.max(qk_0, axis=1),
                        tl.max(qk_1, axis=1),
                    )
                    m_ij = tl.maximum(m_i, pair_max)

                    p_0 = tl.exp2(qk_0 - m_ij[:, None])
                    p_1 = tl.exp2(qk_1 - m_ij[:, None])
                    l_ij = (
                        tl.sum(p_0, axis=1)
                        + tl.sum(p_1, axis=1)
                    )

                    alpha = tl.exp2(m_i - m_ij)
                    acc_o = acc_o * alpha[:, None]
                    l_i = l_i * alpha + l_ij

                    acc_o += tl.dot(p_0.to(v_0.dtype), v_0)

                    # Load V1 only after PV0, avoiding two simultaneous V operands.
                    v_ptrs_1 = tl.make_block_ptr(
                        base=v_base_ptr,
                        shape=(PAGE_SIZE_K, head_dim),
                        strides=(stride_kv_pos, stride_kv_d),
                        offsets=(tile_offset_1, 0),
                        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                        order=(1, 0),
                    )
                    v_1 = tl.load(
                        v_ptrs_1,
                        boundary_check=(0, 1),
                        padding_option="zero",
                    )
                    if USE_FP8:
                        v_1 = v_1.to(q.dtype)
                        if KV_SCALE_MODE == 2:
                            v_scale_1 = tl.load(
                                v_scale_ptr
                                + pid_kh * stride_vs_h
                                + (
                                    page * PAGE_SIZE_K
                                    + tile_offset_1
                                    + off_n
                                )
                                * stride_vs_t,
                                mask=pos_mask_1,
                                other=1.0,
                            )
                            v_1 = (v_1 * v_scale_1[:, None]).to(q.dtype)

                    acc_o += tl.dot(p_1.to(v_1.dtype), v_1)
                    m_i = m_ij

            inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
            acc_o = acc_o * inv_l[:, None]
            acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
            o_ptrs = tl.make_block_ptr(
                base=o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh,
                shape=(q_len, gqa_group_size, head_dim),
                strides=(stride_on, stride_oh, stride_od),
                offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
                block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(2, 1, 0),
            )
            tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode batches are flattened request-major, with a
# runtime query length used to map each query token back to its request metadata.
# This parallelizes over the selected top-k blocks, producing partials that the
# merge kernel combines (flash-decoding). All chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. Base-2 (exp2/log2)
# softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # page-internal compute tile
    PAGE_SIZE_K: tl.constexpr,  # sparse page, 128
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, page)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + PAGE_SIZE_K - 1) // PAGE_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        page_start = blk * PAGE_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        v_base_ptr = k_base_ptr + head_dim * stride_kv_d
        # Paired tiles share one online-softmax update, matching prefill.
        for tile_pair_idx in tl.static_range(0, PAGE_SIZE_K // (2 * BLOCK_SIZE_K)):
            tile_offset_0 = tile_pair_idx * (2 * BLOCK_SIZE_K)
            tile_offset_1 = tile_offset_0 + BLOCK_SIZE_K
            pos_0 = page_start + tile_offset_0 + off_n
            pos_1 = page_start + tile_offset_1 + off_n
            pos_mask_0 = pos_0 < kv_len
            pos_mask_1 = pos_1 < kv_len

            k_ptrs_0 = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(head_dim, PAGE_SIZE_K),
                strides=(stride_kv_d, stride_kv_pos),
                offsets=(0, tile_offset_0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            )
            k_0 = tl.load(k_ptrs_0, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                k_0 = k_0.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k_0 = (k_0 * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale_0 = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * PAGE_SIZE_K + tile_offset_0 + off_n) * stride_ks_t,
                        mask=pos_mask_0,
                        other=1.0,
                    )
                    k_0 = (k_0 * k_scale_0[None, :]).to(q.dtype)
            qk_0 = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk_0 += tl.where(pos_mask_0[None, :], 0, float("-inf"))
            qk_0 += tl.dot(q, k_0) * sm_scale_log2e

            k_ptrs_1 = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(head_dim, PAGE_SIZE_K),
                strides=(stride_kv_d, stride_kv_pos),
                offsets=(0, tile_offset_1),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            )
            k_1 = tl.load(k_ptrs_1, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                k_1 = k_1.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k_1 = (k_1 * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale_1 = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * PAGE_SIZE_K + tile_offset_1 + off_n) * stride_ks_t,
                        mask=pos_mask_1,
                        other=1.0,
                    )
                    k_1 = (k_1 * k_scale_1[None, :]).to(q.dtype)
            qk_1 = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk_1 += tl.where(pos_mask_1[None, :], 0, float("-inf"))
            qk_1 += tl.dot(q, k_1) * sm_scale_log2e

            v_ptrs_0 = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(PAGE_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(tile_offset_0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_0 = tl.load(v_ptrs_0, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                v_0 = v_0.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v_0 = (v_0 * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale_0 = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * PAGE_SIZE_K + tile_offset_0 + off_n) * stride_vs_t,
                        mask=pos_mask_0,
                        other=1.0,
                    )
                    v_0 = (v_0 * v_scale_0[:, None]).to(q.dtype)

            pair_max = tl.maximum(tl.max(qk_0, axis=1), tl.max(qk_1, axis=1))
            m_ij = tl.maximum(m_i, pair_max)
            p_0 = tl.exp2(qk_0 - m_ij[:, None])
            p_1 = tl.exp2(qk_1 - m_ij[:, None])
            l_ij = tl.sum(p_0, axis=1) + tl.sum(p_1, axis=1)
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
            acc_o += tl.dot(p_0.to(v_0.dtype), v_0)

            v_ptrs_1 = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(PAGE_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(tile_offset_1, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v_1 = tl.load(v_ptrs_1, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                v_1 = v_1.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v_1 = (v_1 * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale_1 = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * PAGE_SIZE_K + tile_offset_1 + off_n) * stride_vs_t,
                        mask=pos_mask_1,
                        other=1.0,
                    )
                    v_1 = (v_1 * v_scale_1[:, None]).to(q.dtype)
            acc_o += tl.dot(p_1.to(v_1.dtype), v_1)
            m_i = m_ij
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
_KV_SCALE_NONE = 0
_KV_SCALE_SCALAR = 1
_KV_SCALE_PER_TOKEN_HEAD = 2


def _kv_scale_args(
    output: torch.Tensor,
    num_kv_heads: int,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    if k_scale is None and v_scale is None:
        return output, output, 0, 0, 0, 0, _KV_SCALE_NONE
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale must be both provided or both None")
    if k_scale.device != output.device or v_scale.device != output.device:
        raise ValueError("k_scale and v_scale must be on the same dvice as output")
    if k_scale.numel() == 1 and v_scale.numel() == 1:
        return k_scale, v_scale, 0, 0, 0, 0, _KV_SCALE_SCALAR
    if k_scale.dim() == 2 and v_scale.dim() == 2:
        if k_scale.shape[0] != num_kv_heads or v_scale.shape[0] != num_kv_heads:
            raise ValueError(
                "per-token/head KV scales must have shape "
                f"[{num_kv_heads}, max_kv_tokens]"
            )
        if k_scale.shape != v_scale.shape:
            raise ValueError("k_scale and v_scale must have matching shapes")
        return (
            k_scale,
            v_scale,
            k_scale.stride(0),
            k_scale.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            _KV_SCALE_PER_TOKEN_HEAD,
        )
    raise ValueError(
        "MiniMax-M3 sparse attention supports scalar KV scales or "
        "[num_kv_heads, max_kv_tokens] per-token/head scales"
    )


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )

    kernel_kv_cache = kv_cache
    kernel_use_fp8 = use_fp8
    kernel_kv_scale_mode = kv_scale_mode

    if use_fp8 and kv_scale_mode == _KV_SCALE_SCALAR:
        kernel_kv_cache = kv_cache.to(dtype=q.dtype)

        kernel_kv_cache[..., :head_dim].mul_(k_scale_arg)
        kernel_kv_cache[..., head_dim:].mul_(v_scale_arg)

        kernel_use_fp8 = False
        kernel_kv_scale_mode = _KV_SCALE_NONE
    grid = (max_query_len, num_kv_heads, batch)
    block_size_h = triton.next_power_of_2(gqa_group_size)

    # num_stages=1 keeps shared memory under the 128KiB hardware limit.
    _gqa_sparse_fwd_direct[grid](
        q,
        kernel_kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kernel_kv_cache.stride(0),
        kernel_kv_cache.stride(1),
        kernel_kv_cache.stride(2),
        kernel_kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        PAGE_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_K=_PREFILL_DIRECT_BLOCK_SIZE_K,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_QH=block_size_h,
        USE_FP8=kernel_use_fp8,
        KV_SCALE_MODE=kernel_kv_scale_mode,
        SPLIT_D=(
            head_dim == 128
            and q.dtype == torch.bfloat16
            and kernel_kv_cache.dtype == torch.bfloat16
        ),
        num_warps=2,
        num_stages=1,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
        )
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    target = max(
        1, min(max_topk, _DECODE_TARGET_GRID // max(1, total_q * num_kv_heads))
    )
    num_topk_chunks = 1 << (target.bit_length() - 1)
    # One chunk already finishes online-softmax; write the final tensor and
    # skip the identity merge, matching prefill's single-pass output.
    direct_write = num_topk_chunks == 1
    if direct_write:
        o_buf = output
        stride_o_c = 0
        stride_o_b = output.stride(0)
        stride_o_h = output.stride(1)
        stride_o_d = output.stride(2)
    else:
        o_buf = torch.empty(
            num_topk_chunks,
            total_q,
            num_heads,
            head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        stride_o_c = o_buf.stride(0)
        stride_o_b = o_buf.stride(1)
        stride_o_h = o_buf.stride(2)
        stride_o_d = o_buf.stride(3)
    lse_buf = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    stride_l_c = lse_buf.stride(0)
    stride_l_b = lse_buf.stride(1)
    stride_l_h = lse_buf.stride(2)
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        o_buf,
        lse_buf,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        stride_o_c,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        stride_l_c,
        stride_l_b,
        stride_l_h,
        block_table.stride(0),
        BLOCK_SIZE_K=_DECODE_DIRECT_BLOCK_SIZE_K,
        PAGE_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        num_warps=_DECODE_NUM_WARPS,
        num_stages=1,
    )
    if direct_write:
        return
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_buf,
        lse_buf,
        output,
        head_dim,
        stride_o_c,
        stride_o_b,
        stride_o_h,
        stride_o_d,
        stride_l_c,
        stride_l_b,
        stride_l_h,
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
    )
