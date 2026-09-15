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

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None

from flag_attn.utils import current_platform

# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128

# A 64-token double buffer amortizes its extra softmax/barrier work only for
# the smallest benchmark GQA tile. Larger tiles reuse one full-page KV stage.
_PREFILL_HALF_KV_MAX_BLOCK_SIZE_QH = 8

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_fallback_kernel(
    q_ptr,
    kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
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
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
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
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < seq_len
            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            k_ptrs = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(head_dim, BLOCK_SIZE_K),
                strides=(stride_kv_d, stride_kv_pos),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            )
            k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                k = k.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k = (k * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    k = (k * k_scale[None, :]).to(q.dtype)
            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len
            qk = tl.zeros(
                (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32
            )
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += tl.dot(q, k) * sm_scale_log2e
            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_i = l_i * alpha + l_ij
            v_base_ptr = k_base_ptr + head_dim * stride_kv_d
            v_ptrs = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
            if USE_FP8:
                v = v.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v = (v * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    v = (v * v_scale[:, None]).to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
        acc_o = acc_o * tl.where(l_i > 0, 1.0 / l_i, 0.0)[:, None]
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_kernel(
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
    num_kv_heads,
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
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
):
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
    if USE_FP8 and KV_SCALE_MODE == 1:
        # Scalar scales are invariant across every selected page. Fold K into
        # the logits scale and delay V until the normalized output.
        qk_scale = sm_scale_log2e * tl.load(k_scale_ptr)
        v_scale_scalar = tl.load(v_scale_ptr)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # Valid block count from seq position (no sentinel): block_size_q == 1.
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
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
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        # Keep the tl.dot kernel's PV accumulator in [QH, D] order. The
        # transposed [D, QH] form spills heavily for the FP8 QH=32 tile.
        acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        if USE_FP8:
            # Keep QK in FP8 Tensor Core form.  Q is per-head dynamically
            # scaled once, then its scale is restored on the FP32 logits.
            # This keeps the existing K cache in FP8 through tl.dot instead
            # of materializing a BF16 K tile for every selected page.
            qk_q_scale = tl.maximum(
                tl.max(tl.abs(q), axis=1) * (1.0 / 448.0), 1.0e-8
            )
            q_qk = (q / qk_q_scale[:, None]).to(tl.float8e4nv)
        else:
            q_qk = q
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)

            pos = c + off_n
            pos_mask = pos < seq_len

            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            k_ptrs = tl.make_block_ptr(
                base=k_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            k = tl.load(
                k_ptrs,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
            qk_dot = tl.dot(q_qk, tl.trans(k))
            if USE_FP8:
                qk_dot *= qk_q_scale[:, None]

            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq    = (c + BLOCK_SIZE_K) <= seq_len

            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))

            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            if USE_FP8:
                if KV_SCALE_MODE == 1:
                    qk += qk_dot * qk_scale
                elif KV_SCALE_MODE == 2:
                    # Q @ (K * scale[token]) is equivalent to scaling the
                    # smaller [QH, K] logits instead of the [K, D] K tile.
                    qk += qk_dot * (sm_scale_log2e * k_scale[None, :])
                else:
                    qk += qk_dot * sm_scale_log2e
            else:
                qk += qk_dot * sm_scale_log2e

            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            l_i = l_i * alpha + l_ij

            v_base_ptr = k_base_ptr + head_dim * stride_kv_d
            v_ptrs = tl.make_block_ptr(
                base=v_base_ptr,
                shape=(BLOCK_SIZE_K, head_dim),
                strides=(stride_kv_pos, stride_kv_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            v = tl.load(
                v_ptrs,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )

            if USE_FP8 and BLOCK_SIZE_H < 32:
                # The 8/16-head tiles lower efficiently as FP8 PV MMAs.
                # Fold a per-token V scale into P before quantizing P
                # row-wise, and keep V in its FP8 cache format.
                pv_p = p
                if KV_SCALE_MODE == 2:
                    pv_p *= v_scale[None, :]
                pv_p_scale = tl.maximum(
                    tl.max(tl.abs(pv_p), axis=1) * (1.0 / 448.0), 1.0e-8
                )
                p_dot = (pv_p / pv_p_scale[:, None]).to(tl.float8e4nv)
                acc_o += tl.dot(p_dot, v) * pv_p_scale[:, None]
            else:
                # QH=32 uses the native PV accumulator layout above, but its
                # FP8 PV lowering spills heavily; retain BF16 operands there.
                if USE_FP8:
                    v = v.to(q.dtype)
                p_dot = p.to(v.dtype)
                if USE_FP8 and KV_SCALE_MODE == 2:
                    p_dot = (p * v_scale[None, :]).to(v.dtype)
                acc_o += tl.dot(p_dot, v)
            m_i = m_ij

        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        if USE_FP8 and KV_SCALE_MODE == 1:
            inv_l *= v_scale_scalar
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


@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"]
        * triton.next_power_of_2(args["gqa_group_size"]),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,  # flattened [num_blocks*num_kv_heads*128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
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
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    USE_TL_DOT_PATH: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
    USE_HALF_KV_PIPE: tl.constexpr,
):
    if USE_TL_DOT_PATH:
        _gqa_sparse_fwd_kernel(
            q_ptr,
            kv_cache_ptr,
            k_scale_ptr,
            v_scale_ptr,
            t_ptr,
            o_ptr,
            block_table_ptr,
            cu_seqlens_q,
            cu_seqblocks_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
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
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            BLOCK_SIZE_D,
            BLOCK_SIZE_H,
            BLOCK_SIZE_QH,
            USE_FP8,
            KV_SCALE_MODE,
        )
        return

    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    q_tile_start = pid_q * BLOCK_SIZE_Q
    if q_tile_start >= q_block_len:
        return

    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    q_abs = prefix_len + q_tile_start + off_q
    real_topk = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(q_tile_start, 0, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Four warps form one Hopper warpgroup. Q is staged once and reused by all
    # selected pages as the transposed WGMMA B operand.
    q_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_QH, BLOCK_SIZE_D],
        dtype=q_ptr.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q)
    tl.debug_barrier()

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o_t = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_QH), dtype=tl.float32)

    loop_blocks = tl.max(real_topk, axis=0)
    topk_ptr = t_ptr + pid_kh * stride_th + (q_block_start + q_tile_start) * stride_tn

    if USE_HALF_KV_PIPE:
        HALF_K: tl.constexpr = BLOCK_SIZE_K // 2
        off_half = tl.arange(0, HALF_K)
        causal_offsets = q_abs[:, None] - off_half[None, :]
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, HALF_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_smem = tle.gpu.alloc(
            [2, HALF_K, BLOCK_SIZE_D],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_stage_bytes: tl.constexpr = HALF_K * BLOCK_SIZE_D * 2
        kv_empty = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=1,
            init=tle.gpu.READY,
        )
        kv_full = tle.gpu.alloc_barriers(
            num_barriers=2,
            arrive_count=1,
            expect_bytes=kv_stage_bytes,
        )
        loop_tiles = loop_blocks * 2
        # The K prefetch already resolves the next tile's logical block and
        # physical page. Keep these two scalar addresses live across the loop
        # so the V copy does not reload topk_idx and block_table for that tile.
        cur_kv_row = tl.full((), 0, dtype=tl.int32)
        cur_c = tl.full((), 0, dtype=tl.int32)

        # Prologue: stage the first K half before entering the ping-pong loop.
        if loop_tiles > 0:
            first_blk = tl.load(topk_ptr).to(tl.int32)
            first_page = tl.load(bt_row + first_blk).to(tl.int32)
            first_row = (first_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
            cur_kv_row = first_row
            cur_c = first_blk * BLOCK_SIZE_K
            tle.gpu.barrier_wait(kv_empty[0], phaseIdx=0)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(0),
                [HALF_K, BLOCK_SIZE_D],
                [first_row, 0],
                barrier=kv_full[0],
            )

        for tile_iter in tl.range(loop_tiles, disable_licm=True, num_stages=1):
            block_iter = tile_iter // 2
            # half_idx = tile_iter % 2
            buf_idx = tile_iter % 2
            reuse_iter = tile_iter // 2
            k_phase = reuse_iter * 2
            v_phase = k_phase + 1

            kv_row = cur_kv_row
            c = cur_c
            pos = c + off_half
            pos_mask = pos < seq_len

            tle.gpu.barrier_wait(kv_full[buf_idx], phaseIdx=k_phase)
            qk_t = tle.gpu.wgmma(
                kv_smem.slot(buf_idx),
                q_smem.slot(0),
                out_dtype=tl.float32,
                trans_b=True,
            )
            qk_t = tle.gpu.wgmma_wait(0, qk_t)
            qk_t *= sm_scale_log2e
            qk = tl.reshape(
                tl.trans(qk_t),
                (BLOCK_SIZE_Q, BLOCK_SIZE_H, HALF_K),
            )

            # Reuse this slot for V while the other slot receives the next K.
            tle.gpu.barrier_arrive(kv_empty[buf_idx], phaseIdx=k_phase)
            tle.gpu.barrier_wait(kv_empty[buf_idx], phaseIdx=v_phase)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(buf_idx),
                [HALF_K, BLOCK_SIZE_D],
                [kv_row, BLOCK_SIZE_D],
                barrier=kv_full[buf_idx],
            )

            if tile_iter + 1 < loop_tiles:
                next_tile = tile_iter + 1
                next_block_iter = next_tile // 2
                next_half_idx = next_tile % 2
                next_buf_idx = next_tile % 2
                next_reuse_iter = next_tile // 2
                next_k_phase = next_reuse_iter * 2
                next_blk = tl.load(
                    topk_ptr + next_block_iter * stride_tk
                ).to(tl.int32)
                next_page = tl.load(bt_row + next_blk).to(tl.int32)
                next_kv_row = (
                    next_page * num_kv_heads + pid_kh
                ) * BLOCK_SIZE_K + next_half_idx * HALF_K
                next_c = next_blk * BLOCK_SIZE_K + next_half_idx * HALF_K
                tle.gpu.barrier_wait(
                    kv_empty[next_buf_idx], phaseIdx=next_k_phase
                )
                tle.gpu.copy(
                    kv_cache_desc,
                    kv_smem.slot(next_buf_idx),
                    [HALF_K, BLOCK_SIZE_D],
                    [next_kv_row, 0],
                    barrier=kv_full[next_buf_idx],
                )
                cur_kv_row = next_kv_row
                cur_c = next_c

            if (c + HALF_K) > (prefix_len + q_tile_start):
                qk += tl.where(causal_offsets[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, HALF_K)
            if (c + HALF_K) > seq_len:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc_o_t *= alpha[None, :]
            tl.store(
                tle.gpu.local_ptr(p_smem.slot(0)),
                p.to(kv_cache_ptr.dtype.element_ty),
            )

            tle.gpu.barrier_wait(kv_full[buf_idx], phaseIdx=v_phase)
            tl.debug_barrier()
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(buf_idx),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
                trans_b=True,
            )
            acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)
            tle.gpu.barrier_arrive(kv_empty[buf_idx], phaseIdx=v_phase)

            l_i = tl.math.fma(l_i, alpha, l_ij)
            m_i = m_ij
    else:
        causal_offsets = q_abs[:, None] - off_n[None, :]
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, BLOCK_SIZE_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        kv_stage_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D * 2
        kv_empty = tle.gpu.alloc_barriers(
            num_barriers=1,
            arrive_count=1,
            init=tle.gpu.READY,
        )
        kv_full = tle.gpu.alloc_barriers(
            num_barriers=1,
            arrive_count=1,
            expect_bytes=kv_stage_bytes,
        )
        # Resolve page(0) before the loop. Each iteration consumes the carried
        # row/offset, then resolves page(i + 1) while V(i) is in flight.
        cur_kv_row = tl.full((), 0, dtype=tl.int32)
        cur_c = tl.full((), 0, dtype=tl.int32)
        if loop_blocks > 0:
            first_blk = tl.load(topk_ptr).to(tl.int32)
            first_page = tl.load(bt_row + first_blk).to(tl.int32)
            cur_kv_row = (
                first_page * num_kv_heads + pid_kh
            ) * BLOCK_SIZE_K
            cur_c = first_blk * BLOCK_SIZE_K

        for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
            k_phase = block_iter * 2
            v_phase = k_phase + 1
            kv_row = cur_kv_row
            c = cur_c
            pos = c + off_n
            pos_mask = pos < seq_len

            tle.gpu.barrier_wait(kv_empty[0], phaseIdx=k_phase)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(0),
                [BLOCK_SIZE_K, BLOCK_SIZE_D],
                [kv_row, 0],
                barrier=kv_full[0],
            )
            tle.gpu.barrier_wait(kv_full[0], phaseIdx=k_phase)

            qk_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                q_smem.slot(0),
                out_dtype=tl.float32,
                trans_b=True,
            )
            qk_t = tle.gpu.wgmma_wait(0, qk_t)
            qk_t *= sm_scale_log2e
            qk = tl.reshape(
                tl.trans(qk_t),
                (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K),
            )

            tle.gpu.barrier_arrive(kv_empty[0], phaseIdx=k_phase)
            tle.gpu.barrier_wait(kv_empty[0], phaseIdx=v_phase)
            tle.gpu.copy(
                kv_cache_desc,
                kv_smem.slot(0),
                [BLOCK_SIZE_K, BLOCK_SIZE_D],
                [kv_row, BLOCK_SIZE_D],
                barrier=kv_full[0],
            )

            if block_iter + 1 < loop_blocks:
                next_blk = tl.load(
                    topk_ptr + (block_iter + 1) * stride_tk
                ).to(tl.int32)
                next_page = tl.load(bt_row + next_blk).to(tl.int32)
                cur_kv_row = (
                    next_page * num_kv_heads + pid_kh
                ) * BLOCK_SIZE_K
                cur_c = next_blk * BLOCK_SIZE_K

            if (c + BLOCK_SIZE_K) > (prefix_len + q_tile_start):
                qk += tl.where(causal_offsets[:, None, :] >= c, 0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            if (c + BLOCK_SIZE_K) > seq_len:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc_o_t *= alpha[None, :]
            tl.store(
                tle.gpu.local_ptr(p_smem.slot(0)),
                p.to(kv_cache_ptr.dtype.element_ty),
            )

            tle.gpu.barrier_wait(kv_full[0], phaseIdx=v_phase)
            tl.debug_barrier()
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
                trans_b=True,
            )
            acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)
            tle.gpu.barrier_arrive(kv_empty[0], phaseIdx=v_phase)

            l_i = tl.math.fma(l_i, alpha, l_ij)
            m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o_t *= inv_l[None, :]
    acc_o = tl.trans(acc_o_t)
    acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(q_tile_start, 0, 0),
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
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
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
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    USE_PDL: tl.constexpr,
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

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, blk)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
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
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        k = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[None, :] * stride_kv_pos
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                k_scale = tl.load(
                    k_scale_ptr
                    + pid_kh * stride_ks_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                    mask=pos_mask,
                    other=1.0,
                )
                k = (k * k_scale[None, :]).to(q.dtype)
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_cache_ptr
            + page * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                v_scale = tl.load(
                    v_scale_ptr
                    + pid_kh * stride_vs_h
                    + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask,
                    other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

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
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

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
    grid = (max_query_len, num_kv_heads, batch)
    block_size_h = triton.next_power_of_2(gqa_group_size)

    if not use_fp8 and block_size_h >= 8 and tle is None:
        _gqa_sparse_fwd_fallback_kernel[grid](
            q,
            kv_cache,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            output,
            block_table,
            cu_seqlens_q,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            topk,
            1,
            sm_scale,
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
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=1,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_QH=block_size_h,
            USE_FP8=False,
            KV_SCALE_MODE=_KV_SCALE_NONE,
        )
        return

    # Keep cases that cannot form a legal WGMMA result out of the TLE kernel:
    # FP8 KV has a dtype mismatch with BF16 Q, and a GQA tile below eight heads
    # has an illegal WGMMA N dimension.
    if use_fp8 or block_size_h < 8:
        _gqa_sparse_fwd_kernel[grid](
            q,
            kv_cache,
            k_scale_arg,
            v_scale_arg,
            topk_idx,
            output,
            block_table,
            cu_seqlens_q,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            topk,
            1,  # num_q_loop
            sm_scale,
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
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=1,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
            BLOCK_SIZE_H=block_size_h,
            BLOCK_SIZE_QH=block_size_h,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            num_warps=4,
            num_stages=3,
        )
        return

    from triton.tools.tensor_descriptor import TensorDescriptor

    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=kv_cache.device)

    triton.set_allocator(alloc_fn)
    use_tl_dot_path = False
    use_half_kv_pipe = (
        not use_tl_dot_path
        and block_size_h <= _PREFILL_HALF_KV_MAX_BLOCK_SIZE_QH
    )
    kv_tma_rows = SPARSE_BLOCK_SIZE // 2 if use_half_kv_pipe else SPARSE_BLOCK_SIZE
    kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
    kv_cache_desc = TensorDescriptor(
        kv_cache_2d,
        shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
        strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
        block_shape=[kv_tma_rows, head_dim],
    )
    _gqa_sparse_fwd_tle_kernel[grid](
        q,
        kv_cache,
        kv_cache_desc,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
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
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=1,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        USE_TL_DOT_PATH=use_tl_dot_path,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        USE_HALF_KV_PIPE=use_half_kv_pipe,
        num_warps=4,
        num_stages=3 if use_tl_dot_path else 1,
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
    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    o_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device
    )
    lse_partial = torch.empty(
        num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
    )
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        o_partial,
        lse_partial,
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
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
