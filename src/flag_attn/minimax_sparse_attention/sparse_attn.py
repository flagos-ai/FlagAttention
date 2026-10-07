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

# P = total_q * num_kv_heads. Inside this band a 3*SM grid target
# collapses the split-K chunk count to 1, which enables the
# merge-skip; outside it the fixed 256 target is better.
_MSA_DECODE_GRID_BAND_LO = 96
_MSA_DECODE_GRID_BAND_HI = 160
_MSA_DECODE_TARGET_GRID = 256

# Below this P the bf16-specialized decode kernel's higher occupancy does not
# pay (measured 0.97-1.00x at P = 128..256 vs 1.09-1.16x at P >= 512).
_MSA_DECODE_FUSED_MIN_PARALLELISM = 512

# The TLE prefill kernel keeps P in shared memory as
# [BLOCK_SIZE_K, BLOCK_SIZE_QH] so the QK result never has to be transposed.
# That layout needs BLOCK_SIZE_QH >= 16; at 8 the transposed PV operand view
# runs out of bounds. Smaller GQA tiles go to the tl.dot kernel instead.
_MSA_PREFILL_UNTRANSPOSED_MIN_QH = 16

# Register cap for the fp8 TLE prefill kernel. Staging K in smem frees
# 64 regs/thread, but Triton will not spend that budget on occupancy unless
# told to: without a cap the kernel still compiles to 255 registers and gains
# nothing (1.00x). On H20 (78 SM) the curve was single-peaked at 128.
#
# On H100 (132 SM) the right value depends on the shape, and 128 is the better
# single choice -- it was re-verified here, not inherited.
#
# At GQA 24 with a long chunk, 128 does not hold the tile and spills to DRAM
# (local-load sectors 5.51e9 vs 1.85e9 at 168; dram__bytes 50.91 GB vs 3.40 GB),
# which costs that one shape 0.915x of upstream; 168 turns it into 1.144x.
# But GQA 16 at short chunks goes the other way, and far more sharply. Measured
# against upstream on an idle card, n=4 interleaved, launch-verified
# (total_q = batch * query_len):
#   shape          total_q   maxnreg=168   maxnreg=128
#   1x128x4096         128        0.900x        1.052x
#   1x128x16384        128        0.885x        1.040x
#   1x128x65536        128        0.816x        1.025x
#   4x128x4096         512        1.068x        1.113x
# 128 is the only value that beats upstream at total_q = 128, where low
# occupancy cannot be hidden; none/192/232 all lose there too. Over the 21-shape
# GQA-16 prefill set, 128 keeps every shape at or above upstream while 168
# regresses three of them by up to 18.6%.
#
# So this stays 128. A shape-conditional cap (168 only for QH >= 32, which is
# where the spill actually happens) would capture both, but that needs a
# conditional at the launch site rather than a constant.
_MSA_PREFILL_FP8_MAXNREG = 128

_SM_COUNT_CACHE: dict[int, int] = {}
_PDL_SUPPORTED: bool | None = None


def _pdl_supported() -> bool:
    """Whether this platform supports programmatic dependent launch.

    Measured at 4.7 us per call on this box, against a decode kernel of
    17-25 us at small batch -- so querying it per launch is a real cost, not a
    rounding error. The answer is fixed for the process.
    """
    global _PDL_SUPPORTED
    if _PDL_SUPPORTED is None:
        _PDL_SUPPORTED = current_platform.is_arch_support_pdl()
    return _PDL_SUPPORTED


def _sm_count(device) -> int:
    """SM count, cached per device.

    `torch.cuda.get_device_properties` is not free, and the decode wrapper runs
    it on every call while the decode kernel itself is only ~37 us at small
    batch -- so the uncached query showed up as a ~0.9x regression on this
    repo's small-batch benchmark shapes.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _SM_COUNT_CACHE.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = count
    return count

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


# ---------------------------------------------------------------------------
# Optimized kernels (branch: sparseattention-opt)
#
# Measured on H20 vs the upstream kernels in this file, median over the
# benchmark shape sets: prefill 1.132x (21 shapes, 0 regressions, all
# bit-exact), decode 1.126x for P>64 (9 shapes, CUDA graph). Accuracy
# against an fp64 reference is at the bf16 output-rounding floor
# (2.1-2.9e-03 rel_rmse) for both upstream and these kernels.
#
# Upstream's kernels are all retained and still dispatched to: the
# fallback and tl.dot prefill kernels (fp8, BLOCK_SIZE_QH < 16, no TLE)
# and `_gqa_sparse_decode_kernel` (general decode).
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: triton.next_power_of_2(
            args["gqa_group_size"]
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,  # TMA view: [num_blocks*num_kv_heads*128, 2*head_dim]
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
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,  # == BLOCK_SIZE_H, since BLOCK_SIZE_Q == 1
    KEEP_TRANS: tl.constexpr,
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
    if pid_q >= q_block_len:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    # BLOCK_SIZE_Q == 1: one query token per program, so the valid block count
    # is a scalar rather than a 1-element tensor reduced with tl.max.
    q_abs = prefix_len + pid_q
    loop_blocks = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    causal_offsets = q_abs - off_n

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Four warps form one Hopper warpgroup. Q is staged once and reused by every
    # selected page as the transposed WGMMA B operand.
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

    topk_ptr = t_ptr + pid_kh * stride_th + (q_block_start + pid_q) * stride_tn

    # p_smem normally holds P as [BLOCK_SIZE_K, BLOCK_SIZE_QH] -- the layout
    # WGMMA already produces -- so the QK result never has to be transposed.
    # ncu attributed 99.5% of this kernel's shared-bank conflicts to that one
    # store, and removing the transpose cut instructions by 14%.
    #
    # KEEP_TRANS restores the transposed [QH, K] form for BLOCK_SIZE_QH == 8,
    # where `wgmma(v, p, acc, trans_a=True)` silently computes a WRONG result
    # (an N=8 transposed-A WGMMA returned 96.0 for an exact-128.0 reduction;
    # N>=16 is correct). Everything else in this kernel still applies there.
    if KEEP_TRANS:
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_QH, BLOCK_SIZE_K],
            dtype=kv_cache_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
    else:
        p_smem = tle.gpu.alloc(
            [1, BLOCK_SIZE_K, BLOCK_SIZE_QH],
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
    # Only the RAW barrier survives. Upstream also allocates a `kv_empty` WAR
    # barrier and pays 4 extra sync ops per page on it; `wgmma_wait(0)` below
    # already drains each operand read, so the slot is provably free without it.
    kv_full = tle.gpu.alloc_barriers(
        num_barriers=1,
        arrive_count=1,
        expect_bytes=kv_stage_bytes,
    )

    # Resolve page(0) before the loop; each iteration consumes the carried
    # row/offset and resolves page(i + 1) while V(i) is in flight.
    cur_kv_row = tl.full((), 0, dtype=tl.int32)
    cur_c = tl.full((), 0, dtype=tl.int32)
    if loop_blocks > 0:
        first_blk = tl.load(topk_ptr).to(tl.int32)
        first_page = tl.load(bt_row + first_blk).to(tl.int32)
        cur_kv_row = (first_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
        cur_c = first_blk * BLOCK_SIZE_K

    for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
        k_phase = block_iter * 2
        v_phase = k_phase + 1
        kv_row = cur_kv_row
        c = cur_c
        pos = c + off_n
        pos_mask = pos < seq_len

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
        # Drains the QK WGMMA: K's smem read has completed, so the V copy below
        # cannot clobber a live operand and needs no WAR handshake.
        qk_t = tle.gpu.wgmma_wait(0, qk_t)
        qk_t *= sm_scale_log2e

        tle.gpu.copy(
            kv_cache_desc,
            kv_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [kv_row, BLOCK_SIZE_D],
            barrier=kv_full[0],
        )

        if block_iter + 1 < loop_blocks:
            next_blk = tl.load(topk_ptr + (block_iter + 1) * stride_tk).to(tl.int32)
            next_page = tl.load(bt_row + next_blk).to(tl.int32)
            cur_kv_row = (next_page * num_kv_heads + pid_kh) * BLOCK_SIZE_K
            cur_c = next_blk * BLOCK_SIZE_K

        # qk_t is [BLOCK_SIZE_K, BLOCK_SIZE_QH] straight out of the WGMMA.
        # Keeping that native layout and reducing along axis 0 (the KV axis)
        # removes the transpose and the register shuffle its smem store
        # implied. Both masks are per-block scalar-guarded, so the vector
        # select is only materialized for the diagonal / tail page.
        if KEEP_TRANS:
            qk = tl.reshape(tl.trans(qk_t), (BLOCK_SIZE_QH, BLOCK_SIZE_K))
            if (c + BLOCK_SIZE_K) > q_abs:
                qk += tl.where(causal_offsets[None, :] >= c, 0, float("-inf"))
            if (c + BLOCK_SIZE_K) > seq_len:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
        else:
            qk = qk_t
            if (c + BLOCK_SIZE_K) > q_abs:
                qk += tl.where(causal_offsets[:, None] >= c, 0, float("-inf"))
            if (c + BLOCK_SIZE_K) > seq_len:
                qk += tl.where(pos_mask[:, None], 0, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp2(qk - m_ij[None, :])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=0)
        acc_o_t *= alpha[None, :]
        tl.store(
            tle.gpu.local_ptr(p_smem.slot(0)),
            p.to(kv_cache_ptr.dtype.element_ty),
        )

        tle.gpu.barrier_wait(kv_full[0], phaseIdx=v_phase)
        tl.debug_barrier()
        # acc[D, QH] = V^T @ P, with V=[K, D] in kv_smem. In the default
        # layout p_smem is already K-major so trans_b is unnecessary; under
        # KEEP_TRANS it is [QH, K] and needs transposing, as upstream does.
        if KEEP_TRANS:
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
                trans_b=True,
            )
        else:
            acc_o_t = tle.gpu.wgmma(
                kv_smem.slot(0),
                p_smem.slot(0),
                acc_o_t,
                trans_a=True,
            )
        acc_o_t = tle.gpu.wgmma_wait(0, acc_o_t)

        l_i = tl.math.fma(l_i, alpha, l_ij)
        m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    acc_o_t *= inv_l[None, :]
    acc_o = tl.trans(acc_o_t)
    acc_o = tl.reshape(acc_o, 1, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
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


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_fused_kernel(
    q_ptr,
    kv_cache_ptr,
    t_ptr,
    o_ptr,
    lse_ptr,
    block_table_ptr,
    seq_lens,
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
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
    SINGLE_CHUNK: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
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
    # Bound by the constexpr BLOCK_SIZE_H rather than the runtime
    # gqa_group_size so Triton can prove the tile is fully in range and drop
    # the boundary masks (registers 213 -> 168).
    #
    # REQUIRES gqa_group_size == BLOCK_SIZE_H. Since
    # BLOCK_SIZE_H = max(16, next_pow2(gqa_group_size)), any gqa_group_size
    # below 16 makes BLOCK_SIZE_H larger than the number of heads that actually
    # exist, and these block pointers then run past this kv_head into the next
    # one -- measured max_diff 7.8e-02 at (nkv=8, gqa=8). The wrapper enforces
    # the equality before selecting this kernel; do not relax it.
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(BLOCK_SIZE_H, head_dim),
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
        # One base address shared by the K and V loads.
        kv_base = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        k = tl.load(
            kv_base
            + off_n[None, :] * stride_kv_pos
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        )
        qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(pos_mask[None, :], 0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            kv_base
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i),
                     tl.zeros_like(lse_i))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b
        + pid_h * stride_o_h,
        shape=(BLOCK_SIZE_H, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    # With a single chunk there is no merge to consume lse: the store is dead.
    if not SINGLE_CHUNK:
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr
            + pid_c * stride_l_c
            + pid_b * stride_l_b
            + pid_h * stride_l_h,
            shape=(BLOCK_SIZE_H,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty),
                 boundary_check=(0,))


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_mergeskip_kernel(
    q_ptr,
    kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,
    o_ptr,
    lse_ptr,
    block_table_ptr,
    seq_lens,
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
    BLOCK_SIZE_K: tl.constexpr,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,
    KV_SCALE_MODE: tl.constexpr,
    USE_PDL: tl.constexpr,
    SINGLE_CHUNK: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
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
    # Full-CG padding uses zero-length request rows. Clamp to an empty attention
    # range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

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
    # With a single chunk there is no merge to consume lse: the store is dead.
    if not SINGLE_CHUNK:
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr
            + pid_c * stride_l_c
            + pid_b * stride_l_b
            + pid_h * stride_l_h,
            shape=(gqa_group_size,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


def _tma_allocator(size: int, align: int, stream):
    _ = align
    _ = stream
    return torch.empty(size, dtype=torch.int8, device="cuda")


# ---------------------------------------------------------------------------
# fp8 prefill on the TLE path. Upstream sends all fp8 to the tl.dot kernel,
# which is 2.16x SLOWER than its own bf16 path while every memory channel sits
# idle (DRAM 0.4%, L2 13.8%): ncu puts the cost in 255 registers + 6 spills and
# 22% math_pipe_throttle, not in bandwidth. Meanwhile a direct probe puts the
# fp8 WGMMA issue ceiling at 252.7 TFLOPS against bf16's 133.5.
#
# This kernel stages K through TMA and feeds it to WGMMA still in fp8, so the
# [K, D] K tile is never materialized in registers -- that frees 64
# regs/thread. The freed space only pays off under an explicit maxnreg cap:
# measured at (1,4096,4096), upstream is 4.388 ms, upstream+maxnreg=128 is
# 4.216 ms (1.03x, the cap alone achieves almost nothing), and this kernel with
# maxnreg=128 is 3.213 ms -- so the two are complementary, not alternatives.
#
# maxnreg=128 is a single-peaked optimum, verified across five shapes:
#   none 1.00x | 168 1.26x | 144 1.26x | 128 1.33x | 120 1.27x | 96 1.06x
#   | 80 0.84x | 64 0.71x
#
# PV stays on tl.dot: it contracts over K, so an fp8 WGMMA would need V as
# [D, K], and TMA cannot read the cache's [K, D] V transposed -- a descriptor
# with swapped strides is rejected ("strides must be 16-byte aligned", fp8's
# innermost stride being 1 byte).
#
# Scalar KV scales only (KV_SCALE_MODE == 1); other modes keep the tl.dot path.
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(args["gqa_group_size"]),
        "BLOCK_SIZE_QH": lambda args: triton.next_power_of_2(
            args["gqa_group_size"]
        ),
    }
)
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_tle_fp8_kernel(
    q_ptr,  # [total_q, num_heads, head_dim] bf16
    kv_cache_ptr,  # [num_blocks, num_kv_heads, 128, 2*head_dim] fp8
    k_desc,  # TMA view of the K half, fp8
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,
    o_ptr,
    block_table_ptr,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    if pid_q >= q_len:
        return
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    q_abs = prefix_len + pid_q
    loop_blocks = tl.minimum(max_topk, (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K)
    causal_offsets = q_abs - off_n

    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_qn, stride_qh, stride_qd),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
    q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)

    # Scalar KV scales are invariant across pages: fold K's into the logits
    # scale and defer V's to the normalized output, exactly as upstream does.
    qk_scale = sm_scale_log2e * tl.load(k_scale_ptr)
    v_scale_scalar = tl.load(v_scale_ptr)

    # Q is quantized ONCE, outside the loop, per head. Its scale is restored on
    # the fp32 logits, so no accuracy is traded for the fp8 operand.
    q_absmax = tl.max(tl.abs(q), axis=1)
    q_scale = tl.maximum(q_absmax * (1.0 / 448.0), 1.0e-8)
    q_fp8 = (q / q_scale[:, None]).to(tl.float8e4nv)

    # Staged in smem as [QH, D]: D innermost, which is what the K-major-only
    # fp8 WGMMA requires of its B operand under trans_b.
    q_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_QH, BLOCK_SIZE_D],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
    )
    tl.store(tle.gpu.local_ptr(q_smem.slot(0)), q_fp8)
    tl.debug_barrier()

    # K staged by TMA, kept in fp8 -- never materialized as a register tile.
    # This is the change that is supposed to drop the register count.
    k_smem = tle.gpu.alloc(
        [1, BLOCK_SIZE_K, BLOCK_SIZE_D],
        dtype=tl.float8e4nv,
        layout=None,
        scope=tle.gpu.smem,
    )
    k_bytes: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_D  # fp8: 1 byte/elem
    k_full = tle.gpu.alloc_barriers(
        num_barriers=1, arrive_count=1, expect_bytes=k_bytes
    )

    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)

    topk_ptr = t_ptr + pid_kh * stride_th + (q_start + pid_q) * stride_tn

    cur_page = tl.full((), 0, dtype=tl.int32)
    cur_c = tl.full((), 0, dtype=tl.int32)
    if loop_blocks > 0:
        b0 = tl.load(topk_ptr).to(tl.int32)
        cur_page = tl.load(bt_row + b0).to(tl.int32)
        cur_c = b0 * BLOCK_SIZE_K

    for block_iter in tl.range(loop_blocks, disable_licm=True, num_stages=1):
        page = cur_page
        c = cur_c
        pos = c + off_n
        pos_mask = pos < seq_len
        k_row = (page * num_kv_heads + pid_kh) * BLOCK_SIZE_K

        tle.gpu.copy(
            k_desc,
            k_smem.slot(0),
            [BLOCK_SIZE_K, BLOCK_SIZE_D],
            [k_row, 0],
            barrier=k_full[0],
        )
        tle.gpu.barrier_wait(k_full[0], phaseIdx=block_iter)

        # acc[K, QH] = K[K, D] @ Q[QH, D]^T -- both fp8, both D-innermost.
        qk_t = tle.gpu.wgmma(
            k_smem.slot(0),
            q_smem.slot(0),
            out_dtype=tl.float32,
            trans_b=True,
        )
        qk_t = tle.gpu.wgmma_wait(0, qk_t)

        # Resolve the next page while the WGMMA result is still being consumed.
        if block_iter + 1 < loop_blocks:
            nb = tl.load(topk_ptr + (block_iter + 1) * stride_tk).to(tl.int32)
            cur_page = tl.load(bt_row + nb).to(tl.int32)
            cur_c = nb * BLOCK_SIZE_K

        # Restore Q's quantization scale on the fp32 logits and fold in both
        # the softmax base-2 factor and the scalar K scale.
        qk = tl.trans(qk_t) * (q_scale[:, None] * qk_scale)

        if (c + BLOCK_SIZE_K) > q_abs:
            qk += tl.where(causal_offsets[None, :] >= c, 0, float("-inf"))
        if (c + BLOCK_SIZE_K) > seq_len:
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc_o *= alpha[:, None]

        # PV stays on tl.dot: it contracts over K, so fp8 WGMMA would need V as
        # [D, K] and TMA cannot produce that (see the module docstring). V is
        # loaded to registers and dequantized to Q's dtype, as upstream does.
        v = tl.load(
            kv_cache_ptr
            + page.to(tl.int64) * stride_kv_blk
            + pid_kh * stride_kv_h
            + off_n[:, None] * stride_kv_pos
            + (head_dim + off_d[None, :]) * stride_kv_d,
            mask=pos_mask[:, None],
            other=0.0,
        ).to(q_ptr.dtype.element_ty)
        acc_o += tl.dot(p.to(v.dtype), v)

        l_i = tl.math.fma(l_i, alpha, l_ij)
        m_i = m_ij

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0) * v_scale_scalar
    acc_o *= inv_l[:, None]
    acc_o = tl.reshape(acc_o, 1, BLOCK_SIZE_H, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + q_start * stride_on + pid_h * stride_oh,
        shape=(q_len, gqa_group_size, head_dim),
        strides=(stride_on, stride_oh, stride_od),
        offsets=(pid_q, 0, 0),
        block_shape=(1, BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(2, 1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


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
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
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
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    grid = (max_query_len, num_kv_heads, batch)
    block_size_h = triton.next_power_of_2(gqa_group_size)

    # Dispatch order mirrors upstream: no-TLE fallback, then the tl.dot kernel
    # for cases that cannot form a legal WGMMA (fp8 KV has a dtype mismatch with
    # BF16 Q; a GQA tile below eight heads has an illegal WGMMA N), then TLE.
    if not use_fp8 and block_size_h >= 8 and tle is None:
        _gqa_sparse_fwd_fallback_kernel[grid](
            q, kv_cache, k_scale_arg, v_scale_arg, topk_idx, output,
            block_table, cu_seqlens_q, cu_seqlens_q, seq_lens, prefix_lens,
            num_kv_heads, gqa_group_size, head_dim, topk, 1, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
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

    # Route to the tl.dot kernel for fp8 (dtype mismatch against BF16 Q) and
    # for GQA tiles the TLE kernel cannot form a legal WGMMA for. Upstream's
    # bound is block_size_h < 8; this fork needs < 16 because storing P as
    # [BLOCK_SIZE_K, BLOCK_SIZE_QH] and issuing PV as
    # `wgmma(kv, p, acc, trans_a=True)` reads out of bounds at QH == 8
    # (measured: illegal memory access at gqa 6/7/8, exact at gqa >= 10).
    # fp8 with scalar scales goes to the TLE kernel; the tl.dot path remains
    # for per-token scales (not yet ported) and for GQA tiles too small to form
    # a legal WGMMA.
    use_fp8_tle = (
        use_fp8
        and tle is not None
        and kv_scale_mode == _KV_SCALE_SCALAR
        and block_size_h >= _MSA_PREFILL_UNTRANSPOSED_MIN_QH
    )
    if use_fp8_tle:
        from triton.tools.tensor_descriptor import TensorDescriptor

        triton.set_allocator(_tma_allocator)
        kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
        k_desc = TensorDescriptor(
            kv_cache_2d,
            shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
            strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
            block_shape=[SPARSE_BLOCK_SIZE, head_dim],
        )
        _gqa_sparse_fwd_tle_fp8_kernel[grid](
            q, kv_cache, k_desc, k_scale_arg, v_scale_arg, topk_idx, output,
            block_table, cu_seqlens_q, seq_lens, prefix_lens,
            num_kv_heads, gqa_group_size, head_dim, topk, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            kv_cache.stride(3),
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
            maxnreg=_MSA_PREFILL_FP8_MAXNREG,
        )
        return

    if use_fp8 or block_size_h < 8:
        _gqa_sparse_fwd_kernel[grid](
            q, kv_cache, k_scale_arg, v_scale_arg, topk_idx, output,
            block_table, cu_seqlens_q, cu_seqlens_q, seq_lens, prefix_lens,
            num_kv_heads, gqa_group_size, head_dim, topk, 1, sm_scale,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
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

    triton.set_allocator(_tma_allocator)
    kv_cache_2d = kv_cache.view(-1, 2 * head_dim)
    kv_cache_desc = TensorDescriptor(
        kv_cache_2d,
        shape=[kv_cache_2d.shape[0], kv_cache_2d.shape[1]],
        strides=[kv_cache_2d.stride(0), kv_cache_2d.stride(1)],
        block_shape=[SPARSE_BLOCK_SIZE, head_dim],
    )
    _gqa_sparse_fwd_tle_kernel[grid](
        q, kv_cache, kv_cache_desc, topk_idx, output, block_table,
        cu_seqlens_q, cu_seqlens_q, seq_lens, prefix_lens,
        num_kv_heads, gqa_group_size, head_dim, topk, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        KEEP_TRANS=block_size_h < _MSA_PREFILL_UNTRANSPOSED_MIN_QH,
        num_warps=4,
        num_stages=1,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
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
        else (output, output, 0, 0, 0, 0, _KV_SCALE_NONE)
    )
    use_pdl = _pdl_supported()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}

    # split-K over the selected blocks; chunk count is shape-constant (cuda
    # graph). 3*SM pushes a narrow band down to chunks == 1, which activates the
    # merge-skip; outside it the fixed 256 target is better.
    #
    # The band was derived for 78 SM, where 3*SM == 234 reaches chunks == 1 for
    # P in 118..128. At 132 SM it cannot: 3*SM == 396 yields chunks == 2 there,
    # and for P in 129..198 that is strictly worse than the fixed 256 target,
    # which already gives 1. Measured on H100, the shipped band and no band at
    # all are indistinguishable at every default and user shape (0.0647 vs
    # 0.0647 ms at P=128) -- it is a dead configuration here.
    #
    # Reaching chunks == 1 at P == 128 needs a target in [128, 255], and the
    # payoff splits by dtype, because halving the grid also halves occupancy
    # (0.97 -> 0.48 waves/SM) and only bf16 is bandwidth-bound enough to come
    # out ahead. Measured at P=128, n=5, GQA 16:
    #            bf16                      fp8
    #   chunks=2 (shipped)  0.0647 ms      0.0593 ms
    #   chunks=1 (tg 255)   0.0626 ms      0.0740 ms
    #                       1.034x faster  0.801x slower
    # So the merge-skip is taken on the bf16 path only. P=64 and P=256 are
    # unaffected either way (1.000-1.001x), confirming this is scoped to the band.
    p = max(1, total_q * num_kv_heads)
    if _MSA_DECODE_GRID_BAND_LO <= p <= _MSA_DECODE_GRID_BAND_HI:
        target_grid = 3 * _sm_count(q.device)
    else:
        target_grid = _MSA_DECODE_TARGET_GRID
    if not use_fp8 and p <= _MSA_DECODE_TARGET_GRID:
        # bf16: if a target just under the fixed one would collapse the chunk
        # count to 1 while the selected target does not, take it -- that is the
        # merge-skip the band was meant to reach. On 78 SM the band already
        # gets there, so this changes nothing; it only fires where 3*SM
        # overshoots. max_topk is in the min() above, so compare post-clamp.
        skip_grid = _MSA_DECODE_TARGET_GRID - 1
        if max(1, min(max_topk, skip_grid // p)) == 1 and (
            max(1, min(max_topk, target_grid // p)) > 1
        ):
            target_grid = skip_grid
    target = max(1, min(max_topk, target_grid // p))
    num_topk_chunks = 1 << (target.bit_length() - 1)

    # chunks == 1 makes the merge a pure copy (single-chunk softmax weights are
    # identically 1.0): allocate no partials and write final output directly.
    # The constexpr-bounded block pointers in the bf16 kernel are only in
    # range when the GQA group exactly fills BLOCK_SIZE_H; see the comment
    # there. Any other head configuration falls through to the general kernel.
    # Ordered so the cheap integer tests short-circuit before the
    # next_power_of_2 call: at small batch this wrapper's Python cost is
    # comparable to the kernel itself (~17 us), so it has to stay lean.
    # Cheap integer tests first; next_power_of_2 is a Python call and this
    # wrapper runs against a ~17 us kernel, so it must not be evaluated on the
    # fallback path.
    use_fused_decode = (
        not use_fp8
        and p >= _MSA_DECODE_FUSED_MIN_PARALLELISM
        and gqa_group_size == max(16, triton.next_power_of_2(gqa_group_size))
    ) if (not use_fp8 and p >= _MSA_DECODE_FUSED_MIN_PARALLELISM) else False

    # Fast path: neither optimization applies when there are multiple chunks
    # (nothing to merge-skip) and the bf16 kernel is ineligible. The decode
    # kernel is only ~17 us at small batch, so this wrapper's own Python cost
    # is a measurable share of wall time -- taking the shortest route here is
    # worth more than any kernel tweak. Measured: without this, small-batch
    # shapes ran 0.89-0.95x of upstream purely on host overhead.
    # `single_chunk` selects the kernel and whether a merge pass runs; it is
    # deliberately NOT folded into USE_PDL. Doing so forks that constexpr and
    # splits the shared decode kernel into extra compiled variants, so the
    # same shape resolves to different cache entries depending on the call
    # site -- measured as a systematic 1.4-3.8% loss on the chunks>1 shapes.
    single_chunk = num_topk_chunks == 1
    if not single_chunk and not use_fused_decode:
        o_partial = torch.empty(
            num_topk_chunks, total_q, num_heads, head_dim,
            dtype=q.dtype, device=q.device,
        )
        lse_partial = torch.empty(
            num_topk_chunks, total_q, num_heads,
            dtype=torch.float32, device=q.device,
        )
        _gqa_sparse_decode_kernel[(total_q * num_topk_chunks, num_kv_heads)](
            q, kv_cache, k_scale_arg, v_scale_arg, topk_idx, o_partial,
            lse_partial, block_table, seq_lens,
            total_q, gqa_group_size, head_dim, max_topk, sm_scale,
            decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            kv_cache.stride(3),
            stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1),
            lse_partial.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_FP8=use_fp8,
            KV_SCALE_MODE=kv_scale_mode,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
        _merge_topk_attn_out_kernel[(total_q, num_heads)](
            o_partial, lse_partial, output, head_dim,
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1),
            lse_partial.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
        return

    if single_chunk:
        o_partial = output
        lse_partial = output  # unused; keeps the launch signature uniform
        stride_o_c = 0
        stride_l_c = stride_l_b = stride_l_h = 0
        stride_o_b, stride_o_h, stride_o_d = (
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
        pdl_launch = {}
    else:
        o_partial = torch.empty(
            num_topk_chunks, total_q, num_heads, head_dim,
            dtype=q.dtype, device=q.device,
        )
        lse_partial = torch.empty(
            num_topk_chunks, total_q, num_heads,
            dtype=torch.float32, device=q.device,
        )
        stride_o_c, stride_o_b, stride_o_h, stride_o_d = (
            o_partial.stride(0),
            o_partial.stride(1),
            o_partial.stride(2),
            o_partial.stride(3),
        )
        stride_l_c, stride_l_b, stride_l_h = (
            lse_partial.stride(0),
            lse_partial.stride(1),
            lse_partial.stride(2),
        )

    grid = (total_q * num_topk_chunks, num_kv_heads)
    # The bf16-specialized kernel bounds its Q/O block pointers by the
    # constexpr BLOCK_SIZE_H, which lets Triton drop the boundary masks and
    # brings registers 213 -> 168, i.e. the occupancy limiter from 2 to 3
    # blocks/SM. That is a large win where there are enough CTAs to use the
    # extra concurrency and a small LOSS where there are not: measured
    # 1.09-1.16x for P >= 512 but 0.97-1.00x at P = 128..256, where the chunk
    # count is already 1-2 and the shapes are latency- rather than
    # occupancy-limited. So it is scoped to the range where it was measured
    # to win, exactly like the TARGET_GRID band above.
    if use_fused_decode:
        _gqa_sparse_decode_fused_kernel[grid](
            q, kv_cache, topk_idx, o_partial, lse_partial, block_table,
            seq_lens,
            total_q, gqa_group_size, head_dim, max_topk, sm_scale,
            decode_query_len,
            q.stride(0), q.stride(1), q.stride(2),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            kv_cache.stride(3),
            topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
            stride_o_c, stride_o_b, stride_o_h, stride_o_d,
            stride_l_c, stride_l_b, stride_l_h,
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            SINGLE_CHUNK=single_chunk,
            **pdl_launch,
        )
    else:
        # Only the merge-skip path needs the modified kernel. When there are
        # multiple chunks there is no merge to skip, so use upstream's kernel
        # verbatim: the extra constexpr bought nothing and measured 0.89-0.92x
        # on this repo's small-batch benchmark shapes.
        if single_chunk:
            _gqa_sparse_decode_mergeskip_kernel[grid](
                q, kv_cache, k_scale_arg, v_scale_arg, topk_idx, o_partial,
                lse_partial, block_table, seq_lens,
                total_q, gqa_group_size, head_dim, max_topk, sm_scale,
                decode_query_len,
                q.stride(0), q.stride(1), q.stride(2),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3),
                stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                stride_o_c, stride_o_b, stride_o_h, stride_o_d,
                stride_l_c, stride_l_b, stride_l_h,
                block_table.stride(0),
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                NUM_TOPK_CHUNKS=num_topk_chunks,
                USE_FP8=use_fp8,
                KV_SCALE_MODE=kv_scale_mode,
                USE_PDL=use_pdl,
                SINGLE_CHUNK=single_chunk,
                **pdl_launch,
            )
        else:
            _gqa_sparse_decode_kernel[grid](
                q, kv_cache, k_scale_arg, v_scale_arg, topk_idx, o_partial,
                lse_partial, block_table, seq_lens,
                total_q, gqa_group_size, head_dim, max_topk, sm_scale,
                decode_query_len,
                q.stride(0), q.stride(1), q.stride(2),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                kv_cache.stride(3),
                stride_ks_h, stride_ks_t, stride_vs_h, stride_vs_t,
                topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
                stride_o_c, stride_o_b, stride_o_h, stride_o_d,
                stride_l_c, stride_l_b, stride_l_h,
                block_table.stride(0),
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                NUM_TOPK_CHUNKS=num_topk_chunks,
                USE_FP8=use_fp8,
                KV_SCALE_MODE=kv_scale_mode,
                USE_PDL=use_pdl,
                **pdl_launch,
            )
    if not single_chunk:
        merge_grid = (total_q, num_heads)
        _merge_topk_attn_out_kernel[merge_grid](
            o_partial, lse_partial, output, head_dim,
            o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
            o_partial.stride(3),
            lse_partial.stride(0), lse_partial.stride(1),
            lse_partial.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_launch,
        )
