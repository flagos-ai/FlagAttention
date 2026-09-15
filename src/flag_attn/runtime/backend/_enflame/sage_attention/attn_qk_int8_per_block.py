# Copyright 2024 SageAttention Team
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

"""Enflame S60 SageAttention forward for per-block INT8 Q/K tensors."""

import torch
import triton
import triton.language as tl

_Q_SCALE_BLOCK = 128
_K_SCALE_BLOCK = 64


@triton.jit
def _dequant_k_fp16(
    K,
    K_scale,
    K_fp16,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_okz,
    stride_okh,
    stride_okb,
    stride_okd,
    stride_okn,
    stride_ksz,
    stride_ksh,
    kv_len,
    H_KV: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    K_SCALE_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Dequantize each K element once before the tiled attention kernel."""
    pid = tl.program_id(0)
    block_n = pid % NUM_K_BLOCKS
    off_hz = pid // NUM_K_BLOCKS
    off_h = off_hz % H_KV
    off_z = off_hz // H_KV

    offs_n_in_block = tl.arange(0, BLOCK_N)
    offs_n = block_n * BLOCK_N + offs_n_in_block
    offs_k = tl.arange(0, HEAD_DIM)
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_k[None, :]
    )
    if TILE_MAJOR:
        out_ptrs = (
            K_fp16
            + off_z * stride_okz
            + off_h * stride_okh
            + block_n * stride_okb
            + offs_k[None, :] * stride_okd
            + offs_n_in_block[:, None] * stride_okn
        )
    else:
        out_ptrs = (
            K_fp16
            + off_z * stride_okz
            + off_h * stride_okh
            + offs_n[:, None] * stride_okn
            + offs_k[None, :] * stride_okd
        )
    if EVEN_N:
        k = tl.load(k_ptrs).to(tl.float16)
    else:
        n_mask = offs_n[:, None] < kv_len
        k = tl.load(k_ptrs, mask=n_mask, other=0).to(tl.float16)

    scale_ptr = (
        K_scale
        + off_z * stride_ksz
        + off_h * stride_ksh
        + (block_n * BLOCK_N) // K_SCALE_BLOCK
    )
    if BLOCK_N == K_SCALE_BLOCK:
        scale = tl.load(scale_ptr).to(tl.float16)
    else:
        scale_0 = tl.load(scale_ptr)
        if EVEN_N:
            scale_1 = tl.load(scale_ptr + 1)
        else:
            scale_1 = tl.load(
                scale_ptr + 1,
                mask=(block_n * BLOCK_N + K_SCALE_BLOCK) < kv_len,
                other=0.0,
            )
        scale = tl.where(
            offs_n_in_block < K_SCALE_BLOCK, scale_0, scale_1
        ).to(tl.float16)
    if BLOCK_N == K_SCALE_BLOCK:
        k = k * scale
    else:
        k = k * scale[:, None]

    if EVEN_N:
        tl.store(out_ptrs, k)
    else:
        tl.store(out_ptrs, k, mask=offs_n[:, None] < kv_len)

@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    qo_len,
    kv_len,
    K_ptrs,
    K_scale_ptr,
    V_ptrs,
    stride_kblock,
    stride_vn,
    mask_ptrs,
    stride_maskn,
    Q_SCALE_BLOCK: tl.constexpr,
    K_SCALE_BLOCK: tl.constexpr,
    QK_SCALE_BEFORE_DOT: tl.constexpr,
    PREFETCH_V: tl.constexpr,
    PREFETCH_K_NEXT: tl.constexpr,
    K_PREDEQUANT: tl.constexpr,
    K_TILE_MAJOR: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
):
    lo, hi = 0, kv_len
    if PREFETCH_K_NEXT:
        k_raw = tl.load(K_ptrs)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(
                    mask_ptrs + start_n * stride_maskn,
                    mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n),
                    other=False,
                )
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(
                    mask_ptrs + start_n * stride_maskn,
                    mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n),
                    other=-1.0e6,
                )
        if not skip:
            # INT8 tl.dot does not lower to an efficient matrix path with the
            # current S60 Triton backend. Convert the quantized values to FP16
            # while retaining INT8 storage and the per-block scales.
            if PREFETCH_K_NEXT:
                k = k_raw
            elif EVEN_N:
                k = tl.load(K_ptrs)
            else:
                k_mask = offs_n[None, :] < (kv_len - start_n)
                k = tl.load(K_ptrs, mask=k_mask, other=0)
            if PREFETCH_V:
                if EVEN_N:
                    v = tl.load(V_ptrs)
                else:
                    v = tl.load(
                        V_ptrs,
                        mask=offs_n[:, None] < (kv_len - start_n),
                        other=0.0,
                    )
            if not K_PREDEQUANT:
                k = k.to(tl.float16)
                if BLOCK_N == K_SCALE_BLOCK:
                    k_scale = tl.load(K_scale_ptr)
                else:
                    k_scale_0 = tl.load(K_scale_ptr)
                    if EVEN_N:
                        k_scale_1 = tl.load(K_scale_ptr + 1)
                    else:
                        k_scale_1 = tl.load(
                            K_scale_ptr + 1,
                            mask=K_SCALE_BLOCK < (kv_len - start_n),
                            other=0.0,
                        )
                    k_scale = tl.where(
                        offs_n < K_SCALE_BLOCK, k_scale_0, k_scale_1
                    )

            if QK_SCALE_BEFORE_DOT:
                # Dequantize the smaller K tile before the matrix multiply.
                # This replaces a BLOCK_M x BLOCK_N FP32 score scaling pass
                # with a HEAD_DIM x BLOCK_N conversion/scaling pass. Q was
                # already dequantized once outside the KV loop. Cast the
                # scale first so the multiply itself stays in FP16 instead of
                # producing an FP32 K tile that is immediately narrowed.
                if not K_PREDEQUANT:
                    k_scale = k_scale.to(tl.float16)
                    if BLOCK_N == K_SCALE_BLOCK:
                        k = k * k_scale
                    else:
                        k = k * k_scale[None, :]
                qk = tl.dot(q, k, out_dtype=tl.float32)
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32)
                if K_PREDEQUANT:
                    if BLOCK_M > Q_SCALE_BLOCK:
                        qk *= q_scale[:, None]
                    else:
                        qk *= q_scale
                else:
                    if BLOCK_M > Q_SCALE_BLOCK:
                        if BLOCK_N == K_SCALE_BLOCK:
                            qk *= q_scale[:, None] * k_scale
                        else:
                            qk *= q_scale[:, None] * k_scale[None, :]
                    else:
                        if BLOCK_N == K_SCALE_BLOCK:
                            qk *= q_scale * k_scale
                        else:
                            qk *= q_scale * k_scale[None, :]

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            elif not EVEN_N:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            # PV consumes FP16 probabilities.  Narrow before the sum as well
            # so the numerator and denominator use the same probabilities.
            # On gcu300 this also halves the temporary traffic generated by
            # the second score-tile reduction; keep the row sum itself FP32.
            p = tl.math.exp2(qk).to(tl.float16)
            l_ij = tl.sum(p, 1).to(tl.float32)

            alpha = tl.math.exp2(m_i - m_ij)
            acc = acc * alpha[:, None]

            if PREFETCH_K_NEXT:
                if K_TILE_MAJOR:
                    next_k_ptrs = K_ptrs + HEAD_DIM * BLOCK_N
                else:
                    next_k_ptrs = K_ptrs + stride_kblock
                k_raw_next = tl.load(
                    next_k_ptrs,
                    mask=offs_n[None, :] < (hi - start_n - BLOCK_N),
                    other=0,
                )
            if not PREFETCH_V:
                if EVEN_N:
                    v = tl.load(V_ptrs)
                else:
                    v = tl.load(
                        V_ptrs,
                        mask=offs_n[:, None] < (kv_len - start_n),
                        other=0.0,
                    )
            # Accumulate PV directly into the FP32 output accumulator.  S60
            # measurements show that this fused path is faster than forming
            # an FP16 dot result followed by a separate FP32 add.
            acc = tl.dot(p, v, acc)

            # Keep the online-softmax state update at the end of the loop so
            # its scheduling remains identical to the measured baseline.
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            if PREFETCH_K_NEXT:
                k_raw = k_raw_next
        if K_TILE_MAJOR:
            K_ptrs += HEAD_DIM * BLOCK_N
        else:
            K_ptrs += stride_kblock
        if not K_PREDEQUANT:
            K_scale_ptr += BLOCK_N // K_SCALE_BLOCK
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    Out,
    mask,
    Lse,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_kblock,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    stride_qsz,
    stride_qsh,
    stride_ksz,
    stride_ksh,
    stride_maskz,
    stride_maskh,
    stride_maskm,
    stride_maskn,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
    Q_SCALE_BLOCK: tl.constexpr,
    K_SCALE_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QK_SCALE_BEFORE_DOT: tl.constexpr,
    PREFETCH_V: tl.constexpr,
    PREFETCH_K_NEXT: tl.constexpr,
    K_PREDEQUANT: tl.constexpr,
    K_TILE_MAJOR: tl.constexpr,
    FIRST_TILE_FAST: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    # Flattening avoids depending on backend-specific multi-dimensional grid
    # mapping. Keep the decomposition constexpr so power-of-two shapes use
    # cheap div/mod operations.
    pid = tl.program_id(0)
    start_m = pid % NUM_Q_BLOCKS
    off_hz = pid // NUM_Q_BLOCKS
    off_h = off_hz % H
    off_z = off_hz // H

    offs_m_in_block = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_in_block
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_k[None, :]
    )
    q_scale_block = (start_m * BLOCK_M) // Q_SCALE_BLOCK
    Q_scale_ptr = Q_scale + off_z * stride_qsz + off_h * stride_qsh + q_scale_block
    if K_TILE_MAJOR:
        # Tile-major K is physically [B, H, K_BLOCK, D, BLOCK_N].
        # Its two matrix dimensions are fixed by the kernel tile, so avoid
        # routing these inner strides through the dynamic-stride load path.
        K_ptrs = (
            K
            + off_z * stride_kz
            + (off_h // num_kv_groups) * stride_kh
            + offs_n[None, :]
            + offs_k[:, None] * BLOCK_N
        )
    else:
        K_ptrs = (
            K
            + off_z * stride_kz
            + (off_h // num_kv_groups) * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_k[:, None] * stride_kk
        )
    K_scale_ptr = (
        K_scale
        + off_z * stride_ksz
        + (off_h // num_kv_groups) * stride_ksh
    )
    V_ptrs = (
        V
        + off_z * stride_vz
        + (off_h // num_kv_groups) * stride_vh
        + offs_n[:, None] * stride_vn
        + offs_k[None, :]
    )
    O_block_ptr = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_on
        + offs_k[None, :]
    )
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = (
            mask
            + off_z * stride_maskz
            + off_h * stride_maskh
            + offs_m[:, None] * stride_maskm
            + offs_n[None, :] * stride_maskn
        )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if EVEN_M:
        q = tl.load(Q_ptrs).to(tl.float16)
    else:
        q = tl.load(
            Q_ptrs, mask=offs_m[:, None] < qo_len, other=0
        ).to(tl.float16)
    if BLOCK_M > Q_SCALE_BLOCK:
        # The D64 fast path combines two 128-row Q-scale blocks into one
        # program so the same K/V tile can serve both blocks.
        q_scale_0 = tl.load(Q_scale_ptr)
        if EVEN_M:
            q_scale_1 = tl.load(Q_scale_ptr + 1)
        else:
            q_scale_1 = tl.load(
                Q_scale_ptr + 1,
                mask=(start_m * BLOCK_M + Q_SCALE_BLOCK) < qo_len,
                other=0.0,
            )
        q_scale = tl.where(
            offs_m_in_block < Q_SCALE_BLOCK, q_scale_0, q_scale_1
        )
    else:
        q_scale = tl.load(Q_scale_ptr)
    if QK_SCALE_BEFORE_DOT:
        # Q has one scalar scale for the whole BLOCK_M tile, so pay this
        # conversion once instead of multiplying every score in every KV
        # iteration by q_scale. Narrow the scale before multiplying so the
        # dequantized Q tile is never materialized as an FP32 intermediate.
        q_scale_f16 = q_scale.to(tl.float16)
        if BLOCK_M > Q_SCALE_BLOCK:
            q = q * q_scale_f16[:, None]
        else:
            q = q * q_scale_f16

    if FIRST_TILE_FAST:
        # For the first dense tile m_i=-inf and acc=0, so alpha is exactly
        # zero. Initialize the online-softmax state directly and let the
        # regular loop handle only the remaining tiles.
        k = tl.load(K_ptrs)
        if PREFETCH_V:
            v = tl.load(V_ptrs)
        if not K_PREDEQUANT:
            k = k.to(tl.float16)
            if BLOCK_N == K_SCALE_BLOCK:
                k_scale = tl.load(K_scale_ptr).to(tl.float16)
                k = k * k_scale
            else:
                k_scale_0 = tl.load(K_scale_ptr)
                k_scale_1 = tl.load(K_scale_ptr + 1)
                k_scale = tl.where(
                    offs_n < K_SCALE_BLOCK, k_scale_0, k_scale_1
                ).to(tl.float16)
                k = k * k_scale[None, :]
        qk = tl.dot(q, k, out_dtype=tl.float32)
        m_i = tl.max(qk, 1)
        p = tl.math.exp2(qk - m_i[:, None]).to(tl.float16)
        l_i = tl.sum(p, 1).to(tl.float32)
        if not PREFETCH_V:
            v = tl.load(V_ptrs)
        acc = tl.dot(p, v, acc)

        if K_TILE_MAJOR:
            K_ptrs += HEAD_DIM * BLOCK_N
        else:
            K_ptrs += stride_kblock
        if not K_PREDEQUANT:
            K_scale_ptr += BLOCK_N // K_SCALE_BLOCK
        V_ptrs += BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            qo_len,
            kv_len - BLOCK_N,
            K_ptrs,
            K_scale_ptr,
            V_ptrs,
            stride_kblock,
            stride_vn,
            mask_ptrs,
            stride_maskn,
            Q_SCALE_BLOCK,
            K_SCALE_BLOCK,
            QK_SCALE_BEFORE_DOT,
            PREFETCH_V,
            PREFETCH_K_NEXT,
            K_PREDEQUANT,
            K_TILE_MAJOR,
            EVEN_N,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_scale,
            qo_len,
            kv_len,
            K_ptrs,
            K_scale_ptr,
            V_ptrs,
            stride_kblock,
            stride_vn,
            mask_ptrs,
            stride_maskn,
            Q_SCALE_BLOCK,
            K_SCALE_BLOCK,
            QK_SCALE_BEFORE_DOT,
            PREFETCH_V,
            PREFETCH_K_NEXT,
            K_PREDEQUANT,
            K_TILE_MAJOR,
            EVEN_N,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
        )
    # Form one reciprocal per query row and broadcast the multiply.  Spelling
    # this as a matrix/scalar division can make backends emit a division for
    # every output element instead of hoisting the row-wise reciprocal.
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    if EVEN_M:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    else:
        tl.store(
            O_block_ptr,
            acc.to(Out.type.element_ty),
            mask=offs_m[:, None] < qo_len,
        )

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        if EVEN_M:
            tl.store(lse_ptrs, l_i)
        else:
            tl.store(lse_ptrs, l_i, mask=offs_m < qo_len)

def _select_s60_config(
    head_dim, block_m=None, block_n=None, num_warps=None, num_stages=None
):
    """Choose a low-register-pressure launch configuration for Enflame S60."""
    if head_dim not in (64, 128):
        raise ValueError(f"head_dim {head_dim} not supported; expected 64 or 128")

    # With the FP16 QK compute path, one warp is both correct and about four
    # times faster than four warps in S60 measurements. Additional warps only
    # add scheduling and resource overhead for this tile.
    default_block_m = 128
    block_m = default_block_m if block_m is None else block_m
    # S60 measurements across the 20-shape benchmark table show that a
    # 128-column KV tile consistently outperforms the original 64-column
    # tile. Each tile applies two independent 64-row K scales in the kernel.
    block_n = 128 if block_n is None else block_n
    num_warps = 1 if num_warps is None else num_warps
    num_stages = 1 if num_stages is None else num_stages

    if block_m not in (64, 128) or _Q_SCALE_BLOCK % block_m != 0:
        raise ValueError("block_m must be 64 or 128 and divide the 128-row Q scale block")
    if block_n not in (64, 128) or block_n % _K_SCALE_BLOCK != 0:
        raise ValueError(
            "block_n must be 64 or 128 and be a multiple of the "
            "64-row K scale block"
        )
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive")
    return block_m, block_n, num_warps, num_stages

def forward(
    q,
    k,
    v,
    q_scale,
    k_scale,
    tensor_layout="HND",
    attn_mask=None,
    output_dtype=torch.float16,
    return_lse=False,
    maxnreg=None,
    block_m=None,
    num_warps=None,
    num_stages=None,
    block_n=None,
    qk_scale_before_dot=True,
    predequantize_k=True,
    tile_major_predequant_k=True,
    prefetch_v=True,
    prefetch_k_next=True,
    prefetch_d64=False,
    first_tile_d128=True,
):
    """Run per-block INT8 QK attention on Enflame S60.

    All supported configurations use the Triton attention kernel. Dense
    self-attention stays on the same path so S60 does not depend on the
    torch-gcu SDPA implementation.

    D128 prefetch and first-tile optimizations are enabled by default. D64 V
    prefetch defaults to disabled because it regresses the long-sequence path.
    """

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = (
            q.stride(0),
            q.stride(1),
            q.stride(2),
        )
        stride_bz_k, stride_h_k, stride_seq_k, stride_dim_k = (
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
        )
        stride_bz_v, stride_h_v, stride_seq_v = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
        )
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = (
            q.stride(0),
            q.stride(2),
            q.stride(1),
        )
        stride_bz_k, stride_h_k, stride_seq_k, stride_dim_k = (
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
        )
        stride_bz_v, stride_h_v, stride_seq_v = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
        )
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask = attn_mask.stride(0)
        stride_h_mask = attn_mask.stride(1)
        stride_m_mask = attn_mask.stride(2)
        stride_n_mask = attn_mask.stride(3)
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv
    BLOCK_M, BLOCK_N, launch_num_warps, launch_num_stages = _select_s60_config(
        head_dim, block_m, block_n, num_warps, num_stages
    )
    if maxnreg is not None and maxnreg <= 0:
        raise ValueError("maxnreg must be positive")

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    if tensor_layout == "HND":
        stride_bz_o, stride_h_o, stride_seq_o = (
            o.stride(0),
            o.stride(1),
            o.stride(2),
        )
    else:
        stride_bz_o, stride_h_o, stride_seq_o = (
            o.stride(0),
            o.stride(2),
            o.stride(1),
        )

    # D64 has enough accumulator headroom to merge two logical Q tiles. This
    # halves the number of programs and lets both 128-row Q blocks consume the
    # same K/V load. Keep D128 and masked attention on the lower-pressure tile.
    if head_dim == 64 and BLOCK_M == 128 and attn_mask is None:
        BLOCK_M = 2 * BLOCK_M
    stride_kblock = BLOCK_N * stride_seq_k
    # K is consumed once per Q program, so its inline INT8 conversion and
    # scaling are repeated qo_len / BLOCK_M times. Pre-dequantizing once trades
    # one linear preprocessing kernel and an FP16 temporary for a simpler QK
    # matrix path in every attention program.
    k_reuse_blocks = triton.cdiv(qo_len, BLOCK_M) * num_kv_groups
    # D64 amortizes preprocessing from 32 reuse blocks onward.  On D128 the
    # 32-reuse/T4096 shape is slower and substantially less stable with the
    # temporary FP16 K path; enable it from 64 reuse blocks (T8192) instead.
    min_k_reuse_blocks = 32 if head_dim == 64 else 64
    use_predequant_k = (
        predequantize_k
        and qk_scale_before_dot
        and attn_mask is None
        and k_reuse_blocks >= min_k_reuse_blocks
    )
    use_tile_major_k = (
        use_predequant_k
        and tile_major_predequant_k
        and head_dim == 128
    )
    k_attn = k
    if use_predequant_k:
        # The attention dot consumes K as [D, BLOCK_N].  Keep each KV tile in
        # that exact local layout so N is contiguous without making the D-row
        # stride grow with the full sequence length.  The row-major temporary
        # remains available as a direct A/B and compatibility fallback.
        # The tile-local transpose is consistently beneficial for D128, but
        # D64 can enter a much slower S60 scheduling regime at T8192.  Keep
        # D64 on the measured-stable row-major predequantized path.
        dequant_block_n = BLOCK_N if use_tile_major_k else 2 * _K_SCALE_BLOCK
        num_k_blocks = triton.cdiv(kv_len, dequant_block_n)
        if use_tile_major_k:
            k_attn = torch.empty(
                (b, h_kv, num_k_blocks, head_dim, dequant_block_n),
                dtype=torch.float16,
                device=k.device,
            )
            stride_okz = k_attn.stride(0)
            stride_okh = k_attn.stride(1)
            stride_okb = k_attn.stride(2)
            stride_okd = k_attn.stride(3)
            stride_okn = k_attn.stride(4)
        else:
            k_attn = torch.empty_like(k, dtype=torch.float16)
            stride_okz = k_attn.stride(0)
            stride_okh = (
                k_attn.stride(1)
                if tensor_layout == "HND"
                else k_attn.stride(2)
            )
            stride_okb = 0
            stride_okd = k_attn.stride(3)
            stride_okn = (
                k_attn.stride(2)
                if tensor_layout == "HND"
                else k_attn.stride(1)
            )
        _dequant_k_fp16[(b * h_kv * num_k_blocks,)](
            k,
            k_scale,
            k_attn,
            stride_bz_k,
            stride_h_k,
            stride_seq_k,
            stride_okz,
            stride_okh,
            stride_okb,
            stride_okd,
            stride_okn,
            k_scale.stride(0),
            k_scale.stride(1),
            kv_len,
            H_KV=h_kv,
            NUM_K_BLOCKS=num_k_blocks,
            K_SCALE_BLOCK=_K_SCALE_BLOCK,
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_N=dequant_block_n,
            TILE_MAJOR=use_tile_major_k,
            EVEN_N=(kv_len % dequant_block_n == 0),
            num_warps=1,
            num_stages=1,
        )
        if use_tile_major_k:
            stride_bz_k, stride_h_k, stride_kblock = (
                k_attn.stride(0),
                k_attn.stride(1),
                k_attn.stride(2),
            )
            stride_dim_k = k_attn.stride(3)
            stride_seq_k = k_attn.stride(4)
        elif tensor_layout == "HND":
            stride_bz_k, stride_h_k, stride_seq_k, stride_dim_k = (
                k_attn.stride(0),
                k_attn.stride(1),
                k_attn.stride(2),
                k_attn.stride(3),
            )
            stride_kblock = BLOCK_N * stride_seq_k
        else:
            stride_bz_k, stride_h_k, stride_seq_k, stride_dim_k = (
                k_attn.stride(0),
                k_attn.stride(2),
                k_attn.stride(1),
                k_attn.stride(3),
            )
            stride_kblock = BLOCK_N * stride_seq_k
    if return_lse:
        lse = torch.empty((b, h_qo, qo_len), dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty((0,), dtype=torch.float32, device="cpu")

    num_q_blocks = triton.cdiv(qo_len, BLOCK_M)
    grid = (num_q_blocks * h_qo * b,)
    even_m = qo_len % BLOCK_M == 0
    even_n = kv_len % BLOCK_N == 0
    first_tile_fast = (
        (
            head_dim == 64
            or (
                first_tile_d128
                and head_dim == 128
                and BLOCK_M == 128
                and BLOCK_N == 128
            )
        )
        and attn_mask is None
        and qk_scale_before_dot
        and even_m
        and even_n
        and kv_len >= BLOCK_N
    )
    # D64 benefits from overlapping the V load with QK/softmax except at the
    # first K-predequant reuse tier.  That boundary configuration exhibits a
    # reproducible fast/slow scheduling split on S60, so keep it on the
    # stable load order and re-enable prefetch once reuse reaches 64 blocks.
    enable_d64_prefetch_v = (
        prefetch_d64
        and head_dim == 64
        and BLOCK_M == 256
        and (
            not use_predequant_k
            or k_reuse_blocks >= 2 * min_k_reuse_blocks
        )
    )
    enable_prefetch_v = (
        prefetch_v
        and (
            (head_dim == 128 and BLOCK_M == 128)
            or enable_d64_prefetch_v
        )
        and BLOCK_N == 128
        and attn_mask is None
        and even_m
        and even_n
    )
    enable_prefetch_k_next = (
        prefetch_k_next
        and head_dim == 128
        and BLOCK_M == 128
        and BLOCK_N == 128
        and attn_mask is None
        and even_n
        and kv_len >= 2 * BLOCK_N
    )
    launch_options = {}
    if maxnreg is not None:
        # `maxnreg` is a CUDA-specific launch option; only pass it to the
        # kernel on CUDA devices. Other backends (e.g. Enflame GCU) do not
        # recognise it and would raise a KeyError.
        if q.device.type == "cuda":
            launch_options["maxnreg"] = maxnreg
    _attn_fwd[grid](
        q,
        k_attn,
        v,
        q_scale,
        k_scale,
        o,
        attn_mask,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_dim_k,
        stride_kblock,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        stride_bz_mask,
        stride_h_mask,
        stride_m_mask,
        stride_n_mask,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        NUM_Q_BLOCKS=num_q_blocks,
        Q_SCALE_BLOCK=_Q_SCALE_BLOCK,
        K_SCALE_BLOCK=_K_SCALE_BLOCK,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_K,
        QK_SCALE_BEFORE_DOT=qk_scale_before_dot,
        PREFETCH_V=enable_prefetch_v,
        PREFETCH_K_NEXT=enable_prefetch_k_next,
        K_PREDEQUANT=use_predequant_k,
        K_TILE_MAJOR=use_tile_major_k,
        FIRST_TILE_FAST=first_tile_fast,
        EVEN_M=even_m,
        EVEN_N=even_n,
        RETURN_LSE=return_lse,
        num_warps=launch_num_warps,
        num_stages=launch_num_stages,
        **launch_options,
    )

    return o, lse
