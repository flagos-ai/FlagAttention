# Copyright 2024 SageAttention Team
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

import os

import torch
import triton
import triton.language as tl


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
    stride_kn,
    stride_vn,
    mask_ptrs,
    stride_maskn,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    USE_INT8_DOT: tl.constexpr,
    EVEN_KV: tl.constexpr,
):
    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_valid = offs_n[None, :] < (kv_len - start_n)
        if EVEN_KV:
            k = tl.load(K_ptrs)
        else:
            k = tl.load(K_ptrs, mask=k_valid, other=0)
        k_scale = tl.load(K_scale_ptr)

        if USE_INT8_DOT:
            qk = tl.dot(q, k).to(tl.float32)
        else:
            qk = tl.dot(q.to(tl.float16), k.to(tl.float16), out_dtype=tl.float32)
        qk *= q_scale * k_scale

        if HAS_MASK:
            mask_valid = (offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n)
            if MASK_IS_BOOL:
                mask_block = tl.load(
                    mask_ptrs + start_n * stride_maskn,
                    mask=mask_valid,
                    other=False,
                )
                qk += tl.where(mask_block, 0.0, -1.0e6)
            else:
                mask_block = tl.load(
                    mask_ptrs + start_n * stride_maskn,
                    mask=mask_valid,
                    other=-1.0e6,
                )
                qk += mask_block
        elif not EVEN_KV:
            qk += tl.where(k_valid, 0.0, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc *= alpha[:, None]

        if EVEN_KV:
            v = tl.load(V_ptrs)
        else:
            v = tl.load(
                V_ptrs,
                mask=offs_n[:, None] < (kv_len - start_n),
                other=0.0,
            )
        # MP31 benefits from an FP16 local PV result at D=64, while D=128 is
        # substantially faster with the native FP32 dot-accumulation path.
        # HEAD_DIM is constexpr, so this produces one specialized path per D
        # without a runtime branch.  The long-lived online accumulator remains
        # FP32 in both cases.
        if HEAD_DIM == 64:
            acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        else:
            acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_ij

        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
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
    stride_vz,
    stride_vh,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_on,
    stride_maskz,
    stride_maskh,
    stride_maskm,
    stride_maskn,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SCALE_BLOCK: tl.constexpr,
    K_SCALE_BLOCK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_BOOL: tl.constexpr,
    USE_INT8_DOT: tl.constexpr,
    EVEN_Q: tl.constexpr,
    EVEN_KV: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, Q_SCALE_BLOCK)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, K_SCALE_BLOCK)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    Q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qn + offs_k[None, :]
    # Programs may be smaller than the 128-row Q quantization group; adjacent
    # programs then share the same scale without changing quantization semantics.
    if BLOCK_M == Q_SCALE_BLOCK:
        q_scale_idx = start_m
    else:
        q_scale_idx = (start_m * BLOCK_M) // Q_SCALE_BLOCK
    Q_scale_ptr = Q_scale + q_scale_offset + q_scale_idx
    K_ptrs = (
        K
        + off_z * stride_kz
        + (off_h // num_kv_groups) * stride_kh
        + offs_n[None, :] * stride_kn
        + offs_k[:, None] * stride_kk
    )
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = (
        V + off_z * stride_vz + (off_h // num_kv_groups) * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :]
    )
    O_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_on + offs_k[None, :]

    if HAS_MASK:
        mask_ptrs = (
            mask
            + off_z * stride_maskz
            + off_h * stride_maskh
            + offs_m[:, None] * stride_maskm
            + offs_n[None, :] * stride_maskn
        )
    else:
        mask_ptrs = mask

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if EVEN_Q:
        q = tl.load(Q_ptrs)
    else:
        q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len, other=0)
    q_scale = tl.load(Q_scale_ptr)
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
        stride_kn,
        stride_vn,
        mask_ptrs,
        stride_maskn,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        offs_m,
        offs_n,
        HAS_MASK,
        MASK_IS_BOOL,
        USE_INT8_DOT,
        EVEN_KV,
    )

    acc /= l_i[:, None]
    if EVEN_Q:
        tl.store(O_ptrs, acc.to(Out.type.element_ty))
    else:
        tl.store(O_ptrs, acc.to(Out.type.element_ty), mask=offs_m[:, None] < qo_len)

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        if EVEN_Q:
            tl.store(lse_ptrs, tl.log2(l_i) + m_i)
        else:
            tl.store(lse_ptrs, tl.log2(l_i) + m_i, mask=offs_m < qo_len)


def _shape_and_strides(q, k, v, o, tensor_layout):
    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        q_strides = (q.stride(0), q.stride(1), q.stride(2))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2))
        o_strides = (o.stride(0), o.stride(1), o.stride(2))
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        q_strides = (q.stride(0), q.stride(2), q.stride(1))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1))
        o_strides = (o.stride(0), o.stride(2), o.stride(1))
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    return (
        b,
        h_qo,
        h_kv,
        qo_len,
        kv_len,
        head_dim,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
    )


def _validate_forward_inputs(q, k, v, q_scale, k_scale, tensor_layout, attn_mask):
    if q.dtype != torch.int8 or k.dtype != torch.int8:
        raise TypeError("MUSA SageAttention expects q and k to have dtype torch.int8")
    if v.dtype != torch.float16:
        raise TypeError("MUSA SageAttention currently expects v to have dtype torch.float16")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("MUSA SageAttention scales must have dtype torch.float32")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors")
    if tensor_layout not in {"HND", "NHD"}:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if attn_mask is not None and (
        attn_mask.ndim != 4 or attn_mask.dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32)
    ):
        raise TypeError("attn_mask must be a rank-4 bool or floating-point tensor")
    if not q_scale.is_contiguous() or not k_scale.is_contiguous():
        raise ValueError("q_scale and k_scale must be contiguous")


def _select_launch_config(
    head_dim,
    qo_len,
    kv_len,
    k_stride_n,
    has_mask,
    return_lse,
):
    """Select a device-verified MP31 launch configuration.

    The smaller-M specializations are limited to the packed-K, aligned core
    used by the production quantizer and benchmarks.  Ordinary contiguous K,
    masks, LSE, and partial blocks keep the previous fallback configuration.
    """
    block_n = 64
    use_packed_aligned_core = (
        k_stride_n == 1 and not has_mask and not return_lse and qo_len % 128 == 0 and kv_len % block_n == 0
    )
    if use_packed_aligned_core:
        if head_dim == 64:
            return 64, 8, 1
        return 32, 4, 1

    if head_dim == 64:
        return 128, 8, 1

    block_m = 64
    use_two_stages = not has_mask and not return_lse and qo_len % block_m == 0 and kv_len % block_n == 0
    return block_m, 8, 2 if use_two_stages else 1


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
):
    if maxnreg is not None and maxnreg <= 0:
        raise ValueError("maxnreg must be positive")
    _validate_forward_inputs(q, k, v, q_scale, k_scale, tensor_layout, attn_mask)

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)
    (
        b,
        h_qo,
        h_kv,
        qo_len,
        kv_len,
        head_dim,
        q_strides,
        k_strides,
        v_strides,
        o_strides,
    ) = _shape_and_strides(q, k, v, o, tensor_layout)

    if head_dim not in (64, 128):
        raise ValueError(f"MUSA SageAttention supports head_dim 64 or 128, got {head_dim}")
    if h_qo % h_kv != 0:
        raise ValueError("The number of query heads must be divisible by KV heads")
    if qo_len <= 0 or kv_len <= 0:
        raise ValueError("Query and KV sequence lengths must be positive")

    if tensor_layout == "HND":
        expected_k_shape = (b, h_kv, kv_len, head_dim)
        expected_v_shape = expected_k_shape
    else:
        expected_k_shape = (b, kv_len, h_kv, head_dim)
        expected_v_shape = expected_k_shape
    if tuple(k.shape) != expected_k_shape:
        raise ValueError(f"k must have shape {expected_k_shape}, got {tuple(k.shape)}")
    if tuple(v.shape) != expected_v_shape:
        raise ValueError(f"v must have shape {expected_v_shape}, got {tuple(v.shape)}")

    expected_q_blocks = triton.cdiv(qo_len, 128)
    expected_k_blocks = triton.cdiv(kv_len, 64)
    if q_scale.shape != (b, h_qo, expected_q_blocks):
        raise ValueError(f"q_scale must have shape {(b, h_qo, expected_q_blocks)}, got {tuple(q_scale.shape)}")
    if k_scale.shape != (b, h_kv, expected_k_blocks):
        raise ValueError(f"k_scale must have shape {(b, h_kv, expected_k_blocks)}, got {tuple(k_scale.shape)}")

    if attn_mask is not None:
        expected_mask = (b, h_qo, qo_len, kv_len)
        if tuple(attn_mask.shape) != expected_mask:
            raise ValueError(f"attn_mask must have shape {expected_mask}, got {tuple(attn_mask.shape)}")
        mask_strides = attn_mask.stride()
    else:
        mask_strides = (0, 0, 0, 0)

    if return_lse:
        lse = torch.empty((b, h_qo, qo_len), dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty((0,), dtype=torch.float32, device="cpu")

    # Q is quantized in 128-row groups.  The packed aligned core uses two M64
    # programs per group for D64 and four M32 programs per group for D128.
    # Fallbacks retain the previously verified M128/M64 configurations.
    q_scale_block = 128
    block_n = 64
    block_m, num_warps, num_stages = _select_launch_config(
        head_dim,
        qo_len,
        kv_len,
        k_strides[2],
        attn_mask is not None,
        return_lse,
    )
    # On MP31 the predicate-free aligned specialization improves D=128, but
    # changes D=64 scheduling/register allocation enough to regress it.  Keep
    # the original predicated D=64 specialization even for aligned shapes.
    use_aligned_io = head_dim == 128
    grid = (triton.cdiv(qo_len, block_m), h_qo, b)
    _attn_fwd[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        o,
        attn_mask,
        lse,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *mask_strides,
        qo_len,
        kv_len,
        H=h_qo,
        num_kv_groups=h_qo // h_kv,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        Q_SCALE_BLOCK=q_scale_block,
        K_SCALE_BLOCK=64,
        RETURN_LSE=return_lse,
        HAS_MASK=attn_mask is not None,
        MASK_IS_BOOL=attn_mask is not None and attn_mask.dtype == torch.bool,
        USE_INT8_DOT=os.environ.get("FLAG_ATTN_MTHREADS_INT8_DOT", "1") != "0",
        EVEN_Q=use_aligned_io and qo_len % block_m == 0,
        EVEN_KV=use_aligned_io and kv_len % block_n == 0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse
