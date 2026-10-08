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

import torch
import triton
import triton.language as tl


@triton.jit
def quant_per_block_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_id,
    stride_oz,
    stride_oh,
    stride_on,
    stride_od,
    stride_sz,
    stride_sh,
    sm_scale,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)
    valid = offs_n[:, None] < L

    input_ptrs = (
        Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :] * stride_id
    )
    output_ptrs = (
        Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :] * stride_od
    )
    scale_ptr = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=valid, other=0.0).to(tl.float32)
    x *= sm_scale
    max_abs = tl.max(tl.abs(x))
    scale = tl.where(max_abs > 0.0, max_abs / 127.0, 1.0)
    x_scaled = x / scale
    x_scaled += 0.5 * tl.where(x_scaled >= 0.0, 1.0, -1.0)
    tl.store(output_ptrs, x_scaled.to(tl.int8), mask=valid)
    tl.store(scale_ptr, scale)


def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q must have dtype torch.float16 or torch.bfloat16")
    if k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("k must have dtype torch.float16 or torch.bfloat16")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must be rank-4 tensors")
    if BLKQ <= 0 or BLKK <= 0:
        raise ValueError("BLKQ and BLKK must be positive")

    if km is not None:
        k = k - km

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        b_k, h_kv, kv_len, k_head_dim = k.shape
        # Keep K's public/logical shape as [B, H, N, D], but store it in a
        # [B, H, D, N] backing allocation.  The returned transpose view has a
        # unit N stride, so the attention kernel can load its logical [D, N]
        # SQMMA operand without re-transposing a row-major [N, D] tile.
        k_storage = torch.empty((b_k, h_kv, k_head_dim, kv_len), dtype=torch.int8, device=k.device)
        k_int8 = k_storage.transpose(-2, -1)
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        qo_strides = (
            q_int8.stride(0),
            q_int8.stride(1),
            q_int8.stride(2),
            q_int8.stride(3),
        )
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        ko_strides = (
            k_int8.stride(0),
            k_int8.stride(1),
            k_int8.stride(2),
            k_int8.stride(3),
        )
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        b_k, kv_len, h_kv, k_head_dim = k.shape
        # NHD exposes [B, N, H, D].  Use the same [B, H, D, N] backing
        # allocation and permute it into the requested logical layout.
        k_storage = torch.empty((b_k, h_kv, k_head_dim, kv_len), dtype=torch.int8, device=k.device)
        k_int8 = k_storage.permute(0, 3, 1, 2)
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        qo_strides = (
            q_int8.stride(0),
            q_int8.stride(2),
            q_int8.stride(1),
            q_int8.stride(3),
        )
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        ko_strides = (
            k_int8.stride(0),
            k_int8.stride(2),
            k_int8.stride(1),
            k_int8.stride(3),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    if b != b_k or head_dim != k_head_dim:
        raise ValueError("q and k must have the same batch size and head dimension")
    if head_dim not in (64, 128):
        raise ValueError(f"MUSA SageAttention supports head_dim 64 or 128, got {head_dim}")

    q_scale = torch.empty((b, h_qo, triton.cdiv(qo_len, BLKQ)), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, triton.cdiv(kv_len, BLKK)), device=k.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q_grid = (triton.cdiv(qo_len, BLKQ), h_qo, b)
    quant_per_block_int8_kernel[q_grid](
        q,
        q_int8,
        q_scale,
        qo_len,
        *q_strides,
        *qo_strides,
        q_scale.stride(0),
        q_scale.stride(1),
        sm_scale=sm_scale * 1.44269504,
        C=head_dim,
        BLK=BLKQ,
        num_warps=4,
        num_stages=1,
    )

    k_grid = (triton.cdiv(kv_len, BLKK), h_kv, b)
    quant_per_block_int8_kernel[k_grid](
        k,
        k_int8,
        k_scale,
        kv_len,
        *k_strides,
        *ko_strides,
        k_scale.stride(0),
        k_scale.stride(1),
        sm_scale=1.0,
        C=head_dim,
        BLK=BLKK,
        num_warps=4,
        num_stages=1,
    )
    return q_int8, q_scale, k_int8, k_scale
