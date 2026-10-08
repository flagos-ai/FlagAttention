# Copyright 2026 FlagOS Contributors
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

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

_TILE_CONFIGS = [
    triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
    for block_m, block_n in ((64, 64), (128, 64), (128, 32))
    for num_warps in (4, 8)
]


def _prune_tile_configs(configs, named_args, **kwargs):
    expected_warps = 4 if kwargs["HEAD_DIM"] == 64 else 8
    return [config for config in configs if config.num_warps == expected_warps]

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                    K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                    start_m, mask_ptrs, stride_maskn,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                    K_SCALE_BLOCK: tl.constexpr,
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=False)
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=-1.0e6)
        if not skip:
            k_mask = offs_n[None, :] < (kv_len - start_n)
            k = tl.load(K_ptrs, mask=k_mask, other=0)
            k_scale = tl.load(K_scale_ptr + start_n // K_SCALE_BLOCK)

            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            else:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]

            v = tl.load(
                V_ptrs,
                mask=offs_n[:, None] < (kv_len - start_n),
                other=0.0,
            )
            p = p.to(tl.float16)

            acc += tl.dot(p, v, out_dtype=tl.float32)
            m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        V_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_inner_static(acc, l_i, m_i, q, q_scale, qo_len, kv_len,
                           K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                           start_m, mask_ptrs, stride_maskn,
                           BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                           STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                           KV_BLOCKS: tl.constexpr, FULL_KV: tl.constexpr,
                           K_SCALE_BLOCK: tl.constexpr,
                           ):
    for block_id in tl.static_range(0, KV_BLOCKS):
        start_n = block_id * BLOCK_N
        K_block_ptrs = K_ptrs + start_n * stride_kn
        K_scale_block_ptr = K_scale_ptr + (block_id * BLOCK_N) // K_SCALE_BLOCK
        V_block_ptrs = V_ptrs + start_n * stride_vn
        mask_block = None
        skip = False
        if mask_ptrs is not None:
            if mask_ptrs.dtype.element_ty == tl.int1:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=False)
                if tl.max(mask_block) == 0:
                    skip = True
            else:
                mask_block = tl.load(mask_ptrs + start_n * stride_maskn, mask=(offs_m[:, None] < qo_len) & (offs_n[None, :] < kv_len - start_n), other=-1.0e6)
        if not skip:
            if FULL_KV:
                k = tle.load(K_block_ptrs, is_async=True)
            else:
                k_mask = offs_n[None, :] < (kv_len - start_n)
                k = tle.load(K_block_ptrs, mask=k_mask, other=0, is_async=True)
            k_scale = tl.load(K_scale_block_ptr)

            qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)

            if mask_block is not None:
                if mask_block.dtype == tl.int1:
                    qk = qk + tl.where(mask_block, 0, -1.0e6)
                else:
                    qk = qk + mask_block
            elif not FULL_KV:
                qk += tl.where(k_mask, 0, -1.0e6)

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)

            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            acc = acc * alpha[:, None]

            if FULL_KV:
                v = tle.load(V_block_ptrs, is_async=True)
            else:
                v = tle.load(
                    V_block_ptrs,
                    mask=offs_n[:, None] < (kv_len - start_n),
                    other=0,
                    is_async=True,
                )
            p = p.to(tl.float16)

            acc += tl.dot(p, v, out_dtype=tl.float32)
            m_i = m_ij
    return acc, l_i, m_i

@triton.autotune(
    configs=_TILE_CONFIGS,
    key=["qo_len", "kv_len", "H", "HEAD_DIM", "num_kv_groups", "RETURN_LSE"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    cache_results=True,
)
@triton.heuristics({
    "KV_BLOCKS": lambda args: triton.cdiv(args["kv_len"], args["BLOCK_N"]),
    "FULL_KV": lambda args: args["kv_len"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, mask, Lse,
              stride_qz, stride_qh, stride_qn,
              stride_kz, stride_kh, stride_kn,
              stride_vz, stride_vh, stride_vn,
              stride_oz, stride_oh, stride_on,
              stride_maskz, stride_maskh, stride_maskm, stride_maskn,
              qo_len, kv_len, H: tl.constexpr, num_kv_groups: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              RETURN_LSE: tl.constexpr,
              STATIC_KV: tl.constexpr,
              KV_BLOCKS: tl.constexpr,
              FULL_KV: tl.constexpr,
              Q_SCALE_BLOCK: tl.constexpr,
              K_SCALE_BLOCK: tl.constexpr,
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, Q_SCALE_BLOCK)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(kv_len, K_SCALE_BLOCK)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset + (start_m * BLOCK_M) // Q_SCALE_BLOCK
    K_ptrs = K + (off_z * stride_kz + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (off_z * stride_vz + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    if mask is None:
        mask_ptrs = None
    else:
        mask_ptrs = mask + (off_z * stride_maskz + off_h * stride_maskh) + offs_m[:, None] * stride_maskm + offs_n[None, :] * stride_maskn

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask = offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    if STATIC_KV:
        acc, l_i, m_i = _attn_fwd_inner_static(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                               start_m, mask_ptrs, stride_maskn,
                                               BLOCK_M, HEAD_DIM, BLOCK_N,
                                               4 - STAGE, offs_m, offs_n, KV_BLOCKS, FULL_KV,
                                               K_SCALE_BLOCK
                                               )
    else:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, qo_len, kv_len, K_ptrs, K_scale_ptr, V_ptrs, stride_kn, stride_vn,
                                        start_m, mask_ptrs, stride_maskn,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,
                                        4 - STAGE, offs_m, offs_n, K_SCALE_BLOCK
                                        )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < qo_len))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask = (offs_m < qo_len))

def forward(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None,
            output_dtype=torch.float16, return_lse=False, maxnreg=None):
    stage = 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    if attn_mask is not None:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    else:
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = 0, 0, 0, 0

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device='cpu')

    def grid(meta):
        return (triton.cdiv(qo_len, meta["BLOCK_M"]), h_qo, b)

    # Full-static experiment: specialize every shape by its KV block count.
    static_kv = False
    launch_options = {}
    if maxnreg is not None:
        if maxnreg <= 0:
            raise ValueError("maxnreg must be positive")
        launch_options["maxnreg"] = maxnreg

    _attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, num_kv_groups,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage, RETURN_LSE=return_lse, STATIC_KV=static_kv,
        Q_SCALE_BLOCK=128, K_SCALE_BLOCK=64,
        **launch_options)

    return o, lse
