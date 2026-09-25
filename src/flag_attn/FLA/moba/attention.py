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

"""Dense and routed attention kernels for Triton MoBA."""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.moba.routing import _copy_triton, cdiv


_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _varlen_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_lse_h,
    stride_lse_t,
    num_heads: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    MAX_SEQLEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    program = tl.program_id(0)
    query_block = program % NUM_BLOCKS
    batch_head = program // NUM_BLOCKS
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads
    kv_head = head // GROUP_SIZE

    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM
    q = tl.load(
        q_ptr + (q_start + offs_m[:, None]) * stride_q_t + head * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    scale_log2 = softmax_scale * _LOG2E

    for n_start in tl.range(0, MAX_SEQLEN_K, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_len
        k = tl.load(
            k_ptr + (k_start + offs_n[:, None]) * stride_k_t + kv_head * stride_k_h + offs_d[None, :] * stride_k_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale_log2
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            # FlashAttention uses a bottom-right aligned causal mask for
            # unequal query/key lengths.  MoBA's causal branch has equal
            # lengths, but retaining this definition keeps the primitive sane.
            valid &= offs_n[None, :] <= (offs_m[:, None] + k_len - q_len)
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp2(row_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        acc *= alpha[:, None]

        v = tl.load(
            v_ptr + (k_start + offs_n[:, None]) * stride_v_t + kv_head * stride_v_h + offs_d[None, :] * stride_v_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc += tl.dot(probabilities.to(v.dtype), v, out_dtype=tl.float32)
        row_max = new_max

    acc /= row_sum[:, None]
    tl.store(
        out_ptr + (q_start + offs_m[:, None]) * stride_o_t + head * stride_o_h + offs_d[None, :] * stride_o_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )
    lse = (row_max + tl.log2(row_sum)) / _LOG2E
    tl.store(
        lse_ptr + head * stride_lse_h + (q_start + offs_m) * stride_lse_t,
        lse,
        mask=mask_m,
    )


@triton.jit
def _varlen_attn_bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    dout_ptr,
    lse_ptr,
    dq_ptr,
    cu_q_ptr,
    cu_k_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_do_t,
    stride_do_h,
    stride_do_d,
    stride_lse_h,
    stride_lse_t,
    stride_dq_t,
    stride_dq_h,
    stride_dq_d,
    num_heads: tl.constexpr,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    MAX_SEQLEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    program = tl.program_id(0)
    query_block = program % NUM_BLOCKS
    batch_head = program // NUM_BLOCKS
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM
    q_offsets = (q_start + offs_m[:, None]) * stride_q_t + head * stride_q_h + offs_d[None, :] * stride_q_d
    o_offsets = (q_start + offs_m[:, None]) * stride_o_t + head * stride_o_h + offs_d[None, :] * stride_o_d
    do_offsets = (q_start + offs_m[:, None]) * stride_do_t + head * stride_do_h + offs_d[None, :] * stride_do_d
    q = tl.load(q_ptr + q_offsets, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    out = tl.load(out_ptr + o_offsets, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    dout = tl.load(dout_ptr + do_offsets, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    delta = tl.sum(out.to(tl.float32) * dout.to(tl.float32), axis=1)
    lse = tl.load(
        lse_ptr + head * stride_lse_h + (q_start + offs_m) * stride_lse_t,
        mask=mask_m,
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    scale_log2 = softmax_scale * _LOG2E

    for n_start in tl.range(0, MAX_SEQLEN_K, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_len
        k_offsets = (k_start + offs_n[:, None]) * stride_k_t + head * stride_k_h + offs_d[None, :] * stride_k_d
        v_offsets = (k_start + offs_n[:, None]) * stride_v_t + head * stride_v_h + offs_d[None, :] * stride_v_d
        k = tl.load(k_ptr + k_offsets, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptr + v_offsets, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale_log2
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid &= offs_n[None, :] <= offs_m[:, None] + k_len - q_len
        p = tl.exp2(scores - lse[:, None] * _LOG2E)
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(dout, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)

    dq *= softmax_scale
    tl.store(
        dq_ptr + (q_start + offs_m[:, None]) * stride_dq_t + head * stride_dq_h + offs_d[None, :] * stride_dq_d,
        dq,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _varlen_attn_bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    dout_ptr,
    lse_ptr,
    dk_ptr,
    dv_ptr,
    cu_q_ptr,
    cu_k_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_o_t,
    stride_o_h,
    stride_o_d,
    stride_do_t,
    stride_do_h,
    stride_do_d,
    stride_lse_h,
    stride_lse_t,
    stride_dk_t,
    stride_dk_h,
    stride_dk_d,
    stride_dv_t,
    stride_dv_h,
    stride_dv_d,
    num_heads: tl.constexpr,
    softmax_scale,
    HEAD_DIM: tl.constexpr,
    MAX_SEQLEN_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    program = tl.program_id(0)
    key_block = program % NUM_BLOCKS
    batch_head = program // NUM_BLOCKS
    batch = batch_head // num_heads
    head = batch_head - batch * num_heads
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start

    offs_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < k_len
    mask_d = offs_d < HEAD_DIM
    k_offsets = (k_start + offs_n[:, None]) * stride_k_t + head * stride_k_h + offs_d[None, :] * stride_k_d
    k = tl.load(k_ptr + k_offsets, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    scale_log2 = softmax_scale * _LOG2E

    for m_start in tl.range(0, MAX_SEQLEN_Q, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < q_len
        q_offsets = (q_start + offs_m[:, None]) * stride_q_t + head * stride_q_h + offs_d[None, :] * stride_q_d
        o_offsets = (q_start + offs_m[:, None]) * stride_o_t + head * stride_o_h + offs_d[None, :] * stride_o_d
        do_offsets = (q_start + offs_m[:, None]) * stride_do_t + head * stride_do_h + offs_d[None, :] * stride_do_d
        q = tl.load(q_ptr + q_offsets, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        out = tl.load(out_ptr + o_offsets, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        dout = tl.load(dout_ptr + do_offsets, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        delta = tl.sum(out.to(tl.float32) * dout.to(tl.float32), axis=1)
        lse = tl.load(
            lse_ptr + head * stride_lse_h + (q_start + offs_m) * stride_lse_t,
            mask=mask_m,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale_log2
        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid &= offs_n[None, :] <= offs_m[:, None] + k_len - q_len
        p = tl.exp2(scores - lse[:, None] * _LOG2E)
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(
            dout,
            tl.trans(
                tl.load(
                    v_ptr + (k_start + offs_n[:, None]) * stride_v_t + head * stride_v_h + offs_d[None, :] * stride_v_d,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0.0,
                )
            ),
            out_dtype=tl.float32,
        )
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p.to(dout.dtype)), dout, out_dtype=tl.float32)

    dk *= softmax_scale
    tl.store(
        dk_ptr + (k_start + offs_n[:, None]) * stride_dk_t + head * stride_dk_h + offs_d[None, :] * stride_dk_d,
        dk,
        mask=mask_n[:, None] & mask_d[None, :],
    )
    tl.store(
        dv_ptr + (k_start + offs_n[:, None]) * stride_dv_t + head * stride_dv_h + offs_d[None, :] * stride_dv_d,
        dv,
        mask=mask_n[:, None] & mask_d[None, :],
    )


def _attention_launch_config(
    head_dim: int,
    device: torch.device,
) -> tuple[int, int, int, int, int]:
    # tl.dot requires a reduction dimension of at least 16. Padding also lets
    # the same kernels support every head dimension up to FlashMoBA's D=256.
    block_d = max(16, triton.next_power_of_2(head_dim))
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    gpu_arch = torch.cuda.get_device_capability(device_index)

    if gpu_arch == (9, 0):
        # Preserve the launch policy tuned and validated on SM90/H100.
        if head_dim <= 64:
            return 64, 64, block_d, 4, 3
        if head_dim <= 128:
            return 128, 128, block_d, 8, 3
        # A 128x128 tile with BLOCK_D=256 exceeds Hopper's shared-memory budget.
        return 64, 64, block_d, 8, 2

    # Conservative fallback for every non-SM90 CUDA architecture. In
    # particular, the 32x32 D=256 tile remains below the tighter shared-memory
    # limits found on some GPUs, without maintaining per-model tables.
    if head_dim <= 128:
        return 64, 64, block_d, 4, 2
    return 32, 32, block_d, 4, 1


def _triton_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    group_size = num_heads // num_kv_heads
    num_sequences = cu_seqlens_q.numel() - 1
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty((num_heads, total_q), device=q.device, dtype=torch.float32)
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    num_blocks = cdiv(max_seqlen_q, block_m)
    grid = (num_blocks * num_sequences * num_heads,)
    _varlen_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        num_heads=num_heads,
        GROUP_SIZE=group_size,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        MAX_SEQLEN_K=max_seqlen_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=num_blocks,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def _triton_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    _, num_heads, head_dim = q.shape
    num_sequences = cu_seqlens_q.numel() - 1
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    # 128-wide backward tiles exceed the H100 shared-memory budget for D=128.
    # Forward benefits from them, while backward is feasible at 64x64.
    block_m = min(block_m, 64)
    block_n = min(block_n, 64)
    common = (
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
    )
    strides = (
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
    )
    num_q_blocks = cdiv(max_seqlen_q, block_m)
    dq_grid = (num_q_blocks * num_sequences * num_heads,)
    _varlen_attn_bwd_dq_kernel[dq_grid](
        *common,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        *strides,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        num_heads=num_heads,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        MAX_SEQLEN_K=max_seqlen_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=num_q_blocks,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    num_k_blocks = cdiv(max_seqlen_k, block_n)
    dkdv_grid = (num_k_blocks * num_sequences * num_heads,)
    _varlen_attn_bwd_dkdv_kernel[dkdv_grid](
        *common,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        *strides,
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        num_heads=num_heads,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        MAX_SEQLEN_Q=max_seqlen_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=num_k_blocks,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )


class _TritonVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, causal):
        scale = q.shape[-1] ** -0.5
        out, lse = _triton_varlen_forward(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
        ctx.max_q, ctx.max_k, ctx.scale, ctx.causal = max_q, max_k, scale, causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _triton_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            ctx.scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None


def _triton_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    return _TritonVarlenAttention.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


@triton.jit
def _routed_gqa_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    query_indices_ptr,
    expert_offsets_ptr,
    descriptor_experts_ptr,
    descriptor_starts_ptr,
    cu_seqlens_ptr,
    out_ptr,
    lse_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_out_t,
    stride_out_d,
    softmax_scale,
    NUM_Q_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MAX_CHUNKS_PER_SEQUENCE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    descriptor = tl.program_id(0)
    expert = tl.load(descriptor_experts_ptr + descriptor)
    if expert < 0:
        return
    route_start = tl.load(descriptor_starts_ptr + descriptor)
    expert_end = tl.load(expert_offsets_ptr + expert + 1)
    q_head = expert % NUM_Q_HEADS
    kv_head = q_head // GROUP_SIZE
    chunk_slot = expert // NUM_Q_HEADS
    batch = chunk_slot // MAX_CHUNKS_PER_SEQUENCE
    local_chunk = chunk_slot - batch * MAX_CHUNKS_PER_SEQUENCE
    sequence_start = tl.load(cu_seqlens_ptr + batch)
    key_start = sequence_start + local_chunk * CHUNK_SIZE

    offs_m = tl.arange(0, BLOCK_M)
    route_positions = route_start + offs_m
    mask_m = route_positions < expert_end
    flat_query_indices = tl.load(query_indices_ptr + route_positions, mask=mask_m, other=0)
    query_tokens = flat_query_indices // NUM_Q_HEADS
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q = tl.load(
        q_ptr + query_tokens[:, None] * stride_q_t + q_head * stride_q_h + offs_d[None, :] * stride_q_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    scale_log2 = softmax_scale * _LOG2E
    for n_start in tl.range(0, CHUNK_SIZE, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < CHUNK_SIZE
        k_values = tl.load(
            k_ptr + (key_start + offs_n[:, None]) * stride_k_t + kv_head * stride_k_h + offs_d[None, :] * stride_k_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k_values), out_dtype=tl.float32) * scale_log2
        valid = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        alpha = tl.exp2(row_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        acc *= alpha[:, None]
        v_values = tl.load(
            v_ptr + (key_start + offs_n[:, None]) * stride_v_t + kv_head * stride_v_h + offs_d[None, :] * stride_v_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc += tl.dot(probabilities.to(v_values.dtype), v_values, out_dtype=tl.float32)
        row_max = new_max

    acc /= row_sum[:, None]
    tl.store(
        out_ptr + route_positions[:, None] * stride_out_t + offs_d[None, :] * stride_out_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )
    lse = (row_max + tl.log2(row_sum)) / _LOG2E
    tl.store(lse_ptr + route_positions, lse, mask=mask_m)


def _routed_sparse_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    descriptor_experts: torch.Tensor,
    descriptor_starts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_chunks_per_sequence: int,
    chunk_size: int,
):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    head_dim = q.shape[2]
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    route_out = torch.empty((query_indices.numel(), head_dim), device=q.device, dtype=q.dtype)
    route_lse = torch.empty((query_indices.numel(),), device=q.device, dtype=torch.float32)
    _routed_gqa_attn_fwd_kernel[(descriptor_experts.numel(),)](
        q,
        k,
        v,
        query_indices,
        expert_offsets,
        descriptor_experts,
        descriptor_starts,
        cu_seqlens,
        route_out,
        route_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        route_out.stride(0),
        route_out.stride(1),
        q.shape[-1] ** -0.5,
        NUM_Q_HEADS=num_q_heads,
        GROUP_SIZE=group_size,
        MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
        CHUNK_SIZE=chunk_size,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return route_out, route_lse


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 32}, num_warps=1),
        triton.Config({"BLOCK_D": 64}, num_warps=1),
        triton.Config({"BLOCK_D": 128}, num_warps=1),
        triton.Config({"BLOCK_D": 64}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4),
    ],
    key=["head_dim", "TOPK", "ROW_BUCKET", "GPU_ARCH"],
)
@triton.jit
def _fused_merge_softmax_kernel(
    self_out_ptr,
    self_lse_ptr,
    moba_out_ptr,
    moba_lse_ptr,
    route_positions_ptr,
    out_ptr,
    final_lse_ptr,
    num_rows,
    head_dim,
    stride_self_row,
    stride_self_d,
    stride_self_lse_head,
    stride_self_lse_token,
    stride_moba_row,
    stride_moba_d,
    stride_route_rank,
    stride_route_row,
    stride_out_row,
    stride_out_d,
    stride_final_lse_head,
    stride_final_lse_token,
    num_heads: tl.constexpr,
    ROW_BUCKET: tl.constexpr,
    GPU_ARCH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    NUM_ROUTES: tl.constexpr,
):
    """
    Fused kernel to combine Self-Attention and MoBA-Attention results.

    It performs the standard FlashAttention output merging:
    O = (O_self * exp(LSE_self - LSE_max) + sum(O_moba * exp(LSE_moba - LSE_max))) / exp(LSE_new - LSE_max)

    This avoids multiple passes of reading/writing large output tensors.
    """
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    offs_k = tl.arange(0, TOPK)
    route_mask = offs_k < NUM_ROUTES
    positions = tl.load(
        route_positions_ptr + offs_k * stride_route_rank + row_id * stride_route_row,
        mask=route_mask,
        other=-1,
    )
    mask_k = route_mask & (positions >= 0)
    safe_positions = tl.where(mask_k, positions, 0)

    token = row_id // num_heads
    head = row_id - token * num_heads
    # Load LSEs and compute global Max LSE
    self_lse = tl.load(self_lse_ptr + head * stride_self_lse_head + token * stride_self_lse_token).to(tl.float32)
    moba_lse = tl.load(
        moba_lse_ptr + safe_positions,
        mask=mask_k,
        other=-float("inf"),
    ).to(tl.float32)

    max_moba = tl.max(moba_lse, axis=0)
    max_lse = tl.maximum(self_lse, max_moba)

    # Compute weights
    self_se = tl.exp(self_lse - max_lse)
    moba_se = tl.exp(moba_lse - max_lse)
    moba_se = tl.where(mask_k, moba_se, 0.0)
    total_se = self_se + tl.sum(moba_se, axis=0)
    merged_lse = tl.log(total_se) + max_lse
    tl.store(
        final_lse_ptr + head * stride_final_lse_head + token * stride_final_lse_token,
        merged_lse,
    )

    # Compute weighted output
    inv_total = 1.0 / total_se
    self_factor = self_se * inv_total
    moba_factor = moba_se * inv_total

    offs_d = tl.arange(0, BLOCK_D)

    for d_start in range(0, head_dim, BLOCK_D):
        mask_d = (d_start + offs_d) < head_dim

        self_ptrs = self_out_ptr + row_id * stride_self_row + (d_start + offs_d) * stride_self_d
        self_vals = tl.load(self_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc = self_vals * self_factor

        moba_ptrs = (
            moba_out_ptr + safe_positions[:, None] * stride_moba_row + (d_start + offs_d)[None, :] * stride_moba_d
        )
        moba_vals = tl.load(
            moba_ptrs,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        scaled = moba_vals * moba_factor[:, None]
        acc = acc + tl.sum(scaled, axis=0)

        out_ptrs = out_ptr + row_id * stride_out_row + (d_start + offs_d) * stride_out_d
        tl.store(out_ptrs, acc, mask=mask_d)


def _fused_merge_softmax_triton(
    self_out: torch.Tensor,
    self_lse: torch.Tensor,
    moba_out: torch.Tensor,
    moba_lse: torch.Tensor,
    route_positions: torch.Tensor,
    final_out: torch.Tensor,
    final_lse: torch.Tensor,
):
    """Wrapper for the fused merge kernel"""
    num_rows, head_dim = self_out.shape
    topk = route_positions.shape[0]

    if topk == 0 or moba_out.numel() == 0:
        _copy_triton(self_out, final_out)
        _copy_triton(self_lse, final_lse)
        return

    block_topk = triton.next_power_of_2(topk)

    device_index = self_out.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    gpu_arch = major * 10 + minor
    row_bucket = triton.next_power_of_2(num_rows)

    grid = (num_rows,)

    _fused_merge_softmax_kernel[grid](
        self_out,
        self_lse,
        moba_out,
        moba_lse,
        route_positions,
        final_out,
        final_lse,
        num_rows,
        head_dim,
        self_out.stride(0),
        self_out.stride(1),
        self_lse.stride(0),
        self_lse.stride(1),
        moba_out.stride(0),
        moba_out.stride(1),
        route_positions.stride(0),
        route_positions.stride(1),
        final_out.stride(0),
        final_out.stride(1),
        final_lse.stride(0),
        final_lse.stride(1),
        num_heads=self_lse.shape[0],
        TOPK=block_topk,
        NUM_ROUTES=topk,
        ROW_BUCKET=row_bucket,
        GPU_ARCH=gpu_arch,
    )
