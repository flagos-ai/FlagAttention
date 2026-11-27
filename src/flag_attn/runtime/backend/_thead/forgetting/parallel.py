# Copyright 2026 FlagOS Contributors
# Copyright 2023 BAAI
# SPDX-License-Identifier: Apache-2.0

"""T-Head inference-only ACP Forgetting Attention."""

import math
from functools import lru_cache
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def maybe_contiguous(x):
    return x.contiguous() if x.stride(-1) != 1 else x


@lru_cache(maxsize=128, typed=True)
def _scalar_threshold(device_index, stream_id, batch, heads, value):
    """Reuse immutable scalar parameters; never retain caller tensors.

    The stream identity is part of the key so a cached allocation is only
    consumed on the stream that initialized it, without cross-stream waits.
    """
    scalar = torch.tensor(value, dtype=torch.float32, device=torch.device("cuda", device_index))
    return scalar.broadcast_to((batch, heads))


@torch.no_grad()
def forgetting_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_fgate: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    adaptive_threshold: Optional[Union[Literal["auto"], float, torch.Tensor]] = None,
):
    """Causal ACP forward with optional pruning and left padding.

    Inputs use BTHD (BHTD when head_first=True), FP16/BF16 Q/K/V and
    FP32 log gates. Query heads may group KV heads, D is one of
    16/32/60/64/100/128 and 0 < M <= N. Threshold accepts None
    (no pruning), "auto", or a value broadcastable to (batch, heads).
    Default scaling is 1/sqrt(D). Returns only the attention output.
    """
    for name, entry in (("q", q), ("k", k), ("v", v)):
        assert entry.dtype in (torch.float16, torch.bfloat16), f"{name} must be FP16 or BF16"
    if not head_first:
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        log_fgate = log_fgate.transpose(1, 2)
    b, h, m, d = q.shape
    n = k.shape[2]
    assert d == k.shape[-1] == v.shape[-1], "q/k/v feature dimensions must match"
    assert d in (16, 32, 60, 64, 100, 128), "unsupported head dimension"
    kv_heads = k.shape[1]
    assert kv_heads > 0 and h > 0 and h % kv_heads == 0, "query heads must be a positive multiple of KV heads"
    assert v.shape == k.shape, "k/v shapes must match"
    assert 0 < m <= n, "require 0 < query length <= KV length"
    assert log_fgate.shape == (b, h, n), "log_fgate shape must match batch, heads and KV length"
    assert log_fgate.dtype == torch.float32, "log_fgate must be FP32"
    scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale

    with torch.cuda.device(q.device):
        # Preserve scale and zero-pad feature lanes within kernel loads.
        original_d = d
        d = 1 << (d - 1).bit_length()
        # Decode reduces across the feature axis. Preserve its original lane
        # reduction layout for non-power-of-two D to keep bitwise semantics.
        padded_decode = m == 1 and d != original_d
        kernel_d = d if padded_decode else original_d
        if padded_decode:
            q, k, v = (F.pad(t, (0, d - original_d)) for t in (q, k, v))
        # Consecutive query heads address the same original KV head in the kernel.
        groups = h // kv_heads

        if adaptive_threshold is not None:
            if isinstance(adaptive_threshold, str):
                assert adaptive_threshold == "auto", 'string threshold must be "auto"'
                max_q = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
                max_k = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
                if groups != 1:
                    # Expand only B x Hkv norms, never the full K/V tensors.
                    max_k = max_k.repeat_interleave(groups, dim=1)
                logit_upper_bound = max_q * max_k * scale
                adaptive_threshold = -(2 * logit_upper_bound + math.log(n)) - 10
            if isinstance(adaptive_threshold, (int, float)):
                stream_id = torch.cuda.current_stream(q.device).cuda_stream
                adaptive_threshold = _scalar_threshold(q.device.index, stream_id, b, h, adaptive_threshold)
            else:
                adaptive_threshold = torch.as_tensor(adaptive_threshold, dtype=torch.float32, device=q.device)
                adaptive_threshold = torch.broadcast_to(adaptive_threshold, (b, h))

        has_seq_start = seq_start is not None
        if has_seq_start:
            assert seq_start.shape == (b,), "seq_start must have shape (batch,)"
            log_fgate = log_fgate.clone()
            mask = torch.arange(n, device=q.device)[None, None, :] < seq_start[:, None, None]
            log_fgate.masked_fill_(mask, 0.0)
        prefix = torch.cumsum(log_fgate, dim=-1, dtype=torch.float32)
        q, k, v = (maybe_contiguous(t) for t in (q, k, v))

        # Native T-Head configuration; no NVIDIA architecture dispatch.
        if m == 1:
            bm, bn, stages = 1, min(128, max(16, 1 << (n - 1).bit_length())), 1
            qm = 1
        else:
            qm = 64 if d <= 64 else 128
            bm, bn, stages = 64, 32, (3 if n > 128 else 2)
        grid = ((m + bm - 1) // bm, h, b)
        divisible_m, divisible_n = m % bm == 0, n % bn == 0
        prefix_strides = prefix.stride()
        threshold_strides = (0, 0) if adaptive_threshold is None else adaptive_threshold.stride()
        out = torch.empty_like(q)
        # Up to four KV tiles cannot reliably amortize the prefill pipeline.
        kernel = _fwd_pipelined_kernel if m > 1 and n > 128 else _fwd_kernel
        kernel[grid](
            q,
            k,
            v,
            prefix,
            seq_start,
            adaptive_threshold,
            scale,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *prefix_strides,
            *threshold_strides,
            *out.stride(),
            m,
            n,
            n - m,
            groups,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_DMODEL=d,
            ACTUAL_D=kernel_d,
            QM=qm,
            STAGES=stages,
            PRUNE_BLOCKS=1 << (((n + bn - 1) // bn - 1).bit_length()),
            HAS_SEQ_START=has_seq_start,
            IS_ADAPTIVE=adaptive_threshold is not None,
            DIVISIBLE_M=divisible_m,
            DIVISIBLE_N=divisible_n,
            num_warps=4,
            num_stages=stages,
        )
    if not head_first:
        out = out.transpose(1, 2)
    return out[..., :original_d].contiguous() if padded_decode else out


parallel_forgetting_attn = forgetting_attention


@triton.jit
def _fwd_pipelined_kernel(
    Q,
    K,
    V,
    LOG_LAMBDA,
    SEQ_START,
    THRESHOLD,
    sm_scale,
    O,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_log_lambda_z,
    stride_log_lambda_h,
    stride_log_lambda_n,
    stride_threshold_z,
    stride_threshold_h,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    PRUNE_BLOCKS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QM: tl.constexpr,
    STAGES: tl.constexpr,
    HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".ca")
    else:
        mask_m = offs_m < M
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".ca")
    hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0
    if IS_ADAPTIVE:
        threshold = tl.load(THRESHOLD + off_z * stride_threshold_z + off_h * stride_threshold_h)
        anchor = tl.load(LOG_LAMBDA + (P_SEQ + (start_m * BLOCK_M // QM) * QM) * stride_log_lambda_n)
        block_ids = tl.arange(0, PRUNE_BLOCKS)
        valid_blocks = block_ids < (N + BLOCK_N - 1) // BLOCK_N
        key_pos = tl.minimum((block_ids + 1) * BLOCK_N - 1, N - 1)
        endpoints = tl.load(LOG_LAMBDA + key_pos * stride_log_lambda_n, mask=valid_blocks, other=0)
        first = tl.sum((valid_blocks & (anchor - endpoints < threshold)).to(tl.int32), 0)
        lo = tl.maximum(first * BLOCK_N, lo)
    lo = lo // BLOCK_N * BLOCK_N
    offs_n_init += lo
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    log_lambda_in_ptrs = LOG_LAMBDA + offs_n_init * stride_log_lambda_n
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=STAGES, disable_licm=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        if DIVISIBLE_M:
            q = tl.load(q_ptrs, mask=(offs_k[None, :] < ACTUAL_D) & ((ACTUAL_D != BLOCK_DMODEL) | (P_SEQ + offs_m[:, None] >= start_n)), other=0, cache_modifier=".ca")
        else:
            q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < ACTUAL_D) & ((ACTUAL_D != BLOCK_DMODEL) | (P_SEQ + offs_m[:, None] >= start_n)), other=0, cache_modifier=".ca")
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, mask=offs_k[:, None] < ACTUAL_D, other=0, cache_modifier=".ca")
            log_lambda_in = _load_gate_bits(log_lambda_in_ptrs, tl.full((BLOCK_N,), True, tl.int1))
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < ACTUAL_D), other=0, cache_modifier=".ca")
            log_lambda_in = _load_gate_bits(log_lambda_in_ptrs, mask_n)
        s = tl.dot(q, k, input_precision="ieee") * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s = tl.fma(decay_bias, log2e, s)
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, float("-inf"))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(s - m_i_new[:, None])
        p_sum = tl.sum(p, 1)
        if DIVISIBLE_N:
            v = tl.load(v_ptrs, mask=offs_k[None, :] < ACTUAL_D, other=0, cache_modifier=".ca")
        else:
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < ACTUAL_D), other=0, cache_modifier=".ca")
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n
    if HAS_SEQ_START:
        is_empty_line = offs_m + P_SEQ < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
    else:
        acc = acc * (1.0 / l_i[:, None])
    if DIVISIBLE_M:
        tl.store(o_ptrs, acc.to(input_dtype), mask=offs_k[None, :] < ACTUAL_D, cache_modifier=".cg")
    else:
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None] & (offs_k[None, :] < ACTUAL_D), cache_modifier=".cg")


@triton.jit
def _load_gate_bits(ptr, valid):
    bits_ptr = ptr.to(tl.pointer_type(tl.uint16))
    low = tl.load(bits_ptr, mask=valid, other=0, cache_modifier=".ca").to(tl.uint32)
    high = tl.load(bits_ptr + 1, mask=valid, other=0, cache_modifier=".ca").to(tl.uint32)
    bits = low | (high << 16)
    return bits.to(tl.float32, bitcast=True)


# Keep the original load order for decode and short prefill.
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    LOG_LAMBDA,
    SEQ_START,
    THRESHOLD,
    sm_scale,
    O,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_log_lambda_z,
    stride_log_lambda_h,
    stride_log_lambda_n,
    stride_threshold_z,
    stride_threshold_h,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    M,
    N,
    P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr,
    PRUNE_BLOCKS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QM: tl.constexpr,
    STAGES: tl.constexpr,
    HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".ca")
    else:
        mask_m = offs_m < M
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".ca")
    hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0
    if IS_ADAPTIVE:
        threshold = tl.load(THRESHOLD + off_z * stride_threshold_z + off_h * stride_threshold_h)
        anchor = tl.load(LOG_LAMBDA + (P_SEQ + (start_m * BLOCK_M // QM) * QM) * stride_log_lambda_n)
        if BLOCK_M == 1:
            # Preserve the original decode search order while fusing its
            # result into attention instead of passing a scratch tensor.
            scan_start = 0
            scan_end = tl.minimum(BLOCK_N - 1, N - 1)
            endpoint = tl.load(LOG_LAMBDA + scan_end * stride_log_lambda_n)
            while anchor - endpoint < threshold:
                scan_start += BLOCK_N
                scan_end = tl.minimum(scan_start + BLOCK_N - 1, N - 1)
                endpoint = tl.load(LOG_LAMBDA + scan_end * stride_log_lambda_n)
            lo = tl.maximum(scan_start, lo)
        else:
            # Compare block endpoints in parallel; only nonpositive log gates are
            # valid, so the retained region is contiguous in prefix order.
            block_ids = tl.arange(0, PRUNE_BLOCKS)
            valid_blocks = block_ids < (N + BLOCK_N - 1) // BLOCK_N
            key_pos = tl.minimum((block_ids + 1) * BLOCK_N - 1, N - 1)
            endpoints = tl.load(LOG_LAMBDA + key_pos * stride_log_lambda_n, mask=valid_blocks, other=0)
            first = tl.sum((valid_blocks & (anchor - endpoints < threshold)).to(tl.int32), 0)
            lo = tl.maximum(first * BLOCK_N, lo)
    lo = lo // BLOCK_N * BLOCK_N
    offs_n_init += lo
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    log_lambda_in_ptrs = LOG_LAMBDA + offs_n_init * stride_log_lambda_n
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=STAGES, disable_licm=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        if DIVISIBLE_M:
            q = tl.load(q_ptrs, mask=offs_k[None, :] < ACTUAL_D, other=0, cache_modifier=".ca")
        else:
            q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < ACTUAL_D), other=0, cache_modifier=".ca")
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, mask=offs_k[:, None] < ACTUAL_D, other=0, cache_modifier=".ca")
            v = tl.load(v_ptrs, mask=offs_k[None, :] < ACTUAL_D, other=0, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".ca")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < ACTUAL_D), other=0, cache_modifier=".ca")
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < ACTUAL_D), other=0, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".ca")
        if BLOCK_M > 1:
            s = tl.dot(q, k, input_precision="ieee") * qk_scale
        else:
            s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask, s, float("-inf"))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(s - m_i_new[:, None])
        p_sum = tl.sum(p, 1)
        acc *= alpha[:, None]
        if BLOCK_M > 1:
            acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
        else:
            acc += tl.sum(p.T * v, axis=0, keep_dims=True)
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n
    if HAS_SEQ_START:
        is_empty_line = offs_m + P_SEQ < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
    else:
        acc = acc * (1.0 / l_i[:, None])
    if DIVISIBLE_M:
        tl.store(o_ptrs, acc.to(input_dtype), mask=offs_k[None, :] < ACTUAL_D, cache_modifier=".cg")
    else:
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None] & (offs_k[None, :] < ACTUAL_D), cache_modifier=".cg")
