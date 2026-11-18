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
"""Ascend backend for ACP-enabled Forgetting Attention (forward only).

The attention kernel uses the bundled reference's online softmax order and
reads grouped K/V heads directly. ACP carries the first retained key block
between query tiles. Device FP32 prefix sums can round in different directions
at adjacent block ends, even when every log gate is nonpositive.
"""

import math

import torch
import triton
import triton.language as tl

# Tile geometry is part of the numerical contract: BLOCK_N is the online-softmax
# chunk; BLOCK_M stays at 64 wherever the loop walks more than one key block,
# because a fully masked block still multiplies ``0`` by a NaN/Inf in the
# accumulator. The single-block ACP path can use 128 rows.
_BLOCK_N = 128
_BLOCK_M = 64
_BLOCK_M_SINGLE = 128
_NUM_WARPS = 4
_NUM_STAGES = 1


@triton.jit
def _decode_qk_kernel(Q, K, SCORES, N, num_groups,
                      sqb, sqh, sqd, skb, skh, skn, skd,
                      HEADS, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    """Compute independent decode scores in tiles that fit Ascend's vector buffer."""
    h, b = tl.program_id(1), tl.program_id(2)
    n = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    d = tl.arange(0, BLOCK_D)
    q = tl.load(Q + b * sqb + h * sqh + d[:, None] * sqd, mask=(d < HEAD_DIM)[:, None], other=0.0)
    k = tl.load(K + b * skb + (h // num_groups) * skh + d[:, None] * skd + n[None, :] * skn,
                mask=(d < HEAD_DIM)[:, None] & (n < N)[None, :], other=0.0)
    scores = tl.sum((q * k).to(tl.float32), axis=0)
    tl.store(SCORES + (b * HEADS + h) * N + n, scores, mask=n < N)


@triton.jit
def _fwd_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, START_INDEX, SCORES, sm_scale, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_lz, stride_lh, stride_ln,
    stride_sz, stride_sh, stride_sm,
    stride_oz, stride_oh, stride_om, stride_ok,
    M, N, P_SEQ, num_groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr, DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr, SINGLE_KEY: tl.constexpr, PRECOMPUTED: tl.constexpr,
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
    LOG_LAMBDA += off_z * stride_lz + off_h * stride_lh
    O += off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    mask_d = tl.full((BLOCK_DMODEL,), True, tl.int1)
    if HEAD_DIM != BLOCK_DMODEL:
        mask_d = offs_k < HEAD_DIM
    mask_m = tl.full((BLOCK_M,), True, tl.int1)
    if not DIVISIBLE_M:
        mask_m = offs_m < M

    if BLOCK_M == 1 and not PRECOMPUTED:
        # Load the query as a column for the decode reduction.
        q = tl.load(Q + offs_k[:, None] * stride_qk, mask=mask_d[:, None], other=0.0)
    elif BLOCK_M > 1:
        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                    mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    log_lambda_out = tl.load(LOG_LAMBDA + (P_SEQ + offs_m) * stride_ln,
                             mask=mask_m, other=0.0)

    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
    # Skip the causal mask on blocks ending at or before the first query.
    diagonal_start = ((P_SEQ + start_m * BLOCK_M + 1) // BLOCK_N) * BLOCK_N

    if HAS_SEQ_START:
        seq_start = tl.load(SEQ_START + off_z)
        lo = tl.minimum(tl.maximum(seq_start, 0), hi)
    else:
        lo = 0
        seq_start = 0

    if IS_ADAPTIVE:
        start_index = tl.load(
            START_INDEX + off_z * stride_sz + off_h * stride_sh + start_m * stride_sm
        )
        lo = tl.maximum(start_index, lo)
    lo = (lo // BLOCK_N) * BLOCK_N

    # A static one-iteration loop avoids a loop-carried accumulator in decode.
    if SINGLE_KEY:
        lo, hi = 0, 1
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        # Addresses are recomputed rather than advanced: Ascend's MLIR
        # bytecode bridge cannot serialise loop-carried pointer tensors.
        k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn)
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        log_lambda_in_ptrs = LOG_LAMBDA + (offs_n * stride_ln)

        mask_n = tl.full((BLOCK_N,), True, tl.int1)
        if not DIVISIBLE_N:
            mask_n = offs_n < N
        if not PRECOMPUTED:
            k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, other=0.0)

        if PRECOMPUTED:
            s = tl.load(SCORES + (off_z * tl.num_programs(1) + off_h) * N + offs_n,
                        mask=mask_n, other=0.0)[None, :] * qk_scale
        elif BLOCK_M > 1:
            s = tl.dot(q, k, input_precision="ieee") * qk_scale
        else:
            s = tl.sum((q * k).to(tl.float32), axis=0, keep_dims=True) * qk_scale
        s += (log_lambda_out[:, None] - log_lambda_in[None, :]) * log2e

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if start_n >= diagonal_start:
            s = tl.where((P_SEQ + offs_m[:, None]) >= offs_n[None, :], s, float("-inf"))
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

    if HAS_SEQ_START:
        is_empty_line = (offs_m + P_SEQ) < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
    else:
        acc = acc * (1.0 / l_i[:, None])

    o_ptrs = O + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _start_index_kernel(
    LOG_LAMBDA, THRESHOLD, START_INDEX, thr_scalar,
    stride_lz, stride_lh, stride_ln,
    stride_sz, stride_sh, stride_sm,
    stride_tz, stride_th,
    M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALAR_THRESHOLD: tl.constexpr,
):
    """Scan consecutive eligible blocks in key order, carrying the start per head."""
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)

    LOG_LAMBDA += off_z * stride_lz + off_h * stride_lh
    START_INDEX += off_z * stride_sz + off_h * stride_sh

    if SCALAR_THRESHOLD:
        threshold = thr_scalar
    else:
        THRESHOLD += off_z * stride_tz + off_h * stride_th
        threshold = tl.load(THRESHOLD)

    carry = 0
    for start_m in range(0, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        log_lambda_out = tl.load(LOG_LAMBDA + (P_SEQ + start_m) * stride_ln)
        start_index = carry
        offset_n = tl.minimum(start_index + BLOCK_N - 1, N - 1)
        decay = log_lambda_out - tl.load(LOG_LAMBDA + offset_n * stride_ln)
        # The block containing the anchor includes its self key. An FP32
        # prefix-sum rounding reversal must never prune that block.
        while (decay < threshold) and (offset_n < P_SEQ + start_m):
            start_index += BLOCK_N
            offset_n = tl.minimum(start_index + BLOCK_N - 1, N - 1)
            decay = log_lambda_out - tl.load(LOG_LAMBDA + offset_n * stride_ln)
        carry = start_index

        tl.store(
            START_INDEX + (start_m // BLOCK_M) * stride_sm,
            start_index.to(START_INDEX.dtype.element_ty),
        )


def _head_first_strides(tensor, head_first):
    """The kernel's ``(batch, head, seq, dim)`` strides for a caller tensor.

    ``head_first=True`` already supplies that layout; otherwise the same
    descriptor comes from permuting the caller's strides, so no view is made.
    """
    s = tensor.stride()
    return s if head_first else (s[0], s[2], s[1], s[3])


def _auto_threshold(q, k, sm_scale, N, num_groups, head_first):
    """Apply the reference's norm-based threshold on its padded reduction axis."""
    width = triton.next_power_of_2(q.shape[-1])
    if width != q.shape[-1]:
        padding = (0, width - q.shape[-1])
        q, k = torch.nn.functional.pad(q, padding), torch.nn.functional.pad(k, padding)
    if not head_first:
        q, k = q.transpose(1, 2), k.transpose(1, 2)
    max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
    max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
    if num_groups != 1:
        max_k_norm = max_k_norm.repeat_interleave(num_groups, dim=1)
    upper = max_q_norm * max_k_norm * sm_scale
    return -(2 * upper + math.log(N)) - 10


def _resolve_threshold(adaptive_threshold, q, k, sm_scale, B, H, N, num_groups,
                       head_first):
    """Return ``(value, is_scalar)``.

    A Python number is passed to the kernel as fp32 without a device tensor.
    A caller threshold tensor may need a device or dtype conversion, followed
    by a device-to-host check that it has no positive entries.
    """
    if adaptive_threshold is None:
        return None, False
    if isinstance(adaptive_threshold, str):
        if adaptive_threshold != "auto":
            raise ValueError(
                'adaptive_threshold must be "auto", a float, None or a tensor, '
                f"but got {adaptive_threshold!r}"
            )
        value = _auto_threshold(q, k, sm_scale, N, num_groups, head_first)
    elif isinstance(adaptive_threshold, (int, float)):
        value = float(adaptive_threshold)
        if value > 0:
            raise ValueError(
                "adaptive_threshold must be nonpositive for the ACP path, "
                f"but got {value!r}"
            )
        return value, True
    elif torch.is_tensor(adaptive_threshold):
        value = adaptive_threshold
    else:
        raise ValueError(
            'adaptive_threshold must be "auto", a number, None or a tensor, '
            f"but got {type(adaptive_threshold).__name__}"
        )
    value = torch.as_tensor(value, dtype=torch.float32, device=q.device)
    try:
        value = torch.broadcast_to(value, (B, H))
    except RuntimeError:
        raise RuntimeError(
            "adaptive_threshold must be broadcastable to (batch, heads)="
            f"({B}, {H}), but got {tuple(value.shape)}"
        ) from None
    # Nonnegative scale makes the automatic threshold nonpositive by construction.
    if (not isinstance(adaptive_threshold, str) or sm_scale < 0) and bool((value > 0).any()):
        raise ValueError("adaptive_threshold must be nonpositive for the ACP path")
    return value, False


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    """ACP-enabled Forgetting Attention forward pass on Ascend.

    Pass q/k/v in the same fp16 or bf16 dtype and ``log_fgate`` in fp32. The
    operator is causal and forward only, with ``0 < M <= N`` and
    ``16 <= D <= 128``. Query heads may be grouped over key/value heads.

    ACP thresholds must be nonpositive. Callers must supply nonpositive
    ``log_fgate`` values. The scan drops consecutive stale blocks before the
    first query in the tile.
    """
    if q.device.type != "npu":
        raise NotImplementedError(
            f"the Ascend implementation requires NPU tensors, got {q.device}"
        )
    if k.device != q.device or v.device != q.device or log_fgate.device != q.device:
        raise ValueError("q, k, v and log_fgate must be on the same NPU device")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k and v must have the same fp16 or bf16 dtype")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4-D, got {tuple(tensor.shape)}")
    if log_fgate.dtype != torch.float32:
        raise ValueError(f"log_fgate must be float32, got {log_fgate.dtype}")

    # Triton launches on the current device's stream.
    with torch.npu.device(q.device.index):
        forward = _ForwardOnly.apply if torch.is_grad_enabled() else _forward
        return forward(q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold)


class _ForwardOnly(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold):
        return _forward(q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Ascend forgetting attention is forward-only")


def _forward(q, k, v, log_fgate, head_first, seq_start, sm_scale, adaptive_threshold):
    if head_first:
        B, H, M, D = q.shape
        Hkv, N = k.shape[1:3]
    else:
        B, M, H, D = q.shape
        N, Hkv = k.shape[1:3]
    if k.shape[0] != B or v.shape != k.shape:
        raise ValueError(
            "q/k/v disagree on the batch size or the key/value shape: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if k.shape[-1] != D:
        raise ValueError(
            f"q/k/v must share the head dimension, got q={D}, k={k.shape[-1]}"
        )
    if not (0 < M <= N):
        raise ValueError(f"require 0 < M <= N, got M={M}, N={N}")
    if H <= 0 or Hkv <= 0 or H % Hkv:
        raise ValueError(
            "query heads must be a positive multiple of key/value heads, got "
            f"q heads={H}, k/v heads={Hkv}"
        )
    expected_gate = (B, H, N) if head_first else (B, N, H)
    if tuple(log_fgate.shape) != expected_gate:
        raise ValueError(f"log_fgate must be {expected_gate}, got {tuple(log_fgate.shape)}")
    if not (16 <= D <= 128):
        raise ValueError(
            f"head dimension must be in [16, 128]: the dot needs at least 16 "
            f"lanes and BLOCK_DMODEL is capped at 128, got {D}"
        )

    block_d = 1 << (D - 1).bit_length()

    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    sq, sk, sv = (_head_first_strides(t, head_first) for t in (q, k, v))
    gh = log_fgate if head_first else log_fgate.transpose(1, 2)

    scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    if isinstance(scale, torch.Tensor):
        raise ValueError(
            "sm_scale must be a Python number, got a tensor; pass float(sm_scale)"
        )
    if not math.isfinite(scale):
        raise ValueError(f"sm_scale must be a finite number, got {sm_scale!r}")

    num_groups = H // Hkv
    threshold, scalar_threshold = _resolve_threshold(
        adaptive_threshold, q, k, scale, B, H, N, num_groups, head_first
    )

    has_seq_start = seq_start is not None
    if has_seq_start:
        if tuple(seq_start.shape) != (B,):
            raise ValueError(f"seq_start must have shape ({B},)")
        seq_start = seq_start.to(device=q.device, dtype=torch.int64).contiguous()
        gate_mask = torch.arange(N, device=q.device)[None, None, :] < seq_start[:, None, None]
        gh = gh.masked_fill(gate_mask, 0.0)
    # For one key the prefix is the gate itself. Keep the ordinary attention
    # arithmetic so that signed zeros, non-finites and score overflow propagate.
    log_lambda = gh if N == 1 else torch.cumsum(gh, dim=-1, dtype=torch.float32)

    block_n = _BLOCK_N if M > 1 else min(128, max(16, triton.next_power_of_2(N)))
    single_block = threshold is not None and N <= block_n
    block_m = _BLOCK_M_SINGLE if single_block and not has_seq_start else _BLOCK_M
    if M == 1:
        block_m = 1

    # ACP retains the block containing its first query's self key. With one
    # key block the start is therefore zero, independently of the prefix bits.
    use_adaptive = threshold is not None and not single_block
    num_tiles = triton.cdiv(M, block_m)
    start_index = None
    start_index_strides = (1, 1, 1)
    if use_adaptive:
        start_index = torch.empty((B, H, num_tiles), device=q.device, dtype=torch.int32)
        start_index_strides = start_index.stride()
        thr_tensor = None if scalar_threshold else threshold
        thr_strides = (1, 1) if scalar_threshold else threshold.stride()
        _start_index_kernel[(H, B)](
            log_lambda, thr_tensor, start_index, threshold if scalar_threshold else 0.0,
            *log_lambda.stride(), *start_index_strides, *thr_strides,
            M, N, N - M, BLOCK_M=block_m, BLOCK_N=block_n,
            SCALAR_THRESHOLD=scalar_threshold, num_warps=1,
        )

    out = torch.empty_like(q)
    so = _head_first_strides(out, head_first)

    # A full 128-by-128 decode product exceeds the 192 KiB vector buffer.
    # Compute independent QK scores in smaller key tiles. Softmax still uses
    # the same 128 keys per block and its original accumulation order.
    precompute = M == 1 and N > 1 and block_d == 128
    scores = None
    if precompute:
        scores = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        _decode_qk_kernel[(triton.cdiv(N, 16), H, B)](
            q, k, scores, N, num_groups, sq[0], sq[1], sq[3], *sk,
            H, D, block_d, 16, num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        )
    _fwd_kernel[(num_tiles, H, B)](
        q, k, v, log_lambda, seq_start, start_index, scores, scale, out,
        *sq, *sk, *sv, *log_lambda.stride(),
        *start_index_strides, *so,
        M, N, N - M, num_groups,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_DMODEL=block_d, HEAD_DIM=D,
        HAS_SEQ_START=has_seq_start,
        IS_ADAPTIVE=use_adaptive,
        DIVISIBLE_M=M % block_m == 0, DIVISIBLE_N=N % block_n == 0, SINGLE_KEY=N == 1, PRECOMPUTED=precompute,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )

    return out
