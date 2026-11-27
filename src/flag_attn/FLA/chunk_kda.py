# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the source
# repository: https://github.com/fla-org/flash-linear-attention

"""Portable Kimi Delta Attention inference using ordinary Triton primitives.

Each program keeps a tile of the recurrent state in registers. This fallback
prioritizes portability and numerical stability over prefill throughput.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_attn import runtime

__all__ = ["chunk_kda_fwd_infer"]


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=["T"])
def _kda_recurrent_kernel(
    Q,
    K_in,
    V_in,
    G,
    Beta,
    A_log,
    Dt_bias,
    H0,
    O,
    HT,
    Cu,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    SCALE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    VARLEN: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    iv, inh = tl.program_id(0), tl.program_id(1)
    sequence, head = inh // HV, inh % HV
    qhead = head // (HV // H)
    if VARLEN:
        begin = tl.load(Cu + sequence).to(tl.int64)
        end = tl.load(Cu + sequence + 1).to(tl.int64)
    else:
        begin = sequence.to(tl.int64) * T
        end = begin + T

    keys = tl.arange(0, BK)
    values = iv * BV + tl.arange(0, BV)
    mask_k, mask_v = keys < K, values < V
    state_base = inh.to(tl.int64) * K * V
    if STATE_V_FIRST:
        state_offsets = state_base + values[:, None] * K + keys[None, :]
    else:
        state_offsets = state_base + keys[None, :] * V + values[:, None]
    state_mask = mask_v[:, None] & mask_k[None, :]
    if HAS_INITIAL_STATE:
        state = tl.load(H0 + state_offsets, state_mask, other=0).to(tl.float32)
    else:
        state = tl.full((BV, BK), 0, tl.float32)

    gate_scale = tl.exp(tl.load(A_log + head).to(tl.float32))
    if HAS_BIAS:
        bias = tl.load(Dt_bias + head * K + keys, mask_k, other=0).to(tl.float32)
    else:
        bias = tl.full((BK,), 0, tl.float32)
    for token in range(begin, end):
        q_offsets = (token * H + qhead) * K + keys
        gate_offsets = (token * HV + head) * K + keys
        value_offsets = (token * HV + head) * V + values
        q = tl.load(Q + q_offsets, mask_k, other=0).to(tl.float32)
        k = tl.load(K_in + q_offsets, mask_k, other=0).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6)
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        gate_input = tl.load(G + gate_offsets, mask_k, other=0).to(tl.float32) + bias
        if SAFE_GATE:
            log_gate = LOWER_BOUND * tl.sigmoid(gate_scale * gate_input)
        else:
            softplus = tl.maximum(gate_input, 0) + tl.log(1 + tl.exp(-tl.abs(gate_input)))
            log_gate = -gate_scale * softplus
        decay = tl.exp(log_gate)
        beta = tl.sigmoid(tl.load(Beta + token * HV + head).to(tl.float32))
        v = tl.load(V_in + value_offsets, mask_v, other=0).to(tl.float32)
        state *= decay[None, :]
        residual = (v - tl.sum(state * k[None, :], 1)) * beta
        state += residual[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1) * SCALE
        tl.store(O + value_offsets, out, mask_v)
    if STORE_FINAL_STATE:
        tl.store(HT + state_offsets, state, state_mask)


def _validate_inputs(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    state_v_first,
    cu_seqlens,
    chunk_indices,
    chunk_size,
    safe_gate,
    lower_bound,
    A_log,
    dt_bias,
):
    if torch.is_grad_enabled():
        raise RuntimeError("KDA only supports inference/no-grad mode")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype != q.dtype or tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("q, k, v, g, beta must have the same FP16, BF16 or FP32 dtype")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        raise ValueError("q/k/v/g must be 4D, beta must be 3D, and q/k shapes must match")
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    if B <= 0 or H <= 0 or K <= 0 or HV <= 0 or V <= 0 or HV % H:
        raise ValueError("head dimensions must be positive and the q/k head count must divide the value head count")
    if v.shape[:2] != (B, T) or g.shape != (B, T, HV, K) or beta.shape != (B, T, HV):
        raise ValueError("v, g and beta shapes must match the batch, sequence and value heads")
    if scale is not None and not math.isfinite(float(scale)):
        raise ValueError("scale must be finite")
    if safe_gate and (lower_bound is None or not math.isfinite(float(lower_bound)) or lower_bound >= 0):
        raise ValueError("safe_gate requires a finite negative lower_bound")
    for name, tensor, shape in (("A_log", A_log, (HV,)), ("dt_bias", dt_bias, (HV, K))):
        if tensor is None and name == "dt_bias":
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.shape != shape:
            raise ValueError(f"{name} must be a float32 tensor with shape {shape}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    N = B
    if cu_seqlens is not None:
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens requires B=1 and a 1D tensor with at least two entries")
        if cu_seqlens.device != q.device or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be an int32/int64 tensor on the input device")
        N = cu_seqlens.numel() - 1
    if chunk_indices is not None:
        if (
            chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_indices.device != q.device
            or chunk_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("chunk_indices must have shape [num_chunks, 2] and integer dtype on the input device")
    if initial_state is not None:
        shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
        if initial_state.shape != shape or initial_state.device != q.device:
            raise ValueError(f"initial_state must have shape {shape} on the input device")
        if initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("initial_state must be FP16, BF16 or FP32")


def chunk_kda_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize q/k, sigmoid beta, compute the gate, and apply the KDA recurrence.

    The state is FP32, with [N, HV, V, K] or [N, HV, K, V] layout selected by
    state_v_first. chunk_size and chunk_indices retain the chunk API contract;
    the portable recurrence uses sequence boundaries directly.
    """
    _validate_inputs(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        state_v_first,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        A_log,
        dt_bias,
    )
    q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias = (
        tensor.contiguous() if tensor is not None else None
        for tensor in (q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)
    )
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    output = torch.empty_like(v)
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    final_state = torch.empty(state_shape, device=q.device, dtype=torch.float32) if output_final_state else None
    device_context = getattr(runtime.torch_device_fn, "device", None)
    with device_context(q.device) if device_context is not None else nullcontext():
        _kda_recurrent_kernel[(triton.cdiv(V, 16), N * HV)](
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            output,
            final_state,
            cu_seqlens,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            SCALE=K**-0.5 if scale is None else float(scale),
            LOWER_BOUND=lower_bound if safe_gate else 0.0,
            SAFE_GATE=safe_gate,
            HAS_BIAS=dt_bias is not None,
            HAS_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            STATE_V_FIRST=state_v_first,
            VARLEN=cu_seqlens is not None,
            BK=triton.next_power_of_2(K),
            BV=16,
            num_warps=4,
        )
    return output, final_state
