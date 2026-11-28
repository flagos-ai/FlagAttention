# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Forgetting Attention with lazy runtime specialization.

GPU kernels use standard Triton. Input adaptation and prefix construction
use backend-neutral PyTorch operations shared with the ACP reference.
"""

import importlib
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flag_attn.runtime import torch_device_fn
from flag_attn.runtime.backend import resolve_operator
from .launch import capture, cdiv, pow2, record_launch


@triton.jit
def _general_starts(
    P, T, S, M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, BN: tl.constexpr, QM: tl.constexpr, NK: tl.constexpr
):
    h, b, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = group * 16 + tl.arange(0, 16)
    j = tl.arange(0, NK)
    pp = P + (b * H + h) * N
    a = tl.load(pp + N - M + r * QM, r < tl.cdiv(M, QM), other=0)
    end = tl.minimum(j * BN + BN - 1, N - 1)
    e = tl.load(pp + end, j < tl.cdiv(N, BN), other=0)
    threshold = tl.load(T)
    old = (j[None, :] < tl.cdiv(N, BN)) & ((a[:, None] - e[None, :]) < threshold)
    starts = tl.sum(old.to(tl.int32), 1) * BN
    tl.store(S + (b * H + h) * tl.cdiv(M, QM) + r, starts, r < tl.cdiv(M, QM))


def starts(prefix, threshold, M, N, H, BN, QM, fast=True):
    B = prefix.shape[0]
    NQ = cdiv(M, QM)
    out = torch.empty((B, H, NQ), device=prefix.device, dtype=torch.int32)
    grid = (H, B, cdiv(NQ, 16))
    kwargs = dict(num_warps=4)
    with torch_device_fn.device(prefix.device):
        compiled = _general_starts[grid](prefix, threshold, out, M, N, H, BN, QM, pow2(cdiv(N, BN)), **kwargs)
    record_launch(_general_starts, compiled, grid, kwargs)
    return out


prepare_starts = starts


def exact(gate, threshold, m, bn, qm, anchor_mode=False, fast=True):
    prefix = torch.cumsum(gate, dim=-1, dtype=torch.float32)
    return prefix, starts(prefix, threshold, m, gate.shape[-1], gate.shape[1], bn, qm, fast)


def _generic_prepare(q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold, prep_mode, fast):
    if head_first or seq_start is not None or not torch.is_inference_mode_enabled():
        raise NotImplementedError("preparation supports BTHD inference without seq_start")
    _validate(q, k, v, gate, head_first, sm_scale, adaptive_threshold)
    if not isinstance(adaptive_threshold, (int, float)):
        raise NotImplementedError("preparation requires a nonpositive scalar ACP threshold")
    B, M, H, D = q.shape
    N = k.shape[1]
    BN = min(128, max(16, pow2(N))) if M == 1 else (64 if D <= 64 else 128)
    QM = 1 if M == 1 else 128
    threshold = torch.tensor(float(adaptive_threshold), dtype=torch.float32, device=q.device)
    prefix, begin = exact(gate.transpose(1, 2), threshold, M, BN, QM, fast=fast)
    out = torch.empty((*q.shape[:-1], pow2(D)), device=q.device, dtype=q.dtype)
    scale = 1.0 / math.sqrt(D) if sm_scale is None else sm_scale
    return prefix, begin, out, scale, BN


def prepare(q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold, prep_mode, fast):
    return resolve_operator("prepare_forgetting_attention", __name__, "_generic_prepare")(
        q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold, prep_mode, fast
    )


def _validate(q, k, v, gate, head_first, sm_scale, adaptive_threshold):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("q/k/v must be rank four and K/V shapes must match")
    if q.device.type == "cpu" or q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("require accelerator BF16/FP16 inputs")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
        raise ValueError("q/k/v must have the same dtype and device")
    head_axis, seq_axis = (1, 2) if head_first else (2, 1)
    B, H, M = q.shape[0], q.shape[head_axis], q.shape[seq_axis]
    N = k.shape[seq_axis]
    D, HK = q.shape[-1], k.shape[head_axis]
    if D not in (16, 32, 60, 64, 100, 128) or k.shape[-1] != D:
        raise ValueError("require matching supported q/k/v head dimensions")
    if HK <= 0 or H % HK:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    if B <= 0 or H <= 0 or k.shape[0] != B or not 0 < M <= N:
        raise ValueError("require 0 < M <= N and matching positive batch/head sizes")
    expected_gate = (B, H, N) if head_first else (B, N, H)
    if gate.shape != expected_gate or gate.dtype != torch.float32 or gate.device != q.device:
        raise ValueError("gate must have the matching FP32 shape and device")
    if sm_scale is not None and (not isinstance(sm_scale, (int, float)) or not math.isfinite(sm_scale)):
        raise ValueError("scale must be a finite scalar")
    if isinstance(adaptive_threshold, (int, float)) and (
        not math.isfinite(adaptive_threshold) or adaptive_threshold > 0
    ):
        raise ValueError("require a finite nonpositive scalar ACP threshold")


def _generic_parallel_forgetting_attn(
    q,
    k,
    v,
    log_fgate,
    *,
    head_first=False,
    seq_start=None,
    sm_scale=None,
    adaptive_threshold=None,
):
    _validate(q, k, v, log_fgate, head_first, sm_scale, adaptive_threshold)
    dimension = q.shape[-1]
    if dimension not in (16, 32, 60, 64, 100, 128):
        raise ValueError("head dimension must be one of 16, 32, 60, 64, 100, 128")
    if k.shape[-1] != dimension or v.shape[-1] != dimension:
        raise ValueError("q, k and v must have the same head dimension")
    head_axis = 1 if head_first else 2
    heads, kv_heads = q.shape[head_axis], k.shape[head_axis]
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads or v.shape[head_axis] != kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    # Keep the scale of the original dimension when zero-padding q/k/v.
    scale = 1.0 / math.sqrt(dimension) if sm_scale is None else sm_scale
    padded = 1 << (dimension - 1).bit_length()
    if dimension != padded:
        q, k, v = (F.pad(t, (0, padded - dimension)) for t in (q, k, v))
    groups = heads // kv_heads
    if groups != 1:
        k, v = (t.repeat_interleave(groups, dim=head_axis) for t in (k, v))
    entry = importlib.import_module(".naive", __package__).forgetting_attention
    with torch_device_fn.device(q.device):
        output = entry(
            q,
            k,
            v,
            log_fgate,
            head_first=head_first,
            seq_start=seq_start,
            sm_scale=scale,
            adaptive_threshold=adaptive_threshold,
        )
    return output if dimension == padded else output[..., :dimension].contiguous()


def parallel_forgetting_attn(
    q,
    k,
    v,
    log_fgate,
    *,
    head_first=False,
    seq_start=None,
    sm_scale=None,
    adaptive_threshold=None,
):
    """Run causal Forgetting Attention through the selected runtime backend."""
    return resolve_operator("parallel_forgetting_attn", __name__, "_generic_parallel_forgetting_attn")(
        q,
        k,
        v,
        log_fgate,
        head_first=head_first,
        seq_start=seq_start,
        sm_scale=sm_scale,
        adaptive_threshold=adaptive_threshold,
    )


forgetting_attention = parallel_forgetting_attn
