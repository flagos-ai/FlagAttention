# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Inclusive gate prefix sums with an FP32 carry across sequence tiles."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefix_sum_kernel(
    g,
    out,
    T,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    SCALE: tl.constexpr,
):
    i_k = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh - i_b * HQ

    rk = i_k * BK + tl.arange(0, BK)
    rt_local = tl.arange(0, BT)
    m_k = rk < K
    # Multiplying by a lower-triangular ones matrix computes the local prefix.
    b_l = (rt_local[:, None] >= rt_local[None, :]).to(tl.float32)
    base = ((i_b * T) * HQ + i_hq) * K
    carry = tl.zeros([BK], dtype=tl.float32)

    for i_t in range(0, tl.cdiv(T, BT)):
        rt = i_t * BT + rt_local
        m_t = rt < T
        ptrs = base + rt[:, None] * (HQ * K) + rk[None, :]
        mask = m_t[:, None] & m_k[None, :]
        x = tl.load(g + ptrs, mask=mask, other=0.0).to(tl.float32)
        c = tl.dot(b_l, x) + carry[None, :]
        tl.store(out + ptrs, c * SCALE, mask=mask)
        carry += tl.sum(x, 0)


def gate_prefix_sum(g: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Scaled ``cumsum(g, dim=1)`` in float32, implemented as a Triton scan."""
    B, T, HQ, K = g.shape
    out = torch.empty((B, T, HQ, K), dtype=torch.float32, device=g.device)
    BK = 64
    grid = (triton.cdiv(K, BK), B * HQ)
    _prefix_sum_kernel[grid](g, out, T, HQ=HQ, K=K, BK=BK, BT=64, SCALE=scale)
    return out


__all__ = ["gate_prefix_sum", "gate_prefix_sum_scalar"]


@triton.jit
def _prefix_sum_scalar_kernel(
    g,
    out,
    T,
    HQ: tl.constexpr,
    BT: tl.constexpr,
):
    """Prefix sum for the FoX-style scalar gate, shape [B, T, HQ]."""
    i_bh = tl.program_id(0)
    i_b = i_bh // HQ
    i_hq = i_bh - i_b * HQ
    rt_local = tl.arange(0, BT)
    b_l = (rt_local[:, None] >= rt_local[None, :]).to(tl.float32)
    pad = tl.arange(0, 16)
    base = i_b * T * HQ + i_hq
    carry = 0.0
    for i_t in range(0, tl.cdiv(T, BT)):
        rt = i_t * BT + rt_local
        m_t = rt < T
        x = tl.load(g + base + rt * HQ, mask=m_t, other=0.0).to(tl.float32)
        # Pad to the Cube tile width. Only column zero carries gate values.
        tile = tl.where(pad[None, :] == 0, x[:, None], tl.zeros([BT, 16], dtype=tl.float32))
        c = tl.dot(b_l, tile) + carry
        col0 = tl.sum(tl.where(pad[None, :] == 0, c, tl.zeros([BT, 16], dtype=tl.float32)), 1)
        tl.store(out + base + rt * HQ, col0, mask=m_t)
        carry += tl.sum(x, 0)


def gate_prefix_sum_scalar(g: torch.Tensor) -> torch.Tensor:
    """``cumsum(g, dim=1)`` for a ``[B, T, HQ]`` gate, in float32."""
    B, T, HQ = g.shape
    out = torch.empty((B, T, HQ), dtype=torch.float32, device=g.device)
    grid = (B * HQ,)
    _prefix_sum_scalar_kernel[grid](g, out, T, HQ=HQ, BT=64)
    return out


@triton.jit
def _gate_scan_kernel(G, P, T, C: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
                      SCALE: tl.constexpr, REVERSE: tl.constexpr):
    batch = tl.program_id(1)
    cols = tl.program_id(0) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BT)
    carry = tl.zeros((BC,), tl.float32)
    for tile in range(tl.cdiv(T, BT)):
        step = tile * BT + rows
        ts = T - 1 - step if REVERSE else step
        offsets = ((batch * T + ts[:, None]) * C) + cols[None, :]
        valid = (step[:, None] < T) & (cols[None, :] < C)
        gate = tl.load(G + offsets, valid, 0).to(tl.float32)
        prefix = tl.cumsum(gate, axis=0) + carry[None, :]
        tl.store(P + offsets, prefix * SCALE, valid)
        carry += tl.sum(gate, axis=0)


def gate_prefix_scan(
    g: torch.Tensor, scale: float, out: torch.Tensor | None = None, *, reverse: bool = False,
) -> torch.Tensor:
    """Scan contiguous scalar or channel gates with FP32 accumulation."""
    B, T = g.shape[:2]
    C = g.shape[2] if g.ndim == 3 else g.shape[2] * g.shape[3]
    if out is None:
        out = torch.empty_like(g, dtype=torch.float32)
    bc = min(16, triton.next_power_of_2(C))
    _gate_scan_kernel[triton.cdiv(C, bc), B](g, out, T, C=C, BT=64, BC=bc, SCALE=scale, REVERSE=reverse)
    return out
