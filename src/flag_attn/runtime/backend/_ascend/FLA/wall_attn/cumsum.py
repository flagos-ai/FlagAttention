# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Triton implementation of the per-channel gate prefix sum.

``P[b, t, h, k] = sum_{t' <= t} g[b, t', h, k]`` for every ``(b, h, k)`` lane.

The sequence is walked in tiles of ``BT`` positions: the in-tile inclusive
prefix is a dot product with a lower-triangular ones matrix (``O(BT^2 BK)`` on
the cube, cheap), and a per-channel carry is accumulated across tiles.  That
keeps the whole scan in one Triton kernel with a fixed launch shape — no
library cumsum involved.
"""

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
):
    i_k = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh - i_b * HQ

    rk = i_k * BK + tl.arange(0, BK)
    rt_local = tl.arange(0, BT)
    m_k = rk < K
    # lower-triangular ones: prefix = L @ x
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
        tl.store(out + ptrs, c, mask=mask)
        carry += tl.sum(x, 0)


def gate_prefix_sum(g: torch.Tensor) -> torch.Tensor:
    """``cumsum(g, dim=1)`` in float32, implemented as a Triton scan."""
    B, T, HQ, K = g.shape
    out = torch.empty((B, T, HQ, K), dtype=torch.float32, device=g.device)
    BK = 64
    grid = (triton.cdiv(K, BK), B * HQ)
    _prefix_sum_kernel[grid](g, out, T, HQ=HQ, K=K, BK=BK, BT=64)
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
        # column 0 carries the values, the other columns stay zero so the dot is legal
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
