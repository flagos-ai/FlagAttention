# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Upstream: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/wall_attn

"""Ascend kernels for the public Wall attention API.

Global and split kernels handle dense FP16/BF16 inputs with K=V=64.
The fused forward covers other supported inputs and saves FP32 statistics
for the tiled backward.
"""

import math

import torch
import triton

from flag_attn.FLA.wall_attn import parallel as _generic
from flag_attn.FLA.wall_attn.parallel import RCP_LN2

from .backward import cast_output, wall_attn_backward
from .cumsum import gate_prefix_scan
from .global_fwd import _wall_global_impl
from .kernels import _wall_rescale_bounds, _wall_stable_fwd
from .split import _wall_split_impl

_BK = 64
_BV = 64
_WARPS = 4
# Route names returned by get_last_route().
_ROUTE_FAST_SPLIT = "ascend_fast_split"
_ROUTE_FAST_GLOBAL = "ascend_fast_global"
_ROUTE_FALLBACK = "ascend_fused_fallback"
_ROUTE_PACKED = "ascend_packed"


def _route(name: str | None) -> None:
    _generic._LAST_ROUTE.set(name)


def _global_path_supported(q, k, v, g, sink_bias, scale, window_size):
    """Additional requirements after the common dense forward checks."""
    B, T, HQ, _ = q.shape
    return (
        sink_bias is None
        and window_size is None
        and 128 <= T <= 4096
        and T % 128 == 0
        and B > 0
        and B * HQ * (T // 128) <= 32768
        and isinstance(scale, (int, float))
        and math.isfinite(scale)
    )


def _fast_path_supported(q, k, v, g, g_scalar, sink_bias, window_size, cu_seqlens) -> bool:
    if cu_seqlens is not None:
        return False
    tensors = (q, k, v, g)
    if any(tensor.ndim != 4 for tensor in tensors):
        return False
    if any(tensor.layout != torch.strided for tensor in tensors):
        return False
    if q.device.type != "npu" or any(tensor.device != q.device for tensor in tensors):
        return False
    if q.shape[-1] != _BK or k.shape[-1] != _BK or v.shape[-1] != _BV:
        return False
    if q.shape != g.shape or q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        return False
    if min(q.shape[:3]) <= 0 or k.shape[2] <= 0:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not (k.dtype == v.dtype == g.dtype == q.dtype):
        return False
    if q.shape[2] % k.shape[2] != 0:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and g.is_contiguous()):
        return False
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        return False
    if sink_bias is not None and (
        sink_bias.device != q.device
        or sink_bias.layout != torch.strided
        or not sink_bias.is_contiguous()
    ):
        return False
    if g_scalar is not None:
        return False
    if window_size is not None and window_size <= 0:
        return False
    return True


def _wall_fwd_impl(q, k, v, g, sink_bias, scale, window_size, g_scalar, *, save_stats=False, buffers=None):
    """Run the fused Ascend fallback with base-2 gate prefixes."""
    B, T, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[2]
    if buffers is None:
        o = v.new_empty((B, T, HQ, V), dtype=torch.float32 if save_stats else v.dtype)
        stats = q.new_empty((B, T, HQ, 2), dtype=torch.float32) if save_stats else None
        P = c = None
    else:
        o, stats, P, c = buffers
    if not q.numel():
        if save_stats:
            P = torch.empty_like(g, dtype=torch.float32)
            c = torch.empty_like(g_scalar, dtype=torch.float32) if g_scalar is not None else None
        return o, stats, P, c
    P = gate_prefix_scan(g, RCP_LN2, out=P)
    if g_scalar is not None:
        c = gate_prefix_scan(g_scalar, RCP_LN2, out=c)
    bt, bs = 32, 64
    bk = max(16, triton.next_power_of_2(K))
    bv = min(_BV, max(16, triton.next_power_of_2(V)))
    tiles = triton.cdiv(T, bt)
    safe = torch.empty((B * HQ, tiles), dtype=torch.int32, device=q.device)
    w = window_size if window_size is not None else 0
    _wall_rescale_bounds[tiles, B * HQ](
        q, k, P, safe, T, w, H=H, HQ=HQ, K=K, BT=bt, BS=bt, BK=bk,
        USE_WINDOW=window_size is not None,
    )
    _wall_stable_fwd[triton.cdiv(V, bv), tiles, B * HQ](
        q,
        k,
        v,
        P,
        c if c is not None else q,
        sink_bias if sink_bias is not None else q,
        safe,
        o,
        stats if stats is not None else o,
        scale,
        T,
        w,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        BT=bt,
        BS=bs,
        BK=bk,
        BV=bv,
        USE_WINDOW=window_size is not None,
        USE_SINK=sink_bias is not None,
        USE_SCALAR_G=c is not None,
        SAVE_STATS=save_stats,
        num_warps=_WARPS,
    )
    return o, stats, P, c


class _WallAscendFunction(torch.autograd.Function):
    """Save gate prefixes, FP32 output and row statistics for backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, sink_bias, scale, window_size, g_scalar, offsets):
        if offsets is not None:
            o, stats, P, c = _wall_packed_forward(
                q, k, v, g, sink_bias, scale, window_size, g_scalar, offsets, save_stats=True,
            )
        else:
            o, stats, P, c = _wall_fwd_impl(q, k, v, g, sink_bias, scale, window_size, g_scalar, save_stats=True)
        ctx.save_for_backward(q, k, v, o, P, stats, sink_bias, c)
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.offsets = (0, q.shape[1]) if offsets is None else offsets
        ctx.gate_dtype = g.dtype
        ctx.scalar_dtype = g_scalar.dtype if g_scalar is not None else None
        return cast_output(o, v.dtype)

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, P, stats, sink_bias, c = ctx.saved_tensors
        dq, dk, dv, dg, dsink, dg_scalar = wall_attn_backward(
            q, k, v, o, P, c, stats, sink_bias, do, ctx.scale, ctx.window_size,
            ctx.offsets, ctx.gate_dtype, ctx.scalar_dtype,
        )
        return dq, dk, dv, dg, dsink, None, None, dg_scalar, None


def _validate_forward(q, k, v, g, sink_bias, g_scalar, scale, window_size):
    tensors = (q, k, v, g)
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("Wall attention requires four-dimensional Q, K, V and g")
    if q.shape != g.shape:
        raise ValueError("`g` must be [B, T, HQ, K] matching `q`")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q, K and V must agree on batch, sequence and head shapes")
    if not k.shape[2] or not q.shape[2] or q.shape[2] % k.shape[2]:
        raise ValueError("HQ must be divisible by a positive number of KV heads")
    if not 0 < q.shape[-1] <= 256 or not v.shape[-1]:
        raise ValueError("Wall attention requires 0 < K <= 256 and V > 0")
    operands = (*tensors, *(t for t in (sink_bias, g_scalar) if t is not None))
    if q.device.type != "npu" or any(t.device != q.device or t.layout != torch.strided for t in operands):
        raise ValueError("Wall attention requires strided tensors on the same NPU")
    if any(t.dtype not in (torch.float16, torch.bfloat16, torch.float32) for t in operands):
        raise TypeError("Wall attention on Ascend supports FP16, BF16 and FP32 tensors")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("Wall attention requires a finite numeric scale")
    if window_size is not None and (not isinstance(window_size, int) or window_size <= 0):
        raise ValueError("`window_size` must be a positive integer")


def _wall_dense_forward(q, k, v, g, sink_bias, scale, window_size, g_scalar, fast):
    if fast:
        if _global_path_supported(q, k, v, g, sink_bias, scale, window_size):
            return _wall_global_impl(q, k, v, g, scale), _ROUTE_FAST_GLOBAL
        split = _wall_split_impl(q, k, v, g, sink_bias, scale, window_size, None, g_scalar)
        if split is not None:
            return split[0], _ROUTE_FAST_SPLIT
    o, _, _, _ = _wall_fwd_impl(q, k, v, g, sink_bias, scale, window_size, g_scalar)
    return o, _ROUTE_FALLBACK


def _packed_offsets(cu_seqlens, length):
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("`cu_seqlens` must be a one-dimensional integer tensor")
    offsets = cu_seqlens.detach().cpu().tolist()
    if not offsets or offsets[0] != 0 or offsets[-1] != length or any(
        start > end for start, end in zip(offsets, offsets[1:])
    ):
        raise ValueError("`cu_seqlens` must be nondecreasing and span the packed sequence")
    return offsets


def _wall_packed_forward(q, k, v, g, sink_bias, scale, window_size, g_scalar, offsets, *, save_stats=False):
    out = v.new_empty((*q.shape[:-1], v.shape[-1]), dtype=torch.float32 if save_stats else v.dtype)
    stats = q.new_empty((*q.shape[:-1], 2), dtype=torch.float32) if save_stats else None
    P = torch.empty_like(g, dtype=torch.float32) if save_stats else None
    c = torch.empty_like(g_scalar, dtype=torch.float32) if save_stats and g_scalar is not None else None
    for start, end in zip(offsets, offsets[1:]):
        if start == end:
            continue
        rows = slice(start, end)
        qs, ks, vs, gs = (t[:, rows] for t in (q, k, v, g))
        scalar = g_scalar[:, rows] if g_scalar is not None else None
        fast = _fast_path_supported(qs, ks, vs, gs, scalar, sink_bias, window_size, None)
        if not save_stats and fast and _global_path_supported(qs, ks, vs, gs, sink_bias, scale, window_size):
            _wall_global_impl(qs, ks, vs, gs, scale, out=out[:, rows])
        else:
            buffers = (
                out[:, rows],
                stats[:, rows] if stats is not None else None,
                P[:, rows] if P is not None else None,
                c[:, rows] if c is not None else None,
            )
            _wall_fwd_impl(
                qs, ks, vs, gs, sink_bias, scale, window_size, scalar, save_stats=save_stats, buffers=buffers,
            )
    return out, stats, P, c


def parallel_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Wall attention forward using the Ascend kernels."""
    _route(None)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    fast = _fast_path_supported(q, k, v, g, g_scalar, sink_bias, window_size, cu_seqlens)
    if not fast:
        _validate_forward(q, k, v, g, sink_bias, g_scalar, scale, window_size)
        q, k, v, g = (t.contiguous() for t in (q, k, v, g))
        sink_bias = sink_bias.contiguous() if sink_bias is not None else None
        g_scalar = g_scalar.contiguous() if g_scalar is not None else None
        fast = _fast_path_supported(q, k, v, g, g_scalar, sink_bias, window_size, cu_seqlens)
    requires_backward = _generic._requires_backward(q, k, v, g, sink_bias, g_scalar)
    offsets = _packed_offsets(cu_seqlens, q.shape[1]) if cu_seqlens is not None else None
    if requires_backward:
        o = _WallAscendFunction.apply(q, k, v, g, sink_bias, scale, window_size, g_scalar, offsets)
        _route(_ROUTE_PACKED if offsets is not None else _ROUTE_FALLBACK)
        return o
    if offsets is not None:
        o, _, _, _ = _wall_packed_forward(q, k, v, g, sink_bias, scale, window_size, g_scalar, offsets)
        _route(_ROUTE_PACKED)
        return o
    if q.numel() == 0:
        o = v.new_empty((*q.shape[:-1], v.shape[-1]))
        _route(_ROUTE_FALLBACK)
        return o
    o, route = _wall_dense_forward(q, k, v, g, sink_bias, scale, window_size, g_scalar, fast)
    _route(route)
    return o
