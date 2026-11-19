# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend forward specializations for the public Wall attention API.

Dense FP16/BF16 inputs with K=V=64 use Ascend kernels; scalar gates and other
contracts use the generic implementation. Backward delegates to the generic
kernels and has not been validated on Ascend.
"""

import math

import torch
import triton

from flag_attn.FLA.cumsum import chunk_global_cumsum
from flag_attn.FLA.wall_attn import parallel as _generic
from flag_attn.FLA.wall_attn.parallel import RCP_LN2

from .cumsum import gate_prefix_sum, gate_prefix_sum_scalar
from .global_fwd import _wall_global_impl
from .kernels import _wall_fwd_kernel
from .split import _wall_split_impl

_BT = 128
_BS = 128
_BK = 64
_BV = 64
_WARPS = 4
#: Route names reported through the generic module's ``get_last_route()`` so the
#: repo benchmark keeps reporting something meaningful on Ascend.
_ROUTE_FAST = "ascend_fast"
#: Dense forward with materialized scores for inputs outside the global path.
_ROUTE_FAST_SPLIT = "ascend_fast_split"
_ROUTE_FAST_GLOBAL = "ascend_fast_global"
_ROUTE_FALLBACK = "official_fallback"


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
        # the scalar-gate prefix kernel crashes the bishengir compiler on this
        # CANN build, so keep that combination on the generic implementation.
        return False
    if window_size is not None and window_size <= 0:
        return False
    return True


def _wall_fwd_impl(q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar):
    """Run the Ascend kernel and return ``(o, P, lse, sink_scaled, c)``."""
    P = gate_prefix_sum(g, scale=RCP_LN2)
    c = (
        gate_prefix_sum_scalar(g_scalar) * RCP_LN2
        if g_scalar is not None
        else None
    )
    sink_bias_scaled = sink_bias * RCP_LN2 if sink_bias is not None else None

    B, T, HQ, K = q.shape
    V = v.shape[-1]
    H = k.shape[2]
    o = torch.empty((B, T, HQ, V), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(V, _BV), triton.cdiv(T, _BT), B * HQ)
    _wall_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        P,
        c if c is not None else q,
        sink_bias_scaled if sink_bias_scaled is not None else q,
        lse,
        scale,
        T,
        window_size if window_size is not None else 0,
        B=B,
        H=H,
        HQ=HQ,
        K=K,
        V=V,
        BT=_BT,
        BS=_BS,
        BK=_BK,
        BV=_BV,
        USE_WINDOW=window_size is not None,
        USE_SINK=sink_bias_scaled is not None,
        USE_SCALAR_G=c is not None,
        PV_FP32=q.dtype == torch.float16,
        num_warps=_WARPS,
    )
    return o, P, lse, sink_bias_scaled, c


class _WallAscendFunction(torch.autograd.Function):
    """Forward on the Ascend kernel; backward on the generic Triton kernels."""

    @staticmethod
    def forward(ctx, q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar=None):
        # The dense fast path uses the dedicated Triton gate prefix kernel.
        P = gate_prefix_sum(g, scale=RCP_LN2)
        c = (
            gate_prefix_sum_scalar(g_scalar) * RCP_LN2
            if g_scalar is not None
            else None
        )
        sink_bias_scaled = sink_bias * RCP_LN2 if sink_bias is not None else None

        B, T, HQ, K = q.shape
        V = v.shape[-1]
        H = k.shape[2]
        o = torch.empty((B, T, HQ, V), dtype=q.dtype, device=q.device)
        lse = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
        grid = (triton.cdiv(V, _BV), triton.cdiv(T, _BT), B * HQ)
        _wall_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            P,
            c if c is not None else q,
            sink_bias_scaled if sink_bias_scaled is not None else q,
            lse,
            scale,
            T,
            window_size if window_size is not None else 0,
            B=B,
            H=H,
            HQ=HQ,
            K=K,
            V=V,
            BT=_BT,
            BS=_BS,
            BK=_BK,
            BV=_BV,
            USE_WINDOW=window_size is not None,
            USE_SINK=sink_bias_scaled is not None,
            USE_SCALAR_G=c is not None,
            PV_FP32=q.dtype == torch.float16,
            num_warps=_WARPS,
        )
        ctx.save_for_backward(
            q, k, v, o, P, lse,
            sink_bias_scaled if sink_bias_scaled is not None else torch.empty(0, device=q.device, dtype=q.dtype),
            c if c is not None else torch.empty(0, device=q.device, dtype=q.dtype),
        )
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.cu_seqlens = cu_seqlens
        ctx.has_scalar_g = g_scalar is not None
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, P, lse, sink_or_empty, c_or_empty = ctx.saved_tensors
        sink_bias_scaled = sink_or_empty if sink_or_empty.numel() else None
        c = c_or_empty if ctx.has_scalar_g else None
        dq, dk, dv, dP, dsink_bias, dc = _generic.parallel_wall_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=P,
            lse=lse,
            do=do,
            sink_bias=sink_bias_scaled,
            scale=ctx.scale,
            g_scalar_cumsum=c,
            window_size=ctx.window_size,
            cu_seqlens=ctx.cu_seqlens,
        )
        dg = chunk_global_cumsum(dP, cu_seqlens=ctx.cu_seqlens, reverse=True, scale=RCP_LN2)
        dg_scalar = (
            chunk_global_cumsum(dc, cu_seqlens=ctx.cu_seqlens, reverse=True)
            if dc is not None
            else None
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, dsink_bias, None, None, None, dg_scalar


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
    """Wall attention forward (Ascend kernel) with a generic fallback."""
    _route(None)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    if _fast_path_supported(q, k, v, g, g_scalar, sink_bias, window_size, cu_seqlens):
        if _generic._requires_backward(q, k, v, g, sink_bias, g_scalar):
            o = _WallAscendFunction.apply(q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar)
            _route(_ROUTE_FAST)
            return o
        if _global_path_supported(q, k, v, g, sink_bias, scale, window_size):
            o = _wall_global_impl(q, k, v, g, scale)
            _route(_ROUTE_FAST_GLOBAL)
            return o
        split = _wall_split_impl(q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar)
        if split is not None:
            _route(_ROUTE_FAST_SPLIT)
            return split[0]
        o, _, _, _, _ = _wall_fwd_impl(q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar)
        _route(_ROUTE_FAST)
        return o
    o = _generic._parallel_wall_attn_default(
        q,
        k,
        v,
        g,
        g_scalar=g_scalar,
        sink_bias=sink_bias,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )
    _route(_ROUTE_FALLBACK)
    return o


__all__ = ["parallel_wall_attn"]
