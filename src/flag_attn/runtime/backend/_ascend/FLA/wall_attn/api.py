# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend implementation of Wall attention (forward only).

The public entry mirrors ``flag_attn.FLA.wall_attn.parallel.parallel_wall_attn``.
The forward pass uses the dedicated Ascend Triton kernel for the benchmark
contract (K = V = 64, dense non-varlen, causal + optional window/sink/scalar
gate); every other combination falls back to the generic Triton implementation.

The backward pass is **not** implemented here: it falls back to the generic
Triton dq/dkv kernels (read-only import).  Those kernels have not been validated
on Ascend yet -- treat training as unsupported until that is re-measured.
"""

import torch
import triton

from flag_attn.FLA.cumsum import chunk_global_cumsum
from flag_attn.FLA.wall_attn import parallel as _generic
from flag_attn.FLA.wall_attn.parallel import RCP_LN2

from .kernels import _wall_fwd_kernel

_BT = 64
_BS = 16
_BK = 64
_BV = 64
_WARPS = 4
#: Route names reported through the generic module's ``get_last_route()`` so the
#: repo benchmark keeps reporting something meaningful on Ascend.
_ROUTE_FAST = "ascend_fast"
_ROUTE_FALLBACK = "official_fallback"


def _route(name: str) -> None:
    try:
        _generic._LAST_ROUTE.set(name)
    except Exception:  # pragma: no cover - reporting only
        pass


def _fast_path_supported(q, k, v, g, g_scalar, sink_bias, window_size, cu_seqlens) -> bool:
    if cu_seqlens is not None:
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or g.dim() != 4:
        return False
    if q.shape[-1] != _BK or k.shape[-1] != _BK or v.shape[-1] != _BV:
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
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        return False
    if window_size is not None and window_size <= 0:
        return False
    return True


class _WallAscendFunction(torch.autograd.Function):
    """Forward on the Ascend kernel; backward on the generic Triton kernels."""

    @staticmethod
    def forward(ctx, q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar=None):
        # Fast path is non-varlen only, so the prefix is a plain cumsum over T.
        # The generic Triton chunked scan costs 1.1 ms at T=512 and 6.7 ms at
        # T=4096 on Ascend (measured), which dominated the whole operator; the
        # native torch scan is ~10x cheaper and numerically equivalent.
        P = torch.cumsum(g.float(), dim=1) * RCP_LN2
        c = (
            torch.cumsum(g_scalar.float(), dim=1) * RCP_LN2
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
    q,
    k,
    v,
    g,
    *,
    g_scalar=None,
    sink_bias=None,
    scale=None,
    window_size=None,
    cu_seqlens=None,
):
    """Wall attention forward (Ascend kernel) with a generic fallback."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    if _fast_path_supported(q, k, v, g, g_scalar, sink_bias, window_size, cu_seqlens):
        _route(_ROUTE_FAST)
        return _WallAscendFunction.apply(q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar)
    _route(_ROUTE_FALLBACK)
    return _generic._parallel_wall_attn_default(
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


__all__ = ["parallel_wall_attn"]
