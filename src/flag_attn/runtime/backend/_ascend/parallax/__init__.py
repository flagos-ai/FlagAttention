"""Ascend Triton implementation of Parallax attention."""

from __future__ import annotations

import torch

from .backward import parallax_backward
from .decode import parallax_attn_with_kvcache, parallax_decode
from .forward import parallax_forward


class ParallaxFunction(torch.autograd.Function):
    """Autograd bridge joining the Ascend forward and backward kernels."""

    @staticmethod
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        q, r, k, v = (tensor.contiguous() for tensor in (q, r, k, v))
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous()
        output, barv, d1, bart, m = parallax_forward(
            q, r, k, v, scale, cu_seqlens=cu_seqlens, window_size_left=window_size_left
        )
        ctx.save_for_backward(q, r, k, v, output, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        q, r, k, v, output, barv, d1, bart, m = ctx.saved_tensors
        grad_q, grad_r, grad_k, grad_v = parallax_backward(
            q,
            r,
            k,
            v,
            output,
            barv,
            d1,
            bart,
            m,
            grad_output,
            ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            window_size_left=ctx.window_size_left,
        )
        return grad_q, grad_r, grad_k, grad_v, None, None, None


def parallel_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Causal Parallax attention with Ascend Triton autograd support.

    Tensors use ``[B, T, H, D]`` layout. ``window_size`` is the number of
    causal keys retained (including the diagonal); ``None`` or ``-1`` disables
    the window. ``cu_seqlens`` selects
    packed variable-length sequences with batch size one.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning("head_first has been removed; use [B, T, H, D] layout")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"parallel_parallax got unexpected keyword argument(s): {unexpected}")
    if window_size is not None and (
        not isinstance(window_size, int) or (window_size != -1 and window_size <= 0)
    ):
        raise ValueError("window_size must be a positive integer, -1, or None")
    if scale is None:
        scale = k.shape[-1] ** -0.5
    window_size_left = -1 if window_size is None else int(window_size)
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)


def parallel_parallax_fwd(q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1):
    """Low-level forward interface returning output and saved normalization state."""
    if chunk_indices is not None:
        raise NotImplementedError(
            "custom chunk_indices scheduling is not implemented by the Ascend backend"
        )
    return parallax_forward(q, r, k, v, scale, cu_seqlens, window_size_left)


def parallel_parallax_bwd(
    q, r, k, v, o, barv, d1, bart, m, grad_o, scale,
    cu_seqlens=None, chunk_indices=None, window_size_left=-1,
):
    """Low-level backward interface matching the NVIDIA Parallax API."""
    if chunk_indices is not None:
        raise NotImplementedError(
            "custom chunk_indices scheduling is not implemented by the Ascend backend"
        )
    return parallax_backward(
        q, r, k, v, o, barv, d1, bart, m, grad_o, scale,
        cu_seqlens=cu_seqlens, window_size_left=window_size_left,
    )


__all__ = [
    "ParallaxFunction",
    "parallel_parallax",
    "parallel_parallax_bwd",
    "parallel_parallax_fwd",
    "parallax_attn_with_kvcache",
    "parallax_backward",
    "parallax_decode",
    "parallax_forward",
]
