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


"""Hopper inference dispatch with a portable training and numerical fallback.

FLAG_ATTN_FORCE_GENERIC=1 disables this optimization for validation.
"""

import os
from functools import lru_cache
from importlib import import_module
from numbers import Real

import torch
import triton

from flag_attn.FLA.wall_attn.parallel import (
    _LAST_ROUTE,
    _parallel_wall_attn_default,
    get_last_route,
)
from flag_attn.utils import has_triton_tle

_FAST_T = frozenset((512, 1024, 2048, 4096))
_FAST_H = frozenset((2, 8))
_FAST_BF16_D = frozenset((64, 128))


@lru_cache(maxsize=1)
def _load_fwd_module():
    """Load Hopper-only kernels after metadata dispatch accepts an inference call."""
    module = import_module(f"{__package__}.parallel_fwd")
    triton.set_allocator(module.allocate_descriptor)
    return module


def _is_sm90(device_index: int) -> bool:
    """Return whether ``device_index`` is an NVIDIA Hopper SM90 device."""
    return torch.cuda.get_device_capability(device_index) == (9, 0)


def _requires_backward(*tensors: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in tensors)


def select_route(
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
) -> str:
    """Select a candidate path from metadata; preprocessing checks numerical safety."""
    if os.environ.get("FLAG_ATTN_FORCE_GENERIC", "0") == "1":
        return "official_fallback"
    if any(value is not None for value in (g_scalar, sink_bias, window_size, cu_seqlens)):
        return "official_fallback"
    if scale is not None and not isinstance(scale, Real):
        return "official_fallback"
    if any(tensor.layout != torch.strided for tensor in (q, k, v, g)):
        return "official_fallback"
    if not all(tensor.is_cuda for tensor in (q, k, v, g)):
        return "official_fallback"
    if not (q.device == k.device == v.device == g.device):
        return "official_fallback"
    if not all(tensor.is_contiguous() for tensor in (q, k, v, g)):
        return "official_fallback"
    if any(tensor.ndim != 4 for tensor in (q, k, v, g)):
        return "official_fallback"

    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if not _is_sm90(device_index):
        return "official_fallback"
    if not has_triton_tle(required_api=(
        "gpu.copy", "gpu.wgmma", "gpu.wgmma_wait", "gpu.warp_specialize",
        "gpu.alloc", "gpu.alloc_barriers",
    )):
        return "official_fallback"

    b, t, hq, d = q.shape
    if b != 1 or t not in _FAST_T or hq != 8:
        return "official_fallback"
    h = k.shape[2]
    if h not in _FAST_H or hq % h != 0:
        return "official_fallback"
    if k.shape != (b, t, h, d) or v.shape != (b, t, h, d) or g.shape != q.shape:
        return "official_fallback"
    if _requires_backward(q, k, v, g):
        return "official_fallback"
    # Numerical validation reads a device flag. Keep graph capture on the
    # general implementation, which does not require a device-to-host read.
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            return "official_fallback"

    if q.dtype == k.dtype == v.dtype == g.dtype == torch.bfloat16 and d in _FAST_BF16_D:
        return "hopper_bf16"
    if q.dtype == k.dtype == v.dtype == g.dtype == torch.float16 and d == 64:
        return "hopper_fp16"
    return "official_fallback"


def _parallel_wall_attn_fast_path(
    route: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    fwd = _load_fwd_module()
    b, t, hq, d = q.shape
    cache_dtype = q.dtype if route == "hopper_bf16" else torch.bfloat16
    q_cache = torch.empty((b, hq, t, d), device=q.device, dtype=cache_dtype)
    k_cache = torch.empty_like(q_cache)
    anchor = torch.empty((b, hq, d), device=q.device, dtype=torch.float32)
    output = torch.empty((b, t, hq, d), device=q.device, dtype=v.dtype)
    lse = torch.empty((b, t, hq), device=q.device, dtype=torch.float32)
    unsafe = torch.zeros((), device=q.device, dtype=torch.int32)

    fwd.prepare_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe)
    if unsafe.item() != 0:
        return None

    fwd.parallel_wall_attn_fwd(q_cache, k_cache, v, output, lse, capacity=2, scale=scale)
    return output


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
    """Run a numerically checked H100 path or the full official implementation.

    Fast paths require nonpositive gates and finite, non-underflowing BF16 Q/K
    operands with gauge exponents within +/-125. Other data use the general
    implementation. Validation synchronizes with the host; graph capture uses
    the general implementation instead.
    """
    _LAST_ROUTE.set(None)
    route = select_route(
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
    if route != "official_fallback":
        with torch.cuda.device(q.device):
            output = _parallel_wall_attn_fast_path(
                route,
                q,
                k,
                v,
                g,
                scale=k.shape[-1] ** -0.5 if scale is None else float(scale),
            )
        if output is not None:
            _LAST_ROUTE.set(route)
            return output
    output = _parallel_wall_attn_default(
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
    _LAST_ROUTE.set("official_fallback")
    return output


__all__ = ["get_last_route", "parallel_wall_attn", "select_route"]
