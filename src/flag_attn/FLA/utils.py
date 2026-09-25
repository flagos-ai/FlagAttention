# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import inspect
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


# ---------------------------------------------------------------------------
# FLA hardware capability helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def get_device_capability(device_index: int | None = None) -> tuple[int, int] | None:
    """Return a CUDA device capability, or None when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device_index)
    except (AssertionError, RuntimeError):
        return None


def is_nvidia_hopper_device(device_index: int | None = None) -> bool:
    """Return whether the selected device is an NVIDIA Hopper GPU."""
    capability = get_device_capability(device_index)
    return capability is not None and capability[0] == 9


def is_nvidia_blackwell_device(device_index: int | None = None) -> bool:
    """Return whether the selected device is an NVIDIA Blackwell GPU."""
    capability = get_device_capability(device_index)
    return capability is not None and capability[0] in (10, 12)


def is_cuda_graph_capturing() -> bool:
    """Return whether the active CUDA stream is being captured."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


@lru_cache(maxsize=None)
def get_num_sms(device_index: int | None = None) -> int:
    """Return the selected CUDA device's multiprocessor count."""
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except (AssertionError, RuntimeError):
        return 0


@lru_cache(maxsize=None)
def has_shared_memory(required_bytes: int, device_index: int | None = None) -> bool:
    """Return whether the selected device exposes enough shared memory."""
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
        return properties["max_shared_mem"] >= required_bytes
    except (AttributeError, KeyError, RuntimeError, TypeError):
        return False


def _detect_nvidia_hopper() -> bool:
    """Return whether the active CUDA device is Hopper or newer."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 9
    except Exception:
        return False


is_nvidia_hopper = _detect_nvidia_hopper()
is_tma_supported = is_nvidia_hopper and (
    hasattr(triton.language, "_experimental_make_tensor_descriptor")
    or hasattr(triton.language, "make_tensor_descriptor")
)


# ---------------------------------------------------------------------------
# FLA Triton math and descriptor helpers
# ---------------------------------------------------------------------------


def get_exp():
    """Select the FLA fast exponential implementation when requested."""
    return tldevice.fast_expf if os.environ.get("FLAG_ATTN_USE_FAST_OPS", "0") == "1" else tl.exp


exp = get_exp()


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit
def log(x):
    return tl.log(x.to(tl.float32))


try:
    _SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
except Exception:
    _SUPPORTS_AUTOTUNE_CACHE = False

autotune_cache_kwargs = {"cache_results": True} if _SUPPORTS_AUTOTUNE_CACHE else {}


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:

    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        return None


# Environment settings
SUPPRESS_LEVEL = int(os.getenv("FLAG_ATTN_GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))
FLA_GDN_FIX_BT = os.getenv("FLAG_ATTN_GDN_FIX_BT", "0") == "1"

use_cuda_graph = os.environ.get("FLAG_ATTN_USE_CUDA_GRAPH", "0") == "1"


# ===========================================================================
# Bitonic sort utilities (used by NSA top-k selection kernel)
# ===========================================================================


@triton.jit
def _log2(x):
    """Compute log2 of a compile-time constant integer."""
    return x.bit_length() - 1


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr = None, descending: tl.constexpr = tl.core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: tl.constexpr = _log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids
