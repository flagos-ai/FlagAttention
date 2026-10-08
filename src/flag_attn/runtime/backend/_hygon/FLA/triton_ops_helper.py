# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


def get_exp():
    """Return exp implementation (fast or accurate) based on env flag."""
    return (
        tldevice.fast_expf if os.environ.get("FLA_USE_FAST_OPS", "0") == "1" else tl.exp
    )


# Default exported exp to be imported by kernels.
exp = get_exp()


@triton.jit
def log(x):
    """Wrapper around tl.log that casts to fp32 first (matching the original FLA op.py)."""
    return tl.log(x.to(tl.float32))


@triton.jit
def exp2(x):
    """Base-2 exponential with fp32 computation."""
    return tl.math.exp2(x.to(tl.float32))


try:
    import inspect

    _SUPPORTS_AUTOTUNE_CACHE = (
        "cache_results" in inspect.signature(triton.autotune).parameters
    )
except Exception:
    _SUPPORTS_AUTOTUNE_CACHE = False
autotune_cache_kwargs = {"cache_results": True} if _SUPPORTS_AUTOTUNE_CACHE else {}


if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
