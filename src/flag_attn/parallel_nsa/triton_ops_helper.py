# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Compatibility helpers used by the S60 NSA kernels.

The exponential and logarithm helpers are derived from the
flash-linear-attention project, originally licensed under the
MIT License:

Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""

from __future__ import annotations

import inspect
import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


try:
    import triton.experimental.tle.language as tle

    HAS_TLE = True
except ImportError:
    tle = None
    HAS_TLE = False


def _get_exp():
    """Select the configured Triton exponential implementation."""
    if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
        return tldevice.fast_expf
    return tl.exp


exp = _get_exp()


@triton.jit
def log(x):
    """Compute a natural logarithm using fp32 input."""
    return tl.log(x.to(tl.float32))


try:
    _SUPPORTS_AUTOTUNE_CACHE = (
        "cache_results"
        in inspect.signature(
            triton.autotune
        ).parameters
    )
except Exception:
    _SUPPORTS_AUTOTUNE_CACHE = False


autotune_cache_kwargs = (
    {"cache_results": True}
    if _SUPPORTS_AUTOTUNE_CACHE
    else {}
)


__all__ = [
    "HAS_TLE",
    "autotune_cache_kwargs",
    "exp",
    "log",
    "tle",
]
