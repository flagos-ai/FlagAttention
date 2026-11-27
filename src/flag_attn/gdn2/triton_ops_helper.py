"""Triton helpers shared by the GDN2 kernels."""

import triton
import triton.language as tl

from flag_attn.runtime import autotune_cache_kwargs as _autotune_cache_kwargs

autotune_cache_kwargs = _autotune_cache_kwargs()
exp = tl.exp


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


__all__ = ["autotune_cache_kwargs", "exp", "exp2"]
