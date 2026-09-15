"""Triton helpers shared by the GDN2 kernels."""

import triton
import triton.language as tl

from flag_attn.gated_delta_rule.triton_ops_helper import autotune_cache_kwargs, exp


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


__all__ = ["autotune_cache_kwargs", "exp", "exp2"]
