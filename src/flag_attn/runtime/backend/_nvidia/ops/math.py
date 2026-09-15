"""Floating-point helpers for NVIDIA kernels."""

import triton
import triton.language as tl


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))
