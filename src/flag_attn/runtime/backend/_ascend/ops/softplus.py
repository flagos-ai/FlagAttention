"""Portable softplus used by Ascend kernels."""
import triton
import triton.language as tl

@triton.jit
def softplus(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)
