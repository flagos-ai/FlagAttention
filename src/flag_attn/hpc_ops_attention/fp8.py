# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""E4M3 storage conversion using standard integer and floating-point Triton.

The portable kernels consume FP8 storage even on targets without native FP8
conversion or matrix multiplication. Every finite E4M3 value is exact in FP16.
"""

import triton
import triton.language as tl


@triton.jit
def decode_e4m3(bits):
    bits = bits.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    value = tl.where(
        exponent == 0,
        mantissa.to(tl.float32) * (2.0**-9),
        (8 + mantissa).to(tl.float32) * tl.exp2((exponent - 10).to(tl.float32)),
    )
    value = tl.where((exponent == 15) & (mantissa == 7), float("nan"), value)
    return tl.where((bits & 128) != 0, -value, value).to(tl.float16)


@triton.jit
def round_e4m3(value):
    """Round nonnegative softmax values in [0, 256] to E4M3, ties to even."""
    exponent = ((value.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    exponent = tl.maximum(exponent, -6)
    step = ((exponent - 3 + 127) << 23).to(tl.float32, bitcast=True)
    scaled = value / step
    lower = tl.floor(scaled)
    fraction = scaled - lower
    round_up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    return ((lower + round_up.to(tl.float32)) * step).to(tl.float16)
