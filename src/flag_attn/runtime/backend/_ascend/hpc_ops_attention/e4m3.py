# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""E4M3FN decoding primitives shared by the Ascend FP8 decode and prefill kernels.

Before this module the FP8 paths carried five separate decoders -- three in the
decode implementation (one of them already dead) and two in the prefill kernel --
which duplicated the same bit manipulation while documenting different numeric
contracts.  This is now the only place an E4M3FN byte becomes BF16.  Four public
entries:

=====================  =========================================================
Public entry           Contract
=====================  =========================================================
``e4m3_exact``         decode producer and query decoding: unmasked, bit-exact,
                       ``0x80`` -> -0.0
``e4m3_exact_masked``  prefill arithmetic tier: masked, bit-exact, ``0x80`` ->
                       +0.0 (bit-identical to the LUT tier)
``e4m3_lut``           prefill LUT tier: 256-entry table, equivalent to
                       ``e4m3_exact_masked``
``e4m3_bits16``        approximate 16-bit-integer composition, only for decode's
                       standalone decoding pass
=====================  =========================================================

**Why the two exact tiers are not merged into a single expression.**  Three
constraints, all of them measured:

1. **Zero sign.**  Exhaustively over all 256 byte patterns the two tiers differ
   only at ``0x80``: decode yields ``0x8000`` (-0.0), prefill yields ``0x0000``
   (+0.0); the values are equal.  Prefill switches between its LUT and arithmetic
   tiers at run time depending on shape, and the LUT (torch's ``e4m3 -> float``)
   yields +0.0, so both tiers have to agree on +0.0.  Decode has no LUT tier and
   keeps its original integer-OR sign path so that its output stays bit-for-bit
   what it was.
2. **UB allocation is sensitive to expression shape.**  Merging the two into one
   expression that builds the magnitude in the uint32 domain and then applies the
   sign per ``constexpr`` was bit-identical on 12 shapes, but it pushed the fused
   producer over the UB edge: MTP=2 ``uniform_512`` and MTP=4 ``skewed_extreme``
   (``Q_ROWS`` 16/32) raised a run-time aicore exception (``ub address out of
   bounds``) on a tree where the pre-refactor code ran fine.  Both paths therefore
   keep their original expressions; do not rewrite them to remove the
   duplication.
3. **Type domain.**  Decode works in the int32 bit domain and finishes with a
   single fp32 bitcast; prefill works in the uint32 domain and applies the sign
   with a float select.  Changing the domain changes the instruction sequence the
   compiler emits.

This module provides uniqueness of *location*, not of *expression*.  Any change
that does merge the two exact tiers has to pass the full MTP=1/2/4 safety net
(132 configurations) to show it does not cross the UB boundary.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def e4m3_exact(ptrs):
    """Decode E4M3FN to BF16 without a mask -- decode's producer and Q decoding.

    The missing mask is a hard requirement: a masked load drops the access out of
    the DMA lowering the producer depends on, so this tier deliberately takes no
    mask argument.

    The expression is verbatim the pre-refactor one (constraint 2 in the module
    docstring): the normal path is "shifted magnitude + exponent bias" and must
    use ``+``, not ``|`` (for exponent >= 8 the magnitude field overlaps the bias
    bits, so an OR would drop the carry); the subnormal path keeps its single fp32
    multiply; NaN (``0x7F``/``0xFF``) is preserved because real KV bytes can
    contain it and the prefill LUT tier maps it to NaN as well.
    """
    b = tl.load(ptrs).to(tl.int32) & 255
    m = b & 127
    bits = tl.where(
        m < 8,
        (m.to(tl.float32) * 0.001953125).to(tl.int32, bitcast=True),
        (m << 20) + 0x3C000000,
    ) | ((b & 128) << 24)
    bits = tl.where(m == 127, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True).to(tl.bfloat16)


@triton.jit
def e4m3_exact_masked(ptrs, mask):
    """Masked, bit-exact E4M3FN to BF16 -- prefill's arithmetic tier.

    The differences against ``e4m3_exact`` are the three constraints listed in the
    module docstring (zero sign, expression shape, type domain).  The details
    below were measured and must not be "simplified":

    1. the normal magnitude is ``(mag << 20) + 0x3C000000`` and the ``+`` is
       required for the same carry reason as above; ``mag < 8`` is exactly
       "exponent == 0", so the subnormal select needs no separate exponent field;
    2. the NaN test is ``(bits & 0x7F) == 0x7F`` (an equivalent simplification of
       two field compares plus an AND), but the branch has to stay: the LUT tier
       maps ``0x7F``/``0xFF`` to NaN and dropping it would break tier parity;
    3. the sign is applied as a float select rather than OR-ed into the value bits
       because ``0x80`` must decode to +0.0 to stay bit-identical to the LUT tier.
    """
    bits = tl.load(ptrs, mask=mask, other=0).to(tl.int32) & 255
    mag = bits & 127
    normal_bits = (mag << 20) + 0x3C000000
    subnormal = (mag & 7).to(tl.float32) * 0.001953125
    magnitude = tl.where(
        mag < 8, subnormal.to(tl.uint32, bitcast=True), normal_bits.to(tl.uint32)
    ).to(tl.float32, bitcast=True)
    value = tl.where((bits & 128) != 0, -magnitude, magnitude)
    nan = tl.full(value.shape, 2143289344, tl.uint32).to(tl.float32, bitcast=True)
    return tl.where(mag == 127, nan, value).to(tl.bfloat16)


@triton.jit
def e4m3_lut(lut, ptrs, mask):
    """Masked E4M3FN to BF16 through the 256-entry table -- prefill's LUT tier.

    Bit-identical to ``e4m3_exact_masked``, but every element needs an indexed
    load attached to it.  An isolated decode probe measured it 285x slower than
    the arithmetic tier, which is why decode no longer uses it; prefill keeps it
    as the existing switchable path.
    """
    bits = tl.load(ptrs, mask=mask, other=0).to(tl.int32) & 255
    return tl.load(lut + bits, mask=mask, other=0.0)


@triton.jit
def e4m3_bits16(ptrs):
    """E4M3FN to BF16 bits composed directly in the 16-bit integer domain.

    The BF16 fields are already known from the byte -- the exponent field is
    ``(m >> 3) + 120`` and the top three mantissa bits are ``m & 7`` -- so the
    packed word is exactly ``((m << 4) + 0x3C00)`` with the sign moved from bit 7
    to bit 15.  Seven integer ops, no FP32 round trip and no select, against the
    fifteen ops plus two selects of ``e4m3_exact``; measured on the decoding pass
    the vector pipe drops from 513us to 221us and the pass from 0.63ms to 0.31ms.

    Only the standalone decoding pass may use this form, and only because it just
    writes BF16 to GM.  Feeding BF16 that was bitcast out of an int16 tensor into
    ``tl.trans`` or ``tl.dot`` trips the backend with ``Cannot find root
    memref.alloc for mB``, so the producers and the query decoding must keep
    ``e4m3_exact``.

    Exactly 16 of the 256 byte patterns deviate from the exact tier:

    * ``m == 0`` (``0x00``/``0x80``) decodes to +/-0.0078 instead of +/-0;
    * the 14 subnormal patterns (``0x01``-``0x07``/``0x81``-``0x87``) decode to
      ``2**-7 * (1 + m/8)`` instead of ``m * 2**-9``, at most 0.0078 E4M3 units
      off.

    0.0078 is 0.0017% of the E4M3 full scale, and subnormals occur in
    0.007-0.013% of the official cache -- more than two orders of magnitude below
    the error the FP8 KV cache itself imposes.  NaN (``m == 127``) is corrected
    with one select to ``0x7FFF`` (bit-identical to the exact tier): it has to
    stay NaN, otherwise it silently becomes 480 and poisons every downstream
    score.
    """
    b = tl.load(ptrs).to(tl.int16)
    m = b & 127
    bits = (m << 4) + 0x3C00
    bits = bits | ((b & 128) << 8)
    bits = tl.where(m == 127, 0x7FFF, bits)
    return bits.to(tl.bfloat16, bitcast=True)


__all__ = ["e4m3_bits16", "e4m3_exact", "e4m3_exact_masked", "e4m3_lut"]
