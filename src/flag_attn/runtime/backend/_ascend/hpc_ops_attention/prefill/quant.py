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

"""E4M3 quantization helpers used by the prefill kernels.

``_positive_e4m3_round`` is the device-side rounding of the P matrix to E4M3
(both the arithmetic and the LUT decoding tiers round through it), and
``_e4m3fn_lut`` builds the host-side 256-entry decoding table.
"""

import torch
import triton
import triton.language as tl


def _e4m3fn_lut(device: torch.device) -> torch.Tensor:
    bits = torch.arange(256, dtype=torch.uint8)
    return bits.view(torch.float8_e4m3fn).float().to(torch.bfloat16).to(device)


@triton.jit
def _positive_e4m3_round(x):
    """Round non-negative FP32 values to E4M3FN and return BF16 values.

    Three-bit mantissa round on the FP32 bits: add 1<<19 to carry the mantissa
    (the carry propagates into the exponent = round-half-up), clear the low 20
    bits. x < 2**-6 uses E4M3's 2**-9 subnormal grid, so that branch uses
    floor(x*512+0.5)*2**-9. **Bit-equivalent** to the old log2/exp2/divide formula
    (6e5 points over 1e-9..4.6e2, all hit), but drops two transcendentals and one
    FP32 divide: q=512/kv=32768 softmax 11.82 -> 6.5ms, whole op 32.3 -> 27.2ms.

    min(rounded, 448) is no longer needed: callers pass p * 256, and
    p = exp2(scores - max) <= 1 (max from the row maximum; masked lanes give 0),
    so the input is always <= 256 < 448 -- the clamp is dead code; removing it
    saves another whole-tile minimum (measured 1.2ms).
    """
    rounded = ((x.to(tl.uint32, bitcast=True) + 524288) & 4293918720).to(
        tl.float32, bitcast=True
    )
    # x < 2**-6 (subnormal) is not special-cased: bit rounding uses a finer fp32
    # grid (2**(e-3)) and the fp8 store re-rounds onto 2**-9, matching direct
    # rounding. Measured: with the "x*256 < 2**-6" lane share at 60% (enlarged
    # q_scale panel), the probabilities buffer written is **bit-identical per
    # lane** to the "explicit subnormal branch" version (6.3M lanes x 3 spans, 0
    # differences); the error is identical too (region contributes <=2**-9/256);
    # the bound is 1 subnormal ulp (2**-9), only at exact grid midpoints.
    #
    # Must leave **no equivalent form**: replacing the subnormal branch with a
    # lower-op formula or adding a data-dependent guard measured slower
    # (+1.1% / +1.2% / -0.6%, but +6.4% on short shapes) -- the cost is compiling
    # the boundary case in (UB/register pressure); only deleting it wins 11%.
    return rounded.to(tl.bfloat16)
