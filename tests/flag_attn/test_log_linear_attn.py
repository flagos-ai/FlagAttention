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


import math

import pytest
import torch

from flag_attn.testing import backend as test_backend
from flag_attn.testing.log_linear_attn import log_linear_attn_reference

pytestmark = pytest.mark.log_linear_attn


def _make_inputs(batch, sequence, heads, dim):
    levels = math.ceil(math.log2(sequence)) + 1
    q = torch.randn(batch, sequence, 1, dim, device=test_backend.device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, heads, dim, device=test_backend.device, dtype=torch.bfloat16)
    g = -0.05 * torch.rand(batch, sequence, heads, device=test_backend.device, dtype=torch.float32)
    level_scales = torch.sigmoid(
        torch.randn(
            batch,
            sequence,
            heads,
            levels,
            device=test_backend.device,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    return q, k, v, g, level_scales


def _assert_close(name, expected, actual, ratio=0.004):
    error = (expected.float() - actual.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt()
    error_ratio = (error / (reference + 1e-8)).item()
    assert not torch.isnan(actual).any(), f"{name}: NaN detected"
    assert error_ratio < ratio, f"{name}: RMS error ratio {error_ratio:.6f}"


def test_chunk_log_linear_attn_public_export():
    import flag_attn
    from flag_attn.runtime.backend import resolve_operator

    exports = (
        ("log_linear_attn", "flag_attn.FLA.log_linear_attn.chunk_tle"),
        ("chunk_log_linear_attn", "flag_attn.FLA.log_linear_attn"),
    )

    for name, module in exports:
        assert name in flag_attn.__all__
        actual = getattr(flag_attn, name)
        assert callable(actual), f"{name} is not callable"

        expected = resolve_operator(name, module, name)
        assert actual is expected, f"{name} resolves to an unexpected implementation"


@pytest.mark.skipif(not test_backend.supports_operator("log_linear_attn", tle=True), reason="selected Log-Linear implementation is unavailable")
@pytest.mark.parametrize(
    ("batch", "sequence", "heads", "dim"),
    [
        (1, 64, 2, 64),
        (1, 128, 2, 128),
        (1, 256, 2, 256),
    ],
)
@torch.inference_mode()
def test_chunk_log_linear_attn_matches_reference(batch, sequence, heads, dim):
    torch.manual_seed(42)
    inputs = _make_inputs(batch, sequence, heads, dim)
    expected = log_linear_attn_reference(*inputs)
    from flag_attn import log_linear_attn

    actual = log_linear_attn(*inputs)
    _assert_close("output", expected, actual)
