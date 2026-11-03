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

import hashlib
import importlib
from pathlib import Path

import pytest
import torch

from benchmark.recording import benchmark_metric, record_benchmark_result
from flag_attn import fused_recurrent_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_packed_decode
from flag_attn.testing.gdn import measure_providers
from flag_attn.testing.packed_gdn import make_packed_case, reference_packed
from flag_attn.testing.recurrent_gdn import check_close, make_case, reference

# Per-rank Q/K and value heads for grouping ratios 2, 3 and 4.
HEADS = [(16, 32), (8, 16), (4, 8), (2, 4), (16, 48), (8, 24), (4, 12), (4, 16), (2, 8)]
DECODE_SHAPES = [(n, 1, h, hv) for h, hv in HEADS for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
MTP_SHAPES = [(n, length, h, hv) for h, hv in HEADS for n in (1, 4, 16, 64, 128) for length in (2, 3, 4, 8)]
EXTRA_BATCH_SHAPES = [(n, 1, h, hv) for h, hv in HEADS for n in (24, 48, 96, 192, 512)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def vllm_gdn():
    try:
        importlib.import_module("vllm")
    except ModuleNotFoundError as exc:
        if exc.name == "vllm":
            pytest.skip("the vLLM baseline must be installed to run decode benchmarks")
        raise
    for name in (
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent",
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
    ):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if not exc.name or not (name == exc.name or name.startswith(exc.name + ".")):
                raise
    pytest.fail("installed vLLM does not expose the required FLA recurrent operators")


def _record(record_property, op_name, shape, state_dtype, module, latency, measurement):
    source = Path(module.__file__)
    import vllm

    record_benchmark_result(
        record_property,
        op_name=op_name,
        dtype="bfloat16",
        mode="operator_gpu_graph_" + measurement["cache"],
        result=[
            benchmark_metric(
                case_id="N{}_L{}_H{}_HV{}_{}_{}".format(
                    *shape, str(state_dtype).removeprefix("torch."), measurement["cache"]
                ),
                shape_detail=dict(
                    zip(("N", "L", "H", "HV"), shape),
                    K=128,
                    V=128,
                    state_dtype=str(state_dtype),
                    cache=measurement["cache"],
                    workload="mtp" if shape[1] > 1 else "decode",
                ),
                latency=latency["candidate"],
                latency_base=latency["vllm"],
                speedup=latency["vllm"] / latency["candidate"],
                accuracy="passed",
            )
        ],
        baseline_name="vllm",
        baseline_version=vllm.__version__,
        baseline_source=str(source),
        baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        measurement=measurement,
    )


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.parametrize("shape", DECODE_SHAPES + MTP_SHAPES, ids=lambda s: "N{}_L{}_H{}_HV{}".format(*s))
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"])
@pytest.mark.parametrize("cache", ["cold", "warm"])
@torch.inference_mode()
def test_fused_recurrent_gdn_benchmark(shape, state_dtype, cache, vllm_gdn, record_property):
    n, length, h, hv = shape
    case = make_case(n, length, h, hv, state_dtype=state_dtype, spec=length > 1, pool_min=1024, seed=2026)
    original = case["initial_state"].clone()
    expected = reference(case)
    # vLLM requires contiguous Q/K/V; their conversion is outside this kernel comparison.
    baseline_case = {**case, **{key: case[key].contiguous() for key in ("q", "k", "v", "g", "beta")}}
    providers = {
        "vllm": lambda: vllm_gdn.fused_recurrent_gated_delta_rule_fwd(**baseline_case),
        "candidate": lambda: fused_recurrent_gated_delta_rule_fwd(**case),
    }
    for fn in providers.values():
        case["initial_state"].copy_(original)
        check_close(*fn(), *expected)
    case["initial_state"].copy_(original)
    # make_case allocates all accessed slots in [1, N*L]; other pool rows stay untouched.
    latency, measurement = measure_providers(providers, state=case["initial_state"][: n * length + 1], cache=cache)
    _record(record_property, "fused_recurrent_gated_delta_rule_fwd", shape, state_dtype, vllm_gdn, latency, measurement)


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize("shape", DECODE_SHAPES + EXTRA_BATCH_SHAPES, ids=lambda s: "N{}_L{}_H{}_HV{}".format(*s))
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"])
@pytest.mark.parametrize("cache", ["cold", "warm"])
@torch.inference_mode()
def test_packed_gdn_benchmark(shape, state_dtype, cache, vllm_gdn, record_property):
    n, _, h, hv = shape
    case = make_packed_case(n, h, hv, state_dtype=state_dtype, pool_min=1024, seed=2026)
    original = case["initial_state"].clone()
    expected = reference_packed(case)
    providers = {
        "vllm": lambda: vllm_gdn.fused_recurrent_gated_delta_rule_packed_decode(**case),
        "candidate": lambda: fused_recurrent_gated_delta_rule_packed_decode(**case),
    }
    for fn in providers.values():
        case["initial_state"].copy_(original)
        check_close(*fn(), *expected)
    case["initial_state"].copy_(original)
    latency, measurement = measure_providers(providers, state=case["initial_state"][: n + 1], cache=cache)
    _record(
        record_property,
        "fused_recurrent_gated_delta_rule_packed_decode",
        shape,
        state_dtype,
        vllm_gdn,
        latency,
        measurement,
    )
