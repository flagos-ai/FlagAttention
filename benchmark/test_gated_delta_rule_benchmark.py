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

"""Gated delta rule benchmarks against the external vLLM implementation."""

import hashlib
import importlib
import inspect
import os
import statistics
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable

import pytest
import torch
import triton

from benchmark.recording import BenchmarkRecorder, benchmark_metric, record_benchmark_result

from flag_attn.testing import backend as test_backend
from flag_attn import chunk_gated_delta_rule
from flag_attn.utils import has_triton_tle
from gdn_utils import check_close, make_case, make_packed_case, reference, reference_packed


ASSERT_RATIO = 0.01
WARMUP = 10
REPETITIONS = 50
FULL_TLE_ENV = "FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE"
RECOMPUTE_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_RECOMPUTE_TLE"
TWO_KERNEL_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE"
HYBRID_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_HYBRID_TLE"

SHAPES = [
    (1, 8192, 96, 128, 128),
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]
DTYPES = (torch.float16, torch.bfloat16)

HEADS = [(16, 32), (8, 16), (4, 8), (2, 4), (16, 48), (8, 24), (4, 12), (4, 16), (2, 8)]
PREFILL_SHAPES = [
    (batch, length, h, hv, 128, 128) for h, hv in HEADS for batch, length in ((1, 512), (1, 2048), (4, 2048), (1, 8192))
] + [(b, t, h, h, k, v) for b, t, h, k, v in SHAPES]
DECODE_SHAPES = [(n, 1, h, hv) for h, hv in HEADS for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
MTP_SHAPES = [(n, length, h, hv) for h, hv in HEADS for n in (1, 4, 16, 64, 128) for length in (2, 3, 4, 8)]
EXTRA_BATCH_SHAPES = [(n, 1, h, hv) for h, hv in HEADS for n in (24, 48, 96, 192, 512)]


def _cuda_tle_available() -> bool:
    return test_backend.is_nvidia() and has_triton_tle(3, 6, 0)


def _require_vllm_reference():
    pytest.importorskip("vllm")
    return importlib.import_module("vllm.model_executor.layers.fla.ops.chunk").chunk_gated_delta_rule_fwd


@contextmanager
def _use_optimized_tle():
    names = (
        FULL_TLE_ENV,
        RECOMPUTE_TLE_ENV,
        TWO_KERNEL_TLE_ENV,
        HYBRID_TLE_ENV,
    )
    old_values = {name: os.environ.get(name) for name in names}
    os.environ[FULL_TLE_ENV] = "1"
    os.environ[RECOMPUTE_TLE_ENV] = "0"
    os.environ[TWO_KERNEL_TLE_ENV] = "1"
    os.environ[HYBRID_TLE_ENV] = "1"
    try:
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make_inputs(shape: tuple[int, int, int, int, int], dtype: torch.dtype):
    B, T, H, K, V = shape
    q = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, H, V, device=test_backend.device, dtype=dtype)
    g = (-torch.rand(B, T, H, device=test_backend.device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device=test_backend.device, dtype=dtype).sigmoid()
    return q, k, v, g, beta, K**-0.5


def _call_vllm_reference(inputs):
    q, k, v, g, beta, scale = inputs
    value = _require_vllm_reference()(q, k, v, g, beta, scale=scale, initial_state=None, output_final_state=True)
    return value[1], value[3]


def _call_flag_attn(inputs):
    q, k, v, g, beta, scale = inputs
    o, state = chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, scale=scale, output_final_state=True)
    return o, state.transpose(-1, -2).contiguous()


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    error = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    baseline = expected.float().flatten().square().mean().sqrt().item()
    return error / (baseline + 1e-8)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isfinite(actual).all(), f"{name}: non-finite candidate values"
    assert torch.isfinite(expected).all(), f"{name}: non-finite baseline values"
    actual = actual.float()
    expected = expected.float()
    absolute_error = (actual - expected).abs().max().item()
    if absolute_error <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    ratio = _err_ratio(expected, actual)
    assert ratio < ASSERT_RATIO, f"{name} diff: abs={absolute_error:.6f} ratio={ratio:.6f} limit={ASSERT_RATIO}"


@torch.inference_mode()
def _benchmark_case(dtype: torch.dtype, shape: tuple[int, int, int, int, int]):
    torch.manual_seed(42)
    inputs = _make_inputs(shape, dtype)
    compare_vllm = test_backend.is_nvidia()
    vllm_ms = None
    if compare_vllm:
        triton.set_allocator(
            lambda size, alignment, stream: torch.empty(size, device=inputs[0].device, dtype=torch.uint8)
        )
        expected_o, expected_final_state = _call_vllm_reference(inputs)
        vllm_ms = test_backend.do_bench(
            lambda: _call_vllm_reference(inputs), warmup=WARMUP, rep=REPETITIONS, return_mode="median"
        )
    with _use_optimized_tle() if _cuda_tle_available() else nullcontext():
        actual_o, actual_final_state = _call_flag_attn(inputs)
        if compare_vllm:
            _assert_close("o", actual_o, expected_o)
            _assert_close("final_state", actual_final_state, expected_final_state)
        else:
            assert torch.isfinite(actual_o).all()
            assert torch.isfinite(actual_final_state).all()
        flag_attn_ms = test_backend.do_bench(
            lambda: _call_flag_attn(inputs), warmup=WARMUP, rep=REPETITIONS, return_mode="median"
        )
    return vllm_ms, flag_attn_ms


def _shape_name(shape: tuple[int, int, int, int, int]) -> str:
    return f"B{shape[0]}_T{shape[1]}_H{shape[2]}_K{shape[3]}_V{shape[4]}"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _speedup(vllm_ms: float | None, flag_attn_ms: float):
    return vllm_ms / flag_attn_ms if vllm_ms is not None and flag_attn_ms > 0 else None


def _print_header() -> None:
    print(f"\n{'=' * 92}")
    print("  chunk_gated_delta_rule benchmark — FWD ONLY")
    print("  provider: vllm_chunk_gated_delta_rule vs flag_attn_chunk_gated_delta_rule")
    print(f"  warmup={WARMUP}ms  rep={REPETITIONS}ms")
    print(f"{'=' * 92}")
    print(
        f"{'B':>3} {'T':>6} {'H':>4} {'K':>4} {'V':>4} {'dtype':>8} "
        f"{'vllm(ms)':>10} {'flag_attn(ms)':>14} {'vllm/flag_attn':>14}"
    )


def _print_row(dtype, shape, vllm_ms, flag_attn_ms) -> None:
    B, T, H, K, V = shape
    baseline = "N/A" if vllm_ms is None else f"{vllm_ms:.3f}"
    ratio = _speedup(vllm_ms, flag_attn_ms)
    speedup = "N/A" if ratio is None else f"{ratio:.2f}x"
    print(
        f"{B:>3} {T:>6} {H:>4} {K:>4} {V:>4} {_dtype_name(dtype):>8} {baseline:>10} {flag_attn_ms:>14.3f} {speedup:>14}"
    )


def _print_footer() -> None:
    print(f"\n{'=' * 92}")
    print("  All done.")
    print(f"{'=' * 92}\n")


def _record_result(
    record_property: Callable[[str, object], None] | None,
    dtype: torch.dtype,
    shape: tuple[int, int, int, int, int],
    vllm_ms: float,
    flag_attn_ms: float,
) -> None:
    if record_property is None:
        return
    prefix = f"{_shape_name(shape)}_{_dtype_name(dtype)}"
    record_property(f"{prefix}.vllm_ms", vllm_ms)
    record_property(f"{prefix}.flagattention_ms", flag_attn_ms)
    record_property(f"{prefix}.speedup_vs_vllm", _speedup(vllm_ms, flag_attn_ms))


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.skipif(
    not test_backend.supports_operator("chunk_gated_delta_rule"), reason="requires an available accelerator"
)
def test_chunk_gated_delta_rule_benchmark(
    record_property: Callable[[str, object], None],
) -> None:
    if _cuda_tle_available() and torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("Hopper coverage is parametrized in test_chunk_gdn_forward_benchmark")
    run_benchmark(record_property)


def run_benchmark(
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    if not test_backend.is_available():
        raise RuntimeError("GDN benchmark requires an available accelerator")
    compare_vllm = test_backend.is_nvidia()

    if record_property is not None:
        record_property("scope", "operator_only")
        record_property("baseline", "vLLM" if compare_vllm else None)

    _print_header()
    for dtype in DTYPES:
        recorder = BenchmarkRecorder(
            record_property,
            op_name="chunk_gated_delta_rule",
            dtype=str(dtype),
            baseline="vLLM" if compare_vllm else None,
            phase="forward",
        )
        print("\ndtype:", dtype)
        for shape in SHAPES:
            vllm_ms, flag_attn_ms = _benchmark_case(dtype, shape)
            _print_row(dtype, shape, vllm_ms, flag_attn_ms)
            recorder.add(shape_detail=shape, latency_base=vllm_ms, latency=flag_attn_ms)
            _record_result(
                record_property,
                dtype,
                shape,
                vllm_ms,
                flag_attn_ms,
            )
            test_backend.device_fn.empty_cache()
        recorder.record()
    _print_footer()


def _measure_providers(providers, *, state=None, cache="warm", warmup_ms=1000, rep_ms=100, rounds=7):
    """Return GPU milliseconds; state restoration and cache eviction are untimed."""
    original = state.clone() if state is not None else None
    eviction = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8) if cache == "cold" else None
    measurements = {}

    def capture(fn):
        if state is not None:
            state.copy_(original)
        fn()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            start.record()
            fn()
            end.record()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return graph, start, end

    def samples(name, budget_ms):
        graph, start, end = measurements[name]
        stop = time.perf_counter() + budget_ms / 1000
        values = []
        while len(values) < 5 or time.perf_counter() < stop:
            if state is not None:
                state.copy_(original)
            if eviction is not None:
                eviction.zero_()
            graph.replay()
            end.synchronize()
            values.append(start.elapsed_time(end))
        return values

    try:
        for name, fn in providers.items():
            measurements[name] = capture(fn)
        for name in providers:
            samples(name, warmup_ms)
        medians = {name: [] for name in providers}
        counts = {name: [] for name in providers}
        names = list(providers)
        for index in range(rounds):
            order = names[index % len(names) :] + names[: index % len(names)]
            for name in order:
                values = samples(name, rep_ms)
                medians[name].append(statistics.median(values))
                counts[name].append(len(values))
        return {name: statistics.median(values) for name, values in medians.items()}, {
            "round_medians_ms": medians,
            "samples_per_round": counts,
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "cache": cache,
            "scope": "one operator call per CUDA Graph; external events; reset/eviction excluded",
        }
    finally:
        if state is not None:
            state.copy_(original)


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


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("shape", PREFILL_SHAPES, ids=lambda s: "B{}_T{}_H{}_HV{}_K{}_V{}".format(*s))
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("cache", ["cold", "warm"])
@pytest.mark.parametrize("baseline", ["vllm"], ids=["vllm_reference"])
@torch.inference_mode()
def test_chunk_gdn_forward_benchmark(shape, dtype, cache, baseline, record_property):
    if not _cuda_tle_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper and TLE required")
    from flag_attn import chunk_gated_delta_rule_fwd

    reference = _require_vllm_reference()
    package = importlib.import_module("vllm")

    batch, length, h, hv, dk, dv = shape
    torch.manual_seed(2026)
    q = torch.randn((batch, length, h, dk), device="cuda", dtype=dtype) / dk**0.5
    k = torch.randn_like(q) / dk**0.5
    v = torch.randn((batch, length, hv, dv), device="cuda", dtype=dtype)
    g = -torch.rand((batch, length, hv), device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand_like(g)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=q.device, dtype=torch.uint8))
    providers = {
        baseline: lambda: reference(q, k, v, g, beta, scale=dk**-0.5, initial_state=None, output_final_state=True),
        "candidate": lambda: chunk_gated_delta_rule_fwd(q, k, v, g, beta, dk**-0.5, None, True),
    }

    def candidate_vk():
        value = providers["candidate"]()
        return (*value[:3], value[3].transpose(-1, -2).contiguous(), *value[4:])

    providers["candidate_vk"] = candidate_vk
    expected = providers[baseline]()
    actual = providers["candidate"]()
    expected_o = expected[1]
    expected_state = expected[3].transpose(-1, -2)
    _assert_close("o", actual[1], expected_o)
    _assert_close("final_state", actual[3], expected_state)
    adapted = providers["candidate_vk"]()
    _assert_close("o_vk", adapted[1], expected[1])
    _assert_close("final_state_vk", adapted[3], expected[3])
    assert adapted[3].is_contiguous()
    latency, measurement = _measure_providers(providers, cache=cache)
    source = Path(inspect.getfile(inspect.unwrap(reference)))
    for provider in ("candidate", "candidate_vk"):
        mode = "raw_forward_vllm_" + ("state_abi_" if provider == "candidate_vk" else "native_") + cache
        record_benchmark_result(
            record_property,
            op_name="chunk_gated_delta_rule",
            dtype=_dtype_name(dtype),
            mode=mode,
            result=[
                benchmark_metric(
                    case_id="B{}_T{}_H{}_HV{}_K{}_V{}_{}".format(*shape, cache),
                    shape_detail=dict(zip(("B", "T", "H", "HV", "K", "V"), shape), gate_dtype="float32", cache=cache),
                    latency=latency[provider],
                    latency_base=latency[baseline],
                    speedup=latency[baseline] / latency[provider],
                    accuracy="passed",
                )
            ],
            baseline_name=baseline,
            baseline_version=getattr(package, "__version__", None),
            baseline_source=str(source),
            baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            baseline_state_layout="V,K",
            candidate_state_layout="V,K" if provider == "candidate_vk" else "K,V",
            state_conversion_timed=provider == "candidate_vk",
            measurement=measurement,
        )


@pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.parametrize("shape", DECODE_SHAPES + MTP_SHAPES, ids=lambda s: "N{}_L{}_H{}_HV{}".format(*s))
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"])
@pytest.mark.parametrize("cache", ["cold", "warm"])
@torch.inference_mode()
def test_fused_recurrent_gdn_benchmark(shape, state_dtype, cache, vllm_gdn, record_property):
    from flag_attn import fused_recurrent_gated_delta_rule_fwd

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
    latency, measurement = _measure_providers(providers, state=case["initial_state"][: n * length + 1], cache=cache)
    _record(record_property, "fused_recurrent_gated_delta_rule_fwd", shape, state_dtype, vllm_gdn, latency, measurement)


@pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.parametrize("shape", DECODE_SHAPES + EXTRA_BATCH_SHAPES, ids=lambda s: "N{}_L{}_H{}_HV{}".format(*s))
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16], ids=["fp32_state", "bf16_state"])
@pytest.mark.parametrize("cache", ["cold", "warm"])
@torch.inference_mode()
def test_packed_gdn_benchmark(shape, state_dtype, cache, vllm_gdn, record_property):
    from flag_attn import fused_recurrent_gated_delta_rule_packed_decode

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
    latency, measurement = _measure_providers(providers, state=case["initial_state"][: n + 1], cache=cache)
    _record(
        record_property,
        "fused_recurrent_gated_delta_rule_packed_decode",
        shape,
        state_dtype,
        vllm_gdn,
        latency,
        measurement,
    )
