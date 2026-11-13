"""Official ACP vs the selected Forgetting Attention implementation. Full-call CUDA Event latency, never CUDA Graph.

Run with pytest -m parallel_forgetting_attn. Latencies are in ms.
The original reference adapters and input generation live in the single
correctness test file. Operator marker: parallel_forgetting_attn.
"""
from __future__ import annotations


import math
import statistics


import pytest
import torch

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:  # Execution from the benchmark directory.
    from recording import benchmark_metric, record_benchmark_result

from flag_attn.testing import backend as test_backend
from test_forgetting_attention import (
    DEFAULT_CASES, case_id, make_inputs,
    call_optimized, call_official, assert_bitwise,
)

pytestmark = [
    pytest.mark.parallel_forgetting_attn,
    pytest.mark.skipif(not test_backend.supports_operator("parallel_forgetting_attn"), reason="selected Forgetting Attention implementation is unavailable"),
]


def event_samples(fn, warmup, rep):
    if warmup < 1 or rep < 1:
        raise ValueError("warmup and rep must both be positive iteration counts")
    if not hasattr(test_backend.device_fn, "Event"):
        return [test_backend.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")]
    for _ in range(warmup):
        fn()
    test_backend.device_fn.synchronize()
    pairs = []
    for _ in range(rep):
        begin = test_backend.device_fn.Event(enable_timing=True)
        end = test_backend.device_fn.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        pairs.append((begin, end))
    test_backend.device_fn.synchronize()
    samples = [a.elapsed_time(b) for a, b in pairs]
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError("invalid CUDA Event timing")
    return samples


@torch.inference_mode()
def benchmark_case(case, warmup=40, rep=400, rounds=3, seed=0):
    if rounds < 1:
        raise ValueError("rounds must be positive")
    inputs = make_inputs(case, seed)
    providers = {"flag_attn": lambda: call_optimized(inputs, case)}
    compare_official = test_backend.device == "cuda"
    if compare_official:
        expected = call_official(inputs, case)
        assert_bitwise(call_optimized(inputs, case), expected)
        assert_bitwise(call_optimized(inputs, case), expected)
        del expected
        providers["official"] = lambda: call_official(inputs, case)
    measurements = []
    for index in range(rounds):
        order = list(providers)
        if (index + int(case["id"][1:])) % 2:
            order.reverse()
        result = {"round": index, "provider_order": order}
        for name in order:
            values = event_samples(providers[name], warmup, rep)
            result[name] = {"median_ms": statistics.median(values), "samples_ms": values}
        measurements.append(result)
    baseline = statistics.median(r["official"]["median_ms"] for r in measurements) if compare_official else None
    candidate = statistics.median(r["flag_attn"]["median_ms"] for r in measurements)
    return {"case": case, "baseline": ("official_acp" if case["reference_kind"] == "native" else "official_acp_adapted") if compare_official else None,
            "latency_base": baseline, "latency": candidate, "speedup": baseline / candidate if baseline is not None else None,
            "accuracy": "bitwise_pass" if compare_official else "not_checked", "rounds": measurements}


def print_row(row):
    case = row["case"]
    shape = ",".join(str(case[k]) for k in ("B", "M", "N", "HQ", "H", "D"))
    baseline = "N/A" if row["latency_base"] is None else f'{row["latency_base"]:.6f}'
    speedup = "N/A" if row["speedup"] is None else f'{row["speedup"]:.3f}x'
    print(f'{case["id"]:<5} {shape:<28} {case["dtype"]:<9} {case["reference_kind"]:<7} {baseline:>10} {row["latency"]:>10.6f} {speedup:>9}', flush=True)


def metric_for(row):
    case = row["case"]
    return benchmark_metric(
        shape_detail=[
            [case[k] for k in ("B", "M", "N", "HQ", "H", "D")],
            {"scale": case["scale"], "threshold": -10.0},
        ],
        latency_base=row["latency_base"], latency=row["latency"],
        speedup=row["speedup"], accuracy=row["accuracy"],
    )


@pytest.mark.parametrize("case", DEFAULT_CASES, ids=case_id)
def test_forgetting_attention_benchmark(case, record_property):
    row = benchmark_case(case)
    print_row(row)
    record_benchmark_result(
        record_property,
        op_name="parallel_forgetting_attn",
        dtype=str(getattr(torch, case["dtype"])),
        mode="forward",
        level="end_to_end",
        result=[metric_for(row)],
    )
