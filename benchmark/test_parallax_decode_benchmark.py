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

"""Benchmark the FlagAttention Parallax decode API with pytest.

Each case is checked against the fp32 reference before it is timed. On NVIDIA
SM90, the upstream Parallax CuTe decode kernel is timed with the same inputs
and protocol to provide a speedup baseline.
"""

from __future__ import annotations

from importlib import import_module
import math
import os
import statistics

import torch

import pytest
from flag_attn.testing import backend as test_backend
from flag_attn import parallel_parallax

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Execution from the benchmark directory.
    from recording import BenchmarkRecorder


pytestmark = pytest.mark.parallel_parallax


DEFAULT_SHAPES = (
    (1, 512, 8, 8, 64),
    (1, 2048, 8, 8, 128),
    (2, 2048, 8, 2, 128),
    (1, 8192, 8, 2, 128),
)


DEFAULT_DTYPES = (torch.bfloat16,)
WINDOW_SIZE_LEFT = int(os.getenv("PARALLAX_DECODE_BENCH_WINDOW", "-1"))
WARMUP = int(os.getenv("PARALLAX_DECODE_BENCH_WARMUP", "25"))
CALLS_PER_GRAPH = int(os.getenv("PARALLAX_DECODE_BENCH_ITER", "100"))
SAMPLES = int(os.getenv("PARALLAX_DECODE_BENCH_SAMPLES", "12"))


def _capture_graph(fn, calls_per_graph: int) -> "torch.cuda.CUDAGraph":
    """Capture ``calls_per_graph`` stable-buffer operator invocations."""
    graph = test_backend.device_fn.CUDAGraph()
    capture_stream = test_backend.device_fn.Stream()
    capture_stream.wait_stream(test_backend.device_fn.current_stream())
    with test_backend.device_fn.stream(capture_stream):
        # Finish lazy backend work before capture on this side stream.
        for _ in range(3):
            fn()
        capture_stream.synchronize()
        with test_backend.device_fn.graph(graph, stream=capture_stream):
            for _ in range(calls_per_graph):
                fn()
    test_backend.device_fn.current_stream().wait_stream(capture_stream)
    test_backend.device_fn.synchronize()
    return graph


def _graph_sample_ms(graph: "torch.cuda.CUDAGraph", calls_per_graph: int) -> float:
    """Time one graph replay and return per-operator device latency."""
    start = test_backend.device_fn.Event(enable_timing=True)
    end = test_backend.device_fn.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls_per_graph


def _median_and_mad_percent(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return median, 100.0 * mad / max(median, 1e-12)


def _sample_median_and_mad(
    graph: "torch.cuda.CUDAGraph",
    calls_per_graph: int,
    samples: int,
) -> tuple[float, float]:
    """Replay the selected implementation graph and report median latency and MAD%."""
    values = [_graph_sample_ms(graph, calls_per_graph) for _ in range(samples)]
    return _median_and_mad_percent(values)


def _rel_err(actual, expected):
    denominator = max(expected.float().abs().max().item(), 1e-6)
    return (actual.float() - expected.float()).abs().max().item() / denominator


def _decode_reference(q, r, k, v, scale, window_size_left=-1):
    """FP32 Q=1 oracle without the degenerate cuBLAS einsum path."""
    HQ, H = q.shape[2], k.shape[2]
    repeat = HQ // H
    qf = q[:, 0].float()
    rf = r[:, 0].float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    if repeat > 1:
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    s1 = (qf[:, :, None, :] * kf).sum(dim=-1) * scale
    s2 = (rf[:, :, None, :] * kf).sum(dim=-1)
    if window_size_left >= 0:
        first = max(k.shape[1] - window_size_left, 0)
        valid = torch.arange(k.shape[1], device=k.device) >= first
        s1 = s1.masked_fill(~valid[None, None, :], float("-inf"))
    pivot = s1.amax(dim=-1, keepdim=True)
    pivot_safe = torch.where(torch.isfinite(pivot), pivot, torch.zeros_like(pivot))
    p1 = torch.exp(s1 - pivot_safe)
    p2 = p1 * s2
    d1 = p1.sum(dim=-1, keepdim=True)
    d2 = p2.sum(dim=-1, keepdim=True)
    o1 = (p1[..., None] * vf).sum(dim=2)
    o2 = (p2[..., None] * vf).sum(dim=2)
    inv_d1 = torch.where(d1 > 0, d1.reciprocal(), torch.zeros_like(d1))
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    return result[:, None].contiguous()


def _make_inputs(B, L, HQ, H, D, dtype, seed):
    generator = torch.Generator(device="cpu" if test_backend.device == "gcu" else test_backend.device).manual_seed(seed)
    q = torch.randn(B, 1, HQ, D, device=test_backend.device, dtype=dtype, generator=generator)
    r = torch.randn(B, 1, HQ, D, device=test_backend.device, dtype=dtype, generator=generator) * 0.5
    k = torch.randn(B, L, H, D, device=test_backend.device, dtype=dtype, generator=generator)
    v = torch.randn(B, L, H, D, device=test_backend.device, dtype=dtype, generator=generator)
    return q, r, k, v


def _load_cute_baseline():
    if not test_backend.is_nvidia():
        return None, "Parallax CuTe decode requires NVIDIA CUDA"
    try:
        return import_module("parallax.cute.parallax_decode").parallax_decode, None
    except Exception as exc:
        return None, f"Parallax CuTe decode unavailable: {type(exc).__name__}: {exc}"


@pytest.mark.skipif(
    not test_backend.is_available(),
    reason="requires an available accelerator",
)
@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "num_query_heads", "num_kv_heads", "head_dim"),
    DEFAULT_SHAPES,
)
@pytest.mark.parametrize(
    "dtype",
    DEFAULT_DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
def test_perf_parallax_decode(
    batch_size: int,
    sequence_length: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    record_property,
) -> None:
    if not test_backend.supports_operator("parallax_decode"):
        pytest.skip("selected Parallax decode implementation is unavailable")

    assert WARMUP > 0
    assert CALLS_PER_GRAPH > 0
    assert SAMPLES > 0

    baseline_op, baseline_reason = _load_cute_baseline()
    metadata = {
        "phase": "decode",
        "baseline": "Parallax CuTe SM90" if baseline_op is not None else None,
    }
    if baseline_reason:
        metadata["native_baseline_skip_reason"] = baseline_reason
    recorder = BenchmarkRecorder(
        record_property, op_name="parallel_parallax", dtype=str(dtype), **metadata,
    )
    shape_detail = {
        "B": batch_size, "L": sequence_length, "HQ": num_query_heads,
        "H": num_kv_heads, "D": head_dim,
    }

    seed = (
        2026 + batch_size * 1_000_003 + sequence_length * 10_007 + num_query_heads * 101 + num_kv_heads * 17 + head_dim
    )
    q, r, k, v = _make_inputs(
        batch_size,
        sequence_length,
        num_query_heads,
        num_kv_heads,
        head_dim,
        dtype,
        seed=seed,
    )
    scale = 1.0 / math.sqrt(head_dim)
    out_actual = torch.empty_like(q)
    out_baseline = torch.empty_like(q) if baseline_op is not None else None

    def flag_attn_fn():
        return parallel_parallax(
            q,
            r,
            k,
            v,
            scale,
            stage="decode",
            window_size_left=WINDOW_SIZE_LEFT,
            out=out_actual,
        )

    def baseline_fn():
        return baseline_op(
            q, r, k, v, scale,
            window_size_left=WINDOW_SIZE_LEFT,
            out=out_baseline,
        )

    # Compile outside the timed region and reject numerically invalid rows.
    flag_attn_fn()
    test_backend.device_fn.synchronize()
    reference = _decode_reference(q, r, k, v, scale, WINDOW_SIZE_LEFT)
    flag_attn_ref_error = _rel_err(out_actual, reference)
    assert flag_attn_ref_error < 1e-2
    baseline_ref_error = None
    if baseline_op is not None:
        baseline_fn()
        test_backend.device_fn.synchronize()
        baseline_ref_error = _rel_err(out_baseline, reference)
        assert baseline_ref_error < 1e-2, (
            f"Parallax CuTe decode differs from FP32 reference: {baseline_ref_error:.3e}"
        )
    else:
        print(f"[baseline] {baseline_reason}", flush=True)

    if not test_backend.graph_available():
        flag_attn_samples = [
            test_backend.do_bench(flag_attn_fn, warmup=WARMUP, rep=100, return_mode="median")
            for _ in range(SAMPLES)
        ]
        flag_attn_ms = statistics.median(flag_attn_samples)
        baseline_ms = None
        if baseline_op is not None:
            baseline_samples = [
                test_backend.do_bench(baseline_fn, warmup=WARMUP, rep=100, return_mode="median")
                for _ in range(SAMPLES)
            ]
            baseline_ms = statistics.median(baseline_samples)
        speedup = baseline_ms / flag_attn_ms if baseline_ms is not None else None
        baseline_text = f"{baseline_ms:.6f} ms" if baseline_ms is not None else "N/A"
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "N/A"
        print(
            f"B={batch_size} L={sequence_length} dtype={dtype} "
            f"FlagAttention={flag_attn_ms:.6f} ms CuTe={baseline_text} "
            f"speedup={speedup_text} (without graph capture)",
            flush=True,
        )
        recorder.add(shape_detail=shape_detail, latency=flag_attn_ms, latency_base=baseline_ms)
        recorder.record()
        return

    # Warm every lazy path before capture. Stable out/workspace tensors are
    # supplied by the closure, so replay performs no allocator work.
    for _ in range(WARMUP):
        flag_attn_fn()
        if baseline_op is not None:
            baseline_fn()
    test_backend.device_fn.synchronize()
    graph = _capture_graph(flag_attn_fn, CALLS_PER_GRAPH)
    baseline_graph = (
        _capture_graph(baseline_fn, CALLS_PER_GRAPH)
        if baseline_op is not None else None
    )
    for _ in range(3):
        graph.replay()
        if baseline_graph is not None:
            baseline_graph.replay()
    test_backend.device_fn.synchronize()

    flag_attn_ms, flag_attn_mad = _sample_median_and_mad(graph, CALLS_PER_GRAPH, SAMPLES)
    baseline_ms = None
    if baseline_graph is not None:
        baseline_ms, _ = _sample_median_and_mad(
            baseline_graph, CALLS_PER_GRAPH, SAMPLES
        )
    speedup = baseline_ms / flag_attn_ms if baseline_ms is not None else None
    baseline_text = f"{baseline_ms:.6f} ms" if baseline_ms is not None else "N/A"
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "N/A"
    print(
        f"B={batch_size} L={sequence_length} HQ={num_query_heads} "
        f"HKV={num_kv_heads} D={head_dim} dtype={dtype} "
        f"FlagAttention={flag_attn_ms:.6f} ms CuTe={baseline_text} speedup={speedup_text} "
        f"MAD={flag_attn_mad:.3f}% ref={flag_attn_ref_error:.3e} "
        f"cute_ref={baseline_ref_error if baseline_ref_error is not None else 'N/A'}"
    )
    recorder.add(shape_detail=shape_detail, latency=flag_attn_ms, latency_base=baseline_ms)
    recorder.record()
