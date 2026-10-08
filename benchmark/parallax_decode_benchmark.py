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

"""Benchmark the FlagAttention Parallax TLE decode kernel with pytest.

Each case is checked against the fp32 reference before it is timed. The local
FlagAttention TLE operator is captured into a CUDA graph with multiple calls
per graph, then timed by graph replay. No external operator package is
imported.
"""

from __future__ import annotations

import math
import os
import statistics

import torch

import pytest
from flag_attn.FLA.parallax.decode import (
    HAS_TLE,
    parallax_decode as parallax_decode_tle,
)


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


def _capture_graph(fn, calls_per_graph: int) -> torch.cuda.CUDAGraph:
    """Capture ``calls_per_graph`` stable-buffer operator invocations."""
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        # Finish lazy backend work before capture on this side stream.
        for _ in range(3):
            fn()
        capture_stream.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(calls_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph


def _graph_sample_ms(graph: torch.cuda.CUDAGraph, calls_per_graph: int) -> float:
    """Time one graph replay and return per-operator device latency."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
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
    graph: torch.cuda.CUDAGraph,
    calls_per_graph: int,
    samples: int,
) -> tuple[float, float]:
    """Replay the local TLE graph and report median latency and MAD%."""
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
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, 1, HQ, D, device="cuda", dtype=dtype, generator=generator)
    r = torch.randn(B, 1, HQ, D, device="cuda", dtype=dtype, generator=generator) * 0.5
    k = torch.randn(B, L, H, D, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(B, L, H, D, device="cuda", dtype=dtype, generator=generator)
    return q, r, k, v


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="parallax decode benchmark requires CUDA",
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
) -> None:
    if not HAS_TLE:
        pytest.skip("FlagTree TLE is unavailable")

    assert WARMUP > 0
    assert CALLS_PER_GRAPH > 0
    assert SAMPLES > 0

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
    out_tle = torch.empty_like(q)

    def tle_fn():
        return parallax_decode_tle(
            q,
            r,
            k,
            v,
            scale,
            window_size_left=WINDOW_SIZE_LEFT,
            out=out_tle,
        )

    # Compile outside the timed region and reject numerically invalid rows.
    tle_fn()
    torch.cuda.synchronize()
    reference = _decode_reference(q, r, k, v, scale, WINDOW_SIZE_LEFT)
    tle_ref_error = _rel_err(out_tle, reference)
    assert tle_ref_error < 1e-2

    # Warm every lazy path before capture. Stable out/workspace tensors are
    # supplied by the closure, so replay performs no allocator work.
    for _ in range(WARMUP):
        tle_fn()
    torch.cuda.synchronize()
    graph = _capture_graph(tle_fn, CALLS_PER_GRAPH)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    tle_ms, tle_mad = _sample_median_and_mad(graph, CALLS_PER_GRAPH, SAMPLES)
    print(
        f"B={batch_size} L={sequence_length} HQ={num_query_heads} "
        f"HKV={num_kv_heads} D={head_dim} dtype={dtype} "
        f"TLE={tle_ms:.6f} ms MAD={tle_mad:.3f}% ref={tle_ref_error:.3e}"
    )
