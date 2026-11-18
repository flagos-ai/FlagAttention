# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Official-contract Wall-Attention benchmark against upstream FLA."""

from __future__ import annotations


import math
import statistics
import subprocess
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch
import triton

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:
    from recording import benchmark_metric, record_benchmark_result

from flag_attn.testing import backend as test_backend
from flag_attn import parallel_wall_attn
from flag_attn.FLA.wall_attn import get_last_route, select_route
from flag_attn.utils import has_triton_tle

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096)
SHAPE_FAMILIES = (("mha", 8), ("gqa", 2))
WARMUP_CALLS = 30
MEASUREMENT_MS = 200
REPEATS = 7


def _source_revision(directory: Path) -> str | None:
    """Record a Git revision when running from a checkout, without requiring Git."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _environment():
    try:
        fla_version = version("flash-linear-attention")
    except PackageNotFoundError:
        fla_version = None
    try:
        import fla
        fla_directory = Path(fla.__file__).resolve().parent.parent
    except ImportError:
        fla_directory = None

    # Installed wheels may have no Git metadata. Keep the package version and
    # report an absent revision explicitly rather than guessing a source SHA.
    device = test_backend.device_fn.current_device()
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        driver_version = driver.stdout.splitlines()[0] if driver.returncode == 0 and driver.stdout else None
    except (OSError, subprocess.TimeoutExpired):
        driver_version = None
    return {
        "gpu_model": test_backend.get_device_name(device),
        "compute_capability": list(test_backend.cuda_capability(device)),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "triton_version": str(triton.__version__),
        "driver_version": driver_version,
        "fla_version": fla_version,
        "fla_revision": _source_revision(fla_directory) if fla_directory is not None else None,
        "candidate_revision": _source_revision(Path(__file__).resolve().parent.parent),
    }


@lru_cache(maxsize=1)
def _load_fla_provider():
    if not test_backend.is_nvidia():
        return None
    try:
        from fla.ops.wall_attn import parallel_wall_attn as fla_wall_attn
    except (ImportError, OSError):
        return None
    return fla_wall_attn


def _make_inputs(t: int, h: int, dtype: torch.dtype):
    torch.manual_seed(0)
    q = torch.randn(1, t, 8, 64, device=test_backend.device, dtype=dtype)
    k = torch.randn(1, t, h, 64, device=test_backend.device, dtype=dtype)
    v = torch.randn(1, t, h, 64, device=test_backend.device, dtype=dtype)
    g = (-torch.randn(1, t, 8, 64, device=test_backend.device).abs() * 0.05).to(dtype)
    return q, k, v, g


def _bench_ms(fn: Callable[[], torch.Tensor]) -> float:
    return float(
        test_backend.do_bench(
            fn,
            warmup=0,
            rep=MEASUREMENT_MS,
            return_mode="median",
        )
    )


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


@torch.inference_mode()
def _run_benchmark(dtype_name, dtype, record_property=None):
    fla_wall_attn = _load_fla_provider()
    environment = _environment()
    metrics = []
    speedups = []

    print("dtype family T FLA_ms Final_ms speedup")
    for family, h in SHAPE_FAMILIES:
        for t in SEQUENCE_LENGTHS:
            inputs = _make_inputs(t, h, dtype)
            scale = 64**-0.5
            providers = {"optimized": lambda: parallel_wall_attn(*inputs, scale=scale)}
            if fla_wall_attn is not None:
                providers["fla"] = lambda: fla_wall_attn(*inputs, scale=scale)
            generic_hopper = (test_backend.is_nvidia() and test_backend.cuda_capability() == (9, 0)
                              and has_triton_tle() and not test_backend.has_specialization("parallel_wall_attn"))
            if generic_hopper:
                expected_route = "hopper_bf16" if dtype is torch.bfloat16 else "hopper_fp16"
                assert select_route(*inputs, scale=scale) == expected_route

            outputs = {name: fn() for name, fn in providers.items()}
            actual_route = get_last_route() if not test_backend.has_specialization("parallel_wall_attn") else test_backend.runtime.device.vendor_name
            test_backend.device_fn.synchronize()
            for name, output in outputs.items():
                assert torch.isfinite(output).all(), f"{name} produced NaN/Inf"
            if "fla" in outputs:
                torch.testing.assert_close(outputs["optimized"], outputs["fla"], atol=0.05, rtol=0.05)

            provider_names = tuple(providers)
            for warmup_index in range(WARMUP_CALLS):
                order = provider_names if warmup_index % 2 == 0 else provider_names[::-1]
                for name in order:
                    providers[name]()
            test_backend.device_fn.synchronize()

            samples = {name: [] for name in provider_names}
            for repeat in range(REPEATS):
                order = provider_names if repeat % 2 == 0 else provider_names[::-1]
                for name in order:
                    samples[name].append(_bench_ms(providers[name]))

            latency = {name: _summary(values) for name, values in samples.items()}
            fla_ms = float(latency["fla"]["median_ms"]) if "fla" in latency else None
            optimized_ms = float(latency["optimized"]["median_ms"])
            speedup = fla_ms / optimized_ms if fla_ms is not None else None
            if speedup is not None:
                speedups.append(speedup)
            print(
                dtype_name,
                family,
                t,
                "N/A" if fla_ms is None else f"{fla_ms:.6f}",
                f"{optimized_ms:.6f}",
                "N/A" if speedup is None else f"{speedup:.4f}",
                flush=True,
            )
            metrics.append(
                benchmark_metric(
                    shape_detail={
                        "B": 1,
                        "T": t,
                        "H": h,
                        "HQ": 8,
                        "D": 64,
                        "family": family,
                    },
                    latency_base=fla_ms,
                    latency=optimized_ms,
                    speedup=speedup,
                    provider_route=actual_route,
                    samples=latency,
                )
            )

    geomean_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups)) if speedups else None
    if geomean_speedup is not None:
        print(f"geomean {dtype_name}: {geomean_speedup:.4f}x vs FLA", flush=True)
    record_benchmark_result(
        record_property,
        op_name="parallel_wall_attn",
        dtype=str(dtype),
        result=metrics,
        baseline="official_fla" if fla_wall_attn is not None else None,
        phase="forward",
        hardware=test_backend.get_device_name(),
        input_contract="gate_scale=0.05, seed=0",
        warmup_calls=WARMUP_CALLS,
        measurement_ms=MEASUREMENT_MS,
        repeats=REPEATS,
        geomean_speedup_vs_fla=geomean_speedup,
        environment=environment,
    )


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "dtype_name,dtype",
    (("bf16", torch.bfloat16), ("fp16", torch.float16)),
    ids=("bf16", "fp16"),
)
@pytest.mark.skipif(
    not test_backend.supports_operator("parallel_wall_attn"),
    reason="requires an available accelerator",
)
def test_parallel_wall_attn_benchmark(dtype_name, dtype, record_property):
    _run_benchmark(dtype_name, dtype, record_property)
