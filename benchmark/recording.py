# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Structured result helpers for pytest-driven FlagAttention benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


BENCHMARK_RESULT_PROPERTY = "flag_attn_benchmark_result"
RecordProperty = Callable[[str, object], None]

_METRIC_DEFAULTS = {
    "legacy_shape": None,
    "shape_detail": None,
    "latency_base": None,
    "latency": None,
    "gbps_base": None,
    "gbps": None,
    "speedup": None,
    "accuracy": None,
    "tflops": None,
    "utilization": None,
    "compared_speedup": None,
    "error_msg": None,
}


def benchmark_metric(
    *,
    shape_detail: Any,
    latency: float,
    latency_base: float | None = None,
    speedup: float | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build one FlagGems-compatible benchmark metric row."""

    metric = dict(_METRIC_DEFAULTS)
    metric.update(
        shape_detail=shape_detail,
        latency_base=latency_base,
        latency=latency,
        speedup=speedup,
    )
    metric.update(extra)
    return metric


def record_benchmark_result(
    record_property: RecordProperty | None,
    *,
    op_name: str,
    dtype: str,
    result: Iterable[Mapping[str, Any]],
    mode: str = "kernel",
    level: str = "comprehensive",
    **metadata: Any,
) -> None:
    """Submit one dtype/phase result to the repository pytest recorder."""

    if record_property is None:
        return
    detail = {
        "level": level,
        "op_name": op_name,
        "dtype": dtype,
        "mode": mode,
        "result": [dict(metric) for metric in result],
    }
    detail.update(metadata)
    record_property(BENCHMARK_RESULT_PROPERTY, detail)
