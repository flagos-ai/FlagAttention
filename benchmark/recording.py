# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Structured result helpers for pytest-driven FlagAttention benchmarks."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any


BENCHMARK_RESULT_PROPERTY = "flag_attn_benchmark_result"
RecordProperty = Callable[[str, object], None]
_MULTIPHASE_LOG_OPERATORS = {"chunk_gla", "minimax_m3_sparse_attn"}

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
    latency: float | None,
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
    """Write a FlagGems log record and submit it to the pytest JSON recorder."""

    detail = {
        "level": level,
        "op_name": op_name,
        "dtype": dtype,
        "mode": mode,
        "result": [dict(metric) for metric in result],
    }
    detail.update(metadata)
    log_detail = detail.copy()
    if op_name in _MULTIPHASE_LOG_OPERATORS and metadata.get("phase"):
        # FlagGems' summary groups by op_name and dtype but ignores phase.
        # Distinct names retain every phase when one log contains both.
        log_detail["op_name"] = f"{op_name}_{metadata['phase']}"
    # FlagGems' summary_for_plot.parse_log reads only single-line JSON records
    # prefixed by "[INFO] ". Direct script execution has no pytest recorder,
    # so the console record is the common output for both entry points.
    print(f"[INFO] {json.dumps(_json_safe(log_detail), default=str)}", flush=True)
    if record_property is not None:
        record_property(BENCHMARK_RESULT_PROPERTY, detail)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def record_triton_report(
    report: Any,
    dataframes: Any,
    *,
    op_name: str,
    provider: str,
    baseline: str,
    latency_from_value: Callable[[float, Mapping[str, Any]], float],
) -> None:
    """Record the provider columns returned by a Triton perf_report run.

    Triton prints throughput or latency tables, but FlagGems' parser needs
    per-shape latency and speedup fields. The caller supplies the inverse of
    its benchmark's return-value formula so these fields retain their units.
    """

    configs = report.benchmarks
    if not isinstance(configs, (list, tuple)):
        configs = [configs]
        dataframes = [dataframes]
    if len(configs) != len(dataframes):
        raise ValueError("Triton benchmark configuration/result count mismatch")

    by_dtype: dict[str, list[dict[str, Any]]] = {}
    baseline_present: dict[str, bool] = {}
    for config, dataframe in zip(configs, dataframes):
        if provider not in config.line_vals:
            raise ValueError(f"{op_name}: missing provider {provider!r}")
        provider_name = config.line_names[config.line_vals.index(provider)]
        provider_column = f"{provider_name} ({config.ylabel})"
        baseline_column = None
        if baseline in config.line_vals:
            baseline_name = config.line_names[config.line_vals.index(baseline)]
            baseline_column = f"{baseline_name} ({config.ylabel})"

        dtype = str(config.args.get("dtype", "unknown"))
        metrics = by_dtype.setdefault(dtype, [])
        baseline_present[dtype] = baseline_present.get(dtype, False) or baseline_column is not None
        for _, row in dataframe.iterrows():
            shape_detail = {**config.args}
            for name in config.x_names:
                value = row[name]
                shape_detail[name] = int(value) if float(value).is_integer() else float(value)

            measured = float(row[provider_column])
            latency = latency_from_value(measured, shape_detail)
            baseline_latency = None
            if baseline_column is not None:
                baseline_measured = float(row[baseline_column])
                baseline_latency = latency_from_value(baseline_measured, shape_detail)

            valid = math.isfinite(latency) and latency > 0
            baseline_valid = (
                baseline_latency is not None
                and math.isfinite(baseline_latency)
                and baseline_latency > 0
            )
            error_msg = None
            if not valid:
                error_msg = "provider latency is unavailable"
            elif baseline_column is not None and not baseline_valid:
                error_msg = f"{baseline} baseline latency is unavailable"

            metrics.append(
                benchmark_metric(
                    shape_detail=shape_detail,
                    latency=latency if valid else None,
                    latency_base=baseline_latency if baseline_valid else None,
                    speedup=(baseline_latency / latency if valid and baseline_valid else None),
                    error_msg=error_msg,
                )
            )

    for dtype, metrics in by_dtype.items():
        record_benchmark_result(
            None,
            op_name=op_name,
            dtype=dtype,
            result=metrics,
            baseline=baseline if baseline_present[dtype] else None,
        )
