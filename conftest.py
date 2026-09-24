# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Repository-wide pytest configuration."""

from __future__ import annotations

import ast
import fcntl
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import pytest


_BUILTIN_MARKS = {
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "timeout",
    "tryfirst",
    "trylast",
    "usefixtures",
    "xfail",
}
_BENCHMARK_RESULT_PROPERTY = "flag_attn_benchmark_result"
_DEFAULT_ACCURACY_REPORT_FILE = "accuracy_result.json"
_DEFAULT_BENCHMARK_REPORT_FILE = "benchmark_result.json"
_BENCHMARK_DETAIL_FIELDS = ("op_name", "dtype", "mode", "level", "result")
_BENCHMARK_METRIC_FIELDS = (
    "legacy_shape",
    "shape_detail",
    "latency_base",
    "latency",
    "gbps_base",
    "gbps",
    "speedup",
    "accuracy",
    "tflops",
    "utilization",
    "compared_speedup",
    "error_msg",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flagattention", "FlagAttention test reporting")
    group.addoption(
        "--record",
        action="store",
        choices=("none", "json"),
        default="none",
        dest="flag_attn_record",
        help="record test results (supported format: json)",
    )
    group.addoption(
        "--output",
        action="store",
        dest="flag_attn_output",
        metavar="PATH",
        help=(
            "JSON result path (defaults: accuracy_result.json for tests, "
            "benchmark_result.json for benchmarks)"
        ),
    )
    group.addoption(
        "--quick",
        action="store_true",
        default=False,
        help=(
            "Compatibility option for the shared FlagOS test runner. "
            "FlagAttention tests currently keep their declared parameter sets."
        ),
    )
    group.addoption(
        "--collect-marks",
        action="store",
        default=None,
        metavar="PATH",
        help="Write collected non-built-in pytest marks to PATH as JSON/YAML data.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("flag_attn_record") == "json":
        _include_marked_benchmarks(config)
        output = config.getoption("flag_attn_output")
        if output is None:
            output = (
                _DEFAULT_BENCHMARK_REPORT_FILE
                if _is_benchmark_invocation(config)
                else _DEFAULT_ACCURACY_REPORT_FILE
            )
        recorder = _JsonResultRecorder(
            Path(output), config.rootpath, _selected_operator_mark(config)
        )
        config.pluginmanager.register(recorder, "flag-attention-json-result-recorder")

    marks_output = config.getoption("collect_marks")
    if marks_output:
        collector = _MarkCollector(Path(marks_output))
        config.pluginmanager.register(collector, "flag-attention-mark-collector")


def _selected_operator_mark(config: pytest.Config) -> str | None:
    """Use the operator named by a simple ``-m OP_NAME`` expression."""

    expression = config.getoption("markexpr").strip()
    return expression if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", expression) else None


def _include_marked_benchmarks(config: pytest.Config) -> None:
    """Make the path-free upload command collect benchmarks from the repo root."""

    invocation_dir = Path(str(config.invocation_params.dir)).resolve()
    if (
        config.getoption("markexpr")
        and invocation_dir == config.rootpath.resolve()
        and config.args == list(config.getini("testpaths"))
        and (config.rootpath / "benchmark").is_dir()
    ):
        config.args.append("benchmark")


def _operator_key(marks: list[str], selected_mark: str | None, default: str) -> str:
    if selected_mark in marks:
        return selected_mark
    return marks[0] if marks else default


class _MarkCollector:
    """Collect operator marks for the shared FlagOS runner."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.items: list[dict[str, Any]] = []

    def pytest_collection_modifyitems(
        self, session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        for item in items:
            marks = [
                mark.name
                for mark in item.iter_markers()
                if mark.name not in _BUILTIN_MARKS
            ]
            self.items.append({"nodeid": item.nodeid, "marks": sorted(set(marks))})

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: pytest.ExitCode
    ) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as stream:
            json.dump(self.items, stream, indent=2)
            stream.write("\n")


class _JsonResultRecorder:
    """Collect FlagGems details and a platform-uploadable run summary."""

    def __init__(
        self, output: Path, rootpath: Path, selected_mark: str | None = None
    ) -> None:
        self.output = output
        self.rootpath = rootpath
        self.selected_mark = selected_mark
        self.results: dict[str, dict[str, Any]] = {}
        self.item_keys: dict[str, str] = {}
        self.benchmark_items: set[str] = set()
        self.collection_errors: list[tuple[str, bool, bool]] = []
        self.collection_nodes: set[str] = set()
        self.accuracy_error_items: set[str] = set()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None):
        callspec = getattr(item, "callspec", None)
        params = dict(callspec.params) if callspec is not None else {}
        operator_marks = [
            mark.name
            for mark in item.iter_markers()
            if mark.name not in _BUILTIN_MARKS
        ]
        if _is_benchmark_path(Path(str(item.path)), self.rootpath):
            key = _operator_key(operator_marks, self.selected_mark, item.nodeid)
            self.item_keys[item.nodeid] = key
            self.benchmark_items.add(item.nodeid)
            self.results.setdefault(
                key,
                {
                    "details": [],
                    "result": None,
                    "test_case": item.nodeid,
                    "reason": None,
                },
            )
        else:
            self.item_keys[item.nodeid] = item.nodeid
            self.results[item.nodeid] = {
                "params": params,
                "result": None,
                "opname": operator_marks,
                "reason": None,
            }

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        key = self.item_keys.get(report.nodeid, report.nodeid)
        if report.nodeid in self.benchmark_items:
            result = self.results.setdefault(
                key,
                {
                    "details": [],
                    "result": None,
                    "test_case": report.nodeid,
                    "reason": None,
                },
            )
            if report.when == "call":
                for name, value in report.user_properties:
                    if name == _BENCHMARK_RESULT_PROPERTY:
                        detail = _normalize_benchmark_detail(value)
                        identity = tuple(
                            detail[field] for field in _BENCHMARK_DETAIL_FIELDS[:-1]
                        )
                        existing = next(
                            (
                                item
                                for item in result["details"]
                                if tuple(
                                    item[field]
                                    for field in _BENCHMARK_DETAIL_FIELDS[:-1]
                                )
                                == identity
                            ),
                            None,
                        )
                        if existing is None:
                            result["details"].append(detail)
                        else:
                            existing["result"].extend(detail["result"])
        else:
            result = self.results.setdefault(
                key,
                {"params": {}, "result": None, "opname": [], "reason": None},
            )

        # Keep the first failure because a call-phase assertion normally gives
        # a more useful reason than a later teardown failure.
        if report.failed:
            if report.nodeid not in self.benchmark_items and report.when != "call":
                self.accuracy_error_items.add(report.nodeid)
            if result["result"] != "failed":
                result["result"] = "failed"
                result["reason"] = _report_reason(report)
            return

        if report.skipped:
            if result["result"] not in {"failed", "passed"}:
                result["result"] = "skipped"
                result["reason"] = _report_reason(report)
            return

        if report.when == "call" and report.passed and result["result"] != "failed":
            result["result"] = "passed"
            result["reason"] = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.passed:
            return
        is_benchmark = _is_benchmark_path(Path(report.nodeid), self.rootpath)
        self.collection_errors.append((report.nodeid, is_benchmark, report.skipped))
        self.collection_nodes.add(report.nodeid)
        self.results[report.nodeid] = {
            "params": {},
            "result": "skipped" if report.skipped else "failed",
            "opname": [],
            "reason": _report_reason(report),
        }

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: pytest.ExitCode
    ) -> None:
        summary = _upload_summary(
            self.results,
            self.selected_mark,
            self.collection_errors,
            self.collection_nodes,
            self.accuracy_error_items,
            self.rootpath,
        )
        _merge_json_report(self.output, self.results, summary)


def _normalize_benchmark_detail(value: Any) -> dict[str, Any]:
    """Keep the exact BenchmarkResult/BenchmarkMetrics fields used by FlagGems."""

    if not isinstance(value, Mapping):
        raise TypeError("benchmark result property must be a mapping")
    detail = {field: value[field] for field in _BENCHMARK_DETAIL_FIELDS[:-1]}
    phase = value.get("phase")
    metrics = []
    for source in value["result"]:
        if not isinstance(source, Mapping):
            raise TypeError("benchmark metric must be a mapping")
        # Neither the upload summary nor FlagGems can compute a comparable
        # speedup without two valid timings. Keep baseline-free measurements
        # out of both reports rather than inventing a comparison.
        if not _has_valid_baseline_speedup(source):
            continue
        metric = {field: source.get(field) for field in _BENCHMARK_METRIC_FIELDS}
        metric["latency_base"] = float(metric["latency_base"])
        metric["latency"] = float(metric["latency"])
        metric["speedup"] = metric["latency_base"] / metric["latency"]
        shape = metric["shape_detail"]
        if isinstance(shape, tuple) and all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in shape
        ):
            # A single tensor shape is one entry in FlagGems' argument list.
            shape = [list(shape)]
        if phase:
            # FlagGems represents positional inputs plus keyword arguments as
            # [input_shapes, kwargs]. This keeps equal prefill/decode shapes
            # distinct after the details for one dtype are merged below.
            shape = [shape, {"phase": phase}]
        metric["shape_detail"] = shape
        metrics.append(metric)
    detail["result"] = metrics
    return detail


def _has_valid_baseline_speedup(metric: Mapping[str, Any]) -> bool:
    if metric.get("error_msg") is not None:
        return False
    baseline, latency = metric.get("latency_base"), metric.get("latency")
    if any(not isinstance(value, Real) or isinstance(value, bool)
           for value in (baseline, latency)):
        return False
    try:
        baseline, latency = float(baseline), float(latency)
        ratio = baseline / latency
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        return False
    return all(math.isfinite(value) and value > 0
               for value in (baseline, latency, ratio))


def _upload_shape_key(shape: Any) -> str:
    """Give the platform one stable shape key, retaining distinct phases."""

    phase = None
    if (
        isinstance(shape, list)
        and len(shape) == 2
        and isinstance(shape[1], Mapping)
        and "phase" in shape[1]
    ):
        shape, phase = shape
        phase = phase["phase"]
    if isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], list):
        shape = shape[0]
    if phase and phase != "forward":
        shape = [shape, {"phase": phase}]
    return json.dumps(
        _json_safe(shape), ensure_ascii=False, sort_keys=True, default=str,
        allow_nan=False,
    )


def _upload_phase(shape: Any) -> str:
    if (
        isinstance(shape, list)
        and len(shape) == 2
        and isinstance(shape[1], Mapping)
        and "phase" in shape[1]
    ):
        return str(shape[1]["phase"])
    return ""


def _upload_performance_data(details: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert recorded timings into the guide's dtype and shape summaries."""

    rows_by_dtype: dict[
        str, dict[tuple[str, str, str, str, str], list[tuple[float, float]]]
    ] = {}
    for detail in details:
        dtype = str(detail["dtype"]).removeprefix("torch.")
        shapes = rows_by_dtype.setdefault(dtype, {})
        for metric in detail["result"]:
            if not _has_valid_baseline_speedup(metric):
                continue
            baseline = float(metric["latency_base"])
            latency = float(metric["latency"])
            key = (
                _upload_shape_key(metric["shape_detail"]),
                str(detail["op_name"]),
                str(detail["mode"]),
                str(detail["level"]),
                _upload_phase(metric["shape_detail"]),
            )
            shapes.setdefault(key, []).append((baseline, latency))

    data = {}
    for dtype, shapes in rows_by_dtype.items():
        if not shapes:
            continue
        shape_details = {}
        shape_counts: dict[str, int] = {}
        for shape, _, _, _, _ in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        for (shape, op_name, mode, level, phase), measurements in shapes.items():
            count = len(measurements)
            baseline = math.fsum(row[0] / count for row in measurements)
            latency = math.fsum(row[1] / count for row in measurements)
            # The guide stores details under a string key. Preserve a simple
            # shape key when unique, but distinguish different measurements of
            # the same shape instead of averaging incompatible baselines.
            key = shape
            if shape_counts[shape] > 1:
                identity = {
                    "shape": json.loads(shape), "op_name": op_name,
                    "mode": mode, "level": level,
                }
                if phase:
                    identity["phase"] = phase
                key = json.dumps(
                    identity,
                    ensure_ascii=False, sort_keys=True, allow_nan=False,
                )
            shape_details[key] = {
                "base": baseline,
                "gems": latency,
                "speedup": baseline / latency,
            }
        data[dtype] = {
            "speedup": math.fsum(
                row["speedup"] / len(shape_details)
                for row in shape_details.values()
            ),
            "details": shape_details,
        }
    return data


def _upload_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {"python_version": platform.python_version()}
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        torch_version = getattr(torch_module, "__version__", None)
        if torch_version is not None:
            environment["torch_version"] = str(torch_version)
        cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        if cuda_version:
            environment["cuda_version"] = cuda_version
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            try:
                if cuda.is_available():
                    environment["gpu_model"] = cuda.get_device_name()
            except (AssertionError, RuntimeError):
                pass
    return environment


def _upload_summary(
    results: dict[str, dict[str, Any]],
    selected_mark: str | None,
    collection_errors: list[tuple[str, bool, bool]],
    collection_nodes: set[str],
    accuracy_error_items: set[str],
    rootpath: Path,
) -> dict[str, Any]:
    """Build the upload guide's ``result[operator].performance.data`` tree."""

    groups: dict[str, dict[str, Any]] = {}

    def group(name: str) -> dict[str, Any]:
        return groups.setdefault(
            name,
            {"accuracy_items": [], "accuracy_errors": 0, "accuracy_skips": 0,
             "benchmark": None, "benchmark_errors": 0},
        )

    for key, entry in results.items():
        if key in collection_nodes:
            continue
        if "test_case" in entry and "details" in entry:
            group(key)["benchmark"] = entry
        elif "opname" in entry:
            marks = entry.get("opname") or []
            op_name = _operator_key(marks, selected_mark, key)
            group(op_name)["accuracy_items"].append((key, entry))

    for nodeid, is_benchmark, skipped in collection_errors:
        # Pytest reports module-level importorskip/collection failures before
        # their marks are known. An unrelated module must not change the
        # selected operator's accuracy counts or benchmark status.
        if selected_mark and not _collection_matches_mark(nodeid, selected_mark, rootpath):
            continue
        selected_group = group(selected_mark or nodeid)
        if is_benchmark:
            if not skipped:
                selected_group["benchmark_errors"] += 1
        elif skipped:
            selected_group["accuracy_skips"] += 1
        else:
            selected_group["accuracy_errors"] += 1

    if selected_mark:
        group(selected_mark)

    operators = {}
    for name, entries in groups.items():
        counts = {"passed": 0, "failed": 0, "skipped": entries["accuracy_skips"],
                  "errors": entries["accuracy_errors"]}
        for nodeid, item in entries["accuracy_items"]:
            if nodeid in accuracy_error_items:
                counts["errors"] += 1
            elif item["result"] in counts:
                counts[item["result"]] += 1
        total = sum(counts.values())
        if counts["failed"] or counts["errors"]:
            accuracy_status = "Failed"
        elif counts["passed"]:
            accuracy_status = "Passed"
        elif total:
            accuracy_status = "Skipped"
        else:
            accuracy_status = "NotRun"
        accuracy = {
            **counts,
            "total": total,
            "exit_code": 1 if accuracy_status == "Failed" else 5 if not total else 0,
            "status": accuracy_status,
        }

        benchmark = entries["benchmark"]
        if entries["benchmark_errors"] or (benchmark and benchmark["result"] == "failed"):
            performance = {"status": "Failed", "data": {}}
        elif benchmark and benchmark["result"] == "passed":
            data = _upload_performance_data(benchmark["details"])
            performance = {"status": "Passed" if data else "Skipped", "data": data}
        else:
            performance = {"status": "Skipped", "data": {}}
        operators[name] = {
            "labels": [],
            "accuracy": accuracy,
            "performance": performance,
            "customized": bool(
                entries["accuracy_items"] or benchmark
                or entries["accuracy_errors"] or entries["accuracy_skips"]
                or entries["benchmark_errors"]
            ),
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": _upload_environment(),
        "result": operators,
    }


def _collection_matches_mark(nodeid: str, mark: str, rootpath: Path) -> bool:
    """Attribute an import failure only when its source declares this mark."""

    source_path = Path(nodeid.split("::", 1)[0])
    if not source_path.is_absolute():
        source_path = rootpath / source_path
    filename_matches = source_path.stem in {mark, f"test_{mark}", f"bench_{mark}"}
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError:
        return filename_matches
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return filename_matches or bool(
            re.search(rf"\bpytest\.mark\.{re.escape(mark)}\b", source)
        )
    return filename_matches or any(
        isinstance(node, ast.Attribute)
        and node.attr == mark
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        for node in ast.walk(tree)
    )


def _report_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    if report.skipped and hasattr(report, "wasxfail"):
        return str(report.wasxfail)
    longrepr = report.longrepr
    if hasattr(longrepr, "reprcrash"):
        return str(longrepr.reprcrash.message)
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        return str(longrepr[2])
    if longrepr:
        return str(longrepr)
    return report.outcome


def _is_benchmark_path(path: Path, rootpath: Path) -> bool:
    try:
        if not path.is_absolute():
            path = rootpath / path
        relative = path.resolve().relative_to(rootpath.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == "benchmark"


def _is_benchmark_invocation(config: pytest.Config) -> bool:
    invocation_dir = Path(str(config.invocation_params.dir))
    if _is_benchmark_path(invocation_dir, config.rootpath):
        return True

    for argument in config.args:
        candidate = str(argument).split("::", 1)[0]
        path = Path(candidate)
        if not path.is_absolute():
            path = invocation_dir / path
        if _is_benchmark_path(path, config.rootpath):
            return True
    return False


def _merge_json_report(
    output: Path, results: dict[str, dict[str, Any]], summary: dict[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a+", encoding="utf-8") as report_file:
        fcntl.flock(report_file, fcntl.LOCK_EX)
        try:
            report_file.seek(0)
            content = report_file.read()
            if content.strip():
                existing = json.loads(content)
                if not isinstance(existing, dict):
                    raise ValueError(
                        f"existing JSON report must contain an object: {output}"
                    )
            else:
                existing = {}

            existing.update(results)
            # Keep the FlagGems entries for existing consumers, but the upload
            # summary must describe only this invocation. A reused --output
            # path must not upload an earlier operator's measurements.
            existing.update(summary)
            serialized = json.dumps(
                _json_safe(existing), indent=2, default=str, allow_nan=False,
            )
            report_file.seek(0)
            report_file.truncate()
            report_file.write(serialized + "\n")
            report_file.flush()
            os.fsync(report_file.fileno())
        finally:
            fcntl.flock(report_file, fcntl.LOCK_UN)


def _json_safe(value: Any) -> Any:
    """Replace non-finite optional metrics with JSON null."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
