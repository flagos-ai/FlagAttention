# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Repository-wide pytest configuration."""

from __future__ import annotations

import fcntl
import json
import os
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
        output = config.getoption("flag_attn_output")
        if output is None:
            output = (
                _DEFAULT_BENCHMARK_REPORT_FILE
                if _is_benchmark_invocation(config)
                else _DEFAULT_ACCURACY_REPORT_FILE
            )
        recorder = _JsonResultRecorder(Path(output), config.rootpath)
        config.pluginmanager.register(recorder, "flag-attention-json-result-recorder")

    marks_output = config.getoption("collect_marks")
    if marks_output:
        collector = _MarkCollector(Path(marks_output))
        config.pluginmanager.register(collector, "flag-attention-mark-collector")


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
    """Collect pytest outcomes using the JSON schema consumed by FlagGems."""

    def __init__(self, output: Path, rootpath: Path) -> None:
        self.output = output
        self.rootpath = rootpath
        self.results: dict[str, dict[str, Any]] = {}
        self.item_keys: dict[str, str] = {}
        self.benchmark_items: set[str] = set()

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
            key = operator_marks[0] if operator_marks else item.nodeid
            self.item_keys[item.nodeid] = key
            self.benchmark_items.add(item.nodeid)
            result = self.results.setdefault(key, {"details": []})
            result.update(
                {
                    "result": None,
                    "test_case": item.nodeid,
                    "reason": None,
                }
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
                        result["details"].append(value)
        else:
            result = self.results.setdefault(
                key,
                {"params": {}, "result": None, "opname": [], "reason": None},
            )

        # Keep the first failure because a call-phase assertion normally gives
        # a more useful reason than a later teardown failure.
        if report.failed:
            if result["result"] != "failed":
                result["result"] = "failed"
                result["reason"] = _report_reason(report)
            return

        if report.skipped:
            if result["result"] != "failed":
                result["result"] = "skipped"
                result["reason"] = _report_reason(report)
            return

        if report.when == "call" and report.passed:
            result["result"] = "passed"
            result["reason"] = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.passed:
            return
        self.results[report.nodeid] = {
            "params": {},
            "result": "skipped" if report.skipped else "failed",
            "opname": [],
            "reason": _report_reason(report),
        }

    def pytest_sessionfinish(
        self, session: pytest.Session, exitstatus: pytest.ExitCode
    ) -> None:
        _merge_json_report(self.output, self.results)


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
    output: Path, results: dict[str, dict[str, Any]]
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
            report_file.seek(0)
            report_file.truncate()
            json.dump(existing, report_file, indent=2, default=str)
            report_file.write("\n")
            report_file.flush()
            os.fsync(report_file.fileno())
        finally:
            fcntl.flock(report_file, fcntl.LOCK_UN)
