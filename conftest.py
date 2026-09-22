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
_DEFAULT_REPORT_FILE = "accuracy_result.json"


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
        help=f"JSON result path (default: {_DEFAULT_REPORT_FILE})",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("flag_attn_record") != "json":
        return

    output = config.getoption("flag_attn_output") or _DEFAULT_REPORT_FILE
    recorder = _JsonResultRecorder(Path(output))
    config.pluginmanager.register(recorder, "flag-attention-json-result-recorder")


class _JsonResultRecorder:
    """Collect pytest outcomes using the JSON schema consumed by FlagGems."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.results: dict[str, dict[str, Any]] = {}

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None):
        callspec = getattr(item, "callspec", None)
        params = dict(callspec.params) if callspec is not None else {}
        operator_marks = [
            mark.name
            for mark in item.iter_markers()
            if mark.name not in _BUILTIN_MARKS
        ]
        self.results[item.nodeid] = {
            "params": params,
            "result": None,
            "opname": operator_marks,
            "reason": None,
        }

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        result = self.results.setdefault(
            report.nodeid,
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
