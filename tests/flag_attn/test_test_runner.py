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

"""Host-only regressions for test outcomes and the scheduler's exit status."""

import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    # The environment may contain an unrelated installed package named tools.
    # Keep the script importable under its own name for spawned workers too.
    tools_path = str(Path(__file__).resolve().parents[2] / "tools")
    sys.path.insert(0, tools_path)
    try:
        yield importlib.import_module("run_tests")
    finally:
        sys.path.remove(tools_path)


@pytest.mark.parametrize(
    "source,exit_code,status,counts",
    [
        (
            'import pytest\npytest.importorskip("flagattention_missing_test_accelerator")\n',
            5,
            "Skipped",
            {"passed": 0, "failed": 0, "errors": 0, "skipped": 1},
        ),
        (
            "VALUE = 1\n",
            5,
            "NotFound",
            {"passed": 0, "failed": 0, "errors": 0, "skipped": 0},
        ),
        (
            'raise RuntimeError("collection failed")\n',
            2,
            "Error",
            {"passed": 0, "failed": 0, "errors": 1, "skipped": 0},
        ),
        (
            "def test_failure():\n    assert False\n",
            1,
            "Failed",
            {"passed": 0, "failed": 1, "errors": 0, "skipped": 0},
        ),
    ],
    ids=["module-skip", "no-tests", "collection-error", "test-failure"],
)
def test_pytest_junit_outcomes(tmp_path, runner, source, exit_code, status, counts):
    test_file = tmp_path / "test_outcome.py"
    test_file.write_text(source)
    junit = tmp_path / "results.xml"
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            f"--junitxml={junit}",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == exit_code, completed.stdout + completed.stderr
    result = runner.parse_junit(junit, completed.returncode)
    assert result["status"] == status
    assert {key: result[key] for key in counts} == counts
    if status == "Skipped":
        assert "flagattention_missing_test_accelerator" in result["details"]["skipped"][0]["reason"]


@pytest.mark.parametrize("case", ["accuracy", "performance", "no-tests", "no-benchmark"])
def test_run_summary_and_exit_code(tmp_path, monkeypatch, runner, case):
    test_file = tmp_path / "test_operator.py"
    test_file.write_text(
        "VALUE = 1\n" if case == "no-tests" else f"def test_operator():\n    assert {case != 'accuracy'}\n"
    )
    benchmark = tmp_path / "benchmark.py"
    benchmark.write_text(f"raise SystemExit({int(case == 'performance')})\n")
    operator = {
        "id": "host_regression",
        "current_stage": "alpha",
        "tests": [str(test_file)],
        "benchmarks": [] if case == "no-benchmark" else [str(benchmark)],
        "benchmark_requires": [],
    }
    monkeypatch.setattr(runner, "load_inventory", lambda: [operator])
    monkeypatch.setattr(runner, "probe_environment", lambda: {"torch": {"device_count": 1}})
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    output_dir = tmp_path / "results"
    previous_handler = signal.getsignal(signal.SIGTERM)
    previous_workers = list(runner.WORKER_PROCESSES)
    try:
        exit_code = runner.main(
            ["--stages", "all", "--output", str(output_dir), "--color", "never"]
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        runner.WORKER_PROCESSES[:] = previous_workers

    summary = json.loads((output_dir / "summary.json").read_text())
    assert exit_code == (0 if case == "no-benchmark" else 1)
    assert summary["status"] == ("passed" if case == "no-benchmark" else "failed")
    phase = "performance" if case in {"performance", "no-benchmark"} else "accuracy"
    status = "NotFound" if case in {"no-tests", "no-benchmark"} else "Failed"
    assert summary["result"]["host_regression"][phase]["status"] == status


def test_incomplete_worker_results_are_failures(tmp_path, runner):
    results = runner.aggregate_results([{"id": "unfinished"}], [0], tmp_path)
    assert results["unfinished"]["accuracy"]["status"] == "Error"
    assert results["unfinished"]["performance"]["status"] == "Error"
    assert runner.has_failures(results)


def test_standalone_benchmark_keeps_flaggems_compatible_log(tmp_path, runner):
    benchmark = tmp_path / "synthetic_benchmark.py"
    benchmark.write_text(
        "from benchmark.recording import benchmark_metric, record_benchmark_result\n"
        "record_benchmark_result(None, op_name='synthetic', dtype='torch.float16', "
        "result=[benchmark_metric(shape_detail=(2, 128), latency_base=2.0, "
        "latency=1.0, speedup=2.0)])\n"
    )
    output_dir = tmp_path / "results"
    result = runner.run_performance(
        {
            "id": "synthetic",
            "benchmarks": [str(benchmark)],
            "benchmark_requires": [],
        },
        0,
        {
            "root": str(Path(__file__).resolve().parents[2]),
            "output_dir": str(output_dir),
            "python": sys.executable,
            "benchmark_timeout": 30,
            "dump_output": False,
            "skip_benchmarks": False,
        },
    )

    assert result["status"] == "Passed"
    stdout = (output_dir / "synthetic" / "performance_stdout.log").read_text()
    records = [
        json.loads(line[len("[INFO] ") :])
        for line in stdout.splitlines()
        if line.startswith("[INFO] {")
    ]
    assert len(records) == 1
    assert records[0]["op_name"] == "synthetic"
    assert records[0]["result"][0]["speedup"] == 2.0
    assert result["details"][0]["record_count"] == 1


@pytest.mark.parametrize(
    "output",
    [
        "plain benchmark table",
        "[INFO] {malformed json}",
        "[INFO] " + json.dumps({"op_name": "synthetic", "result": [{"speedup": 2.0}]}),
        "[INFO] "
        + json.dumps(
            {
                "op_name": "synthetic",
                "dtype": "float16",
                "mode": "kernel",
                "level": "comprehensive",
                "result": [],
            }
        ),
        "[INFO] "
        + json.dumps(
            {
                "op_name": "synthetic",
                "dtype": "float16",
                "mode": "kernel",
                "level": "comprehensive",
                "result": [{"shape_detail": [1]}],
            }
        ),
        None,
    ],
    ids=[
        "no-record",
        "invalid-json",
        "missing-header",
        "empty-result",
        "missing-metric",
        "silent",
    ],
)
def test_standalone_benchmark_requires_parseable_record(tmp_path, runner, output):
    benchmark = tmp_path / "synthetic_benchmark.py"
    benchmark.write_text("pass\n" if output is None else f"print({output!r})\n")
    output_dir = tmp_path / "results"
    result = runner.run_performance(
        {
            "id": "synthetic",
            "benchmarks": [str(benchmark)],
            "benchmark_requires": [],
        },
        0,
        {
            "root": str(Path(__file__).resolve().parents[2]),
            "output_dir": str(output_dir),
            "python": sys.executable,
            "benchmark_timeout": 30,
            "dump_output": False,
            "skip_benchmarks": False,
        },
    )

    assert result["status"] == "Failed"
    assert result["details"][0]["exit_code"] == 0
    assert result["details"][0]["record_count"] == 0
    assert "valid [INFO] JSON result record" in result["details"][0]["error"]


def test_standalone_benchmark_validates_each_appended_script(tmp_path, runner):
    first = tmp_path / "first.py"
    first.write_text(
        "from benchmark.recording import benchmark_metric, record_benchmark_result\n"
        "record_benchmark_result(None, op_name='synthetic', dtype='torch.float16', "
        "result=[benchmark_metric(shape_detail=(2, 128), latency_base=2.0, "
        "latency=1.0, speedup=2.0)])\n"
    )
    second = tmp_path / "second.py"
    second.write_text("print('human-readable table only')\n")
    output_dir = tmp_path / "results"
    result = runner.run_performance(
        {
            "id": "synthetic",
            "benchmarks": [str(first), str(second)],
            "benchmark_requires": [],
        },
        0,
        {
            "root": str(Path(__file__).resolve().parents[2]),
            "output_dir": str(output_dir),
            "python": sys.executable,
            "benchmark_timeout": 30,
            "dump_output": False,
            "skip_benchmarks": False,
        },
    )

    assert result["status"] == "Failed"
    assert [item["status"] for item in result["details"]] == ["Passed", "Failed"]
    assert [item["record_count"] for item in result["details"]] == [1, 0]


def test_flaggems_record_requires_numeric_timing(tmp_path, runner):
    def log_record(**metric_overrides):
        metric = {
            "shape_detail": [1, 128],
            "latency_base": 2.0,
            "latency": 1.0,
            "speedup": 2.0,
        }
        metric.update(metric_overrides)
        return "[INFO] " + json.dumps(
            {
                "op_name": "synthetic",
                "dtype": "torch.float16",
                "mode": "kernel",
                "level": "comprehensive",
                "result": [metric],
            }
        )

    stdout = tmp_path / "performance_stdout.log"
    stdout.write_text(
        "\n".join(
            [
                log_record(speedup="2x"),
                log_record(latency=float("nan")),
                log_record(latency=None),
                log_record(latency=0),
                log_record(speedup=True),
                log_record(latency_base=None, speedup=None),
            ]
        )
        + "\n"
    )
    assert runner.count_flaggems_records(stdout) == 1
