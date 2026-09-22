# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-only integration tests for the repository's pytest JSON recorder."""

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_pytest(tmp_path, test_name, source, *extra_args):
    test_file = tmp_path / test_name
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(source, encoding="utf-8")
    return _invoke_pytest(
        tmp_path, [str(test_file.relative_to(tmp_path))], *extra_args
    )


def _invoke_pytest(tmp_path, test_names, *extra_args):
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PROJECT_ROOT), pythonpath) if part
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *test_names,
            "-p",
            "conftest",
            "-p",
            "no:cacheprovider",
            "-q",
            *extra_args,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_record_json_outcomes_params_markers_and_merge(tmp_path):
    report = tmp_path / "reports" / "accuracy.json"
    first_run = _run_pytest(
        tmp_path,
        "test_outcomes.py",
        """
import pytest

@pytest.mark.operator_name
@pytest.mark.parametrize("value", [3])
def test_pass(value):
    assert value == 3

def test_fail():
    assert False, "accuracy mismatch"

@pytest.mark.skip(reason="accelerator unavailable")
def test_skip():
    pass

@pytest.mark.xfail(reason="known accuracy issue")
def test_xfail():
    assert False, "internal numeric mismatch"

@pytest.fixture
def broken_setup():
    raise RuntimeError("setup failed")

def test_setup_error(broken_setup):
    pass

@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("teardown failed")

def test_teardown_error(broken_teardown):
    pass
""",
        "--record",
        "json",
        "--output",
        str(report),
    )
    assert first_run.returncode == 1, first_run.stdout + first_run.stderr

    data = json.loads(report.read_text(encoding="utf-8"))
    passed = data["test_outcomes.py::test_pass[3]"]
    assert passed == {
        "params": {"value": 3},
        "result": "passed",
        "opname": ["operator_name"],
        "reason": None,
    }
    assert data["test_outcomes.py::test_fail"]["result"] == "failed"
    assert "accuracy mismatch" in data["test_outcomes.py::test_fail"]["reason"]
    assert data["test_outcomes.py::test_skip"]["result"] == "skipped"
    assert "accelerator unavailable" in data["test_outcomes.py::test_skip"]["reason"]
    assert data["test_outcomes.py::test_xfail"]["result"] == "skipped"
    assert data["test_outcomes.py::test_xfail"]["reason"] == "known accuracy issue"
    assert data["test_outcomes.py::test_setup_error"]["result"] == "failed"
    assert "setup failed" in data["test_outcomes.py::test_setup_error"]["reason"]
    assert data["test_outcomes.py::test_teardown_error"]["result"] == "failed"
    assert "teardown failed" in data["test_outcomes.py::test_teardown_error"]["reason"]

    second_run = _run_pytest(
        tmp_path,
        "test_more.py",
        "def test_more():\n    assert True\n",
        "--record",
        "json",
        "--output",
        str(report),
    )
    assert second_run.returncode == 0, second_run.stdout + second_run.stderr
    merged = json.loads(report.read_text(encoding="utf-8"))
    assert "test_outcomes.py::test_pass[3]" in merged
    assert merged["test_more.py::test_more"]["result"] == "passed"

    bad_collection = tmp_path / "test_bad_collection.py"
    bad_collection.write_text(
        'raise RuntimeError("collection failed")\n', encoding="utf-8"
    )
    good_collection = tmp_path / "test_good_collection.py"
    good_collection.write_text(
        "def test_still_runs():\n    assert True\n", encoding="utf-8"
    )
    skipped_collection = tmp_path / "test_skipped_collection.py"
    skipped_collection.write_text(
        'import pytest\npytest.importorskip("missing_test_accelerator")\n',
        encoding="utf-8",
    )
    collection_run = _invoke_pytest(
        tmp_path,
        [bad_collection.name, good_collection.name, skipped_collection.name],
        "--continue-on-collection-errors",
        "--record",
        "json",
        "--output",
        str(report),
    )
    assert collection_run.returncode == 1, collection_run.stdout + collection_run.stderr
    collected = json.loads(report.read_text(encoding="utf-8"))
    assert collected["test_bad_collection.py"]["result"] == "failed"
    assert "collection failed" in collected["test_bad_collection.py"]["reason"]
    assert collected["test_good_collection.py::test_still_runs"]["result"] == "passed"
    assert collected["test_skipped_collection.py"]["result"] == "skipped"
    assert "missing_test_accelerator" in collected["test_skipped_collection.py"]["reason"]


def test_output_alone_does_not_enable_recording(tmp_path):
    report = tmp_path / "ignored.json"
    completed = _run_pytest(
        tmp_path,
        "test_pass.py",
        "def test_pass():\n    assert True\n",
        "--output",
        str(report),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not report.exists()

    default_output = _run_pytest(
        tmp_path,
        "test_default_output.py",
        "def test_default_output():\n    assert True\n",
        "--record",
        "json",
    )
    assert default_output.returncode == 0, default_output.stdout + default_output.stderr
    default_report = tmp_path / "accuracy_result.json"
    data = json.loads(default_report.read_text(encoding="utf-8"))
    assert data["test_default_output.py::test_default_output"]["result"] == "passed"


def test_benchmark_record_uses_flaggems_schema(tmp_path):
    report = tmp_path / "benchmark_custom.json"
    completed = _run_pytest(
        tmp_path,
        "benchmark/test_perf.py",
        """
import pytest

from benchmark.recording import benchmark_metric, record_benchmark_result

@pytest.mark.demo_benchmark
def test_perf(record_property):
    metrics = [
        benchmark_metric(
            shape_detail=(1, 128),
            latency_base=2.5,
            latency=1.25,
            speedup=2.0,
        ),
        benchmark_metric(
            shape_detail=(2, 256),
            latency_base=4.5,
            latency=1.5,
            speedup=3.0,
        ),
    ]
    record_benchmark_result(
        record_property,
        op_name="demo_benchmark",
        dtype="torch.float16",
        result=metrics,
        baseline="torch",
        phase="forward",
    )

@pytest.mark.partial_benchmark
def test_partial_failure(record_property):
    record_benchmark_result(
        record_property,
        op_name="partial_benchmark",
        dtype="torch.bfloat16",
        result=[benchmark_metric(shape_detail=(4,), latency=0.5)],
    )
    assert False, "benchmark failed after one result"

@pytest.mark.skipped_benchmark
@pytest.mark.skip(reason="benchmark accelerator unavailable")
def test_skipped():
    pass
""",
        "--record",
        "json",
        "--output",
        str(report),
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr

    data = json.loads(report.read_text(encoding="utf-8"))
    benchmark = data["demo_benchmark"]
    assert benchmark["result"] == "passed"
    assert benchmark["reason"] is None
    assert benchmark["test_case"] == "benchmark/test_perf.py::test_perf"
    assert len(benchmark["details"]) == 1
    detail = benchmark["details"][0]
    assert detail["op_name"] == "demo_benchmark"
    assert detail["dtype"] == "torch.float16"
    assert detail["baseline"] == "torch"
    assert detail["phase"] == "forward"
    assert detail["result"][0]["shape_detail"] == [1, 128]
    assert detail["result"][0]["latency_base"] == 2.5
    assert detail["result"][0]["latency"] == 1.25
    assert detail["result"][0]["speedup"] == 2.0

    partial = data["partial_benchmark"]
    assert partial["result"] == "failed"
    assert "benchmark failed after one result" in partial["reason"]
    assert len(partial["details"]) == 1
    skipped = data["skipped_benchmark"]
    assert skipped["details"] == []
    assert skipped["result"] == "skipped"
    assert "benchmark accelerator unavailable" in skipped["reason"]

    default_run = _run_pytest(
        tmp_path,
        "benchmark/test_default_report.py",
        """
import pytest

@pytest.mark.default_benchmark
def test_default_report():
    pass
""",
        "--record",
        "json",
    )
    assert default_run.returncode == 0, default_run.stdout + default_run.stderr
    default_report = tmp_path / "benchmark_result.json"
    default_data = json.loads(default_report.read_text(encoding="utf-8"))
    assert default_data["default_benchmark"]["result"] == "passed"
