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

"""Regression tests for the FlagGems-compatible FlagAttention runner."""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner():
    """Import the script as a module without requiring accelerator hardware."""

    tools_path = str(Path(__file__).resolve().parents[2] / "tools")
    sys.path.insert(0, tools_path)
    fake_modules = {}
    sentinel = object()

    try:
        fake_flag_attn = types.ModuleType("flag_attn")
        fake_flag_attn.__version__ = "test-version"
        fake_flag_attn.vendor_name = "nvidia"
        fake_flag_attn.device = "cuda"
        fake_modules["flag_attn"] = sys.modules.get("flag_attn", sentinel)
        sys.modules["flag_attn"] = fake_flag_attn

        fake_distro = types.ModuleType("distro")
        fake_distro.id = lambda: "test-os"
        fake_distro.version = lambda: "test-version"
        fake_modules["distro"] = sys.modules.get("distro", sentinel)
        sys.modules["distro"] = fake_distro

        yield importlib.import_module("run_tests")
    finally:
        sys.modules.pop("run_tests", None)
        sys.path.remove(tools_path)
        for name, previous in fake_modules.items():
            if previous is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_parse_perf_data_emits_upload_shape_and_dtype_tree(tmp_path, runner):
    report = tmp_path / "benchmark.json"
    report.write_text(
        json.dumps(
            {
                "sample": {
                    "result": "passed",
                    "test_case": "benchmark/test_sample.py::test_sample",
                    "details": [
                        {
                            "dtype": "torch.float16",
                            "result": [
                                {
                                    "shape_detail": [1, 64],
                                    "latency_base": 2.0,
                                    "latency": 1.0,
                                    "speedup": 2.0,
                                },
                                {
                                    "shape_detail": [128, 256],
                                    "latency_base": 4.0,
                                    "latency": 2.0,
                                    "speedup": 2.0,
                                },
                            ],
                        },
                        {
                            "dtype": "torch.float32",
                            "result": [
                                {
                                    "shape_detail": [1, 64],
                                    "latency_base": 3.0,
                                    "latency": 2.0,
                                    "speedup": 1.5,
                                }
                            ],
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.parse_perf_data("sample", report)

    assert result["status"] == "Passed"
    assert result["test_case"].endswith("test_sample")
    assert result["data"]["fp16"]["speedup"] == 2.0
    assert result["data"]["fp16"]["details"]["[1,64]"] == {
        "base": 2.0,
        "gems": 1.0,
        "speedup": 2.0,
    }
    assert result["data"]["fp32"]["speedup"] == 1.5


def test_parse_perf_data_keeps_missing_speedup_as_null(tmp_path, runner):
    report = tmp_path / "benchmark.json"
    report.write_text(
        json.dumps(
            {
                "sample": {
                    "result": "passed",
                    "details": [
                        {
                            "dtype": "torch.float8_e5m2",
                            "native_baseline_skip_reason": "baseline unavailable",
                            "result": [
                                {
                                    "shape_detail": [1],
                                    "latency_base": None,
                                    "latency": 1.0,
                                    "speedup": None,
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.parse_perf_data("sample", report)

    assert result["status"] == "Passed"
    assert result["data"]["float8_e5m2"]["result"] == "OK"
    assert result["data"]["float8_e5m2"]["speedup"] is None


def test_probe_flag_attn_records_library_metadata(runner):
    runner.ENV_INFO.clear()
    runner._probe_flagattn()

    assert runner.ENV_INFO["flag_attn"] == {
        "version": "test-version",
        "vendor": "nvidia",
        "device": "cuda",
    }


def test_probe_env_preserves_all_shared_environment_sections(monkeypatch, runner):
    runner.ENV_INFO.clear()

    monkeypatch.setattr(runner.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runner.platform, "python_version", lambda: "3.12.0")
    monkeypatch.setattr(runner.distro, "id", lambda: "ubuntu")
    monkeypatch.setattr(runner.distro, "version", lambda: "24.04")

    def probe_torch():
        runner.ENV_INFO["torch"] = {"version": "test-torch"}

    def probe_triton():
        runner.ENV_INFO["flagtree"] = "test-flagtree"
        runner.ENV_INFO["triton"] = {"version": "test-triton"}

    def probe_flag_attn():
        runner.ENV_INFO["flag_attn"] = {"version": "test-flag-attn"}

    def probe_vllm():
        runner.ENV_INFO["vllm"] = {"version": "test-vllm"}

    def probe_sdk():
        runner.ENV_INFO["sdk"] = {"name": "test-sdk", "version": "test-sdk-version"}

    monkeypatch.setattr(runner, "_probe_torch", probe_torch)
    monkeypatch.setattr(runner, "_probe_triton", probe_triton)
    monkeypatch.setattr(runner, "_probe_flagattn", probe_flag_attn)
    monkeypatch.setattr(runner, "_probe_vllm", probe_vllm)
    monkeypatch.setattr(runner, "_probe_sdk", probe_sdk)

    runner.probe_env()

    assert runner.ENV_INFO == {
        "architecture": "x86_64",
        "os_name": "ubuntu",
        "os_release": "24.04",
        "python": "3.12.0",
        "torch": {"version": "test-torch"},
        "flagtree": "test-flagtree",
        "triton": {"version": "test-triton"},
        "flag_attn": {"version": "test-flag-attn"},
        "vllm": {"version": "test-vllm"},
        "sdk": {"name": "test-sdk", "version": "test-sdk-version"},
    }


def test_get_env_matches_shared_vendor_masking_without_path_injection(
    monkeypatch, runner
):
    runner.ENV_INFO.clear()
    runner.ENV_INFO["flag_attn"] = {"vendor": "nvidia"}
    monkeypatch.setenv("PYTHONPATH", "caller-controlled-path")

    env = runner.get_env("2")

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["PYTHONPATH"] == "caller-controlled-path"


def test_collect_marks_only_collects_tests(monkeypatch, tmp_path, runner):
    ops = [f"op_{index}" for index in range(11)]
    calls = []

    def fake_call(command, **kwargs):
        calls.append((command, kwargs))
        marks_path = next(
            Path(argument.split("=", 1)[1])
            for argument in command
            if argument.startswith("--collect-marks=")
        )
        marks_path.write_text("[]\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(runner.subprocess, "call", fake_call)
    accuracy_marks, benchmark_marks = runner.collect_marks(ops)

    assert accuracy_marks == set(ops)
    assert benchmark_marks == set(ops)
    assert [command[-1] for command, _ in calls] == ["tests/", "benchmark/"]
    assert all("--collect-only" in command for command, _ in calls)


def test_runner_commands_and_summary_processing_match_shared_flow(
    monkeypatch, tmp_path, runner
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    runner.ENV_INFO.clear()
    runner.ENV_INFO["flag_attn"] = {"vendor": "nvidia"}
    runner.CFG.output_dir = tmp_path / "results"
    runner.CFG.dump_output = False
    runner.CFG.quick = False
    runner.CFG.all_op_ids = {"sample"}
    runner.CFG.customized_ops = {"sample"}
    runner.CFG.output_dir.mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "benchmark").mkdir()

    commands = []

    def fake_run_cmd(op, cmd, cwd=None, env=None, timeout=1800, flavor=None):
        commands.append((cmd, flavor, cwd, env))
        if flavor == "accuracy":
            (Path(cwd) / "accuracy_sample.json").write_text(
                json.dumps({"sample[case]": {"result": "passed", "params": {}}}),
                encoding="utf-8",
            )
        else:
            (Path(cwd) / "benchmark_sample.json").write_text(
                json.dumps(
                    {
                        "sample": {
                            "result": "passed",
                            "details": [
                                {
                                    "dtype": "torch.float16",
                                    "result": [
                                        {
                                            "shape_detail": [1],
                                            "latency_base": 2.0,
                                            "latency": 1.0,
                                            "speedup": 2.0,
                                        }
                                    ],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
        return 0

    monkeypatch.setattr(runner, "run_cmd", fake_run_cmd)

    accuracy = runner.run_accuracy_q(0, "sample")
    performance = runner.run_benchmark_q(0, "sample")

    assert commands[0][0] == (
        'pytest -m "sample" --record json --output accuracy_sample.json '
        "--continue-on-collection-errors -vs"
    )
    assert commands[1][0] == (
        'pytest -m "sample" --level core --record json '
        "--output benchmark_sample.json --continue-on-collection-errors"
    )
    assert accuracy["status"] == "Passed"
    assert performance["data"]["fp16"]["details"]["[1]"]["speedup"] == 2.0
    assert (runner.CFG.output_dir / "sample" / "accuracy_result.json").exists()
    assert (runner.CFG.output_dir / "sample" / "performance_result.json").exists()


def test_inventory_targets_preserve_explicit_test_nodes(monkeypatch, tmp_path, runner):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "get_ops_from_inventory", lambda: [{
        "id": "sample",
        "tests": ["tests/test_sample.py::test_boundary"],
        "benchmark": "benchmark/test_sample.py",
    }])
    import shlex

    assert shlex.split(runner.inventory_targets("sample", "tests")) == [
        str(tmp_path / "tests/test_sample.py::test_boundary")
    ]
    assert shlex.split(runner.inventory_targets("sample", "benchmark")) == [
        str(tmp_path / "benchmark/test_sample.py")
    ]
    assert runner.inventory_targets("unknown", "tests") == ""


def test_accuracy_only_reports_benchmark_skipped(monkeypatch, runner):
    monkeypatch.setattr(runner.CFG, "skip_benchmarks", True, raising=False)
    assert runner.run_benchmark_q(0, "sample") == {
        "status": "Skipped", "exit_code": 0, "duration": 0, "data": {},
    }


@pytest.mark.parametrize("case,expected", [
    ("passed", 0), ("failed", 1), ("error", 1), ("timeout", 1),
    ("notfound", 1), ("all_skipped", 1), ("zero_passed", 1),
    ("partial_skip", 0), ("missing_operator", 1), ("missing_worker", 1),
    ("invalid_worker", 1), ("performance_failed", 1),
    ("performance_skipped", 1), ("skip_benchmarks", 0),
    ("worker_crashed", 1), ("missing_accuracy", 1),
    ("missing_performance", 1), ("non_strict_failure", 0),
])
def test_strict_results_actual_cli_exit_code(tmp_path, case, expected):
    import subprocess

    script = tmp_path / "simulate_runner.py"
    tools = Path(__file__).resolve().parents[2] / "tools"
    script.write_text(
        '''import json,sys,types
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
case = sys.argv.pop(1)
fake = types.ModuleType("flag_attn")
fake.__version__ = "test"
sys.modules["flag_attn"] = fake
distro = types.ModuleType("distro")
sys.modules["distro"] = distro
import run_tests as r
accuracy = dict(status="Passed", passed=1, failed=0, skipped=0, exit_code=0)
performance = dict(status="Passed", exit_code=0, data={})
if case in ("failed","error","timeout","notfound"):
    accuracy["status"] = dict(failed="Failed",error="Error",timeout="Timeout",notfound="NotFound")[case]
if case == "all_skipped": accuracy.update(status="Skipped",passed=0,skipped=2)
if case == "zero_passed": accuracy["passed"] = 0
if case == "partial_skip": accuracy["skipped"] = 2
if case == "performance_failed": performance["status"] = "Failed"
if case in ("performance_skipped","skip_benchmarks"): performance["status"] = "Skipped"
if case == "non_strict_failure": accuracy["status"] = "Failed"
entry = dict(accuracy=accuracy,performance=performance)
if case == "missing_accuracy": entry.pop("accuracy")
if case == "missing_performance": entry.pop("performance")
class FakeProcess:
    def __init__(self, **kw): self.exitcode = 7 if case == "worker_crashed" else 0
    def start(self):
        if case != "missing_worker":
            data = {} if case == "missing_operator" else {"sample":entry}
            (r.CFG.output_dir / "summary0.json").write_text("invalid" if case == "invalid_worker" else json.dumps(data))
    def join(self): pass
class Display:
    def __init__(self,*a,**kw): pass
    def init(self): pass
    def finish(self): pass
r.Process=FakeProcess
r.LiveDisplay=Display
r.display_loop=lambda *a:None
r.probe_env=lambda:None
r.collect_marks=lambda ops:(set(ops),set(ops))
r.get_ops_to_test=lambda:["sample"]
sys.exit(r.main())
''', encoding="utf-8",
    )
    command = [sys.executable, str(script), str(tools), case, "--ops", "sample",
               "--gpus", "0", "--output-dir", str(tmp_path / "results"), "--color", "never"]
    if case != "non_strict_failure":
        command.append("--strict-results")
    if case == "skip_benchmarks":
        command.append("--skip-benchmarks")
    result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    assert result.returncode == expected, result.stdout + result.stderr
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert "result" in summary
    if case == "partial_skip":
        assert summary["result"]["sample"]["accuracy"]["skipped"] == 2


def test_interrupt_still_cleans_up_and_exits_nonzero(monkeypatch, runner):
    calls = []
    monkeypatch.setattr(runner, "INTERRUPTED", False)
    monkeypatch.setattr(runner, "terminate_workers", lambda: calls.append("workers"))
    monkeypatch.setattr(runner, "cleanup_intermediate_files", lambda: calls.append("files"))
    with pytest.raises(SystemExit) as exc:
        runner.handle_interrupt(None, None)
    assert exc.value.code == 1
    assert calls == ["workers", "files"]


@pytest.mark.parametrize("kind", ["tests", "benchmark"])
def test_inventory_nodes_are_pytest_collectable_from_worker_cwd(monkeypatch, tmp_path, runner, kind):
    import shlex
    import subprocess

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    folder = tmp_path / kind
    folder.mkdir()
    (folder / "test_sample.py").write_text("def test_boundary(): pass\ndef test_other(): pass\n", encoding="utf-8")
    (folder / "test_optional_backend.py").write_text("raise ImportError('unrelated optional backend')\n", encoding="utf-8")
    monkeypatch.setattr(runner, "get_ops_from_inventory", lambda: [{
        "id": "sample", "tests": ["tests/test_sample.py::test_boundary"],
        "benchmark": "benchmark/test_sample.py::test_boundary",
    }])
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *shlex.split(runner.inventory_targets("sample", kind))],
        cwd=folder, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_boundary" in result.stdout and "test_other" not in result.stdout
