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

    monkeypatch.setattr(runner, "_probe_torch", probe_torch)
    monkeypatch.setattr(runner, "_probe_triton", probe_triton)
    monkeypatch.setattr(runner, "_probe_flagattn", probe_flag_attn)
    monkeypatch.setattr(runner, "_probe_vllm", probe_vllm)

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
