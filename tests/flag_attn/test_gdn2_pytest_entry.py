# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""The operator marker alone must collect correctness tests and benchmarks."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def repository_conftest():
    path = Path(__file__).resolve().parents[2] / "conftest.py"
    spec = importlib.util.spec_from_file_location("gdn2_pytest_entry_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(root, *, record="none", mark="chunk_gdn2", arguments=None, cwd=None):
    options = dict(
        flag_attn_record=record,
        flag_attn_output=str(root / "results.json"),
        markexpr=mark,
        collect_marks=None,
    )
    return SimpleNamespace(
        rootpath=root,
        invocation_params=SimpleNamespace(dir=cwd or root),
        args=list(arguments if arguments is not None else ["tests"]),
        getoption=options.__getitem__,
        getini=lambda name: ["tests"],
        pluginmanager=SimpleNamespace(register=lambda *args: None),
    )


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("record", ["none", "json"])
def test_marker_collects_benchmarks_without_requiring_record(
    repository_conftest, tmp_path, record
):
    (tmp_path / "benchmark").mkdir()
    (tmp_path / "benchmark/test_latency.py").write_text(
        "import pytest\n@pytest.mark.chunk_gdn2\ndef test_latency():\n    assert True\n"
    )
    config = _config(tmp_path, record=record)
    repository_conftest.pytest_configure(config)
    assert config.args == ["tests", "benchmark/test_latency.py"]


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "case", ["no_marker", "explicit_path", "subdirectory", "no_benchmark"]
)
def test_collection_keeps_explicit_scope(repository_conftest, tmp_path, case):
    if case != "no_benchmark":
        (tmp_path / "benchmark").mkdir()
    kwargs = {}
    if case == "no_marker":
        kwargs["mark"] = ""
    elif case == "explicit_path":
        kwargs["arguments"] = ["tests/flag_attn/test_gdn2.py"]
    elif case == "subdirectory":
        kwargs["cwd"] = tmp_path / "tests"
    config = _config(tmp_path, **kwargs)
    original = list(config.args)
    repository_conftest.pytest_configure(config)
    assert config.args == original


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("record", [False, True])
def test_marker_only_command_runs_accuracy_and_benchmark(
    repository_conftest, tmp_path, record
):
    source = Path(repository_conftest.__file__).read_text()
    (tmp_path / "conftest.py").write_text(source)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        'markers = ["chunk_gdn2: GDN2 operator"]\n'
    )
    for directory, name in (("tests", "accuracy"), ("benchmark", "latency")):
        path = tmp_path / directory
        path.mkdir()
        (path / f"test_{name}.py").write_text(
            f"import pytest\n@pytest.mark.chunk_gdn2\ndef test_{name}():\n    assert True\n"
        )
    (tmp_path / "benchmark/test_unrelated.py").write_text(
        'import pytest\nraise AssertionError("Unrelated benchmark imported")\n'
        "@pytest.mark.unrelated_operator\ndef test_unrelated():\n    assert True\n"
    )
    command = [sys.executable, "-m", "pytest", "-m", "chunk_gdn2"]
    if record:
        command.extend(["--record", "json"])
    env = dict(
        os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1"
    )
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout, result.stdout
