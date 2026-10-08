# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for grouped benchmark result collection."""

import json
import math

import pytest

from benchmark.recording import (
    BENCHMARK_LOG_PATH_ENV,
    BENCHMARK_RESULT_PROPERTY,
    BenchmarkRecorder,
)


def test_recorder_submits_one_group_with_each_shape_and_metadata():
    properties = []
    recorder = BenchmarkRecorder(
        lambda key, value: properties.append((key, value)),
        op_name="chunk_gla",
        dtype="torch.bfloat16",
        phase="forward",
        baseline="FLA",
    )
    first = recorder.add(
        shape_detail=(1, 128),
        latency_base=2.0,
        latency=1.0,
        speedup=999.0,
        accuracy=0.99,
        case_id="example-forward-1",
        candidate_source=None,
    )
    recorder.add(shape_detail=(2, 256), latency_base=6.0, latency=2.0)

    assert properties == []
    assert first["speedup"] == 2.0
    assert first["accuracy"] == 0.99
    assert first["case_id"] == "example-forward-1"
    assert first["candidate_source"] is None
    recorder.record()

    assert len(properties) == 1
    key, detail = properties[0]
    assert key == BENCHMARK_RESULT_PROPERTY
    assert (detail["op_name"], detail["dtype"], detail["phase"], detail["baseline"]) == (
        "chunk_gla",
        "torch.bfloat16",
        "forward",
        "FLA",
    )
    assert [row["shape_detail"] for row in detail["result"]] == [(1, 128), (2, 256)]
    assert [row["speedup"] for row in detail["result"]] == [2.0, 3.0]
    assert detail["result"][1]["case_id"] is None
    recorder.record()
    assert len(properties) == 1


@pytest.mark.parametrize(
    "baseline,latency",
    [
        (None, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (2.0, 0.0),
        (math.inf, 1.0),
        (2.0, math.nan),
        (1e308, 1e-308),
        (True, 1.0),
        ("2.0", 1.0),
    ],
)
def test_recorder_does_not_invent_speedup_for_invalid_timings(baseline, latency):
    recorder = BenchmarkRecorder(None, op_name="example", dtype="torch.float16")
    metric = recorder.add(shape_detail=(1,), latency_base=baseline, latency=latency)
    assert metric["latency_base"] == baseline
    assert metric["speedup"] is None


def test_recorder_uses_existing_direct_script_stdout_sink(monkeypatch, capsys):
    monkeypatch.delenv(BENCHMARK_LOG_PATH_ENV, raising=False)
    recorder = BenchmarkRecorder(None, op_name="example", dtype="torch.float16")
    recorder.add(shape_detail=(1, 64), latency_base=2.0, latency=1.0)
    recorder.record()

    [line] = capsys.readouterr().out.splitlines()
    assert line.startswith("[INFO] {")
    assert json.loads(line[len("[INFO] ") :])["result"][0]["speedup"] == 2.0


def test_recorder_uses_existing_direct_script_sidecar_sink(monkeypatch, tmp_path, capsys):
    sidecar = tmp_path / "benchmark.log"
    monkeypatch.setenv(BENCHMARK_LOG_PATH_ENV, str(sidecar))
    recorder = BenchmarkRecorder(None, op_name="example", dtype="torch.float16")
    recorder.add(shape_detail=(1, 64), latency_base=2.0, latency=1.0)
    recorder.record()

    assert capsys.readouterr().out == ""
    [line] = sidecar.read_text(encoding="utf-8").splitlines()
    assert line.startswith("[INFO] {")
    assert json.loads(line[len("[INFO] ") :])["result"][0]["speedup"] == 2.0
