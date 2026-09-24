# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the FlagGems benchmark log contract."""

import json
from types import SimpleNamespace

import pytest

from benchmark.recording import (
    benchmark_metric,
    record_benchmark_result,
    record_triton_report,
)


class _Frame:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return enumerate(self.rows)


def _records(stdout):
    return [
        json.loads(line[len("[INFO] ") :])
        for line in stdout.splitlines()
        if line.startswith("[INFO] {")
    ]


def test_triton_report_records_latency_and_speedup(capsys):
    config = SimpleNamespace(
        line_vals=["flag_attn", "torch"],
        line_names=["flag_attn", "torch"],
        ylabel="tflop/s",
        args={"dtype": "torch.float16", "mode": "fwd"},
        x_names=["N_CTX"],
    )
    frame = _Frame(
        [
            {
                "N_CTX": 512.0,
                "flag_attn (tflop/s)": 20.0,
                "torch (tflop/s)": 10.0,
            },
            {
                "N_CTX": 1024.0,
                "flag_attn (tflop/s)": 30.0,
                "torch (tflop/s)": 0.0,
            },
        ]
    )
    report = SimpleNamespace(benchmarks=[config])
    record_triton_report(
        report,
        [frame],
        op_name="flash_attention",
        provider="flag_attn",
        baseline="torch",
        latency_from_value=lambda value, shape: 20.0 / value if value > 0 else float("inf"),
    )

    [record] = _records(capsys.readouterr().out)
    assert (record["op_name"], record["dtype"], record["mode"], record["level"]) == (
        "flash_attention",
        "torch.float16",
        "kernel",
        "comprehensive",
    )
    first, missing_baseline = record["result"]
    assert first["shape_detail"]["N_CTX"] == 512
    assert first["latency_base"] == 2.0
    assert first["latency"] == 1.0
    assert first["speedup"] == 2.0
    assert missing_baseline["latency_base"] is None
    assert missing_baseline["speedup"] is None
    assert missing_baseline["error_msg"]


def test_triton_report_without_baseline_does_not_invent_speedup(capsys):
    config = SimpleNamespace(
        line_vals=["triton"],
        line_names=["triton"],
        ylabel="ms",
        args={"dtype": "torch.float16"},
        x_names=["context_len"],
    )
    frame = _Frame([{"context_len": 512, "triton (ms)": 0.5}])
    record_triton_report(
        SimpleNamespace(benchmarks=[config]),
        [frame],
        op_name="paged_attention",
        provider="triton",
        baseline="vllm",
        latency_from_value=lambda value, shape: value,
    )

    [record] = _records(capsys.readouterr().out)
    [metric] = record["result"]
    assert metric["latency"] == 0.5
    assert metric["latency_base"] is None
    assert metric["speedup"] is None
    assert metric["error_msg"] is None
    assert record["baseline"] is None


@pytest.mark.parametrize(
    "op_name,phases",
    [
        ("chunk_gla", ["forward", "forward_backward"]),
        ("minimax_m3_sparse_attn", ["prefill", "decode"]),
    ],
)
def test_multi_phase_records_keep_distinct_log_names(capsys, op_name, phases):
    properties = []
    for phase in phases:
        record_benchmark_result(
            lambda key, value: properties.append((key, value)),
            op_name=op_name,
            dtype="torch.bfloat16",
            result=[
                benchmark_metric(
                    shape_detail={"phase": phase},
                    latency_base=2.0,
                    latency=1.0,
                    speedup=2.0,
                )
            ],
            phase=phase,
        )

    assert [record["op_name"] for record in _records(capsys.readouterr().out)] == [
        f"{op_name}_{phase}" for phase in phases
    ]
    assert [value["op_name"] for _, value in properties] == [op_name] * len(phases)
