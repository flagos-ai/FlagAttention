# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pytest recording entry; the standalone CLI expands the complete frozen suite."""

import json

import pytest
import torch

pytest.importorskip("triton.experimental.tle.language", reason="FlagTree/TLE required")

from benchmark.bench_gdn_recurrent import main
from benchmark.recording import benchmark_metric, record_benchmark_result

pytestmark = [
    pytest.mark.gdn_forward,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required",
    ),
]


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param("recurrent", marks=pytest.mark.fused_recurrent_gated_delta_rule_fwd),
        pytest.param("packed", marks=pytest.mark.fused_recurrent_gated_delta_rule_packed_decode),
    ],
)
@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
def test_fused_recurrent_gdn_benchmark(monkeypatch, tmp_path, record_property, operator, state_dtype):
    output = tmp_path / "recurrent.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_gdn_recurrent.py",
            "--operator",
            operator,
            "--state-dtype",
            state_dtype,
            "--presets",
            "g2_tp2",
            "--n",
            "1",
            "16",
            "128",
            "--length",
            "1",
            *(["4"] if operator == "recurrent" else []),
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads(output.read_text())
    metrics = [
        benchmark_metric(
            case_id=row["case_id"] + "_state_" + state_dtype,
            shape_detail={**{key: row[key] for key in ("N", "L", "H", "HV")}, "state_dtype": state_dtype},
            latency=row["tle_us"] / 1000,
            latency_base=row["base_us"] / 1000,
            speedup=row["speedup"],
            accuracy=row["correctness"],
        )
        for row in result["results"]
    ]
    record_benchmark_result(
        record_property,
        op_name="fused_recurrent_gated_delta_rule_fwd"
        if operator == "recurrent"
        else "fused_recurrent_gated_delta_rule_packed_decode",
        dtype="bfloat16",
        mode="kernel_graph",
        result=metrics,
        provenance=result["provenance"],
    )
    assert len(result["results"]) == (6 if operator == "recurrent" else 3)
