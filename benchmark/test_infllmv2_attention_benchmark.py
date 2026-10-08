# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Pytest-native InfLLM-V2 performance suite.

The suite owns its data generation, timing and reporting; it does not import
``infllmv2_attention_benchmark.py`` or either correctness-test module.

Environment controls:
    INFLLMV2_BENCH_WARMUP_MS (default: 20)
    INFLLMV2_BENCH_REP_MS    (default: 100)

Examples:
    pytest benchmark/test_infllmv2_attention_benchmark.py -m infllmv2_attention -q
    pytest benchmark/test_infllmv2_attention_benchmark.py -m infllmv2_decode -q
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import pytest
import torch
import triton

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:  # Direct invocation from the benchmark directory.
    from recording import benchmark_metric, record_benchmark_result

from flag_attn import InfLLMV2Config, infllmv2_attention, infllmv2_decode


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for performance tests"
)
ATTENTION = pytest.mark.infllmv2_attention
DECODE = pytest.mark.infllmv2_decode
WARMUP_MS = int(os.getenv("INFLLMV2_BENCH_WARMUP_MS", "20"))
REP_MS = int(os.getenv("INFLLMV2_BENCH_REP_MS", "100"))
_RESULTS: list[dict[str, object]] = []
_COLUMNS = (
    ("operator", "operator"),
    ("case", "case"),
    ("mode", "mode"),
    ("batch", "B"),
    ("q_length", "Q"),
    ("kv_length", "KV"),
    ("head_dim", "D"),
    ("dtype", "dtype"),
    ("warmup_ms", "warmup_ms"),
    ("rep_ms", "rep_ms"),
    ("median_ms", "median_ms"),
)
_RIGHT_ALIGNED = {
    "batch",
    "q_length",
    "kv_length",
    "head_dim",
    "warmup_ms",
    "rep_ms",
    "median_ms",
}


def _display_value(key: str, value: object) -> str:
    if key == "median_ms":
        return f"{float(value):.4f}"
    return str(value)


def _write_benchmark_summary(terminalreporter) -> None:
    """Print the selected benchmark rows as one borderless table."""
    if not _RESULTS:
        return

    rows = [
        [_display_value(key, result[key]) for key, _ in _COLUMNS]
        for result in _RESULTS
    ]
    widths = [
        max(len(title), *(len(row[index]) for row in rows))
        for index, (_, title) in enumerate(_COLUMNS)
    ]

    def format_row(values: list[str]) -> str:
        cells = []
        for index, ((key, _), value) in enumerate(zip(_COLUMNS, values)):
            cells.append(
                value.rjust(widths[index])
                if key in _RIGHT_ALIGNED
                else value.ljust(widths[index])
            )
        return "  ".join(cells).rstrip()

    terminalreporter.ensure_newline()
    terminalreporter.write_line(
        format_row([title for _, title in _COLUMNS]), bold=True
    )
    for row in rows:
        terminalreporter.write_line(format_row(row))


class _BenchmarkSummaryPlugin:
    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config) -> None:
        _write_benchmark_summary(terminalreporter)


@pytest.fixture(scope="module", autouse=True)
def _register_benchmark_summary(request):
    """Register this file's terminal-summary hook without a conftest plugin."""
    _RESULTS.clear()
    plugin_name = "infllmv2-benchmark-summary"
    if not request.config.pluginmanager.hasplugin(plugin_name):
        request.config.pluginmanager.register(_BenchmarkSummaryPlugin(), plugin_name)
    yield


@dataclass(frozen=True)
class AttentionCase:
    name: str
    batch: int
    q_length: int
    kv_length: int
    head_dim: int
    dense: bool = False


@dataclass(frozen=True)
class DecodeCase:
    name: str
    batch: int
    kv_length: int
    head_dim: int


PREFILL_CASES = (
    AttentionCase("dense-1x1024-d64", 1, 1024, 1024, 64, True),
    AttentionCase("full-1x8192-d64", 1, 8192, 8192, 64),
    AttentionCase("full-1x8192-d128", 1, 8192, 8192, 128),
    AttentionCase("full-1x32768-d64", 1, 32768, 32768, 64),
    AttentionCase("full-1x32768-d128", 1, 32768, 32768, 128),
    AttentionCase("full-1x65536-d64", 1, 65536, 65536, 64),
    AttentionCase("full-1x65536-d128", 1, 65536, 65536, 128),
    AttentionCase("chunk-1x128x16384-d64", 1, 128, 16384, 64),
    AttentionCase("chunk-1x128x65536-d128", 1, 128, 65536, 128),
)

BACKWARD_CASES = (
    AttentionCase("backward-1x128x8192-d64", 1, 128, 8192, 64),
    AttentionCase("backward-1x128x8192-d128", 1, 128, 8192, 128),
)

DECODE_CASES = (
    DecodeCase("decode-1x8192-d64", 1, 8192, 64),
    DecodeCase("decode-1x8192-d128", 1, 8192, 128),
    DecodeCase("decode-1x65536-d64", 1, 65536, 64),
    DecodeCase("decode-1x65536-d128", 1, 65536, 128),
    DecodeCase("decode-8x32768-d64", 8, 32768, 64),
    DecodeCase("decode-8x32768-d128", 8, 32768, 128),
)


def _packed_data(
    batch: int,
    q_length: int,
    kv_length: int,
    head_dim: int,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(31)
    q = torch.randn(
        (batch * q_length, 32, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(requires_grad)
    k = torch.randn(
        (batch * kv_length, 2, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(requires_grad)
    v = torch.randn(
        (batch * kv_length, 2, head_dim),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(requires_grad)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * q_length
    cu_k = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * kv_length
    return q, k, v, cu_q, cu_k


def _measure(fn: Callable[[], object]) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=WARMUP_MS,
            rep=REP_MS,
            return_mode="median",
        )
    )


def _check_output(output: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    assert output.shape == expected_shape
    flat = output.reshape(-1)
    step = max(1, flat.numel() // 4096)
    assert torch.isfinite(flat[::step]).all()


def _report(
    record_property,
    *,
    operator: str,
    case: str,
    mode: str,
    batch: int,
    q_length: int,
    kv_length: int,
    head_dim: int,
    median_ms: float,
) -> None:
    fields = {
        "operator": operator,
        "case": case,
        "mode": mode,
        "batch": batch,
        "q_length": q_length,
        "kv_length": kv_length,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "warmup_ms": WARMUP_MS,
        "rep_ms": REP_MS,
        "median_ms": median_ms,
    }
    _RESULTS.append(fields)
    record_benchmark_result(
        record_property,
        op_name=operator,
        dtype="torch.bfloat16",
        mode=mode,
        result=[
            benchmark_metric(
                shape_detail={
                    "case": case,
                    "batch": batch,
                    "q_length": q_length,
                    "kv_length": kv_length,
                    "head_dim": head_dim,
                },
                latency=median_ms,
            )
        ],
        baseline="none",
        warmup_ms=WARMUP_MS,
        rep_ms=REP_MS,
    )


@ATTENTION
@pytest.mark.parametrize("case", PREFILL_CASES, ids=lambda case: case.name)
def test_infllmv2_attention_benchmark(
    case: AttentionCase, record_property
) -> None:
    q, k, v, cu_q, cu_k = _packed_data(
        case.batch, case.q_length, case.kv_length, case.head_dim
    )
    config = InfLLMV2Config(
        topk=64,
        dense_len=case.kv_length + 1 if case.dense else 8192,
    )

    def forward() -> torch.Tensor:
        return infllmv2_attention(
            q,
            k,
            v,
            cu_q,
            cu_k,
            case.q_length,
            case.kv_length,
            config=config,
        )

    output = forward()
    _check_output(output, q.shape)
    median_ms = _measure(forward)
    assert median_ms > 0
    _report(
        record_property,
        operator="infllmv2_attention",
        case=case.name,
        mode="prefill",
        batch=case.batch,
        q_length=case.q_length,
        kv_length=case.kv_length,
        head_dim=case.head_dim,
        median_ms=median_ms,
    )


@ATTENTION
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=lambda case: case.name)
def test_infllmv2_attention_backward_benchmark(
    case: AttentionCase, record_property
) -> None:
    q, k, v, cu_q, cu_k = _packed_data(
        case.batch,
        case.q_length,
        case.kv_length,
        case.head_dim,
        requires_grad=True,
    )
    grad = torch.randn_like(q)
    config = InfLLMV2Config(topk=64, dense_len=8192)

    def forward_backward():
        output = infllmv2_attention(
            q,
            k,
            v,
            cu_q,
            cu_k,
            case.q_length,
            case.kv_length,
            config=config,
        )
        return torch.autograd.grad(output, (q, k, v), grad)

    grads = forward_backward()
    assert all(torch.isfinite(tensor.reshape(-1)[:: max(1, tensor.numel() // 1024)]).all() for tensor in grads)
    median_ms = _measure(forward_backward)
    assert median_ms > 0
    _report(
        record_property,
        operator="infllmv2_attention",
        case=case.name,
        mode="forward_backward",
        batch=case.batch,
        q_length=case.q_length,
        kv_length=case.kv_length,
        head_dim=case.head_dim,
        median_ms=median_ms,
    )


@DECODE
@pytest.mark.parametrize("case", DECODE_CASES, ids=lambda case: case.name)
def test_infllmv2_decode_benchmark(
    case: DecodeCase, record_property
) -> None:
    q, k, v, _, cu_k = _packed_data(
        case.batch, 1, case.kv_length, case.head_dim
    )
    config = InfLLMV2Config(topk=64, dense_len=8192)

    def forward() -> torch.Tensor:
        return infllmv2_decode(
            q,
            k,
            v,
            cu_k,
            case.kv_length,
            config=config,
        )

    output = forward()
    _check_output(output, q.shape)
    median_ms = _measure(forward)
    assert median_ms > 0
    _report(
        record_property,
        operator="infllmv2_decode",
        case=case.name,
        mode="decode",
        batch=case.batch,
        q_length=1,
        kv_length=case.kv_length,
        head_dim=case.head_dim,
        median_ms=median_ms,
    )
