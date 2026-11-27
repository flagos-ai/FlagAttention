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

"""Hygon gfx936 dynamic decode A/B benchmark against the portable Triton baseline.

The upstream decode benchmarks time the selected Hy3 route against the
official CUDA implementation, which is unavailable on Hygon.  This benchmark
instead times the same route against the portable pure-Triton dynamic decode
fallback that runs on non-SM90 devices, using the upstream Graph-replay
measurement and the official workload matrix.  Correctness against the
PyTorch reference is covered by the ``hy3_attention`` test suite; here both
implementations must agree with each other before timing.
"""

from __future__ import annotations


import gc
import math
import sys
from pathlib import Path

import torch
import triton
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flag_attn import hy3_attention, hy3_attention_prepare  # noqa: E402
from flag_attn.hpc_ops_attention.decode import (  # noqa: E402
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
)
from flag_attn.hpc_ops_attention.decode.hygon import is_hygon_decode_device  # noqa: E402
from flag_attn.testing import backend as test_backend  # noqa: E402

import test_hpc_ops_decode_bf16_perf as bf16_perf  # noqa: E402
import test_hpc_ops_decode_fp8_perf as fp8_perf  # noqa: E402


# Both implementations must agree; the PyTorch reference gate lives in the
# correctness suite.  The value matches the cross-implementation tolerance
# used by the upstream decode benchmarks.
BENCH_CROSS_ATOL = 0.032
PERF_RESULTS = []


def _portable_baseline(inputs, quant: str, mtp: int):
    """Return the portable pure-Triton fallback call for one decode step."""

    if mtp == 1:
        workspace = prepare_pure_triton_mtp1_workspace(inputs, quant)
        return lambda: attention_decode_pure_triton_mtp1(inputs, workspace)
    workspace = prepare_pure_triton_mtp_workspace(inputs, quant)
    return lambda: attention_decode_pure_triton_mtp(inputs, workspace)


def _measure_ab(case, quant, mtp, layout, chunk_pages, calls, suite):
    for call in calls.values():
        output = call()
        test_backend.device_fn.synchronize()
        assert torch.isfinite(output).all()
    outputs = {name: call() for name, call in calls.items()}
    test_backend.device_fn.synchronize()
    torch.testing.assert_close(
        outputs["hygon"], outputs["portable"], atol=BENCH_CROSS_ATOL, rtol=1e-5,
    )
    timing = suite._measure(
        calls, suite.BENCH_WARMUP, suite.BENCH_ITERS, suite.BENCH_REPEAT, suite.BENCH_GRAPH,
    )
    PERF_RESULTS.append({
        "case": case,
        "quant": quant,
        "mtp": mtp,
        "layout": layout,
        "chunk_pages": chunk_pages,
        "portable_ms": timing["portable"][0],
        "hygon_ms": timing["hygon"][0],
        "speedup": timing["portable"][0] / timing["hygon"][0],
    })


def _print_performance_table():
    if not PERF_RESULTS:
        return
    headers = (
        "Case",
        "Quant",
        "MTP",
        "Layout",
        "Chunk",
        "Portable (ms)",
        "Hygon (ms)",
        "Speedup",
    )
    rows = [headers]
    for result in PERF_RESULTS:
        rows.append((
            result["case"],
            result["quant"],
            str(result["mtp"]),
            result["layout"],
            str(result["chunk_pages"]),
            f'{result["portable_ms"]:.4f}',
            f'{result["hygon_ms"]:.4f}',
            f'{result["speedup"]:.3f}x',
        ))

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row):
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    print("\n\nHygon dynamic decode A/B summary (CUDA Graph replay median)")
    print(f"Cross-implementation tolerance: {BENCH_CROSS_ATOL}")
    for row in rows:
        print(separator)
        print(format_row(row))
    print(separator)
    for quant in dict.fromkeys(result["quant"] for result in PERF_RESULTS):
        group = [result["speedup"] for result in PERF_RESULTS if result["quant"] == quant]
        geomean = math.exp(sum(math.log(value) for value in group) / len(group))
        print(f"{quant}: geomean {geomean:.3f}x over {len(group)} cases")


@pytest.fixture(scope="module", autouse=True)
def report_performance_results():
    triton.set_allocator(lambda size, _align, _stream: torch.empty(
        size, dtype=torch.int8, device=test_backend.device,
    ))
    yield
    _print_performance_table()


@pytest.mark.skipif(not test_backend.is_available(), reason="requires an available accelerator")
@pytest.mark.parametrize("mtp", bf16_perf.BENCH_MTP, ids=lambda value: f"mtp{value}")
@pytest.mark.parametrize("case", bf16_perf.BENCH_CASES)
@pytest.mark.hy3_attention
@pytest.mark.parametrize("layout", bf16_perf.BENCH_LAYOUTS)
def test_hygon_attention_decode_bf16_perf(mtp, case, layout):
    if not is_hygon_decode_device(torch.device(test_backend.device)):
        pytest.skip("Hygon gfx936 dynamic decode route required")
    if not test_backend.graph_available():
        pytest.skip("HIP graph timing required")

    panel = bf16_perf.make_inputs(
        bf16_perf.OFFICIAL_CASES[case], mtp, bf16_perf.BENCH_NUM_HEAD_KV,
        bf16_perf.BENCH_NUM_HEAD_Q, layout,
    )
    inputs = bf16_perf.DynamicBF16Inputs(
        panel.q, panel.k, panel.v, panel.block_ids, panel.kv_lens, layout,
    )
    workspace = hy3_attention_prepare(inputs, variant="bf16_dynamic")
    if getattr(workspace, "route", "") != "hygon_split_k":
        pytest.skip("Hygon dynamic decode route unavailable")

    portable_call = _portable_baseline(inputs, "bf16", mtp)
    hygon_call = lambda: hy3_attention(inputs, workspace, stage="decode", variant="bf16_dynamic")
    _measure_ab(
        case, "bf16", mtp, layout, workspace.chunk_pages,
        {"portable": portable_call, "hygon": hygon_call},
        bf16_perf,
    )

    del panel, inputs, workspace
    gc.collect()
    test_backend.device_fn.empty_cache()


@pytest.mark.skipif(not test_backend.is_available(), reason="requires an available accelerator")
@pytest.mark.parametrize("quant", fp8_perf.BENCH_QUANT_TYPES)
@pytest.mark.parametrize("mtp", fp8_perf.BENCH_MTP, ids=lambda value: f"mtp{value}")
@pytest.mark.parametrize("case", fp8_perf.BENCH_CASES)
@pytest.mark.hy3_attention
@pytest.mark.parametrize("layout", fp8_perf.BENCH_LAYOUTS)
def test_hygon_attention_decode_fp8_perf(quant, mtp, case, layout):
    if not is_hygon_decode_device(torch.device(test_backend.device)):
        pytest.skip("Hygon gfx936 dynamic decode route required")
    if not test_backend.graph_available():
        pytest.skip("HIP graph timing required")

    panel = fp8_perf.make_inputs(
        fp8_perf.OFFICIAL_CASES[case], mtp, fp8_perf.BENCH_NUM_HEAD_KV,
        fp8_perf.BENCH_NUM_HEAD_Q, layout, quant,
    )
    inputs = fp8_perf._inputs(panel)
    variant = "fp8_qk_dynamic" if fp8_perf.QUANT_TYPES[quant] == 0 else "fp8_kv_dynamic"
    workspace = hy3_attention_prepare(inputs, variant=variant)
    if getattr(workspace, "route", "") != "hygon_split_k":
        pytest.skip("Hygon dynamic decode route unavailable")

    portable_call = _portable_baseline(inputs, quant, mtp)
    hygon_call = lambda: hy3_attention(inputs, workspace, stage="decode", variant=variant)
    _measure_ab(
        case, quant, mtp, layout, workspace.chunk_pages,
        {"portable": portable_call, "hygon": hygon_call},
        fp8_perf,
    )

    del panel, inputs, workspace
    gc.collect()
    test_backend.device_fn.empty_cache()
