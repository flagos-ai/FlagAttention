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

"""Benchmark FP8 static and dynamic decode against the official CUDA op."""

from __future__ import annotations

import gc
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from flag_attn.hpc_ops_attention.decode.dynamic import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_dynamic as fp8_qk_dynamic,
    fp8_qpertoken_perhead_kvpertensor_dynamic as fp8_kv_dynamic,
)
from flag_attn.hpc_ops_attention.decode import HAS_TLE  # noqa: E402
from flag_attn.hpc_ops_attention.decode.static import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_static as fp8_qk_static,
    fp8_qpertoken_perhead_kvpertensor_static as fp8_kv_static,
)


BLOCK_SIZE = 64
HEAD_DIM = 128
SUPPORTED_MTP = (1, 2, 4)
QUANT_TYPES = {
    "qkpertoken_perhead_vperhead": 0,
    "qpertoken_perhead_kvpertensor": 1,
}
OFFICIAL_CASES = fp8_qk_static.OFFICIAL_CASES
FP8DecodeInputs = fp8_qk_static.FP8DecodeInputs
_IMPLEMENTATIONS = {
    ("static", "qkpertoken_perhead_vperhead"): fp8_qk_static,
    ("static", "qpertoken_perhead_kvpertensor"): fp8_kv_static,
    ("dynamic", "qkpertoken_perhead_vperhead"): fp8_qk_dynamic,
    ("dynamic", "qpertoken_perhead_kvpertensor"): fp8_kv_dynamic,
}

# Fixed pytest benchmark matrix.  Select a subset with a pytest node ID or
# ``-k`` instead of command-line benchmark arguments.
BENCH_MTP = SUPPORTED_MTP
BENCH_QUANT_TYPES = tuple(QUANT_TYPES)
BENCH_SCHEDULES = ("static", "dynamic")
BENCH_CASES = tuple(OFFICIAL_CASES)
BENCH_LAYOUTS = ("NHD", "HND")
BENCH_NUM_HEAD_KV = 1
BENCH_NUM_HEAD_Q = 8
BENCH_WARMUP = 50
BENCH_ITERS = 300
BENCH_REPEAT = 5
BENCH_MIN_PROCESS_LEN = 512
BENCH_GRAPH = True
BENCH_CHECK = False
PERF_RESULTS = []


def _implementation(schedule: str, quant_type: str):
    return _IMPLEMENTATIONS[(schedule, quant_type)]


@dataclass
class Panel:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


def _load_hpc():
    try:
        import hpc
        return hpc
    except (ImportError, OSError, AssertionError):
        # A source-tree ``hpc`` package may shadow the built extension and
        # assert because no local _C*.so lives beside its __init__.py.
        sys.modules.pop("hpc", None)
    build_roots = (REPO_ROOT / "build", REPO_ROOT.parent / "build")
    build_libs = sorted(
        path
        for build_root in build_roots
        for path in build_root.glob("lib.*")
        if list((path / "hpc").glob("_C*.so"))
    )
    if not build_libs:
        searched = ", ".join(str(path) for path in build_roots)
        print(f"hpc CUDA baseline unavailable ({searched}); omitting CUDA columns",
              file=sys.stderr)
        return None
    sys.path.insert(0, str(build_libs[0]))
    try:
        import hpc
    except (ImportError, OSError, AssertionError) as error:
        sys.modules.pop("hpc", None)
        print(f"hpc CUDA baseline unavailable ({error}); omitting CUDA columns",
              file=sys.stderr)
        return None
    return hpc


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _quantize_k_per_token(storage: torch.Tensor) -> torch.Tensor:
    blocks, _, heads, dim = storage.shape
    scale = (
        storage[:, :BLOCK_SIZE].float().abs().amax(-1).clamp_min(1e-6)
        / 448.0
    )
    result = torch.empty_like(storage, dtype=torch.float8_e4m3fn)
    result[:, :BLOCK_SIZE] = (
        storage[:, :BLOCK_SIZE] / scale[..., None]
    ).to(torch.float8_e4m3fn)
    packed = (
        scale.permute(0, 2, 1).contiguous().view(torch.float8_e4m3fn)
        .reshape(blocks, heads, -1, dim).permute(0, 2, 1, 3).contiguous()
    )
    result[:, BLOCK_SIZE:] = packed
    return result


def _quantize_v_per_head(
    storage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = storage.shape[2]
    scale = (
        storage[:, :BLOCK_SIZE].float().abs().permute(2, 0, 1, 3)
        .reshape(heads, -1).amax(-1).clamp_min(1e-6) / 448.0
    )
    quantized = (
        storage.float() / scale[None, None, :, None]
    ).to(torch.float8_e4m3fn)
    return quantized, scale


def make_inputs(
    lengths, mtp: int, hkv: int, hq: int, layout: str, quant_type: str,
) -> Panel:
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8

    # Match the official benchmark's seeded RNG order exactly:
    # Q -> K/V -> scales -> block IDs.
    q_bf16 = torch.randn(
        (len(lengths) * mtp, hq, HEAD_DIM),
        dtype=torch.bfloat16, device="cuda",
    ) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.float().abs().amax(-1).clamp_min(1e-6)
    q = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)

    if QUANT_TYPES[quant_type] == 0:
        raw_k = torch.randn(
            (capacity, BLOCK_SIZE + 2, hkv, HEAD_DIM),
            dtype=torch.bfloat16, device="cuda",
        )
        raw_v = torch.randn_like(raw_k)
        k_storage = _quantize_k_per_token(raw_k)
        v_storage, v_scale = _quantize_v_per_head(raw_v)
        if layout == "HND":
            k_storage = _as_hnd_view(k_storage)
            v_storage = _as_hnd_view(v_storage)
        k_cache = k_storage[:, :BLOCK_SIZE]
        v_cache = v_storage[:, :BLOCK_SIZE]
        k_scale = k_storage[:, BLOCK_SIZE:]
    else:
        k_cache = (
            torch.randn(
                (capacity, BLOCK_SIZE, hkv, HEAD_DIM),
                dtype=torch.bfloat16, device="cuda",
            ) / math.sqrt(HEAD_DIM)
        ).to(torch.float8_e4m3fn)
        v_cache = torch.randn(
            (capacity, BLOCK_SIZE, hkv, HEAD_DIM),
            dtype=torch.bfloat16, device="cuda",
        ).to(torch.float8_e4m3fn)
        if layout == "HND":
            k_cache = _as_hnd_view(k_cache)
            v_cache = _as_hnd_view(v_cache)
        k_scale = torch.rand(
            (1,), dtype=torch.float32, device="cuda"
        ).clamp_min(1e-6)
        v_scale = torch.rand(
            (1,), dtype=torch.float32, device="cuda"
        ).clamp_min(1e-6)

    packed_ids = torch.randperm(capacity, device="cuda")[:total_blocks].int()
    block_ids = torch.empty(
        (len(lengths), int(block_counts.max().item())),
        dtype=torch.int32, device="cuda",
    )
    offset = 0
    for batch, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch, :count] = packed_ids[offset:offset + count]
        offset += count

    return Panel(
        q, k_cache, v_cache, block_ids, kv_lens,
        q_scale, k_scale, v_scale,
    )


def _inputs(panel: Panel) -> FP8DecodeInputs:
    return FP8DecodeInputs(
        panel.q, panel.k_cache, panel.v_cache, panel.block_ids,
        panel.kv_lens, panel.q_scale, panel.k_scale, panel.v_scale,
    )


def _cuda_quant_type(hpc, quant_type: str):
    return (
        hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD
        if QUANT_TYPES[quant_type] == 0 else
        hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR
    )


def _cuda_task_map(hpc, panel: Panel, mtp: int, min_process_len: int):
    task_map = hpc.get_attention_decode_task_workspace(
        len(panel.kv_lens), int(panel.kv_lens.max()),
        panel.k_cache.shape[2], min_process_len=min_process_len,
    )
    hpc.assign_attention_decode_task(
        panel.kv_lens, task_map, panel.k_cache.shape[2], mtp, True,
        min_process_len=min_process_len,
    )
    return task_map


def _run_cuda(
    hpc, panel: Panel, output: torch.Tensor, mtp: int,
    quant_type: str, task_map,
):
    return hpc.attention_decode_fp8(
        panel.q, panel.k_cache, panel.v_cache, panel.block_ids,
        panel.kv_lens, panel.q_scale, panel.k_scale, panel.v_scale,
        mtp=mtp - 1, new_kv_included=True,
        quant_type=_cuda_quant_type(hpc, quant_type), splitk=True,
        task_map=task_map, output=output,
    )


def _bench_ms(call, warmup: int, iters: int, graph_mode: bool) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    if graph_mode:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        call = graph.replay
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in events)
    return values[len(values) // 2]


def _measure(calls, warmup, iters, repeat, graph_mode):
    names = list(calls)
    samples = {name: [] for name in names}
    for index in range(repeat):
        order = names[index % len(names):] + names[:index % len(names)]
        if index & 1:
            order.reverse()
        for name in order:
            samples[name].append(
                _bench_ms(calls[name], warmup, iters, graph_mode)
            )
    return {
        name: (statistics.median(values), min(values), max(values))
        for name, values in samples.items()
    }


@pytest.fixture(scope="module")
def hpc_baseline():
    triton.set_allocator(lambda size, _align, _stream: torch.empty(
        size, dtype=torch.int8, device="cuda"
    ))
    return _load_hpc()


def _print_performance_table():
    if not PERF_RESULTS:
        return
    headers = (
        "Case",
        "Schedule",
        "MTP",
        "Layout",
        "Quant Type",
        "FlagAttention Impl",
        "FlagAttention (ms)",
        "HPC CUDA (ms)",
        "HPC/FlagAttention",
    )
    rows = [headers]
    for result in PERF_RESULTS:
        rows.append(
            (
                result["case"],
                result["schedule"],
                str(result["mtp"]),
                result["layout"],
                result["quant_type"],
                result["flagattention_impl"],
                f'{result["flagattention_ms"]:.4f}',
                (
                    f'{result["hpc_ms"]:.4f}'
                    if result["hpc_ms"] is not None else "N/A"
                ),
                (
                    f'{result["hpc_ms"] / result["flagattention_ms"]:.3f}x'
                    if result["hpc_ms"] is not None else "N/A"
                ),
            )
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row):
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    print("\n\nHPC-Opc Decode Attention FP8 performance summary (CUDA Graph replay median)")
    print(
        f"Warmup replays: {BENCH_WARMUP}; Timed samples: {BENCH_ITERS}; "
        f"Repeats: {BENCH_REPEAT}"
    )
    print(separator)
    print(format_row(rows[0]))
    print(separator)
    for row in rows[1:]:
        print(format_row(row))
    print(separator)
    print(
        "HPC/FlagAttention > 1: FlagAttention is faster; "
        "HPC/FlagAttention < 1: HPC CUDA is faster."
    )


@pytest.fixture(scope="module", autouse=True)
def report_performance_results():
    yield
    _print_performance_table()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mtp", BENCH_MTP, ids=lambda value: f"mtp{value}")
@pytest.mark.parametrize("quant_type", BENCH_QUANT_TYPES)
@pytest.mark.parametrize("schedule", BENCH_SCHEDULES)
@pytest.mark.parametrize("case", BENCH_CASES)
@pytest.mark.parametrize("layout", BENCH_LAYOUTS)
def test_attention_decode_fp8_perf(
    hpc_baseline, mtp, quant_type, schedule, case, layout,
):
    if not HAS_TLE and mtp != 1:
        pytest.skip("pure Triton fallback supports MTP=1 only")

    hpc = hpc_baseline
    panel = make_inputs(
        OFFICIAL_CASES[case], mtp, BENCH_NUM_HEAD_KV,
        BENCH_NUM_HEAD_Q, layout, quant_type,
    )
    inputs = _inputs(panel)
    implementation = _implementation(schedule, quant_type)
    workspace = implementation.prepare_decode_workspace(inputs)
    cuda_output = (
        torch.empty_like(panel.q, dtype=torch.bfloat16) if hpc is not None else None
    )
    task_map = (
        _cuda_task_map(hpc, panel, mtp, BENCH_MIN_PROCESS_LEN)
        if hpc is not None and schedule == "dynamic" else None
    )
    tle_call = lambda: implementation.attention_decode_fp8(inputs, workspace)
    cuda_call = (
        None if hpc is None else
        lambda: _run_cuda(hpc, panel, cuda_output, mtp, quant_type, task_map)
    )

    if BENCH_CHECK:
        actual = tle_call().detach().clone()
        torch.cuda.synchronize()
        reset = implementation.workspace_is_reset(workspace)
        assert torch.isfinite(actual).all()
        assert reset, "decode workspace was not reset"
        if cuda_call is not None:
            expected = cuda_call().detach().clone()
            torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)

    calls = {"tle": tle_call}
    if cuda_call is not None:
        calls["cuda"] = cuda_call
    timing = _measure(
        calls, BENCH_WARMUP, BENCH_ITERS, BENCH_REPEAT, BENCH_GRAPH,
    )
    flagattention_ms = timing["tle"][0]
    hpc_ms = timing["cuda"][0] if cuda_call is not None else None
    PERF_RESULTS.append({
        "case": case,
        "schedule": schedule,
        "mtp": mtp,
        "layout": layout,
        "quant_type": quant_type,
        "flagattention_impl": "TLE" if HAS_TLE else "Triton",
        "flagattention_ms": flagattention_ms,
        "hpc_ms": hpc_ms,
    })

    del panel, inputs, workspace, cuda_output, task_map
    gc.collect()
    torch.cuda.empty_cache()
