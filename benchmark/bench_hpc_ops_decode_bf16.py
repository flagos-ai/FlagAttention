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

"""Benchmark the final BF16 static and dynamic policies against official CUDA."""

from __future__ import annotations

import gc
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from flag_attn.hpc_ops_attention.decode.dynamic.bf16_dynamic import (  # noqa: E402
    DynamicBF16Inputs,
    attention_decode_bf16_dynamic,
    bf16_dynamic_workspace_is_reset,
    prepare_dynamic_bf16_workspace,
)
from flag_attn.hpc_ops_attention.decode import HAS_TLE  # noqa: E402
from flag_attn.hpc_ops_attention.decode.static.bf16_static import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    OFFICIAL_CASES,
    StaticBF16Inputs,
    attention_decode_bf16_tle,
    prepare_static_bf16_workspace,
)


# Fixed pytest benchmark matrix.  Use pytest node IDs or ``-k`` to select a
# subset without changing the benchmark configuration in the source file.
BENCH_MTP = (1, 2, 3)
BENCH_CASES = tuple(OFFICIAL_CASES)
BENCH_METHODS = ("static", "dynamic")
BENCH_LAYOUTS = ("NHD", "HND")
BENCH_NUM_HEAD_KV = 1
BENCH_NUM_HEAD_Q = 8
BENCH_WARMUP = 100
BENCH_ITERS = 300
BENCH_REPEAT = 5
BENCH_MIN_PROCESS_LEN = 64
BENCH_GRAPH = True
BENCH_CHECK = False
PERF_RESULTS = []


@dataclass
class Panel:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor


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


def _make_cuda_task_map(
    hpc,
    kv_lens: torch.Tensor,
    num_head_kv: int,
    num_seq_q: int,
    min_process_len: int,
) -> torch.Tensor:
    """Build the official CUDA dynamic schedule outside timed launches."""
    task_map = hpc.get_attention_decode_task_workspace(
        len(kv_lens), int(kv_lens.max().item()), num_head_kv,
        min_process_len=min_process_len,
    )
    hpc.assign_attention_decode_task(
        kv_lens,
        task_map,
        num_head_kv,
        num_seq_q,
        True,
        min_process_len=min_process_len,
    )
    return task_map


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def make_inputs(
    lengths, mtp: int, num_head_kv: int, num_head_q: int, layout: str,
) -> Panel:
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    kv_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    history = kv_lens - mtp
    if bool(torch.any(history < 0).item()):
        raise ValueError("every final KV length must be at least MTP")
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8
    q = torch.randn(
        (len(lengths) * mtp, num_head_q, HEAD_DIM),
        device="cuda", dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)
    k = torch.randn(
        (capacity, BLOCK_SIZE, num_head_kv, HEAD_DIM),
        device="cuda", dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)
    v = torch.randn(
        (capacity, BLOCK_SIZE, num_head_kv, HEAD_DIM),
        device="cuda", dtype=torch.bfloat16,
    )
    packed = torch.randperm(capacity, device="cuda")[:total_blocks].int()
    block_ids = torch.empty(
        (len(lengths), int(block_counts.max().item())),
        device="cuda", dtype=torch.int32,
    )
    cursor = 0
    for batch, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch, :count] = packed[cursor:cursor + count]
        cursor += count
    if layout == "HND":
        k, v = _as_hnd_view(k), _as_hnd_view(v)
    return Panel(q, k, v, block_ids, kv_lens)


def pytorch_reference(panel: Panel, mtp: int) -> torch.Tensor:
    batch = panel.kv_lens.numel()
    hq, hkv = panel.q.shape[1], panel.k.shape[2]
    heads_per_group = hq // hkv
    q = panel.q.reshape(batch, mtp, hq, HEAD_DIM)
    outputs = []
    for batch_idx in range(batch):
        length = int(panel.kv_lens[batch_idx])
        pages = triton.cdiv(length, BLOCK_SIZE)
        ids = panel.block_ids[batch_idx, :pages]
        k = panel.k[ids].reshape(-1, hkv, HEAD_DIM)[:length]
        v = panel.v[ids].reshape(-1, hkv, HEAD_DIM)[:length]
        k = k.transpose(0, 1).repeat_interleave(heads_per_group, 0).float()
        v = v.transpose(0, 1).repeat_interleave(heads_per_group, 0).float()
        scores = q[batch_idx].transpose(0, 1).float() @ k.transpose(-1, -2)
        scores /= math.sqrt(HEAD_DIM)
        history = length - mtp
        causal = torch.cat((
            torch.ones((mtp, history), dtype=torch.bool, device=panel.q.device),
            torch.tril(torch.ones(
                (mtp, mtp), dtype=torch.bool, device=panel.q.device,
            )),
        ), dim=-1)
        scores.masked_fill_(~causal[None], -float("inf"))
        outputs.append((F.softmax(scores, -1) @ v).transpose(0, 1))
    return torch.stack(outputs).reshape(batch * mtp, hq, HEAD_DIM).bfloat16()


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
    samples = sorted(start.elapsed_time(end) for start, end in events)
    return samples[len(samples) // 2]


def _measure(calls, warmup, iters, repeat, graph_mode):
    names = list(calls)
    samples = {name: [] for name in names}
    for repeat_idx in range(repeat):
        order = names[repeat_idx % len(names):] + names[:repeat_idx % len(names)]
        if repeat_idx & 1:
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
        size, dtype=torch.int8, device="cuda",
    ))
    return _load_hpc()


def _print_performance_table():
    if not PERF_RESULTS:
        return
    headers = (
        "Case",
        "Method",
        "MTP",
        "Layout",
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
                result["method"],
                str(result["mtp"]),
                result["layout"],
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

    print("\n\nHPC-Opc Decode Attention BF16 performance summary (CUDA Graph replay median)")
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
@pytest.mark.parametrize("case", BENCH_CASES)
@pytest.mark.parametrize("method", BENCH_METHODS)
@pytest.mark.parametrize("layout", BENCH_LAYOUTS)
def test_attention_decode_bf16_perf(hpc_baseline, mtp, case, method, layout):
    if not HAS_TLE and mtp != 1:
        pytest.skip("pure Triton fallback supports MTP=1 only")

    hpc = hpc_baseline
    panel = make_inputs(
        OFFICIAL_CASES[case], mtp, BENCH_NUM_HEAD_KV,
        BENCH_NUM_HEAD_Q, layout,
    )
    reference = pytorch_reference(panel, mtp) if BENCH_CHECK else None
    cuda_out = torch.empty_like(panel.q) if hpc is not None else None

    if method == "static":
        inputs = StaticBF16Inputs(
            panel.q, panel.k, panel.v, panel.block_ids, panel.kv_lens, layout,
        )
        workspace = prepare_static_bf16_workspace(inputs)
        cuda_task_map = None
        cuda_call = (
            None if hpc is None else lambda: hpc.attention_decode_bf16(
                panel.q, panel.k, panel.v, panel.block_ids, panel.kv_lens,
                mtp=mtp - 1, new_kv_included=True, splitk=True,
                output=cuda_out,
            )
        )
        tle_call = lambda: attention_decode_bf16_tle(inputs, workspace)
    else:
        inputs = DynamicBF16Inputs(
            panel.q, panel.k, panel.v, panel.block_ids, panel.kv_lens, layout,
        )
        workspace = prepare_dynamic_bf16_workspace(inputs)
        if HAS_TLE:
            assert getattr(workspace, "mtp", None) == mtp
            assert getattr(workspace, "route", "")
        cuda_task_map = (
            _make_cuda_task_map(
                hpc, panel.kv_lens, BENCH_NUM_HEAD_KV, mtp,
                BENCH_MIN_PROCESS_LEN,
            ) if hpc is not None else None
        )
        cuda_call = (
            None if hpc is None else lambda: hpc.attention_decode_bf16(
                panel.q, panel.k, panel.v, panel.block_ids, panel.kv_lens,
                mtp=mtp - 1, new_kv_included=True, splitk=True,
                task_map=cuda_task_map, output=cuda_out,
            )
        )
        tle_call = lambda: attention_decode_bf16_dynamic(inputs, workspace)

    if BENCH_CHECK:
        actual = tle_call().detach().clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, reference, atol=0.016, rtol=1e-5)
        assert torch.isfinite(actual).all()
        if cuda_call is not None:
            expected = cuda_call().detach().clone()
            torch.testing.assert_close(expected, reference, atol=0.016, rtol=1e-5)
            torch.testing.assert_close(actual, expected, atol=0.032, rtol=1e-5)
        if method == "dynamic":
            assert bf16_dynamic_workspace_is_reset(workspace)

    calls = {"triton+tle": tle_call}
    if cuda_call is not None:
        calls["cuda"] = cuda_call
    timing = _measure(
        calls, BENCH_WARMUP, BENCH_ITERS, BENCH_REPEAT, BENCH_GRAPH,
    )
    flagattention_ms = timing["triton+tle"][0]
    hpc_ms = timing["cuda"][0] if cuda_call is not None else None
    PERF_RESULTS.append({
        "case": case,
        "method": method,
        "mtp": mtp,
        "layout": layout,
        "flagattention_impl": "TLE" if HAS_TLE else "Triton",
        "flagattention_ms": flagattention_ms,
        "hpc_ms": hpc_ms,
    })

    del panel, reference, inputs, workspace, cuda_out, cuda_task_map
    gc.collect()
    torch.cuda.empty_cache()
