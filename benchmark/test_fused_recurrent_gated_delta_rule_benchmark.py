# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import sys

_BASELINE_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _BASELINE_ROOT.parent
for _root in (_PROJECT_ROOT / "src", _PROJECT_ROOT, _BASELINE_ROOT, Path(__file__).parent):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import pytest
import torch
import triton

from flag_attn import runtime
from flag_attn.testing import backend as test_backend
def build(case, dtype, device, normalize=True):
    name, lengths, H, HV, K, V, mode, inplace, strided = case
    torch.manual_seed(2701)
    N = len(lengths)
    plain = mode in ("plain", "none")
    B, T = (N, lengths[0]) if plain else (1, sum(lengths))

    def rand(shape, dt=dtype):
        if strided:
            x = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dt)[..., ::2]
        else:
            x = torch.randn(shape, dtype=dt)
        # Preserve the tested strides after device transfer.
        out = torch.empty_strided(x.shape, x.stride(), dtype=dt, device=device)
        out.copy_(x)
        return out

    q, k, v = rand((B, T, H, K)), rand((B, T, H, K)), rand((B, T, HV, V))
    k.mul_(0.05)
    g = rand((B, T, HV), torch.float32)
    g.copy_(-g.abs() * 0.1)
    beta = rand((B, T, HV, V) if name == "vector_beta" else (B, T, HV))
    beta.copy_(beta.sigmoid())
    width = max(max(lengths), 4)
    pool = N * width + 3
    state_dtype = dtype if name == "low_state" else torch.float32
    state = rand((pool, HV, V, K), state_dtype)
    state.mul_(0.05)
    indices = None
    if mode in ("index", "table", "spec"):
        if mode == "index":
            indices = torch.arange(1, N + 1, dtype=torch.int32)
            if name == "padding":
                indices[:2] = torch.tensor([0, -1], dtype=torch.int32)
        else:
            indices = torch.arange(1, N * width + 1, dtype=torch.int64).view(N, width)
        if strided:
            buf = torch.empty(
                (*indices.shape[:-1], indices.shape[-1] * 2), dtype=indices.dtype
            )
            buf[..., ::2] = indices
            indices = buf[..., ::2]
        d = torch.empty_strided(
            indices.shape, indices.stride(), dtype=indices.dtype, device=device
        )
        d.copy_(indices)
        indices = d
    cu = (
        None
        if plain
        else torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
    )
    accepted = (
        torch.tensor([3] * N, dtype=torch.int32, device=device)
        if mode == "spec"
        else None
    )
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=None if mode == "none" else state,
        inplace_final_state=inplace,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=normalize,
    )


def reference(data):
    q, k, v = (data[x].detach().cpu().double() for x in ("q", "k", "v"))
    g = data["g"].cpu().double() if data["g"] is not None else None
    beta = data["beta"].cpu().double() if data["beta"] is not None else None
    state = data["initial_state"]
    state = state.cpu().clone() if state is not None else None
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    indices = data["ssm_state_indices"]
    indices = indices.cpu() if indices is not None else None
    cu = data["cu_seqlens"]
    bounds = (
        list(zip(cu.cpu().tolist()[:-1], cu.cpu().tolist()[1:]))
        if cu is not None
        else [(0, T)] * B
    )
    accepted = data["num_accepted_tokens"]
    output = torch.zeros(v.shape, dtype=torch.float64)
    final = (
        state.clone()
        if data["inplace_final_state"]
        else torch.zeros(
            (B * T, HV, V, K), dtype=state.dtype if state is not None else torch.float32
        )
    )
    for n, (start, end) in enumerate(bounds):
        batch = 0 if cu is not None else n
        slot = n
        if indices is not None:
            col = int(accepted[n].item()) - 1 if accepted is not None else 0
            slot = int(indices[n, col] if indices.ndim == 2 else indices[n])
            if slot <= 0 or slot >= len(state):
                continue
        acc = (
            state[slot].double().clone()
            if state is not None
            else torch.zeros((HV, V, K), dtype=torch.float64)
        )
        for t in range(start, end):
            qv, kv = q[batch, t], k[batch, t]
            if data["use_qk_l2norm_in_kernel"]:
                qv = qv / (qv.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                kv = kv / (kv.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            qv = qv.repeat_interleave(HV // H, 0) * data["scale"]
            kv = kv.repeat_interleave(HV // H, 0)
            if g is not None:
                acc = acc * g[batch, t].exp()[:, None, None]
            residual = v[batch, t] - (acc * kv[:, None, :]).sum(-1)
            if beta is not None:
                bv = beta[batch, t]
                residual = residual * (bv[:, None] if bv.ndim == 1 else bv)
            acc = acc + residual[:, :, None] * kv[:, None, :]
            output[batch, t] = (acc * qv[:, None, :]).sum(-1)
            if data["inplace_final_state"]:
                if indices is not None and indices.ndim == 2:
                    dest = int(indices[n, t - start])
                    if 0 < dest < len(final):
                        final[dest] = acc
                elif t == end - 1:
                    final[slot] = acc
            else:
                final[batch * T + t] = acc
    return output.to(data["v"].dtype), final


def assert_result(result, expected, dtype):
    tol = {
        torch.float16: (3e-3, 3e-3),
        torch.bfloat16: (2e-2, 2e-2),
        torch.float32: (3e-5, 3e-5),
    }
    for got, ref in zip(result, expected):
        rtol, atol = tol[got.dtype]
        torch.testing.assert_close(got.detach().cpu(), ref, rtol=rtol, atol=atol)

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

pytestmark = pytest.mark.benchmark
SOURCE = "fused-recurrent-gdn-decode-and-mtp"
WARMUP_CALLS, SAMPLE_COUNT = 5, 25
GRAPH_CALLS = 64
# (id, request count, tokens per request, key heads, value heads).
PERF_CASES = [
    ("decode_b1_h16_hv32", 1, 1, 16, 32),
    ("decode_b2_h16_hv32", 2, 1, 16, 32),
    ("decode_b4_h16_hv32", 4, 1, 16, 32),
    ("decode_b64_h16_hv32", 64, 1, 16, 32),
    ("decode_b64_h16_hv64", 64, 1, 16, 64),
    ("decode_b2_h4_hv8", 2, 1, 4, 8),
    ("decode_b4_h4_hv8", 4, 1, 4, 8),
    ("decode_b32_h4_hv8", 32, 1, 4, 8),
    ("decode_b128_h4_hv8", 128, 1, 4, 8),
    ("mtp_b2_t2", 2, 2, 4, 8),
    ("mtp_b2_t4", 2, 4, 4, 8),
    ("mtp_b2_t8", 2, 8, 4, 8),
    ("mtp_b1_t8", 1, 8, 4, 8),
    ("mtp_b4_t8", 4, 8, 4, 8),
    ("mtp_b8_t8", 8, 8, 4, 8),
]
DTYPES = (torch.float16, torch.bfloat16)


@lru_cache(maxsize=1)
def _implementation():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )
    return fused_recurrent_gated_delta_rule_fwd


def _clone_state(state):
    result = torch.empty_strided(state.shape, state.stride(), dtype=state.dtype, device=state.device)
    return result.copy_(state)


def _check(call, data, pristine, expected, dtype, written):
    state = data["initial_state"]
    state.copy_(pristine)
    out, final = call()
    test_backend.device_fn.synchronize()
    assert out.dtype == dtype and final.dtype == pristine.dtype
    assert final.data_ptr() == state.data_ptr(), "in-place state alias changed"
    assert_result((out, final), expected, dtype)
    untouched = torch.ones(len(pristine), dtype=torch.bool, device=state.device)
    untouched[written] = False
    assert torch.equal(final[untouched], pristine[untouched]), "untouched state slots changed"


def _capture(call, state, pristine):
    device = test_backend.device_fn
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    with device.stream(stream):
        for _ in range(WARMUP_CALLS):
            state.copy_(pristine)
            call()
    device.current_stream().wait_stream(stream)
    device.synchronize()
    state.copy_(pristine)
    device.synchronize()
    graph = device.CUDAGraph()
    with device.graph(graph, stream=stream):
        for _ in range(GRAPH_CALLS):
            result = call()
    device.synchronize()

    def replay():
        graph.replay()
        return result

    return replay


def _replay_reference(data):
    # Compact the CPU state pool to slots used by this workload. The large
    # unused pool remains covered by the exact untouched-slot check.
    cpu = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
           for key, value in data.items()}
    slots = cpu["ssm_state_indices"].long().unique()
    remap = torch.zeros(len(cpu["initial_state"]), dtype=torch.int64)
    remap[slots] = torch.arange(1, len(slots) + 1)
    cpu["ssm_state_indices"] = remap[cpu["ssm_state_indices"].long()]
    original = cpu["initial_state"]
    cpu["initial_state"] = torch.cat((torch.zeros_like(original[:1]), original[slots]))
    for _ in range(GRAPH_CALLS):
        expected = reference(cpu)
        cpu["initial_state"] = expected[1]
    final = original.clone()
    final[slots] = expected[1][1:]
    return expected[0], final


def _measure(replay, state, pristine):
    device = test_backend.device_fn
    for _ in range(WARMUP_CALLS):
        state.copy_(pristine)
        replay()
    device.synchronize()
    events = [(device.Event(enable_timing=True), device.Event(enable_timing=True))
              for _ in range(SAMPLE_COUNT)]
    samples = []
    for sample in range(SAMPLE_COUNT):
        start, end = events[sample]
        state.copy_(pristine)
        device.synchronize()
        start.record()
        replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / GRAPH_CALLS)
    assert len(samples) == SAMPLE_COUNT
    assert all(math.isfinite(value) and value > 0 for value in samples), samples
    return samples


def _save(row, record_property):
    raw_path = os.environ.get("GDN_RESULTS")
    if raw_path:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
    recorder = BenchmarkRecorder(record_property, op_name="fused_recurrent_gated_delta_rule_fwd",
                                 dtype=row["dtype"], phase="forward")
    metadata = {k: v for k, v in row.items() if k not in ("flag_ms", "config")}
    recorder.add(shape_detail=row["config"], latency=row.get("flag_ms"),
                 latency_unit="ms", **metadata)
    recorder.record()


@pytest.fixture(scope="module")
def benchmark_table(request):
    _implementation()
    manager = request.config.pluginmanager
    reporter = manager.get_plugin("terminalreporter")
    prefix = request.node.nodeid + "::"
    class QuietCases:
        @pytest.hookimpl(tryfirst=True)
        def pytest_report_teststatus(self, report, config):
            if report.nodeid.startswith(prefix):
                if report.when == "call" and report.passed:
                    return "passed", "", ""
                if report.skipped:
                    return "skipped", "", ""
    quiet = QuietCases()
    manager.register(quiet)
    def write(text):
        with manager.get_plugin("capturemanager").global_and_fixture_disabled():
            if reporter is None:
                print(text, flush=True)
            else:
                reporter.write(text + "\n", flush=True)
    write(f"\nT-Head {torch.cuda.get_device_name()}; torch={torch.__version__}; triton={triton.__version__}")
    write(f"Precomputed g/beta and int32 state indices; FP32 state; Graph: {GRAPH_CALLS} calls/replay.")
    write("5 warmups / 25 samples; reset outside timing.")
    write("Median event time per call; capture/preparation excluded.")
    write(f"{'Case':>22} {'QKV':>5} {'B':>4} {'T/req':>5} {'H':>3} {'HV':>3} {'K':>3} {'V':>3} "
          f"{'Flag(us)':>11} {'Status':>12}")
    write("-" * 95)
    def emit(row):
        c = row["config"]
        def us(key):
            value = row.get(key)
            return "N/A" if value is None else f"{value * 1000:.3f}"
        dtype = row["dtype"].replace("torch.", "").replace("bfloat16", "bf16").replace("float16", "fp16")
        write(f"{row['case']:>22} {dtype:>5} {len(c['lengths']):>4} {max(c['lengths']):>5} "
              f"{c['H']:>3} {c['HV']:>3} {c['K']:>3} {c['V']:>3} "
              f"{us('flag_ms'):>11} {row['status']:>12}")
        if row.get("reason"):
            write("    " + row["reason"])
    try:
        yield emit
    finally:
        manager.unregister(quiet)


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("case", PERF_CASES, ids=[case[0] for case in PERF_CASES])
@pytest.mark.parametrize("dtype", DTYPES, ids=("fp16", "bf16"))
@torch.inference_mode()
def test_fused_recurrent_gated_delta_rule_benchmark(case, dtype, record_property, benchmark_table):
    flag = _implementation()
    name, batch, tokens, heads, value_heads = case
    lengths = [tokens] * batch
    workload = (name, lengths, heads, value_heads, 128, 128,
                "spec" if tokens > 1 else "index", True, False)
    data = build(workload, dtype, test_backend.device)
    # Use the index dtype expected by the T-Head kernel.
    data["ssm_state_indices"] = data["ssm_state_indices"].to(torch.int32)
    if tokens > 1:
        # Legal accepted positions vary across requests and never exceed T.
        data["num_accepted_tokens"] = (
            torch.arange(batch, dtype=torch.int32, device=test_backend.device) % tokens + 1
        )
    source_file = Path(inspect.getfile(flag))
    row = dict(
        case=name, mode="mtp" if tokens > 1 else "decode", dtype=str(dtype), backend="thead", graph=True,
        unit="ms", source=SOURCE,
        gate_scope="precomputed_g_beta_recurrence_with_optional_qk_normalization",
        flag_source=str(source_file), flag_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        sample_count=SAMPLE_COUNT, warmup_calls=WARMUP_CALLS,
        protocol="graph-64-sequential-calls-reset-outside-5warmup-25-samples-median-per-call",
        graph_calls=GRAPH_CALLS, state_reset="before_each_replay", graph_state_policy="sequential_updates",
        flag_accuracy="NOT_RUN", status="NOT_RUN",
        config=dict(lengths=lengths, H=heads, HV=value_heads, K=128, V=128,
                    seed=2701, scale=data["scale"], qk_normalization=True, state_dtype="torch.float32",
                    tensors={k: dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype))
                             for k, v in data.items() if isinstance(v, torch.Tensor)}),
    )
    pristine = _clone_state(data["initial_state"])
    provider = dict(data, initial_state=_clone_state(pristine))
    def run_flag():
        return flag(**provider)
    row["state_index_dtype"] = "torch.int32"
    indices = data["ssm_state_indices"]
    written = (indices[:, :tokens] if indices.ndim == 2 else indices).long().flatten()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected = reference(data)
        try:
            _check(run_flag, provider, pristine, expected, dtype, written)
            row["flag_accuracy"] = "PASS"
        except Exception as exc:
            row.update(status="FAIL", reason=f"FlagAttention correctness failed: {exc}")
            row["flag_accuracy"] = "FAIL"
            _save(row, record_property)
            benchmark_table(row)
            raise
        del expected
        expected = _replay_reference(data)
        try:
            replay = _capture(run_flag, provider["initial_state"], pristine)
            _check(replay, provider, pristine, expected, dtype, written)
            row["flag_graph_accuracy"] = "PASS"
        except Exception as exc:
            row.update(status="FAIL", reason=f"FlagAttention graph correctness failed: {exc}")
            row["flag_graph_accuracy"] = "FAIL"
            _save(row, record_property)
            benchmark_table(row)
            raise
    finally:
        torch.set_num_threads(threads)
    del expected
    values = _measure(replay, provider["initial_state"], pristine)
    row["flag_samples_ms"] = values
    row["flag_ms"] = statistics.median(values)
    row["flag_p20_ms"] = torch.quantile(torch.tensor(values), 0.2).item()
    row["flag_p80_ms"] = torch.quantile(torch.tensor(values), 0.8).item()
    row["status"] = "PASS"
    _save(row, record_property)
    benchmark_table(row)
