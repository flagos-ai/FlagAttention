# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = _BASELINE_ROOT.parent
for _import_root in (_PROJECT_ROOT / "src", _PROJECT_ROOT, _BASELINE_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))


from functools import lru_cache
import math
import time
import warnings

import pytest
import torch
import triton

from flag_attn import runtime
from flag_attn.testing import backend as test_backend

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder


# Batch sizes covered by this benchmark.
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
STATE_DTYPES = (torch.float32, torch.bfloat16)
# Two head configurations with TP=4.
HEAD_CONFIGS = ((4, 4, 8), (4, 4, 16))
HEAD_DIM = 128
DTYPE = torch.bfloat16
SEED = 42
INITIAL_WARMUP = 10
WARMUP_MS, REP_MS = 50, 200
QUANTILES = (0.5, 0.2, 0.8)

def _assert_close(name, actual, expected):
    atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (2e-2, 1e-2)
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: {message}",
    )


def _measure(call):
    try:
        values = triton.testing.do_bench(
            call, quantiles=list(QUANTILES), warmup=WARMUP_MS, rep=REP_MS,
        )
        timing = "triton_do_bench_device_event"
    except (RuntimeError, NotImplementedError, AttributeError) as error:
        warnings.warn(f"do_bench unavailable; using 200-call wall timing: {error}", stacklevel=2)
        test_backend.device_fn.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            call()
        test_backend.device_fn.synchronize()
        values = [(time.perf_counter() - start) / 200 * 1000, None, None]
        timing = "synchronized_wall_mean_200_calls"
    assert math.isfinite(values[0]) and values[0] > 0, values
    return [value * 1000 if value is not None else None for value in values], timing

def _shape_matrix():
    return [
        (tp, batch, h, hv, state_dtype)
        for state_dtype in STATE_DTYPES
        for tp, h, hv in HEAD_CONFIGS
        for batch in BATCHES
    ]


@lru_cache(maxsize=1)
def _load_implementation():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    return fused_recurrent_gated_delta_rule_packed_decode


def _make_inputs(batch, h, hv, state_dtype):
    # Keep input generation deterministic across cases.
    torch.manual_seed(SEED)
    device = test_backend.device
    mixed = torch.randn(batch, (2 * h + hv) * HEAD_DIM, device=device, dtype=DTYPE)
    b = torch.randn(batch, hv, device=device, dtype=DTYPE)
    alog = torch.randn(hv, device=device, dtype=torch.float32)
    a = torch.randn(batch, hv, device=device, dtype=DTYPE)
    bias = torch.randn(hv, device=device, dtype=DTYPE)
    initial = torch.randn(
        max(512, batch + 16), hv, HEAD_DIM, HEAD_DIM, device=device, dtype=state_dtype,
    ) * 0.1
    return dict(
        mixed_qkv=mixed, a=a, b=b, A_log=alog, dt_bias=bias,
        initial_state=initial,
        ssm_state_indices=torch.arange(1, batch + 1, device=device, dtype=torch.int32),
    )

def _reference(data, h, hv):
    """Independent CPU FP32 packed decoding, including gates and one update."""
    mixed = data["mixed_qkv"].detach().float().cpu()
    q, k, v = torch.split(mixed, (h * HEAD_DIM, h * HEAD_DIM, hv * HEAD_DIM), dim=-1)
    batch = mixed.shape[0]
    q, k = (x.reshape(batch, h, HEAD_DIM).repeat_interleave(hv // h, dim=1) for x in (q, k))
    v = v.reshape(batch, hv, HEAD_DIM)
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    a, b, alog, bias = (data[name].detach().float().cpu() for name in ("a", "b", "A_log", "dt_bias"))
    g = -alog.exp() * torch.nn.functional.softplus(a + bias, beta=1.0, threshold=20.0)
    beta = b.sigmoid()  # Keep beta FP32 through the recurrence.
    indices = data["ssm_state_indices"].long()
    initial = data["initial_state"][indices].detach().float().cpu()
    state = initial * g.exp()[:, :, None, None]
    residual = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
    state = state + residual[:, :, :, None] * k[:, :, None, :]
    out = (state * (q * HEAD_DIM ** -0.5)[:, :, None, :]).sum(-1)
    return out.unsqueeze(1), state


@torch.inference_mode()
def run_benchmark(record_property, shape):
    tp, batch, h, hv, state_dtype = shape
    flag = _load_implementation()
    data = _make_inputs(batch, h, hv, state_dtype)
    initial = data["initial_state"]
    state = initial.clone()
    output = torch.empty(batch, 1, hv, HEAD_DIM, device=initial.device, dtype=DTYPE)

    def run_flag():
        return flag(
            data["mixed_qkv"],
            data["a"],
            data["b"],
            data["A_log"],
            data["dt_bias"],
            HEAD_DIM ** -0.5,
            state,
            output,
            data["ssm_state_indices"],
            True,
        )[0]

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_state = _reference(data, h, hv)
        indices = data["ssm_state_indices"].long()
        actual = run_flag()
        test_backend.device_fn.synchronize()
        _assert_close("output", actual, expected)
        _assert_close("state", state[indices], expected_state)
        assert torch.equal(state[0], initial[0]), "null state slot changed"
        assert actual.data_ptr() == output.data_ptr()
    finally:
        torch.set_num_threads(threads)
    del expected, expected_state

    state.copy_(initial)
    for _ in range(INITIAL_WARMUP):
        run_flag()
    test_backend.device_fn.synchronize()
    state.copy_(initial)
    values, method = _measure(run_flag)
    flag_us = values[0]

    recorder = BenchmarkRecorder(
        record_property, op_name="fused_recurrent_gated_delta_rule_packed_decode",
        dtype=str(DTYPE), phase="forward",
    )
    recorder.add(
        shape_detail=dict(
            TP=tp, B=batch, T=1, H=h, HV=hv, K=HEAD_DIM, V=HEAD_DIM,
            qkv_dtype=str(DTYPE), state_dtype=str(state_dtype),
            mixed_qkv_shape=list(data["mixed_qkv"].shape), state_shape=list(initial.shape),
            qk_normalization=True, scale=HEAD_DIM ** -0.5, seed=SEED,
        ),
        latency=flag_us, latency_unit="us",
        initial_warmup=INITIAL_WARMUP, warmup_ms=WARMUP_MS, rep_ms=REP_MS,
        quantiles=list(QUANTILES), quantiles_us=values,
        execution_kind="eager", timing_method=method,
        latency_scope="packed_qkv_gates_recurrence",
        cache_protocol="triton_driver_clear_cache" if method.startswith("triton") else "repeated_no_flush",
        correctness="cpu_fp32_single_step",
        shape_selection="two_head_configurations_tp4_fp32_and_bf16_states",
    )
    recorder.record()
    return (*shape, flag_us)



@pytest.fixture(scope="module")
def benchmark_table(request):
    prefix = request.node.nodeid + "::"

    class QuietPassedCases:
        @pytest.hookimpl(tryfirst=True)
        def pytest_report_teststatus(self, report, config):
            if report.nodeid.startswith(prefix) and report.when == "call" and report.passed:
                return "passed", "", ""

    plugin = QuietPassedCases()
    manager = request.config.pluginmanager
    manager.register(plugin)
    reporter = manager.get_plugin("terminalreporter")

    def write(line):
        with manager.get_plugin("capturemanager").global_and_fixture_disabled():
            if reporter is not None:
                reporter.write(line + "\n", flush=True)
            else:
                print(line, flush=True)

    emitted_dtype = None

    def emit(row):
        tp, batch, h, hv, state_dtype, flag_us = row
        nonlocal emitted_dtype
        if state_dtype != emitted_dtype:
            emitted_dtype = state_dtype
            write("")
            write("State: BF16" if state_dtype == torch.bfloat16 else "State: FP32")
            write(
                f"{'TP':>2} {'B':>3} {'H':>2} {'HV':>2} {'K':>3} {'V':>3} "
                f"{'QKV':>5} {'State':>5} {'flag_attn(us)':>14}"
            )
            write("-" * 66)
        state = "BF16" if state_dtype == torch.bfloat16 else "FP32"
        write(
            f"{tp:>2} {batch:>3} {h:>2} {hv:>2} {HEAD_DIM:>3} {HEAD_DIM:>3} "
            f"{'BF16':>5} {state:>5} {flag_us:>14.3f}"
        )

    try:
        yield emit
    finally:
        manager.unregister(plugin)


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shape",
    _shape_matrix(),
    ids=[
        f"TP{tp}-B{batch}-H{h}-HV{hv}-state-{'bf16' if dtype == torch.bfloat16 else 'fp32'}"
        for tp, batch, h, hv, dtype in _shape_matrix()
    ],
)
def test_fused_recurrent_packed_decode_benchmark(record_property, shape, benchmark_table):
    benchmark_table(run_benchmark(record_property, shape))
