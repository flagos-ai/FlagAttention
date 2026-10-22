# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head packed GDN versus bundled official PLA, with an independent CPU reference.

Four representative head configurations are each measured with both BF16 and
FP32 state, covering identical batches. QKV remains BF16. Latency is in us,
using seven rounds of 40 Event samples over 32-call Graph replays.
"""

from functools import lru_cache
from importlib import import_module
import math
import statistics

import pytest
import torch

from flag_attn import runtime
from flag_attn.testing import backend as test_backend

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

pytestmark = pytest.mark.fused_recurrent_gated_delta_rule_packed_decode

BATCHES = (1, 2, 4, 8, 16, 32, 64)
STATE_DTYPES = (torch.bfloat16, torch.float32)
# Head configurations come from t-head/pla caselist/decode and cover HV/H=2,3,4.
HEAD_CONFIGS = (
    (2, 8, 16),
    (4, 4, 8),
    (2, 8, 24),
    (2, 8, 32),
)
HEAD_DIM = 128
DTYPE = torch.bfloat16
SEED = 1234
ASSERT_RATIO = 0.01
ROUNDS, REPEAT_ITERS, DRY_RUN_ITERS = 7, 40, 5
INITIAL_WARMUP, ROUND_WARMUP = 10, 30
CALLS_PER_GRAPH = 32
PLA_IMPORT_ERROR = None


def _assert_close(name, actual, expected):
    # Select tolerances before converting: output and state may differ in dtype.
    atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (2e-2, 1e-2)
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert torch.isfinite(actual).all(), f"{name}: nonfinite actual"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol, msg=lambda message: f"{name}: elementwise check failed\n{message}",
    )
    max_abs = (actual - expected).abs().max().item()
    ratio = (actual - expected).square().mean().sqrt().item() / (expected.square().mean().sqrt().item() + 1e-8)
    assert (
        max_abs <= 1e-6 or ratio < ASSERT_RATIO
    ), f"{name}: relative_rms={ratio:.7g}, max_abs={max_abs:.7g}, limit={ASSERT_RATIO}"
    return ratio


def _check_provider(name, call, state, initial, expected, expected_state):
    out = call()
    test_backend.device_fn.synchronize()
    output_error = state_error = 0.0
    # Every row has one token and a distinct valid slot, numbered 1..B.
    for row in range(out.shape[0]):
        output_error = max(output_error, _assert_close(f"{name} output[{row}]", out[row], expected[row]))
        slot = row + 1
        state_error = max(state_error, _assert_close(f"{name} state[{slot}]", state[slot], expected_state[slot]),)
    assert torch.equal(state[0], initial[0]), f"{name}: null state slot changed"
    return output_error, state_error


def _capture(call, state, initial, name):
    device = test_backend.device_fn
    state.copy_(initial)
    for _ in range(INITIAL_WARMUP):
        call()
    stream = device.Stream()
    stream.wait_stream(device.current_stream())
    graph = device.CUDAGraph()
    with device.graph(graph, stream=stream):
        for _ in range(CALLS_PER_GRAPH):
            out = call()
    device.current_stream().wait_stream(stream)
    device.synchronize()

    # Graph correctness is checked within each provider. Cross-provider
    # 32-step drift is a separate stability test, not a single-step gate.
    state.copy_(initial)
    graph.replay()
    saved_out, saved_state = out.clone(), state.clone()
    state.copy_(initial)
    for _ in range(CALLS_PER_GRAPH):
        eager_out = call()
    device.synchronize()
    assert torch.equal(saved_out, eager_out), f"{name}: Graph/eager output mismatch"
    assert torch.equal(saved_state, state), f"{name}: Graph/eager state mismatch"
    assert torch.equal(state[0], initial[0]), f"{name}: null state slot changed"
    return graph, out


def _measure_pair(calls, states, initial):
    device = test_backend.device_fn
    if not test_backend.graph_available() or not hasattr(device, "Event"):
        pytest.fail("this benchmark requires T-Head Graph and Event support")
    names = ("FlagAttention", "PLA packed")
    captured = [_capture(call, state, initial, name) for call, state, name in zip(calls, states, names)]
    events = [
        [(device.Event(enable_timing=True), device.Event(enable_timing=True)) for _ in range(REPEAT_ITERS)]
        for _ in calls
    ]
    for pairs in events:
        for start, end in pairs:
            start.record()
            end.record()
    device.synchronize()
    samples = [[] for _ in calls]
    for round_id in range(ROUNDS):
        # Alternate which provider runs first to reduce ordering bias.
        for provider in range(len(calls)) if round_id % 2 == 0 else reversed(range(len(calls))):
            state = states[provider]
            state.copy_(initial)
            for _ in range(ROUND_WARMUP):
                calls[provider]()
            graph, out = captured[provider]
            for _ in range(DRY_RUN_ITERS):
                graph.replay()
            device.synchronize()
            for start, end in events[provider]:
                start.record()
                graph.replay()
                end.record()
            device.synchronize()
            values = [start.elapsed_time(end) * 1000.0 / CALLS_PER_GRAPH for start, end in events[provider]]
            assert all(math.isfinite(value) and value > 0 for value in values)
            assert torch.isfinite(out).all(), f"{names[provider]}: nonfinite output"
            assert torch.isfinite(state).all(), f"{names[provider]}: nonfinite state"
            assert torch.equal(state[0], initial[0]), f"{names[provider]}: null slot changed"
            samples[provider].append(values)
    return samples


def _shape_matrix():
    return [
        (tp, batch, h, hv, state_dtype)
        for state_dtype in STATE_DTYPES
        for tp, h, hv in HEAD_CONFIGS
        for batch in BATCHES
    ]


@lru_cache(maxsize=1)
def _load_implementations():
    global PLA_IMPORT_ERROR
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    try:
        import_module("flag_attn.runtime.backend._thead.FLA.gated_delta_rule." "pla_gdn.official.pla.decode._C")
        from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.pla_gdn import packed_decode
    except (ImportError, OSError) as exc:
        PLA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return fused_recurrent_gated_delta_rule_packed_decode, None
    PLA_IMPORT_ERROR = None
    return fused_recurrent_gated_delta_rule_packed_decode, packed_decode


def _make_inputs(batch, h, hv, state_dtype):
    # Per-case generators make pytest filtering reproducible. The two state
    # dtypes share identical QKV/gates and the same FP32 source initial state.
    device = test_backend.device
    generator = torch.Generator(device=device).manual_seed(SEED)
    mixed = torch.randn(batch, (2 * h + hv) * HEAD_DIM, device=device, dtype=DTYPE, generator=generator) * 0.1
    a = torch.randn(batch, hv, device=device, dtype=DTYPE, generator=generator)
    b = torch.randn(batch, hv, device=device, dtype=DTYPE, generator=generator)
    initial = (
        torch.randn(batch + 1, hv, HEAD_DIM, HEAD_DIM, device=device, dtype=torch.float32, generator=generator) * 0.1
    )
    return dict(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=torch.zeros(hv, device=device, dtype=torch.float32),
        dt_bias=torch.zeros(hv, device=device, dtype=torch.float32),
        initial_state=initial.to(state_dtype),
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
    initial = data["initial_state"].detach().float().cpu()
    indices = data["ssm_state_indices"].detach().long().cpu()
    state = initial[indices] * g.exp()[:, :, None, None]
    residual = (v - (state * k[:, :, None, :]).sum(-1)) * beta[:, :, None]
    state = state + residual[:, :, :, None] * k[:, :, None, :]
    out = (state * (q * HEAD_DIM ** -0.5)[:, :, None, :]).sum(-1)
    final = initial.clone()
    final[indices] = state
    return out.unsqueeze(1), final


@torch.inference_mode()
def run_benchmark(record_property, shape):
    tp, batch, h, hv, state_dtype = shape
    flag, pla = _load_implementations()
    data = _make_inputs(batch, h, hv, state_dtype)
    initial = data["initial_state"]
    states = (initial.clone(), initial.clone())
    outputs = tuple(torch.empty(batch, 1, hv, HEAD_DIM, device=initial.device, dtype=DTYPE) for _ in states)
    # PLA reads dt_bias at activation dtype. All entries are zero, so this
    # conversion preserves every input value and is outside the timed call.
    pla_bias = data["dt_bias"].to(DTYPE)
    cu = torch.arange(batch + 1, device=initial.device, dtype=torch.int32)

    def run_flag():
        return flag(
            data["mixed_qkv"],
            data["a"],
            data["b"],
            data["A_log"],
            data["dt_bias"],
            HEAD_DIM ** -0.5,
            states[0],
            outputs[0],
            data["ssm_state_indices"],
            True,
        )[0]

    def run_pla():
        # is_kda=False, is_sglang=False: slot zero is invalid on both sides.
        return pla(
            data["mixed_qkv"],
            data["a"],
            data["b"],
            data["A_log"],
            pla_bias,
            1.0,
            20.0,
            HEAD_DIM ** -0.5,
            states[1],
            data["ssm_state_indices"],
            outputs[1],
            True,
            cu,
            False,
            False,
        )

    calls = (run_flag, run_pla) if pla is not None else (run_flag,)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_state = _reference(data, h, hv)
        errors = [
            _check_provider(name, call, state, initial, expected, expected_state)
            for name, call, state in zip(("FlagAttention", "PLA packed"), calls, states)
        ]
        # Both entries must write the caller-provided output buffer.
        assert run_flag().data_ptr() == outputs[0].data_ptr()
        if pla is not None:
            assert run_pla().data_ptr() == outputs[1].data_ptr()
    finally:
        torch.set_num_threads(threads)
    del expected, expected_state
    samples = _measure_pair(calls, states, initial)
    medians = [[statistics.median(row) for row in provider] for provider in samples]
    flag_us = statistics.median(medians[0])
    pla_us = statistics.median(medians[1]) if pla is not None else None
    recorder = BenchmarkRecorder(
        record_property,
        op_name="fused_recurrent_gated_delta_rule_packed_decode",
        dtype=str(DTYPE),
        phase="forward",
        baseline="pla_official_packed" if pla is not None else None,
    )
    recorder.add(
        shape_detail=dict(
            TP=tp,
            B=batch,
            T=1,
            H=h,
            HV=hv,
            K=HEAD_DIM,
            V=HEAD_DIM,
            qkv_dtype=str(DTYPE),
            state_dtype=str(state_dtype),
            mixed_qkv_shape=list(data["mixed_qkv"].shape),
            state_shape=list(initial.shape),
            qk_normalization=True,
            scale=HEAD_DIM ** -0.5,
            seed=SEED,
        ),
        latency=flag_us,
        latency_base=pla_us,
        latency_unit="us",
        sample_count=ROUNDS * REPEAT_ITERS,
        calls_per_graph=CALLS_PER_GRAPH,
        rounds=ROUNDS,
        samples_per_round=REPEAT_ITERS,
        initial_warmup=INITIAL_WARMUP,
        round_warmup=ROUND_WARMUP,
        dry_run_iters=DRY_RUN_ITERS,
        execution_kind="multi_call_graph_replay",
        timing_method="device_event",
        latency_scope="packed_qkv_gates_recurrence_amortized_graph_update",
        cache_protocol="warm_repeated_no_flush",
        samples_us=samples[0],
        pla_samples_us=samples[1] if pla is not None else [],
        round_medians_us=medians[0],
        pla_round_medians_us=medians[1] if pla is not None else [],
        output_relative_rms=errors[0][0],
        state_relative_rms=errors[0][1],
        pla_output_relative_rms=errors[1][0] if pla is not None else None,
        pla_state_relative_rms=errors[1][1] if pla is not None else None,
        correctness="each_provider_cpu_fp32_single_step_and_graph_vs_eager",
        pla_available=pla is not None,
        pla_unavailable_reason=PLA_IMPORT_ERROR,
        pla_entrypoint=f"{pla.__module__}.{pla.__name__}" if pla is not None else None,
        pla_dispatch="official_compiled_lut_or_default" if pla is not None else None,
        shape_selection="four_head_configurations_paired_bf16_fp32_states_fixed_batches",
    )
    recorder.record()
    return (*shape, flag_us, pla_us)


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
        tp, batch, h, hv, state_dtype, flag_us, pla_us = row
        nonlocal emitted_dtype
        if state_dtype != emitted_dtype:
            emitted_dtype = state_dtype
            write("")
            write("State: BF16" if state_dtype == torch.bfloat16 else "State: FP32")
            write(
                f"{'TP':>2} {'B':>3} {'H':>2} {'HV':>2} {'K':>3} {'V':>3} "
                f"{'QKV':>5} {'State':>5} {'flag_attn(us)':>14} {'pla(us)':>11} {'pla/flag_attn':>14}"
            )
            write("-" * 93)
        state = "BF16" if state_dtype == torch.bfloat16 else "FP32"
        pla_text = "N/A" if pla_us is None else f"{pla_us:.3f}"
        ratio_text = "N/A" if pla_us is None else f"{pla_us / flag_us:.3f}"
        write(
            f"{tp:>2} {batch:>3} {h:>2} {hv:>2} {HEAD_DIM:>3} {HEAD_DIM:>3} "
            f"{'BF16':>5} {state:>5} {flag_us:>14.3f} {pla_text:>11} {ratio_text:>14}"
        )

    try:
        yield emit
    finally:
        manager.unregister(plugin)


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
