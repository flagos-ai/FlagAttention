# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare T-Head ordinary GDN forward with the bundled PLA precomputed adapter.

Run with pytest benchmark/test_fused_recurrent_gated_delta_rule_benchmark.py.
Both providers receive identical precomputed g/beta and independent copies of
the same BF16 state. Gate preparation and state restoration are not timed.
Latency is the median of seven round medians, in microseconds, using 32-call
Graph replays. PLA/FlagAttention > 1 means FlagAttention is faster.
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

pytestmark = pytest.mark.fused_recurrent_gated_delta_rule_fwd

ASSERT_RATIO = 0.01
HEAD_DIM = 128
DTYPE = STATE_DTYPE = torch.bfloat16
SEED = 0
ROUNDS, REPEAT_ITERS, DRY_RUN_ITERS = 7, 40, 5
INITIAL_WARMUP, ROUND_WARMUP = 10, 30
CALLS_PER_GRAPH = 32
PLA_IMPORT_ERROR = None
# Head configurations are frozen from PLA caselist/decode. Batches use the
# common fixed matrix below.
# TP denotes local head counts; no distributed run is started.
BATCHES = (1, 2, 4, 8, 16, 32, 64)
HEAD_CONFIGS = (
    (1, 16, 32),
    (2, 8, 16),
    (4, 4, 8),
    (1, 16, 48),
    (2, 8, 24),
    (1, 16, 64),
    (2, 8, 32),
    (4, 4, 16),
)


def _load_fwd():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )

    return fused_recurrent_gated_delta_rule_fwd


def _reference(inputs):
    """CPU FP32 recurrence; state ABI is [slots, HV, V, K]."""
    q, k, v, g, beta = [inputs[name].detach().float().cpu() for name in ("q", "k", "v", "g", "beta")]
    state0 = inputs["initial_state"].detach().float().cpu()
    indices = inputs["ssm_state_indices"].detach().cpu()
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    cu = inputs["cu_seqlens"]
    spans = (
        list(zip(cu.cpu().tolist()[:-1], cu.cpu().tolist()[1:]))
        if cu is not None
        else [(b * T, (b + 1) * T) for b in range(B)]
    )
    q, k = q.reshape(-1, H, K), k.reshape(-1, H, K)
    v, g = v.reshape(-1, HV, V), g.reshape(-1, HV)
    beta = beta.reshape(-1, HV, V) if beta.ndim == 4 else beta.reshape(-1, HV, 1)
    out = torch.zeros_like(v)
    final = state0.clone() if inputs["inplace_final_state"] else torch.empty(B * T, HV, V, K)
    valid = torch.zeros(B * T, dtype=torch.bool)
    written = set()
    accepted = inputs.get("num_accepted_tokens")
    accepted = None if accepted is None else accepted.detach().cpu()
    for n, (start, end) in enumerate(spans):
        if start == end:
            continue
        row = indices[n].reshape(-1)
        read_pos = int(accepted[n]) - 1 if accepted is not None else 0
        slot = int(row[read_pos])
        if slot <= 0:
            continue
        state = state0[slot].clone()
        for t, pos in enumerate(range(start, end)):
            qt = q[pos].repeat_interleave(HV // H, dim=0)
            kt = k[pos].repeat_interleave(HV // H, dim=0)
            if inputs["use_qk_l2norm_in_kernel"]:
                qt = qt / torch.sqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            state = state * torch.exp(g[pos])[:, None, None]
            residual = (v[pos] - (state * kt[:, None, :]).sum(-1)) * beta[pos]
            state = state + residual[:, :, None] * kt[:, None, :]
            out[pos] = (state * (qt * inputs["scale"])[:, None, :]).sum(-1)
            valid[pos] = True
            dest = int(row[t]) if inputs["inplace_final_state"] else pos
            if not inputs["inplace_final_state"] or dest > 0:
                final[dest] = state
                written.add(dest)
    return out.reshape(B, T, HV, V), final, valid, sorted(written)


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


def _make_inputs(batch, H, HV):
    torch.manual_seed(SEED)
    device = test_backend.device
    K = V = HEAD_DIM
    q = torch.randn(batch, 1, H, K, device=device, dtype=DTYPE)
    k = torch.randn(batch, 1, H, K, device=device, dtype=DTYPE)
    v = torch.randn(batch, 1, HV, V, device=device, dtype=DTYPE)
    g = -torch.rand(batch, 1, HV, device=device, dtype=torch.float32)
    beta = torch.rand(batch, 1, HV, device=device, dtype=DTYPE)
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    state = torch.randn(batch + 1, HV, V, K, device=device, dtype=STATE_DTYPE)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K ** -0.5,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


@lru_cache(maxsize=1)
def _load_pla():
    global PLA_IMPORT_ERROR
    try:
        pla = import_module(
            "flag_attn.runtime.backend._thead.FLA.gated_delta_rule." "pla_gdn.precomputed.pla_precomputed"
        )
    except (ImportError, OSError) as exc:
        PLA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None
    PLA_IMPORT_ERROR = None
    return pla


def _shape_matrix():
    return [(tp, batch, h, hv) for tp, h, hv in HEAD_CONFIGS for batch in BATCHES]



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
    names = ("FlagAttention", "PLA precomputed")
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


@torch.inference_mode()
def run_benchmark(record_property, shape):
    tp, batch, h, hv = shape
    fwd = _load_fwd()
    pla = _load_pla()
    inputs = _make_inputs(batch, h, hv)
    initial = inputs["initial_state"].clone()
    states = (inputs["initial_state"], initial.clone())

    def run_flag():
        return fwd(**inputs)[0]

    def run_pla():
        return pla.run(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            states[1],
            inputs["ssm_state_indices"],
            inputs["scale"],
            True,
        )

    calls = (run_flag, run_pla) if pla is not None else (run_flag,)
    config = list(pla.config(batch, h, hv)) if pla is not None else None
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_state, _, _ = _reference(inputs)
        errors = [
            _check_provider(name, call, state, initial, expected, expected_state)
            for name, call, state in zip(("FlagAttention", "PLA precomputed"), calls, states)
        ]
    finally:
        torch.set_num_threads(threads)
    del expected, expected_state
    samples = _measure_pair(calls, states, initial)
    medians = [[statistics.median(row) for row in provider] for provider in samples]
    flag_us = statistics.median(medians[0])
    pla_us = statistics.median(medians[1]) if pla is not None else None
    quantiles = [
        torch.quantile(torch.tensor([v for row in provider for v in row]), torch.tensor([0.2, 0.8])).tolist()
        for provider in samples
    ]
    recorder = BenchmarkRecorder(
        record_property,
        op_name="fused_recurrent_gated_delta_rule_fwd",
        dtype=str(DTYPE),
        phase="forward",
        baseline="pla_precomputed" if pla is not None else None,
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
            q_shape=list(inputs["q"].shape),
            state_shape=list(initial.shape),
            state_dtype=str(STATE_DTYPE),
            gate_dtype=str(inputs["g"].dtype),
            beta_dtype=str(inputs["beta"].dtype),
            scale=inputs["scale"],
            qk_normalization=True,
            inplace=True,
            seed=SEED,
        ),
        latency=flag_us,
        latency_base=pla_us,
        latency_unit="us",
        p20_us=quantiles[0][0],
        p80_us=quantiles[0][1],
        pla_p20_us=quantiles[1][0] if pla is not None else None,
        pla_p80_us=quantiles[1][1] if pla is not None else None,
        sample_count=ROUNDS * REPEAT_ITERS,
        calls_per_graph=CALLS_PER_GRAPH,
        rounds=ROUNDS,
        samples_per_round=REPEAT_ITERS,
        initial_warmup=INITIAL_WARMUP,
        round_warmup=ROUND_WARMUP,
        dry_run_iters=DRY_RUN_ITERS,
        execution_kind="multi_call_graph_replay",
        timing_method="device_event",
        latency_scope="precomputed_gates_forward_amortized_graph_update",
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
        pla_entrypoint=pla.__file__ if pla is not None else None,
        pla_config=config,
        pla_available=pla is not None,
        pla_unavailable_reason=PLA_IMPORT_ERROR,
        shape_selection="official_head_configs_fixed_batches_1_2_4_8_16_32_64",
    )
    recorder.record()
    return (*shape, flag_us, pla_us)


@pytest.fixture(scope="module")
def benchmark_table(request):
    """Stream table rows while retaining pytest's diagnostics."""
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

    write("")
    write(
        f"{'TP':>2} {'B':>3} {'T':>2} {'H':>2} {'HV':>2} "
        f"{'K':>3} {'V':>3} {'QKV/State':>9} {'flag_attn(us)':>14} "
        f"{'pla(us)':>11} {'pla/flag_attn':>14}"
    )
    write("-" * 92)

    def emit(row):
        tp, batch, h, hv, flag_us, pla_us = row
        pla_text = "N/A" if pla_us is None else f"{pla_us:.3f}"
        ratio_text = "N/A" if pla_us is None else f"{pla_us / flag_us:.3f}"
        write(
            f"{tp:>2} {batch:>3} {1:>2} {h:>2} {hv:>2} "
            f"{HEAD_DIM:>3} {HEAD_DIM:>3} {'BF16/BF16':>9} "
            f"{flag_us:>14.3f} {pla_text:>11} {ratio_text:>14}"
        )

    try:
        yield emit
    finally:
        manager.unregister(plugin)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shape",
    _shape_matrix(),
    ids=[f"TP{tp}-B{batch}-T1-H{h}-HV{hv}-K128-V128-bf16" for tp, batch, h, hv in _shape_matrix()],
)
def test_fused_recurrent_gated_delta_rule_benchmark(record_property, shape, benchmark_table):
    benchmark_table(run_benchmark(record_property, shape))
