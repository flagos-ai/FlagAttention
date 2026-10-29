# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark T-Head ordinary GDN forward with an independent CPU reference.

Run with pytest benchmark/test_fused_recurrent_gated_delta_rule_benchmark.py.
Inputs use precomputed g/beta and BF16 state.
Gate preparation and state restoration are not timed.
Latency is the median of seven round medians, in microseconds, using 32-call
Graph replays.
"""

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

pytestmark = pytest.mark.chunk_gated_delta_rule

ASSERT_RATIO = 0.01
HEAD_DIM = 128
DTYPE = STATE_DTYPE = torch.bfloat16
SEED = 0
ROUNDS, REPEAT_ITERS, DRY_RUN_ITERS = 7, 40, 5
INITIAL_WARMUP, ROUND_WARMUP = 10, 30
CALLS_PER_GRAPH = 32
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


def _shape_matrix():
    return [(tp, batch, h, hv) for tp, h, hv in HEAD_CONFIGS for batch in BATCHES]



def _check_result(call, state, initial, expected, expected_state):
    out = call()
    test_backend.device_fn.synchronize()
    output_error = state_error = 0.0
    # Every row has one token and a distinct valid slot, numbered 1..B.
    for row in range(out.shape[0]):
        output_error = max(output_error, _assert_close(f"output[{row}]", out[row], expected[row]))
        slot = row + 1
        state_error = max(state_error, _assert_close(f"state[{slot}]", state[slot], expected_state[slot]))
    assert torch.equal(state[0], initial[0]), "null state slot changed"
    return output_error, state_error


def _capture(call, state, initial):
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

    # Check that a Graph replay produces the same updates as eager execution.
    state.copy_(initial)
    graph.replay()
    saved_out, saved_state = out.clone(), state.clone()
    state.copy_(initial)
    for _ in range(CALLS_PER_GRAPH):
        eager_out = call()
    device.synchronize()
    assert torch.equal(saved_out, eager_out), "Graph/eager output mismatch"
    assert torch.equal(saved_state, state), "Graph/eager state mismatch"
    assert torch.equal(state[0], initial[0]), "null state slot changed"
    return graph, out


def _measure(call, state, initial):
    device = test_backend.device_fn
    if not test_backend.graph_available() or not hasattr(device, "Event"):
        pytest.fail("this benchmark requires T-Head Graph and Event support")
    graph, out = _capture(call, state, initial)
    events = [
        (device.Event(enable_timing=True), device.Event(enable_timing=True)) for _ in range(REPEAT_ITERS)
    ]
    for start, end in events:
        start.record()
        end.record()
    device.synchronize()
    samples = []
    for _ in range(ROUNDS):
        state.copy_(initial)
        for _ in range(ROUND_WARMUP):
            call()
        for _ in range(DRY_RUN_ITERS):
            graph.replay()
        device.synchronize()
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        device.synchronize()
        values = [start.elapsed_time(end) * 1000.0 / CALLS_PER_GRAPH for start, end in events]
        assert all(math.isfinite(value) and value > 0 for value in values)
        assert torch.isfinite(out).all(), "nonfinite output"
        assert torch.isfinite(state).all(), "nonfinite state"
        assert torch.equal(state[0], initial[0]), "null slot changed"
        samples.append(values)
    return samples


@torch.inference_mode()
def run_benchmark(record_property, shape):
    tp, batch, h, hv = shape
    fwd = _load_fwd()
    inputs = _make_inputs(batch, h, hv)
    initial = inputs["initial_state"].clone()
    state = inputs["initial_state"]

    def run_flag():
        return fwd(**inputs)[0]

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, expected_state, _, _ = _reference(inputs)
        output_error, state_error = _check_result(run_flag, state, initial, expected, expected_state)
    finally:
        torch.set_num_threads(threads)
    del expected, expected_state
    samples = _measure(run_flag, state, initial)
    medians = [statistics.median(row) for row in samples]
    flag_us = statistics.median(medians)
    quantiles = torch.quantile(
        torch.tensor([v for row in samples for v in row]), torch.tensor([0.2, 0.8])
    ).tolist()
    recorder = BenchmarkRecorder(
        record_property,
        op_name="fused_recurrent_gated_delta_rule_fwd",
        dtype=str(DTYPE),
        phase="forward",
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
        latency_unit="us",
        p20_us=quantiles[0],
        p80_us=quantiles[1],
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
        samples_us=samples,
        round_medians_us=medians,
        output_relative_rms=output_error,
        state_relative_rms=state_error,
        correctness="cpu_fp32_single_step_and_graph_vs_eager",
        shape_selection="eight_head_configurations_fixed_batches_1_2_4_8_16_32_64",
    )
    recorder.record()
    return (*shape, flag_us)


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
        f"{'K':>3} {'V':>3} {'QKV/State':>9} {'flag_attn(us)':>14}"
    )
    write("-" * 66)

    def emit(row):
        tp, batch, h, hv, flag_us = row
        write(
            f"{tp:>2} {batch:>3} {1:>2} {h:>2} {hv:>2} "
            f"{HEAD_DIM:>3} {HEAD_DIM:>3} {'BF16/BF16':>9} "
            f"{flag_us:>14.3f}"
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
