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

"""Log-Linear Attention performance tests for the selected implementation."""

import importlib
import math
import statistics

import pytest
import torch

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:
    from recording import benchmark_metric, record_benchmark_result

from flag_attn import log_linear_attn
from flag_attn.testing import backend as test_backend

pytestmark = pytest.mark.log_linear_attn

BENCHMARK_SHAPES = [
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (8, 1024, 8, 64),
]
BENCHMARK_ROUNDS = 7
BENCHMARK_WARMUP_MS = 1000
BENCHMARK_REP_MS = 100


def _make_inputs(batch, sequence, heads, dim):
    levels = math.ceil(math.log2(sequence)) + 1
    q = torch.randn(batch, sequence, 1, dim, device=test_backend.device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, heads, dim, device=test_backend.device, dtype=torch.bfloat16)
    g = -0.05 * torch.rand(batch, sequence, heads, device=test_backend.device, dtype=torch.float32)
    level_scales = torch.sigmoid(
        torch.randn(
            batch,
            sequence,
            heads,
            levels,
            device=test_backend.device,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    return q, k, v, g, level_scales


def _load_fla_reference():
    if not test_backend.is_nvidia():
        return None, "FLA Log-Linear Attention requires NVIDIA CUDA"
    try:
        op = importlib.import_module("fla.ops.log_linear_attn").chunk_log_linear_attn
    except (ImportError, AttributeError) as exc:
        return None, str(exc)
    return op, None


@torch.inference_mode()
def run_benchmark(
    *,
    shapes=None,
    rounds=BENCHMARK_ROUNDS,
    warmup=BENCHMARK_WARMUP_MS,
    rep=BENCHMARK_REP_MS,
    record_property=None,
):
    if not test_backend.supports_operator("log_linear_attn", tle=True):
        pytest.skip(test_backend.skip_reason("log_linear_attn"))
    if rounds < 1 or warmup < 0 or rep <= 0:
        raise ValueError("rounds and rep must be positive; warmup must be nonnegative")

    shapes = BENCHMARK_SHAPES if shapes is None else shapes
    fla_op, fla_error = _load_fla_reference()
    if fla_op is None:
        print(f"[baseline] FLA unavailable; measuring FlagAttention only: {fla_error}")

    print(f"Device={test_backend.get_device_name()} dtype=BF16", flush=True)
    print(f"rounds={rounds}, warmup={warmup} ms, rep={rep} ms", flush=True)
    print("shape\tlatency_ms\tfla_ms\tspeedup_vs_fla", flush=True)
    metrics = []
    for batch, sequence, heads, dim in shapes:
        torch.manual_seed(0)
        inputs = _make_inputs(batch, sequence, heads, dim)

        def run_flag():
            return log_linear_attn(*inputs)

        output = run_flag()
        test_backend.device_fn.synchronize()
        assert torch.isfinite(output).all(), "Log-Linear Attention produced NaN/Inf"

        fla_rounds = []
        if fla_op is not None:
            def run_fla():
                return fla_op(*inputs)[0]

            baseline_output = run_fla()
            test_backend.device_fn.synchronize()
            assert torch.isfinite(baseline_output).all(), "FLA baseline produced NaN/Inf"
            error = (baseline_output.float() - output.float()).square().mean().sqrt()
            reference = output.float().square().mean().sqrt()
            assert (error / (reference + 1e-8)).item() < 0.004, (
                "FLA baseline differs from Log-Linear Attention"
            )

        flag_rounds = []
        for round_index in range(rounds):
            if fla_op is not None and round_index % 2 == 0:
                fla_rounds.append(
                    test_backend.do_bench(
                        run_fla, warmup=warmup, rep=rep, return_mode="median"
                    )
                )
            flag_rounds.append(
                test_backend.do_bench(
                    run_flag, warmup=warmup, rep=rep, return_mode="median"
                )
            )
            if fla_op is not None and round_index % 2 == 1:
                fla_rounds.append(
                    test_backend.do_bench(
                        run_fla, warmup=warmup, rep=rep, return_mode="median"
                    )
                )

        latency = statistics.median(flag_rounds)
        baseline_latency = statistics.median(fla_rounds) if fla_rounds else None
        speedup = baseline_latency / latency if baseline_latency is not None else None
        shape = f"B{batch}_T{sequence}_H{heads}_D{dim}"
        baseline_text = f"{baseline_latency:.6f}" if baseline_latency is not None else "N/A"
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "N/A"
        print(f"{shape}\t{latency:.6f}\t{baseline_text}\t{speedup_text}", flush=True)
        metrics.append(
            benchmark_metric(
                shape_detail={"B": batch, "T": sequence, "H": heads, "D": dim},
                latency=latency,
                latency_base=baseline_latency,
                speedup=speedup,
                measurement_scope="public_api_forward",
            )
        )

    record_benchmark_result(
        record_property,
        op_name="log_linear_attn",
        dtype=str(torch.bfloat16),
        result=metrics,
        mode="api",
        baseline="FLA" if fla_op is not None else None,
        phase="forward",
        rounds=rounds,
        warmup_ms=warmup,
        rep_ms=rep,
    )
    return metrics


def test_log_linear_attn_benchmark(record_property):
    run_benchmark(record_property=record_property)
