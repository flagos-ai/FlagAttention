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

"""SageAttention speedup benchmark against PyTorch SDPA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
import torch.nn.functional as F
import triton

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:  # Direct script execution.
    from recording import benchmark_metric, record_benchmark_result

from flag_attn.sage_attention import forward, per_block_int8


SDPA_BASELINE_NAME = "torch_sdpa"
BENCHMARK_SCOPE = "attention_kernel_only"
DEFAULT_WARMUP = 25
DEFAULT_REPETITIONS = 100
MAX_ACCURACY_ELEMENTS = 1_048_576
DEFAULT_OUTPUT_DTYPE = torch.bfloat16

# Reuse the first three workloads from the original benchmark defaults while
# keeping pytest runtime bounded. The longer defaults remain available through
# the command-line entry point.
PYTEST_CASES = (
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
)


@dataclass(frozen=True)
class BenchmarkResult:
    baseline_ms: float
    flagattention_ms: float
    cosine_similarity: float
    relative_l2: float

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.flagattention_ms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=(1024, 2048, 4096, 8192, 16384, 32768),
    )
    parser.add_argument(
        "--output-dtype",
        "--dtype",
        dest="output_dtype",
        choices=("float16", "bfloat16"),
        default=str(DEFAULT_OUTPUT_DTYPE).removeprefix("torch."),
        help=(
            "SageAttention output dtype; Q, K, and V remain FP16 before Q/K "
            "quantization. BF16 baseline timing includes the SDPA output cast"
        ),
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--rep", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--maxnreg", type=int)
    return parser.parse_args(argv)


def _make_inputs(shape: tuple[int, int, int, int]):
    torch.manual_seed(42)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    return q, k, v, q_int8, q_scale, k_int8, k_scale


def _run_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
    )


def _run_flagattention(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    maxnreg: int | None,
) -> torch.Tensor:
    output, _ = forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        output_dtype=output_dtype,
        maxnreg=maxnreg,
    )
    return output


def _accuracy_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float]:
    actual = actual.flatten()
    expected = expected.flatten()
    if actual.numel() > MAX_ACCURACY_ELEMENTS:
        stride = (actual.numel() + MAX_ACCURACY_ELEMENTS - 1) // MAX_ACCURACY_ELEMENTS
        actual = actual[::stride]
        expected = expected[::stride]
    actual = actual.float()
    expected = expected.float()
    cosine = F.cosine_similarity(actual, expected, dim=0).item()
    relative_l2 = ((actual - expected).norm() / expected.norm().clamp_min(1e-8)).item()
    return cosine, relative_l2


@torch.inference_mode()
def _benchmark_case(
    shape: tuple[int, int, int, int],
    *,
    output_dtype: torch.dtype = torch.float16,
    warmup: int = DEFAULT_WARMUP,
    rep: int = DEFAULT_REPETITIONS,
    maxnreg: int | None = None,
) -> BenchmarkResult:
    q, k, v, q_int8, q_scale, k_int8, k_scale = _make_inputs(shape)

    baseline_output = _run_baseline(q, k, v).to(output_dtype)
    flagattention_output = _run_flagattention(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        output_dtype=output_dtype,
        maxnreg=maxnreg,
    )
    cosine, relative_l2 = _accuracy_metrics(flagattention_output, baseline_output)
    assert cosine >= 0.99, f"SageAttention cosine similarity is too low: {cosine:.6f}"
    assert relative_l2 <= 0.05, (
        f"SageAttention relative L2 error is too high: {relative_l2:.6f}"
    )
    del baseline_output, flagattention_output

    baseline_ms = triton.testing.do_bench(
        lambda: _run_baseline(q, k, v).to(output_dtype),
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )
    flagattention_ms = triton.testing.do_bench(
        lambda: _run_flagattention(
            q_int8,
            k_int8,
            v,
            q_scale,
            k_scale,
            output_dtype=output_dtype,
            maxnreg=maxnreg,
        ),
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )
    return BenchmarkResult(
        baseline_ms=baseline_ms,
        flagattention_ms=flagattention_ms,
        cosine_similarity=cosine,
        relative_l2=relative_l2,
    )


def _shape_name(shape: tuple[int, int, int, int]) -> str:
    batch_size, num_heads, seq_len, head_dim = shape
    return f"B{batch_size}_H{num_heads}_T{seq_len}_D{head_dim}"


def _baseline_name(output_dtype: torch.dtype) -> str:
    if output_dtype == torch.float16:
        return SDPA_BASELINE_NAME
    return f"{SDPA_BASELINE_NAME}_plus_{str(output_dtype).removeprefix('torch.')}_cast"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _print_header(
    output_dtype: torch.dtype,
    warmup: int,
    rep: int,
) -> None:
    print(f"\n{'=' * 104}")
    print("  sage_attention benchmark — FWD ONLY")
    print(f"  provider: {_baseline_name(output_dtype)} vs flag_attn_sage_attention")
    print("  timing: Q/K quantization excluded from both providers")
    print(f"  warmup={warmup}ms  rep={rep}ms")
    print(f"{'=' * 104}")
    print(
        f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
        f"{'sdpa(ms)':>10} {'flag_attn(ms)':>14} {'sdpa/flag_attn':>14} "
        f"{'cosine':>10} {'rel_l2':>10}"
    )


def _print_row(
    shape: tuple[int, int, int, int],
    output_dtype: torch.dtype,
    result: BenchmarkResult,
) -> None:
    B, H, T, D = shape
    print(
        f"{B:>3} {T:>6} {H:>4} {D:>4} {_dtype_name(output_dtype):>8} "
        f"{result.baseline_ms:>10.3f} {result.flagattention_ms:>14.3f} "
        f"{result.speedup:>14.2f}x {result.cosine_similarity:>10.6f} "
        f"{result.relative_l2:>10.6f}"
    )


def _print_footer() -> None:
    print(f"\n{'=' * 104}")
    print("  All done.")
    print(f"{'=' * 104}\n")


def _record_result(
    record_property: Callable[[str, object], None] | None,
    shape: tuple[int, int, int, int],
    output_dtype: torch.dtype,
    result: BenchmarkResult,
) -> None:
    if record_property is None:
        return
    prefix = f"{_shape_name(shape)}_{_dtype_name(output_dtype)}"
    record_property(f"{prefix}.baseline_ms", result.baseline_ms)
    record_property(f"{prefix}.flagattention_ms", result.flagattention_ms)
    record_property(f"{prefix}.speedup_vs_sdpa", result.speedup)
    record_property(f"{prefix}.cosine_similarity", result.cosine_similarity)
    record_property(f"{prefix}.relative_l2", result.relative_l2)


def _run_benchmark_cases(
    shapes: tuple[tuple[int, int, int, int], ...],
    *,
    output_dtype: torch.dtype,
    warmup: int,
    rep: int,
    maxnreg: int | None,
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    if record_property is not None:
        record_property("scope", BENCHMARK_SCOPE)
        record_property("baseline", _baseline_name(output_dtype))
        record_property("output_dtype", str(output_dtype))

    _print_header(output_dtype, warmup, rep)
    print("\ndtype:", output_dtype)
    metrics = []
    for shape in shapes:
        result = _benchmark_case(
            shape,
            output_dtype=output_dtype,
            warmup=warmup,
            rep=rep,
            maxnreg=maxnreg,
        )
        _print_row(shape, output_dtype, result)
        metrics.append(
            benchmark_metric(
                shape_detail=shape,
                latency_base=result.baseline_ms,
                latency=result.flagattention_ms,
                speedup=result.speedup,
                accuracy=result.cosine_similarity,
                cosine_similarity=result.cosine_similarity,
                relative_l2=result.relative_l2,
            )
        )
        _record_result(record_property, shape, output_dtype, result)
        torch.cuda.empty_cache()
    record_benchmark_result(
        record_property,
        op_name="sage_attention",
        dtype=str(output_dtype),
        result=metrics,
        baseline=_baseline_name(output_dtype),
        phase="forward",
        scope=BENCHMARK_SCOPE,
    )
    _print_footer()


@pytest.mark.sage_attention
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SageAttention benchmark requires CUDA"
)
def test_sage_attention_benchmark(
    record_property: Callable[[str, object], None],
) -> None:
    _run_benchmark_cases(
        PYTEST_CASES,
        output_dtype=DEFAULT_OUTPUT_DTYPE,
        warmup=DEFAULT_WARMUP,
        rep=DEFAULT_REPETITIONS,
        maxnreg=None,
        record_property=record_property,
    )


def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SageAttention benchmark requires CUDA")

    output_dtype = getattr(torch, args.output_dtype)
    shapes = tuple(
        (args.batch_size, args.num_heads, seq_len, args.head_dim)
        for seq_len in args.seq_lens
    )
    _run_benchmark_cases(
        shapes,
        output_dtype=output_dtype,
        warmup=args.warmup,
        rep=args.rep,
        maxnreg=args.maxnreg,
    )


if __name__ == "__main__":
    benchmark(parse_args())
