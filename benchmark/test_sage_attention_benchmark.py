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

"""SageAttention Q/K quantization + attention benchmark against PyTorch SDPA."""

from __future__ import annotations


import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import triton

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Direct script execution.
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn import sage_attention
from flag_attn.sage_attention import per_block_int8


SDPA_BASELINE_NAME = "torch_sdpa"
BENCHMARK_SCOPE = "quantization_and_attention"
DEFAULT_WARMUP = 25
DEFAULT_REPETITIONS = 100
MAX_ACCURACY_ELEMENTS = 1_048_576
DEFAULT_OUTPUT_DTYPE = torch.bfloat16

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
    parser.add_argument("--is-causal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _make_inputs(shape: tuple[int, int, int, int]):
    torch.manual_seed(42)
    q = torch.randn(shape, device=test_backend.device, dtype=torch.float16)
    k = torch.randn(shape, device=test_backend.device, dtype=torch.float16)
    v = torch.randn(shape, device=test_backend.device, dtype=torch.float16)
    return q, k, v


def _run_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = True) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=is_causal,
    )


def _run_flagattention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    maxnreg: int | None,
    is_causal: bool = True,
) -> torch.Tensor:
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    output, _ = sage_attention(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        output_dtype=output_dtype,
        maxnreg=maxnreg,
        is_causal=is_causal,
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
    is_causal: bool = True,
) -> BenchmarkResult:
    q, k, v = _make_inputs(shape)

    baseline_output = _run_baseline(q, k, v, is_causal=is_causal).to(output_dtype)
    flagattention_output = _run_flagattention(
        q,
        k,
        v,
        output_dtype=output_dtype,
        maxnreg=maxnreg,
        is_causal=is_causal,
    )
    cosine, relative_l2 = _accuracy_metrics(flagattention_output, baseline_output)
    assert cosine >= 0.99, f"SageAttention cosine similarity is too low: {cosine:.6f}"
    assert relative_l2 <= 0.05, (
        f"SageAttention relative L2 error is too high: {relative_l2:.6f}"
    )
    del baseline_output, flagattention_output

    baseline_ms = test_backend.do_bench(
        lambda: _run_baseline(q, k, v, is_causal=is_causal).to(output_dtype),
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )
    flagattention_ms = test_backend.do_bench(
        lambda: _run_flagattention(
            q,
            k,
            v,
            output_dtype=output_dtype,
            maxnreg=maxnreg,
            is_causal=is_causal,
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
    is_causal: bool = True,
) -> None:
    print(f"\n{'=' * 104}")
    print("  sage_attention benchmark — FWD ONLY")
    print(f"  provider: {_baseline_name(output_dtype)} vs flag_attn_sage_attention")
    print("  timing: SageAttention includes Q/K quantization + attention; SDPA includes its output cast")
    print(f"  is_causal={is_causal}; both providers start from the same FP16 Q/K/V inputs")
    print(f"  warmup={warmup}ms  rep={rep}ms")
    print(f"{'=' * 104}")
    print(
        f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
        f"{'sdpa(ms)':>10} {'sage_attn(ms)':>14} {'sdpa/sage_attn':>14} "
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
    is_causal: bool = True,
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    if record_property is not None:
        record_property("scope", BENCHMARK_SCOPE)
        record_property("baseline", _baseline_name(output_dtype))
        record_property("output_dtype", str(output_dtype))
        record_property("is_causal", is_causal)

    _print_header(output_dtype, warmup, rep, is_causal)
    print("\ndtype:", output_dtype)
    recorder = BenchmarkRecorder(
        record_property,
        op_name="sage_attention",
        dtype=str(output_dtype),
        mode="api",
        baseline=_baseline_name(output_dtype),
        phase="forward",
        scope=BENCHMARK_SCOPE,
        is_causal=is_causal,
    )
    for shape in shapes:
        result = _benchmark_case(
            shape,
            output_dtype=output_dtype,
            warmup=warmup,
            rep=rep,
            maxnreg=maxnreg,
            is_causal=is_causal,
        )
        _print_row(shape, output_dtype, result)
        recorder.add(
            shape_detail=shape,
            latency_base=result.baseline_ms,
            latency=result.flagattention_ms,
            accuracy=result.cosine_similarity,
            cosine_similarity=result.cosine_similarity,
            relative_l2=result.relative_l2,
        )
        _record_result(record_property, shape, output_dtype, result)
        test_backend.device_fn.empty_cache()
    recorder.record()
    _print_footer()


def benchmark(
    args: argparse.Namespace,
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    if not test_backend.supports_operator("sage_attention"):
        raise RuntimeError("SageAttention benchmark requires an available accelerator")

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
        is_causal=args.is_causal,
        record_property=record_property,
    )
