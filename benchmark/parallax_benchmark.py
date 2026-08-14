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

from __future__ import annotations
import gc
from dataclasses import dataclass
from enum import Enum
import math
import os
from statistics import median
from typing import Callable

# Set the Triton floating-point mode before importing Triton-backed kernels.
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import pytest
import torch
import triton

from fla.ops.parallax.parallel import (
    _block_size as fla_block_size,
    parallel_parallax_bwd as fla_parallel_parallax_bwd,
    parallel_parallax_fwd as fla_parallel_parallax_fwd,
)
from fla.ops.utils import prepare_chunk_indices as fla_prepare_chunk_indices
from flag_attn.parallax.index import (
    prepare_chunk_indices as flag_attn_prepare_chunk_indices,
)
from flag_attn.parallax.parallel import (
    _block_size as flag_attn_block_size,
    parallel_parallax_bwd as flag_attn_parallel_parallax_bwd,
    parallel_parallax_fwd as flag_attn_parallel_parallax_fwd,
)


class BenchMode(Enum):
    KERNEL = "kernel"
    OPERATOR = "operator"
    WRAPPER = "wrapper"


class BenchConfig:
    """Self-contained configuration for this benchmark.

    Environment variables:
      PARALLAX_BENCH_MODE: kernel, operator, or wrapper (default: kernel)
      PARALLAX_BENCH_WARMUP: warm-up budget (default: 100)
      PARALLAX_BENCH_ITER: measurement budget (default: 100)
      PARALLAX_BENCH_DTYPES: comma-separated float16,bfloat16 (default: bfloat16)
    """

    def __init__(self) -> None:
        mode = os.getenv("PARALLAX_BENCH_MODE", "kernel").lower()
        try:
            self.mode = BenchMode(mode)
        except ValueError as exc:
            raise ValueError(
                "PARALLAX_BENCH_MODE must be kernel, operator, or wrapper"
            ) from exc

        self.warm_up = int(os.getenv("PARALLAX_BENCH_WARMUP", "100"))
        self.repetition = int(os.getenv("PARALLAX_BENCH_ITER", "100"))
        if self.warm_up < 0:
            raise ValueError("PARALLAX_BENCH_WARMUP must be non-negative")
        if self.repetition <= 0:
            raise ValueError("PARALLAX_BENCH_ITER must be positive")

        dtype_names = os.getenv("PARALLAX_BENCH_DTYPES", "").strip()
        self.user_desired_dtypes = None
        if dtype_names:
            supported = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            requested = [name.strip() for name in dtype_names.split(",")]
            unknown = [name for name in requested if name not in supported]
            if unknown:
                raise ValueError(
                    "PARALLAX_BENCH_DTYPES only supports float16,bfloat16; "
                    f"got {','.join(unknown)}"
                )
            self.user_desired_dtypes = [supported[name] for name in requested]


Config = BenchConfig()


DEFAULT_DTYPES = (torch.bfloat16,)
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

PAIR_WARMUP_CYCLES = 3
BALANCED_MEASUREMENT_CYCLES = 6
VALIDATION_SAMPLE_COUNT = 16_384
ACCURACY_TOLERANCE = {
    torch.float16: 5e-3,
    torch.bfloat16: 2e-2,
}
MAD_WARNING_THRESHOLD_PCT = 1.0
SHORT_LATENCY_THRESHOLD_MS = 0.2
SHORT_LATENCY_MAD_WARNING_PCT = 2.0
TABLE_WIDTH = 123


@dataclass(frozen=True)
class ParallaxCase:
    B: int
    T: int
    H: int
    HQ: int
    D: int
    window_size: int | None = None
    cu_seqlens: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PhaseBenchmarkResult:
    fla_ms: float
    flag_attn_ms: float
    fla_mad_pct: float
    flag_attn_mad_pct: float

    @property
    def speedup(self) -> float:
        # FLA is the baseline. A value greater than 1 means that
        # FlagAttention has lower latency and is therefore faster.
        return self.fla_ms / self.flag_attn_ms


DEFAULT_CASES = (
    ParallaxCase(B=1, T=15, H=2, HQ=2, D=64),
    ParallaxCase(B=1, T=63, H=1, HQ=1, D=64),
    ParallaxCase(B=1, T=111, H=2, HQ=2, D=64),
    ParallaxCase(B=2, T=200, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=256, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=512, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=1024, H=2, HQ=2, D=64),
    ParallaxCase(B=2, T=2048, H=2, HQ=8, D=64),
    ParallaxCase(B=2, T=4096, H=2, HQ=2, D=64),
    ParallaxCase(B=2, T=8192, H=2, HQ=8, D=64),
    ParallaxCase(B=3, T=111, H=2, HQ=2, D=100),
    ParallaxCase(B=3, T=1024, H=2, HQ=8, D=128),
    ParallaxCase(B=2, T=2048, H=2, HQ=8, D=128),
    ParallaxCase(B=2, T=8192, H=8, HQ=16, D=128),
)


def _build_inputs(
    case: ParallaxCase,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    torch.Tensor | None,
    int,
]:
    if case.HQ % case.H != 0:
        raise ValueError("HQ must be divisible by H")

    device = "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(_case_seed(case, dtype))

    query_shape = (
        case.B,
        case.T,
        case.HQ,
        case.D,
    )
    kv_shape = (
        case.B,
        case.T,
        case.H,
        case.D,
    )

    q = torch.randn(
        query_shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    r = torch.randn(
        query_shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        kv_shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        kv_shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    cu_seqlens = None

    if case.cu_seqlens is not None:
        if (
            case.B != 1
            or case.cu_seqlens[0] != 0
            or case.cu_seqlens[-1] != case.T
        ):
            raise ValueError(
                "A variable-length case requires B=1 and "
                "cu_seqlens spanning [0, T]"
            )

        cu_seqlens = torch.tensor(
            case.cu_seqlens,
            dtype=torch.long,
            device=device,
        )

    scale = case.D**-0.5

    window_size_left = (
        -1
        if case.window_size is None
        else case.window_size
    )

    return (
        q,
        r,
        k,
        v,
        scale,
        cu_seqlens,
        window_size_left,
    )


def _case_seed(case: ParallaxCase, dtype: torch.dtype) -> int:
    """Create a stable, order-independent seed for one benchmark case."""

    values = [
        case.B,
        case.T,
        case.H,
        case.HQ,
        case.D,
        case.window_size if case.window_size is not None else -1,
        0 if dtype == torch.float16 else 1,
    ]
    if case.cu_seqlens is not None:
        values.extend(case.cu_seqlens)

    seed = 42
    modulus = 2**63 - 1
    for value in values:
        seed = (seed * 1_000_003 + value) % modulus
    return seed


def _build_chunk_indices(
    case: ParallaxCase,
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
]:

    if cu_seqlens is None:
        return None, None

    device_index = q.device.index

    fla_bt = fla_block_size(
        case.D,
        device_index,
    )
    flag_attn_bt = flag_attn_block_size(
        case.D,
        device_index,
    )

    fla_chunk_indices = fla_prepare_chunk_indices(
        cu_seqlens,
        fla_bt,
    )
    flag_attn_chunk_indices = flag_attn_prepare_chunk_indices(
        cu_seqlens,
        flag_attn_bt,
    )

    return (
        fla_chunk_indices,
        flag_attn_chunk_indices,
    )


def _bench_ms(fn: Callable[[], object]) -> float:
    """Return average milliseconds per provider invocation.

    Kernel mode uses Triton's per-invocation device timing so short kernels do
    not absorb host launch and allocator gaps. Operator/wrapper modes retain
    count-based CUDA-event batch timing.
    """

    torch.cuda.synchronize()

    if Config.mode.value == "kernel":
        result = float(
            triton.testing.do_bench(
                fn,
                # Re-stabilize clocks after every provider switch. ABBA/BAAB
                # applies the same warm-up budget symmetrically to both paths.
                warmup=Config.warm_up,
                rep=Config.repetition,
                return_mode="median",
            )
        )
        torch.cuda.synchronize()
        return result

    repetitions = max(1, int(Config.repetition))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(repetitions):
        fn()

    end.record()

    torch.cuda.synchronize()

    return (
        start.elapsed_time(end)
        / repetitions
    )


def _warm_up_pair(
    fla_fn: Callable[[], object],
    flag_attn_fn: Callable[[], object],
) -> None:
    """Compile and warm both providers without including warm-up in timing."""

    if Config.mode.value == "kernel":
        # Compile both paths once. Kernel-mode clock stabilization is applied
        # inside every measured sample by _bench_ms.
        for fn in (fla_fn, flag_attn_fn):
            fn()
    else:
        for cycle in range(Config.warm_up):
            functions = (
                (fla_fn, flag_attn_fn)
                if cycle % 2 == 0
                else (flag_attn_fn, fla_fn)
            )
            for fn in functions:
                fn()

    for cycle in range(PAIR_WARMUP_CYCLES):
        for fn in _measurement_order(cycle, fla_fn, flag_attn_fn):
            fn()

    torch.cuda.synchronize()


def _measurement_order(
    cycle: int,
    fla_fn: Callable[[], object],
    flag_attn_fn: Callable[[], object],
) -> tuple[Callable[[], object], ...]:


    if cycle % 2 == 0:
        return (
            fla_fn,
            flag_attn_fn,
            flag_attn_fn,
            fla_fn,
        )

    return (
        flag_attn_fn,
        fla_fn,
        fla_fn,
        flag_attn_fn,
    )


def _median_and_mad_pct(
    samples: list[float],
) -> tuple[float, float]:

    center = float(median(samples))

    if center == 0.0:
        return center, 0.0

    absolute_deviations = [
        abs(sample - center)
        for sample in samples
    ]
    mad = float(median(absolute_deviations))

    return center, mad / center * 100.0


def _bench_balanced_pair(
    fla_fn: Callable[[], object],
    flag_attn_fn: Callable[[], object],
) -> PhaseBenchmarkResult:
    _warm_up_pair(fla_fn, flag_attn_fn)

    fla_samples: list[float] = []
    flag_attn_samples: list[float] = []

    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for cycle in range(BALANCED_MEASUREMENT_CYCLES):
            for fn in _measurement_order(
                cycle,
                fla_fn,
                flag_attn_fn,
            ):
                sample_ms = _bench_ms(fn)
                if not math.isfinite(sample_ms) or sample_ms <= 0.0:
                    raise RuntimeError(
                        f"invalid latency sample: {sample_ms}"
                    )

                if fn is fla_fn:
                    fla_samples.append(sample_ms)
                else:
                    flag_attn_samples.append(sample_ms)
    finally:
        if gc_was_enabled:
            gc.enable()

    fla_ms, fla_mad_pct = _median_and_mad_pct(
        fla_samples
    )
    flag_attn_ms, flag_attn_mad_pct = (
        _median_and_mad_pct(
            flag_attn_samples
        )
    )

    return PhaseBenchmarkResult(
        fla_ms=fla_ms,
        flag_attn_ms=flag_attn_ms,
        fla_mad_pct=fla_mad_pct,
        flag_attn_mad_pct=flag_attn_mad_pct,
    )


def _sample_tensor(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= VALIDATION_SAMPLE_COUNT:
        return flat.float()

    indices = torch.linspace(
        0,
        flat.numel() - 1,
        steps=VALIDATION_SAMPLE_COUNT,
        device=flat.device,
        dtype=torch.float64,
    ).long()
    return flat[indices].float()


def _relative_rms(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> float:
    if reference.shape != actual.shape:
        raise AssertionError(
            f"{name}: shape mismatch: {reference.shape} != {actual.shape}"
        )

    reference_sample = _sample_tensor(reference)
    actual_sample = _sample_tensor(actual)
    if not torch.isfinite(reference_sample).all():
        raise AssertionError(f"{name}: FLA result contains NaN or Inf")
    if not torch.isfinite(actual_sample).all():
        raise AssertionError(f"{name}: FlagAttention result contains NaN or Inf")

    error = (reference_sample - actual_sample).square().mean().sqrt()
    scale = reference_sample.square().mean().sqrt()
    return float((error / (scale + 1e-8)).item())


def _validate_pair(
    fla_fn: Callable[[], object],
    flag_attn_fn: Callable[[], object],
    phase: str,
    dtype: torch.dtype,
) -> float:
    """Run one untimed sampled parity check before measuring latency."""

    with torch.no_grad():
        fla_result = fla_fn()
        flag_attn_result = flag_attn_fn()
    torch.cuda.synchronize()

    if not isinstance(fla_result, tuple) or not isinstance(
        flag_attn_result,
        tuple,
    ):
        raise AssertionError("Parallax providers must return tuples")

    if phase == "fwd":
        names = ("o",)
        fla_tensors = fla_result[:1]
        flag_attn_tensors = flag_attn_result[:1]
    elif phase == "fwd_bwd":
        names = ("dq", "dr", "dk", "dv")
        fla_tensors = fla_result
        flag_attn_tensors = flag_attn_result
    else:
        raise ValueError(f"Unsupported benchmark phase: {phase}")

    if len(fla_tensors) != len(names) or len(flag_attn_tensors) != len(names):
        raise AssertionError(f"{phase}: unexpected result count")

    ratios = [
        _relative_rms(name, reference, actual)
        for name, reference, actual in zip(
            names,
            fla_tensors,
            flag_attn_tensors,
        )
    ]
    max_ratio = max(ratios)
    tolerance = ACCURACY_TOLERANCE[dtype]
    if max_ratio >= tolerance:
        details = ", ".join(
            f"{name}={ratio:.3e}"
            for name, ratio in zip(names, ratios)
        )
        raise AssertionError(
            f"{phase} sampled relative RMS error {max_ratio:.3e} "
            f"exceeded tolerance {tolerance:.3e}: {details}"
        )
    return max_ratio


def _benchmark_case(
    case: ParallaxCase,
    dtype: torch.dtype,
    phase: str,
) -> PhaseBenchmarkResult:
    (
        q,
        r,
        k,
        v,
        scale,
        cu_seqlens,
        window_size_left,
    ) = _build_inputs(
        case,
        dtype,
    )

    (
        fla_chunk_indices,
        flag_attn_chunk_indices,
    ) = _build_chunk_indices(
        case,
        q,
        cu_seqlens,
    )

    grad_output = torch.randn_like(q)

    def run_fla_fwd():
        return fla_parallel_parallax_fwd(
            q,
            r,
            k,
            v,
            scale,
            cu_seqlens,
            fla_chunk_indices,
            window_size_left,
        )

    def run_fla_fwd_bwd():
        o, barv, d1, bart, m = run_fla_fwd()

        return fla_parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            grad_output,
            scale,
            cu_seqlens,
            fla_chunk_indices,
            window_size_left,
        )

    def run_flag_attn_fwd():
        return flag_attn_parallel_parallax_fwd(
            q,
            r,
            k,
            v,
            scale,
            cu_seqlens,
            flag_attn_chunk_indices,
            window_size_left,
        )

    def run_flag_attn_fwd_bwd():
        o, barv, d1, bart, m = run_flag_attn_fwd()

        return flag_attn_parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            grad_output,
            scale,
            cu_seqlens,
            flag_attn_chunk_indices,
            window_size_left,
        )

    if phase == "fwd":
        fla_fn = run_fla_fwd
        flag_attn_fn = run_flag_attn_fwd
    elif phase == "fwd_bwd":
        fla_fn = run_fla_fwd_bwd
        flag_attn_fn = run_flag_attn_fwd_bwd
    else:
        raise ValueError(
            f"Unsupported benchmark phase: {phase}"
        )

    _validate_pair(
        fla_fn=fla_fn,
        flag_attn_fn=flag_attn_fn,
        phase=phase,
        dtype=dtype,
    )

    return _bench_balanced_pair(
        fla_fn=fla_fn,
        flag_attn_fn=flag_attn_fn,
    )


def _selected_dtypes() -> list[torch.dtype]:
    dtypes = (
        Config.user_desired_dtypes
        or DEFAULT_DTYPES
    )

    unsupported = [
        dtype
        for dtype in dtypes
        if dtype not in SUPPORTED_DTYPES
    ]

    if unsupported:
        names = ", ".join(
            str(dtype)
            for dtype in unsupported
        )
        raise ValueError(
            f"parallel_parallax does not support: {names}"
        )

    return list(dtypes)


def _print_header(
    title: str,
    fla_column: str,
    flag_attn_column: str,
) -> None:
    print()
    print("=" * TABLE_WIDTH)
    print(title)
    print("=" * TABLE_WIDTH)
    print(
        f"{'B':>3} "
        f"{'T':>7} "
        f"{'H':>4} "
        f"{'HQ':>4} "
        f"{'D':>4} "
        f"{'dtype':>9} "
        f"{fla_column:>18} "
        f"{flag_attn_column:>23} "
        f"{'speedup':>11} "
        f"{'FLA-MAD(%)':>12} "
        f"{'FlagAttention-MAD(%)':>18} "
    )

    print("-" * TABLE_WIDTH)


def _print_result(
    case: ParallaxCase,
    dtype: torch.dtype,
    result: PhaseBenchmarkResult,
) -> None:
    dtype_name = str(dtype).removeprefix(
        "torch."
    )

    print(
        f"{case.B:>3} "
        f"{case.T:>7} "
        f"{case.H:>4} "
        f"{case.HQ:>4} "
        f"{case.D:>4} "
        f"{dtype_name:>9} "
        f"{result.fla_ms:>18.6f} "
        f"{result.flag_attn_ms:>23.6f} "
        f"{result.speedup:>10.3f}x "
        f"{result.fla_mad_pct:>11.3f}% "
        f"{result.flag_attn_mad_pct:>17.3f}% "
    )

    max_mad = max(result.fla_mad_pct, result.flag_attn_mad_pct)
    is_short_latency = max(result.fla_ms, result.flag_attn_ms) < (
        SHORT_LATENCY_THRESHOLD_MS
    )
    warning_threshold = (
        SHORT_LATENCY_MAD_WARNING_PCT
        if is_short_latency
        else MAD_WARNING_THRESHOLD_PCT
    )
    if max_mad > warning_threshold:
        print(
            f"[WARN unstable] maximum MAD={max_mad:.3f}% "
            f"exceeds {warning_threshold:.1f}%"
        )


def _run_phase_table(
    phase: str,
    title: str,
    fla_column: str,
    flag_attn_column: str,
) -> None:
    _print_header(
        title=title,
        fla_column=fla_column,
        flag_attn_column=flag_attn_column,
    )

    for dtype in _selected_dtypes():
        for case in DEFAULT_CASES:
            result = _benchmark_case(
                case=case,
                dtype=dtype,
                phase=phase,
            )
            _print_result(
                case=case,
                dtype=dtype,
                result=result,
            )

    print("-" * TABLE_WIDTH)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="parallel_parallax benchmark requires CUDA",
)
def test_perf_parallel_parallax() -> None:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    iter_unit = (
        "ms/sample"
        if Config.mode.value == "kernel"
        else "calls/sample"
    )

    print(
        "\n"
        "[parallel_parallax: "
        "FLA baseline vs FlagAttention]"
    )

    print(
        f"device={torch.cuda.get_device_name()} "
        f"mode={Config.mode.value} "
        f"warmup={Config.warm_up} "
        f"iter={Config.repetition} {iter_unit}"
    )

    print(
        "measurement_order = mirrored ABBA/BAAB; "
        f"samples/provider = "
        f"{BALANCED_MEASUREMENT_CYCLES * 2}"
    )
    if Config.mode.value == "kernel":
        print(
            "timing = Triton per-invocation device timing; "
            "warmup/iter are per-sample millisecond budgets"
        )
    else:
        print(
            "timing = CUDA Event batch elapsed time / calls per sample; "
            "latency values are milliseconds"
        )
    print(
        "baseline = FLA; "
        "speedup = FLA latency / FlagAttention latency; "
        ">1 means FlagAttention is faster"
    )
    print(
        "MAD% = relative median absolute deviation; "
        "lower is more stable"
    )
    print("fwd+bwd = forward + backward")

    _run_phase_table(
        phase="fwd",
        title="[Forward]",
        fla_column="FLA-fwd(ms)",
        flag_attn_column="FlagAttention-fwd(ms)",
    )

    _run_phase_table(
        phase="fwd_bwd",
        title="[Forward + Backward]",
        fla_column="FLA-fwdbwd(ms)",
        flag_attn_column="FlagAttention-fwdbwd(ms)",
    )
