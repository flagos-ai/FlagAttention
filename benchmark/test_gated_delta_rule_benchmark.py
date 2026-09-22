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

"""Gated delta rule speedup benchmark against the FLA implementation."""

import os
from contextlib import contextmanager
from typing import Callable

import pytest
import torch
import triton

from flag_attn import chunk_gated_delta_rule
from flag_attn.utils import has_triton_tle


ASSERT_RATIO = 0.01
WARMUP = 10
REPETITIONS = 50
FULL_TLE_ENV = "FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE"
RECOMPUTE_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_RECOMPUTE_TLE"
TWO_KERNEL_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE"
HYBRID_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_HYBRID_TLE"

SHAPES = [
    (1, 8192, 96, 128, 128),
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]
DTYPES = (torch.float16, torch.bfloat16)


def _load_fla_reference():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdn

        return fla_chunk_gdn, None
    except Exception as exc:
        return None, str(exc)


FLA_CHUNK_GDN, FLA_IMPORT_ERROR = _load_fla_reference()


def _cuda_tle_available() -> bool:
    return torch.cuda.is_available() and has_triton_tle(3, 6, 0)


def _require_fla_reference():
    if FLA_CHUNK_GDN is None:
        raise RuntimeError(
            f"external FLA GDN implementation is unavailable: {FLA_IMPORT_ERROR}"
        )
    return FLA_CHUNK_GDN


@contextmanager
def _use_optimized_tle():
    names = (
        FULL_TLE_ENV,
        RECOMPUTE_TLE_ENV,
        TWO_KERNEL_TLE_ENV,
        HYBRID_TLE_ENV,
    )
    old_values = {name: os.environ.get(name) for name in names}
    os.environ[FULL_TLE_ENV] = "1"
    os.environ[RECOMPUTE_TLE_ENV] = "0"
    os.environ[TWO_KERNEL_TLE_ENV] = "1"
    os.environ[HYBRID_TLE_ENV] = "1"
    try:
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _make_inputs(shape: tuple[int, int, int, int, int], dtype: torch.dtype):
    B, T, H, K, V = shape
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = (-torch.rand(B, T, H, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
    return q, k, v, g, beta, K**-0.5


def _call_fla_reference(inputs):
    fla_chunk_gdn = _require_fla_reference()
    q, k, v, g, beta, scale = inputs
    return fla_chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        output_final_state=True,
    )


def _call_flag_attn(inputs):
    q, k, v, g, beta, scale = inputs
    return chunk_gated_delta_rule(
        q,
        k,
        v,
        beta,
        g,
        head_first=False,
        scale=scale,
        output_final_state=True,
    )


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    error = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    baseline = expected.float().flatten().square().mean().sqrt().item()
    return error / (baseline + 1e-8)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    absolute_error = (actual - expected).abs().max().item()
    if absolute_error <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    ratio = _err_ratio(expected, actual)
    assert ratio < ASSERT_RATIO, (
        f"{name} diff: abs={absolute_error:.6f} ratio={ratio:.6f} "
        f"limit={ASSERT_RATIO}"
    )


@torch.inference_mode()
def _benchmark_case(dtype: torch.dtype, shape: tuple[int, int, int, int, int]):
    torch.manual_seed(42)
    inputs = _make_inputs(shape, dtype)

    expected_o, expected_final_state = _call_fla_reference(inputs)
    fla_ms = triton.testing.do_bench(
        lambda: _call_fla_reference(inputs),
        warmup=WARMUP,
        rep=REPETITIONS,
        return_mode="median",
    )
    with _use_optimized_tle():
        actual_o, actual_final_state = _call_flag_attn(inputs)
        _assert_close("o", actual_o, expected_o)
        _assert_close("final_state", actual_final_state, expected_final_state)
        flag_attn_ms = triton.testing.do_bench(
            lambda: _call_flag_attn(inputs),
            warmup=WARMUP,
            rep=REPETITIONS,
            return_mode="median",
        )
    return fla_ms, flag_attn_ms


def _shape_name(shape: tuple[int, int, int, int, int]) -> str:
    return f"B{shape[0]}_T{shape[1]}_H{shape[2]}_K{shape[3]}_V{shape[4]}"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _speedup(fla_ms: float, flag_attn_ms: float) -> float:
    return fla_ms / flag_attn_ms if flag_attn_ms > 0 else float("inf")


def _print_header() -> None:
    print(f"\n{'=' * 92}")
    print("  chunk_gated_delta_rule benchmark — FWD ONLY")
    print(
        "  provider: fla_chunk_gated_delta_rule vs "
        "flag_attn_chunk_gated_delta_rule"
    )
    print(f"  warmup={WARMUP}ms  rep={REPETITIONS}ms")
    print(f"{'=' * 92}")
    print(
        f"{'B':>3} {'T':>6} {'H':>4} {'K':>4} {'V':>4} {'dtype':>8} "
        f"{'fla(ms)':>10} {'flag_attn(ms)':>14} {'fla/flag_attn':>14}"
    )


def _print_row(
    dtype: torch.dtype,
    shape: tuple[int, int, int, int, int],
    fla_ms: float,
    flag_attn_ms: float,
) -> None:
    B, T, H, K, V = shape
    speedup = _speedup(fla_ms, flag_attn_ms)
    print(
        f"{B:>3} {T:>6} {H:>4} {K:>4} {V:>4} {_dtype_name(dtype):>8} "
        f"{fla_ms:>10.3f} {flag_attn_ms:>14.3f} {speedup:>14.2f}x"
    )


def _print_footer() -> None:
    print(f"\n{'=' * 92}")
    print("  All done.")
    print(f"{'=' * 92}\n")


def _record_result(
    record_property: Callable[[str, object], None] | None,
    dtype: torch.dtype,
    shape: tuple[int, int, int, int, int],
    fla_ms: float,
    flag_attn_ms: float,
) -> None:
    if record_property is None:
        return
    prefix = f"{_shape_name(shape)}_{_dtype_name(dtype)}"
    record_property(f"{prefix}.fla_ms", fla_ms)
    record_property(f"{prefix}.flagattention_ms", flag_attn_ms)
    record_property(f"{prefix}.speedup_vs_fla", _speedup(fla_ms, flag_attn_ms))


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN external benchmark requires CUDA/TLE"
)
@pytest.mark.skipif(
    FLA_CHUNK_GDN is None,
    reason=f"external FLA GDN implementation is unavailable: {FLA_IMPORT_ERROR}",
)
def test_chunk_gated_delta_rule_benchmark(
    record_property: Callable[[str, object], None],
) -> None:
    run_benchmark(record_property)


def run_benchmark(
    record_property: Callable[[str, object], None] | None = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GDN benchmark requires CUDA")
    if not has_triton_tle(3, 6, 0):
        raise RuntimeError("GDN benchmark requires a compatible Triton TLE build")
    _require_fla_reference()

    if record_property is not None:
        record_property("scope", "operator_only")
        record_property("baseline", "FLA")

    _print_header()
    for dtype in DTYPES:
        print("\ndtype:", dtype)
        for shape in SHAPES:
            fla_ms, flag_attn_ms = _benchmark_case(dtype, shape)
            _print_row(dtype, shape, fla_ms, flag_attn_ms)
            _record_result(
                record_property,
                dtype,
                shape,
                fla_ms,
                flag_attn_ms,
            )
            torch.cuda.empty_cache()
    _print_footer()


if __name__ == "__main__":
    run_benchmark()
