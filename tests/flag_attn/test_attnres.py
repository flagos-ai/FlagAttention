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

import importlib
import os
import statistics

import pytest
import torch
import triton

import flag_attn


ATTNRES_EXTERNAL_BENCHMARK_SHAPES = [
    (2, 1),
    (5, 128),
    (5, 1024),
    (9, 128),
    (9, 8192),
]
ATTNRES_BENCHMARK_ROUNDS = 7
ATTNRES_BENCHMARK_WARMUP_MS = 1000
ATTNRES_BENCHMARK_REP_MS = 100
FLA_BASELINE_COMMIT = "5aea42b7740f9968f6418c6c60b78ea785ce6140"
FLAGTREE_COMMIT = "aaa420f9366440e18bd83a0391c2d2dbcc5e0b81"


def _load_fla_reference():
    try:
        fla_attnres = importlib.import_module("fla.ops.attnres.fused")

        return fla_attnres.fused_attnres, fla_attnres, None
    except Exception as exc:
        return None, None, str(exc)


FLA_FUSED_ATTNRES, FLA_ATTNRES_MODULE, FLA_IMPORT_ERROR = _load_fla_reference()
FLAG_ATTNRES_MODULE = importlib.import_module("flag_attn.FLA.attnres.fused")


def _tle_available() -> bool:
    try:
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


def _require_fla_reference():
    if FLA_FUSED_ATTNRES is None:
        pytest.skip(f"external FLA AttnRes implementation is unavailable: {FLA_IMPORT_ERROR}")
    return FLA_FUSED_ATTNRES


def _bench(fn):
    return triton.testing.do_bench(
        fn,
        warmup=ATTNRES_BENCHMARK_WARMUP_MS,
        rep=ATTNRES_BENCHMARK_REP_MS,
        return_mode="median",
    )


def test_fused_attnres_public_export():
    from flag_attn.FLA.attnres import fused_attnres

    assert flag_attn.fused_attnres is fused_attnres
    assert fused_attnres is FLAG_ATTNRES_MODULE.fused_attnres
    assert "fused_attnres" in flag_attn.__all__


@pytest.mark.parametrize(
    ("num_sources", "batch", "tokens", "hidden_size", "output_norm", "dtype"),
    [
        (1, 1, 3, 7168, False, torch.bfloat16),
        (5, 1, 3, 7168, False, torch.bfloat16),
        (9, 1, 3, 7168, True, torch.bfloat16),
        (5, 2, 7, 4096, True, torch.float16),
    ],
)
@torch.inference_mode()
def test_fused_attnres(
    num_sources: int,
    batch: int,
    tokens: int,
    hidden_size: int,
    output_norm: bool,
    dtype: torch.dtype,
):
    if not _tle_available():
        pytest.skip("fused_attnres requires CUDA and triton.experimental.tle")

    torch.manual_seed(42)
    residuals = [
        torch.randn(batch, tokens, hidden_size, device="cuda", dtype=dtype)
        for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device="cuda", dtype=dtype)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_rms_weight = (
        torch.randn(hidden_size, device="cuda", dtype=dtype) if output_norm else None
    )

    expected, expected_weights = flag_attn.testing.fused_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        scale=hidden_size**-0.5,
        return_weights=True,
    )
    actual, actual_weights = flag_attn.fused_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        scale=hidden_size**-0.5,
        return_weights=True,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_weights, expected_weights, atol=5e-5, rtol=5e-5)


@torch.inference_mode()
def test_fused_attnres_without_weights():
    if not _tle_available():
        pytest.skip("fused_attnres requires CUDA and triton.experimental.tle")

    hidden_size = 7168
    residuals = [
        torch.randn(2, hidden_size, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    ]
    query = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)

    output = flag_attn.fused_attnres(query, residuals, rms_weight, scale=hidden_size**-0.5)
    assert isinstance(output, torch.Tensor)
    assert output.shape == residuals[0].shape


def test_fused_attnres_rejects_empty_residuals():
    query = torch.empty(128)
    rms_weight = torch.empty(128)
    with pytest.raises(ValueError, match="at least one"):
        flag_attn.fused_attnres(query, [], rms_weight)


@pytest.mark.skipif(
    not _tle_available(), reason="AttnRes external benchmark requires CUDA/TLE"
)
@pytest.mark.skipif(
    os.environ.get("FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS", "0") != "1",
    reason="set FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1 to run AttnRes benchmarks",
)
@pytest.mark.parametrize("output_norm", [False, True])
@pytest.mark.parametrize(("num_sources", "num_rows"), ATTNRES_EXTERNAL_BENCHMARK_SHAPES)
@torch.inference_mode()
def test_fused_attnres_kernel_benchmark(
    num_sources, num_rows, output_norm, record_property
):
    fla_fused_attnres = _require_fla_reference()
    torch.manual_seed(0)
    hidden_size = 7168
    dtype = torch.bfloat16
    residuals = [
        torch.randn(num_rows, hidden_size, device="cuda", dtype=dtype)
        for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device="cuda", dtype=dtype)
    rms_weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    output_rms_weight = (
        torch.randn(hidden_size, device="cuda", dtype=dtype) if output_norm else None
    )
    scale = hidden_size**-0.5

    expected = fla_fused_attnres(
        query, residuals, rms_weight, output_rms_weight, scale=scale
    )
    actual = flag_attn.fused_attnres(
        query, residuals, rms_weight, output_rms_weight, scale=scale
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=1e-2)

    # Prebuild pointer structures and allocate every output before timing. The
    # benchmark below measures the existing forward kernels, excluding Python
    # wrappers, pointer-table construction, tensor allocation, and postprocessing.
    fla_ptrs = FLA_ATTNRES_MODULE._build_ptr_table(residuals)
    fla_output = torch.empty_like(residuals[0])
    stats_shape = (num_sources, num_rows)
    fla_probability = torch.empty(stats_shape, device="cuda", dtype=torch.float32)
    fla_rstd = torch.empty_like(fla_probability)
    fla_score_mean = torch.empty_like(fla_probability)
    fla_output_rstd = (
        torch.empty((num_rows,), device="cuda", dtype=torch.float32)
        if output_norm
        else None
    )
    fla_block_l = max(8, triton.next_power_of_2(num_sources))
    fla_dtype = FLA_ATTNRES_MODULE._TORCH_TO_TL_DTYPE[dtype]

    flat_residuals = tuple(
        residual.reshape(-1, hidden_size) for residual in residuals
    )
    flag_ptrs = FLAG_ATTNRES_MODULE._padded_residual_tuple(flat_residuals)
    flag_output = torch.empty_like(residuals[0])

    def run_fla_kernel():
        FLA_ATTNRES_MODULE.attnres_fwd_kernel[(num_rows,)](
            q=query,
            res=fla_ptrs,
            w=rms_weight,
            o=fla_output,
            p=fla_probability,
            rstd=fla_rstd,
            score_mean=fla_score_mean,
            ow=output_rms_weight,
            o_rstd=fla_output_rstd,
            N=num_rows,
            L=num_sources,
            D=hidden_size,
            eps=1e-6,
            scale=scale,
            BL=fla_block_l,
            HAS_ONORM=output_norm,
            DTYPE=fla_dtype,
        )

    def run_flag_kernel():
        FLAG_ATTNRES_MODULE._fused_attnres_fwd_kernel[(num_rows,)](
            query=query,
            residuals=flag_ptrs,
            rms_weight=rms_weight,
            output_rms_weight=output_rms_weight,
            output=flag_output,
            logit=None,
            lse=None,
            N=num_rows,
            L=num_sources,
            L2=len(flag_ptrs),
            D=hidden_size,
            N_BUCKET=(0 if num_rows <= 128 else (1 if num_rows <= 1024 else 2)),
            eps=1e-6,
            scale=scale,
            BD=triton.next_power_of_2(hidden_size),
            HAS_ONORM=output_norm,
            RETURN_WEIGHTS=False,
            ASYNC_LOAD=num_sources <= 2 and not output_norm,
        )

    run_fla_kernel()
    run_flag_kernel()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        flag_output.float(), fla_output.float(), atol=5e-3, rtol=1e-2
    )

    fla_rounds = []
    flag_attn_rounds = []
    for round_index in range(ATTNRES_BENCHMARK_ROUNDS):
        if round_index % 2 == 0:
            fla_rounds.append(_bench(run_fla_kernel))
            flag_attn_rounds.append(_bench(run_flag_kernel))
        else:
            flag_attn_rounds.append(_bench(run_flag_kernel))
            fla_rounds.append(_bench(run_fla_kernel))

    fla_ms = statistics.median(fla_rounds)
    flag_attn_ms = statistics.median(flag_attn_rounds)
    speedup = fla_ms / flag_attn_ms
    shape_name = f"L{num_sources}_N{num_rows}_D{hidden_size}"

    record_property("benchmark_scope", "preallocated_forward_kernel")
    record_property("fla_baseline_commit", FLA_BASELINE_COMMIT)
    record_property("flagtree_commit", FLAGTREE_COMMIT)
    record_property("shape", shape_name)
    record_property("dtype", str(dtype))
    record_property("output_norm", output_norm)
    record_property("fla_ms", fla_ms)
    record_property("flag_attn_ms", flag_attn_ms)
    record_property("speedup_vs_fla", speedup)
    record_property("fla_rounds_ms", ",".join(map(str, fla_rounds)))
    record_property("flag_attn_rounds_ms", ",".join(map(str, flag_attn_rounds)))
    print(
        f"\n{shape_name} output_norm={output_norm}: FLA kernel={fla_ms:.6f} ms, "
        f"FlagAttention kernel={flag_attn_ms:.6f} ms, speedup={speedup:.3f}x, "
        f"FLA rounds={fla_rounds}, FlagAttention rounds={flag_attn_rounds}"
    )
