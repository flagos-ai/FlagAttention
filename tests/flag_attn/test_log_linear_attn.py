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

import importlib.util
import math
import os
from pathlib import Path
import statistics

import pytest
import torch
import triton

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_source_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TLE_MODULE = _load_source_module(
    "flag_attn_log_linear_attn_chunk_tle",
    "src/flag_attn/FLA/log_linear_attn/chunk_tle.py",
)
REFERENCE_MODULE = _load_source_module(
    "flag_attn_log_linear_attn_reference",
    "src/flag_attn/testing/log_linear_attn.py",
)
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


def _tle_available():
    return torch.cuda.is_available() and TLE_MODULE.HAS_TLE_LOG_LINEAR_ATTN


def _make_inputs(batch, sequence, heads, dim):
    levels = math.ceil(math.log2(sequence)) + 1
    q = torch.randn(batch, sequence, 1, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, heads, dim, device="cuda", dtype=torch.bfloat16)
    g = -0.05 * torch.rand(batch, sequence, heads, device="cuda", dtype=torch.float32)
    level_scales = torch.sigmoid(
        torch.randn(
            batch,
            sequence,
            heads,
            levels,
            device="cuda",
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    return q, k, v, g, level_scales


def _assert_close(name, expected, actual, ratio=0.004):
    error = (expected.float() - actual.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt()
    error_ratio = (error / (reference + 1e-8)).item()
    assert not torch.isnan(actual).any(), f"{name}: NaN detected"
    assert error_ratio < ratio, f"{name}: RMS error ratio {error_ratio:.6f}"


def _bench(fn):
    return triton.testing.do_bench(
        fn,
        warmup=BENCHMARK_WARMUP_MS,
        rep=BENCHMARK_REP_MS,
        return_mode="median",
    )


def test_chunk_log_linear_attn_public_export():
    root_init = (REPO_ROOT / "src/flag_attn/__init__.py").read_text()
    operator_init = (
        REPO_ROOT / "src/flag_attn/FLA/log_linear_attn/__init__.py"
    ).read_text()
    assert '"chunk_log_linear_attn"' in root_init
    assert "flag_attn.FLA.log_linear_attn" in root_init
    assert "from flag_attn.FLA.log_linear_attn.chunk_tle import" in operator_init


@pytest.mark.skipif(not _tle_available(), reason="requires CUDA and FlagTree/TLE")
@pytest.mark.parametrize(
    ("batch", "sequence", "heads", "dim"),
    [
        (1, 64, 2, 64),
        (1, 128, 2, 128),
        (1, 256, 2, 256),
    ],
)
@torch.inference_mode()
def test_chunk_log_linear_attn_matches_reference(batch, sequence, heads, dim):
    torch.manual_seed(42)
    inputs = _make_inputs(batch, sequence, heads, dim)
    expected = REFERENCE_MODULE.log_linear_attn_reference(*inputs)
    actual = TLE_MODULE.chunk_log_linear_attn(*inputs)
    _assert_close("output", expected, actual)


@pytest.mark.skipif(not _tle_available(), reason="requires CUDA and FlagTree/TLE")
@pytest.mark.skipif(
    os.environ.get("FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS", "0") != "1",
    reason="set FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1 to run TileLang benchmarks",
)
@pytest.mark.parametrize(("batch", "sequence", "heads", "dim"), BENCHMARK_SHAPES)
@torch.inference_mode()
def test_chunk_log_linear_attn_tilelang_benchmark(
    batch, sequence, heads, dim, record_property
):
    pytest.importorskip("tilelang")
    from baselines.log_linear_attn_tilelang import _prepare_tilelang_forward

    torch.manual_seed(0)
    inputs = _make_inputs(batch, sequence, heads, dim)
    tile_kernel, tile_args, tile_output = _prepare_tilelang_forward(*inputs)
    run_tilelang = lambda: tile_kernel(*tile_args)
    run_tle, tle_output = TLE_MODULE._prepare_tle_forward(*inputs)

    run_tilelang()
    run_tle()
    torch.cuda.synchronize()
    _assert_close("TileLang/TLE", tile_output, tle_output)

    tilelang_rounds = []
    tle_rounds = []
    for round_index in range(BENCHMARK_ROUNDS):
        if round_index % 2 == 0:
            tilelang_rounds.append(_bench(run_tilelang))
            tle_rounds.append(_bench(run_tle))
        else:
            tle_rounds.append(_bench(run_tle))
            tilelang_rounds.append(_bench(run_tilelang))

    tilelang_ms = statistics.median(tilelang_rounds)
    tle_ms = statistics.median(tle_rounds)
    speedup = tilelang_ms / tle_ms
    shape = f"B{batch}_T{sequence}_H{heads}_D{dim}"
    record_property("shape", shape)
    record_property("tilelang_ms", tilelang_ms)
    record_property("tle_ms", tle_ms)
    record_property("speedup_vs_tilelang", speedup)
    print(
        f"\n{shape}: TileLang={tilelang_ms:.6f} ms, "
        f"TLE={tle_ms:.6f} ms, speedup={speedup:.3f}x"
    )
