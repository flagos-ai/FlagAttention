import os
from contextlib import contextmanager

import pytest
import torch
import triton

from flag_attn import chunk_gated_delta_rule
from flag_attn.FLA.gated_delta_rule import chunk_gated_delta_rule_fwd
from flag_attn.utils import has_triton_tle

ASSERT_RATIO = 0.01
RECOMPUTE_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_RECOMPUTE_TLE"
FULL_TLE_ENV = "FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE"
TWO_KERNEL_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE"
HYBRID_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_HYBRID_TLE"

GDN_RECOMPUTE_TEST_SHAPES = [
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]

GDN_FUSED_FWD_TEST_SHAPES = [
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
]

GDN_TWO_KERNEL_TEST_SHAPES = [
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 2048, 32, 256, 256),
]

GDN_EXTERNAL_TEST_SHAPES = [
    (2, 2048, 16, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]

GDN_EXTERNAL_BENCHMARK_SHAPES = [
    (1, 8192, 96, 128, 128),
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]


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
        pytest.skip(f"external FLA GDN implementation is unavailable: {FLA_IMPORT_ERROR}")
    return FLA_CHUNK_GDN


@contextmanager
def _set_gdn_tle(
    *,
    full_tle: bool,
    recompute_tle: bool,
    two_kernel_tle: bool,
    hybrid_tle: bool = False,
):
    old_full = os.environ.get(FULL_TLE_ENV)
    old_recompute = os.environ.get(RECOMPUTE_TLE_ENV)
    old_two_kernel = os.environ.get(TWO_KERNEL_TLE_ENV)
    old_hybrid = os.environ.get(HYBRID_TLE_ENV)
    os.environ[FULL_TLE_ENV] = "1" if full_tle else "0"
    os.environ[RECOMPUTE_TLE_ENV] = "1" if recompute_tle else "0"
    os.environ[TWO_KERNEL_TLE_ENV] = "1" if two_kernel_tle else "0"
    os.environ[HYBRID_TLE_ENV] = "1" if hybrid_tle else "0"
    try:
        yield
    finally:
        if old_full is None:
            os.environ.pop(FULL_TLE_ENV, None)
        else:
            os.environ[FULL_TLE_ENV] = old_full
        if old_recompute is None:
            os.environ.pop(RECOMPUTE_TLE_ENV, None)
        else:
            os.environ[RECOMPUTE_TLE_ENV] = old_recompute
        if old_two_kernel is None:
            os.environ.pop(TWO_KERNEL_TLE_ENV, None)
        else:
            os.environ[TWO_KERNEL_TLE_ENV] = old_two_kernel
        if old_hybrid is None:
            os.environ.pop(HYBRID_TLE_ENV, None)
        else:
            os.environ[HYBRID_TLE_ENV] = old_hybrid


def _make_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    *,
    use_initial_state: bool,
):
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    initial_state = (
        torch.zeros(B, H, K, V, device=device, dtype=dtype)
        if use_initial_state
        else None
    )
    return q, k, v, g, beta, K**-0.5, initial_state, True, None


def _call_fwd(
    args,
    *,
    full_tle: bool,
    recompute_tle: bool,
    two_kernel_tle: bool = False,
    hybrid_tle: bool = False,
):
    with _set_gdn_tle(
        full_tle=full_tle,
        recompute_tle=recompute_tle,
        two_kernel_tle=two_kernel_tle,
        hybrid_tle=hybrid_tle,
    ):
        return chunk_gated_delta_rule_fwd(*args)


def _call_public_hybrid(args):
    q, k, v, g, beta, scale, _, _, _ = args
    with _set_gdn_tle(
        full_tle=True,
        recompute_tle=False,
        two_kernel_tle=True,
        hybrid_tle=True,
    ):
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


def _call_fla_reference(args):
    fla_chunk_gdn = _require_fla_reference()
    q, k, v, g, beta, scale, _, _, _ = args
    return fla_chunk_gdn(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        output_final_state=True,
    )


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    err = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    base = expected.float().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs().max().item()
    if abs_err <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    ratio = _err_ratio(expected, actual)
    assert ratio < ASSERT_RATIO, (
        f"{name} diff: abs={abs_err:.6f} ratio={ratio:.6f} " f"limit={ASSERT_RATIO}"
    )


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN recompute TLE tests require CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_RECOMPUTE_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_recompute_tle_matches_native(dtype, shape):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=True)

    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    actual = _call_fwd(args, full_tle=False, recompute_tle=True)

    _assert_close("g", actual[0], baseline[0])
    _assert_close("o", actual[1], baseline[1])
    _assert_close("A", actual[2], baseline[2])
    _assert_close("final_state", actual[3], baseline[3])


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN fused TLE tests require CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_FUSED_FWD_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_full_tle_matches_native(dtype, shape):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    actual = _call_fwd(args, full_tle=True, recompute_tle=True)

    _assert_close("g", actual[0], baseline[0])
    _assert_close("o", actual[1], baseline[1])
    _assert_close("A", actual[2], baseline[2])
    _assert_close("final_state", actual[3], baseline[3])


@pytest.mark.skipif(not _cuda_tle_available(), reason="GDN public API test requires CUDA/TLE")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_public_api_matches_native(dtype):
    torch.manual_seed(42)
    args = _make_inputs(1, 256, 2, 128, 128, dtype, use_initial_state=False)
    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    q, k, v, g, beta, scale, _, _, _ = args

    with _set_gdn_tle(
        full_tle=True,
        recompute_tle=True,
        two_kernel_tle=False,
    ):
        o, final_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            beta,
            g,
            head_first=False,
            scale=scale,
            output_final_state=True,
        )

    _assert_close("o", o, baseline[1])
    _assert_close("final_state", final_state, baseline[3])


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN two-kernel tests require CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_TWO_KERNEL_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_two_kernel_matches_native(dtype, shape):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    actual = _call_fwd(
        args,
        full_tle=False,
        recompute_tle=False,
        two_kernel_tle=True,
    )

    _assert_close("g", actual[0], baseline[0])
    _assert_close("o", actual[1], baseline[1])
    _assert_close("A", actual[2], baseline[2])
    _assert_close("final_state", actual[3], baseline[3])


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN hybrid tests require CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_hybrid_matches_two_kernel(dtype):
    torch.manual_seed(42)
    args = _make_inputs(4, 2048, 16, 128, 128, dtype, use_initial_state=False)

    expected = _call_fwd(
        args,
        full_tle=False,
        recompute_tle=False,
        two_kernel_tle=True,
    )
    actual = _call_fwd(
        args,
        full_tle=True,
        recompute_tle=False,
        two_kernel_tle=True,
        hybrid_tle=True,
    )

    _assert_close("g", actual[0], expected[0])
    _assert_close("o", actual[1], expected[1])
    _assert_close("A", actual[2], expected[2])
    _assert_close("final_state", actual[3], expected[3])


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN external comparison requires CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_EXTERNAL_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_hybrid_matches_fla(dtype, shape):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

    expected_o, expected_final_state = _call_fla_reference(args)
    actual_o, actual_final_state = _call_public_hybrid(args)

    _assert_close("o", actual_o, expected_o)
    _assert_close("final_state", actual_final_state, expected_final_state)


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN external benchmark requires CUDA/TLE"
)
@pytest.mark.skipif(
    os.environ.get("FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS", "0") != "1",
    reason="set FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1 to run GDN benchmarks",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_EXTERNAL_BENCHMARK_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_hybrid_benchmark(dtype, shape, record_property):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

    expected_o, expected_final_state = _call_fla_reference(args)
    actual_o, actual_final_state = _call_public_hybrid(args)
    _assert_close("o", actual_o, expected_o)
    _assert_close("final_state", actual_final_state, expected_final_state)

    fla_ms = triton.testing.do_bench(
        lambda: _call_fla_reference(args), warmup=10, rep=50
    )
    flag_attn_ms = triton.testing.do_bench(
        lambda: _call_public_hybrid(args), warmup=10, rep=50
    )
    speedup = fla_ms / flag_attn_ms
    shape_name = f"B{shape[0]}_T{shape[1]}_H{shape[2]}_K{shape[3]}_V{shape[4]}"

    record_property("shape", shape_name)
    record_property("dtype", str(dtype))
    record_property("fla_ms", fla_ms)
    record_property("flag_attn_ms", flag_attn_ms)
    record_property("speedup_vs_fla", speedup)
    print(
        f"\n{dtype} {shape_name}: FLA={fla_ms:.6f} ms, "
        f"FlagAttention={flag_attn_ms:.6f} ms, speedup={speedup:.3f}x"
    )


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN public two-kernel test requires CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_public_api_uses_two_kernel(dtype):
    torch.manual_seed(42)
    args = _make_inputs(4, 2048, 16, 128, 128, dtype, use_initial_state=False)
    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    q, k, v, g, beta, scale, _, _, _ = args

    with _set_gdn_tle(
        full_tle=False,
        recompute_tle=False,
        two_kernel_tle=True,
    ):
        o, final_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            beta,
            g,
            head_first=False,
            scale=scale,
            output_final_state=True,
        )

    _assert_close("o", o, baseline[1])
    _assert_close("final_state", final_state, baseline[3])
