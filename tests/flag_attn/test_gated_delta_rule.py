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

"""GDN forward contracts for chunk prefill, recurrent and packed decode."""

import os
from contextlib import contextmanager

import pytest
import torch
import triton

from flag_attn import chunk_gated_delta_rule
from flag_attn.FLA.gated_delta_rule import (
    chunk_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_fwd,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from flag_attn.testing import backend as test_backend
from flag_attn.utils import has_triton_tle
from gdn_utils import check_close, make_case, make_packed_case, reference, reference_packed, unpack_reference_case


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


def _load_fla_reference():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdn

        return fla_chunk_gdn, None
    except Exception as exc:
        return None, str(exc)


FLA_CHUNK_GDN, FLA_IMPORT_ERROR = _load_fla_reference()


def _cuda_tle_available() -> bool:
    return test_backend.is_nvidia() and has_triton_tle(3, 6, 0)


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
    old_fused = os.environ.get("FLAG_ATTN_CHUNK_GDR_FUSED_TLE")
    os.environ["FLAG_ATTN_CHUNK_GDR_FUSED_TLE"] = "1" if full_tle else "0"
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
        if old_fused is None:
            os.environ.pop("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", None)
        else:
            os.environ["FLAG_ATTN_CHUNK_GDR_FUSED_TLE"] = old_fused
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
    device = torch.device(test_backend.device)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype) if use_initial_state else None
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
    assert ratio < ASSERT_RATIO, f"{name} diff: abs={abs_err:.6f} ratio={ratio:.6f} limit={ASSERT_RATIO}"


def _two_kernel_configs():
    if not _cuda_tle_available():
        return [None]
    from flag_attn.FLA.gated_delta_rule.chunk_fused_forward import (
        _chunk_gdn_two_kernel_fwd_kernel,
    )

    return [
        pytest.param(
            config,
            id=f"BV{config.kwargs['BV']}-warps{config.num_warps}-stages{config.num_stages}",
        )
        for config in _chunk_gdn_two_kernel_fwd_kernel.configs
    ]


def rms_close(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    assert error / (expected.float().square().mean().sqrt() + 1e-8) < 0.01


@pytest.mark.chunk_gated_delta_rule
class TestChunkGatedDeltaRule:
    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN recompute TLE tests require CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", GDN_RECOMPUTE_TEST_SHAPES)
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_fwd_recompute_tle_matches_native(self, dtype, shape):
        torch.manual_seed(42)
        args = _make_inputs(*shape, dtype=dtype, use_initial_state=True)

        baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
        actual = _call_fwd(args, full_tle=False, recompute_tle=True)

        _assert_close("g", actual[0], baseline[0])
        _assert_close("o", actual[1], baseline[1])
        _assert_close("A", actual[2], baseline[2])
        _assert_close("final_state", actual[3], baseline[3])

    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN fused TLE tests require CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", GDN_FUSED_FWD_TEST_SHAPES)
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_fwd_full_tle_matches_native(self, dtype, shape):
        torch.manual_seed(42)
        args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

        baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
        actual = _call_fwd(args, full_tle=True, recompute_tle=True)

        _assert_close("g", actual[0], baseline[0])
        _assert_close("o", actual[1], baseline[1])
        _assert_close("A", actual[2], baseline[2])
        _assert_close("final_state", actual[3], baseline[3])

    @pytest.mark.skipif(
        not test_backend.supports_operator("chunk_gated_delta_rule"), reason="requires an available accelerator"
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_public_api_matches_native(self, dtype):
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

    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN two-kernel tests require CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", GDN_TWO_KERNEL_TEST_SHAPES)
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_fwd_two_kernel_matches_native(self, dtype, shape, monkeypatch):
        import flag_attn.FLA.gated_delta_rule.chunk as implementation

        calls = []
        original = implementation.chunk_gdn_two_kernel_fwd

        def tracked(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(implementation, "chunk_gdn_two_kernel_fwd", tracked)
        torch.manual_seed(42)
        args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

        baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
        actual = _call_fwd(
            args,
            full_tle=False,
            recompute_tle=False,
            two_kernel_tle=True,
        )

        wide_hopper = shape[3] > 128 and torch.cuda.get_device_capability()[0] == 9
        assert len(calls) == (0 if wide_hopper else 1)
        _assert_close("g", actual[0], baseline[0])
        _assert_close("o", actual[1], baseline[1])
        _assert_close("A", actual[2], baseline[2])
        _assert_close("final_state", actual[3], baseline[3])

    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN autotune configurations require CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("config", _two_kernel_configs())
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_two_kernel_configs(self, dtype, config):
        from flag_attn.FLA.gated_delta_rule.chunk_fused_forward import (
            _chunk_gdn_two_kernel_fwd_kernel,
        )

        torch.manual_seed(42)
        args = _make_inputs(4, 2048, 16, 128, 128, dtype, use_initial_state=False)
        baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
        q, k, v, _, beta, scale, _, _, _ = args
        output = torch.empty_like(v)
        final_state = torch.empty_like(baseline[3])
        # Exercise every candidate: autotuning measures speed without validating
        # the recurrent state, so checking only the winner can hide bad configs.
        _chunk_gdn_two_kernel_fwd_kernel.fn[(triton.cdiv(128, config.kwargs["BV"]), 64)](
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=baseline[2],
            g=baseline[0],
            o=output,
            final_state=final_state,
            scale=scale,
            T=2048,
            H=16,
            Hg=16,
            K=128,
            V=128,
            BT=64,
            **config.all_kwargs(),
        )
        _assert_close("o", output, baseline[1])
        _assert_close("final_state", final_state, baseline[3])

    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN hybrid tests require CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_fwd_hybrid_matches_two_kernel(self, dtype):
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

    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN external comparison requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", GDN_EXTERNAL_TEST_SHAPES)
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_hybrid_matches_fla(self, dtype, shape):
        torch.manual_seed(42)
        args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

        expected_o, expected_final_state = _call_fla_reference(args)
        actual_o, actual_final_state = _call_public_hybrid(args)

        _assert_close("o", actual_o, expected_o)
        _assert_close("final_state", actual_final_state, expected_final_state)

    @pytest.mark.skipif(not _cuda_tle_available(), reason="GDN public two-kernel test requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @torch.inference_mode()
    def test_chunk_gated_delta_rule_public_api_uses_two_kernel(self, dtype):
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

    @pytest.mark.skipif(not torch.cuda.is_available() or not has_triton_tle(3, 6, 0), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(2, 16384, 16), (4, 2048, 16), (4, 4096, 64), (2, 2048, 16)])
    @pytest.mark.parametrize("seed", [42, 123])
    @torch.inference_mode()
    def test_fused_tle_matches_fla(self, monkeypatch, dtype, shape, seed):
        import triton.experimental.tle.language as tle

        if not hasattr(tle, "pipe") or torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("requires Hopper and TLE pipes")
        reference = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
        import flag_attn.FLA.gated_delta_rule.chunk_fused_tle as fused

        original_fwd = fused._tle_chunk_gated_delta_rule_fwd
        calls = []

        def tracked_fwd(*args, **kwargs):
            calls.append(True)
            return original_fwd(*args, **kwargs)

        monkeypatch.setattr(fused, "_tle_chunk_gated_delta_rule_fwd", tracked_fwd)

        B, T, H = shape
        torch.manual_seed(seed)
        q = torch.randn(B, T, H, 128, device="cuda", dtype=dtype) / 128**0.5
        k = torch.randn_like(q) / 128**0.5
        v = torch.randn_like(q)
        g = (-torch.rand(B, T, H, device="cuda") * 0.1).to(dtype)
        beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "1")
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE", "1")
        expected = reference(q, k, v, g, beta, output_final_state=True)
        previous = None
        for _ in range(5):
            actual = chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, output_final_state=True)
            for a, b in zip(actual, expected):
                assert torch.isfinite(a).all()
                ratio = (a.float() - b.float()).square().mean().sqrt() / (b.float().square().mean().sqrt() + 1e-8)
                assert ratio < 0.01, ratio.item()
            if previous is not None:
                for a, b in zip(actual, previous):
                    assert torch.equal(a, b), "nondeterministic pipeline output"
            previous = actual
        assert len(calls) == 5, "regression cases must exercise the fused TLE kernel"

    @pytest.mark.skipif(not torch.cuda.is_available() or not has_triton_tle(3, 6, 0), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "case",
        [
            "gva",
            "head_first",
            "normalize",
            "zero_scale",
            "zero_gate",
            "strong_gate",
            "initial_state",
            "no_final_state",
            "tail",
            "fp32_gate",
        ],
    )
    @torch.inference_mode()
    def test_fused_tle_api_cases(self, monkeypatch, dtype, case):
        reference = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
        torch.manual_seed(7)
        B, T, H, Hg = 2, 2048, 16, 4 if case == "gva" else 16
        if case == "tail":
            T += 7
        q = torch.randn(B, T, Hg, 128, device="cuda", dtype=dtype) / 128**0.5
        k = torch.randn_like(q) / 128**0.5
        v = torch.randn(B, T, H, 128, device="cuda", dtype=dtype)
        g = (-torch.rand(B, T, H, device="cuda") * 0.1).to(dtype)
        beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
        if case == "zero_gate":
            g.zero_()
        if case == "strong_gate":
            g *= 20
        if case == "fp32_gate":
            g = g.float()
        h0 = torch.randn(B, H, 128, 128, device="cuda", dtype=dtype) * 0.1 if case == "initial_state" else None
        kw = dict(
            scale=0.0 if case == "zero_scale" else 128**-0.5,
            initial_state=h0,
            output_final_state=case != "no_final_state",
            use_qk_l2norm_in_kernel=case == "normalize",
        )
        expected = reference(q, k, v, g, beta, **kw)
        if case == "head_first":
            q, k, v, beta, g = [x.transpose(1, 2) for x in (q, k, v, beta, g)]
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "1")
        if case == "fp32_gate":
            with pytest.raises(ValueError, match="g must have the same dtype as q"):
                chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, **kw)
            return
        actual = chunk_gated_delta_rule(q, k, v, beta, g, head_first=case == "head_first", **kw)
        for index, (a, b) in enumerate(zip(actual, expected)):
            if b is None:
                assert a is None
                continue
            if case == "head_first" and index == 0:
                a = a.transpose(1, 2)
            assert torch.isfinite(a).all()
            ratio = (a.float() - b.float()).square().mean().sqrt() / (b.float().square().mean().sqrt() + 1e-8)
            assert ratio < 0.01, ratio.item()

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("fp32_gates", [False, True])
    @torch.inference_mode()
    def test_raw_chunk_fused_contract(self, dtype, fp32_gates, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("fused chunk path requires Hopper")
        import flag_attn.FLA.gated_delta_rule.chunk_fused_tle as module

        calls = []
        original = module._tle_chunk_gated_delta_rule_fwd

        def tracked(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        torch.manual_seed(93)
        q = torch.randn((2, 2048, 4, 128), device="cuda", dtype=dtype) / 128**0.5
        k = torch.randn_like(q) / 128**0.5
        v = torch.randn((2, 2048, 16, 128), device="cuda", dtype=dtype)
        g = -torch.rand((2, 2048, 16), device="cuda", dtype=dtype) * 0.1
        if fp32_gates:
            g = g.float()
        beta = torch.rand_like(g)
        args = dict(q=q, k=k, v=v, g=g, beta=beta, scale=128**-0.5, initial_state=None, output_final_state=True)
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE", "0")
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE", "0")
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "0")
        expected = chunk_gated_delta_rule_fwd(**args)
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "1")
        monkeypatch.setattr(module, "_tle_chunk_gated_delta_rule_fwd", tracked)
        actual = chunk_gated_delta_rule_fwd(**args)
        assert len(calls) == 1 and len(actual) == len(expected) == 7
        assert actual[0].dtype == expected[0].dtype == g.dtype
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                rms_close(a, b)
        # Complete initialization of the returned triangular auxiliary tensor.
        a = actual[2].view(2, 32, 64, 16, 64).permute(0, 1, 3, 2, 4)
        assert torch.count_nonzero(a.triu(1)) == 0

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
    @torch.inference_mode()
    def test_prefill_packed_and_speculative_state_handoff(self, dtype, state_dtype):
        torch.manual_seed(29)
        n, h, hv, dk = 2, 4, 16, 128
        q = torch.randn((n, 2048, h, dk), device="cuda", dtype=dtype) / dk**0.5
        k = torch.randn_like(q) / dk**0.5
        v = torch.randn((n, 2048, hv, dk), device="cuda", dtype=dtype)
        g = -torch.rand((n, 2048, hv), device="cuda", dtype=dtype) * 0.1
        beta = torch.rand_like(g)
        _, final = chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, output_final_state=True)
        packed = make_packed_case(n, h, hv, dtype=dtype, state_dtype=state_dtype)
        ids = packed["ssm_state_indices"].long()
        # Chunk returns [N,HV,K,V]; decode cache ABI is [pool,HV,V,K].
        packed["initial_state"][ids] = final.transpose(-1, -2).to(state_dtype)
        expected = reference_packed(packed)
        check_close(*fused_recurrent_gated_delta_rule_packed_decode(**packed), *expected)
        spec = make_case(n, 4, h, hv, dtype=dtype, state_dtype=state_dtype, spec=True)
        accepted = spec["num_accepted_tokens"].long() - 1
        read = spec["ssm_state_indices"][torch.arange(n, device="cuda"), accepted].long()
        spec["initial_state"][read] = packed["initial_state"][ids]
        expected = reference(spec)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**spec), *expected)

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @torch.inference_mode()
    def test_stateful_packed_chunk_keeps_fallback(self, dtype, monkeypatch):
        fla = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
        import flag_attn.FLA.gated_delta_rule.chunk_fused_tle as module

        def unexpected_fast_path(*args, **kwargs):
            raise AssertionError("stateful ragged prefill must retain its existing path")

        monkeypatch.setattr(module, "_tle_chunk_gated_delta_rule_fwd", unexpected_fast_path)
        torch.manual_seed(113)
        q = torch.randn((1, 23, 4, 128), device="cuda", dtype=dtype) / 128**0.5
        k = torch.randn_like(q) / 128**0.5
        v = torch.randn((1, 23, 8, 128), device="cuda", dtype=dtype)
        g = -torch.rand((1, 23, 8), device="cuda", dtype=torch.float32)
        beta = torch.rand_like(g)
        initial = torch.randn((2, 8, 128, 128), device="cuda", dtype=torch.float32) * 0.05
        cu = torch.tensor([0, 7, 23], device="cuda", dtype=torch.int64)
        expected = fla(q, k, v, g, beta, initial_state=initial, output_final_state=True, cu_seqlens=cu)
        actual = chunk_gated_delta_rule_fwd(q, k, v, g, beta, 128**-0.5, initial, True, cu)
        rms_close(actual[1], expected[0])
        rms_close(actual[3], expected[1])

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @torch.inference_mode()
    def test_raw_chunk_materialized_auxiliary_contract(self, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("Hopper fusion contract")
        fla = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
        import flag_attn.FLA.gated_delta_rule.api as api
        import flag_attn.FLA.gated_delta_rule.chunk as implementation
        import flag_attn.FLA.gated_delta_rule.chunk_fused_forward as two_kernel
        import flag_attn.FLA.gated_delta_rule.chunk_fused_tle as fused

        def unexpected_fusion(*args, **kwargs):
            raise AssertionError("materialized auxiliaries require the staged path")

        for module in (api, implementation, two_kernel):
            monkeypatch.setattr(module, "SUPPRESS_LEVEL", 3)
        monkeypatch.setattr(api, "_tle_chunk_gated_delta_rule_fwd", unexpected_fusion)
        monkeypatch.setattr(fused, "_tle_chunk_gated_delta_rule_fwd", unexpected_fusion)
        monkeypatch.setattr(implementation, "chunk_gdn_two_kernel_fwd", unexpected_fusion)
        torch.manual_seed(73)
        q = torch.randn((2, 2048, 4, 128), device="cuda", dtype=torch.bfloat16) / 128**0.5
        k = torch.randn_like(q) / 128**0.5
        v = torch.randn((2, 2048, 16, 128), device="cuda", dtype=torch.bfloat16)
        g = -torch.rand((2, 2048, 16), device="cuda") * 0.1
        beta = torch.rand_like(g)
        expected = fla(q, k, v, g, beta, output_final_state=True)
        result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, 128**-0.5, None, True)
        assert len(result) == 7 and all(x is not None for x in result)
        assert all(torch.isfinite(x).all() for x in result)
        assert result[4].shape == (2, 2048, 16, 128)
        assert result[6].shape == v.shape
        rms_close(result[1], expected[0])
        rms_close(result[3], expected[1])

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("case", ["short_gva", "tail", "ragged"])
    @torch.inference_mode()
    def test_shared_preparation_poison_and_repeat(self, dtype, case, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("register preparation requires Hopper")
        torch.manual_seed(2091)
        batch, length, h, hv, dk, dv = (1, 512, 16, 32, 128, 128)
        cu = None
        if case == "tail":
            batch, length, h, hv, dk, dv = 2, 71, 2, 6, 60, 80
        if case == "ragged":
            batch, length, h, hv, dk, dv = 1, 82, 2, 6, 128, 64
            cu = torch.tensor([0, 0, 7, 71, 82], device="cuda", dtype=torch.int64)
        n = batch if cu is None else cu.numel() - 1
        q = torch.randn((batch, length, h, dk), device="cuda", dtype=dtype) / dk**0.5
        k = torch.randn_like(q) / dk**0.5
        v = torch.randn((batch, length, hv, dv), device="cuda", dtype=dtype)
        g = -torch.rand((batch, length, hv), device="cuda", dtype=torch.float32) * 0.1
        if case == "short_gva":
            g = g.to(dtype)
        beta = torch.rand_like(g)
        initial = None if case == "short_gva" else torch.randn((n, hv, dk, dv), device="cuda") * 0.05
        cache = (
            torch.zeros((n, hv, dv, dk), device="cuda") if initial is None else initial.transpose(-1, -2).contiguous()
        )
        expected = reference(
            dict(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=dk**-0.5,
                initial_state=cache,
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=None,
                num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=False,
            )
        )
        expected = (expected[0], expected[1].transpose(-1, -2))
        original_empty, original_like, original_new = torch.empty, torch.empty_like, torch.Tensor.new_empty

        def poison(t):
            if t.is_cuda and t.is_floating_point():
                t.fill_(float("nan"))
            return t

        monkeypatch.setattr(torch, "empty", lambda *a, **kw: poison(original_empty(*a, **kw)))
        monkeypatch.setattr(torch, "empty_like", lambda *a, **kw: poison(original_like(*a, **kw)))
        monkeypatch.setattr(torch.Tensor, "new_empty", lambda self, *a, **kw: poison(original_new(self, *a, **kw)))
        monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "0")
        monkeypatch.setenv("FLAG_ATTN_GDN_REGISTER_PREP", "1")
        previous = None
        for _ in range(3):
            result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, dk**-0.5, initial, True, cu)
            assert result[0].dtype == g.dtype
            for actual, target in zip((result[1], result[3]), expected):
                assert torch.isfinite(actual).all()
                ratio = (actual.float() - target.float()).square().mean().sqrt() / (
                    target.float().square().mean().sqrt() + 1e-8
                )
                assert ratio < 0.01, ratio.item()
            if previous is not None:
                assert all(torch.equal(a, b) for a, b in zip(result[:4], previous))
            previous = [x.clone() for x in result[:4]]

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("length", [7, 64, 71])
    @pytest.mark.parametrize("block_k", [32, 64])
    @pytest.mark.parametrize("warps", [1, 2, 4, 8])
    @pytest.mark.parametrize("outputs", [(True, False), (False, True), (True, True)], ids=["ungated", "gated", "both"])
    @torch.inference_mode()
    def test_preparation_all_configs_initialize_dense_outputs(self, dtype, length, block_k, warps, outputs):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("register preparation requires Hopper")
        from flag_attn.FLA.gated_delta_rule.chunk_fused_tle_prepare import _prepare_kernel

        torch.manual_seed(73)
        dk = 128 if length == 64 else 60
        k = torch.randn((1, length, 2, dk), device="cuda", dtype=dtype) / dk**0.5
        g = -torch.rand((1, length, 6), device="cuda") * 0.1
        beta = torch.rand_like(g)
        gc = torch.full_like(g, float("nan"))
        ungated, gated_output = outputs
        a = torch.full((1, length, 6, 64), float("nan"), device="cuda", dtype=dtype) if ungated else None
        ag = torch.full((1, length, 6, 64), float("nan"), device="cuda", dtype=dtype) if gated_output else None
        chunks = (length + 63) // 64
        _prepare_kernel.fn.fn[(chunks, 6)](
            k=k,
            g=g,
            g_out=gc,
            beta=beta,
            A=a,
            AG=ag,
            cu_seqlens=None,
            chunk_indices=None,
            T=length,
            H=2,
            HV=6,
            K=dk,
            BT=64,
            BC=16,
            BK=block_k,
            USE_G=not ungated,
            IS_VARLEN=False,
            OUTPUT_GATED=gated_output,
            OUTPUT_UNGATED=ungated,
            MERGE_PRECISION="ieee",
            FULL_CHUNKS=length % 64 == 0,
            num_warps=warps,
        )
        pad = chunks * 64 - length
        kg = torch.nn.functional.pad(k.repeat_interleave(3, 2).cpu().double(), (0, 0, 0, 0, 0, pad))
        kg = kg.reshape(1, chunks, 64, 6, dk).transpose(2, 3)
        bg = torch.nn.functional.pad(beta.cpu().double(), (0, 0, 0, pad)).reshape(1, chunks, 64, 6).transpose(2, 3)
        gp = (
            torch.nn.functional.pad(g.cpu().double(), (0, 0, 0, pad))
            .reshape(1, chunks, 64, 6)
            .transpose(2, 3)
            .cumsum(-1)
        )
        eye = torch.eye(64, dtype=torch.float64).expand(1, chunks, 6, 64, 64)
        lower = torch.tril((kg @ kg.transpose(-1, -2)) * bg[..., :, None], diagonal=-1)
        inverse = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
        mask = torch.ones((64, 64), dtype=torch.bool).tril()
        difference = torch.where(mask, gp[..., :, None] - gp[..., None, :], 0)
        gated = torch.where(mask, inverse * difference.exp(), 0)
        for actual, target in ((a, inverse), (ag, gated)):
            if actual is None:
                continue
            target = (
                target.transpose(2, 3).reshape(1, chunks * 64, 6, 64)[:, :length].to(device=actual.device, dtype=dtype)
            )
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, target, rtol=1e-2, atol=2e-3)
            upper = torch.arange(64, device="cuda")[None, :] > (torch.arange(length, device="cuda") % 64)[:, None]
            assert torch.count_nonzero(actual.masked_select(upper[None, :, None, :])) == 0
        expected_gc = (
            gp.transpose(2, 3).reshape(1, chunks * 64, 6)[:, :length].to(device=gc.device, dtype=torch.float32)
        )
        torch.testing.assert_close(gc, expected_gc, rtol=1e-6, atol=1e-6)

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("length", [7, 23, 64, 71])
    @pytest.mark.parametrize("gate_mode", ["decay", "zero", "mixed"])
    @pytest.mark.parametrize("batch", [1, 3])
    @torch.inference_mode()
    def test_native_gated_preparation_boundaries(self, dtype, length, gate_mode, batch, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("register preparation requires Hopper")
        from gdn_utils import make_case
        import flag_attn.FLA.gated_delta_rule.chunk_fused_tle_prepare as preparation

        calls = []
        original = preparation._prepare_kernel.run

        def record(*args, **kwargs):
            calls.append(kwargs["T"])
            return original(*args, **kwargs)

        monkeypatch.setattr(preparation._prepare_kernel, "run", record)

        case = make_case(batch, length, 2, 6, 60, 80, dtype=dtype, strided=False)
        for name in ("q", "k", "v", "g", "beta"):
            x = case[name]
            case[name] = x.reshape(batch, length, *x.shape[2:])
        case["cu_seqlens"] = case["ssm_state_indices"] = None
        case["use_qk_l2norm_in_kernel"] = False
        case["initial_state"] = case["initial_state"][:batch].contiguous()
        for name in ("q", "k"):
            case[name] /= 60**0.5
        if gate_mode == "zero":
            case["g"].zero_()
        elif gate_mode == "mixed":
            case["g"] = torch.rand_like(case["g"]) * 0.2 - 0.1
        expected = reference(case)
        actual = chunk_gated_delta_rule_fwd(
            case["q"],
            case["k"],
            case["v"],
            case["g"],
            case["beta"],
            case["scale"],
            case["initial_state"].transpose(-1, -2).contiguous(),
            True,
        )
        assert actual[2].shape[-1] == min(64, max(16, 1 << (length - 1).bit_length()))
        assert calls == [batch * length if length % 64 == 0 else length]
        for result, target in ((actual[1], expected[0]), (actual[3].transpose(-1, -2), expected[1])):
            assert torch.isfinite(result).all()
            ratio = (result.float() - target.float()).square().mean().sqrt() / (
                target.float().square().mean().sqrt() + 1e-8
            )
            assert ratio < 0.01, ratio.item()

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("dims", [(60, 80), (128, 128)])
    @pytest.mark.parametrize("warps", [2, 4])
    @pytest.mark.parametrize("stages", [2, 3, 4])
    @torch.inference_mode()
    def test_narrow_state_configs(self, dtype, dims, warps, stages, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("narrow state configurations require Hopper")
        import flag_attn.FLA.chunk_delta_h as state
        from gdn_utils import make_case

        dk, dv = dims
        case = make_case(2, 71, 2, 6, dk, dv, dtype=dtype, strided=False)
        for name in ("q", "k", "v", "g", "beta"):
            x = case[name]
            case[name] = x.reshape(2, 71, *x.shape[2:])
        case["cu_seqlens"] = case["ssm_state_indices"] = None
        case["use_qk_l2norm_in_kernel"] = False
        case["initial_state"] = case["initial_state"][:2].contiguous()
        for name in ("q", "k"):
            case[name] /= dk**0.5
        expected = reference(case)
        raw = state.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn.fn
        calls = []

        class ForcedState:
            def __getitem__(self, grid):
                def launch(**kwargs):
                    calls.append(kwargs["N"])
                    return raw[grid({"BV": 16})](
                        **kwargs,
                        USE_G=kwargs["g"] is not None,
                        USE_GK=kwargs["gk"] is not None,
                        USE_INITIAL_STATE=kwargs["h0"] is not None,
                        STORE_FINAL_STATE=kwargs["ht"] is not None,
                        SAVE_NEW_VALUE=kwargs["v_new"] is not None,
                        IS_VARLEN=kwargs["cu_seqlens"] is not None,
                        BV=16,
                        num_warps=warps,
                        num_stages=stages,
                    )

                return launch

        monkeypatch.setattr(state, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", ForcedState())
        actual = chunk_gated_delta_rule_fwd(
            case["q"],
            case["k"],
            case["v"],
            case["g"],
            case["beta"],
            case["scale"],
            case["initial_state"].transpose(-1, -2).contiguous(),
            True,
        )
        assert calls == [2]
        for result, target in ((actual[1], expected[0]), (actual[3].transpose(-1, -2), expected[1])):
            assert torch.isfinite(result).all()
            ratio = (result.float() - target.float()).square().mean().sqrt() / (
                target.float().square().mean().sqrt() + 1e-8
            )
            assert ratio < 0.01, ratio.item()

    @pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("heads", [(16, 16, 64), (8, 16, 80)])
    @pytest.mark.parametrize("gates", [(True, True), (True, False), (False, True)])
    @pytest.mark.parametrize("config_index", [0, 1])
    @torch.inference_mode()
    def test_single_key_tile_fp32_gates(self, dtype, heads, gates, config_index, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("FP32 two-kernel gates require Hopper")
        import flag_attn.FLA.gated_delta_rule.chunk_fused_forward as fused
        from gdn_utils import make_case

        h, hv, dv = heads
        case = make_case(4, 128, h, hv, 64, dv, dtype=dtype, strided=False)
        for name in ("q", "k", "v", "g", "beta"):
            x = case[name]
            case[name] = x.reshape(4, 128, *x.shape[2:])
        for name, fp32 in zip(("g", "beta"), gates):
            if not fp32:
                case[name] = case[name].to(dtype)
        case["cu_seqlens"] = case["ssm_state_indices"] = None
        case["use_qk_l2norm_in_kernel"] = False
        case["initial_state"] = case["initial_state"][:4].contiguous().zero_()
        for name in ("q", "k"):
            case[name] /= 64**0.5
        expected = reference(case)
        original = fused._chunk_gdn_two_kernel_fwd_kernel
        config = original.configs[config_index]
        calls = []

        class ForcedPipeline:
            def __getitem__(self, grid):
                def launch(**kwargs):
                    calls.append(kwargs["g"].dtype)
                    assert kwargs["beta"].dtype == case["beta"].dtype
                    return original.fn[grid(config.kwargs)](**kwargs, **config.all_kwargs())

                return launch

        monkeypatch.setattr(fused, "_chunk_gdn_two_kernel_fwd_kernel", ForcedPipeline())
        actual = chunk_gated_delta_rule_fwd(
            case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], None, True
        )
        assert calls == [torch.float32]
        assert actual[0].dtype == case["g"].dtype
        prefix = case["g"].float().reshape(4, 2, 64, hv).cumsum(2).reshape(4, 128, hv)
        torch.testing.assert_close(actual[0], prefix.to(case["g"].dtype), rtol=1e-3, atol=1e-5)
        for result, target in ((actual[1], expected[0]), (actual[3].transpose(-1, -2), expected[1])):
            assert torch.isfinite(result).all()
            ratio = (result.float() - target.float()).square().mean().sqrt() / (
                target.float().square().mean().sqrt() + 1e-8
            )
            assert ratio < 0.01, ratio.item()


@pytest.mark.fused_recurrent_gated_delta_rule_fwd
@pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
class TestRecurrentGatedDeltaRule:
    @pytest.mark.parametrize("h,hv", [(2, 4), (4, 12), (2, 8)])
    @pytest.mark.parametrize("length", [1, 3, 8])
    @pytest.mark.parametrize("strided", [False, True])
    @pytest.mark.parametrize("seed", [0, 17])
    def test_recurrent_stateful(self, h, hv, length, strided, seed):
        case = make_case(4, length, h, hv, strided=strided, spec=length > 1, seed=seed)
        expected = reference(case)
        actual = fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(*actual, *expected)
        assert actual[1].data_ptr() == case["initial_state"].data_ptr()

    @pytest.mark.parametrize("kd,vd", [(1, 1), (60, 80), (64, 128), (128, 64), (256, 128), (128, 256)])
    @pytest.mark.parametrize("vector_beta", [False, True])
    def test_recurrent_tails_and_vector_beta(self, kd, vd, vector_beta):
        case = make_case(3, 3, 2, 6, kdim=kd, vdim=vd, vector_beta=vector_beta, spec=True)
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.parametrize(
        "dtype,state_dtype",
        [(torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)],
    )
    def test_recurrent_dtypes(self, dtype, state_dtype):
        case = make_case(3, 4, 2, 6, dtype=dtype, state_dtype=state_dtype, spec=True)
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.parametrize("inplace", [False, True])
    def test_recurrent_dense_without_indices(self, inplace):
        n, length = 3, 4
        case = make_case(n, length, 2, 4)
        for name in ("q", "k", "v", "g", "beta"):
            x = case[name]
            case[name] = x.reshape(n, length, *x.shape[2:])
        case.update(cu_seqlens=None, ssm_state_indices=None, inplace_final_state=inplace)
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.parametrize("inplace", [False, True])
    def test_recurrent_ragged_and_empty(self, inplace):
        case = make_case(5, 8, 2, 6, lengths=[0, 1, 8, 3, 0], spec=True)
        case["inplace_final_state"] = inplace
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    def test_recurrent_padding_and_index_strides(self):
        case = make_case(4, 4, 2, 6, padding=True, spec=True)
        case["ssm_state_indices"][-2] = -1
        idx = case["ssm_state_indices"]
        wide = torch.full((4, 8), -99, device="cuda", dtype=idx.dtype)
        wide[:, ::2] = idx
        case["ssm_state_indices"] = wide[:, ::2]
        original = case["initial_state"].clone()
        expected = reference(case)
        out, st = fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(out, st, *expected)
        assert torch.equal(st[0], original[0])
        assert torch.count_nonzero(out[:, -4:]) == 0

    def test_recurrent_continuation_and_repeatability(self):
        case = make_case(3, 8, 2, 6)
        original = case["initial_state"].clone()
        expected = reference(case)
        full = fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(*full, *expected)
        full_o, full_s = full[0].clone(), full[1].clone()
        case["initial_state"].copy_(original)
        repeated = fused_recurrent_gated_delta_rule_fwd(**case)
        assert torch.equal(repeated[0], full_o) and torch.equal(repeated[1], full_s)
        # Feed one token from each sequence per call, writing one stable slot per request.
        case["initial_state"].copy_(original)
        ids = case["ssm_state_indices"]
        initial = original[ids[:, 0].long()].clone()
        stable = torch.arange(1, 4, device="cuda", dtype=torch.int32)
        pool = torch.zeros_like(original)
        pool[stable.long()] = initial
        outputs = []
        for t in range(8):
            step = {**case, "initial_state": pool, "cu_seqlens": None, "ssm_state_indices": stable}
            for name in ("q", "k", "v", "g", "beta"):
                x = case[name].reshape(3, 8, *case[name].shape[2:])
                step[name] = x[:, t : t + 1]
            outputs.append(fused_recurrent_gated_delta_rule_fwd(**step)[0])
        stacked = torch.cat(outputs, dim=1).reshape_as(full_o)
        torch.testing.assert_close(stacked, full_o, rtol=1.0e-2, atol=2.0e-3)
        torch.testing.assert_close(pool[stable.long()], full_s[ids[:, -1].long()], rtol=2.0e-4, atol=2.0e-5)

    def test_recurrent_zero_decay_beta_limits(self):
        case = make_case(3, 4, 2, 6)
        case["g"].zero_()
        case["beta"][..., 0] = 0
        case["beta"][..., 1] = 1
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    def test_recurrent_preallocated_kernel_is_exercised(self):
        from flag_attn.FLA.gated_delta_rule.fused_recurrent import _launch

        case = make_case(2, 4, 2, 4, spec=True)
        expected = reference(case)
        out = torch.full_like(case["v"], float("nan"))
        kernel = _launch(
            case["q"],
            case["k"],
            case["v"],
            case["g"],
            case["beta"],
            case["scale"],
            case["initial_state"],
            True,
            case["cu_seqlens"],
            case["ssm_state_indices"],
            case["num_accepted_tokens"],
            True,
            out,
            case["initial_state"],
        )
        assert "recurrent_fwd_kernel" in kernel.name
        check_close(out, case["initial_state"], *expected)

    @pytest.mark.parametrize("strided_state", [False, True])
    def test_recurrent_gate_and_state_strides(self, strided_state):
        case = make_case(3, 3, 2, 6, kdim=60, vdim=80)
        for name in ("g", "beta"):
            x = case[name]
            wide = torch.empty((*x.shape[:-1], x.shape[-1] * 2), device="cuda", dtype=x.dtype)
            wide[..., ::2] = x
            case[name] = wide[..., ::2]
        if strided_state:
            state = case["initial_state"]
            backing = torch.empty((*state.shape[:2], 60, 80), device="cuda", dtype=state.dtype)
            case["initial_state"] = backing.transpose(-1, -2)
            case["initial_state"].copy_(state)
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    def test_recurrent_public_export(self):
        import flag_attn

        case = make_case(2, 1, 2, 4)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    def test_recurrent_padded_projection_buffer(self):
        case = make_case(3, 4, 2, 6, spec=True, projection_padding=64)
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    def test_recurrent_padding_snapshots_are_initialized(self, monkeypatch):
        case = make_case(4, 4, 2, 6, padding=True, spec=True)
        case["inplace_final_state"] = False
        expected = reference(case)
        empty = torch.empty

        def poisoned_empty(*args, **kwargs):
            tensor = empty(*args, **kwargs)
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
            return tensor

        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", poisoned_empty)
            out, final = fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(out, final, *expected)
        assert torch.count_nonzero(final[-4:]) == 0

    def test_recurrent_invalid_write_slot_keeps_cache(self):
        case = make_case(3, 4, 2, 6, spec=True)
        case["num_accepted_tokens"].fill_(1)
        case["ssm_state_indices"][0, -1] = 0
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.parametrize("values", ["zero", "impulse"])
    def test_recurrent_longer_verification_values(self, values):
        case = make_case(3, 16, 2, 6, kdim=127, vdim=129, spec=True)
        case["g"].zero_()
        case["q"].zero_()
        case["k"].zero_()
        case["v"].zero_()
        if values == "impulse":
            case["q"][..., 0] = 1
            case["k"][..., 0] = 1
            case["v"][:, 0, :, 0] = 1
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    def test_recurrent_pre_normalized_inputs(self):
        case = make_case(4, 8, 2, 6, spec=True)
        for name in ("q", "k"):
            x = case[name]
            case[name] = torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype)
        case["use_qk_l2norm_in_kernel"] = False
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.parametrize("lengths", [[0, 0, 0], [0, 1, 0]])
    def test_recurrent_empty_batch_segments(self, lengths):
        case = make_case(3, 1, 2, 4, lengths=lengths)
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.parametrize("varlen", [False, True])
    def test_recurrent_single_read_index_without_inplace_writes(self, varlen):
        case = make_case(2, 4, 2, 4)
        case["inplace_final_state"] = False
        case["ssm_state_indices"] = case["ssm_state_indices"][:, :1]
        if not varlen:
            for name in ("q", "k", "v", "g", "beta"):
                x = case[name]
                case[name] = x.reshape(2, 4, *x.shape[2:])
            case["cu_seqlens"] = None
        initial = case["initial_state"].clone()
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)
        torch.testing.assert_close(case["initial_state"], initial, rtol=0, atol=0)

    @pytest.mark.parametrize("invalid", ["v_rank", "g_dtype", "beta_dtype", "empty_indices", "short_indices"])
    def test_recurrent_invalid_metadata(self, invalid):
        case = make_case(2, 4, 2, 4)
        if invalid == "v_rank":
            case["v"] = case["v"][..., 0]
        elif invalid == "g_dtype":
            case["g"] = case["g"].to(torch.int32)
        elif invalid == "beta_dtype":
            case["beta"] = case["beta"].to(torch.int32)
        elif invalid == "empty_indices":
            case["ssm_state_indices"] = case["ssm_state_indices"][:, :0]
        else:
            for name in ("q", "k", "v", "g", "beta"):
                x = case[name]
                case[name] = x.reshape(2, 4, *x.shape[2:])
            case["cu_seqlens"] = None
            case["ssm_state_indices"] = case["ssm_state_indices"][:, :2]
        with pytest.raises(ValueError):
            fused_recurrent_gated_delta_rule_fwd(**case)

    @pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("vector_beta", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    @torch.inference_mode()
    def test_recurrent_large_snapshot_layout(self, state_dtype, vector_beta, normalize, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("Hopper layout path")
        from gdn_utils import record_recurrent_launch

        calls = record_recurrent_launch(monkeypatch)
        case = make_case(32, 4, 4, 16, state_dtype=state_dtype, spec=True, vector_beta=vector_beta)
        if not normalize:
            for name in ("q", "k"):
                case[name] = torch.nn.functional.normalize(case[name].float(), dim=-1).to(case[name].dtype)
            case["use_qk_l2norm_in_kernel"] = False
        expected = reference(case)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)
        assert len(calls) == 1 and calls[0]["LAYOUT"] is not None


@pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
@pytest.mark.skipif(not _cuda_tle_available(), reason="requires CUDA/TLE")
class TestPackedGatedDeltaRule:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("dims", [(1, 2, 128, 128), (2, 6, 60, 80), (2, 8, 127, 129), (1, 1, 256, 64)])
    @pytest.mark.parametrize("seed", [0, 17])
    @torch.inference_mode()
    def test_packed_reference(self, dtype, dims, seed, monkeypatch):
        import flag_attn.FLA.gated_delta_rule.packed_decode as module

        original = module._launch
        calls = []

        def tracked(*args, **kw):
            assert kw["gate_parameters"] is not None
            calls.append(True)
            return original(*args, **kw)

        monkeypatch.setattr(module, "_launch", tracked)
        c = make_packed_case(n=3, h=dims[0], hv=dims[1], kdim=dims[2], vdim=dims[3], dtype=dtype, seed=seed)
        expected = reference_packed(c)
        state0 = c["initial_state"].clone()
        previous = None
        for _ in range(3):
            c["initial_state"].copy_(state0)
            c["out"].fill_(float("nan"))
            result = fused_recurrent_gated_delta_rule_packed_decode(**c)
            assert result[0] is c["out"] and result[1] is c["initial_state"]
            check_close(*result, *expected)
            if previous is not None:
                assert torch.equal(result[0], previous[0]) and torch.equal(result[1], previous[1])
            previous = [x.clone() for x in result]
        assert len(calls) == 3

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @torch.inference_mode()
    def test_packed_beta_rounding_in_fp32_cache(self, dtype):
        c = make_packed_case(n=1, h=1, hv=1, kdim=1, vdim=1, dtype=dtype)
        c["mixed_qkv"].fill_(1)
        c["initial_state"].zero_()
        c["b"].fill_(0.1)
        c["scale"] = 1.0
        c["use_qk_l2norm_in_kernel"] = False
        expected = c["b"].float().sigmoid().to(dtype).float()
        _, state = fused_recurrent_gated_delta_rule_packed_decode(**c)
        slot = c["ssm_state_indices"].long()
        torch.testing.assert_close(state[slot, 0, 0, 0], expected[:, 0], rtol=2e-4, atol=2e-5)
        untouched = state.clone()
        untouched[slot] = 0
        assert torch.count_nonzero(untouched) == 0

    @pytest.mark.parametrize(
        "case",
        [
            "padding",
            "strides",
            "threshold",
            "gate_extremes",
            "zero",
            "impulse",
            "zero_scale",
            "no_norm",
            "bf16_state",
            "fp16_state",
            "empty",
        ],
    )
    @torch.inference_mode()
    def test_packed_boundaries(self, case):
        c = make_packed_case(
            n=4,
            h=2,
            hv=6,
            projection_padding=64,
            state_dtype=torch.bfloat16
            if case == "bf16_state"
            else torch.float16
            if case == "fp16_state"
            else torch.float32,
        )
        if case == "padding":
            c["ssm_state_indices"][0], c["ssm_state_indices"][-1] = 0, -1
        if case == "strides":
            for name in ("a", "b", "ssm_state_indices"):
                x = c[name]
                backing = torch.zeros((x.shape[0] * 2, *x.shape[1:]), device=x.device, dtype=x.dtype)
                backing[::2] = x
                c[name] = backing[::2]
            st = c["initial_state"]
            c["initial_state"] = st.transpose(-1, -2).contiguous().transpose(-1, -2)
        if case == "threshold":
            c["a"][:] = torch.tensor([-20, -1, 0, 19.875, 20, 20.125], device="cuda")
            c["dt_bias"].zero_()
        if case == "gate_extremes":
            c["a"][:] = torch.tensor([-80, -20, 0, 20, 80, 100], device="cuda")
            c["b"][:] = torch.tensor([-80, -20, 0, 20, 80, 100], device="cuda")
        if case in ("zero", "impulse"):
            c["mixed_qkv"].zero_()
            if case == "impulse":
                c["mixed_qkv"][:, ::128] = 1
        if case == "zero_scale":
            c["scale"] = 0.0
        if case == "no_norm":
            c["use_qk_l2norm_in_kernel"] = False
            c["mixed_qkv"] *= 0.05
        if case == "empty":
            for name in ("mixed_qkv", "a", "b", "out", "ssm_state_indices"):
                c[name] = c[name][:0]
        expected = reference_packed(c)
        check_close(*fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)

    @torch.inference_mode()
    def test_packed_continuation_matches_recurrent(self):
        from flag_attn import fused_recurrent_gated_delta_rule_fwd

        c = make_packed_case()
        separate = c["initial_state"].clone()
        for seed in range(4):
            torch.manual_seed(seed)
            c["mixed_qkv"].normal_()
            expected_case = unpack_reference_case({**c, "initial_state": separate})
            expected = fused_recurrent_gated_delta_rule_fwd(**expected_case)
            check_close(*fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)

    @torch.inference_mode()
    def test_packed_split_grid(self):
        n, hv = 21846, 3
        c = dict(
            mixed_qkv=torch.zeros((n, 2 + hv), device="cuda", dtype=torch.bfloat16),
            a=torch.zeros((n, hv), device="cuda"),
            b=torch.zeros((n, hv), device="cuda"),
            A_log=torch.zeros(hv, device="cuda"),
            dt_bias=torch.zeros(hv, device="cuda"),
            scale=1.0,
            initial_state=torch.zeros((n + 1, hv, 1, 1), device="cuda"),
            out=torch.full((n, 1, hv, 1), float("nan"), device="cuda", dtype=torch.bfloat16),
            ssm_state_indices=torch.arange(1, n + 1, device="cuda", dtype=torch.int64),
        )
        out, state = fused_recurrent_gated_delta_rule_packed_decode(**c)
        assert torch.count_nonzero(out) == 0 and torch.count_nonzero(state) == 0

    @torch.inference_mode()
    def test_public_exports_and_untouched_padding(self):
        import flag_attn
        from flag_attn.FLA.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

        assert flag_attn.chunk_gated_delta_rule_fwd is chunk_gated_delta_rule_fwd
        assert callable(flag_attn.fused_recurrent_gated_delta_rule_fwd)
        c = make_packed_case(n=3)
        c["ssm_state_indices"][0] = 0
        c["ssm_state_indices"][2] = -1
        original = c["initial_state"].clone()
        expected = reference_packed(c)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)
        untouched = torch.ones(original.shape[0], dtype=torch.bool, device="cuda")
        untouched[c["ssm_state_indices"][1].long()] = False
        assert torch.equal(c["initial_state"][untouched], original[untouched])

    @pytest.mark.parametrize("axis", ["projection", "state_head", "state_value", "state_key"])
    @torch.inference_mode()
    def test_large_relative_offsets(self, axis):
        if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            pytest.skip("large-offset addressing check requires 12 GiB free")
        c = make_packed_case(n=3, h=1, hv=3, kdim=3, vdim=3)
        stride = 2**30 + 64
        if axis == "projection":
            width = c["mixed_qkv"].shape[1]
            storage = torch.empty(2 * stride + width, device="cuda", dtype=c["mixed_qkv"].dtype)
            view = storage.as_strided((3, width), (stride, 1))
            view.copy_(c["mixed_qkv"])
            c["mixed_qkv"] = view
        else:
            state = c["initial_state"]
            strides = [64, 9, 3, 1]
            strides[{"state_head": 1, "state_value": 2, "state_key": 3}[axis]] = stride
            size = 1 + sum((d - 1) * s for d, s in zip(state.shape, strides))
            storage = torch.empty(size, device="cuda", dtype=state.dtype)
            view = storage.as_strided(state.shape, strides)
            view.copy_(state)
            c["initial_state"] = view
        expected = reference_packed(c)
        check_close(*fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)
        assert torch.count_nonzero(c["out"]) > 0

    @pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("dims", [(128, 128), (256, 64), (64, 256)])
    @torch.inference_mode()
    def test_low_precision_cache_large_parallelism(self, state_dtype, dims):
        from flag_attn.FLA.gated_delta_rule.packed_decode import _launch_packed
        from flag_attn import fused_recurrent_gated_delta_rule_fwd
        from gdn_utils import make_case, reference

        c = make_packed_case(n=64, h=4, hv=8, kdim=dims[0], vdim=dims[1], state_dtype=state_dtype)
        expected = reference_packed(c)
        kernel = _launch_packed(**c)
        assert kernel.n_spills == 0
        check_close(c["out"], c["initial_state"], *expected)
        recurrent = make_case(64, 3, 4, 8, dims[0], dims[1], state_dtype=state_dtype, spec=True)
        expected = reference(recurrent)
        check_close(*fused_recurrent_gated_delta_rule_fwd(**recurrent), *expected)

    @pytest.mark.parametrize("case", ["width", "out", "indices", "gate", "parameter", "grad"])
    def test_packed_invalid(self, case):
        c = make_packed_case()
        if case == "width":
            c["mixed_qkv"] = c["mixed_qkv"][:, :-1]
        if case == "out":
            c["out"] = c["out"][:, 0]
        if case == "indices":
            c["ssm_state_indices"] = c["ssm_state_indices"].float()
        if case == "gate":
            c["a"] = c["a"][:, :-1]
        if case == "parameter":
            c["dt_bias"] = c["dt_bias"][:-1]
        if case == "grad":
            c["mixed_qkv"].requires_grad_(True)
        with pytest.raises((ValueError, RuntimeError)):
            fused_recurrent_gated_delta_rule_packed_decode(**c)

    @pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
    @torch.inference_mode()
    def test_packed_large_cache_reuses_read_slot(self, state_dtype, monkeypatch):
        if torch.cuda.get_device_capability()[0] != 9:
            pytest.skip("Hopper cache scheduling path")
        from gdn_utils import record_recurrent_launch

        calls = record_recurrent_launch(monkeypatch)
        c = make_packed_case(n=192, h=4, hv=16, state_dtype=state_dtype)
        c["ssm_state_indices"][0] = 0
        c["ssm_state_indices"][-1] = -1
        expected = reference_packed(c)
        check_close(*fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)
        assert len(calls) == 1 and calls[0]["REUSE_PACKED_STATE"]
