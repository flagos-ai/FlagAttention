# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared GDN contracts across prefill, packed decode and speculative decode."""

import pytest
import torch

pytest.importorskip("triton.experimental.tle.language", reason="FlagTree/TLE required")

from flag_attn.FLA.gated_delta_rule import chunk_gated_delta_rule, chunk_gated_delta_rule_fwd
from flag_attn import fused_recurrent_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_packed_decode
from flag_attn.testing.packed_gdn import make_packed_case, reference_packed
from flag_attn.testing.recurrent_gdn import make_case, reference, check_close

pytestmark = [pytest.mark.gdn_forward, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def rms_close(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    assert error / (expected.float().square().mean().sqrt() + 1e-8) < 0.01


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp32_gates", [False, True])
@torch.inference_mode()
def test_raw_chunk_fused_contract(dtype, fp32_gates, monkeypatch):
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
    for a, b in zip(actual, expected):
        if b is None:
            assert a is None
        else:
            rms_close(a, b)
    # Complete initialization of the returned triangular auxiliary tensor.
    a = actual[2].view(2, 32, 64, 16, 64).permute(0, 1, 3, 2, 4)
    assert torch.count_nonzero(a.triu(1)) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_prefill_packed_and_speculative_state_handoff(dtype, state_dtype):
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_stateful_packed_chunk_keeps_fallback(dtype, monkeypatch):
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


@pytest.mark.chunk_gated_delta_rule
@torch.inference_mode()
def test_raw_chunk_materialized_auxiliary_contract(monkeypatch):
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
