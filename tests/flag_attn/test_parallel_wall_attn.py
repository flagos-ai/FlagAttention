# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness and routing tests for the public Wall-Attention provider."""

from __future__ import annotations


import pytest
import torch

from flag_attn.testing import backend as test_backend
from flag_attn import parallel_wall_attn
from flag_attn.FLA import wall_attn
from flag_attn.FLA.cumsum import chunk_global_cumsum
from flag_attn.FLA.wall_attn import parallel as provider
from flag_attn.utils import has_triton_tle

# Cover each dtype, head layout, and supported sequence length with representative pairs.
FAST_CASES = [
    pytest.param("mha", 8, 512, "bf16", torch.bfloat16, id="bf16-t512-mha"),
    pytest.param("gqa", 2, 4096, "bf16", torch.bfloat16, id="bf16-t4096-gqa"),
    pytest.param("gqa", 2, 512, "fp16", torch.float16, id="fp16-t512-gqa"),
    pytest.param("mha", 8, 4096, "fp16", torch.float16, id="fp16-t4096-mha"),
    pytest.param("mha", 8, 1024, "bf16", torch.bfloat16, id="bf16-t1024-mha"),
    pytest.param("gqa", 2, 2048, "fp16", torch.float16, id="fp16-t2048-gqa"),
]


def _has_h100_tle() -> bool:
    return test_backend.supports_operator("parallel_wall_attn") and test_backend.cuda_capability() == (9, 0) and has_triton_tle()


def _make_inputs(t: int, h: int, d: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(1, t, 8, d, device=test_backend.device, dtype=dtype)
    k = torch.randn(1, t, h, d, device=test_backend.device, dtype=dtype)
    v = torch.randn(1, t, h, d, device=test_backend.device, dtype=dtype)
    g = (-torch.randn(1, t, 8, d, device=test_backend.device).abs() * 0.05).to(dtype)
    return q, k, v, g


@pytest.mark.parallel_wall_attn
def test_parallel_wall_attn_is_public_api():
    from flag_attn.runtime.backend import resolve_operator

    assert parallel_wall_attn is resolve_operator("parallel_wall_attn", "flag_attn.FLA.wall_attn", "parallel_wall_attn")


@pytest.mark.parallel_wall_attn
def test_parallel_wall_attn_official_helpers_are_exported():
    from flag_attn.FLA.wall_attn.decode import build_wall_kv_cache, parallel_wall_attn_decode
    from flag_attn.FLA.wall_attn.naive import naive_wall_attn

    assert wall_attn.build_wall_kv_cache is build_wall_kv_cache
    assert wall_attn.parallel_wall_attn_decode is parallel_wall_attn_decode
    assert wall_attn.naive_wall_attn is naive_wall_attn
    assert {"build_wall_kv_cache", "parallel_wall_attn_decode", "naive_wall_attn"} <= set(wall_attn.__all__)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "h,option",
    [
        pytest.param(8, "default", id="mha-default"),
        pytest.param(2, "default", id="gqa-default"),
        pytest.param(8, "window", id="mha-window"),
        pytest.param(2, "varlen", id="gqa-varlen"),
        pytest.param(8, "sink_scalar", id="mha-sink-scalar"),
    ],
)
@pytest.mark.skipif(not test_backend.supports_operator("parallel_wall_attn"), reason="naive reference comparison requires accelerator cumsum")
@torch.inference_mode()
def test_parallel_wall_attn_default_matches_naive(h, option):
    q, k, v, g = _make_inputs(17, h, 16, torch.float32, seed=7)
    kwargs = {"scale": 16**-0.5}
    if option == "window":
        kwargs["window_size"] = 5
    elif option == "varlen":
        kwargs["cu_seqlens"] = torch.tensor([0, 5, 17], device=q.device, dtype=torch.long)
    elif option == "sink_scalar":
        kwargs["sink_bias"] = torch.randn(8, device=q.device) * 0.1
        kwargs["g_scalar"] = -torch.rand(1, 17, 8, device=q.device) * 0.01
    expected = wall_attn.naive_wall_attn(q, k, v, g, **kwargs)
    actual = provider._parallel_wall_attn_default(q, k, v, g, **kwargs).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "h,dtype,use_extra_gates",
    [
        pytest.param(8, torch.bfloat16, False, id="mha-bf16-default"),
        pytest.param(2, torch.bfloat16, True, id="gqa-bf16-sink-scalar"),
        pytest.param(8, torch.float16, True, id="mha-fp16-sink-scalar"),
        pytest.param(2, torch.float16, False, id="gqa-fp16-default"),
    ],
)
@pytest.mark.skipif(not test_backend.supports_operator("parallel_wall_attn"), reason=test_backend.skip_reason("parallel_wall_attn"))
@torch.inference_mode()
def test_parallel_wall_attn_decode_matches_naive(h, dtype, use_extra_gates):
    # One tail query, three cache chunks, including a partial final chunk.
    q, k, v, g = _make_inputs(129, h, 32, dtype, seed=1)
    prefix = chunk_global_cumsum(g, scale=provider.RCP_LN2)
    k_tilde, anchors = wall_attn.build_wall_kv_cache(k, prefix, chunk_size=64)
    kwargs = {"scale": 32**-0.5}
    scalar_prefix = None
    if use_extra_gates:
        kwargs["sink_bias"] = torch.randn(8, device=q.device) * 0.1
        kwargs["g_scalar"] = -torch.rand(1, 129, 8, device=q.device) * 0.01
        scalar_prefix = chunk_global_cumsum(kwargs["g_scalar"], scale=provider.RCP_LN2)
    expected = wall_attn.naive_wall_attn(q, k, v, g, **kwargs)[:, -1:].contiguous()
    actual, lse = wall_attn.parallel_wall_attn_decode(
        q=q[:, -1:].contiguous(),
        v=v,
        p_curr=prefix[:, -1:].contiguous(),
        k_tilde=k_tilde,
        r_cache=anchors,
        sink_bias=kwargs.get("sink_bias"),
        scale=kwargs["scale"],
        cache_chunk_size=64,
        g_scalar_cumsum=scalar_prefix,
    )
    assert k_tilde.shape == (1, 129, 8, 32)
    assert anchors.shape == (1, 3, 8, 32)
    assert k_tilde.dtype == dtype
    assert actual.shape == (1, 1, 8, 32)
    assert lse.shape == (1, 1, 8)
    assert torch.isfinite(actual).all() and torch.isfinite(lse).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("family,h,t,dtype_name,dtype", FAST_CASES)
@pytest.mark.skipif(not _has_h100_tle(), reason="fast-path matrix requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_fast_matrix_matches_default(family, h, t, dtype_name, dtype):
    del family
    q, k, v, g = _make_inputs(t, h, 64, dtype, seed=0)
    g = g * 0.2
    expected_route = "hopper_bf16" if dtype is torch.bfloat16 else "hopper_fp16"
    assert provider.select_route(q, k, v, g, scale=64**-0.5) == expected_route

    expected = provider._parallel_wall_attn_default(q, k, v, g, scale=64**-0.5).float()
    actual = parallel_wall_attn(q, k, v, g, scale=64**-0.5).float()
    assert provider.get_last_route() == expected_route
    test_backend.device_fn.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    if dtype_name == "fp16":
        assert (actual - expected).abs().max().item() <= 0.02


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "h,t",
    [
        pytest.param(2, 512, id="512-gqa"),
        pytest.param(8, 4096, id="4096-mha"),
    ],
)
@pytest.mark.skipif(not _has_h100_tle(), reason="D128 fast path requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_bf16_d128_matches_default(h, t):
    q, k, v, g = _make_inputs(t, h, 128, torch.bfloat16, seed=0)
    # Keep the long sequence within the fast path numerical range.
    g = g * 0.2
    assert provider.select_route(q, k, v, g) == "hopper_bf16"
    expected = provider._parallel_wall_attn_default(q, k, v, g).float()
    actual = parallel_wall_attn(q, k, v, g).float()
    assert provider.get_last_route() == "hopper_bf16"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "change",
    (
        "fp16_d128",
        "unsupported_t",
        "noncontiguous",
        "requires_grad",
        "g_scalar",
        "sink_bias",
        "window_size",
        "cu_seqlens",
    ),
)
@pytest.mark.skipif(not test_backend.supports_operator("parallel_wall_attn"), reason="routing test requires accelerator tensors")
def test_parallel_wall_attn_unsupported_contract_routes_to_fallback(change, monkeypatch):
    q, k, v, g = _make_inputs(512, 8, 64, torch.bfloat16, seed=0)
    kwargs = {}
    if change == "fp16_d128":
        q, k, v, g = _make_inputs(512, 8, 128, torch.float16, seed=0)
    elif change == "unsupported_t":
        q, k, v, g = _make_inputs(256, 8, 64, torch.bfloat16, seed=0)
    elif change == "noncontiguous":
        q = q.transpose(1, 2)
    elif change == "requires_grad":
        q.requires_grad_(True)
    elif change == "g_scalar":
        kwargs[change] = torch.zeros(1, 512, 8, device=test_backend.device, dtype=q.dtype)
    elif change == "sink_bias":
        kwargs[change] = torch.zeros(8, device=test_backend.device, dtype=q.dtype)
    elif change == "window_size":
        kwargs[change] = 128
    elif change == "cu_seqlens":
        kwargs[change] = torch.tensor([0, 512], device=test_backend.device, dtype=torch.long)

    assert provider.select_route(q, k, v, g, **kwargs) == "official_fallback"
    sentinel = torch.empty(0, device=test_backend.device)
    monkeypatch.setattr(provider, "_parallel_wall_attn_default", lambda *args, **kw: sentinel)
    assert provider.parallel_wall_attn(q, k, v, g, **kwargs) is sentinel


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(not test_backend.supports_operator("parallel_wall_attn"), reason=test_backend.skip_reason("parallel_wall_attn"))
@pytest.mark.skipif(not test_backend.is_nvidia(), reason="tests the generic NVIDIA implementation contract")
def test_parallel_wall_attn_non_sm90_routes_to_fallback(monkeypatch):
    inputs = _make_inputs(512, 8, 64, torch.bfloat16, seed=0)
    monkeypatch.setattr(provider, "_is_sm90", lambda device_index: False)
    assert provider.select_route(*inputs) == "official_fallback"


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.skipif(not _has_h100_tle(), reason="numerical guard test requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_constant_gate_uses_safe_fallback(dtype):
    t, gate = 4096, -0.05
    q = torch.ones(1, t, 8, 64, device=test_backend.device, dtype=dtype)
    k, v = torch.ones_like(q), torch.ones_like(q)
    g = torch.full_like(q, gate)
    actual = parallel_wall_attn(q, k, v, g)
    assert provider.get_last_route() == "official_fallback"
    assert torch.isfinite(actual).all()
    # All value rows are one, so any finite normalized causal attention is one.
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=0.01, rtol=0.01)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("change", ("tail", "noncontiguous", "window_size", "sink_bias", "g_scalar"))
@pytest.mark.skipif(not test_backend.supports_operator("parallel_wall_attn"), reason=test_backend.skip_reason("parallel_wall_attn"))
@torch.inference_mode()
def test_parallel_wall_attn_real_fallback_matches_naive(change):
    t = 257 if change == "tail" else 256
    q, k, v, g = _make_inputs(t, 2, 64, torch.bfloat16, seed=1)
    kwargs = {}
    if change == "noncontiguous":
        q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not q.is_contiguous()
    elif change == "window_size":
        kwargs[change] = 128
    elif change == "sink_bias":
        kwargs[change] = torch.zeros(8, device=q.device, dtype=q.dtype)
    elif change == "g_scalar":
        kwargs[change] = torch.zeros(1, t, 8, device=q.device, dtype=q.dtype)
    expected = wall_attn.naive_wall_attn(q, k, v, g, scale=64**-0.5, **kwargs)
    actual = parallel_wall_attn(q, k, v, g, **kwargs)
    assert provider.get_last_route() == "official_fallback"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(not test_backend.supports_operator("parallel_wall_attn"), reason=test_backend.skip_reason("parallel_wall_attn"))
def test_parallel_wall_attn_real_fallback_gradients_match_naive(monkeypatch):
    from flag_attn.FLA.wall_attn import naive as naive_provider

    def differentiable_cumsum(x, *, cu_seqlens=None, scale=1.0):
        assert cu_seqlens is None
        return torch.cumsum(x.float(), dim=1) * scale

    monkeypatch.setattr(naive_provider, "chunk_global_cumsum", differentiable_cumsum)
    inputs = _make_inputs(128, 2, 64, torch.bfloat16, seed=7)
    reference_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    candidate_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    expected = wall_attn.naive_wall_attn(*reference_inputs, scale=64**-0.5)
    actual = parallel_wall_attn(*candidate_inputs)
    assert provider.get_last_route() == "official_fallback"
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad)
    actual_grads = torch.autograd.grad(actual, candidate_inputs, grad)
    torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.05)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(not _has_h100_tle(), reason="capture policy test requires H100/SM90 with TLE")
def test_parallel_wall_attn_capture_selects_fallback(monkeypatch):
    inputs = _make_inputs(512, 8, 64, torch.bfloat16, seed=0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert provider.select_route(*inputs) == "official_fallback"
