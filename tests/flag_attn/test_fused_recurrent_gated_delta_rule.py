# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head fused recurrent gated delta rule forward tests."""

import pytest
import torch
import torch.nn.functional as F

from flag_attn import runtime
from flag_attn.testing import backend as test_backend

pytestmark = pytest.mark.fused_recurrent_gated_delta_rule_fwd

ASSERT_RATIO = 0.01
K = V = 128

# Exact activation/state dtype pairs from vLLM's packed-decode reference test.
DECODE_CASES = [
    pytest.param(32, 4, 8, torch.float16, torch.float16, id="N32-H4-HV8-fp16"),
    pytest.param(32, 4, 8, torch.bfloat16, torch.bfloat16, id="N32-H4-HV8-bf16"),
    pytest.param(32, 4, 8, torch.float32, torch.float32, id="N32-H4-HV8-fp32"),
    pytest.param(1, 16, 64, torch.bfloat16, torch.float32, id="N1-H16-HV64-bf16-fp32state"),
    pytest.param(9, 4, 12, torch.bfloat16, torch.float32, id="N9-H4-HV12-bf16-fp32state"),
    pytest.param(1, 4, 12, torch.bfloat16, torch.bfloat16, id="N1-H4-HV12-bf16"),
]


def _load_fwd():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )
    return fused_recurrent_gated_delta_rule_fwd


def _reference(inputs):
    """CPU FP32 recurrence; state ABI is [slots, HV, V, K]."""
    q, k, v, g, beta = [inputs[name].detach().float().cpu() for name in ("q", "k", "v", "g", "beta")]
    state0 = inputs["initial_state"].detach().float().cpu()
    indices = inputs["ssm_state_indices"].detach().cpu()
    B, T, H, K = q.shape
    HV, V = v.shape[-2:]
    cu = inputs["cu_seqlens"]
    spans = list(zip(cu.cpu().tolist()[:-1], cu.cpu().tolist()[1:])) if cu is not None else [
        (b * T, (b + 1) * T) for b in range(B)
    ]
    q, k = q.reshape(-1, H, K), k.reshape(-1, H, K)
    v, g = v.reshape(-1, HV, V), g.reshape(-1, HV)
    beta = beta.reshape(-1, HV, V) if beta.ndim == 4 else beta.reshape(-1, HV, 1)
    out = torch.zeros_like(v)
    final = state0.clone() if inputs["inplace_final_state"] else torch.empty(B * T, HV, V, K)
    valid = torch.zeros(B * T, dtype=torch.bool)
    written = set()
    accepted = inputs.get("num_accepted_tokens")
    accepted = None if accepted is None else accepted.detach().cpu()
    for n, (start, end) in enumerate(spans):
        if start == end:
            continue
        row = indices[n].reshape(-1)
        read_pos = int(accepted[n]) - 1 if accepted is not None else 0
        slot = int(row[read_pos])
        if slot <= 0:
            continue
        state = state0[slot].clone()
        for t, pos in enumerate(range(start, end)):
            qt = q[pos].repeat_interleave(HV // H, dim=0)
            kt = k[pos].repeat_interleave(HV // H, dim=0)
            if inputs["use_qk_l2norm_in_kernel"]:
                qt = qt / torch.sqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            state = state * torch.exp(g[pos])[:, None, None]
            residual = (v[pos] - (state * kt[:, None, :]).sum(-1)) * beta[pos]
            state = state + residual[:, :, None] * kt[:, None, :]
            out[pos] = (state * (qt * inputs["scale"])[:, None, :]).sum(-1)
            valid[pos] = True
            dest = int(row[t]) if inputs["inplace_final_state"] else pos
            if not inputs["inplace_final_state"] or dest > 0:
                final[dest] = state
                written.add(dest)
    return out.reshape(B, T, HV, V), final, valid, sorted(written)


def _assert_close(name, actual, expected):
    # Select tolerances before converting: output and state may differ in dtype.
    atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (2e-2, 1e-2)
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert torch.isfinite(actual).all(), f"{name}: nonfinite actual"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: elementwise check failed\n{message}",
    )
    max_abs = (actual - expected).abs().max().item()
    ratio = (actual - expected).square().mean().sqrt().item() / (
        expected.square().mean().sqrt().item() + 1e-8
    )
    assert max_abs <= 1e-6 or ratio < ASSERT_RATIO, (
        f"{name}: relative_rms={ratio:.7g}, max_abs={max_abs:.7g}, limit={ASSERT_RATIO}"
    )
    return ratio



def _check_case_impl(inputs):
    fwd = _load_fwd()
    expected_o, expected_state, valid, written = _reference(inputs)
    original = inputs["initial_state"].clone()
    actual_o, actual_state = fwd(**inputs)
    test_backend.device_fn.synchronize()
    assert actual_o.shape == expected_o.shape
    assert actual_state.shape == expected_state.shape
    actual_flat = actual_o.reshape(-1, *actual_o.shape[-2:])
    expected_flat = expected_o.reshape(actual_flat.shape)
    output_ratio = 0.0
    for pos in valid.nonzero().flatten().tolist():
        output_ratio = max(output_ratio, _assert_close(f"output[{pos}]", actual_flat[pos], expected_flat[pos]))
    state_ratio = 0.0
    for slot in written:
        state_ratio = max(state_ratio, _assert_close(f"state[{slot}]", actual_state[slot], expected_state[slot]))
    if inputs["inplace_final_state"]:
        assert actual_state.data_ptr() == inputs["initial_state"].data_ptr()
        untouched = [slot for slot in range(len(original)) if slot not in written]
        assert torch.equal(actual_state[untouched], original[untouched]), "untouched state slots changed"
    else:
        assert torch.equal(inputs["initial_state"], original), "non-inplace call modified initial state"
        assert actual_state.data_ptr() != inputs["initial_state"].data_ptr()
    return output_ratio, state_ratio



def _check_case(inputs, record_property):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        output_error, state_error = _check_case_impl(inputs)
    finally:
        torch.set_num_threads(threads)
    record_property("output_relative_rms", output_error)
    record_property("state_relative_rms", state_error)


def _make_decode_inputs(batch, H, HV, dtype, state_dtype, strided_mixed_qkv):
    """Use the ordinary-forward reference inputs from the first vLLM test."""
    device = test_backend.device
    generator = torch.Generator(device=device).manual_seed(0)
    qkv_dim = 2 * H * K + HV * V
    if strided_mixed_qkv:
        projection = torch.randn(
            batch, qkv_dim + 64, device=device, dtype=dtype, generator=generator
        )
        mixed_qkv = projection[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn(
            batch, qkv_dim, device=device, dtype=dtype, generator=generator
        )
    a = torch.randn(batch, HV, device=device, dtype=dtype, generator=generator)
    b = torch.randn(batch, HV, device=device, dtype=dtype, generator=generator)
    A_log = torch.randn(HV, device=device, dtype=dtype, generator=generator)
    dt_bias = torch.randn(HV, device=device, dtype=dtype, generator=generator)
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    if batch > 3:
        indices[-3:] = -1
        indices[-1] = 0
    state = torch.randn(
        batch + 1, HV, V, K, device=device, dtype=state_dtype, generator=generator
    )
    q, k, v = torch.split(mixed_qkv, (H * K, H * K, HV * V), dim=-1)
    # Upstream materializes all three reference inputs, even when the packed
    # projection is strided. Do not add a separate Q/K/V-view test here.
    q = q.view(batch, H, K).unsqueeze(1).contiguous()
    k = k.view(batch, H, K).unsqueeze(1).contiguous()
    v = v.view(batch, HV, V).unsqueeze(1).contiguous()
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)
    return dict(q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
                initial_state=state, inplace_final_state=True, cu_seqlens=None,
                ssm_state_indices=indices, num_accepted_tokens=None,
                use_qk_l2norm_in_kernel=True)


def _make_gating_reference_inputs(num_reqs, dtype, num_speculative_tokens=None):
    """Mirror ordinary-forward reference construction from vLLM's gating tests."""
    torch.manual_seed(0)
    device = test_backend.device
    H, HV = 16, 32  # upstream tp_size=1
    length = 1 if num_speculative_tokens is None else num_speculative_tokens + 1
    tokens = num_reqs * length
    entries = tokens * 2
    mixed_qkv = torch.rand(tokens, 2 * H * K + HV * V, device=device, dtype=dtype)
    q, k, v = torch.split(mixed_qkv, (H * K, H * K, HV * V), dim=-1)
    q, k, v = q.view(1, tokens, H, K), k.view(1, tokens, H, K), v.view(1, tokens, HV, V)
    A_log = torch.rand(HV, device=device, dtype=dtype)
    dt_bias = torch.rand(HV, device=device, dtype=dtype)
    a = torch.rand(tokens, HV, device=device, dtype=dtype)
    b = torch.rand(tokens, HV, device=device, dtype=dtype)
    state = torch.rand(entries + 1, HV, V, K, device=device, dtype=dtype)
    indices = (torch.randperm(entries, device=device, dtype=torch.int32) + 1)[:tokens]
    accepted = None
    if num_speculative_tokens is not None:
        indices = indices.view(num_reqs, length)
        # Preserve upstream's exclusive upper bound, not randint(1, length+1).
        accepted = torch.randint(
            1, num_speculative_tokens + 1, (num_reqs,), device=device, dtype=torch.int32
        )
    cu = torch.arange(0, tokens + 1, length, device=device, dtype=torch.int32)
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return dict(q=q, k=k, v=v, g=g.unsqueeze(0), beta=beta.unsqueeze(0),
                scale=K**-0.5, initial_state=state, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=indices, num_accepted_tokens=accepted,
                use_qk_l2norm_in_kernel=True)


@pytest.mark.parametrize("strided_mixed_qkv", [False, True], ids=["packed-contiguous", "packed-strided"])
@pytest.mark.parametrize("batch,H,HV,dtype,state_dtype", DECODE_CASES)
@torch.inference_mode()
def test_decode_from_vllm_reference(batch, H, HV, dtype, state_dtype, strided_mixed_qkv, record_property):
    _load_fwd()
    inputs = _make_decode_inputs(batch, H, HV, dtype, state_dtype, strided_mixed_qkv)
    _check_case(inputs, record_property)


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.inference_mode()
def test_non_spec_from_vllm_gating_reference(num_reqs, dtype, record_property):
    _load_fwd()
    inputs = _make_gating_reference_inputs(num_reqs, dtype)
    _check_case(inputs, record_property)


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("num_speculative_tokens", [1, 3], ids=["length2", "length4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.inference_mode()
def test_spec_from_vllm_gating_reference(num_reqs, num_speculative_tokens, dtype, record_property):
    _load_fwd()
    inputs = _make_gating_reference_inputs(num_reqs, dtype, num_speculative_tokens)
    _check_case(inputs, record_property)


@pytest.mark.parametrize("layout", ["fixed-batch2-length3", "packed-lengths1-and3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.inference_mode()
def test_non_inplace_returns_every_token_state(layout, dtype, record_property):
    """Non-inplace output covers all tokens and leaves the state pool intact."""
    _load_fwd()
    device = test_backend.device
    generator = torch.Generator(device=device).manual_seed(0)
    H, HV = 2, 4
    if layout == "fixed-batch2-length3":
        batch, length = 2, 3
        cu = None
    else:
        batch, length = 1, 4
        cu = torch.tensor([0, 1, 4], device=device, dtype=torch.int32)
    q = torch.randn(batch, length, H, K, device=device, dtype=dtype, generator=generator)
    k = torch.randn(batch, length, H, K, device=device, dtype=dtype, generator=generator)
    v = torch.randn(batch, length, HV, V, device=device, dtype=dtype, generator=generator)
    g = -torch.rand(batch, length, HV, device=device, dtype=torch.float32, generator=generator)
    beta = torch.rand(batch, length, HV, device=device, dtype=dtype, generator=generator)
    state = torch.randn(3, HV, V, K, device=device, dtype=dtype, generator=generator)
    indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
    inputs = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
        initial_state=state, inplace_final_state=False, cu_seqlens=cu,
        ssm_state_indices=indices, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )
    _check_case(inputs, record_property)
