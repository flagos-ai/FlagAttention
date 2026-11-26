# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""T-Head packed GDN decode tests.

PLA GDN shapes, seed=42 and activation-dtype tolerances, adapted from
t-head/pla@3065b5a7238b5023f92bbeda534fca8bde3e3675,
scripts/decode/test_fused_sigmoid_gating_recurrent_k_last_packed.py.
Reference remains explicit QKV/gating followed by the ordinary forward.
Only GDN cases apply to this operator; existing layout/regression tests remain.
"""

import pytest
import torch

from flag_attn import runtime
from flag_attn.testing import backend as test_backend


pytestmark = pytest.mark.chunk_gated_delta_rule


@pytest.fixture(scope="module")
def thead_ops():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent_packed_decode import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    return fused_recurrent_gated_delta_rule_fwd, fused_recurrent_gated_delta_rule_packed_decode


def _assert_close(name, actual, expected, dtype=None):
    dtype = actual.dtype if dtype is None else dtype
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 1e-2)
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: {message}",
    )

@pytest.mark.parametrize("strided_mixed_qkv", [False, True], ids=["contiguous", "strided"])
@pytest.mark.parametrize(
    "batch,H,HV,dtype,state_dtype",
    [
        pytest.param(32, 4, 8, torch.float16, torch.float16, id="B32-H4-HV8-fp16"),
        pytest.param(32, 4, 8, torch.bfloat16, torch.bfloat16, id="B32-H4-HV8-bf16"),
        pytest.param(32, 4, 8, torch.float32, torch.float32, id="B32-H4-HV8-fp32"),
        pytest.param(1, 16, 64, torch.bfloat16, torch.float32, id="B1-H16-HV64-bf16-fp32state"),
        pytest.param(9, 4, 12, torch.bfloat16, torch.float32, id="B9-H4-HV12-bf16-fp32state"),
        pytest.param(1, 4, 12, torch.bfloat16, torch.bfloat16, id="B1-H4-HV12-bf16"),
    ],
)
@torch.inference_mode()
def test_fused_recurrent_packed_decode_matches_reference(
    thead_ops, batch, H, HV, dtype, state_dtype, strided_mixed_qkv,
):
    ordinary, packed = thead_ops
    device = test_backend.device
    generator = torch.Generator(device=device).manual_seed(0)
    K = V = 128
    qkv_dim = 2 * H * K + HV * V
    if strided_mixed_qkv:
        projection = torch.randn(
            batch, qkv_dim + 64, device=device, dtype=dtype, generator=generator,
        )
        mixed_qkv = projection[:, :qkv_dim]
        assert mixed_qkv.stride(0) == qkv_dim + 64
    else:
        mixed_qkv = torch.randn(
            batch, qkv_dim, device=device, dtype=dtype, generator=generator,
        )
    a = torch.randn(batch, HV, device=device, dtype=dtype, generator=generator)
    b = torch.randn(batch, HV, device=device, dtype=dtype, generator=generator)
    A_log = torch.randn(HV, device=device, dtype=dtype, generator=generator)
    dt_bias = torch.randn(HV, device=device, dtype=dtype, generator=generator)
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    if batch > 3:
        indices[-3:] = -1
        indices[-1] = 0
    state0 = torch.randn(
        batch + 1, HV, V, K, device=device, dtype=state_dtype, generator=generator,
    )
    state_ref, state_packed = state0.clone(), state0.clone()
    # NaN sentinels expose output rows that the packed implementation fails to write.
    out_packed = torch.full((batch, 1, HV, V), float("nan"), device=device, dtype=dtype)

    q, k, v = torch.split(mixed_qkv, (H * K, H * K, HV * V), dim=-1)
    q = q.view(batch, H, K).unsqueeze(1).contiguous()
    k = k.view(batch, H, K).unsqueeze(1).contiguous()
    v = v.view(batch, HV, V).unsqueeze(1).contiguous()
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x,
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)
    out_ref, final_ref = ordinary(
        q=q, k=k, v=v, g=g, beta=beta, scale=K**-0.5,
        initial_state=state_ref, inplace_final_state=True,
        cu_seqlens=None, ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    returned_out, returned_state = packed(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        scale=K**-0.5, initial_state=state_packed, out=out_packed,
        ssm_state_indices=indices, use_qk_l2norm_in_kernel=True,
    )
    test_backend.device_fn.synchronize()
    valid = indices > 0
    _assert_close("output", out_packed[valid], out_ref[valid])
    _assert_close("state", state_packed, final_ref, dtype)
    # Ordinary forward leaves invalid outputs undefined; packed decode writes zero.
    assert torch.count_nonzero(out_packed[~valid]).item() == 0
    touched = torch.zeros(batch + 1, device=device, dtype=torch.bool)
    touched[indices[valid].long()] = True
    assert torch.equal(state_packed[~touched], state0[~touched]), "unused state slots changed"
    assert returned_out.data_ptr() == out_packed.data_ptr()
    assert returned_state.data_ptr() == state_packed.data_ptr()


@torch.inference_mode()
def test_packed_decode_keeps_beta_in_fp32(thead_ops):
    _, packed = thead_ops
    device = test_backend.device
    dtype = torch.bfloat16
    mixed_qkv = torch.ones(1, 3, device=device, dtype=dtype)
    a = torch.zeros(1, 1, device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros(1, device=device, dtype=dtype)
    indices = torch.ones(1, device=device, dtype=torch.int32)
    state = torch.zeros(2, 1, 1, 1, device=device, dtype=torch.float32)
    out = torch.full((1, 1, 1, 1), float("nan"), device=device, dtype=dtype)
    packed(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=params, dt_bias=params,
        scale=1.0, initial_state=state, out=out, ssm_state_indices=indices,
    )
    # With k=v=1, zero state and no normalization, state equals sigmoid(b.fp32).
    torch.testing.assert_close(
        state[1, 0, 0, 0], torch.sigmoid(b.float()).squeeze(), atol=1e-6, rtol=1e-6,
    )
    assert torch.count_nonzero(state[0]).item() == 0


@torch.inference_mode()
def test_packed_decode_supports_large_batch_head_grid(thead_ops):
    _, packed = thead_ops
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = test_backend.device
    # All slots are null: the kernel must return before reading these inputs.
    gates = torch.empty(B, HV, device=device)
    params = torch.empty(HV, device=device)
    state = torch.zeros(1, HV, V, K, device=device)
    out = torch.full((B, 1, HV, V), float("nan"), device=device)
    packed(
        mixed_qkv=torch.empty(B, 2 * H * K + HV * V, device=device),
        a=gates, b=gates, A_log=params, dt_bias=params, scale=1.0,
        initial_state=state, out=out,
        ssm_state_indices=torch.zeros(B, device=device, dtype=torch.int32),
    )
    assert torch.count_nonzero(out).item() == 0
    assert torch.count_nonzero(state).item() == 0


# Exact GDN configurations from PLA run_correctness(), including its repeats.
PLA_GDN_SHAPES = (
    (1, 8, 16, 128, 128, 32),
    (4, 8, 16, 128, 128, 32),
    (16, 8, 16, 128, 128, 64),
    (32, 8, 16, 128, 128, 128),
    (64, 8, 16, 128, 128, 128),
    (128, 8, 16, 128, 128, 256),
    (256, 8, 16, 128, 128, 512),
    (1, 16, 32, 128, 128, 32),
    (32, 16, 32, 128, 128, 128),
    (64, 16, 32, 128, 128, 128),
    (32, 16, 16, 128, 128, 128),
    (64, 16, 16, 128, 128, 128),
    (32, 8, 16, 128, 128, 128),
    (1, 8, 16, 128, 128, 32),
    (2, 8, 16, 128, 128, 32),
)
PLA_PAD_SHAPES = (
    (32, 8, 16, 128, 128, 128, 8),
    (16, 16, 32, 128, 128, 64, 4),
    (4, 8, 16, 128, 128, 32, 2),
    (1, 8, 16, 128, 128, 32, 1),
)
PLA_BF16_STATE_SHAPES = (
    (4, 8, 16, 128, 128, 32),
    (32, 8, 16, 128, 128, 128),
    (64, 16, 32, 128, 128, 128),
    (128, 8, 16, 128, 128, 256),
)


def _make_pla_inputs(shape, state_dtype=torch.float32):
    batch, h, hv, k, v, pool_size = shape
    torch.manual_seed(42)
    device = test_backend.device
    dtype = torch.bfloat16
    mixed = torch.randn(batch, 2 * h * k + hv * v, device=device, dtype=dtype)
    b = torch.randn(batch, hv, device=device, dtype=dtype)
    alog = torch.randn(hv, device=device, dtype=torch.float32)
    a = torch.randn(batch, hv, device=device, dtype=dtype)
    bias = torch.randn(hv, device=device, dtype=dtype)
    state = torch.randn(pool_size, hv, v, k, device=device, dtype=state_dtype) * 0.1
    return dict(
        mixed_qkv=mixed, a=a, b=b, A_log=alog, dt_bias=bias,
        initial_state=state,
        # Same unique-slot workload; slot zero is reserved by FlagAttention.
        ssm_state_indices=torch.arange(1, batch + 1, device=device, dtype=torch.int32),
    )


def _run_pla_case(ops, data, shape, ordinary=False):
    batch, h, hv, k, v, _ = shape
    state = data["initial_state"].clone()
    if ordinary:
        q, key, value = torch.split(data["mixed_qkv"], (h * k, h * k, hv * v), dim=-1)
        g = -data["A_log"].float().exp() * torch.nn.functional.softplus(
            data["a"].float() + data["dt_bias"].float(), beta=1.0, threshold=20.0,
        )
        return ops[0](
            q=q.reshape(batch, 1, h, k).contiguous(),
            k=key.reshape(batch, 1, h, k).contiguous(),
            v=value.reshape(batch, 1, hv, v).contiguous(),
            g=g.unsqueeze(1), beta=data["b"].float().sigmoid().unsqueeze(1),
            scale=k ** -0.5, initial_state=state, inplace_final_state=True,
            ssm_state_indices=data["ssm_state_indices"], use_qk_l2norm_in_kernel=True,
        )
    out = torch.full((batch, 1, hv, v), float("nan"), device=state.device, dtype=data["mixed_qkv"].dtype)
    result = ops[1](
        **{name: value for name, value in data.items() if name != "initial_state"},
        initial_state=state, out=out, scale=k ** -0.5, use_qk_l2norm_in_kernel=True,
    )
    assert result[0].data_ptr() == out.data_ptr()
    assert result[1].data_ptr() == state.data_ptr()
    return result


def _check_pla_pair(ops, data, shape):
    expected, ref_state = _run_pla_case(ops, data, shape, ordinary=True)
    actual, state = _run_pla_case(ops, data, shape)
    test_backend.device_fn.synchronize()
    valid = data["ssm_state_indices"] > 0
    indices = data["ssm_state_indices"][valid].long()
    _assert_close("PLA GDN output", actual[valid], expected[valid])
    # Match official state tolerance to activation dtype, not FP32 state dtype.
    _assert_close("PLA GDN state", state[indices], ref_state[indices], actual.dtype)
    assert torch.count_nonzero(actual[~valid]).item() == 0
    untouched = torch.ones(state.shape[0], device=state.device, dtype=torch.bool)
    untouched[indices] = False
    assert torch.equal(state[untouched], data["initial_state"][untouched])


@pytest.mark.parametrize("shape", PLA_GDN_SHAPES)
@torch.inference_mode()
def test_packed_decode_pla_gdn_shapes(thead_ops, shape):
    _check_pla_pair(thead_ops, _make_pla_inputs(shape), shape)


@pytest.mark.parametrize("case", PLA_PAD_SHAPES)
@torch.inference_mode()
def test_packed_decode_pla_padding(thead_ops, case):
    shape, padded = case[:6], case[6]
    data = _make_pla_inputs(shape)
    data["ssm_state_indices"][-padded:] = -1
    _check_pla_pair(thead_ops, data, shape)


@pytest.mark.parametrize("shape", PLA_BF16_STATE_SHAPES)
@torch.inference_mode()
def test_packed_decode_pla_bf16_state(thead_ops, shape):
    # Official BF16-state test compares the same packed operator in both modes.
    fp32 = _make_pla_inputs(shape, torch.float32)
    bf16 = _make_pla_inputs(shape, torch.bfloat16)
    expected, ref_state = _run_pla_case(thead_ops, fp32, shape)
    actual, state = _run_pla_case(thead_ops, bf16, shape)
    indices = bf16["ssm_state_indices"].long()
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(state[indices], ref_state[indices].to(torch.bfloat16), atol=5e-2, rtol=2e-2)


@torch.inference_mode()
def test_packed_decode_pla_bf16_state_padding(thead_ops):
    shape = (8, 8, 16, 128, 128, 32)
    data = _make_pla_inputs(shape, torch.bfloat16)
    data["ssm_state_indices"][-3:] = -1
    fp32 = dict(data, initial_state=data["initial_state"].float())
    expected, ref_state = _run_pla_case(thead_ops, fp32, shape)
    actual, state = _run_pla_case(thead_ops, data, shape)
    valid = data["ssm_state_indices"] > 0
    indices = data["ssm_state_indices"][valid].long()
    torch.testing.assert_close(actual[valid], expected[valid], atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(state[indices], ref_state[indices].to(torch.bfloat16), atol=5e-2, rtol=2e-2)
    assert torch.count_nonzero(actual[~valid]).item() == 0
    untouched = torch.ones(state.shape[0], device=state.device, dtype=torch.bool)
    untouched[indices] = False
    assert torch.equal(state[untouched], data["initial_state"][untouched])
