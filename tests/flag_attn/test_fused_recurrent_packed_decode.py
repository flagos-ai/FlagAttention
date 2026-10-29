# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""T-Head packed GDN decode tests.

Reference: explicit QKV and gating, then the ordinary forward.
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


def _assert_close(name, actual, expected):
    # Preserve the ordinary-forward harness's per-tensor dtype policy.
    atol, rtol = (1e-4, 1e-4) if actual.dtype == torch.float32 else (2e-2, 1e-2)
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all(), f"{name}: nonfinite actual"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    torch.testing.assert_close(
        actual, expected, atol=atol, rtol=rtol,
        msg=lambda message: f"{name}: elementwise check failed\n{message}",
    )
    diff = actual - expected
    ratio = diff.square().mean().sqrt().item() / (
        expected.square().mean().sqrt().item() + 1e-8
    )
    assert diff.abs().max().item() <= 1e-6 or ratio < 0.01, (
        f"{name}: relative_rms={ratio:.7g}, limit=0.01"
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
    _assert_close("state", state_packed, final_ref)
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
