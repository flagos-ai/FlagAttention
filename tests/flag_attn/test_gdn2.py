"""NVIDIA GDN2 against an independent FP32 recurrent reference."""

import pytest
import torch
import torch.nn.functional as F

from flag_attn import chunk_gdn2
from flag_attn.runtime.backend import get_backend_name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_backend_name() != "nvidia",
    reason="NVIDIA CUDA is required",
)


def recurrent_reference(q, k, v, g, b, w, initial_state=None, scale=None):
    q, k, v, g, b, w = (x.float() for x in (q, k, v, g, b, w))
    B, T, H, K = q.shape
    state = torch.zeros(B, H, K, v.shape[-1], device=q.device)
    if initial_state is not None:
        state = initial_state.clone()
    scale = K**-0.5 if scale is None else scale
    out = []
    for t in range(T):
        state = state * g[:, t].exp().unsqueeze(-1)
        erased = ((b[:, t] * k[:, t]).unsqueeze(-1) * state).sum(-2)
        residual = w[:, t] * v[:, t] - erased
        state = state + k[:, t].unsqueeze(-1) * residual.unsqueeze(-2)
        out.append((q[:, t].unsqueeze(-1) * state).sum(-2) * scale)
    return torch.stack(out, dim=1), state


def assert_relative_rms(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected).square().mean().sqrt()
    rms = expected.square().mean().sqrt()
    assert error / rms.clamp_min(1e-8) < 0.01


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("T,K,V", [(16, 64, 64), (33, 128, 128), (129, 256, 512)])
@pytest.mark.parametrize("initial,state_v_first", [(False, False), (True, False), (True, True)])
@torch.inference_mode()
def test_gdn2_recurrence(dtype, T, K, V, initial, state_v_first):
    torch.manual_seed(42)
    B, H = 2, 2
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) / K**0.5
    k = torch.randn_like(q) / K**0.5
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    g = -torch.rand_like(q) * 0.1
    b, w = torch.rand_like(q), torch.rand_like(v)
    h0 = torch.randn(B, H, K, V, device="cuda") * 0.1 if initial else None
    expected, expected_state = recurrent_reference(q, k, v, g, b, w, h0)
    state_input = h0.transpose(-1, -2).contiguous() if initial and state_v_first else h0
    actual, state = chunk_gdn2(q, k, v, g, b, w, initial_state=state_input,
                               output_final_state=True, state_v_first=state_v_first)
    if state_v_first:
        state = state.transpose(-1, -2)
    assert_relative_rms(actual, expected)
    assert_relative_rms(state, expected_state)


@pytest.mark.parametrize("lower_bound", [None, -3.0])
@torch.inference_mode()
def test_gdn2_fused_gate_and_normalization(lower_bound):
    torch.manual_seed(7)
    q = torch.randn(1, 33, 2, 64, device="cuda", dtype=torch.float16)
    k, v, g = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    b, w = torch.rand_like(q), torch.rand_like(q)
    A_log = torch.zeros(2, device="cuda")
    bias = torch.randn(2, 64, device="cuda")
    activated = (-F.softplus(g.float() + bias) if lower_bound is None
                 else lower_bound * torch.sigmoid(g.float() + bias))
    expected, expected_state = recurrent_reference(
        F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1),
        v, activated, b, w,
    )
    actual, state = chunk_gdn2(
        q, k, v, g, b, w, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        A_log=A_log, dt_bias=bias, lower_bound=lower_bound,
        safe_gate=lower_bound is not None, output_final_state=True,
    )
    assert_relative_rms(actual, expected)
    assert_relative_rms(state, expected_state)
