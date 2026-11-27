# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent Torch KDA recurrence and inputs shared by the contract tests."""

import torch
import torch.nn.functional as F

from flag_attn.testing import backend as test_backend

LONG_CASES = {
    "fixed-8192": [8192],
    "varlen-skewed": [1300, 547, 2048, 963, 271, 3063],
    "varlen-uniform": [1024] * 8,
}


def make_case(
    *,
    B=1,
    T=33,
    H=2,
    HV=None,
    K=128,
    V=128,
    dtype=torch.bfloat16,
    lengths=None,
    state_v_first=True,
    state_dtype=torch.float32,
    initial=True,
    final=True,
    noncontiguous=False,
    safe_gate=True,
    chunk_size=16,
    cu_dtype=torch.int64,
    bias=True,
):
    torch.manual_seed(42)
    HV = H if HV is None else HV
    if lengths is not None:
        B, T = 1, sum(lengths)
    N = B if lengths is None else len(lengths)
    device = test_backend.device

    def layout(tensor):
        if not noncontiguous:
            return tensor
        storage = torch.empty((*tensor.shape[:-1], tensor.shape[-1] * 2), device=device, dtype=tensor.dtype)
        view = storage[..., ::2]
        view.copy_(tensor)
        return view

    q, k = [layout(torch.randn(B, T, H, K, device=device, dtype=dtype) / K**0.5) for _ in range(2)]
    v = layout(torch.randn(B, T, HV, V, device=device, dtype=dtype))
    g = layout(torch.randn(B, T, HV, K, device=device, dtype=dtype))
    beta = layout(torch.randn(B, T, HV, device=device, dtype=dtype))
    A_log = layout(torch.rand(HV, device=device, dtype=torch.float32))
    dt_bias = layout(torch.rand(HV, K, device=device, dtype=torch.float32)) if bias else None
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    state = layout(torch.randn(state_shape, device=device, dtype=state_dtype)) if initial else None
    cu = None
    if lengths is not None:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu = torch.tensor(offsets, device=device, dtype=cu_dtype)
    return (q, k, v, g, beta), dict(
        scale=K**-0.5,
        initial_state=state,
        output_final_state=final,
        state_v_first=state_v_first,
        cu_seqlens=cu,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=-5.0 if safe_gate else None,
        A_log=A_log,
        dt_bias=dt_bias,
    )


def reference(args, kwargs):
    q, k, v, g, beta = [x.float() for x in args]
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(HV // H, dim=2).reshape(B * T, HV, K)
    k = k.repeat_interleave(HV // H, dim=2).reshape(B * T, HV, K)
    if kwargs.get("dt_bias") is not None:
        g = g + kwargs["dt_bias"].float().view(HV, K)
    A = kwargs["A_log"].float().exp().view(HV, 1)
    if kwargs["safe_gate"]:
        g = kwargs["lower_bound"] * (g * A).sigmoid()
    else:
        g = -A * F.softplus(g)
    decay = g.exp().reshape(B * T, HV, K)
    beta = beta.sigmoid().reshape(B * T, HV)
    v = v.reshape(B * T, HV, V)
    output = torch.empty_like(v)
    cu = kwargs.get("cu_seqlens")
    offsets = [i * T for i in range(B + 1)] if cu is None else cu.cpu().tolist()
    original = kwargs.get("initial_state")
    if original is not None:
        original = original.float()
        if kwargs["state_v_first"]:
            original = original.transpose(-1, -2)
    final = []
    scale = K**-0.5 if kwargs.get("scale") is None else kwargs["scale"]
    for n, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = q.new_zeros(HV, K, V) if original is None else original[n].clone()
        for token in range(start, end):
            state = state * decay[token, :, :, None]
            residual = (v[token] - (state * k[token, :, :, None]).sum(-2)) * beta[token, :, None]
            state = state + k[token, :, :, None] * residual[:, None, :]
            output[token] = (state * q[token, :, :, None]).sum(-2) * scale
        final.append(state)
    state = None
    if kwargs["output_final_state"]:
        state = torch.stack(final)
        if kwargs["state_v_first"]:
            state = state.transpose(-1, -2).contiguous()
    return output.reshape(B, T, HV, V).to(args[2].dtype), state


def assert_close(actual, expected):
    for result, target in zip(actual, expected):
        if target is None:
            assert result is None
            continue
        assert result.shape == target.shape and result.dtype == target.dtype
        assert torch.isfinite(result).all() and torch.isfinite(target).all()
        if not result.numel():
            continue
        error = (result.float() - target.float()).square().mean().sqrt()
        magnitude = target.float().square().mean().sqrt()
        assert error <= 1e-6 or error / (magnitude + 1e-8) < 0.005
