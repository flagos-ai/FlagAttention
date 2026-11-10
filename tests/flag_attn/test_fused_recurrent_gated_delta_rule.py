# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""T-Head recurrent GDN correctness, adapted from vLLM-SAIL.

Source: vllm-project/vllm-sail@95c24b3c21a132a4987f7daba99b69fccf7d6190
tests/e2e/gdn/test_gdn_pla_decode.py (update and state-isolation cases).
Precompute g/beta for FlagAttention; replace plugin dispatch mocks with an
independent CPU FP32 recurrence. Retain the existing non-inplace regression.
All test inputs use the source suite's dense state and contiguous index tables.
"""

import pytest
import torch
import torch.nn.functional as F

from flag_attn import runtime
from flag_attn.testing import backend as test_backend

pytestmark = pytest.mark.chunk_gated_delta_rule
H, HV = 4, 8
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
OUTPUT_TOL = {torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-3, 1.6e-2), torch.float32: (1e-5, 1.3e-6)}


@pytest.fixture(scope="module")
def fwd():
    if not test_backend.is_available() or runtime.device.vendor_name != "thead":
        pytest.skip("requires the T-Head accelerator")
    from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd,
    )
    return fused_recurrent_gated_delta_rule_fwd


@pytest.fixture(scope="module", autouse=True)
def cpu_reference_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _inputs(lengths, dtype, *, state_dtype=torch.float32, dim=128, seed=0, slots=None):
    """Keep the official tensor generation order, scales, and null-slot convention."""
    torch.manual_seed(seed)
    device = test_backend.device
    batch, total = len(lengths), sum(lengths)
    slots = slots or batch + 1
    def rand(shape, dt=dtype):
        return torch.randn(shape, dtype=dt, device=device)
    q, k = rand((1, total, H, dim)) * 0.1, rand((1, total, H, dim)) * 0.1
    v = rand((1, total, HV, dim)) * 0.1
    a, b = rand((total, HV)) * 0.1, rand((total, HV)) * 0.1
    a_log, bias = rand((HV,), torch.float32) * 0.1, rand((HV,)) * 0.1
    state = rand((slots, HV, dim, dim), state_dtype) * 0.05
    indices = (torch.randperm(slots - 1)[:batch] + 1).to(device=device, dtype=torch.int32)
    return dict(
        q=q, k=k, v=v,
        g=(-a_log.exp() * F.softplus(a.float() + bias.float())).unsqueeze(0),
        beta=b.float().sigmoid().unsqueeze(0), scale=dim**-0.5,
        initial_state=state, inplace_final_state=True,
        cu_seqlens=torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), device=device, dtype=torch.int32),
        ssm_state_indices=indices, num_accepted_tokens=None, use_qk_l2norm_in_kernel=True,
    )


def _reference(data):
    """CPU FP32 math; keep the accumulator unrounded within each kernel call."""
    q, k, v, g, beta = [data[name].detach().float().cpu() for name in ("q", "k", "v", "g", "beta")]
    batch, length = q.shape[:2]
    dim = q.shape[-1]
    q, k = q.reshape(-1, H, dim), k.reshape(-1, H, dim)
    v, g, beta = v.reshape(-1, HV, dim), g.reshape(-1, HV), beta.reshape(-1, HV)
    initial = data["initial_state"].detach().float().cpu()
    indices = data["ssm_state_indices"].cpu()
    accepted = data["num_accepted_tokens"]
    accepted = None if accepted is None else accepted.cpu()
    cu = data["cu_seqlens"]
    bounds = cu.cpu().tolist() if cu is not None else list(range(0, batch * length + 1, length))
    out = torch.zeros_like(v)
    final = initial.clone() if data["inplace_final_state"] else torch.empty(batch * length, HV, dim, dim)
    valid = torch.zeros(batch * length, dtype=torch.bool)
    written = set()
    for n, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        row = indices[n].reshape(-1)
        slot = int(row[int(accepted[n]) - 1 if accepted is not None else 0])
        if slot <= 0 or start == end:
            continue
        state = initial[slot].clone()
        for step, token in enumerate(range(start, end)):
            qt, kt = q[token], k[token]
            if data["use_qk_l2norm_in_kernel"]:
                qt = qt / torch.sqrt(qt.square().sum(-1, keepdim=True) + 1e-6)
                kt = kt / torch.sqrt(kt.square().sum(-1, keepdim=True) + 1e-6)
            qt = qt.repeat_interleave(HV // H, dim=0) * data["scale"]
            kt = kt.repeat_interleave(HV // H, dim=0)
            state = state * g[token].exp()[:, None, None]
            residual = (v[token] - (state * kt[:, None, :]).sum(-1)) * beta[token, :, None]
            state = state + residual[:, :, None] * kt[:, None, :]
            out[token] = (state * qt[:, None, :]).sum(-1)
            valid[token] = True
            dest = int(row[step]) if data["inplace_final_state"] else token
            if dest > 0 or not data["inplace_final_state"]:
                final[dest] = state
                written.add(dest)
    return out.reshape_as(data["v"]), final, valid, written


@torch.inference_mode()
def _check(fwd, data):
    expected_out, expected_state, valid, written = _reference(data)
    original = data["initial_state"].clone()
    out, state = fwd(**data)
    test_backend.device_fn.synchronize()
    assert out.shape == expected_out.shape and out.dtype == data["v"].dtype
    assert state.shape == expected_state.shape and state.dtype == original.dtype
    atol, rtol = OUTPUT_TOL[out.dtype]
    torch.testing.assert_close(out.float().cpu().reshape(-1, HV, out.shape[-1])[valid],
                               expected_out.reshape(-1, HV, out.shape[-1])[valid], atol=atol, rtol=rtol)
    # State tolerances follow the official PLA/Triton parity test.
    torch.testing.assert_close(state.float().cpu(), expected_state, atol=1e-3, rtol=1e-2)
    if data["inplace_final_state"]:
        assert state.data_ptr() == data["initial_state"].data_ptr()
        untouched = [slot for slot in range(len(original)) if slot not in written]
        assert torch.equal(state[untouched], original[untouched])
    else:
        assert state.data_ptr() != data["initial_state"].data_ptr()
        assert torch.equal(data["initial_state"], original)
    return out, state


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lengths,state_dtype,dim,seed", [
    pytest.param([1, 1, 1, 1], torch.float32, 128, 4, id="decode"),
    pytest.param([2, 1, 3], torch.float32, 128, 6, id="varlen"),
    pytest.param([1, 1, 1, 1], torch.bfloat16, 128, 11, id="bf16-state"),
    pytest.param([1, 1, 1, 1], torch.float32, 64, 13, id="dim64"),
])
def test_update(fwd, dtype, lengths, state_dtype, dim, seed):
    data = _inputs(lengths, dtype, state_dtype=state_dtype, dim=dim, seed=seed)
    # The ordinary vLLM kernel writes one index per token. Repeat each request's
    # slot in a real contiguous table, exactly as the official Triton reference.
    data["ssm_state_indices"] = data["ssm_state_indices"][:, None].repeat(1, max(lengths))
    _check(fwd, data)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("accepted_dtype", (torch.int32, torch.int64))
def test_speculative_update(fwd, dtype, accepted_dtype):
    data = _inputs([2, 2, 2], dtype, slots=7, seed=12)
    data["ssm_state_indices"] = (torch.randperm(6).reshape(3, 2) + 1).to(
        device=test_backend.device, dtype=torch.int32)
    data["num_accepted_tokens"] = torch.tensor([1, 2, 1], device=test_backend.device, dtype=accepted_dtype)
    _check(fwd, data)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("null_slot", (0, -1))
def test_null_state_isolation(fwd, dtype, null_slot):
    data = _inputs([1] * 4, dtype, slots=8, seed=14)
    initial = data["initial_state"].clone()
    data["ssm_state_indices"][1] = null_slot
    out, state = _check(fwd, data)
    keep = [0, 2, 3]
    compact = dict(data, initial_state=initial,
                   ssm_state_indices=data["ssm_state_indices"][keep].clone(),
                   cu_seqlens=torch.arange(4, device=test_backend.device, dtype=torch.int32))
    for name in ("q", "k", "v", "g", "beta"):
        compact[name] = data[name][:, keep].contiguous()
    ref_out, ref_state = _check(fwd, compact)
    # Invalid-request output is undefined; compare only the real requests.
    torch.testing.assert_close(out[:, keep], ref_out, atol=0, rtol=0)
    assert torch.equal(state, ref_state)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fixed_length", (False, True), ids=("packed", "fixed"))
def test_non_inplace_returns_every_token_state(fwd, dtype, fixed_length):
    data = _inputs([3, 3] if fixed_length else [1, 3], dtype, seed=15)
    if fixed_length:
        for name in ("q", "k", "v", "g", "beta"):
            tensor = data[name]
            data[name] = tensor.reshape(2, 3, *tensor.shape[2:])
        data["cu_seqlens"] = None
    data["inplace_final_state"] = False
    _check(fwd, data)
