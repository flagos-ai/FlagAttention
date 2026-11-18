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

"""Accuracy tests for the generic pure-Triton chunk KDA forward.

test_chunk_kda.py covers the Enflame implementation and skips everywhere else,
so these tests cover the generic implementation that every other accelerator
resolves to. They use plain Triton only and therefore run on a Triton without
``triton.experimental.tle``.
"""

import pytest
import torch
import torch.nn.functional as F

from flag_attn.FLA.chunk_kda import chunk_kda_fwd_infer_triton
from flag_attn.testing import backend as test_backend

LOWER_BOUND = -5.0
# bfloat16 inputs through a chunked recurrence; the reference runs in fp32.
RELATIVE_TOLERANCE = 0.02

pytestmark = [
    pytest.mark.chunk_kda,
    pytest.mark.skipif(
        not test_backend.is_available(),
        reason="chunk KDA kernels require an available accelerator",
    ),
]


def _gate(g: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    H = g.shape[-2]
    g = g.float() + dt_bias.view(H, -1)
    return LOWER_BOUND * torch.sigmoid(A_log.view(H, 1).float().exp() * g)


def _recurrent_reference(q, k, v, g, beta, scale, initial_state, output_final_state):
    """Token-by-token KDA recurrence in fp32, used as ground truth."""
    B, T, H, K = q.shape
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * scale
    state = q.new_zeros(B, H, K, v.shape[-1])
    if initial_state is not None:
        state = state + initial_state
    out = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, beta_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        state = state * g_i[..., None].exp()
        state = state + torch.einsum(
            "b h k, b h v -> b h k v",
            beta_i[..., None] * k_i,
            v_i - (k_i[..., None] * state).sum(-2),
        )
        out[:, i] = torch.einsum("b h k, b h k v -> b h v", q_i, state)
    return out, (state if output_final_state else None)


def _reference(q, k, v, g, beta, *, scale, initial_state, output_final_state,
               A_log, dt_bias, cu_seqlens):
    q = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
    k = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    g = _gate(g, A_log, dt_bias)
    beta = beta.float().sigmoid()
    if cu_seqlens is None:
        return _recurrent_reference(
            q, k, v.float(), g, beta, scale, initial_state, output_final_state
        )
    bounds = cu_seqlens.tolist()
    outs, states = [], []
    for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        state_i = initial_state[index : index + 1] if initial_state is not None else None
        out_i, state_out = _recurrent_reference(
            q[:, start:end], k[:, start:end], v[:, start:end].float(),
            g[:, start:end], beta[:, start:end], scale, state_i, output_final_state,
        )
        outs.append(out_i)
        if output_final_state:
            states.append(state_out)
    final = torch.cat(states, dim=0) if output_final_state else None
    return torch.cat(outs, dim=1), final


def _relative_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected, actual = expected.float(), actual.float()
    scale = max(expected.abs().max().item(), 1e-6)
    return (expected - actual).abs().max().item() / scale


def _make_inputs(seq_lens, H, device, seed=0):
    torch.manual_seed(seed)
    K = V = 128
    T = sum(seq_lens)
    q = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, V, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, K, device=device, dtype=torch.float32)
    beta = torch.randn(1, T, H, device=device, dtype=torch.float32)
    A_log = torch.log(torch.empty(H, device=device, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(H * K, device=device, dtype=torch.float32)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
            device=device, dtype=torch.int32,
        )
    return q, k, v, g, beta, A_log, dt_bias, cu_seqlens


@pytest.mark.parametrize(
    "seq_lens, H",
    [
        ([64], 2),
        ([128], 4),
        ([256], 1),
        ([48], 2),
        ([32, 64, 16], 2),
    ],
    ids=["t64h2", "t128h4", "t256h1", "ragged48", "varlen"],
)
@pytest.mark.parametrize("output_final_state", [False, True])
def test_matches_recurrent_reference(seq_lens, H, output_final_state):
    device = test_backend.device
    q, k, v, g, beta, A_log, dt_bias, cu_seqlens = _make_inputs(seq_lens, H, device)
    scale = q.shape[-1] ** -0.5

    expected, expected_state = _reference(
        q, k, v, g, beta, scale=scale, initial_state=None,
        output_final_state=output_final_state, A_log=A_log,
        dt_bias=dt_bias, cu_seqlens=cu_seqlens,
    )
    with torch.no_grad():
        actual, actual_state = chunk_kda_fwd_infer_triton(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            chunk_size=16, lower_bound=LOWER_BOUND, A_log=A_log, dt_bias=dt_bias,
        )

    assert torch.isfinite(actual).all(), "output contains NaN or Inf"
    assert _relative_error(expected, actual) < RELATIVE_TOLERANCE
    if output_final_state:
        assert actual_state is not None
        assert _relative_error(expected_state, actual_state) < RELATIVE_TOLERANCE
    else:
        assert actual_state is None


def test_initial_state_is_carried():
    device = test_backend.device
    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs([64], 2, device)
    scale = q.shape[-1] ** -0.5
    H, K, V = q.shape[2], q.shape[-1], v.shape[-1]
    initial_state = torch.randn(1, H, K, V, device=device, dtype=torch.float32) * 0.1

    expected, _ = _reference(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        output_final_state=False, A_log=A_log, dt_bias=dt_bias, cu_seqlens=None,
    )
    with torch.no_grad():
        actual, _ = chunk_kda_fwd_infer_triton(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=initial_state,
            chunk_size=16, lower_bound=LOWER_BOUND, A_log=A_log, dt_bias=dt_bias,
        )
    assert _relative_error(expected, actual) < RELATIVE_TOLERANCE


def test_rejects_grad_enabled_inputs():
    """The kernels save nothing, so a differentiable call must fail loudly."""
    device = test_backend.device
    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs([64], 2, device)
    with pytest.raises(RuntimeError, match="inference only"):
        chunk_kda_fwd_infer_triton(
            q=q.clone().requires_grad_(True), k=k, v=v, g=g, beta=beta,
            chunk_size=16, lower_bound=LOWER_BOUND, A_log=A_log, dt_bias=dt_bias,
        )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: {"q": t["q"][..., :64], "k": t["k"][..., :64]}, "K=128"),
        (lambda t: {"v": t["v"][..., :64]}, "V=128"),
        (lambda t: {"chunk_size": 32}, "chunk_size=16"),
    ],
    ids=["bad_k", "bad_v", "bad_chunk"],
)
def test_rejects_unsupported_shapes(mutate, message):
    device = test_backend.device
    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs([64], 2, device)
    call = {
        "q": q, "k": k, "v": v, "g": g, "beta": beta, "chunk_size": 16,
        "lower_bound": LOWER_BOUND, "A_log": A_log, "dt_bias": dt_bias,
    }
    call.update(mutate(call))
    with pytest.raises(ValueError, match=message):
        chunk_kda_fwd_infer_triton(**call)
