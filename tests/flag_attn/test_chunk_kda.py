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

import math

import pytest
import torch
import torch.nn.functional as F

from flag_attn.runtime.backend._enflame.kda import chunk_kda

LOWER_BOUND = -5.0
ASSERT_RATIO = 0.005


def _accelerator_available() -> bool:
    return hasattr(torch, "gcu") and torch.gcu.is_available()


pytestmark = [
    pytest.mark.chunk_kda,
    pytest.mark.skipif(
        not _accelerator_available(),
        reason="chunk_kda tests require an available Enflame GCU",
    ),
]


def _naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float,
) -> torch.Tensor:
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return lower_bound * torch.sigmoid(A_log.view(H, 1).float().exp() * g)


def _naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    state = k.new_zeros(B, HV, K, V).to(q)
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

    return out.to(dtype), state if output_final_state else None


def _state_to_kv_layout(
    state: torch.Tensor | None,
    state_v_first: bool,
) -> torch.Tensor | None:
    if state is None:
        return None
    if state_v_first:
        return state.transpose(-1, -2).contiguous()
    return state


def _state_from_kv_layout(
    state: torch.Tensor | None,
    state_v_first: bool,
) -> torch.Tensor | None:
    if state is None:
        return None
    if state_v_first:
        return state.transpose(-1, -2).contiguous()
    return state


def _reference_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
    k = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    g = _naive_kda_lowerbound_gate(g, A_log, dt_bias, LOWER_BOUND)
    beta = beta.float().sigmoid()

    if cu_seqlens is None:
        out, final_state = _naive_recurrent_kda(
            q=q,
            k=k,
            v=v.float(),
            g=g,
            beta=beta,
            scale=scale,
            initial_state=_state_to_kv_layout(initial_state, state_v_first),
            output_final_state=output_final_state,
        )
        return out.to(v.dtype), _state_from_kv_layout(final_state, state_v_first)

    cu_seqlens_list = cu_seqlens.detach().cpu().tolist()
    outs, final_states = [], []
    for i, (start, end) in enumerate(zip(cu_seqlens_list[:-1], cu_seqlens_list[1:])):
        initial_state_i = (
            initial_state[i : i + 1] if initial_state is not None else None
        )
        out_i, final_state_i = _naive_recurrent_kda(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end].float(),
            g=g[:, start:end],
            beta=beta[:, start:end],
            scale=scale,
            initial_state=_state_to_kv_layout(initial_state_i, state_v_first),
            output_final_state=output_final_state,
        )
        outs.append(out_i)
        if output_final_state:
            final_states.append(_state_from_kv_layout(final_state_i, state_v_first))

    out = torch.cat(outs, dim=1).to(v.dtype)
    final_state = torch.cat(final_states, dim=0) if output_final_state else None
    return out, final_state


def _make_inputs(
    *,
    seq_lens: list[int],
    H: int,
    HV: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
    state_v_first: bool,
    normal_inputs: bool,
    use_initial_state: bool,
    output_final_state: bool,
) -> tuple[tuple[torch.Tensor, ...], dict]:
    device = "gcu"
    T = sum(seq_lens)
    N = len(seq_lens)

    rand = torch.randn if normal_inputs else torch.rand
    q = rand(1, T, H, D, device=device, dtype=dtype)
    k = rand(1, T, H, D, device=device, dtype=dtype)
    v = rand(1, T, HV, D, device=device, dtype=dtype)
    g = torch.randn(1, T, HV, D, device=device, dtype=dtype)
    beta = torch.randn(1, T, HV, device=device, dtype=dtype)
    A_log = torch.log(
        torch.empty(HV, device=device, dtype=torch.float32).uniform_(1, 16)
    )
    dt_bias = torch.randn(HV, D, device=device, dtype=torch.float32)

    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(N, HV, D, D, device=device, dtype=torch.float32)

    cu_seqlens = None
    if N > 1:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
            device=device,
            dtype=torch.int32,
        )

    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_v_first": state_v_first,
        "cu_seqlens": cu_seqlens,
        "chunk_size": 16,
    }
    return (q, k, v, g, beta), kwargs


def _abs_err(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return (expected.detach() - actual.detach()).flatten().abs().max().item()


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    err = (expected.detach() - actual.detach()).flatten().square().mean().sqrt().item()
    base = expected.detach().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual = actual.float()
    expected = expected.float()
    abs_err = _abs_err(expected, actual)
    err_ratio = _err_ratio(expected, actual)
    msg = (
        f"{name} diff: {abs_err:.6f} ratio: {err_ratio:.6f} " f"(limit {ASSERT_RATIO})"
    )
    if abs_err <= 1e-6:
        return
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in reference"
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert err_ratio < ASSERT_RATIO, msg


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "seq_lens": [32],
                "H": 2,
                "HV": 2,
                "D": 64,
                "scale": 0.1,
                "dtype": torch.bfloat16,
                "state_v_first": True,
                "normal_inputs": False,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="dense",
        ),
        pytest.param(
            {
                "seq_lens": [32],
                "H": 2,
                "HV": 2,
                "D": 128,
                "scale": 1 / math.sqrt(128),
                "dtype": torch.bfloat16,
                "state_v_first": True,
                "normal_inputs": True,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="strict_tle_compatible",
        ),
        pytest.param(
            {
                "seq_lens": [13, 19, 16],
                "H": 2,
                "HV": 2,
                "D": 128,
                "scale": 1 / math.sqrt(128),
                "dtype": torch.bfloat16,
                "state_v_first": True,
                "normal_inputs": True,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="strict_tle_varlen_compatible",
        ),
        pytest.param(
            {
                "seq_lens": [32],
                "H": 2,
                "HV": 4,
                "D": 64,
                "scale": 0.1,
                "dtype": torch.bfloat16,
                "state_v_first": True,
                "normal_inputs": False,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="dense_gqa",
        ),
        pytest.param(
            {
                "seq_lens": [13, 19, 16],
                "H": 2,
                "HV": 2,
                "D": 64,
                "scale": 0.1,
                "dtype": torch.bfloat16,
                "state_v_first": True,
                "normal_inputs": True,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="varlen",
        ),
        pytest.param(
            {
                "seq_lens": [16],
                "H": 2,
                "HV": 2,
                "D": 64,
                "scale": 1 / math.sqrt(64),
                "dtype": torch.bfloat16,
                "state_v_first": True,
                "normal_inputs": False,
                "use_initial_state": False,
                "output_final_state": False,
            },
            id="no_initial_state",
        ),
        pytest.param(
            {
                "seq_lens": [24],
                "H": 2,
                "HV": 2,
                "D": 64,
                "scale": 0.1,
                "dtype": torch.bfloat16,
                "state_v_first": False,
                "normal_inputs": False,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="state_kv_layout",
        ),
        pytest.param(
            {
                "seq_lens": [32],
                "H": 2,
                "HV": 2,
                "D": 64,
                "scale": 0.1,
                "dtype": torch.float16,
                "state_v_first": True,
                "normal_inputs": False,
                "use_initial_state": True,
                "output_final_state": True,
            },
            id="fp16_dense",
        ),
    ],
)
@torch.inference_mode()
def test_chunk_kda_matches_torch_reference(case):
    torch.manual_seed(42)
    args, kwargs = _make_inputs(
        seq_lens=case["seq_lens"],
        H=case["H"],
        HV=case["HV"],
        D=case["D"],
        scale=case["scale"],
        dtype=case["dtype"],
        state_v_first=case["state_v_first"],
        normal_inputs=case["normal_inputs"],
        use_initial_state=case["use_initial_state"],
        output_final_state=case["output_final_state"],
    )

    actual, actual_final = chunk_kda(*args, **kwargs)
    expected, expected_final = _reference_chunk_kda(
        *args,
        scale=kwargs["scale"],
        initial_state=kwargs["initial_state"],
        output_final_state=kwargs["output_final_state"],
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        state_v_first=kwargs["state_v_first"],
        cu_seqlens=kwargs["cu_seqlens"],
    )

    _assert_close("o", actual, expected)
    if case["output_final_state"]:
        _assert_close("ht", actual_final, expected_final)
    else:
        assert actual_final is None
        assert expected_final is None
