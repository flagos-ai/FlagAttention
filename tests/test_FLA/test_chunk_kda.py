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

pytest.importorskip("triton.experimental.tle.language", reason="chunk_kda requires Triton TLE >= 3.6")
pytest.importorskip("flaggems_vllm", reason="chunk_kda currently uses FlagGems index helpers")

from fla.chunk_kda import chunk_kda_fwd_infer

LOWER_BOUND = -5.0
ASSERT_RATIO = 0.005
D_HEAD = 128

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="chunk_kda tests require CUDA")


def _lower_bound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    H = g.shape[-2]
    return LOWER_BOUND * torch.sigmoid(
        A_log.view(H, 1).float().exp() * (g.float() + dt_bias.view(H, -1))
    )


def _recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    q, k, v, g, beta = (tensor.float() for tensor in (q, k, v, g, beta))
    q = q * scale

    state = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        state = state + initial_state.transpose(-1, -2)

    output = torch.empty_like(v)
    for i in range(T):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        g_i, beta_i = g[:, i], beta[:, i]
        state = state * g_i[..., None].exp()
        residual = v_i - (k_i[..., None] * state).sum(-2)
        state = state + torch.einsum("bhk,bhv->bhkv", beta_i[..., None] * k_i, residual)
        output[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state)

    final_state = state.transpose(-1, -2).contiguous() if output_final_state else None
    return output.to(dtype), final_state


def _reference(
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
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q = F.normalize(q.float(), p=2.0, dim=-1, eps=1e-6)
    k = F.normalize(k.float(), p=2.0, dim=-1, eps=1e-6)
    g = _lower_bound_gate(g, A_log, dt_bias)
    beta = beta.float().sigmoid()

    if cu_seqlens is None:
        return _recurrent_reference(
            q, k, v, g, beta, scale, initial_state, output_final_state
        )

    outputs = []
    final_states = []
    offsets = cu_seqlens.cpu().tolist()
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        state_i = initial_state[i : i + 1] if initial_state is not None else None
        output_i, final_state_i = _recurrent_reference(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            state_i,
            output_final_state,
        )
        outputs.append(output_i)
        if final_state_i is not None:
            final_states.append(final_state_i)

    final_state = torch.cat(final_states) if output_final_state else None
    return torch.cat(outputs, dim=1), final_state


def _make_inputs(
    seq_lens: list[int],
    B: int,
    H: int,
    state_dtype: torch.dtype,
    use_initial_state: bool,
    output_final_state: bool,
    noncontiguous: bool,
):
    device = torch.device("cuda")
    is_varlen = len(seq_lens) > 1
    if is_varlen and B != 1:
        raise ValueError("varlen inputs require B=1")
    T = sum(seq_lens) if is_varlen else seq_lens[0]
    N = len(seq_lens) if is_varlen else B

    def with_layout(tensor):
        if not noncontiguous:
            return tensor
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2),
            device=device,
            dtype=tensor.dtype,
        )
        view = storage[..., ::2]
        view.copy_(tensor)
        return view

    def randn(shape, dtype=torch.bfloat16):
        return with_layout(torch.randn(shape, device=device, dtype=dtype))

    def normalized_qk():
        tensor = F.normalize(
            torch.randn((B, T, H, D_HEAD), device=device, dtype=torch.float32),
            p=2.0,
            dim=-1,
        ).to(torch.bfloat16)
        return with_layout(tensor)

    q = normalized_qk()
    k = normalized_qk()
    v = randn((B, T, H, D_HEAD))
    g = randn((B, T, H, D_HEAD))
    beta = randn((B, T, H))
    A_log = torch.rand(H, device=device, dtype=torch.float32)
    dt_bias = with_layout(torch.rand(H, D_HEAD, device=device, dtype=torch.float32))
    initial_state = None
    if use_initial_state:
        initial_state = randn((N, H, D_HEAD, D_HEAD), dtype=state_dtype)

    cu_seqlens = None
    if is_varlen:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
            device=device,
            dtype=torch.long,
        )

    kwargs = {
        "scale": 1 / math.sqrt(D_HEAD),
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "state_v_first": True,
        "cu_seqlens": cu_seqlens,
        "chunk_size": 16,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    return (q, k, v, g, beta), kwargs


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    assert not torch.isnan(actual).any(), f"{name}: NaN in TLE output"
    assert not torch.isnan(expected).any(), f"{name}: NaN in reference output"
    absolute_error = (actual - expected).abs().max().item()
    relative_error = (actual - expected).square().mean().sqrt() / (
        expected.square().mean().sqrt() + 1e-8
    )
    assert absolute_error <= 1e-6 or relative_error.item() < ASSERT_RATIO, (
        f"{name}: max error={absolute_error:.6f}, RMS ratio={relative_error.item():.6f}, "
        f"limit={ASSERT_RATIO}"
    )


@pytest.mark.parametrize(
    ("seq_lens", "B", "state_dtype", "use_initial_state", "output_final_state", "noncontiguous"),
    [
        pytest.param([32], 1, torch.bfloat16, True, True, False, id="dense-aligned-bf16-state"),
        pytest.param([31], 1, torch.float32, False, False, False, id="dense-tail-no-state"),
        pytest.param([17], 2, torch.bfloat16, True, True, False, id="dense-batched-bf16-state"),
        pytest.param([13, 19, 16], 1, torch.float32, True, True, False, id="varlen-fp32-state"),
        pytest.param([17, 15], 1, torch.bfloat16, True, True, True, id="varlen-noncontiguous-bf16-state"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_matches_recurrent_reference(
    seq_lens,
    B,
    state_dtype,
    use_initial_state,
    output_final_state,
    noncontiguous,
):
    torch.manual_seed(42)
    args, kwargs = _make_inputs(
        seq_lens=seq_lens,
        B=B,
        H=2,
        state_dtype=state_dtype,
        use_initial_state=use_initial_state,
        output_final_state=output_final_state,
        noncontiguous=noncontiguous,
    )

    actual, actual_final = chunk_kda_fwd_infer(*args, **kwargs)
    expected, expected_final = _reference(
        *args,
        scale=kwargs["scale"],
        initial_state=kwargs["initial_state"],
        output_final_state=kwargs["output_final_state"],
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        cu_seqlens=kwargs["cu_seqlens"],
    )

    _assert_close("output", actual, expected)
    if output_final_state:
        assert actual_final.dtype == torch.float32
        _assert_close("final_state", actual_final, expected_final)
    else:
        assert actual_final is None
        assert expected_final is None
