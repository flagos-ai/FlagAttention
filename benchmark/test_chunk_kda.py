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

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from benchmark.base import Benchmark
from flag_attn.runtime.backend._enflame.kda import chunk_kda

DEVICE = "gcu"
LOWER_BOUND = -5.0
DEFAULT_H = 96
DEFAULT_D = 128
FIXED_CASES = [
    [8192],
]
VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]
FLASHKDA_CASES = FIXED_CASES + VARLEN_CASES


def _gcu_available() -> bool:
    return hasattr(torch, "gcu") and torch.gcu.is_available()


def _chunk_kda_inference_op(*args, **kwargs):
    with torch.inference_mode():
        return chunk_kda(*args, **kwargs)


def _naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = LOWER_BOUND,
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
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
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


def _chunk_kda_torch_reference_op(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    safe_gate=True,
    lower_bound=LOWER_BOUND,
    A_log=None,
    dt_bias=None,
    state_v_first=False,
    cu_seqlens=None,
    **kwargs,
):
    if not safe_gate:
        raise ValueError("chunk_kda torch reference currently expects safe_gate=True")
    if A_log is None or dt_bias is None:
        raise ValueError("chunk_kda torch reference requires A_log and dt_bias")

    with torch.inference_mode():
        q = F.normalize(q.float(), p=2.0, dim=-1)
        k = F.normalize(k.float(), p=2.0, dim=-1)
        g = _naive_kda_lowerbound_gate(g, A_log, dt_bias, lower_bound=lower_bound)
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
        for i, (start, end) in enumerate(
            zip(cu_seqlens_list[:-1], cu_seqlens_list[1:])
        ):
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


class ChunkKDABenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        tuple(itertools.chain(seq_lens, [DEFAULT_H, DEFAULT_D]))
        for seq_lens in FLASHKDA_CASES
    ]
    DEFAULT_SHAPE_DESC = "seq_lens..., H, D"

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) < 3 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            *seq_lens, H, D = shape
            yield self._build_inputs(seq_lens, H, D, cur_dtype)

    def _build_inputs(self, seq_lens, H, D, dtype):
        device = DEVICE
        T_total = sum(seq_lens)
        N = len(seq_lens)
        scale = 1.0 / math.sqrt(D)

        q = F.normalize(
            torch.randn((1, T_total, H, D), dtype=torch.float32, device=device),
            p=2.0,
            dim=-1,
        ).to(dtype)
        k = F.normalize(
            torch.randn((1, T_total, H, D), dtype=torch.float32, device=device),
            p=2.0,
            dim=-1,
        ).to(dtype)
        v = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
        g = torch.randn((1, T_total, H, D), dtype=dtype, device=device)
        beta = torch.randn((1, T_total, H), dtype=dtype, device=device)
        A_log = torch.rand(H, dtype=torch.float32, device=device)
        dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
        initial_state = (
            torch.arange(N * H * D * D, dtype=torch.float32, device=device)
            .reshape(N, H, D, D)
            .to(dtype)
            .float()
        )

        cu_seqlens = None
        if N > 1:
            cu_seqlens = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
                dtype=torch.int32,
                device=device,
            )

        return (
            q,
            k,
            v,
            g,
            beta,
            {
                "scale": scale,
                "initial_state": initial_state,
                "output_final_state": True,
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "safe_gate": True,
                "lower_bound": LOWER_BOUND,
                "A_log": A_log,
                "dt_bias": dt_bias,
                "state_v_first": True,
                "cu_seqlens": cu_seqlens,
                "chunk_size": 32,
            },
        )


@pytest.mark.skipif(
    not _gcu_available(),
    reason="chunk_kda benchmark requires an available Enflame GCU",
)
@pytest.mark.chunk_kda
def test_chunk_kda():
    bench = ChunkKDABenchmark(
        op_name="chunk_kda",
        torch_op=_chunk_kda_torch_reference_op,
    )
    bench.set_gems(_chunk_kda_inference_op)
    bench.run()
