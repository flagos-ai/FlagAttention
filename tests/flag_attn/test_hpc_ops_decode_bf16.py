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

"""Official BF16 decode correctness matrix for the selected backend."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import triton


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from flag_attn.hpc_ops_attention.decode import HAS_TLE  # noqa: E402
from flag_attn.hpc_ops_attention.decode.dynamic import (  # noqa: E402
    bf16_dynamic,
)


SUPPORTED_TEST_MTP = (1, 2, 3) if HAS_TLE else (1,)
from flag_attn.hpc_ops_attention.decode.static import (  # noqa: E402
    bf16_static,
)


BLOCK_SIZE = bf16_static.BLOCK_SIZE
HEAD_DIM = bf16_static.HEAD_DIM


@dataclass
class _Panel:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor


@dataclass(frozen=True)
class _DecodeImplementation:
    inputs_type: Callable[..., object]
    prepare_workspace: Callable[[object], object]
    attention_decode: Callable[[object, object], torch.Tensor]


_IMPLEMENTATIONS = {
    "static": _DecodeImplementation(
        bf16_static.StaticBF16Inputs,
        bf16_static.prepare_static_bf16_workspace,
        bf16_static.attention_decode_bf16_tle,
    ),
    "dynamic": _DecodeImplementation(
        bf16_dynamic.DynamicBF16Inputs,
        bf16_dynamic.prepare_dynamic_bf16_workspace,
        bf16_dynamic.attention_decode_bf16_dynamic,
    ),
}


def _implementation(schedule: str) -> _DecodeImplementation:
    return _IMPLEMENTATIONS[schedule]


def _inputs(
    panel: _Panel,
    kvcache_shape: str,
    implementation: _DecodeImplementation,
) -> object:
    return implementation.inputs_type(
        panel.q,
        panel.k,
        panel.v,
        panel.block_ids,
        panel.kv_lens,
        kvcache_shape,
    )


def _make_inputs(
    num_batch: int,
    num_seq_q: int,
    max_seq_kv: int,
    num_head_kv: int,
    num_head_q: int,
    kvcache_shape: str,
) -> _Panel:
    """Reproduce the official BF16 test's random-number call order."""
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    q = torch.randn(
        (num_batch * num_seq_q, num_head_q, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ) / math.sqrt(HEAD_DIM)
    new_k = torch.randn(
        (num_batch * num_seq_q, num_head_kv, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ) / math.sqrt(HEAD_DIM)
    new_v = torch.randn(
        (num_batch * num_seq_q, num_head_kv, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    history = torch.randint(
        1, max_seq_kv, (num_batch,), dtype=torch.int32, device="cuda",
    )
    kv_lens = history + num_seq_q
    block_counts = (
        kv_lens + BLOCK_SIZE - 1
    ) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    max_num_blocks = int(total_blocks * 1.2) + num_batch + 8
    storage = torch.randn(
        (max_num_blocks, 2, BLOCK_SIZE, num_head_kv, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    if kvcache_shape == "HND":
        storage = (
            storage.permute(0, 1, 3, 2, 4)
            .contiguous()
            .permute(0, 1, 3, 2, 4)
        )
    packed_ids = torch.randperm(max_num_blocks)[:total_blocks].to(
        dtype=torch.int32, device="cuda",
    )
    block_ids = torch.empty(
        (num_batch, int(block_counts.max().item())),
        dtype=torch.int32,
        device="cuda",
    )
    new_k = new_k.reshape(
        num_batch, num_seq_q, num_head_kv, HEAD_DIM,
    )
    new_v = new_v.reshape(
        num_batch, num_seq_q, num_head_kv, HEAD_DIM,
    )
    cursor = 0
    for batch_id, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch_id, :count] = packed_ids[cursor:cursor + count]
        cursor += count
        history_length = int(history[batch_id])
        for row in range(num_seq_q):
            position = history_length + row
            physical = block_ids[batch_id, position // BLOCK_SIZE]
            storage[physical, 0, position % BLOCK_SIZE] = new_k[batch_id, row]
            storage[physical, 1, position % BLOCK_SIZE] = new_v[batch_id, row]

    return _Panel(
        q=q,
        k=storage[:, 0],
        v=storage[:, 1],
        block_ids=block_ids,
        kv_lens=kv_lens,
    )


def _pytorch_reference(panel: _Panel, num_seq_q: int) -> torch.Tensor:
    num_batch = int(panel.kv_lens.numel())
    num_head_q = int(panel.q.shape[1])
    num_head_kv = int(panel.k.shape[2])
    heads_per_group = num_head_q // num_head_kv
    q = panel.q.reshape(
        num_batch, num_seq_q, num_head_q, HEAD_DIM,
    )
    output = torch.empty_like(q)
    for batch_id in range(num_batch):
        length = int(panel.kv_lens[batch_id])
        pages = triton.cdiv(length, BLOCK_SIZE)
        ids = panel.block_ids[batch_id, :pages]
        k = panel.k[ids].reshape(-1, num_head_kv, HEAD_DIM)[:length]
        v = panel.v[ids].reshape(-1, num_head_kv, HEAD_DIM)[:length]
        k = k.transpose(0, 1).repeat_interleave(
            heads_per_group, dim=0,
        ).float()
        v = v.transpose(0, 1).repeat_interleave(
            heads_per_group, dim=0,
        ).float()
        scores = (
            q[batch_id].transpose(0, 1).float()
            @ k.transpose(-1, -2)
        ) / math.sqrt(HEAD_DIM)
        history = length - num_seq_q
        causal = torch.cat(
            (
                torch.ones(
                    (num_seq_q, history),
                    dtype=torch.bool,
                    device=panel.q.device,
                ),
                torch.tril(torch.ones(
                    (num_seq_q, num_seq_q),
                    dtype=torch.bool,
                    device=panel.q.device,
                )),
            ),
            dim=-1,
        )
        scores.masked_fill_(~causal[None], -float("inf"))
        output[batch_id] = (
            F.softmax(scores, dim=-1) @ v
        ).transpose(0, 1)
    return output.reshape_as(panel.q)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="BF16 decode validation requires Hopper",
)
@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize(
    "num_seq_q", SUPPORTED_TEST_MTP, ids=lambda value: f"mtp{value}",
)
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8), (4, 32)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
@pytest.mark.parametrize("use_dynamic_sched", [False, True])
@pytest.mark.parametrize("kvcache_shape", ["NHD", "HND"])
@torch.no_grad()
def test_attn_bf16_sm90(
    num_batch: int,
    num_seq_q: int,
    max_seq_kv: int,
    block_size: int,
    kv_head_q_head: tuple[int, int],
    head_dim: int,
    new_kv_included: bool,
    use_output: bool,
    splitk: bool,
    use_dynamic_sched: bool,
    kvcache_shape: str,
):
    triton.set_allocator(lambda size, _align, _stream: torch.empty(
        size, dtype=torch.int8, device="cuda",
    ))
    assert block_size == BLOCK_SIZE
    assert head_dim == HEAD_DIM
    assert new_kv_included
    assert not use_output
    assert splitk

    num_head_kv, num_head_q = kv_head_q_head
    panel = _make_inputs(
        num_batch,
        num_seq_q,
        max_seq_kv,
        num_head_kv,
        num_head_q,
        kvcache_shape,
    )
    schedule = "dynamic" if use_dynamic_sched else "static"
    implementation = _implementation(schedule)
    inputs = _inputs(panel, kvcache_shape, implementation)
    workspace = implementation.prepare_workspace(inputs)
    actual = implementation.attention_decode(inputs, workspace)

    actual = actual.detach().clone()
    expected = _pytorch_reference(panel, num_seq_q)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.016, rtol=1e-5)
