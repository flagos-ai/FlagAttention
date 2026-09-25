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

"""Official FP8 qk-per-token/v-per-head decode correctness matrix."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import triton


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from flag_attn.hpc_ops_attention.decode import HAS_TLE  # noqa: E402
from flag_attn.hpc_ops_attention.decode.dynamic import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_dynamic as fp8_dynamic,
)
from flag_attn.hpc_ops_attention.decode.static import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_static as fp8_static,
)


BLOCK_SIZE = fp8_static.BLOCK_SIZE
HEAD_DIM = fp8_static.HEAD_DIM


@dataclass
class _Panel:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


_IMPLEMENTATIONS = {
    "static": fp8_static,
    "dynamic": fp8_dynamic,
}


def _implementation(schedule: str) -> ModuleType:
    return _IMPLEMENTATIONS[schedule]


def _inputs(panel: _Panel, implementation: ModuleType) -> object:
    return implementation.FP8DecodeInputs(
        panel.q,
        panel.k_cache,
        panel.v_cache,
        panel.block_ids,
        panel.kv_lens,
        panel.q_scale,
        panel.k_scale,
        panel.v_scale,
    )


FP8_ATOL = {
    "qkpertoken_perhead_vperhead": 0.1,
    "qpertoken_perhead_kvpertensor": 0.2,
}
QUANT_TYPE = "qkpertoken_perhead_vperhead"
SUPPORTED_TEST_MTP = (1, 2, 4) if HAS_TLE else (1,)


def _supports_sm90_fp8_backend() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch, "float8_e4m3fn")
        and torch.cuda.get_device_capability()[0] >= 9
    )


@pytest.fixture(autouse=True)
def _triton_allocator():
    if torch.cuda.is_available():
        triton.set_allocator(
            lambda size, _align, _stream: torch.empty(
                size, dtype=torch.int8, device="cuda"
            )
        )
    yield


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _quantize_k_per_token(
    storage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks, _, heads, dim = storage.shape
    scale = storage[:, :BLOCK_SIZE].float().abs().amax(-1) / 448.0
    quantized = torch.empty_like(storage, dtype=torch.float8_e4m3fn)
    quantized[:, :BLOCK_SIZE] = (
        storage[:, :BLOCK_SIZE] / scale[..., None]
    ).to(torch.float8_e4m3fn)
    packed_scale = (
        scale.permute(0, 2, 1)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .reshape(blocks, heads, -1, dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    quantized[:, BLOCK_SIZE:] = packed_scale
    return quantized, quantized[:, BLOCK_SIZE:]


def _quantize_v_per_head(
    storage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = storage.shape[2]
    scale = (
        storage[:, :BLOCK_SIZE]
        .float()
        .abs()
        .permute(2, 0, 1, 3)
        .reshape(heads, -1)
        .amax(-1)
        / 448.0
    )
    quantized = (
        storage.float() / scale[None, None, :, None]
    ).to(torch.float8_e4m3fn)
    # This factor is intentional: it exactly matches the official
    # qk-per-token/v-per-head correctness test distribution.
    return quantized, scale * 0.1


def _official_make_inputs(
    num_batch: int,
    mtp: int,
    max_seq_kv: int,
    num_head_kv: int,
    num_head_q: int,
    layout: str,
    quant_type: str,
) -> _Panel:
    """Reproduce the official FP8 correctness-test input distribution."""
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    q_bf16 = torch.randn(
        (num_batch * mtp, num_head_q, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    ) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.float().abs().amax(-1) / 10.0
    q = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)

    # Match the official correctness tests' seeded RNG order exactly:
    # Q -> new K/V -> scales -> history -> cache -> block IDs.
    if quant_type == "qkpertoken_perhead_vperhead":
        new_k = torch.randn(
            (num_batch, mtp, num_head_kv, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        new_v = torch.randn_like(new_k)
        # The official test performs this draw before history generation even
        # though quant_paged_cache_perhead later supplies the effective scale.
        torch.randn((num_head_kv,), dtype=torch.float32, device="cuda")
    else:
        new_k = (
            torch.randn(
                (num_batch, mtp, num_head_kv, HEAD_DIM),
                dtype=torch.bfloat16,
                device="cuda",
            )
            / math.sqrt(HEAD_DIM)
        ).to(torch.float8_e4m3fn)
        new_v = torch.randn(
            (num_batch, mtp, num_head_kv, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        ).to(torch.float8_e4m3fn)
        k_scale = torch.randn((1,), dtype=torch.float32, device="cuda")
        v_scale = torch.randn((1,), dtype=torch.float32, device="cuda")

    history = torch.randint(
        1,
        max_seq_kv,
        (num_batch,),
        dtype=torch.int32,
        device="cuda",
    )
    kv_lens = history + mtp
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(num_batch * max_seq_kv / BLOCK_SIZE * 1.2)

    if quant_type == "qkpertoken_perhead_vperhead":
        scale_rows = BLOCK_SIZE * 4 // HEAD_DIM
        storage = torch.randn(
            (capacity, 2, BLOCK_SIZE + scale_rows, num_head_kv, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        if layout == "HND":
            storage = (
                storage.permute(0, 1, 3, 2, 4)
                .contiguous()
                .permute(0, 1, 3, 2, 4)
            )
    else:
        storage = (
            torch.randn(
                (capacity, 2, BLOCK_SIZE, num_head_kv, HEAD_DIM),
                dtype=torch.bfloat16,
                device="cuda",
            )
            / math.sqrt(HEAD_DIM)
        ).to(torch.float8_e4m3fn)
        if layout == "HND":
            storage = (
                storage.permute(0, 1, 3, 2, 4)
                .contiguous()
                .permute(0, 1, 3, 2, 4)
            )

    # The official correctness tests draw the permutation on CPU and then
    # transfer it, so preserve that generator choice as well as call order.
    packed_ids = torch.randperm(capacity)[:total_blocks].to(
        dtype=torch.int32, device="cuda"
    )
    block_ids = torch.empty(
        (num_batch, int(block_counts.max().item())),
        dtype=torch.int32,
        device="cuda",
    )
    cursor = 0
    for batch_id, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch_id, :count] = packed_ids[cursor:cursor + count]
        cursor += count
    for batch_id, history_length in enumerate(history.cpu().tolist()):
        for row in range(mtp):
            position = history_length + row
            physical = block_ids[batch_id, position // BLOCK_SIZE]
            storage[physical, 0, position % BLOCK_SIZE] = new_k[batch_id, row]
            storage[physical, 1, position % BLOCK_SIZE] = new_v[batch_id, row]

    if quant_type == "qkpertoken_perhead_vperhead":
        k_storage, k_scale = _quantize_k_per_token(storage[:, 0])
        v_storage, v_scale = _quantize_v_per_head(storage[:, 1])
        k_cache = k_storage[:, :BLOCK_SIZE]
        v_cache = v_storage[:, :BLOCK_SIZE]
    else:
        k_cache = storage[:, 0]
        v_cache = storage[:, 1]

    return _Panel(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_ids=block_ids,
        kv_lens=kv_lens,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def _packed_k_scale(panel, batch: int, length: int) -> torch.Tensor:
    heads = panel.k_scale.shape[2]
    blocks = panel.block_ids[
        batch, : (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    ]
    return (
        panel.k_scale[blocks]
        .contiguous()
        .view(torch.float32)
        .permute(0, 1, 3, 2)
        .reshape(-1, heads)[:length]
        .transpose(0, 1)
        .float()
    )


@torch.no_grad()
def _pytorch_reference(panel, mtp: int, quant_type: str) -> torch.Tensor:
    batch = panel.kv_lens.numel()
    num_head_q = panel.q.shape[1]
    num_head_kv = panel.k_cache.shape[2]
    heads_per_group = num_head_q // num_head_kv
    q = panel.q.reshape(batch, mtp, num_head_q, HEAD_DIM)
    q_scale = panel.q_scale.reshape(batch, mtp, num_head_q)
    outputs = []

    for batch_id in range(batch):
        length = int(panel.kv_lens[batch_id])
        block_ids = panel.block_ids[
            batch_id, : (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        ]
        k = panel.k_cache[block_ids].reshape(
            -1, num_head_kv, HEAD_DIM
        )[:length]
        v = panel.v_cache[block_ids].reshape(
            -1, num_head_kv, HEAD_DIM
        )[:length]
        k = k.transpose(0, 1).repeat_interleave(
            heads_per_group, dim=0
        ).float()
        v = v.transpose(0, 1).repeat_interleave(
            heads_per_group, dim=0
        ).float()

        scores = (
            q[batch_id].transpose(0, 1).float()
            @ k.transpose(-1, -2)
        )
        scores *= q_scale[batch_id].transpose(0, 1).float()[:, :, None]
        scores /= math.sqrt(HEAD_DIM)
        if quant_type == "qkpertoken_perhead_vperhead":
            scores *= _packed_k_scale(
                panel, batch_id, length
            ).repeat_interleave(heads_per_group, dim=0)[:, None, :]
        else:
            scores *= panel.k_scale.float()

        history = length - mtp
        causal_mask = torch.cat(
            (
                torch.ones(
                    (mtp, history), dtype=torch.bool, device="cuda"
                ),
                torch.tril(
                    torch.ones(
                        (mtp, mtp), dtype=torch.bool, device="cuda"
                    )
                ),
            ),
            dim=-1,
        )
        scores.masked_fill_(~causal_mask[None, :, :], -float("inf"))
        weights = torch.exp(scores - scores.amax(-1, keepdim=True))
        denominator = weights.sum(-1, keepdim=True)
        weights = (weights * 256.0).to(torch.float8_e4m3fn).float()
        output = weights @ v / denominator
        if quant_type == "qkpertoken_perhead_vperhead":
            output *= (
                panel.v_scale.repeat_interleave(heads_per_group)[:, None, None]
                / 256.0
            )
        else:
            output *= panel.v_scale.float() / 256.0
        outputs.append(output.transpose(0, 1).to(torch.bfloat16))

    return torch.stack(outputs).reshape(
        batch * mtp, num_head_q, HEAD_DIM
    )


@pytest.mark.skipif(
    not _supports_sm90_fp8_backend(), reason="requires SM90 FP8 support"
)
@pytest.mark.parametrize(
    "num_batch", [1, 16, 200], ids=lambda value: f"batch{value}"
)
@pytest.mark.parametrize(
    "num_seq_q", SUPPORTED_TEST_MTP, ids=lambda value: f"mtp{value}"
)
@pytest.mark.parametrize(
    "max_seq_kv", [1024, 4096], ids=lambda value: f"kv{value}"
)
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize(
    "kv_head_q_head",
    [(1, 8), (4, 32)],
    ids=lambda value: f"gqa8-hkv{value[0]}-hq{value[1]}",
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("new_kv_included", [True])
@pytest.mark.parametrize("use_output", [False])
@pytest.mark.parametrize("splitk", [True])
@pytest.mark.parametrize("use_dynamic_sched", [False, True])
@pytest.mark.parametrize("kvcache_shape", ["NHD", "HND"])
@torch.no_grad()
def test_attn_fp8_sm90(
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
    assert block_size == BLOCK_SIZE
    assert head_dim == HEAD_DIM
    assert new_kv_included
    assert splitk
    assert not use_output

    num_head_kv, num_head_q = kv_head_q_head
    panel = _official_make_inputs(
        num_batch,
        num_seq_q,
        max_seq_kv,
        num_head_kv,
        num_head_q,
        kvcache_shape,
        QUANT_TYPE,
    )
    schedule = "dynamic" if use_dynamic_sched else "static"
    implementation = _implementation(schedule)
    inputs = _inputs(panel, implementation)
    workspace = implementation.prepare_decode_workspace(inputs)
    actual = implementation.attention_decode_fp8(
        inputs, workspace
    ).detach().clone()
    expected = _pytorch_reference(panel, num_seq_q, QUANT_TYPE)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    torch.testing.assert_close(
        actual,
        expected,
        atol=FP8_ATOL[QUANT_TYPE],
        rtol=1e-5,
    )
    assert torch.isfinite(actual).all()
    assert implementation.workspace_is_reset(workspace)
    if kvcache_shape == "HND" and num_head_kv > 1:
        assert panel.k_cache.stride(1) != num_head_kv * HEAD_DIM
