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

from flag_attn.hpc_ops_attention.prefill import (
    attention_with_kvcache_blocksparse_prefill_fp8,
)

BLOCK = 128
HEAD_DIM = 128
Q_HEADS = 8
KV_HEADS = 2


def test_attention_blocksparse_prefill_fp8_rejects_cpu():
    q = torch.empty(1, 1, HEAD_DIM)
    cache = torch.empty(1, 32, 1, HEAD_DIM)
    scale = torch.empty(1)
    metadata = torch.empty(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="must be a CUDA tensor"):
        attention_with_kvcache_blocksparse_prefill_fp8(
            q,
            cache,
            cache,
            scale,
            scale,
            scale,
            torch.empty(2, dtype=torch.int32),
            metadata.view(1, 1),
            metadata,
            1,
        )


def _make_mask(q_len, kv_len, masked, device):
    if not masked:
        return None
    q_tiles = math.ceil(q_len / BLOCK)
    kv_tiles = math.ceil(kv_len / BLOCK)
    rows = torch.arange(q_tiles).view(q_tiles, 1)
    cols = torch.arange(kv_tiles).view(1, kv_tiles)
    valid = cols <= rows + (kv_tiles - q_tiles)
    mask = valid.view(1, 1, q_tiles, kv_tiles).expand(1, Q_HEADS, -1, -1).clone()
    mask[:, :, :, 0] = False
    return mask.to(device=device, dtype=torch.uint8).contiguous()


def _make_inputs(quant_type, kv_layout, masked, page_size):
    device = torch.device("cuda")
    fp8 = torch.float8_e4m3fn
    q_len, kv_len = 129, 257
    pages = math.ceil(kv_len / page_size)

    q = (
        torch.randn(q_len, Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        / math.sqrt(HEAD_DIM)
    ).to(fp8)
    k = (
        torch.randn(
            pages,
            page_size,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        / math.sqrt(HEAD_DIM)
    ).to(fp8)
    v = torch.randn(
        pages,
        page_size,
        KV_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8)
    if kv_layout == "hnd":
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
        v = v.transpose(1, 2).contiguous().transpose(1, 2)

    q_scale = torch.rand(1, Q_HEADS, 256, device=device, dtype=torch.float32)
    q_scale[:, :, q_len:] = 0
    if quant_type == 1:
        k_scale = torch.rand(1, device=device, dtype=torch.float32) + 0.5
        v_scale = torch.rand(1, device=device, dtype=torch.float32) + 0.5
    else:
        k_scale = (
            torch.rand(
                pages,
                page_size // 32,
                KV_HEADS,
                HEAD_DIM // 4,
                device=device,
                dtype=torch.float32,
            )
            .clamp_min_(1e-6)
            .view(fp8)
        )
        v_scale = torch.rand(KV_HEADS, device=device, dtype=torch.float32) + 0.5

    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    block_ids = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
    kv_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_mask = _make_mask(q_len, kv_len, masked, device)
    return (
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        cu_seqlens_q,
        block_ids,
        kv_lens,
        q_len,
        quant_type,
        block_mask,
    )


@torch.no_grad()
def _reference(args):
    (
        q,
        k_cache,
        v_cache,
        q_scale,
        k_scale,
        v_scale,
        _cu_seqlens_q,
        block_ids,
        kv_lens,
        q_len,
        quant_type,
        block_mask,
    ) = args
    kv_len = int(kv_lens[0].item())
    page_ids = block_ids[0]
    paged_k = k_cache[page_ids].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len]
    paged_v = v_cache[page_ids].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len]
    packed_k_scale = k_scale.view(torch.float32) if quant_type == 0 else None
    token_k_scale = None
    if packed_k_scale is not None:
        token_k_scale = (
            packed_k_scale[page_ids].permute(0, 1, 3, 2).reshape(-1, KV_HEADS)[:kv_len]
        )

    output = torch.empty_like(q, dtype=torch.bfloat16)
    group_size = Q_HEADS // KV_HEADS
    key_positions = torch.arange(kv_len, device=q.device)
    key_tiles = torch.div(key_positions, BLOCK, rounding_mode="floor")
    q_positions = torch.arange(q_len, device=q.device)
    causal = key_positions.unsqueeze(0) <= (q_positions.unsqueeze(1) + kv_len - q_len)

    for kv_head in range(KV_HEADS):
        head_start = kv_head * group_size
        head_end = head_start + group_size
        q_group = q[:, head_start:head_end].permute(1, 0, 2).float()
        k_head = paged_k[:, kv_head].float()
        v_head = paged_v[:, kv_head].float()
        k_head_t = k_head.transpose(0, 1).contiguous()
        scores = torch.stack(
            [
                torch.mm(q_group[index].contiguous(), k_head_t)
                for index in range(group_size)
            ]
        )
        query_scale = q_scale[0, head_start:head_end, :q_len].unsqueeze(-1)
        if quant_type == 0:
            key_scale = token_k_scale[:, kv_head].view(1, 1, kv_len)
            value_scale = v_scale[kv_head]
        else:
            key_scale = k_scale[0]
            value_scale = v_scale[0]
        scores = scores * query_scale * key_scale / math.sqrt(HEAD_DIM)

        valid = causal.unsqueeze(0)
        if block_mask is not None:
            query_tiles = torch.div(q_positions, BLOCK, rounding_mode="floor")
            sparse = block_mask[0, head_start:head_end].bool()
            sparse = sparse.index_select(1, query_tiles).index_select(2, key_tiles)
            valid = valid & sparse
        scores = scores.masked_fill(~valid, float("-inf"))
        probabilities = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
        denominator = probabilities.sum(dim=-1, keepdim=True)
        probabilities = (probabilities * 256.0).to(torch.float8_e4m3fn).float()
        v_head = v_head.contiguous()
        result = torch.stack(
            [
                torch.mm(probabilities[index].contiguous(), v_head)
                for index in range(group_size)
            ]
        )
        result = result * (value_scale / 256.0) / denominator
        output[:, head_start:head_end] = result.permute(1, 0, 2)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("quant_type", [0, 1])
@pytest.mark.parametrize("kv_layout", ["nhd", "hnd"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("page_size", [32, 64])
def test_attention_blocksparse_prefill_fp8(quant_type, kv_layout, masked, page_size):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    args = _make_inputs(quant_type, kv_layout, masked, page_size)
    reference = _reference(args)
    output = attention_with_kvcache_blocksparse_prefill_fp8(
        *args[:-1],
        block_mask=args[-1],
        sparsity_bucket=2 if masked else None,
    )
    assert torch.isfinite(reference).all(), "PyTorch reference contains NaN or Inf"
    assert torch.isfinite(output).all(), "attention output contains NaN or Inf"
    torch.testing.assert_close(output.float(), reference.float(), atol=0.1, rtol=0.1)
