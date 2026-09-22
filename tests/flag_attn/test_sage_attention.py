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

from flag_attn.sage_attention import forward, per_block_int8


def _expand_scale(scale, block_size, length):
    return scale.repeat_interleave(block_size, dim=-1)[..., :length, None]


def _headwise_matmul(left, right):
    return torch.stack(
        [
            torch.stack([left[batch, head] @ right[batch, head] for head in range(left.shape[1])])
            for batch in range(left.shape[0])
        ]
    )


def _reference(q, k, v, q_scale, k_scale, tensor_layout, attn_mask=None):
    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    q = q.float() * _expand_scale(q_scale, 128, q.shape[-2])
    k = k.float() * _expand_scale(k_scale, 64, k.shape[-2])

    num_groups = q.shape[1] // k.shape[1]
    k = torch.repeat_interleave(k, num_groups, dim=1)
    v = torch.repeat_interleave(v, num_groups, dim=1)

    logits_log2 = _headwise_matmul(q, k.transpose(-1, -2))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits_log2 = logits_log2.masked_fill(~attn_mask, float("-inf"))
        else:
            logits_log2 = logits_log2 + attn_mask

    probabilities = torch.softmax(logits_log2 * math.log(2.0), dim=-1)
    output = _headwise_matmul(probabilities, v.float())
    lse = torch.logsumexp(logits_log2 * math.log(2.0), dim=-1) / math.log(2.0)

    if tensor_layout == "NHD":
        output = output.transpose(1, 2)
    return output, lse


@pytest.mark.sage_attention
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_forward_matches_dequantized_reference(tensor_layout, num_kv_heads):
    torch.manual_seed(2026)
    batch_size, num_query_heads, seq_len, head_dim = 1, 2, 128, 64
    q = torch.randn(batch_size, num_query_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    if tensor_layout == "NHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, tensor_layout=tensor_layout)
    actual, actual_lse = forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        return_lse=True,
    )
    expected, expected_lse = _reference(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout,
    )

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-2, rtol=2e-2)


def test_forward_rejects_non_positive_maxnreg():
    q = torch.zeros((1, 1, 128, 64), device="cuda", dtype=torch.int8)
    k = torch.zeros_like(q)
    v = torch.zeros(q.shape, device="cuda", dtype=torch.float16)
    q_scale = torch.ones((1, 1, 1), device="cuda")
    k_scale = torch.ones((1, 1, 2), device="cuda")

    with pytest.raises(ValueError, match="maxnreg must be positive"):
        forward(q, k, v, q_scale, k_scale, maxnreg=0)


@pytest.mark.sage_attention
@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_forward_supports_masks_and_partial_blocks(mask_kind):
    torch.manual_seed(7)
    q = torch.randn((1, 1, 129, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 1, 70, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    if mask_kind == "bool":
        attn_mask = torch.ones((1, 1, 129, 70), device="cuda", dtype=torch.bool)
        attn_mask[..., ::3] = False
    else:
        attn_mask = torch.zeros((1, 1, 129, 70), device="cuda", dtype=torch.float32)
        attn_mask[..., ::3] = -2.0

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    actual, actual_lse = forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        attn_mask=attn_mask,
        return_lse=True,
        maxnreg=168,
    )
    expected, expected_lse = _reference(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        "HND",
        attn_mask,
    )

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-2, rtol=2e-2)
