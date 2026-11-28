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

"""Pure PyTorch reference for Inkling paged relative attention."""

from __future__ import annotations

import torch


def ref_rel_attn(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute the operator semantics using PyTorch FP32 intermediates."""
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = q.shape[2]
    value_dim = value_cache.shape[3]
    block_size = key_cache.shape[1]
    group_size = num_heads // num_kv_heads
    batch_size = cache_seqlens.shape[0]
    out = torch.empty((*q.shape[:2], value_dim), device=q.device, dtype=q.dtype)

    for seq_id in range(batch_size):
        q_start = int(cu_seqlens_q[seq_id])
        q_end = int(cu_seqlens_q[seq_id + 1])
        q_len = q_end - q_start
        k_len = int(cache_seqlens[seq_id])
        num_blocks = (k_len + block_size - 1) // block_size
        physical_blocks = block_table[seq_id, :num_blocks].long()
        k = key_cache[physical_blocks].reshape(-1, num_kv_heads, head_dim)[:k_len].float()
        v = value_cache[physical_blocks].reshape(-1, num_kv_heads, value_dim)[:k_len].float()
        q_seq = q[q_start:q_end].float()
        rel_seq = rel_logits[q_start:q_end].float()

        q_pos = torch.arange(q_len, device=q.device).view(q_len, 1) + k_len - q_len
        k_pos = torch.arange(k_len, device=q.device).view(1, k_len)
        rel_dist = q_pos - k_pos
        rel_in_range = (rel_dist >= 0) & (rel_dist < rel_extent)
        rel_index = rel_dist.clamp(0, rel_extent - 1)

        mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device)
        if causal:
            mask &= k_pos <= q_pos
        if window_size[0] >= 0:
            mask &= k_pos >= q_pos - window_size[0]
        if window_size[1] >= 0:
            mask &= k_pos <= q_pos + window_size[1]

        for q_head in range(num_heads):
            kv_head = q_head // group_size
            scores = q_seq[:, q_head] @ k[:, kv_head].T
            scores *= softmax_scale
            rel_bias = rel_seq[:, q_head].gather(1, rel_index)
            scores += torch.where(
                rel_in_range,
                rel_bias,
                torch.zeros_like(rel_bias),
            )
            probabilities = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
            probabilities = torch.nan_to_num(probabilities)
            out[q_start:q_end, q_head] = (probabilities @ v[:, kv_head]).to(q.dtype)

    return out


__all__ = ["ref_rel_attn"]
