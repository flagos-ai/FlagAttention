# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""MiniMax M3 Sparse Attention with a paged KV cache.

The cache layout is compatible with vLLM:
  kv_cache: [num_blocks, num_kv_heads, 128, 2*head_dim]  K=[..., :head_dim] V=[..., head_dim:]
  index_kv_cache: [num_blocks, 128, head_dim]
  block_table: [batch, max_blocks]
"""

from .index_topk import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_decode_score,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from .sparse_attn import minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
]
