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

"""Conservative, architecture-independent launch sizes for portable Triton."""

from __future__ import annotations


def get_preset(
    max_seqlen_q: int,
    *,
    max_seqlen_k: int | None = None,
    head_dim_q: int = 128,
    head_dim_v: int = 128,
    q_heads_per_kv_head: int = 1,
    arch: str | None = None,
) -> dict[str, int]:
    """Return portable tile sizes; ``arch`` is retained for API compatibility."""
    del max_seqlen_k, q_heads_per_kv_head, arch
    return {
        "BLOCK_Q": 16 if max_seqlen_q <= 16 else 32,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 1 if max(head_dim_q, head_dim_v) > 128 else 2,
    }


def get_tle_preset(
    max_seqlen_q: int,
    *,
    max_seqlen_k: int | None = None,
    head_dim_q: int = 128,
    head_dim_v: int = 128,
    q_heads_per_kv_head: int = 1,
    arch: str | None = None,
) -> dict[str, int]:
    """Compatibility preset for the portable implementation."""
    return get_preset(
        max_seqlen_q, max_seqlen_k=max_seqlen_k,
        head_dim_q=head_dim_q, head_dim_v=head_dim_v,
        q_heads_per_kv_head=q_heads_per_kv_head, arch=arch,
    )
