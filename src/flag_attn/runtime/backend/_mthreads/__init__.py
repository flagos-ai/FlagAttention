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

"""MThreads attention backend."""

from __future__ import annotations

import importlib


_OPERATOR_EXPORTS = {
    "chunk_gla": (".FLA.gla", "chunk_gla"),
    "chunk_gdn2": (".FLA.gdn2", "chunk_gdn2"),
    "chunk_gdn2_native": (".FLA.gdn2.native.chunk_fwd", "chunk_gdn2_fwd"),
    "chunk_kda": (".FLA.kda", "chunk_kda"),
    "minimax_m3_index_decode": (".minimax_sparse_attention", "minimax_m3_index_decode"),
    "minimax_m3_index_decode_score": (
        ".minimax_sparse_attention",
        "minimax_m3_index_decode_score",
    ),
    "minimax_m3_index_score": (".minimax_sparse_attention", "minimax_m3_index_score"),
    "minimax_m3_index_topk": (".minimax_sparse_attention", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (".minimax_sparse_attention", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (
        ".minimax_sparse_attention",
        "minimax_m3_sparse_attn_decode",
    ),
    "parallel_nsa": (".FLA.nsa", "parallel_nsa"),
    "parallel_nsa_compression": (".FLA.nsa", "parallel_nsa_compression"),
    "sage_attention_forward": (".sage_attention", "forward"),
    "sage_attention_per_block_int8": (".sage_attention", "per_block_int8"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
