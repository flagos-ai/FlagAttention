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

from types import SimpleNamespace

import torch
import importlib
# Hygon PyTorch exposes its device through the CUDA-compatible API.
device = SimpleNamespace(vendor_name="hygon", name="cuda")
torch_device_fn = torch.cuda


class _Error:
    @staticmethod
    def backend_not_support(device_name):
        raise RuntimeError(f"Backend {device_name!r} is not supported")


error = _Error()




_OPERATOR_EXPORTS = {
    "sage_attention": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_forward": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "SPARSE_BLOCK_SIZE": (".msa.index_topk", "SPARSE_BLOCK_SIZE"),
    "chunk_kda_fwd_infer": (".FLA.kda.chunk_kda", "chunk_kda_fwd_infer"),
    "parallel_nsa": (".FLA.nsa.parallel_nsa", "parallel_nsa"),
    "parallel_nsa_compression": (".FLA.nsa.parallel_nsa_compression", "parallel_nsa_compression"),
    "chunk_gla": (".FLA.gla", "chunk_gla"),
    "minimax_m3_index_decode": (".msa", "minimax_m3_index_decode"),
    "minimax_m3_index_decode_score": (".msa", "minimax_m3_index_decode_score"),
    "minimax_m3_index_score": (".msa", "minimax_m3_index_score"),
    "minimax_m3_index_topk": (".msa", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (".msa", "minimax_m3_sparse_attn"),
    # The public minimax_m3_sparse_attn is a stage facade that resolves its
    # prefill primitive under this private name, so the plain key above never
    # reaches the prefill path on its own.
    "_minimax_m3_sparse_attn_prefill": (".msa", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (".msa", "minimax_m3_sparse_attn_decode"),
}


def __getattr__(name: str):
    try:
        module, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module, __name__), symbol)
    globals()[name] = value
    return value


__all__ = ["device", "torch_device_fn", "error", *sorted(_OPERATOR_EXPORTS)]
