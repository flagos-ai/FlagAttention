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

"""CUDA FlashMoBA baseline for parallel_moba."""

from itertools import accumulate

import torch


OP_NAME = "parallel_moba"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SEQUENCE_LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
NUM_HEADS = 16
HEAD_DIM = 128
CHUNK_SIZE = 128
TOPK = 8


def native():
    try:
        from flash_moba import flash_moba_varlen_func
    except (ImportError, OSError):
        return None

    def flash_moba(q, k, v, cu_seqlens, max_seqlen, chunk_size, topk):
        return flash_moba_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            moba_chunk_size=chunk_size, moba_topk=topk, causal=True,
        )

    return flash_moba


def grid():
    return [
        ((length, length), NUM_HEADS, HEAD_DIM, CHUNK_SIZE, TOPK)
        for length in SEQUENCE_LENGTHS
    ]


def build_inputs(binding, dtype, device):
    seq_lens, heads, head_dim, chunk_size, topk = binding
    torch.manual_seed(42)
    total = sum(seq_lens)
    shape = (total, heads, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    cu_seqlens = torch.tensor(
        [0, *accumulate(seq_lens)], dtype=torch.int32, device=device,
    )
    return (q, k, v, cu_seqlens, max(seq_lens), chunk_size, topk), {}


def key_shape(binding):
    return list(binding)


def config(binding, dtype):
    seq_lens, heads, head_dim, chunk_size, topk = binding
    total = sum(seq_lens)

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "dims": {
            "seq_lens": seq_lens, "num_heads": heads, "head_dim": head_dim,
            "chunk_size": chunk_size, "topk": topk,
        },
        "inputs": {
            "q": tensor([total, heads, head_dim]),
            "k": tensor([total, heads, head_dim]),
            "v": tensor([total, heads, head_dim]),
            "cu_seqlens": tensor([len(seq_lens) + 1], torch.int32),
            "max_seqlen": {"scalar": max(seq_lens)},
            "chunk_size": {"scalar": chunk_size},
            "topk": {"scalar": topk},
            "causal": {"scalar": True},
        },
        "outputs": {"output": tensor([total, heads, head_dim])},
        "mutation": "none",
        "graph": False,
    }
