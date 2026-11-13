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

"""Official CuTe forward baseline for Inkling FA4 relative attention."""

import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


OP_NAME = "inkling_fa4_rel_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = ("out",)

HEAD_DIM = 128
BLOCK_SIZE = 16
CASES = (
    ("full_prefill", ((64, 64),), 4, 4, 128, None),
    ("ragged_prefill", ((64, 64), (33, 33), (17, 17)), 8, 2, 128, None),
    ("long_prefill", ((512, 512),), 8, 2, 1024, None),
    ("chunked_prefill", ((32, 512),), 8, 2, 1024, None),
    ("sliding_window", ((64, 512),), 8, 2, 256, 255),
    ("decode_1k", ((1, 1024),), 8, 2, 1024, None),
    ("decode_8k", ((1, 8192),), 8, 2, 1024, None),
)


def native():
    try:
        importlib.import_module("cutlass.cute")
        flash_root = os.getenv("FLASH_ATTN_ROOT")
        if flash_root:
            package_dir = Path(flash_root).expanduser() / "flash_attn"
            if not (package_dir / "cute" / "interface.py").is_file():
                return None
            package = types.ModuleType("flash_attn")
            package.__path__ = [str(package_dir)]
            package.__package__ = "flash_attn"
            sys.modules["flash_attn"] = package
            importlib.invalidate_caches()
        return importlib.import_module("flash_attn.cute").flash_attn_varlen_func
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None


def grid():
    return [
        {
            "case": name, "seq_lens": seq_lens, "num_heads": heads,
            "num_kv_heads": kv_heads, "rel_extent": extent,
            "window_left": window_left, "num_splits": 1, "rel": "real",
            "seed": index,
        }
        for index, (name, seq_lens, heads, kv_heads, extent, window_left)
        in enumerate(CASES)
    ]


def _score_mod(rel_extent):
    import cutlass.cute as cute
    from cutlass.cute import Float32
    from flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        del b_idx
        rel_logits = aux_tensors[0]
        local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + local_offset) - kv_idx
        global_q = seqlen_info.offset_q + q_idx
        distance = rel_dist[0]
        rel_index = distance if distance >= 0 else 0
        rel_index = rel_index if rel_index < rel_extent else rel_extent - 1
        bias = rel_logits[global_q[0], h_idx[0], rel_index]
        bias = Float32(bias) if distance == rel_index else Float32(0.0)
        return scores + bias

    return score_mod_rel_bias


def build_inputs(binding, dtype, device):
    torch.manual_seed(binding["seed"])
    seq_lens = binding["seq_lens"]
    q_lens = [ql for ql, _ in seq_lens]
    kv_lens = [kl for _, kl in seq_lens]
    total_q = sum(q_lens)
    count = len(seq_lens)
    heads = binding["num_heads"]
    kv_heads = binding["num_kv_heads"]
    extent = binding["rel_extent"]
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, heads, HEAD_DIM, device=device, dtype=dtype)
    q = F.normalize(q.float(), dim=-1).to(dtype)
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = count * max_blocks + 1
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM, device=device, dtype=dtype)
    k_cache = F.normalize(k_cache.float(), dim=-1).to(dtype)
    v_cache = torch.randn(num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM, device=device, dtype=dtype)
    block_table = torch.zeros(count, max_blocks, dtype=torch.int32, device=device)
    for seq in range(count):
        block_table[seq] = torch.arange(
            1 + seq * max_blocks, 1 + (seq + 1) * max_blocks,
            dtype=torch.int32, device=device,
        )
    cum_q = [0]
    for q_len in q_lens:
        cum_q.append(cum_q[-1] + q_len)
    cu_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    rel_logits = torch.randn(total_q, heads, extent, device=device, dtype=dtype)
    window = (-1, -1) if binding["window_left"] is None else (binding["window_left"], 0)
    return (), {
        "q": q, "k": k_cache, "v": v_cache,
        "cu_seqlens_q": cu_q, "seqused_k": cache_seqlens,
        "max_seqlen_q": max(q_lens), "page_table": block_table,
        "softmax_scale": scale, "causal": True, "window_size": window,
        "num_splits": binding["num_splits"],
        "score_mod": _score_mod(extent), "aux_tensors": [rel_logits],
        "return_lse": False, "out": torch.empty_like(q),
    }


def key_shape(binding):
    return [binding["case"], binding["num_splits"], binding["rel"]]


def config(binding, dtype):
    q_lens = [ql for ql, _ in binding["seq_lens"]]
    kv_lens = [kl for _, kl in binding["seq_lens"]]
    total_q, count = sum(q_lens), len(q_lens)
    heads, kv_heads = binding["num_heads"], binding["num_kv_heads"]
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = count * max_blocks + 1

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    return {
        "op": OP_NAME, "dtype": str(dtype),
        "dims": {key: binding[key] for key in ("case", "num_splits", "rel")},
        "inputs": {
            "q": tensor([total_q, heads, HEAD_DIM]),
            "k": tensor([num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM]),
            "v": tensor([num_blocks, BLOCK_SIZE, kv_heads, HEAD_DIM]),
            "cu_seqlens_q": tensor([count + 1], torch.int32),
            "seqused_k": tensor([count], torch.int32),
            "max_seqlen_q": {"scalar": max(q_lens)},
            "page_table": tensor([count, max_blocks], torch.int32),
            "softmax_scale": {"scalar": 1.0 / HEAD_DIM},
            "causal": {"scalar": True},
            "window_size": {"scalar": (-1, -1) if binding["window_left"] is None
                            else (binding["window_left"], 0)},
            "num_splits": {"scalar": binding["num_splits"]},
            "aux_tensors": tensor([total_q, heads, binding["rel_extent"]]),
            "return_lse": {"scalar": False},
            "out": tensor([total_q, heads, HEAD_DIM]),
        },
        "outputs": {"out": tensor([total_q, heads, HEAD_DIM])},
        "mutation": "out", "graph": False,
    }
