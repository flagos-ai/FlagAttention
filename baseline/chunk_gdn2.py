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

"""NVIDIA native forward baseline for chunk_gdn2."""

import math

import torch


OP_NAME = "chunk_gdn2"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False

SHAPES = (
    (2, 512, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (2, 2048, 16, 256, 512),
)


def native():
    try:
        from flag_attn.gdn2.native.chunk_fwd import chunk_gdn2_fwd
    except (ImportError, OSError):
        return None
    return chunk_gdn2_fwd


def grid():
    return list(SHAPES)


def build_inputs(binding, dtype, device):
    b_size, seq_len, heads, key_dim, value_dim = binding
    torch.manual_seed(42)
    q = torch.randn(
        b_size, seq_len, heads, key_dim, device=device, dtype=dtype
    ) / math.sqrt(key_dim)
    k = torch.randn(
        b_size, seq_len, heads, key_dim, device=device, dtype=dtype
    ) / math.sqrt(key_dim)
    v = torch.randn(b_size, seq_len, heads, value_dim, device=device, dtype=dtype)
    g = (
        -torch.rand(
            b_size, seq_len, heads, key_dim, device=device, dtype=torch.float32
        ) * 0.1
    ).to(dtype)
    beta = torch.rand(b_size, seq_len, heads, key_dim, device=device, dtype=dtype)
    w = torch.rand(b_size, seq_len, heads, value_dim, device=device, dtype=dtype)
    initial_state = torch.randn(
        b_size, heads, key_dim, value_dim, device=device, dtype=torch.float32
    ) * 0.01
    return (), {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "b": beta,
        "w_gate": w,
        "scale": key_dim**-0.5,
        "initial_state": initial_state,
        "output_final_state": True,
        "chunk_size": 64,
        "disable_recompute": True,
    }


def key_shape(binding):
    return list(binding)


def config(binding, dtype):
    b_size, seq_len, heads, key_dim, value_dim = binding

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "phase": "forward",
        "dims": dict(zip(("B", "T", "H", "K", "V"), binding)),
        "inputs": {
            "q": tensor([b_size, seq_len, heads, key_dim]),
            "k": tensor([b_size, seq_len, heads, key_dim]),
            "v": tensor([b_size, seq_len, heads, value_dim]),
            "g": tensor([b_size, seq_len, heads, key_dim]),
            "b": tensor([b_size, seq_len, heads, key_dim]),
            "w_gate": tensor([b_size, seq_len, heads, value_dim]),
            "initial_state": tensor([b_size, heads, key_dim, value_dim], torch.float32),
            "scale": {"scalar": key_dim**-0.5},
            "output_final_state": {"scalar": True},
            "chunk_size": {"scalar": 64},
            "disable_recompute": {"scalar": True},
        },
        "outputs": {
            "output": tensor([b_size, seq_len, heads, value_dim]),
            "final_state": tensor([b_size, heads, key_dim, value_dim], torch.float32),
        },
        "mutation": "none",
        "graph": False,
    }
