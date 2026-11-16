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

"""CUDA baseline collection contract for chunk_gated_delta_rule."""

import torch


OP_NAME = "chunk_gated_delta_rule"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False

SHAPES = (
    (1, 8192, 96, 128, 128),
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
)


def native():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except (ImportError, OSError):
        return None
    return chunk_gated_delta_rule


def grid():
    return list(SHAPES)


def build_inputs(binding, dtype, device):
    b, t, h, k_size, v_size = binding
    torch.manual_seed(42)
    q = torch.randn(b, t, h, k_size, device=device, dtype=dtype) / k_size**0.5
    k = torch.randn(b, t, h, k_size, device=device, dtype=dtype) / k_size**0.5
    v = torch.randn(b, t, h, v_size, device=device, dtype=dtype)
    g = (-torch.rand(b, t, h, device=device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(b, t, h, device=device, dtype=dtype).sigmoid()
    return (q, k, v, g, beta), {
        "scale": k_size**-0.5,
        "output_final_state": True,
    }


def key_shape(binding):
    return list(binding)


def config(binding, dtype):
    b, t, h, k_size, v_size = binding

    def tensor(shape):
        return {"shape": shape, "dtype": str(dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "dims": dict(zip(("B", "T", "H", "K", "V"), binding)),
        "inputs": {
            "q": tensor([b, t, h, k_size]),
            "k": tensor([b, t, h, k_size]),
            "v": tensor([b, t, h, v_size]),
            "g": tensor([b, t, h]),
            "beta": tensor([b, t, h]),
            "scale": {"scalar": k_size**-0.5},
            "output_final_state": {"scalar": True},
        },
        "outputs": {
            "output": tensor([b, t, h, v_size]),
            "final_state": {
                "shape": [b, h, k_size, v_size],
                "dtype": str(torch.float32),
            },
        },
        "mutation": "none",
        "graph": False,
    }
