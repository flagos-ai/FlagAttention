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

"""CUDA baseline collection contract for parallel_wall_attn."""

import torch


OP_NAME = "parallel_wall_attn"
DTYPES = [torch.bfloat16, torch.float16]
IS_INPLACE = False

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096)
SHAPE_FAMILIES = (("mha", 8), ("gqa", 2))


def native():
    try:
        from fla.ops.wall_attn import parallel_wall_attn
    except (ImportError, OSError):
        return None
    return parallel_wall_attn


def grid():
    return [
        {"B": 1, "T": t, "H": h, "HQ": 8, "D": 64, "family": family}
        for family, h in SHAPE_FAMILIES
        for t in SEQUENCE_LENGTHS
    ]


def build_inputs(binding, dtype, device):
    t, h = binding["T"], binding["H"]
    torch.manual_seed(0)
    q = torch.randn(1, t, 8, 64, device=device, dtype=dtype)
    k = torch.randn(1, t, h, 64, device=device, dtype=dtype)
    v = torch.randn(1, t, h, 64, device=device, dtype=dtype)
    g = (-torch.randn(1, t, 8, 64, device=device).abs() * 0.05).to(dtype)
    return (q, k, v, g), {"scale": 64**-0.5}


def key_shape(binding):
    return [binding[key] for key in ("B", "T", "H", "HQ", "D", "family")]


def config(binding, dtype):
    t, h = binding["T"], binding["H"]

    def tensor(shape):
        return {"shape": shape, "dtype": str(dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "dims": dict(binding),
        "inputs": {
            "q": tensor([1, t, 8, 64]),
            "k": tensor([1, t, h, 64]),
            "v": tensor([1, t, h, 64]),
            "g": tensor([1, t, 8, 64]),
            "scale": {"scalar": 64**-0.5},
        },
        "outputs": {"output": tensor([1, t, 8, 64])},
        "mutation": "none",
        "graph": False,
    }
