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

"""CUDA FLA forward baseline for chunk_gla."""

import torch
import torch.nn.functional as F


OP_NAME = "chunk_gla"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = (
    (1, 4096, 32, 512), (2, 2048, 16, 512), (1, 8192, 96, 128),
    (2, 16384, 16, 128), (4, 2048, 16, 128), (4, 4096, 64, 128),
    (8, 2048, 32, 256), (4, 1024, 8, 512), (8, 1024, 8, 64),
)


def native():
    try:
        from fla.ops.gla import chunk_gla
    except (ImportError, OSError):
        return None
    return chunk_gla


def grid():
    return list(SHAPES)


def build_inputs(binding, dtype, device):
    b, t, h, d = binding
    shape = (b, t, h, d)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    g_logit = torch.randn(shape, device=device, dtype=dtype)
    g = F.logsigmoid(g_logit)
    return (q, k, v, g), {
        "scale": d**-0.5, "initial_state": None,
        "output_final_state": False, "state_v_first": False,
        "cu_seqlens": None,
    }


def key_shape(binding):
    return list(binding)


def config(binding, dtype):
    b, t, h, d = binding
    shape = [b, t, h, d]
    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "phase": "forward",
        "dims": dict(zip(("B", "T", "H", "D"), binding)),
        "inputs": {
            name: {"shape": shape, "dtype": str(dtype)}
            for name in ("q", "k", "v", "g")
        },
        "outputs": {"output": {"shape": shape, "dtype": str(dtype)}},
        "mutation": "none",
        "graph": False,
    }
