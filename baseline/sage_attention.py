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

"""PyTorch CUDA SDPA baseline for the SageAttention pytest benchmark."""

import torch
import torch.nn.functional as F


OP_NAME = "sage_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = (
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
)


def native():
    op = getattr(F, "scaled_dot_product_attention", None)
    if op is None:
        return None

    def sdpa_forward(q, k, v, *, output_dtype):
        return op(q, k, v, dropout_p=0.0, is_causal=False).to(output_dtype)

    return sdpa_forward


def grid():
    return list(SHAPES)


def build_inputs(binding, dtype, device):
    torch.manual_seed(42)
    q = torch.randn(binding, device=device, dtype=torch.float16)
    k = torch.randn(binding, device=device, dtype=torch.float16)
    v = torch.randn(binding, device=device, dtype=torch.float16)
    return (q, k, v), {"output_dtype": dtype}


def key_shape(binding):
    return list(binding)


def config(binding, dtype):
    batch, heads, sequence, dim = binding
    input_tensor = {"shape": list(binding), "dtype": str(torch.float16)}
    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "phase": "forward",
        "dims": {"B": batch, "H": heads, "T": sequence, "D": dim},
        "inputs": {
            "q": dict(input_tensor),
            "k": dict(input_tensor),
            "v": dict(input_tensor),
            "dropout_p": {"scalar": 0.0},
            "is_causal": {"scalar": False},
            "output_dtype": {"scalar": str(dtype)},
        },
        "outputs": {"output": {"shape": list(binding), "dtype": str(dtype)}},
        "mutation": "none",
        "graph": False,
    }
