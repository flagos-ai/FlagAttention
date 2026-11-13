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

"""NVIDIA FLA baseline for Log-Linear Attention forward benchmarks."""

import importlib
import math

import torch


OP_NAME = "log_linear_attn"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

# Keep these cases aligned with benchmark/test_log_linear_attn_benchmark.py.
BENCHMARK_SHAPES = (
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (8, 1024, 8, 64),
)


def native():
    """Return FLA's public CUDA implementation when it is installed."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        op = importlib.import_module("fla.ops.log_linear_attn").chunk_log_linear_attn
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "FLA chunk_log_linear_attn is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return list(BENCHMARK_SHAPES)


def build_inputs(binding, dtype, device):
    batch, sequence, heads, dim = binding
    levels = math.ceil(math.log2(sequence)) + 1
    torch.manual_seed(0)
    q = torch.randn(batch, sequence, 1, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, sequence, heads, dim, device=device, dtype=dtype)
    g = -0.05 * torch.rand(batch, sequence, heads, device=device, dtype=torch.float32)
    level_scales = torch.sigmoid(
        torch.randn(
            batch, sequence, heads, levels, device=device, dtype=torch.float32
        )
    ).to(dtype)
    return (q, k, v, g, level_scales), {}


def key_shape(binding):
    return list(binding)


def config(binding, dtype):
    batch, sequence, heads, dim = binding
    levels = math.ceil(math.log2(sequence)) + 1

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "benchmark_scope": "public_api_forward",
        "phase": "forward",
        "dims": dict(zip(("B", "T", "H", "D"), binding)),
        "inputs": {
            "q": tensor([batch, sequence, 1, dim]),
            "k": tensor([batch, sequence, 1, dim]),
            "v": tensor([batch, sequence, heads, dim]),
            "g": tensor([batch, sequence, heads], torch.float32),
            "level_scales": tensor([batch, sequence, heads, levels]),
        },
        "outputs": {
            "output": tensor([batch, sequence, heads, dim]),
            "final_state": {"scalar": None},
        },
        "input_seed": 0,
        "mutation": "none",
        "graph": False,
    }
