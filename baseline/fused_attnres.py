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

"""NVIDIA FLA baseline for FlagAttention's fused_attnres public API."""

import importlib

import torch


OP_NAME = "fused_attnres"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

# Keep these cases aligned with benchmark/test_fused_attnres_benchmark.py.
BENCHMARK_SHAPES = ((2, 1), (5, 128), (5, 1024), (9, 128), (9, 8192))
HIDDEN_SIZE = 7168
INPUT_SEED = 0
BENCHMARK_SCOPE = "public_api_forward"


def native():
    """Return FLA's CUDA public API, or None when FLA is unavailable."""
    global NATIVE_UNAVAILABLE_REASON
    try:
        op = importlib.import_module("fla.ops.attnres.fused").fused_attnres
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "FLA fused_attnres is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    """Cover every shape and output-norm variant in the pytest benchmark."""
    return [
        {
            "num_sources": num_sources,
            "num_rows": num_rows,
            "hidden_size": HIDDEN_SIZE,
            "output_norm": output_norm,
        }
        for num_sources, num_rows in BENCHMARK_SHAPES
        for output_norm in (False, True)
    ]


def build_inputs(binding, dtype, device):
    """Reproduce the benchmark's public-API inputs and call arguments."""
    num_sources = binding["num_sources"]
    num_rows = binding["num_rows"]
    hidden_size = binding["hidden_size"]
    torch.manual_seed(INPUT_SEED)
    residuals = [
        torch.randn(num_rows, hidden_size, device=device, dtype=dtype)
        for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device=device, dtype=dtype)
    rms_weight = torch.randn(hidden_size, device=device, dtype=dtype)
    output_rms_weight = (
        torch.randn(hidden_size, device=device, dtype=dtype)
        if binding["output_norm"]
        else None
    )
    return (
        query,
        residuals,
        rms_weight,
        output_rms_weight,
    ), {"scale": hidden_size**-0.5, "return_weights": False}


def key_shape(binding):
    """Use all benchmark identity fields; query shape alone is ambiguous."""
    return [
        binding["num_sources"],
        binding["num_rows"],
        binding["hidden_size"],
        binding["output_norm"],
    ]


def config(binding, dtype):
    """Describe the exact public-API workload stored in the baseline JSON."""
    num_sources = binding["num_sources"]
    num_rows = binding["num_rows"]
    hidden_size = binding["hidden_size"]
    output_norm = binding["output_norm"]

    def tensor(shape):
        return {"shape": shape, "dtype": str(dtype)}

    inputs = {
        "query": tensor([hidden_size]),
        "rms_weight": tensor([hidden_size]),
        "output_rms_weight": (
            tensor([hidden_size]) if output_norm else {"scalar": None}
        ),
        "scale": {"scalar": hidden_size**-0.5},
        "return_weights": {"scalar": False},
    }
    inputs.update(
        {
            f"residual_{index}": tensor([num_rows, hidden_size])
            for index in range(num_sources)
        }
    )
    return {
        "op": OP_NAME,
        "benchmark_scope": BENCHMARK_SCOPE,
        "dtype": str(dtype),
        "dims": dict(binding),
        "inputs": inputs,
        "outputs": {"output": tensor([num_rows, hidden_size])},
        "input_seed": INPUT_SEED,
        "mutation": "none",
        "graph": False,
    }
