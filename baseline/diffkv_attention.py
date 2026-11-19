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

"""NVIDIA FA3 baseline for DiffKV Full Attention and SWA decode."""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch


OP_NAME = "diffkv_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None


@lru_cache(maxsize=1)
def _benchmark():
    root = Path(__file__).resolve().parents[1]
    benchmark_dir = root / "benchmark"
    path = benchmark_dir / "test_diffkv_attention_benchmark.py"
    name = "_flag_attn_diffkv_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    inserted = [
        str(directory)
        for directory in (root, benchmark_dir)
        if str(directory) not in sys.path
    ]
    for directory in inserted:
        sys.path.insert(0, directory)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    finally:
        for directory in inserted:
            sys.path.remove(directory)
    return module


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        benchmark = _benchmark()
    except (ImportError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = str(exc)
        return None
    available, reason = benchmark._load_fa3_provider()
    if not available:
        NATIVE_UNAVAILABLE_REASON = reason
        return None
    NATIVE_UNAVAILABLE_REASON = None

    def fa3_native(runner):
        return runner()

    return fa3_native


def grid():
    benchmark = _benchmark()
    return [
        {"mode": mode, "batch": batch, "kv_length": length, "path": path}
        for mode in benchmark._MODES
        for batch, length in benchmark._SHAPES
        for path in benchmark._PATHS
    ]


def _dims(binding):
    benchmark = _benchmark()
    layout = benchmark.DEFAULT_LAYOUT
    return {
        "mode": binding["mode"],
        "batch": binding["batch"],
        "query_length": 1,
        "kv_length": binding["kv_length"],
        "num_query_heads": layout.num_query_heads,
        "num_kv_heads": benchmark.diffkv_impl.num_kv_heads_for_mode(binding["mode"]),
        "head_size_qk": layout.head_size_qk,
        "head_size_v": layout.head_size_v,
        "path": binding["path"],
    }


def build_inputs(binding, dtype, device):
    benchmark = _benchmark()
    dims = _dims(binding)
    window = -1 if binding["mode"] == "full" else benchmark.WINDOW_SIZE
    inputs = benchmark.make_inputs(
        dims["batch"],
        dims["kv_length"],
        dtype,
        dims["num_query_heads"],
        dims["num_kv_heads"],
        dims["head_size_qk"],
        dims["head_size_v"],
        benchmark.DEFAULT_LAYOUT.block_size,
        seed=benchmark.shape_seed(
            benchmark.DEFAULT_SEED,
            dims["mode"],
            dims["batch"],
            dims["kv_length"],
            dims["num_kv_heads"],
        ),
        page_table=benchmark.PAGE_TABLE,
        device=device,
    )
    runner = benchmark.build_fa3_runner(inputs, window)
    return (runner,), {}


def key_shape(binding):
    return list(_dims(binding).values())


def config(binding, dtype):
    benchmark = _benchmark()
    dims = _dims(binding)
    batch = dims["batch"]
    length = dims["kv_length"]
    hq = dims["num_query_heads"]
    hkv = dims["num_kv_heads"]
    dqk = dims["head_size_qk"]
    dv = dims["head_size_v"]
    block_size = benchmark.DEFAULT_LAYOUT.block_size
    pages = (length + block_size - 1) // block_size
    num_pages = batch * pages
    packed_dim = dqk + dv
    cache_stride = [
        hkv * block_size * packed_dim,
        packed_dim,
        block_size * packed_dim,
        1,
    ]
    window = -1 if dims["mode"] == "full" else benchmark.WINDOW_SIZE

    def tensor(shape, tensor_dtype=dtype, stride=None):
        description = {"shape": shape, "dtype": str(tensor_dtype)}
        if stride is not None:
            description["stride"] = stride
        return description

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "native_api": "torch.ops._vllm_fa3_C.fwd",
        "benchmark_scope": "preallocated_forward_kernel",
        "phase": "decode",
        "dims": dims,
        "inputs": {
            "query": tensor([batch, hq, dqk]),
            "key_cache": tensor([num_pages, block_size, hkv, dqk], stride=cache_stride),
            "value_cache": tensor([num_pages, block_size, hkv, dv], stride=cache_stride),
            "context_lens": tensor([batch], torch.int32),
            "block_tables": tensor([batch, pages], torch.int32),
            "cu_seqlens_q": tensor([batch + 1], torch.int32),
            "softmax_scale": {"scalar": dqk**-0.5},
            "window_size": {"scalar": window},
            "causal": {"scalar": True},
        },
        "outputs": {"output": tensor([batch, hq, dv])},
        "input_seed": benchmark.shape_seed(
            benchmark.DEFAULT_SEED, dims["mode"], batch, length, hkv
        ),
        "page_table": benchmark.PAGE_TABLE,
        "mutation": "none",
        "graph": False,
    }
