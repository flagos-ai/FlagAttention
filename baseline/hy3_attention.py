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

"""hpc CUDA baseline for Hy3 BF16 decode and FP8 decode/prefill benchmarks."""

import math
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import torch


OP_NAME = "hy3_attention"
DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
IS_INPLACE = True
MUTATED_INPUTS = ("output",)

_HPC = None
_NATIVE_APIS = {
    "decode_bf16": "attention_decode_bf16",
    "decode_fp8": "attention_decode_fp8",
    "prefill_fp8": "attention_with_kvcache_blocksparse_prefill_fp8",
}


@lru_cache(maxsize=3)
def _load_benchmark(stem):
    path = Path(__file__).resolve().parents[1] / "benchmark" / f"{stem}.py"
    name = f"_flag_attn_baseline_{stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native():
    global _HPC
    try:
        _HPC = _load_benchmark("test_hpc_ops_decode_bf16_perf")._load_hpc()
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None
    if _HPC is None:
        return None

    def hpc_native(*args, variant, **kwargs):
        return getattr(_HPC, _NATIVE_APIS[variant])(*args, **kwargs)

    return hpc_native


def grid():
    bf16 = _load_benchmark("test_hpc_ops_decode_bf16_perf")
    fp8 = _load_benchmark("test_hpc_ops_decode_fp8_perf")
    prefill = _load_benchmark("test_hpc_ops_prefill_fp8_perf")
    return [
        {
            "variant": "decode_bf16", "case": case, "mtp": mtp,
            "method": method, "layout": layout,
        }
        for layout in bf16.BENCH_LAYOUTS
        for method in bf16.BENCH_METHODS
        for case in bf16.BENCH_CASES
        for mtp in bf16.BENCH_MTP
    ] + [
        {
            "variant": "decode_fp8", "case": case, "mtp": mtp,
            "quant_type": quant_type, "schedule": schedule,
            "layout": layout,
        }
        for layout in fp8.BENCH_LAYOUTS
        for case in fp8.BENCH_CASES
        for schedule in fp8.BENCH_SCHEDULES
        for quant_type in fp8.BENCH_QUANT_TYPES
        for mtp in fp8.BENCH_MTP
    ] + [
        {
            "variant": "prefill_fp8", "q_len": q_len, "kv_len": kv_len,
            "quant_type": quant_type, "masked": masked, "kv_layout": layout,
        }
        for layout in ("nhd", "hnd")
        for masked in (False, True)
        for quant_type in (0, 1)
        for q_len, kv_len in prefill.CASES
    ]


def unsupported_reason(binding, dtype):
    expected = torch.bfloat16 if binding["variant"] == "decode_bf16" else torch.float8_e4m3fn
    if dtype != expected:
        return f"{binding['variant']} benchmark uses {expected}, not {dtype}"
    if _HPC is None or not callable(getattr(_HPC, _NATIVE_APIS[binding["variant"]], None)):
        return f"hpc CUDA API {_NATIVE_APIS[binding['variant']]} unavailable"
    if dtype == torch.float8_e4m3fn and not hasattr(_HPC, "QuantType"):
        return "hpc CUDA QuantType unavailable"
    return None


def build_inputs(binding, dtype, device):
    variant = binding["variant"]
    if variant == "decode_bf16":
        bench = _load_benchmark("test_hpc_ops_decode_bf16_perf")
        panel = bench.make_inputs(
            bench.OFFICIAL_CASES[binding["case"]], binding["mtp"],
            bench.BENCH_NUM_HEAD_KV, bench.BENCH_NUM_HEAD_Q, binding["layout"],
        )
        args = (panel.q, panel.k, panel.v, panel.block_ids, panel.kv_lens)
        kwargs = {
            "mtp": binding["mtp"] - 1, "new_kv_included": True,
            "splitk": True, "output": torch.empty_like(panel.q),
        }
        if binding["method"] == "dynamic":
            kwargs["task_map"] = bench._make_cuda_task_map(
                _HPC, panel.kv_lens, bench.BENCH_NUM_HEAD_KV,
                binding["mtp"], bench.BENCH_MIN_PROCESS_LEN,
            )
    elif variant == "decode_fp8":
        bench = _load_benchmark("test_hpc_ops_decode_fp8_perf")
        panel = bench.make_inputs(
            bench.OFFICIAL_CASES[binding["case"]], binding["mtp"],
            bench.BENCH_NUM_HEAD_KV, bench.BENCH_NUM_HEAD_Q,
            binding["layout"], binding["quant_type"],
        )
        args = (
            panel.q, panel.k_cache, panel.v_cache, panel.block_ids,
            panel.kv_lens, panel.q_scale, panel.k_scale, panel.v_scale,
        )
        kwargs = {
            "mtp": binding["mtp"] - 1, "new_kv_included": True,
            "quant_type": bench._cuda_quant_type(_HPC, binding["quant_type"]),
            "splitk": True,
            "output": torch.empty_like(panel.q, dtype=torch.bfloat16),
        }
        if binding["schedule"] == "dynamic":
            kwargs["task_map"] = bench._cuda_task_map(
                _HPC, panel, binding["mtp"], bench.BENCH_MIN_PROCESS_LEN,
            )
    else:
        bench = _load_benchmark("test_hpc_ops_prefill_fp8_perf")
        inputs = bench._make_inputs(
            binding["q_len"], binding["kv_len"], binding["quant_type"],
            binding["masked"], binding["kv_layout"],
        )
        args = (
            inputs.q, inputs.k_cache, inputs.v_cache, inputs.q_scale,
            inputs.k_scale, inputs.v_scale, inputs.cu_seqlens_q,
            inputs.block_ids, inputs.kv_lens, inputs.max_q_len,
        )
        quant_type = (
            _HPC.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD
            if binding["quant_type"] == 0
            else _HPC.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR
        )
        kwargs = {
            "quant_type": quant_type, "block_mask": inputs.block_mask,
            "output": torch.empty_like(inputs.q, dtype=torch.bfloat16),
        }
    return args, {"variant": variant, **kwargs}


def key_shape(binding):
    return [binding[key] for key in binding]


def config(binding, dtype):
    variant = binding["variant"]

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    if variant in ("decode_bf16", "decode_fp8"):
        bench = _load_benchmark(
            "test_hpc_ops_decode_bf16_perf" if variant == "decode_bf16"
            else "test_hpc_ops_decode_fp8_perf"
        )
        lengths = bench.OFFICIAL_CASES[binding["case"]]
        batch = len(lengths)
        max_blocks = max((length + bench.BLOCK_SIZE - 1) // bench.BLOCK_SIZE for length in lengths)
        total_blocks = sum((length + bench.BLOCK_SIZE - 1) // bench.BLOCK_SIZE for length in lengths)
        capacity = int(total_blocks * 1.2) + batch + 8
        q_shape = [batch * binding["mtp"], bench.BENCH_NUM_HEAD_Q, bench.HEAD_DIM]
        cache_shape = [capacity, bench.BLOCK_SIZE, bench.BENCH_NUM_HEAD_KV, bench.HEAD_DIM]
        out = tensor(q_shape, torch.bfloat16)
        inputs = {
            "q": tensor(q_shape),
            "block_ids": tensor([batch, max_blocks], torch.int32),
            "kv_lens": tensor([batch], torch.int32),
            "mtp": {"scalar": binding["mtp"] - 1},
            "new_kv_included": {"scalar": True},
            "splitk": {"scalar": True}, "output": out,
        }
        if variant == "decode_bf16":
            inputs.update(k=tensor(cache_shape), v=tensor(cache_shape))
            if binding["method"] == "dynamic":
                inputs["task_map"] = {
                    "source": "hpc.get_attention_decode_task_workspace",
                    "min_process_len": bench.BENCH_MIN_PROCESS_LEN,
                }
        else:
            inputs.update(
                k_cache=tensor(cache_shape), v_cache=tensor(cache_shape),
                q_scale=tensor([batch * binding["mtp"], bench.BENCH_NUM_HEAD_Q], torch.float32),
                quant_type={"scalar": binding["quant_type"]},
            )
            if binding["quant_type"] == "qkpertoken_perhead_vperhead":
                inputs["k_scale"] = tensor([capacity, 2, bench.BENCH_NUM_HEAD_KV, bench.HEAD_DIM])
                inputs["v_scale"] = tensor([bench.BENCH_NUM_HEAD_KV], torch.float32)
            else:
                inputs["k_scale"] = tensor([1], torch.float32)
                inputs["v_scale"] = tensor([1], torch.float32)
            if binding["schedule"] == "dynamic":
                inputs["task_map"] = {
                    "source": "hpc.get_attention_decode_task_workspace",
                    "min_process_len": bench.BENCH_MIN_PROCESS_LEN,
                }
    else:
        bench = _load_benchmark("test_hpc_ops_prefill_fp8_perf")
        q_len, kv_len = binding["q_len"], binding["kv_len"]
        requested_pages = math.ceil(kv_len / bench.PAGE_SIZE)
        pages = max(requested_pages * 2, requested_pages + 8)
        q_shape = [q_len, bench.Q_HEADS, bench.HEAD_DIM]
        cache_shape = [pages, bench.PAGE_SIZE, bench.KV_HEADS, bench.HEAD_DIM]
        out = tensor(q_shape, torch.bfloat16)
        inputs = {
            "q": tensor(q_shape),
            "k_cache": tensor(cache_shape), "v_cache": tensor(cache_shape),
            "q_scale": tensor(
                [1, bench.Q_HEADS, math.ceil(q_len / bench.BLOCK) * bench.BLOCK],
                torch.float32,
            ),
            "cu_seqlens_q": tensor([2], torch.int32),
            "block_ids": tensor([1, requested_pages], torch.int32),
            "kv_lens": tensor([1], torch.int32),
            "max_q_len": {"scalar": q_len},
            "quant_type": {"scalar": binding["quant_type"]},
            "block_mask": (
                tensor(
                    [1, bench.Q_HEADS, math.ceil(q_len / bench.BLOCK),
                     math.ceil(kv_len / bench.BLOCK)], torch.uint8,
                ) if binding["masked"] else {"scalar": None}
            ),
            "output": out,
        }
        if binding["quant_type"] == 0:
            inputs["k_scale"] = tensor([pages, 2, bench.KV_HEADS, bench.HEAD_DIM])
            inputs["v_scale"] = tensor([bench.KV_HEADS], torch.float32)
        else:
            inputs["k_scale"] = tensor([1], torch.float32)
            inputs["v_scale"] = tensor([1], torch.float32)
    inputs["variant"] = {"scalar": variant}
    return {
        "op": OP_NAME, "dtype": str(dtype), "dims": dict(binding),
        "inputs": inputs, "outputs": {"output": out},
        "mutation": "output", "graph": False,
    }
