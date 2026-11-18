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

"""vLLM forward baseline for MiniMax M3 sparse attention prefill and decode."""

import ast
import inspect
import textwrap

import torch


OP_NAME = "minimax_m3_sparse_attn"
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
DTYPES = [torch.bfloat16] + ([FP8_DTYPE] if FP8_DTYPE is not None else [])
IS_INPLACE = True
MUTATED_INPUTS = ("output",)

BLOCK = 128
HEAD_DIM = 128
KV_SCALE = 0.5
TOPK = 16
INIT_BLOCKS = 1
LOCAL_BLOCKS = 2

PREFILL_SHAPES = (
    (1, 128, 4096, 1, 8), (1, 512, 4096, 1, 8),
    (1, 1024, 4096, 1, 8), (1, 4096, 4096, 1, 8),
    (1, 128, 16384, 1, 8), (1, 512, 16384, 1, 8),
    (1, 1024, 16384, 1, 8), (1, 4096, 16384, 1, 8),
    (1, 8192, 16384, 1, 8), (1, 128, 65536, 1, 8),
    (1, 1024, 65536, 1, 8), (4, 128, 4096, 1, 8),
    (4, 512, 4096, 1, 8), (4, 1024, 4096, 1, 8),
    (8, 128, 4096, 1, 8), (8, 512, 4096, 1, 8),
    (8, 1024, 4096, 1, 8), (16, 128, 4096, 1, 8),
    (16, 512, 4096, 1, 8), (32, 128, 4096, 1, 8),
    (64, 128, 4096, 1, 8),
)
DECODE_SHAPES = (
    (1, 1, 4096, 1, 8), (1, 1, 16384, 1, 8),
    (1, 1, 65536, 1, 8), (1, 1, 131072, 1, 8),
    (4, 1, 4096, 1, 8), (4, 1, 16384, 1, 8),
    (4, 1, 65536, 1, 8), (16, 1, 4096, 1, 8),
    (16, 1, 16384, 1, 8), (32, 1, 4096, 1, 8),
    (32, 1, 16384, 1, 8), (64, 1, 4096, 1, 8),
    (64, 1, 16384, 1, 8), (128, 1, 4096, 1, 8),
    (128, 1, 8192, 1, 8), (256, 1, 4096, 1, 8),
    (512, 1, 3072, 1, 8), (1024, 1, 3072, 1, 8),
)

_VLLM_LAYOUT = None
_FP8_SUPPORTED = False


def _kv_layout(fn):
    source = inspect.getsource(inspect.unwrap(fn))
    tree = ast.parse(textwrap.dedent(source))
    indices = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (isinstance(node.func.value, ast.Name)
                and node.func.value.id == "kv_cache"
                and node.func.attr == "stride"):
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                raise RuntimeError("ambiguous vLLM KV cache layout")
            indices.add(node.args[0].value)
    layout = {frozenset(range(4)): "packed4d",
              frozenset(range(5)): "split5d"}.get(frozenset(indices))
    if layout is None:
        raise RuntimeError("unknown vLLM KV cache layout")
    return layout


def native():
    global _VLLM_LAYOUT, _FP8_SUPPORTED
    try:
        from vllm.models.minimax_m3.common.ops.index_topk import (
            minimax_m3_index_decode, minimax_m3_index_score,
            minimax_m3_index_topk,
        )
        from vllm.models.minimax_m3.common.ops.sparse_attn import (
            minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode,
        )
        prefill_layout = _kv_layout(minimax_m3_sparse_attn)
        decode_layout = _kv_layout(minimax_m3_sparse_attn_decode)
        if prefill_layout != decode_layout:
            return None
        _VLLM_LAYOUT = prefill_layout
        prefill_params = inspect.signature(minimax_m3_sparse_attn).parameters
        decode_params = inspect.signature(minimax_m3_sparse_attn_decode).parameters
        _FP8_SUPPORTED = all(
            name in params
            for params in (prefill_params, decode_params)
            for name in ("k_scale", "v_scale")
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError, SyntaxError):
        return None

    def vllm_msa(*, phase, q, idx_q, kv_cache, index_kv_cache,
                 block_table, cu_q, seq_lens, prefix_lens, sm_scale,
                 k_scale, v_scale, output, seq_len, kv_heads):
        if phase == "prefill":
            query_len = q.shape[0] // seq_lens.numel()
            scores = minimax_m3_index_score(
                idx_q, index_kv_cache, block_table, cu_q, seq_lens,
                prefix_lens, query_len, seq_len, kv_heads,
            )
            topk_idx = minimax_m3_index_topk(
                scores, cu_q, prefix_lens, query_len,
                TOPK, INIT_BLOCKS, LOCAL_BLOCKS,
            )
            kwargs = {
                "q": q, "kv_cache": kv_cache, "topk_idx": topk_idx,
                "block_table": block_table, "sm_scale": sm_scale,
                "output": output, "cu_seqlens_q": cu_q, "seq_lens": seq_lens,
                "prefix_lens": prefix_lens, "max_query_len": query_len,
                "num_kv_heads": kv_heads,
            }
            fn = minimax_m3_sparse_attn
        else:
            decode_qlen = q.shape[0] // seq_lens.numel()
            topk_idx = minimax_m3_index_decode(
                idx_q, index_kv_cache, block_table, seq_lens, seq_len,
                TOPK, INIT_BLOCKS, LOCAL_BLOCKS, kv_heads,
                decode_qlen, decode_qlen,
            )
            kwargs = {
                "q": q, "kv_cache": kv_cache, "topk_idx": topk_idx,
                "block_table": block_table, "sm_scale": sm_scale,
                "output": output, "seq_lens": seq_lens,
                "num_kv_heads": kv_heads, "decode_query_len": decode_qlen,
            }
            fn = minimax_m3_sparse_attn_decode
        if k_scale is not None:
            kwargs.update(k_scale=k_scale, v_scale=v_scale)
        return fn(**kwargs)

    return vllm_msa


def grid():
    keys = ("B", "Q", "S", "KVH", "H")
    return [
        {**dict(zip(keys, shape)), "phase": phase, "shape_index": index}
        for phase, shapes in (("prefill", PREFILL_SHAPES), ("decode", DECODE_SHAPES))
        for index, shape in enumerate(shapes)
    ]


def unsupported_reason(binding, dtype):
    if dtype == FP8_DTYPE and not _FP8_SUPPORTED:
        return "vLLM sparse attention lacks k_scale/v_scale for FP8"
    return None


def _random_storage(shape, device, fp8, scale, generator):
    value = torch.randn(
        shape, device=device, dtype=torch.bfloat16, generator=generator,
    ) * 0.5
    return (value / scale).to(FP8_DTYPE) if fp8 else value


def build_inputs(binding, dtype, device):
    batch, query_len, seq_len, kv_heads, heads = (
        binding[key] for key in ("B", "Q", "S", "KVH", "H")
    )
    phase = binding["phase"]
    fp8 = dtype == FP8_DTYPE
    generator = torch.Generator(device=device).manual_seed(
        binding["shape_index"] + (100_000 if phase == "decode" else 0)
    )
    blocks_per_request = (seq_len + BLOCK - 1) // BLOCK
    total_blocks = batch * blocks_per_request
    total_q = batch * query_len
    q = torch.randn(
        (total_q, heads, HEAD_DIM), device=device,
        dtype=torch.bfloat16, generator=generator,
    ) * 0.5
    idx_q = _random_storage((total_q, kv_heads, HEAD_DIM), device, fp8, 1.0, generator)
    k_cont = _random_storage(
        (total_blocks * BLOCK, kv_heads, HEAD_DIM),
        device, fp8, KV_SCALE, generator,
    )
    v_cont = _random_storage(
        (total_blocks * BLOCK, kv_heads, HEAD_DIM),
        device, fp8, KV_SCALE, generator,
    )
    index_k_cont = _random_storage(
        (total_blocks * BLOCK, HEAD_DIM), device, fp8, 1.0, generator,
    )
    storage_dtype = FP8_DTYPE if fp8 else torch.bfloat16
    kv_cache = torch.empty(
        (total_blocks, kv_heads, BLOCK, 2 * HEAD_DIM),
        device=device, dtype=storage_dtype,
    )
    kv_cache[..., :HEAD_DIM] = k_cont.reshape(
        total_blocks, BLOCK, kv_heads, HEAD_DIM,
    ).permute(0, 2, 1, 3)
    kv_cache[..., HEAD_DIM:] = v_cont.reshape(
        total_blocks, BLOCK, kv_heads, HEAD_DIM,
    ).permute(0, 2, 1, 3)
    index_kv_cache = index_k_cont.reshape(total_blocks, BLOCK, HEAD_DIM)
    physical_pages = torch.randperm(total_blocks, device=device, generator=generator)
    if fp8:
        physical_pages = torch.arange(total_blocks, device=device)
    block_table = physical_pages.reshape(batch, blocks_per_request).to(torch.int32)
    if not fp8:
        order = physical_pages.argsort()
        kv_cache = kv_cache[order].contiguous()
        index_kv_cache = index_kv_cache[order].contiguous()
    if _VLLM_LAYOUT == "split5d":
        kv_cache = kv_cache.reshape(
            total_blocks, kv_heads, BLOCK, 2, HEAD_DIM,
        ).permute(0, 3, 2, 1, 4).contiguous()
    cu_q = torch.arange(
        0, (batch + 1) * query_len, query_len,
        device=device, dtype=torch.int32,
    )
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    prefix_lens = torch.full_like(seq_lens, seq_len - query_len)
    k_scale = torch.tensor([KV_SCALE], device=device, dtype=torch.float32) if fp8 else None
    v_scale = torch.tensor([KV_SCALE], device=device, dtype=torch.float32) if fp8 else None
    output = torch.empty_like(q)
    return (), {
        "phase": phase, "q": q, "idx_q": idx_q, "kv_cache": kv_cache,
        "index_kv_cache": index_kv_cache, "block_table": block_table,
        "cu_q": cu_q, "seq_lens": seq_lens, "prefix_lens": prefix_lens,
        "sm_scale": HEAD_DIM**-0.5, "k_scale": k_scale, "v_scale": v_scale,
        "output": output, "seq_len": seq_len, "kv_heads": kv_heads,
    }


def key_shape(binding):
    return [binding[key] for key in ("B", "Q", "S", "KVH", "H")]


def config(binding, dtype):
    batch, query_len, seq_len, kv_heads, heads = (
        binding[key] for key in ("B", "Q", "S", "KVH", "H")
    )
    blocks_per_request = (seq_len + BLOCK - 1) // BLOCK
    total_blocks = batch * blocks_per_request
    total_q = batch * query_len
    storage_dtype = dtype

    def tensor(shape, tensor_dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    cache_shape = (
        [total_blocks, kv_heads, BLOCK, 2 * HEAD_DIM]
        if _VLLM_LAYOUT == "packed4d"
        else [total_blocks, 2, BLOCK, kv_heads, HEAD_DIM]
    )
    return {
        "op": OP_NAME, "dtype": str(dtype), "phase": binding["phase"],
        "dims": {key: binding[key] for key in ("B", "Q", "S", "KVH", "H", "phase")},
        "inputs": {
            "q": tensor([total_q, heads, HEAD_DIM], torch.bfloat16),
            "idx_q": tensor([total_q, kv_heads, HEAD_DIM], storage_dtype),
            "kv_cache": tensor(cache_shape, storage_dtype),
            "index_kv_cache": tensor([total_blocks, BLOCK, HEAD_DIM], storage_dtype),
            "block_table": tensor([batch, blocks_per_request], torch.int32),
            "cu_q": tensor([batch + 1], torch.int32),
            "seq_lens": tensor([batch], torch.int32),
            "prefix_lens": tensor([batch], torch.int32),
            "k_scale": tensor([1], torch.float32) if dtype == FP8_DTYPE else {"scalar": None},
            "v_scale": tensor([1], torch.float32) if dtype == FP8_DTYPE else {"scalar": None},
            "sm_scale": {"scalar": HEAD_DIM**-0.5},
            "seq_len": {"scalar": seq_len},
            "kv_heads": {"scalar": kv_heads},
            "topk": {"scalar": TOPK},
            "init_blocks": {"scalar": INIT_BLOCKS},
            "local_blocks": {"scalar": LOCAL_BLOCKS},
            "output": tensor([total_q, heads, HEAD_DIM], torch.bfloat16),
        },
        "outputs": {"output": tensor([total_q, heads, HEAD_DIM], torch.bfloat16)},
        "mutation": "output", "graph": False,
    }
