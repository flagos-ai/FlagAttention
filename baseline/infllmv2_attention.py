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

"""Official CUDA packed-varlen baseline for InfLLM-V2 forward stages."""

import torch


OP_NAME = "infllmv2_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = None

# Matches PREFILL_CASES and DECODE_CASES in the InfLLM-V2 benchmark.
CASES = (
    ("dense-1x1024-d64", 1, 1024, 1024, 64, True),
    ("full-1x8192-d64", 1, 8192, 8192, 64, False),
    ("full-1x8192-d128", 1, 8192, 8192, 128, False),
    ("full-1x32768-d64", 1, 32768, 32768, 64, False),
    ("full-1x32768-d128", 1, 32768, 32768, 128, False),
    ("full-1x65536-d64", 1, 65536, 65536, 64, False),
    ("full-1x65536-d128", 1, 65536, 65536, 128, False),
    ("chunk-1x128x16384-d64", 1, 128, 16384, 64, False),
    ("chunk-1x128x65536-d128", 1, 128, 65536, 128, False),
    ("decode-1x8192-d64", 1, 1, 8192, 64, False),
    ("decode-1x8192-d128", 1, 1, 8192, 128, False),
    ("decode-1x65536-d64", 1, 1, 65536, 64, False),
    ("decode-1x65536-d128", 1, 1, 65536, 128, False),
    ("decode-8x32768-d64", 8, 1, 32768, 64, False),
    ("decode-8x32768-d128", 8, 1, 32768, 128, False),
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        from infllm_v2 import infllmv2_attn_varlen_func
    except (ImportError, OSError) as exc:
        NATIVE_UNAVAILABLE_REASON = repr(exc)
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return infllmv2_attn_varlen_func


def grid():
    return [
        {
            "case": name,
            "batch": batch,
            "q_length": q_length,
            "kv_length": kv_length,
            "head_dim": head_dim,
            "dense": dense,
        }
        for name, batch, q_length, kv_length, head_dim, dense in CASES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    q_length = binding["q_length"]
    kv_length = binding["kv_length"]
    head_dim = binding["head_dim"]
    generator = torch.Generator(device=device).manual_seed(31)
    q = torch.randn(
        (batch * q_length, 32, head_dim), generator=generator,
        device=device, dtype=dtype,
    )
    k = torch.randn(
        (batch * kv_length, 2, head_dim), generator=generator,
        device=device, dtype=dtype,
    )
    v = torch.randn(
        (batch * kv_length, 2, head_dim), generator=generator,
        device=device, dtype=dtype,
    )
    cu_q = torch.arange(batch + 1, device=device, dtype=torch.int32) * q_length
    cu_k = torch.arange(batch + 1, device=device, dtype=torch.int32) * kv_length

    topk_idx = None
    if not binding["dense"]:
        from flag_attn.infllmv2.operator import InfLLMV2Config
        from flag_attn.infllmv2.forward import (
            compress_k, pool_scores, select_blocks, stage1,
        )

        config = InfLLMV2Config(topk=64, dense_len=8192)
        k1, cu_k1 = compress_k(
            k, cu_k, config.k1_kernel_size, config.k1_stride
        )
        k2, cu_k2 = compress_k(
            k, cu_k, config.k2_kernel_size, config.k2_stride
        )
        scores = stage1(
            q, k1, k2, cu_q, cu_k1, cu_k2, q_length,
            config.k1_stride, config.k2_stride, None, config.causal, cu_k,
        )
        block_scores = pool_scores(
            scores, cu_q, cu_k1, q_length,
            config.k1_kernel_size, config.k1_stride, config.block_size,
            config.init_blocks, config.local_blocks, cu_k, kv_length,
            _valid_only=True,
        )
        topk_idx = select_blocks(
            block_scores, cu_q, config.topk, config.block_size, cu_k
        )

    return (q, k, v, cu_q, cu_k, q_length, kv_length), {
        "causal": True,
        "topk_idx": topk_idx,
    }


def key_shape(binding):
    return [
        binding["batch"], binding["q_length"],
        binding["kv_length"], binding["head_dim"],
    ]


def config(binding, dtype):
    from flag_attn.infllmv2.operator import InfLLMV2Config

    batch = binding["batch"]
    q_length = binding["q_length"]
    kv_length = binding["kv_length"]
    head_dim = binding["head_dim"]
    op_config = InfLLMV2Config(
        topk=64, dense_len=kv_length + 1 if binding["dense"] else 8192
    )

    def tensor(shape, tensor_dtype=dtype):
        return {"shape": shape, "dtype": str(tensor_dtype)}

    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "phase": "forward",
        "stage": "decode" if binding["case"].startswith("decode-") else "prefill",
        "dims": {
            key: binding[key]
            for key in ("case", "batch", "q_length", "kv_length", "head_dim")
        },
        "inputs": {
            "q": tensor([batch * q_length, 32, head_dim]),
            "k": tensor([batch * kv_length, 2, head_dim]),
            "v": tensor([batch * kv_length, 2, head_dim]),
            "cu_seqlens_q": tensor([batch + 1], torch.int32),
            "cu_seqlens_k": tensor([batch + 1], torch.int32),
            "topk_idx": (
                None if binding["dense"]
                else tensor([2, batch * q_length, op_config.topk], torch.int32)
            ),
            "max_seqlen_q": {"scalar": q_length},
            "max_seqlen_k": {"scalar": kv_length},
            "causal": {"scalar": True},
        },
        "outputs": {"output": tensor([batch * q_length, 32, head_dim])},
        "input_seed": 31,
        "selector": {
            "k1_kernel_size": op_config.k1_kernel_size,
            "k1_stride": op_config.k1_stride,
            "k2_kernel_size": op_config.k2_kernel_size,
            "k2_stride": op_config.k2_stride,
            "block_size": op_config.block_size,
            "topk": op_config.topk,
            "init_blocks": op_config.init_blocks,
            "local_blocks": op_config.local_blocks,
            "dense_len": op_config.dense_len,
        },
        "measurement_scope": "official_varlen_attention",
        "mutation": "none",
        "graph": False,
    }
