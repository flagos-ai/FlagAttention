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

"""Ascend correctness implementation of HY3 BF16 dynamic decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import (
    AscendBF16MTP1Workspace,
    attention_decode_ascend_bf16_mtp1,
    attention_decode_ascend_bf16_mtp2,
    attention_decode_ascend_bf16_mtp3,
    attention_decode_ascend_bf16_mtp4,
    prepare_ascend_bf16_mtp1_workspace,
    prepare_ascend_bf16_mtp2_workspace,
    prepare_ascend_bf16_mtp3_workspace,
    prepare_ascend_bf16_mtp4_workspace,
    refresh_ascend_bf16_mtp1_task_map,
)


BLOCK_SIZE = 64
HEAD_DIM = 128


@dataclass
class DynamicBF16Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    layout: str

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


DynamicBF16Workspace = AscendBF16MTP1Workspace


def _validate(inputs: DynamicBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16 or inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError("q must be BF16 [batch * MTP, Hq, 128]")
    if inputs.q.stride(2) != 1:
        raise ValueError("q head dimension must be contiguous")
    if inputs.mtp not in (1, 2, 3, 4):
        raise NotImplementedError("Ascend BF16 dynamic decode supports MTP=1, MTP=2, MTP=3, or MTP=4")

    tensors = (
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
    )
    if inputs.q.device.type != "npu":
        raise ValueError("Ascend BF16 dynamic decode requires NPU tensors")
    if any(tensor.device != inputs.q.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same NPU device")

    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE or cache.shape[3] != HEAD_DIM:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32 or inputs.kv_lens.ndim != 1:
        raise ValueError("kv_lens must be rank-1 int32")

    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("BF16 dynamic decode requires official GQA8 heads")
    if inputs.v_cache.shape[2] != hkv:
        raise ValueError("K and V caches must have the same number of heads")
    if hkv > 1:
        inferred_k = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
        inferred_v = "HND" if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else "NHD"
        if inferred_k != inputs.layout or inferred_v != inputs.layout:
            raise ValueError("explicit layout does not match cache strides")
    return inputs.mtp, hq, hkv


def prepare_dynamic_bf16_workspace(
    inputs: DynamicBF16Inputs,
) -> DynamicBF16Workspace:
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_ascend_bf16_mtp1_workspace(inputs)
    if inputs.mtp == 2:
        return prepare_ascend_bf16_mtp2_workspace(inputs)
    if inputs.mtp == 3:
        return prepare_ascend_bf16_mtp3_workspace(inputs)
    return prepare_ascend_bf16_mtp4_workspace(inputs)


def attention_decode_bf16_dynamic(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
) -> torch.Tensor:
    _validate(inputs)
    if inputs.mtp == 1:
        return attention_decode_ascend_bf16_mtp1(inputs, workspace)
    if inputs.mtp == 2:
        return attention_decode_ascend_bf16_mtp2(inputs, workspace)
    if inputs.mtp == 3:
        return attention_decode_ascend_bf16_mtp3(inputs, workspace)
    return attention_decode_ascend_bf16_mtp4(inputs, workspace)


def refresh_dynamic_bf16_task_map(
    inputs: DynamicBF16Inputs,
    workspace: DynamicBF16Workspace,
) -> None:
    _validate(inputs)
    refresh_ascend_bf16_mtp1_task_map(inputs, workspace)


def bf16_dynamic_workspace_is_reset(
    workspace: DynamicBF16Workspace,
) -> bool:
    del workspace
    return True


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "DynamicBF16Inputs",
    "DynamicBF16Workspace",
    "attention_decode_bf16_dynamic",
    "bf16_dynamic_workspace_is_reset",
    "prepare_dynamic_bf16_workspace",
    "refresh_dynamic_bf16_task_map",
]
