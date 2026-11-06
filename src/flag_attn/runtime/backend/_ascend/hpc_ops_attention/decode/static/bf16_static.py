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

"""Ascend HY3 BF16 decode with a fixed rectangular producer schedule."""

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
)
from ..dynamic.bf16_dynamic import _validate


BLOCK_SIZE = 64
HEAD_DIM = 128


@dataclass
class StaticBF16Inputs:
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


StaticBF16Workspace = AscendBF16MTP1Workspace


def prepare_static_bf16_workspace(
    inputs: StaticBF16Inputs,
) -> StaticBF16Workspace:
    """Allocate a workspace whose producer always uses the rectangular grid."""
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_ascend_bf16_mtp1_workspace(inputs, static_sched=True)
    if inputs.mtp == 2:
        return prepare_ascend_bf16_mtp2_workspace(inputs, static_sched=True)
    if inputs.mtp == 3:
        return prepare_ascend_bf16_mtp3_workspace(inputs, static_sched=True)
    return prepare_ascend_bf16_mtp4_workspace(inputs, static_sched=True)


def attention_decode_bf16_tle(
    inputs: StaticBF16Inputs,
    workspace: StaticBF16Workspace,
) -> torch.Tensor:
    """Launch the fixed static workspace without rebuilding a task map."""
    _validate(inputs)
    if inputs.mtp == 1:
        return attention_decode_ascend_bf16_mtp1(inputs, workspace)
    if inputs.mtp == 2:
        return attention_decode_ascend_bf16_mtp2(inputs, workspace)
    if inputs.mtp == 3:
        return attention_decode_ascend_bf16_mtp3(inputs, workspace)
    return attention_decode_ascend_bf16_mtp4(inputs, workspace)


def bf16_static_workspace_is_reset(
    workspace: StaticBF16Workspace,
) -> bool:
    del workspace
    return True


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "StaticBF16Inputs",
    "StaticBF16Workspace",
    "attention_decode_bf16_tle",
    "bf16_static_workspace_is_reset",
    "prepare_static_bf16_workspace",
]
