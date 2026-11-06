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

"""Ascend HY3 static FP8 decode: Q per-token/head, KV per-tensor."""

from ..fp8 import (
    BLOCK_SIZE,
    HEAD_DIM,
    SUPPORTED_MTP,
    FP8DecodeInputs,
    FP8DecodeWorkspace,
    QPERTOKEN_KVPERTENSOR,
    attention_decode_fp8 as _attention_decode_fp8,
    prepare_fp8_workspace,
    validate_fp8_inputs,
)

QUANT_TYPE = "qpertoken_perhead_kvpertensor"
QUANT_TYPE_ID = QPERTOKEN_KVPERTENSOR


def prepare_decode_workspace(inputs: FP8DecodeInputs) -> FP8DecodeWorkspace:
    return prepare_fp8_workspace(inputs, QUANT_TYPE_ID, static_sched=True)


def attention_decode_fp8(inputs: FP8DecodeInputs, workspace: FP8DecodeWorkspace):
    return _attention_decode_fp8(inputs, workspace, QUANT_TYPE_ID)


def select_decode_policy(inputs: FP8DecodeInputs) -> dict[str, object]:
    validate_fp8_inputs(inputs, QUANT_TYPE_ID)
    return {"schedule": "static", "mtp": inputs.mtp, "layout": inputs.layout}


def workspace_is_reset(workspace: FP8DecodeWorkspace) -> bool:
    del workspace
    return True


__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "SUPPORTED_MTP", "QUANT_TYPE", "QUANT_TYPE_ID",
    "FP8DecodeInputs", "FP8DecodeWorkspace", "attention_decode_fp8",
    "prepare_decode_workspace", "select_decode_policy", "workspace_is_reset",
]
