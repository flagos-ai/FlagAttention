# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Triton bf16 dynamic decode."""

from .. import (
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp1_workspace,
    prepare_pure_triton_mtp_workspace,
)
from ..inputs import BLOCK_SIZE, HEAD_DIM, OFFICIAL_CASES, DynamicBF16Inputs
from ..policy import GenericDecodePolicy

QUANT_TYPE = "bf16"
SUPPORTED_MTP = (1, 2, 3)
DynamicBF16Policy = GenericDecodePolicy
DynamicBF16Workspace = PureTritonMTP1Workspace


def select_dynamic_bf16_policy(inputs):
    return GenericDecodePolicy(quant_type=QUANT_TYPE, mtp=inputs.mtp)


def prepare_dynamic_bf16_workspace(inputs):
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, QUANT_TYPE)
    return prepare_pure_triton_mtp_workspace(inputs, QUANT_TYPE)


def attention_decode_bf16_dynamic(inputs, workspace):
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    return attention_decode_pure_triton_mtp1(inputs, workspace)


def bf16_dynamic_workspace_is_reset(workspace):
    return isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace))


def refresh_dynamic_bf16_task_map(inputs, workspace):
    """The portable scheduler reads current sequence lengths at every launch."""
    return None


__all__ = [
    "BLOCK_SIZE",
    "DynamicBF16Inputs",
    "DynamicBF16Policy",
    "DynamicBF16Workspace",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "QUANT_TYPE",
    "SUPPORTED_MTP",
    "attention_decode_bf16_dynamic",
    "bf16_dynamic_workspace_is_reset",
    "prepare_dynamic_bf16_workspace",
    "refresh_dynamic_bf16_task_map",
    "select_dynamic_bf16_policy",
]
