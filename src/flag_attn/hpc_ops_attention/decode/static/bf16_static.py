# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Triton bf16 static decode."""

from .. import (
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp1_workspace,
    prepare_pure_triton_mtp_workspace,
)
from ..inputs import BLOCK_SIZE, HEAD_DIM, OFFICIAL_CASES, StaticBF16Inputs
from ..policy import GenericDecodePolicy

QUANT_TYPE = "bf16"
SUPPORTED_MTP = (1, 2, 3)
StaticBF16Policy = GenericDecodePolicy
StaticBF16Workspace = PureTritonMTP1Workspace


def select_static_bf16_policy(inputs):
    return GenericDecodePolicy(quant_type=QUANT_TYPE, mtp=inputs.mtp)


def prepare_static_bf16_workspace(inputs):
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, QUANT_TYPE)
    return prepare_pure_triton_mtp_workspace(inputs, QUANT_TYPE)


def attention_decode_bf16_static(inputs, workspace):
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    return attention_decode_pure_triton_mtp1(inputs, workspace)


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "QUANT_TYPE",
    "SUPPORTED_MTP",
    "StaticBF16Inputs",
    "StaticBF16Policy",
    "StaticBF16Workspace",
    "attention_decode_bf16_static",
    "prepare_static_bf16_workspace",
    "select_static_bf16_policy",
]
