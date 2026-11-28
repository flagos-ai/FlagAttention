# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Triton qpertoken_perhead_kvpertensor dynamic decode."""

from .. import (
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp1_workspace,
    prepare_pure_triton_mtp_workspace,
)
from ..inputs import BLOCK_SIZE, HEAD_DIM, OFFICIAL_CASES, FP8DecodeInputs
from ..policy import GenericDecodePolicy

QUANT_TYPE = "qpertoken_perhead_kvpertensor"
SUPPORTED_MTP = (1, 2, 4)
QUANT_TYPE_ID = 1
QUANT_TYPES = (QUANT_TYPE,)
FP8DecodePolicy = GenericDecodePolicy
FP8DecodeWorkspace = PureTritonMTP1Workspace


def select_decode_policy(inputs):
    return GenericDecodePolicy(quant_type=QUANT_TYPE, mtp=inputs.mtp)


def prepare_decode_workspace(inputs):
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, QUANT_TYPE)
    return prepare_pure_triton_mtp_workspace(inputs, QUANT_TYPE)


def attention_decode_fp8(inputs, workspace):
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    return attention_decode_pure_triton_mtp1(inputs, workspace)


def workspace_is_reset(workspace):
    return isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace))


__all__ = [
    "BLOCK_SIZE",
    "FP8DecodeInputs",
    "FP8DecodePolicy",
    "FP8DecodeWorkspace",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "QUANT_TYPE",
    "QUANT_TYPES",
    "QUANT_TYPE_ID",
    "SUPPORTED_MTP",
    "attention_decode_fp8",
    "prepare_decode_workspace",
    "select_decode_policy",
    "workspace_is_reset",
]
