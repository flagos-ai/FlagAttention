# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend-owned HY3 preparation and execution routing."""

from functools import lru_cache
from importlib import import_module


_DECODE_IMPLEMENTATIONS = {
    "bf16_static": ("static.bf16_static", "prepare_static_bf16_workspace", "attention_decode_bf16_tle"),
    "bf16_dynamic": ("dynamic.bf16_dynamic", "prepare_dynamic_bf16_workspace", "attention_decode_bf16_dynamic"),
    "fp8_qk_static": ("static.fp8_qkpertoken_perhead_vperhead_static", "prepare_decode_workspace", "attention_decode_fp8"),
    "fp8_qk_dynamic": ("dynamic.fp8_qkpertoken_perhead_vperhead_dynamic", "prepare_decode_workspace", "attention_decode_fp8"),
    "fp8_kv_static": ("static.fp8_qpertoken_perhead_kvpertensor_static", "prepare_decode_workspace", "attention_decode_fp8"),
    "fp8_kv_dynamic": ("dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic", "prepare_decode_workspace", "attention_decode_fp8"),
}


@lru_cache(maxsize=None)
def _implementation(variant):
    module, prepare, compute = _DECODE_IMPLEMENTATIONS[variant]
    implementation = import_module(f"{__package__}.decode.{module}")
    return implementation, getattr(implementation, prepare), getattr(implementation, compute)


def _inputs(inputs, variant):
    if not variant.startswith("fp8_"):
        return inputs
    from .decode.fp8 import FP8DecodeInputs

    if isinstance(inputs, FP8DecodeInputs):
        return inputs
    return FP8DecodeInputs(
        inputs.q, inputs.k_cache, inputs.v_cache, inputs.block_ids,
        inputs.kv_lens, inputs.q_scale, inputs.k_scale, inputs.v_scale,
    )


def prepare_workspace(inputs, *, variant):
    _, prepare, _ = _implementation(variant)
    return prepare(_inputs(inputs, variant))


def _attention_decode(inputs, workspace, *, variant):
    _, _, compute = _implementation(variant)
    return compute(_inputs(inputs, variant), workspace)


def attention_fp8_qk_static(inputs, workspace):
    return _attention_decode(inputs, workspace, variant="fp8_qk_static")


def attention_fp8_qk_dynamic(inputs, workspace):
    return _attention_decode(inputs, workspace, variant="fp8_qk_dynamic")


def attention_fp8_kv_static(inputs, workspace):
    return _attention_decode(inputs, workspace, variant="fp8_kv_static")


def attention_fp8_kv_dynamic(inputs, workspace):
    return _attention_decode(inputs, workspace, variant="fp8_kv_dynamic")
