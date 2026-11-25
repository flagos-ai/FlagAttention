# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend-owned HY3 decode workspace preparation."""

from functools import lru_cache
from importlib import import_module


_DECODE_WORKSPACE_PREPARERS = {
    "bf16_static": ("static.bf16_static", "prepare_static_bf16_workspace"),
    "bf16_dynamic": ("dynamic.bf16_dynamic", "prepare_dynamic_bf16_workspace"),
}


@lru_cache(maxsize=None)
def _workspace_preparer(variant):
    if variant not in _DECODE_WORKSPACE_PREPARERS:
        raise ValueError("Ascend HY3 decode only supports bf16_static and bf16_dynamic")
    module, prepare = _DECODE_WORKSPACE_PREPARERS[variant]
    implementation = import_module(f"{__package__}.decode.{module}")
    return getattr(implementation, prepare)


def hy3_attention_prepare(inputs, *, variant):
    return _workspace_preparer(variant)(inputs)
