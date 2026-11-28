# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wall Attention prefill/training, KV-cache decode, and reference APIs."""

from flag_attn.runtime.backend import resolve_operator

from .decode import build_wall_kv_cache, parallel_wall_attn_decode
from .naive import naive_wall_attn

_DISPATCH = {
    "parallel_wall_attn": "parallel_wall_attn",
    "select_route": "wall_attn_select_route",
    "get_last_route": "wall_attn_get_last_route",
}


def __getattr__(name):
    if name not in _DISPATCH:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return resolve_operator(_DISPATCH[name], f"{__name__}.parallel", name)


__all__ = [
    "build_wall_kv_cache",
    "get_last_route",
    "naive_wall_attn",
    "parallel_wall_attn",
    "parallel_wall_attn_decode",
    "select_route",
]
