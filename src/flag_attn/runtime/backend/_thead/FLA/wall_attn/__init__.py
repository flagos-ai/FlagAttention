# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wall-Attention prefill/training, KV-cache decode, and correctness reference."""

from .parallel import get_last_route, parallel_wall_attn, select_route
from .decode import build_wall_kv_cache, parallel_wall_attn_decode
from .naive import naive_wall_attn

__all__ = [
    "build_wall_kv_cache",
    "naive_wall_attn",
    "parallel_wall_attn",
    "parallel_wall_attn_decode",
    "select_route",
    "get_last_route",
]
