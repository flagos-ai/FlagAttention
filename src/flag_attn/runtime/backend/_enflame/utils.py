# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Shared sparse-attention helpers for non-FLA Enflame operators."""

from flag_attn.minimax_sparse_attention.utils import (
    current_platform,
    has_triton_tle,
    round_up,
)

SPARSE_BLOCK_SIZE = 128

__all__ = [
    "SPARSE_BLOCK_SIZE",
    "current_platform",
    "has_triton_tle",
    "round_up",
]
