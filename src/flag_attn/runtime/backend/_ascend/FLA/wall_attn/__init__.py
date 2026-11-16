# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for Wall attention (forward)."""

from .api import parallel_wall_attn

__all__ = ["parallel_wall_attn"]
