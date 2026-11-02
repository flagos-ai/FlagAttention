# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""T-Head Wall Attention prefill and training."""

from .parallel import parallel_wall_attn

__all__ = ["parallel_wall_attn"]
