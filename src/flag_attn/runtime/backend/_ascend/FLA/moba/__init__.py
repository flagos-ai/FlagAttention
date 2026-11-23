# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend MoBA implementation."""

from flag_attn.runtime.backend._ascend.FLA.moba.parallel import parallel_moba

__all__ = ["parallel_moba"]
