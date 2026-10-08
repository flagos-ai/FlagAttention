# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""MThreads gated linear attention implemented with Triton/TLE."""

from .chunk_gla import chunk_gla

__all__ = ["chunk_gla"]
