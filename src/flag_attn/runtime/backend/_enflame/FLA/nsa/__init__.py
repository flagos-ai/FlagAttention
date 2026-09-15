# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 backend for parallel NSA."""

from .parallel_nsa import parallel_nsa
from .parallel_nsa_compression import (
    parallel_nsa_compression,
)

__all__ = [
    "parallel_nsa",
    "parallel_nsa_compression",
]
