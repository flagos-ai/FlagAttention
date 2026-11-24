# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""MThreads implementation of Native Sparse Attention."""

from .parallel import parallel_nsa
from .compression import parallel_nsa_compression

__all__ = ["parallel_nsa", "parallel_nsa_compression"]
