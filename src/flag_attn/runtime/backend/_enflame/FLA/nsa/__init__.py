# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 backend for parallel NSA."""

from typing import Literal

from .parallel import parallel_nsa as _parallel_nsa_full
from .compression import (
    parallel_nsa_compression,
)


def parallel_nsa(*args, mode: Literal["full", "compression"] = "full", **kwargs):
    """Run full/selected NSA or its compression-only component.

    Full mode (the default) forwards to parallel_nsa.parallel_nsa; its
    existing block and gate arguments also select sparse-only attention.
    Compression mode forwards to parallel_nsa_compression with that
    function's original arguments and return value.
    """
    if mode == "full":
        return _parallel_nsa_full(*args, **kwargs)
    if mode == "compression":
        return parallel_nsa_compression(*args, **kwargs)
    raise ValueError(f"Unsupported NSA mode: {mode!r}; expected 'full' or 'compression'")


__all__ = [
    "parallel_nsa",
    "parallel_nsa_compression",
]
