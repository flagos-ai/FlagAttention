"""Parameterized local linear attention implemented with Triton kernels."""

from .parallel import (
    ParallaxFunction,
    parallel_parallax,
    parallel_parallax_bwd,
    parallel_parallax_fwd,
)

__all__ = [
    "ParallaxFunction",
    "parallel_parallax",
    "parallel_parallax_bwd",
    "parallel_parallax_fwd",
]
