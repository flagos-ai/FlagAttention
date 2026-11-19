"""MoBA attention implementations."""

from flag_attn.runtime.backend._hygon.FLA.moba.parallel import (
    parallel_moba,
)

__all__ = [
    "parallel_moba",
]
