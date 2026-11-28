# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Optional NVIDIA kernel capabilities, kept outside portable operators."""

from functools import lru_cache
import os


def force_generic():
    return os.getenv("FLAG_ATTN_FORCE_GENERIC", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def hopper_prefill():
    try:
        import triton.experimental.tle.language as tle
        from triton.tools.tensor_descriptor import TensorDescriptor  # noqa: F401
    except (ImportError, AttributeError):
        return None
    if not all(
        hasattr(getattr(tle, "gpu", None), name)
        for name in (
            "alloc",
            "local_ptr",
            "alloc_barriers",
            "barrier_wait",
            "barrier_arrive",
            "copy",
            "wgmma",
            "wgmma_wait",
            "smem",
            "READY",
        )
    ):
        return None
    from ..hopper.minimax_sparse_attention import sparse_attn

    return sparse_attn
