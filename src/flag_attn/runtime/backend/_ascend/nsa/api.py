# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compatible, lazy public inference adapter for Ascend NSA compression."""
import torch


def parallel_nsa_compression(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Return output/LSE; training through this public override is unsupported.

    Existing internal autograd implementation is preserved. Inputs requiring
    gradients are accepted only inside an explicit no-grad/inference context.
    """
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise RuntimeError("Ascend NSA compression public override is forward-only; use no_grad/inference_mode")
    from .compression import parallel_nsa_compression as implementation

    return implementation(q, k, v, block_size=block_size, scale=scale, cu_seqlens=cu_seqlens)
