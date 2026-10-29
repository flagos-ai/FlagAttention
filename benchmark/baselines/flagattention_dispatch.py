# Derived from FlagAttention 54417c6; original code is MIT licensed.
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""Pinned original dispatch predicate, used only to measure the legacy policy."""

import os
import torch
from flag_attn.FLA.gated_delta_rule.chunk_fused_forward import HAS_TWO_KERNEL_TLE, TWO_KERNEL_TLE_ENV

BASELINE_COMMIT = "54417c6e9e4f85cdca248981ed18dc5b9fa5eb5d"


def can_use_two_kernel_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    enabled = os.environ.get(TWO_KERNEL_TLE_ENV, "1").lower()
    if not (HAS_TWO_KERNEL_TLE and enabled not in {"0", "false", "off", "no"}):
        return False
    if initial_state is not None or not output_final_state or cu_seqlens is not None:
        return False
    if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not all(x.device == q.device and x.dtype == q.dtype for x in (k, v, beta, g)):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if beta.ndim != 3 or g.ndim != 3:
        return False

    B, T, Hg, K = q.shape
    H, V = v.shape[2:]
    return (
        k.shape == (B, T, Hg, K)
        and v.shape[:2] == (B, T)
        and beta.shape == (B, T, H)
        and g.shape == (B, T, H)
        and H % Hg == 0
        and B * H >= 64
        and K in {64, 128, 256}
        and V > 0
        and T > 0
        and T % 64 == 0
    )
