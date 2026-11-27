# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable implementation of the prepared GDN forward interface."""

import torch

from flag_attn.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_attn.FLA.chunk_o import chunk_fwd_o
from .wy_fast import recompute_w_u_fwd


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
    # The portable implementation materializes the intermediate state instead
    # of using a hardware-specific fused pipeline.
    return False


def chunk_gdn_two_kernel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_size = A.shape[-1]
    w, u = recompute_w_u_fwd(k, v, beta, g, A, None)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    return chunk_fwd_o(q, k, v_new, h, g, scale, chunk_size=chunk_size), final_state
