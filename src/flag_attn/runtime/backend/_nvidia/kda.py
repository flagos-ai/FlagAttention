# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Automatic Hopper KDA selection with the portable implementation as fallback."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module

import torch

from flag_attn.FLA import chunk_kda as generic
from flag_attn.utils import has_triton_tle


@lru_cache(maxsize=1)
def _has_hopper_tle():
    if not has_triton_tle(
        3, 6, 0, required_api=("gpu.alloc", "gpu.local_ptr", "gpu.copy", "gpu.warp_specialize", "pipe")
    ):
        return False
    from triton._C.libtriton import ir

    return hasattr(ir.builder, "make_swizzled_shared_encoding_attr")


def chunk_kda_fwd_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 16,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    generic._validate_inputs(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        state_v_first,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        A_log,
        dt_bias,
    )
    q, k, v, g, beta, initial_state, cu_seqlens, chunk_indices, A_log, dt_bias = (
        tensor.contiguous() if tensor is not None else None
        for tensor in (q, k, v, g, beta, initial_state, cu_seqlens, chunk_indices, A_log, dt_bias)
    )
    implementation = generic.chunk_kda_fwd_infer
    if q.device.type == "cuda" and torch.cuda.get_device_capability(q.device)[0] == 9 and _has_hopper_tle():
        hopper = import_module("flag_attn.runtime.backend._nvidia.hopper.kda")
        reason = hopper._tle_input_error(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            True,
            True,
            True,
            False,
            state_v_first,
            cu_seqlens,
            safe_gate,
            lower_bound,
            A_log,
            dt_bias,
            chunk_size,
        )
        if reason is None:
            implementation = hopper.chunk_kda_fwd_infer
    return implementation(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        A_log,
        dt_bias,
    )
