# Copyright 2026 FlagOS Contributors

# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

"""Native GDN pipeline: preprocess, recompute W/U, update state, project output."""

import logging

import torch

from flag_attn.FLA.utils import SUPPRESS_LEVEL

from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fused_wu_h,
)
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.chunk_o import chunk_fwd_o
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.chunk_fused_tail_vblock import (
    can_use_fused_tail_vblock,
    chunk_gated_delta_rule_fused_tail_vblock,
)
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flag_attn.runtime.backend._thead.FLA.gated_delta_rule.wy_fast import recompute_w_u_fwd

logger = logging.getLogger(__name__)


def _chunk_size_for_sequence(T: int, is_varlen: bool) -> int:
    if is_varlen:
        return 64
    return min(64, max(16, 1 << (T - 1).bit_length()))


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    logger.debug("GEMS CHUNK GATED DELTA RULE FWD")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()

    chunk_size = _chunk_size_for_sequence(q.shape[1], cu_seqlens is not None)

    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
    )
    if (
        SUPPRESS_LEVEL < 3 and initial_state is None and cu_seqlens is None
        and output_final_state and chunk_size == 64 and q.shape[1] % 64 == 0
        and q.shape[-1] == 128 and v.shape[-1] == 128
        and k.dtype in (torch.float16, torch.bfloat16)
    ):
        h, v_new, final_state = chunk_gated_delta_rule_fused_wu_h(k, v, beta, g, A)
        o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, chunk_size=64)
        return g, o, A, final_state, None, None, None

    block_v = 128 if v.shape[-1] == 128 else 64
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        block_v=block_v,
    )
    if SUPPRESS_LEVEL < 3 and can_use_fused_tail_vblock(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    ):
        o, final_state = chunk_gated_delta_rule_fused_tail_vblock(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g,
            initial_state=initial_state,
            scale=scale,
        )
        return g, o, A, final_state, None, None, None
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    return g, o, A, final_state, w, h, v_new
