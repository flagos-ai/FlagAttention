# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import logging
import os

import torch

from flag_attn.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_attn.FLA.gated_delta_rule.chunk_fused_tail_vblock import (
    can_use_fused_tail_vblock,
    chunk_gated_delta_rule_fused_tail_vblock,
)
from flag_attn.FLA.gated_delta_rule.chunk_fused_forward import (
    can_use_two_kernel_fused_forward,
    chunk_gdn_two_kernel_fwd,
)
from flag_attn.FLA.chunk_o import chunk_fwd_o
from flag_attn.FLA.gated_delta_rule.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril,
)
from flag_attn.FLA.utils import SUPPRESS_LEVEL
from flag_attn.FLA.gated_delta_rule.wy_fast import (
    maybe_set_tle_recompute_allocator,
    recompute_w_u_fwd,
)

logger = logging.getLogger(__name__)
HYBRID_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_HYBRID_TLE"


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
    q_contiguous = q.is_contiguous()
    k_contiguous = k.is_contiguous()
    v_contiguous = v.is_contiguous()
    g_contiguous = g.is_contiguous()
    beta_contiguous = beta.is_contiguous()
    initial_state_contiguous = initial_state is None or initial_state.is_contiguous()
    cu_seqlens_contiguous = cu_seqlens is None or cu_seqlens.is_contiguous()
    if not (
        q_contiguous
        and k_contiguous
        and v_contiguous
        and g_contiguous
        and beta_contiguous
        and initial_state_contiguous
        and cu_seqlens_contiguous
    ):
        if not q_contiguous:
            q = q.contiguous()
        if not k_contiguous:
            k = k.contiguous()
        if not v_contiguous:
            v = v.contiguous()
        if not g_contiguous:
            g = g.contiguous()
        if not beta_contiguous:
            beta = beta.contiguous()
        if not initial_state_contiguous:
            initial_state = initial_state.contiguous()
        if not cu_seqlens_contiguous:
            cu_seqlens = cu_seqlens.contiguous()

    chunk_size = _chunk_size_for_sequence(q.shape[1], cu_seqlens is not None)
    maybe_set_tle_recompute_allocator(q.device, cu_seqlens)

    if initial_state is None and output_final_state:
        try:
            from flag_attn.FLA.gated_delta_rule.api import (
                _can_use_tle_chunk_gated_delta_rule,
                _tle_chunk_gated_delta_rule_fwd,
            )

            if _can_use_tle_chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            ):
                hybrid_enabled = os.environ.get(HYBRID_TLE_ENV, "1").lower() not in {
                    "0",
                    "false",
                    "off",
                    "no",
                }
                preprocessed = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
                    g=g,
                    k=k,
                    beta=beta,
                    cu_seqlens=cu_seqlens,
                    chunk_size=chunk_size,
                    output_dtype=k.dtype,
                    use_g_in_kkt=False,
                    output_gated_from_ungated=hybrid_enabled,
                )
                if hybrid_enabled:
                    g_tle, A_0, A_g = preprocessed
                else:
                    g_tle, A_0 = preprocessed
                o, final_state = _tle_chunk_gated_delta_rule_fwd(
                    q=q,
                    k=k,
                    v=v,
                    g_cumsum=g_tle,
                    beta=beta,
                    a=A_0,
                    scale=float(scale),
                )
                if not hybrid_enabled:
                    _, A_g = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
                        g=g,
                        k=k,
                        beta=beta,
                        cu_seqlens=cu_seqlens,
                        chunk_size=chunk_size,
                        output_dtype=k.dtype,
                    )
                return g_tle, o, A_g, final_state, None, None, None
        except ImportError:
            pass

    if can_use_two_kernel_fused_forward(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    ):
        g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
            g=g,
            k=k,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            output_dtype=k.dtype,
        )
        o, final_state = chunk_gdn_two_kernel_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            g=g,
            scale=float(scale),
        )
        return g, o, A, final_state, None, None, None

    g, A = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=k.dtype,
    )
    block_v = 128 if v.shape[-1] == 128 else 64
    pipe_stages = 5 if v.shape[-1] >= 256 else (3 if v.shape[-1] == 128 else 2)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        block_v=block_v,
        pipe_stages=pipe_stages,
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
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new
