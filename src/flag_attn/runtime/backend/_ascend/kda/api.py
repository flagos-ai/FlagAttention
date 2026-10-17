# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Map the public inference signature to Ascend KDA without hiding options."""

import torch


def chunk_kda_fwd_infer(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    state_v_first=False, cu_seqlens=None, chunk_indices=None, chunk_size=16,
    safe_gate=False, lower_bound=None, A_log=None, dt_bias=None,
):
    if torch.is_grad_enabled():
        raise RuntimeError("Ascend KDA only supports inference mode")
    if chunk_indices is not None:
        raise NotImplementedError("Ascend KDA does not accept precomputed chunk_indices")
    from .forward import chunk_kda

    return chunk_kda(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, state_v_first=state_v_first,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, safe_gate=safe_gate,
        lower_bound=lower_bound, A_log=A_log, dt_bias=dt_bias,
    )
