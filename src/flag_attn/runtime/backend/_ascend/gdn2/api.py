# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Public GDN2 contract for Ascend's inference-only fused implementation."""

import torch


def chunk_gdn2(
    q, k, v, g, b, w, *, A_log=None, dt_bias=None, scale=None,
    initial_state=None, output_final_state=False, state_v_first=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, safe_gate=False,
    lower_bound=None, chunk_size=16, return_intermediate_states=False,
    cu_seqlens=None, cu_seqlens_cpu=None, chunk_indices=None,
):
    """Run inference; reject differentiable calls instead of losing gradients."""
    tensors = (q, k, v, g, b, w, A_log, dt_bias, initial_state)
    if torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in tensors
    ):
        raise RuntimeError("Ascend GDN2 only supports inference; backward is unavailable")
    from .chunk_fwd_infer import chunk_gdn2_fwd_infer

    # The fused 910B kernel uses K/V-oriented state loads. Normalize the public
    # V/K layout at the boundary, just as the existing Ascend KDA wrapper does.
    kernel_state = initial_state
    if state_v_first and kernel_state is not None:
        kernel_state = kernel_state.transpose(-1, -2).contiguous()
    result = chunk_gdn2_fwd_infer(
        q, k, v, g, b, w, A_log=A_log, dt_bias=dt_bias, scale=scale,
        initial_state=kernel_state, output_final_state=output_final_state,
        state_v_first=False, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel, safe_gate=safe_gate,
        lower_bound=lower_bound, chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states, cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=chunk_indices,
    )
    if not state_v_first:
        return result
    output, final_state, *history = result
    if final_state is not None:
        final_state = final_state.transpose(-1, -2).contiguous()
    if history:
        return output, final_state, history[0].transpose(-1, -2).contiguous()
    return output, final_state
