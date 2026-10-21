# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dispatch the validated Ascend KDA kernels with bounded sequence segments."""

import torch

from .chunk import (
    _ascend_native_forward_single,
    _validate_chunk_kda_inputs,
    chunk_kda_fwd_infer_triton,
)


def chunk_kda(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, allow_neg_eigval=False, safe_gate=True,
    lower_bound=None, state_v_first=False, cu_seqlens=None, chunk_size=16,
    *, A_log=None, dt_bias=None,
):
    """Return KDA output and optional final state on Ascend 910B.

    Q/K use [B,T,H,K]; V, gates and beta use HV heads with HV divisible by H.
    Gates, Q/K normalization and beta sigmoid stay fused into the intra kernel.
    State is carried between 1024-token segments, in K/V layout internally.
    Packed inputs use B=1 and one state per nonempty sequence.
    """
    if q.device.type != "npu":
        raise ValueError("Ascend KDA requires NPU tensors")
    _validate_chunk_kda_inputs(
        q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias, chunk_size,
        state_v_first, use_qk_l2norm_in_kernel, use_gate_in_kernel,
        use_beta_sigmoid_in_kernel, allow_neg_eigval, safe_gate, lower_bound,
    )
    if min(q.shape) <= 0:
        raise ValueError("KDA input dimensions must be positive")
    common = dict(
        scale=scale, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first, chunk_size=chunk_size, safe_gate=safe_gate,
        lower_bound=lower_bound, A_log=A_log, dt_bias=dt_bias,
        forward_impl=chunk_kda_fwd_infer_triton,
    )
    if cu_seqlens is None:
        return _ascend_native_forward_single(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state, **common,
        )
    # Preserve source serial packed dispatch; no batching or fusion changes.
    boundaries = cu_seqlens.detach().cpu().tolist()
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != q.shape[1]:
        raise ValueError("cu_seqlens must span the full packed input")
    if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError("Packed sequences must have positive lengths")
    outputs, states = [], []
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        output, state = _ascend_native_forward_single(
            q=q[:, start:end], k=k[:, start:end], v=v[:, start:end],
            g=g[:, start:end], beta=beta[:, start:end],
            initial_state=initial_state[i:i+1] if initial_state is not None else None,
            **common,
        )
        outputs.append(output)
        if output_final_state:
            states.append(state)
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0) if output_final_state else None
