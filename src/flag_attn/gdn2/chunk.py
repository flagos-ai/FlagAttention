"""Backend-independent Triton implementation of chunked GDN2 forward."""

import torch
import triton
import triton.language as tl

from .native.chunk_fwd import chunk_gdn2_fwd

__all__ = ["chunk_gdn2"]


@triton.jit
def _l2_normalize_kernel(q, k, q_out, k_out, K: tl.constexpr, BLOCK_K: tl.constexpr):
    cols = tl.arange(0, BLOCK_K)
    offsets = tl.program_id(0).to(tl.int64) * K + cols
    mask = cols < K
    q_values = tl.load(q + offsets, mask, other=0).to(tl.float32)
    k_values = tl.load(k + offsets, mask, other=0).to(tl.float32)
    q_inv_norm = tl.rsqrt(tl.sum(q_values * q_values, axis=0) + 1e-12)
    k_inv_norm = tl.rsqrt(tl.sum(k_values * k_values, axis=0) + 1e-12)
    tl.store(q_out + offsets, q_values * q_inv_norm, mask)
    tl.store(k_out + offsets, k_values * k_inv_norm, mask)


def chunk_gdn2(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    A_log=None,
    dt_bias=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    state_v_first=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    chunk_size=16,
    return_intermediate_states=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
):
    """Run GDN2 over fixed-length batches or packed variable-length sequences."""
    if q.ndim != 4 or any(x.shape != q.shape for x in (k, g, b)):
        raise ValueError("q, k, g and b must have the same [B, T, H, K] shape")
    if v.ndim != 4 or w.shape != v.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v and w must have the same [B, T, H, V] shape")
    if chunk_size not in (16, 32, 64):
        raise ValueError("GDN2 supports chunk_size in {16, 32, 64}")
    if not 0 < q.shape[-1] <= 256:
        raise ValueError("GDN2 supports key dimensions in [1, 256]")
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("cu_seqlens requires batch size 1")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be float32")
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log must be provided when use_gate_in_kernel=True")
        if safe_gate and (lower_bound is None or not -5 <= lower_bound < 0):
            raise ValueError("safe_gate requires lower_bound in [-5, 0)")

    q, k, v, g, b, w = (tensor.contiguous() for tensor in (q, k, v, g, b, w))
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel and q.numel():
        K = q.shape[-1]
        q_normalized, k_normalized = torch.empty_like(q), torch.empty_like(k)
        _l2_normalize_kernel[(q.numel() // K,)](
            q, k, q_normalized, k_normalized, K, triton.next_power_of_2(K)
        )
        q, k = q_normalized, k_normalized

    o, final_state, *_, h, _initial_state = chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
    )
    if return_intermediate_states:
        return o, final_state, h
    return o, final_state
