# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable GDN recurrence for inference, with indexed state snapshots.

The token-by-token form of the operator, for decode and short MTP windows
where the chunked decomposition has nothing to amortize. One program owns a
``[BV, BK]`` tile of one request's state and walks its tokens:

    S <- exp(g) * S
    delta = (v - S @ k) * beta
    S <- S + delta outer k
    o = S^T (q * scale)

State uses the ``[pool, HV, V, K]`` cache layout; ``ssm_state_indices``
selects which pool rows to read and write, so speculative decoding can keep
one snapshot per accepted token. Portable Triton only: explicit pointer
arithmetic with masks, FP32 recurrence, no tensor cores.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    indices,
    accepted,
    scale,
    T,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_v_b,
    stride_v_t,
    stride_v_h,
    stride_o_b,
    stride_o_t,
    stride_o_h,
    stride_s_pool,
    stride_s_h,
    stride_s_v,
    stride_g_b,
    stride_g_t,
    stride_idx_n,
    stride_idx_t,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INDICES: tl.constexpr,
    USE_ACCEPTED: tl.constexpr,
    INPLACE: tl.constexpr,
    VECTOR_BETA: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        i_b, t0 = 0, bos
    else:
        seq_len = T
        i_b, t0 = i_n, 0

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < DK
    m_v = o_v < DV
    m_s = m_v[:, None] & m_k[None, :]

    # Pick the pool row to read: with speculative decoding the caller passes
    # accepted[n], and accepted-1 selects the last committed snapshot.
    read_row = i_n
    if USE_INDICES:
        col = 0
        if USE_ACCEPTED:
            col = tl.load(accepted + i_n).to(tl.int32) - 1
        read_row = tl.load(indices + i_n * stride_idx_n + col * stride_idx_t).to(tl.int32)
        # A non-positive row marks a padded request: emit zeros, touch nothing.
        if read_row <= 0:
            for i_t in range(seq_len):
                p_o = o + i_b * stride_o_b + (t0 + i_t) * stride_o_t + i_hv * stride_o_h + o_v
                tl.store(p_o, tl.zeros([BV], dtype=o.dtype.element_ty), mask=m_v)
            return

    # [BV, BK] state tile, mirroring the [pool, HV, V, K] cache layout.
    p_h0 = h0 + read_row.to(tl.int64) * stride_s_pool + i_hv * stride_s_h + o_v[:, None] * stride_s_v + o_k[None, :]
    b_s = tl.load(p_h0, mask=m_s, other=0.0).to(tl.float32)

    for i_t in range(seq_len):
        t = t0 + i_t
        p_q = q + i_b * stride_q_b + t * stride_q_t + i_h * stride_q_h + o_k
        p_k = k + i_b * stride_q_b + t * stride_q_t + i_h * stride_q_h + o_k
        b_q = tl.load(p_q, mask=m_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
        if NORMALIZE:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1.0e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1.0e-6)
        p_v = v + i_b * stride_v_b + t * stride_v_t + i_hv * stride_v_h + o_v
        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        b_g = tl.load(g + i_b * stride_g_b + t * stride_g_t + i_hv).to(tl.float32)
        if VECTOR_BETA:
            b_beta = tl.load(beta + (i_b * stride_g_b + t * stride_g_t + i_hv) * DV + o_v, mask=m_v, other=0.0)
            b_beta = b_beta.to(tl.float32)
        else:
            b_beta = tl.load(beta + i_b * stride_g_b + t * stride_g_t + i_hv).to(tl.float32)

        b_s = b_s * tl.exp(b_g)
        b_delta = (b_v - tl.sum(b_s * b_k[None, :], axis=1)) * b_beta
        b_s = b_s + b_delta[:, None] * b_k[None, :]
        b_o = tl.sum(b_s * (b_q * scale)[None, :], axis=1)
        p_o = o + i_b * stride_o_b + t * stride_o_t + i_hv * stride_o_h + o_v
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=m_v)

        # Snapshot per token when the caller supplied write indices.
        if INPLACE:
            write_row = i_n
            if USE_INDICES:
                write_row = tl.load(indices + i_n * stride_idx_n + i_t * stride_idx_t).to(tl.int32)
            if (not USE_INDICES) or write_row > 0:
                p_ht = (
                    ht
                    + write_row.to(tl.int64) * stride_s_pool
                    + i_hv * stride_s_h
                    + o_v[:, None] * stride_s_v
                    + o_k[None, :]
                )
                tl.store(p_ht, b_s.to(ht.dtype.element_ty), mask=m_s)
        else:
            p_ht = (
                ht
                + (i_b * T + i_t).to(tl.int64) * (HV * DV * DK)
                + i_hv * (DV * DK)
                + o_v[:, None] * DK
                + o_k[None, :]
            )
            tl.store(p_ht, b_s.to(ht.dtype.element_ty), mask=m_s)


def _launch(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    normalize,
    out,
    final_state,
    *,
    config=None,
    use_tle=False,
    gate_parameters=None,
):
    """Launch the recurrence into caller-provided output and state buffers."""
    if gate_parameters is not None or k is None:
        raise NotImplementedError(
            "the portable GDN recurrence does not implement the packed projection "
            "route; it is provided by the Hopper backend"
        )
    del use_tle
    B, T, H, DK = k.shape
    HV, DV = v.shape[2:]
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if N == 0 or T == 0:
        return None

    BK = triton.next_power_of_2(DK)
    BV = min(config[0] if config else 64, triton.next_power_of_2(DV))
    idx_strides = (1, 0)
    if ssm_state_indices is not None:
        idx_strides = (
            (ssm_state_indices.stride(0), 0)
            if ssm_state_indices.ndim == 1
            else ssm_state_indices.stride()
        )

    grid = (triton.cdiv(DV, BV), N * HV)
    _recurrent_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=out,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        indices=ssm_state_indices,
        accepted=num_accepted_tokens,
        scale=float(scale),
        T=T,
        stride_q_b=q.stride(0),
        stride_q_t=q.stride(1),
        stride_q_h=q.stride(2),
        stride_v_b=v.stride(0),
        stride_v_t=v.stride(1),
        stride_v_h=v.stride(2),
        stride_o_b=out.stride(0),
        stride_o_t=out.stride(1),
        stride_o_h=out.stride(2),
        stride_s_pool=initial_state.stride(0),
        stride_s_h=initial_state.stride(1),
        stride_s_v=initial_state.stride(2),
        stride_g_b=g.stride(0),
        stride_g_t=g.stride(1),
        stride_idx_n=idx_strides[0],
        stride_idx_t=idx_strides[1],
        H=H,
        HV=HV,
        DK=DK,
        DV=DV,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        USE_INDICES=ssm_state_indices is not None,
        USE_ACCEPTED=num_accepted_tokens is not None,
        INPLACE=inplace_final_state,
        VECTOR_BETA=beta.ndim == 4,
        NORMALIZE=normalize,
        num_warps=4,
    )
    return None


def fused_recurrent_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    """Compute GDN outputs and updated states (forward inference only).

    Q/K: ``[B,T,H,K]``, V: ``[B,T,HV,V]``, g: ``[B,T,HV]`` (log decay),
    beta: ``[B,T,HV]`` or ``[B,T,HV,V]``; HV must be divisible by H.
    ``initial_state`` uses the ``[pool,HV,V,K]`` cache layout. Packed
    variable-length inputs require ``B=1`` and a contiguous ``cu_seqlens``.
    A selected index <= 0 marks a padded request: its outputs are zero and
    the cache is left untouched.
    """
    tensors = (q, k, v, g, beta, initial_state)
    if any(x.device != q.device for x in tensors):
        raise ValueError("all tensors must be on the same device")
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise RuntimeError("fused recurrent GDN supports forward inference only")
    if q.ndim != 4 or v.ndim != 4 or k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("expected matching [B,T,H,K] Q/K and [B,T,HV,V] V")
    B, T, H, DK = q.shape
    HV, DV = v.shape[2:]
    if HV % H or g.shape != (B, T, HV) or beta.shape not in ((B, T, HV), (B, T, HV, DV)):
        raise ValueError("invalid grouped heads, gate, or beta shape")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, DV, DK):
        raise ValueError("state must use [pool,HV,V,K] layout")
    if cu_seqlens is not None and (B != 1 or cu_seqlens.ndim != 1):
        raise ValueError("packed variable-length inputs require B=1 and 1D offsets")
    if num_accepted_tokens is not None and ssm_state_indices is None:
        raise ValueError("accepted-token counts require state indices")
    if not (0 < DK <= 256 and 0 < DV <= 256):
        raise ValueError("supported K and V dimensions are 1..256")

    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty((B * T, HV, DV, DK), dtype=initial_state.dtype, device=q.device)
    )
    _launch(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
        out,
        final,
    )
    return out, final
