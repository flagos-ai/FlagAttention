# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Inference-only scalar GDN recurrence with vLLM-compatible state snapshots.

States use [pool, HV, V, K]. Q/K/V may be views into a packed projection.
Each CTA owns a disjoint V slice of one request/head, keeping its FP32 state
in registers through all new tokens. All requested per-token cache writes
are retained. No FP32 matmul/TF32 arithmetic is used.
"""

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _load(ptr, mask, USE_TLE: tl.constexpr, ASYNC: tl.constexpr = False):
    if USE_TLE:
        return tle.load(ptr, mask=mask, other=0, is_async=ASYNC)
    return tl.load(ptr, mask=mask, other=0)


@triton.jit(do_not_specialize=["T"])
def _recurrent_fwd_kernel(
    Q,
    Kp,
    Vp,
    G,
    Beta,
    H0,
    HT,
    O,
    Cu,
    Indices,
    Accepted,
    ALog,
    DtBias,
    T: tl.int64,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    HS: tl.constexpr,
    OS: tl.constexpr,
    FS: tl.constexpr,
    IS: tl.constexpr,
    ACCEPT_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    VARLEN: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    SPEC: tl.constexpr,
    INPLACE: tl.constexpr,
    VECTOR_BETA: tl.constexpr,
    NORMALIZE: tl.constexpr,
    USE_TLE: tl.constexpr,
    ASYNC_STATE: tl.constexpr,
    FUSED_GATES: tl.constexpr,
    SPLIT_GRID: tl.constexpr,
    WIDE_OFFSETS: tl.constexpr,
):
    iv = tl.program_id(0)
    if SPLIT_GRID:
        n, hv = tl.program_id(2), tl.program_id(1)
    else:
        nh = tl.program_id(1)
        n, hv = nh // HV, nh % HV
    h = hv // (HV // H)
    if WIDE_OFFSETS:
        n, hv, h = n.to(tl.int64), hv.to(tl.int64), h.to(tl.int64)
    if VARLEN:
        bos = tl.load(Cu + n).to(tl.int64)
        end = tl.load(Cu + n + 1).to(tl.int64)
        length = end - bos
        qb, kb, vb = bos * QS[1], bos * KS[1], bos * VS[1]
        gb, bb = bos * GS[1], bos * BS[1]
        ob = bos * OS[1]
    else:
        bos = n * T
        length = 1 if FUSED_GATES else T
        qb, kb, vb = n * QS[0], n * KS[0], n * VS[0]
        gb, bb = n * GS[0], n * BS[0]
        ob = n * OS[0]
    if length == 0:
        return
    state_idx = n.to(tl.int64)
    if HAS_INDICES:
        it = 0
        if SPEC:
            it = tl.load(Accepted + n * ACCEPT_STRIDE).to(tl.int64) - 1
        state_idx = tl.load(Indices + n * IS[0] + it * IS[1]).to(tl.int64)
    rk = tl.arange(0, BK)
    rv = iv * BV + tl.arange(0, BV)
    if WIDE_OFFSETS:
        rk, rv = rk.to(tl.int64), rv.to(tl.int64)
    mk, mv = rk < DK, rv < DV
    mh = mv[:, None] & mk[None, :]
    if HAS_INDICES and state_idx <= 0:
        for t in range(length):
            pos = t.to(tl.int64) if WIDE_OFFSETS else t
            tl.store(O + ob + pos * OS[1] + hv * OS[2] + rv * OS[3], 0, mask=mv)
            if not INPLACE:
                tl.store(
                    HT + (bos + pos) * FS[0] + hv * FS[1] + rv[:, None] * FS[2] + rk[None, :] * FS[3],
                    0,
                    mask=mh,
                )
        return

    state = _load(
        H0 + state_idx * HS[0] + hv * HS[1] + rv[:, None] * HS[2] + rk[None, :] * HS[3],
        mh,
        USE_TLE,
        ASYNC_STATE,
    ).to(tl.float32)
    for t in range(length):
        pos = t.to(tl.int64) if WIDE_OFFSETS else t
        q = _load(Q + qb + pos * QS[1] + h * QS[2] + rk * QS[3], mk, USE_TLE).to(tl.float32)
        if FUSED_GATES:
            # Packed projection shares one base pointer. Preserve that aliasing
            # information rather than passing three independent kernel pointers.
            k = _load(Q + H * DK + kb + h * KS[2] + rk, mk, USE_TLE).to(tl.float32)
            v = _load(Q + 2 * H * DK + vb + hv * VS[2] + rv, mv, USE_TLE).to(tl.float32)
        else:
            k = _load(Kp + kb + pos * KS[1] + h * KS[2] + rk * KS[3], mk, USE_TLE).to(tl.float32)
            v = _load(Vp + vb + pos * VS[1] + hv * VS[2] + rv * VS[3], mv, USE_TLE).to(tl.float32)
        if NORMALIZE:
            q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6)
            k = k / tl.sqrt(tl.sum(k * k) + 1.0e-6)
        q = q * SCALE
        g = tl.load(G + gb + pos * GS[1] + hv * GS[2]).to(tl.float32)
        if FUSED_GATES:
            a_log = tl.load(ALog + hv).to(tl.float32)
            dt_bias = tl.load(DtBias + hv).to(tl.float32)
            x = g + dt_bias
            g = -tl.exp(a_log) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
        state = state * tl.exp(g)
        delta = v - tl.sum(state * k[None, :], 1)
        if VECTOR_BETA:
            beta = _load(Beta + bb + pos * BS[1] + hv * BS[2] + rv * BS[3], mv, USE_TLE).to(tl.float32)
        else:
            beta = tl.load(Beta + bb + pos * BS[1] + hv * BS[2]).to(tl.float32)
        if FUSED_GATES:
            beta = tl.sigmoid(beta)
        delta = delta * beta
        state = state + delta[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1)
        tl.store(O + ob + pos * OS[1] + hv * OS[2] + rv * OS[3], out, mask=mv)
        if INPLACE:
            target = n.to(tl.int64)
            if HAS_INDICES:
                target = tl.load(Indices + n * IS[0] + pos * IS[1]).to(tl.int64)
            if not HAS_INDICES or target > 0:
                tl.store(
                    HT + target * FS[0] + hv * FS[1] + rv[:, None] * FS[2] + rk[None, :] * FS[3],
                    state,
                    mask=mh,
                )
        else:
            target = bos + pos
            tl.store(
                HT + target * FS[0] + hv * FS[1] + rv[:, None] * FS[2] + rk[None, :] * FS[3],
                state,
                mask=mh,
            )


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
    use_tle=True,
    gate_parameters=None,
):
    """Preallocated launch shared by the public API and the benchmark."""
    direct_packed = gate_parameters is not None and k is None
    if direct_packed:
        HV, DV, DK = initial_state.shape[1:]
        B, T = q.shape[0], 1
        H = (q.shape[1] - HV * DV) // (2 * DK)
    else:
        B, T, H, DK = k.shape
        HV, DV = v.shape[2:]
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if N == 0 or T == 0:
        return None
    # A small V tile increases available CTAs and reduces live state registers.
    # Once request/head parallelism is sufficient, amortize Q/K loads over 8 V rows.
    if config is None:
        # Packed gates add independent work per CTA. Keep 4 rows until the
        # request/head grid supplies enough 8-row CTAs to populate Hopper.
        threshold = 256 if gate_parameters is not None else 128
        tile = 4 if N * HV < threshold else 8
        if initial_state.dtype != torch.float32:
            if gate_parameters is not None:
                # Half-size cache traffic exposes duplicated Q/K and gate work
                # sooner. Use wider tiles once the request/head grid is ample;
                # cap the packed path's live FP32 state at 2048 elements/CTA.
                tile = 4 if N * HV < 64 else 8
                if N * HV >= 512:
                    tile = min(16, 2048 // triton.next_power_of_2(DK))
            elif N * HV >= 128:
                # With half-size snapshots, amortize repeated normalization and
                # metadata sooner once the grid offers enough resident CTAs.
                tile = min(32, 4096 // triton.next_power_of_2(DK))
        config = (tile, 1, False)
    BV, warps, async_state = config
    BV = min(triton.next_power_of_2(DV), BV)
    idx_strides = (1, 0)
    if ssm_state_indices is not None:
        idx_strides = (ssm_state_indices.stride(0), 0) if ssm_state_indices.ndim == 1 else ssm_state_indices.stride()

    # Batch strides are unused for B=1 packed input. Canonicalizing them avoids
    # recompiling identical kernels whenever the packed token count changes.
    def strides(x):
        s = x.stride()
        return (0, *s[1:]) if cu_seqlens is not None else s

    if direct_packed:
        # The kernel needs pointer/stride metadata, not three Tensor views.
        # These are exactly the strides of the canonical packed views.
        qs = ks = (q.stride(0), 0, DK, 1)
        vs = (q.stride(0), 0, DV, 1)
        gs = (g.stride(0), HV, 1)
        bs = (beta.stride(0), HV, 1)
    else:
        qs, ks, vs, gs, bs = map(strides, (q, k, v, g, beta))

    split_grid = N * HV > 65535

    # Pool indices and packed token offsets already use int64. Promote the
    # remaining address products only when a tensor's relative span needs it.
    def span(x, start=0):
        return sum(max(d - 1, 0) * abs(s) for d, s in zip(x.shape[start:], x.stride()[start:]))

    address_inputs = (q, k, v, g, beta, out, ssm_state_indices, num_accepted_tokens)
    wide_offsets = any(x is not None and span(x) > 2**31 - 1 for x in address_inputs)
    state_inputs = (initial_state,) if final_state is initial_state else (initial_state, final_state)
    wide_offsets |= any(span(x, 1) > 2**31 - 1 for x in state_inputs)
    grid = (triton.cdiv(DV, BV), HV, N) if split_grid else (triton.cdiv(DV, BV), N * HV)
    return _recurrent_fwd_kernel[grid](
        q,
        k if gate_parameters is None else None,
        v if gate_parameters is None else None,
        g,
        beta,
        initial_state,
        final_state,
        out,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        None if gate_parameters is None else gate_parameters[0],
        None if gate_parameters is None else gate_parameters[1],
        T,
        H,
        HV,
        DK,
        DV,
        triton.next_power_of_2(DK),
        BV,
        qs,
        ks,
        vs,
        gs,
        bs,
        initial_state.stride(),
        strides(out),
        final_state.stride(),
        idx_strides,
        0 if num_accepted_tokens is None else num_accepted_tokens.stride(0),
        scale,
        cu_seqlens is not None,
        ssm_state_indices is not None,
        num_accepted_tokens is not None,
        inplace_final_state,
        beta.ndim == 4,
        normalize,
        use_tle,
        async_state,
        gate_parameters is not None,
        split_grid,
        wide_offsets,
        num_warps=warps,
        num_stages=1,
    )


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
    """Compute scalar GDN outputs and updated states (forward inference only).

    Q/K: [B,T,H,K], V: [B,T,HV,V], g: [B,T,HV] (log decay),
    beta: [B,T,HV] or [B,T,HV,V]. HV must be divisible by H.
    initial_state: [pool,HV,V,K]; dense no-index mode uses one row per sequence.
    Packed variable lengths require B=1 and contiguous cu_seqlens [N+1].
    K and V must be in 1..256. N and HV must each fit the CUDA Y/Z-grid limit 65535.
    A selected initial index <=0 marks a padded request: its outputs and returned
    snapshots are zero and cache is untouched. For an active request, nonpositive
    per-token write indices suppress only that store; recurrence/output continues.
    In-place writes must not alias another request's read or write slots.

    With indices [N,L], accepted[n]-1 selects the previous state to read and
    indices[n,t] receives every new state. In non-inplace mode all per-token
    snapshots are returned in [B*T,HV,V,K]. Indices and offsets are caller
    metadata; they must be valid and remain device resident (no host sync).
    """
    tensors = (q, k, v, g, beta, initial_state)
    if any(x.device != q.device for x in tensors) or q.device.type != "cuda":
        raise ValueError("all tensors must be on the same CUDA device")
    if any(x.requires_grad for x in tensors) and torch.is_grad_enabled():
        raise RuntimeError("fused recurrent GDN supports forward inference only")
    if q.ndim != 4 or k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("expected matching [B,T,H,K] Q/K and [B,T,HV,V] V")
    B, T, H, DK = q.shape
    HV, DV = v.shape[2:]
    if H <= 0 or HV <= 0 or HV % H or g.shape != (B, T, HV) or beta.shape not in ((B, T, HV), (B, T, HV, DV)):
        raise ValueError("invalid grouped heads, gate, or beta shape")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Q/K/V must have the same FP16, BF16 or FP32 dtype")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (HV, DV, DK):
        raise ValueError("state must use [pool,HV,V,K] layout")
    if initial_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("state must be FP16, BF16 or FP32")
    if cu_seqlens is not None and (B != 1 or cu_seqlens.ndim != 1):
        raise ValueError("packed variable-length inputs require B=1 and 1D offsets")
    if cu_seqlens is not None and cu_seqlens.stride(0) != 1:
        raise ValueError("cu_seqlens must be contiguous")
    if num_accepted_tokens is not None and ssm_state_indices is None:
        raise ValueError("accepted-token counts require state indices")
    if ssm_state_indices is not None and ssm_state_indices.ndim not in (1, 2):
        raise ValueError("state indices must be 1D or 2D")
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if N < 0:
        raise ValueError("cu_seqlens must contain at least one offset")
    if N > 65535 or HV > 65535:
        raise ValueError("request and value-head counts must each not exceed 65535")
    for metadata in (cu_seqlens, ssm_state_indices, num_accepted_tokens):
        if metadata is not None and (metadata.device != q.device or metadata.dtype not in (torch.int32, torch.int64)):
            raise ValueError("offsets, indices and counts must be CUDA int32/int64 tensors")
    if ssm_state_indices is not None and ssm_state_indices.shape[0] != N:
        raise ValueError("state indices need one row per sequence")
    if num_accepted_tokens is not None and ssm_state_indices.ndim != 2:
        raise ValueError("speculative state selection requires 2D state indices")
    if num_accepted_tokens is not None and (num_accepted_tokens.ndim != 1 or num_accepted_tokens.numel() != N):
        raise ValueError("accepted-token counts need one element per sequence")
    if ssm_state_indices is None and initial_state.shape[0] < N:
        raise ValueError("dense state needs one row per sequence")
    if not (0 < DK <= 256 and 0 < DV <= 256):
        raise ValueError("supported K and V dimensions are 1..256")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    final = (
        initial_state
        if inplace_final_state
        else torch.empty(
            (B * T, HV, DV, DK),
            dtype=initial_state.dtype,
            device=q.device,
        )
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
