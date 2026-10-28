# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""State-safe recurrent fixtures and an independent FP32 recurrence."""

import torch


def make_case(
    n=4,
    length=4,
    h=4,
    hv=8,
    kdim=128,
    vdim=128,
    dtype=torch.bfloat16,
    state_dtype=torch.float32,
    seed=0,
    strided=True,
    lengths=None,
    spec=False,
    vector_beta=False,
    padding=False,
    pool_min=0,
    projection_padding=0,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lengths = [length] * n if lengths is None else list(lengths)
    total = sum(lengths)
    width = 2 * h * kdim + hv * vdim
    packed = torch.randn((total, width + projection_padding), device="cuda", dtype=dtype, generator=gen)
    q, k, v = packed[:, :width].split((h * kdim, h * kdim, hv * vdim), dim=-1)
    q, k, v = q.reshape(1, total, h, kdim), k.reshape(1, total, h, kdim), v.reshape(1, total, hv, vdim)
    if not strided:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g = -torch.rand((1, total, hv), device="cuda", dtype=torch.float32, generator=gen)
    beta_shape = (1, total, hv, vdim) if vector_beta else (1, total, hv)
    beta = torch.rand(beta_shape, device="cuda", dtype=torch.float32, generator=gen)
    maxlen = max(max(lengths, default=0), 1)
    pool = max(pool_min, n * maxlen + 7)
    # One distinct row per request/token; slot zero is reserved.
    idx = (torch.randperm(n * maxlen, device="cuda", generator=gen) + 1).reshape(n, maxlen).to(torch.int32)
    if padding and n > 1:
        idx[-1] = 0
    accepted = (torch.arange(n, device="cuda", dtype=torch.int32) % maxlen + 1) if spec else None
    state = torch.randn((pool, hv, vdim, kdim), device="cuda", dtype=state_dtype, generator=gen) * 0.05
    offsets = [0]
    for l in lengths:
        offsets.append(offsets[-1] + l)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=kdim**-0.5,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=idx,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
    )


@torch.no_grad()
def reference(case):
    q, k, v, g, beta = (case[x] for x in ("q", "k", "v", "g", "beta"))
    state0 = case["initial_state"]
    final = (
        state0.clone()
        if case["inplace_final_state"]
        else torch.zeros(
            (q.shape[0] * q.shape[1], v.shape[2], v.shape[3], q.shape[3]), dtype=state0.dtype, device=q.device
        )
    )
    out = torch.zeros_like(v)
    B, T, H, DK = q.shape
    HV = v.shape[2]
    offsets = case["cu_seqlens"]
    offsets = offsets.tolist() if offsets is not None else list(range(0, (B + 1) * T, T))
    indices = case["ssm_state_indices"]
    indices = None if indices is None else indices.tolist()
    accepted = case["num_accepted_tokens"]
    accepted = None if accepted is None else accepted.tolist()
    qf, kf, vf = [x.reshape(-1, *x.shape[2:]).float() for x in (q, k, v)]
    gf, bf = [x.reshape(-1, *x.shape[2:]).float() for x in (g, beta)]
    of = out.reshape(-1, HV, v.shape[-1])
    for n, (bos, eos) in enumerate(zip(offsets, offsets[1:])):
        row = None if indices is None else indices[n]
        if isinstance(row, int):
            row = [row] * max(eos - bos, 1)
        read_idx = n if row is None else row[0 if accepted is None else accepted[n] - 1]
        if row is not None and read_idx <= 0:
            continue
        st = state0[read_idx].float().clone()
        for t, pos in enumerate(range(bos, eos)):
            qq = qf[pos].repeat_interleave(HV // H, 0)
            kk = kf[pos].repeat_interleave(HV // H, 0)
            if case["use_qk_l2norm_in_kernel"]:
                qq = qq / torch.sqrt((qq * qq).sum(-1, keepdim=True) + 1.0e-6)
                kk = kk / torch.sqrt((kk * kk).sum(-1, keepdim=True) + 1.0e-6)
            qq = qq * case["scale"]
            st = st * gf[pos].exp()[:, None, None]
            delta = vf[pos] - (st * kk[:, None, :]).sum(-1)
            bt = bf[pos]
            delta = delta * (bt if bt.ndim == 2 else bt[:, None])
            st = st + delta[:, :, None] * kk[:, None, :]
            of[pos] = (st * qq[:, None, :]).sum(-1).to(out.dtype)
            if case["inplace_final_state"]:
                dest = n if row is None else row[t]
                if row is None or dest > 0:
                    final[dest] = st.to(final.dtype)
            else:
                final[pos] = st.to(final.dtype)
    return out, final


def check_close(out, state, ref_out, ref_state):
    torch.testing.assert_close(out, ref_out, rtol=1.0e-2, atol=2.0e-3)
    tol = 2.0e-4 if state.dtype == torch.float32 else 1.0e-2
    torch.testing.assert_close(state, ref_state, rtol=tol, atol=2.0e-5 if state.dtype == torch.float32 else 2.0e-3)
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
