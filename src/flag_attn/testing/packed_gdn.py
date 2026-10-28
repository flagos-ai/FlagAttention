# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Packed gate inputs adapted to the shared independent GDN reference."""

import torch

from .recurrent_gdn import make_case, reference


def make_packed_case(
    n=4,
    h=4,
    hv=8,
    kdim=128,
    vdim=128,
    dtype=torch.bfloat16,
    state_dtype=torch.float32,
    seed=0,
    pool_min=0,
    projection_padding=0,
):
    c = make_case(
        n, 1, h, hv, kdim, vdim, dtype, state_dtype, seed, pool_min=pool_min, projection_padding=projection_padding
    )
    width = 2 * h * kdim + hv * vdim
    mixed = c["q"].as_strided((n, width), (c["q"].stride(1), 1), storage_offset=0)
    gen = torch.Generator(device="cuda").manual_seed(seed + 71)
    a = torch.randn((n, hv), device="cuda", dtype=dtype, generator=gen)
    b = torch.randn((n, hv), device="cuda", dtype=dtype, generator=gen)
    A_log = torch.randn((hv,), device="cuda", generator=gen) * 0.2
    dt_bias = torch.randn((hv,), device="cuda", generator=gen)
    return dict(
        mixed_qkv=mixed,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=c["scale"],
        initial_state=c["initial_state"],
        out=torch.empty((n, 1, hv, vdim), device="cuda", dtype=dtype),
        ssm_state_indices=c["ssm_state_indices"][:, 0],
        use_qk_l2norm_in_kernel=True,
    )


def unpack_reference_case(case):
    """Gate math in FP32; recurrence is the same reference used for all decode."""
    mixed = case["mixed_qkv"]
    n = mixed.shape[0]
    hv, dv, dk = case["initial_state"].shape[1:]
    h = (mixed.shape[1] - hv * dv) // (2 * dk)
    q, k, v = mixed.split((h * dk, h * dk, hv * dv), -1)
    g = -case["A_log"].float().exp() * torch.nn.functional.softplus(
        case["a"].float() + case["dt_bias"].float(), threshold=20
    )
    beta = case["b"].float().sigmoid()
    return dict(
        q=q.view(n, 1, h, dk),
        k=k.view(n, 1, h, dk),
        v=v.view(n, 1, hv, dv),
        g=g.unsqueeze(1),
        beta=beta.unsqueeze(1),
        scale=case["scale"],
        initial_state=case["initial_state"],
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=case["ssm_state_indices"],
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=case["use_qk_l2norm_in_kernel"],
    )


@torch.no_grad()
def reference_packed(case):
    return reference(unpack_reference_case(case))
