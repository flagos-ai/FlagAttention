# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Allocation-sensitive prefill regression and shared preparation boundaries."""

import pytest
import torch

pytest.importorskip("triton.experimental.tle.language", reason="FlagTree/TLE required")

from flag_attn import chunk_gated_delta_rule_fwd
from flag_attn.testing.recurrent_gdn import reference

pytestmark = [
    pytest.mark.gdn_forward,
    pytest.mark.chunk_gated_delta_rule,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["short_gva", "tail", "ragged"])
@torch.inference_mode()
def test_shared_preparation_poison_and_repeat(dtype, case, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("register preparation requires Hopper")
    torch.manual_seed(2091)
    batch, length, h, hv, dk, dv = (1, 512, 16, 32, 128, 128)
    cu = None
    if case == "tail":
        batch, length, h, hv, dk, dv = 2, 71, 2, 6, 60, 80
    if case == "ragged":
        batch, length, h, hv, dk, dv = 1, 82, 2, 6, 128, 64
        cu = torch.tensor([0, 0, 7, 71, 82], device="cuda", dtype=torch.int64)
    n = batch if cu is None else cu.numel() - 1
    q = torch.randn((batch, length, h, dk), device="cuda", dtype=dtype) / dk**0.5
    k = torch.randn_like(q) / dk**0.5
    v = torch.randn((batch, length, hv, dv), device="cuda", dtype=dtype)
    g = -torch.rand((batch, length, hv), device="cuda", dtype=torch.float32) * 0.1
    if case == "short_gva":
        g = g.to(dtype)
    beta = torch.rand_like(g)
    initial = None if case == "short_gva" else torch.randn((n, hv, dk, dv), device="cuda") * 0.05
    cache = torch.zeros((n, hv, dv, dk), device="cuda") if initial is None else initial.transpose(-1, -2).contiguous()
    expected = reference(
        dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=dk**-0.5,
            initial_state=cache,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=False,
        )
    )
    expected = (expected[0], expected[1].transpose(-1, -2))
    original_empty, original_like, original_new = torch.empty, torch.empty_like, torch.Tensor.new_empty

    def poison(t):
        if t.is_cuda and t.is_floating_point():
            t.fill_(float("nan"))
        return t

    monkeypatch.setattr(torch, "empty", lambda *a, **kw: poison(original_empty(*a, **kw)))
    monkeypatch.setattr(torch, "empty_like", lambda *a, **kw: poison(original_like(*a, **kw)))
    monkeypatch.setattr(torch.Tensor, "new_empty", lambda self, *a, **kw: poison(original_new(self, *a, **kw)))
    monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "0")
    monkeypatch.setenv("FLAG_ATTN_GDN_REGISTER_PREP", "1")
    previous = None
    for _ in range(3):
        result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, dk**-0.5, initial, True, cu)
        assert result[0].dtype == torch.float32
        for actual, target in zip((result[1], result[3]), expected):
            assert torch.isfinite(actual).all()
            ratio = (actual.float() - target.float()).square().mean().sqrt() / (
                target.float().square().mean().sqrt() + 1e-8
            )
            assert ratio < 0.01, ratio.item()
        if previous is not None:
            assert all(torch.equal(a, b) for a, b in zip(result[:4], previous))
        previous = [x.clone() for x in result[:4]]
