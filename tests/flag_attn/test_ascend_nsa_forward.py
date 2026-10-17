# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend selected-only INTERNAL API, not the complete public NSA facade."""
import pytest
import torch

from flag_attn.testing import backend

pytestmark = pytest.mark.skipif(
    not backend.is_available() or backend.runtime.device.vendor_name != "ascend",
    reason="Ascend selected-only validation requires an available NPU",
)


@pytest.mark.parallel_nsa
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [64, 128])
@torch.inference_mode()
def test_ascend_nsa_compression_public_alias(dtype, length):
    import flag_attn
    from flag_attn.parallel_nsa import parallel_nsa_compression
    from flag_attn.runtime.backend._ascend.nsa.api import parallel_nsa_compression as adapter

    assert parallel_nsa_compression is adapter
    assert flag_attn.parallel_nsa_compression is adapter
    torch.manual_seed(48)
    q = torch.randn(1, length, 16, 64).to(dtype)
    k, v = [torch.randn(1, length // 64, 1, 64).to(dtype) for _ in range(2)]
    logits = torch.einsum("bthk,bchk->bthc", q.float(), k.float().repeat_interleave(16, dim=2)) / 8
    visible = torch.arange(length // 64)[None, :] < (torch.arange(length)[:, None] + 1) // 64
    logits = logits.masked_fill(~visible[None, :, None, :], float("-inf"))
    expected = torch.einsum("bthc,bchv->bthv", logits.softmax(-1).nan_to_num(),
                            v.float().repeat_interleave(16, dim=2))
    expected_lse = logits.logsumexp(-1).masked_fill(~visible.any(-1)[None, :, None], 0)
    inputs = [x.to(backend.device) for x in (q, k, v)]
    for implementation in (parallel_nsa_compression, lambda *args: flag_attn.parallel_nsa(*args, mode="compression")):
        output, lse = implementation(*inputs)
        assert not output.requires_grad and not lse.requires_grad
        assert output.dtype == dtype
        torch.testing.assert_close(output.cpu().float(), expected, rtol=1e-1, atol=2e-1)
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=1e-1, atol=2e-1)


@pytest.mark.parallel_nsa
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,packed,token_group", [(32, False, 4), (33, False, 1), (33, True, 1)])
@torch.inference_mode()
def test_ascend_nsa_selected_forward(dtype, length, packed, token_group):
    from flag_attn.runtime.backend._ascend.nsa.forward import parallel_nsa_fwd

    torch.manual_seed(47)
    hosts = [torch.randn(1, length, heads, 32).to(dtype) for heads in (16, 1, 1)]
    indices = torch.empty(1, length, 1, 3, dtype=torch.int32)
    counts = torch.full((1, length, 1), 3, dtype=torch.int32)
    boundaries = [0, 16, length] if packed else [0, length]
    expected = torch.zeros(1, length, 16, 32)
    expected_lse = torch.full((1, length, 16), float("-inf"))
    q, k, v = [x.float() for x in hosts]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for token in range(start, end):
            local = token - start
            indices[0, token, 0] = torch.tensor([local // 16, -1, 99])
            if token_group == 1 and local % 7 == 0:
                counts[0, token, 0] = 0
                continue
            selected_start = start + (local // 16) * 16
            scores = q[0, token] @ k[0, selected_start:token + 1, 0].T / (32 ** 0.5)
            expected[0, token] = scores.softmax(-1) @ v[0, selected_start:token + 1, 0]
            expected_lse[0, token] = scores.logsumexp(-1)
    cu = torch.tensor(boundaries, dtype=torch.int32, device=backend.device) if packed else None
    output, lse = parallel_nsa_fwd(
        *[x.to(backend.device) for x in hosts], indices.to(backend.device),
        counts.to(backend.device) if token_group == 1 else 3, 16,
        cu_seqlens=cu, token_group=token_group,
    )
    assert output.dtype == dtype
    assert lse.dtype == torch.float32
    torch.testing.assert_close(output.cpu().float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-2, rtol=2e-2)
