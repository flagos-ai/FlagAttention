import pytest
import torch

from flag_attn import rotary_embedding


def test_rotary_embedding_matches_reference():
    q = torch.randn(2, 3, 2, 8)
    k = torch.randn(2, 3, 1, 8)
    cos = torch.randn(16, 4)
    sin = torch.randn(16, 4)
    positions = torch.tensor([[3, 5, 7], [0, 2, 4]])

    q_out, k_out = rotary_embedding(q, k, cos, sin, positions)
    selected_cos = cos[positions].unsqueeze(2)
    selected_sin = sin[positions].unsqueeze(2)
    q_first, q_second = q[..., :4], q[..., 4:]
    k_first, k_second = k[..., :4], k[..., 4:]
    q_ref = torch.cat(
        (q_first * selected_cos - q_second * selected_sin,
         q_first * selected_sin + q_second * selected_cos),
        dim=-1,
    )
    k_ref = torch.cat(
        (k_first * selected_cos - k_second * selected_sin,
         k_first * selected_sin + k_second * selected_cos),
        dim=-1,
    )
    torch.testing.assert_close(q_out, q_ref)
    torch.testing.assert_close(k_out, k_ref)


@pytest.mark.skipif(
    not hasattr(torch, "mlu") or not torch.mlu.is_available(), reason="requires MLU"
)
def test_rotary_embedding_mlu_backend():
    device = torch.device("mlu")
    q = torch.randn(1, 4, 2, 16, device=device, dtype=torch.float16)
    k = torch.randn(1, 4, 1, 16, device=device, dtype=torch.float16)
    cos = torch.randn(16, 8, device=device, dtype=torch.float16)
    sin = torch.randn(16, 8, device=device, dtype=torch.float16)
    positions = torch.arange(4, device=device).view(1, 4)

    q_out, k_out = rotary_embedding(q, k, cos, sin, positions)
    q_ref, k_ref = rotary_embedding(
        q.cpu(), k.cpu(), cos.cpu(), sin.cpu(), positions.cpu()
    )
    torch.testing.assert_close(q_out.cpu(), q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out.cpu(), k_ref, atol=2e-2, rtol=2e-2)
