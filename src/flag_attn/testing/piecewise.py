import math

import pytest
import torch


def attention(q1, k1, q2, k2, v, dist_threshold, causal, sm_scale=None, upcast=False):
    input_dtype = q1.dtype
    if upcast:
        q1, k1, q2, k2, v = q1.float(), k1.float(), q2.float(), k2.float(), v.float()
    # (B, H, T, D)
    D = q1.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    kv_seq_len = k1.shape[-2]
    q_seq_len = q1.shape[-2]
    p_seq = kv_seq_len - q_seq_len
    device = q1.device

    ms = torch.arange(q_seq_len, device=device).unsqueeze(-1)
    ns = torch.arange(kv_seq_len, device=device)

    S1 = torch.matmul(q1, k1.transpose(2, 3))
    S2 = torch.matmul(q2, k2.transpose(2, 3))
    long_distance = (ms + p_seq - ns) >= dist_threshold
    S = torch.where(long_distance, S2, S1) * sm_scale

    if causal:
        S = torch.where(ms + p_seq >= ns, S, torch.finfo(S.dtype).min)

    # upcast attention to fp32
    P = torch.softmax(S, dim=-1, dtype=torch.float32).to(v.dtype)
    if causal:
        P = torch.where(ms + p_seq >= ns, P, 0.0)
    attn_output = torch.matmul(P, v)
    return attn_output.to(input_dtype)


def attention_grad(q1, k1, q2, k2, v, w, causal, sm_scale, o, do, upcast=False):
    input_dtype = q1.dtype

    if upcast:
        q1, k1, q2, k2, v, o, do = [item.float() for item in [q1, k1, q2, k2, v, o, do]]
    kv_seq_len = k1.shape[-2]
    q_seq_len = q1.shape[-2]
    p_seq = kv_seq_len - q_seq_len
    device = q1.device

    ms = torch.arange(q_seq_len, device=device).unsqueeze(-1)
    ns = torch.arange(kv_seq_len, device=device)

    S1 = torch.matmul(q1, k1.transpose(2, 3))
    S2 = torch.matmul(q2, k2.transpose(2, 3))
    long_distance = (ms + p_seq - ns) >= w
    S = torch.where(long_distance, S2, S1) * sm_scale

    if causal:
        S = torch.where((ms + p_seq) >= ns, S, torch.finfo(S.dtype).min)

    # upcast attention to fp32
    P = torch.softmax(S, dim=-1, dtype=torch.float32).to(v.dtype)

    # dP & dv
    dv = torch.matmul(P.transpose(2, 3), do)
    dP = torch.matmul(do, v.transpose(2, 3))

    # dS
    delta = (do * o).sum(-1, keepdim=True)  # (B,H,T)
    dS = P * (dP - delta) * sm_scale
    dS2 = torch.where(long_distance, dS, 0.0)
    dS1 = torch.where(long_distance, 0.0, dS)

    # dq & dk
    dq1 = torch.matmul(dS1, k1)
    dk1 = torch.matmul(dS1.transpose(2, 3), q1)

    dq2 = torch.matmul(dS2, k2)
    dk2 = torch.matmul(dS2.transpose(2, 3), q2)

    dq1, dk1, dq2, dk2, dv = [item.to(input_dtype) for item in [dq1, dk1, dq2, dk2, dv]]
    return dq1, dk1, dq2, dk2, dv


@pytest.mark.parametrize(
    "B, H, T, D, P_SEQ", [(2, 3, 1024, 32, 100), (2, 3, 1024, 32, 0)]
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_op(B, H, T, D, P_SEQ, causal, dtype):
    q1 = (
        torch.empty((B, H, T, D), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    q2 = (
        torch.empty((B, H, T, D), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k1 = (
        torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k2 = (
        torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((B, H, T + P_SEQ, D), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    w = 780

    o = attention(q1, k1, q2, k2, v, w, causal, sm_scale)
    do = torch.empty((B, H, T, D), dtype=dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    o.backward(do)
    dq1, dk1, dq2, dk2, dv = attention_grad(
        q1, k1, q2, k2, v, w, causal, sm_scale, o, do
    )

    torch.testing.assert_close(dv, v.grad, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(dq1, q1.grad, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(dk1, k1.grad, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(dq2, q2.grad, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(dk2, k2.grad, atol=1e-2, rtol=0.0)
