import logging

import pytest
import torch

import flag_attn

torch.random.manual_seed(10086)


def max_diff(a, b):
    return (a - b).abs().max().item()


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "B, H, M, N, D",
    [
        (2, 4, 512, 612, 128),
        (2, 4, 1024, 1034, 64),
        (2, 4, 2048, 2048, 32),
        (2, 4, 4096, 4096, 16),
        (2, 4, 4096, 4001, 16),
        (1, 2, 8192, 8192, 16),
        (1, 2, 8192, 8192, 32),
        (1, 2, 8192, 3867, 32),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_fwd(B, H, M, N, D, causal, stride_order, dtype, scale, device_id):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q1 = torch.empty((B, H, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        q2 = torch.empty((B, H, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        k1 = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        k2 = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        v = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
    else:
        q1 = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        q2 = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        k1 = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        k2 = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        v = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
    w = (M // 2) if M < N else (M - N // 2)

    o_ref = flag_attn.testing.piecewise_attention(
        q1, k1, q2, k2, v, w, causal=causal, upcast=True
    )
    o_torch = flag_attn.testing.piecewise_attention(q1, k1, q2, k2, v, w, causal=causal)
    o_hyp = flag_attn.piecewise_attention(q1, k1, q2, k2, v, w, causal=causal)

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    logging.info(
        "torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            torch_max_diff, triton_max_diff
        )
    )
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5
    # assert torch.testing.assert_close(o_hyp, o_ref)


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "B, H, M, N, D",
    [
        (2, 4, 512, 612, 128),
        (2, 4, 1024, 1034, 64),
        (2, 4, 2048, 2048, 32),
        (2, 4, 4096, 4096, 16),
        (2, 4, 4096, 4001, 16),
        (1, 2, 8192, 8192, 16),
        (1, 2, 8192, 8192, 32),
        (1, 2, 8192, 3867, 32),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_bwd(B, H, M, N, D, causal, stride_order, dtype, scale, device_id):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q1 = (
            torch.empty((B, H, M, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        q2 = (
            torch.empty((B, H, M, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        k1 = (
            torch.empty((B, H, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        k2 = (
            torch.empty((B, H, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        v = (
            torch.empty((B, H, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        do = torch.empty((B, H, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
    else:
        q1 = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        q2 = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        k1 = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        k2 = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        v = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        do = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )

    w = (M // 2) if M < N else (M - N // 2)

    o_ref = flag_attn.testing.piecewise_attention(
        q1, k1, q2, k2, v, w, causal=causal, upcast=True
    )
    dq1_ref, dk1_ref, dq2_ref, dk2_ref, dv_ref = torch.autograd.grad(
        o_ref, (q1, k1, q2, k2, v), do
    )

    o_torch = flag_attn.testing.piecewise_attention(
        q1, k1, q2, k2, v, w, causal=causal, upcast=False
    )
    dq1_torch, dk1_torch, dq2_torch, dk2_torch, dv_torch = torch.autograd.grad(
        o_torch, (q1, k1, q2, k2, v), do
    )

    o_hyp = flag_attn.piecewise_attention(q1, k1, q2, k2, v, w, causal=causal)
    dq1_hyp, dk1_hyp, dq2_hyp, dk2_hyp, dv_hyp = torch.autograd.grad(
        o_hyp, (q1, k1, q2, k2, v), do
    )

    o_torch_max_diff = max_diff(o_torch, o_ref)
    dq1_torch_max_diff = max_diff(dq1_torch, dq1_ref)
    dq2_torch_max_diff = max_diff(dq2_torch, dq2_ref)
    dk1_torch_max_diff = max_diff(dk1_torch, dk1_ref)
    dk2_torch_max_diff = max_diff(dk2_torch, dk2_ref)
    dv_torch_max_diff = max_diff(dv_torch, dv_ref)

    o_triton_max_diff = max_diff(o_hyp, o_ref)
    dq1_triton_max_diff = max_diff(dq1_hyp, dq1_ref)
    dq2_triton_max_diff = max_diff(dq2_hyp, dq2_ref)
    dk1_triton_max_diff = max_diff(dk1_hyp, dk1_ref)
    dk2_triton_max_diff = max_diff(dk2_hyp, dk2_ref)
    dv_triton_max_diff = max_diff(dv_hyp, dv_ref)

    logging.info(
        "o torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            o_torch_max_diff, o_triton_max_diff
        )
    )
    logging.info(
        "dq1 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            dq1_torch_max_diff, dq1_triton_max_diff
        )
    )
    logging.info(
        "dq2 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            dq2_torch_max_diff, dq2_triton_max_diff
        )
    )
    logging.info(
        "dk1 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            dk1_torch_max_diff, dk1_triton_max_diff
        )
    )
    logging.info(
        "dk2 torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            dk2_torch_max_diff, dk2_triton_max_diff
        )
    )
    logging.info(
        "dv torch_max_diff: {:.8f}\ttriton_max_diff: {:.8f}".format(
            dv_torch_max_diff, dv_triton_max_diff
        )
    )

    assert o_triton_max_diff <= 2 * o_torch_max_diff + 1e-5
    assert dq1_triton_max_diff <= 2 * dq1_torch_max_diff + 1e-5
    assert dq2_triton_max_diff <= 2 * dq2_torch_max_diff + 1e-5
    assert dk1_triton_max_diff <= 2 * dk1_torch_max_diff + 1e-5
    assert dk2_triton_max_diff <= 2 * dk2_torch_max_diff + 1e-5
    assert dv_triton_max_diff <= 2 * dv_torch_max_diff + 1e-5
