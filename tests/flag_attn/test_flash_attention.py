import pytest
import torch

import flag_attn

torch.random.manual_seed(10086)


def max_diff(a, b):
    return (a - b).abs().max().item()


def zero_percent(a, b):
    diff = (a - b).abs()
    num_non_zeros = diff.nonzero().shape[0]
    return (1.0 - num_non_zeros / diff.numel()) * 100.0


def report(name, actual, expected):
    print(
        f"{name}: \tmax_difference: {max_diff(actual, expected):0.6f}\t"
        f"zero_diff elements: {zero_percent(actual, expected):0.3f}%"
    )


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "B, Hq, Hk, M, N, D",
    [
        (2, 4, 4, 512, 612, 128),
        (2, 4, 4, 1024, 1034, 64),
        (2, 4, 4, 2048, 2048, 32),
        (2, 4, 4, 4096, 4096, 16),
        (2, 4, 4, 4001, 4001, 32),
        (2, 4, 4, 4001, 4096, 64),
        (2, 4, 4, 4096, 4000, 128),
        (1, 2, 2, 8192, 8202, 16),
        (1, 2, 2, 8192, 8192, 32),
        # test for mqa/gqa
        (2, 4, 2, 512, 612, 128),
        (2, 4, 1, 1024, 1034, 64),
        (2, 4, 2, 2048, 2048, 32),
        (2, 4, 1, 4096, 4096, 16),
        (2, 4, 2, 4001, 4001, 32),
        (2, 4, 1, 4001, 4096, 64),
        (2, 4, 2, 4096, 4000, 128),
        (1, 2, 1, 8192, 8202, 16),
        (1, 2, 1, 8192, 8192, 32),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
def test_attention_fwd(
    B, Hq, Hk, M, N, D, causal, stride_order, dtype, scale, device_id
):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q = torch.empty((B, Hq, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        k = torch.empty((B, Hk, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        v = torch.empty((B, Hk, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
    else:
        q = (
            torch.empty((B, M, Hq, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        k = (
            torch.empty((B, N, Hk, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        v = (
            torch.empty((B, N, Hk, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )

    o_ref = flag_attn.testing.flash_attention(q, k, v, causal, upcast=True)
    o_torch = flag_attn.testing.flash_attention(q, k, v, causal, upcast=False)
    o_hyp = flag_attn.flash_attention(q, k, v, causal)

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    report("o hyp", o_hyp, o_ref)
    report("o torch", o_hyp, o_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [10.0])
@pytest.mark.parametrize(
    "B, Hq, Hk, M, N, D",
    [
        (2, 4, 4, 1, 612, 128),
        (2, 4, 4, 1, 1034, 64),
        (2, 4, 4, 1, 2048, 32),
        (2, 4, 4, 1, 4096, 16),
        (2, 4, 4, 1, 4001, 32),
        (2, 4, 4, 1, 4096, 64),
        (2, 4, 4, 2, 4000, 128),
        (1, 2, 2, 4, 8202, 16),
        (1, 2, 2, 1, 8192, 32),
        # test for mqa/gqa
        (2, 4, 2, 1, 612, 128),
        (2, 4, 1, 1, 1034, 64),
        (2, 4, 2, 1, 2048, 32),
        (2, 4, 1, 1, 4096, 16),
        (2, 4, 2, 1, 4001, 32),
        (2, 4, 1, 1, 4096, 64),
        (2, 4, 2, 2, 4000, 128),
        (1, 2, 1, 4, 8202, 16),
        (1, 2, 1, 1, 8192, 32),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
def test_attention_splitkv(
    B, Hq, Hk, M, N, D, causal, stride_order, dtype, scale, device_id
):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q = torch.empty((B, Hq, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        k = torch.empty((B, Hk, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        v = torch.empty((B, Hk, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
    else:
        q = (
            torch.empty((B, M, Hq, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        k = (
            torch.empty((B, N, Hk, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        v = (
            torch.empty((B, N, Hk, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )

    o_ref = flag_attn.testing.flash_attention(q, k, v, causal, upcast=True)
    o_torch = flag_attn.testing.flash_attention(q, k, v, causal, upcast=False)
    o_hyp = flag_attn.flash_attention(q, k, v, causal)

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    report("o hyp", o_hyp, o_ref)
    report("o torch", o_hyp, o_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "B, Hq, Hk, M, N, D",
    [
        (2, 4, 4, 512, 612, 128),
        (2, 4, 4, 1024, 1034, 64),
        (2, 4, 4, 2048, 2048, 32),
        (2, 4, 4, 4096, 4096, 16),
        (2, 4, 4, 4001, 4001, 32),
        (2, 4, 4, 4001, 4096, 64),
        (2, 4, 4, 4096, 4001, 128),
        (1, 2, 2, 8192, 8202, 16),
        (1, 2, 2, 8192, 8192, 32),
        (2, 4, 4, 10006, 10, 128),
        # test for mqa/gqa
        (2, 4, 2, 512, 612, 128),
        (2, 4, 1, 1024, 1034, 64),
        (2, 4, 2, 2048, 2048, 32),
        (2, 4, 1, 4096, 4096, 16),
        (2, 4, 2, 4001, 4001, 32),
        (2, 4, 1, 4001, 4096, 64),
        (2, 4, 2, 4096, 4001, 128),
        (1, 2, 1, 8192, 8202, 16),
        (1, 2, 1, 8192, 8192, 32),
        (2, 4, 2, 10006, 10, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
def test_attention_bwd(
    B, Hq, Hk, M, N, D, causal, stride_order, dtype, scale, device_id
):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q = (
            torch.empty((B, Hq, M, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        k = (
            torch.empty((B, Hk, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        v = (
            torch.empty((B, Hk, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        do = torch.randn((B, Hq, M, D), dtype=dtype, device=device)
    else:
        q = (
            torch.empty((B, M, Hq, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        k = (
            torch.empty((B, N, Hk, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        v = (
            torch.empty((B, N, Hk, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        do = torch.randn((B, M, Hq, D), dtype=dtype, device=device).transpose(1, 2)

    o_ref = flag_attn.testing.flash_attention(q, k, v, causal=causal, upcast=True)
    o_torch = flag_attn.testing.flash_attention(q, k, v, causal=causal, upcast=False)
    o_hyp = flag_attn.flash_attention(q, k, v, causal=causal)

    gq_ref, gk_ref, gv_ref = torch.autograd.grad(o_ref, (q, k, v), do)
    gq_torch, gk_torch, gv_torch = torch.autograd.grad(o_torch, (q, k, v), do)
    gq_hyp, gk_hyp, gv_hyp = torch.autograd.grad(o_hyp, (q, k, v), do)

    o_torch_max_diff = max_diff(o_torch, o_ref)
    gq_torch_max_diff = max_diff(gq_torch, gq_ref)
    gk_torch_max_diff = max_diff(gk_torch, gk_ref)
    gv_torch_max_diff = max_diff(gv_torch, gv_ref)

    o_triton_max_diff = max_diff(o_hyp, o_ref)
    gq_triton_max_diff = max_diff(gq_hyp, gq_ref)
    gk_triton_max_diff = max_diff(gk_hyp, gk_ref)
    gv_triton_max_diff = max_diff(gv_hyp, gv_ref)

    assert o_triton_max_diff < 2 * o_torch_max_diff + 1e-5
    assert gq_triton_max_diff < 2 * gq_torch_max_diff + 1e-5
    assert gk_triton_max_diff < 2 * gk_torch_max_diff + 1e-5
    assert gv_triton_max_diff < 2 * gv_torch_max_diff + 1e-5


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize(
    "B, H, M, N, D",
    [
        (2, 4, 512, 612, 128),
        (2, 4, 1024, 1034, 64),
        (2, 4, 2048, 2048, 32),
        (2, 4, 4096, 4096, 16),
        (2, 4, 4001, 4001, 32),
        (2, 4, 4001, 4096, 64),
        (2, 4, 4096, 4001, 128),
        (1, 2, 8192, 8202, 16),
        (1, 2, 8192, 8192, 32),
    ],
)
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
def test_attention_with_aux_outs(
    B, H, M, N, D, causal, stride_order, dtype, scale, device_id
):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q = torch.empty((B, H, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        k = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        v = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
    else:
        q = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        k = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        v = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )

    o_ref, log_norm_ref, tot_attn_ref = flag_attn.testing.flash_attention(
        q,
        k,
        v,
        causal,
        return_log_normalizer=True,
        return_total_attention=True,
        upcast=True,
    )
    o_torch, log_norm_torch, tot_attn_torch = flag_attn.testing.flash_attention(
        q,
        k,
        v,
        causal,
        return_log_normalizer=True,
        return_total_attention=True,
        upcast=False,
    )
    o_hyp, log_norm_hyp, tot_attn_hyp, *_ = flag_attn.flash_attention(
        q, k, v, causal, return_log_normalizer=True, return_total_attention=True
    )

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5

    torch_max_diff = max_diff(log_norm_torch, log_norm_ref)
    triton_max_diff = max_diff(log_norm_hyp, log_norm_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5

    torch_max_diff = max_diff(tot_attn_torch, tot_attn_ref)
    triton_max_diff = max_diff(tot_attn_hyp, tot_attn_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5


@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize(
    "B, H, M, N, D",
    [
        (2, 4, 512, 612, 128),
        (2, 4, 1024, 1034, 64),
        (2, 4, 2048, 2048, 32),
        (2, 4, 4096, 4096, 16),
        (2, 4, 4001, 4001, 32),
        (2, 4, 4001, 4096, 64),
        (2, 4, 4096, 4000, 128),
        (1, 2, 8192, 8202, 16),
        (1, 2, 8192, 8192, 32),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.5, 0.8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
def test_attention_fwd_dropout(
    B, H, M, N, D, causal, dropout_p, stride_order, dtype, scale, device_id
):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q = torch.empty((B, H, M, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        k = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
        v = torch.empty((B, H, N, D), dtype=dtype, device=device).normal_(
            mean=0.0, std=scale
        )
    else:
        q = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        k = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )
        v = (
            torch.empty((B, N, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
        )

    o_hyp, _, _, seed, offset = flag_attn.flash_attention(
        q, k, v, causal, dropout_p=dropout_p, return_seed_offset=True
    )
    mask = flag_attn.testing.recompute_mask(B, H, M, N, dropout_p, seed, offset, device)
    o_ref = flag_attn.testing.flash_attention(
        q, k, v, causal, dropout_p=dropout_p, dropout_mask=mask, upcast=True
    )
    o_torch = flag_attn.testing.flash_attention(
        q, k, v, causal, dropout_p=dropout_p, dropout_mask=mask, upcast=False
    )

    torch_max_diff = max_diff(o_torch, o_ref)
    triton_max_diff = max_diff(o_hyp, o_ref)
    report("o hyp", o_hyp, o_ref)
    report("o torch", o_torch, o_ref)
    assert triton_max_diff <= 2 * torch_max_diff + 1e-5


# @pytest.mark.parametrize('increment', [random.randint(0, 1000000000) for i in range(100)])
@pytest.mark.parametrize("device_id", list(range(torch.cuda.device_count())))
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.parametrize(
    "B, H, M, N, D",
    [
        (2, 4, 512, 612, 128),
        (2, 4, 1024, 1034, 64),
        (2, 4, 2048, 2048, 32),
        (2, 4, 4096, 4096, 16),
        (2, 4, 4001, 4001, 32),
        (2, 4, 4001, 4096, 64),
        (2, 4, 4096, 4000, 128),
        (1, 2, 8192, 8202, 16),
        (1, 2, 8192, 8192, 32),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dropout_p", [0.5, 0.8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stride_order", ["BHTD", "BTHD"])
def test_attention_bwd_dropout(
    B, H, M, N, D, causal, dropout_p, stride_order, dtype, scale, device_id
):
    device = f"cuda:{device_id}"
    if stride_order == "BHTD":
        q = (
            torch.empty((B, H, M, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        k = (
            torch.empty((B, H, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        v = (
            torch.empty((B, H, N, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .requires_grad_()
        )
        do = torch.randn((B, H, M, D), dtype=dtype, device=device)
    else:
        q = (
            torch.empty((B, M, H, D), dtype=dtype, device=device)
            .normal_(mean=0.0, std=scale)
            .transpose(1, 2)
            .requires_grad_()
        )
        k = (
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
        do = torch.randn((B, M, H, D), dtype=dtype, device=device).transpose(1, 2)

    # from flag_attn.dropout import philox_cuda_seed_offset
    o_hyp, _, _, seed, offset = flag_attn.flash_attention(
        q, k, v, causal=causal, dropout_p=dropout_p, return_seed_offset=True
    )
    mask = flag_attn.testing.recompute_mask(B, H, M, N, dropout_p, seed, offset, device)
    o_ref = flag_attn.testing.flash_attention(
        q, k, v, causal=causal, dropout_p=dropout_p, dropout_mask=mask, upcast=True
    )
    o_torch = flag_attn.testing.flash_attention(
        q, k, v, causal=causal, dropout_p=dropout_p, dropout_mask=mask, upcast=False
    )

    gq_ref, gk_ref, gv_ref = torch.autograd.grad(o_ref, (q, k, v), do)
    gq_torch, gk_torch, gv_torch = torch.autograd.grad(o_torch, (q, k, v), do)
    gq_hyp, gk_hyp, gv_hyp = torch.autograd.grad(o_hyp, (q, k, v), do)

    o_torch_max_diff = max_diff(o_torch, o_ref)
    gq_torch_max_diff = max_diff(gq_torch, gq_ref)
    gk_torch_max_diff = max_diff(gk_torch, gk_ref)
    gv_torch_max_diff = max_diff(gv_torch, gv_ref)

    o_triton_max_diff = max_diff(o_hyp, o_ref)
    gq_triton_max_diff = max_diff(gq_hyp, gq_ref)
    gk_triton_max_diff = max_diff(gk_hyp, gk_ref)
    gv_triton_max_diff = max_diff(gv_hyp, gv_ref)

    assert o_triton_max_diff < 2 * o_torch_max_diff + 1e-5
    assert gq_triton_max_diff < 2 * gq_torch_max_diff + 1e-5
    assert gk_triton_max_diff < 2 * gk_torch_max_diff + 1e-5
    assert gv_triton_max_diff < 2 * gv_torch_max_diff + 1e-5
