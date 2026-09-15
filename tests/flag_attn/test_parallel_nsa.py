# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import pytest
import torch
import triton

torch_gcu = pytest.importorskip("torch_gcu")

from flag_attn.runtime.backend._enflame.nsa.index import (
    prepare_token_indices_enflame as prepare_token_indices,
)
from flag_attn.runtime.backend._enflame.nsa import parallel_nsa

def _device():
    index = int(
        os.environ.get("S60_TEST_DEVICE", "0")
    )
    torch.gcu.set_device(index)
    return torch.device(f"gcu:{index}")


logger = logging.getLogger(__name__)

# BF16 has lower mantissa precision than the upstream FP16 test path.
BF16_ERROR_RATIO = 0.01


# ===========================================================================
# Naive reference implementation (from flash-linear-attention)
# ===========================================================================


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Naive PyTorch reference for NSA selected-sparse attention.

    Args:
        q: queries of shape ``[B, T, HQ, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        block_indices: block indices of shape ``[B, T, H, S]``.
        block_size: selected block size. Default: 64.
        scale: scale factor. If None, defaults to ``K ** -0.5``.
        cu_seqlens: cumulative sequence lengths for variable-length sequences.

    Returns:
        o: outputs of shape ``[B, T, HQ, V]``.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning("head_first has been removed.")
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    k, v, block_indices = (
        x.repeat_interleave(G, dim=2)
        for x in (k, v, block_indices)
    )
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat(
            [
                block_indices.new_tensor(range(0, B * T, T)),
                block_indices.new_tensor([B * T]),
            ]
        )

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, i_b = map(
                lambda x: x[0][cu_seqlens[i] : cu_seqlens[i + 1]],
                (q, k, v, block_indices),
            )

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            q_i = q_b[i_q] * scale
            i_i = i_b[i_q]
            k_i, v_i = map(
                lambda x: x.gather(
                    0, i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])
                ),
                (k_b, v_b),
            )
            attn = (
                torch.einsum("h d, n h d -> n h", q_i, k_i)
                .masked_fill(i_i > i_q, float("-inf"))
                .softmax(0)
            )
            if not varlen:
                o[i, i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)
            else:
                o[0][cu_seqlens[i] + i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)

    return o.to(dtype)


# ===========================================================================
# Testing utilities (from fla.utils._testing)
# ===========================================================================


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    logger.info(msg)
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    if warning:
        if error_rate > ratio:
            logger.warning(msg)
    else:
        assert error_rate < ratio, msg


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("B", "T", "H", "HQ", "D", "S", "block_size", "scale", "dtype"),
    [
        pytest.param(
            *test, id="B{}-T{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-{}".format(*test)
        )
        for test in [
            (1, 63, 1, 16, 64, 16, 32, 1.0, torch.bfloat16),
            (3, 111, 1, 32, 100, 16, 32, 1.0, torch.bfloat16),
            (3, 1024, 2, 32, 60, 16, 32, 0.1, torch.bfloat16),
            # Covers D128 and all 16 selected blocks while
            # keeping the naive Autograd reference below
            # the S60 device-memory limit.
            (1, 511, 2, 32, 128, 16, 32, 0.1, torch.bfloat16),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    """Compare S60 parallel_nsa with the naive reference (forward + backward)."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = _device()

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    ref = naive_nsa(
        q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        scale=scale,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, BF16_ERROR_RATIO)
    assert_close("dq", ref_dq, tri_dq, BF16_ERROR_RATIO)
    assert_close("dk", ref_dk, tri_dk, BF16_ERROR_RATIO)
    assert_close("dv", ref_dv, tri_dv, BF16_ERROR_RATIO)


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("B", "T", "H", "HQ", "D", "S", "block_size", "scale", "dtype"),
    [
        pytest.param(
            *test,
            id=(
                "B{}-T{}-H{}-HQ{}-D{}-S{}-"
                "block_size{}-scale{}-{}"
            ).format(*test),
        )
        for test in [
            (3, 1024, 2, 32, 128, 16, 32, 0.1, torch.bfloat16),
            (4, 2048, 2, 32, 64, 16, 32, 0.1, torch.bfloat16),
        ]
    ],
)
def test_parallel_large_smoke(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    """Exercise large S60 NSA forward and backward configurations.

    The naive per-query Autograd reference retains every selected
    K/V gather until backward and exceeds S60 device memory for
    these shapes. Numerical coverage for D128 and S=16 is provided
    by ``test_parallel``; this test preserves the original large
    production-shape coverage without the infeasible reference.
    """
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = _device()

    q = torch.randn(
        (B, T, HQ, D),
        dtype=dtype,
        device=device,
    ).requires_grad_(True)
    k = torch.randn(
        (B, T, H, D),
        dtype=dtype,
        device=device,
    ).requires_grad_(True)
    v = torch.randn(
        (B, T, H, D),
        dtype=dtype,
        device=device,
    ).requires_grad_(True)
    do = torch.randn(
        (B, T, HQ, D),
        dtype=dtype,
        device=device,
    )

    block_indices = torch.full(
        (B, T, H, S),
        T,
        dtype=torch.long,
        device=device,
    )

    for b in range(B):
        for t in range(T):
            for h in range(H):
                selected = torch.randperm(
                    max(
                        1,
                        triton.cdiv(
                            t,
                            block_size,
                        ),
                    )
                )[:S]

                block_indices[
                    b,
                    t,
                    h,
                    :len(selected),
                ] = selected

    block_indices = block_indices.sort(-1)[0]

    output = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        scale=scale,
    )

    output.backward(do)
    torch.gcu.synchronize()

    tensors = (
        ("output", output),
        ("dq", q.grad),
        ("dk", k.grad),
        ("dv", v.grad),
    )

    for name, tensor in tensors:
        assert tensor is not None, (
            f"{name} was not produced"
        )
        assert torch.isfinite(
            tensor.float()
        ).all(), (
            f"{name} contains non-finite values"
        )


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("H", "HQ", "D", "S", "block_size", "cu_seqlens", "dtype"),
    [
        pytest.param(
            *test, id="H{}-HQ{}-D{}-S{}-block_size{}-cu_seqlens{}-{}".format(*test)
        )
        for test in [
            (1, 16, 64, 16, 32, [0, 15], torch.bfloat16),
            (2, 32, 64, 16, 32, [0, 256, 500, 1000], torch.bfloat16),
            (2, 32, 100, 16, 32, [0, 15, 100, 300, 1200, 2000], torch.bfloat16),
        ]
    ],
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Compare S60 parallel_nsa with naive reference for variable-length sequences."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = _device()

    T = cu_seqlens[-1]
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((1, T, H, S), T, dtype=torch.long, device=device)
    seq_indices = prepare_token_indices(cu_seqlens_t).tolist()

    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[0, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_size=block_size,
        cu_seqlens=cu_seqlens_t,
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        cu_seqlens=cu_seqlens_t,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close("o", ref, tri, 0.004)
    assert_close("dq", ref_dq, tri_dq, BF16_ERROR_RATIO)
    assert_close("dk", ref_dk, tri_dk, BF16_ERROR_RATIO)
    assert_close("dv", ref_dv, tri_dv, BF16_ERROR_RATIO)
