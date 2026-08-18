"""Correctness tests for TLE decode, using the CuTe SM90 kernel as baseline."""

from __future__ import annotations

import math

import pytest
import torch

from flag_attn.parallax.tle import HAS_TLE, parallax_attn_with_kvcache, parallax_decode

try:
    from flag_attn.parallax.cute import (
        parallax_attn_with_kvcache as cute_kvcache,
        parallax_decode as cute_decode,
    )
except Exception as exc:
    cute_kvcache = None
    cute_decode = None
    CUTE_IMPORT_ERROR = exc
else:
    CUTE_IMPORT_ERROR = None


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = max(expected.float().abs().max().item(), 1e-6)
    return (actual.float() - expected.float()).abs().max().item() / scale


def _decode_reference(q, r, k, v, scale, window_size_left=-1):
    """FP32 decode oracle using pointwise reductions instead of cuBLAS einsum.

    Some experimental Torch/Triton stacks reject the degenerate Q=1 batched
    SGEMM selected by einsum. This equivalent formulation also makes a delayed
    custom-kernel CUDA error distinguishable from a reference-backend error.
    """
    HQ, H = q.shape[2], k.shape[2]
    repeat = HQ // H
    qf = q[:, 0].permute(0, 1, 2).float()  # (B, HQ, D)
    rf = r[:, 0].permute(0, 1, 2).float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    if repeat > 1:
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    s1 = (qf[:, :, None, :] * kf).sum(dim=-1) * scale
    s2 = (rf[:, :, None, :] * kf).sum(dim=-1)
    if window_size_left >= 0:
        first = max(k.shape[1] - window_size_left, 0)
        valid = torch.arange(k.shape[1], device=k.device) >= first
        s1 = s1.masked_fill(~valid[None, None, :], float("-inf"))
    pivot = s1.amax(dim=-1, keepdim=True)
    pivot_safe = torch.where(torch.isfinite(pivot), pivot, torch.zeros_like(pivot))
    p1 = torch.exp(s1 - pivot_safe)
    p2 = p1 * s2
    d1 = p1.sum(dim=-1, keepdim=True)
    d2 = p2.sum(dim=-1, keepdim=True)
    o1 = (p1[..., None] * vf).sum(dim=2)
    o2 = (p2[..., None] * vf).sum(dim=2)
    inv_d1 = torch.where(d1 > 0, d1.reciprocal(), torch.zeros_like(d1))
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    return result[:, None].contiguous()


def _inputs(B, L, HQ, H, D, dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, 1, HQ, D, device="cuda", dtype=dtype, generator=generator)
    r = torch.randn(B, 1, HQ, D, device="cuda", dtype=dtype, generator=generator) * 0.5
    k = torch.randn(B, L, H, D, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(B, L, H, D, device="cuda", dtype=dtype, generator=generator)
    return q, r, k, v


def _require_tle():
    if not HAS_TLE:
        pytest.skip("FlagTree TLE is unavailable (triton.experimental.tle)")


@pytest.mark.sm90
def test_cute_baseline_importable():
    """Report the exact optional-baseline import failure without hiding TLE tests."""
    _require_tle()
    if cute_decode is None:
        pytest.skip(f"CuTeDSL baseline is unavailable: {CUTE_IMPORT_ERROR!r}")


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B,L,HQ,H,D,window",
    [
        (1, 512, 8, 8, 64, -1),       # short-cache split-KV path
        (1, 512, 8, 8, 128, -1),      # shape-aware BN=64 path
        (1, 2048, 8, 8, 128, -1),     # long-cache split-KV path
        (2, 1000, 8, 2, 128, -1),     # partial tile + GQA
        (2, 2048, 8, 2, 64, 257),     # GQA + unaligned sliding window
    ],
)
def test_tle_decode_matches_reference_and_cute_when_available(
    B, L, HQ, H, D, window, dtype
):
    _require_tle()
    q, r, k, v = _inputs(B, L, HQ, H, D, dtype, seed=B * 1009 + L + D)
    scale = 1.0 / math.sqrt(D)

    actual = parallax_decode(
        q, r, k, v, scale, window_size_left=window
    )
    torch.cuda.synchronize()
    reference = _decode_reference(q, r, k, v, scale, window)
    torch.cuda.synchronize()

    assert not torch.isnan(actual).any()
    assert _rel_err(actual, reference) < 1e-2
    if cute_decode is not None:
        baseline = cute_decode(
            q, r, k, v, scale, window_size_left=window
        )
        torch.cuda.synchronize()
        assert _rel_err(actual, baseline) < 1e-2


@pytest.mark.sm90
def test_tle_runtime_seqused_k_matches_cute():
    """Exercise the canonical serving API with per-batch active cache lengths."""
    _require_tle()

    B, L, HQ, H, D = 2, 2048, 8, 2, 128
    q, r, k, v = _inputs(B, L, HQ, H, D, torch.bfloat16, seed=123)
    seqlens = torch.tensor([777, 1901], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)

    actual = parallax_attn_with_kvcache(
        q,
        r,
        k,
        v,
        seqused_k=seqlens,
        window_size=(257, 0),
        scale=scale,
        out=out,
    )
    torch.cuda.synchronize()
    references = []
    for batch_index, active_len in enumerate(seqlens.tolist()):
        references.append(
            _decode_reference(
                q[batch_index:batch_index + 1],
                r[batch_index:batch_index + 1],
                k[batch_index:batch_index + 1, :active_len],
                v[batch_index:batch_index + 1, :active_len],
                scale,
                window_size_left=257,
            )
        )
    reference = torch.cat(references, dim=0)

    assert actual is out
    assert _rel_err(actual, reference) < 1e-2
    if cute_kvcache is not None:
        baseline = cute_kvcache(
            q,
            r,
            k,
            v,
            seqused_k=seqlens,
            window_size=(257, 0),
            scale=scale,
        )
        torch.cuda.synchronize()
        assert _rel_err(actual, baseline) < 1e-2


@pytest.mark.sm90
def test_tle_noncontiguous_decode_inputs():
    """Decode commonly receives q[:, -1:], which need not be contiguous."""
    _require_tle()
    B, L, H, D = 2, 1024, 8, 128
    generator = torch.Generator(device="cuda").manual_seed(7)
    q_full = torch.randn(B, 3, H, D, device="cuda", dtype=torch.bfloat16,
                         generator=generator)
    r_full = torch.randn(B, 3, H, D, device="cuda", dtype=torch.bfloat16,
                         generator=generator)
    k = torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16,
                    generator=generator)
    v = torch.randn_like(k)
    q = q_full[:, -1:]
    r = r_full[:, -1:]
    assert not q.is_contiguous()

    actual = parallax_decode(q, r, k, v, D ** -0.5)
    torch.cuda.synchronize()
    reference = _decode_reference(q, r, k, v, D ** -0.5)
    assert _rel_err(actual, reference) < 1e-2
    if cute_decode is not None:
        # CuTe's public contract requires contiguous q/r, whereas the TLE
        # wrapper deliberately accepts the common q[:, -1:] decode view and
        # materializes it internally. Compare values under each backend's
        # documented input contract.
        baseline = cute_decode(
            q.contiguous(), r.contiguous(), k, v, D ** -0.5
        )
        torch.cuda.synchronize()
        assert _rel_err(actual, baseline) < 1e-2
