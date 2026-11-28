# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Pytest-native InfLLM-V2 correctness suite.

The suite is independent of the benchmark. Operator marks are the stable
selection interface:

    pytest tests/flag_attn/test_infllmv2_attention.py -m infllmv2_attention
    pytest tests/flag_attn/test_infllmv2_attention.py -m infllmv2_decode
"""

from __future__ import annotations


import importlib.util
import sys

import pytest
import torch

from flag_attn.testing import backend as test_backend
from flag_attn import (
    InfLLMV2Config,
    infllmv2_attention,
    infllmv2_decode,
    infllmv2_compress_k as compress_k,
    infllmv2_pool_scores as pool_scores,
    infllmv2_select_blocks as select_blocks,
    infllmv2_sparse_attention as sparse_attention,
    infllmv2_stage1 as stage1,
)
from flag_attn.infllmv2.reference import (
    compress_k_ref,
    dense_attention_ref,
    pool_scores_ref,
    select_blocks_ref,
    sparse_attention_ref,
    stage1_ref,
)


HAS_GPU = (test_backend.supports_operator("infllmv2_attention") or test_backend.supports_operator("infllmv2_decode")) and importlib.util.find_spec("triton") is not None
pytestmark = [
    pytest.mark.skipif(not HAS_GPU, reason="requires an available accelerator"),
]


def _cu(lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=test_backend.device,
        dtype=torch.int32,
    )


def _packed_inputs(
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    *,
    d: int = 128,
    hq: int = 32,
    hkv: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = False,
    seed: int = 17,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(q_lengths) == len(kv_lengths)
    assert all(0 < q_len <= kv_len for q_len, kv_len in zip(q_lengths, kv_lengths))
    generator = torch.Generator(device=test_backend.device)
    generator.manual_seed(seed)
    q = torch.randn(
        (sum(q_lengths), hq, d), generator=generator, device=test_backend.device, dtype=dtype
    ).mul_(0.2).requires_grad_(requires_grad)
    k = torch.randn(
        (sum(kv_lengths), hkv, d), generator=generator, device=test_backend.device, dtype=dtype
    ).mul_(0.2).requires_grad_(requires_grad)
    v = torch.randn(
        (sum(kv_lengths), hkv, d), generator=generator, device=test_backend.device, dtype=dtype
    ).mul_(0.2).requires_grad_(requires_grad)
    return q, k, v, _cu(q_lengths), _cu(kv_lengths)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
    config: InfLLMV2Config,
) -> torch.Tensor:
    if max_k < config.dense_len:
        return dense_attention_ref(q, k, v, cu_q, cu_k, causal=config.causal)
    with torch.no_grad():
        k1, cu_k1 = compress_k_ref(k, cu_k, config.k1_kernel_size, config.k1_stride)
        k2, cu_k2 = compress_k_ref(k, cu_k, config.k2_kernel_size, config.k2_stride)
        score = stage1_ref(
            q,
            k1,
            k2,
            cu_q,
            cu_k1,
            cu_k2,
            config.k1_stride,
            config.k2_stride,
            causal=config.causal,
            cu_seqlens_k=cu_k,
        )
        pooled = pool_scores_ref(
            score,
            cu_q,
            cu_k1,
            max_q,
            config.k1_kernel_size,
            config.k1_stride,
            config.block_size,
            config.init_blocks,
            config.local_blocks,
            cu_k,
            max_k,
        )
        selected = select_blocks_ref(
            pooled, cu_q, config.topk, config.block_size, cu_k
        )
    return sparse_attention_ref(
        q,
        k,
        v,
        selected,
        cu_q,
        cu_k,
        config.block_size,
        causal=config.causal,
    )


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
def test_selector_pipeline_matches_reference(d: int) -> None:
    q, k, _, cu_q, cu_k = _packed_inputs((96, 128), (192, 256), d=d)
    config = InfLLMV2Config(topk=4, local_blocks=2, dense_len=0)

    k1, cu_k1 = compress_k(k, cu_k, config.k1_kernel_size, config.k1_stride)
    k2, cu_k2 = compress_k(k, cu_k, config.k2_kernel_size, config.k2_stride)
    ref_k1, ref_cu_k1 = compress_k_ref(k, cu_k, config.k1_kernel_size, config.k1_stride)
    ref_k2, ref_cu_k2 = compress_k_ref(k, cu_k, config.k2_kernel_size, config.k2_stride)
    torch.testing.assert_close(cu_k1, ref_cu_k1)
    torch.testing.assert_close(cu_k2, ref_cu_k2)
    torch.testing.assert_close(k1, ref_k1, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(k2, ref_k2, atol=2e-3, rtol=2e-3)

    score = stage1(
        q, k1, k2, cu_q, cu_k1, cu_k2, 128, cu_seqlens_k=cu_k
    )
    ref_score = stage1_ref(
        q, k1, k2, cu_q, cu_k1, cu_k2, cu_seqlens_k=cu_k
    )
    torch.testing.assert_close(score, ref_score, atol=2e-2, rtol=2e-2)

    pooled = pool_scores(
        score,
        cu_q,
        cu_k1,
        128,
        local_blocks=2,
        cu_seqlens_k=cu_k,
        max_seqlen_k=256,
    )
    ref_pooled = pool_scores_ref(
        score,
        cu_q,
        cu_k1,
        128,
        local_blocks=2,
        cu_seqlens_k=cu_k,
        max_seqlen_k=256,
    )
    torch.testing.assert_close(pooled, ref_pooled, atol=0, rtol=0, equal_nan=True)
    actual_ids = select_blocks(pooled, cu_q, 4, cu_seqlens_k=cu_k)
    expected_ids = select_blocks_ref(ref_pooled, cu_q, 4, cu_seqlens_k=cu_k)
    torch.testing.assert_close(actual_ids, expected_ids)


ATTENTION_CASES = (
    pytest.param((192,), (192,), 64, torch.bfloat16, id="full-d64-bf16"),
    pytest.param((192,), (192,), 128, torch.bfloat16, id="full-d128-bf16"),
    pytest.param((64,), (256,), 64, torch.bfloat16, id="prefix-d64-bf16"),
    pytest.param((128,), (512,), 128, torch.bfloat16, id="chunk-d128-bf16"),
    pytest.param((64, 96), (128, 192), 64, torch.float16, id="varlen-d64-fp16"),
    pytest.param((64, 96), (128, 192), 128, torch.float16, id="varlen-d128-fp16"),
)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("q_lengths,kv_lengths,d,dtype", ATTENTION_CASES)
@pytest.mark.skipif(not test_backend.supports_operator("infllmv2_attention"), reason=test_backend.skip_reason("infllmv2_attention"))
def test_attention_full_prefix_chunk_varlen(
    q_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    d: int,
    dtype: torch.dtype,
) -> None:
    q, k, v, cu_q, cu_k = _packed_inputs(
        q_lengths, kv_lengths, d=d, dtype=dtype
    )
    max_q, max_k = max(q_lengths), max(kv_lengths)
    config = InfLLMV2Config(topk=4, local_blocks=2, dense_len=0)
    actual = infllmv2_attention(
        q, k, v, cu_q, cu_k, max_q, max_k, config=config
    )
    expected = _reference_attention(q, k, v, cu_q, cu_k, max_q, max_k, config)
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
@pytest.mark.skipif(not test_backend.supports_operator("infllmv2_attention"), reason=test_backend.skip_reason("infllmv2_attention"))
def test_attention_dense_path(d: int) -> None:
    q, k, v, cu_q, cu_k = _packed_inputs((96, 128), (128, 192), d=d)
    config = InfLLMV2Config(dense_len=8192)
    actual = infllmv2_attention(
        q, k, v, cu_q, cu_k, 128, 192, config=config
    )
    expected = dense_attention_ref(q, k, v, cu_q, cu_k)
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.skipif(not test_backend.supports_operator("infllmv2_attention"), reason=test_backend.skip_reason("infllmv2_attention"))
def test_attention_sparse_backward(dtype: torch.dtype) -> None:
    q, k, v, cu_q, cu_k = _packed_inputs(
        (64,), (128,), d=64, hq=8, hkv=2, dtype=dtype, requires_grad=True
    )
    q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    config = InfLLMV2Config(topk=4, local_blocks=2, dense_len=0)
    actual = infllmv2_attention(
        q, k, v, cu_q, cu_k, 64, 128, config=config
    )
    expected = _reference_attention(
        q_ref, k_ref, v_ref, cu_q, cu_k, 64, 128, config
    )
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=9e-2, rtol=9e-2)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=7e-2, rtol=7e-2)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
@pytest.mark.skipif(not test_backend.supports_operator("infllmv2_decode"), reason=test_backend.skip_reason("infllmv2_decode"))
def test_decode_varlen_matches_reference(d: int, dense: bool) -> None:
    kv_lengths = (192, 256)
    q, k, v, cu_q, cu_k = _packed_inputs((1, 1), kv_lengths, d=d, seed=29)
    config = InfLLMV2Config(
        topk=4,
        local_blocks=2,
        dense_len=8192 if dense else 0,
    )
    actual = infllmv2_attention(q, k, v, cu_k, max(kv_lengths), stage="decode", config=config)
    expected = _reference_attention(
        q, k, v, cu_q, cu_k, 1, max(kv_lengths), config
    )
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@pytest.mark.infllmv2_attention
def test_wide_topk_tle_or_standard_matches_streaming() -> None:
    generator = torch.Generator(device=test_backend.device)
    generator.manual_seed(41)
    max_q, max_blocks = 1024, 257
    score = torch.randn(
        (2, max_q, max_blocks),
        generator=generator,
        device=test_backend.device,
        dtype=torch.bfloat16,
    )
    cu_q = torch.tensor([0, max_q], device=test_backend.device, dtype=torch.int32)
    cu_k = torch.tensor([0, max_blocks * 64], device=test_backend.device, dtype=torch.int32)
    expected = select_blocks(score, cu_q, 64, cu_seqlens_k=cu_k, _allow_tle=False)
    actual = select_blocks(score, cu_q, 64, cu_seqlens_k=cu_k)
    torch.testing.assert_close(actual, expected)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
def test_stage1_dispatch_matches_standard(d: int) -> None:
    q, k, _, cu_q, cu_k = _packed_inputs((1024,), (8192,), d=d, seed=37)
    k1, cu_k1 = compress_k(k, cu_k, 32, 16)
    k2, cu_k2 = compress_k(k, cu_k, 128, 64)
    # max_seqlen_q is an upper bound: this also checks masked trailing CTAs
    # in the long-prefill dispatch without building a quadratic reference.
    expected = stage1(
        q, k1, k2, cu_q, cu_k1, cu_k2, 65536, cu_seqlens_k=cu_k, _allow_tle=False
    )
    actual = stage1(q, k1, k2, cu_q, cu_k1, cu_k2, 65536, cu_seqlens_k=cu_k)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.infllmv2_attention
def test_stage2_dispatch_matches_standard() -> None:
    q, k, v, cu_q, cu_k = _packed_inputs((1024,), (8192,), d=128, seed=23)
    ids = torch.arange(64, device=test_backend.device, dtype=torch.int32)
    selected = ids.view(1, 1, 64).expand(2, 1024, 64).contiguous()
    expected = sparse_attention(
        q, k, v, selected, cu_q, cu_k, 1024, _allow_tle=False
    )
    actual = sparse_attention(q, k, v, selected, cu_q, cu_k, 1024)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("capability", [None, (8, 0), (10, 0)], ids=["force-generic", "ampere", "blackwell"])
def test_public_fallback_avoids_hopper(monkeypatch, capability) -> None:
    q, k, v, cu_q, cu_k = _packed_inputs((1024,), (1024,), d=128)
    if capability is None:
        monkeypatch.setenv("FLAG_ATTN_FORCE_GENERIC", "1")
    elif test_backend.is_nvidia():
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    calls = []

    def trace(frame, event, arg):
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            if ".hopper.infllmv2" in module:
                calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        actual = infllmv2_attention(
            q, k, v, cu_q, cu_k, 1024, 1024,
            config=InfLLMV2Config(topk=16, dense_len=0),
        )
    finally:
        sys.setprofile(previous)
    assert not calls
    expected = dense_attention_ref(q, k, v, cu_q, cu_k)
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@pytest.mark.infllmv2_attention
@pytest.mark.parametrize("max_q", [1, 32, 96])
def test_portable_dense_backward(monkeypatch, max_q) -> None:
    monkeypatch.setenv("FLAG_ATTN_FORCE_GENERIC", "1")
    q, k, v, cu_q, cu_k = _packed_inputs(
        (max_q, max_q + 1), (128, 193), d=64, hq=8, hkv=2, requires_grad=True,
    )
    qr, kr, vr = [x.detach().clone().requires_grad_(True) for x in (q, k, v)]
    actual = infllmv2_attention(q, k, v, cu_q, cu_k, max_q + 1, 193)
    expected = dense_attention_ref(qr, kr, vr, cu_q, cu_k)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for tensor, ref in zip((q, k, v), (qr, kr, vr)):
        torch.testing.assert_close(tensor.grad, ref.grad, atol=7e-2, rtol=7e-2)


@pytest.mark.infllmv2_attention
def test_direct_decode_matches_stage_facade() -> None:
    q, k, v, _, cu_k = _packed_inputs((1, 1), (31, 129), d=64)
    config = InfLLMV2Config(topk=4, dense_len=0)
    direct = infllmv2_decode(q, k, v, cu_k, 129, config=config)
    staged = infllmv2_attention(q, k, v, cu_k, 129, stage="decode", config=config)
    torch.testing.assert_close(direct, staged)


@pytest.mark.infllmv2_attention
def test_invalid_attention_input_is_rejected() -> None:
    q, k, v, cu_q, cu_k = _packed_inputs((32,), (64,), d=64)
    with pytest.raises(ValueError, match="maximum sequence lengths"):
        infllmv2_attention(q, k, v, cu_q, cu_k, 0, 64)


@pytest.mark.infllmv2_attention
@pytest.mark.skipif(not test_backend.supports_operator("infllmv2_decode"), reason=test_backend.skip_reason("infllmv2_decode"))
def test_invalid_decode_batch_is_rejected() -> None:
    _, k, v, _, cu_k = _packed_inputs((1, 1), (32, 64), d=64)
    bad_q = torch.empty((1, 32, 64), device=test_backend.device, dtype=k.dtype)
    with pytest.raises(ValueError, match="decode q"):
        infllmv2_attention(bad_q, k, v, cu_k, 64, stage="decode")
