# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Pytest-native InfLLM-V2 correctness suite.

The suite is independent of the benchmark. Operator marks are the stable
selection interface:

    pytest tests/flag_attn/test_infllmv2_attention.py -m infllmv2_attention
    pytest tests/flag_attn/test_infllmv2_attention.py -m infllmv2_decode
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from flag_attn import InfLLMV2Config, infllmv2_attention, infllmv2_decode
from flag_attn.infllmv2 import forward as forward_impl
from flag_attn.infllmv2.forward import (
    _sparse_attention_forward,
    compressed_lengths,
    compress_k,
    pool_scores,
    select_blocks,
    sparse_attention_tle_hopper,
    stage1,
    stage1_tle_hopper,
)
from flag_attn.infllmv2.reference import (
    compress_k_ref,
    dense_attention_ref,
    pool_scores_ref,
    select_blocks_ref,
    sparse_attention_ref,
    stage1_ref,
)


HAS_GPU = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA, PyTorch and Triton are required")
ATTENTION = pytest.mark.infllmv2_attention
DECODE = pytest.mark.infllmv2_decode

HAS_TLE = forward_impl._TLE_AVAILABLE


def _cu(lengths: tuple[int, ...]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
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
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn(
        (sum(q_lengths), hq, d), generator=generator, device="cuda", dtype=dtype
    ).mul_(0.2).requires_grad_(requires_grad)
    k = torch.randn(
        (sum(kv_lengths), hkv, d), generator=generator, device="cuda", dtype=dtype
    ).mul_(0.2).requires_grad_(requires_grad)
    v = torch.randn(
        (sum(kv_lengths), hkv, d), generator=generator, device="cuda", dtype=dtype
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


@ATTENTION
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


@ATTENTION
@pytest.mark.parametrize("q_lengths,kv_lengths,d,dtype", ATTENTION_CASES)
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


@ATTENTION
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
def test_attention_dense_path(d: int) -> None:
    q, k, v, cu_q, cu_k = _packed_inputs((96, 128), (128, 192), d=d)
    config = InfLLMV2Config(dense_len=8192)
    actual = infllmv2_attention(
        q, k, v, cu_q, cu_k, 128, 192, config=config
    )
    expected = dense_attention_ref(q, k, v, cu_q, cu_k)
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@ATTENTION
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
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


@DECODE
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
def test_decode_varlen_matches_reference(d: int, dense: bool) -> None:
    kv_lengths = (192, 256)
    q, k, v, cu_q, cu_k = _packed_inputs((1, 1), kv_lengths, d=d, seed=29)
    config = InfLLMV2Config(
        topk=4,
        local_blocks=2,
        dense_len=8192 if dense else 0,
    )
    actual = infllmv2_decode(q, k, v, cu_k, max(kv_lengths), config=config)
    expected = _reference_attention(
        q, k, v, cu_q, cu_k, 1, max(kv_lengths), config
    )
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@ATTENTION
def test_wide_topk_tle_or_standard_matches_streaming() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(41)
    max_q, max_blocks = 1024, 257
    score = torch.randn(
        (2, max_q, max_blocks),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cu_q = torch.tensor([0, max_q], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, max_blocks * 64], device="cuda", dtype=torch.int32)
    expected = select_blocks(score, cu_q, 64, cu_seqlens_k=cu_k, _allow_tle=False)
    actual = select_blocks(score, cu_q, 64, cu_seqlens_k=cu_k)
    torch.testing.assert_close(actual, expected)


@ATTENTION
@pytest.mark.parametrize("d", [64, 128], ids=lambda value: f"d{value}")
def test_tle_stage1_matches_standard_when_available(d: int) -> None:
    if not HAS_TLE or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("TLE Stage1 requires Hopper")
    q, k, _, cu_q, cu_k = _packed_inputs((1024,), (8192,), d=d, seed=37)
    k1, cu_k1 = compress_k(k, cu_k, 32, 16)
    k2, cu_k2 = compress_k(k, cu_k, 128, 64)
    expected = stage1(q, k1, k2, cu_q, cu_k1, cu_k2, 1024, cu_seqlens_k=cu_k)
    actual = stage1_tle_hopper(
        q, k1, k2, cu_q, cu_k1, cu_k2, 1024, cu_seqlens_k=cu_k
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@ATTENTION
def test_tle_stage2_matches_standard_when_available() -> None:
    if not HAS_TLE or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("TLE Stage2 requires Hopper")
    q, k, v, cu_q, cu_k = _packed_inputs((128,), (8192,), d=128, seed=23)
    ids = torch.arange(64, device="cuda", dtype=torch.int32)
    selected = ids.view(1, 1, 64).expand(2, 128, 64).contiguous()
    expected = _sparse_attention_forward(
        q, k, v, selected, cu_q, cu_k, 128, store_lse=False
    )[0]
    actual = sparse_attention_tle_hopper(
        q, k, v, selected, cu_q, cu_k, 128
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@ATTENTION
def test_d64_tle_stage1_dispatch_and_standard_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    kernels = importlib.import_module("flag_attn.infllmv2.forward")
    q = torch.zeros((1, 32, 64), device="cuda", dtype=torch.bfloat16)
    k1 = torch.zeros((1, 2, 64), device="cuda", dtype=torch.bfloat16)
    k2 = torch.zeros_like(k1)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    sentinel = torch.empty((2, 1, 1), device="cuda", dtype=torch.bfloat16)

    monkeypatch.setattr(kernels, "_TLE_AVAILABLE", True)
    monkeypatch.setattr(kernels, "stage1_tle_hopper", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert kernels.stage1(q, k1, k2, cu, cu, cu, 65536) is sentinel

    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(grid)

            return launch

    monkeypatch.setattr(kernels, "_TLE_AVAILABLE", False)
    monkeypatch.setattr(kernels, "stage1_tle_hopper", lambda *args, **kwargs: pytest.fail("TLE called"))
    monkeypatch.setattr(kernels, "_stage1_kernel", FakeKernel())
    fallback = kernels.stage1(q, k1, k2, cu, cu, cu, 65536)
    assert fallback.shape == (2, 1, 1)
    assert launches


@ATTENTION
def test_invalid_attention_input_is_rejected() -> None:
    q, k, v, cu_q, cu_k = _packed_inputs((32,), (64,), d=64)
    with pytest.raises(ValueError, match="maximum sequence lengths"):
        infllmv2_attention(q, k, v, cu_q, cu_k, 0, 64)


@DECODE
def test_invalid_decode_batch_is_rejected() -> None:
    _, k, v, _, cu_k = _packed_inputs((1, 1), (32, 64), d=64)
    bad_q = torch.empty((1, 32, 64), device="cuda", dtype=k.dtype)
    with pytest.raises(ValueError, match="decode q"):
        infllmv2_decode(bad_q, k, v, cu_k, 64)
