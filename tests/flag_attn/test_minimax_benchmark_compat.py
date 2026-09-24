# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Host checks for MiniMax benchmark baseline compatibility and accuracy guards."""

from __future__ import annotations

import pytest
import torch

from benchmark import test_minimax_sparse_attention_benchmark as benchmark


def _packed_prefill(kv_cache):
    return (
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
    )


def _packed_decode(kv_cache):
    return (
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
    )


def _split_prefill(kv_cache):
    return (
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
    )


def _split_decode(kv_cache):
    return (
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
    )


def _dynamic_stride(kv_cache, axis):
    return kv_cache.stride(axis)


def _partial_stride(kv_cache):
    return kv_cache.stride(0)


@pytest.mark.parametrize(
    "prefill,decode,expected",
    [
        (_packed_prefill, _packed_decode, "packed4d"),
        (_split_prefill, _split_decode, "split5d"),
    ],
)
def test_vllm_layout_follows_both_wrapper_stride_contracts(prefill, decode, expected):
    assert benchmark._vllm_kv_layout(prefill, decode) == expected


def test_vllm_layout_rejects_mixed_prefill_and_decode_contracts():
    with pytest.raises(RuntimeError, match="prefill/decode KV cache layouts differ"):
        benchmark._vllm_kv_layout(_packed_prefill, _split_decode)


@pytest.mark.parametrize("wrapper", [_dynamic_stride, _partial_stride, len])
def test_vllm_layout_rejects_uninspectable_or_ambiguous_contracts(wrapper):
    with pytest.raises(RuntimeError, match="layout|stride"):
        benchmark._sparse_kv_layout(wrapper)


def _data_with_remapped_pages():
    # Logical pages [0, 1, 2] refer to physical pages [2, 0, 1].
    cache = torch.arange(3 * 2 * 4 * 2 * benchmark.HEAD_DIM, dtype=torch.int32)
    cache = cache.reshape(3, 2, 4, 2 * benchmark.HEAD_DIM)
    return benchmark.MSAData(
        q=torch.empty(1),
        idx_q=torch.empty(1),
        kv_cache=cache,
        index_kv_cache=torch.empty(1),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.int32),
        cu_q=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([3 * 4], dtype=torch.int32),
        prefix_lens=torch.tensor([0], dtype=torch.int32),
        sm_scale=1.0,
        k_scale=None,
        v_scale=None,
    )


def test_vllm_legacy_cache_keeps_each_physical_page_and_kv_channel():
    data = _data_with_remapped_pages()
    legacy = benchmark._vllm_data(data, "split5d")
    assert legacy.kv_cache.shape == (3, 2, 4, 2, benchmark.HEAD_DIM)
    assert legacy.kv_cache.is_contiguous()
    assert legacy.kv_cache is not data.kv_cache
    assert legacy.block_table is data.block_table
    assert legacy.index_kv_cache is data.index_kv_cache

    for logical_page in range(3):
        physical_page = data.block_table[0, logical_page].item()
        for head in range(2):
            torch.testing.assert_close(
                legacy.kv_cache[physical_page, 0, :, head],
                data.kv_cache[physical_page, head, :, : benchmark.HEAD_DIM],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                legacy.kv_cache[physical_page, 1, :, head],
                data.kv_cache[physical_page, head, :, benchmark.HEAD_DIM :],
                rtol=0,
                atol=0,
            )


def test_vllm_packed_cache_reuses_original_data():
    data = _data_with_remapped_pages()
    assert benchmark._vllm_data(data, "packed4d") is data


def _check_outputs(flag, vllm, dtype_name="bf16"):
    return benchmark._check_outputs(
        flag,
        vllm,
        shape=(1, 128, 2, 4),
        dtype_name=dtype_name,
        mode="prefill",
        layout="split5d",
    )


@pytest.mark.parametrize("dtype_name", ["bf16", "fp8"])
def test_output_check_accepts_identical_results(dtype_name):
    output = torch.tensor([0.25, -0.5], dtype=torch.float32)
    accuracy = _check_outputs(output, output.clone(), dtype_name)
    assert accuracy == pytest.approx(1.0)
    assert accuracy <= 1.0


def test_output_check_treats_matching_zero_outputs_as_exact():
    output = torch.zeros(4)
    assert _check_outputs(output, output.clone()) == 1.0


@pytest.mark.parametrize("dtype_name", ["bf16", "fp8"])
def test_output_check_rejects_wrong_baseline_results(dtype_name):
    with pytest.raises(AssertionError, match="output mismatch.*max_abs_diff"):
        _check_outputs(torch.tensor([0.0]), torch.tensor([1.0]), dtype_name)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_output_check_rejects_nonfinite_results(bad_value):
    with pytest.raises(AssertionError, match="vLLM produced NaN or Inf"):
        _check_outputs(torch.tensor([0.0]), torch.tensor([bad_value]))
