# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

from flag_attn.sage_attention import forward, per_block_int8
from flag_attn.runtime.backend import get_backend_name
from flag_attn.testing import backend as test_backend

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_backend_name() != "nvidia",
    reason="NVIDIA CUDA is required",
)


def _expand_scale(scale, block_size, length):
    return scale.repeat_interleave(block_size, dim=-1)[..., :length, None]


def _headwise_matmul(left, right):
    return torch.stack(
        [
            torch.stack([left[batch, head] @ right[batch, head] for head in range(left.shape[1])])
            for batch in range(left.shape[0])
        ]
    )


def _reference(q, k, v, q_scale, k_scale, tensor_layout, attn_mask=None, is_causal=False):
    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    q = q.float() * _expand_scale(q_scale, 128, q.shape[-2])
    k = k.float() * _expand_scale(k_scale, 64, k.shape[-2])

    num_groups = q.shape[1] // k.shape[1]
    k = torch.repeat_interleave(k, num_groups, dim=1)
    v = torch.repeat_interleave(v, num_groups, dim=1)

    logits_log2 = _headwise_matmul(q, k.transpose(-1, -2))
    if is_causal:
        causal_mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).tril()
        logits_log2 = logits_log2.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            logits_log2 = logits_log2.masked_fill(~attn_mask, float("-inf"))
        else:
            logits_log2 = logits_log2 + attn_mask

    probabilities = torch.softmax(logits_log2 * math.log(2.0), dim=-1)
    output = _headwise_matmul(probabilities, v.float())
    lse = torch.logsumexp(logits_log2 * math.log(2.0), dim=-1) / math.log(2.0)

    if tensor_layout == "NHD":
        output = output.transpose(1, 2)
    return output, lse


@pytest.mark.sage_attention
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
def test_forward_matches_dequantized_reference(tensor_layout, num_kv_heads, head_dim, output_dtype, tuned, is_causal):
    torch.manual_seed(2026)
    batch_size, num_query_heads, seq_len = 1, 2, 128
    q = torch.randn(batch_size, num_query_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    if tensor_layout == "NHD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, tensor_layout=tensor_layout)
    actual, actual_lse = forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        return_lse=True,
        output_dtype=output_dtype,
        maxnreg=(168 if head_dim == 64 else 128) if tuned else None,
        is_causal=is_causal,
    )
    expected, expected_lse = _reference(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout,
        is_causal=is_causal,
    )

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.sage_attention
def test_forward_rejects_non_positive_maxnreg():
    q = torch.zeros((1, 1, 128, 64), device="cuda", dtype=torch.int8)
    k = torch.zeros_like(q)
    v = torch.zeros(q.shape, device="cuda", dtype=torch.float16)
    q_scale = torch.ones((1, 1, 1), device="cuda")
    k_scale = torch.ones((1, 1, 2), device="cuda")

    with pytest.raises(ValueError, match="maxnreg must be positive"):
        forward(q, k, v, q_scale, k_scale, maxnreg=0)


@pytest.mark.sage_attention
@pytest.mark.parametrize("mask_kind", ["bool", "additive"])
def test_forward_supports_masks_and_partial_blocks(mask_kind):
    torch.manual_seed(7)
    q = torch.randn((1, 1, 129, 128), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 1, 70, 128), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    if mask_kind == "bool":
        attn_mask = torch.ones((1, 1, 129, 70), device="cuda", dtype=torch.bool)
        attn_mask[..., ::3] = False
    else:
        attn_mask = torch.zeros((1, 1, 129, 70), device="cuda", dtype=torch.float32)
        attn_mask[..., ::3] = -2.0

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    actual, actual_lse = forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        attn_mask=attn_mask,
        return_lse=True,
        maxnreg=168,
        is_causal=False,
    )
    expected, expected_lse = _reference(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        "HND",
        attn_mask,
    )

    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.sage_attention
@pytest.mark.parametrize("length", [64, 129, 257])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
def test_causal_default_and_partial_tiles(length, head_dim, output_dtype):
    torch.manual_seed(42)
    q = torch.randn((1, 2, length, head_dim), device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    qi, qs, ki, ks = per_block_int8(q, k)
    actual, actual_lse = forward(qi, ki, v, qs, ks, output_dtype=output_dtype, return_lse=True)
    expected, expected_lse = _reference(qi, ki, v, qs, ks, "HND", is_causal=True)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.sage_attention
def test_causal_future_values_do_not_affect_prefix():
    torch.manual_seed(7)
    q = torch.randn((2, 2, 256, 64), device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    qi, qs, ki, ks = per_block_int8(q, k)
    actual, _ = forward(qi, ki, v, qs, ks)
    changed_v = v.clone()
    changed_v[..., 64:, :] += 10
    changed, _ = forward(qi, ki, changed_v, qs, ks)
    torch.testing.assert_close(actual[..., :64, :], changed[..., :64, :], atol=0, rtol=0)


@pytest.mark.sage_attention
def test_causal_rejects_unequal_lengths_and_explicit_mask():
    q = torch.ones((1, 1, 128, 64), device="cuda", dtype=torch.int8)
    k = torch.ones((1, 1, 64, 64), device="cuda", dtype=torch.int8)
    v = torch.ones(k.shape, device="cuda", dtype=torch.float16)
    qs, ks = torch.ones((1, 1, 1), device="cuda"), torch.ones((1, 1, 1), device="cuda")
    with pytest.raises(ValueError, match="equal"):
        forward(q, k, v, qs, ks)
    with pytest.raises(ValueError, match="attn_mask"):
        forward(q, q, v, qs, ks, attn_mask=torch.ones((1, 1, 128, 128), device="cuda", dtype=torch.bool))


@pytest.mark.sage_attention
def test_benchmark_defaults_to_causal(monkeypatch):
    from benchmark.test_sage_attention_benchmark import parse_args

    monkeypatch.setattr("sys.argv", ["sage_attention_benchmark.py"])
    assert parse_args().is_causal is True
    monkeypatch.setattr("sys.argv", ["sage_attention_benchmark.py", "--no-is-causal"])
    assert parse_args().is_causal is False

    import benchmark.test_sage_attention_benchmark as bench

    args = parse_args([])
    shapes = tuple((args.batch_size, args.num_heads, length, args.head_dim) for length in args.seq_lens)
    assert shapes == tuple((4, 32, length, 128) for length in (1024, 2048, 4096, 8192, 16384, 32768))
    assert args.is_causal is True
    assert args.output_dtype == "bfloat16"
    assert (args.warmup, args.rep, args.maxnreg) == (25, 100, None)

    calls, records = [], {}

    def measure(shape, **kwargs):
        calls.append(shape)
        assert kwargs == dict(output_dtype=torch.bfloat16, warmup=25, rep=100, maxnreg=None, is_causal=True)
        return bench.BenchmarkResult(2.0, 1.0, 1.0, 0.0)

    monkeypatch.setattr(bench, "_benchmark_case", measure)
    monkeypatch.setattr(bench, "_print_header", lambda *args: None)
    monkeypatch.setattr(bench, "_print_row", lambda *args: None)
    monkeypatch.setattr(bench, "_print_footer", lambda: None)
    monkeypatch.delenv("FLAG_ATTN_BENCHMARK_LOG_PATH", raising=False)
    monkeypatch.setattr("sys.argv", ["pytest", "-m", "sage_attention", "-v", "-s"])
    test_sage_attention_benchmark(lambda key, value: records.update({key: value}))
    assert tuple(calls) == shapes
    result = records["flag_attn_benchmark_result"]
    assert result["dtype"] == "torch.bfloat16"
    assert result["is_causal"] is True
    assert tuple(tuple(row["shape_detail"]) for row in result["result"]) == shapes
    for shape in shapes:
        prefix = f"B{shape[0]}_H{shape[1]}_T{shape[2]}_D{shape[3]}_bfloat16"
        assert records[f"{prefix}.baseline_ms"] == 2.0
        assert records[f"{prefix}.flagattention_ms"] == 1.0
        assert records[f"{prefix}.speedup_vs_sdpa"] == 2.0
        assert records[f"{prefix}.cosine_similarity"] == 1.0
        assert records[f"{prefix}.relative_l2"] == 0.0


@pytest.mark.sage_attention
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("maxnreg", [None, 128])
def test_benchmark_times_quantization_and_attention(monkeypatch, is_causal, maxnreg):
    import benchmark.test_sage_attention_benchmark as bench

    state = {"timed": False, "provider": 0}
    quant_calls, attention_calls, baseline_calls = [], [], []

    def quantize(q, k):
        assert q.dtype == k.dtype == torch.float16
        quant_calls.append((state["timed"], state["provider"]))
        scale = torch.ones((1, 1, 1), device=q.device)
        return q.to(torch.int8), scale, k.to(torch.int8), scale

    def attention(q, k, v, q_scale, k_scale, **kwargs):
        assert q.dtype == k.dtype == torch.int8
        assert kwargs["is_causal"] is is_causal
        assert kwargs["maxnreg"] == maxnreg
        attention_calls.append((state["timed"], state["provider"]))
        return v.to(kwargs["output_dtype"]), None

    def baseline(q, k, v, **kwargs):
        assert kwargs["is_causal"] is is_causal
        baseline_calls.append((state["timed"], state["provider"]))
        return v

    def timer(function, **kwargs):
        state["provider"] += 1
        state["timed"] = True
        try:
            for _ in range(2):
                function()
        finally:
            state["timed"] = False
        return 1.0

    monkeypatch.setattr(bench, "per_block_int8", quantize)
    monkeypatch.setattr(bench, "sage_attention", attention)
    monkeypatch.setattr(bench, "_run_baseline", baseline)
    monkeypatch.setattr(bench, "_accuracy_metrics", lambda actual, expected: (1.0, 0.0))
    monkeypatch.setattr(bench.test_backend, "do_bench", timer)
    bench._benchmark_case((1, 1, 128, 64), output_dtype=torch.bfloat16,
                          is_causal=is_causal, maxnreg=maxnreg)
    assert [call for call in quant_calls if call[0]] == [(True, 2), (True, 2)]
    assert [call for call in attention_calls if call[0]] == [(True, 2), (True, 2)]
    assert [call for call in baseline_calls if call[0]] == [(True, 1), (True, 1)]
    assert bench.BENCHMARK_SCOPE == "quantization_and_attention"


@pytest.mark.sage_attention
@pytest.mark.parametrize("is_causal", [False, True])
def test_benchmark_records_total_scope_and_causal(monkeypatch, is_causal):
    import benchmark.test_sage_attention_benchmark as bench

    monkeypatch.setattr(bench, "_benchmark_case", lambda *args, **kwargs: bench.BenchmarkResult(2.0, 1.0, 1.0, 0.0))
    records = {}
    bench._run_benchmark_cases(((1, 1, 128, 64),), output_dtype=torch.bfloat16,
                               warmup=1, rep=1, maxnreg=None, is_causal=is_causal,
                               record_property=lambda key, value: records.update({key: value}))
    result = records["flag_attn_benchmark_result"]
    assert result["mode"] == "api"
    assert result["scope"] == "quantization_and_attention"
    assert result["is_causal"] is is_causal
    assert result["result"][0]["latency"] == 1.0


@pytest.mark.sage_attention
@pytest.mark.skipif(
    not test_backend.supports_operator("sage_attention"), reason=test_backend.skip_reason("sage_attention")
)
def test_sage_attention_benchmark(record_property):
    from benchmark.test_sage_attention_benchmark import benchmark, parse_args

    benchmark(parse_args([]), record_property=record_property)
