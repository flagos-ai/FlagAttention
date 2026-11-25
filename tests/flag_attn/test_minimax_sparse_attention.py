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

"""Cross-backend correctness tests for the MiniMax M3 paged MSA kernels.

Manual B200 matrix (120 cases, no MiniMax dependency; NVFP4 inputs need TE)::

    pytest -q tests/flag_attn/test_minimax_sparse_attention.py::test_nvidia_msa_full_chain

Use --stage/--dtype/-k to filter the manual matrix. Disable forced TF32 for
the independent FP32 reference (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 and
NVIDIA_TF32_OVERRIDE=0). B200 tests skip on other architectures; no B200 CI.
"""

from __future__ import annotations


import importlib
from dataclasses import dataclass, replace
from typing import Sequence

import pytest
import torch
import triton.knobs

from flag_attn.testing import backend as test_backend
from flag_attn import minimax_m3_sparse_attn
from flag_attn.minimax_sparse_attention import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from flag_attn.minimax_sparse_attention import minimax_m3_sparse_attn_decode

triton.knobs.autotuning.adjust_block_size = False
BLOCK = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
BF16_SCORE_ATOL = 2e-2
BF16_SCORE_RTOL = 2e-2
FP8_SCORE_ATOL = 2e-1
FP8_SCORE_RTOL = 1.5e-1
BF16_ATOL = 2e-2
BF16_RTOL = 2e-2
FP8_ATOL = 6e-2
FP8_RTOL = 8e-2


pytestmark = pytest.mark.skipif(
    not (test_backend.supports_operator("minimax_m3_sparse_attn") or test_backend.supports_operator("minimax_m3_sparse_attn_decode")), reason="requires an available accelerator"
)


@pytest.mark.minimax_m3_sparse_attn
def test_public_exports_use_active_backend() -> None:
    from flag_attn import runtime

    try:
        backend = importlib.import_module(f"flag_attn.runtime.backend._{runtime.device.vendor_name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"flag_attn.runtime.backend._{runtime.device.vendor_name}":
            raise
        backend = None
    if backend is not None and "minimax_m3_sparse_attn_decode" in vars(backend).get("_OPERATOR_EXPORTS", {}):
        assert minimax_m3_sparse_attn_decode is backend.minimax_m3_sparse_attn_decode
    else:
        from flag_attn.minimax_sparse_attention.sparse_attn import minimax_m3_sparse_attn_decode as generic_decode

        assert minimax_m3_sparse_attn_decode is generic_decode


@dataclass
class MSAData:
    q: torch.Tensor
    idx_q: torch.Tensor
    kv_cache: torch.Tensor
    index_kv_cache: torch.Tensor
    block_table: torch.Tensor
    cu_q: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    query_lens: tuple[int, ...]
    max_seq_len: int
    sm_scale: float
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None


def _encode_fp8(value: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    if FP8_DTYPE is None:
        pytest.skip("PyTorch does not provide float8_e4m3fn")
    try:
        return (value / scale).to(FP8_DTYPE)
    except NotImplementedError:
        pytest.skip("the active backend does not support FP8 conversion")


def _storage(
    shape: tuple[int, ...],
    device: torch.device,
    use_fp8: bool,
    scale: float = 1.0,
) -> torch.Tensor:
    value = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.5
    return _encode_fp8(value, scale) if use_fp8 else value


def make_data(
    seq_lens: Sequence[int],
    num_kv_heads: int,
    group_size: int,
    *,
    decode: bool,
    decode_qlen: int = 1,
    prefix_lens: Sequence[int] | None = None,
    mode: str = "bf16",
    seed: int = 42,
) -> MSAData:
    if mode not in {"bf16", "fp8_index", "fp8_kv", "fp8_full"}:
        raise ValueError(f"unsupported mode: {mode}")
    if mode != "bf16" and not test_backend.supports_msa_fp8():
        pytest.skip("FP8 tests require accelerator FP8 support")
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError("seq_lens must contain positive lengths")
    if num_kv_heads <= 0 or group_size <= 0:
        raise ValueError("head counts must be positive")
    if decode and decode_qlen > min(seq_lens):
        raise ValueError("decode_qlen must not exceed the shortest sequence")

    torch.manual_seed(seed)
    device = torch.device(test_backend.device)
    batch = len(seq_lens)
    max_seq_len = max(seq_lens)
    if decode:
        query_lens = (decode_qlen,) * batch
        prefix_values = (0,) * batch
    else:
        prefix_values = tuple(prefix_lens) if prefix_lens is not None else (0,) * batch
        if len(prefix_values) != batch:
            raise ValueError("prefix_lens and seq_lens must have equal lengths")
        if any(
            prefix < 0 or prefix >= seq for prefix, seq in zip(prefix_values, seq_lens)
        ):
            raise ValueError("each prefix length must be in [0, seq_len)")
        query_lens = tuple(seq - prefix for seq, prefix in zip(seq_lens, prefix_values))

    if num_kv_heads * group_size <= 0:
        raise ValueError("invalid number of query heads")
    total_q = sum(query_lens)
    num_heads = num_kv_heads * group_size
    index_fp8 = mode in {"fp8_index", "fp8_full"}
    kv_fp8 = mode in {"fp8_kv", "fp8_full"}
    kv_scale_value = 0.5

    q = (
        torch.randn((total_q, num_heads, HEAD_DIM), device=device, dtype=torch.bfloat16)
        * 0.5
    )
    idx_q = _storage((total_q, num_kv_heads, HEAD_DIM), device, index_fp8)

    blocks_per_request = [(seq + BLOCK - 1) // BLOCK for seq in seq_lens]
    total_blocks = sum(blocks_per_request)
    max_blocks = max(blocks_per_request)
    logical_kv = torch.empty(
        (total_blocks, num_kv_heads, BLOCK, 2 * HEAD_DIM),
        device=device,
        dtype=torch.float8_e4m3fn if kv_fp8 else torch.bfloat16,
    )
    logical_k = _storage(
        (total_blocks * BLOCK, num_kv_heads, HEAD_DIM),
        device,
        kv_fp8,
        kv_scale_value,
    )
    logical_v = _storage(
        (total_blocks * BLOCK, num_kv_heads, HEAD_DIM),
        device,
        kv_fp8,
        kv_scale_value,
    )
    logical_k = logical_k.reshape(total_blocks, BLOCK, num_kv_heads, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    logical_v = logical_v.reshape(total_blocks, BLOCK, num_kv_heads, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    logical_kv[..., :HEAD_DIM] = logical_k
    logical_kv[..., HEAD_DIM:] = logical_v

    logical_index_k = _storage(
        (total_blocks * BLOCK, HEAD_DIM), device, index_fp8
    ).reshape(total_blocks, BLOCK, HEAD_DIM)
    physical_pages = torch.randperm(total_blocks, device=device)
    block_table = torch.zeros((batch, max_blocks), device=device, dtype=torch.int32)
    offset = 0
    for request, num_blocks in enumerate(blocks_per_request):
        block_table[request, :num_blocks] = physical_pages[offset : offset + num_blocks]
        offset += num_blocks

    kv_cache = torch.empty_like(logical_kv)
    index_kv_cache = torch.empty_like(logical_index_k)
    for cache, logical in ((kv_cache, logical_kv), (index_kv_cache, logical_index_k)):
        if cache.dtype != FP8_DTYPE:
            cache[physical_pages] = logical
            continue
        try:
            cache[physical_pages] = logical
        except NotImplementedError:
            # FP8 occupies one byte; use the same permutation on its bit pattern.
            try:
                cache.view(torch.uint8)[physical_pages] = logical.view(torch.uint8)
            except NotImplementedError:
                pytest.skip("the active backend cannot scatter FP8 cache pages")

    cu_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    prefix_lens_tensor = torch.tensor(prefix_values, device=device, dtype=torch.int32)
    if kv_fp8:
        k_scale = torch.tensor([kv_scale_value], device=device, dtype=torch.float32)
        v_scale = torch.tensor([kv_scale_value], device=device, dtype=torch.float32)
    else:
        k_scale = v_scale = None
    return MSAData(
        q=q,
        idx_q=idx_q,
        kv_cache=kv_cache,
        index_kv_cache=index_kv_cache,
        block_table=block_table,
        cu_q=cu_q,
        seq_lens=seq_lens_tensor,
        prefix_lens=prefix_lens_tensor,
        query_lens=tuple(query_lens),
        max_seq_len=max_seq_len,
        sm_scale=HEAD_DIM**-0.5,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def _ref_index_score(data: MSAData) -> torch.Tensor:
    batch = len(data.query_lens)
    num_idx_heads = data.idx_q.shape[1]
    max_blocks = (data.max_seq_len + BLOCK - 1) // BLOCK
    scores = torch.full(
        (num_idx_heads, data.q.shape[0], max_blocks),
        float("-inf"),
        device=data.q.device,
        dtype=torch.float32,
    )
    for request in range(batch):
        q_start = int(data.cu_q[request].item())
        q_end = int(data.cu_q[request + 1].item())
        seq_len = int(data.seq_lens[request].item())
        prefix_len = int(data.prefix_lens[request].item())
        query_len = q_end - q_start
        for block in range((seq_len + BLOCK - 1) // BLOCK):
            page = int(data.block_table[request, block].item())
            valid_tokens = min(BLOCK, seq_len - block * BLOCK)
            keys = data.index_kv_cache[page, :valid_tokens].float()
            queries = data.idx_q[q_start:q_end].float()
            qk = torch.einsum("qhd,kd->qhk", queries, keys)
            key_positions = torch.arange(
                block * BLOCK,
                block * BLOCK + valid_tokens,
                device=data.q.device,
            )
            query_positions = torch.arange(query_len, device=data.q.device) + prefix_len
            causal = key_positions[None, :] <= query_positions[:, None]
            qk = qk.masked_fill(~causal[:, None, :], float("-inf"))
            scores[:, q_start:q_end, block] = qk.amax(dim=-1).transpose(0, 1)
    return scores


def _ref_topk(
    scores: torch.Tensor,
    data: MSAData,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    num_idx_heads, total_q, _ = scores.shape
    result = torch.full(
        (num_idx_heads, total_q, topk),
        -1,
        device=scores.device,
        dtype=torch.int32,
    )
    for request, query_len in enumerate(data.query_lens):
        q_start = int(data.cu_q[request].item())
        prefix_len = int(data.prefix_lens[request].item())
        for query_index in range(query_len):
            query_pos = prefix_len + query_index
            valid_blocks = (query_pos + BLOCK) // BLOCK
            num_selected = min(topk, valid_blocks)
            if num_selected == 0:
                continue
            current = scores[:, q_start + query_index, :valid_blocks].clone()
            local_start = max(0, valid_blocks - local_blocks)
            current[:, local_start:valid_blocks] = 1e29
            init_end = min(init_blocks, valid_blocks)
            if init_end:
                current[:, :init_end] = torch.where(
                    current[:, :init_end] < 1e29,
                    torch.full_like(current[:, :init_end], 1e30),
                    current[:, :init_end],
                )
            result[:, q_start + query_index, :num_selected] = current.topk(
                num_selected, dim=-1
            ).indices.to(torch.int32)
    return result


def _dequantized_kv(
    data: MSAData, page: int, kv_head: int, valid_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = data.q.shape[-1]
    keys = data.kv_cache[page, kv_head, :valid_tokens, :head_dim].float()
    values = data.kv_cache[page, kv_head, :valid_tokens, head_dim:].float()
    if data.k_scale is not None:
        keys = keys * data.k_scale.item()
        values = values * data.v_scale.item()
    return keys, values


def _ref_sparse_attn(data: MSAData, topk_idx: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(data.q)
    num_kv_heads = data.kv_cache.shape[1]
    group_size = data.q.shape[1] // num_kv_heads
    for request, query_len in enumerate(data.query_lens):
        q_start = int(data.cu_q[request].item())
        seq_len = int(data.seq_lens[request].item())
        prefix_len = int(data.prefix_lens[request].item())
        for query_index in range(query_len):
            query_id = q_start + query_index
            max_k = min(prefix_len + query_index + 1, seq_len)
            for kv_head in range(num_kv_heads):
                selected = topk_idx[kv_head, query_id]
                keys_parts = []
                values_parts = []
                for block_value in selected[selected >= 0]:
                    block = int(block_value.item())
                    block_start = block * BLOCK
                    valid_tokens = min(BLOCK, max_k - block_start)
                    if valid_tokens <= 0:
                        continue
                    page = int(data.block_table[request, block].item())
                    keys, values = _dequantized_kv(data, page, kv_head, valid_tokens)
                    keys_parts.append(keys)
                    values_parts.append(values)
                if not keys_parts:
                    continue
                keys = torch.cat(keys_parts, dim=0)
                values = torch.cat(values_parts, dim=0)
                head_start = kv_head * group_size
                head_end = head_start + group_size
                query = data.q[query_id, head_start:head_end].float()
                logits = (query @ keys.transpose(0, 1)) * data.sm_scale
                weights = torch.softmax(logits, dim=-1)
                output[query_id, head_start:head_end] = (weights @ values).to(
                    output.dtype
                )
    return output


def _ref_decode_index(
    data: MSAData, topk: int, init_blocks: int, local_blocks: int, decode_qlen: int
) -> torch.Tensor:
    num_requests = len(data.query_lens)
    num_idx_heads = data.idx_q.shape[1]
    max_blocks = (data.max_seq_len + BLOCK - 1) // BLOCK
    scores = torch.full(
        (num_idx_heads, data.q.shape[0], max_blocks),
        float("-inf"),
        device=data.q.device,
        dtype=torch.float32,
    )
    for request in range(num_requests):
        seq_len = int(data.seq_lens[request].item())
        for query_index in range(decode_qlen):
            query_id = request * decode_qlen + query_index
            query_pos = seq_len - decode_qlen + query_index
            kv_len = query_pos + 1
            valid_blocks = (kv_len + BLOCK - 1) // BLOCK
            query = data.idx_q[query_id].float()
            for block in range(valid_blocks):
                page = int(data.block_table[request, block].item())
                valid_tokens = min(BLOCK, kv_len - block * BLOCK)
                keys = data.index_kv_cache[page, :valid_tokens].float()
                scores[:, query_id, block] = torch.einsum(
                    "hd,kd->hk", query, keys
                ).amax(dim=-1)
            init_end = min(init_blocks, valid_blocks)
            scores[:, query_id, :init_end] = 1e30
            local_start = max(0, valid_blocks - local_blocks)
            scores[:, query_id, local_start:valid_blocks] = 1e29

    result = torch.full(
        (num_idx_heads, data.q.shape[0], topk),
        -1,
        device=data.q.device,
        dtype=torch.int32,
    )
    for request in range(num_requests):
        seq_len = int(data.seq_lens[request].item())
        for query_index in range(decode_qlen):
            query_id = request * decode_qlen + query_index
            query_pos = seq_len - decode_qlen + query_index
            valid_blocks = (query_pos + BLOCK) // BLOCK
            num_selected = min(topk, valid_blocks)
            result[:, query_id, :num_selected] = (
                scores[:, query_id, :valid_blocks]
                .topk(num_selected, dim=-1)
                .indices.to(torch.int32)
            )
    return result


def _ref_decode_attn(
    data: MSAData, topk_idx: torch.Tensor, decode_qlen: int
) -> torch.Tensor:
    output = torch.zeros_like(data.q)
    num_requests = len(data.query_lens)
    num_kv_heads = data.kv_cache.shape[1]
    group_size = data.q.shape[1] // num_kv_heads
    for request in range(num_requests):
        seq_len = int(data.seq_lens[request].item())
        for query_index in range(decode_qlen):
            query_id = request * decode_qlen + query_index
            max_k = seq_len - decode_qlen + query_index + 1
            for kv_head in range(num_kv_heads):
                keys_parts = []
                values_parts = []
                for block_value in topk_idx[kv_head, query_id]:
                    if block_value < 0:
                        continue
                    block = int(block_value.item())
                    block_start = block * BLOCK
                    valid_tokens = min(BLOCK, max_k - block_start)
                    if valid_tokens <= 0:
                        continue
                    page = int(data.block_table[request, block].item())
                    keys, values = _dequantized_kv(data, page, kv_head, valid_tokens)
                    keys_parts.append(keys)
                    values_parts.append(values)
                if not keys_parts:
                    continue
                keys = torch.cat(keys_parts, dim=0)
                values = torch.cat(values_parts, dim=0)
                head_start = kv_head * group_size
                head_end = head_start + group_size
                query = data.q[query_id, head_start:head_end].float()
                logits = (query @ keys.transpose(0, 1)) * data.sm_scale
                weights = torch.softmax(logits, dim=-1)
                output[query_id, head_start:head_end] = (weights @ values).to(
                    output.dtype
                )
    return output


def _assert_score_match(
    actual: torch.Tensor, expected: torch.Tensor, data: MSAData, decode: bool
) -> None:
    atol = FP8_SCORE_ATOL if data.idx_q.dtype == FP8_DTYPE else BF16_SCORE_ATOL
    rtol = FP8_SCORE_RTOL if data.idx_q.dtype == FP8_DTYPE else BF16_SCORE_RTOL
    for request, query_len in enumerate(data.query_lens):
        q_start = int(data.cu_q[request].item())
        for query_index in range(query_len):
            if decode:
                query_pos = int(data.seq_lens[request].item()) - query_len + query_index
            else:
                query_pos = int(data.prefix_lens[request].item()) + query_index
            valid_blocks = (query_pos + BLOCK) // BLOCK
            torch.testing.assert_close(
                actual[:, q_start + query_index, :valid_blocks],
                expected[:, q_start + query_index, :valid_blocks],
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )


def _assert_topk_match(
    actual: torch.Tensor, expected: torch.Tensor, data: MSAData, topk: int, decode: bool
) -> None:
    for request, query_len in enumerate(data.query_lens):
        q_start = int(data.cu_q[request].item())
        for query_index in range(query_len):
            query_id = q_start + query_index
            if decode:
                query_pos = int(data.seq_lens[request].item()) - query_len + query_index
            else:
                query_pos = int(data.prefix_lens[request].item()) + query_index
            valid_blocks = (query_pos + BLOCK) // BLOCK
            selected = min(topk, valid_blocks)
            actual_set = actual[:, query_id, :selected].sort(dim=-1).values
            expected_set = expected[:, query_id, :selected].sort(dim=-1).values
            if not torch.equal(actual_set, expected_set):
                raise AssertionError(
                    f"top-k set mismatch at request={request}, query={query_index}"
                )
            if selected < topk and not torch.all(actual[:, query_id, selected:] == -1):
                raise AssertionError(
                    f"top-k padding mismatch at request={request}, query={query_index}"
                )


def _assert_attention_match(
    actual: torch.Tensor, expected: torch.Tensor, data: MSAData
) -> None:
    if not torch.isfinite(actual.float()).all():
        raise AssertionError("MSA output contains NaN or Inf")
    fp8 = data.k_scale is not None
    atol = FP8_ATOL if fp8 else BF16_ATOL
    rtol = FP8_RTOL if fp8 else BF16_RTOL
    diff = (actual.float() - expected.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(-1), expected.float().reshape(-1), dim=0
    ).item()
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(
            f"attention mismatch: cosine={cosine:.6f}, "
            f"max_diff={diff.max().item():.6e}"
        ) from exc


def _run_prefill(case: tuple, mode: str) -> None:
    seq_lens, prefixes, num_kv_heads, group_size, topk, init_blocks, local_blocks = case
    data = make_data(
        seq_lens,
        num_kv_heads,
        group_size,
        decode=False,
        prefix_lens=prefixes,
        mode=mode,
    )
    scores = minimax_m3_index_score(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        max(data.query_lens),
        data.max_seq_len,
        num_kv_heads,
    )
    topk_idx = minimax_m3_index_topk(
        scores,
        data.cu_q,
        data.prefix_lens,
        max(data.query_lens),
        topk,
        init_blocks,
        local_blocks,
    )
    output = torch.empty_like(data.q)
    sparse_kwargs = {}
    if data.k_scale is not None:
        sparse_kwargs = {"k_scale": data.k_scale, "v_scale": data.v_scale}
    minimax_m3_sparse_attn(
        data.q,
        data.kv_cache,
        topk_idx,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        max(data.query_lens),
        num_kv_heads,
        data.sm_scale,
        output,
        **sparse_kwargs,
    )
    test_backend.device_fn.synchronize()

    ref_scores = _ref_index_score(data)
    ref_topk = _ref_topk(ref_scores, data, topk, init_blocks, local_blocks)
    ref_output = _ref_sparse_attn(data, ref_topk)
    _assert_score_match(scores, ref_scores, data, decode=False)
    _assert_topk_match(topk_idx, ref_topk, data, topk, decode=False)
    _assert_attention_match(output, ref_output, data)


def _run_decode(case: tuple, mode: str) -> None:
    seq_lens, num_kv_heads, group_size, topk, init_blocks, local_blocks, decode_qlen = (
        case
    )
    data = make_data(
        seq_lens,
        num_kv_heads,
        group_size,
        decode=True,
        decode_qlen=decode_qlen,
        mode=mode,
    )
    topk_idx = minimax_m3_index_decode(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.seq_lens,
        data.max_seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_qlen,
        decode_qlen,
    )
    output = torch.empty_like(data.q)
    sparse_kwargs = {}
    if data.k_scale is not None:
        sparse_kwargs = {"k_scale": data.k_scale, "v_scale": data.v_scale}
    minimax_m3_sparse_attn(
        data.q,
        data.kv_cache,
        topk_idx,
        data.block_table,
        data.seq_lens,
        num_kv_heads,
        data.sm_scale,
        output,
        decode_qlen,
        stage="decode",
        **sparse_kwargs,
    )
    test_backend.device_fn.synchronize()

    ref_topk = _ref_decode_index(data, topk, init_blocks, local_blocks, decode_qlen)
    ref_output = _ref_decode_attn(data, ref_topk, decode_qlen)
    _assert_topk_match(topk_idx, ref_topk, data, topk, decode=True)
    _assert_attention_match(output, ref_output, data)


PREFILL_CASES = [
    # Boundary and padding: valid_blocks=2, topk=4.
    ((129,), (0,), 1, 16, 4, 1, 2),
    # Selection: 1024 tokens -> 8 blocks, topk=4.
    ((1024,), (0,), 2, 8, 4, 1, 2),
    # Ragged/prefix selection: late queries see 32/16/9 blocks, topk=8.
    ((4096, 2048, 1025), (0, 256, 128), 4, 4, 8, 1, 2),
    # The same ragged selection with eight query heads per KV head.
    ((4096, 2048, 1025), (0, 256, 128), 4, 8, 8, 1, 2),
]

DECODE_CASES = [
    # Boundary and selection: 512 tokens -> 4 blocks, topk=3.
    ((512,), 1, 16, 3, 1, 2, 1),
    # Ragged selection: late queries see 16/8/5 blocks, topk=4.
    ((2048, 1024, 513), 2, 8, 4, 1, 2, 4),
    # Long GQA case with both selection and a short padding request.
    ((8192, 2048, 1025, 129), 4, 4, 8, 2, 3, 8),
]

DECODE_SELECTION_CASES = [
    ((4097,), 1, 1, 8, 1, 2, 1),
    ((1025,) * 16, 4, 1, 4, 1, 2, 1),
]

DECODE_K16_CASES = [
    ((2048, 1025), 2, 8, 16, 1, 2, 4),
    ((4100,), 1, 1, 16, 0, 0, 4),
]


@pytest.mark.parametrize(
    "case", PREFILL_CASES, ids=("boundary", "selection", "long_ragged", "long_ragged_gqa8")
)
@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn"), reason=test_backend.skip_reason("minimax_m3_sparse_attn"))
def test_prefill_bf16(case: tuple) -> None:
    _run_prefill(case, "bf16")


@pytest.mark.parametrize(
    "case", DECODE_CASES, ids=("boundary", "selection", "long_gqa")
)
@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn_decode"), reason=test_backend.skip_reason("minimax_m3_sparse_attn_decode"))
def test_decode_bf16(case: tuple) -> None:
    _run_decode(case, "bf16")


@pytest.mark.parametrize(
    "case", DECODE_SELECTION_CASES, ids=("split_k", "single_chunk")
)
@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn_decode"), reason=test_backend.skip_reason("minimax_m3_sparse_attn_decode"))
def test_decode_topk_selection_bf16(case: tuple) -> None:
    """Cover N > K for both multi-chunk and single-chunk selection."""
    _run_decode(case, "bf16")


@pytest.mark.parametrize(
    "case", DECODE_K16_CASES, ids=("identity_ragged", "spec_causal")
)
@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn_decode"), reason=test_backend.skip_reason("minimax_m3_sparse_attn_decode"))
def test_decode_topk_k16_bf16(case: tuple) -> None:
    """Cover the configured K=16 Identity and causal selection paths."""
    _run_decode(case, "bf16")


@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn_decode"), reason=test_backend.skip_reason("minimax_m3_sparse_attn_decode"))
def test_decode_topk_identity_out_and_score_out_bf16() -> None:
    """Identity must preserve out aliasing and populate an explicit score buffer."""
    seq_lens = (2048, 1025)
    num_kv_heads = 2
    topk = 16
    decode_qlen = 4
    data = make_data(
        seq_lens,
        num_kv_heads,
        8,
        decode=True,
        decode_qlen=decode_qlen,
        mode="bf16",
    )
    total_q = data.q.shape[0]
    max_blocks = (data.max_seq_len + BLOCK - 1) // BLOCK
    out = torch.full(
        (num_kv_heads, total_q + 1, topk),
        -2,
        device=test_backend.device,
        dtype=torch.int32,
    )
    score_out = torch.full(
        (num_kv_heads, total_q, max_blocks),
        float("nan"),
        device=test_backend.device,
        dtype=torch.float32,
    )

    actual = minimax_m3_index_decode(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.seq_lens,
        data.max_seq_len,
        topk,
        1,
        2,
        num_kv_heads,
        decode_qlen,
        decode_qlen,
        out=out,
        score_out=score_out,
    )
    test_backend.device_fn.synchronize()

    expected = _ref_decode_index(data, topk, 1, 2, decode_qlen)
    _assert_topk_match(actual, expected, data, topk, decode=True)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.all(out[:, total_q] == -2)
    for request, seq_len in enumerate(seq_lens):
        for query_index in range(decode_qlen):
            query_id = request * decode_qlen + query_index
            kv_len = seq_len - decode_qlen + query_index + 1
            valid_blocks = (kv_len + BLOCK - 1) // BLOCK
            assert torch.all(torch.isfinite(score_out[:, query_id, :valid_blocks]))


if test_backend.supports_msa_fp8():
    @pytest.mark.parametrize("mode", ("fp8_index", "fp8_kv", "fp8_full"))
    @pytest.mark.parametrize("case", PREFILL_CASES[:2], ids=("boundary", "selection"))
    @pytest.mark.minimax_m3_sparse_attn
    @pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn"), reason=test_backend.skip_reason("minimax_m3_sparse_attn"))
    def test_prefill_fp8(mode: str, case: tuple) -> None:
        _run_prefill(case, mode)


    @pytest.mark.parametrize("mode", ("fp8_index", "fp8_kv", "fp8_full"))
    @pytest.mark.parametrize("case", DECODE_CASES[:2], ids=("boundary", "selection"))
    @pytest.mark.minimax_m3_sparse_attn
    @pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn_decode"), reason=test_backend.skip_reason("minimax_m3_sparse_attn_decode"))
    def test_decode_fp8(mode: str, case: tuple) -> None:
        _run_decode(case, mode)



def _non_pow2_prefill_inputs(head_dim: int, group_size: int, num_kv_heads: int):
    """Minimal packed-KV prefill inputs at an arbitrary head_dim.

    ``make_data`` is pinned to the module-level ``HEAD_DIM``, so this builds the
    tensors directly to reach head dimensions that are not a power of two.
    """
    device = test_backend.device
    query_lens = (17, 13)
    context = 4 * BLOCK
    total_q = sum(query_lens)
    batch = len(query_lens)
    pages_per_req = context // BLOCK
    total_pages = pages_per_req * batch
    topk = pages_per_req

    generator = torch.Generator(device="cpu" if device == "gcu" else device).manual_seed(1234)
    q = torch.randn(
        (total_q, num_kv_heads * group_size, head_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = (
        torch.randn(
            (total_pages, num_kv_heads, BLOCK, 2 * head_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.3
    ).to(torch.bfloat16)
    block_table = (
        torch.randperm(total_pages, generator=generator, device=device)
        .to(torch.int32)
        .view(batch, pages_per_req)
    )
    cu_q = torch.tensor(
        [0, query_lens[0], total_q], device=device, dtype=torch.int32
    )
    seq_lens = torch.full((batch,), context, device=device, dtype=torch.int32)
    prefix_lens = seq_lens - torch.tensor(
        query_lens, device=device, dtype=torch.int32
    )
    # Every request selects all of its pages, so the loop, the causal diagonal
    # and the tail mask are all exercised.
    topk_idx = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .view(1, 1, topk)
        .expand(num_kv_heads, total_q, topk)
        .contiguous()
    )
    return (
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_q,
        seq_lens,
        prefix_lens,
        max(query_lens),
        num_kv_heads,
        head_dim**-0.5,
    )


# Exercise non-power-of-two head dimensions against the same independent
# PyTorch attention reference used by the regular prefill cases.
@pytest.mark.parametrize("head_dim", (96, 100, 128))
@pytest.mark.parametrize("group_size", (8, 16))
@pytest.mark.minimax_m3_sparse_attn
@pytest.mark.skipif(not test_backend.supports_operator("minimax_m3_sparse_attn"), reason=test_backend.skip_reason("minimax_m3_sparse_attn"))
def test_prefill_bf16_non_pow2_head_dim(head_dim: int, group_size: int) -> None:
    args = _non_pow2_prefill_inputs(head_dim, group_size, num_kv_heads=1)
    actual = torch.empty_like(args[0])
    minimax_m3_sparse_attn(*args, output=actual)
    test_backend.device_fn.synchronize()

    q, kv_cache, topk_idx, block_table, cu_q, seq_lens, prefix_lens, _, _, scale = args
    query_lens = tuple((cu_q[1:] - cu_q[:-1]).tolist())
    data = MSAData(
        q=q,
        idx_q=q,
        kv_cache=kv_cache,
        index_kv_cache=kv_cache,
        block_table=block_table,
        cu_q=cu_q,
        seq_lens=seq_lens,
        prefix_lens=prefix_lens,
        query_lens=query_lens,
        max_seq_len=int(seq_lens.max().item()),
        sm_scale=scale,
        k_scale=None,
        v_scale=None,
    )
    expected = _ref_sparse_attn(data, topk_idx)
    _assert_attention_match(actual, expected, data)


# Manual SM100 workload preparation, reused by the MiniMax benchmark.
DTYPES = {
    "bf16": "BF16 / BF16 / BF16",
    "mixed": "BF16 / FP8 / FP8",
    "fp8": "FP8 / FP8 / FP8",
    "nvfp4-bf16": "BF16 / NVFP4 / NVFP4",
    "nvfp4-fp8": "FP8 / NVFP4 / NVFP4",
}
WORKLOADS = {
    "prefill": (("P1", 1, 2048, 2048), ("P2", 1, 8192, 8192),
                ("P3", 1, 512, 65536), ("P4", 1, 8192, 65536)),
    "decode": (("D1", 1, 1, 8192), ("D2", 64, 1, 8192),
               ("D3", 1, 1, 65536), ("D4", 64, 1, 65536)),
}


@dataclass(frozen=True)
class MSACase:
    stage: str
    dtype: str
    batch: int
    q: int
    kv: int
    heads: int
    gqa: int
    label: str
    tp: int = 0
    seed: int = 42
    topk: int = 16

    @property
    def id(self):
        return (f"{self.stage}-{self.dtype}-tp{self.tp}-{self.label}-"
                f"b{self.batch}-q{self.q}-s{self.kv}-h{self.heads}-g{self.gqa}")

    @property
    def packed(self):
        return self.dtype.startswith("nvfp4")

    @property
    def tolerance(self):
        return (0.02, 0.02) if self.dtype in ("bf16", "nvfp4-bf16") else (0.06, 0.08)


def msa_cases(*, stage="all", dtype="all"):
    """120 SM100 workloads: five input types, three TP layouts, two stages."""
    if dtype not in (*DTYPES, "all", "both"):
        raise ValueError(f"Unknown MSA dtype {dtype!r}; choose {tuple(DTYPES)} or all")
    cases = []
    for phase in ("prefill", "decode"):
        if stage not in ("all", phase):
            continue
        for mode in DTYPES:
            if dtype not in ("all", "both", mode):
                continue
            for tp, heads, gqa in ((1, 4, 16), (4, 1, 16), (8, 1, 8)):
                for label, batch, q, kv in WORKLOADS[phase]:
                    cases.append(MSACase(phase, mode, batch, q, kv, heads, gqa, label, tp))
    return cases


def swizzle_nvfp4_scales(scales):
    logical = scales.reshape(-1, 8).view(torch.uint8)
    row = torch.arange(logical.shape[0], device=scales.device)[:, None]
    col = torch.arange(8, device=scales.device)[None, :]
    offsets = ((row // 128) * 2 + col // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
    storage = torch.empty_like(logical).flatten()
    storage[offsets.flatten()] = logical.flatten()
    return storage.reshape(-1, 8)


def quantize_nvfp4(x, max_elements=1 << 27):
    """TE rowwise quantization, bounded preparation memory, one global scale.

    A calibration page carries the full tensor amax into each chunk. It is
    discarded before use; page boundaries align with 128-row scale tiles.
    No kernel or third-party function is monkey-patched.
    """
    from transformer_engine.pytorch.tensor import NVFP4Quantizer

    quantizer = NVFP4Quantizer(rowwise=True, columnwise=False)
    chunk_pages = max(1, max_elements // x[0].numel())
    chunked = x.numel() > max_elements
    calibration = torch.zeros_like(x[:1]) if chunked else None
    if chunked:
        calibration.flatten()[0] = torch.maximum(x.amax().abs(), x.amin().abs())
    data, scales, global_scale, first_rng = [], [], None, None
    for start in range(0, len(x), chunk_pages):
        part = x[start:start + chunk_pages]
        source = torch.cat((part, calibration)) if chunked else part
        meta = quantizer.quantize(source.contiguous()).get_metadata()
        current = meta["amax_rowwise"].float() / (448 * 6)
        if global_scale is not None:
            torch.testing.assert_close(current, global_scale, atol=0, rtol=0)
        global_scale = current
        data.append(meta["rowwise_data"].view(torch.uint8)[:len(part)].clone())
        rows = part.numel() // 128
        scales.append(meta["rowwise_scale_inv"][:rows, :8].clone().view(torch.float8_e4m3fn)
                      .reshape(*part.shape[:-1], 8))
        if first_rng is None:
            first_rng = torch.cuda.get_rng_state()
    if chunked:
        torch.cuda.set_rng_state(first_rng)
    return torch.cat(data), torch.cat(scales), global_scale.contiguous()


@dataclass
class MSAInputs:
    case: MSACase
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kv: torch.Tensor
    iq: torch.Tensor
    ik: torch.Tensor
    pages: torch.Tensor
    cuq: torch.Tensor
    seq: torch.Tensor
    prefix: torch.Tensor
    output: torch.Tensor
    ks: torch.Tensor | None = None
    vs: torch.Tensor | None = None
    kg: torch.Tensor | None = None
    vg: torch.Tensor | None = None
    iqs: torch.Tensor | None = None
    iks: torch.Tensor | None = None
    ks_storage: torch.Tensor | None = None
    vs_storage: torch.Tensor | None = None

    def index_kwargs(self):
        return dict(fp4_format="nvfp4", q_scale=self.iqs, k_scale=self.iks) if self.case.packed else {}

    def scores(self):
        c = self.case
        return minimax_m3_index_score(self.iq, self.ik, self.pages, self.cuq, self.seq, self.prefix,
                                     c.q, c.kv, c.heads, **self.index_kwargs())

    def select(self):
        c = self.case
        if c.stage == "decode":
            return minimax_m3_index_decode(self.iq, self.ik, self.pages, self.seq, c.kv, c.topk,
                                           0, 0, c.heads, c.q, c.q, **self.index_kwargs())
        return minimax_m3_index_topk(self.scores(), self.cuq, self.prefix, c.q, c.topk, 0, 0)

    def attention(self, selected):
        c = self.case
        extras = dict(k_scale=self.kg, v_scale=self.vg)
        if c.packed:
            extras = dict(kv_format="nvfp4", k_scale=self.ks_storage, v_scale=self.vs_storage,
                          k_global_scale=self.kg, v_global_scale=self.vg)
        if c.stage == "decode":
            minimax_m3_sparse_attn(self.q, self.kv, selected, self.pages, self.seq, c.heads,
                                  128**-0.5, self.output, c.q, stage="decode", **extras)
        else:
            minimax_m3_sparse_attn(self.q, self.kv, selected, self.pages, self.cuq, self.seq,
                                  self.prefix, c.q, c.heads, 128**-0.5, self.output, **extras)
        return self.output


def make_msa_inputs(case):
    c = case
    torch.manual_seed(c.seed)

    def rand(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * .5

    nb = (c.kv + 127) // 128
    npages = c.batch * nb
    q = rand((c.batch * c.q, c.heads * c.gqa, 128))
    if c.dtype in ("fp8", "nvfp4-fp8"):
        q = q.to(torch.float8_e4m3fn)
    pages = torch.randperm(npages, device="cuda").int().reshape(c.batch, nb)
    cuq = torch.arange(c.batch + 1, device="cuda", dtype=torch.int32) * c.q
    seq = torch.full((c.batch,), c.kv, device="cuda", dtype=torch.int32)
    kw = {}
    shape = (npages, c.heads, 128, 128)
    if c.packed:
        k, ks, kg = quantize_nvfp4(rand(shape))
        v, vs, vg = quantize_nvfp4(rand(shape))
        iq = torch.randint(256, (c.batch * c.q, c.heads, 64), device="cuda", dtype=torch.uint8)
        ik = torch.randint(256, (npages, c.heads, 128, 64), device="cuda", dtype=torch.uint8)
        iqs = torch.exp2(torch.randint(-3, 2, (*iq.shape[:-1], 8), device="cuda").float()).to(torch.float8_e4m3fn)
        iks = torch.exp2(torch.randint(-3, 2, (*ik.shape[:-1], 8), device="cuda").float()).to(torch.float8_e4m3fn)
        kw = dict(ks=ks, vs=vs, kg=kg, vg=vg, iqs=iqs, iks=iks,
                  ks_storage=swizzle_nvfp4_scales(ks), vs_storage=swizzle_nvfp4_scales(vs))
    else:
        dtype = torch.bfloat16 if c.dtype == "bf16" else torch.float8_e4m3fn
        k, v = rand(shape).to(dtype), rand(shape).to(dtype)
        iq = rand((c.batch * c.q, c.heads, 128)).to(dtype)
        ik = rand((npages, 128, 128)).to(dtype)
    return MSAInputs(c, q, k, v, torch.cat((k, v), -1), iq, ik, pages, cuq, seq, seq - c.q,
                     torch.empty(q.shape, device="cuda", dtype=torch.bfloat16), **kw)


# SM100-only additions; existing SM90 test coverage is unchanged.
def pytest_generate_tests(metafunc):
    if "msa_case" in metafunc.fixturenames:
        cases = msa_cases(stage=metafunc.config.getoption("--stage"),
                          dtype=metafunc.config.getoption("--dtype"))
        metafunc.parametrize("msa_case", cases, ids=lambda case: case.id)


_requires_blackwell = pytest.mark.skipif(
    not test_backend.is_nvidia() or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0),
    reason="requires NVIDIA SM100",
)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
def test_nvidia_msa_full_chain(msa_case, record_property):
    """Manual SM100 matrix: 5 dtypes x 12 workloads x prefill/decode."""
    from flag_attn.testing.minimax_sparse_attention import (
        reference_selection, reference_attention, selection_difference,
    )
    from flag_attn.runtime.backend._nvidia.minimax_sparse_attention import sparse_attn, index_topk

    data = make_msa_inputs(msa_case)
    selected_ref = reference_selection(data)
    expected = reference_attention(data, selected_ref)
    # Only clear recipe caches; compilation caches are not claimed to be cold.
    for cache in (sparse_attn._PERSISTENT_LAUNCH_CACHE, sparse_attn._FP8_LAUNCH_CACHE,
                  sparse_attn._NVFP4_LAUNCH_CACHE, sparse_attn._DECODE_LAUNCH_CACHE,
                  index_topk._DECODE_INDEX_LAUNCH_CACHE):
        cache.clear()
    diagnostics = []
    atol, rtol = msa_case.tolerance
    for call in range(2):
        selected = data.select()
        actual = data.attention(selected)
        difference = (actual.float() - expected.float()).abs()
        bad = (~torch.isfinite(actual)) | (difference > atol + rtol * expected.float().abs())
        different, total = selection_difference(data, selected, selected_ref)
        diagnostics.append(dict(call=call, topk_different_rows=different, topk_total_rows=total,
                                max_abs=float(difference.max()), failures=int(bad.sum()), elements=actual.numel()))
        record_property("msa_correctness", diagnostics.copy())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_fp8_query_early_causal_no_requantization(seed):
    """Keep the original large-input RNG sequence, reduce to its first page."""

    data = make_data((8192,), 4, 16, decode=False, prefix_lens=(0,), mode="fp8_full", seed=seed)
    q = data.q[:3].to(torch.float8_e4m3fn).contiguous()
    page = int(data.block_table[0, 0])
    kv = data.kv_cache[page : page + 1].contiguous()
    pages = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    selected = torch.full((4, 3, 16), -1, device="cuda", dtype=torch.int32)
    selected[:, :, 0] = 0
    cu = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    seq = torch.tensor([3], device="cuda", dtype=torch.int32)
    prefix = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    minimax_m3_sparse_attn(
        q,
        kv,
        selected,
        pages,
        cu,
        seq,
        prefix,
        3,
        4,
        data.sm_scale,
        output,
        k_scale=torch.ones_like(data.k_scale),
        v_scale=torch.ones_like(data.v_scale),
    )
    # Elementwise reduction avoids any TF32-dependent reference GEMM.
    for token in range(3):
        for head in range(64):
            keys = kv[0, head // 16, : token + 1, :128].float()
            values = kv[0, head // 16, : token + 1, 128:].float()
            logits = (q[token, head].float()[None, :] * keys).sum(-1) * data.sm_scale
            expected = (logits.softmax(-1)[:, None] * values).sum(0)
            torch.testing.assert_close(output[token, head].float(), expected, atol=0.06, rtol=0.08)


def unpack_fp4(data, scale, fmt):
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=data.device
    )
    nibbles = torch.stack((data & 15, data >> 4), -1).flatten(-2)
    values = lut[nibbles.long()]
    if fmt == "mxfp4":
        scales = torch.exp2(scale.view(torch.uint8).float() - 127)
        vec = 32
    else:
        scales = scale.view(torch.float8_e4m3fn).float()
        vec = 16
    return values * scales.repeat_interleave(vec, dim=-1)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("key_heads", [1, 2])
@pytest.mark.parametrize("query_lengths", [(17, 65), (128, 129)])
def test_fp4_index_scores(fmt, key_heads, query_lengths):
    torch.manual_seed(19)
    device = "cuda"
    heads = 4
    lengths = (257, 513)
    prefixes = tuple(k - q for k, q in zip(lengths, query_lengths))
    page_counts = tuple((n + 127) // 128 for n in lengths)
    total_pages = sum(page_counts)
    q = torch.randint(0, 256, (sum(query_lengths), heads, 64), device=device, dtype=torch.uint8)
    k = torch.randint(0, 256, (total_pages, key_heads, 128, 64), device=device, dtype=torch.uint8)
    vec = 32 if fmt == "mxfp4" else 16

    def scales(shape):
        if fmt == "mxfp4":
            return torch.randint(124, 129, shape, device=device, dtype=torch.uint8)
        return (torch.rand(shape, device=device) + 0.25).to(torch.float8_e4m3fn).view(torch.uint8)

    qs, ks = scales((*q.shape[:2], 128 // vec)), scales((*k.shape[:3], 128 // vec))
    if query_lengths[0] == 128:
        dtype = getattr(torch, "float8_e8m0fnu", torch.uint8) if vec == 32 else torch.float8_e4m3fn
        qs, ks = qs.view(dtype), ks.view(dtype)
    permutation = torch.randperm(total_pages, device=device).to(torch.int32)
    pages = torch.full((2, max(page_counts)), -1, device=device, dtype=torch.int32)
    pages[0, : page_counts[0]] = permutation[: page_counts[0]]
    pages[1, : page_counts[1]] = permutation[page_counts[0] :]
    cu = torch.tensor([0, query_lengths[0], sum(query_lengths)], device=device, dtype=torch.int32)
    seq = torch.tensor(lengths, device=device, dtype=torch.int32)
    prefix = torch.tensor(prefixes, device=device, dtype=torch.int32)
    # Exercise both the existing single-index-head cache rank and the grouped
    # upstream-style index-key layout.
    key_arg, ks_arg = (k[:, 0], ks[:, 0]) if key_heads == 1 else (k, ks)
    q_arg = q
    if key_heads == 2:
        q_arg, key_arg = q.view(torch.int8), key_arg.view(torch.int8)
    scores = minimax_m3_index_score(
        q_arg,
        key_arg,
        pages,
        cu,
        seq,
        prefix,
        max(query_lengths),
        max(lengths),
        heads,
        fp4_format=fmt,
        q_scale=qs,
        k_scale=ks_arg,
    )
    dq, dk = unpack_fp4(q, qs, fmt), unpack_fp4(k, ks, fmt)
    ref = torch.full_like(scores, float("-inf"))
    offset = 0
    for batch, (nq, nk, pref) in enumerate(zip(query_lengths, lengths, prefixes)):
        for h in range(heads):
            keys = dk[pages[batch, : page_counts[batch]].long(), h // (heads // key_heads)]
            keys = keys.reshape(-1, 128)
            logits = dq[offset : offset + nq, h] @ keys.T
            positions = torch.arange(keys.shape[0], device=device)
            visible = positions[None, :] <= (pref + torch.arange(nq, device=device)[:, None])
            visible &= positions[None, :] < nk
            logits.masked_fill_(~visible, float("-inf"))
            ref[h, offset : offset + nq, : page_counts[batch]] = logits.reshape(nq, page_counts[batch], 128).amax(-1)
        offset += nq
    # Existing BF16 index-score tolerance, comparing exactly the quantized
    # stored values; quantization distortion is not counted as kernel error.
    torch.testing.assert_close(scores, ref, atol=2e-2, rtol=2e-2)


def swizzle_scales(scales):
    rows = scales.numel() // 8
    row = torch.arange(rows, device=scales.device)[:, None]
    col = torch.arange(8, device=scales.device)[None, :]
    offsets = ((row // 128) * 2 + col // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
    result = torch.empty(rows * 8, dtype=torch.uint8, device=scales.device)
    result[offsets.flatten()] = scales.view(torch.uint8).flatten()
    return result


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("gqa", [1, 4, 16])
@pytest.mark.parametrize("q_fp8", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_nvfp4_sparse_attention(gqa, q_fp8, empty, index_fmt=None):
    torch.manual_seed(27)
    nq, nk, heads_k = 17, 1025 if index_fmt else 257, 2
    page_count = (nk + 127) // 128
    device = "cuda"
    q = torch.randn((nq, heads_k * gqa, 128), device=device, dtype=torch.bfloat16) * 0.5
    if q_fp8:
        q = q.to(torch.float8_e4m3fn)
    kp = torch.randint(0, 256, (page_count, heads_k, 128, 64), device=device, dtype=torch.uint8)
    vp = torch.randint(0, 256, kp.shape, device=device, dtype=torch.uint8)
    ks = (torch.rand((page_count, heads_k, 128, 8), device=device) * 0.5 + 0.125).to(torch.float8_e4m3fn)
    vs = (torch.rand(ks.shape, device=device) * 0.5 + 0.125).to(torch.float8_e4m3fn)
    kg, vg = torch.tensor(0.25, device=device), torch.tensor(0.5, device=device)
    packed = torch.cat((kp, vp), -1)
    pages = torch.randperm(page_count, device=device).to(torch.int32)[None]
    selected = torch.full((heads_k, nq, 4), -1, device=device, dtype=torch.int32)
    if not empty:
        selected[..., :3] = torch.arange(3, device=device, dtype=torch.int32)
    cu = torch.tensor([0, nq], device=device, dtype=torch.int32)
    seq = torch.tensor([nk], device=device, dtype=torch.int32)
    prefix = torch.tensor([nk - nq], device=device, dtype=torch.int32)
    if index_fmt:
        iq = torch.randint(0, 256, (nq, heads_k, 64), device=device, dtype=torch.uint8)
        ik = torch.randint(0, 256, (page_count, heads_k, 128, 64), device=device, dtype=torch.uint8)
        groups, scale = (4, 127) if index_fmt == "mxfp4" else (8, 56)
        iqs = torch.full((*iq.shape[:2], groups), scale, device=device, dtype=torch.uint8)
        iks = torch.full((*ik.shape[:3], groups), scale, device=device, dtype=torch.uint8)
        scores = minimax_m3_index_score(
            iq, ik, pages, cu, seq, prefix, nq, nk, heads_k, fp4_format=index_fmt, q_scale=iqs, k_scale=iks
        )
        selected = minimax_m3_index_topk(scores, cu, prefix, nq, 4, 1, 1)
    output = torch.empty(q.shape, device=device, dtype=torch.bfloat16)
    ks_storage, vs_storage = swizzle_scales(ks), swizzle_scales(vs)
    minimax_m3_sparse_attn(
        q,
        packed,
        selected,
        pages,
        cu,
        seq,
        prefix,
        nq,
        heads_k,
        128**-0.5,
        output,
        kv_format="nvfp4",
        k_scale=ks_storage,
        v_scale=vs_storage,
        k_global_scale=kg,
        v_global_scale=vg,
    )
    expected = torch.zeros_like(output)
    if not empty:
        dk = unpack_fp4(kp, ks, "nvfp4")
        dv = unpack_fp4(vp, vs, "nvfp4")
        # Reference the stored NVFP4 values, not an additional FP8 rounding
        # selected by the implementation's query dtype.
        dk, dv = dk.float(), dv.float()
        for h in range(heads_k):
            keys = dk[pages[0].long(), h].reshape(-1, 128)
            values = dv[pages[0].long(), h].reshape(-1, 128)
            for t in range(nq):
                count = nk - nq + t + 1
                query = q[t, h * gqa : (h + 1) * gqa].float()
                positions = torch.arange(page_count * 128, device=device)
                visible = positions < count
                visible &= (positions[:, None] // 128 == selected[h, t][None, :]).any(-1)
                logits = query @ keys[visible].T * (128**-0.5) * kg
                expected[t, h * gqa : (h + 1) * gqa] = (logits.softmax(-1) @ values[visible] * vg).to(torch.bfloat16)
    assert torch.isfinite(output).all()
    atol, rtol = (6e-2, 8e-2) if q_fp8 else (2e-2, 2e-2)
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("index_fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("q_fp8", [False, True])
def test_quantized_full_chain(index_fmt, q_fp8):
    """Real Top-K selection feeds packed attention, including forced pages."""
    test_nvfp4_sparse_attention(4, q_fp8, False, index_fmt=index_fmt)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_packed_index_requires_format(dtype):
    q = torch.zeros((1, 1, 64), device="cuda", dtype=dtype)
    k = torch.zeros((1, 128, 64), device="cuda", dtype=dtype)
    pages = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seq = torch.tensor([128], device="cuda", dtype=torch.int32)
    prefix = seq - 1
    with pytest.raises(ValueError, match="explicit fp4_format"):
        minimax_m3_index_score(q, k, pages, cu, seq, prefix, 1, 128, 1)
    with pytest.raises(ValueError, match="explicit fp4_format"):
        minimax_m3_index_decode(q, k, pages, seq, 128, 4, 0, 0, 1, 1, 1)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("gqa", [1, 4, 16])
@pytest.mark.parametrize("q_fp8", [False, True])
def test_sm100_scalar_scale_loop(gqa, q_fp8):
    """Exercise scalar KV scales through the public prefill entry."""
    data = make_data((257, 513), 2, gqa, decode=False, prefix_lens=(240, 496), mode="fp8_kv")
    q = data.q.to(torch.float8_e4m3fn) if q_fp8 else data.q
    ks, vs = torch.tensor(0.37, device="cuda"), torch.tensor(0.73, device="cuda")
    data = replace(data, q=q.to(torch.bfloat16), k_scale=ks, v_scale=vs)
    selected = torch.arange(4, device="cuda", dtype=torch.int32).expand(2, q.shape[0], 4).clone()
    selected[:, :17, 3] = -1
    output = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    minimax_m3_sparse_attn(
        q,
        data.kv_cache,
        selected,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        17,
        2,
        data.sm_scale,
        output,
        k_scale=ks,
        v_scale=vs,
    )
    torch.testing.assert_close(output, _ref_sparse_attn(data, selected), atol=0.06, rtol=0.08)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("q_fp8", [False, True])
def test_nvfp4_graph_replay_refreshes_cache(q_fp8):
    """Graph replay reads current packed values, not values from capture."""
    nq, nk, npage = 2048, 2305, 19
    q = torch.randn((nq, 4, 128), device="cuda", dtype=torch.bfloat16)
    if q_fp8:
        q = q.to(torch.float8_e4m3fn)
    kv = torch.randint(0, 256, (npage, 1, 128, 128), device="cuda", dtype=torch.uint8)
    scales = swizzle_scales(torch.ones((npage, 1, 128, 8), device="cuda").to(torch.float8_e4m3fn))
    global_scale = torch.ones((), device="cuda")
    selected = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32).expand(1, nq, 4).contiguous()
    pages = torch.randperm(npage, device="cuda").int()[None]
    cu = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
    seq = torch.tensor([nk], device="cuda", dtype=torch.int32)
    prefix = seq - nq
    output = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)

    def run():
        minimax_m3_sparse_attn(
            q,
            kv,
            selected,
            pages,
            cu,
            seq,
            prefix,
            nq,
            1,
            128**-0.5,
            output,
            kv_format="nvfp4",
            k_scale=scales,
            v_scale=scales,
            k_global_scale=global_scale,
            v_global_scale=global_scale,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    assert output.float().abs().max() > 0
    kv.zero_()
    graph.replay()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.minimax_m3_sparse_attn
@_requires_blackwell
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("qlen,topk", [(1, 4), (4, 4), (4, 16)])
def test_fp4_decode_score_out_and_topk(fmt, qlen, topk):
    torch.manual_seed(31)
    device = "cuda"
    heads, total_pages, max_blocks = 4, 14, 9
    q = torch.randint(0, 256, (2 * qlen, heads, 64), device=device, dtype=torch.uint8)
    k = torch.randint(0, 256, (total_pages, 128, 64), device=device, dtype=torch.uint8)
    vec = 32 if fmt == "mxfp4" else 16
    qs = torch.full((*q.shape[:2], 128 // vec), 127 if vec == 32 else 56, device=device, dtype=torch.uint8)
    ks = torch.full((*k.shape[:2], 128 // vec), 127 if vec == 32 else 56, device=device, dtype=torch.uint8)
    permutation = torch.randperm(total_pages, device=device).to(torch.int32)
    pages = torch.full((2, 9), -1, device=device, dtype=torch.int32)
    pages[0, :5], pages[1] = permutation[:5], permutation[5:]
    seq = torch.tensor([513, 1025], device=device, dtype=torch.int32)
    cu = torch.arange(3, device=device, dtype=torch.int32) * qlen
    prefix = seq - qlen
    raw = minimax_m3_index_score(
        q, k, pages, cu, seq, prefix, qlen, 1025, heads, fp4_format=fmt, q_scale=qs, k_scale=ks
    )[..., :max_blocks]
    expected = raw.clone()
    for batch in range(2):
        for token in range(qlen):
            qid = batch * qlen + token
            visible = (int(seq[batch]) - qlen + token + 128) // 128
            expected[:, qid, : min(1, visible)] = 1.0e30
            expected[:, qid, max(0, visible - 2) : visible] = 1.0e29
    score_out = torch.full((heads, max_blocks, 2 * qlen), float("nan"), device=device).transpose(1, 2)
    out = torch.full((heads, 2 * qlen + 3, topk), -99, device=device, dtype=torch.int32)
    selected = minimax_m3_index_decode(
        q,
        k,
        pages,
        seq,
        1025,
        topk,
        1,
        2,
        heads,
        qlen,
        qlen,
        out=out,
        score_out=score_out,
        fp4_format=fmt,
        q_scale=qs,
        k_scale=ks,
    )
    assert selected.data_ptr() == out.data_ptr()
    assert torch.all(out[:, 2 * qlen :] == -99)
    torch.testing.assert_close(score_out, expected, atol=0.02, rtol=0.02)
    selected_scores = score_out.gather(-1, selected.clamp(min=0).long())
    selected_scores = selected_scores.masked_fill(selected < 0, float("-inf"))
    padded = torch.nn.functional.pad(expected, (0, max(0, topk - max_blocks)), value=float("-inf"))
    best_scores = padded.topk(topk, dim=-1).values
    torch.testing.assert_close(selected_scores.sort(-1).values, best_scores.sort(-1).values, atol=0.02, rtol=0.02)
