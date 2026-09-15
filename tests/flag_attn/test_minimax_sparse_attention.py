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

"""CUDA correctness tests for the MiniMax M3 paged MSA kernels."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Sequence

import pytest
import torch
import triton.knobs

from flag_attn.minimax_sparse_attention import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

index_topk_module = importlib.import_module(minimax_m3_index_topk.__module__)

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


def _supports_fp8() -> bool:
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    # NVIDIA FP8 Tensor Core support starts with Ada (8.9) and Hopper (9.0).
    return torch.cuda.get_device_capability() >= (8, 9)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MSA v1 tests require CUDA"
)


def test_public_exports_use_active_backend() -> None:
    from flag_attn.runtime.backend import is_metax_backend

    module_name = minimax_m3_sparse_attn.__module__
    if is_metax_backend():
        assert module_name.startswith(
            "flag_attn.runtime.backend._metax.minimax_sparse_attention"
        )
    else:
        assert module_name.startswith("flag_attn.minimax_sparse_attention")


def test_metax_public_exports() -> None:
    from flag_attn.runtime.backend import is_metax_backend

    if not is_metax_backend():
        pytest.skip("requires the MetaX backend")

    import flag_attn
    from flag_attn.runtime.backend import _metax

    assert flag_attn.chunk_gdn2 is _metax.chunk_gdn2
    assert flag_attn.minimax_m3_sparse_attn is _metax.minimax_m3_sparse_attn


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
    return (value / scale).to(FP8_DTYPE)


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
    if mode != "bf16" and not _supports_fp8():
        pytest.skip("FP8 tests require an NVIDIA GPU with FP8 support")
    if not seq_lens or any(length <= 0 for length in seq_lens):
        raise ValueError("seq_lens must contain positive lengths")
    if num_kv_heads <= 0 or group_size <= 0:
        raise ValueError("head counts must be positive")
    if decode and decode_qlen > min(seq_lens):
        raise ValueError("decode_qlen must not exceed the shortest sequence")

    torch.manual_seed(seed)
    device = torch.device("cuda")
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
    kv_cache[physical_pages] = logical_kv
    index_kv_cache[physical_pages] = logical_index_k

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
    keys = data.kv_cache[page, kv_head, :valid_tokens, :HEAD_DIM].float()
    values = data.kv_cache[page, kv_head, :valid_tokens, HEAD_DIM:].float()
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
    torch.cuda.synchronize()

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
    minimax_m3_sparse_attn_decode(
        data.q,
        data.kv_cache,
        topk_idx,
        data.block_table,
        data.seq_lens,
        num_kv_heads,
        data.sm_scale,
        output,
        decode_qlen,
        **sparse_kwargs,
    )
    torch.cuda.synchronize()

    ref_topk = _ref_decode_index(data, topk, init_blocks, local_blocks, decode_qlen)
    ref_output = _ref_decode_attn(data, ref_topk, decode_qlen)
    _assert_topk_match(topk_idx, ref_topk, data, topk, decode=True)
    _assert_attention_match(output, ref_output, data)


@pytest.mark.minimax_sparse_attention_topk
def test_prefill_topk_streaming_partial_tile_excludes_padding() -> None:
    """Invalid lanes must lose even when every valid score is negative infinity."""
    num_score_blocks = 96
    valid_blocks = 70
    topk = 16
    score = torch.full(
        (1, 1, num_score_blocks),
        -float("inf"),
        device="cuda",
        dtype=torch.float32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor(
        [(valid_blocks - 1) * BLOCK], device="cuda", dtype=torch.int32
    )

    assert index_topk_module._select_prefill_topk_path(score, topk, 1) == "streaming"
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        1,
        topk,
        0,
        0,
    )
    torch.cuda.synchronize()

    expected = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, 1, topk)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not index_topk_module._HAS_TLE_RADIX,
    reason="requires a backend with TLE radix top-k support",
)
@pytest.mark.minimax_sparse_attention_topk
def test_prefill_topk_radix_path() -> None:
    """Exercise the actual wide-row TLE path instead of accepting fallback."""
    num_score_blocks = 1024
    topk = 16
    generator = torch.Generator(device="cuda")
    generator.manual_seed(123)
    score = torch.randn(
        (1, 1, num_score_blocks),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    prefix_lens = torch.tensor(
        [(num_score_blocks - 1) * BLOCK], device="cuda", dtype=torch.int32
    )
    block_size_k, _ = index_topk_module._radix_prefill_launch_config(num_score_blocks)
    config_key = (block_size_k, topk)
    index_topk_module._FAILED_RADIX_CONFIGS.discard(config_key)

    assert index_topk_module._select_prefill_topk_path(score, topk, 1) == "radix"
    actual = minimax_m3_index_topk(
        score,
        cu_seqlens_q,
        prefix_lens,
        1,
        topk,
        0,
        0,
    )
    torch.cuda.synchronize()

    assert config_key not in index_topk_module._FAILED_RADIX_CONFIGS
    expected = torch.topk(score, topk, dim=-1).indices.to(torch.int32)
    expected = expected.sort(dim=-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


PREFILL_CASES = [
    # Boundary and padding: valid_blocks=2, topk=4.
    ((129,), (0,), 1, 16, 4, 1, 2),
    # Selection: 1024 tokens -> 8 blocks, topk=4.
    ((1024,), (0,), 2, 8, 4, 1, 2),
    # Ragged/prefix selection: late queries see 32/16/9 blocks, topk=8.
    ((4096, 2048, 1025), (0, 256, 128), 4, 4, 8, 1, 2),
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
    "case", PREFILL_CASES, ids=("boundary", "selection", "long_ragged")
)
@pytest.mark.minimax_sparse_attention_prefill
def test_prefill_bf16(case: tuple) -> None:
    _run_prefill(case, "bf16")


@pytest.mark.parametrize(
    "case", DECODE_CASES, ids=("boundary", "selection", "long_gqa")
)
@pytest.mark.minimax_sparse_attention_decode
def test_decode_bf16(case: tuple) -> None:
    _run_decode(case, "bf16")


@pytest.mark.parametrize(
    "case", DECODE_SELECTION_CASES, ids=("split_k", "single_chunk")
)
@pytest.mark.minimax_sparse_attention_decode
def test_decode_topk_selection_bf16(case: tuple) -> None:
    """Cover N > K for both multi-chunk and single-chunk selection."""
    _run_decode(case, "bf16")


@pytest.mark.parametrize(
    "case", DECODE_K16_CASES, ids=("identity_ragged", "spec_causal")
)
@pytest.mark.minimax_sparse_attention_decode
def test_decode_topk_k16_bf16(case: tuple) -> None:
    """Cover the configured K=16 Identity and causal selection paths."""
    _run_decode(case, "bf16")


@pytest.mark.minimax_sparse_attention_decode
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
        device="cuda",
        dtype=torch.int32,
    )
    score_out = torch.full(
        (num_kv_heads, total_q, max_blocks),
        float("nan"),
        device="cuda",
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
    torch.cuda.synchronize()

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


@pytest.mark.skipif(
    not _supports_fp8(),
    reason="FP8 tests require an NVIDIA GPU with FP8 support",
)
@pytest.mark.parametrize("mode", ("fp8_index", "fp8_kv", "fp8_full"))
@pytest.mark.parametrize("case", PREFILL_CASES[:2], ids=("boundary", "selection"))
@pytest.mark.minimax_sparse_attention_prefill
def test_prefill_fp8(mode: str, case: tuple) -> None:
    _run_prefill(case, mode)


@pytest.mark.skipif(
    not _supports_fp8(),
    reason="FP8 tests require an NVIDIA GPU with FP8 support",
)
@pytest.mark.parametrize("mode", ("fp8_index", "fp8_kv", "fp8_full"))
@pytest.mark.parametrize("case", DECODE_CASES[:2], ids=("boundary", "selection"))
@pytest.mark.minimax_sparse_attention_decode
def test_decode_fp8(mode: str, case: tuple) -> None:
    _run_decode(case, mode)
