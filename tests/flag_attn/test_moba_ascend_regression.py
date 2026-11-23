"""Regression tests for the Ascend MoBA K10-R routing fixes.

Covers the three P0 defects found by the K10 audit with *independent* oracles:

* T1 grouped top-k dropped the last candidate group -> the plan must span every
  candidate lane, and the selector is checked against a CPU top-k oracle at
  ``max_seqlen=66048`` (previously 1317 wrong selections).
* T3 ``tokens_per_park`` could be zero -> the plan must keep it >= 1, and the
  selector is exercised at ``topk=64, max_seqlen=33000`` (previously a
  ZeroDivisionError).
* T4 the 256 MiB park budget must bound one launch's scratch; the plan reports
  the exact byte count so that invariant is a unit test.
"""

import os

import pytest
import torch

# This suite is Ascend-specific. Importing torch_npu registers torch.npu; skip
# the module on vendors where it is not installed instead of failing collection.
if not hasattr(torch, "npu"):
    pytest.importorskip("torch_npu")

from flag_attn import parallel_moba
from flag_attn.FLA.moba.naive_moba import moba_attn_varlen_naive
from flag_attn.runtime.backend._ascend.FLA.moba import parallel as ascend_parallel
from flag_attn.runtime.backend._ascend.FLA.moba.routing import (
    _TOPK_PARK_BUDGET_BYTES,
    _gather_blocks_per_launch,
    _streaming_tile_shape,
    _topk_park_plan,
)
from flag_attn.testing import backend as test_backend

pytestmark = [
    pytest.mark.parallel_moba,
    pytest.mark.skipif(
        not test_backend.supports_operator("parallel_moba"),
        reason=test_backend.skip_reason("parallel_moba"),
    ),
]

MAX_CHUNKS = [
    1, 8, 64, 96, 128, 160, 224, 256,
    257, 258, 512, 513, 544, 768, 992, 993, 1024, 1025, 1504, 1505, 1536, 2048,
    4096,
]
TOPKS = [1, 2, 3, 4, 7, 8, 9, 16, 17, 31, 32, 33, 63, 64]


def _tile_shape(max_chunks: int, topk: int):
    """The production tile policy, as the selector applies it."""
    block_topk = max(2, 1 << (topk - 1).bit_length())
    block_tok, block_chunk = _streaming_tile_shape(max_chunks, block_topk)
    return block_topk, block_tok, block_chunk


@pytest.mark.parametrize("max_chunks", MAX_CHUNKS)
@pytest.mark.parametrize("topk", TOPKS)
def test_topk_plan_spans_every_candidate(max_chunks, topk):
    block_topk, block_tok, block_chunk = _tile_shape(max_chunks, topk)
    plan = _topk_park_plan(max_chunks, block_chunk, block_topk, block_tok, 16, 2)

    # T1: ceil groups cover all candidate lanes; a floor would miss the tail.
    assert plan["num_groups"] * plan["group_cand"] >= plan["total_cand_lanes"]
    assert plan["num_groups"] * plan["group_cand"] - plan["total_cand_lanes"] < plan["group_cand"]
    assert plan["group_cand"] <= 128
    assert plan["total_cand_lanes"] == plan["cand_tiles"] * block_topk

    # T3: stage-2 token tiling never becomes degenerate.
    assert plan["group_tok"] <= block_tok
    assert plan["tokens_per_park"] >= 1

    # T4: one park launch stays inside the declared scratch budget.
    programs = plan["blocks_per_launch"] * 16 * min(2, plan["batches_per_launch"])
    assert programs <= plan["budget"]
    assert programs * plan["park_program_bytes"] <= _TOPK_PARK_BUDGET_BYTES

    # Stage 2 may only consume the blocks parked by its own stage-1 launch.
    for batch_extent in (1, 2):
        gather_blocks = _gather_blocks_per_launch(plan, batch_extent)
        assert 1 <= gather_blocks <= plan["blocks_per_launch"]


def test_topk_plan_regression_shapes():
    # 516 chunks is the 66048-token T1 shape: 17 tiles x 8 ranks = 136 lanes.
    block_topk, block_tok, block_chunk = _tile_shape(516, 8)
    plan = _topk_park_plan(516, block_chunk, block_topk, block_tok, 16, 2)
    assert plan["cand_tiles"] == 17
    assert plan["total_cand_lanes"] == 136
    assert plan["num_groups"] == 2

    # 258 chunks + topk 64 is the T3 shape: previously tokens_per_park == 0.
    block_topk, block_tok, block_chunk = _tile_shape(258, 64)
    plan = _topk_park_plan(258, block_chunk, block_topk, block_tok, 16, 2)
    assert plan["tokens_per_park"] == 1


def _device_index() -> int:
    spec = os.environ.get("FLAG_ATTN_TEST_DEVICE", "npu:0")
    return int(spec.split(":")[-1]) if ":" in spec else 0


def _selector_case(max_seqlen: int, topk: int):
    """Run the inference selector on a deterministic 2-sequence batch."""
    dev_index = _device_index()
    torch.npu.set_device(dev_index)
    assert torch.npu.current_device() == dev_index
    device = f"npu:{dev_index}"

    batch, heads, dim, chunk = 2, 16, 128, 128
    tokens = batch * max_seqlen
    torch.manual_seed(20261121)
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    cu = torch.tensor(
        [i * max_seqlen for i in range(batch + 1)], device=device, dtype=torch.int32
    )
    with torch.no_grad():
        cu_chunk, chunk_end, _, valid, maxc = (
            ascend_parallel._build_dense_chunk_metadata_triton(cu, max_seqlen, chunk)
        )
        means = ascend_parallel._compute_dense_chunk_means_triton(
            k, cu_chunk, valid, chunk
        )
        hist = min(topk - 1, int(maxc) - 1)
        selected = ascend_parallel._compute_selected_chunks_batched_streaming_triton(
            q, means, chunk_end, cu, max_seqlen, int(maxc), hist, chunk
        )
        torch.npu.synchronize()
    return (
        q.cpu().float(),
        means.cpu().float(),
        chunk_end.cpu().long(),
        cu.cpu().long().tolist(),
        int(maxc),
        hist,
        selected.cpu(),
        chunk,
    )


def _check_against_cpu_topk(case, token_indices):
    q, means, chunk_end, cu, maxc, topk, selected, chunk = case
    heads = q.shape[1]
    kv_heads = means.shape[1]
    group = heads // kv_heads
    tol = 5e-2  # only near-ties may reorder; scores are O(10)

    for t in token_indices:
        batch = max(i for i in range(len(cu) - 1) if cu[i] <= t)
        start = cu[batch]
        slot_base = batch * maxc
        slots = torch.arange(slot_base, slot_base + maxc)
        ends = chunk_end[slots]
        valid = slots[(ends > start) & (t >= ends)]
        want = min(topk, valid.numel())
        for head in range(heads):
            kv = head // group
            scores = (means[valid, kv, :] * q[t, head, :]).sum(dim=1)
            order = torch.argsort(scores, descending=True, stable=True)
            ref = valid[order[:want]]
            got = selected[:, head, t]
            got = got[got >= 0]
            assert got.numel() == want, (t, head, got.numel(), want)
            got_scores = (means[got, kv, :] * q[t, head, :]).sum(dim=1)
            if not torch.equal(got.sort().values, ref.sort().values):
                # Only a near-tie may reorder: the worst selected score must be
                # within ``tol`` of the best excluded candidate.
                assert want < valid.numel(), (t, head)
                first_out = scores[order[want]]
                assert float(got_scores.min()) >= float(first_out) - tol, (t, head)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")
def test_selector_tail_group_matches_cpu_topk():
    """66048 tokens: the last candidate tile (chunk 512+) must still be ranked."""
    case = _selector_case(66048, topk=8)
    seq = 66048
    tokens = [seq + seq - 1, seq + seq - 2, seq - 1, 0, seq + 32, seq + seq - 130]
    _check_against_cpu_topk(case, tokens)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")
def test_selector_large_topk_does_not_divide_by_zero():
    """33000 tokens + topk 64: ``tokens_per_park`` was zero before the fix."""
    case = _selector_case(33000, topk=64)
    seq = 33000
    _check_against_cpu_topk(case, [seq - 1, seq + seq - 1, 0, seq + 5])


def _naive_forward(q, k, v, cu, max_seqlen, chunk, topk):
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return moba_attn_varlen_naive(
        q, k, v, cu, max_seqlen, moba_chunk_size=chunk, moba_topk=topk
    )


FORWARD_CASES = [
    ((383, 384, 385), 4, 4, 64, 128, 3, torch.bfloat16),
    ((512, 513, 1024), 8, 2, 128, 128, 4, torch.bfloat16),
    ((1025, 129), 4, 1, 64, 256, 2, torch.float16),
    ((127, 256, 385), 8, 2, 128, 128, 8, torch.bfloat16),
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU is required")
@pytest.mark.parametrize(
    ("seq_lens", "heads", "kv_heads", "dim", "chunk", "topk", "dtype"),
    FORWARD_CASES,
)
def test_forward_irregular_matches_naive(seq_lens, heads, kv_heads, dim, chunk, topk, dtype):
    dev_index = _device_index()
    torch.npu.set_device(dev_index)
    assert torch.npu.current_device() == dev_index
    device = f"npu:{dev_index}"

    torch.manual_seed(7)
    total = sum(seq_lens)
    q = torch.randn(total, heads, dim, device=device, dtype=dtype)
    k = torch.randn(total, kv_heads, dim, device=device, dtype=dtype)
    v = torch.randn(total, kv_heads, dim, device=device, dtype=dtype)
    cu = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    max_seqlen = max(seq_lens)

    with torch.no_grad():
        out = parallel_moba(q, k, v, cu, max_seqlen, chunk, topk)
        ref = _naive_forward(q, k, v, cu, max_seqlen, chunk, topk)

    diff = (out.float() - ref.float()).abs()
    assert diff.max().item() < 4e-2, diff.max().item()
    assert diff.mean().item() < 4e-4, diff.mean().item()
