"""Regression tests for Ascend MoBA routing and rank planning.

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

from flag_attn.runtime.backend._ascend.FLA.moba import routing

pytestmark = pytest.mark.parallel_moba

MAX_CHUNKS = [
    1, 8, 64, 96, 128, 160, 224, 256,
    257, 258, 512, 513, 544, 768, 992, 993, 1024, 1025, 1504, 1505, 1536, 2048,
    4096,
]
TOPKS = [1, 2, 3, 4, 7, 8, 9, 16, 17, 31, 32, 33, 63, 64]


def _tile_shape(max_chunks: int, topk: int):
    """The production tile policy, as the selector applies it."""
    block_topk = max(2, 1 << (topk - 1).bit_length())
    block_tok, block_chunk = routing._streaming_tile_shape(max_chunks, block_topk)
    return block_topk, block_tok, block_chunk


@pytest.mark.parametrize("max_chunks", MAX_CHUNKS)
@pytest.mark.parametrize("topk", TOPKS)
def test_topk_plan_spans_every_candidate(max_chunks, topk):
    block_topk, block_tok, block_chunk = _tile_shape(max_chunks, topk)
    plan = routing._topk_park_plan(max_chunks, block_chunk, block_topk, block_tok, 16, 2)

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
    assert programs * plan["park_program_bytes"] <= routing._TOPK_PARK_BUDGET_BYTES

    # Stage 2 may only consume the blocks parked by its own stage-1 launch.
    for batch_extent in (1, 2):
        gather_blocks = routing._gather_blocks_per_launch(plan, batch_extent)
        assert 1 <= gather_blocks <= plan["blocks_per_launch"]


def test_topk_plan_regression_shapes():
    # 516 chunks is the 66048-token T1 shape: 17 tiles x 8 ranks = 136 lanes.
    block_topk, block_tok, block_chunk = _tile_shape(516, 8)
    plan = routing._topk_park_plan(516, block_chunk, block_topk, block_tok, 16, 2)
    assert plan["cand_tiles"] == 17
    assert plan["total_cand_lanes"] == 136
    assert plan["num_groups"] == 2

    # 258 chunks + topk 64 is the T3 shape: previously tokens_per_park == 0.
    block_topk, block_tok, block_chunk = _tile_shape(258, 64)
    plan = routing._topk_park_plan(258, block_chunk, block_topk, block_tok, 16, 2)
    assert plan["tokens_per_park"] == 1


def test_topk_plan_bounds_minimum_launch():
    max_chunks, block_chunk, block_topk, block_tok = 16385, 32, 64, 16
    plan = routing._topk_park_plan(max_chunks, block_chunk, block_topk, block_tok, 64, 1)
    assert plan["head_limit"] == 63
    program_bytes = plan["park_program_bytes"]
    assert 63 * program_bytes <= routing._TOPK_PARK_BUDGET_BYTES < 64 * program_bytes

    with pytest.raises(ValueError, match="one MoBA top-k park program"):
        routing._topk_park_plan(2**22, block_chunk, block_topk, block_tok, 1, 1)

    # With many batches, one park block can expand beyond the gather cap.
    plan = routing._topk_park_plan(8, 32, 2, 128, 16, 1024)
    batch_extent = plan["batches_per_launch"]
    gather_blocks = routing._gather_blocks_per_launch(plan, batch_extent)
    assert (
        gather_blocks * 16 * batch_extent * plan["tokens_per_park"] <= plan["gather_budget"]
    )


@pytest.fixture
def npu_device():
    from flag_attn.testing import backend

    if backend.device != "npu" or not backend.is_available():
        pytest.skip("An active Ascend NPU backend is required")
    spec = os.environ.get("FLAG_ATTN_TEST_DEVICE", "npu:0")
    index = int(spec.split(":")[-1]) if ":" in spec else 0
    torch.npu.set_device(index)
    assert torch.npu.current_device() == index
    return f"npu:{index}"


def _selector_case(device: str, max_seqlen: int, topk: int):
    """Run the inference selector on a deterministic 2-sequence batch."""
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
            routing._build_dense_chunk_metadata_triton(cu, max_seqlen, chunk)
        )
        means = routing._compute_dense_chunk_means_triton(
            k, cu_chunk, valid, chunk
        )
        hist = min(topk - 1, int(maxc) - 1)
        selected = routing._compute_selected_chunks_batched_streaming_triton(
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
    )


def _check_against_cpu_topk(case, token_indices):
    q, means, chunk_end, cu, maxc, topk, selected = case
    heads = q.shape[1]
    kv_heads = means.shape[1]
    group = heads // kv_heads
    tol = 5e-2  # only near-ties may reorder; scores are O(10)
    changed_ranks = replaced_ids = near_ties = 0
    max_gap = 0.0
    errors = []

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
            ranks = selected[:, head, t]
            got = ranks[:want]
            assert torch.all(ranks[want:] == -1), (t, head, ranks)
            assert torch.isin(got, valid).all(), (t, head, got, valid)
            assert got.unique().numel() == want, (t, head, got)
            changed_ranks += (got != ref).sum().item()

            missing = ref[~torch.isin(ref, got)]
            if missing.numel():
                extra = got[~torch.isin(got, ref)]
                replaced_ids += missing.numel()
                # Compare the best missed candidate with the worst replacement.
                best_missed = scores[torch.searchsorted(valid, missing)].max()
                worst_extra = scores[torch.searchsorted(valid, extra)].min()
                gap = (best_missed - worst_extra).item()
                max_gap = max(max_gap, gap)
                if gap <= tol:
                    near_ties += 1
                else:
                    errors.append((t, head, gap, missing.tolist(), extra.tolist()))

    diagnostics = (
        f"checks={len(token_indices) * heads}, changed_ranks={changed_ranks}, "
        f"replaced_ids={replaced_ids}, max_boundary_gap={max_gap:.6g}, "
        f"near_tie_rows={near_ties}, wrong_rows={len(errors)}"
    )
    print(diagnostics)
    assert not errors, (diagnostics, errors[:3])


def test_selector_tail_group_matches_cpu_topk(npu_device):
    """66048 tokens: the last candidate tile (chunk 512+) must still be ranked."""
    case = _selector_case(npu_device, 66048, topk=8)
    seq = 66048
    tokens = [seq + seq - 1, seq + seq - 2, seq - 1, 0, seq + 32, seq + seq - 130]
    _check_against_cpu_topk(case, tokens)


def test_selector_large_topk_does_not_divide_by_zero(npu_device):
    """33000 tokens + topk 64: ``tokens_per_park`` was zero before the fix."""
    case = _selector_case(npu_device, 33000, topk=64)
    seq = 33000
    _check_against_cpu_topk(case, [seq - 1, seq + seq - 1, 0, seq + 5])


@pytest.mark.parametrize("head_limit", [1, 4])
def test_selector_head_slices_preserve_gqa(npu_device, monkeypatch, head_limit):
    chunk_size, seq_len, max_chunks, topk = 4, 25, 7, 2
    cu = torch.tensor([0, 9, 9 + seq_len], dtype=torch.int32, device=npu_device)
    _, chunk_end, _, _, _ = routing._build_dense_chunk_metadata_triton(cu, seq_len, chunk_size)
    q = torch.zeros((34, 6, 32), dtype=torch.bfloat16, device=npu_device)
    q[:, :, 0] = 1
    means = torch.zeros((2 * max_chunks, 3, 32), dtype=q.dtype, device=npu_device)
    scores = torch.arange(max_chunks, dtype=torch.float32, device=npu_device)
    means[:, 0, 0] = scores.repeat(2)
    means[:, 1, 0] = -scores.repeat(2)

    # A one-head limit splits KV groups. Four heads fit two whole groups.
    args = q, means, chunk_end, cu, seq_len, max_chunks, topk, chunk_size
    expected = routing._compute_selected_chunks_batched_streaming_triton(*args)
    assert expected[:, :, -1].cpu().tolist() == [
        [12, 12, 7, 7, 7, 7],
        [11, 11, 8, 8, 8, 8],
    ]
    plan = routing._topk_park_plan(max_chunks, 32, 2, 128, 6, 2)
    monkeypatch.setattr(routing, "_TOPK_PARK_BUDGET_BYTES", head_limit * plan["park_program_bytes"])
    park_kernel = routing._chunk_topk_park_kernel
    launch_programs = []

    class ObservedParkKernel:
        def __getitem__(self, grid):
            blocks, heads, batches = grid
            launch_programs.append(blocks * heads * batches)
            return park_kernel[grid]

    monkeypatch.setattr(routing, "_chunk_topk_park_kernel", ObservedParkKernel())
    actual = routing._compute_selected_chunks_batched_streaming_triton(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert launch_programs and max(launch_programs) <= head_limit


@pytest.mark.parametrize("topk, tail_start", [(1, 2048), (2, 2048), (7, 512)])
def test_selectors_exact_ids(npu_device, topk, tail_start):
    """Exercise a partially filled gather group and compare both selectors."""
    chunk_size, heads, dim = 4, 5, 32
    history = (2, tail_start + 2)
    short_len, long_len = (count * chunk_size + 1 for count in history)
    offsets = [0, short_len, short_len + long_len]
    max_chunks = history[1] + 1
    block_topk, block_tok, block_chunk = _tile_shape(max_chunks, topk)
    plan = routing._topk_park_plan(max_chunks, block_chunk, block_topk, block_tok, heads, 2)
    assert plan["num_groups"] == 2
    assert plan["total_cand_lanes"] % plan["group_cand"] > 0

    # One-hot queries produce exact BF16 scores. The last head uses two base-64
    # digits to rank the newest historical chunk first at each causal boundary.
    query = torch.zeros((heads, dim), dtype=torch.bfloat16)
    query[:4, :4] = torch.eye(4, dtype=query.dtype)
    query[4, 4:6] = torch.tensor([64, 1])
    q = query.expand(offsets[-1], -1, -1).contiguous().to(npu_device)
    means = torch.zeros((2 * max_chunks, 1, dim), dtype=torch.bfloat16)
    means[:, :, :6] = 64  # high scores in padding and the final local chunk
    dense_ids, ends, batch_ends = [], [], []
    for batch, count in enumerate(history):
        base = batch * max_chunks
        means[base:base + count] = 0
        ids = torch.arange(count)
        means[base:base + count, 0, 4] = ids // 64
        means[base:base + count, 0, 5] = ids % 64
        dense_ids.extend(range(base, base + count))
        ends.extend(offsets[batch] + (i + 1) * chunk_size for i in range(count))
        batch_ends.extend([offsets[batch + 1]] * count)

    tail = max_chunks + tail_start
    # Head 0 has all ties. Head 1 has a unique maximum in the second group.
    # Head 2 ties for first within that group. Head 3 ties across groups at
    # the last retained rank for topk=7.
    means[tail:tail + 2, 0, 1] = torch.tensor([3, 4])
    means[tail:tail + 2, 0, 2] = 4
    means[max_chunks:max_chunks + 6, 0, 3] = 3
    means[max_chunks + 6:max_chunks + 8, 0, 3] = 2
    means[tail:tail + 2, 0, 3] = 2

    cu = torch.tensor(offsets, device=npu_device, dtype=torch.int32)
    _, dense_ends, _, _, _ = routing._build_dense_chunk_metadata_triton(
        cu, long_len, chunk_size
    )
    inference = routing._compute_selected_chunks_batched_streaming_triton(
        q, means.to(npu_device), dense_ends, cu, long_len, max_chunks, topk, chunk_size
    ).cpu()
    training = routing._compute_selected_chunks_streaming_triton(
        q,
        means[dense_ids].to(npu_device),
        torch.tensor(ends, device=npu_device, dtype=torch.int32),
        torch.tensor(batch_ends, device=npu_device, dtype=torch.int32),
        topk,
    ).cpu()

    compact_base = 0
    for batch, count in enumerate(history):
        start = offsets[batch]
        dense_base = batch * max_chunks
        scores = (
            query.float() @ means[dense_base:dense_base + count, 0].float().T
        ).tolist()
        causal_tokens = ends[compact_base:compact_base + count]
        for selected, base in ((inference, dense_base), (training, compact_base)):
            # Give every historical candidate an opportunity to win Top-1 on
            # head 4 at the first token where that candidate is causal.
            got = selected[0, 4, causal_tokens].tolist()
            assert got == list(range(base, base + count))

            # Derive legal candidates from the sequence geometry, independent
            # of device metadata, then compare the full ordered list and padding.
            sample = {0, chunk_size - 1, chunk_size, count * chunk_size - 1, count * chunk_size}
            if batch == 1:
                sample.update({tail_start * chunk_size, (tail_start + 1) * chunk_size})
            for local_token in sorted(sample):
                eligible = range(local_token // chunk_size)
                for head, score in enumerate(scores):
                    ranked = sorted(eligible, key=lambda i: (-score[i], base + i))[:topk]
                    expected = [base + i for i in ranked] + [-1] * (topk - len(ranked))
                    actual = selected[:, head, start + local_token].tolist()
                    assert actual == expected, (batch, local_token, head, base, actual, expected)
        compact_base += count


def _naive_forward(q, k, v, cu, max_seqlen, chunk, topk):
    from flag_attn.FLA.moba.naive_moba import moba_attn_varlen_naive

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


@pytest.mark.parametrize(
    ("seq_lens", "heads", "kv_heads", "dim", "chunk", "topk", "dtype"),
    FORWARD_CASES,
)
def test_forward_irregular_matches_naive(
    npu_device, seq_lens, heads, kv_heads, dim, chunk, topk, dtype
):
    from flag_attn import parallel_moba
    from flag_attn.runtime.backend._ascend.FLA.moba import parallel as ascend_parallel

    assert parallel_moba is ascend_parallel.parallel_moba

    torch.manual_seed(7)
    total = sum(seq_lens)
    q = torch.randn(total, heads, dim, device=npu_device, dtype=dtype)
    k = torch.randn(total, kv_heads, dim, device=npu_device, dtype=dtype)
    v = torch.randn(total, kv_heads, dim, device=npu_device, dtype=dtype)
    cu = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=npu_device, dtype=torch.int32
    )
    max_seqlen = max(seq_lens)

    with torch.no_grad():
        out = parallel_moba(q, k, v, cu, max_seqlen, chunk, topk)
        ref = _naive_forward(q, k, v, cu, max_seqlen, chunk, topk)

    diff = (out.float() - ref.float()).abs()
    assert diff.max().item() < 4e-2, diff.max().item()
    assert diff.mean().item() < 4e-4, diff.mean().item()
