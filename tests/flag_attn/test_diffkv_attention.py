# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

from __future__ import annotations

# Run the DiffKV correctness tests from the repository root:
#   pytest -q -m diffkv_attention \\
#     tests/flag_attn/test_diffkv_attention.py

import importlib
import pathlib
import sys

import pytest
import torch


triton = pytest.importorskip("triton")

# Keep the pytest marker aligned with ``OP_NAME`` and the ``id``/``for`` entry
# in ``conf/operators.yaml``.
pytestmark = pytest.mark.diffkv_attention

# Keep the test runnable directly from a source checkout.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from flag_attn.diffkv_attention.api import (
    OP_NAME,
    diffkv_attention,
    unified_attention_diffkv,
)

diffkv_impl = importlib.import_module(
    "flag_attn.diffkv_attention.diffkv"
)


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DiffKV Triton tests require CUDA"
)


NUM_QUERY_HEADS = 64
HEAD_SIZE_QK = 192
HEAD_SIZE_V = 128
BLOCK_SIZE = 16


def test_diffkv_op_name_export():
    """The public operator is addressable by its standard benchmark name."""
    import flag_attn

    assert OP_NAME == "diffkv_attention"
    assert getattr(flag_attn, OP_NAME) is diffkv_attention


def _reference(query, key_cache, value_cache, context_lens, block_tables,
               scale, window_size):
    batch, num_query_heads, _ = query.shape
    num_kv_heads = key_cache.shape[2]
    group = num_query_heads // num_kv_heads
    output = torch.empty(
        batch, num_query_heads, value_cache.shape[-1],
        device=query.device, dtype=torch.float32,
    )
    for b in range(batch):
        length = int(context_lens[b])
        blocks = block_tables[b, : (length + key_cache.shape[1] - 1) // key_cache.shape[1]]
        keys = key_cache[blocks].reshape(-1, num_kv_heads, key_cache.shape[-1])[:length]
        values = value_cache[blocks].reshape(-1, num_kv_heads, value_cache.shape[-1])[:length]
        positions = torch.arange(length, device=query.device)
        for h in range(num_query_heads):
            # Elementwise reduction avoids a cuBLAS GEMV dispatch for this
            # narrow (192-wide) vector and is numerically equivalent to the
            # reference dot product.
            scores = (
                keys[:, h // group, :].float() * query[b, h].float()
            ).sum(dim=-1) * scale
            if window_size > 0:
                scores = scores.masked_fill(
                    positions < length - window_size, float("-inf")
                )
            probs = torch.softmax(scores, dim=0)
            output[b, h] = torch.sum(probs[:, None] * values[:, h // group].float(), dim=0)
    return output.to(query.dtype)


def _make_case(
    batch,
    seq_len,
    *,
    num_kv_heads=4,
    dtype=torch.bfloat16,
    seed=0,
):
    """Create a deterministic paged-cache case with a physical page shuffle.

    The cache manager does not promise that logical pages are stored in
    identity order.  Using a seeded permutation here exercises the page-table
    address calculation while keeping every test case exactly reproducible.
    """
    device = "cuda"
    blocks_per_seq = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * blocks_per_seq
    generator = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(
        batch,
        NUM_QUERY_HEADS,
        HEAD_SIZE_QK,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    key_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        HEAD_SIZE_QK,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    value_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        HEAD_SIZE_V,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    physical_pages = torch.randperm(
        num_blocks, device=device, dtype=torch.int64, generator=generator
    )
    block_tables = physical_pages.reshape(batch, blocks_per_seq).to(torch.int32)
    context_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return query, key_cache, value_cache, context_lens, block_tables


def _baseline_num_segments(query, key_cache, context_lens):
    """Use the shared launch policy for the standard Triton 3D workspace."""
    batch = query.shape[0]
    num_queries_per_kv = query.shape[1] // key_cache.shape[2]
    block_m = (
        diffkv_impl._LAUNCH.query_block_m
        if num_queries_per_kv <= diffkv_impl._LAUNCH.query_block_m
        else triton.next_power_of_2(num_queries_per_kv)
    )
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = query.shape[0] // block_q + batch
    return diffkv_impl.get_num_par_softmax_segments(
        int(context_lens.max().item()),
        batch,
        True,
        total_num_q_blocks=total_num_q_blocks,
        num_kv_heads=key_cache.shape[2],
        num_sms=diffkv_impl._device_num_sms(query.device),
        block_size=key_cache.shape[1],
    )


def _run_baseline(query, key_cache, value_cache, context_lens, block_tables,
                  scale, window_size, path):
    """Invoke the standard Triton fallback with the production layout.

    The benchmark uses this path for shapes rejected by the TLE acceptance
    policy; keeping a direct correctness check here prevents a benchmark-only
    wrapper from hiding regressions in the fallback implementation.
    """
    batch, num_query_heads, _ = query.shape
    seq_len = int(context_lens.max().item())
    block_size = key_cache.shape[1]
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    out = torch.empty(
        batch, num_query_heads, value_cache.shape[-1],
        device=query.device, dtype=query.dtype,
    )
    use_3d = path == "3d"
    if use_3d:
        num_segments = _baseline_num_segments(query, key_cache, context_lens)
        segm_output = torch.empty(
            batch, num_query_heads, num_segments,
            value_cache.shape[-1], device=query.device, dtype=torch.float32,
        )
        segm_max = torch.empty(
            batch, num_query_heads, num_segments,
            device=query.device, dtype=torch.float32,
        )
        segm_expsum = torch.empty_like(segm_max)
        threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        threshold = None
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=context_lens,
        softmax_scale=scale,
        causal=True,
        window_size=(window_size - 1, 0) if window_size > 0 else (-1, -1),
        block_table=block_tables,
        softcap=0.0,
        max_seqlen_q=1,
        seq_threshold_3D=threshold,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        max_seqlen_k=seq_len,
        path=path,
        backend="triton",
    )
    return out


@pytest.mark.parametrize(
    "batch,seq_len",
    [
        (1, 512),
        (1, 513),
        (1, 2048),
        (8, 2049),
        (1, 8192),
        (4, 512),
        (2, 2048),
        (8, 512),
        (4, 4096),
        (8, 2048),
        (32, 8192),
    ],
)
@pytest.mark.parametrize("path", ["2d", "3d"])
@pytest.mark.parametrize("window_size", [-1, 128])
@CUDA_REQUIRED
def test_diffkv_attention_matches_reference(batch, seq_len, path, window_size):
    torch.manual_seed(0)
    query, key_cache, value_cache, context_lens, block_tables = _make_case(
        batch, seq_len
    )
    scale = query.shape[-1] ** -0.5
    actual = diffkv_attention(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale=scale,
        window_size=window_size,
        path=path,
    )
    expected = _reference(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        scale,
        window_size,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("batch,seq_len,path,window_size", [
    (1, 512, "3d", -1),
    (8, 2048, "3d", 128),
])
@CUDA_REQUIRED
def test_diffkv_baseline_matches_reference(batch, seq_len, path, window_size):
    """Validate the standard Triton implementation used by benchmark fallback."""
    torch.manual_seed(0)
    query, key_cache, value_cache, context_lens, block_tables = _make_case(
        batch, seq_len, seed=1
    )
    scale = query.shape[-1] ** -0.5
    actual = _run_baseline(
        query, key_cache, value_cache, context_lens, block_tables,
        scale, window_size, path,
    )
    expected = _reference(
        query, key_cache, value_cache, context_lens, block_tables,
        scale, window_size,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("batch", "seq_len", "path", "window_size"),
    [
        (1, 513, "3d", -1),
        (8, 2049, "3d", 128),
    ],
)
@CUDA_REQUIRED
def test_diffkv_swa_gqa_matches_reference(
    batch, seq_len, path, window_size
):
    """Cover the production SWA/GQA layout with eight KV heads."""
    query, key_cache, value_cache, context_lens, block_tables = _make_case(
        batch, seq_len, num_kv_heads=8, seed=2
    )
    scale = query.shape[-1] ** -0.5
    actual = diffkv_attention(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale=scale,
        window_size=window_size,
        path=path,
    )
    expected = _reference(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        scale,
        window_size,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@CUDA_REQUIRED
def test_public_api_triton_3d_matches_reference():
    """Exercise the public API's explicit non-TLE 3D path."""
    query, key_cache, value_cache, context_lens, block_tables = _make_case(
        1, 513, seed=3
    )
    scale = query.shape[-1] ** -0.5
    actual = diffkv_attention(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale=scale,
        window_size=-1,
        path="3d",
        backend="triton",
    )
    expected = _reference(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        scale,
        -1,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@CUDA_REQUIRED
def test_make_case_uses_non_identity_physical_pages():
    """Ensure correctness tests do not accidentally use logical page ids."""
    _, _, _, _, block_tables = _make_case(2, 513, seed=4)
    identity = torch.arange(
        block_tables.numel(), device=block_tables.device, dtype=block_tables.dtype
    ).reshape_as(block_tables)
    assert not torch.equal(block_tables, identity)
    assert torch.unique(block_tables).numel() == block_tables.numel()


@CUDA_REQUIRED
def test_public_api_rejects_invalid_paged_cache_inputs():
    """The public API rejects invalid structural launch arguments.

    Page-table ids and context lengths are produced by the cache manager.  The
    public operator deliberately does not scan those GPU tensors for range
    errors, because doing so would synchronize the host on every invocation.
    """
    query, key_cache, value_cache, context_lens, block_tables = _make_case(1, 512)

    with pytest.raises(ValueError, match="window_size"):
        diffkv_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            window_size=0,
        )


def test_unified_diffkv_is_lazily_exported_from_flag_attn():
    import flag_attn

    assert callable(flag_attn.unified_attention_diffkv)


@pytest.mark.parametrize(
    ("max_seqlen_k", "expected"),
    [
        (None, None),
        (512, "short"),
        (1024, "short"),
        (1025, "medium"),
        (8192, "medium"),
        (8193, "long"),
    ],
)
def test_workload_class_boundaries(max_seqlen_k, expected):
    assert diffkv_impl._tle_workload_class(max_seqlen_k) == expected


@pytest.mark.parametrize(
    ("batch", "seq_len"),
    [
        (2, 513),
        (4, 1024),
        (12, 2049),
        (24, 4096),
        (64, 16384),
    ],
)
def test_segment_policy_accepts_holdout_geometries(batch, seq_len):
    """The host policy must handle unseen batch/KV combinations generically."""
    segments = diffkv_impl.get_num_par_softmax_segments(
        seq_len,
        batch,
        True,
        total_num_q_blocks=2 * batch,
        num_kv_heads=4,
        num_sms=108,
        block_size=BLOCK_SIZE,
    )
    assert segments > 0
    assert segments & (segments - 1) == 0


def test_path_normalization():
    assert diffkv_impl._normalize_path(" 3D ") == "3d"
    assert diffkv_impl._normalize_path("2d") == "2d"
    with pytest.raises(ValueError, match="path must be 2d or 3d"):
        diffkv_impl._normalize_path("auto")


@pytest.mark.parametrize("value", ["0", "false", "off", "no", " OFF "])
def test_tle_environment_switch_disables_tle(monkeypatch, value):
    monkeypatch.setenv("FLAG_ATTN_DIFFKV_TLE", value)
    assert diffkv_impl._tle_env_enabled() is False


def test_tle_environment_switch_enabled_by_default(monkeypatch):
    monkeypatch.delenv("FLAG_ATTN_DIFFKV_TLE", raising=False)
    assert diffkv_impl._tle_env_enabled() is True


def _select_config(**overrides):
    values = {
        "head_size_qk": 192,
        "head_size_v": 128,
        "max_seqlen_q": 1,
        "max_seqlen_k": 512,
        "num_seqs": 8,
        "num_query_heads": 64,
        "num_kv_heads": 4,
        "block_size": 16,
        "num_query_tokens": 8,
        "is_decode": True,
        "path": "2d",
        "num_par_softmax_segments": None,
        "has_softmax_buffers": False,
        "num_sms": 108,
        "fused_reducer_available": False,
    }
    values.update(overrides)
    return diffkv_impl._select_tle_launch_config(**values)


def test_short_2d_policy_uses_wide_tile_for_batch_eight():
    config = _select_config()
    assert config.use_3d is False
    assert config.split_heads is False
    assert config.tile_size == 128
    assert config.loop_num_stages == 5


def test_short_3d_policy_uses_geometry_split_and_light_pipeline():
    config = _select_config(
        num_seqs=1,
        num_query_tokens=1,
        path="3d",
        num_par_softmax_segments=32,
        has_softmax_buffers=True,
    )
    assert config.use_3d is True
    assert config.num_segments == diffkv_impl._geometry_split_segments(
        512, 2, 4, 108, config.tile_size
    )
    assert config.split_heads is True
    assert config.loop_num_stages == 1


@pytest.mark.parametrize(
    (
        "num_seqs",
        "num_query_tokens",
        "max_seqlen_k",
        "segments",
        "expected_split",
        "expected_block_m",
    ),
    [
        (1, 1, 8192, 64, True, 8),
        (8, 8, 2048, 8, False, 16),
    ],
)
def test_3d_head_split_uses_active_grid_geometry(
    num_seqs,
    num_query_tokens,
    max_seqlen_k,
    segments,
    expected_split,
    expected_block_m,
):
    config = _select_config(
        num_seqs=num_seqs,
        num_query_tokens=num_query_tokens,
        max_seqlen_k=max_seqlen_k,
        num_sms=132,
        path="3d",
        num_par_softmax_segments=segments,
        has_softmax_buffers=True,
    )
    assert config.split_heads is expected_split
    assert config.block_m == expected_block_m
