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

"""Pytest benchmark for local Inkling FA4 relative attention.

Uses the same triton.testing.perf_report / do_bench convention as the other
FlagAttention benchmarks. Providers share inputs, split counts and timing.
CuTe is an optional comparison baseline; no vLLM installation is required.
"""

from __future__ import annotations


import pytest

import argparse
import importlib
import os
import sys
import types
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton.testing as triton_testing

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn.inkling_fa4 import get_backend, tle_available


pytestmark = pytest.mark.inkling_fa4_rel_attention

HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
BACKEND_NAMES = ("triton", "tle", "official_cute")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    seq_lens: tuple[tuple[int, int], ...]
    num_heads: int
    num_kv_heads: int
    rel_extent: int
    window_left: int | None = None


@dataclass(frozen=True)
class Backend:
    name: str
    fn: Callable[..., Any]


@dataclass
class Prepared:
    config: BenchmarkCase
    q: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    rel_logits: torch.Tensor
    window_size: tuple[int, int]
    scale: float
    num_splits: int
    max_q: int
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)


def build_cases() -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase("full_prefill", ((64, 64),), 4, 4, 128),
        BenchmarkCase("ragged_prefill", ((64, 64), (33, 33), (17, 17)), 8, 2, 128),
        BenchmarkCase("long_prefill", ((512, 512),), 8, 2, 1024),
        BenchmarkCase("chunked_prefill", ((32, 512),), 8, 2, 1024),
        BenchmarkCase("sliding_window", ((64, 512),), 8, 2, 256, window_left=255),
        BenchmarkCase("decode_1k", ((1, 1024),), 8, 2, 1024),
        BenchmarkCase("decode_8k", ((1, 8192),), 8, 2, 1024),
    ]
    return cases


@cache
def make_relative_score_mod(rel_extent: int):
    """Build the CuTe DSL score-mod closure for the official FA4 operator."""
    import cutlass.cute as cute
    from cutlass.cute import Float32
    from flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        del b_idx
        rel_logits = aux_tensors[0]
        local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + local_offset) - kv_idx
        global_q = seqlen_info.offset_q + q_idx
        distance = rel_dist[0]
        rel_index = distance if distance >= 0 else 0
        rel_index = rel_index if rel_index < rel_extent else rel_extent - 1
        bias = rel_logits[global_q[0], h_idx[0], rel_index]
        bias = Float32(bias) if distance == rel_index else Float32(0.0)
        return scores + bias

    return score_mod_rel_bias


def _install_namespace_package(name: str, package_dir: Path) -> None:
    """Expose ``flash_attn.cute`` without executing ``flash_attn/__init__``.

    The upstream ``__init__`` eagerly imports the ``flash_attn_2_cuda``
    extension, which the CuTe DSL path does not need.
    """
    existing = sys.modules.get(name)
    if existing is not None and not hasattr(existing, "__path__"):
        del sys.modules[name]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(package_dir)]
        package.__package__ = name
        sys.modules[name] = package
    importlib.invalidate_caches()


def _load_official_cute(flash_root: str | Path | None) -> tuple[Any, str]:
    try:
        importlib.import_module("cutlass.cute")
    except Exception as exc:
        return None, f"cutlass.cute unavailable ({type(exc).__name__}: {exc})"

    if flash_root is not None:
        package_dir = Path(flash_root).expanduser() / "flash_attn"
        if not (package_dir / "cute" / "interface.py").is_file():
            return None, f"no flash_attn/cute/interface.py under {package_dir}"
        _install_namespace_package("flash_attn", package_dir)

    try:
        module = importlib.import_module("flash_attn.cute")
        operator = getattr(module, "flash_attn_varlen_func", None)
    except Exception as exc:
        return None, f"flash_attn.cute unavailable ({type(exc).__name__}: {exc})"
    if operator is None:
        return None, "flash_attn.cute has no flash_attn_varlen_func"
    return operator, ""


def prepare_case(
    config: BenchmarkCase,
    rel_mode: str = "real",
    seed: int = 0,
    num_splits: int = 1,
    backends: tuple[str, ...] = BACKEND_NAMES,
) -> Prepared:
    torch.manual_seed(seed)
    device = test_backend.device

    q_lens = [ql for ql, _ in config.seq_lens]
    kv_lens = [kl for _, kl in config.seq_lens]
    total_q = sum(q_lens)
    num_seq = len(config.seq_lens)
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, config.num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = F.normalize(q.float(), dim=-1).to(DTYPE)

    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seq * max_blocks + 1
    key_cache = torch.randn(num_blocks, BLOCK_SIZE, config.num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE)
    key_cache = F.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn(num_blocks, BLOCK_SIZE, config.num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE)

    block_table = torch.zeros(num_seq, max_blocks, dtype=torch.int32, device=device)
    for seq in range(num_seq):
        block_table[seq] = torch.arange(
            1 + seq * max_blocks,
            1 + (seq + 1) * max_blocks,
            dtype=torch.int32,
            device=device,
        )

    cum_q = [0]
    for ql in q_lens:
        cum_q.append(cum_q[-1] + ql)
    cu_seqlens_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    if rel_mode == "real":
        rel_logits = torch.randn(total_q, config.num_heads, config.rel_extent, device=device, dtype=DTYPE)
    else:
        rel_logits = torch.zeros(total_q, config.num_heads, config.rel_extent, device=device, dtype=DTYPE)

    window_size = (-1, -1) if config.window_left is None else (config.window_left, 0)

    return Prepared(
        config=config,
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        rel_logits=rel_logits,
        window_size=window_size,
        scale=scale,
        num_splits=num_splits,
        max_q=max(q_lens),
        outputs={name: torch.empty_like(q) for name in backends},
    )


def make_runner(backend: Backend, prep: Prepared) -> Callable[[], Any]:
    """Bind arguments before timing; measure only the operator invocation."""
    if backend.name == "official_cute":
        return partial(
            backend.fn,
            q=prep.q,
            k=prep.key_cache,
            v=prep.value_cache,
            cu_seqlens_q=prep.cu_seqlens_q,
            seqused_k=prep.cache_seqlens,
            max_seqlen_q=prep.max_q,
            page_table=prep.block_table,
            softmax_scale=prep.scale,
            causal=True,
            window_size=prep.window_size,
            num_splits=prep.num_splits,
            score_mod=make_relative_score_mod(prep.config.rel_extent),
            aux_tensors=[prep.rel_logits],
            return_lse=False,
            out=prep.outputs[backend.name],
        )
    return partial(
        backend.fn,
        q=prep.q,
        key_cache=prep.key_cache,
        value_cache=prep.value_cache,
        block_table=prep.block_table,
        cache_seqlens=prep.cache_seqlens,
        cu_seqlens_q=prep.cu_seqlens_q,
        max_seqlen_q=prep.max_q,
        softmax_scale=prep.scale,
        causal=True,
        window_size=prep.window_size,
        rel_extent=prep.config.rel_extent,
        rel_logits=prep.rel_logits,
        num_splits=prep.num_splits,
        out=prep.outputs[backend.name],
    )


def resolve_backends(names: list[str], flash_root: str | None) -> dict[str, Backend]:
    """Skip unavailable optional providers; propagate local implementation bugs."""
    resolved = {}
    capability = test_backend.cuda_capability()
    for name in dict.fromkeys(names):
        if name == "official_cute":
            fn, reason = _load_official_cute(flash_root) if test_backend.is_nvidia() else (None, "CuTe requires NVIDIA")
        elif name == "tle":
            available, reason = tle_available()
            if capability[0] != 9:
                available, reason = False, "Hopper TLE requires SM90"
            fn = get_backend(name) if available else None
        else:
            from flag_attn import inkling_fa4_rel_attention
            from functools import partial

            fn, reason = partial(inkling_fa4_rel_attention, backend=name), ""
        if fn is None:
            print(f"SKIP {name}: {reason}")
        else:
            resolved[name] = Backend(name, fn)
    return resolved



def run_benchmark(args, record_property=None) -> None:
    if not test_backend.supports_operator("inkling_fa4_rel_attention"):
        pytest.skip("this benchmark requires an available accelerator")
    if test_backend.is_nvidia() and test_backend.cuda_capability() < (8, 0):
        pytest.skip("the BF16 benchmark requires SM80 or newer")

    backends = resolve_backends(args.providers, args.flash_attn_root)
    if not backends:
        pytest.skip("none of the requested providers is available")
    cases = [c for c in build_cases() if args.cases is None or c.name in args.cases]
    # Reuse the exact same tensors across providers and split configurations.
    prepared = {
        c.name: prepare_case(c, rel_mode=args.rel, seed=index, backends=tuple(backends))
        for index, c in enumerate(cases)
    }
    configs = [
        triton_testing.Benchmark(
            x_names=["case"],
            x_vals=[c.name for c in cases],
            line_arg="provider",
            line_vals=list(backends),
            line_names=list(backends),
            styles=[("red", "-"), ("blue", "-"), ("green", "-")][: len(backends)],
            ylabel="ms",
            plot_name=f"inkling_fa4-bf16-splits-{splits}-rel-{args.rel}",
            args={"num_splits": splits},
        )
        for splits in dict.fromkeys(args.num_splits)
    ]

    @triton_testing.perf_report(configs)
    def bench_relative_attention(case: str, provider: str, num_splits: int) -> float:
        data = prepared[case]
        data.num_splits = num_splits
        run = make_runner(backends[provider], data)
        # Compile and initialize the provider before entering timed measurement.
        for _ in range(3):
            run()
        test_backend.device_fn.synchronize()
        return float(
            test_backend.do_bench(
                run,
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )
        )

    print(f"GPU: {test_backend.get_device_name()} | dtype: bf16 | latency: median ms")
    print(f"warmup={args.warmup} ms | rep={args.rep} ms | relative logits={args.rel}")
    with torch.inference_mode():
        dataframes = bench_relative_attention.run(
            print_data=True, show_plots=False, save_path="", return_df=True,
        )
    if not isinstance(dataframes, (list, tuple)):
        dataframes = [dataframes]
    for config, dataframe in zip(configs, dataframes):
        if "triton" not in backends:
            continue
        splits = config.args["num_splits"]
        recorder = BenchmarkRecorder(
            record_property, op_name="inkling_fa4_rel_attention",
            dtype=str(DTYPE), baseline="official_cute" if "official_cute" in backends else None,
        )
        for _, row in dataframe.iterrows():
            baseline_ms = (
                float(row["official_cute (ms)"]) if "official_cute" in backends else None
            )
            recorder.add(
                shape_detail={"case": row["case"], "num_splits": splits, "rel": args.rel},
                latency=float(row["triton (ms)"]), latency_base=baseline_ms,
            )
        recorder.record()


def test_inkling_fa4_rel_attention_benchmark(record_property):
    args = argparse.Namespace(
        cases=None, providers=list(BACKEND_NAMES), num_splits=[1],
        warmup=25, rep=100, rel="real", flash_attn_root=os.getenv("FLASH_ATTN_ROOT"),
    )
    run_benchmark(args, record_property)
