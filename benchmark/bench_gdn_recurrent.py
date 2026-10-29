# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Frozen scalar-GDN decode suite. State reset and cache eviction are untimed.

Example:
 PYTHONPATH=src:<verified-FlagTree-build> python benchmark/bench_gdn_recurrent.py \
   --manifest benchmark/gdn_shapes.json --suite all --cache warm \
   --rounds 7 --warmup-ms 1000 --rep-ms 100 --output result.json

Graph timing captures exactly ONE operator call and two external CUDA events.
Every replay restores the original in-place state before (and outside) the
timed graph. Event nodes provide GPU time without including reset or host gaps.
Non-graph GPU-event timing and API GPU-work timing are separate modes.
This is an operator microbenchmark, not a measured serving distribution.
"""

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent))
from baselines.vllm_recurrent import BASELINE_COMMIT
from baselines.vllm_recurrent import fused_recurrent_gated_delta_rule_fwd as baseline_api
from baselines.vllm_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as baseline_kernel
from baselines.vllm_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as baseline_packed_api,
)
from flag_attn.FLA.gated_delta_rule.fused_recurrent import _launch, fused_recurrent_gated_delta_rule_fwd
from flag_attn.FLA.gated_delta_rule.packed_decode import _launch_packed, fused_recurrent_gated_delta_rule_packed_decode
from flag_attn.testing.recurrent_gdn import make_case, reference, check_close
from flag_attn.testing.packed_gdn import make_packed_case, reference_packed


def baseline_launch(c, out):
    q, k, v, g, beta = (c[x] for x in ("q", "k", "v", "g", "beta"))
    N = c["cu_seqlens"].numel() - 1
    H, DK = q.shape[2:]
    HV, DV = v.shape[2:]
    st = c["initial_state"]
    idx = c["ssm_state_indices"]
    return baseline_kernel[(1, triton.cdiv(DV, 32), N * HV)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=out,
        h0=st,
        ht=st,
        cu_seqlens=c["cu_seqlens"],
        ssm_state_indices=idx,
        num_accepted_tokens=c["num_accepted_tokens"],
        scale=c["scale"],
        N=N,
        T=q.shape[1],
        B=1,
        H=H,
        HV=HV,
        K=DK,
        V=DV,
        BK=triton.next_power_of_2(DK),
        BV=min(triton.next_power_of_2(DV), 32),
        stride_init_state_token=st.stride(0),
        stride_final_state_token=st.stride(0),
        stride_indices_seq=idx.stride(0),
        stride_indices_tok=idx.stride(1),
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=beta.ndim == 4,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=False,
        num_warps=1,
        num_stages=3,
    )


def candidate_launch(c, out, config=None, use_tle=True):
    return _launch(
        c["q"],
        c["k"],
        c["v"],
        c["g"],
        c["beta"],
        c["scale"],
        c["initial_state"],
        True,
        c["cu_seqlens"],
        c["ssm_state_indices"],
        c["num_accepted_tokens"],
        True,
        out,
        c["initial_state"],
        config=config,
        use_tle=use_tle,
    )


def record_kernel(kernel, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    info = dict(name=kernel.name, registers=kernel.n_regs, spills=kernel.n_spills, shared=kernel.metadata.shared)
    for ext in ("ttir", "ttgir", "llir", "ptx"):
        if ext in kernel.asm:
            (directory / f"{name}.{ext}").write_text(kernel.asm[ext])
    (directory / f"{name}.json").write_text(json.dumps(info, indent=2))
    return info


class Measurement:
    def __init__(self, fn, state, original, *, graph, cache, eviction, reset_rows=None):
        self.fn, self.state, self.original = fn, state, original
        self.cache, self.eviction = cache, eviction
        self.reset_state = state if reset_rows is None else state[:reset_rows]
        self.reset_original = original if reset_rows is None else original[:reset_rows]
        self.start = torch.cuda.Event(enable_timing=True, external=graph)
        self.end = torch.cuda.Event(enable_timing=True, external=graph)
        self.graph = None
        if state is not None:
            state.copy_(original)
        fn()
        torch.cuda.synchronize()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.start.record()
                    fn()
                    self.end.record()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()

    def once(self):
        if self.reset_state is not None:
            self.reset_state.copy_(self.reset_original)
        if self.cache == "cold":
            self.eviction.zero_()
        if self.graph is not None:
            self.graph.replay()
        else:
            self.start.record()
            self.fn()
            self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1000

    def samples(self, budget_ms):
        stop = time.perf_counter() + budget_ms / 1000
        values = []
        while len(values) < 5 or time.perf_counter() < stop:
            values.append(self.once())
        return values


def cases(manifest, suite, preset_filter, n_filter, l_filter):
    suites = ["ordinary_decode", "speculative_decode"] if suite == "all" else [suite]
    for group in suites:
        spec = manifest["suites"][group]
        for preset in manifest["presets"]:
            if preset_filter and preset["id"] not in preset_filter:
                continue
            for n in spec["request_counts"]:
                if n_filter and n not in n_filter:
                    continue
                for length in spec["tokens_per_request"]:
                    if l_filter and length not in l_filter:
                        continue
                    yield group, preset, n, length


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator", choices=["recurrent", "packed"], default="recurrent")
    p.add_argument("--state-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("gdn_shapes.json"))
    p.add_argument(
        "--suite", choices=["ordinary_decode", "speculative_decode", "batch_extensions", "all"], default="all"
    )
    p.add_argument("--presets", nargs="*")
    p.add_argument("--n", nargs="*", type=int)
    p.add_argument("--length", nargs="*", type=int)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--warmup-ms", type=float, default=1000)
    p.add_argument("--rep-ms", type=float, default=100)
    p.add_argument("--cache", choices=["warm", "cold"], default="warm")
    p.add_argument("--engine", choices=["graph", "eager"], default="graph")
    p.add_argument("--scope", choices=["kernel", "public"], default="kernel")
    p.add_argument("--candidate-contiguous", action="store_true", help="Controlled same-layout ablation")
    p.add_argument("--projection-padding", type=int, default=0, help="Extra elements per packed QKV row (stress case)")
    p.add_argument("--no-tle", action="store_true", help="Same tiling with standard tl.load (ablation)")
    p.add_argument("--config", type=int, nargs=2, metavar=("BV", "WARPS"))
    p.add_argument("--dump", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.operator == "packed":
        if a.suite == "all":
            a.suite = "ordinary_decode"
        if a.suite == "speculative_decode" or (a.length and a.length != [1]):
            p.error("packed decode processes exactly one token per request")
        if a.candidate_contiguous:
            p.error("packed decode consumes the packed projection in both providers")
    if a.scope == "public" and (a.config is not None or a.no_tle or a.candidate_contiguous):
        p.error("configuration/load/layout ablations apply only to the preallocated kernel scope")
    raw = a.manifest.read_bytes()
    manifest = json.loads(raw)
    provenance = dict(
        candidate_source_sha256=hashlib.sha256(Path(sys.modules[_launch.__module__].__file__).read_bytes()).hexdigest(),
        baseline_source_sha256=hashlib.sha256(
            Path(sys.modules[baseline_api.__module__].__file__).read_bytes()
        ).hexdigest(),
        fixture_source_sha256=hashlib.sha256(Path(sys.modules[make_case.__module__].__file__).read_bytes()).hexdigest(),
        benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        model_sources_sha256=hashlib.sha256(
            Path(__file__).with_name("gdn_model_sources.json").read_bytes()
        ).hexdigest(),
        canonical_runner_sha256=hashlib.sha256(
            Path(__file__).with_name("bench_gdn_forward.py").read_bytes()
        ).hexdigest(),
        baseline_commit=BASELINE_COMMIT,
        triton=triton.__version__,
        triton_path=triton.__file__,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        state_reset="all potentially accessed state rows restored before each call; unused pool rows unchanged; reset excluded from timing",
        cache_note="warm means no forced eviction after reset; cold adds a 256 MiB eviction kernel before the timed call",
        graph="one operator invocation; external CUDA event nodes inside graph" if a.engine == "graph" else None,
        scope_note="baseline kernel inputs precontiguous; candidate uses strided views"
        if a.scope == "kernel"
        else (
            "API GPU work captured, including baseline contiguous copies; Python allocation/validation is not replayed"
            if a.engine == "graph"
            else "CUDA event interval around API calls; host enqueue gaps can contribute; not a host wall timer"
        ),
        probability="No production-frequency claim; groups reported separately",
    )
    if a.operator == "packed":
        provenance.update(
            packed_source_sha256=hashlib.sha256(
                Path(sys.modules[_launch_packed.__module__].__file__).read_bytes()
            ).hexdigest(),
            packed_fixture_sha256=hashlib.sha256(
                Path(sys.modules[make_packed_case.__module__].__file__).read_bytes()
            ).hexdigest(),
            scope_note="one packed decode kernel with fused gate transforms and all output/cache stores; identical packed layouts; no allocations in either public API",
        )
    results = []
    eviction = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.float32) if a.cache == "cold" else None
    config = None if a.config is None else (*a.config, False)

    def save():
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(dict(provenance=provenance, results=results), indent=2))

    for group, preset, n, length in cases(manifest, a.suite, a.presets, a.n, a.length):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        case_id = f"{preset['id']}_N{n}_L{length}"
        if a.operator == "packed":
            c = make_packed_case(
                n,
                preset["H"],
                preset["HV"],
                pool_min=1024,
                seed=2026,
                projection_padding=a.projection_padding,
                state_dtype=getattr(torch, a.state_dtype),
            )
        else:
            c = make_case(
                n,
                length,
                preset["H"],
                preset["HV"],
                spec=length > 1,
                strided=True,
                pool_min=1024,
                seed=2026,
                projection_padding=a.projection_padding,
                state_dtype=getattr(torch, a.state_dtype),
            )
        original = c["initial_state"].clone()
        expected = reference_packed(c) if a.operator == "packed" else reference(c)
        out_base = torch.empty_like(c["out"] if a.operator == "packed" else c["v"])
        out_tle = torch.empty_like(out_base)
        bc = c if a.operator == "packed" else {**c, **{x: c[x].contiguous() for x in ("q", "k", "v", "g", "beta")}}
        cc = bc if a.candidate_contiguous else c
        if a.operator == "packed":

            def fn_base(case=c, out=out_base):
                return baseline_packed_api(**{**case, "out": out})

            def fn_tle(case=c, out=out_tle):
                if a.scope == "public":
                    return fused_recurrent_gated_delta_rule_packed_decode(**{**case, "out": out})
                return _launch_packed(**{**case, "out": out}, config=config, use_tle=not a.no_tle)
        elif a.scope == "kernel":

            def fn_base(case=bc, out=out_base):
                return baseline_launch(case, out)

            def fn_tle(case=cc, out=out_tle):
                return candidate_launch(case, out, config, not a.no_tle)
        else:

            def fn_base(case=c):
                return baseline_api(**{**case, **{x: case[x].contiguous() for x in ("q", "k", "v", "g", "beta")}})

            def fn_tle(case=c):
                return fused_recurrent_gated_delta_rule_fwd(**case)

        resources = {}
        for provider, fn, out in (("base", fn_base, out_base), ("tle", fn_tle, out_tle)):
            c["initial_state"].copy_(original)
            result = fn()
            if a.scope == "public" or (a.operator == "packed" and provider == "base"):
                check_close(*result, *expected)
            else:
                check_close(out, c["initial_state"], *expected)
                resources[provider] = dict(
                    registers=result.n_regs, spills=result.n_spills, shared=result.metadata.shared
                )
                if a.dump:
                    record_kernel(result, a.dump, case_id + "_" + provider)
        del expected
        fixtures = {
            name: Measurement(
                fn,
                c["initial_state"],
                original,
                graph=a.engine == "graph",
                cache=a.cache,
                eviction=eviction,
                reset_rows=n * length + 1,
            )
            for name, fn in (("base", fn_base), ("tle", fn_tle))
        }
        for fixture in fixtures.values():
            fixture.samples(a.warmup_ms)
        rounds = {name: [] for name in fixtures}
        samples = {name: [] for name in fixtures}
        for r in range(a.rounds):
            order = ("base", "tle") if r % 2 == 0 else ("tle", "base")
            for name in order:
                vals = fixtures[name].samples(a.rep_ms)
                rounds[name].append(statistics.median(vals))
                samples[name].append(len(vals))
        us = {name: statistics.median(vals) for name, vals in rounds.items()}
        row = dict(
            case_id=case_id,
            group=group,
            N=n,
            L=length,
            H=preset["H"],
            HV=preset["HV"],
            base_us=us["base"],
            tle_us=us["tle"],
            speedup=us["base"] / us["tle"],
            round_medians=rounds,
            samples=samples,
            resources=resources,
            qkv_strides={
                name: list(c[name].stride())
                for name in (("mixed_qkv", "a", "b") if a.operator == "packed" else ("q", "k", "v"))
            },
            operator=a.operator,
            state_dtype=a.state_dtype,
            correctness="passed",
        )
        results.append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k not in ("round_medians", "samples")}), flush=True)
        del fixtures, fn_base, fn_tle, c, bc, cc, original, out_base, out_tle
    summary = {}
    for group in ("ordinary_decode", "speculative_decode"):
        rows = [r for r in results if r["group"] == group]
        if rows:
            summary[group] = dict(
                cases=len(rows),
                geomean=math.exp(statistics.mean(math.log(r["speedup"]) for r in rows)),
                minimum=min(r["speedup"] for r in rows),
                wins=sum(r["speedup"] > 1 for r in rows),
            )
    print("SUMMARY", json.dumps(summary), flush=True)
    save()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
