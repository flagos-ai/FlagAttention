# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unified GDN forward benchmark.

--operator all runs prefill, indexed recurrent/MTP, and fused-gate packed decode
serially, with one frozen model-geometry manifest and one timing implementation.
Output JSONs retain separate semantic scopes; no cross-phase speedup is pooled.

Example: PYTHONPATH=src:<FlagTree-build> python benchmark/bench_gdn_forward.py
 --operator all --cache cold --rounds 7 --warmup-ms 1000 --rep-ms 100 --output results

Decode-only filters and ablations are accepted by the underlying decode runner.
"""

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent))
from bench_gdn_recurrent import Measurement, main as decode_main


def _rms_check(actual, expected):
    assert torch.isfinite(actual).all()
    ratio = (actual.float() - expected.float()).square().mean().sqrt() / (
        expected.float().square().mean().sqrt() + 1e-8
    )
    assert ratio < 0.01, ratio.item()


@torch.inference_mode()
def chunk_main(argv):
    from baselines import flagattention_dispatch as legacy_dispatch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla
    from flag_attn.FLA.gated_delta_rule import chunk_gated_delta_rule_fwd
    from flag_attn.FLA.gated_delta_rule.api import _can_use_fused_tle_chunk
    import flag_attn.FLA.gated_delta_rule.chunk as implementation

    parser = argparse.ArgumentParser(description="Prefill with all raw forward auxiliary outputs retained")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("gdn_shapes.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-ms", type=float, default=1000)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--cache", choices=["cold", "warm"], default="cold")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--gate-dtype", choices=["same", "float32"], default="same")
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args(argv)
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    shapes = []
    if not args.historical_only:
        for preset in manifest["presets"]:
            for batch, length in manifest["prefill"]["model_geometry"]["batch_sequence_pairs"]:
                shapes.append((batch, length, preset["H"], preset["HV"], 128, 128))
    shapes.extend(map(tuple, manifest["prefill"]["historical_shapes"]))
    shapes = list(dict.fromkeys(shapes))
    dtype = getattr(torch, args.dtype)
    eviction = torch.empty(64 * 1024 * 1024, device="cuda") if args.cache == "cold" else None
    rows = []
    sources = [("chunk", implementation), ("legacy_dispatch", legacy_dispatch), ("fla", sys.modules[fla.__module__])]
    sources += [
        (name, importlib.import_module("flag_attn.FLA.gated_delta_rule." + name))
        for name in (
            "api",
            "chunk_fused_tle",
            "chunk_fused_forward",
            "chunk_fused_tle_prepare",
            "fused_cumsum_kkt_solve_tril",
            "wy_fast",
        )
    ]
    sources += [(name, importlib.import_module("flag_attn.FLA." + name)) for name in ("chunk_delta_h", "chunk_o")]
    provenance = dict(
        operator="chunk_gated_delta_rule_fwd",
        baseline="staged recurrence/output dispatch with shared register preparation; old/new fused main disabled",
        upstream="54417c6 dispatch policy with common boundary zero-fill fixes; both new flags disabled; timed only when valid",
        reference="external FLA; see imported module path and source hashes",
        triton=triton.__version__,
        triton_path=triton.__file__,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        model_sources_sha256=hashlib.sha256(
            Path(__file__).with_name("gdn_model_sources.json").read_bytes()
        ).hexdigest(),
        benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_paths={name: str(Path(module.__file__).resolve()) for name, module in sources},
        source_sha256={
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in sources
        },
        options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        scope="one raw forward GPU call including preparation, required zero fills, gated inverse and final state; FLA public forward is a separate reference",
        graph="one invocation per graph; external CUDA events; replay excludes Python validation/allocation",
        cache="256 MiB eviction outside timing" if args.cache == "cold" else "no extra eviction",
        precision="FP32 states and IEEE FP32 inverse merges by default; no production frequency weights",
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(provenance=provenance, results=rows), indent=2))

    for batch, length, h, hv, dk, dv in shapes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(2026)
        q = torch.randn((batch, length, h, dk), device="cuda", dtype=dtype) / dk**0.5
        k = torch.randn_like(q) / dk**0.5
        v = torch.randn((batch, length, hv, dv), device="cuda", dtype=dtype)
        gate_dtype = torch.float32 if args.gate_dtype == "float32" else dtype
        g = (-torch.rand((batch, length, hv), device="cuda") * 0.1).to(gate_dtype)
        beta = torch.rand((batch, length, hv), device="cuda", dtype=gate_dtype)
        kw = dict(q=q, k=k, v=v, g=g, beta=beta, scale=dk**-0.5, initial_state=None, output_final_state=True)

        def flag_call(enabled, register="1", call_kwargs=kw):
            old_dispatch = implementation.can_use_two_kernel_fused_forward
            if register == "0":
                implementation.can_use_two_kernel_fused_forward = legacy_dispatch.can_use_two_kernel_fused_forward
            old = os.environ.get("FLAG_ATTN_CHUNK_GDR_FUSED_TLE")
            legacy_flags = ("FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE", "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE")
            legacy_values = {key: os.environ.get(key) for key in legacy_flags}
            if register == "1" and enabled == "0":
                for key in legacy_flags:
                    os.environ[key] = "0"
            old_register = os.environ.get("FLAG_ATTN_GDN_REGISTER_PREP")
            os.environ["FLAG_ATTN_CHUNK_GDR_FUSED_TLE"] = enabled
            os.environ["FLAG_ATTN_GDN_REGISTER_PREP"] = register
            try:
                return chunk_gated_delta_rule_fwd(**call_kwargs)
            finally:
                implementation.can_use_two_kernel_fused_forward = old_dispatch
                for key, value in legacy_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                if old is None:
                    os.environ.pop("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", None)
                else:
                    os.environ["FLAG_ATTN_CHUNK_GDR_FUSED_TLE"] = old
                if old_register is None:
                    os.environ.pop("FLAG_ATTN_GDN_REGISTER_PREP", None)
                else:
                    os.environ["FLAG_ATTN_GDN_REGISTER_PREP"] = old_register

        fns = {
            "base": lambda fn=flag_call: fn("0"),
            "tle": lambda fn=flag_call: fn("1"),
            "fla": lambda qv=q, kv=k, vv=v, gate=g, bt=beta, dim=dk: fla(
                qv, kv, vv, gate, bt, scale=dim**-0.5, output_final_state=True
            ),
        }
        expected = fns["fla"]()
        base = fns["base"]()
        candidate = fns["tle"]()
        for result in (base, candidate):
            _rms_check(result[1], expected[0])
            _rms_check(result[3], expected[1])
        # The candidate must preserve the raw API's cumulative gate and inverse too.
        _rms_check(candidate[0], base[0])
        _rms_check(candidate[2], base[2])
        # An inaccurate historical provider is evidence, not a passing benchmark
        # or a speedup baseline. Keep its failed checks in the output JSON.
        upstream_errors = []
        upstream_repeatable = True
        previous = None
        for _ in range(3):
            result = flag_call("0", "0")
            pair = (result[1], result[3])
            errors = []
            for actual, target in zip(pair, expected):
                ratio = (
                    (actual.float() - target.float()).square().mean().sqrt()
                    / (target.float().square().mean().sqrt() + 1e-8)
                ).item()
                errors.append(ratio if math.isfinite(ratio) else None)
            upstream_errors.append(errors)
            if previous is not None:
                upstream_repeatable &= all(torch.equal(a, b) for a, b in zip(pair, previous))
            previous = [t.clone() for t in pair]
        upstream_valid = upstream_repeatable and all(e is not None and e < 0.01 for es in upstream_errors for e in es)
        if upstream_valid:
            fns["upstream"] = lambda fn=flag_call: fn("0", "0")
        del result, pair, previous
        del expected, base, candidate
        fixtures = {
            name: Measurement(fn, None, None, graph=True, cache=args.cache, eviction=eviction)
            for name, fn in fns.items()
        }
        for fixture in fixtures.values():
            fixture.samples(args.warmup_ms)
        rounds = {name: [] for name in fns}
        samples = {name: [] for name in fns}
        providers = list(fns)
        for r in range(args.rounds):
            shift = r % len(providers)
            order = providers[shift:] + providers[:shift]
            for name in order:
                vals = fixtures[name].samples(args.rep_ms)
                rounds[name].append(statistics.median(vals))
                samples[name].append(len(vals))
        us = {name: statistics.median(vals) for name, vals in rounds.items()}
        row = dict(
            case_id=f"B{batch}_T{length}_H{h}_HV{hv}_K{dk}_V{dv}",
            B=batch,
            T=length,
            H=h,
            HV=hv,
            K=dk,
            V=dv,
            dtype=args.dtype,
            gate_dtype=str(gate_dtype),
            operator="chunk",
            group="prefill",
            base_us=us["base"],
            tle_us=us["tle"],
            fla_us=us["fla"],
            speedup=us["base"] / us["tle"],
            fla_speedup=us["fla"] / us["tle"],
            upstream_us=us.get("upstream"),
            upstream_speedup=us["upstream"] / us["tle"] if upstream_valid else None,
            upstream_correctness=dict(passed=upstream_valid, repeatable=upstream_repeatable, errors=upstream_errors),
            round_medians=rounds,
            samples=samples,
            fused_eligible=_can_use_fused_tle_chunk(q, k, v, beta, g, None, True, None),
            correctness="candidate_and_staged_control_passed",
        )
        rows.append(row)
        save()
        print(json.dumps({k: v for k, v in row.items() if k not in ("round_medians", "samples")}), flush=True)
        del fixtures, fns, flag_call, q, k, v, g, beta, kw
    print(
        "SUMMARY",
        json.dumps(dict(cases=len(rows), geomean=math.exp(statistics.mean(math.log(r["speedup"]) for r in rows)))),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=["all", "chunk", "recurrent", "packed"], default="all")
    parser.add_argument("--state-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", choices=["cold", "warm"], default="cold")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-ms", type=float, default=1000)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("gdn_shapes.json"))
    args, extra = parser.parse_known_args()
    common = [
        "--cache",
        args.cache,
        "--rounds",
        str(args.rounds),
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
        "--manifest",
        str(args.manifest),
    ]
    operators = ["chunk", "recurrent", "packed"] if args.operator == "all" else [args.operator]
    for operator in operators:
        output = (
            args.output if len(operators) == 1 and args.output.suffix == ".json" else args.output / (operator + ".json")
        )
        if operator == "chunk":
            chunk_main(["--output", str(output), *common, *extra])
        else:
            previous = sys.argv
            sys.argv = [
                previous[0],
                "--operator",
                operator,
                "--state-dtype",
                args.state_dtype,
                "--output",
                str(output),
                *common,
                *extra,
            ]
            try:
                decode_main()
            finally:
                sys.argv = previous


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
