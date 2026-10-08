"""Official ACP vs V7.6 TLE. Full-call CUDA Event latency, never CUDA Graph.

Supports pytest discovery and direct CLI execution. Latencies are in ms.
Run this file with --full for all 90 shapes, or --ids S011,S024 for a subset.
The original reference adapters and input generation live in the single
correctness test file; no JSON manifest or custom pytest markers are needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

# Direct script execution works without installing or modifying the environment.
_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "tests/flag_attn"), str(_ROOT)]

import pytest
import torch
import triton

from flag_attn.forgetting_attention import has_tle
from test_forgetting_attention import (
    CASES, DEFAULT_CASES, OFFICIAL_COMMIT, case_id, make_inputs,
    call_optimized, call_official, assert_bitwise,
)

CUDA_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = [
    pytest.mark.skipif(not CUDA_SM90, reason="ACP benchmark requires Hopper SM90 CUDA"),
    pytest.mark.skipif(not has_tle(), reason="ACP benchmark requires Triton 3.6 TLE"),
]


def event_samples(fn, warmup, rep):
    if warmup < 1 or rep < 1:
        raise ValueError("warmup and rep must both be positive iteration counts")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(rep):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        pairs.append((begin, end))
    torch.cuda.synchronize()
    samples = [a.elapsed_time(b) for a, b in pairs]
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError("invalid CUDA Event timing")
    return samples


@torch.inference_mode()
def benchmark_case(case, warmup=40, rep=400, rounds=3, seed=0):
    if rounds < 1:
        raise ValueError("rounds must be positive")
    inputs = make_inputs(case, seed)
    expected = call_official(inputs, case)
    assert_bitwise(call_optimized(inputs, case), expected)
    assert_bitwise(call_optimized(inputs, case), expected)
    del expected
    providers = {"official": lambda: call_official(inputs, case),
                 "tle_v76": lambda: call_optimized(inputs, case)}
    measurements = []
    for index in range(rounds):
        order = list(providers)
        if (index + int(case["id"][1:])) % 2:
            order.reverse()
        result = {"round": index, "provider_order": order}
        for name in order:
            values = event_samples(providers[name], warmup, rep)
            result[name] = {"median_ms": statistics.median(values), "samples_ms": values}
        measurements.append(result)
    baseline = statistics.median(r["official"]["median_ms"] for r in measurements)
    candidate = statistics.median(r["tle_v76"]["median_ms"] for r in measurements)
    return {"case": case, "baseline": "official_acp" if case["reference_kind"] == "native" else "official_acp_adapted",
            "latency_base": baseline, "latency": candidate, "speedup": baseline / candidate,
            "accuracy": "bitwise_pass", "rounds": measurements}


def print_row(row):
    case = row["case"]
    shape = ",".join(str(case[k]) for k in ("B", "M", "N", "HQ", "H", "D"))
    print(f'{case["id"]:<5} {shape:<28} {case["dtype"]:<9} {case["reference_kind"]:<7} '
          f'{row["latency_base"]:>10.6f} {row["latency"]:>10.6f} {row["speedup"]:>8.3f}x', flush=True)


def detail_for(row):
    case = row["case"]
    return {"op_name": "forgetting_attention", "dtype": case["dtype"], "mode": "forward", "level": "end_to_end",
            "result": [{"shape_detail": [[case[k] for k in ("B", "M", "N", "HQ", "H", "D")],
                                         {"scale": case["scale"], "threshold": -10.0,
                                          "baseline_kind": row["baseline"]}],
                        "latency_base": row["latency_base"], "latency": row["latency"],
                        "speedup": row["speedup"], "accuracy": row["accuracy"]}]}


@pytest.mark.parametrize("case", DEFAULT_CASES, ids=case_id)
def test_forgetting_attention_benchmark(case, record_property):
    row = benchmark_case(case)
    print_row(row)
    # Optional forward compatibility with PR #70+ recorders; PR #69 ignores this property.
    record_property("flag_attn_benchmark_result", detail_for(row))


def _positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="all 90 historical shape/dtype configurations")
    parser.add_argument("--ids", help="comma-separated historical IDs, e.g. S011,S024")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=_positive, default=40, help="warmup iterations per provider per round")
    parser.add_argument("--rep", type=_positive, default=400, help="CUDA Event samples per provider per round")
    parser.add_argument("--rounds", type=_positive, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional detailed JSON, refuses overwrite")
    args = parser.parse_args(argv)
    if not CUDA_SM90 or not has_tle():
        parser.error("Hopper SM90 CUDA and compatible Triton 3.6 TLE are required")
    if args.output and args.output.exists():
        parser.error(f"refusing overwrite: {args.output}")
    cases = CASES if args.full or args.ids else DEFAULT_CASES
    if args.ids:
        wanted = args.ids.split(",")
        unknown = set(wanted) - {c["id"] for c in CASES}
        if unknown:
            parser.error(f"unknown case IDs: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in wanted]
    if args.dtype:
        cases = [c for c in cases if c["dtype"] == args.dtype]
    if not cases:
        parser.error("no matching cases")
    payload = {"implementation": "V7.6 TLE", "timer": "full_operator_cuda_event",
               "graph_used": False, "official_commit": OFFICIAL_COMMIT,
               "torch": torch.__version__, "triton": triton.__version__,
               "triton_path": triton.__file__, "python": sys.executable,
               "gpu": torch.cuda.get_device_name(), "threshold": -10.0,
               "gate": "logsigmoid(U[0,10])", "seed": args.seed,
               "warmup": args.warmup, "rep": args.rep, "rounds": args.rounds,
               "complete": False, "rows": []}
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}))
    print("ID    B,M,N,Hq,Hkv,D                dtype     ref     official_ms     tle_ms  speedup")
    for case in cases:
        row = benchmark_case(case, args.warmup, args.rep, args.rounds, args.seed)
        payload["rows"].append(row)
        print_row(row)
        print("[INFO] " + json.dumps(detail_for(row)), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
    payload["complete"] = True
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    main()
