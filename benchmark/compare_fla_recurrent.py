# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""FLA common-semantics comparison: L=1, dense input/output FP32 states.

This deliberately uses no indexed cache and no intermediate snapshots.
Both providers load separate immutable initial states and write a preallocated
final state. It is NOT the vLLM indexed-cache workload in the main benchmark.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as fla_kernel
from flag_attn.FLA.gated_delta_rule.fused_recurrent import _launch
from flag_attn.testing.recurrent_gdn import make_case, reference, check_close
from bench_gdn_recurrent import Measurement


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("gdn_shapes.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n", type=int, nargs="*", default=[1, 16, 128])
    p.add_argument("--warmup-ms", type=float, default=1000)
    p.add_argument("--rep-ms", type=float, default=100)
    p.add_argument("--rounds", type=int, default=7)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    rows = []
    for preset in manifest["presets"]:
        h, hv = preset["H"], preset["HV"]
        for n in a.n:
            c = make_case(n, 1, h, hv, strided=False)
            for name in ("q", "k", "v", "g", "beta"):
                c[name] = c[name].reshape(n, 1, *c[name].shape[2:]).contiguous()
            c.update(
                initial_state=c["initial_state"][:n].contiguous(),
                cu_seqlens=None,
                ssm_state_indices=None,
                num_accepted_tokens=None,
                inplace_final_state=False,
            )
            initial = c["initial_state"].clone()
            expected = reference(c)
            fo = torch.empty_like(c["v"])
            to = torch.empty_like(fo)
            fs = torch.empty_like(initial)
            ts = torch.empty_like(initial)

            def fla():
                return fla_kernel[(16, n * hv)](
                    q=c["q"],
                    k=c["k"],
                    v=c["v"],
                    g=c["g"],
                    gk=None,
                    gv=None,
                    beta=c["beta"],
                    A_log=None,
                    dt_bias=None,
                    o=fo,
                    h0=c["initial_state"],
                    ht=fs,
                    cu_seqlens=None,
                    scale=c["scale"],
                    T=1,
                    H=h,
                    HV=hv,
                    K=128,
                    V=128,
                    BK=128,
                    BV=8,
                    IS_BETA_HEADWISE=True,
                    USE_QK_L2NORM_IN_KERNEL=True,
                    STATE_V_FIRST=True,
                    APPLY_BETA_SIGMOID=False,
                    ALLOW_NEG_EIGVAL=False,
                    num_warps=1,
                    num_stages=3,
                )

            def tle():
                return _launch(
                    c["q"],
                    c["k"],
                    c["v"],
                    c["g"],
                    c["beta"],
                    c["scale"],
                    c["initial_state"],
                    False,
                    None,
                    None,
                    None,
                    True,
                    to,
                    ts,
                )

            fla()
            tle()
            check_close(fo, fs, *expected)
            check_close(to, ts, *expected)
            measurements = {
                name: Measurement(fn, c["initial_state"], initial, graph=True, cache="warm", eviction=None)
                for name, fn in (("fla", fla), ("tle", tle))
            }
            for m in measurements.values():
                m.samples(a.warmup_ms)
            rounds = {name: [] for name in measurements}
            for r in range(a.rounds):
                for name in ("fla", "tle") if r % 2 == 0 else ("tle", "fla"):
                    rounds[name].append(statistics.median(measurements[name].samples(a.rep_ms)))
            us = {name: statistics.median(t) for name, t in rounds.items()}
            row = dict(
                preset=preset["id"],
                N=n,
                L=1,
                H=h,
                HV=hv,
                fla_us=us["fla"],
                tle_us=us["tle"],
                speedup=us["fla"] / us["tle"],
                rounds=rounds,
                correctness="passed",
            )
            rows.append(row)
            a.output.write_text(
                json.dumps(
                    dict(
                        fla_module=__import__("fla").__file__,
                        scope="dense L=1 separate immutable input and output state; preallocated; graph/no forced eviction",
                        results=rows,
                    ),
                    indent=2,
                )
            )
            print(json.dumps(row), flush=True)
    print("GEOMEAN", math.exp(statistics.mean(math.log(r["speedup"]) for r in rows)), flush=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
