import argparse
import json
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent))
from flag_attn.testing.recurrent_gdn import make_case, reference, check_close
from baselines.vllm_recurrent import fused_recurrent_gated_delta_rule_fwd as baseline
from flag_attn.FLA.gated_delta_rule.fused_recurrent import _launch, fused_recurrent_gated_delta_rule_fwd

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--length", type=int, default=4)
parser.add_argument("--h", type=int, default=4)
parser.add_argument("--hv", type=int, default=8)
parser.add_argument("--sweep", action="store_true")
parser.add_argument("--compact", action="store_true")
parser.add_argument("--profile", choices=["base", "tle"])
args = parser.parse_args()
c = make_case(args.n, args.length, args.h, args.hv, strided=True, spec=args.length > 1)
ref_o, ref_s = reference(c)
original = c["initial_state"].clone()
o, s = fused_recurrent_gated_delta_rule_fwd(**c)
check_close(o, s, ref_o, ref_s)
print("candidate correctness passed", flush=True)
c["initial_state"].copy_(original)
bc = {**c, **{x: c[x].contiguous() for x in ("q", "k", "v", "g", "beta")}}
bo, bs = baseline(**bc)
check_close(bo, bs, ref_o, ref_s)
print("baseline correctness passed", flush=True)
out = torch.empty_like(c["v"])
c["initial_state"].copy_(original)


def base_launch():
    from baselines.vllm_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as kernel

    N = args.n
    return kernel[(1, triton.cdiv(c["v"].shape[-1], 32), N * args.hv)](
        q=bc["q"],
        k=bc["k"],
        v=bc["v"],
        g=bc["g"],
        beta=bc["beta"],
        o=out,
        h0=c["initial_state"],
        ht=c["initial_state"],
        cu_seqlens=c["cu_seqlens"],
        ssm_state_indices=c["ssm_state_indices"],
        num_accepted_tokens=c["num_accepted_tokens"],
        scale=c["scale"],
        N=N,
        T=c["q"].shape[1],
        B=1,
        H=args.h,
        HV=args.hv,
        K=128,
        V=128,
        BK=128,
        BV=32,
        stride_init_state_token=c["initial_state"].stride(0),
        stride_final_state_token=c["initial_state"].stride(0),
        stride_indices_seq=c["ssm_state_indices"].stride(0),
        stride_indices_tok=1,
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=False,
        num_warps=1,
        num_stages=3,
    )


def candidate_launch(config, use_tle=True):
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


if args.profile:
    fn = base_launch if args.profile == "base" else lambda: candidate_launch(None)
    fn()
    torch.cuda.synchronize()
    print("PROFILE_READY", flush=True)
    torch.cuda.nvtx.range_push("target")
    fn()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    sys.exit()


def measure(fn):
    c["initial_state"].copy_(original)
    fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            start.record()
            fn()
            end.record()
    torch.cuda.current_stream().wait_stream(stream)
    times = []
    for i in range(30):
        c["initial_state"].copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return sorted(times)[len(times) // 2]


print(json.dumps({"baseline_us": measure(base_launch)}), flush=True)
configs = (
    [(4, 1, False), (8, 1, False), (16, 1, False), (32, 1, False), (16, 2, False)]
    if args.compact
    else [(16, 4, False)]
    if not args.sweep
    else [
        (bv, w, a)
        for bv in (4, 8, 16, 32, 64, 128)
        for w in (1, 2, 4, 8)
        for a in (False, True)
        if bv * 128 // (w * 32) >= 4
    ]
)
rows = []
for cfg in configs:
    try:
        c["initial_state"].copy_(original)
        kern = candidate_launch(cfg)
        check_close(out, c["initial_state"], ref_o, ref_s)
        us = measure(lambda: candidate_launch(cfg))
        row = dict(config=cfg, us=us, regs=kern.n_regs, spills=kern.n_spills, shared=kern.metadata.shared)
        rows.append(row)
        print(json.dumps(row), flush=True)
    except Exception as e:
        print(json.dumps({"config": cfg, "error": str(e)}), flush=True)
print("BEST", json.dumps(sorted(rows, key=lambda x: x["us"])[:8]), flush=True)
