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

import argparse
import math

import torch
import torch.nn.functional as F
import triton

from fla.chunk_kda import chunk_kda_fwd_infer

try:
    import flash_kda
except ImportError:
    flash_kda = None

LOWER_BOUND = -5.0
D_HEAD = 128
DEFAULT_HEADS = 96
BENCHMARK_CASES = {
    "fixed-8192": [8192],
    "varlen-skewed": [1300, 547, 2048, 963, 271, 3063],
    "varlen-uniform": [1024] * 8,
}


def _make_inputs(seq_lens: list[int], H: int):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    T = sum(seq_lens)
    N = len(seq_lens)
    q = F.normalize(torch.randn(1, T, H, D_HEAD, device=device), dim=-1).to(dtype)
    k = F.normalize(torch.randn(1, T, H, D_HEAD, device=device), dim=-1).to(dtype)
    v = torch.randn(1, T, H, D_HEAD, device=device, dtype=dtype)
    g = torch.randn(1, T, H, D_HEAD, device=device, dtype=dtype)
    beta = torch.randn(1, T, H, device=device, dtype=dtype)
    # Match FlashKDA's backend benchmark input distribution.
    A_log = torch.rand(H, device=device, dtype=torch.float32)
    dt_bias = torch.rand(H, D_HEAD, device=device, dtype=torch.float32)
    initial_state = torch.randn(N, H, D_HEAD, D_HEAD, device=device)

    cu_seqlens = None
    if N > 1:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(seq_lens).cumsum(0).tolist(),
            device=device,
            dtype=torch.long,
        )

    kwargs = {
        "scale": 1 / math.sqrt(D_HEAD),
        "initial_state": initial_state,
        "output_final_state": True,
        "state_v_first": True,
        "cu_seqlens": cu_seqlens,
        "chunk_size": 16,
        "safe_gate": True,
        "lower_bound": LOWER_BOUND,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    return (q, k, v, g, beta), kwargs


def benchmark_case(
    name: str,
    seq_lens: list[int],
    H: int,
    warmup: int,
    rep: int,
    providers: list[str],
) -> dict[str, tuple[float, float]]:
    args, kwargs = _make_inputs(seq_lens, H)
    results = {}

    for provider in providers:
        if provider == "tle":

            def run():
                with torch.inference_mode():
                    return chunk_kda_fwd_infer(*args, **kwargs)

        else:
            if flash_kda is None:
                raise RuntimeError("--provider flash-kda requires the FlashKDA extension")

            def run():
                # Include output allocation to match chunk_kda_fwd_infer's public API.
                output = torch.empty_like(args[2])
                final_state = torch.empty_like(kwargs["initial_state"])
                with torch.inference_mode():
                    flash_kda.fwd(
                        *args,
                        kwargs["scale"],
                        output,
                        kwargs["A_log"],
                        kwargs["dt_bias"],
                        kwargs["lower_bound"],
                        initial_state=kwargs["initial_state"],
                        final_state=final_state,
                        cu_seqlens=kwargs["cu_seqlens"],
                    )
                return output, final_state

        run()
        torch.cuda.synchronize()
        latency_ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
        token_heads_per_second = sum(seq_lens) * H / latency_ms * 1e3
        results[provider] = latency_ms, token_heads_per_second
        print(
            f"{provider:<10} {name:<18} T={sum(seq_lens):>6} N={len(seq_lens):>2} H={H:>3} "
            f"latency={latency_ms:>8.3f} ms token-head/s={token_heads_per_second:>12.0f}"
        )

    if "tle" in results and "flash-kda" in results:
        speedup = results["flash-kda"][0] / results["tle"][0]
        print(f"{'comparison':<10} {name:<18} TLE speedup vs FlashKDA={speedup:.3f}x")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the TLE chunk KDA kernel against an optional FlashKDA baseline"
    )
    parser.add_argument("--case", choices=BENCHMARK_CASES, action="append")
    parser.add_argument("--seq-lens", type=int, nargs="+", help="custom sequence lengths; overrides --case")
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--provider", choices=("tle", "flash-kda"), action="append")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("chunk_kda benchmark requires CUDA")
    if args.heads <= 0 or args.warmup < 0 or args.rep <= 0:
        parser.error("--heads and --rep must be positive; --warmup must be non-negative")

    if args.seq_lens:
        cases = [("custom", args.seq_lens)]
    else:
        names = args.case or list(BENCHMARK_CASES)
        cases = [(name, BENCHMARK_CASES[name]) for name in names]

    providers = args.provider or ["tle"] + (["flash-kda"] if flash_kda is not None else [])

    torch.manual_seed(42)
    print(f"GPU: {torch.cuda.get_device_name()} | dtype=bfloat16 | D={D_HEAD}")
    for name, seq_lens in cases:
        benchmark_case(name, seq_lens, args.heads, args.warmup, args.rep, providers)


if __name__ == "__main__":
    main()
