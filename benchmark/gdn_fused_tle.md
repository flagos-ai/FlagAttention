# Fused GDN forward with TLE pipes

The Hopper fast path prepares gate prefix sums and ungated inverse tiles in one
kernel, retaining the triangular solve in registers. A second kernel prefetches
inputs through TLE pipes, runs the recurrent state/value computation in one
worker, and overlaps output computation in another worker. Every pipe owns its
storage; consumers retain slots until dependent results have been materialized.
There is no global `h` or `v_new` intermediate.

This replaces the experimental v3 design that shared A storage across independent
pipes and split the recurrent dependency chain across several workers. The
regression test compares repeated identical calls bit for bit in addition to
checking output and final-state RMS error against external FLA (limit 1%).

Supported fast path: Hopper, FP16/BF16, fixed-length sequences divisible by 64,
T >= 2048, K=V=128, B*H >= 32, no initial state, final state requested. The public
wrapper retains its layout conversion and optional q/k normalization. Other
inputs use the existing implementation. Existing input validation and forward-only
API behavior remain unchanged. The block solve uses TF32, matching the original
FLA v3 preprocessing; mixed-precision outputs are checked against FLA.

Set `FLAG_ATTN_CHUNK_GDR_FUSED_TLE=0` to compare the unchanged upstream dispatch.
The existing `FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE=0` also disables this path.

## Validation environment and reproduction

NVIDIA H100 80GB HBM3; PyTorch 2.10.0+cu128; FlagTree's Python 3.12 build with
Triton/TLE pipes. Base commit: `54417c6e9e4f85cdca248981ed18dc5b9fa5eb5d`.
Install external FLA for the reference tests and benchmark. Put the FlagTree
build and this repository's `src` on PYTHONPATH **before starting Python**.

```sh
python -m pytest tests/flag_attn/test_gdn_fused_tle.py -q
python benchmark/bench_gdn_fused_tle.py
```

The benchmark uses inference mode, three warmup calls, then Triton do_bench with
100 ms warmup, 300 ms measurement and median latency. Measurements are sequential
on an otherwise idle GPU and include preprocessing and final-state writes.

| dtype | B,T,H (D=128) | upstream ms | fused ms | speedup |
|---|---|---:|---:|---:|
| FP16 | 2,16384,16 | 0.8622 | 0.8816 | 0.978x |
| FP16 | 4,2048,16 | 0.2276 | 0.1772 | 1.284x |
| FP16 | 4,4096,64 | 1.6221 | 1.0001 | 1.622x |
| BF16 | 2,16384,16 | 0.8532 | 0.8811 | 0.968x |
| BF16 | 4,2048,16 | 0.2261 | 0.1796 | 1.259x |
| BF16 | 4,4096,64 | 1.6213 | 1.0038 | 1.615x |

The long low-head-count shape has a 2–3% latency regression against current
upstream and is not claimed as a speedup. All three shapes improve against the
original FLA baseline. Hardware beyond H100 is not validated.
