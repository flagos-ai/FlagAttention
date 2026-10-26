# Add a fused Hopper GDN forward path with TLE pipes

Gated delta rule forward evaluation materializes intermediate state and value tensors between stages. This PR adds a Hopper TLE path that keeps the recurrent state in registers, passes chunk data through shared-memory pipes, and overlaps output computation with the recurrence to reduce intermediate global-memory traffic.

## Implementation

- Fuse gate prefix sums, lower-triangular KKT construction, and the triangular solve into one preparation kernel. Keep the solve's intermediate blocks in registers.
- Compute ungated inverse tiles and restore gate factors in the main kernel using the diagonal-similarity identity `(I + D L D^-1)^-1 = D (I + L)^-1 D^-1`. Gate contributions are relocated, rather than omitted.
- Use independent input producers and separate recurrence/output workers. Each pipe owns its backing storage; named readers retain their slots until their dependent computations finish.
- Keep the recurrent state in FP32 registers, retaining zero initialization and the required zero-fill for unwritten upper-triangular tiles. Write the full output and requested final state.
- Add guarded API dispatch, repeated-call regression tests, and a forward benchmark.

## Scope and precision

The new path requires Hopper, TLE pipes, FP16/BF16 inputs, `K=V=128`, dense sequences with `T>=2048` divisible by 64, `B*H>=32`, no initial state, and `output_final_state=True`. Existing paths handle other inputs. The wrapper retains grouped-value-head mapping, layout conversion, and optional q/k normalization.

This is a dimension-specialized forward path. It does not select individual benchmark tuples or inspect input values. Aligned sequences need no padding within this path; sequences with tails use existing dispatch.

The FP32 block-merge dots in the preparation kernel currently use **TF32**. This precision choice is part of the implementation and must be assessed separately from fusion and scheduling. Algebraic equivalence does not imply identical floating-point results.

Set `FLAG_ATTN_CHUNK_GDR_FUSED_TLE=0` to disable the new path. The existing `FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE` switch also applies. This PR adds no backward implementation.

## Validation

On NVIDIA H100 80GB HBM3, PyTorch 2.10.0+cu128, and FlagTree Triton/TLE 3.6.0:

- The existing and new GDN test suites completed with **74 passed**.
- The new tests compare output and final-state relative RMS error against external FLA with a 1% threshold, and check bitwise repeatability across repeated identical calls. Fast-path regression cases assert that the new kernel was invoked.
- Additional audit cases exercised unseen batch/sequence/head combinations, uniform inputs, zero values, an impulse input, zero gates, and stronger decay. Completed allocation-poison checks filled floating `torch.empty` allocations with NaNs and verified unchanged results, while preserving required zero initialization.
- Independent FP32 sequential-recurrence spot checks were also run. For the BF16 zero-gate impulse case `(B,T,H)=(1,2048,32)`, the new path's output and final-state relative RMS errors were 0.319% and 0.804%. These spot checks do not establish exhaustive numerical coverage.

```sh
python -m pytest tests/flag_attn/test_gdn_fused_tle.py -q
python benchmark/bench_gdn_fused_tle.py
```

The reference tests and benchmark require external FLA. Put the FlagTree build and this repository's `src` on `PYTHONPATH` before starting Python.

## Performance

End-to-end forward latency on H100, including preparation, allocation, and final-state writes. Inputs and options are identical within each comparison, with `K=V=128`, zero initial state, and final state requested. Measurements use inference mode, three warmup calls, and Triton `do_bench` with 100 ms warmup, 300 ms measurement, and median latency.

The table compares external FLA, the existing FlagAttention GDN implementation from [PR #40](https://github.com/flagos-ai/FlagAttention/pull/40), and the new path. FlagAttention was measured at commit `54417c6e9e4f85cdca248981ed18dc5b9fa5eb5d`, which contains the merged PR #40 implementation, with the new path disabled. These are fresh measurements from the same benchmark run, rather than numbers copied from PR #40's description.

| dtype | B,T,H | FLA baseline ms | FlagAttention (PR #40) ms | New path ms | Speedup vs FLA | Speedup vs PR #40 |
|---|---|---:|---:|---:|---:|---:|
| FP16 | 2,16384,16 | 1.1419 | 0.8622 | 0.8816 | 1.295x | 0.978x |
| FP16 | 4,2048,16 | 0.3823 | 0.2276 | 0.1772 | 2.157x | 1.284x |
| FP16 | 4,4096,64 | 2.0693 | 1.6221 | 1.0001 | 2.069x | 1.622x |
| BF16 | 2,16384,16 | 1.1454 | 0.8532 | 0.8811 | 1.300x | 0.968x |
| BF16 | 4,2048,16 | 0.3778 | 0.2261 | 0.1796 | 2.104x | 1.259x |
| BF16 | 4,4096,64 | 2.0698 | 1.6213 | 1.0038 | 2.062x | 1.615x |

Speedup is baseline latency divided by new-path latency, calculated from unrounded measurements.

The long, low-head-count case regresses by approximately 2–3% against FlagAttention. This PR does not claim a speedup for every shape.

Supplementary measurements compare with the unchanged dense GDN operator sources from [vLLM commit f3871c3](https://github.com/vllm-project/vllm/tree/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/third_party/flash_linear_attention/ops), using the same FlagTree compiler environment. A standalone harness loads the operator sources without the vLLM server; these are operator measurements, not full-engine measurements. Each entry below is the median of three benchmark runs.

| dtype | B,T,H | vLLM ms | New path ms | Speedup |
|---|---|---:|---:|---:|
| FP16 | 2,16384,16 | 1.2774 | 0.8796 | 1.452x |
| FP16 | 4,2048,16 | 0.3455 | 0.1782 | 1.939x |
| FP16 | 4,4096,64 | 2.4612 | 0.9982 | 2.466x |
| BF16 | 2,16384,16 | 1.2688 | 0.8808 | 1.441x |
| BF16 | 4,2048,16 | 0.3457 | 0.1786 | 1.936x |
| BF16 | 4,4096,64 | 2.4594 | 1.0004 | 2.458x |

vLLM's triangular solve defaults to IEEE, whereas the new path above uses TF32 for block merging. An additional ablation changed only those block-merge dots to IEEE; the measured speedups against vLLM were 1.359–2.174x across the same six cases. That ablation is **not** the implementation included in this patch.

Only H100 has been validated. The committed benchmark compares FlagAttention and external FLA; the supplementary vLLM harness is currently outside this patch.
