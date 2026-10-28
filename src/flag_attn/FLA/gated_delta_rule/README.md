# GDN forward operators

The three inference entry points cover the same scalar GDN recurrence:

```
decayed = exp(g) * state
delta = beta * (v - decayed @ k)
state = decayed + delta outer k
output = state @ (scale * q)
```

Grouped heads map each value head to its corresponding Q/K head. Optional
Q/K normalization uses the entry point's existing contract. GDN2 and KDA
have different gates and are not substituted for this recurrence.

| Entry point | Workload and inputs | State contract |
|---|---|---|
| `chunk_gated_delta_rule_fwd` | Chunk prefill, precomputed log gate `g` and `beta` | Input/final `[N,HV,K,V]`; preserves seven raw forward results, including cumulative gates and gated inverse |
| `fused_recurrent_gated_delta_rule_fwd` | Stateful single/multi-token decode, packed lengths, precomputed `g`/`beta` | Indexed `[pool,HV,V,K]`, accepted-prefix state selection, all requested new-token snapshots |
| `fused_recurrent_gated_delta_rule_packed_decode` | One token per request, packed QKV and raw gate projections | Same indexed cache ABI, fused gate transform, caller-provided output, in-place cache update |

All three names are exported from `flag_attn` and this package. The higher-level
`chunk_gated_delta_rule` retains layout conversion, normalization and fallback
dispatch. No vLLM engine route is changed by importing this library.

## Shared implementation and state handoff

The two decode wrappers use one recurrent TLE kernel. Packed decode specializes
it to one token and reads Q/K/V from one projection pointer. It computes
`g=-exp(A_log)*softplus(a+dt_bias, threshold=20)` and `beta=sigmoid(b)` in the
kernel, without allocating intermediate gate or QKV buffers. A selected initial
index `<=0` produces zero output for that request and leaves cache unchanged.
In recurrent multi-token calls, a nonpositive per-token destination skips only
that cache store while keeping the recurrence and output. In-place writes must
not overlap another request's reads or writes. Metadata validation does not copy
device values to the host.

Chunk prefill uses a shared register-only KKT/triangular preparation and a
guarded Hopper pipeline with independent pipe storage and retained zero initialization.
Both public and raw chunk dispatch share the same
eligibility predicate. When the raw API needs its gated inverse, preparation
restores gate factors from the FP32 inverse blocks in the same kernel. The
public API avoids this auxiliary allocation when it is not requested. Existing
chunk fallback dispatch also uses this preparation on Hopper, including ragged
and tail inputs. This removes a materialized KKT readback that showed
allocation-sensitive errors in the previous implementation. Prefix gates are
kept in FP32; invalid load lanes and unwritten upper inverse tiles are zero.

Chunk final states use `[K,V]` orientation; recurrent caches use `[V,K]`.
Transpose the last two dimensions and cast to the cache dtype before placing
a prefill state in its decode cache slot. `tests/flag_attn/test_gdn_forward.py` checks prefill -> packed decode
-> accepted-prefix multi-token decode, as well as the raw chunk return contract.

## Scope and precision

The new chunk pipeline requires Hopper, TLE pipes, FP16/BF16 inputs, K=V=128,
dense T>=2048 divisible by 64, B*HV>=32, no initial state and a requested final
state. The raw API accepts gates/beta in the QKV dtype or FP32; the higher-level
wrapper keeps its existing same-dtype validation. Other chunk calls retain their existing recurrence/output dispatch, with
shared preparation where supported. Hopper wide-key calls (K>128) use the staged
path because the older fused route regresses there; legacy benchmark policy is
restored with its pinned predicate. FP32 inverse merges now default to IEEE;
the previous TF32 choice is an explicit preparation ablation. Decode uses FP32 recurrence
without TF32. Its K/V range is 1..256, with FP16/BF16/FP32 inputs and states;
N and HV must each fit the 65535 CUDA Y/Z grid limit. Packed projection/gate
rows may be padded, with contiguous last dimensions. State strides are honored.
Address products use int64 when a relative tensor span exceeds signed int32;
ordinary layouts keep their existing address arithmetic.

Chunk calls requesting materialized `w/h/v_new` via
`FLAG_ATTN_GDN_RECOMPUTE_SUPPRESS_LEVEL=3` retain the staged path.

The decode APIs support forward inference only. They reject gradient-enabled
inputs requiring gradients; they do not provide a backward implementation.
Caller-provided output/cache allocations must not overlap other input buffers.

## Unified validation and measurement

With the verified FlagTree build on PYTHONPATH before Python starts:

The chunk external-reference tests and prefill benchmark also need FLA;
tests skip that optional comparison when FLA is unavailable.

```sh
python -m pytest tests/flag_attn/test_gdn_forward.py \
  tests/flag_attn/test_gdn_preparation.py \
  tests/flag_attn/test_gdn_fused_tle.py \
  tests/flag_attn/test_gdn_packed_decode.py \
  tests/flag_attn/test_fused_recurrent_gdn.py \
  tests/flag_attn/test_gated_delta_rule.py -q

python -m pytest --record json -m fused_recurrent_gated_delta_rule_fwd
python -m pytest --record json -m fused_recurrent_gated_delta_rule_packed_decode

python benchmark/bench_gdn_forward.py --operator all --cache cold \
  --rounds 7 --warmup-ms 1000 --rep-ms 100 --output results/cold
python benchmark/bench_gdn_forward.py --operator all --cache warm \
  --rounds 7 --warmup-ms 1000 --rep-ms 100 --output results/warm

# FP32 precomputed gates match the raw vLLM gate ABI.
python benchmark/bench_gdn_forward.py --operator chunk --gate-dtype float32 \
  --cache cold --output results/prefill-fp32-gates.json

# BF16 state is a distinct cache configuration, using identical state precision
# in both providers. Run both decode entries with the same frozen shapes.
python benchmark/bench_gdn_forward.py --operator recurrent --state-dtype bfloat16 \
  --cache cold --output results/bf16-state-recurrent.json
python benchmark/bench_gdn_forward.py --operator packed --state-dtype bfloat16 \
  --cache cold --output results/bf16-state-packed.json
```

`benchmark/gdn_shapes.json` freezes the per-rank head geometries, request/token
counts, prefill lengths, and historical regression shapes. Official model
configuration revisions are locked in `benchmark/gdn_model_sources.json`.
The decode benchmark defaults to explicit FP32 state for continuity with the
original protocol. Also report BF16 state separately: in the pinned vLLM
configuration, `mamba_ssm_cache_dtype=auto` follows the model/cache dtype, so
an otherwise unmodified BF16 configuration resolves to BF16 recurrent state.
An explicit FP32 cache setting remains supported. Both providers always receive
the same state dtype; changing it is not credited as an implementation speedup.
The canonical runner executes phases serially and uses the same state-safe
single-call CUDA Graph/event sampler. State restoration and optional 256 MiB
eviction occur outside timing; required initialization/output/state stores
remain timed. Graph replay does not measure Python validation/allocation cost.
Samples and per-round medians are retained.

Prefill compares a staged recurrence/output control with both the legacy and new
fused mains disabled, and reports external FLA separately. The new eligibility
check is independent of the legacy TLE switch. It also checks the upstream
dispatch policy with the common boundary zero-fill fixes by disabling both
`FLAG_ATTN_CHUNK_GDR_FUSED_TLE` and `FLAG_ATTN_GDN_REGISTER_PREP`. Cases in that policy
that fail accuracy or bitwise repeatability are retained as failures and excluded
from speedup baselines. Decode compares the pinned vLLM source under
the same compiler; only imports/platform queries are adapted. The vendored vLLM
functions are pinned to [0232bb3060cb345912aedb96e7eec2a2d8fc9741](https://github.com/vllm-project/vllm/blob/0232bb3060cb345912aedb96e7eec2a2d8fc9741/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py).
The measured external FLA checkout is `df587d1df9772e5c854a2b3bff73ef83d4bdfc88`;
its imported module paths and hashes are recorded in every prefill table. All cases check
outputs and requested states before timing. The standalone decode module also
provides non-Graph, layout, load and configuration ablations. FLA dense L=1
common-contract measurements remain a separate supplementary comparison.

Keep prefill, recurrent/MTP and packed decode results separate. Model geometry
does not establish deployment frequency. Static shape averages do not measure
whole-model TTFT/TPOT, and no production-frequency weighting is assumed.
