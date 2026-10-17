# Ascend alignment to locked upstream (2026-10-16)

## Scope and rollback

- Fork baseline: `cc675a5f7f456416ab75e08ac48b079c774d533f`.
- Locked upstream: `17a2f54463e61f1906f3a684dc1da7413938fffd`.
- Common ancestor: `d9adc17cfc2b4ac3776537548886da3de20e8085`.
- Divergence: fork 15 commits, upstream 20 commits.
- Local migration branch: `chore/align-upstream-20261016`.
- Local backup: `backup/npu-ghd-pre-merge` at the fork baseline.
- During the original migration, `npu-ghd` stayed at its baseline and no push
  was authorized. On 2026-10-16 the user subsequently authorized documenting
  this checkpoint and publishing it to the fork's `npu-ghd` branch, without a PR.
- Rollback: use the backup checkout. Kernel sources, math, tile sizes, launch
  configuration and numerical policies are preserved during migration.

## Environment

Validated inside server container `ghd-sageattn`, project
`/data/ghd/pr/flagattention-sage-integration`:

| Component | Observed value |
| --- | --- |
| Python | `/usr/local/python3.11.15/bin/python` |
| PyTorch | `2.9.0+cpu` (NPU extension available) |
| torch_npu | `2.9.0.post2` |
| Triton | `3.5.1` (Ascend/TLE build) |
| Chip | `Ascend910B4-1` |
| Device selection | `ASCEND_RT_VISIBLE_DEVICES=4` |
| CANN | `latest` resolves to `/usr/local/Ascend/cann-9.0.0` |

The locked upstream CI specifies runner `910b-flaggems-weekly`, container
`harbor.baai.ac.cn/flagos-app/vllm0.24.0-ascend-cann9.0.0:2.2.0-0.3.0`.
Local validation does not prove that GitHub self-hosted CI is accessible.

## P0 audit and file ownership

Both the original local branch and server worktree were clean at baseline.
The origin branch was fetched and confirmed at `cc675a5`; the upstream object
was fetched by its locked SHA, without advancing the target to a newer main.

| Classification | Paths | Action |
| --- | --- | --- |
| Inherit upstream | Public namespaces, `op_registry.py`, `testing/backend.py`, runtime, inventory, runner, packaging, CI and generic tests/benchmarks | Keep the locked upstream framework |
| Preserve fork | Ascend GLA, KDA, NSA, MiniMax and common primitives listed below | Keep implementations; register only after checking contracts |
| Manual integration | `_ascend/__init__.py`, GDN2 exports and validation, both Sage sources, `ops/__init__.py`, `.gitignore` | Keep fork fixes and adopt registry entry points |
| Rename-aware integration | `benchmark/parallel_nsa_benchmark.py` -> upstream `benchmark/test_parallel_nsa_benchmark.py` | Retain upstream rename and review lost fork content before accepting |

The 16 fork-only paths in a no-rename two-tree comparison are:

1. `benchmark/parallel_nsa_benchmark.py` (upstream renamed this benchmark).
2. `src/flag_attn/runtime/backend/_ascend/gla/__init__.py`.
3. `src/flag_attn/runtime/backend/_ascend/gla/chunk.py`.
4. `src/flag_attn/runtime/backend/_ascend/kda/__init__.py`.
5. `src/flag_attn/runtime/backend/_ascend/kda/chunk.py`.
6. `src/flag_attn/runtime/backend/_ascend/kda/forward.py`.
7. `src/flag_attn/runtime/backend/_ascend/minimax_sparse_attention/__init__.py`.
8. `src/flag_attn/runtime/backend/_ascend/minimax_sparse_attention/index_topk.py`.
9. `src/flag_attn/runtime/backend/_ascend/minimax_sparse_attention/sparse_attn.py`.
10. `src/flag_attn/runtime/backend/_ascend/nsa/__init__.py`.
11. `src/flag_attn/runtime/backend/_ascend/nsa/compression.py`.
12. `src/flag_attn/runtime/backend/_ascend/nsa/forward.py`.
13. `src/flag_attn/runtime/backend/_ascend/nsa/kernels.py`.
14. `src/flag_attn/runtime/backend/_ascend/ops/bwd_preprocess.py`.
15. `src/flag_attn/runtime/backend/_ascend/ops/cumsum.py`.
16. `src/flag_attn/runtime/backend/_ascend/ops/mean_pooling.py`.

## Migration-before validation

Logs remain local to the server in ignored
`BaselineBenchmark/migration-20261016/`. They must not enter Git.

```sh
ASCEND_RT_VISIBLE_DEVICES=4 PYTHONPATH=src:. /usr/local/python3.11.15/bin/python \
  -m pytest BaselineBenchmark/validation-current/test_gdn2.py \
  src/flag_attn/runtime/backend/_ascend/test_sage_attention.py -q --disable-warnings
```

Result: **10 passed, 1 warning in 13.58s**, exit 0; `pre-accuracy.log`.
The suite checks GDN2 lengths 16/33 against recurrence and Sage layouts,
grouped heads, masks and partial blocks. It does not prove all supported shapes.

Performance protocol: fixed seed 42, same physical device, synchronized wall
time including host dispatch; 5 warmups and 30 repetitions, median/P10/P90.
GDN2 `[B=1,T=128,H=1,K=64,V=64]` BF16; Sage `[B=1,H=2,T=128/256,D=64]`
FP16, HND, unmasked, quantization excluded. Source and log are
`protocol.py` and `pre-performance.log` in the same ignored directory.

Before-migration latency (ms):

| Operator | P10 | Median | P90 |
| --- | ---: | ---: | ---: |
| GDN2 T128 | 0.637028 | 0.640668 | 0.661935 |
| Sage T128 | 0.296561 | 0.299299 | 0.312621 |
| Sage T256 | 0.297467 | 0.300845 | 0.312095 |

Protocol exit 0. The old NSA benchmark is byte-identical to the common
ancestor; accepting its upstream rename does not drop fork-specific changes.

## Registration audit (P2)

| Operator | Contract boundary | Initial decision |
| --- | --- | --- |
| GDN2 | Inference only, BT=16 | Registered through `gdn2/api.py`, exact public parameter order; differentiable calls fail explicitly; original internal entry preserved |
| Sage | Public forward aliases and shared quantization; existing masked framework path | Registered with upstream aliases, fork mask/maxnreg behavior preserved |
| GLA | Same parameters/defaults, output/final state, variable length and backward | Registered to `gla.chunk_gla`; BLOCKED in P3: existing shared state kernel has no Ascend tuning entries (BK/BV) |
| KDA | Registry key `chunk_kda_fwd_infer`; Ascend wrapper has distinct defaults/options | Registered through matching inference signature, preserves `safe_gate=False` default; rejects precomputed chunk_indices explicitly |
| NSA selected | Returns output/LSE, lacks full backward API | Keep internal; do not register as full `parallel_nsa` |
| NSA compression | Matching signature/output but backward is unreliable in the full regression suite | Intentionally NOT registered after P3 evidence; original internal implementation is retained. See failures below |
| MiniMax | Prefill and four index primitives match parameter/default/return contracts | Registered prefill and index primitives; decode attention intentionally not registered because fork simply reexports Generic; no independent Ascend decode implementation |
| ops | Internal shared primitives without public generic contracts | Keep internal |

## Stage gates

- P0: PASS; audit, backup and migration branches exist; pre-migration accuracy
  and fixed-shape timing captured on 910B; original `npu-ghd` unchanged.
- P1: PASS. Merged the locked upstream, resolving `.gitignore` by retaining
  local MLA exclusion and Sage forward by preserving fork mask/maxnreg fixes
  plus upstream's named grid function. All 15 unique Ascend source files remain.
  `python -m compileall -q src` passed locally and on the server.
  Runtime/runner pytest initially had 36 pass / 1 failure: upstream #99 added
  SDK metadata without updating the environment dictionary assertion. Added
  a deterministic SDK probe mock/expectation; rerun: **37 passed**, exit 0.
  Logs: `/tmp/migration-p1-tests{,-rerun}.log` in the container.
  Local Windows lacks torch/Triton; real imports/tests ran in the existing
  Ascend container using the isolated migration directory.
- P2: dispatch/contracts PASS, NPU correctness pending P3. Added adapters and
  the single `_ascend._OPERATOR_EXPORTS` table; no second dispatcher. Registry
  override/fallback/dependency propagation, exact adapter parameter order,
  inference guards and internal primitive isolation tested with CPU tensors/mocks.
  Runtime + runner + registry: **52 passed**, exit 0, `/tmp/migration-p2-tests.log`.
  These results do not establish NPU mathematical correctness.
- P3: **NOT PASSED**. See acceptance evidence below. Work is checkpointed for
  review, not declared complete. P4 acceptance has not started because P3 is red.

## P3 evidence and remaining blockers

### Changes

- Added `tests/flag_attn/test_ascend_attention.py` using public APIs and independent
  CPU references. Tests cover GDN2 FP16/BF16, lengths 16/33, nonzero initial
  state, both state layouts, shape/dtype, invalid BT; GLA output/state/gradients;
  KDA recurrence and MiniMax stage-facade/index/prefill.
- CUDA native/TLE GDN2 comparison tests are explicitly skipped on Ascend,
  alongside separately executed NPU recurrence tests; skipped is not passed.
- Extended existing inventory entries, without creating a second dispatcher.
- Added inventory-path scoping to the existing runner so unrelated optional
  modules are not eagerly imported. Added `--skip-benchmarks` with an explicit
  `Skipped` performance result (the locked upstream did not accept this flag).
- Added matching runner regressions. Initial environment import required
  `distro==1.9.0`, installed only in `/tmp/flagattention-migration-deps`.
- GDN2 V-first initial-state recurrence failed at relative RMS 0.0256607;
  reproduced the exact error using the original fork implementation. Adapted
  public V/K states to K/V before calling the unchanged kernel and transposed
  final/intermediate states back. The internal kernel/API remains preserved.
- NSA compression was initially registered, then withdrawn after a reproducible
  full-suite backward failure. Its internal kernels remain intact.

### Commands and results

All NPU tests use physical device 4 in `ghd-sageattn`; source checkout used
`/data/ghd/pr/flagattention-ascend-migration-20261016`. Validation source was
transferred from the local migration branch; original `npu-ghd` is untouched.

```sh
export ASCEND_RT_VISIBLE_DEVICES=4 FLAG_ATTN_BACKEND=ascend PYTHONPATH=src:.
/usr/local/python3.11.15/bin/python -m pytest -q \
  tests/flag_attn/test_runtime.py tests/flag_attn/test_test_runner.py \
  tests/flag_attn/test_ascend_registry.py --disable-warnings
```

Result before final registry withdrawal: **54 passed**, `/tmp/migration-p3-host-tests.log`
on the host. Final registry withdrawal rerun: **53 passed**, exit 0,
host `/tmp/migration-final-host-tests.log`.

```sh
PATH=/usr/local/python3.11.15/bin:$PATH \
PYTHONPATH=src:.:/tmp/flagattention-migration-deps \
/usr/local/python3.11.15/bin/python tools/run_tests.py \
  --ops chunk_gdn2,sage_attention --skip-benchmarks --dump-output \
  --gpus 0 --output-dir /tmp/migration-p3-runner-cann --color never
```

Result: GDN2 **9 passed / 83 skipped**, Sage **7 passed**, both accuracy entries
**Passed**; performance entries **Skipped**.
Read JSON per-operator statuses: upstream runner can return shell exit 0 even
when an operator fails. CUDA-only GDN2 comparisons (83 cases) remain skipped.
Runner logs and JSON are in the container at `/tmp/migration-p3-runner-cann/`.
The first runner attempt failed to import `distro`; the next rejected the
missing CLI flag; a later run lost the CANN compiler from a manually overridden
PATH. None of these failed runs is counted as a pass. The command above retains
CANN PATH and is the successful execution.

```sh
/usr/local/python3.11.15/bin/python -m pytest -q \
  tests/flag_attn/test_ascend_attention.py --disable-warnings
```

Result after GDN2 layout adapter: **12 passed, 2 failed**. Failures are GLA
lengths 17/32: `dynamic_func() missing ... BK and BV`. Both old and new
`runtime.get_tuned_config('chunk_fwd_kernel_h')` return `[]`, and neither
checkout contains `_ascend/tune_configs.yaml`. Supplying new launch/tile
configuration is outside this task's explicit no-tuning constraint. Host log:
`/tmp/migration-p3-ascend-rerun-host.log`. This is a configuration BLOCKED
condition backed by real test failures, not a missing-hardware skip.

```sh
/usr/local/python3.11.15/bin/python -m pytest -q \
  tests/flag_attn/test_sage_attention.py \
  tests/flag_attn/test_parallel_nsa_compression.py --disable-warnings
```

Initial result: **14 passed, 1 failed**; complete rerun: **13 passed, 2 failed**.
NSA compression backward at T128/T256 has nonfinite gradients in the full suite;
the T128 case passed in isolation on old and new sources, and passed three
additional isolated runs. Cause is not established; do not label it a migration
math regression or claim stable correctness. Container logs:
`/tmp/migration-public-existing{,-rerun}.log`,
`/tmp/migration-{old-compression,new-compression-rerun}.log`,
`/tmp/migration-nsa-repeat-{1,2,3}.log`.

Direct project benchmark functions were invoked through `runpy.run_path`, since
the upstream modules have pytest entry points and no script `__main__`:

```python
import runpy
runpy.run_path('benchmark/test_gdn2_benchmark.py')['run_benchmark'](
    ['--shape', '1,128,1,64,64', '--warmup', '5', '--rep', '20'])
m = runpy.run_path('benchmark/test_sage_attention_benchmark.py')
m['benchmark'](m['parse_args']([
    '--batch-size', '1', '--num-heads', '2', '--head-dim', '64',
    '--seq-lens', '128', '256', '--warmup', '5', '--rep', '20', '--dtype', 'float16']))
```

GDN2 BF16 public API: **0.715145 ms** (NPU event timing, no native CUDA baseline).
Sage FP16 T128/T256: **0.318420 / 0.321860 ms**; cosine
**0.999930 / 0.999925**, relative L2 **0.011816 / 0.012276**, within the upstream
benchmark's thresholds. SDPA speedups on these small cases are **0.419 / 0.410**,
so these cases are slower than SDPA. Host logs:
`/tmp/migration-{gdn,sage}-bench-host.log`. These use a different timing protocol
from P0 and must NOT be compared directly to P0 or called no-regression proof.

### Static checks and limitations

- `compileall` and `git diff --check` passed.
- Structural AST comparison of all **43** existing Ascend JIT functions against
  `cc675a5` found **zero changed kernels**.
- Ruff passes for the new adapters/tests, backend export table and modified
  GDN2/runner tests. Full runner-file Ruff reports **35** pre-existing findings;
  running the same checker on locked upstream produced the same **35**
  code/message pairs: **zero new findings**. They are not fixed through
  unrelated cleanup.
- Repository pre-commit execution is **BLOCKED** while fetching hook repos:
  the configured global Git proxy at `127.0.0.1:7890` is unreachable. Did not
  alter global Git settings or disable hooks to claim success.
- GitHub CI runner/container permissions and actual self-hosted workflow run
  are **BLOCKED / not verified**; the locked 910B CI config was inherited intact.
- P4 before/after performance acceptance and complete migration sign-off remain
  **PENDING**. The subsequent forward-only scope excludes NSA backward from
  acceptance, but does not resolve the GLA forward failure or missing coverage.

## Superseding forward-only checkpoint (2026-10-16)

The user subsequently limited acceptance to **forward for every operator**.
NSA backward failures above remain historical evidence, not current acceptance
blockers. This scope update is documentation-only: it does not restore the NSA
compression registry override, change tests, or represent a new hardware run.
See [the forward-only checkpoint](forward-only-checkpoint-20261016.md) for the
current per-operator status, architecture, and authorized publication scope.

## Next scoped work

1. Provide previously validated Ascend state-kernel tuning metadata, or explicitly
   authorize restoring the missing launch configuration. Do not invent/tune it
   under the current task constraints.
2. Under the forward-only scope, restore and validate the appropriate NSA public
   inference registration in a separate change; preserve the implementation here.
   Backward diagnosis is not required by the current acceptance scope.
3. Rerun P3, then perform P4 same-protocol before/after comparisons and final CI
   acceptance. No kernel performance optimization has been performed. Publication
   of this checkpoint is now authorized; this document alone is not proof that
   the push succeeded.
