# Ascend forward acceptance — 2026-10-17

## Scope and audit

- Baseline: `75e0c09a881d589f4dfc939ca757638b72008428`.
- Locked upstream: `17a2f54463e61f1906f3a684dc1da7413938fffd`, already an ancestor.
- Work branch: `fix/ascend-forward-acceptance-20261017`.
- Actual GitHub `npu-ghd` was checked with `ls-remote` and matches the baseline.
  The local `origin/npu-ghd` ref is stale (`7c88f51`), because this shared Git
  store has no configured origin fetch refspec; `fetch origin npu-ghd` updates
  FETCH_HEAD, not that tracking ref. It is not a remote divergence.
- Worktree was clean before creating this branch. Original local/server
  `npu-ghd` refs are not changed; no merge, push, PR or global Git changes.
- Server original checkout is still `cc675a5`; validation will use a new,
  isolated checkout transferred from this work branch.
- Only forward/inference is in scope. Backward is OUT OF SCOPE, never PASS.
- Raw device logs and benchmark data stay outside tracked files.
- Source plan: [ascend-post-sync-plan.md](ascend-post-sync-plan.md).
- Engineering references: [PR #63](https://github.com/flagos-ai/FlagAttention/pull/63)
  and [PR #91](https://github.com/flagos-ai/FlagAttention/pull/91); actual source
  contracts govern adapters, registry and tests.

## Environment (fresh detection)

Container `ghd-sageattn` is running. Its existing device mask is **4,5**;
it is not overwritten. `npu-smi` reports Ascend910B4-1. Tests select logical
device 0 within that existing mask. Fresh imports report Python 3.11.15,
PyTorch 2.9.0+cpu, torch_npu 2.9.0.post2, Triton 3.5.1; NPU available=True,
device_count=2 and logical device 0 is Ascend910B4-1. CANN environment is loaded
from the existing toolkit without replacing its compiler PATH.

## Stage gates

| Stage | State | Evidence |
| --- | --- | --- |
| P0 audit | PASS | Clean baseline, isolated branch, fresh NPU detection |
| P1 GLA forward | PASS | 17 passed, exit 0, real 910B |
| P2 NSA forward | PASS | Final combined run: 39 passed, exit 0 |
| P3 strict runner | PASS | 30 CPU/CLI tests passed locally and in container |
| P4 NPU regression | PASS | 83 CPU/contract tests, 54 NPU forward/guard cases; strict runner exit 0 |
| P5 performance/CI | BLOCKED | Measurement PASS; self-hosted CI access/execution not verified |

## Fork ownership review

`git diff --name-only 17a2f54...75e0c09` identifies existing fork changes in
Ascend implementations/exports, configuration inventory, migration docs,
Ascend tests and the runner. P1 will add an Ascend YAML and split its own GLA
test; P2 will change only a new adapter/exports/tests; P3 only runner/tests.
Existing JIT source is outside the implementation boundary.

## P1 — GLA configuration and separate forward gate

The new CPU-capable configuration test was executed before adding YAML:
**1 failed**, exit 1, because the loader returned `[]` (log `p1-red.log`).
The original projects and Git history contain no Ascend YAML for this kernel.
The single candidate is copied from the locked NVIDIA example: BK=32, BV=32,
num_warps=4, num_stages=3. It is a correctness restoration candidate, not a
performance tuning result. No backward configuration is invented.

`test_gla_forward` covers lengths 17/32, FP16/BF16, both state layouts, with
and without a nonzero initial state, using the existing 0.01 relative-RMS
threshold and an independent CPU recurrence. Existing gradient assertions
remain in `test_gla_backward`, outside the forward gate. The inventory node
now points to the new forward test; official cross-vendor tests remain intact.

Command (from the isolated checkout, existing CANN environment and device mask):

```sh
python -m pytest -q \
  tests/flag_attn/test_ascend_attention.py::test_gla_forward \
  tests/flag_attn/test_ascend_registry.py::test_ascend_forward_state_config_is_loadable \
  --disable-warnings
```

Hardware result: **17 passed, 2 warnings**, exit **0**, 1682.24 seconds on
910B (16 numerical cases plus the configuration check). The lengthy first run
compiled the existing autotune candidates unchanged. The candidate is now
validated for this matrix, not claimed optimal. Backward: OUT OF SCOPE.
Logs: `/tmp/flagattention-ascend-acceptance-20261017/p1-green.log`.
Pre-commit on the new YAML passed all applicable hooks; unmatched hooks skipped.
Standalone Ruff on both modified test files passed.
Static `compileall` and `git diff --check` pass. AST comparison with the baseline
covers **45 JIT functions** (43 Ascend plus 2 shared state functions): zero
changed function ASTs. CPU/mock checks are not hardware correctness evidence.

## P2 — NSA forward-only public contract

P1 commit: `96977eb`. Added only a lazy `nsa/api.py` adapter and the single
compression registry key, preserving exact public parameter names/defaults.
Training with any q/k/v requiring gradients is explicitly rejected while
autograd is enabled; no detach, no no-grad decorator, no silent fallback.
Explicit no-grad/inference contexts work. Full `parallel_nsa` is not registered.

Tests-first: 8 newly selected NSA contract cases failed before implementation
(exit 1, `p2-red.log`). The first combined implementation run had **12 failed,
27 passed, 2 teardown errors**: the new test incorrectly monkeypatched a frozen
device dataclass field, preventing mock cleanup and returning mock strings in
subsequent compression tests. This is a test-isolation bug, not NPU precision
evidence. Fixed the test by replacing the lightweight runtime device object
within monkeypatch instead; no production device metadata or kernel changes.
The failed log `p2-green.log` is retained, not counted as a pass.

Fresh isolated reruns: registry **23 passed**, exit 0 (`p2-contract-rerun.log`);
NPU **16 passed**, exit 0 (`p2-forward-rerun.log`). The latter comprises 4 new
public alias/facade tests, 6 selected-only internal tests, and 6 original
compression forward/no-grad tests. FP16/BF16 compression covers T64/128 and
GQA, output plus natural-log LSE; selected-only covers grouped/masked branches,
negative/future indices, empty selections, T32/33 and packed sequence lengths.
The existing numerical tolerances and backward tests are unchanged.

```sh
python -m pytest -q tests/flag_attn/test_ascend_registry.py --disable-warnings
python -m pytest -q tests/flag_attn/test_ascend_nsa_forward.py \
  tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_fwd_accuracy \
  tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_no_grad \
  --disable-warnings
```

Final combined run confirms mock isolation: **39 passed**, exit **0**, 13.04s
(`p2-final.log`). Applicable pre-commit hooks passed (`p2-hooks.log`, exit 0).
Backward remains OUT OF SCOPE; selected-only is explicitly INTERNAL API.

## P3 — Opt-in strict result gate

P2 commit: `f7059e4`. Added `--strict-results`, evaluated after writing the
unchanged summary JSON. It rejects missing/invalid worker summaries, crashed
workers, missing requested operators/phases, non-Passed accuracy, nonzero
subprocess exits, zero executed passes and non-Passed performance. Explicit
`--skip-benchmarks` permits a genuinely Skipped performance entry, never a
performance PASS. Partial Skip plus positive Pass is accepted with counts
retained. Default non-strict exit behavior is preserved and regression-tested.

Tests-first CLI run: **17 failed, 1 passed**, exit 1 (unknown strict flag before
implementation). Final local suite: **30 passed**, exit 0; same suite in the
container: **30 passed**, exit 0 (`p3-regression.log`). Includes 18 real
subprocess exit-code scenarios, Ctrl-C cleanup and actual pytest node collection
from both worker cwd layouts while an unrelated optional-backend file would
raise if accidentally imported. Pinned Ruff 0.1.14 passes runner and tests.

```sh
# CPU-only runner tests; confcutdir avoids unrelated hardware conftest locally.
python -m pytest -q --confcutdir=tests/flag_attn \
  tests/flag_attn/test_test_runner.py --disable-warnings
```

No NPU mathematical correctness is inferred from these CPU/mock tests.

## P4 — Fresh integrated forward gate

Production source at P3 commit `1a097ec`, plus only the P4 test additions,
was transferred to the isolated container checkout. Before tests, assertions
verified vendor `ascend`, device `npu`, actual NPU availability and that the
imported package resides in the current checkout (not an installed old copy).

The first integrated command passed 83 CPU/contract and 53 forward/guard tests,
and the strict runner, but **exited 2** at Linux `git diff --check`: source
transport inherited the Windows host's system `core.autocrlf=true`. Committed
Git blobs were confirmed LF. Even `git archive` applies working-tree conversion,
so the first archive retry still failed. Fixed transport with command-local
`git -c core.autocrlf=false -c core.eol=lf archive`; no global config changes,
kernel rewriting or ignored whitespace checks. Failed logs are retained as
`p4-acceptance.log` and `p4-static.log`; successful static retry is
`p4-static-lf.log`, exit 0.

Added one explicit Sage top-level/namespace forward alias test with uniform
attention output/LSE and invalid quantization block size; its separate run
passed 1 case. Final complete rerun (`p4-final.log`) exited **0**:

| Check | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Runtime / registry / runner CPU and mock contracts | 83 | 0 | 0 |
| Explicit 910B forward and error-contract matrix | 54 | 0 | 0 |
| Strict inventory runner: GDN2 accuracy | 9 | 0 | 83 |
| Strict inventory runner: Sage accuracy | 7 | 0 | 0 |

Runner tests repeat some matrix cases; do not sum these rows as unique tests.
Its performance status is **SKIPPED**, not PASS. Those 83 GDN2 skips are CUDA
native/TLE resource comparisons, not skipped Ascend recurrence cases.

| Forward matrix component | Cases | Coverage boundary |
| --- | ---: | --- |
| GDN2 | 9 | FP16/BF16, T16/33, nonzero initial/final state, both layouts, invalid chunk size |
| GLA | 16 | FP16/BF16, T17/32, both layouts, zero/nonzero initial state, output/final state |
| KDA | 3 | BF16 T16/33 recurrence/final state, public rejection of precomputed indices |
| MiniMax | 2 | Prefill + score/top-k composite reference; invalid stage |
| Sage | 8 | HND/NHD, GQA, bool/additive mask, tails, quantization/LSE, aliases, error guards |
| NSA compression | 10 | Public facade/namespace, FP16/BF16 T64/128, output/LSE, no-grad |
| NSA selected | 6 | INTERNAL API only; grouped/masked, packed/boundary/empty/invalid indices, output/LSE |

All backward: **OUT OF SCOPE**, original code/tests retained. MiniMax decode,
full public NSA and independent all-shape primitive validation: **NOT RUN**;
no claim of independent optimized Ascend decode/full NSA support.

Reproduction: use the explicit nodes in the execution plan, replacing the
Sage file with its three `test_forward_*` nodes, and adding:

```sh
python -m pytest -q \
  tests/flag_attn/test_ascend_nsa_forward.py::test_ascend_nsa_compression_public_alias \
  tests/flag_attn/test_ascend_nsa_forward.py::test_ascend_nsa_selected_forward \
  tests/flag_attn/test_ascend_attention.py::test_kda_public_rejects_precomputed_chunk_indices \
  tests/flag_attn/test_ascend_attention.py::test_minimax_rejects_unknown_stage \
  tests/flag_attn/test_ascend_attention.py::test_sage_public_forward_alias_and_quantization_error \
  --disable-warnings
python tools/run_tests.py --ops chunk_gdn2,sage_attention \
  --skip-benchmarks --strict-results --gpus 4 --dump-output --color never \
  --output-dir /tmp/flagattention-ascend-acceptance-20261017/runner
```

The existing runner interprets `--gpus` as child physical masks; `--gpus 4`
keeps its workers on physical 4, a subset of the existing parent mask `4,5`.
Direct tests use logical device 0 in that unchanged parent mask. Global mask
and runner device-selection implementation are not changed.

Final `compileall`, `git diff --check`, pinned Ruff on changed Python files,
and applicable source pre-commit hooks pass (`p4-hooks.log`); unmatched hooks
are Skipped. AST audit still reports **45 unchanged JIT functions**.

## P5 — Comparable performance freeze and CI boundary

P4 code/test commit: `c240113fad480fcc97eb62e6a6686711c5fd039f`. The isolated
server work branch was advanced to that same commit using the transferred
Git bundle and LF archive, without touching the original project or `npu-ghd`.
Its generated `tests/summary.json` was moved to the untracked evidence directory,
not deleted or staged. The isolated worktree was then confirmed clean.

### Protocol and entries

- Before: exported source from `cc675a5f7f456416ab75e08ac48b079c774d533f`.
- After: `c240113fad480fcc97eb62e6a6686711c5fd039f`.
- Same Python/container, CANN 9.0.0, torch_npu/Triton, logical device 0 in
  existing physical mask `4,5`; same input fingerprints verified for every case.
- Seed 42; 10 warmups, 50 repetitions; synchronize before each timer and after
  each call; `time.perf_counter`, milliseconds; median and inclusive P10/P90.
- **Total internal optimized API call time**, including host dispatch and
  allocation/synchronization, not pure kernel time. Public facade/adapter cost
  is **not** measured: before lacked the same public Sage facade, so both
  snapshots use identical internal entry paths instead of unlike interfaces.
- GDN2 entry: `_ascend.gdn2.chunk_fwd_infer.chunk_gdn2_fwd_infer`, BF16,
  B1/T128/H1/K64/V64, default inference options, no initial state.
- Sage entry: `_ascend.attn_qk_int8_per_block.forward`, FP16 V, HND,
  B1/H2/D64/T128 or T256, unmasked, return_lse=False; quantization outside timer.
- One identical independent harness; SHA256:
  `29232ce1cc03a01bac1755a37025ff972f93f4836cb5ff68a64c8286d4208779`.
- The harness, validation script and raw data are retained locally under the
  ignored `BaselineBenchmark/forward-acceptance-20261017/` and in the untracked
  evidence archive. Neither benchmark data nor BaselineBenchmark enters Git.

### Measured results (ms)

| Case | Before P10 | Before median | Before P90 | After P10 | After median | After P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN2 BF16 T128 | 0.572142 | 0.575827 | 0.596078 | 0.580177 | 0.584006 | 0.603780 |
| Sage FP16 T128 | 0.296456 | 0.299255 | 0.305370 | 0.300392 | 0.304056 | 0.310882 |
| Sage FP16 T256 | 0.296950 | 0.298546 | 0.303300 | 0.301156 | 0.302940 | 0.306655 |

Before/after median ratios: **0.985995 / 0.984210 / 0.985494**. These are ratios
between two optimized snapshots, not speedups against a naive implementation.
This single run's after medians are approximately 1.4–1.6% higher. No threshold
was specified for performance acceptance; no repeated interleaved statistical
study or hardware exclusivity was established. Therefore **do not claim a
speedup or a proven zero regression**, nor attribute this small difference to
a kernel regression. Output norms also match, but norms alone are not an
accuracy proof; use the independent P4 references for correctness.

Both benchmark commands exit **0** (`p5-before.log`, `p5-after.log`), each with
all 50 raw samples per case. A separate comparison verifies environment,
protocol and input hashes match: **COMPARABLE**, exit 0. Historical wall-clock
and NPU-event numbers in previous reports are not substituted for this run.

```sh
# In the existing CANN/Python environment; preserve its device mask.
python BaselineBenchmark/forward-acceptance-20261017/perf.py \
  --root /tmp/flagattention-perf-cc675a5 --label cc675a5
python BaselineBenchmark/forward-acceptance-20261017/perf.py \
  --root "$PWD" --label c240113
```

### CI and final limits

`.github/workflows/ops-test.yaml` and `.github/configs/backends/910B.yaml` are
unchanged from the baseline. The inherited runner label is
`910b-flaggems-weekly`; a copied configuration is not evidence of access.
**CI: BLOCKED — self-hosted runner access and actual execution not verified.**
No workflow was triggered and no branch was pushed, as required by this plan.
No empty GitHub status is interpreted as CI PASS. Measurement and documentation
are complete; P5 is not marked fully PASS because the CI item is unresolved.

Remaining work: obtain/confirm runner access and explicitly authorize the remote
workflow step if desired; broaden the shape matrix and measure public-wrapper
cost in a separate, scoped follow-up. No kernel performance optimization was
performed in this execution.

## Per-stage handoff (task-book fields)

Common baseline SHA: `75e0c09a881d589f4dfc939ca757638b72008428`.
Common hardware/software: fresh environment table above, Ascend910B4-1;
Python 3.11.15 / torch 2.9.0+cpu / torch_npu 2.9.0.post2 / Triton 3.5.1 /
CANN 9.0.0. Raw commands, exit codes, counts and failure histories appear in
each stage above and the untracked log archive. **Kernel changed: NO**, verified
by unchanged ASTs of 43 Ascend + 2 shared JIT functions, not only visual diff.

### 阶段：P0 — 冻结与审计

- 状态：PASS。
- 基线 SHA / 当前 SHA：`75e0c09` / `75cdfda`。
- 修改文件：本验收记录、任务书副本。
- 关键改动与原因：独立工作分支、远端真实 SHA/陈旧跟踪引用区分、真机环境检测。
- 执行命令与退出码：fetch、ls-remote、祖先检查、版本/可用性检测均成功。
- 通过/失败/跳过计数：非测试 Gate；NPU available=True，2 个可见设备。
- 验证范围和未验证内容：冻结与环境，不代表数值验收。
- Kernel 是否改动：否；遗留风险：后续阶段待执行。
- 本地 commit SHA：`75cdfda`。

### 阶段：P1 — GLA forward 配置

- 状态：PASS。
- 基线 SHA / 当前 SHA：`75e0c09` / `96977eb`。
- 修改文件：Ascend YAML、Ascend 数值/注册测试、inventory、本记录。
- 关键改动与原因：恢复有来源的 BK/BV 候选，拆分 forward/backward，修正 inventory 节点。
- 执行命令与退出码：P1 显式 pytest，最终 exit 0；Ruff/hooks/静态检查通过。
- 通过/失败/跳过计数：最终 17/0/0；先验配置回归 0/1/0 保留。
- 验证范围和未验证内容：声明矩阵的前向；backward OUT OF SCOPE；非最优调参结论。
- Kernel 是否改动：否；遗留风险：未覆盖全形状。
- 本地 commit SHA：`96977eb`。

### 阶段：P2 — NSA 推理接入

- 状态：PASS。
- 基线 SHA / 当前 SHA：`75e0c09` / `f7059e4`。
- 修改文件：NSA api.py、后端注册表、注册/前向测试、本记录。
- 关键改动与原因：compression 公共薄适配，显式拒绝训练，selected-only 保持内部。
- 执行命令与退出码：P2 最终组合 pytest exit 0；pre-commit exit 0。
- 通过/失败/跳过计数：最终 39/0/0；首轮测试隔离错误详见前文，不计 PASS。
- 验证范围和未验证内容：公共 compression 和内部 selected；完整 NSA/backward 不验收。
- Kernel 是否改动：否；遗留风险：不代表全模式、全形状支持。
- 本地 commit SHA：`f7059e4`。

### 阶段：P3 — Runner 严格结果 Gate

- 状态：PASS。
- 基线 SHA / 当前 SHA：`75e0c09` / `1a097ec`。
- 修改文件：tools/run_tests.py、Runner 测试、本记录。
- 关键改动与原因：opt-in 严格退出、有效 summary 检查，保留默认行为与 JSON 格式。
- 执行命令与退出码：本地/容器 Runner pytest 均 exit 0；Ruff 通过。
- 通过/失败/跳过计数：最终各 30/0/0；包含真实 CLI 子进程、collection 和 Ctrl-C 测试。
- 验证范围和未验证内容：CPU/mock 与调度契约，不作为 NPU 精度证明。
- Kernel 是否改动：否；遗留风险：CI 须显式启用严格开关。
- 本地 commit SHA：`1a097ec`。

### 阶段：P4 — 910B forward 综合回归

- 状态：PASS。
- 基线 SHA / 当前 SHA：`75e0c09` / `c240113`。
- 修改文件：补充公共入口/错误路径测试、本记录。
- 关键改动与原因：显式前向节点、真机/源码来源检测、真实严格 Runner。
- 执行命令与退出码：p4-final 整体 exit 0；首次 CRLF 静态失败 exit 2 保留。
- 通过/失败/跳过计数：CPU 83/0/0，NPU 54/0/0；Runner GDN2 9/0/83、Sage 7/0/0。
- 验证范围和未验证内容：本轮声明矩阵；未覆盖 decode/full NSA/所有形状；性能 Skip 非 PASS。
- Kernel 是否改动：否；遗留风险：广覆盖与 CI 权限仍需后续验证。
- 本地 commit SHA：`c240113`。

### 阶段：P5 — 性能冻结与交付

- 状态：BLOCKED（仅 CI）；同协议基线测量 PASS。
- 基线 SHA / 当前 SHA：性能对比 `cc675a5` / `c240113`；最终附加提交仅记录文档。
- 修改文件：本记录；harness/原始数据只保存在忽略目录与本地证据包。
- 关键改动与原因：统一时间口径、输入指纹、分位数和 CI 状态，防止虚报无回退。
- 执行命令与退出码：before/after benchmark 与可比性校验均 exit 0；CI NOT RUN/BLOCKED。
- 通过/失败/跳过计数：两个快照各 3 个案例、每例 50 次；未执行 CI 不算 PASS。
- 验证范围和未验证内容：小尺寸内部优化入口，不是 public-wrapper 成本或全形状收益。
- Kernel 是否改动：否；遗留风险与下一阶段入口：自托管权限、授权后的 CI 和更广性能矩阵。
- 本地 commit SHA：本节随 `docs(ascend): record forward acceptance and perf baseline` 提交；
  使用 `git log -1 --format=%H` 获取最终文档提交，避免自引用 SHA。
