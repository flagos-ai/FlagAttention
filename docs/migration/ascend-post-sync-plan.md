# FlagAttention Ascend 架构同步后修复与验收 — Codex 执行任务书

> **用途**：把本文放到仓库 `docs/migration/ascend-post-sync-plan.md`，交给 Codex 按顺序执行。本文是实施指令，不是已完成报告。
>
> **检查基线（2026-10-16）**：`Ghd-2077/FlagAttention:npu-ghd` = `75e0c09a881d589f4dfc939ca757638b72008428`；`flagos-ai/FlagAttention:main` = `17a2f54463e61f1906f3a684dc1da7413938fffd`；GitHub compare 显示 **ahead 21 / behind 0**。若实际 HEAD 不同，先记录差异并重新审计，不得把本文的历史结果当作新 HEAD 的测试结果。
>
> **本轮范围**：仅让 Ascend **forward/inference** 公共路径可靠、可测、可复现。**不验收 backward，不重写 Kernel，不进行性能调优，不重新合并上游。** 现有 backward 代码保留。上游 GitHub PR [#63](https://github.com/flagos-ai/FlagAttention/pull/63)、[#91](https://github.com/flagos-ai/FlagAttention/pull/91) 与仓库的 `docs/migration/ascend-alignment.md`、`forward-only-checkpoint-20261016.md` 是工程规范和历史证据；以实际源码的 API 契约为准。

## 0. 给 Codex 的总体执行规则（必须遵守）

1. **顺序执行 P0 → P1 → P2 → P3 → P4 → P5。** 每阶段先读源码和现有测试，再做最小修改、单独验证、记录结果、创建独立本地 commit；如果验收失败，不得将该阶段标记为 PASS，不得跳过阻塞项声称全部完成。
2. 只在新工作分支修改。**不要 force-push、不要修改 `npu-ghd` 或 `upstream/main`、不要自动创建 PR/推送远端**。工作区非干净时先保护既有改动，不能擅自 `reset --hard`、`clean -fd` 或覆盖用户文件。
3. **禁止改动现有 Triton/JIT Kernel 数学语义、tile/launch 策略、签名、数值容差**；本轮只允许必要的配置文件、轻量公共适配层、导出注册表、测试、Runner、文档。若修复必须涉及 Kernel，标记 **BLOCKED** 并说明原因，不能悄悄越界。
4. 遵守官方 `op_registry` 的规则：仅注册与公共接口兼容的实现；**已注册的实现异常必须暴露，不允许静默退回 Generic**；不得创建平行注册框架或在 Generic 模块硬编码 Ascend 分支。导入应保持 lazy；不要为了通过测试而静默吞异常、假造 Skip、放宽数值阈值。
5. 所有报告必须严格区分 `PASS` / `FAIL` / `SKIPPED` / `BLOCKED` / `NOT RUN`。CPU/mock 测试不能替代 910B 数值测试，旧日志不能作为本次执行通过的证据。
6. 保留原有许可证/署名；遵循 `.pre-commit-config.yaml`、`pyproject.toml` 及 PR #63 的导出、测试、Benchmark 规范。**只清理本轮引入的 lint 问题**，不做无关格式化。
7. 有硬件就按本文真机执行；没有权限/设备时，运行可行的静态和 CPU 测试，记录 `BLOCKED: NPU unavailable`，**不要假称完成 P1/P2/P4/P5**。

### 初始动作（P0：冻结当前状态，不能直接开始改代码）

在仓库根目录执行（若已有项目目录则不重新 clone）：

```bash
set -e
pwd
git status --short
git remote -v
if ! git remote get-url upstream >/dev/null 2>&1; then
  git remote add upstream https://github.com/flagos-ai/FlagAttention.git
fi
git fetch origin npu-ghd
git fetch upstream main
BASE=75e0c09a881d589f4dfc939ca757638b72008428
UP=17a2f54463e61f1906f3a684dc1da7413938fffd
test -z "$(git status --porcelain)" || { echo 'Dirty worktree: protect user changes before continuing'; exit 2; }
test "$(git rev-parse HEAD)" = "$BASE" || { echo 'HEAD changed: re-audit first'; exit 2; }
git merge-base --is-ancestor "$UP" HEAD || { echo 'Upstream not merged: re-audit first'; exit 2; }
git switch -c fix/ascend-forward-acceptance-20261016
```

- 若工作区不干净：先报告文件清单并保护用户变更；不得强行执行分支切换。若 `UP` 不在历史里，或当前 HEAD 不是 `BASE` 对应的迁移后快照，**先记录新的 HEAD、`git log` 和 `git diff`，重新分析差异**。如果工作分支已存在，不自动删除或覆盖；另取明确命名的安全分支。
- **仅 fetch，不 merge**，也不要修改 Git 全局配置。
- 记录 `python --version`、`torch.__version__`、`torch_npu.__version__`、`triton.__version__`、CANN、芯片型号、可见设备映射；**不自动覆盖 `ASCEND_RT_VISIBLE_DEVICES`**。历史验证容器为 `ghd-sageattn`、Python `/usr/local/python3.11.15/bin/python`、Ascend910B4-1、CANN 9.0.0、Triton 3.5.1；这些只是参考，不假定本次环境相同。
- 对每个待改文件执行 `git diff --name-only "$UP"...HEAD` 并记录 fork 差异；创建本地验收记录 `docs/migration/ascend-forward-acceptance.md`，初始每阶段状态 `NOT RUN`。
- 建立不提交的日志目录（例如 `/tmp/flagattention-ascend-acceptance-<date>/`）。**不得提交个人环境路径、Benchmark 大文件、设备日志、私有服务地址或凭据。**

**P0 Gate**：工作分支独立；仓库状态、版本、设备信息、锁定 SHA 已记录；原始 `npu-ghd` 没有被修改。

---

## P1. 修复 GLA forward 的 `BK/BV` 缺失（当前明确阻塞项）

**问题证据**：`tests/flag_attn/test_ascend_attention.py::test_gla_forward_and_backward` 的长度 17 / 32 在此前 910B 上报 `dynamic_func() missing ... BK and BV`。公共 `src/flag_attn/FLA/chunk_h.py` 的 `chunk_fwd_kernel_h` 通过 `runtime.get_tuned_config('chunk_fwd_kernel_h')` 获取配置；`runtime/configs_loader.py` 读取 `_ascend/tune_configs.yaml`，但当前仓库不存在该文件。历史记录显示前向 2 个 FAIL；这不是已修复的问题。

### 修改边界

- **可改**：`src/flag_attn/runtime/backend/_ascend/tune_configs.yaml`（新增）；`tests/flag_attn/test_ascend_attention.py`；必要时 `conf/operators.yaml`、`pyproject.toml`、独立的配置加载测试。
- **不可改**：`src/flag_attn/FLA/chunk_h.py` 及 `_ascend/gla/chunk.py` 的 JIT Kernel/运算逻辑；不可全局更改 `TunedConfigLoader` 以绕过问题。

### 执行步骤

1. 确认调用链：`flag_attn.chunk_gla` → `resolve_operator('chunk_gla')` → `_ascend.gla.chunk_gla` → `chunk_fwd_h` → `chunk_fwd_kernel_h`。用 CPU/AST 检查配置加载：`FLAG_ATTN_BACKEND=ascend` 下 `runtime.get_tuned_config('chunk_fwd_kernel_h')` 当前应为空；记录结果。
2. 优先检索**确实在 Ascend 环境验证过**的配置（本地已保存的运行记录或合法引用）。找到则记录来源、设备和验证方式；未找到时，**仅为恢复功能采用一个待验证的保守候选**，例如参考上游 `_nvidia/tune_configs.yaml` 中同名 Kernel 的 `{BK: 32, BV: 32}, num_warps: 4, num_stages: 3`，放入 Ascend 独立 YAML。注意：**这个值来自 NVIDIA 示例，不代表在 910B 上已验证或最优**。只有 910B 编译 + 数值测试成功后才可标记“有效”；资源不足可在上游已存在的有限候选 `{32,32},{32,64},{64,32},{64,64}` 中逐个尝试并记录，不做性能寻优。若全失败，保留证据并标记 BLOCKED，不能更改核心 Kernel 绕过。
3. 最小 YAML 结构范例（只是**待验证候选**）：

   ```yaml
   # Correctness-first candidate derived from upstream NVIDIA config;
   # NOT an Ascend performance tuning claim. Validate on 910B.
   chunk_fwd_kernel_h:
     - META: {BK: 32, BV: 32}
       num_warps: 4
       num_stages: 3
   ```

   如 forward 路径还访问其他 `get_tuned_config(...)`，按实际缺失项逐一补齐；**不要为本轮不验收的 backward 盲目添加 `chunk_bwd_kernel_dh`**。
4. 把 `test_gla_forward_and_backward` 拆成 `test_gla_forward` 和单独的 `test_gla_backward`。保留已有 backward 验证但**不纳入本轮 forward Gate**；不得把 forward 与 backward 混写为一个必过用例。**必须**把 `conf/operators.yaml` 里旧的 `tests/flag_attn/test_ascend_attention.py::test_gla_forward_and_backward` 节点改成新 `::test_gla_forward`，避免运行器收集已不存在的测试；原有官方跨厂商测试不可粗暴删除。如其包含 backward，则在验收报告中说明 `run_tests.py --ops chunk_gla` 与“forward-only”并非同一测试范围。本轮 forward 以**显式 pytest 节点**为准。必要时新建专用 Ascend forward marker，但不得伪装完整算子测试通过。
5. 新增 CPU 可运行的配置回归测试：`get_tuned_config('chunk_fwd_kernel_h')` 非空，并检查 `BK/BV`、`num_warps`、`num_stages` 合法；这只能证明配置**可加载**，不是 Kernel 正确。
6. 在 910B 上运行（先确认 NPU 可用、vendor 是 `ascend`，`PYTHONPATH` 指向当前 checkout）：

   ```bash
   export FLAG_ATTN_BACKEND=ascend
   export PYTHONPATH="src:.:${PYTHONPATH:-}"
   python -m pytest -q \
     tests/flag_attn/test_ascend_attention.py::test_gla_forward \
     --disable-warnings
   ```

   至少验证长度 17/32、FP16/BF16、output、final state、`state_v_first` 及非零 `initial_state`（若该调用契约支持）。参考独立 CPU recurrence；不得降低原有误差阈值来求通过。**先保证原有 17/32 两个前向案例正常**，再扩充边界组合。

**P1 Gate**：当前 checkout 的公共 `flag_attn.chunk_gla` 在真实 910B 上完成已覆盖的 forward 正确性验证，无 `BK/BV` 错误；配置来源/候选说明清楚；无 JIT Kernel 逻辑更改；backward 记为 `OUT OF SCOPE`（不是 PASS）。提交 `fix(ascend): restore GLA forward config and isolate forward tests`。

---

## P2. NSA forward-only 接入与契约验证

**已知状态**：`_ascend/nsa/compression.py` 实现 `parallel_nsa_compression`；当前 `_ascend/__init__.py::_OPERATOR_EXPORTS` **没有** `parallel_nsa_compression`、`parallel_nsa`。此前前向相关测试通过过，但在出现 backward 非有限梯度后撤销了公共注册。`_ascend/nsa/forward.py::parallel_nsa_fwd` 仅实现 selected attention、返回 `(output, LSE)`，**不等价于**上游完整 `parallel_nsa` API。

### P2A：安全恢复 compression 公共入口

- **可改**：`src/flag_attn/runtime/backend/_ascend/nsa/api.py`（建议新增薄适配层）、`_ascend/__init__.py`、`tests/flag_attn/test_ascend_registry.py`、forward-only 数值测试和验收文档。
- **不可改**：NSA Kernel 及 autograd 实现；不可通过 `@torch.no_grad` 静默吞掉用户所需梯度，不得用装饰器取代输入契约检查。

执行：

1. 对照上游 `flag_attn.parallel_nsa.parallel_nsa_compression` 的参数**名称、位置、默认值、返回值和 dtype**，实现签名一致的薄 wrapper；调用原 `_ascend.nsa.compression.parallel_nsa_compression`。
2. 当 autograd 开启且 `q/k/v` 需要梯度时，**清晰抛出 `RuntimeError` / `NotImplementedError`**，说明本次公共 override 仅支持 forward-only；在 `torch.no_grad()` / `torch.inference_mode()` 的真实推理场景允许执行。不能隐式 detach，也不能失败时切回 Generic。
3. 只在 `_ascend/__init__.py::_OPERATOR_EXPORTS` 新增 `"parallel_nsa_compression": (".nsa.api", "parallel_nsa_compression")`；不要改顶层官方 `parallel_nsa` facade，也不要注册 `parallel_nsa` 全路径。
4. 新增 Registry/mock 契约测试：`resolve_operator('parallel_nsa_compression', ..., vendor='ascend')` 绑定 Ascend adapter；上游公共 `parallel_nsa(mode='compression', ...)` 与 `from flag_attn.parallel_nsa import parallel_nsa_compression` 指向同一 Ascend 实现；`parallel_nsa` 全路径**仍不注册**；依赖错误必须传播；需要梯度时必须拒绝。
5. NPU 前向测试必须经**公共入口**调用，覆盖 FP16/BF16、T=64/128、合法 GQA、`block_size=64`、输出与 LSE、无梯度。可沿用上游已定义的两个 forward 用例节点，但**不要将 backward 用例纳入 forward-only 验收**：

   ```bash
   python -m pytest -q \
     tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_fwd_accuracy \
     tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_no_grad \
     --disable-warnings
   ```

   旧的 6 项 forward 历史结果不算本次 PASS，必须重新执行。记录公共 API 是否真正调用 Ascend wrapper，而不是误走 Generic。

### P2B：NSA selected-only 保持独立，补最小测试

- **不得**将 `_ascend.nsa.forward.parallel_nsa_fwd` 直接注册成完整 `parallel_nsa`，它缺少 full/compression/window 等模式的公共契约，且返回语义不同。
- 保留内部路径，在 `tests/flag_attn/` 新增或整理显式的 `test_ascend_nsa_selected_forward`：实际从内部入口调用，和独立参考对齐 output + LSE、有效/无效 block index、边界/变长。报告明确标注 `internal API`；未来需要产品化时再独立设计兼容的 stage facade / 单独公开接口。
- 原 NSA backward 测试保留，历史非有限梯度记录保留，但其结果不作为本轮 forward Gate 的成败依据。

**P2 Gate**：compression 的**两种公共调用方式**在 910B 正确运行，需梯度时明确拒绝；selected-only 的内部前向验证单独可复现；没有错误注册 `parallel_nsa`；未改任何 NSA 计算 Kernel。提交 `fix(ascend): expose NSA compression with forward-only contract`。

---

## P3. 修复测试 Runner 的假成功风险（保持上游兼容）

**证据**：`tools/run_tests.py` 当前在写入 `summary.json` 后自然返回 0；即使 `accuracy` 记录为 `Failed`/`Error`，只检查进程退出码的 CI 仍可能误判通过。

**允许改动**：`tools/run_tests.py`、`tests/flag_attn/test_test_runner.py`、必要的使用文档；不要重构整个调度器/结果 JSON 格式。

1. 添加**显式 opt-in 严格模式** `--strict-results`（默认保持原有行为，避免破坏官方已有结果上传/汇总工作流）。该模式在 `summary.json` 写入后检查每个请求算子的状态并返回非零退出码：
   - Accuracy `Failed` / `Error` / `Timeout` / `NotFound` → **失败**；若目标算子仅 `Skipped` / `0 passed`，也不能被当作前向验收成功。
   - Performance 在未使用 `--skip-benchmarks` 时 `Failed` / `Error` / `Timeout` / `NotFound` → 失败；使用 `--skip-benchmarks` 时 `Skipped` 可以接受，但绝不可记录为性能通过。
   - 请求算子缺失 `summary`，或工作进程未产出有效结果 → **失败**。测试集存在部分 Skip 时保留明细，并明确记录 `passed/failed/skipped` 数量。
2. 该检查应兼容现有 `summary.json` 结构、worker 汇总、异常退出、Ctrl-C，不删改指标字段。增添单元测试模拟 `Passed`、`Failed`、`NotFound`、全 Skip、部分 Skip+Pass、缺失 worker summary、显式性能 Skip，验证实际 CLI 退出码。
3. 验证 `inventory_targets()` 在 `tests/` 和 `benchmark/` cwd 下生成的路径含 `::test_node` 时可被 pytest 正确识别，不触发其他可选 backend 的 import。保留上游兼容逻辑，不扩大测试扫描范围。
4. 不要直接让默认命令在非严格模式变更退出行为；新增的严格开关用于本轮独立验收和有权限时的 Ascend CI 配置。**禁止“失败但 exit 0”再被文档描述为 PASS。**

**P3 Gate**：CPU mock 回归 + CLI 非零退出码模拟测试均通过；`--strict-results` 的错误传播可见；无无关 Runner 大重构。提交 `fix(testing): add opt-in strict runner result gate`。

---

## P4. Forward-only 910B 综合验收与测试清单

### 前置检测

在执行任何 NPU 测试前，使用**当前 checkout 的同一 Python 环境**：

```bash
export FLAG_ATTN_BACKEND=ascend
export PYTHONPATH="src:.:${PYTHONPATH:-}"
python - <<'PY'
import torch
import flag_attn
try:
    import torch_npu
except ImportError as exc:
    raise RuntimeError('torch_npu unavailable; hardware acceptance BLOCKED') from exc
print('torch=', torch.__version__, 'torch_npu=', torch_npu.__version__)
print('vendor=', flag_attn.vendor_name, 'device=', flag_attn.device)
assert flag_attn.vendor_name == 'ascend' and flag_attn.device == 'npu'
assert torch.npu.is_available(), 'NPU unavailable; hardware acceptance BLOCKED'
PY
```

历史环境使用容器 `ghd-sageattn` + `ASCEND_RT_VISIBLE_DEVICES=4`，但**请根据真实容器设备映射决定是否设置该环境变量**，不可无依据固定为 4。保留 CANN 的编译器 `PATH`，不得用全局更换 Python/Triton 来回避错误。

### 当前应验收的矩阵

| 入口 | 必须跑到的检查 | 期望结果 |
|---|---|---|
| `flag_attn.chunk_gdn2` | FP16/BF16；T=16/33；非零初始状态；`state_v_first` 双布局；错误 chunk size | forward 输出 + final state 对齐独立 reference；训练梯度明确拒绝 |
| `flag_attn.sage_attention` 与 `sage_attention.forward` | HND/NHD、GQA、mask、尾块、LSE、quantization、非法参数 | 已注册 Ascend 实现，输出精度及异常一致 |
| `flag_attn.chunk_gla` | T=17/32；至少 FP16/BF16；输出/状态/布局 | forward 全绿、无缺失 BK/BV；backward 另记 OUT OF SCOPE |
| `flag_attn.chunk_kda` | T=16/33；BF16；final state；参数/门控及不支持 `chunk_indices` 行为 | 独立 recurrence 对齐；不得据两个案例宣称全形状支持 |
| `flag_attn.parallel_nsa(mode='compression')` | FP16/BF16；T=64/128；output/LSE；无梯度；autograd 错误路径 | 经 Ascend 公共 wrapper 且参考对齐 |
| 内部 `_ascend.nsa.forward.parallel_nsa_fwd` | Selected-only index、长度边界、LSE | **内部实现**单独通过，不冒充完整公共 `parallel_nsa` |
| MiniMax stage facade/index | prefill、index score/topk、异常/边界；decode 要标记现状 | prefill/index 对齐参考；没有独立 Ascend decode 时不声称已优化/已通过 |

运行具体的**显式 pytest 节点**，避免当前官方测试文件内的 backward 案例被误算为 forward-only Gate：

```bash
python -m pytest -q \
  tests/flag_attn/test_runtime.py \
  tests/flag_attn/test_ascend_registry.py \
  tests/flag_attn/test_test_runner.py \
  --disable-warnings

python -m pytest -q \
  tests/flag_attn/test_ascend_attention.py::test_gdn2_recurrence \
  tests/flag_attn/test_ascend_attention.py::test_gdn2_rejects_unsupported_chunk_size \
  tests/flag_attn/test_ascend_attention.py::test_gla_forward \
  tests/flag_attn/test_ascend_attention.py::test_kda_recurrence \
  tests/flag_attn/test_ascend_attention.py::test_minimax_prefill_stage_and_index \
  tests/flag_attn/test_sage_attention.py \
  tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_fwd_accuracy \
  tests/flag_attn/test_parallel_nsa_compression.py::test_parallel_nsa_compression_no_grad \
  --disable-warnings
```

> 上述 `test_gla_forward` 是 P1 **计划创建的节点**；新增 selected-only 测试的精确节点名由 P2 确定并加入上述命令。若既有 `test_sage_attention.py` 包含非 forward 用例，应先枚举节点，改为显式列出 forward 节点，不能用整个文件名掩盖范围。所有真实运行要保存终端命令、退出码和 pytest 明细。

如需跑官方 Runner，只能将结果视为**对应 inventory 的整体测试**（其中可能包含 backward）：

```bash
python tools/run_tests.py \
  --ops chunk_gdn2,sage_attention \
  --skip-benchmarks --strict-results --gpus 0 \
  --dump-output --color never \
  --output-dir /tmp/flagattention-forward-acceptance-runner
```

不要在未审查 inventory / markers 时用 `--ops chunk_gla,parallel_nsa` 的结果声称“仅 forward 完成”；如果出现 backward 失败，应保留完整记录并另行区分 forward Gate 与全量测试。检查 `summary.json` 的每个算子 `accuracy.status`、计数以及 `performance.status`，禁止只看 shell exit code。

### P4 Gate（硬条件）

- 所有**本轮宣称支持的**公共 forward 路径在真实 910B 上至少有一个成功执行的数值测试，且必要边界案例通过；不允许全部 Skip。
- P1 GLA T17/32 通过；P2 compression 通过；GDN2、Sage 既有回归重新执行通过；KDA、MiniMax 等报告准确写明覆盖范围。
- NPU 不可用、编译失败、公共路径 fallback 错误、配置问题均为 BLOCKED/FAIL，不是 PASS。失败时保存最小复现和栈信息，不扩大 Kernel 修改范围。
- `compileall`、`git diff --check`、本轮变更文件的 Ruff / pre-commit（可访问时）通过；如果 pre-commit 因外网或代理失效，写 `BLOCKED`，不能伪造成功。

完成后提交 `test(ascend): gate supported forward APIs on 910B`（如果只改测试/文档）。

---

## P5. 冻结性能基线、CI 检查与最终交付

本轮**不优化性能**，只为下一轮 GDN2 / SageAttention 优化建立可信的测量方式：

1. 使用 **同一套独立 Benchmark harness** 对迁移前 `cc675a5f7f456416ab75e08ac48b079c774d533f` 与当前修复后的 HEAD 分别测量。可以创建**只读/隔离** Git worktree；测试两者时不覆盖用户工作区。不要把之前“同步 wall-clock”的数字与本次“NPU event”的数字直接作前后对比。
2. 固定 910B 实际设备、Python、torch_npu、Triton、CANN、random seed、dtype、shape、layout、mask、预热次数、重复次数、同步方式以及量化是否包含在计时中。至少记录 median / P10 / P90 及单位（ms）；标明总调用时间还是纯 Kernel 时间。比较相同输入和相同计时口径；若不能保证一致，写 `NOT COMPARABLE`。
3. 最小 benchmark 形状：GDN2 BF16 `[B,T,H,K,V]=[1,128,1,64,64]`；Sage FP16 `B=1,H=2,D=64,T=128/256,HND,unmasked`。以后扩大完整 shape matrix，**本轮只保留优化入口与原始数据**。
4. 迁移记录中旧结果：GDN2 0.715145 ms，Sage T128/T256 0.318420/0.321860 ms；这组使用 NPU event，**仅作历史参考**。Sage 在该小尺寸下相对 SDPA 速度比 0.419/0.410，不可宣称全面提速。
5. 检查 `.github/workflows/ops-test.yaml` 与 `.github/configs/backends/910B.yaml`：继承的 910B self-hosted runner 不等于 fork 有访问权限。只有真正运行并检验结果才可写 CI PASS；无权触发时写 `BLOCKED: self-hosted runner access`。不得把 GitHub status 为空误读为通过。
6. 运行变更审查并确认没有核心 JIT Kernel 改动：

   ```bash
   git diff --check
   python -m compileall -q src tests/flag_attn
   git diff 75e0c09a881d589f4dfc939ca757638b72008428 -- \
     src/flag_attn/runtime/backend/_ascend \
     src/flag_attn/FLA/chunk_h.py
   git status --short
   ```

   如果需要以更强证据证明 Kernel 未改变，使用 AST 或已知 JIT 函数签名比较，并将校验方式写入报告；不要仅凭主观查看 diff。

**P5 Gate**：复现命令、环境表、正确性结果、`PASS/FAIL/SKIPPED/BLOCKED/NOT RUN` 分类、性能原始结果/协议、未解决限制及下一轮优化候选全部落盘。仅在每项 Gate 完成时称“阶段通过”。提交 `docs(ascend): record forward acceptance and perf baseline`。

---

## 交付清单与最终报告格式（每一阶段都要更新）

最终应有：

- **代码**：仅必要的 `_ascend/tune_configs.yaml`、NSA 薄 adapter/注册表、测试拆分与补充、Runner 严格退出模式；不得有不相关 Kernel diff。
- **证据**：`docs/migration/ascend-forward-acceptance.md`，列出基线 SHA、实际 HEAD、测试环境、逐条命令、结果计数、真实硬件证明、失败日志位置、benchmark 协议和还未支持的 API。
- **提交**：分阶段本地 commit，可审查、可 cherry-pick、可回滚；**不得自动推送**。

Codex 每阶段结束必须使用以下模板汇报：

```text
阶段：P<N> — <名称>
状态：PASS | FAIL | BLOCKED | NOT RUN
基线 SHA / 当前 SHA：...
修改文件：...
关键改动与原因：...
执行的命令与退出码：...
通过/失败/跳过计数：...
硬件与软件环境：...
验证范围和未验证内容：...
Kernel 是否改动：否（给出检查证据）
遗留风险与下一阶段入口：...
本地 commit SHA：...（无 commit 时写无）
```

### 明确的停止条件

- 当前 HEAD 与锁定基线不一致、已有用户未提交修改、发现新上游架构变化：**先审计，不可盲目套用补丁**。
- 需要修改 Kernel 算法、隐式禁用梯度、强行改变 Public API、无来源地声明性能收益：**停止该项并记录 BLOCKED**。
- NPU 不可用、910B Kernel 无法编译或参考精度失败：**不标记 PASS**，保留最小复现；其它互不依赖阶段可以继续，但最终验收不能宣称完成。
- backward 测试失败：**保留原始失败记录，标记 OUT OF SCOPE，不为了 forward-only 通过而删掉测试或 Kernel**。

## Codex 首次启动时直接发送的指令

> 请阅读 `docs/migration/ascend-post-sync-plan.md`，以 `75e0c09a881d589f4dfc939ca757638b72008428` 为目标基线，**按照 P0 → P5 顺序直接实施**，每一阶段按文档的允许修改范围、命令、Gate 和汇报模板执行；通过一个 Gate 后自动进入下一阶段，不必为已授权的本地代码修改反复询问。重点依次完成 GLA 的 `BK/BV` 配置与 forward 测试拆分、NSA compression 的安全公共入口、Runner 严格退出模式、910B forward-only 回归与同协议性能基线。严格禁止修改现有 JIT Kernel 计算逻辑、吞掉失败或跳过的事实、覆盖用户文件、修改原 `npu-ghd` 分支或推送远端。如果目标 HEAD 已变化，先核对并更新审计。若遇设备/依赖等外部阻塞，保留原始错误并标记 BLOCKED；仍可继续不依赖该阻塞的单元测试与文档工作，但不可宣称最终验收通过。每阶段独立本地 commit，最后提供逐阶段的审查报告和待办。
