# Ascend 架构迁移：Forward-only 阶段记录

记录日期：2026-10-16。本文留存现有结果，不代表重新执行了一轮硬件测试。

## 版本与验收范围

- 原分支基线：`cc675a5f7f456416ab75e08ac48b079c774d533f`。
- 本次合入的固定官方版本：`17a2f54463e61f1906f3a684dc1da7413938fffd`；不是对未来最新 main 的承诺。
- 被记录的代码提交：`3c54ad09e6f8d4bd7add183617bf1ab889c66332`。
- 迁移工作分支：`chore/align-upstream-20261016`。
- 用户授权将此版本及记录推送至 `Ghd-2077/FlagAttention` 的 `npu-ghd`，不创建 PR。
- **所有算子只验收 forward；NSA backward 不在本次验收范围内。**
- 未测试、跳过、仅内部实现通过，不能等同于公共入口或全部配置通过。

## 逐算子结果

| 算子 | Forward 精度状态 | 性能状态 | 边界及待办 |
| --- | --- | --- | --- |
| GDN2 | 9 项通过，包含 FP16/BF16、长度 16/33、非零初始状态及两种状态布局等检查 | Benchmark 能运行；BF16 `[1,128,1,64,64]` 为 0.715145 ms | 83 项 CUDA 专属比较跳过，不计通过；尚无统一协议的重构前后性能验收 |
| SageAttention | 7 项通过，包含布局、分组头、mask、尾块等检查 | FP16 B1/H2/D64，T128/T256 为 0.318420/0.321860 ms | 小尺寸相对 SDPA 的加速比分别为 0.419/0.410，即慢于 SDPA；不能声称全面加速 |
| KDA | 已测 2 个长度的 BF16 前向与独立 CPU 递推参考一致 | 本轮尚未完成性能验收 | 仅覆盖已测用例，不代表全配置通过 |
| MiniMax | 1 项组合测试通过，覆盖 prefill、index score、index top-k | 本轮尚未完成性能验收 | decode 等路径未完成本轮数值验证；稀疏 attention decode 没有独立 Ascend 优化覆盖 |
| NSA compression | 6 个前向相关用例通过 | 本轮尚未完成性能验收 | 证据来自撤下公共注册之前；当前公共 override 仍未恢复，不可把内部通过等同于公共入口已接通 |
| NSA selected attention | 本轮未重新验证 | 本轮未重新验证 | selected-only 实现保留为内部入口，未注册为完整 `parallel_nsa` |
| GLA | **未通过**：长度 17/32 前向调用失败 | 被前向错误阻塞 | `dynamic_func() missing ... BK and BV`；现有调优配置为空，尚未恢复已验证配置 |
| 公共小算子 | 本轮未逐个独立完成全量验证 | 本轮未逐个验证 | 不能概括为全部通过 |

补充结果：最终 runtime/runner/registry 回归 **53 passed**，主要是 CPU/mock 契约验证，不能替代 NPU 数值验证。
新增 Ascend 数值测试集结果 **12 passed / 2 failed**，两项失败均为 GLA 前向调用。
历史 NSA compression backward 非有限梯度失败仍保留原始记录，但不计入 forward-only 的失败项。

**结论：当前是可追溯的阶段性检查点，不是所有算子精度与性能均通过的最终验收版本。**

## 架构变化

主要变化是统一公共入口、注册与必要的参数适配，不是重写优化 kernel。
已对照原基线检查 43 个 Ascend JIT 函数，kernel AST 无变化。

```text
公共 API → 官方统一注册表 → Ascend 注册入口 → 必要的适配层 → 原有优化 kernel
```

以下为相关目录，省略未变化的官方文件及其他随官方同步的算子目录。
`+` 表示相对原分支基线新增，`~` 表示修改。

```text
FlagAttention/
├── src/flag_attn/
│   ├── FLA/                         # 官方通用实现，保留
│   ├── gdn2/                        # 官方公共入口/通用实现，保留
│   ├── sage_attention/
│   ├── parallel_nsa/
│   ├── minimax_sparse_attention/
│   ├── runtime/backend/
│   │   ├── op_registry.py           + 官方统一注册表
│   │   └── _ascend/
│   │       ├── __init__.py          ~ 单一注册表
│   │       ├── ops/                   后端公共小算子
│   │       │   ├── cumsum.py
│   │       │   ├── mean_pooling.py
│   │       │   ├── quantization.py
│   │       │   ├── softplus.py
│   │       │   └── bwd_preprocess.py
│   │       ├── gdn2/
│   │       │   ├── api.py           + 状态布局与公共接口适配
│   │       │   └── chunk_fwd_infer.py
│   │       ├── gla/chunk.py
│   │       ├── kda/
│   │       │   ├── api.py           + 公共接口适配
│   │       │   ├── chunk.py
│   │       │   └── forward.py
│   │       ├── nsa/
│   │       │   ├── compression.py
│   │       │   ├── forward.py
│   │       │   └── kernels.py
│   │       ├── minimax_sparse_attention/
│   │       │   ├── index_topk.py
│   │       │   └── sparse_attn.py
│   │       ├── attn_qk_int8_per_block.py
│   │       └── sage_attention.py
│   └── testing/backend.py           + 官方设备/计时辅助
├── tests/flag_attn/
│   ├── test_ascend_attention.py      + Ascend 数值测试
│   └── test_ascend_registry.py       + 注册契约测试
├── benchmark/
│   └── test_parallel_nsa_benchmark.py  官方重命名，原为 parallel_nsa_benchmark.py
├── conf/operators.yaml              ~ 测试选择配置
├── tools/run_tests.py                ~ 收集范围与跳过性能测试选项
├── docs/migration/
│   ├── ascend-alignment.md           + 完整迁移证据
│   └── forward-only-checkpoint-20261016.md + 本记录
├── BaselineBenchmark/                  仅服务器本地存在，忽略、不提交
└── flash_mla/                          本地忽略、不提交
```

GDN2 的适配层将 V/K 初始状态转换为 kernel 接受的 K/V 布局，并转换返回状态。
KDA 的适配层对齐官方接口签名与默认值。两者均没有把融合计算拆成新的小 kernel。
公共 ops 与各算子的专属实现保持分离；不会将独立 kernel 中仅功能相似的代码强行提取为额外 launch。

## 测试环境、复现入口与限制

- 容器：`ghd-sageattn`；Python：`/usr/local/python3.11.15/bin/python`。
- 隔离验证路径：`/data/ghd/pr/flagattention-ascend-migration-20261016`。
- Ascend 910B4-1，物理设备 4；CANN 9.0.0，torch_npu 2.9.0.post2，Triton 3.5.1。
- 测试命令、原始日志位置、计时协议详见 [完整迁移记录](ascend-alignment.md)。
- GDN2/Sage benchmark 使用 NPU event 计时；迁移前记录使用同步 wall-clock，两组数据不能直接用于无性能回退结论。
- 尚未完成 P4 同协议前后对比；GitHub 自托管 NPU CI 未验证。
- 本次只记录并发布既有代码，不修复 GLA、不恢复 NSA 注册、不新增硬件测试。

## 后续事项（本次发布不包含）

1. 恢复并验证 GLA 所需的已验证 Ascend 配置。
2. 按 forward-only 契约恢复 NSA compression 公共注册并重新验证公共入口。
3. 补齐 NSA selected、MiniMax 未覆盖路径及其他算子的前向和 benchmark。
4. 执行统一协议的重构前后性能对比，再进行完整验收。
