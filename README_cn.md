<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# FlagAttention

<p align="center">
    <img src="./assets/logo/horizontal-blue.png" width="400" alt="FlagAttention">
</p>

[English](./README.md)

FlagAttention 是一个使用 [Triton](https://github.com/triton-lang/triton) 实现的内存高效 Attention 算子集合，面向需要自定义 attention score 变换、paged/sparse KV cache 布局或递归线性注意力 kernel 的训练与推理任务。

与 [FlashAttention](https://arxiv.org/abs/2205.14135) 类似，稠密算子通过分块和重计算避免实体化完整的 attention matrix。仓库还包含解码、块稀疏、量化和递归算子，它们并不局限于标准 scaled dot-product attention 接口。

> [!IMPORTANT]
> 仓库中的算子成熟度不同。目前只有 FlashAttention 和 Piecewise Attention 标记为 `stable`，其余算子族均为 `alpha`。算子阶段、测试和 benchmark 入口以 [`conf/operators.yaml`](./conf/operators.yaml) 为准。

## 算子概览

| 算子族 | 公开导入入口 | 主要布局/用途 | 梯度支持 | 阶段 |
| --- | --- | --- | --- | --- |
| FlashAttention | `flag_attn.flash_attention` | 稠密注意力；`q: [B,Hq,M,D]`、`k/v: [B,Hkv,N,D]`；支持 MQA/GQA、dropout 和辅助输出 | 前向 + 反向 | Stable |
| Piecewise Attention | `flag_attn.piecewise_attention` | 根据 token 距离在两套 Q/K 之间选择的稠密注意力 | 前向 + 反向 | Stable |
| Split-KV FlashAttention | `flag_attn.flash_attention_split_kv` | 长 KV 解码及 query 并行度较低的任务 | 前向 | Alpha |
| Paged Attention | `flag_attn.paged_attention` | 在 paged KV cache 上执行单 token query | 前向 | Alpha |
| MiniMax M3 稀疏注意力 | 六个 `flag_attn.minimax_m3_*` 函数 | 以 128 token page 为单位打分、Top-K、稀疏 prefill/decode | 推理 | Alpha |
| Chunk GLA | `flag_attn.chunk_gla` | 递归 gated linear attention；`[B,T,H,D]`，支持定长及 packed varlen | 前向 + 反向 | Alpha |
| Chunk Gated Delta Rule | `flag_attn.chunk_gated_delta_rule` | 分块 delta-rule 递归；支持 head-first/sequence-first | 前向 | Alpha |
| GDN2 | `flag_attn.chunk_gdn2` | 原生 Triton/TLE 及厂商路径的分块 GDN2 prefill | 前向/推理 | Alpha |
| Kimi Delta Attention | `flag_attn.chunk_kda` | 特化的分块 KDA 推理；约束随当前后端变化 | 推理 | Alpha |
| SageAttention | `flag_attn.sage_attention.forward`、`per_block_int8` | Q/K 分块 INT8 量化及 attention；支持 HND/NHD | 前向 | Alpha |
| Parallel NSA | `flag_attn.parallel_nsa.parallel_nsa`、`parallel_nsa_compression` | Native Sparse Attention 及压缩；支持定长和 packed varlen | 前向 + 反向 | Alpha |

顶层包导出了稠密、paged、递归、KDA/GDN2 和 MiniMax API；SageAttention 与 Parallel NSA 使用子模块入口。`flag_attn.runtime.backend._<vendor>` 下还存在厂商开发接口，但这些路径不视为稳定公开 API。

## 依赖与后端

- Python 3.10 或更高版本。
- 与目标加速器匹配的 PyTorch build。
- Triton 2.2 或更高版本，或提供兼容 Triton API 的厂商运行时。
- 执行 kernel 需要支持的加速器；纯 CPU 环境可以读取包元数据，但不能运行 attention kernel。

[`pyproject.toml`](./pyproject.toml) 没有将 PyTorch 和加速器运行时声明为硬依赖，因为正确的软件包取决于设备和驱动栈。请先安装适合目标设备的 PyTorch，再安装 FlagAttention。

运行时能够识别 `nvidia`、`amd`、`hygon`、`iluvatar`、`metax`、`enflame`、`ascend`、`cambricon`、`mthreads`、`intel` 和 `cpu`，用于设备元数据及显式选择。能够识别后端并不表示每个算子都已在该后端实现或验证；算子级覆盖情况请查看 [`conf/operators.yaml`](./conf/operators.yaml)、[`src/flag_attn/runtime/backend`](./src/flag_attn/runtime/backend) 中的后端包以及对应测试。

后端派发也因算子而异。顶层 GDN2/KDA API 会为 Enflame、MetaX 和 MThreads 选择特化实现，MiniMax M3 会选择 MetaX 实现；其他厂商扩展可能需要使用对应的后端子模块。不能假定每个顶层 API 都会派发到所有已识别厂商。

设备元数据接口遵循 FlagGems/FlagGems-vLLM 约定：

```python
import flag_attn

print(flag_attn.vendor_name)  # 例如 "nvidia"、"metax"、"enflame"
print(flag_attn.vendor)       # vendor_name 的别名
print(flag_attn.device)       # 例如 "cuda"、"gcu"、"npu"
print(flag_attn.backend_info) # 结构化 DeviceDetector 对象
```

默认根据可用的 PyTorch 设备和 Triton target 自动识别，也可以在导入包之前覆盖：

```sh
FLAG_ATTN_BACKEND=metax python your_program.py
# FLAG_ATTN_VENDOR 是别名；FLAG_ATTN_BACKEND 优先级更高。
```

## 安装

克隆仓库，并先安装适合目标加速器的 PyTorch/运行时。标准上游 Triton 开发环境可以执行：

```sh
git clone https://github.com/flagos-ai/FlagAttention.git
cd FlagAttention
pip install -e ".[triton,test]"
```

如果当前环境已经提供 Triton 或兼容的厂商 fork，不要安装 `triton` extra：

```sh
pip install -e ".[test]"
```

当前包初始化过程会加载 YAML 调优配置和 PyTorch 参考实现，因此目前所有安装方式都还需要 PyYAML 和 pytest；可编辑安装的 `test` extra 会安装这两个依赖。仅当某个算子明确要求尚未发布的功能（例如匹配版本的 TLE）时，才建议使用 Triton nightly。

构建 wheel 或源码发行包：

```sh
pip install -U build setuptools setuptools-scm
python -m build --no-isolation
pip install PyYAML pytest
pip install dist/flag_attn-*.whl
```

项目没有 `setup.py`，构建使用 PEP 517 和 setuptools-scm。Debian/RPM 运行时说明见 [`packaging/INSTALL.md`](./packaging/INSTALL.md)。当前发行包的依赖声明与安装说明不一定会同时提供 PyYAML 和 pytest，请确保运行 `import flag_attn` 的 Python 环境中已安装这两个包。

## 快速开始

### FlashAttention

```python
import torch
from flag_attn import flash_attention

B, Hq, Hkv, M, N, D = 2, 16, 4, 2048, 4096, 128
q = torch.randn(B, Hq, M, D, device="cuda", dtype=torch.float16,
                requires_grad=True)
k = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16,
                requires_grad=True)
v = torch.randn(B, Hkv, N, D, device="cuda", dtype=torch.float16,
                requires_grad=True)

out = flash_attention(q, k, v, causal=True)
out.sum().backward()
```

完整接口为：

```python
flash_attention(
    q, k, v,
    causal=False,
    sm_scale=None,
    dropout_p=0.0,
    return_log_normalizer=False,
    return_total_attention=False,
    return_seed_offset=False,
)
```

`Hq` 必须能被 `Hkv` 整除，`D` 必须是 `16`、`32`、`64` 或 `128`。当 Q/KV 长度不同时，causal mask 采用右下角对齐。实现会根据 GPU 占用率自动选择普通前向或 Split-KV 前向。

只要启用任意辅助返回开关，函数就固定返回五个值；未启用的字段为 `None`：

```python
out, lse, total, seed, offset = flash_attention(
    q, k, v,
    return_log_normalizer=True,
    return_total_attention=True,
)
```

- `lse`：`[B, Hq, M]`，每行的 log-normalizer。
- `total`：`[B, Hq, N]`，attention probability 沿 query 轴求和的结果。
- `seed` 和 `offset`：启用 dropout 且请求返回时使用的 Philox 状态。

当占用率启发式选择 Split-KV 路径时不支持 dropout；这类形状需要使用非 Split-KV 配置或设置 `dropout_p=0`。

### Piecewise Attention

```python
import torch
from flag_attn import piecewise_attention

B, H, M, N, D = 1, 2, 128, 128, 64
q1 = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
q2 = torch.randn_like(q1)
k1 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k2 = torch.randn_like(k1)
v = torch.randn_like(k1)

out = piecewise_attention(
    q1, k1, q2, k2, v,
    dist_threshold=32,
    causal=True,
)
```

对于 query 行 `m` 和 key 列 `n`，当带符号的右下角对齐偏移 `N - M + m - n >= dist_threshold` 时使用 `q2 @ k2`，否则使用 `q1 @ k1`。随后采用与 FlashAttention 相同的分块 online softmax，反向阶段再把 `dScore` 分配给两套 Q/K。`q1/q2` 的布局为 `[B,H,M,D]`，`k1/k2/v` 为 `[B,H,N,D]`；所有输入的 head 数相同（不支持 GQA），`D` 必须是 `16`、`32`、`64` 或 `128`。

Piecewise Attention 最初用于 NLPE（Non-Linearized Position Embedding），[Aquila-2-34B](https://github.com/FlagAI-Open/Aquila2) 使用它在距离超过阈值时切换位置表示。

[`examples`](./examples) 中还包含面向 CUDA 的脚本，但其中部分属于历史示例。例如，[`examples/flash_attention_with_aux_outputs.py`](./examples/flash_attention_with_aux_outputs.py) 仍只解包三个返回值，而当前辅助输出 API 固定返回五个值；在该脚本更新前请使用上面的五值示例。测试代码是更权威的调用示例。用于验证的 PyTorch 参考实现通过 `flag_attn.testing` 暴露。

## 推理与稀疏 API

### Split-KV 与 Paged Attention

```python
from flag_attn import flash_attention_split_kv, paged_attention
```

`flash_attention_split_kv` 是面向长 KV 任务的显式纯前向 API。它接收 `q: [B,Hq,M,D]` 和 `k/v: [B,Hkv,N,D]`，要求 `Hq % Hkv == 0`，并支持 `D` 属于 `{16,32,64,128}`。当实际选择多个 split 时，每个 split 分别计算局部输出和 log-normalizer，第二个 kernel 再通过全局 logsumexp 合并；只有一个 split 时直接返回结果。

`paged_attention` 的输入为：

```text
query:        [num_sequences, num_query_heads, head_size]
key_cache:    [num_blocks, num_kv_heads, block_size, head_size]
value_cache:  [num_blocks, num_kv_heads, block_size, head_size]
context_lens: [num_sequences]
block_tables: [num_sequences, max_blocks_per_sequence]
```

算子会自动选择单次计算或 partition + reduce 实现，也可用 `num_splits` 覆盖选择。K/V cache 必须具有相同的 shape 和 stride；支持的 head size 为 `{16,32,64,128,256,512}`，当 `num_query_heads > num_kv_heads` 时，cache block size 必须至少为 16 个 token。显式设置 `num_splits > 1` 时，计算得到的 partition size 还必须不小于 cache block size，且能被它整除。完整示例见 [`examples/paged_example.py`](./examples/paged_example.py)。

调用方式为 `paged_attention(query, key_cache, value_cache, context_lens, block_tables, attn_scale, max_context_len, num_splits=0)`，返回与 `query` 同形状的输出。`attn_scale` 是 softmax 缩放系数（通常为 `head_size**-0.5`），`max_context_len` 是 batch 内最大的上下文长度。

### MiniMax M3 稀疏注意力

M3 路径使用兼容 vLLM 的 paged cache，稀疏 block 固定为 128 token：

```text
Prefill: minimax_m3_index_score
      -> minimax_m3_index_topk
      -> minimax_m3_sparse_attn

Decode: minimax_m3_index_decode（score + top-k）
     -> minimax_m3_sparse_attn_decode

仅计算 decode score：minimax_m3_index_decode_score
```

稀疏 attention kernel 支持 GQA 和 BF16 KV cache，并在支持的硬件上支持带 scale 的 FP8 cache。这些函数仅用于推理，由调用方提供 paged cache、序列元数据和 block table。稀疏 prefill/decode 必须传入 `output` buffer，结果写入该 buffer，函数返回 `None`；索引打分/Top-K 函数返回 tensor，其中部分函数可选择传入复用的输出 buffer。精确参数见 [`src/flag_attn/minimax_sparse_attention`](./src/flag_attn/minimax_sparse_attention) 中的函数签名，端到端用法见 [`tests/flag_attn/test_minimax_sparse_attention.py`](./tests/flag_attn/test_minimax_sparse_attention.py)。

### SageAttention

```python
import torch
from flag_attn.sage_attention import forward as sage_attention
from flag_attn.sage_attention import per_block_int8

B, H, N, D = 1, 2, 128, 64
q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)

q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, tensor_layout="HND")
out, lse = sage_attention(
    q_int8, k_int8, v, q_scale, k_scale,
    tensor_layout="HND",
    return_lse=True,
)
```

支持 `HND`（`[B,H,N,D]`）和 `NHD`（`[B,N,H,D]`）布局。默认 Q 以 128 token 为一块量化，K 以 64 token 为一块量化；前向 kernel 支持 bool mask 和 additive mask。

`forward` 始终返回 `(out, lse)`；当 `return_lse=False` 时，`lse` 是一个空的 CPU tensor。公开的 `flag_attn.sage_attention` 子模块会在可用 NPU 上自动选择 Ascend 实现；其他厂商的 SageAttention 特化实现需要使用其后端开发模块。

## 递归与线性注意力 API

GLA、GDN2 和 KDA 使用 sequence-first `[B,T,H,D]`。Gated Delta Rule 默认 `head_first=True`，布局为 `[B,H,T,D]`，同时也接受 sequence-first 输入。大多数递归接口返回 `(output, final_state)`，并通过 `cu_seqlens` 支持 packed varlen。

- [`chunk_gla`](./src/flag_attn/FLA/gated_linear_attention/chunk_gla.py) 实现带 autograd 的分块 gated linear attention，支持初始/最终递归状态。
- [`chunk_gated_delta_rule`](./src/flag_attn/FLA/gated_delta_rule/api.py) 是纯前向 delta-rule 实现；`head_first=True` 时布局为 `[B,H,T,D]`，当前 `BT` 固定为 64。
- [`chunk_gdn2`](./src/flag_attn/gdn2/chunk.py) 在原生 Triton、TLE 和厂商特化的 GDN2 前向路径之间派发。
- [`chunk_kda`](./src/flag_attn/FLA/chunk_kda.py) 是推理 API。通用 CUDA 路径要求 Triton TLE 3.6 或更高版本、inference mode、BF16 输入、`K=V=128`、chunk size 16、V-first state 和 gate 参数；厂商实现的约束可能不同。
- [`parallel_nsa`](./src/flag_attn/parallel_nsa) 包含 Native Sparse Attention 选择及压缩算子，支持定长和 packed varlen。`parallel_nsa` 必须提供预计算的 `block_indices` 或 `g_cmp`，且 `Hq/Hkv` 必须是至少 16 的 2 的幂；组合 sliding-window 时还需要 `g_swa` 和外部 `flash-attn` 包。对于定长输入，Compression API 接收全长 Q，以及时间维为 `ceil(T / block_size)` 的 K/V，返回 `(output, lse)`；packed 输入会依次存放各序列的压缩块。登记的 NSA 测试直接导入 Enflame 专用模块，不能据此推断公开的 `flag_attn.parallel_nsa` 子模块已在其他后端得到验证。

这些 alpha API 针对特定模型布局进行了优化。集成前请阅读对应 docstring 和测试，其参数与后端约束仍可能变化。

## 测试

安装开发依赖，并在不探测 GPU 的情况下查看完整算子清单：

```sh
pip install -e ".[triton,test]"
python tools/run_tests.py --list-ops --stages all
```

在兼容的 CUDA 环境中执行完整 pytest：

```sh
pytest tests
```

稠密算子会与 FP32 PyTorch reference 比较。精度检查要求 Triton 相对 FP32 reference 的误差不超过同精度 PyTorch 误差的大约两倍，并留有很小的绝对误差余量。当前通用测试中仍有不少用例直接假定 CUDA；在其他加速器上应选择对应厂商测试或算子清单项，而不能假定所有 CUDA 用例都会自动跳过。

基于算子清单的调度器默认只执行 `stable` 算子，并可把任务分发到指定 CUDA GPU：

```sh
python tools/run_tests.py --stages stable --gpus 0 --skip-benchmarks
python tools/run_tests.py --stages all --gpus all --dump-output
```

使用与 FlagGems 兼容的 pytest recorder 输出用例级 JSON：

```sh
pytest -m "sage_attention" --record json --output accuracy_sage_attention.json -vs
```

省略 `--output` 时默认写入 `accuracy_result.json`，已有报告按 pytest node ID 合并。

## 性能测试

已有的 benchmark 入口会随算子登记在 [`conf/operators.yaml`](./conf/operators.yaml) 中，并非每个清单项都有 benchmark。可以通过统一调度器运行，也可以按文件类型直接执行：

```sh
python tools/run_tests.py --stages all --gpus 0

python benchmark/flash_benchmark.py
python benchmark/piecewise_benchmark.py
python benchmark/flash_decoding_benchmark.py

cd benchmark
pytest -m "sage_attention" --record json --output benchmark_sage_attention.json -vs
```

调度器命令会先执行精度测试、再运行 benchmark，并非仅测性能。每个 benchmark 的 `performance_stdout.log` 会保存在输出目录的算子子目录；以 `[INFO] {` 开头的单行 JSON 与 FlagGems benchmark log 格式一致，包含输入形状、基线延迟、FlagAttention 延迟和可计算时的加速比。没有基线的测试会把 `speedup` 记为 `null`。Benchmark 同时保留原有的表格、图表和基于矩阵乘运算量计算的吞吐率；`flash_decoding_benchmark.py` 的图表纵轴现在正确标为毫秒。历史 v0.2 图表仍保存在 [`assets/v0.2`](./assets/v0.2)；评估当前代码、Triton 和硬件时应重新运行当前 benchmark。

## 仓库结构

```text
FlagAttention/
├── src/flag_attn/
│   ├── flash.py, piecewise.py, split_kv.py, paged.py
│   ├── minimax_sparse_attention/   # MiniMax M3 索引与稀疏注意力
│   ├── sage_attention/             # INT8 Q/K 量化与注意力
│   ├── parallel_nsa/               # NSA 选择与压缩
│   ├── FLA/                        # GLA、Gated Delta Rule、KDA 辅助实现
│   ├── gdn2/                       # 通用 GDN2 实现
│   ├── runtime/backend/            # 设备识别与厂商后端
│   └── testing/                    # PyTorch 参考实现
├── tests/                          # 精度与派发测试
├── benchmark/                      # 性能测试入口
├── examples/                       # 小型使用示例
├── conf/operators.yaml             # 算子阶段/测试/benchmark 清单
├── tools/run_tests.py              # 多设备测试调度器
└── packaging/                      # Debian/RPM 打包
```

## 当前限制

- 所有计算 kernel 都需要支持的加速器；CPU 仅用于元数据和参考实现。
- 不同后端的算子与 dtype 覆盖范围不同，不能仅根据后端识别成功就推断算子可用。
- FlashAttention 和 Piecewise Attention 的 head dimension 必须属于 `{16, 32, 64, 128}`；paged 和特化算子有各自约束。
- 当 FlashAttention 自动派发到 Split-KV 时不能使用 dropout。
- Split-KV、Paged、MiniMax M3、递归/线性、SageAttention 和 NSA API 当前仍为 alpha。
- 一些 TLE 路径要求特定的较新 Triton build，shape 约束也比原生 fallback 更严格。

## 更多

FlagAttention 属于 FlagOS/FlagOpen 开源生态。更多项目请访问 [FlagOpen](https://flagopen.baai.ac.cn/)。

[<img src="./assets/logo/baai-flagopen.jpeg" alt="BAAI FlagOpen">](https://flagopen.baai.ac.cn/)
