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
> 仓库中的算子成熟度不同。目前只有 FlashAttention 和 Piecewise Attention 标记为 `stable`，其余算子族均为 `alpha`，其参数和后端约束仍可能变化。算子阶段、测试和 benchmark 入口以 [`conf/operators.yaml`](./conf/operators.yaml) 为准。

## 算子概览

[`conf/operators.yaml`](./conf/operators.yaml) 中登记了 26 个清单项，覆盖下列算子族。只有 FlashAttention 和 Piecewise Attention 为 `stable`，其余均为 `alpha`。

### 稠密注意力

| 算子族 | 公开导入入口 | 主要布局/用途 | 梯度支持 | 阶段 |
| --- | --- | --- | --- | --- |
| FlashAttention | `flag_attn.flash_attention` | 稠密注意力；`q: [B,Hq,M,D]`、`k/v: [B,Hkv,N,D]`；支持 MQA/GQA、dropout 和辅助输出 | 前向 + 反向 | Stable |
| Piecewise Attention | `flag_attn.piecewise_attention` | 根据 token 距离在两套 Q/K 之间选择的稠密注意力 | 前向 + 反向 | Stable |
| Split-KV FlashAttention | `flag_attn.flash_attention_split_kv` | 长 KV 解码及 query 并行度较低的任务 | 前向 | Alpha |
| Paged Attention | `flag_attn.paged_attention` | 在 paged KV cache 上执行单 token query | 前向 | Alpha |

### 稀疏与块稀疏注意力

| 算子族 | 公开导入入口 | 主要布局/用途 | 梯度支持 | 阶段 |
| --- | --- | --- | --- | --- |
| MiniMax M3 稀疏注意力 | 六个 `flag_attn.minimax_m3_*` 函数 | 以 128 token page 为单位打分、Top-K、稀疏 prefill/decode | 推理 | Alpha |
| Parallel NSA | `flag_attn.parallel_nsa`、`flag_attn.parallel_nsa_compression` | Native Sparse Attention 及压缩；支持定长和 packed varlen | 前向 + 反向 | Alpha |
| InfLLM-V2 | `flag_attn.infllmv2_attention`、`infllmv2_decode` | 连续 packed 因果 prefill 配合块选择，以及变长 decode | 前向 + 反向（prefill） | Alpha |
| Parallel MoBA | `flag_attn.parallel_moba` | 变长 Mixture-of-Block-Attention，按因果 chunk 路由 | 前向 + 反向 | Alpha |
| Hy3 Attention | `flag_attn.hy3_attention` | FP8 paged 块稀疏 prefill 及六种 BF16/FP8 decode 变体 | 推理 | Alpha |
| Inkling FA4 相对注意力 | `flag_attn.inkling_fa4_rel_attention` | 带相对位置 logits 的 paged varlen 注意力；Triton 与 Hopper TLE | 推理 | Alpha |

### 递归与线性注意力

| 算子族 | 公开导入入口 | 主要布局/用途 | 梯度支持 | 阶段 |
| --- | --- | --- | --- | --- |
| Chunk GLA | `flag_attn.chunk_gla` | 递归 gated linear attention；`[B,T,H,D]`，支持定长及 packed varlen | 前向 + 反向 | Alpha |
| Chunk Gated Delta Rule | `flag_attn.chunk_gated_delta_rule` | 分块 delta-rule 递归；支持 head-first/sequence-first | 前向 | Alpha |
| GDN2 | `flag_attn.chunk_gdn2` | 原生 Triton/TLE 及厂商路径的分块 GDN2 prefill | 前向/推理 | Alpha |
| Kimi Delta Attention | `flag_attn.chunk_kda` | 特化的分块 KDA 推理；约束随当前后端变化 | 推理 | Alpha |
| Log-Linear Attention | `flag_attn.log_linear_attn`、`chunk_log_linear_attn` | 带多层级门控缩放的稠密对数线性注意力；TLE kernel | 前向 | Alpha |
| Parallax | `flag_attn.parallel_parallax` | 参数化局部线性注意力；含 prefill、decode 和 KV-cache decode 三个阶段 | 前向 + 反向（prefill） | Alpha |
| Wall Attention | `flag_attn.parallel_wall_attn` | 带门控的窗口注意力，可选 sink bias，支持 packed varlen | 前向 + 反向 | Alpha |

### 量化与融合算子

| 算子族 | 公开导入入口 | 主要布局/用途 | 梯度支持 | 阶段 |
| --- | --- | --- | --- | --- |
| SageAttention | `flag_attn.sage_attention`、`from flag_attn.sageattention import per_block_int8` | Q/K 分块 INT8 量化及 attention；支持 HND/NHD | 前向 | Alpha |
| DiffKV Attention | `flag_attn.diffkv_attention` | K/V head dimension 不对称的 paged decode；Triton 与 TLE 路径 | 推理 | Alpha |
| Forgetting Attention | `flag_attn.parallel_forgetting_attn` | 带逐 token 对数遗忘门的注意力；自动选择 Hopper TLE 或通用 Triton | 前向 | Alpha |
| Fused AttnRes | `flag_attn.fused_attnres` | 对 RMS 归一化后的残差源做 softmax 混合，可选融合输出 RMSNorm | 推理 | Alpha |

[`conf/operators.yaml`](./conf/operators.yaml) 中列出的算子均可通过 `flag_attn.<name>` 访问。SageAttention、Native Sparse Attention、DiffKV 和 Forgetting Attention 的实现分别位于 `flag_attn.sageattention`、`flag_attn.nsa`、`flag_attn.diffkv` 和 `flag_attn.forgetting`。`flag_attn.runtime.backend._<vendor>` 下还存在厂商开发接口，但这些路径不视为稳定公开 API。

## 依赖与后端

- Python 3.10 或更高版本。
- 与目标加速器匹配的 PyTorch build。
- Triton 2.2 或更高版本，或提供兼容 Triton API 的厂商运行时。
- 执行 kernel 需要支持的加速器；纯 CPU 环境可以读取包元数据，但不能运行 attention kernel。

[`pyproject.toml`](./pyproject.toml) 没有将 PyTorch 和加速器运行时声明为硬依赖，因为正确的软件包取决于设备和驱动栈。请先安装适合目标设备的 PyTorch，再安装 FlagAttention。

运行时能够识别 `nvidia`、`amd`、`hygon`、`iluvatar`、`metax`、`thead`、`enflame`、`ascend`、`cambricon`、`mthreads`、`intel` 和 `cpu`，用于设备元数据及显式选择。能够识别后端并不表示每个算子都已在该后端实现或验证；算子级覆盖情况请查看 [`conf/operators.yaml`](./conf/operators.yaml)、[`src/flag_attn/runtime/backend`](./src/flag_attn/runtime/backend) 中的后端包以及对应测试。

后端派发因算子而异，且基于名字查表。每个厂商包声明一张 `_OPERATOR_EXPORTS` 表：当表中出现某个公开 API 的名字时解析到厂商实现，否则保留通用 Triton 实现。已声明但导入或执行失败的覆盖会直接报错，而不是静默回落。通用实现以 NVIDIA 为基准，因此 `_nvidia` 不声明任何覆盖，只提供调优配置。

各厂商包当前提供覆盖的算子族如下：

| 后端 | 设备类型 | 提供厂商实现的算子族 |
| --- | --- | --- |
| `_nvidia` | `cuda` | 无覆盖；使用通用 Triton 路径及 `tune_configs.yaml` |
| `_metax` | `cuda` | MiniMax M3、SageAttention、NSA、GLA、KDA、GDN2、AttnRes |
| `_hygon` | `cuda` | MiniMax M3、SageAttention、NSA、GLA、KDA、AttnRes，以及私有的 `mla` 算子族 |
| `_mthreads` | `musa` | MiniMax M3、SageAttention、NSA、GLA、KDA、GDN2、MoBA |
| `_enflame` | `gcu` | MiniMax M3、SageAttention、NSA、GLA、KDA、GDN2、AttnRes、Parallax |
| `_ascend` | `npu` | Gated Delta Rule、Parallax、GDN2、SageAttention |
| `_thead` | `cuda` | MiniMax M3、Wall Attention |
| `_iluvatar` | `cuda` | 尚未声明覆盖 |

需要注意的是，两个 `stable` 的稠密算子族 FlashAttention 和 Piecewise Attention 没有任何厂商覆盖，在所有后端上都运行通用 Triton kernel。不能假定每个顶层 API 都会派发到所有已识别厂商。

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
from flag_attn import sage_attention
from flag_attn.sageattention import per_block_int8

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

`sage_attention` 始终返回 `(out, lse)`。当 `return_lse=False` 时，`lse` 是一个空的 CPU tensor。`flag_attn.sage_attention` 根据检测到的后端选择已注册的实现，例如在可用 NPU 上选择 Ascend 实现。

## 递归、线性与其余算子 API

### 递归与线性注意力

GLA、GDN2 和 KDA 使用 sequence-first `[B,T,H,D]`。Gated Delta Rule 默认 `head_first=True`，布局为 `[B,H,T,D]`，同时也接受 sequence-first 输入。大多数递归接口返回 `(output, final_state)`，并通过 `cu_seqlens` 支持 packed varlen。

- [`chunk_gla`](./src/flag_attn/FLA/gated_linear_attention/chunk.py) 实现带 autograd 的分块 gated linear attention，支持初始/最终递归状态。
- [`chunk_gated_delta_rule`](./src/flag_attn/FLA/gated_delta_rule/api.py) 是纯前向 delta-rule 实现；`head_first=True` 时布局为 `[B,H,T,D]`，当前 `BT` 固定为 64。
- [`chunk_gdn2`](./src/flag_attn/gdn2/__init__.py) 通过厂商注册表派发。[通用前向实现](./src/flag_attn/gdn2/chunk.py) 使用标准 Triton。[NVIDIA 后端](./src/flag_attn/runtime/backend/_nvidia/gdn2/chunk.py) 在 GPU 和 TLE 编译器均兼容时使用 16-token chunk 融合路径。
- [`chunk_kda`](./src/flag_attn/FLA/chunk_kda.py) 是推理 API。通用 CUDA 路径需要提供 TLE 的 Triton build，并且会拒绝开启梯度的调用、非 BF16 或非连续输入、`K != 128`、`V != 128` 以及 GVA。chunk size 默认为 16；厂商实现的约束可能不同。
- [`log_linear_attn`](./src/flag_attn/FLA/log_linear_attn/chunk_tle.py) 是基于 TLE kernel 的稠密纯前向 Log-Linear Attention 入口，接收 `(q, k, v, g, level_scales)`，其中 `level_scales` 提供各层级的门控缩放。`chunk_log_linear_attn` 是同一函数的清单名称。
- [`parallel_parallax`](./src/flag_attn/FLA/parallax) 是参数化局部线性注意力，单一公开入口通过 `stage=` 选择三个阶段。Prefill（`"prefill"`，默认）接收 `(q, r, k, v)`，其中 `q/r: [B,T,HQ,D]`、`k/v: [B,T,H,D]`，支持 GQA、autograd、可选 `window_size`，并通过 `cu_seqlens` 支持 packed varlen。注意 `r` 不会被 `scale` 预先缩放。`"decode"` 和 `"kvcache"` 阶段则路由到 KV-cache decode kernel。
- [`parallel_wall_attn`](./src/flag_attn/FLA/wall_attn/parallel.py) 是带门控的窗口注意力，接收 `(q, k, v, g)`，可选 `g_scalar`、`sink_bias`、`window_size` 和 `cu_seqlens`，提供前向与反向。公开入口会校验输入并在经过数值检查的快速路径与通用实现之间选择；校验需要与 host 同步，因此 graph capture 场景始终使用通用实现。

### Native Sparse Attention

[`parallel_nsa`](./src/flag_attn/nsa) 包含 Native Sparse Attention 选择及压缩算子，支持定长和 packed varlen，通过 `mode=` 选择。`parallel_nsa` 必须提供预计算的 `block_indices` 或 `g_cmp`，且 `Hq/Hkv` 必须是至少 16 的 2 的幂；组合 sliding-window 时还需要 `g_swa` 和外部 `flash-attn` 包。对于定长输入，Compression API 接收全长 Q，以及时间维为 `ceil(T / block_size)` 的 K/V，返回 `(output, lse)`；packed 输入会依次存放各序列的压缩块。

### 其他稀疏与融合算子

- [`infllmv2_attention`](./src/flag_attn/infllmv2) 默认执行连续 packed 因果 prefill，`stage="decode"` 时执行变长 decode。Prefill 接收 packed 的 `q/k/v` 以及 `cu_seqlens_q`、`cu_seqlens_k` 和两个最大长度；每个 Q 序列都是其 KV 序列的后缀，因此 KV 必须包含已缓存前缀和当前 chunk。块选择参数位于 `InfLLMV2Config`，目前仅支持因果模式。梯度可以回传到传入的所有 Q/K/V。该算子不执行 paged cache 或原地 cache 更新。
- [`parallel_moba`](./src/flag_attn/FLA/moba/parallel.py) 是面向 packed `[seqlen, heads, head_dim]` 张量的因果变长 MoBA，接收 `cu_seqlens`、`max_seqlen`、`moba_chunk_size` 和 `moba_topk`。`moba_topk` 包含当前因果 chunk，与 FlashMoBA 一致。支持 MHA/GQA/MQA、最大 256 的 head dimension 以及最大 64 的 top-k。推理路径有意跳过对依赖具体数值的元数据的 host 侧校验，autograd 路径仍保留该校验。
- [`hy3_attention`](./src/flag_attn/hpc_ops_attention) 提供 FP8 paged 块稀疏 prefill（`stage="prefill"`）以及由 `variant=` 选择的六种 decode 实现：`bf16_static`、`bf16_dynamic`、`fp8_qk_static`、`fp8_qk_dynamic`、`fp8_kv_static`、`fp8_kv_dynamic`。其中 `qk` 表示逐 token 量化 Q/K 并逐 head 量化 V，`kv` 表示逐 token 量化 Q 并逐 tensor 量化 K/V；`static` 与 `dynamic` 选择调度策略。Decode 会原样转发 `(inputs, workspace)`，因此需要先用对应实现模块准备 workspace。该算子族要求 PyTorch 2.7.0。
- [`inkling_fa4_rel_attention`](./src/flag_attn/inkling_fa4) 是仅用于推理的 paged varlen 注意力，并叠加相对位置 logits，参数与同名 vLLM 算子一致。Cache 使用 `[num_blocks, block_size, num_kv_heads, head_dim]` 布局。`backend=` 接受 `"auto"`（默认）、`"triton"` 或 `"tle"`，环境变量 `INKLING_FA4_BACKEND` 可做同样选择；auto 优先使用 TLE，当 TLE 不可用或布局不支持时回落到 Triton。
- [`diffkv_attention`](./src/flag_attn/diffkv/api.py) 是 K/V head dimension 不对称的 paged decode：`query: [B,Hq,Dqk]`、`key_cache: [NB,BS,Hkv,Dqk]`、`value_cache: [NB,BS,Hkv,Dv]`，返回 `[B,Hq,Dv]`。`window_size=-1` 表示完整上下文；`path` 选择 `"2d"` 或 `"3d"` launch，`num_segments` 控制 3D 路径的 split-KV；`backend` 接受 `"auto"`、`"tle"` 或 `"triton"`。
- [`parallel_forgetting_attn`](./src/flag_attn/forgetting/parallel.py) 通过 `log_fgate` 引入逐 token 的对数遗忘门，另有 `head_first=`、`seq_start=`、`sm_scale=` 和 `adaptive_threshold=`。通用实现保留在 `forgetting/`，使用标准 Triton 内核和跨后端的主机侧预处理；Hopper TMA/WGMMA 内核及专用前缀扫描位于 `runtime/backend/_nvidia/hopper/forgetting`，NVIDIA 启动配置位于 `runtime/backend/_nvidia/forgetting`。运行时自动选择兼容的 Hopper 实现，否则使用通用路径，支持 16、32、60、64、100、128 这些 head dimension。该接口仅支持前向。启动 Python 前设置 `FLAG_ATTN_FORCE_GENERIC=1`，即可通过相同顶层 API 和 `pytest -m parallel_forgetting_attn` 验证通用实现。顶层 `naive_forgetting_attention`、`prepare_forgetting_attention` 和 `capture_forgetting_attention` 分别提供测试使用的参考实现、前缀与剪枝边界准备、启动跟踪接口。
- [`fused_attnres`](./src/flag_attn/FLA/attnres/fused.py) 聚合多个残差源：每个源先做 RMS 归一化用于与 `query` 打分，再由源维度上的 softmax 混合原始残差，`output_rms_weight` 可选地融合下一子层所需的 RMSNorm。该算子仅用于推理、不提供 autograd 公式，针对 Kimi K3 形状优化，支持最多 16 个源和最大 8192 的 hidden size。

这些 alpha API 针对特定模型布局进行了优化。集成前请阅读对应 docstring 和测试，其参数与后端约束仍可能变化。

## 测试

安装开发依赖：

```sh
pip install -e ".[triton,test]"
```

算子清单本身就是 [`conf/operators.yaml`](./conf/operators.yaml)，直接阅读该文件即可了解已登记的阶段、测试和 benchmark 入口。

在兼容的 CUDA 环境中执行完整 pytest：

```sh
pytest tests
```

稠密算子会与 FP32 PyTorch reference 比较。精度检查要求 Triton 相对 FP32 reference 的误差不超过同精度 PyTorch 误差的大约两倍，并留有很小的绝对误差余量。当前通用测试中仍有不少用例直接假定 CUDA；在其他加速器上应选择对应厂商测试或算子清单项，而不能假定所有 CUDA 用例都会自动跳过。

基于算子清单的调度器默认只执行 `stable` 算子，并可把任务分发到指定 CUDA GPU：

```sh
python tools/run_tests.py --stages stable --gpus 0
python tools/run_tests.py --stages all --gpus all --dump-output
python tools/run_tests.py --ops "sage_attention,chunk_gla" --gpus 0
```

算子选择的优先级为 `--ops` > `--op-list-file` > `--stages`。

使用 pytest recorder 输出与 FlagGems 兼容的用例级 JSON；同目录另写算子级上传汇总 `summary.json`：

```sh
pytest -m "sage_attention" --record json --output accuracy_sage_attention.json --continue-on-collection-errors -vs
```

`--output` 文件只保留与 FlagGems 兼容的原始用例或算子记录；同目录的 `summary.json` 顶层包含 `timestamp`、`env` 和 `result`。每个 `result[算子名]` 分别记录 `accuracy` 与 `performance`；`performance.data` 按 dtype 汇总具有有效基线的各 shape 延迟和加速比。从仓库根目录无路径运行 `--record json -m OP_NAME` 时，也会收集 `benchmark/`。省略 `--output` 时，包含 benchmark 的命令默认写入 `benchmark_result.json`，仅运行精度测试时写入 `accuracy_result.json`。即使复用输出路径，上传汇总也只描述本次运行。

`OP_NAME` 是 pytest marker。算子清单中的 `chunk_gated_delta_rule_fwd` 和 `chunk_kda_enflame` 与原有的短名称 marker 都可使用。没有匹配测试的 marker 会得到 `NotRun` 精度、`Skipped` 性能，且没有加速比数据。

## 性能测试

已有的 benchmark 入口会随算子登记在 [`conf/operators.yaml`](./conf/operators.yaml) 中，并非每个清单项都有 benchmark。可以通过统一调度器运行，也可以按文件类型直接执行：

```sh
python tools/run_tests.py --stages all --gpus 0

python benchmark/flash_benchmark.py
python benchmark/piecewise_benchmark.py
python benchmark/flash_decoding_benchmark.py

cd benchmark
pytest -m "sage_attention" --record json --output performance_result.json --continue-on-collection-errors -vs
```

指定的 `performance_result.json` 会在算子键下记录每个 shape 的基线延迟、FlagAttention 延迟及 `speedup = 基线延迟 / FlagAttention 延迟`。只运行 benchmark 时，单独的 `summary.json` 会把精度标为未运行；benchmark 通过不代表精度测试集通过。没有有效基线的 shape 不会在任一文件中虚构加速比。上传平台 zip 时，将生成的 `summary.json` 置于压缩包根目录或一级子目录。

无路径的 pytest 命令只收集 `benchmark/` 下的 pytest 性能测试。独立 benchmark 脚本仍需直接运行，它们的数据不会写入这个 pytest JSON 文件。

### 与 NVIDIA 基线对比

[`baseline/`](./baseline) 中保存了 NVIDIA 参考实现，每个算子一个模块，各自声明形状、dtype、输入构造方式以及所测量的上游可调用对象。将由这些参考实现产出的基线 JSON 传入，即可逐形状对比非 NVIDIA 平台的运行结果：

```sh
pytest benchmark -m "chunk_gla" --record json --base-data nvidia_baseline.json -vs
```

匹配成功的形状会在本机 `speedup` 之外额外记录 `compared_speedup`。当基线为当前芯片提供了 `_scaling_factors` 时，该对比值会除以与该形状瓶颈资源对应的缩放系数，使结果反映效率而非单纯的硬件差异。未匹配的形状会记录 `base_miss_reason`；既无本机参考又未匹配到基线的形状会使本次运行失败，而不是给出一个缺乏前提的数字。不传 `--base-data` 时对比功能直接关闭。

调度器命令会先执行精度测试、再运行 benchmark，并非仅测性能。每个算子的输出目录中，可读的 benchmark 输出保存在 `performance_stdout.log`；各 benchmark 脚本的结构化结果分别保存在 `performance_records_0.log`、`performance_records_1.log` 等文件中。这些结果使用与 FlagGems 兼容的 `[INFO] {` 单行 JSON 格式，包含输入形状、基线延迟、FlagAttention 延迟和可计算时的加速比。没有基线时，原始记录中的 `speedup` 为 `null`；与 FlagGems 兼容的 `--record json` 报告会略过这些形状，使 FlagGems 将该 dtype 标为 `Unknown`。

为兼容旧版调度器，未设置记录路径时直接运行脚本仍会向标准输出打印机器记录；如需干净的直接运行输出，可设置 `FLAG_ATTN_BENCHMARK_LOG_PATH=/path/to/records.log`。Benchmark 同时保留原有的表格、图表和基于矩阵乘运算量计算的吞吐率；`flash_decoding_benchmark.py` 的图表纵轴现在正确标为毫秒。历史 v0.2 图表仍保存在 [`assets/v0.2`](./assets/v0.2)；评估当前代码、Triton 和硬件时应重新运行当前 benchmark。

## 仓库结构

包结构分为三层：以 NVIDIA 为基准编写的通用 Triton 实现、负责识别厂商并按名字解析算子的运行时，以及按算子逐个覆盖的厂商包。

```text
FlagAttention/
├── src/flag_attn/
│   ├── __init__.py                 # 公共 API 门面；按算子懒加载派发
│   │
│   │   # 第 1 层 —— 通用实现，纯 Triton，以 NVIDIA 为基准
│   ├── flash.py, piecewise.py, split_kv.py, total.py, paged.py, dropout.py
│   ├── utils.py                    # has_triton_tle()、is_sm90_device()、tensor_cache
│   ├── FLA/                        # GLA、Gated Delta Rule、KDA、log-linear、MoBA、
│   │                               #   Parallax、Wall Attention、AttnRes 及共享 chunk 原语
│   ├── gdn2/                       # GDN2 门面 + native/ Triton kernel
│   ├── parallel_nsa/               # NSA 选择与压缩
│   ├── minimax_sparse_attention/   # MiniMax M3 索引与稀疏注意力
│   ├── infllmv2/                   # InfLLM-V2 prefill、反向、decode
│   ├── hpc_ops_attention/          # Hy3 FP8 prefill + 六种 decode 变体
│   ├── sage_attention/             # INT8 Q/K 量化与注意力
│   ├── diffkv_attention/           # Paged DiffKV，Triton 与 TLE 路径
│   ├── inkling_fa4/                # FA4 相对注意力，Triton 与 TLE 路径
│   ├── forgetting_attention/       # Forgetting Attention 及 naive 参考实现
│   │
│   │   # 第 2 层 —— 厂商分发
│   ├── runtime/
│   │   ├── accelerator.py          # 可选后端的 torch 设备抽象
│   │   ├── configs_loader.py       # 读取 backend/_<vendor>/tune_configs.yaml
│   │   └── backend/
│   │       ├── device_finder.py    # 厂商识别与设备类型映射
│   │       ├── op_registry.py      # resolve_operator()：厂商覆盖或通用实现
│   │       │
│   │       │   # 第 3 层 —— 厂商实现，可以使用 TLE
│   │       ├── _nvidia/            # 无覆盖；仅提供 tune_configs.yaml
│   │       ├── _metax/  _hygon/  _mthreads/  _enflame/
│   │       └── _ascend/  _thead/  _iluvatar/
│   │
│   └── testing/                    # PyTorch 参考实现与测试辅助
├── tests/flag_attn/                # 精度与派发测试，每个算子一个文件
├── benchmark/                      # 性能测试入口
├── baseline/                       # 用于对比的 NVIDIA 参考实现
├── examples/                       # 小型使用示例
├── conf/operators.yaml             # 算子阶段/测试/benchmark 清单
├── tools/run_tests.py              # 多设备测试调度器
├── .github/configs/backends/       # 各机型 CI 配置
└── packaging/                      # Debian/RPM 打包
```

## 当前限制

- 所有计算 kernel 都需要支持的加速器；CPU 仅用于元数据和参考实现。
- 不同后端的算子与 dtype 覆盖范围不同，不能仅根据后端识别成功就推断算子可用。
- FlashAttention 和 Piecewise Attention 的 head dimension 必须属于 `{16, 32, 64, 128}`；paged 和特化算子有各自约束。
- 当 FlashAttention 自动派发到 Split-KV 时不能使用 dropout。
- 除 FlashAttention 和 Piecewise Attention 外，其余所有算子族当前均为 alpha，包括上文列出的全部稀疏、递归、量化与融合算子。
- 一些 TLE 路径要求提供 TLE 的 Triton build，shape 约束也比原生 fallback 更严格。Hy3 额外要求 PyTorch 2.7.0，NSA 的 sliding-window 组合需要外部 `flash-attn` 包。
- 许多算子只提供前向或仅支持推理；具体的梯度支持情况见算子概览表格。

## 更多

FlagAttention 属于 FlagOS/FlagOpen 开源生态。更多项目请访问 [FlagOpen](https://flagopen.baai.ac.cn/)。

[<img src="./assets/logo/baai-flagopen.jpeg" alt="BAAI FlagOpen">](https://flagopen.baai.ac.cn/)
