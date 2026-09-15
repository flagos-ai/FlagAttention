# FlagAttention

<p align="center">
    <img src="./assets/logo/horizontal-blue.png" width = "400" alt="flag-attention" >
</p>

[中文版](./README_cn.md)

FlagAttention is a project for memory-efficient attention operators implemented in the [Triton language](https://github.com/triton-lang/triton). Motivated by the need for non-standard attention operators in language modeling, it starts as an extension of multi-head attention.

It saves memory footprint and traffic like [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention v2](https://tridao.me/publications/flash2/flash2.pdf). Implemented in the Triton language, it is easier to understand and modify. The original implementation of FlashAttention in CUDA([flash-attention](https://github.com/Dao-AILab/flash-attention)) provides a good example of how to design an algorithm that takes different levels of memory into account. By tiling and re-computation, FlashAttention avoids materializing the attention scores, whose capacity is proportional to the square of the sequence length. However, custom transformation to the attention scores is not possible when using FlashAttention, unless it is supported by FlashAttention out-of-the-box.
While extending FlashAttention requires proficiency in CUDA programming, FlagAttention implemented in the Triton language is easier to modify.

For standard attention workloads, modern alternatives such as [FlashAttention-3](https://arxiv.org/abs/2407.08608), PyTorch `scaled_dot_product_attention`, and [FlexAttention](https://arxiv.org/abs/2412.05496) are also worth considering. FlagAttention remains useful when the attention score computation, KV-cache layout, or inference path needs project-specific customization.

FlagAttention now offers several operators.

1. **flash_attention**: FlashAttention v2-style attention implemented in the Triton language, with support for MQA/GQA, dropout, and auxiliary outputs.
2. **piecewise_attention**: An extension used for NLPE(Non-Linearized position embedding) in both training and inference of the [Aquila-2-34B](https://github.com/FlagAI-Open/Aquila2) model.
3. **flash_attention_split_kv**: A split-KV flash decoding operator for long KV sequences and grouped-query layouts.
4. **paged_attention**: A paged KV-cache attention operator for inference.
5. **fused_attnres**: A forward-only TLE kernel for Kimi K3 Attention Residuals aggregation, with optional output RMSNorm fusion.

When further customization is required, FlagAttention serves as an example.

## Changelog

### v0.1

Add piecewise_attention & flash_attention.

### v0.2

Optimization of operators.
1. applying mask only when needed.
2. use a separate kernel to compute the gradien of q to avoid atomic RMW to global memory.


## Requirements

FlagAttention requires Python 3.10+, PyTorch, and a Triton-compatible runtime. The package does not install Triton by default, so environments that already provide Triton APIs, for example through Triton itself or a compatible fork, can install FlagAttention without pulling another Triton package.

For a standard Triton installation, use either the optional extra:

```sh
pip install -e ".[triton]"
```

or install Triton directly:

```sh
pip install triton
```

Triton nightly builds are only recommended when you need a specific unreleased Triton feature or bug fix.

FlagAttention requires a CUDA-capable GPU supported by PyTorch and Triton. It has been tested on Ampere Nvidia GPUs(e.g. A100, RTX-3090, ...). Other GPUs may work but have not been tested yet. When installing PyTorch, choose a build that matches your driver and CUDA runtime; PyTorch pip wheels commonly bundle the CUDA runtime and do not require a full local CUDA Toolkit installation for normal use.

## Installation

FlagAttention can be installed in either way below.

1. Editable Installation. Changes to the code in the local source tree are effective without re-installation.
2. Build a distribution and then install. Only the package is installed.

### Editable Installation

Editable installation with pip.

```sh
git clone https://github.com/FlagOpen/FlagAttention && cd FlagAttention
pip install -e .
```

If Triton is not already available in the environment, install the Triton extra:

```sh
pip install -e ".[triton]"
```

### Build a Distribution & Install

Following modern Python packaging convention(PEP-517), FlagAttention is configured by [`pyproject.toml`](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/), and no `setup.py` is provided. To build a distribution, either a source distribution or a binary distribution, python package `build` is recommended.

First, install `build` package via pip.

```sh
pip install build
```

Then build the package.

```sh
git clone https://github.com/FlagOpen/FlagAttention && cd FlagAttention
# to build in `no-isolation` mode requires installing build requirements manually
pip install -U setuptools setuptools-scm
python -m build --no-isolation
```

The built package is in `dist/` for installation.

```sh
pip install dist/flag_attn-xxx.whl
```

## Usage

FlagAttention provides customized operators for attention. When an operator is equivalent to a torch function, it can be used as a drop-in replacement.

## Run the Tests

A recent version of `pytest`(>=7.1.0) is required to run the tests in `tests/`. Operators in `FlagAttention` are tested against [reference implementations](src/flag_attn/testing) in Pytorch provided by `flag_attn.testing`, both for the forward and backward operators. For operators with support for inputs of `float16` or `bfloat16`, three different implementations are included for numerical accuracy testing.

1. **Reference Implementation in Pytorch**: This implementation upcasts the inputs to `float32` and performs the computations in `float32` all the way through before casting the outputs to `float16` or `bfloat16`.
2. **Triton Implementation**: The Triton implementation uses `float16` or `bfloat16` for MMA(matrix multiplication accumulation) inputs and `float32` for MMA outputs and other computations.
3. **Pytorch Implementation**: This implementation mirrors the computations in the reference implementation, except that the precision is the same as the Triton implementation.

The tests for numerical accuracy enforce that the maximum difference between the Triton implementation and reference implementation is not greater than twice the maximanum difference between the Pytorch implementation and reference implementation.

```sh
pytest .
```

## Run the Benchmark

Benchmarks are included to quantify the achieved `TFLOP/s`, which serves as a metric of speed operators. The calculation of FLOPs for an operator considers only the matmul operation. The resulting FLOPs are then divided by the median runtime to determine the achieved FLOPs/s.

The benchmarking process involves comparing the Triton implementations with counterparts in Pytorch. When the input size is large, resulting in memory exhaustion in the Pytorch implementation, the FLOP/s is considered zero.

```sh
cd benchmark/
python flash_benchmark.py
python piecewise_benchmark.py
```

The optional AttnRes benchmark uses the same pytest test module as its
correctness coverage and compares preallocated forward kernels against FLA.
Python wrappers, tensor allocation, and residual pointer-table construction are
excluded. The benchmark uses public FLA commit `5aea42b7740f9968f6418c6c60b78ea785ce6140`
and FlagTree commit `aaa420f9366440e18bd83a0391c2d2dbcc5e0b81`.

```sh
FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1 pytest tests/flag_attn/test_attnres.py -k benchmark -s
```

## Operators

### fused_attnres

`fused_attnres` scores RMS-normalized residual sources with a learned query,
applies a softmax over the source/depth axis, and mixes the original residuals.
It uses an online softmax so each residual element is read once and can fuse the
output RMSNorm used by the following sublayer. The current implementation is a
forward-only inference kernel and requires a Triton build with TLE support.

The implementation lives in the [src/flag_attn/FLA/attnres/](src/flag_attn/FLA/attnres/)
package: `fused.py` contains the TLE kernel and forward API, and `__init__.py` exports the public function.
The public `flag_attn.fused_attnres` entry point is preserved through a lazy export.

```python
from flag_attn import fused_attnres

fused_attnres(
    query,
    residuals,
    rms_weight,
    output_rms_weight=None,
    rms_eps=1e-6,
    scale=1.0,
    return_weights=False,
)
```

### flash_attention

The implementation of FlashAttention in the Triton language. The interface is.

```python
flash_attention(q, k, v, causal=False, sm_scale=None, return_log_normalizer=False, return_total_attention=False)
```

In addition to the attention outputs, it can return some extra outputs dependes on `return_log_normalizer` and `return_total_attention`.

1. log_normalizer: shape (batch_size, num_heads, seqlen_q). The log normalizer of the softmax inside attention operation.
2. total_attention: shape (batch_size, num_heads, seqlen_k). The sum of attention weights along q's sequence axis.

### piecewise_attention

The first extension to FlashAttention is [piecewise_attention](src/flag_attn/piecewise.py). This operator enhances FlashAttention by using two `q`'s and two `k`'s to calculate the attention scores(S) before applying softmax to obtain the attention weights(P).

The rationale behind this design is rooted in the observations that a transformer with rotary position embedding struggles with predicting sequences longer than the maximum sequence length it is trained on. Pairs of `(q, k)` yield unexpectedly high attention scores when the distance exceeds the maximum sequence length in the training set.

To address this issue, BAAI proposes NLPE(Non-Linearized Position Embedding), which applies two different position embeddings to `q` and `k` based on whether the distance between `q` and `k` exceeds a pre-defined threshold, producing `q1, q2` and `k1, k2`. Then the attention score is computed as the dot product of `q1, k1` or `q2, k2` depending on the distance between `q` and `k`.



The interface is shown below.

![piecewise_attention_interface](./assets/piecewise_attention_interface.png)

```python
piecewise_attention(q1, k1, q2, k2, v, dist_threshold, causal=False, sm_scale=None)
```

It splices two attention scores(S) in the forward computation and splits the gradient of S in the backward computation.

![piecewise attention](assets/piecewise_attention.png)

#### Usage

```python
# piecewise_attention
import torch
from flag_attn import piecewise_attention

B, H, T, D = 2, 16, 8192, 128
dist_threshold = T // 2

q1 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
q2 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
k1 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
k2 = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
v = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
o = piecewise_attention(q1, k1, q2, k2, v, dist_threshold, causal=True)
print(o)

go = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
gq1, gk1, gq2, gk2, gv = torch.autograd.grad(
    o, (q1, k1, q2, k2, v), go
)
print(gq1)
```

```python
# flash_attention
import torch
from flag_attn import flash_attention

B, H, T, D = 2, 16, 8192, 128

q = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
k = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
v = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0").requires_grad_()
o = flash_attention(q, k, v, causal=True)
print(o)

go = torch.randn((B, H, T, D), dtype=torch.float16, device="cuda:0")
gq, gk, gv = torch.autograd.grad(
    o, (q, k, v), go
)
print(gq)
```

#### Performance

Benchmark is performed under such conditions.

1. seqlen in `[512, 1k, 2k, 4k, 16k, 32k]`;
2. batch size: `32k / seqlen`;
3. headdim in`[64, 128]`；
4. num_heads: `2048 / headdim`.

##### flash_attention

The performance of flash_attention with causal masking is shown below.

![headdim64](./assets/v0.2/flash_attention_d64.png)

![headdim128](./assets/v0.2/flash_attention.png)

The forward operator runs as fast as, and in some cases, faster than FlashAttention(CUDA), but the backward operator is generally slower than FlashAttention. We first follow the paper and update the gradient of Q with atomic addition in the backward operator, which runs extremely slowly. Then we split the backward operator into two kernels, one to compute the gradient of k and v, the other to compute the gradient of q. This alternation avoids atomic additions but introduces more re-computation. Although this strategy yields a 4x to 5x speedup in the backward operator, it is still slower than FlashAttention(CUDA).

The same split-kernel trick is also applied to `piecewise_attention` for efficiency.

##### piecewise_attention

The performance of piecewise_attention has improved compared to that in v0.1. In the case where the head dim is 128 and causal masking is applied, the forward and backward operator is faster than that in v0.1 by 36% and 9%, respectively.

![piecewise_attention](./assets/v0.2/piecewise_attention.png)

#### Features

- support for [Nvidia](https://www.nvidia.com/) Ampere GPU(Tested on RTX-3090 and A100)；
- support for [Iluvatar CoreX](https://www.iluvatar.com/) GPU(Tested on Iluvatar CoreX MR-V100)；
- datatype support, `float16` and `bfloat16` for Ampere Nvidia GPUs;
- support causal and non-causal modes;
- support forward & backward modes;
- the sequence length of k/v can be different from that of q;
- support computation of total attention of each `k` gets from all `q`'s;
- supports returning accumulative attention of each keys.
- supports [MQA](https://arxiv.org/abs/1911.02150) and [GQA](https://arxiv.org/pdf/2305.13245).
- supports dropout of attention weights.

#### Limitations

- `headdim` should be in `[16, 32, 64, 128]`.

## TODOs

1. Test on other GPUs;
2. Test on more versions of triton；
3. Improve performance of attention operators(especially for the backward op);
4. Support other extensions to flash attention.

## More

For more about the open source system for large models from BAAI, please with [BAAI/FlagOpen](https://flagopen.baai.ac.cn/).
[<img src="./assets/logo/baai-flagopen.jpeg">](https://flagopen.baai.ac.cn/)
