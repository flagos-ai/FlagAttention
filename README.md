# FlagAttention

<p align="center">
    <img src="./assets/logo/horizontal-blue.png" width="400" alt="FlagAttention">
</p>

[中文版](./README_cn.md)

FlagAttention is a collection of memory-efficient attention operators implemented with [Triton](https://github.com/triton-lang/triton). It targets model training and inference workloads that need custom attention-score transformations, paged or sparse KV-cache layouts, or recurrent linear-attention kernels.

Like [FlashAttention](https://arxiv.org/abs/2205.14135), the dense kernels tile the computation and recompute intermediates instead of materializing the full attention matrix. The repository also contains decoding, block-sparse, quantized, and recurrent operators that do not fit the standard scaled-dot-product-attention interface.

> [!IMPORTANT]
> The repository contains operators at different maturity levels. Only FlashAttention and Piecewise Attention are currently marked `stable`; the remaining operator families are `alpha`. [`conf/operators.yaml`](./conf/operators.yaml) is the source of truth for stages, tests, and benchmark entry points.

## Operator overview

| Operator family | Public import | Main layout/use | Gradient support | Stage |
| --- | --- | --- | --- | --- |
| FlashAttention | `flag_attn.flash_attention` | Dense attention; `q: [B,Hq,M,D]`, `k/v: [B,Hkv,N,D]`; MQA/GQA, dropout, auxiliary outputs | Forward + backward | Stable |
| Piecewise Attention | `flag_attn.piecewise_attention` | Dense attention with two Q/K pairs selected by token distance | Forward + backward | Stable |
| Split-KV FlashAttention | `flag_attn.flash_attention_split_kv` | Long-KV decoding and low-query-parallelism workloads | Forward | Alpha |
| Paged Attention | `flag_attn.paged_attention` | Single-token queries over a paged KV cache | Forward | Alpha |
| MiniMax M3 sparse attention | Six `flag_attn.minimax_m3_*` functions | 128-token page scoring, top-k selection, sparse prefill and decode | Inference | Alpha |
| Chunk GLA | `flag_attn.chunk_gla` | Recurrent gated linear attention; `[B,T,H,D]`, fixed or packed variable lengths | Forward + backward | Alpha |
| Chunk Gated Delta Rule | `flag_attn.chunk_gated_delta_rule` | Chunked delta-rule recurrence; head-first or sequence-first | Forward | Alpha |
| GDN2 | `flag_attn.chunk_gdn2` | Chunked GDN2 prefill with native Triton/TLE and vendor paths | Forward/inference | Alpha |
| Kimi Delta Attention | `flag_attn.chunk_kda` | Specialized chunked KDA inference; constraints depend on the active backend | Inference | Alpha |
| SageAttention | `flag_attn.sage_attention.forward`, `per_block_int8` | Per-block INT8 Q/K quantization and attention; HND or NHD layout | Forward | Alpha |
| Parallel NSA | `flag_attn.parallel_nsa.parallel_nsa`, `parallel_nsa_compression` | Native sparse attention and compression; fixed or packed variable lengths | Forward + backward | Alpha |

The top-level package exports the dense, paged, recurrent, KDA/GDN2, and MiniMax APIs. SageAttention and Parallel NSA are submodule APIs. Vendor-specific development interfaces are available under `flag_attn.runtime.backend._<vendor>` but are not treated as stable public APIs.

## Requirements and backends

- Python 3.10 or newer.
- A PyTorch build for the target accelerator.
- Triton 2.2 or newer, or a vendor runtime that provides compatible Triton APIs.
- A supported accelerator for kernel execution. CPU-only hosts can inspect package metadata, but cannot run the attention kernels.

PyTorch and accelerator runtimes are intentionally not hard dependencies in [`pyproject.toml`](./pyproject.toml), because the correct packages depend on the device and driver stack. Install the appropriate PyTorch build before FlagAttention.

The runtime recognizes `nvidia`, `amd`, `hygon`, `iluvatar`, `metax`, `enflame`, `ascend`, `cambricon`, `mthreads`, `intel`, and `cpu` for device metadata and explicit selection. Recognition does not mean that every operator is implemented or tested on every backend; consult [`conf/operators.yaml`](./conf/operators.yaml), the backend packages in [`src/flag_attn/runtime/backend`](./src/flag_attn/runtime/backend), and the relevant tests for operator-level coverage.

Backend routing is also operator-specific. The top-level GDN2/KDA APIs select specialized Enflame, MetaX, and MThreads implementations; MiniMax M3 selects its MetaX implementation. Other vendor extensions may require their backend submodule. Do not assume that every top-level API dispatches to every detected vendor.

Device metadata follows the FlagGems/FlagGems-vLLM convention:

```python
import flag_attn

print(flag_attn.vendor_name)  # for example: "nvidia", "metax", "enflame"
print(flag_attn.vendor)       # alias for vendor_name
print(flag_attn.device)       # for example: "cuda", "gcu", "npu"
print(flag_attn.backend_info) # structured DeviceDetector object
```

Detection normally uses the available PyTorch device and Triton target. It can be overridden before importing the package:

```sh
FLAG_ATTN_BACKEND=metax python your_program.py
# FLAG_ATTN_VENDOR is an alias; FLAG_ATTN_BACKEND takes precedence.
```

## Installation

Clone the repository and first install the PyTorch/runtime build appropriate for your accelerator. For a standard upstream Triton development environment:

```sh
git clone https://github.com/flagos-ai/FlagAttention.git
cd FlagAttention
pip install -e ".[triton,test]"
```

If the environment already supplies Triton or a compatible vendor fork, do not install the `triton` extra:

```sh
pip install -e ".[test]"
```

The current package initialization loads YAML tuning configuration and PyTorch reference helpers. Consequently, every installation currently needs PyYAML and pytest; the `test` extra installs both for editable installs. Triton nightly builds are only recommended when an operator explicitly needs an unreleased feature such as a matching TLE version.

To build a wheel or source distribution:

```sh
pip install -U build setuptools setuptools-scm
python -m build --no-isolation
pip install PyYAML pytest
pip install dist/flag_attn-*.whl
```

There is no `setup.py`; packaging uses PEP 517 and setuptools-scm. Debian/RPM runtime notes are in [`packaging/INSTALL.md`](./packaging/INSTALL.md). The current distribution-package metadata and install notes may not supply both PyYAML and pytest, so ensure they are available in the Python environment used to import `flag_attn`.

## Quick start

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

The complete interface is:

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

`Hq` must be divisible by `Hkv`, and `D` must be one of `16`, `32`, `64`, or `128`. Rectangular causal attention is bottom-right aligned. The implementation automatically selects the regular or split-KV forward path according to GPU occupancy.

When any auxiliary-return flag is enabled, the function always returns five values; disabled fields are `None`:

```python
out, lse, total, seed, offset = flash_attention(
    q, k, v,
    return_log_normalizer=True,
    return_total_attention=True,
)
```

- `lse`: `[B, Hq, M]`, the row log-normalizer.
- `total`: `[B, Hq, N]`, attention probabilities summed over the query axis.
- `seed` and `offset`: Philox state when dropout is active and requested.

Dropout is not supported when the occupancy heuristic selects the split-KV path. Use a non-split shape or `dropout_p=0` for those workloads.

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

For query row `m` and key column `n`, the operator uses `q2 @ k2` when the signed, bottom-right-aligned offset `N - M + m - n >= dist_threshold`; otherwise it uses `q1 @ k1`. It then applies the same tiled online softmax strategy as FlashAttention and splits `dScore` across both Q/K pairs in backward. `q1/q2` use `[B,H,M,D]`, `k1/k2/v` use `[B,H,N,D]`, all inputs have the same head count (no GQA), and `D` is one of `16`, `32`, `64`, or `128`.

Piecewise Attention was introduced for NLPE (Non-Linearized Position Embedding), used by [Aquila-2-34B](https://github.com/FlagAI-Open/Aquila2) to switch positional representations beyond a distance threshold.

Additional CUDA-oriented scripts are available in [`examples`](./examples), but some are historical. In particular, [`examples/flash_attention_with_aux_outputs.py`](./examples/flash_attention_with_aux_outputs.py) still unpacks three values although the current auxiliary-output API returns five; use the five-value example above until that script is updated. The tests are the authoritative usage examples. PyTorch reference implementations used for validation are exposed as `flag_attn.testing`.

## Inference and sparse APIs

### Split-KV and paged attention

```python
from flag_attn import flash_attention_split_kv, paged_attention
```

`flash_attention_split_kv` is an explicit forward-only API for long-KV workloads. It accepts `q: [B,Hq,M,D]` and `k/v: [B,Hkv,N,D]`, requires `Hq % Hkv == 0`, and supports `D` in `{16,32,64,128}`. When multiple splits are selected, each computes a partial output and log-normalizer; a second kernel combines them with a global logsumexp. A single split returns its output directly.

`paged_attention` accepts:

```text
query:        [num_sequences, num_query_heads, head_size]
key_cache:    [num_blocks, num_kv_heads, block_size, head_size]
value_cache:  [num_blocks, num_kv_heads, block_size, head_size]
context_lens: [num_sequences]
block_tables: [num_sequences, max_blocks_per_sequence]
```

It selects a one-pass or partitioned/reduced implementation automatically; `num_splits` can override that choice. K and V caches must have the same shape and strides. Supported head sizes are `{16,32,64,128,256,512}`; when `num_query_heads > num_kv_heads`, the cache block size must be at least 16 tokens. Explicit `num_splits > 1` also requires the resulting partition size to be at least the cache block size and divisible by it. See [`examples/paged_example.py`](./examples/paged_example.py).

Call `paged_attention(query, key_cache, value_cache, context_lens, block_tables, attn_scale, max_context_len, num_splits=0)`; it returns an output with the same shape as `query`. `attn_scale` is the softmax scale (typically `head_size**-0.5`), and `max_context_len` is the largest context length represented by the batch.

### MiniMax M3 sparse attention

The M3 path uses a vLLM-compatible paged cache with a fixed sparse block size of 128 tokens:

```text
Prefill: minimax_m3_index_score
      -> minimax_m3_index_topk
      -> minimax_m3_sparse_attn

Decode: minimax_m3_index_decode (score + top-k)
     -> minimax_m3_sparse_attn_decode

Score-only decode API: minimax_m3_index_decode_score
```

The sparse attention kernels support GQA and BF16 KV caches, with FP8 cache scaling on supported hardware. These functions are inference-only and use caller-provided paged caches, sequence metadata, and block tables. Sparse prefill/decode require an `output` buffer and write into it, returning `None`; index-scoring/top-k functions return tensors, and some accept optional reusable output buffers. See the function signatures in [`src/flag_attn/minimax_sparse_attention`](./src/flag_attn/minimax_sparse_attention) and the end-to-end tests in [`tests/flag_attn/test_minimax_sparse_attention.py`](./tests/flag_attn/test_minimax_sparse_attention.py).

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

Both `HND` (`[B,H,N,D]`) and `NHD` (`[B,N,H,D]`) layouts are supported. Q is quantized in 128-token blocks and K in 64-token blocks by default. The forward kernel accepts boolean or additive attention masks.

`forward` always returns `(out, lse)`; when `return_lse=False`, `lse` is an empty CPU tensor. The public `flag_attn.sage_attention` submodule automatically selects the Ascend implementation on an available NPU; other vendor-specific SageAttention implementations use their backend development modules.

## Recurrent and linear attention APIs

GLA, GDN2, and KDA use sequence-first `[B,T,H,D]`. Gated Delta Rule defaults to head-first `[B,H,T,D]` with `head_first=True`, and also accepts sequence-first input. Most recurrent APIs return `(output, final_state)` and support packed variable-length sequences through `cu_seqlens`.

- [`chunk_gla`](./src/flag_attn/FLA/gated_linear_attention/chunk_gla.py) implements chunked gated linear attention with autograd and optional initial/final recurrent states.
- [`chunk_gated_delta_rule`](./src/flag_attn/FLA/gated_delta_rule/api.py) provides a forward-only delta-rule implementation. `head_first=True` uses `[B,H,T,D]`; `BT` is currently fixed at 64.
- [`chunk_gdn2`](./src/flag_attn/gdn2/chunk.py) dispatches between native Triton, TLE, and vendor-specific GDN2 forward paths.
- [`chunk_kda`](./src/flag_attn/FLA/chunk_kda.py) is an inference API. The generic CUDA path requires Triton TLE 3.6 or newer, inference mode, BF16 inputs, `K=V=128`, chunk size 16, V-first state, and gate parameters; vendor implementations may differ.
- [`parallel_nsa`](./src/flag_attn/parallel_nsa) contains Native Sparse Attention selection and compression operators with fixed-length and packed-variable-length support. `parallel_nsa` needs either precomputed `block_indices` or `g_cmp`, and `Hq/Hkv` must be a power of two and at least 16. Sliding-window composition additionally needs `g_swa` and the external `flash-attn` package. For fixed-length input, the compression API consumes full-length Q and K/V with time dimension `ceil(T / block_size)`, then returns `(output, lse)`; packed input stores the per-sequence compressed blocks consecutively. The registered NSA tests import the Enflame-specific modules, so they do not establish coverage for the public `flag_attn.parallel_nsa` submodule on other backends.

These alpha APIs are optimized for specific model layouts. Read their docstrings and tests before integrating them; arguments and backend constraints may change.

## Tests

Install the development dependencies and list the operator inventory without probing a GPU:

```sh
pip install -e ".[triton,test]"
python tools/run_tests.py --list-ops --stages all
```

Run the full pytest suite in a compatible CUDA environment:

```sh
pytest tests
```

The dense operators are compared against FP32 PyTorch references. Their accuracy checks require the Triton error relative to the FP32 reference to be no greater than roughly twice the same-precision PyTorch error, with a small absolute allowance. Much of the generic suite currently assumes CUDA; on other accelerators, select the relevant vendor tests or inventory entries instead of expecting every CUDA test to skip.

The inventory-driven scheduler runs `stable` operators by default and distributes work across selected CUDA GPUs:

```sh
python tools/run_tests.py --stages stable --gpus 0 --skip-benchmarks
python tools/run_tests.py --stages all --gpus all --dump-output
```

Use the FlagGems-compatible pytest recorder for case-level JSON output:

```sh
pytest -m "sage_attention" --record json --output accuracy_sage_attention.json -vs
```

If `--output` is omitted, the report is written to `accuracy_result.json`. Existing reports are merged by pytest node ID.

## Benchmarks

Available benchmark entry points are registered alongside their operators in [`conf/operators.yaml`](./conf/operators.yaml); not every inventory entry has a benchmark. Run them through the scheduler or directly, depending on the file:

```sh
python tools/run_tests.py --stages all --gpus 0

python benchmark/flash_benchmark.py
python benchmark/piecewise_benchmark.py
python benchmark/flash_decoding_benchmark.py

cd benchmark
pytest -m "sage_attention" --record json --output benchmark_sage_attention.json -vs
```

The scheduler runs accuracy tests before benchmarks. Each operator's `performance_stdout.log` is retained under its output directory. Single-line JSON records beginning with `[INFO] {` follow the FlagGems benchmark log format and contain shapes, baseline latency, FlagAttention latency, and speedup when a baseline is available. Benchmarks without a baseline record `speedup` as `null`. Existing tables, plots, and matmul-based throughput remain available; the `flash_decoding_benchmark.py` plot axis now correctly says milliseconds. Historical v0.2 plots remain under [`assets/v0.2`](./assets/v0.2); rerun the current benchmark for conclusions about current code, Triton, and hardware.

## Repository layout

```text
FlagAttention/
├── src/flag_attn/
│   ├── flash.py, piecewise.py, split_kv.py, paged.py
│   ├── minimax_sparse_attention/   # MiniMax M3 index + sparse attention
│   ├── sage_attention/             # INT8 Q/K quantization + attention
│   ├── parallel_nsa/               # NSA selection and compression
│   ├── FLA/                        # GLA, Gated Delta Rule, KDA helpers
│   ├── gdn2/                       # Generic GDN2 implementation
│   ├── runtime/backend/            # Device detection and vendor backends
│   └── testing/                    # PyTorch reference implementations
├── tests/                          # Accuracy and dispatch tests
├── benchmark/                      # Performance entry points
├── examples/                       # Small usage programs
├── conf/operators.yaml             # Operator stage/test/benchmark inventory
├── tools/run_tests.py              # Multi-device test scheduler
└── packaging/                      # Debian/RPM packaging
```

## Current limitations

- All compute kernels require a supported accelerator; CPU is metadata/reference-only.
- Operator and dtype coverage varies by backend. Do not infer support only from successful backend detection.
- FlashAttention and Piecewise Attention require head dimensions in `{16, 32, 64, 128}`; paged and specialized operators have their own constraints.
- FlashAttention dropout cannot run when the automatic dispatcher selects split-KV.
- Split-KV, paged, MiniMax M3, recurrent/linear, SageAttention, and NSA APIs are currently alpha.
- Several TLE paths require a specific recent Triton build and stricter shapes than their native fallbacks.

## More

FlagAttention is part of the FlagOS/FlagOpen open-source ecosystem. See [FlagOpen](https://flagopen.baai.ac.cn/) for more projects.

[<img src="./assets/logo/baai-flagopen.jpeg" alt="BAAI FlagOpen">](https://flagopen.baai.ac.cn/)
