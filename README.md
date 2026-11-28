# FlagAttention

<p align="center">
    <img src="./assets/logo/horizontal-blue.png" width="400" alt="FlagAttention">
</p>

[中文版](./README_cn.md)

FlagAttention is a collection of memory-efficient attention operators implemented with [Triton](https://github.com/triton-lang/triton). It targets model training and inference workloads that need custom attention-score transformations, paged or sparse KV-cache layouts, or recurrent linear-attention kernels.

Like [FlashAttention](https://arxiv.org/abs/2205.14135), the dense kernels tile the computation and recompute intermediates instead of materializing the full attention matrix. The repository also contains decoding, block-sparse, quantized, and recurrent operators that do not fit the standard scaled-dot-product-attention interface.

> [!IMPORTANT]
> The repository contains operators at different maturity levels. Only FlashAttention and Piecewise Attention are currently marked `stable`; the remaining operator families are `alpha`, and their arguments and backend constraints may still change. [`conf/operators.yaml`](./conf/operators.yaml) is the source of truth for stages, tests, and benchmark entry points.

## Operator overview

Twenty-six inventory entries are registered in [`conf/operators.yaml`](./conf/operators.yaml), covering the operator families below. Only FlashAttention and Piecewise Attention are `stable`; everything else is `alpha`.

### Dense attention

| Operator family | Public import | Main layout/use | Gradient support | Stage |
| --- | --- | --- | --- | --- |
| FlashAttention | `flag_attn.flash_attention` | Dense attention; `q: [B,Hq,M,D]`, `k/v: [B,Hkv,N,D]`; MQA/GQA, dropout, auxiliary outputs | Forward + backward | Stable |
| Piecewise Attention | `flag_attn.piecewise_attention` | Dense attention with two Q/K pairs selected by token distance | Forward + backward | Stable |
| Split-KV FlashAttention | `flag_attn.flash_attention_split_kv` | Long-KV decoding and low-query-parallelism workloads | Forward | Alpha |
| Paged Attention | `flag_attn.paged_attention` | Single-token queries over a paged KV cache | Forward | Alpha |

### Sparse and block-sparse attention

| Operator family | Public import | Main layout/use | Gradient support | Stage |
| --- | --- | --- | --- | --- |
| MiniMax M3 sparse attention | Six `flag_attn.minimax_m3_*` functions | 128-token page scoring, top-k selection, sparse prefill and decode | Inference | Alpha |
| Parallel NSA | `flag_attn.parallel_nsa`, `flag_attn.parallel_nsa_compression` | Native sparse attention and compression; fixed or packed variable lengths | Forward + backward | Alpha |
| InfLLM-V2 | `flag_attn.infllmv2_attention`, `infllmv2_decode` | Continuous-packed causal prefill with block selection, plus variable-length decode | Forward + backward (prefill) | Alpha |
| Parallel MoBA | `flag_attn.parallel_moba` | Variable-length Mixture-of-Block-Attention with causal chunk routing | Forward + backward | Alpha |
| Hy3 attention | `flag_attn.hy3_attention` | FP8 paged block-sparse prefill and six BF16/FP8 decode variants | Inference | Alpha |
| Inkling FA4 relative attention | `flag_attn.inkling_fa4_rel_attention` | Paged varlen attention with relative-position logits; Triton and Hopper TLE | Inference | Alpha |

### Recurrent and linear attention

| Operator family | Public import | Main layout/use | Gradient support | Stage |
| --- | --- | --- | --- | --- |
| Chunk GLA | `flag_attn.chunk_gla` | Recurrent gated linear attention; `[B,T,H,D]`, fixed or packed variable lengths | Forward + backward | Alpha |
| Chunk Gated Delta Rule | `flag_attn.chunk_gated_delta_rule` | Chunked delta-rule recurrence; head-first or sequence-first | Forward | Alpha |
| GDN2 | `flag_attn.chunk_gdn2` | Chunked GDN2 prefill with native Triton/TLE and vendor paths | Forward/inference | Alpha |
| Kimi Delta Attention | `flag_attn.chunk_kda` | Specialized chunked KDA inference; constraints depend on the active backend | Inference | Alpha |
| Log-Linear Attention | `flag_attn.log_linear_attn`, `chunk_log_linear_attn` | Dense log-linear attention with multi-level gate scales; TLE kernels | Forward | Alpha |
| Parallax | `flag_attn.parallel_parallax` | Parameterized local linear attention; prefill, decode and KV-cache decode stages | Forward + backward (prefill) | Alpha |
| Wall Attention | `flag_attn.parallel_wall_attn` | Gated windowed attention with optional sink bias and packed variable lengths | Forward + backward | Alpha |

### Quantized and fused operators

| Operator family | Public import | Main layout/use | Gradient support | Stage |
| --- | --- | --- | --- | --- |
| SageAttention | `flag_attn.sage_attention`, `from flag_attn.sageattention import per_block_int8` | Per-block INT8 Q/K quantization and attention; HND or NHD layout | Forward | Alpha |
| DiffKV attention | `flag_attn.diffkv_attention` | Paged decode with asymmetric K/V head dimensions; Triton and TLE paths | Inference | Alpha |
| Forgetting Attention | `flag_attn.parallel_forgetting_attn` | Attention with per-token log forget gates; SM90 TLE path with Triton fallback | Forward + backward | Alpha |
| Fused AttnRes | `flag_attn.fused_attnres` | Softmax mixing of RMS-normalized residual sources, with optional fused output RMSNorm | Inference | Alpha |

Every operator named in [`conf/operators.yaml`](./conf/operators.yaml) is available as `flag_attn.<name>`. The implementation packages for SageAttention, Native Sparse Attention, DiffKV, and Forgetting Attention are `flag_attn.sageattention`, `flag_attn.nsa`, `flag_attn.diffkv`, and `flag_attn.forgetting`. Vendor-specific development interfaces are available under `flag_attn.runtime.backend._<vendor>` but are not treated as stable public APIs.

## Requirements and backends

- Python 3.10 or newer.
- A PyTorch build for the target accelerator.
- Triton 2.2 or newer, or a vendor runtime that provides compatible Triton APIs.
- A supported accelerator for kernel execution. CPU-only hosts can inspect package metadata, but cannot run the attention kernels.

PyTorch and accelerator runtimes are intentionally not hard dependencies in [`pyproject.toml`](./pyproject.toml), because the correct packages depend on the device and driver stack. Install the appropriate PyTorch build before FlagAttention.

The runtime recognizes `nvidia`, `amd`, `hygon`, `iluvatar`, `metax`, `thead`, `enflame`, `ascend`, `cambricon`, `mthreads`, `intel`, and `cpu` for device metadata and explicit selection. Recognition does not mean that every operator is implemented or tested on every backend; consult [`conf/operators.yaml`](./conf/operators.yaml), the backend packages in [`src/flag_attn/runtime/backend`](./src/flag_attn/runtime/backend), and the relevant tests for operator-level coverage.

Backend routing is operator-specific and name-based. Each vendor package declares an `_OPERATOR_EXPORTS` table; a public API resolves to the vendor implementation when that table names it, and otherwise keeps the generic Triton implementation. A declared override that fails to import or run is an error rather than a silent fallback. The generic implementations target NVIDIA, so `_nvidia` declares no overrides and supplies only tuning configuration.

Vendor packages currently ship overrides for these operator families:

| Backend | Device | Operator families with vendor implementations |
| --- | --- | --- |
| `_nvidia` | `cuda` | None; generic Triton paths plus `tune_configs.yaml` |
| `_metax` | `cuda` | MiniMax M3, SageAttention, NSA, GLA, KDA, GDN2, AttnRes |
| `_hygon` | `cuda` | MiniMax M3, SageAttention, NSA, GLA, KDA, AttnRes, and a private `mla` family |
| `_mthreads` | `musa` | MiniMax M3, SageAttention, NSA, GLA, KDA, GDN2, MoBA |
| `_enflame` | `gcu` | MiniMax M3, SageAttention, NSA, GLA, KDA, GDN2, AttnRes, Parallax |
| `_ascend` | `npu` | Gated Delta Rule, Parallax, GDN2, SageAttention |
| `_thead` | `cuda` | MiniMax M3, Wall Attention |
| `_iluvatar` | `cuda` | None declared yet |

Note that the two `stable` dense families, FlashAttention and Piecewise Attention, have no vendor overrides at all and run their generic Triton kernels everywhere. Do not assume that every top-level API dispatches to every detected vendor.

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

Both `HND` (`[B,H,N,D]`) and `NHD` (`[B,N,H,D]`) layouts are supported. Q is quantized in 128-token blocks and K in 64-token blocks by default. The forward kernel accepts boolean or additive attention masks.

`sage_attention` always returns `(out, lse)`. When `return_lse=False`, `lse` is an empty CPU tensor. `flag_attn.sage_attention` selects the implementation registered for the detected backend, including the Ascend implementation on an available NPU.

## Recurrent, linear, and remaining operator APIs

### Recurrent and linear attention

GLA, GDN2, and KDA use sequence-first `[B,T,H,D]`. Gated Delta Rule defaults to head-first `[B,H,T,D]` with `head_first=True`, and also accepts sequence-first input. Most recurrent APIs return `(output, final_state)` and support packed variable-length sequences through `cu_seqlens`.

- [`chunk_gla`](./src/flag_attn/FLA/gated_linear_attention/chunk.py) implements chunked gated linear attention with autograd and optional initial/final recurrent states.
- [`chunk_gated_delta_rule`](./src/flag_attn/FLA/gated_delta_rule/api.py) provides a forward-only delta-rule implementation. `head_first=True` uses `[B,H,T,D]`; `BT` is currently fixed at 64.
- [`chunk_gdn2`](./src/flag_attn/gdn2/__init__.py) dispatches through the vendor registry. The [generic forward](./src/flag_attn/gdn2/chunk.py) uses standard Triton. The [NVIDIA override](./src/flag_attn/runtime/backend/_nvidia/gdn2/chunk.py) uses the fused path for 16-token chunks on supported GPUs with a compatible TLE compiler.
- [`chunk_kda`](./src/flag_attn/FLA/chunk_kda.py) is an inference API. The generic CUDA path needs a Triton build providing TLE, and rejects grad-enabled calls, non-BF16 or non-contiguous inputs, `K != 128`, `V != 128`, and GVA. Chunk size defaults to 16; vendor implementations may differ.
- [`log_linear_attn`](./src/flag_attn/FLA/log_linear_attn/chunk_tle.py) is a dense, forward-only Log-Linear Attention entry built on TLE kernels. It takes `(q, k, v, g, level_scales)`, where `level_scales` supplies the per-level gate scales. `chunk_log_linear_attn` is the same function under its inventory name.
- [`parallel_parallax`](./src/flag_attn/FLA/parallax) is parameterized local linear attention with a single public entry and three stages selected by `stage=`. Prefill (`"prefill"`, the default) takes `(q, r, k, v)` with `q/r: [B,T,HQ,D]` and `k/v: [B,T,H,D]`, supports GQA, autograd, an optional `window_size`, and packed variable lengths via `cu_seqlens`. Note that `r` is not pre-scaled by `scale`. The `"decode"` and `"kvcache"` stages route to the KV-cache decode kernels.
- [`parallel_wall_attn`](./src/flag_attn/FLA/wall_attn/parallel.py) is gated windowed attention taking `(q, k, v, g)` with optional `g_scalar`, `sink_bias`, `window_size`, and `cu_seqlens`. It provides forward and backward. The public entry validates its inputs and picks a numerically checked fast path or the general implementation; validation synchronizes with the host, so graph capture always uses the general path.

### Native Sparse Attention

[`parallel_nsa`](./src/flag_attn/nsa) contains Native Sparse Attention selection and compression operators with fixed-length and packed-variable-length support, selected through `mode=`. `parallel_nsa` needs either precomputed `block_indices` or `g_cmp`, and `Hq/Hkv` must be a power of two and at least 16. Sliding-window composition additionally needs `g_swa` and the external `flash-attn` package. For fixed-length input, the compression API consumes full-length Q and K/V with time dimension `ceil(T / block_size)`, then returns `(output, lse)`; packed input stores the per-sequence compressed blocks consecutively.

### Other sparse and fused operators

- [`infllmv2_attention`](./src/flag_attn/infllmv2) runs continuous-packed causal prefill by default and variable-length decode with `stage="decode"`. Prefill takes packed `q/k/v` with `cu_seqlens_q`, `cu_seqlens_k`, and the two max lengths; each Q sequence is the suffix of its KV sequence, so KV must contain the cached prefix plus the current chunk. Block selection parameters live in `InfLLMV2Config`, which currently supports causal mode only. Gradients flow through all supplied Q/K/V. The operator performs no paged-cache or in-place cache updates.
- [`parallel_moba`](./src/flag_attn/FLA/moba/parallel.py) is causal variable-length MoBA over packed `[seqlen, heads, head_dim]` tensors, taking `cu_seqlens`, `max_seqlen`, `moba_chunk_size`, and `moba_topk`. `moba_topk` counts the current causal chunk, matching FlashMoBA. It supports MHA/GQA/MQA, head dimensions up to 256, and top-k up to 64. The inference path deliberately skips host-side validation of value-dependent metadata; the autograd path retains it.
- [`hy3_attention`](./src/flag_attn/hpc_ops_attention) exposes FP8 paged block-sparse prefill (`stage="prefill"`) and six decode implementations selected by `variant=`: `bf16_static`, `bf16_dynamic`, `fp8_qk_static`, `fp8_qk_dynamic`, `fp8_kv_static`, `fp8_kv_dynamic`. Here `qk` means per-token Q/K with per-head V quantization, and `kv` means per-token Q with per-tensor K/V quantization; `static` and `dynamic` select the scheduling policy. Decode forwards `(inputs, workspace)` unchanged. Prepare it through `flag_attn.hy3_attention_prepare(inputs, variant=...)`; `flag_attn.Hy3StaticBF16Inputs`, `flag_attn.Hy3DynamicBF16Inputs`, and `flag_attn.Hy3FP8DecodeInputs` expose the input contracts. The runtime automatically selects the available backend implementation. Portable kernels use standard Triton, including software E4M3 conversion; NVIDIA Hopper kernels live under `runtime/backend/_nvidia/hopper/hpc_ops_attention`. Set `FLAG_ATTN_FORCE_GENERIC=1` before starting Python to validate the portable implementation using the same public API and `pytest -m hy3_attention`. This family requires PyTorch 2.7.0.
- [`inkling_fa4_rel_attention`](./src/flag_attn/inkling_fa4) is inference-only paged varlen attention that adds relative-position logits, with arguments mirroring the vLLM operator of the same name. Caches use `[num_blocks, block_size, num_kv_heads, head_dim]`. `backend=` accepts `"auto"` (default), `"triton"`, or `"tle"`, and `INKLING_FA4_BACKEND` sets the same choice; auto prefers TLE and falls back to Triton when TLE is unavailable or the layout is unsupported.
- [`diffkv_attention`](./src/flag_attn/diffkv/api.py) is paged decode with asymmetric K/V head dimensions: `query: [B,Hq,Dqk]`, `key_cache: [NB,BS,Hkv,Dqk]`, `value_cache: [NB,BS,Hkv,Dv]`, returning `[B,Hq,Dv]`. `window_size=-1` means full context; `path` selects the `"2d"` or `"3d"` launch, with `num_segments` controlling split-KV on the 3D path; `backend` accepts `"auto"`, `"tle"`, or `"triton"`.
- [`parallel_forgetting_attn`](./src/flag_attn/forgetting/parallel.py) adds per-token log forget gates through `log_fgate`, with `head_first=`, `seq_start=`, `sm_scale=`, and `adaptive_threshold=`. On SM90 with compatible TLE it uses the TLE kernel; otherwise the ordinary Triton kernel handles head dimensions 16, 32, 60, 64, 100, and 128 by zero-padding without changing attention scaling or pruning.
- [`fused_attnres`](./src/flag_attn/FLA/attnres/fused.py) aggregates residual sources: each source is RMS-normalized for scoring against `query`, then a softmax over the source axis mixes the original residuals, with `output_rms_weight` optionally fusing the next sublayer's RMSNorm. It is inference-only with no autograd formula, tuned for Kimi K3 shapes, and supports up to 16 sources and hidden sizes up to 8192.

These alpha APIs are optimized for specific model layouts. Read their docstrings and tests before integrating them; arguments and backend constraints may change.

## Tests

Install the development dependencies:

```sh
pip install -e ".[triton,test]"
```

The operator inventory itself is [`conf/operators.yaml`](./conf/operators.yaml); read it directly to see which stages, tests, and benchmark entry points are registered.

Run the full pytest suite in a compatible CUDA environment:

```sh
pytest tests
```

The dense operators are compared against FP32 PyTorch references. Their accuracy checks require the Triton error relative to the FP32 reference to be no greater than roughly twice the same-precision PyTorch error, with a small absolute allowance. Much of the generic suite currently assumes CUDA; on other accelerators, select the relevant vendor tests or inventory entries instead of expecting every CUDA test to skip.

The inventory-driven scheduler runs `stable` operators by default and distributes work across selected CUDA GPUs:

```sh
python tools/run_tests.py --stages stable --gpus 0
python tools/run_tests.py --stages all --gpus all --dump-output
python tools/run_tests.py --ops "sage_attention,chunk_gla" --gpus 0
```

Operator selection follows `--ops` > `--op-list-file` > `--stages`.

Use the pytest recorder for FlagGems-compatible case-level JSON. It also writes an operator-level upload summary as `summary.json` beside the requested output:

```sh
pytest -m "sage_attention" --record json --output accuracy_sage_attention.json --continue-on-collection-errors -vs
```

The `--output` file keeps only the original FlagGems-compatible case or operator records. Its sibling `summary.json` contains top-level `timestamp`, `env`, and `result` fields. Each `result[operator]` has separate `accuracy` and `performance` results; `performance.data` groups valid shape-level baseline and FlagAttention latencies and speedups by dtype. From the repository root, a path-free `--record json -m OP_NAME` command also collects `benchmark/`. If `--output` is omitted, benchmark invocations write `benchmark_result.json`, while accuracy-only invocations write `accuracy_result.json`. The upload summary describes the current run even when an output path is reused.

`OP_NAME` is a pytest marker. The inventory names `chunk_gated_delta_rule_fwd` and `chunk_kda_enflame` are supported alongside their shorter legacy markers. A marker with no matching tests produces `NotRun` accuracy, `Skipped` performance, and no speedup data.

## Benchmarks

Available benchmark entry points are registered alongside their operators in [`conf/operators.yaml`](./conf/operators.yaml); not every inventory entry has a benchmark. Run them through the scheduler or directly, depending on the file:

```sh
python tools/run_tests.py --stages all --gpus 0

python benchmark/flash_benchmark.py
python benchmark/piecewise_benchmark.py
python benchmark/flash_decoding_benchmark.py

cd benchmark
pytest -m "sage_attention" --record json --output performance_result.json --continue-on-collection-errors -vs
```

The requested `performance_result.json` contains each measured shape's baseline latency, FlagAttention latency, and `speedup = baseline / FlagAttention` under the operator key. When only benchmark tests run, the separate `summary.json` marks accuracy as not run; a passing benchmark does not claim that the accuracy suite passed. Without a valid baseline, neither file invents a speedup for that shape. For a platform upload zip, place the generated `summary.json` at the archive root or one directory below it.

The path-free pytest command collects the pytest benchmark tests under `benchmark/`. Standalone benchmark scripts are run directly and do not contribute data to this pytest JSON file.

### Comparing against an NVIDIA baseline

[`baseline/`](./baseline) holds NVIDIA reference implementations, one module per operator, each declaring its shapes, dtypes, input builders, and the upstream callable it measures. Pass a baseline JSON produced from those references to compare a non-NVIDIA run shape by shape:

```sh
pytest benchmark -m "chunk_gla" --record json --base-data nvidia_baseline.json -vs
```

Each matched shape then carries a `compared_speedup` alongside the on-machine `speedup`. When the baseline supplies `_scaling_factors` for the running chip, the comparison is divided by the factor matching that shape's bottleneck resource, so the number reflects efficiency rather than raw hardware difference. Unmatched shapes record a `base_miss_reason`; a shape with neither an on-machine reference nor a baseline match fails the run instead of reporting an unqualified number. Without `--base-data` the comparison is simply disabled.

The scheduler runs accuracy tests before benchmarks. Under each operator's output directory, human-readable benchmark output is saved as `performance_stdout.log`, while structured records from each benchmark script are saved separately as `performance_records_0.log`, `performance_records_1.log`, and so on. These records use FlagGems-compatible `[INFO] {` JSON lines containing shapes, baseline latency, FlagAttention latency, and speedup when a baseline is available. Raw records use `speedup: null` without a baseline; the FlagGems-compatible `--record json` report omits those shape rows so FlagGems reports the dtype as `Unknown`.

Direct script execution without a record path still prints machine records to stdout for older schedulers. Set `FLAG_ATTN_BENCHMARK_LOG_PATH=/path/to/records.log` to keep direct console output free of JSON. Existing tables, plots, and matmul-based throughput remain available; the `flash_decoding_benchmark.py` plot axis now correctly says milliseconds. Historical v0.2 plots remain under [`assets/v0.2`](./assets/v0.2); rerun the current benchmark for conclusions about current code, Triton, and hardware.

## Repository layout

The package has three layers: generic Triton implementations written against NVIDIA, a runtime that detects the vendor and resolves each operator by name, and vendor packages that override individual operators.

```text
FlagAttention/
├── src/flag_attn/
│   ├── __init__.py                 # Public API facade; lazy per-operator dispatch
│   │
│   │   # Layer 1 -- generic implementations, pure Triton, NVIDIA as the reference
│   ├── flash.py, piecewise.py, split_kv.py, total.py, paged.py, dropout.py
│   ├── utils.py                    # has_triton_tle(), is_sm90_device(), tensor_cache
│   ├── FLA/                        # GLA, Gated Delta Rule, KDA, log-linear, MoBA,
│   │                               #   Parallax, Wall Attention, AttnRes, shared chunk helpers
│   ├── gdn2/                       # GDN2 facade + native/ Triton kernels
│   ├── parallel_nsa/               # NSA selection and compression
│   ├── minimax_sparse_attention/   # MiniMax M3 index + sparse attention
│   ├── infllmv2/                   # InfLLM-V2 prefill, backward, decode
│   ├── hpc_ops_attention/          # Hy3 FP8 prefill + six decode variants
│   ├── sage_attention/             # INT8 Q/K quantization + attention
│   ├── diffkv_attention/           # Paged DiffKV, Triton and TLE paths
│   ├── inkling_fa4/                # FA4 relative attention, Triton and TLE paths
│   ├── forgetting_attention/       # Forgetting Attention + naive reference
│   │
│   │   # Layer 2 -- vendor dispatch
│   ├── runtime/
│   │   ├── accelerator.py          # Torch device abstraction for optional backends
│   │   ├── configs_loader.py       # Reads backend/_<vendor>/tune_configs.yaml
│   │   └── backend/
│   │       ├── device_finder.py    # Vendor detection and device-type mapping
│   │       ├── op_registry.py      # resolve_operator(): vendor override or generic
│   │       │
│   │       │   # Layer 3 -- vendor implementations, may use TLE
│   │       ├── _nvidia/            # No overrides; tune_configs.yaml only
│   │       ├── _metax/  _hygon/  _mthreads/  _enflame/
│   │       └── _ascend/  _thead/  _iluvatar/
│   │
│   └── testing/                    # PyTorch reference implementations + test helpers
├── tests/flag_attn/                # Accuracy and dispatch tests, one file per operator
├── benchmark/                      # Performance entry points
├── baseline/                       # NVIDIA reference implementations for comparison
├── examples/                       # Small usage programs
├── conf/operators.yaml             # Operator stage/test/benchmark inventory
├── tools/run_tests.py              # Multi-device test scheduler
├── .github/configs/backends/       # Per-machine CI configuration
└── packaging/                      # Debian/RPM packaging
```

## Current limitations

- All compute kernels require a supported accelerator; CPU is metadata/reference-only.
- Operator and dtype coverage varies by backend. Do not infer support only from successful backend detection.
- FlashAttention and Piecewise Attention require head dimensions in `{16, 32, 64, 128}`; paged and specialized operators have their own constraints.
- FlashAttention dropout cannot run when the automatic dispatcher selects split-KV.
- Every operator family except FlashAttention and Piecewise Attention is currently alpha, including all of the sparse, recurrent, quantized, and fused operators listed above.
- Several TLE paths require a Triton build that provides TLE and accept stricter shapes than their native fallbacks. Hy3 additionally requires PyTorch 2.7.0, and NSA sliding-window composition requires the external `flash-attn` package.
- Many operators are forward- or inference-only; the operator overview tables state which provide gradients.

## More

FlagAttention is part of the FlagOS/FlagOpen open-source ecosystem. See [FlagOpen](https://flagopen.baai.ac.cn/) for more projects.

[<img src="./assets/logo/baai-flagopen.jpeg" alt="BAAI FlagOpen">](https://flagopen.baai.ac.cn/)
