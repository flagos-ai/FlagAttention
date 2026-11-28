# InfLLM-V2

Use `flag_attn.infllmv2_attention` for packed prefill or
`flag_attn.infllmv2_attention(..., stage="decode")` / `flag_attn.infllmv2_decode`
for decode. Existing parameters, causal alignment, configuration and gradients
are preserved.

`forward.py` and `backward.py` contain complete standard Triton implementations.
Dense attention uses the same portable attention kernel with every KV block
selected. Compression, pooling, selection and backward remain in this directory.

NVIDIA dispatch lives in `runtime/backend/_nvidia/infllmv2.py`. The TLE kernels
live in `runtime/backend/_nvidia/hopper/infllmv2/forward.py` and are loaded only
on SM90 with the required TLE interfaces available. Current crossover rules are:

| Stage | Hopper specialization |
| --- | --- |
| Stage 1 | BF16, GQA group 16, D64/D128, maximum Q length >= 65536 |
| TopK | BF16, maximum Q length >= 1024, 257–4096 blocks, TopK <= 64 |
| Stage 2 | Inference, BF16, GQA group 16, D128, block size 64, maximum Q length >= 1024 |

Other shapes and missing TLE use the standard Triton implementation. NVIDIA
dense attention retains its existing FlashAttention path. Set
`FLAG_ATTN_FORCE_GENERIC=1` to validate the complete portable path on NVIDIA,
including dense attention. No operator arguments are needed to select hardware.

Individual stages are also available through top-level imports:
`infllmv2_compress_k`, `infllmv2_stage1`, `infllmv2_pool_scores`,
`infllmv2_select_blocks`, and `infllmv2_sparse_attention`. They use the same
backend registry as the full pipeline. The existing `_allow_tle=False` stage
keyword selects the standard implementation for comparison.

Run accuracy and performance tests with:

```sh
pytest -m infllmv2_attention tests/flag_attn/test_infllmv2_attention.py benchmark/test_infllmv2_attention_benchmark.py
FLAG_ATTN_FORCE_GENERIC=1 pytest -m infllmv2_attention tests/flag_attn/test_infllmv2_attention.py benchmark/test_infllmv2_attention_benchmark.py
```
