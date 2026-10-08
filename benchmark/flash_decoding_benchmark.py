# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import math
import pathlib
import torch
import triton

import flag_attn

try:
    from benchmark.recording import record_triton_report
except ModuleNotFoundError:  # Direct execution from the benchmark directory.
    from recording import record_triton_report


try:
    from flash_attn import flash_attn_func
    FLASH_VER = 2
except BaseException:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None


configs = [triton.testing.Benchmark(
    x_names=['N_CTX'],
    x_vals=[2**i for i in range(9, 20)],
    line_arg='provider',
    line_vals=['flag_attn', 'torch', ] + (['flash'] if HAS_FLASH else []),
    line_names=['flag_attn', 'torch', ] + ([f'flash-{FLASH_VER}'] if HAS_FLASH else []),
    styles=[('red', '-'), ('green', '-'), ('blue', '-'), ('cyan', '-')],
    ylabel='ms',
    plot_name=f'attention_d-{D_HEAD}_dtype-{dtype} (ms)',
    args={'D_HEAD': D_HEAD, 'dtype': dtype}
) for D_HEAD in [64, 128]
    for dtype in [torch.float16]]

@triton.testing.perf_report(configs)
def bench_flash_attention(N_CTX, D_HEAD, provider, dtype=torch.float16):
    BATCH = 2
    H = 2048 // D_HEAD
    causal = False
    if provider == "flag_attn":
        q = torch.randn((BATCH, H, 1, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        fn = lambda: flag_attn.flash_attention(q, k, v, causal=causal)
        ms = triton.testing.do_bench(fn)
    if provider == "torch":
        q = torch.randn((BATCH, H, 1, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        try:
            fn = lambda: flag_attn.testing.flash_attention(q, k, v, causal=causal, upcast=False)
            ms = triton.testing.do_bench(fn)
        except torch.cuda.OutOfMemoryError as e:
            logging.info(f"torch OOM for batch_size: {BATCH}, num_heads: {H}, seqlen: {N_CTX}, headdim: {D_HEAD}")
            ms = float("inf")
    if provider == "flash":
        q = torch.randn((BATCH, 1, H, D_HEAD), dtype=dtype, device="cuda")
        k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device="cuda")
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        ms = triton.testing.do_bench(fn)

    return ms
    # # total TFLOPS: following Flash Attention v2, only gemms are counted.
    # macs = 2. * BATCH * H * N_CTX * D_HEAD # Q@K, P@V
    # total_flops = 2 * macs
    # return total_flops / ms * 1e-9

# only works on post-Ampere GPUs right now
today = datetime.date.today().strftime(format("%Y%m%d"))
output_dir = pathlib.Path(f"results_flash_attention_with_split_kv_{today}")
output_dir.mkdir(exist_ok=True)
results = bench_flash_attention.run(save_path=output_dir, print_data=True, return_df=True)
record_triton_report(
    bench_flash_attention,
    results,
    op_name='flash_attention_split_kv',
    provider='flag_attn',
    baseline='torch',
    latency_from_value=lambda value, shape: value if math.isfinite(value) else math.inf,
)
