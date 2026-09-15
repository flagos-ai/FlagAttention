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

import argparse

import torch
import triton

from flag_attn.runtime.backend._ascend import forward


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SageAttention QK INT8 / PV FP16 forward")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=(1024, 2048, 4096, 8192, 16384, 32768))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--maxnreg", type=int)
    return parser.parse_args()


def benchmark(args):
    if hasattr(torch, "npu") and torch.npu.is_available():
        device = "npu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("SageAttention benchmark requires a CUDA or NPU device")

    dtype = getattr(torch, args.dtype)
    print(f"device: {device}")
    print("seq_len\tlatency_ms\ttflops")

    for seq_len in args.seq_lens:
        shape = (args.batch_size, args.num_heads, seq_len, args.head_dim)
        q = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
        k = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
        v = torch.randn(shape, device=device, dtype=torch.float16)
        q_scale = torch.rand(
            (args.batch_size, args.num_heads, triton.cdiv(seq_len, 128)),
            device=device,
            dtype=torch.float32,
        )
        k_scale = torch.rand(
            (args.batch_size, args.num_heads, triton.cdiv(seq_len, 64)),
            device=device,
            dtype=torch.float32,
        )

        def run():
            return forward(
                q,
                k,
                v,
                q_scale,
                k_scale,
                output_dtype=dtype,
                maxnreg=args.maxnreg,
            )

        latency_ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
        flops = 4 * args.batch_size * args.num_heads * seq_len * seq_len * args.head_dim
        tflops = flops / latency_ms * 1e-9
        print(f"{seq_len}\t{latency_ms:.4f}\t{tflops:.2f}")


if __name__ == "__main__":
    benchmark(parse_args())
