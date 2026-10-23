import os
import torch
import triton
from flag_attn.FLA.gated_delta_rule import chunk_gated_delta_rule as op
from flag_attn.FLA.gated_delta_rule.api import _set_tle_descriptor_allocator
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla

_set_tle_descriptor_allocator(torch.device("cuda"))
with torch.inference_mode():
    for dtype in [torch.float16, torch.bfloat16]:
        for B, T, H in [(2, 16384, 16), (4, 2048, 16), (4, 4096, 64)]:
            torch.manual_seed(42)
            q = torch.randn(B, T, H, 128, device="cuda", dtype=dtype) / 128**0.5
            k = torch.randn_like(q) / 128**0.5
            v = torch.randn_like(q)
            g = (-torch.rand(B, T, H, device="cuda") * 0.1).to(dtype)
            beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()

            def f():
                return op(q, k, v, beta, g, head_first=False, output_final_state=True)

            def ref():
                return fla(q, k, v, g, beta, output_final_state=True)

            timings = {}
            for name, enabled, fn in [("upstream", "0", f), ("fused", "1", f), ("fla", "0", ref)]:
                os.environ["FLAG_ATTN_CHUNK_GDR_FUSED_TLE"] = enabled
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                timings[name] = float(triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median"))
            print(dtype, (B, T, H), timings, "speedup", timings["upstream"] / timings["fused"], flush=True)
