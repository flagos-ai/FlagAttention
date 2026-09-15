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

import torch
import torch.nn.functional as F
import triton

from flag_attn.FLA.gated_linear_attention import chunk_gla as flag_attn_chunk_gla

DEFAULT_WARMUP = 10
DEFAULT_REP = 20

# optional FLA reference
_HAS_FLA_CHUNK = False
_fla_chunk_gla = None

try:
    from fla.ops.gla import chunk_gla as _fla_chunk_gla

    _HAS_FLA_CHUNK = True
except Exception:
    _HAS_FLA_CHUNK = False


def _fla_chunk_wrapper(q, k, v, g, **kwargs):
    return _fla_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=kwargs.get("scale", None),
        initial_state=kwargs.get("initial_state", None),
        output_final_state=kwargs.get("output_final_state", False),
        state_v_first=kwargs.get("state_v_first", False),
        cu_seqlens=kwargs.get("cu_seqlens", None),
    )



def _bench_ms(fn, warmup: int, rep: int) -> float:
    # do_bench performs an untimed first call and warmup before recording events,
    # so JIT/autotune work is excluded from the reported steady-state latency.
    return triton.testing.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


def _build_inputs(B, T, H, D, dtype, requires_grad=False):
    device = torch.device("cuda")
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    g_logit = torch.randn(
        B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )

    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }
    if requires_grad:
        return q, k, v, g_logit, kwargs
    else:
        g = F.logsigmoid(g_logit)
        return q, k, v, g, kwargs


# ---------------------------
# fwd+bwd benchmark helper
# ---------------------------
def _bench_fwd_bwd_ms(fn, q, k, v, g_logit, kwargs, warmup: int, rep: int) -> float:
    """Measure forward + backward pass time for a chunk_gla function.

    ``g_logit`` is the pre-logsigmoid raw tensor (a leaf).  F.logsigmoid is
    called *inside* the timed closure so each backward iteration gets a fresh
    computation graph.
    """

    params = [q, k, v, g_logit]

    def _fwd_bwd():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        g = F.logsigmoid(g_logit)
        out = fn(q, k, v, g, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()

    return triton.testing.do_bench(
        _fwd_bwd,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


_SHAPES = [
    (1, 4096, 32, 512),
    (2, 2048, 16, 512),
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (2, 2048, 16, 512),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
]

_DTYPES = [
    torch.bfloat16,
    # torch.float32,
    # torch.float16,
]


def _print_header(title: str, subtitle: str, warmup: int, rep: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {subtitle}")
    print(f"  warmup={warmup}ms  rep={rep}ms")
    print(f"{'=' * 70}")

    if _HAS_FLA_CHUNK:
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'fla(ms)':>10} {'flag_attn(ms)':>14} {'fla/flag_attn':>14}"
        )
    else:
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'flag_attn(ms)':>14}"
        )


def _print_row(B, T, H, D, dtype, ms_fla, ms_flag_attn) -> None:
    dtype_str = str(dtype).split(".")[-1]
    if _HAS_FLA_CHUNK:
        speedup = ms_fla / ms_flag_attn if ms_flag_attn > 0 else float("inf")
        print(
            f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} "
            f"{ms_fla:>10.3f} {ms_flag_attn:>14.3f} {speedup:>14.2f}x"
        )
    else:
        print(
            f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} "
            f"{ms_flag_attn:>14.3f}"
        )

def run_benchmark(warmup: int = DEFAULT_WARMUP, rep: int = DEFAULT_REP) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("chunk_gla benchmark requires CUDA")

    # ============================================================
    # Part 1: forward only
    # ============================================================
    _print_header(
        "chunk_gla benchmark — FWD ONLY",
        "provider: fla_chunk vs flag_attn_chunk",
        warmup,
        rep,
    )

    for dtype in _DTYPES:
        print("\ndtype:", dtype)
        for B, T, H, D in _SHAPES:
            q, k, v, g, kwargs = _build_inputs(B, T, H, D, dtype)

            ms_fla = None
            if _HAS_FLA_CHUNK:
                ms_fla = _bench_ms(
                    lambda: _fla_chunk_wrapper(q, k, v, g, **kwargs), warmup, rep
                )
            ms_flag_attn = _bench_ms(
                lambda: flag_attn_chunk_gla(q, k, v, g, **kwargs), warmup, rep
            )
            _print_row(B, T, H, D, dtype, ms_fla, ms_flag_attn)

    # ============================================================
    # Part 2: forward + backward
    # ============================================================
    _print_header(
        "chunk_gla benchmark — FWD + BWD",
        "provider: fla_chunk vs flag_attn_chunk  (forward + backward)",
        warmup,
        rep,
    )

    for dtype in _DTYPES:
        print("\ndtype:", dtype)
        for B, T, H, D in _SHAPES:
            q, k, v, g_logit, kwargs = _build_inputs(
                B, T, H, D, dtype, requires_grad=True
            )

            ms_fla = None
            if _HAS_FLA_CHUNK:
                ms_fla = _bench_fwd_bwd_ms(
                    _fla_chunk_wrapper,
                    q,
                    k,
                    v,
                    g_logit,
                    kwargs,
                    warmup,
                    rep,
                )
            ms_flag_attn = _bench_fwd_bwd_ms(
                flag_attn_chunk_gla,
                q,
                k,
                v,
                g_logit,
                kwargs,
                warmup,
                rep,
            )
            _print_row(B, T, H, D, dtype, ms_fla, ms_flag_attn)

    print(f"\n{'=' * 70}")
    print("  All done. ")
    print(f"{'=' * 70}\n")

run_benchmark()
