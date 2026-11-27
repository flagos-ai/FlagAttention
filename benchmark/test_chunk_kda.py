# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable KDA inference timings through the two public API names."""

import math
import os
import sys

import pytest
import torch

import flag_attn
from flag_attn.testing import backend as test_backend
from flag_attn.utils import has_triton_tle
from benchmark.recording import benchmark_metric, record_benchmark_result
from benchmark.test_FLA.bench_chunk_kda import (
    BENCHMARK_CASES,
    DEFAULT_HEADS,
    _make_inputs,
    flash_kda,
)

pytestmark = [
    pytest.mark.KDA,
    pytest.mark.skipif(not test_backend.is_available(), reason="requires an available accelerator"),
]


@pytest.mark.parametrize("api", ["chunk_kda", "chunk_kda_fwd_infer"])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("case", list(BENCHMARK_CASES))
@torch.inference_mode()
def test_chunk_kda(case, chunk_size, api, record_property):
    """Retain all three original 8192-token layouts and the BT=32 coverage."""
    torch.manual_seed(42)
    lengths = BENCHMARK_CASES[case]
    args, kwargs = _make_inputs(lengths, DEFAULT_HEADS)
    kwargs["chunk_size"] = chunk_size
    operator = getattr(flag_attn, api)

    def candidate():
        return operator(*args, **kwargs)

    candidate()  # Compile and autotune before observing dispatch or measuring.
    sources = []
    previous_profile = sys.getprofile()

    def record_call(frame, event, arg):
        if event == "call" and frame.f_code.co_name.startswith("chunk_kda"):
            if frame.f_globals.get("__name__", "").startswith("flag_attn."):
                sources.append(frame.f_code.co_filename)

    sys.setprofile(record_call)
    try:
        actual = candidate()
    finally:
        sys.setprofile(previous_profile)
    assert sources
    candidate_source = sources[-1]
    force_generic = os.environ.get("FLAG_ATTN_FORCE_GENERIC", "0").lower() in {"1", "true", "yes", "on"}
    if force_generic:
        assert candidate_source.endswith("/FLA/chunk_kda.py")
    elif test_backend.is_nvidia():
        required = ("gpu.alloc", "gpu.local_ptr", "gpu.copy", "gpu.warp_specialize", "pipe")
        hopper = test_backend.cuda_capability()[0] == 9 and has_triton_tle(3, 6, 0, required_api=required)
        if hopper:
            from triton._C.libtriton import ir

            hopper = hasattr(ir.builder, "make_swizzled_shared_encoding_attr")
        expected = "/runtime/backend/_nvidia/hopper/kda.py" if hopper and chunk_size == 16 else "/FLA/chunk_kda.py"
        assert candidate_source.endswith(expected)
    assert all(torch.isfinite(value).all() for value in actual)
    baseline_ms = None
    if flash_kda is not None:

        def baseline():
            output = torch.empty_like(args[2])
            final = torch.empty_like(kwargs["initial_state"])
            flash_kda.fwd(
                *args,
                kwargs["scale"],
                output,
                kwargs["A_log"],
                kwargs["dt_bias"],
                kwargs["lower_bound"],
                initial_state=kwargs["initial_state"],
                final_state=final,
                cu_seqlens=kwargs["cu_seqlens"],
            )
            return output, final

        expected = baseline()
        for result, reference in zip(actual, expected):
            assert torch.isfinite(reference).all()
            error = (result.float() - reference.float()).square().mean().sqrt()
            magnitude = reference.float().square().mean().sqrt()
            assert error / (magnitude + 1e-8) < 0.005
        baseline_ms = test_backend.do_bench(baseline, warmup=25, rep=100, return_mode="median")
    latency = test_backend.do_bench(candidate, warmup=25, rep=100, return_mode="median")
    assert math.isfinite(latency) and latency > 0
    record_benchmark_result(
        record_property,
        op_name="KDA",
        dtype="bfloat16",
        mode="inference",
        result=[
            benchmark_metric(
                shape_detail={
                    "case": case,
                    "seq_lens": lengths,
                    "H": DEFAULT_HEADS,
                    "K": 128,
                    "V": 128,
                    "chunk_size": chunk_size,
                    "api": api,
                },
                latency=latency,
                candidate_source=candidate_source,
                latency_base=baseline_ms,
                speedup=baseline_ms / latency if baseline_ms is not None else None,
                accuracy="passed" if baseline_ms is not None else None,
            )
        ],
        baseline_name="FlashKDA" if flash_kda is not None else None,
        warmup_ms=25,
        rep_ms=100,
        return_mode="median",
    )
    print(f"{api} {case} chunk_size={chunk_size}: {latency:.6f} ms")
