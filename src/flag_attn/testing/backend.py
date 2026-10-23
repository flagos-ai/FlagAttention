# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Device selection and timing shared by operator tests and benchmarks.

CUDA extensions and NVIDIA architecture checks must use is_nvidia(), rather
than assuming every backend exposing torch.cuda is an NVIDIA device.
"""

import importlib
import statistics
import time

import torch

from flag_attn import runtime

device = runtime.device.name
device_fn = runtime.torch_device_fn

# These generic APIs explicitly reject non-CUDA tensors. A vendor export
# overrides this restriction; a CUDA-compatible vendor retains the generic API.
_CUDA_GENERIC_OPERATORS = {
    "parallel_moba",
    "infllmv2_attention",
    "infllmv2_decode",
    "inkling_fa4_rel_attention",
    "diffkv_attention",
    "hy3_attention",
    "chunk_gated_delta_rule",
    "parallel_parallax",
    "parallax_decode",
    "parallax_attn_with_kvcache",
    "parallel_wall_attn",
    "log_linear_attn",
    "parallel_forgetting_attn",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_index_topk",
}


def is_available():
    return device != "cpu" and device_fn is not None and device_fn.is_available()


def is_nvidia():
    return is_available() and runtime.device.vendor_name == "nvidia"


def cuda_capability(*args):
    """NVIDIA SM version; (0, 0) means that NVIDIA SM checks do not apply."""
    return torch.cuda.get_device_capability(*args) if is_nvidia() else (0, 0)


def supports_msa_fp8() -> bool:
    """Whether the active MSA backend accepts FP8 inputs, for tests and benchmarks."""
    if runtime.device.vendor_name == "hygon":
        # Hygon MSA receives dequantized inputs from the framework.
        return False
    if getattr(torch, "float8_e4m3fn", None) is None:
        return False
    if not (
        supports_operator("minimax_m3_sparse_attn")
        or supports_operator("minimax_m3_sparse_attn_decode")
    ):
        return False
    # NVIDIA FP8 Tensor Core support starts with Ada (8.9) and Hopper (9.0).
    return not is_nvidia() or cuda_capability() >= (8, 9)


def get_device_name(*args):
    getter = getattr(device_fn, "get_device_name", None)
    return getter(*args) if getter is not None else runtime.device.vendor_name


def has_specialization(name):
    module_name = f"flag_attn.runtime.backend._{runtime.device.vendor_name}"
    try:
        backend = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return False
    return name in vars(backend).get("_OPERATOR_EXPORTS", {}) and name not in vars(backend).get("_GENERIC_ONLY_OPS", ())


def graph_available():
    return is_available() and all(
        hasattr(device_fn, name) for name in ("CUDAGraph", "graph", "Stream", "stream", "current_stream")
    )


def do_bench(fn, warmup=25, rep=100, quantiles=None, return_mode="mean", grad_to_none=None):
    """Preserve NVIDIA's Triton protocol; use backend events on other devices.

    Without device events, synchronized wall time is reported. Such latency
    includes host dispatch and must not be presented as kernel-only timing.
    Compilation runs before timing. warmup and rep are millisecond budgets.
    """
    if is_nvidia():
        from triton.testing import do_bench as triton_do_bench

        return triton_do_bench(
            fn, warmup=warmup, rep=rep, quantiles=quantiles, return_mode=return_mode, grad_to_none=grad_to_none
        )
    if not is_available():
        raise RuntimeError("No accelerator is available for benchmarking")
    if return_mode not in {"min", "max", "mean", "median", "all"}:
        raise ValueError(f"Unsupported return_mode: {return_mode}")
    fn()
    device_fn.synchronize()
    deadline = time.perf_counter() + warmup / 1000
    while time.perf_counter() < deadline:
        fn()
        device_fn.synchronize()
    values = []
    deadline = time.perf_counter() + rep / 1000
    event_type = getattr(device_fn, "Event", None)
    while not values or time.perf_counter() < deadline:
        if grad_to_none:
            for tensor in grad_to_none:
                tensor.grad = None
        if event_type is not None:
            start, end = event_type(enable_timing=True), event_type(enable_timing=True)
            start.record()
            fn()
            end.record()
            device_fn.synchronize()
            values.append(start.elapsed_time(end))
        else:
            device_fn.synchronize()
            start = time.perf_counter()
            fn()
            device_fn.synchronize()
            values.append((time.perf_counter() - start) * 1000)
    if quantiles is not None:
        return torch.quantile(torch.tensor(values), torch.tensor(quantiles)).tolist()
    if return_mode == "all":
        return values
    return {"min": min, "max": max, "mean": statistics.mean, "median": statistics.median}[return_mode](values)


def supports_operator(name, *, min_sm=None, exact_sm=None, tle=False):
    """Apply generic NVIDIA kernel requirements only without a vendor override."""
    if not is_available():
        return False
    if has_specialization(name):
        return True
    if name in _CUDA_GENERIC_OPERATORS and device != "cuda":
        return False
    if min_sm is not None and cuda_capability() < min_sm:
        return False
    if exact_sm is not None and cuda_capability() != exact_sm:
        return False
    if tle:
        from flag_attn.utils import has_triton_tle

        return is_nvidia() and has_triton_tle(3, 6, 0)
    return True


def skip_reason(name):
    if not is_available():
        return f"{name}: selected {runtime.device.vendor_name} accelerator is unavailable"
    if not has_specialization(name) and name in _CUDA_GENERIC_OPERATORS and device != "cuda":
        return f"{name}: generic implementation requires CUDA tensors; no {runtime.device.vendor_name} override is registered"
    return f"{name}: selected implementation does not support this device or configuration"
