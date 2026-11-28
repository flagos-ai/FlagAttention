# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Automatically select compatible Hopper kernels before launch."""

from importlib import import_module
from functools import lru_cache
import torch
from flag_attn.utils import has_triton_tle


@lru_cache(maxsize=1)
def _tle_available():
    # Compiler APIs are fixed for this process; do not re-probe on hot launches.
    return has_triton_tle(
        3, 6, 0,
        required_api=(
            "gpu.types.BlockEncoding", "gpu.alloc", "gpu.alloc_barrier",
            "gpu.alloc_barriers", "gpu.copy", "gpu.barrier_wait", "gpu.wgmma",
            "gpu.wgmma_wait", "gpu.local_ptr", "gpu.set_layout",
        ),
    )


def _supports_hopper(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (9, 0) and _tle_available()


def parallel_forgetting_attn(
    q,
    k,
    v,
    log_fgate,
    *,
    head_first=False,
    seq_start=None,
    sm_scale=None,
    adaptive_threshold=None,
):
    if _supports_hopper(q.device):
        implementation = import_module(f"{__package__}.parallel")._tle_forgetting_attention
    else:
        from flag_attn.forgetting.parallel import _generic_parallel_forgetting_attn

        implementation = _generic_parallel_forgetting_attn
    return implementation(
        q,
        k,
        v,
        log_fgate,
        head_first=head_first,
        seq_start=seq_start,
        sm_scale=sm_scale,
        adaptive_threshold=adaptive_threshold,
    )


def prepare_forgetting_attention(q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold, prep_mode, fast):
    if _supports_hopper(q.device):
        implementation = import_module(f"{__package__}.parallel").prepare
    else:
        from flag_attn.forgetting.parallel import _generic_prepare

        implementation = _generic_prepare
    return implementation(q, k, v, gate, head_first, seq_start, sm_scale, adaptive_threshold, prep_mode, fast)
