# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select Parallax kernels using the input device and available extensions."""

from importlib import import_module

import torch

from flag_attn.utils import has_triton_tle


def _implementation(q, stage):
    hopper = q.is_cuda and torch.cuda.get_device_capability(q.device)[0] == 9
    if hopper and (stage == "parallel" or has_triton_tle(required_api=("load",))):
        return import_module(f"flag_attn.runtime.backend._nvidia.hopper.parallax.{stage}")
    return import_module(f"flag_attn.FLA.parallax.{stage}")


def parallel_parallax(q, r, k, v, scale=None, window_size=None, cu_seqlens=None, **kwargs):
    return _implementation(q, "parallel").parallel_parallax(
        q, r, k, v, scale, window_size, cu_seqlens, **kwargs
    )


def parallax_decode(q, r, k, v, qk_scale=None, *, window_size_left=-1, out=None):
    return _implementation(q, "decode").parallax_decode(
        q, r, k, v, qk_scale, window_size_left=window_size_left, out=out
    )


def parallax_attn_with_kvcache(
    q, r, k_cache, v_cache, *, page_table=None, seqused_k=None, window_size=None, scale=None, out=None
):
    return _implementation(q, "decode").parallax_attn_with_kvcache(
        q, r, k_cache, v_cache, page_table=page_table, seqused_k=seqused_k,
        window_size=window_size, scale=scale, out=out,
    )
