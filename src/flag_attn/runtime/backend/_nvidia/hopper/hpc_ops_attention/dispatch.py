# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select Hopper kernels before launch; propagate compilation/runtime errors."""

from importlib import import_module

import torch

from flag_attn.hpc_ops_attention import _HY3_DECODE_IMPLEMENTATIONS
from flag_attn.hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8 import (
    attention_with_kvcache_blocksparse_prefill_fp8_triton,
)
from flag_attn.utils import has_triton_tle


def _supports_hopper(device):
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (9, 0)
        and has_triton_tle(3, 6, 0, required_api=("device_mesh", "gpu.wgmma", "gpu.warp_specialize"))
    )


def _decode_module(inputs, variant):
    module, _ = _HY3_DECODE_IMPLEMENTATIONS[variant]
    package = __package__ if _supports_hopper(inputs.q.device) else "flag_attn.hpc_ops_attention"
    return import_module(f"{package}.decode.{module}")


def hy3_attention_prepare(inputs, *, variant):
    module = _decode_module(inputs, variant)
    prepare = {
        "bf16_static": "prepare_static_bf16_workspace",
        "bf16_dynamic": "prepare_dynamic_bf16_workspace",
    }.get(variant, "prepare_decode_workspace")
    return getattr(module, prepare)(inputs)


def hy3_attention_workspace_is_reset(workspace, *, variant):
    if variant == "bf16_static":
        return True  # Static BF16 has no cooperative completion counters.
    module, _ = _HY3_DECODE_IMPLEMENTATIONS[variant]
    module = import_module(f"{__package__}.decode.{module}")
    reset = "bf16_dynamic_workspace_is_reset" if variant == "bf16_dynamic" else "workspace_is_reset"
    return getattr(module, reset)(workspace)


def _decode(inputs, workspace, variant):
    module = _decode_module(inputs, variant)
    return getattr(module, _HY3_DECODE_IMPLEMENTATIONS[variant][1])(inputs, workspace)


def hy3_attention_decode_bf16_static(inputs, workspace):
    return _decode(inputs, workspace, "bf16_static")


def hy3_attention_decode_bf16_dynamic(inputs, workspace):
    return _decode(inputs, workspace, "bf16_dynamic")


def hy3_attention_decode_fp8_qk_static(inputs, workspace):
    return _decode(inputs, workspace, "fp8_qk_static")


def hy3_attention_decode_fp8_qk_dynamic(inputs, workspace):
    return _decode(inputs, workspace, "fp8_qk_dynamic")


def hy3_attention_decode_fp8_kv_static(inputs, workspace):
    return _decode(inputs, workspace, "fp8_kv_static")


def hy3_attention_decode_fp8_kv_dynamic(inputs, workspace):
    return _decode(inputs, workspace, "fp8_kv_dynamic")


def hy3_attention(
    q,
    kcache,
    vcache,
    qscale,
    kscale,
    vscale,
    cu_seqlens_q,
    block_ids,
    seqlens_kvcache,
    max_seqlens_q,
    quant_type=1,
    block_mask=None,
    output=None,
    *,
    sparsity_bucket=None,
):
    implementation = attention_with_kvcache_blocksparse_prefill_fp8_triton
    if _supports_hopper(q.device):
        implementation = import_module(
            f"{__package__}.prefill.attention_blocksparse_prefill_fp8"
        ).attention_with_kvcache_blocksparse_prefill_fp8
    return implementation(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type,
        block_mask,
        output,
        sparsity_bucket=sparsity_bucket,
    )
