# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Select Hopper GDN kernels for the input device, with a portable fallback."""

from functools import lru_cache
from importlib import import_module

import torch

from flag_attn.utils import has_triton_tle


@lru_cache(maxsize=1)
def _has_hopper_tle():
    if not has_triton_tle(3, 6, 0, required_api=("gpu.alloc", "gpu.local_ptr", "gpu.warp_specialize", "pipe")):
        return False
    # Some older installations expose the Python APIs without the IR bindings
    # used by shared-memory pipes. They must keep the portable implementation.
    from triton._C.libtriton import ir

    return hasattr(ir.builder, "make_swizzled_shared_encoding_attr")


def _implementation(q, module, name):
    hopper = q.device.type == "cuda" and torch.cuda.get_device_capability(q.device)[0] == 9 and _has_hopper_tle()
    package = (
        "flag_attn.runtime.backend._nvidia.hopper.gated_delta_rule" if hopper else "flag_attn.FLA.gated_delta_rule"
    )
    return getattr(import_module(f"{package}.{module}"), name)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _implementation(q, "api", "chunk_gated_delta_rule")(
        q, k, v, beta, g, BT, initial_state, output_final_state, cu_seqlens, head_first, scale, use_qk_l2norm_in_kernel
    )


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    return _implementation(q, "chunk", "chunk_gated_delta_rule_fwd")(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
    )


def fused_recurrent_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False,
):
    return _implementation(q, "fused_recurrent", "fused_recurrent_gated_delta_rule_fwd")(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _implementation(mixed_qkv, "packed_decode", "fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out, ssm_state_indices, use_qk_l2norm_in_kernel
    )
