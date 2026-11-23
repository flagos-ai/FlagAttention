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

"""Single-token GDN recurrence with packed QKV and fused gate transforms."""

import torch

from .fused_recurrent import _launch


def _launch_packed(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
    *,
    config=None,
    use_tle=True,
):
    return _launch(
        mixed_qkv,
        None,
        None,
        a,
        b,
        scale,
        initial_state,
        True,
        None,
        ssm_state_indices,
        None,
        use_qk_l2norm_in_kernel,
        out,
        initial_state,
        config=config,
        use_tle=use_tle,
        gate_parameters=(A_log, dt_bias),
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
    """Update one token per request, returning the supplied out and cache tensors.

    mixed_qkv: [N, 2*H*K + HV*V], a/b: [N,HV], A_log/dt_bias: [HV].
    State: [pool,HV,V,K], output: [N,1,HV,V], indices: [N] CUDA int32/int64.
    g=-exp(A_log)*softplus(a+dt_bias, threshold=20), beta=sigmoid(b).
    Beta is rounded to b.dtype before the FP32 state update, matching vLLM.
    H divides HV. K/V are in 1..256; N and HV each must be <=65535.
    Projection and gate rows may be padded; their last dimension is contiguous.
    State strides are honored. Indices <=0 give zero output and leave cache
    untouched; positive indices must be in bounds and unique across requests.
    Physical cache writes must not alias another request's reads or writes.
    The output must be contiguous and disjoint from the input/cache allocations.
    Forward inference only, with FP32 recurrence and no TF32 arithmetic.
    """
    tensors = (mixed_qkv, a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    dev = mixed_qkv.device
    if dev.type != "cuda" or any(x.device != dev for x in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise RuntimeError("packed GDN decode supports forward inference only")
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv must be 2D with contiguous last dimension")
    if initial_state.ndim != 4:
        raise ValueError("state must use [pool,HV,V,K] layout")
    n = mixed_qkv.shape[0]
    hv, dv, dk = initial_state.shape[1:]
    if not (0 < dk <= 256 and 0 < dv <= 256 and 0 < hv <= 65535 and n <= 65535):
        raise ValueError("unsupported dimensions or CUDA grid limits")
    qk_dim = mixed_qkv.shape[1] - hv * dv
    if qk_dim <= 0 or qk_dim % (2 * dk) or hv % (qk_dim // (2 * dk)):
        raise ValueError("invalid packed QKV width or grouped head ratio")
    for gate in (a, b):
        if gate.shape != (n, hv) or gate.stride(-1) != 1:
            raise ValueError("a/b must have shape [N,HV] with contiguous last dimension")
    for parameter in (A_log, dt_bias):
        if parameter.shape != (hv,) or parameter.stride(0) != 1:
            raise ValueError("A_log/dt_bias must be contiguous [HV] tensors")
    if ssm_state_indices.shape != (n,) or ssm_state_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("state indices must be int32/int64 [N]")
    if out.shape != (n, 1, hv, dv) or not out.is_contiguous():
        raise ValueError("out must be contiguous with shape [N,1,HV,V]")
    floating = (torch.float16, torch.bfloat16, torch.float32)
    if any(x.dtype not in floating for x in tensors[:-1]) or out.dtype != mixed_qkv.dtype:
        raise ValueError("floating inputs must be FP16/BF16/FP32 and out must match mixed_qkv")
    _launch_packed(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
    )
    return out, initial_state
