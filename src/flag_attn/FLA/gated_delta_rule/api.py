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

"""Public wrapper for the portable chunked gated delta rule forward."""

import torch
import triton
import triton.language as tl

from flag_attn.FLA.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd


@triton.jit
def _l2_normalize_last_dim_kernel(
    x,
    out,
    n_rows: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    h = row % H
    row_bt = row // H
    t = row_bt % n_rows
    b = row_bt // n_rows
    x_base = x + b * stride_x_b + t * stride_x_t + h * stride_x_h
    values = tl.load(x_base + offs * stride_x_k, mask=mask, other=0.0).to(tl.float32)
    inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(values * values, axis=0)), 1e-6)
    tl.store(out + row * K + offs, values * inv_norm, mask=mask)


def _l2_normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize the last dimension of a ``[B, T, H, K]`` tensor."""
    B, T, H, K = x.shape
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    _l2_normalize_last_dim_kernel[(B * T * H,)](
        x=x,
        out=out,
        n_rows=T,
        H=H,
        K=K,
        stride_x_b=x.stride(0),
        stride_x_t=x.stride(1),
        stride_x_h=x.stride(2),
        stride_x_k=x.stride(3),
        BLOCK_K=triton.next_power_of_2(K),
    )
    return out


def _as_seq_first(x: torch.Tensor, name: str, head_first: bool, expected_ndim: int) -> torch.Tensor:
    if x.ndim != expected_ndim:
        raise ValueError(f"{name} must be {expected_ndim}-dimensional, got {x.ndim}")
    return x.transpose(1, 2) if head_first else x


def _validate_inputs(q, k, v, beta, g, initial_state, cu_seqlens) -> None:
    B, T, Hg, K = q.shape
    H, V = v.shape[2], v.shape[3]
    if k.shape != (B, T, Hg, K):
        raise ValueError(f"k must have shape {(B, T, Hg, K)}, got {tuple(k.shape)}")
    if v.shape[:2] != (B, T):
        raise ValueError(f"v must start with shape {(B, T)}, got {tuple(v.shape[:2])}")
    if beta.shape != (B, T, H) or g.shape != (B, T, H):
        raise ValueError("beta and g must have shape [B, T, H] matching the value heads")
    if H % Hg:
        raise ValueError(f"value head count {H} must be a multiple of the query/key head count {Hg}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must share one dtype")
    if initial_state is not None and initial_state.shape[-2:] != (K, V):
        raise ValueError(f"initial_state must end with shape {(K, V)}, got {tuple(initial_state.shape[-2:])}")
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("variable-length inputs must be packed into a single batch row")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1-D tensor of offsets")


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
    """Portable chunked gated delta rule forward.

    Inputs follow the usual FLA layouts:
    - ``head_first=True``: q/k/v are ``[B, H, T, D]``, beta/g are ``[B, H, T]``.
    - ``head_first=False``: q/k/v are ``[B, T, H, D]``, beta/g are ``[B, T, H]``.

    q/k may use fewer heads than v when the q/k head count divides the v head
    count. Returns ``(o, final_state)``; ``final_state`` is ``None`` unless
    ``output_final_state`` is set.
    """
    if BT != 64:
        raise ValueError("chunk_gated_delta_rule currently supports only BT=64")

    q_seq = _as_seq_first(q, name="q", head_first=head_first, expected_ndim=4)
    k_seq = _as_seq_first(k, name="k", head_first=head_first, expected_ndim=4)
    v_seq = _as_seq_first(v, name="v", head_first=head_first, expected_ndim=4)
    beta_seq = _as_seq_first(beta, name="beta", head_first=head_first, expected_ndim=3)
    g_seq = _as_seq_first(g, name="g", head_first=head_first, expected_ndim=3)

    _validate_inputs(q_seq, k_seq, v_seq, beta_seq, g_seq, initial_state, cu_seqlens)

    if scale is None:
        scale = k_seq.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q_seq = _l2_normalize_last_dim(q_seq.contiguous())
        k_seq = _l2_normalize_last_dim(k_seq.contiguous())

    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        g=g_seq,
        beta=beta_seq,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    if head_first:
        o = o.transpose(1, 2)
    return o, final_state
