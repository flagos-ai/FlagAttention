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

"""Public ``chunk_gla`` entry point for the Ascend backend.

Fixed-length calls in the default ``[K, V]`` state layout use the chunked
forward/backward pair (``chunk.py`` / ``chunk_bwd.py``).  Packed variable-length
inputs (``cu_seqlens`` with ``B == 1``) use the chunked varlen wrapper once the
average segment is long enough for its per-segment launch overhead to pay off;
shorter varlen inputs and ``state_v_first`` fall back to the recurrent
implementation in ``recurrent.py``, which is the numerical reference for both.
"""

import torch

from .chunk import MAX_HEAD_DIM, chunk_gla_fwd_chunked
from .chunk_bwd import chunk_gla_bwd_chunked
from .chunk_varlen import chunk_gla_varlen
from .recurrent import chunk_gla as _recurrent_chunk_gla

_CHUNKED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The chunked varlen path splits by segment on the host, so it only beats the
# single-launch recurrent kernel once a segment averages this many tokens.
_VARLEN_CHUNKED_MIN_TOKENS = 1024


class _ChunkGLAFunction(torch.autograd.Function):
    """Autograd wrapper around the chunked forward and backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        o, ht, aux = chunk_gla_fwd_chunked(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            return_intermediates=True,
        )
        ctx.save_for_backward(q, k, v, g, initial_state, aux["h"], aux["A"], aux["g_cum"])
        ctx.scale = scale
        return o, ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, initial_state, h, A, g_cum = ctx.saved_tensors
        if do is None:
            do = torch.zeros_like(v)
        dq, dk, dv, dg, dh0 = chunk_gla_bwd_chunked(
            q,
            k,
            v,
            g,
            h,
            A,
            g_cum,
            do.contiguous(),
            None if dht is None else dht.contiguous(),
            ctx.scale,
            initial_state=initial_state,
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dg.to(g.dtype), None, dh0, None


def _use_chunked(q: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether the chunked kernels cover this call."""
    return q.dtype in _CHUNKED_DTYPES and max(q.shape[-1], v.shape[-1]) <= MAX_HEAD_DIM


def _varlen_use_chunked(T: int, cu_seqlens, cu_seqlens_cpu) -> bool:
    """Whether the packed varlen call is long enough for the chunked path."""
    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    try:
        offsets = cu.tolist()
    except Exception:
        return False
    segments = len(offsets) - 1
    return segments > 0 and T // segments >= _VARLEN_CHUNKED_MIN_TOKENS


def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ascend GLA, chunked when possible and recurrent otherwise.

    The signature follows flag_attn.FLA.gated_linear_attention.chunk_gla.
    ``g`` is a per-key gate of shape ``[B, T, H, K]``; ``initial_state`` is
    float32.  ``cu_seqlens`` (variable length) and ``state_v_first`` are served
    by the recurrent implementation.
    """
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape[:3] == q.shape[:3], "v must share batch, sequence and head dimensions."
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not state_v_first and _use_chunked(q, v):
        if cu_seqlens is None and cu_seqlens_cpu is None:
            return _ChunkGLAFunction.apply(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g.contiguous(),
                scale,
                None if initial_state is None else initial_state.contiguous(),
                output_final_state,
            )
        if q.shape[0] == 1 and _varlen_use_chunked(q.shape[1], cu_seqlens, cu_seqlens_cpu):
            return chunk_gla_varlen(
                q,
                k,
                v,
                g,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
    return _recurrent_chunk_gla(
        q,
        k,
        v,
        g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
