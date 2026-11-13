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

"""Host-split packed variable-length GLA for Ascend.

Inputs have shape [1,T,H,*], with cu_seqlens delimiting independent sequences
and one initial/final state per sequence. The host validates boundaries,
preferring cu_seqlens_cpu to avoid a device-to-host read, then invokes the
fixed-length chunked forward/backward on contiguous sequence views.
Each segment chooses its own chunk size, including short and tail chunks.

Forward intermediates are saved per segment; backward does not rerun forward.
Outputs and gradients are concatenated in sequence order, with no copy for a
single segment. Empty sequences preserve their state and state gradient.
The cost grows with segment count through launches and concatenation.

api.py selects this path with FLAG_ATTN_GLA_VARLEN_SPLIT=1. It retains the
fixed-length panel exponent handling for strong gates. The public autograd
wrapper requires B=1 and the default [K,V] state layout; state_v_first varlen
calls are dispatched to recurrent. The last boundary must equal T."""

import torch

from .chunk import MAX_HEAD_DIM, chunk_gla_fwd_chunked
from .chunk_bwd import chunk_gla_bwd_chunked

__all__ = [
    "chunk_gla_varlen",
    "chunk_gla_varlen_fwd",
    "chunk_gla_varlen_bwd",
]


def _segment_bounds(
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    T: int,
) -> list[int]:
    """Host-side ``cu_seqlens`` as a Python list of ints, validated against ``T``.

    ``cu_seqlens_cpu`` is preferred when given: it saves the device-to-host
    copy that ``.tolist()`` on a device tensor would trigger.
    """
    cu = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if cu is None:
        raise ValueError(
            "the variable-length chunked GLA needs cu_seqlens (or cu_seqlens_cpu)."
        )
    # ``tolist`` is the only host read on this path (and a no-op copy when the
    # boundaries already live on the CPU); keep it as the single torch op here.
    bounds = [int(v) for v in (cu.tolist() if isinstance(cu, torch.Tensor) else cu)]
    if len(bounds) < 2:
        raise ValueError(f"cu_seqlens must hold at least two entries, got {len(bounds)}.")
    if bounds[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {bounds[0]}.")
    if any(b < a for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"cu_seqlens must be non-decreasing, got {bounds}.")
    if bounds[-1] != T:
        raise ValueError(
            f"the packed sequence length T={T} must equal cu_seqlens[-1]={bounds[-1]}."
        )
    return bounds


def _check_inputs(
    q: torch.Tensor, v: torch.Tensor, cu_seqlens, cu_seqlens_cpu, T: int
) -> tuple[list[int], int]:
    """Shape/dtype checks shared by the forward and the backward."""
    B = q.shape[0]
    if B != 1:
        raise ValueError(
            f"the batch size is expected to be 1 when using cu_seqlens, got {B}; "
            "flatten variable-length inputs before processing."
        )
    K, V = q.shape[-1], v.shape[-1]
    if K > MAX_HEAD_DIM or V > MAX_HEAD_DIM:
        raise ValueError(
            f"chunked GLA supports K,V <= {MAX_HEAD_DIM}, got K={K}, V={V}."
        )
    bounds = _segment_bounds(cu_seqlens, cu_seqlens_cpu, T)
    return bounds, len(bounds) - 1


def _pack(parts: list[torch.Tensor], dim: int) -> torch.Tensor:
    """Concatenate segments along ``dim``; return a single segment without copying."""
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


def _check_initial_state(
    initial_state: torch.Tensor | None,
    N: int,
    H: int,
    K: int,
    V: int,
    state_v_first: bool,
) -> None:
    if initial_state is None:
        return
    if initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be in float32.")
    expected = (N, H, V, K) if state_v_first else (N, H, K, V)
    if tuple(initial_state.shape) != expected:
        raise ValueError(
            f"initial_state must be {expected} for a variable-length call "
            f"(one state per sequence), got {tuple(initial_state.shape)}."
        )


def chunk_gla_varlen_fwd(
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
    chunk_size: int | None = None,
    return_intermediates: bool = False,
):
    """Chunked GLA forward for packed variable-length inputs.

    Shapes (packed layout, ``B == 1``): ``q/k/g`` are ``[1, T, H, K]``,
    ``v``/``o`` are ``[1, T, H, V]``, ``cu_seqlens`` holds ``N + 1``
    boundaries with ``cu_seqlens[0] == 0`` and ``cu_seqlens[-1] == T``.
    ``initial_state`` is float32 ``[N, H, K, V]`` (``[N, H, V, K]`` when
    ``state_v_first``).

    Returns ``(o, ht)`` with ``o`` in ``v.dtype`` and ``ht`` the float32 final
    state ``[N, H, K, V]`` (``None`` unless ``output_final_state``).  With
    ``return_intermediates=True`` a third element is returned:
    ``{"cu_seqlens": [ints], "segs": [{"start", "stop", "g_cum", "h", "A"}]}``,
    the per-segment forward intermediates consumed by
    :func:`chunk_gla_varlen_bwd` (segments of length 0 are absent).
    """
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape[:3] == q.shape[:3], "v must share batch, sequence and head dimensions."
    B, T, H, K = q.shape
    V = v.shape[-1]

    bounds, N = _check_inputs(q, v, cu_seqlens, cu_seqlens_cpu, T)
    _check_initial_state(initial_state, N, H, K, V, state_v_first)
    if scale is None:
        scale = K ** -0.5

    state_shape = (V, K) if state_v_first else (K, V)
    o_parts = []
    ht_parts = [] if output_final_state else None
    segs = []
    for i in range(N):
        s, e = bounds[i], bounds[i + 1]
        h0_i = None if initial_state is None else initial_state[i : i + 1]
        if e <= s:
            # Empty sequence: no kernel work, the state just passes through.
            if ht_parts is not None:
                if h0_i is None:
                    ht_parts.append(torch.zeros((1, H, *state_shape), device=q.device, dtype=torch.float32))
                else:
                    ht_parts.append(h0_i)
            continue
        o_i, ht_i, aux = chunk_gla_fwd_chunked(
            q[:, s:e],
            k[:, s:e],
            v[:, s:e],
            g[:, s:e],
            scale=scale,
            initial_state=h0_i,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            chunk_size=chunk_size,
            return_intermediates=True,
        )
        o_parts.append(o_i)
        if ht_parts is not None:
            ht_parts.append(ht_i)
        if return_intermediates:
            segs.append(
                {
                    "start": s,
                    "stop": e,
                    "g_cum": aux["g_cum"],
                    "h": aux["h"],
                    "A": aux["A"],
                }
            )

    # The segments tile [0, T] contiguously, so packing is one ``cat`` (and no
    # copy at all when a single segment holds the whole packed batch).
    o = o_parts[0] if len(o_parts) == 1 else torch.cat(o_parts, dim=1)
    ht = None
    if ht_parts is not None:
        ht = ht_parts[0] if len(ht_parts) == 1 else torch.cat(ht_parts, dim=0)

    if return_intermediates:
        return o, ht, {"cu_seqlens": bounds, "segs": segs}
    return o, ht


def chunk_gla_varlen_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor | None,
    dht: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int | None = None,
    parts: dict | None = None,
):
    """Chunked GLA backward for packed variable-length inputs.

    ``do`` is the gradient of the forward output (``None`` means zeros) and
    ``dht`` the gradient of the final state, ``[N, H, K, V]`` (``None`` when
    the forward did not return it).  ``parts`` is the third element returned by
    :func:`chunk_gla_varlen_fwd` with ``return_intermediates=True``; when it is
    ``None`` the forward is re-run to rebuild the per-segment intermediates
    (``cu_seqlens`` is then required).

    Returns ``(dq, dk, dv, dg, dh0)`` exactly as ``chunk_gla_bwd_chunked``
    does (float32 for the accumulated gradients, plus whatever dtype that
    function writes), with ``dh0`` matching ``initial_state``'s shape and
    ``None`` when ``initial_state`` is ``None``.  Cast at the boundary.
    """
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape[:3] == q.shape[:3], "v must share batch, sequence and head dimensions."
    B, T, H, K = q.shape
    V = v.shape[-1]

    if parts is None:
        bounds, _ = _check_inputs(q, v, cu_seqlens, cu_seqlens_cpu, T)
        _, _, parts = chunk_gla_varlen_fwd(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=dht is not None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            return_intermediates=True,
        )
    else:
        bounds = parts["cu_seqlens"]
    N = len(bounds) - 1
    segs = parts["segs"]
    if len(segs) != sum(1 for i in range(N) if bounds[i + 1] > bounds[i]):
        raise ValueError("the forward intermediates do not match cu_seqlens.")

    if do is None:
        do = torch.zeros_like(v)
    # Autograd hands out zero-stride gradients for reductions such as ``o.sum()``;
    # the segment kernels index rows with the usual contiguous strides.
    do = do.contiguous()
    if dht is not None:
        dht = dht.contiguous()

    dh0_parts = [] if initial_state is not None else None
    dq_parts, dk_parts, dv_parts, dg_parts = [], [], [], []
    i_seg = 0
    for i in range(N):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            # Empty sequence: the state gradient passes straight through.
            if dh0_parts is not None:
                if dht is None:
                    dh0_parts.append(
                        torch.zeros((1, *initial_state.shape[1:]), device=q.device, dtype=torch.float32)
                    )
                else:
                    dh0_parts.append(dht[i : i + 1])
            continue
        seg = segs[i_seg]
        i_seg += 1
        h0_i = None if initial_state is None else initial_state[i : i + 1]
        dht_i = None if dht is None else dht[i : i + 1]
        seg_dq, seg_dk, seg_dv, seg_dg, seg_dh0 = chunk_gla_bwd_chunked(
            q[:, s:e],
            k[:, s:e],
            v[:, s:e],
            g[:, s:e],
            seg["h"],
            seg["A"],
            seg["g_cum"],
            do[:, s:e],
            dht_i,
            scale,
            initial_state=h0_i,
        )
        dq_parts.append(seg_dq)
        dk_parts.append(seg_dk)
        dv_parts.append(seg_dv)
        dg_parts.append(seg_dg)
        if dh0_parts is not None:
            dh0_parts.append(seg_dh0)

    dq = _pack(dq_parts, dim=1)
    dk = _pack(dk_parts, dim=1)
    dv = _pack(dv_parts, dim=1)
    dg = _pack(dg_parts, dim=1)
    dh0 = _pack(dh0_parts, dim=0) if dh0_parts is not None else None

    return dq, dk, dv, dg, dh0


class _ChunkGLAVarlenFunction(torch.autograd.Function):
    """Autograd wrapper around the per-segment chunked forward/backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state, chunk_size, cu_seqlens):
        o, ht, parts = chunk_gla_varlen_fwd(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            return_intermediates=True,
        )
        ctx.save_for_backward(q, k, v, g, initial_state)
        ctx.parts = parts
        ctx.scale = scale
        return o, ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, initial_state = ctx.saved_tensors
        dq, dk, dv, dg, dh0 = chunk_gla_varlen_bwd(
            q,
            k,
            v,
            g,
            do,
            dht,
            ctx.scale,
            initial_state=initial_state,
            parts=ctx.parts,
        )
        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dg.to(g.dtype),
            None,
            dh0,
            None,
            None,
            None,
        )


def chunk_gla_varlen(
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
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autograd-capable chunked GLA for packed variable-length inputs.

    Drop-in replacement for the varlen branch of ``api.chunk_gla``: same
    arguments, same ``(o, ht)`` outputs, one state per sequence.  Only the
    default ``[K, V]`` state layout is differentiable here (``state_v_first``
    raises ``NotImplementedError``; keep that case on the recurrent path).
    """
    if cu_seqlens is None and cu_seqlens_cpu is None:
        raise ValueError("chunk_gla_varlen requires cu_seqlens (or cu_seqlens_cpu).")
    if state_v_first:
        raise NotImplementedError(
            "the chunked variable-length backward only supports the [K, V] state "
            "layout; keep state_v_first on the recurrent implementation."
        )
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape[:3] == q.shape[:3], "v must share batch, sequence and head dimensions."
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _ChunkGLAVarlenFunction.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        scale,
        None if initial_state is None else initial_state.contiguous(),
        output_final_state,
        chunk_size,
        # Prefer CPU boundaries to avoid a device-to-host read.
        cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens,
    )
