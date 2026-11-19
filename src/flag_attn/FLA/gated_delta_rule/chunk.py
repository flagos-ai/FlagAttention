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

"""Stage orchestration for the portable chunked gated delta rule forward."""

import torch

from flag_attn.FLA.gated_delta_rule.output import gdn_fwd_output
from flag_attn.FLA.gated_delta_rule.prepare import gdn_fwd_prepare
from flag_attn.FLA.gated_delta_rule.state import gdn_fwd_state

# Chunk length of the decomposition. 64 keeps the UT transform's [BT, BT]
# tile within registers on every backend while staying tensor-core friendly.
CHUNK_SIZE = 64


def _chunk_size_for_sequence(T: int, is_varlen: bool) -> int:
    if is_varlen:
        return CHUNK_SIZE
    return min(CHUNK_SIZE, max(16, 1 << (max(T, 1) - 1).bit_length()))


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """Run the chunked forward and return the staged intermediates.

    Returns ``(g_cumsum, o, A, final_state, w, h, v_new)``, matching the
    established tuple shape. ``A`` is ``None`` here: the portable route keeps
    the UT transform inside the prepare kernel's registers rather than
    materializing it, so there is no attention-like matrix to hand back.
    """
    gate_dtype = g.dtype
    tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    for name, tensor in tensors.items():
        if not tensor.is_contiguous():
            tensors[name] = tensor.contiguous()
    q, k, v, g, beta = (tensors[name] for name in ("q", "k", "v", "g", "beta"))
    if initial_state is not None and not initial_state.is_contiguous():
        initial_state = initial_state.contiguous()
    if cu_seqlens is not None and not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    chunk_size = _chunk_size_for_sequence(q.shape[1], cu_seqlens is not None)

    g_cumsum, w, u = gdn_fwd_prepare(
        k=k,
        v=v,
        g=g,
        beta=beta,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = gdn_fwd_state(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = gdn_fwd_output(
        q=q,
        k=k,
        v_new=v_new,
        h=h,
        g=g_cumsum,
        scale=float(scale),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return g_cumsum.to(gate_dtype), o, None, final_state, w, h, v_new
