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

import torch


def log_linear_attn_reference(q, k, v, g, level_scales):
    """Float32 reference for dense Log-Linear Attention."""
    sequence = q.shape[1]
    rows = torch.arange(sequence, device=q.device)[:, None]
    cols = torch.arange(sequence, device=q.device)[None, :]
    xor = torch.bitwise_xor(rows, cols)
    levels = torch.where(
        xor == 0,
        torch.zeros_like(xor),
        torch.floor(torch.log2(xor.float())).to(torch.int64) + 1,
    )

    gate = g.float().permute(0, 2, 1).cumsum(dim=-1)
    decay = torch.exp(gate[..., :, None] - gate[..., None, :])
    decay = torch.where(rows >= cols, decay, 0.0)
    scales = level_scales.float().permute(0, 2, 1, 3)
    scale_index = levels[None, None].expand(*scales.shape[:2], -1, -1)
    hierarchical_scale = torch.gather(scales, dim=-1, index=scale_index)

    content = torch.einsum("btqk,bsqk->bts", q.float(), k.float())
    weights = content[:, None] * decay * hierarchical_scale
    values = v.float().permute(0, 2, 1, 3).contiguous()
    output = (weights[..., None] * values[:, :, None]).sum(dim=3)
    return output.permute(0, 2, 1, 3).to(v.dtype)


__all__ = ["log_linear_attn_reference"]
