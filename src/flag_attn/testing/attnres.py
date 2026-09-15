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

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for Attention Residuals."""
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    values = torch.stack([residual.reshape(-1, hidden_size) for residual in residuals]).float()
    keys = F.rms_norm(values, (hidden_size,), rms_weight.reshape(-1).float(), rms_eps)
    logits = (keys * (query.reshape(-1).float() * scale)).sum(dim=-1)
    probability = logits.softmax(dim=0)
    output = (probability[..., None] * values).sum(dim=0).view(output_shape)
    if output_rms_weight is not None:
        output = F.rms_norm(output, (hidden_size,), output_rms_weight.reshape(-1).float(), rms_eps)
    output = output.to(residuals[0].dtype)
    if return_weights:
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
