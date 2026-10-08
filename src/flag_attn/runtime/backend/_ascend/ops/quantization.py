# Copyright 2024 SageAttention Team
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
import torch.nn.functional as F


def _quantize(x, block_size, sm_scale):
    batch, heads, seq_len, head_dim = x.shape
    blocks = (seq_len + block_size - 1) // block_size
    padded_len = blocks * block_size
    if padded_len != seq_len:
        x = F.pad(x, (0, 0, 0, padded_len - seq_len))
    x = x.float() * sm_scale
    x = x.reshape(batch, heads, blocks, block_size, head_dim)
    scale = torch.amax(torch.abs(x), dim=(-1, -2)).clamp_min(1e-8) / 127.0
    x_int8 = torch.round(x / scale[..., None, None]).clamp(-128, 127).to(torch.int8)
    return x_int8.reshape(batch, heads, padded_len, head_dim)[..., :seq_len, :], scale


def quant_per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    if tensor_layout == "HND":
        q_hnd, k_hnd = q, k
        restore = False
    elif tensor_layout == "NHD":
        q_hnd, k_hnd = q.transpose(1, 2), k.transpose(1, 2)
        restore = True
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    if BLKQ <= 0 or BLKK <= 0:
        raise ValueError("BLKQ and BLKK must be positive")
    if q_hnd.shape[-1] != k_hnd.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if sm_scale is None:
        sm_scale = q_hnd.shape[-1] ** -0.5
    if km is not None:
        if tensor_layout == "NHD" and km.ndim == 3:
            km = km.transpose(1, 2)
        k_hnd = k_hnd - km
    q_int8, q_scale = _quantize(q_hnd, BLKQ, sm_scale * 1.44269504)
    k_int8, k_scale = _quantize(k_hnd, BLKK, 1.0)
    if restore:
        q_int8 = q_int8.transpose(1, 2)
        k_int8 = k_int8.transpose(1, 2)
    return q_int8, q_scale, k_int8, k_scale
