"""A clean version of moba implementation for educational purposes"""

import torch
import math


def moba_attn_varlen_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
) -> torch.Tensor:
    """Implement the moba brute-force setting for reference

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        max_seqlen (int): the max sequence length of the batch, same definition in flash attn

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """

    # qkv shape = [ S, H, D ]
    batch = cu_seqlens.numel() - 1
    softmax_scale = q.shape[-1] ** (-0.5)

    o = torch.zeros_like(q)
    for batch_idx in range(batch):
        batch_start = cu_seqlens[batch_idx].item()
        batch_end = cu_seqlens[batch_idx + 1].item()
        # get qkv of this batch
        q_ = q[batch_start:batch_end]
        k_ = k[batch_start:batch_end]
        v_ = v[batch_start:batch_end]
        o_ = o[batch_start:batch_end]
        # calc key gate weight
        key_gate_weight = []
        batch_size = batch_end - batch_start
        num_block = math.ceil(batch_size / moba_chunk_size)
        for block_idx in range(0, num_block):
            block_start = block_idx * moba_chunk_size
            block_end = min(batch_size, block_start + moba_chunk_size)
            key_gate_weight.append(k_[block_start:block_end].mean(dim=0, keepdim=True))
        key_gate_weight = torch.cat(key_gate_weight, dim=0)  # [ N, H, D ]
        # calc & mask gate
        # use fp32 to avoid precision issue in bf16
        q_ = q_.type(torch.float32)
        key_gate_weight = key_gate_weight.type(torch.float32)
        gate = torch.einsum("shd,nhd->hsn", q_, key_gate_weight)  # [ H, S, N ]
        key_gate_weight = key_gate_weight.type_as(k)
        q_ = q_.type_as(k)
        for i in range(num_block):
            # select the future Qs that can attend to KV chunk i
            gate[:, : (i + 1) * moba_chunk_size, i] = float("-inf")
            gate[:, i * moba_chunk_size : (i + 1) * moba_chunk_size, i] = float("inf")
        # gate_top_k_idx = gate_top_k_val = [ H S K ]
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=min(moba_topk, num_block), dim=-1, largest=True, sorted=False
        )
        gate_top_k_val, _ = gate_top_k_val.min(dim=-1)  # [ H, S ]
        need_attend = gate >= gate_top_k_val.unsqueeze(-1)
        # add gate_idx_mask in case of there is cornercases of same topk val been selected
        gate_idx_mask = torch.zeros(
            need_attend.shape, dtype=torch.bool, device=q.device
        )
        gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=True)
        need_attend = torch.logical_and(need_attend, gate_idx_mask)
        gate[need_attend] = 0
        gate[~need_attend] = -float("inf")
        gate = gate.repeat_interleave(moba_chunk_size, dim=-1)[
            :, :, :batch_size
        ]  # [ H, S, S ]
        gate.masked_fill_(
            torch.ones_like(gate, dtype=torch.bool).tril().logical_not(), -float("inf")
        )

        # calc qk = qk^t
        q_ = q_.type(torch.float32)
        k_ = k_.type(torch.float32)
        v_ = v_.type(torch.float32)
        qk = torch.einsum("xhd,yhd->hxy", q_, k_)
        # mask
        qk += gate
        qk *= softmax_scale
        # calc o
        p = qk.softmax(dim=-1)
        o_ += torch.einsum("hxy,yhd->xhd", p, v_)
        o = o.type_as(q)

    return o
