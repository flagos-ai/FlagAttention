# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent PyTorch references for the MSA stored-value contract."""

import torch


def unpack_nvfp4(packed, scales):
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                       device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    return lut[codes] * scales.float().repeat_interleave(16, -1)


def reference_selection(data):
    """FP32 causal index max-score, stable ties by logical block ID."""
    if torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("MSA reference requires IEEE FP32; disable forced TF32 before starting Python")
    c = data.case
    nb = data.pages.shape[1]
    scores = torch.empty((c.heads, c.batch * c.q, nb), device="cuda", dtype=torch.float32)
    iq = unpack_nvfp4(data.iq, data.iqs) if c.packed else data.iq.float()
    for batch in range(c.batch):
        pages = data.pages[batch].long()
        keys = (unpack_nvfp4(data.ik[pages], data.iks[pages]) if c.packed else
                data.ik[pages].float().unsqueeze(1).expand(-1, c.heads, -1, -1))
        keys = keys.permute(1, 0, 2, 3).reshape(c.heads, nb * 128, 128)
        for start in range(0, c.q, 128):
            end = min(start + 128, c.q)
            q = iq[batch * c.q + start:batch * c.q + end].permute(1, 0, 2)
            logits = torch.bmm(q, keys.transpose(1, 2))
            positions = torch.arange(start, end, device="cuda") + c.kv - c.q
            logits.masked_fill_(torch.arange(nb * 128, device="cuda")[None, :] > positions[:, None], -float("inf"))
            scores[:, batch * c.q + start:batch * c.q + end] = logits.reshape(c.heads, end-start, nb, 128).amax(-1)
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :c.topk]
    ids = ids.masked_fill(~torch.isfinite(scores.gather(-1, ids)), nb).sort(-1).values
    return ids.masked_fill(ids == nb, -1).int().contiguous()


def reference_attention(data, selected):
    """FP32 attention on stored Q/K/V values, rounded once to BF16 output."""
    c = data.case
    result = torch.empty_like(data.output)
    for batch in range(c.batch):
        offset = batch * c.q
        for h in range(c.heads):
            for start in range(0, c.q, 64):
                end = min(start + 64, c.q)
                ids = selected[h, offset+start:offset+end].long()
                physical = data.pages[batch, ids.clamp_min(0)].long()
                if c.packed:
                    k = unpack_nvfp4(data.k[physical, h], data.ks[physical, h])
                    v = unpack_nvfp4(data.v[physical, h], data.vs[physical, h])
                else:
                    k, v = data.k[physical, h].float(), data.v[physical, h].float()
                k, v = k.flatten(1, 2), v.flatten(1, 2)
                if data.kg is not None:
                    k, v = k * data.kg, v * data.vg
                q = data.q[offset+start:offset+end, h*c.gqa:(h+1)*c.gqa].float()
                logits = torch.bmm(q, k.transpose(1, 2)) * (128**-0.5)
                positions = (ids[..., None] * 128 + torch.arange(128, device="cuda")).flatten(1)
                visible = (positions >= 0) & (positions < c.kv)
                visible &= positions <= (c.kv - c.q + torch.arange(start, end, device="cuda")[:, None])
                logits.masked_fill_(~visible[:, None, :], -float("inf"))
                result[offset+start:offset+end, h*c.gqa:(h+1)*c.gqa] = torch.bmm(
                    logits.softmax(-1).nan_to_num(), v).to(torch.bfloat16)
    return result


def selection_difference(data, selected, reference):
    c = data.case
    nb = data.pages.shape[1]
    positions = (torch.arange(c.q, device="cuda") + c.kv - c.q).repeat(c.batch)
    effective = (selected >= 0) & (selected < nb) & (selected * 128 <= positions[None, :, None])
    actual = selected.masked_fill(~effective, nb).sort(-1).values
    target = reference.masked_fill(reference < 0, nb).sort(-1).values
    return int((actual != target).any(-1).sum()), target.shape[0] * target.shape[1]
