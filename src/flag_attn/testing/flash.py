import math

import torch


def attention(
    q,
    k,
    v,
    causal,
    dropout_p=0.0,
    dropout_mask=None,
    sm_scale=None,
    return_log_normalizer=False,
    return_total_attention=False,
    upcast=False,
):
    input_dtype = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    # (B, H, T, D)
    D = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    num_heads_q = q.shape[1]
    num_heads_k = k.shape[1]
    assert num_heads_q % num_heads_k == 0
    num_groups = num_heads_q // num_heads_k

    if num_groups > 1:
        k = torch.repeat_interleave(k, repeats=num_groups, dim=1)
        v = torch.repeat_interleave(v, repeats=num_groups, dim=1)
    kv_seq_len = k.shape[-2]
    q_seq_len = q.shape[-2]
    p_seq = kv_seq_len - q_seq_len
    device = q.device

    ms = torch.arange(q_seq_len, device=device).unsqueeze(-1)
    ns = torch.arange(kv_seq_len, device=device)

    S = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        S = torch.where(ms + p_seq >= ns, S, float("-inf"))

    S = S.to(torch.float32)
    if return_log_normalizer:
        log_normalizer = torch.logsumexp(S, dim=-1)

    # upcast attention to fp32
    P = torch.softmax(S, dim=-1, dtype=torch.float32)
    if causal:
        P = torch.where(ms + p_seq >= ns, P, 0.0)

    if return_total_attention:
        tot_attn = torch.sum(P, dim=-2)

    # Applies dropout
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        P = P.masked_fill(~dropout_mask, 0.0)

    attn_output = torch.matmul(P.to(v.dtype), v) * dropout_scaling
    attn_output = attn_output.to(input_dtype)

    has_extra_return = return_log_normalizer or return_total_attention
    if has_extra_return:
        outs = (
            attn_output,
            log_normalizer if return_log_normalizer else None,
            tot_attn if return_total_attention else None,
        )
        return outs
    else:
        return attn_output
