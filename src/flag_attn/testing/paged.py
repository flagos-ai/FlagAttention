import torch


def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> None:
    output = torch.empty_like(query)

    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_size = value_cache.shape[2]
    head_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache[block_number, :, block_offset, :]
            keys.append(k)
            v = value_cache[block_number, :, block_offset, :]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        S = torch.bmm(q.transpose(0, 1).float(), keys.permute(1, 2, 0).float()) * scale
        P = torch.softmax(S, dim=-1)
        out = torch.bmm(P, values.transpose(0, 1).float()).transpose(0, 1)
        out = out.to(values.dtype)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)

    return output
