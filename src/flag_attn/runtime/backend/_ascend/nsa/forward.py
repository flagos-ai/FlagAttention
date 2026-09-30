"""Selected sparse attention on Ascend; compression and backward are not included."""

import math

import torch
import triton

from flag_attn.FLA.index import prepare_token_indices

from .kernels import parallel_nsa_fwd_kernel_tle, parallel_nsa_fwd_kernel_tle_grouped


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor | int,
    block_size: int,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    token_indices: torch.Tensor | None = None,
    *,
    token_group: int = 4,
):
    """Return output and natural-log LSE for causal selected attention.

    Q is [B, T, HQ, K], K/V are [B, T, H, K/V], and indices are
    [B, T, H, S]. Negative and future blocks are ignored. Indices may differ
    for every token. Empty selections produce zero output and -inf LSE.
    Tensor counts are [B, T, H] with values in [0, S]. Packed variable-length
    inputs use B=1 and cumulative sequence lengths starting at zero and ending
    at T. Metadata tensors must be contiguous and on the same NPU as Q.

    Full tiles use packed K/V and grouped queries; other inputs retain masked
    asynchronous loads. This entry point is forward-only.
    """
    tensors = (q, k, v, block_indices)
    if any(x.ndim != 4 for x in tensors):
        raise ValueError("Q, K, V and block_indices must have four dimensions")
    if q.device.type != "npu" or any(x.device != q.device for x in tensors):
        raise ValueError("All inputs must be on the same NPU")
    if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        raise ValueError("Q, K and V must have the same FP16 or BF16 dtype")
    if any(x.requires_grad for x in (q, k, v)):
        raise NotImplementedError("Ascend NSA currently supports forward inference only")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("Inputs must be contiguous")
    B, T, H, K = k.shape
    HQ, V, S = q.shape[2], v.shape[-1], block_indices.shape[-1]
    if min(B, T, H, K, HQ, V, S) <= 0:
        raise ValueError("Input dimensions must be positive")
    if q.shape[:2] != (B, T) or q.shape[-1] != K or v.shape[:3] != (B, T, H):
        raise ValueError("Incompatible Q/K/V shapes")
    if block_indices.shape[:3] != (B, T, H) or HQ % H:
        raise ValueError("Incompatible head counts or block_indices shape")
    G = HQ // H
    if G < 16 or G & (G - 1) or K > 128:
        raise NotImplementedError("Requires a power-of-two GQA group >=16 and K<=128")
    if not isinstance(block_size, int) or block_size < 16 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two >=16")
    if not isinstance(token_group, int) or token_group < 1:
        raise ValueError("token_group must be a positive integer")
    if block_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_indices must be INT32 or INT64")

    def check_metadata(x, name):
        if x.device != q.device or not x.is_contiguous() or x.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be contiguous INT32/INT64 on the same NPU")

    if isinstance(block_counts, torch.Tensor):
        check_metadata(block_counts, "block_counts")
        if block_counts.shape != (B, T, H):
            raise ValueError("block_counts must have shape [B,T,H]")
    elif not isinstance(block_counts, int) or not 0 <= block_counts <= S:
        raise ValueError("block_counts must be an integer in [0,S] or a tensor")
    if cu_seqlens is not None:
        check_metadata(cu_seqlens, "cu_seqlens")
        if B != 1 or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("Variable-length inputs require B=1 and cumulative lengths")
        if token_indices is None:
            token_indices = prepare_token_indices(cu_seqlens)
        check_metadata(token_indices, "token_indices")
        if token_indices.shape != (T, 2):
            raise ValueError("token_indices must have shape [T,2]")
    elif token_indices is not None:
        raise ValueError("token_indices requires cu_seqlens")
    scale = K ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")

    BK, BV = triton.next_power_of_2(K), min(128, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    o = torch.empty((B, T, HQ, V), dtype=v.dtype, device=q.device)
    lse = torch.empty((B, T, HQ), dtype=torch.float32, device=q.device)
    full_tiles = T % block_size == 0 and K == BK and V == BV
    grouped = (
        token_group > 1 and cu_seqlens is None and full_tiles
        and isinstance(block_counts, int) and block_counts == S
        and triton.cdiv(T, token_group) * NV * B * H <= 65535
    )
    if grouped:
        packed_k = k.transpose(1, 2).contiguous()
        packed_v = v.transpose(1, 2).contiguous()
        parallel_nsa_fwd_kernel_tle_grouped[(triton.cdiv(T, token_group), NV, B * H)](
            q, packed_k, packed_v, o, lse, scale, block_indices, T,
            H, HQ, G, K, V, S, block_size, BK, BV, token_group,
        )
    else:
        max_tokens = 65535 // (NV * B * H)
        if max_tokens < 1:
            raise NotImplementedError("Batch/head/value dimensions exceed the Ascend grid limit")
        chunk_t = min(max_tokens, 256 if T >= 65536 or B * H >= 64 else 1024)
        for start in range(0, T, chunk_t):
            parallel_nsa_fwd_kernel_tle[(min(chunk_t, T - start), NV, B * H)](
                q, k, v, o, lse, scale, block_indices, block_counts,
                cu_seqlens, token_indices, T, start,
                H, HQ, G, K, V, S, block_size, BK, BV,
                PACKED_KV=False, FAST_VALID_BLOCKS=False, FULL_TILES=full_tiles,
            )
    return o, lse
