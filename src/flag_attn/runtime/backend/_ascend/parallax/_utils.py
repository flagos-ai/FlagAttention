"""Host-side scheduling helpers shared by Parallax backward implementations."""

from contextlib import contextmanager, nullcontext

import torch


def supports_staged(q, window_size_left):
    """Check staged-kernel length, dtype, and launch-grid limits."""
    batch, time, heads, dim = q.shape
    if not 128 <= time <= 4096:
        return False
    if q.dtype != torch.bfloat16 or dim not in (64, 128):
        return False
    if dim == 64 and window_size_left >= 0:
        return False

    padded_time = 1 << (time - 1).bit_length()
    row_block = min(8, 8192 // padded_time)
    head_programs = batch * heads
    softmax_programs = head_programs * ((time + row_block - 1) // row_block)

    if dim == 64:
        tiles = (time + 127) // 128
        score_tiles = tiles * (tiles + 1) // 2
    elif window_size_left < 0:
        tiles = (time + 127) // 128
        score_tiles = tiles * (tiles + 1) // 2 * 2
    else:
        query_tiles = (time + 63) // 64
        key_tiles = (window_size_left + 64 + 128 - 2 + 127) // 128
        score_tiles = query_tiles * key_tiles

    # Bound softmax and score grids; this also limits the four BF16
    # intermediates to less than 4 GiB.
    return softmax_programs <= 65535 and score_tiles * head_programs <= 65535


@contextmanager
def backward_streams(device, concurrent=True):
    """Join independent derivatives before returning to the caller's stream."""
    if not concurrent or torch.npu.is_current_stream_capturing():
        yield nullcontext(), nullcontext()
        return

    current = torch.npu.current_stream(device)
    dqr = torch.npu.Stream(device=device)
    dkv = torch.npu.Stream(device=device)
    dqr.wait_stream(current)
    dkv.wait_stream(current)
    try:
        yield torch.npu.stream(dqr), torch.npu.stream(dkv)
    finally:
        current.wait_stream(dqr)
        current.wait_stream(dkv)
