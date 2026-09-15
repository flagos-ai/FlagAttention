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

import os

import pytest
import torch

torch_gcu = pytest.importorskip("torch_gcu")

from flag_attn.runtime.backend._enflame.nsa.parallel_nsa_compression import (
    parallel_nsa_compression,
)


def _device():
    index = int(
        os.environ.get("S60_TEST_DEVICE", "0")
    )
    torch.gcu.set_device(index)
    return torch.device(f"gcu:{index}")



def naive_nsa_compression_fwd(q, k, v, scale, block_size):
    """
    Naive reference implementation of NSA compression forward.

    The compression represents each block of BS tokens as a single key/value
    representation. For query token at position t, it attends to compression
    representations from blocks that are *complete* before t, i.e.
    blocks whose last token index < t:  block i covers tokens [i*BS, (i+1)*BS),
    so only blocks where (i+1)*BS - 1 < t  =>  i < (t+1)//BS are visible.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    V = v.shape[-1]
    G = HQ // H
    BS = block_size

    q = q.float()
    k = k.float()
    v = v.float()

    o = torch.zeros(B, T, HQ, V, device=q.device, dtype=torch.float32)
    lse = torch.zeros(B, T, HQ, device=q.device, dtype=torch.float32)

    for b in range(B):
        for t in range(T):
            # number of complete blocks before token t
            NC = (t + 1) // BS
            if NC == 0:
                lse[b, t, :] = 0.0
                continue

            for hq in range(HQ):
                h = hq // G
                q_t = q[b, t, hq, :]  # (K,)

                scores = []
                vals = []
                for i_c in range(NC):
                    # compression representation at block i_c
                    # k[b, i_c, h, :]  -- already compressed
                    k_c = k[b, i_c, h, :]  # (K,)
                    v_c = v[b, i_c, h, :]  # (V,)

                    score = (q_t * scale) @ k_c  # scalar
                    scores.append(score)
                    vals.append(v_c)

                scores = torch.tensor(scores, device=q.device)  # (NC,)
                vals = torch.stack(vals, dim=0)  # (NC, V)

                # online-safe softmax
                m = scores.max()
                p = torch.exp(scores - m)
                acc = p.sum()
                o_vals = (p[:, None] * vals).sum(0) / acc

                o[b, t, hq, :] = o_vals
                lse[b, t, hq] = m + torch.log(acc)

    return o, lse


@pytest.mark.parallel_nsa_compression
@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("K", [64])
@pytest.mark.parametrize("V", [64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_parallel_nsa_compression_fwd_accuracy(B, T, H, HQ, K, V, block_size, dtype):
    """Compare triton forward with a naive token-by-token reference."""
    device = _device()
    torch.manual_seed(42)

    TC = T // block_size
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, TC, H, K, device=device, dtype=dtype)
    v = torch.randn(B, TC, H, V, device=device, dtype=dtype)
    scale = K**-0.5

    # ensure T is a multiple of block_size for deterministic comparison
    if T % block_size != 0:
        pytest.skip("T must be a multiple of block_size for this test")

    ref_o, ref_lse = naive_nsa_compression_fwd(q, k, v, scale, block_size)

    # triton forward
    res_o, res_lse = parallel_nsa_compression(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        scale=scale,
        cu_seqlens=None,
    )

    torch.testing.assert_close(res_o.float(), ref_o, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(res_lse.float(), ref_lse, rtol=1e-1, atol=2e-1)


@pytest.mark.parallel_nsa_compression
@pytest.mark.parametrize("T", [128, 256])
@pytest.mark.parametrize("block_size", [64])
def test_parallel_nsa_compression_bwd_accuracy(T, block_size):
    """Verify backward produces finite gradients."""
    device = _device()
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # H=4, HQ=64 ensures G=HQ//H=16 which satisfies tl.dot minimum K dimension
    B, H, HQ, K, V = 1, 4, 64, 64, 64
    TC = T // block_size
    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, TC, H, K, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, TC, H, V, device=device, dtype=dtype, requires_grad=True)
    scale = K**-0.5

    if T % block_size != 0:
        pytest.skip("T must be a multiple of block_size")

    o, lse = parallel_nsa_compression(
        q=q,
        k=k,
        v=v,
        block_size=block_size,
        scale=scale,
        cu_seqlens=None,
    )

    grad = torch.randn_like(o)
    o.backward(grad)

    # verify gradients are finite
    assert q.grad is not None, "dq is None"
    assert k.grad is not None, "dk is None"
    assert v.grad is not None, "dv is None"
    assert torch.isfinite(q.grad).all(), "dq has non-finite values"
    assert torch.isfinite(k.grad).all(), "dk has non-finite values"
    assert torch.isfinite(v.grad).all(), "dv has non-finite values"


@pytest.mark.parallel_nsa_compression
@pytest.mark.parametrize("T", [64, 128])
def test_parallel_nsa_compression_no_grad(T):
    """Test forward-only (no grad) path for various shapes."""
    device = _device()
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, H, HQ, K, V = 1, 4, 16, 64, 64
    block_size = 64
    TC = T // block_size

    q = torch.randn(B, T, HQ, K, device=device, dtype=dtype)
    k = torch.randn(B, TC, H, K, device=device, dtype=dtype)
    v = torch.randn(B, TC, H, V, device=device, dtype=dtype)
    scale = K**-0.5

    if T % block_size != 0:
        pytest.skip("T must be a multiple of block_size")

    with torch.no_grad():
        o, lse = parallel_nsa_compression(
            q=q,
            k=k,
            v=v,
            block_size=block_size,
            scale=scale,
            cu_seqlens=None,
        )

    assert o.shape == (B, T, HQ, V)
    assert lse.shape == (B, T, HQ)
    assert torch.isfinite(o).all()
    assert torch.isfinite(lse).all()
