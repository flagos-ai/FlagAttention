# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend public APIs against independent CPU recurrences."""

import pytest
import torch

from flag_attn.testing import backend

pytestmark = pytest.mark.skipif(
    not backend.is_available() or backend.runtime.device.vendor_name != "ascend",
    reason="Ascend public API regressions require an available NPU",
)


def assert_relative(actual, expected, limit=0.01):
    actual = actual.detach().cpu().float()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    ratio = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    assert ratio < limit, f"relative RMS error {ratio.item()} >= {limit}"


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("length", [16, 33])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("state_v_first", [False, True])
@torch.inference_mode()
def test_gdn2_recurrence(length, dtype, state_v_first):
    from flag_attn import chunk_gdn2

    torch.manual_seed(42)
    shape = (1, length, 1, 64)
    q, k = [torch.randn(shape).to(dtype) / 8 for _ in range(2)]
    v = torch.randn(shape).to(dtype)
    g = -torch.rand(shape).to(dtype) * 0.1
    b, w = [torch.rand(shape).to(dtype) for _ in range(2)]
    state = torch.randn(1, 1, 64, 64) * 0.01
    h0 = state.transpose(-1, -2).contiguous() if state_v_first else state.clone()
    output, final = chunk_gdn2(
        *[x.to(backend.device) for x in (q, k, v, g, b, w)],
        initial_state=h0.to(backend.device), output_final_state=True,
        state_v_first=state_v_first,
    )
    assert output.dtype == dtype
    assert final.dtype == torch.float32
    expected = []
    q, k, v, g, b, w = [x.float() for x in (q, k, v, g, b, w)]
    for token in range(length):
        state = state * g[:, token].exp().unsqueeze(-1)
        erase = torch.einsum("bhk,bhkv->bhv", b[:, token] * k[:, token], state)
        state = state + k[:, token].unsqueeze(-1) * (v[:, token] * w[:, token] - erase).unsqueeze(-2)
        expected.append(torch.einsum("bhk,bhkv->bhv", q[:, token] / 8, state))
    assert_relative(output, torch.stack(expected, 1))
    assert_relative(final.transpose(-1, -2) if state_v_first else final, state)


@pytest.mark.chunk_gdn2
@torch.inference_mode()
def test_gdn2_rejects_unsupported_chunk_size():
    from flag_attn import chunk_gdn2

    x = torch.zeros(1, 16, 1, 64, device=backend.device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="chunk_size=16"):
        chunk_gdn2(x, x, x, x, x, x, chunk_size=64)


@pytest.mark.chunk_gla
@pytest.mark.parametrize("length", [17, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("with_initial_state", [False, True])
@torch.inference_mode()
def test_gla_forward(length, dtype, state_v_first, with_initial_state):
    from flag_attn import chunk_gla

    torch.manual_seed(43)
    shape = (1, length, 1, 32)
    hosts = [torch.randn(shape).to(dtype) / 8 for _ in range(3)]
    hosts.append(-torch.rand(shape).to(dtype) * 0.1)
    state = torch.randn(1, 1, 32, 32) * 0.01 if with_initial_state else torch.zeros(1, 1, 32, 32)
    initial = state.transpose(-1, -2).contiguous() if state_v_first else state.clone()
    output, final = chunk_gla(
        *[x.to(backend.device) for x in hosts], output_final_state=True,
        initial_state=initial.to(backend.device) if with_initial_state else None,
        state_v_first=state_v_first,
    )
    q, k, v, g = [x.float() for x in hosts]
    expected = []
    for token in range(length):
        state = state * g[:, token].exp().unsqueeze(-1)
        state = state + k[:, token].unsqueeze(-1) * v[:, token].unsqueeze(-2)
        expected.append(torch.einsum("bhk,bhkv->bhv", q[:, token] / (32 ** 0.5), state))
    assert output.dtype == dtype
    assert_relative(output, torch.stack(expected, 1))
    assert_relative(final.transpose(-1, -2) if state_v_first else final, state)


@pytest.mark.chunk_gla
@pytest.mark.parametrize("length", [17, 32])
def test_gla_backward(length):
    from flag_attn import chunk_gla

    torch.manual_seed(43)
    shape = (1, length, 1, 32)
    hosts = [torch.randn(shape).to(torch.float16) / 8 for _ in range(3)]
    hosts.append(-torch.rand(shape).to(torch.float16) * 0.1)
    refs = [x.float().requires_grad_() for x in hosts]
    inputs = [x.to(backend.device).requires_grad_() for x in hosts]
    output, final = chunk_gla(*inputs, output_final_state=True)
    q, k, v, g = refs
    state = torch.zeros(1, 1, 32, 32)
    expected = []
    for token in range(length):
        state = state * g[:, token].exp().unsqueeze(-1)
        state = state + k[:, token].unsqueeze(-1) * v[:, token].unsqueeze(-2)
        expected.append(torch.einsum("bhk,bhkv->bhv", q[:, token] / (32 ** 0.5), state))
    expected = torch.stack(expected, 1)
    assert_relative(output, expected.detach())
    assert_relative(final, state.detach())
    expected_grads = torch.autograd.grad(expected.square().sum(), refs)
    actual_grads = torch.autograd.grad(output.float().square().sum(), inputs)
    for actual, reference in zip(actual_grads, expected_grads):
        assert_relative(actual, reference, limit=0.02)


@pytest.mark.chunk_kda
@pytest.mark.parametrize("length", [16, 33])
@torch.inference_mode()
def test_kda_recurrence(length):
    from flag_attn import chunk_kda

    torch.manual_seed(44)
    shape = (1, length, 1, 64)
    q, k, v, g = [torch.randn(shape).to(torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1]).to(torch.bfloat16)
    a_log, dt_bias = torch.zeros(1), torch.zeros(1, 64)
    output, final = chunk_kda(
        *[x.to(backend.device) for x in (q, k, v, g, beta)],
        safe_gate=True, lower_bound=-5, A_log=a_log.to(backend.device),
        dt_bias=dt_bias.to(backend.device), output_final_state=True,
    )
    q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    decay = (-5 * g.sigmoid()).exp()
    beta = beta.sigmoid()
    state = torch.zeros(1, 1, 64, 64)
    expected = []
    for token in range(length):
        state = state * decay[:, token].unsqueeze(-1)
        residual = v[:, token] - torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        state = state + k[:, token].unsqueeze(-1) * (residual * beta[:, token, :, None]).unsqueeze(-2)
        expected.append(torch.einsum("bhk,bhkv->bhv", q[:, token] / 8, state))
    assert_relative(output, torch.stack(expected, 1))
    assert_relative(final, state)


@pytest.mark.minimax_m3_sparse_attn
@torch.inference_mode()
def test_minimax_prefill_stage_and_index():
    from flag_attn import minimax_m3_index_score, minimax_m3_index_topk, minimax_m3_sparse_attn

    torch.manual_seed(45)
    q = torch.randn(3, 1, 64, device=backend.device, dtype=torch.float16)
    keys = torch.randn(1, 128, 64, device=backend.device, dtype=q.dtype)
    kv = torch.randn(1, 1, 128, 128, device=backend.device, dtype=q.dtype)
    table = torch.zeros(1, 1, device=backend.device, dtype=torch.int32)
    cu = torch.tensor([0, 3], device=backend.device, dtype=torch.int32)
    lens = torch.tensor([19], device=backend.device, dtype=torch.int32)
    prefix = lens - 3
    scores = minimax_m3_index_score(q, keys, table, cu, lens, prefix, 3, 19, 1)
    expected_scores = torch.stack([(q[t].cpu().float() @ keys[0, :17+t].cpu().float().T).amax(-1) for t in range(3)])
    torch.testing.assert_close(scores[0, :3, 0].cpu(), expected_scores[:, 0], atol=0.03, rtol=0.005)
    ids = minimax_m3_index_topk(scores, cu, prefix, 3, 1, 0, 0)
    assert torch.equal(ids.cpu(), torch.zeros_like(ids.cpu()))
    output = torch.empty_like(q)
    result = minimax_m3_sparse_attn(q, kv, ids, table, cu, lens, prefix, 3, 1, 0.125, output, stage="prefill")
    assert result is None
    expected = []
    for token in range(3):
        cache = kv[0, 0, :17+token].cpu().float()
        probability = (q[token, 0].cpu().float() @ cache[:, :64].T / 8).softmax(-1)
        expected.append(probability @ cache[:, 64:])
    assert_relative(output, torch.stack(expected)[:, None], limit=0.01)


@pytest.mark.minimax_m3_sparse_attn
def test_minimax_rejects_unknown_stage():
    from flag_attn import minimax_m3_sparse_attn

    with pytest.raises(ValueError, match="Unsupported MiniMax stage"):
        minimax_m3_sparse_attn(stage="unsupported")


@pytest.mark.chunk_kda
@torch.inference_mode()
def test_kda_public_rejects_precomputed_chunk_indices():
    from flag_attn import chunk_kda

    x = torch.zeros(1, 16, 1, 64, device=backend.device, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="chunk_indices"):
        chunk_kda(x, x, x, x, x[..., 0], chunk_indices=object())


@pytest.mark.sage_attention
@torch.inference_mode()
def test_sage_public_forward_alias_and_quantization_error():
    from flag_attn import sage_attention
    from flag_attn.sage_attention import forward, per_block_int8

    assert forward is sage_attention
    q = torch.zeros(1, 1, 128, 64, device=backend.device, dtype=torch.int8)
    v = torch.ones_like(q, dtype=torch.float16)
    q_scale = torch.ones(1, 1, 1, device=backend.device)
    k_scale = torch.ones(1, 1, 2, device=backend.device)
    for implementation in (sage_attention, forward):
        output, lse = implementation(q, q, v, q_scale, k_scale, return_lse=True)
        torch.testing.assert_close(output, v, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, torch.full_like(lse, 7), atol=2e-2, rtol=2e-2)
    with pytest.raises(ValueError, match="positive"):
        per_block_int8(v, v, BLKQ=0)
