import math

import pytest
import torch

import flag_attn
from flag_attn.testing import backend as test_backend


ASSERT_RATIO = 0.01

# These cases cover the large head count and asymmetric key/value shapes used
# by GDN2. The Triton autotuner may take several minutes on a cold cache.
GDN2_TEST_CASES = [
    ((1, 4096, 16, 64, 64), (torch.float16, torch.bfloat16)),
    ((1, 8192, 96, 128, 128), (torch.float16, torch.bfloat16)),
    ((2, 2048, 16, 256, 512), (torch.float16,)),
]

pytestmark = [
    pytest.mark.chunk_gdn2,
    pytest.mark.skipif(
        not test_backend.supports_operator("chunk_gdn2"),
        reason=test_backend.skip_reason("chunk_gdn2"),
    ),
]


def _make_inputs(shape, dtype):
    B, T, H, K, V = shape
    device = test_backend.device
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)
    return q, k, v, g, b, w


def _assert_close(name, actual, expected):
    assert actual.shape == expected.shape, name
    actual = actual.float()
    expected = expected.float()
    assert torch.isfinite(actual).all(), f"{name}: nonfinite result"
    assert torch.isfinite(expected).all(), f"{name}: nonfinite reference"
    diff = actual - expected
    abs_err = diff.abs().max().item()
    ratio = diff.square().mean().sqrt().item() / (expected.square().mean().sqrt().item() + 1e-8)
    assert abs_err <= 1e-6 or ratio < ASSERT_RATIO, (
        f"{name} diff: abs={abs_err:.6f} ratio={ratio:.6f} limit={ASSERT_RATIO}"
    )


@pytest.mark.skipif(
    flag_attn.vendor_name in {"ascend", "enflame", "metax", "mthreads"},
    reason="These backends have different chunk-size or packed state-history constraints",
)
@pytest.mark.parametrize(
    "chunk_size,varlen,gate_mode",
    [(16, True, "safe"), (32, False, "log"), (64, False, "softplus")],
)
@torch.inference_mode()
def test_chunk_gdn2_matches_token_recurrence(chunk_size, varlen, gate_mode):
    torch.manual_seed(42)
    lengths = [chunk_size + 3, 5] if varlen else [chunk_size + 3] * 2
    B, T = (1, sum(lengths)) if varlen else (2, lengths[0])
    H, K, V = 2, 32, 24
    inputs = _make_inputs((B, T, H, K, V), torch.float16)
    device = inputs[0].device
    state = torch.randn(2, H, K, V, device=device) * 0.01
    cu_seqlens = torch.tensor([0, lengths[0], T], device=device) if varlen else None
    chunk_indices = torch.tensor([[0, 0], [0, 1], [1, 0]], device=device) if varlen else None
    A_log = torch.zeros(H, device=device)
    dt_bias = torch.full((H * K,), -3.0, device=device)
    lower_bound = -2.0 if gate_mode == "safe" else None

    actual, final_state, snapshots = flag_attn.chunk_gdn2(
        *inputs,
        initial_state=state.transpose(-1, -2).contiguous(),
        state_v_first=True,
        output_final_state=True,
        return_intermediate_states=True,
        use_gate_in_kernel=gate_mode != "log",
        use_qk_l2norm_in_kernel=True,
        safe_gate=gate_mode == "safe",
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )

    q, k, v, g, b, w = (tensor.float() for tensor in inputs)
    q = torch.nn.functional.normalize(q, dim=-1) * K**-0.5
    k = torch.nn.functional.normalize(k, dim=-1)
    if gate_mode != "log":
        gate = g + dt_bias.view(H, K)
        g = lower_bound * gate.sigmoid() if lower_bound is not None else -torch.nn.functional.softplus(gate)

    expected = torch.empty_like(v)
    expected_states = []
    expected_snapshots = []
    offset = 0
    for sequence, length in enumerate(lengths):
        batch = 0 if varlen else sequence
        h = state[sequence]
        for token in range(length):
            if token % chunk_size == 0:
                expected_snapshots.append(h.transpose(-1, -2))
            index = (batch, offset + token)
            h = h * g[index].exp().unsqueeze(-1)
            erase = (h * (b[index] * k[index]).unsqueeze(-1)).sum(-2)
            h = h + k[index].unsqueeze(-1) * (w[index] * v[index] - erase).unsqueeze(-2)
            expected[index] = (q[index].unsqueeze(-1) * h).sum(-2)
        expected_states.append(h.transpose(-1, -2))
        if varlen:
            offset += length

    _assert_close("output", actual, expected)
    _assert_close("final state", final_state, torch.stack(expected_states))
    _assert_close("state history", snapshots, torch.stack(expected_snapshots).view(B, -1, H, V, K))


@pytest.mark.parametrize(
    "shape,dtype",
    [(shape, dtype) for shape, dtypes in GDN2_TEST_CASES for dtype in dtypes],
)
@torch.inference_mode()
def test_chunk_gdn2_resumes_from_final_state(shape, dtype):
    torch.manual_seed(42)
    inputs = _make_inputs(shape, dtype)
    midpoint = shape[1] // 2

    output, final_state = flag_attn.chunk_gdn2(*inputs, output_final_state=True)
    prefix, prefix_state = flag_attn.chunk_gdn2(
        *(tensor[:, :midpoint].contiguous() for tensor in inputs), output_final_state=True
    )
    suffix, resumed_state = flag_attn.chunk_gdn2(
        *(tensor[:, midpoint:].contiguous() for tensor in inputs),
        initial_state=prefix_state,
        output_final_state=True,
    )

    _assert_close("prefix output", prefix, output[:, :midpoint])
    _assert_close("suffix output", suffix, output[:, midpoint:])
    _assert_close("final state", resumed_state, final_state)
