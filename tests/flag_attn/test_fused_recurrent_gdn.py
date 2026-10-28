# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytest.importorskip("triton.experimental.tle.language", reason="FlagTree/TLE required")

from flag_attn.FLA.gated_delta_rule.fused_recurrent import _launch, fused_recurrent_gated_delta_rule_fwd
from flag_attn.testing.recurrent_gdn import make_case, reference, check_close

pytestmark = [
    pytest.mark.gdn_forward,
    pytest.mark.fused_recurrent_gated_delta_rule_fwd,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("h,hv", [(2, 4), (4, 12), (2, 8)])
@pytest.mark.parametrize("length", [1, 3, 8])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("seed", [0, 17])
def test_recurrent_stateful(h, hv, length, strided, seed):
    case = make_case(4, length, h, hv, strided=strided, spec=length > 1, seed=seed)
    expected = reference(case)
    actual = fused_recurrent_gated_delta_rule_fwd(**case)
    check_close(*actual, *expected)
    assert actual[1].data_ptr() == case["initial_state"].data_ptr()


@pytest.mark.parametrize("kd,vd", [(1, 1), (60, 80), (64, 128), (128, 64), (256, 128), (128, 256)])
@pytest.mark.parametrize("vector_beta", [False, True])
def test_recurrent_tails_and_vector_beta(kd, vd, vector_beta):
    case = make_case(3, 3, 2, 6, kdim=kd, vdim=vd, vector_beta=vector_beta, spec=True)
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


@pytest.mark.parametrize(
    "dtype,state_dtype",
    [(torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)],
)
def test_recurrent_dtypes(dtype, state_dtype):
    case = make_case(3, 4, 2, 6, dtype=dtype, state_dtype=state_dtype, spec=True)
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


@pytest.mark.parametrize("inplace", [False, True])
def test_recurrent_dense_without_indices(inplace):
    n, length = 3, 4
    case = make_case(n, length, 2, 4)
    for name in ("q", "k", "v", "g", "beta"):
        x = case[name]
        case[name] = x.reshape(n, length, *x.shape[2:])
    case.update(cu_seqlens=None, ssm_state_indices=None, inplace_final_state=inplace)
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


@pytest.mark.parametrize("inplace", [False, True])
def test_recurrent_ragged_and_empty(inplace):
    case = make_case(5, 8, 2, 6, lengths=[0, 1, 8, 3, 0], spec=True)
    case["inplace_final_state"] = inplace
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


def test_recurrent_padding_and_index_strides():
    case = make_case(4, 4, 2, 6, padding=True, spec=True)
    case["ssm_state_indices"][-2] = -1
    idx = case["ssm_state_indices"]
    wide = torch.full((4, 8), -99, device="cuda", dtype=idx.dtype)
    wide[:, ::2] = idx
    case["ssm_state_indices"] = wide[:, ::2]
    original = case["initial_state"].clone()
    expected = reference(case)
    out, st = fused_recurrent_gated_delta_rule_fwd(**case)
    check_close(out, st, *expected)
    assert torch.equal(st[0], original[0])
    assert torch.count_nonzero(out[:, -4:]) == 0


def test_recurrent_continuation_and_repeatability():
    case = make_case(3, 8, 2, 6)
    original = case["initial_state"].clone()
    expected = reference(case)
    full = fused_recurrent_gated_delta_rule_fwd(**case)
    check_close(*full, *expected)
    full_o, full_s = full[0].clone(), full[1].clone()
    case["initial_state"].copy_(original)
    repeated = fused_recurrent_gated_delta_rule_fwd(**case)
    assert torch.equal(repeated[0], full_o) and torch.equal(repeated[1], full_s)
    # Feed one token from each sequence per call, writing one stable slot per request.
    case["initial_state"].copy_(original)
    ids = case["ssm_state_indices"]
    initial = original[ids[:, 0].long()].clone()
    stable = torch.arange(1, 4, device="cuda", dtype=torch.int32)
    pool = torch.zeros_like(original)
    pool[stable.long()] = initial
    outputs = []
    for t in range(8):
        step = {**case, "initial_state": pool, "cu_seqlens": None, "ssm_state_indices": stable}
        for name in ("q", "k", "v", "g", "beta"):
            x = case[name].reshape(3, 8, *case[name].shape[2:])
            step[name] = x[:, t : t + 1]
        outputs.append(fused_recurrent_gated_delta_rule_fwd(**step)[0])
    stacked = torch.cat(outputs, dim=1).reshape_as(full_o)
    torch.testing.assert_close(stacked, full_o, rtol=1.0e-2, atol=2.0e-3)
    torch.testing.assert_close(pool[stable.long()], full_s[ids[:, -1].long()], rtol=2.0e-4, atol=2.0e-5)


def test_recurrent_zero_decay_beta_limits():
    case = make_case(3, 4, 2, 6)
    case["g"].zero_()
    case["beta"][..., 0] = 0
    case["beta"][..., 1] = 1
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


def test_recurrent_preallocated_kernel_is_exercised():
    case = make_case(2, 4, 2, 4, spec=True)
    expected = reference(case)
    out = torch.full_like(case["v"], float("nan"))
    kernel = _launch(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["scale"],
        case["initial_state"],
        True,
        case["cu_seqlens"],
        case["ssm_state_indices"],
        case["num_accepted_tokens"],
        True,
        out,
        case["initial_state"],
    )
    assert "recurrent_fwd_kernel" in kernel.name
    check_close(out, case["initial_state"], *expected)


@pytest.mark.parametrize("strided_state", [False, True])
def test_recurrent_gate_and_state_strides(strided_state):
    case = make_case(3, 3, 2, 6, kdim=60, vdim=80)
    for name in ("g", "beta"):
        x = case[name]
        wide = torch.empty((*x.shape[:-1], x.shape[-1] * 2), device="cuda", dtype=x.dtype)
        wide[..., ::2] = x
        case[name] = wide[..., ::2]
    if strided_state:
        state = case["initial_state"]
        backing = torch.empty((*state.shape[:2], 60, 80), device="cuda", dtype=state.dtype)
        case["initial_state"] = backing.transpose(-1, -2)
        case["initial_state"].copy_(state)
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


def test_recurrent_public_export():
    import flag_attn

    case = make_case(2, 1, 2, 4)
    expected = reference(case)
    check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)


def test_recurrent_padded_projection_buffer():
    case = make_case(3, 4, 2, 6, spec=True, projection_padding=64)
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


def test_recurrent_padding_snapshots_are_initialized(monkeypatch):
    case = make_case(4, 4, 2, 6, padding=True, spec=True)
    case["inplace_final_state"] = False
    expected = reference(case)
    empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if tensor.is_floating_point():
            tensor.fill_(float("nan"))
        return tensor

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", poisoned_empty)
        out, final = fused_recurrent_gated_delta_rule_fwd(**case)
    check_close(out, final, *expected)
    assert torch.count_nonzero(final[-4:]) == 0


def test_recurrent_invalid_write_slot_keeps_cache():
    case = make_case(3, 4, 2, 6, spec=True)
    case["num_accepted_tokens"].fill_(1)
    case["ssm_state_indices"][0, -1] = 0
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


@pytest.mark.parametrize("values", ["zero", "impulse"])
def test_recurrent_longer_verification_values(values):
    case = make_case(3, 16, 2, 6, kdim=127, vdim=129, spec=True)
    case["g"].zero_()
    case["q"].zero_()
    case["k"].zero_()
    case["v"].zero_()
    if values == "impulse":
        case["q"][..., 0] = 1
        case["k"][..., 0] = 1
        case["v"][:, 0, :, 0] = 1
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


def test_recurrent_pre_normalized_inputs():
    case = make_case(4, 8, 2, 6, spec=True)
    for name in ("q", "k"):
        x = case[name]
        case[name] = torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype)
    case["use_qk_l2norm_in_kernel"] = False
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)


@pytest.mark.parametrize("lengths", [[0, 0, 0], [0, 1, 0]])
def test_recurrent_empty_batch_segments(lengths):
    case = make_case(3, 1, 2, 4, lengths=lengths)
    expected = reference(case)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**case), *expected)
