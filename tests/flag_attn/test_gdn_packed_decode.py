# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytest.importorskip("triton.experimental.tle.language", reason="FlagTree/TLE required")

from flag_attn.FLA.gated_delta_rule import fused_recurrent_gated_delta_rule_packed_decode as op
from flag_attn.testing.packed_gdn import make_packed_case, reference_packed, unpack_reference_case
from flag_attn.testing.recurrent_gdn import check_close

pytestmark = [
    pytest.mark.gdn_forward,
    pytest.mark.fused_recurrent_gated_delta_rule_packed_decode,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dims", [(1, 2, 128, 128), (2, 6, 60, 80), (2, 8, 127, 129), (1, 1, 256, 64)])
@pytest.mark.parametrize("seed", [0, 17])
@torch.inference_mode()
def test_packed_reference(dtype, dims, seed, monkeypatch):
    import flag_attn.FLA.gated_delta_rule.packed_decode as module

    original = module._launch
    calls = []

    def tracked(*args, **kw):
        assert kw["gate_parameters"] is not None
        calls.append(True)
        return original(*args, **kw)

    monkeypatch.setattr(module, "_launch", tracked)
    c = make_packed_case(n=3, h=dims[0], hv=dims[1], kdim=dims[2], vdim=dims[3], dtype=dtype, seed=seed)
    expected = reference_packed(c)
    state0 = c["initial_state"].clone()
    previous = None
    for _ in range(3):
        c["initial_state"].copy_(state0)
        c["out"].fill_(float("nan"))
        result = op(**c)
        assert result[0] is c["out"] and result[1] is c["initial_state"]
        check_close(*result, *expected)
        if previous is not None:
            assert torch.equal(result[0], previous[0]) and torch.equal(result[1], previous[1])
        previous = [x.clone() for x in result]
    assert len(calls) == 3


@pytest.mark.parametrize(
    "case",
    [
        "padding",
        "strides",
        "threshold",
        "gate_extremes",
        "zero",
        "impulse",
        "zero_scale",
        "no_norm",
        "bf16_state",
        "fp16_state",
        "empty",
    ],
)
@torch.inference_mode()
def test_packed_boundaries(case):
    c = make_packed_case(
        n=4,
        h=2,
        hv=6,
        projection_padding=64,
        state_dtype=torch.bfloat16
        if case == "bf16_state"
        else torch.float16
        if case == "fp16_state"
        else torch.float32,
    )
    if case == "padding":
        c["ssm_state_indices"][0], c["ssm_state_indices"][-1] = 0, -1
    if case == "strides":
        for name in ("a", "b", "ssm_state_indices"):
            x = c[name]
            backing = torch.zeros((x.shape[0] * 2, *x.shape[1:]), device=x.device, dtype=x.dtype)
            backing[::2] = x
            c[name] = backing[::2]
        st = c["initial_state"]
        c["initial_state"] = st.transpose(-1, -2).contiguous().transpose(-1, -2)
    if case == "threshold":
        c["a"][:] = torch.tensor([-20, -1, 0, 19.875, 20, 20.125], device="cuda")
        c["dt_bias"].zero_()
    if case == "gate_extremes":
        c["a"][:] = torch.tensor([-80, -20, 0, 20, 80, 100], device="cuda")
        c["b"][:] = torch.tensor([-80, -20, 0, 20, 80, 100], device="cuda")
    if case in ("zero", "impulse"):
        c["mixed_qkv"].zero_()
        if case == "impulse":
            c["mixed_qkv"][:, ::128] = 1
    if case == "zero_scale":
        c["scale"] = 0.0
    if case == "no_norm":
        c["use_qk_l2norm_in_kernel"] = False
        c["mixed_qkv"] *= 0.05
    if case == "empty":
        for name in ("mixed_qkv", "a", "b", "out", "ssm_state_indices"):
            c[name] = c[name][:0]
    expected = reference_packed(c)
    check_close(*op(**c), *expected)


@torch.inference_mode()
def test_packed_continuation_matches_recurrent():
    from flag_attn import fused_recurrent_gated_delta_rule_fwd

    c = make_packed_case()
    separate = c["initial_state"].clone()
    for seed in range(4):
        torch.manual_seed(seed)
        c["mixed_qkv"].normal_()
        expected_case = unpack_reference_case({**c, "initial_state": separate})
        expected = fused_recurrent_gated_delta_rule_fwd(**expected_case)
        check_close(*op(**c), *expected)


@torch.inference_mode()
def test_packed_split_grid():
    n, hv = 21846, 3
    c = dict(
        mixed_qkv=torch.zeros((n, 2 + hv), device="cuda", dtype=torch.bfloat16),
        a=torch.zeros((n, hv), device="cuda"),
        b=torch.zeros((n, hv), device="cuda"),
        A_log=torch.zeros(hv, device="cuda"),
        dt_bias=torch.zeros(hv, device="cuda"),
        scale=1.0,
        initial_state=torch.zeros((n + 1, hv, 1, 1), device="cuda"),
        out=torch.full((n, 1, hv, 1), float("nan"), device="cuda", dtype=torch.bfloat16),
        ssm_state_indices=torch.arange(1, n + 1, device="cuda", dtype=torch.int64),
    )
    out, state = op(**c)
    assert torch.count_nonzero(out) == 0 and torch.count_nonzero(state) == 0


@torch.inference_mode()
def test_public_exports_and_untouched_padding():
    import flag_attn
    from flag_attn.FLA.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

    assert flag_attn.chunk_gated_delta_rule_fwd is chunk_gated_delta_rule_fwd
    assert callable(flag_attn.fused_recurrent_gated_delta_rule_fwd)
    c = make_packed_case(n=3)
    c["ssm_state_indices"][0] = 0
    c["ssm_state_indices"][2] = -1
    original = c["initial_state"].clone()
    expected = reference_packed(c)
    check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)
    untouched = torch.ones(original.shape[0], dtype=torch.bool, device="cuda")
    untouched[c["ssm_state_indices"][1].long()] = False
    assert torch.equal(c["initial_state"][untouched], original[untouched])


@pytest.mark.parametrize("axis", ["projection", "state_head", "state_value", "state_key"])
@torch.inference_mode()
def test_large_relative_offsets(axis):
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("large-offset addressing check requires 12 GiB free")
    c = make_packed_case(n=3, h=1, hv=3, kdim=3, vdim=3)
    stride = 2**30 + 64
    if axis == "projection":
        width = c["mixed_qkv"].shape[1]
        storage = torch.empty(2 * stride + width, device="cuda", dtype=c["mixed_qkv"].dtype)
        view = storage.as_strided((3, width), (stride, 1))
        view.copy_(c["mixed_qkv"])
        c["mixed_qkv"] = view
    else:
        state = c["initial_state"]
        strides = [64, 9, 3, 1]
        strides[{"state_head": 1, "state_value": 2, "state_key": 3}[axis]] = stride
        size = 1 + sum((d - 1) * s for d, s in zip(state.shape, strides))
        storage = torch.empty(size, device="cuda", dtype=state.dtype)
        view = storage.as_strided(state.shape, strides)
        view.copy_(state)
        c["initial_state"] = view
    expected = reference_packed(c)
    check_close(*op(**c), *expected)
    assert torch.count_nonzero(c["out"]) > 0


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dims", [(128, 128), (256, 64), (64, 256)])
@torch.inference_mode()
def test_low_precision_cache_large_parallelism(state_dtype, dims):
    from flag_attn.FLA.gated_delta_rule.packed_decode import _launch_packed
    from flag_attn import fused_recurrent_gated_delta_rule_fwd
    from flag_attn.testing.recurrent_gdn import make_case, reference

    c = make_packed_case(n=64, h=4, hv=8, kdim=dims[0], vdim=dims[1], state_dtype=state_dtype)
    expected = reference_packed(c)
    kernel = _launch_packed(**c)
    assert kernel.n_spills == 0
    check_close(c["out"], c["initial_state"], *expected)
    recurrent = make_case(64, 3, 4, 8, dims[0], dims[1], state_dtype=state_dtype, spec=True)
    expected = reference(recurrent)
    check_close(*fused_recurrent_gated_delta_rule_fwd(**recurrent), *expected)


@pytest.mark.parametrize("case", ["width", "out", "indices", "gate", "parameter", "grad"])
def test_packed_invalid(case):
    c = make_packed_case()
    if case == "width":
        c["mixed_qkv"] = c["mixed_qkv"][:, :-1]
    if case == "out":
        c["out"] = c["out"][:, 0]
    if case == "indices":
        c["ssm_state_indices"] = c["ssm_state_indices"].float()
    if case == "gate":
        c["a"] = c["a"][:, :-1]
    if case == "parameter":
        c["dt_bias"] = c["dt_bias"][:-1]
    if case == "grad":
        c["mixed_qkv"].requires_grad_(True)
    with pytest.raises((ValueError, RuntimeError)):
        op(**c)
