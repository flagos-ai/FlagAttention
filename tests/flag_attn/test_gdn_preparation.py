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

"""Allocation-sensitive prefill regression and shared preparation boundaries."""

import pytest
import torch

pytest.importorskip("triton.experimental.tle.language", reason="FlagTree/TLE required")

from flag_attn import chunk_gated_delta_rule_fwd
from flag_attn.testing.recurrent_gdn import reference

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["short_gva", "tail", "ragged"])
@torch.inference_mode()
def test_shared_preparation_poison_and_repeat(dtype, case, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("register preparation requires Hopper")
    torch.manual_seed(2091)
    batch, length, h, hv, dk, dv = (1, 512, 16, 32, 128, 128)
    cu = None
    if case == "tail":
        batch, length, h, hv, dk, dv = 2, 71, 2, 6, 60, 80
    if case == "ragged":
        batch, length, h, hv, dk, dv = 1, 82, 2, 6, 128, 64
        cu = torch.tensor([0, 0, 7, 71, 82], device="cuda", dtype=torch.int64)
    n = batch if cu is None else cu.numel() - 1
    q = torch.randn((batch, length, h, dk), device="cuda", dtype=dtype) / dk**0.5
    k = torch.randn_like(q) / dk**0.5
    v = torch.randn((batch, length, hv, dv), device="cuda", dtype=dtype)
    g = -torch.rand((batch, length, hv), device="cuda", dtype=torch.float32) * 0.1
    if case == "short_gva":
        g = g.to(dtype)
    beta = torch.rand_like(g)
    initial = None if case == "short_gva" else torch.randn((n, hv, dk, dv), device="cuda") * 0.05
    cache = torch.zeros((n, hv, dv, dk), device="cuda") if initial is None else initial.transpose(-1, -2).contiguous()
    expected = reference(
        dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=dk**-0.5,
            initial_state=cache,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=False,
        )
    )
    expected = (expected[0], expected[1].transpose(-1, -2))
    original_empty, original_like, original_new = torch.empty, torch.empty_like, torch.Tensor.new_empty

    def poison(t):
        if t.is_cuda and t.is_floating_point():
            t.fill_(float("nan"))
        return t

    monkeypatch.setattr(torch, "empty", lambda *a, **kw: poison(original_empty(*a, **kw)))
    monkeypatch.setattr(torch, "empty_like", lambda *a, **kw: poison(original_like(*a, **kw)))
    monkeypatch.setattr(torch.Tensor, "new_empty", lambda self, *a, **kw: poison(original_new(self, *a, **kw)))
    monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "0")
    monkeypatch.setenv("FLAG_ATTN_GDN_REGISTER_PREP", "1")
    previous = None
    for _ in range(3):
        result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, dk**-0.5, initial, True, cu)
        assert result[0].dtype == g.dtype
        for actual, target in zip((result[1], result[3]), expected):
            assert torch.isfinite(actual).all()
            ratio = (actual.float() - target.float()).square().mean().sqrt() / (
                target.float().square().mean().sqrt() + 1e-8
            )
            assert ratio < 0.01, ratio.item()
        if previous is not None:
            assert all(torch.equal(a, b) for a, b in zip(result[:4], previous))
        previous = [x.clone() for x in result[:4]]


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [7, 64, 71])
@pytest.mark.parametrize("block_k", [32, 64])
@pytest.mark.parametrize("warps", [1, 2, 4, 8])
@pytest.mark.parametrize("outputs", [(True, False), (False, True), (True, True)], ids=["ungated", "gated", "both"])
@torch.inference_mode()
def test_preparation_all_configs_initialize_dense_outputs(dtype, length, block_k, warps, outputs):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("register preparation requires Hopper")
    from flag_attn.FLA.gated_delta_rule.chunk_fused_tle_prepare import _prepare_kernel

    torch.manual_seed(73)
    dk = 128 if length == 64 else 60
    k = torch.randn((1, length, 2, dk), device="cuda", dtype=dtype) / dk**0.5
    g = -torch.rand((1, length, 6), device="cuda") * 0.1
    beta = torch.rand_like(g)
    gc = torch.full_like(g, float("nan"))
    ungated, gated_output = outputs
    a = torch.full((1, length, 6, 64), float("nan"), device="cuda", dtype=dtype) if ungated else None
    ag = torch.full((1, length, 6, 64), float("nan"), device="cuda", dtype=dtype) if gated_output else None
    chunks = (length + 63) // 64
    _prepare_kernel.fn.fn[(chunks, 6)](
        k=k,
        g=g,
        g_out=gc,
        beta=beta,
        A=a,
        AG=ag,
        cu_seqlens=None,
        chunk_indices=None,
        T=length,
        H=2,
        HV=6,
        K=dk,
        BT=64,
        BC=16,
        BK=block_k,
        USE_G=not ungated,
        IS_VARLEN=False,
        OUTPUT_GATED=gated_output,
        OUTPUT_UNGATED=ungated,
        MERGE_PRECISION="ieee",
        FULL_CHUNKS=length % 64 == 0,
        num_warps=warps,
    )
    pad = chunks * 64 - length
    kg = torch.nn.functional.pad(k.repeat_interleave(3, 2).cpu().double(), (0, 0, 0, 0, 0, pad))
    kg = kg.reshape(1, chunks, 64, 6, dk).transpose(2, 3)
    bg = torch.nn.functional.pad(beta.cpu().double(), (0, 0, 0, pad)).reshape(1, chunks, 64, 6).transpose(2, 3)
    gp = torch.nn.functional.pad(g.cpu().double(), (0, 0, 0, pad)).reshape(1, chunks, 64, 6).transpose(2, 3).cumsum(-1)
    eye = torch.eye(64, dtype=torch.float64).expand(1, chunks, 6, 64, 64)
    lower = torch.tril((kg @ kg.transpose(-1, -2)) * bg[..., :, None], diagonal=-1)
    inverse = torch.linalg.solve_triangular(eye + lower, eye, upper=False, unitriangular=True)
    mask = torch.ones((64, 64), dtype=torch.bool).tril()
    difference = torch.where(mask, gp[..., :, None] - gp[..., None, :], 0)
    gated = torch.where(mask, inverse * difference.exp(), 0)
    for actual, target in ((a, inverse), (ag, gated)):
        if actual is None:
            continue
        target = target.transpose(2, 3).reshape(1, chunks * 64, 6, 64)[:, :length].to(device=actual.device, dtype=dtype)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, target, rtol=1e-2, atol=2e-3)
        upper = torch.arange(64, device="cuda")[None, :] > (torch.arange(length, device="cuda") % 64)[:, None]
        assert torch.count_nonzero(actual.masked_select(upper[None, :, None, :])) == 0
    expected_gc = gp.transpose(2, 3).reshape(1, chunks * 64, 6)[:, :length].to(device=gc.device, dtype=torch.float32)
    torch.testing.assert_close(gc, expected_gc, rtol=1e-6, atol=1e-6)


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [7, 23, 64, 71])
@pytest.mark.parametrize("gate_mode", ["decay", "zero", "mixed"])
@pytest.mark.parametrize("batch", [1, 3])
@torch.inference_mode()
def test_native_gated_preparation_boundaries(dtype, length, gate_mode, batch, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("register preparation requires Hopper")
    from flag_attn.testing.recurrent_gdn import make_case
    import flag_attn.FLA.gated_delta_rule.chunk_fused_tle_prepare as preparation

    calls = []
    original = preparation._prepare_kernel.run

    def record(*args, **kwargs):
        calls.append(kwargs["T"])
        return original(*args, **kwargs)

    monkeypatch.setattr(preparation._prepare_kernel, "run", record)

    case = make_case(batch, length, 2, 6, 60, 80, dtype=dtype, strided=False)
    for name in ("q", "k", "v", "g", "beta"):
        x = case[name]
        case[name] = x.reshape(batch, length, *x.shape[2:])
    case["cu_seqlens"] = case["ssm_state_indices"] = None
    case["use_qk_l2norm_in_kernel"] = False
    case["initial_state"] = case["initial_state"][:batch].contiguous()
    for name in ("q", "k"):
        case[name] /= 60**0.5
    if gate_mode == "zero":
        case["g"].zero_()
    elif gate_mode == "mixed":
        case["g"] = torch.rand_like(case["g"]) * 0.2 - 0.1
    expected = reference(case)
    actual = chunk_gated_delta_rule_fwd(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["scale"],
        case["initial_state"].transpose(-1, -2).contiguous(),
        True,
    )
    assert actual[2].shape[-1] == min(64, max(16, 1 << (length - 1).bit_length()))
    assert calls == [batch * length if length % 64 == 0 else length]
    for result, target in ((actual[1], expected[0]), (actual[3].transpose(-1, -2), expected[1])):
        assert torch.isfinite(result).all()
        ratio = (result.float() - target.float()).square().mean().sqrt() / (
            target.float().square().mean().sqrt() + 1e-8
        )
        assert ratio < 0.01, ratio.item()


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(60, 80), (128, 128)])
@pytest.mark.parametrize("warps", [2, 4])
@pytest.mark.parametrize("stages", [2, 3, 4])
@torch.inference_mode()
def test_narrow_state_configs(dtype, dims, warps, stages, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("narrow state configurations require Hopper")
    import flag_attn.FLA.chunk_delta_h as state
    from flag_attn.testing.recurrent_gdn import make_case

    dk, dv = dims
    case = make_case(2, 71, 2, 6, dk, dv, dtype=dtype, strided=False)
    for name in ("q", "k", "v", "g", "beta"):
        x = case[name]
        case[name] = x.reshape(2, 71, *x.shape[2:])
    case["cu_seqlens"] = case["ssm_state_indices"] = None
    case["use_qk_l2norm_in_kernel"] = False
    case["initial_state"] = case["initial_state"][:2].contiguous()
    for name in ("q", "k"):
        case[name] /= dk**0.5
    expected = reference(case)
    raw = state.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn.fn
    calls = []

    class ForcedState:
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append(kwargs["N"])
                return raw[grid({"BV": 16})](
                    **kwargs,
                    USE_G=kwargs["g"] is not None,
                    USE_GK=kwargs["gk"] is not None,
                    USE_INITIAL_STATE=kwargs["h0"] is not None,
                    STORE_FINAL_STATE=kwargs["ht"] is not None,
                    SAVE_NEW_VALUE=kwargs["v_new"] is not None,
                    IS_VARLEN=kwargs["cu_seqlens"] is not None,
                    BV=16,
                    num_warps=warps,
                    num_stages=stages,
                )

            return launch

    monkeypatch.setattr(state, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", ForcedState())
    actual = chunk_gated_delta_rule_fwd(
        case["q"],
        case["k"],
        case["v"],
        case["g"],
        case["beta"],
        case["scale"],
        case["initial_state"].transpose(-1, -2).contiguous(),
        True,
    )
    assert calls == [2]
    for result, target in ((actual[1], expected[0]), (actual[3].transpose(-1, -2), expected[1])):
        assert torch.isfinite(result).all()
        ratio = (result.float() - target.float()).square().mean().sqrt() / (
            target.float().square().mean().sqrt() + 1e-8
        )
        assert ratio < 0.01, ratio.item()


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [(16, 16, 64), (8, 16, 80)])
@pytest.mark.parametrize("gates", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("config_index", [0, 1])
@torch.inference_mode()
def test_single_key_tile_fp32_gates(dtype, heads, gates, config_index, monkeypatch):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FP32 two-kernel gates require Hopper")
    import flag_attn.FLA.gated_delta_rule.chunk_fused_forward as fused
    from flag_attn.testing.recurrent_gdn import make_case

    h, hv, dv = heads
    case = make_case(4, 128, h, hv, 64, dv, dtype=dtype, strided=False)
    for name in ("q", "k", "v", "g", "beta"):
        x = case[name]
        case[name] = x.reshape(4, 128, *x.shape[2:])
    for name, fp32 in zip(("g", "beta"), gates):
        if not fp32:
            case[name] = case[name].to(dtype)
    case["cu_seqlens"] = case["ssm_state_indices"] = None
    case["use_qk_l2norm_in_kernel"] = False
    case["initial_state"] = case["initial_state"][:4].contiguous().zero_()
    for name in ("q", "k"):
        case[name] /= 64**0.5
    expected = reference(case)
    original = fused._chunk_gdn_two_kernel_fwd_kernel
    config = original.configs[config_index]
    calls = []

    class ForcedPipeline:
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append(kwargs["g"].dtype)
                assert kwargs["beta"].dtype == case["beta"].dtype
                return original.fn[grid(config.kwargs)](**kwargs, **config.all_kwargs())

            return launch

    monkeypatch.setattr(fused, "_chunk_gdn_two_kernel_fwd_kernel", ForcedPipeline())
    actual = chunk_gated_delta_rule_fwd(
        case["q"], case["k"], case["v"], case["g"], case["beta"], case["scale"], None, True
    )
    assert calls == [torch.float32]
    assert actual[0].dtype == case["g"].dtype
    prefix = case["g"].float().reshape(4, 2, 64, hv).cumsum(2).reshape(4, 128, hv)
    torch.testing.assert_close(actual[0], prefix.to(case["g"].dtype), rtol=1e-3, atol=1e-5)
    for result, target in ((actual[1], expected[0]), (actual[3].transpose(-1, -2), expected[1])):
        assert torch.isfinite(result).all()
        ratio = (result.float() - target.float()).square().mean().sqrt() / (
            target.float().square().mean().sqrt() + 1e-8
        )
        assert ratio < 0.01, ratio.item()
