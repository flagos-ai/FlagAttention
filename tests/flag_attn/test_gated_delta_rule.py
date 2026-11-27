# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""GDN contracts through the public API, for both generic and vendor dispatch."""

import os
import sys
from contextlib import contextmanager

import pytest
import torch

import flag_attn
from flag_attn import chunk_gated_delta_rule, chunk_gated_delta_rule_fwd
from flag_attn.testing import backend as test_backend
from gdn_utils import check_close, make_case, make_packed_case, reference, reference_packed, unpack_reference_case


def rms_close(actual, expected):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).square().mean().sqrt()
    assert error / (expected.float().square().mean().sqrt() + 1e-8) < 0.01


@contextmanager
def _record_calls():
    """Observe automatic dispatch without importing or selecting a backend."""
    calls = set()
    previous = sys.getprofile()

    def record(frame, event, arg):
        if event == "call":
            calls.add((frame.f_globals.get("__name__", ""), frame.f_code.co_name))

    sys.setprofile(record)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _chunk_reference(q, k, v, g, beta, scale, initial_state=None, cu_seqlens=None, normalize=False):
    """Independent FP32 token recurrence; state layout is [N, H, K, V]."""
    q, k, v, g, beta = [x.float() for x in (q, k, v, g, beta)]
    if normalize:
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
    B, T, Hg, K = q.shape
    H, V = v.shape[2:]
    q, k = [x.repeat_interleave(H // Hg, dim=2) for x in (q, k)]
    out = torch.empty_like(v)
    if cu_seqlens is None:
        state = torch.zeros(B, H, K, V, device=q.device) if initial_state is None else initial_state.float().clone()
        for t in range(T):
            state *= g[:, t].exp()[..., None, None]
            delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
            state += k[:, t, :, :, None] * delta[:, :, None, :]
            out[:, t] = (state * q[:, t, :, :, None]).sum(-2) * scale
        return out, state
    offsets = cu_seqlens.cpu().tolist()
    states = []
    for n, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        o, state = _chunk_reference(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            g[:, begin:end],
            beta[:, begin:end],
            scale,
            None if initial_state is None else initial_state[n : n + 1],
        )
        out[:, begin:end] = o
        states.append(state)
    return out, torch.cat(states)


def _chunk_inputs(shape, dtype, *, fp32_gates=False):
    B, T, Hg, H, K, V = shape
    torch.manual_seed(42)
    q, k = [torch.randn(B, T, Hg, K, device=test_backend.device, dtype=dtype) / K**0.5 for _ in range(2)]
    v = torch.randn(B, T, H, V, device=test_backend.device, dtype=dtype)
    gate_dtype = torch.float32 if fp32_gates else dtype
    g = -torch.rand(B, T, H, device=test_backend.device, dtype=gate_dtype) * 0.1
    beta = torch.rand_like(g).sigmoid()
    return q, k, v, g, beta


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.skipif(not test_backend.is_available(), reason="requires an accelerator")
class TestChunkGatedDeltaRule:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 193])
    @pytest.mark.parametrize("stateful", [False, True])
    @torch.inference_mode()
    def test_chunk_boundaries(self, dtype, length, stateful):
        q, k, v, g, beta = _chunk_inputs((2, length, 2, 4, 60, 80), dtype)
        state = torch.randn(2, 4, 60, 80, device=q.device, dtype=dtype) * 0.01 if stateful else None
        expected = _chunk_reference(q, k, v, g, beta, 0.125, state)
        actual = chunk_gated_delta_rule(
            q, k, v, beta, g, initial_state=state, head_first=False, scale=0.125, output_final_state=True
        )
        for a, e in zip(actual, expected):
            rms_close(a, e)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("dims", [(32, 64), (64, 64), (128, 64), (64, 128), (128, 128), (256, 256), (256, 512)])
    @pytest.mark.parametrize("fp32_gates", [False, True])
    @torch.inference_mode()
    def test_raw_chunk_contract(self, dtype, dims, fp32_gates):
        q, k, v, g, beta = _chunk_inputs((2, 129, 2, 4, *dims), dtype, fp32_gates=fp32_gates)
        initial = torch.randn(2, 4, *dims, device=q.device, dtype=torch.float32) * 0.01
        expected = _chunk_reference(q, k, v, g, beta, dims[0] ** -0.5, initial)
        result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, dims[0] ** -0.5, initial, True)
        assert len(result) == 7 and result[0].shape == g.shape and result[0].dtype == g.dtype
        assert result[2].shape == (*g.shape, 64)
        rms_close(result[1], expected[0])
        rms_close(result[3], expected[1])
        if os.environ.get("FLAG_ATTN_GDN_RECOMPUTE_SUPPRESS_LEVEL", "0") == "3":
            assert all(x is not None for x in result[4:])
        result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, dims[0] ** -0.5, initial, False)
        assert result[3] is None
        rms_close(result[1], expected[0])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("head_first", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    @pytest.mark.parametrize("packed", [False, True])
    @torch.inference_mode()
    def test_layout_normalization_and_varlen(self, dtype, head_first, normalize, packed):
        q, k, v, g, beta = _chunk_inputs((1, 193, 2, 6, 64, 80), dtype)
        cu = torch.tensor([0, 1, 64, 129, 193], device=q.device, dtype=torch.long) if packed else None
        initial = torch.randn(4 if packed else 1, 6, 64, 80, device=q.device, dtype=dtype) * 0.01
        original = initial.clone()
        expected = _chunk_reference(q, k, v, g, beta, 0.2, initial, cu, normalize)
        inputs = []
        for x in (q, k, v, beta, g):
            padded = torch.empty((*x.shape[:-1], x.shape[-1] * 2), device=x.device, dtype=x.dtype)
            padded[..., ::2] = x
            x = padded[..., ::2]
            inputs.append(x.transpose(1, 2) if head_first else x)
        o, state = chunk_gated_delta_rule(
            *inputs,
            head_first=head_first,
            scale=0.2,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=normalize,
        )
        rms_close(o.transpose(1, 2) if head_first else o, expected[0])
        rms_close(state, expected[1])
        assert torch.equal(initial, original)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape",
        [(2, 2048, 16, 16, 128, 128), (1, 2048, 8, 32, 128, 128), (2, 1024, 4, 8, 64, 64), (2, 2048, 4, 8, 256, 256)],
    )
    @torch.inference_mode()
    def test_prefill_and_automatic_dispatch(self, dtype, shape):
        q, k, v, g, beta = _chunk_inputs(shape, dtype)
        expected = _chunk_reference(q, k, v, g, beta, shape[-2] ** -0.5)
        with _record_calls() as calls:
            actual = chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, output_final_state=True)
        for a, e in zip(actual, expected):
            rms_close(a, e)
        generic = os.environ.get("FLAG_ATTN_FORCE_GENERIC", "0").lower() in {"1", "true", "yes", "on"}
        if generic:
            assert not any("runtime.backend" in module for module, name in calls if "gated_delta" in module)
        elif test_backend.is_nvidia() and torch.cuda.get_device_capability()[0] == 9:
            from flag_attn.utils import has_triton_tle

            if has_triton_tle(3, 6, 0):
                assert any("._nvidia.hopper.gated_delta_rule" in module for module, name in calls)
                if shape[-2:] == (128, 128) and os.environ.get("FLAG_ATTN_GDN_RECOMPUTE_SUPPRESS_LEVEL", "0") != "3":
                    assert any(name == "_tle_chunk_gated_delta_rule_fwd" for module, name in calls)
        raw = chunk_gated_delta_rule_fwd(q, k, v, g, beta, shape[-2] ** -0.5, None, True)
        rms_close(raw[1], expected[0])
        rms_close(raw[3], expected[1])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("packed", [False, True])
    @torch.inference_mode()
    def test_poisoned_allocations_and_empty_segments(self, dtype, packed, monkeypatch):
        q, k, v, g, beta = _chunk_inputs((1, 129, 2, 4, 64, 64), dtype)
        cu = torch.tensor([0, 0, 1, 64, 64, 129], device=q.device, dtype=torch.long) if packed else None
        state = torch.randn(5 if packed else 1, 4, 64, 64, device=q.device, dtype=dtype) * 0.01
        expected = _chunk_reference(q, k, v, g, beta, 0.125, state, cu)
        # Warm up autotuning before deliberately poisoning all scratch buffers.
        chunk_gated_delta_rule_fwd(q, k, v, g, beta, 0.125, state, True, cu)
        empty, empty_like, new_empty = torch.empty, torch.empty_like, torch.Tensor.new_empty

        def poison(x):
            return x.fill_(float("nan")) if x.is_floating_point() else x

        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty", lambda *a, **kw: poison(empty(*a, **kw)))
            patch.setattr(torch, "empty_like", lambda *a, **kw: poison(empty_like(*a, **kw)))
            patch.setattr(torch.Tensor, "new_empty", lambda self, *a, **kw: poison(new_empty(self, *a, **kw)))
            result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, 0.125, state, True, cu)
        rms_close(result[1], expected[0])
        rms_close(result[3], expected[1])
        assert torch.isfinite(result[2]).all()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @torch.inference_mode()
    def test_prefill_decode_handoff(self, dtype):
        q, k, v, g, beta = _chunk_inputs((2, 132, 2, 4, 64, 80), dtype)
        expected_o, expected_state = _chunk_reference(q, k, v, g, beta, 0.125)
        o, state = chunk_gated_delta_rule(
            q[:, :129],
            k[:, :129],
            v[:, :129],
            beta[:, :129],
            g[:, :129],
            head_first=False,
            output_final_state=True,
        )
        decoded, final = flag_attn.fused_recurrent_gated_delta_rule_fwd(
            q[:, 129:],
            k[:, 129:],
            v[:, 129:],
            g[:, 129:],
            beta[:, 129:],
            0.125,
            state.transpose(-1, -2).contiguous(),
        )
        rms_close(torch.cat((o, decoded), dim=1), expected_o)
        rms_close(final.transpose(-1, -2), expected_state)

    @pytest.mark.parametrize("case", ["BT", "heads", "gate", "state", "offset_dtype", "packed_batch"])
    def test_invalid_inputs(self, case):
        q, k, v, g, beta = _chunk_inputs((2, 32, 2, 4, 32, 64), torch.float16)
        kw = dict(head_first=False)
        if case == "BT":
            kw["BT"] = 32
        elif case == "heads":
            v = v[:, :, :3]
        elif case == "gate":
            beta = beta[..., :1]
        elif case == "state":
            kw["initial_state"] = torch.zeros(2, 4, 64, 32, device=q.device, dtype=q.dtype)
        else:
            kw["cu_seqlens"] = torch.tensor(
                [0, 32, 64], device=q.device, dtype=torch.int32 if case == "offset_dtype" else torch.int64
            )
        with pytest.raises(ValueError):
            chunk_gated_delta_rule(q, k, v, beta, g, **kw)


@pytest.mark.skipif(not test_backend.is_available(), reason="requires an accelerator")
class TestRecurrentGatedDeltaRule:
    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("h,hv", [(2, 4), (4, 12), (2, 8)])
    @pytest.mark.parametrize("length", [1, 3, 8])
    @pytest.mark.parametrize("strided", [False, True])
    @pytest.mark.parametrize("seed", [0, 17])
    def test_recurrent_stateful(self, h, hv, length, strided, seed):
        case = make_case(4, length, h, hv, strided=strided, spec=length > 1, seed=seed)
        expected = reference(case)
        actual = flag_attn.fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(*actual, *expected)
        assert actual[1].data_ptr() == case["initial_state"].data_ptr()

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("kd,vd", [(1, 1), (60, 80), (64, 128), (128, 64), (256, 128), (128, 256)])
    @pytest.mark.parametrize("vector_beta", [False, True])
    def test_recurrent_tails_and_vector_beta(self, kd, vd, vector_beta):
        case = make_case(3, 3, 2, 6, kdim=kd, vdim=vd, vector_beta=vector_beta, spec=True)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize(
        "dtype,state_dtype",
        [(torch.float16, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32)],
    )
    def test_recurrent_dtypes(self, dtype, state_dtype):
        case = make_case(3, 4, 2, 6, dtype=dtype, state_dtype=state_dtype, spec=True)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("inplace", [False, True])
    def test_recurrent_dense_without_indices(self, inplace):
        n, length = 3, 4
        case = make_case(n, length, 2, 4)
        for name in ("q", "k", "v", "g", "beta"):
            x = case[name]
            case[name] = x.reshape(n, length, *x.shape[2:])
        case.update(cu_seqlens=None, ssm_state_indices=None, inplace_final_state=inplace)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("inplace", [False, True])
    def test_recurrent_ragged_and_empty(self, inplace):
        case = make_case(5, 8, 2, 6, lengths=[0, 1, 8, 3, 0], spec=True)
        case["inplace_final_state"] = inplace
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_padding_and_index_strides(self):
        case = make_case(4, 4, 2, 6, padding=True, spec=True)
        case["ssm_state_indices"][-2] = -1
        idx = case["ssm_state_indices"]
        wide = torch.full((4, 8), -99, device=test_backend.device, dtype=idx.dtype)
        wide[:, ::2] = idx
        case["ssm_state_indices"] = wide[:, ::2]
        original = case["initial_state"].clone()
        expected = reference(case)
        out, st = flag_attn.fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(out, st, *expected)
        assert torch.equal(st[0], original[0])
        assert torch.count_nonzero(out[:, -4:]) == 0

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_continuation_and_repeatability(self):
        case = make_case(3, 8, 2, 6)
        original = case["initial_state"].clone()
        expected = reference(case)
        full = flag_attn.fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(*full, *expected)
        full_o, full_s = full[0].clone(), full[1].clone()
        case["initial_state"].copy_(original)
        repeated = flag_attn.fused_recurrent_gated_delta_rule_fwd(**case)
        assert torch.equal(repeated[0], full_o) and torch.equal(repeated[1], full_s)
        # Feed one token from each sequence per call, writing one stable slot per request.
        case["initial_state"].copy_(original)
        ids = case["ssm_state_indices"]
        initial = original[ids[:, 0].long()].clone()
        stable = torch.arange(1, 4, device=test_backend.device, dtype=torch.int32)
        pool = torch.zeros_like(original)
        pool[stable.long()] = initial
        outputs = []
        for t in range(8):
            step = {**case, "initial_state": pool, "cu_seqlens": None, "ssm_state_indices": stable}
            for name in ("q", "k", "v", "g", "beta"):
                x = case[name].reshape(3, 8, *case[name].shape[2:])
                step[name] = x[:, t : t + 1]
            outputs.append(flag_attn.fused_recurrent_gated_delta_rule_fwd(**step)[0])
        stacked = torch.cat(outputs, dim=1).reshape_as(full_o)
        torch.testing.assert_close(stacked, full_o, rtol=1.0e-2, atol=2.0e-3)
        torch.testing.assert_close(pool[stable.long()], full_s[ids[:, -1].long()], rtol=2.0e-4, atol=2.0e-5)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_zero_decay_beta_limits(self):
        case = make_case(3, 4, 2, 6)
        case["g"].zero_()
        case["beta"][..., 0] = 0
        case["beta"][..., 1] = 1
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("strided_state", [False, True])
    def test_recurrent_gate_and_state_strides(self, strided_state):
        case = make_case(3, 3, 2, 6, kdim=60, vdim=80)
        for name in ("g", "beta"):
            x = case[name]
            wide = torch.empty((*x.shape[:-1], x.shape[-1] * 2), device=test_backend.device, dtype=x.dtype)
            wide[..., ::2] = x
            case[name] = wide[..., ::2]
        if strided_state:
            state = case["initial_state"]
            backing = torch.empty((*state.shape[:2], 60, 80), device=test_backend.device, dtype=state.dtype)
            case["initial_state"] = backing.transpose(-1, -2)
            case["initial_state"].copy_(state)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_public_export(self):
        case = make_case(2, 1, 2, 4)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_padded_projection_buffer(self):
        case = make_case(3, 4, 2, 6, spec=True, projection_padding=64)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_padding_snapshots_are_initialized(self, monkeypatch):
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
            out, final = flag_attn.fused_recurrent_gated_delta_rule_fwd(**case)
        check_close(out, final, *expected)
        assert torch.count_nonzero(final[-4:]) == 0

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_invalid_write_slot_keeps_cache(self):
        case = make_case(3, 4, 2, 6, spec=True)
        case["num_accepted_tokens"].fill_(1)
        case["ssm_state_indices"][0, -1] = 0
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("values", ["zero", "impulse"])
    def test_recurrent_longer_verification_values(self, values):
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
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    def test_recurrent_pre_normalized_inputs(self):
        case = make_case(4, 8, 2, 6, spec=True)
        for name in ("q", "k"):
            x = case[name]
            case[name] = torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype)
        case["use_qk_l2norm_in_kernel"] = False
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("lengths", [[0, 0, 0], [0, 1, 0]])
    def test_recurrent_empty_batch_segments(self, lengths):
        case = make_case(3, 1, 2, 4, lengths=lengths)
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("varlen", [False, True])
    def test_recurrent_single_read_index_without_inplace_writes(self, varlen):
        case = make_case(2, 4, 2, 4)
        case["inplace_final_state"] = False
        case["ssm_state_indices"] = case["ssm_state_indices"][:, :1]
        if not varlen:
            for name in ("q", "k", "v", "g", "beta"):
                x = case[name]
                case[name] = x.reshape(2, 4, *x.shape[2:])
            case["cu_seqlens"] = None
        initial = case["initial_state"].clone()
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)
        torch.testing.assert_close(case["initial_state"], initial, rtol=0, atol=0)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("invalid", ["v_rank", "g_dtype", "beta_dtype", "empty_indices", "short_indices"])
    def test_recurrent_invalid_metadata(self, invalid):
        case = make_case(2, 4, 2, 4)
        if invalid == "v_rank":
            case["v"] = case["v"][..., 0]
        elif invalid == "g_dtype":
            case["g"] = case["g"].to(torch.int32)
        elif invalid == "beta_dtype":
            case["beta"] = case["beta"].to(torch.int32)
        elif invalid == "empty_indices":
            case["ssm_state_indices"] = case["ssm_state_indices"][:, :0]
        else:
            for name in ("q", "k", "v", "g", "beta"):
                x = case[name]
                case[name] = x.reshape(2, 4, *x.shape[2:])
            case["cu_seqlens"] = None
            case["ssm_state_indices"] = case["ssm_state_indices"][:, :2]
        with pytest.raises(ValueError):
            flag_attn.fused_recurrent_gated_delta_rule_fwd(**case)

    @pytest.mark.fused_recurrent_gated_delta_rule_fwd
    @pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("vector_beta", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    @torch.inference_mode()
    def test_recurrent_large_snapshot_layout(self, state_dtype, vector_beta, normalize, monkeypatch):
        case = make_case(32, 4, 4, 16, state_dtype=state_dtype, spec=True, vector_beta=vector_beta)
        if not normalize:
            for name in ("q", "k"):
                case[name] = torch.nn.functional.normalize(case[name].float(), dim=-1).to(case[name].dtype)
            case["use_qk_l2norm_in_kernel"] = False
        expected = reference(case)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**case), *expected)


@pytest.mark.skipif(not test_backend.is_available(), reason="requires an accelerator")
class TestPackedGatedDeltaRule:
    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("dims", [(1, 2, 128, 128), (2, 6, 60, 80), (2, 8, 127, 129), (1, 1, 256, 64)])
    @pytest.mark.parametrize("seed", [0, 17])
    @torch.inference_mode()
    def test_packed_reference(self, dtype, dims, seed, monkeypatch):
        c = make_packed_case(n=3, h=dims[0], hv=dims[1], kdim=dims[2], vdim=dims[3], dtype=dtype, seed=seed)
        expected = reference_packed(c)
        state0 = c["initial_state"].clone()
        previous = None
        for _ in range(3):
            c["initial_state"].copy_(state0)
            c["out"].fill_(float("nan"))
            result = flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c)
            assert result[0] is c["out"] and result[1] is c["initial_state"]
            check_close(*result, *expected)
            if previous is not None:
                assert torch.equal(result[0], previous[0]) and torch.equal(result[1], previous[1])
            previous = [x.clone() for x in result]

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @torch.inference_mode()
    def test_packed_beta_rounding_in_fp32_cache(self, dtype):
        c = make_packed_case(n=1, h=1, hv=1, kdim=1, vdim=1, dtype=dtype)
        c["mixed_qkv"].fill_(1)
        c["initial_state"].zero_()
        c["b"].fill_(0.1)
        c["scale"] = 1.0
        c["use_qk_l2norm_in_kernel"] = False
        expected = c["b"].float().sigmoid().to(dtype).float()
        _, state = flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c)
        slot = c["ssm_state_indices"].long()
        torch.testing.assert_close(state[slot, 0, 0, 0], expected[:, 0], rtol=2e-4, atol=2e-5)
        untouched = state.clone()
        untouched[slot] = 0
        assert torch.count_nonzero(untouched) == 0

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
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
    def test_packed_boundaries(self, case):
        c = make_packed_case(
            n=4,
            h=2,
            hv=6,
            projection_padding=64,
            state_dtype=(
                torch.bfloat16 if case == "bf16_state" else torch.float16 if case == "fp16_state" else torch.float32
            ),
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
            c["a"][:] = torch.tensor([-20, -1, 0, 19.875, 20, 20.125], device=test_backend.device)
            c["dt_bias"].zero_()
        if case == "gate_extremes":
            c["a"][:] = torch.tensor([-80, -20, 0, 20, 80, 100], device=test_backend.device)
            c["b"][:] = torch.tensor([-80, -20, 0, 20, 80, 100], device=test_backend.device)
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
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @torch.inference_mode()
    def test_packed_continuation_matches_recurrent(self):
        c = make_packed_case()
        separate = c["initial_state"].clone()
        for seed in range(4):
            torch.manual_seed(seed)
            c["mixed_qkv"].normal_()
            expected_case = unpack_reference_case({**c, "initial_state": separate})
            expected = flag_attn.fused_recurrent_gated_delta_rule_fwd(**expected_case)
            check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @torch.inference_mode()
    def test_packed_split_grid(self):
        n, hv = 21846, 3
        c = dict(
            mixed_qkv=torch.zeros((n, 2 + hv), device=test_backend.device, dtype=torch.bfloat16),
            a=torch.zeros((n, hv), device=test_backend.device),
            b=torch.zeros((n, hv), device=test_backend.device),
            A_log=torch.zeros(hv, device=test_backend.device),
            dt_bias=torch.zeros(hv, device=test_backend.device),
            scale=1.0,
            initial_state=torch.zeros((n + 1, hv, 1, 1), device=test_backend.device),
            out=torch.full((n, 1, hv, 1), float("nan"), device=test_backend.device, dtype=torch.bfloat16),
            ssm_state_indices=torch.arange(1, n + 1, device=test_backend.device, dtype=torch.int64),
        )
        out, state = flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c)
        assert torch.count_nonzero(out) == 0 and torch.count_nonzero(state) == 0

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @torch.inference_mode()
    def test_public_exports_and_untouched_padding(self):
        assert callable(flag_attn.chunk_gated_delta_rule_fwd)
        assert callable(flag_attn.fused_recurrent_gated_delta_rule_fwd)
        c = make_packed_case(n=3)
        c["ssm_state_indices"][0] = 0
        c["ssm_state_indices"][2] = -1
        original = c["initial_state"].clone()
        expected = reference_packed(c)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)
        untouched = torch.ones(original.shape[0], dtype=torch.bool, device=test_backend.device)
        untouched[c["ssm_state_indices"][1].long()] = False
        assert torch.equal(c["initial_state"][untouched], original[untouched])

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @pytest.mark.parametrize("axis", ["projection", "state_head", "state_value", "state_key"])
    @torch.inference_mode()
    def test_large_relative_offsets(self, axis):
        if not test_backend.is_nvidia() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
            pytest.skip("large-offset addressing check requires 12 GiB free")
        c = make_packed_case(n=3, h=1, hv=3, kdim=3, vdim=3)
        stride = 2**30 + 64
        if axis == "projection":
            width = c["mixed_qkv"].shape[1]
            storage = torch.empty(2 * stride + width, device=test_backend.device, dtype=c["mixed_qkv"].dtype)
            view = storage.as_strided((3, width), (stride, 1))
            view.copy_(c["mixed_qkv"])
            c["mixed_qkv"] = view
        else:
            state = c["initial_state"]
            strides = [64, 9, 3, 1]
            strides[{"state_head": 1, "state_value": 2, "state_key": 3}[axis]] = stride
            size = 1 + sum((d - 1) * s for d, s in zip(state.shape, strides))
            storage = torch.empty(size, device=test_backend.device, dtype=state.dtype)
            view = storage.as_strided(state.shape, strides)
            view.copy_(state)
            c["initial_state"] = view
        expected = reference_packed(c)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)
        assert torch.count_nonzero(c["out"]) > 0

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("dims", [(128, 128), (256, 64), (64, 256)])
    @torch.inference_mode()
    def test_low_precision_cache_large_parallelism(self, state_dtype, dims):
        from gdn_utils import make_case, reference

        c = make_packed_case(n=64, h=4, hv=8, kdim=dims[0], vdim=dims[1], state_dtype=state_dtype)
        expected = reference_packed(c)
        flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c)
        check_close(c["out"], c["initial_state"], *expected)
        recurrent = make_case(64, 3, 4, 8, dims[0], dims[1], state_dtype=state_dtype, spec=True)
        expected = reference(recurrent)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_fwd(**recurrent), *expected)

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @pytest.mark.parametrize("case", ["width", "out", "indices", "gate", "parameter", "grad"])
    def test_packed_invalid(self, case):
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
            flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c)

    @pytest.mark.fused_recurrent_gated_delta_rule_packed_decode
    @pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
    @torch.inference_mode()
    def test_packed_large_cache_reuses_read_slot(self, state_dtype, monkeypatch):
        c = make_packed_case(n=192, h=4, hv=16, state_dtype=state_dtype)
        c["ssm_state_indices"][0] = 0
        c["ssm_state_indices"][-1] = -1
        expected = reference_packed(c)
        check_close(*flag_attn.fused_recurrent_gated_delta_rule_packed_decode(**c), *expected)


@pytest.mark.chunk_gated_delta_rule
def test_portable_import_without_tle():
    import subprocess

    script = """
import importlib.abc
import os
import sys
os.environ['FLAG_ATTN_FORCE_GENERIC'] = '1'
class NoTLE(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('triton.experimental.tle'):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, NoTLE())
import flag_attn
for name in ('chunk_gated_delta_rule', 'chunk_gated_delta_rule_fwd',
             'fused_recurrent_gated_delta_rule_fwd',
             'fused_recurrent_gated_delta_rule_packed_decode'):
    fn = getattr(flag_attn, name)
    assert fn.__module__.startswith('flag_attn.FLA.gated_delta_rule'), fn.__module__
assert not any('triton.experimental.tle' in name for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.skipif(not test_backend.is_available(), reason="requires an accelerator")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8192, 96, 96, 128, 128),
        (2, 16384, 16, 16, 128, 128),
        (4, 2048, 16, 16, 128, 128),
        (4, 4096, 64, 64, 128, 128),
        (8, 1024, 8, 8, 64, 64),
        (8, 2048, 32, 32, 256, 256),
    ],
)
@torch.inference_mode()
def test_long_prefill_reference(dtype, shape):
    q, k, v, g, beta = _chunk_inputs(shape, dtype)
    expected_o, expected_state = _chunk_reference(q, k, v, g, beta, shape[-2] ** -0.5)
    o, state = chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, output_final_state=True)
    rms_close(o, expected_o)
    rms_close(state, expected_state)
    raw = chunk_gated_delta_rule_fwd(q, k, v, g, beta, shape[-2] ** -0.5, None, True)
    rms_close(raw[1], expected_o)
    rms_close(raw[3], expected_state)


@pytest.mark.chunk_gated_delta_rule
@pytest.mark.skipif(not test_backend.is_available(), reason="requires an accelerator")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_raw_auxiliary_values(dtype, packed):
    q, k, v, g, beta = _chunk_inputs((1, 65, 2, 4, 64, 32), dtype, fp32_gates=True)
    cu = torch.tensor([0, 1, 33, 65], device=q.device, dtype=torch.long) if packed else None
    result = chunk_gated_delta_rule_fwd(q, k, v, g, beta, 0.125, None, True, cu)
    keys = k.float().repeat_interleave(2, dim=2)
    segments = [(0, 1), (1, 33), (33, 65)] if packed else [(0, 64), (64, 65)]
    for begin, end in segments:
        cumulative = g[:, begin:end].cumsum(dim=1)
        torch.testing.assert_close(result[0][:, begin:end], cumulative, rtol=1e-5, atol=1e-6)
        # Small CPU FP64 matrices avoid backend BLAS stride restrictions for
        # single-token tails and provide an independent inverse reference.
        block = keys[0, begin:end].transpose(0, 1).cpu().double()
        decay = cumulative[0].transpose(0, 1).cpu().double()
        gram = block @ block.transpose(-1, -2)
        gram *= beta[0, begin:end].transpose(0, 1).cpu().double()[..., None]
        gram *= (decay[..., :, None] - decay[..., None, :]).exp()
        identity = torch.eye(end - begin, dtype=torch.float64).expand(4, -1, -1)
        inverse = torch.linalg.solve_triangular(identity + gram.tril(-1), identity, upper=False)
        actual = result[2][0, begin:end, :, : end - begin].transpose(0, 1)
        rms_close(actual, inverse.to(q.device))
        assert torch.count_nonzero(result[2][0, begin:end, :, end - begin :]) == 0
