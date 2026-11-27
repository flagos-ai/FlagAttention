# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""KDA contracts through public APIs, independent of vendor implementation imports."""

import inspect
import os
import subprocess
import sys
from contextlib import contextmanager

import pytest
import torch

import flag_attn
from flag_attn.testing import backend as test_backend
from flag_attn.utils import has_triton_tle
from kda_utils import LONG_CASES, assert_close, make_case, reference

pytestmark = [
    pytest.mark.KDA,
    pytest.mark.skipif(not test_backend.is_available(), reason="requires an available accelerator"),
    pytest.mark.skipif(
        test_backend.has_specialization("chunk_kda_fwd_infer")
        and not test_backend.is_nvidia()
        and os.environ.get("FLAG_ATTN_FORCE_GENERIC", "0").lower() not in {"1", "true", "yes", "on"},
        reason="existing vendor input contracts differ; set FLAG_ATTN_FORCE_GENERIC=1 to validate the portable contract",
    ),
]
APIS = ["chunk_kda", "chunk_kda_fwd_infer"]


def _force_generic():
    return os.environ.get("FLAG_ATTN_FORCE_GENERIC", "0").lower() in {"1", "true", "yes", "on"}


@contextmanager
def _record_calls():
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


def test_public_signatures():
    names = [
        "q",
        "k",
        "v",
        "g",
        "beta",
        "scale",
        "initial_state",
        "output_final_state",
        "state_v_first",
        "cu_seqlens",
        "chunk_indices",
        "chunk_size",
        "safe_gate",
        "lower_bound",
        "A_log",
        "dt_bias",
    ]
    for name in APIS:
        parameters = inspect.signature(getattr(flag_attn, name)).parameters
        assert list(parameters)[:16] == names
        if test_backend.is_nvidia() or _force_generic():
            assert len(parameters) == 16
        assert parameters["chunk_size"].default == 16
        assert parameters["output_final_state"].default is False


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65])
@torch.inference_mode()
def test_boundaries(api, length):
    args, kwargs = make_case(T=length, initial=bool(length % 2))
    assert_close(getattr(flag_attn, api)(*args, **kwargs), reference(args, kwargs))


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,state_v_first",
    [
        ((2, 33, 2, 4, 30, 48), False),
        ((1, 65, 2, 6, 64, 80), True),
        ((1, 17, 2, 2, 192, 256), False),
        ((2, 7, 2, 4, 256, 129), True),
    ],
)
@torch.inference_mode()
def test_grouped_asymmetric_noncontiguous(api, dtype, shape, state_v_first):
    B, T, H, HV, K, V = shape
    args, kwargs = make_case(
        B=B, T=T, H=H, HV=HV, K=K, V=V, dtype=dtype, state_v_first=state_v_first, state_dtype=dtype, noncontiguous=True
    )
    original = [tensor.clone() for tensor in (*args, kwargs["initial_state"])]
    actual = getattr(flag_attn, api)(*args, **kwargs)
    assert_close(actual, reference(args, kwargs))
    for tensor, copy in zip((*args, kwargs["initial_state"]), original):
        assert torch.equal(tensor, copy)


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("lengths", [[0, 1, 0, 16, 17, 0], [0, 0], [1, 0, 65, 2]])
@torch.inference_mode()
def test_varlen_empty_segments_and_indices(api, cu_dtype, lengths):
    args, kwargs = make_case(lengths=lengths, cu_dtype=cu_dtype)
    indices = [[n, i] for n, length in enumerate(lengths) for i in range((length + 15) // 16)]
    kwargs["chunk_indices"] = torch.tensor(indices, device=args[0].device, dtype=cu_dtype).reshape(-1, 2)
    assert_close(getattr(flag_attn, api)(*args, **kwargs), reference(args, kwargs))


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("state_v_first", [False, True])
@torch.inference_mode()
def test_softplus_gate_and_optional_bias(api, bias, state_v_first):
    args, kwargs = make_case(
        K=64, V=32, HV=4, dtype=torch.float16, safe_gate=False, bias=bias, state_v_first=state_v_first, chunk_size=32
    )
    kwargs["scale"] = None
    assert_close(getattr(flag_attn, api)(*args, **kwargs), reference(args, kwargs))


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_automatic_dispatch(api, packed):
    args, kwargs = make_case(T=80, B=2, lengths=[13, 19, 16] if packed else None)
    with _record_calls() as calls:
        actual = getattr(flag_attn, api)(*args, **kwargs)
    assert_close(actual, reference(args, kwargs))
    if _force_generic():
        assert ("flag_attn.FLA.chunk_kda", "chunk_kda_fwd_infer") in calls
        assert not any("runtime.backend" in module for module, name in calls if name == "chunk_kda_fwd_infer")
    elif test_backend.is_nvidia():
        required = ("gpu.alloc", "gpu.local_ptr", "gpu.copy", "gpu.warp_specialize", "pipe")
        eligible = torch.cuda.get_device_capability(args[0].device)[0] == 9 and has_triton_tle(
            3, 6, 0, required_api=required
        )
        if eligible:
            from triton._C.libtriton import ir

            eligible = hasattr(ir.builder, "make_swizzled_shared_encoding_attr")
        module = "flag_attn.runtime.backend._nvidia.hopper.kda" if eligible else "flag_attn.FLA.chunk_kda"
        assert (module, "chunk_kda_fwd_infer") in calls


@pytest.mark.parametrize("api", APIS)
@torch.inference_mode()
def test_prefill_resume(api):
    args, kwargs = make_case(T=65)
    expected = reference(args, kwargs)
    op = getattr(flag_attn, api)
    prefix, state = op(*(x[:, :31] for x in args), **kwargs)
    tail, final = op(*(x[:, 31:] for x in args), **{**kwargs, "initial_state": state})
    assert_close((torch.cat((prefix, tail), dim=1), final), expected)


@pytest.mark.parametrize("case", list(LONG_CASES))
@torch.inference_mode()
def test_production_lengths(case):
    lengths = LONG_CASES[case]
    args, kwargs = make_case(H=96, T=sum(lengths), lengths=lengths if len(lengths) > 1 else None)
    actual = flag_attn.chunk_kda(*args, **kwargs)
    assert_close(actual, reference(args, kwargs))


def test_grad_mode_rejected():
    args, kwargs = make_case(K=64, V=64, T=1)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference"):
        flag_attn.chunk_kda(*args, **kwargs)


@pytest.mark.parametrize("force_generic", ["0", "1"])
def test_import_and_dispatch_without_tle(force_generic):
    if force_generic == "0" and test_backend.has_specialization("chunk_kda_fwd_infer") and not test_backend.is_nvidia():
        pytest.skip("automatic missing-TLE fallback is provided by the NVIDIA dispatcher")
    program = r"""
import importlib.abc
import sys
class NoTLE(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('triton.experimental.tle'):
            raise ModuleNotFoundError('TLE intentionally unavailable', name=fullname)
blocker = NoTLE()
sys.meta_path.insert(0, blocker)
import torch
import flag_attn
from flag_attn.testing import backend
from triton.runtime.jit import JITFunction
launches = []
original_run = JITFunction.run
def record_run(self, *args, **kwargs):
    launches.append((self.fn.__module__, self.fn.__name__))
JITFunction.run = record_run
q = torch.zeros(1, 1, 1, 4, device=backend.device, dtype=torch.bfloat16)
v = torch.ones(1, 1, 1, 2, device=backend.device, dtype=torch.bfloat16)
beta = torch.zeros(1, 1, 1, device=backend.device, dtype=torch.bfloat16)
A = torch.zeros(1, device=backend.device)
with torch.inference_mode():
    for op in (flag_attn.chunk_kda, flag_attn.chunk_kda_fwd_infer):
        o, state = op(q, q, v, q, beta, A_log=A, output_final_state=True)
        assert o.shape == v.shape and state.shape == (1, 1, 4, 2)
assert launches == [('flag_attn.FLA.chunk_kda', '_kda_recurrent_kernel')] * 2
assert not any(name.startswith('triton.experimental.tle') for name in sys.modules)
# FlagTree's compiler itself imports its TLE dialect, including for plain
# Triton. Allow compiler internals after checking operator import/dispatch.
JITFunction.run = original_run
sys.meta_path.remove(blocker)
with torch.inference_mode():
    for op in (flag_attn.chunk_kda, flag_attn.chunk_kda_fwd_infer):
        o, state = op(q, q, v, q, beta, A_log=A, output_final_state=True)
        assert torch.count_nonzero(o) == 0 and torch.count_nonzero(state) == 0
"""
    env = {**os.environ, "FLAG_ATTN_FORCE_GENERIC": force_generic}
    result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
