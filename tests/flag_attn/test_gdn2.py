
import importlib
import inspect
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from flag_attn.testing import backend as test_backend
from flag_attn import chunk_gdn2
from flag_attn.gdn2.native.chunk_fwd import chunk_gdn2_fwd
from flag_attn.gdn2.native.output import chunk_gla_fwd_kernel_o

ASSERT_RATIO = 0.01

# H20 cold-cache history puts these main cases at about 19 minutes. Different
# (H, K, V, dtype) combinations can each trigger several minutes of compilation;
# the 4096-token case also warms the output-config tests below.
GDN2_TEST_CASES = [
    # ((2, 512, 8, 64, 64), (torch.float16, torch.bfloat16)),
    # ((4, 1024, 8, 64, 64), (torch.float16, torch.bfloat16)),
    # ((1, 2048, 8, 64, 64), (torch.float16, torch.bfloat16)),
    ((1, 4096, 16, 64, 64), (torch.float16, torch.bfloat16)),
    ((1, 8192, 96, 128, 128), (torch.float16, torch.bfloat16)),
    ((2, 2048, 16, 256, 512), (torch.float16,)),
    # ((2, 2048, 16, 256, 512), (torch.bfloat16,)),
    # ((2, 16384, 16, 128, 128), (torch.float16, torch.bfloat16)),
    # ((4, 1024, 8, 256, 512), (torch.float16, torch.bfloat16)),
    # ((4, 2048, 16, 128, 128), (torch.float16, torch.bfloat16)),
    # ((4, 4096, 64, 128, 128), (torch.float16, torch.bfloat16)),
    # ((8, 1024, 8, 64, 64), (torch.float16, torch.bfloat16)),
    # ((8, 2048, 32, 256, 256), (torch.float16, torch.bfloat16)),
]


def _cuda_available() -> bool:
    return test_backend.supports_operator("chunk_gdn2")


pytestmark = [
    pytest.mark.skipif(not _cuda_available(), reason="chunk_gdn2 tests require CUDA"),
]


@pytest.mark.chunk_gdn2
@pytest.mark.skipif(
    not test_backend.supports_operator("chunk_gdn2"),
    reason=test_backend.skip_reason("chunk_gdn2"),
)
def test_chunk_gdn2_benchmark(record_property) -> None:
    from benchmark.test_gdn2_benchmark import run_benchmark

    run_benchmark([], record_property=record_property)


def _set_public_gdn2_native_for_call(force_native: bool):
    module = importlib.import_module("flag_attn.gdn2.chunk")
    old = module.HAS_TLE_GDN2
    if force_native:
        module.HAS_TLE_GDN2 = False
    return module, old


@pytest.mark.chunk_gdn2
def test_gdn2_public_registry_and_signature():
    from flag_attn.gdn2 import chunk_gdn2 as facade
    from flag_attn.runtime.backend import resolve_operator

    generic = importlib.import_module("flag_attn.gdn2.chunk")
    selected = resolve_operator("chunk_gdn2", "flag_attn.gdn2.chunk", vendor="nvidia")
    assert selected.__module__ == "flag_attn.gdn2.chunk"
    assert inspect.signature(selected) == inspect.signature(generic.chunk_gdn2)
    if test_backend.is_nvidia():
        assert chunk_gdn2 is facade is selected
    for vendor in ("cpu", "amd"):
        assert resolve_operator("chunk_gdn2", "flag_attn.gdn2.chunk", vendor=vendor) is generic.chunk_gdn2


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("vendor", ["cpu", "amd"])
def test_gdn2_non_nvidia_lazy_import(vendor):
    code = (
        "import sys; import flag_attn; import flag_attn.gdn2; "
        "assert 'flag_attn.gdn2.chunk' not in sys.modules; "
        "assert 'triton.experimental.tle.language' not in sys.modules"
    )
    env = dict(os.environ, FLAG_ATTN_BACKEND=vendor, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _native_gdn2_reference(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    scale,
    initial_state,
    output_final_state,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_size=16,
):
    (
        o,
        final_state,
        _g,
        _Aqk,
        _Akk,
        _w_wy,
        _u_wy,
        _qg,
        _kg,
        _v_new,
        _h,
        _initial_state,
    ) = chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        state_v_first=state_v_first,
    )
    return o, final_state


def _make_inputs(*, B, T, H, K, V, dtype, state_v_first):
    device = test_backend.device
    scale = K**-0.5

    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)
    initial_state = None
    g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.1).to(dtype)

    A_log = None
    dt_bias = None
    lower_bound = None

    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_gate_in_kernel": False,
        "safe_gate": False,
        "lower_bound": lower_bound,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_v_first": state_v_first,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
        "chunk_size": 64,
    }
    return (q, k, v, g, b, w), kwargs


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    err = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    base = expected.float().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs().max().item()
    ratio = _err_ratio(expected, actual)
    if abs_err <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    assert ratio < ASSERT_RATIO, (
        f"{name} diff: abs={abs_err:.6f} ratio={ratio:.6f} " f"limit={ASSERT_RATIO}"
    )


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "impl",
    [pytest.param("tle", id="tle"), pytest.param("native", id="native")],
)
@pytest.mark.parametrize(
    "shape,dtype",
    [(shape, dtype) for shape, dtypes in GDN2_TEST_CASES for dtype in dtypes],
)
@torch.inference_mode()
def test_chunk_gdn2_matches_native_triton(impl, dtype, shape):
    module = importlib.import_module(chunk_gdn2.__module__)
    if impl == "tle" and not module.HAS_TLE_GDN2:
        pytest.skip("TLE GDN2 path is unavailable in this environment")

    torch.manual_seed(42)
    B, T, H, K, V = shape
    args, kwargs = _make_inputs(
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        dtype=dtype,
        state_v_first=False,
    )

    expected, expected_final = _native_gdn2_reference(*args, **kwargs)

    force_native = impl == "native"
    actual_kwargs = dict(kwargs)
    if impl == "tle":
        # Public TLE inference currently requires BT=16. The baseline remains
        # the original native Triton path with BT=64.
        actual_kwargs["chunk_size"] = 16

    module, old = _set_public_gdn2_native_for_call(force_native)
    try:
        # Native coverage must bypass the vendor override explicitly. Changing
        # generic.HAS_TLE_GDN2 cannot control a registered NVIDIA kernel.
        implementation = module.chunk_gdn2 if impl == "tle" else importlib.import_module("flag_attn.gdn2.chunk").chunk_gdn2
        actual, actual_final = implementation(*args, **actual_kwargs)
    finally:
        module.HAS_TLE_GDN2 = old

    _assert_close("o", actual, expected)
    _assert_close("ht", actual_final, expected_final)


@pytest.mark.chunk_gdn2
def test_k1_tle_resource_controls_are_enabled():
    module = importlib.import_module(chunk_gdn2.__module__)
    if not module.HAS_TLE_GDN2:
        pytest.skip("TLE GDN2 path is unavailable in this environment")

    assert module.K1_TLE_MAXNREG_CANDIDATES == (64, 72, 96)
    assert module.K1_TLE_SMEM_REUSE_MIN_K == 256
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "tle.gpu.alloc" in source
    assert "tle.gpu.local_ptr" in source


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("intermediate", [False, True])
def test_nvidia_gdn2_missing_tle_uses_native(monkeypatch, intermediate):
    module = importlib.import_module("flag_attn.gdn2.chunk")
    native = importlib.import_module("flag_attn.gdn2.native.chunk_fwd")
    monkeypatch.setattr(module, "HAS_TLE_GDN2", False)
    expected = tuple(object() for _ in range(12))
    captured = {}

    def native_forward(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(module, "chunk_gdn2_fwd", native_forward)
    q = torch.empty(1, 16, 1, 64)
    state = object()
    result = module.chunk_gdn2(
        q, q, q, q, q, q, initial_state=state, output_final_state=True,
        state_v_first=True, return_intermediate_states=intermediate, chunk_size=64,
    )
    assert result == ((expected[0], expected[1], expected[10]) if intermediate else expected[:2])
    assert captured["initial_state"] is state
    assert captured["state_v_first"] is True
    assert captured["chunk_size"] == 64
    assert captured["scale"] == 64**-0.5


@pytest.mark.chunk_gdn2
def test_nvidia_gdn2_tle_errors_propagate(monkeypatch):
    module = importlib.import_module("flag_attn.gdn2.chunk")
    monkeypatch.setattr(module, "HAS_TLE_GDN2", True)

    def failure(**kwargs):
        raise RuntimeError("TLE compilation failed")

    monkeypatch.setattr(module, "chunk_gdn2_fwd_infer", failure, raising=False)
    with pytest.raises(RuntimeError, match="TLE compilation failed"):
        module.chunk_gdn2(None, None, None, None, None, None)


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("final_state,intermediate", [(False, False), (True, True)])
@torch.inference_mode()
def test_nvidia_gdn2_native_fallback_matches_generic(monkeypatch, dtype, final_state, intermediate):
    module = importlib.import_module("flag_attn.gdn2.chunk")
    generic = importlib.import_module("flag_attn.gdn2.chunk")
    monkeypatch.setattr(module, "HAS_TLE_GDN2", False)
    monkeypatch.setattr(generic, "HAS_TLE_GDN2", False)
    args, kwargs = _make_inputs(B=1, T=4096, H=16, K=64, V=64, dtype=dtype, state_v_first=False)
    kwargs.update(output_final_state=final_state, return_intermediate_states=intermediate)
    reference_kwargs = dict(kwargs)
    reference_kwargs.pop("use_qk_l2norm_in_kernel", None)
    reference = chunk_gdn2_fwd(
        q=args[0], k=args[1], v=args[2], g=args[3], b=args[4], w_gate=args[5],
        **reference_kwargs,
    )
    expected = (reference[0], reference[1], reference[10]) if intermediate else reference[:2]
    actual = module.chunk_gdn2(*args, **kwargs)
    assert len(actual) == len(expected) == (3 if intermediate else 2)
    for index, (value, reference) in enumerate(zip(actual, expected)):
        if reference is None:
            assert value is None
        else:
            _assert_close(f"fallback result {index}", value, reference)


def _recurrent_gdn2_reference(args, kwargs):
    """FP32 GDN2 recurrence, independent of chunk kernels and native layouts."""
    q, k, v, g, b, w = (value.float() for value in args)
    if kwargs["use_gate_in_kernel"]:
        bias = kwargs["dt_bias"]
        gate = g + bias
        a = kwargs["A_log"].float().exp().view(-1, 1)
        g = (-a * F.softplus(gate) if kwargs["lower_bound"] is None
             else kwargs["lower_bound"] * torch.sigmoid(a * gate))
    initial = kwargs["initial_state"]
    if kwargs["state_v_first"]:
        initial = initial.transpose(-1, -2)
    boundaries = (kwargs["cu_seqlens_cpu"].tolist() if kwargs["cu_seqlens_cpu"] is not None
                  else [0, q.shape[1]])
    outputs, final_states = [], []
    for sequence, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state = initial[sequence].float().clone()
        for t in range(start, end):
            state = state * g[0, t].exp().unsqueeze(-1)
            erased = (state * (b[0, t] * k[0, t]).unsqueeze(-1)).sum(-2)
            residual = w[0, t] * v[0, t] - erased
            state = state + k[0, t].unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append((q[0, t].unsqueeze(-1) * state).sum(-2) * kwargs["scale"])
        final_states.append(state)
    return torch.stack(outputs).unsqueeze(0), torch.stack(final_states)


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["initial", "gate", "bounded_gate", "varlen"])
@torch.inference_mode()
def test_nvidia_gdn2_state_gate_varlen(dtype, case):
    module = importlib.import_module(chunk_gdn2.__module__)
    if not test_backend.is_nvidia() or not module.HAS_TLE_GDN2:
        pytest.skip("NVIDIA TLE GDN2 is unavailable")
    torch.manual_seed(17)
    K, V = (128, 128) if case == "varlen" else (64, 64)
    args, kwargs = _make_inputs(B=1, T=128, H=2, K=K, V=V, dtype=dtype, state_v_first=True)
    count = 2 if case == "varlen" else 1
    kwargs["initial_state"] = torch.randn(count, 2, V, K, device=test_backend.device) * 0.01
    if case == "varlen":
        kwargs["cu_seqlens"] = torch.tensor([0, 64, 128], device=test_backend.device, dtype=torch.long)
        kwargs["cu_seqlens_cpu"] = kwargs["cu_seqlens"].cpu()
    if case in ("gate", "bounded_gate"):
        kwargs["use_gate_in_kernel"] = True
        kwargs["A_log"] = torch.zeros(2, device=test_backend.device)
        kwargs["dt_bias"] = torch.zeros(2, K, device=test_backend.device)
    if case == "bounded_gate":
        kwargs.update(safe_gate=True, lower_bound=-3.0)
    expected, expected_state = _recurrent_gdn2_reference(args, kwargs)
    actual, state = chunk_gdn2(*args, **dict(kwargs, chunk_size=16))
    _assert_close("NVIDIA output", actual, expected)
    _assert_close("NVIDIA final state", state.transpose(-1, -2), expected_state)


def _reference_matmul(left, right):
    # Some CUDA compatibility environments reject cublasSgemmStridedBatched.
    # Independent 2D products preserve the same FP32 reference calculation.
    shape = (*left.shape[:-2], left.shape[-2], right.shape[-1])
    left = left.reshape(-1, *left.shape[-2:])
    right = right.reshape(-1, *right.shape[-2:])
    return torch.stack([a @ b for a, b in zip(left, right)]).reshape(shape)


@pytest.fixture(scope="module", params=[torch.float16, torch.bfloat16])
@torch.inference_mode()
def output_composition_case(request):
    """An independent FP32 oracle for inter-chunk and intra-chunk output."""
    torch.manual_seed(42)
    dtype = request.param
    B, T, H, K, V, BT = 1, 4096, 16, 64, 64, 64
    NT = T // BT
    q = torch.randn(B, T, H, K, device=test_backend.device, dtype=dtype) / math.sqrt(K)
    g = -torch.rand(B, T, H, K, device=test_backend.device, dtype=torch.float32) * 3
    v = torch.randn(B, T, H, V, device=test_backend.device, dtype=dtype)
    h = torch.randn(B, NT, H, K, V, device=test_backend.device, dtype=dtype) * 0.01
    scores = torch.randn(B, NT, H, BT, BT, device=test_backend.device, dtype=dtype).tril() * 0.02
    A = scores.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, BT)
    scale = K**-0.5

    qg = (q.float() * g.exp2()).to(dtype)
    qg = qg.view(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)
    values = v.view(B, NT, BT, H, V).permute(0, 1, 3, 2, 4)
    expected = (_reference_matmul(qg.float(), h.float()) * scale
                + _reference_matmul(scores.float(), values.float()))
    expected = expected.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, V)
    return q, v, g, h, A, scale, expected


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "config",
    chunk_gla_fwd_kernel_o.fn.configs,
    ids=lambda config: (
        f"bk{config.kwargs['BK']}-bv{config.kwargs['BV']}-"
        f"warps{config.num_warps}-stages{config.num_stages}"
    ),
)
@torch.inference_mode()
def test_gdn2_output_config_matches_reference(output_composition_case, config):
    q, v, g, h, A, scale, expected = output_composition_case
    B, T, H, K = q.shape
    V, BT = v.shape[-1], A.shape[-1]
    actual = torch.empty_like(v)
    # Launch each production candidate directly so a fast, faulty configuration
    # cannot evade coverage by losing an autotune timing comparison.
    kernel = chunk_gla_fwd_kernel_o.fn.fn
    for _ in range(3):
        kernel[((V + config.kwargs["BV"] - 1) // config.kwargs["BV"], T // BT, B * H)](
            q,
            v,
            g,
            h,
            actual,
            A,
            None,
            None,
            scale,
            T,
            H=H,
            HV=H,
            K=K,
            V=V,
            BT=BT,
            STATE_V_FIRST=False,
            IS_VARLEN=False,
            **config.kwargs,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _assert_close("output composition", actual, expected)


@pytest.fixture
def gdn2_workload_tuning():
    module = importlib.import_module("flag_attn.gdn2._workload_tuning")
    caches = (module._tuning_source, module._workload_configs, module._workload_kernel)
    for cache in caches:
        cache.cache_clear()
    yield module
    for cache in caches:
        cache.cache_clear()


@pytest.mark.chunk_gdn2
class TestGdn2WorkloadTuning:
    K1 = "_chunk_gdn2_fwd_intra_infer_kernel"
    K2 = "_chunk_gdn2_fwd_h_o_infer_kernel"
    KEY = (8, 64, 64, 16, "torch.bfloat16")

    @staticmethod
    def _entry(key=KEY):
        return dict(
            WORKLOAD=list(key), META={"BK": 64, "BV": 64},
            num_warps=1, num_stages=2, num_ctas=1, maxnreg=None,
        )

    def _source(self, module, monkeypatch, tmp_path, entries, count=None):
        from types import SimpleNamespace
        import yaml
        import triton

        (tmp_path / "tune_configs.yaml").write_text(yaml.safe_dump({self.K1: entries}))
        monkeypatch.setattr(module, "files", lambda package: tmp_path)
        configs = [triton.Config({"BK": 64, "BV": 64}, num_warps=1, num_stages=2)]
        if count is not None:
            configs *= count
        else:
            configs *= len(entries)
        monkeypatch.setattr(
            module, "TunedConfigLoader",
            lambda backend: SimpleNamespace(get_tuned_config=lambda name: configs),
        )

    @pytest.mark.parametrize("name", [K1, K2])
    def test_shipped_configs_match_native_records(self, gdn2_workload_tuning, name):
        import yaml
        from importlib.resources import files
        from flag_attn.runtime.configs_loader import TunedConfigLoader

        resource = files("flag_attn.runtime.backend._nvidia").joinpath("tune_configs.yaml")
        entries = yaml.safe_load(resource.read_text())[name]
        before = TunedConfigLoader("nvidia").get_tuned_config(name)
        after = gdn2_workload_tuning._workload_configs(name)
        assert len(entries) == len(before) == len(after) == 8
        for record, config in zip(entries, before):
            selected = after[tuple(record["WORKLOAD"])]
            assert selected.kwargs == config.kwargs == record["META"]
            for field in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
                assert getattr(selected, field) == getattr(config, field) == record[field]

    @pytest.mark.parametrize("case", ["unknown", "duplicate", "missing", "count"])
    def test_invalid_records_fail(self, gdn2_workload_tuning, monkeypatch, tmp_path, case):
        entries, count = [self._entry()], None
        if case == "duplicate":
            entries *= 2
        elif case == "missing":
            entries[0].pop("WORKLOAD")
        elif case == "count":
            count = 0
        self._source(gdn2_workload_tuning, monkeypatch, tmp_path, entries, count)
        if case == "unknown":
            configs = gdn2_workload_tuning._workload_configs(self.K1)
            assert (16, *self.KEY[1:]) not in configs
            with pytest.raises(KeyError, match="No measured configuration.*workload"):
                gdn2_workload_tuning._workload_kernel(self.K1, object(), (16, *self.KEY[1:]))
        else:
            message = {"duplicate": "Duplicate", "missing": "WORKLOAD", "count": "count"}[case]
            with pytest.raises(ValueError, match=message):
                gdn2_workload_tuning._workload_configs(self.K1)

    def test_missing_yaml_keeps_explicit_miss(self, gdn2_workload_tuning, monkeypatch, tmp_path):
        from types import SimpleNamespace

        monkeypatch.setattr(gdn2_workload_tuning, "files", lambda package: tmp_path)
        monkeypatch.setattr(
            gdn2_workload_tuning, "TunedConfigLoader",
            lambda backend: SimpleNamespace(get_tuned_config=lambda name: []),
        )
        with pytest.raises(KeyError, match="No measured configuration"):
            gdn2_workload_tuning._workload_kernel(self.K1, object(), self.KEY)

    def test_native_key_preserves_argument_dtype_order(self, gdn2_workload_tuning, monkeypatch):
        from types import SimpleNamespace

        tuner = SimpleNamespace(keys=["H", "K", "V", "BT"], arg_names=["second", "H", "first", "T"])
        kernel = SimpleNamespace(fn=tuner)
        arguments = dict(first=torch.empty(1, dtype=torch.float32), H=8, K=64, V=64, BT=16,
                         T=2048, second=torch.empty(1, dtype=torch.bfloat16), unused=torch.empty(1))
        monkeypatch.setattr(gdn2_workload_tuning, "_workload_kernel", lambda op, fn, key: key)
        key = gdn2_workload_tuning.get_gdn2_workload_kernel(self.K1, kernel, arguments)
        assert key == (8, 64, 64, 16, "torch.bfloat16", "torch.float32")

    @pytest.mark.parametrize("supports_disk_cache", [False, True])
    def test_launcher_and_source_caches(self, gdn2_workload_tuning, monkeypatch, tmp_path, supports_disk_cache):
        from types import SimpleNamespace
        import gc
        import weakref
        import triton

        self._source(gdn2_workload_tuning, monkeypatch, tmp_path, [self._entry(), self._entry((16, *self.KEY[1:]))])
        reads, wrapped, options_seen = [], [], []
        monkeypatch.setattr(gdn2_workload_tuning, "files", lambda package: reads.append(package) or tmp_path)
        jit = object()

        class Kernel:
            values = {"FLAG": lambda arguments: False}
            fn = SimpleNamespace(keys=["H", "K", "V", "BT"], arg_names=["q", "T"], fn=jit, cache_results=True)

        kernel = Kernel()

        def build(configs, key, options):
            assert len(configs) == 1
            assert key == kernel.fn.keys
            options_seen.append(options)

            def decorate(fn):
                assert fn is jit
                wrapped.append(configs[0])
                return SimpleNamespace(fn=fn, configs=configs)
            return decorate

        def modern(configs, key, cache_results=False):
            return build(configs, key, {"cache_results": cache_results})

        def legacy(configs, key):
            return build(configs, key, {})

        monkeypatch.setattr(triton, "autotune", modern if supports_disk_cache else legacy)
        monkeypatch.setattr(triton, "heuristics", lambda values: lambda fn: (values, fn))
        tensor = torch.empty(1, dtype=torch.bfloat16)
        reference = weakref.ref(tensor)
        arguments = dict(q=tensor, T=2048, H=8, K=64, V=64, BT=16)
        first = gdn2_workload_tuning.get_gdn2_workload_kernel(self.K1, kernel, arguments)
        assert first[0] == kernel.values
        assert gdn2_workload_tuning.get_gdn2_workload_kernel(self.K1, kernel, arguments) is first
        other = gdn2_workload_tuning.get_gdn2_workload_kernel(self.K1, kernel, dict(arguments, H=16))
        assert other is not first
        assert len(wrapped) == 2 and len(reads) == 1
        assert options_seen == ([{"cache_results": True}] * 2 if supports_disk_cache else [{}, {}])
        del arguments, tensor
        gc.collect()
        assert reference() is None

    def test_generic_loader_retains_original_contract(self):
        from flag_attn.runtime.configs_loader import TunedConfigLoader

        loader = TunedConfigLoader("nvidia")
        assert not any(hasattr(loader, name) for name in ("get_workload_config", "get_workload_kernel", "_workload_kernel"))
        first = loader.get_tuned_config("chunk_fwd_kernel_h")
        assert loader.get_tuned_config("chunk_fwd_kernel_h") is first
        assert [config.kwargs for config in first] == [
            {"BK": 32, "BV": 32}, {"BK": 32, "BV": 64},
            {"BK": 64, "BV": 32}, {"BK": 64, "BV": 64},
        ]
        assert TunedConfigLoader("not_installed_backend").get_tuned_config("missing") == []

    def test_resource_loading_is_independent_of_cwd(self, gdn2_workload_tuning, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert len(gdn2_workload_tuning._workload_configs(self.K1)) == 8

    @pytest.mark.parametrize("vendor", ["cpu", "amd"])
    def test_lazy_import_avoids_private_tuning(self, vendor):
        code = (
            "import sys; import flag_attn; import flag_attn.gdn2; "
            "assert 'flag_attn.gdn2._workload_tuning' not in sys.modules; "
            "assert 'triton.experimental.tle.language' not in sys.modules"
        )
        env = dict(os.environ, FLAG_ATTN_BACKEND=vendor, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_no_tle_import_avoids_private_tuning(self):
        code = (
            "import sys; import flag_attn.utils as utils; utils.has_triton_tle=lambda *args: False; "
            "import flag_attn.gdn2.chunk as chunk; assert not chunk.HAS_TLE_GDN2; "
            "assert 'flag_attn.gdn2._workload_tuning' not in sys.modules"
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
