import importlib
import math
from pathlib import Path

import pytest
import torch

from flag_attn import chunk_gdn2
from flag_attn.gdn2.native.chunk_fwd import chunk_gdn2_fwd
from flag_attn.gdn2.native.output import chunk_gla_fwd_kernel_o

ASSERT_RATIO = 0.01

GDN2_TEST_SHAPES = [
    (2, 512, 8, 64, 64),
    (4, 1024, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 96, 128, 128),
    (2, 2048, 16, 256, 512),
    (2, 16384, 16, 128, 128),
    (4, 1024, 8, 256, 512),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = [
    pytest.mark.skipif(not _cuda_available(), reason="chunk_gdn2 tests require CUDA"),
]


def _set_public_gdn2_native_for_call(force_native: bool):
    module = importlib.import_module("flag_attn.gdn2.chunk")
    old = module.HAS_TLE_GDN2
    if force_native:
        module.HAS_TLE_GDN2 = False
    return module, old


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
    device = "cuda"
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


@pytest.mark.parametrize(
    "impl",
    [pytest.param("tle", id="tle"), pytest.param("native", id="native")],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN2_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gdn2_matches_native_triton(impl, dtype, shape):
    module = importlib.import_module("flag_attn.gdn2.chunk")
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
        actual, actual_final = chunk_gdn2(*args, **actual_kwargs)
    finally:
        module.HAS_TLE_GDN2 = old

    _assert_close("o", actual, expected)
    _assert_close("ht", actual_final, expected_final)


def test_k1_tle_resource_controls_are_enabled():
    module = importlib.import_module("flag_attn.gdn2.chunk")
    if not module.HAS_TLE_GDN2:
        pytest.skip("TLE GDN2 path is unavailable in this environment")

    assert module.K1_TLE_MAXNREG_CANDIDATES == (64, 72, 96)
    assert module.K1_TLE_SMEM_REUSE_MIN_K == 256
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "tle.gpu.alloc" in source
    assert "tle.gpu.local_ptr" in source


@pytest.fixture(scope="module", params=[torch.float16, torch.bfloat16])
@torch.inference_mode()
def output_composition_case(request):
    """An independent FP32 oracle for inter-chunk and intra-chunk output."""
    torch.manual_seed(42)
    dtype = request.param
    B, T, H, K, V, BT = 1, 4096, 16, 64, 64, 64
    NT = T // BT
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) / math.sqrt(K)
    g = -torch.rand(B, T, H, K, device="cuda", dtype=torch.float32) * 3
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype)
    h = torch.randn(B, NT, H, K, V, device="cuda", dtype=dtype) * 0.01
    scores = torch.randn(B, NT, H, BT, BT, device="cuda", dtype=dtype).tril() * 0.02
    A = scores.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, BT)
    scale = K**-0.5

    qg = (q.float() * g.exp2()).to(dtype)
    qg = qg.view(B, NT, BT, H, K).permute(0, 1, 3, 2, 4)
    values = v.view(B, NT, BT, H, V).permute(0, 1, 3, 2, 4)
    expected = qg.float() @ h.float() * scale + scores.float() @ values.float()
    expected = expected.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, V)
    return q, v, g, h, A, scale, expected


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
