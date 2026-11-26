
import math
from pathlib import Path

import pytest
import torch

from flag_attn.testing import backend as test_backend
from flag_attn import chunk_gdn2, runtime
from flag_attn.gdn2.chunk import chunk_gdn2 as generic_chunk_gdn2
from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.gdn2.native.chunk_fwd import chunk_gdn2_fwd
from flag_attn.gdn2.native.output import chunk_gla_fwd_kernel_o
from flag_attn.runtime.backend._nvidia.gdn2 import chunk as nvidia_gdn2

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


def _gdn2_available() -> bool:
    return test_backend.supports_operator("chunk_gdn2")


pytestmark = [
    pytest.mark.skipif(not _gdn2_available(), reason="chunk_gdn2 is unavailable on the active backend"),
]


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
    "chunk_size,varlen,gate_mode",
    [(16, True, "safe"), (32, False, "log"), (64, False, "softplus")],
)
@torch.inference_mode()
def test_generic_gdn2_matches_token_recurrence(chunk_size, varlen, gate_mode):
    torch.manual_seed(42)
    lengths = [chunk_size + 3, 5] if varlen else [chunk_size + 3] * 2
    B, T = (1, sum(lengths)) if varlen else (2, lengths[0])
    H, K, V = 2, 32, 24
    args, _ = _make_inputs(B=B, T=T, H=H, K=K, V=V, dtype=torch.float16, state_v_first=True)
    q, k, v, g, b, w = args
    state = torch.randn(2, H, K, V, device=q.device) * 0.01
    cu_seqlens = torch.tensor([0, lengths[0], T], device=q.device) if varlen else None
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if varlen else None
    A_log = torch.zeros(H, device=q.device)
    dt_bias = torch.full((H * K,), -3.0, device=q.device)
    lower_bound = -2.0 if gate_mode == "safe" else None

    kwargs = dict(
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
    actual, final_state, snapshots = generic_chunk_gdn2(*args, **kwargs)

    q, k, v, g, b, w = (tensor.float() for tensor in args)
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

    _assert_close("o", actual, expected)
    _assert_close("ht", final_state, torch.stack(expected_states))
    expected_snapshots = torch.stack(expected_snapshots).view(B, -1, H, V, K)
    assert snapshots.shape == expected_snapshots.shape
    _assert_close("snapshots", snapshots, expected_snapshots)
    if test_backend.is_nvidia():
        output, final, history = chunk_gdn2(*args, **kwargs)
        assert history.shape == expected_snapshots.shape
        _assert_close("NVIDIA o", output, expected)
        _assert_close("NVIDIA ht", final, torch.stack(expected_states))
        _assert_close("NVIDIA snapshots", history, expected_snapshots)


@pytest.mark.chunk_gdn2
@pytest.mark.parametrize(
    "impl",
    [pytest.param("tle", id="tle"), pytest.param("native", id="native")]
    if test_backend.is_nvidia() and nvidia_gdn2.supports_fused_gdn2(torch.device(test_backend.device))
    else [pytest.param("backend", id="backend")],
)
@pytest.mark.parametrize(
    "shape,dtype",
    [(shape, dtype) for shape, dtypes in GDN2_TEST_CASES for dtype in dtypes],
)
@torch.inference_mode()
def test_chunk_gdn2_matches_native_triton(impl, dtype, shape):
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

    actual_kwargs = dict(kwargs)
    if (
        impl == "tle"
        or runtime.device.vendor_name == "metax"
        or (impl == "backend" and runtime.device.vendor_name == "mthreads")
    ):
        # Fused inference requires 16-token chunks on these paths.
        actual_kwargs["chunk_size"] = 16

    implementation = generic_chunk_gdn2 if impl == "native" else chunk_gdn2
    actual, actual_final = implementation(*args, **actual_kwargs)

    _assert_close("o", actual, expected)
    _assert_close("ht", actual_final, expected_final)


if test_backend.is_nvidia() and nvidia_gdn2.HAS_TLE_GDN2:
    @pytest.mark.chunk_gdn2
    def test_k1_tle_resource_controls_are_enabled():
        assert nvidia_gdn2.K1_TLE_MAXNREG_CANDIDATES == (64,)
        assert nvidia_gdn2.K1_TLE_SMEM_REUSE_MIN_K == 256
        source = Path(nvidia_gdn2.__file__).read_text(encoding="utf-8")
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
    expected = qg.cpu().float() @ h.cpu().float() * scale + scores.cpu().float() @ values.cpu().float()
    expected = expected.permute(0, 1, 3, 2, 4).contiguous().view(B, T, H, V).to(q.device)
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
