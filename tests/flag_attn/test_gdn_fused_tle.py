import pytest
import torch

from flag_attn.FLA.gated_delta_rule import chunk_gated_delta_rule
from flag_attn.utils import has_triton_tle


@pytest.mark.skipif(not torch.cuda.is_available() or not has_triton_tle(3, 6, 0), reason="requires CUDA/TLE")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 16384, 16), (4, 2048, 16), (4, 4096, 64), (2, 2048, 16)])
@pytest.mark.parametrize("seed", [42, 123])
@torch.inference_mode()
def test_fused_tle_matches_fla(monkeypatch, dtype, shape, seed):
    import triton.experimental.tle.language as tle

    if not hasattr(tle, "pipe") or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper and TLE pipes")
    reference = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    import flag_attn.FLA.gated_delta_rule.chunk_fused_tle as fused

    original_fwd = fused._tle_chunk_gated_delta_rule_fwd
    calls = []

    def tracked_fwd(*args, **kwargs):
        calls.append(True)
        return original_fwd(*args, **kwargs)

    monkeypatch.setattr(fused, "_tle_chunk_gated_delta_rule_fwd", tracked_fwd)

    B, T, H = shape
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, 128, device="cuda", dtype=dtype) / 128**0.5
    k = torch.randn_like(q) / 128**0.5
    v = torch.randn_like(q)
    g = (-torch.rand(B, T, H, device="cuda") * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
    monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "1")
    monkeypatch.setenv("FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE", "1")
    expected = reference(q, k, v, g, beta, output_final_state=True)
    # Repetition detects timing-dependent pipe lifetime errors.
    previous = None
    for _ in range(5):
        actual = chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, output_final_state=True)
        for a, b in zip(actual, expected):
            assert torch.isfinite(a).all()
            ratio = (a.float() - b.float()).square().mean().sqrt() / (b.float().square().mean().sqrt() + 1e-8)
            assert ratio < 0.01, ratio.item()
        if previous is not None:
            for a, b in zip(actual, previous):
                assert torch.equal(a, b), "nondeterministic pipeline output"
        previous = actual
    assert len(calls) == 5, "regression cases must exercise the fused TLE kernel"


@pytest.mark.skipif(not torch.cuda.is_available() or not has_triton_tle(3, 6, 0), reason="requires CUDA/TLE")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "case",
    [
        "gva",
        "head_first",
        "normalize",
        "zero_scale",
        "zero_gate",
        "strong_gate",
        "initial_state",
        "no_final_state",
        "tail",
        "fp32_gate",
    ],
)
@torch.inference_mode()
def test_fused_tle_api_cases(monkeypatch, dtype, case):
    reference = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    torch.manual_seed(7)
    B, T, H, Hg = 2, 2048, 16, 4 if case == "gva" else 16
    if case == "tail":
        T += 7
    q = torch.randn(B, T, Hg, 128, device="cuda", dtype=dtype) / 128**0.5
    k = torch.randn_like(q) / 128**0.5
    v = torch.randn(B, T, H, 128, device="cuda", dtype=dtype)
    g = (-torch.rand(B, T, H, device="cuda") * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device="cuda", dtype=dtype).sigmoid()
    if case == "zero_gate":
        g.zero_()
    if case == "strong_gate":
        g *= 20
    if case == "fp32_gate":
        g = g.float()
    h0 = torch.randn(B, H, 128, 128, device="cuda", dtype=dtype) * 0.1 if case == "initial_state" else None
    kw = dict(
        scale=0.0 if case == "zero_scale" else 128**-0.5,
        initial_state=h0,
        output_final_state=case != "no_final_state",
        use_qk_l2norm_in_kernel=case == "normalize",
    )
    expected = reference(q, k, v, g, beta, **kw)
    if case == "head_first":
        q, k, v, beta, g = [x.transpose(1, 2) for x in (q, k, v, beta, g)]
    monkeypatch.setenv("FLAG_ATTN_CHUNK_GDR_FUSED_TLE", "1")
    if case == "fp32_gate":
        with pytest.raises(ValueError, match="g must have the same dtype as q"):
            chunk_gated_delta_rule(q, k, v, beta, g, head_first=False, **kw)
        return
    actual = chunk_gated_delta_rule(q, k, v, beta, g, head_first=case == "head_first", **kw)
    for index, (a, b) in enumerate(zip(actual, expected)):
        if b is None:
            assert a is None
            continue
        if case == "head_first" and index == 0:
            a = a.transpose(1, 2)
        assert torch.isfinite(a).all()
        ratio = (a.float() - b.float()).square().mean().sqrt() / (b.float().square().mean().sqrt() + 1e-8)
        assert ratio < 0.01, ratio.item()
