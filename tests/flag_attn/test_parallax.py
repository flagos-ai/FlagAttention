# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from itertools import accumulate

import pytest
import torch

from flag_attn.parallax import parallel_parallax


DTYPES = (torch.float16, torch.bfloat16)
TOLERANCE = {
    torch.float16: 0.005,
    torch.bfloat16: 0.02,
}
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


pytestmark = [
    pytest.mark.skipif(
        not _cuda_available(),
        reason="parallel_parallax tests require CUDA",
    ),
]


@dataclass(frozen=True)
class ParallaxCase:
    name: str
    B: int
    T: int
    H: int
    HQ: int
    D: int
    scale: float | None = None
    window_size: int | None = None
    seq_lens: tuple[int, ...] | None = None


# These cases preserve the dense, sliding-window and variable-length coverage
# from the original improve_parallax tests. In particular, they exercise:
#   * short, multi-tile and long sequences;
#   * MHA and GQA;
#   * non-power-of-two D and D=128;
#   * window boundaries that are not tile-aligned;
#   * single- and multi-sequence packed VarLen inputs.
CASES = (
    # Dense causal.
    ParallaxCase(
        name="dense-B1-T63-H1-HQ1-D64-scale1",
        B=1,
        T=63,
        H=1,
        HQ=1,
        D=64,
        scale=1.0,
    ),
    ParallaxCase(
        name="dense-B3-T111-H2-HQ2-D100-scale1",
        B=3,
        T=111,
        H=2,
        HQ=2,
        D=100,
        scale=1.0,
    ),
    ParallaxCase(
        name="dense-B3-T1024-H2-HQ8-D60-scale0.1",
        B=3,
        T=1024,
        H=2,
        HQ=8,
        D=60,
        scale=0.1,
    ),
    ParallaxCase(
        name="dense-B3-T1024-H2-HQ8-D128-scale0.1",
        B=3,
        T=1024,
        H=2,
        HQ=8,
        D=128,
        scale=0.1,
    ),
    ParallaxCase(
        name="dense-B4-T2048-H2-HQ8-D64-scale0.1",
        B=4,
        T=2048,
        H=2,
        HQ=8,
        D=64,
        scale=0.1,
    ),
    # Sliding-window causal.
    ParallaxCase(
        name="window-B1-T63-H1-HQ1-D64-W16",
        B=1,
        T=63,
        H=1,
        HQ=1,
        D=64,
        window_size=16,
    ),
    ParallaxCase(
        name="window-B3-T111-H2-HQ2-D100-W32",
        B=3,
        T=111,
        H=2,
        HQ=2,
        D=100,
        window_size=32,
    ),
    ParallaxCase(
        name="window-B3-T1024-H2-HQ8-D128-W64",
        B=3,
        T=1024,
        H=2,
        HQ=8,
        D=128,
        window_size=64,
    ),
    ParallaxCase(
        name="window-B2-T2048-H2-HQ8-D64-W256",
        B=2,
        T=2048,
        H=2,
        HQ=8,
        D=64,
        window_size=256,
    ),
    ParallaxCase(
        name="window-B2-T1024-H2-HQ2-D64-W200",
        B=2,
        T=1024,
        H=2,
        HQ=2,
        D=64,
        window_size=200,
    ),
    # Packed variable-length causal.
    ParallaxCase(
        name="varlen-H2-HQ2-D64-cu0-15",
        B=1,
        T=15,
        H=2,
        HQ=2,
        D=64,
        seq_lens=(15,),
    ),
    ParallaxCase(
        name="varlen-H2-HQ8-D64-cu0-256-500-1000",
        B=1,
        T=1000,
        H=2,
        HQ=8,
        D=64,
        seq_lens=(256, 244, 500),
    ),
    ParallaxCase(
        name="varlen-H2-HQ2-D100-cu0-15-100-300-1200-2000",
        B=1,
        T=2000,
        H=2,
        HQ=2,
        D=100,
        seq_lens=(15, 85, 200, 900, 800),
    ),
    # Packed variable-length sliding-window causal.
    ParallaxCase(
        name="varlen-window-H2-HQ2-D64-W16-cu0-111",
        B=1,
        T=111,
        H=2,
        HQ=2,
        D=64,
        window_size=16,
        seq_lens=(111,),
    ),
    ParallaxCase(
        name="varlen-window-H2-HQ8-D100-W32-cu0-256-500-1000",
        B=1,
        T=1000,
        H=2,
        HQ=8,
        D=100,
        window_size=32,
        seq_lens=(256, 244, 500),
    ),
)


def _reference_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None,
    window_size: int | None,
) -> torch.Tensor:
    """FP32 PyTorch reference for one or more equal-length sequences."""
    B, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    scale = D**-0.5 if scale is None else scale
    output_dtype = q.dtype

    q = q.float().reshape(B, T, H, G, D)
    r = r.float().reshape(B, T, H, G, D)
    k = k.float()
    v = v.float()

    row = torch.arange(T, device=q.device)[:, None]
    col = torch.arange(T, device=q.device)[None, :]
    valid = col <= row
    if window_size is not None:
        valid &= col >= row - window_size + 1

    batches = []
    for batch_index in range(B):
        heads = []
        for head_index in range(H):
            groups = []
            key = k[batch_index, :, head_index]
            value = v[batch_index, :, head_index]
            for group_index in range(G):
                query = q[batch_index, :, head_index, group_index]
                correction_query = r[
                    batch_index,
                    :,
                    head_index,
                    group_index,
                ]

                logits = query @ key.T * scale
                probabilities = logits.masked_fill(
                    ~valid,
                    -torch.inf,
                ).softmax(dim=-1)
                correction = probabilities * (correction_query @ key.T)
                mean_value = probabilities @ value
                output = (
                    mean_value
                    * (1.0 + correction.sum(dim=-1, keepdim=True))
                    - correction @ value
                )
                groups.append(output)
            heads.append(torch.stack(groups, dim=1))
        batches.append(torch.stack(heads, dim=1))

    return torch.stack(batches).reshape(B, T, HQ, D).to(output_dtype)


def _reference_varlen(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    seq_lens: tuple[int, ...],
    scale: float | None,
    window_size: int | None,
) -> torch.Tensor:
    offsets = (0, *accumulate(seq_lens))
    outputs = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        outputs.append(
            _reference_parallax(
                q[:, start:end],
                r[:, start:end],
                k[:, start:end],
                v[:, start:end],
                scale=scale,
                window_size=window_size,
            )
        )
    return torch.cat(outputs, dim=1)


def _make_inputs(
    case: ParallaxCase,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    assert case.HQ % case.H == 0, "HQ must be divisible by H"
    if case.seq_lens is not None:
        assert case.B == 1, "VarLen inputs require B == 1"
        assert sum(case.seq_lens) == case.T
        assert all(length > 0 for length in case.seq_lens)

    query_shape = (case.B, case.T, case.HQ, case.D)
    kv_shape = (case.B, case.T, case.H, case.D)
    device = "cuda"
    return (
        torch.randn(query_shape, dtype=dtype, device=device),
        torch.randn(query_shape, dtype=dtype, device=device),
        torch.randn(kv_shape, dtype=dtype, device=device),
        torch.randn(kv_shape, dtype=dtype, device=device),
    )


def _clone_for_grad(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        tensor.detach().clone().requires_grad_(True)
        for tensor in tensors
    )


def _absolute_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return (
        (expected.detach().float() - actual.detach().float())
        .abs()
        .max()
        .item()
    )


def _error_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    error = (
        (expected.detach().float() - actual.detach().float())
        .square()
        .mean()
        .sqrt()
    )
    reference = expected.detach().float().square().mean().sqrt()
    return (error / (reference + 1e-8)).item()


def _assert_close(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    tolerance: float,
    *,
    warning: bool = False,
    abs_tolerance: float = 1e-6,
) -> tuple[float, float]:
    assert expected.shape == actual.shape, (
        f"{name}: shape mismatch: expected {expected.shape}, "
        f"got {actual.shape}"
    )
    assert expected.dtype == actual.dtype, (
        f"{name}: dtype mismatch: expected {expected.dtype}, "
        f"got {actual.dtype}"
    )
    assert torch.isfinite(expected).all(), (
        f"{name}: reference contains NaN or Inf"
    )
    assert torch.isfinite(actual).all(), (
        f"{name}: result contains NaN or Inf"
    )

    absolute_error = _absolute_error(expected, actual)
    ratio = _error_ratio(expected, actual)
    message = (
        f"{name:>16} diff: {absolute_error:.6f} "
        f"ratio: {ratio:.6f}"
    )
    logger.info(message)

    if absolute_error <= abs_tolerance:
        return absolute_error, ratio

    # Preserve the original FLA CI behavior: borderline numerical deviations
    # can be reported as warnings in CI, while local runs remain strict.
    allow_ci_warning = FLA_CI_ENV and (
        ratio < 0.01 or absolute_error <= 0.3
    )
    if warning or allow_ci_warning:
        if ratio >= tolerance:
            warnings.warn(message, stacklevel=2)
        return absolute_error, ratio

    assert ratio < tolerance, (
        f"{message}; tolerance: {tolerance:.6f}"
    )
    return absolute_error, ratio


def _print_error_report(
    case: ParallaxCase,
    dtype: torch.dtype,
    tolerance: float,
    metrics: tuple[tuple[str, float, float], ...],
) -> None:
    dtype_name = str(dtype).removeprefix("torch.")
    title = (
        f"parallel_parallax | {case.name} | "
        f"dtype={dtype_name} | tolerance={tolerance:.6f}"
    )
    separator = "-" * len(title)
    rows = [
        "",
        separator,
        title,
        separator,
        f"{'tensor':<8} {'max_abs_error':>16} {'error_ratio':>16} {'status':>10}",
    ]
    for name, absolute_error, ratio in metrics:
        status = (
            "PASS"
            if absolute_error <= 1e-6 or ratio < tolerance
            else "WARN"
        )
        rows.append(
            f"{name:<8} {absolute_error:>16.6e} "
            f"{ratio:>16.6e} {status:>10}"
        )
    rows.append(separator)
    print("\n".join(rows), flush=True)


def _run_reference(
    case: ParallaxCase,
    inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    if case.seq_lens is None:
        return _reference_parallax(
            *inputs,
            scale=case.scale,
            window_size=case.window_size,
        )
    return _reference_varlen(
        *inputs,
        seq_lens=case.seq_lens,
        scale=case.scale,
        window_size=case.window_size,
    )


def _make_cu_seqlens(
    case: ParallaxCase,
) -> torch.Tensor | None:
    if case.seq_lens is None:
        return None
    return torch.tensor(
        (0, *accumulate(case.seq_lens)),
        dtype=torch.long,
        device="cuda",
    )


@pytest.mark.parametrize(
    "dtype",
    DTYPES,
    ids=("float16", "bfloat16"),
)
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case.name,
)
def test_parallel_parallax_forward_backward(
    case: ParallaxCase,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    base_inputs = _make_inputs(case, dtype)
    reference_inputs = _clone_for_grad(base_inputs)
    actual_inputs = _clone_for_grad(base_inputs)
    del base_inputs

    grad_output = torch.randn(
        (case.B, case.T, case.HQ, case.D),
        dtype=dtype,
        device="cuda",
    )

    # Finish and release the much larger PyTorch reference graph before
    # constructing the parallel_parallax graph. This follows the original
    # reference-first validation order and substantially lowers peak memory.
    expected = _run_reference(case, reference_inputs)
    expected_output = expected.detach()
    expected_grads = tuple(
        grad.detach()
        for grad in torch.autograd.grad(
            expected,
            reference_inputs,
            grad_outputs=grad_output,
        )
    )
    del expected
    del reference_inputs

    actual = parallel_parallax(
        *actual_inputs,
        scale=case.scale,
        window_size=case.window_size,
        cu_seqlens=_make_cu_seqlens(case),
    )
    actual_grads = tuple(
        grad.detach()
        for grad in torch.autograd.grad(
            actual,
            actual_inputs,
            grad_outputs=grad_output,
        )
    )

    assert actual.shape == (
        case.B,
        case.T,
        case.HQ,
        case.D,
    )
    assert actual.dtype == dtype

    tolerance = TOLERANCE[dtype]
    output_absolute_error, output_ratio = _assert_close(
        "o",
        expected_output,
        actual.detach(),
        tolerance,
    )
    metrics = [
        ("o", output_absolute_error, output_ratio),
    ]
    for name, expected_grad, actual_grad in zip(
        ("dq", "dr", "dk", "dv"),
        expected_grads,
        actual_grads,
    ):
        absolute_error, ratio = _assert_close(
            name,
            expected_grad,
            actual_grad,
            tolerance,
        )
        metrics.append((name, absolute_error, ratio))

    _print_error_report(
        case,
        dtype,
        tolerance,
        tuple(metrics),
    )


def test_parallel_parallax_rejects_float32() -> None:
    inputs = tuple(
        torch.randn(
            (1, 8, 1, 64),
            dtype=torch.float32,
            device="cuda",
        )
        for _ in range(4)
    )
    with pytest.raises(TypeError, match="requires bf16 or fp16"):
        parallel_parallax(*inputs)


def test_parallel_parallax_varlen_requires_batch_one() -> None:
    inputs = tuple(
        torch.randn(
            (2, 8, 1, 64),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(4)
    )
    cu_seqlens = torch.tensor(
        [0, 8],
        dtype=torch.long,
        device="cuda",
    )
    with pytest.raises(
        ValueError,
        match="batch size is expected to be 1",
    ):
        parallel_parallax(
            *inputs,
            cu_seqlens=cu_seqlens,
        )
