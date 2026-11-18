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

"""NVIDIA baselines for Parallax forward and decode pytest workloads."""

import torch


OP_NAME = "parallel_parallax"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

PREFILL_CASES = (
    (1, 15, 2, 2, 64), (1, 63, 1, 1, 64), (1, 111, 2, 2, 64),
    (2, 200, 2, 8, 64), (2, 256, 2, 8, 64), (2, 512, 2, 8, 64),
    (2, 1024, 2, 2, 64), (2, 2048, 2, 8, 64), (2, 4096, 2, 2, 64),
    (2, 8192, 2, 8, 64), (3, 111, 2, 2, 100), (3, 1024, 2, 8, 128),
    (2, 2048, 2, 8, 128), (2, 8192, 8, 16, 128),
)
DECODE_CASES = (
    (1, 512, 8, 8, 64),
    (1, 2048, 8, 8, 128),
    (2, 2048, 8, 2, 128),
    (1, 8192, 8, 2, 128),
)


def _decode_reference(q, r, k, v, scale, window_size_left=-1):
    """FP32 Q=1 oracle without the degenerate cuBLAS einsum path."""
    HQ, H = q.shape[2], k.shape[2]
    repeat = HQ // H
    qf = q[:, 0].float()
    rf = r[:, 0].float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    if repeat > 1:
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    s1 = (qf[:, :, None, :] * kf).sum(dim=-1) * scale
    s2 = (rf[:, :, None, :] * kf).sum(dim=-1)
    if window_size_left >= 0:
        first = max(k.shape[1] - window_size_left, 0)
        valid = torch.arange(k.shape[1], device=k.device) >= first
        s1 = s1.masked_fill(~valid[None, None, :], float("-inf"))
    pivot = s1.amax(dim=-1, keepdim=True)
    pivot_safe = torch.where(torch.isfinite(pivot), pivot, torch.zeros_like(pivot))
    p1 = torch.exp(s1 - pivot_safe)
    p2 = p1 * s2
    d1 = p1.sum(dim=-1, keepdim=True)
    d2 = p2.sum(dim=-1, keepdim=True)
    o1 = (p1[..., None] * vf).sum(dim=2)
    o2 = (p2[..., None] * vf).sum(dim=2)
    inv_d1 = torch.where(d1 > 0, d1.reciprocal(), torch.zeros_like(d1))
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    return result[:, None].contiguous()


def native():
    try:
        from fla.ops.parallax.parallel import parallel_parallax_fwd
    except (ImportError, OSError):
        return None

    def parallax_reference(q, r, k, v, scale, *, variant, window_size_left=-1):
        if variant == "prefill":
            return parallel_parallax_fwd(q, r, k, v, scale, None, None, window_size_left)
        return _decode_reference(q, r, k, v, scale, window_size_left)

    return parallax_reference


def grid():
    return [
        {"variant": "prefill", **dict(zip(("B", "T", "H", "HQ", "D"), case))}
        for case in PREFILL_CASES
    ] + [
        {"variant": "decode", **dict(zip(("B", "L", "HQ", "H", "D"), case))}
        for case in DECODE_CASES
    ]


def _prefill_seed(binding, dtype):
    values = [
        binding["B"], binding["T"], binding["H"], binding["HQ"],
        binding["D"], -1, 0 if dtype == torch.float16 else 1,
    ]
    seed = 42
    for value in values:
        seed = (seed * 1_000_003 + value) % (2**63 - 1)
    return seed


def build_inputs(binding, dtype, device):
    b, h, hq, d = (binding[key] for key in ("B", "H", "HQ", "D"))
    if binding["variant"] == "prefill":
        t = binding["T"]
        generator = torch.Generator(device=device).manual_seed(
            _prefill_seed(binding, dtype)
        )
        q_shape, kv_shape = (b, t, hq, d), (b, t, h, d)
    else:
        length = binding["L"]
        seed = 2026 + b * 1_000_003 + length * 10_007 + hq * 101 + h * 17 + d
        generator = torch.Generator(device=device).manual_seed(seed)
        q_shape, kv_shape = (b, 1, hq, d), (b, length, h, d)
    q = torch.randn(q_shape, dtype=dtype, device=device, generator=generator)
    r = torch.randn(q_shape, dtype=dtype, device=device, generator=generator)
    if binding["variant"] == "decode":
        r = r * 0.5
    k = torch.randn(kv_shape, dtype=dtype, device=device, generator=generator)
    v = torch.randn(kv_shape, dtype=dtype, device=device, generator=generator)
    return (q, r, k, v, d**-0.5), {
        "variant": binding["variant"], "window_size_left": -1,
    }


def key_shape(binding):
    keys = (
        ("B", "T", "H", "HQ", "D") if binding["variant"] == "prefill"
        else ("B", "L", "HQ", "H", "D")
    )
    return [binding["variant"], *(binding[key] for key in keys)]


def config(binding, dtype):
    b, h, hq, d = (binding[key] for key in ("B", "H", "HQ", "D"))
    variant = binding["variant"]
    if variant == "prefill":
        t = binding["T"]
        q_shape, kv_shape = [b, t, hq, d], [b, t, h, d]
        dims = {key: binding[key] for key in ("B", "T", "H", "HQ", "D")}
        outputs = {
            "o": {"shape": q_shape, "dtype": str(dtype)},
            "barv": {"shape": q_shape, "dtype": str(dtype)},
            **{
                name: {"shape": [b, t, hq], "dtype": str(torch.float32)}
                for name in ("d1", "bart", "m")
            },
        }
    else:
        length = binding["L"]
        q_shape, kv_shape = [b, 1, hq, d], [b, length, h, d]
        dims = {key: binding[key] for key in ("B", "L", "HQ", "H", "D")}
        outputs = {"output": {"shape": q_shape, "dtype": str(torch.float32)}}

    def tensor(shape):
        return {"shape": shape, "dtype": str(dtype)}
    return {
        "op": OP_NAME,
        "dtype": str(dtype),
        "phase": "forward" if variant == "prefill" else "decode",
        "dims": dims,
        "inputs": {
            "q": tensor(q_shape), "r": tensor(q_shape),
            "k": tensor(kv_shape), "v": tensor(kv_shape),
            "scale": {"scalar": d**-0.5},
            "window_size_left": {"scalar": -1},
        },
        "outputs": outputs,
        "mutation": "none",
        "graph": False,
    }
