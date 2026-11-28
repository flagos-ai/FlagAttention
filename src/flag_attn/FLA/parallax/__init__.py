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

"""Parameterized local linear attention implemented with Triton kernels."""

from typing import Literal

import torch

from flag_attn.runtime.backend import resolve_operator

from .parallel import (
    ParallaxFunction,
    parallel_parallax_bwd,
    parallel_parallax_fwd,
)


def __getattr__(name):
    if name == "HAS_TLE":
        from . import decode

        return decode.HAS_TLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def parallax_decode(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float | None = None,
    *,
    window_size_left: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode through runtime dispatch, preserving the original decode interface."""
    return resolve_operator("parallax_decode", "flag_attn.FLA.parallax.decode")(
        q, r, k, v, qk_scale, window_size_left=window_size_left, out=out
    )


def parallax_attn_with_kvcache(
    q: torch.Tensor,
    r: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    window_size=None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode a KV cache through runtime dispatch, including caller-owned output."""
    return resolve_operator("parallax_attn_with_kvcache", "flag_attn.FLA.parallax.decode")(
        q, r, k_cache, v_cache, page_table=page_table, seqused_k=seqused_k,
        window_size=window_size, scale=scale, out=out,
    )


def parallel_parallax(*args, stage: Literal["prefill", "decode", "kvcache"] = "prefill", **kwargs):
    """Run Parallax attention through a single public entry.

    Arguments and return values follow parallel.parallel_parallax for
    prefill (the default), parallax_decode for decode, and
    parallax_attn_with_kvcache for KV-cache decode. Each interface retains
    its original scale, window and output-buffer conventions.
    """
    if stage == "prefill":
        return resolve_operator("parallel_parallax", "flag_attn.FLA.parallax.parallel")(*args, **kwargs)
    if stage == "decode":
        return parallax_decode(*args, **kwargs)
    if stage == "kvcache":
        return parallax_attn_with_kvcache(*args, **kwargs)
    raise ValueError(f"Unsupported Parallax stage: {stage!r}; expected 'prefill', 'decode' or 'kvcache'")


__all__ = [
    "HAS_TLE",
    "ParallaxFunction",
    "parallax_attn_with_kvcache",
    "parallax_decode",
    "parallel_parallax",
    "parallel_parallax_bwd",
    "parallel_parallax_fwd",
]
