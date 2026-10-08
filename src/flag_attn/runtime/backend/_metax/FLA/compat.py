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

"""Compatibility helpers shared by the MetaX FLA Triton kernels."""

from __future__ import annotations

from collections.abc import Sequence
import inspect
import re
from typing import Any

import triton
import triton.language as tl


def _triton_version() -> tuple[int, int, int]:
    parts = re.findall(r"\d+", str(getattr(triton, "__version__", "0.0.0")))
    values = [int(part) for part in parts[:3]]
    return tuple((values + [0, 0, 0])[:3])


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    """Return whether this Triton build provides the requested TLE extension."""

    if _triton_version() < (major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    return True


HAS_TLE = has_triton_tle()


@triton.jit
def exp2(x):
    """Base-2 exponential with fp32 computation."""

    return tl.math.exp2(x.to(tl.float32))


try:
    _SUPPORTS_AUTOTUNE_CACHE = (
        "cache_results" in inspect.signature(triton.autotune).parameters
    )
except Exception:
    _SUPPORTS_AUTOTUNE_CACHE = False


autotune_cache_kwargs = (
    {"cache_results": True} if _SUPPORTS_AUTOTUNE_CACHE else {}
)


def libtuner(
    *,
    configs: Sequence[triton.Config],
    key: Sequence[str],
    use_cuda_graph: bool = False,
    **kwargs: Any,
):
    """Map FlagGems autotuning metadata to Triton's native autotuner."""
    del use_cuda_graph
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)
