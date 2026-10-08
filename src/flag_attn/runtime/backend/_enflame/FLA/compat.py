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

"""Shared Triton compatibility helpers in the Enflame FLA package."""

import inspect
import os
import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from flag_attn.runtime.configs_loader import TunedConfigLoader

F = TypeVar("F", bound=Callable[..., Any])


def get_exp():
    return (
        tldevice.fast_expf
        if os.environ.get("FLAG_ATTN_USE_FAST_OPS", "0") == "1"
        else tl.exp
    )


exp = get_exp()


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


_SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(
    triton.autotune
).parameters
autotune_cache_kwargs = (
    {"cache_results": True} if _SUPPORTS_AUTOTUNE_CACHE else {}
)
_tuned_config_loader = TunedConfigLoader("enflame")


def get_tuned_config(
    op_name: str, fallback: Sequence[triton.Config]
) -> list[triton.Config]:
    if os.environ.get("FLAG_ATTN_USE_TUNED_CONFIGS", "1") == "0":
        return list(fallback)
    configs = _tuned_config_loader.get_tuned_config(op_name)
    if configs and os.environ.get("FLAG_ATTN_FAST_AUTOTUNE", "0") == "1":
        return [configs[0]]
    return list(configs) if configs else list(fallback)


def libentry() -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        return fn

    return decorator


def libtuner(
    *,
    configs: Sequence[triton.Config],
    key: Sequence[str],
    use_cuda_graph: bool = False,
    **kwargs: Any,
):
    del use_cuda_graph
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)


def _triton_version() -> tuple[int, int, int]:
    parts = re.findall(r"\d+", str(getattr(triton, "__version__", "0.0.0")))
    values = [int(part) for part in parts[:3]]
    return tuple((values + [0, 0, 0])[:3])


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    if _triton_version() < (major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    return True


use_cuda_graph = False
