# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Runtime compatibility helpers for the MetaX MSA kernels."""

import inspect
import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar, overload

import torch


_F = TypeVar("_F", bound=Callable[..., Any])


def _triton_target() -> Any | None:
    """Return Triton's active target without making platform detection fatal."""
    try:
        import triton

        return triton.runtime.driver.active.get_current_target()
    except Exception:
        return None


def _torch_device_name() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        return str(torch.cuda.get_device_name()).lower()
    except (AssertionError, RuntimeError):
        return ""


class _CurrentPlatform:
    """Small platform adapter used by the MSA Triton kernels."""

    def triton_backend(self) -> str:
        target = _triton_target()
        return str(getattr(target, "backend", "")).lower()

    def is_metax(self) -> bool:
        """Return whether the active Triton target is MetaX/MACA.

        MetaX exposes the CUDA PyTorch ABI, so ``torch.version.cuda`` and the
        ``cuda`` device type do not distinguish it from NVIDIA. mcTriton does:
        its active target backend is named ``maca``. The device-name fallback
        keeps detection useful in stripped-down or pre-initialization setups.
        """
        backend = self.triton_backend()
        if backend in {"maca", "metax"}:
            return True
        device_name = _torch_device_name()
        return "metax" in device_name or "曦云" in device_name

    def is_c550(self) -> bool:
        if not self.is_metax():
            return False
        target = _triton_target()
        arch = getattr(target, "arch", 0)
        try:
            if int(arch) >= 90:
                return True
        except (TypeError, ValueError):
            pass
        return "c550" in _torch_device_name()

    def is_arch_support_pdl(self) -> bool:
        if not torch.cuda.is_available():
            return False
        # HIP/ROCm and MetaX mcTriton do not accept CUDA's launch_pdl runtime
        # argument. C550 reports CUDA capability 9.0 through the compatibility
        # ABI, so the backend check must happen before the capability check.
        if torch.version.hip is not None:
            return False
        if self.is_metax():
            return False
        try:
            capability = torch.cuda.get_device_capability()
        except RuntimeError:
            return False
        # PDL is enabled only for CUDA devices with the required capability.
        return capability >= (9, 0)

    def attention_launch_kwargs(
        self,
        *,
        num_warps: int = 4,
        num_stages: int = 2,
        scenario: str = "flashattn-fwd",
    ) -> dict[str, int | str]:
        """Return backend-safe launch options for dot-heavy attention kernels."""
        if not self.is_metax():
            return {}
        # mcTriton accepts these MACAOptions at the launch site. The
        # flash-attention scenario enables its chain-dot scheduling path.
        return {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "pipeline": "basic",
            "scenario": scenario,
        }


current_platform = _CurrentPlatform()


def round_up(n: int, d: int) -> int:
    """Round n up to the nearest multiple of d."""
    return (n + d - 1) // d * d


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    """Return whether the installed Triton exposes the TLE language module."""
    try:
        import triton
        import triton.experimental.tle.language as _tle  # noqa: F401
    except Exception:
        return False
    version = str(getattr(triton, "__version__", "0.0.0"))
    release = [int(value) for value in re.findall(r"\d+", version)[:3]]
    release += [0] * (3 - len(release))
    return tuple(release[:3]) >= (major, minor, patch)


@overload
def triton_jit(fn: _F) -> Any: ...


@overload
def triton_jit(
    fn: None = None,
    *,
    do_not_specialize_on_alignment: Iterable[int | str] | None = None,
    **kwargs: Any,
) -> Callable[[_F], Any]: ...


def triton_jit(
    fn: _F | None = None,
    *,
    do_not_specialize_on_alignment: Iterable[int | str] | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``triton.jit`` across upstream Triton and mcTriton 3.0.

    ``do_not_specialize_on_alignment`` was added after the Triton version used
    by the C550 FlagOS image. On older releases, mapping those arguments to
    ``do_not_specialize`` also removes the pointer-alignment cache key and is a
    conservative compatibility fallback.
    """
    import triton

    alignment_args = tuple(do_not_specialize_on_alignment or ())
    if alignment_args:
        jit_params = inspect.signature(triton.jit).parameters
        if "do_not_specialize_on_alignment" in jit_params:
            kwargs["do_not_specialize_on_alignment"] = alignment_args
        else:
            existing = tuple(kwargs.get("do_not_specialize", ()))
            kwargs["do_not_specialize"] = tuple(dict.fromkeys(existing + alignment_args))

    if fn is not None:
        return triton.jit(fn, **kwargs)
    return triton.jit(**kwargs)
