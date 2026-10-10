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

"""Utility functions extracted from vLLM framework dependencies."""

import re
from typing import Any

import torch

# Hygon GPU architectures. Kept in step with the same list in
# flag_attn.runtime.backend.device_finder.
_HYGON_GPU_ARCHS = ("gfx916", "gfx926", "gfx928", "gfx936", "gfx938", "gfx946")


def _triton_target() -> Any | None:
    """Return Triton's active target without making platform detection fatal."""
    try:
        import triton

        return triton.runtime.driver.active.get_current_target()
    except Exception:
        return None


class _CurrentPlatform:
    """Small platform adapter used by the MSA Triton kernels."""

    def is_hygon(self) -> bool:
        """Return whether the active Triton target is a Hygon DCU.

        Hygon exposes the CUDA PyTorch ABI and names its Triton backend ``hip``,
        exactly as AMD does, so neither the device type nor the backend name
        separates the two. The GPU arch does.
        """
        target = _triton_target()
        if target is None:
            return False
        if str(getattr(target, "backend", "")).lower() != "hip":
            return False
        return str(getattr(target, "arch", "")).lower().startswith(_HYGON_GPU_ARCHS)

    def is_arch_support_pdl(self) -> bool:
        if not torch.cuda.is_available():
            return False
        # HIP/ROCm does not accept CUDA's launch_pdl runtime argument.
        if torch.version.hip is not None:
            return False
        try:
            capability = torch.cuda.get_device_capability()
        except RuntimeError:
            return False
        # PDL is enabled only for CUDA devices with the required capability.
        return capability >= (9, 0)


current_platform = _CurrentPlatform()


def round_up(n: int, d: int) -> int:
    """Round n up to the nearest multiple of d."""
    return (n + d - 1) // d * d


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    """Return whether the installed Triton exposes the TLE language module."""
    try:
        import triton
        import triton.experimental.tle.language as _tle  # noqa: F401
    except ImportError:
        return False
    version = str(getattr(triton, "__version__", "0.0.0"))
    release = [int(value) for value in re.findall(r"\d+", version)[:3]]
    release += [0] * (3 - len(release))
    return tuple(release[:3]) >= (major, minor, patch)
