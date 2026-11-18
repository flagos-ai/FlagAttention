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

"""Architecture names for vendors whose generations need separate kernels.

A vendor package may ship a subpackage per GPU generation, so a kernel written
for one generation never runs on another. ``detect_arch`` names the running
generation; ``op_registry`` then prefers that subpackage's exports over the
vendor's own, and the vendor's over the generic implementation.

Only NVIDIA declares generations today. Other vendors return ``None`` and keep
resolving at vendor level, which leaves their current behavior unchanged.
"""

import os
from functools import lru_cache

import torch

# CUDA compute-capability major versions, which separate the generations whose
# kernels genuinely differ: Hopper has wgmma and TMA, Blackwell replaces wgmma
# with tcgen05 and adds a second TMA generation. Both 10.x (datacenter B100 and
# B200) and 12.x (consumer RTX 50) report as Blackwell, because they share the
# instruction set this dispatch selects on.
_NVIDIA_ARCHS = {
    9: "hopper",
    10: "blackwell",
    12: "blackwell",
}

_ARCH_ENV = "FLAG_ATTN_ARCH"


def _normalize_arch(value: str) -> str:
    arch = value.strip().lower().removeprefix("_")
    if not arch.isidentifier():
        raise ValueError(f"Invalid FlagAttention arch {value!r}")
    return arch


@lru_cache(maxsize=None)
def detect_arch(vendor: str) -> str | None:
    """Return the running GPU generation for vendor, or None when it declares none.

    ``FLAG_ATTN_ARCH`` overrides detection, so a machine can be asked for a
    generation's kernels without one of its GPUs present. An unknown or absent
    generation returns None rather than raising: the caller then resolves at
    vendor level, which is always a valid choice.
    """
    override = os.environ.get(_ARCH_ENV)
    if override:
        return _normalize_arch(override)
    if vendor != "nvidia":
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
    except (AssertionError, OSError, RuntimeError):
        return None
    return _NVIDIA_ARCHS.get(major)


__all__ = ["detect_arch"]
