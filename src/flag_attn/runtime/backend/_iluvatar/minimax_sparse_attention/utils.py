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

import torch


class _CurrentPlatform:
    """Platform surface the MSA benchmark reads at import.

    Iluvatar kernels do not launch PDL. ``is_arch_support_pdl`` stays so the
    unmodified benchmark can cache the decision without touching this backend.
    """

    def is_arch_support_pdl(self) -> bool:
        if not torch.cuda.is_available():
            return False
        if torch.version.hip is not None:
            return False
        try:
            capability = torch.cuda.get_device_capability()
        except RuntimeError:
            return False
        return capability >= (9, 0)


current_platform = _CurrentPlatform()


def round_up(n: int, d: int) -> int:
    """Round n up to the nearest multiple of d."""
    return (n + d - 1) // d * d
