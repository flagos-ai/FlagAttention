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

"""Autotuning configurations loaded from the selected backend's YAML file.

A kernel asks for its configurations by name. The selected backend's
``tune_configs.yaml`` answers when it lists that name, and the kernel's own
fallback answers otherwise, so one generic kernel serves every backend and a
backend that only needs different tiles or launch bounds expresses that in
YAML instead of in a copy of the kernel.
"""

from collections.abc import Sequence
from functools import lru_cache
from inspect import signature
from pathlib import Path

import triton
import yaml

from .backend import get_backend_name

# Reserved top-level key: holds per-kernel tile caps, not an operator's configs.
_TILE_LIMITS_KEY = "tile_limits"


class TunedConfigLoader:
    def __init__(self, backend: str | None = None):
        backend = backend or get_backend_name()
        config_path = (
            Path(__file__).parent / "backend" / f"_{backend}" / "tune_configs.yaml"
        )
        try:
            with config_path.open() as config_file:
                self._configs = yaml.safe_load(config_file) or {}
        except FileNotFoundError:
            self._configs = {}
        self._config_parameters = signature(triton.Config).parameters

    @lru_cache(maxsize=None)
    def get_tuned_config(self, op_name: str) -> list[triton.Config]:
        entries = self._configs.get(op_name)
        # tile_limits is a reserved mapping, not an operator's config list, and
        # a hand-edited file can hold any shape at all; ignore anything that is
        # not a list of config mappings rather than failing the kernel launch.
        if op_name == _TILE_LIMITS_KEY or not isinstance(entries, list):
            return []
        result = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kwargs = {}
            for name in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
                if name in entry and name in self._config_parameters:
                    kwargs[name] = entry[name]
            result.append(triton.Config(entry.get("META", {}), **kwargs))
        return result

    def has_tuned_config(self, op_name: str) -> bool:
        """Return whether this backend's YAML lists configurations for op_name."""
        return bool(self._configs.get(op_name))

    def get_tile_limit(self, op_name: str, tile: str) -> int | None:
        """Return a backend's cap for one tile dimension, or None when unset.

        Tiles that a kernel computes at run time, from the head dimension or
        the available shared memory, cannot be written as a fixed config list.
        A backend that merely needs a smaller tile states the cap under a
        ``tile_limits`` mapping instead of forking the kernel::

            tile_limits:
              parallel_nsa_fwd:
                BV: 64
        """
        limits = self._configs.get(_TILE_LIMITS_KEY)
        if not isinstance(limits, dict):
            return None
        entry = limits.get(op_name)
        if not isinstance(entry, dict):
            return None
        value = entry.get(tile)
        return int(value) if isinstance(value, int) else None


def tuned_config_or(
    op_name: str,
    fallback: Sequence[triton.Config],
) -> list[triton.Config]:
    """Return the selected backend's configurations for op_name, else fallback.

    A kernel passes the search space it would use with no tuning data. A
    backend narrows or replaces that space by listing op_name in its
    ``tune_configs.yaml``, which keeps a tuning-only difference out of the
    kernel source. Importing here, rather than at module scope, keeps this
    module importable before the runtime package has finished initializing.
    """
    from . import config_loader

    tuned = config_loader.get_tuned_config(op_name)
    return list(tuned) if tuned else list(fallback)


def tile_limit(op_name: str, tile: str, value: int) -> int:
    """Clamp a computed tile to the selected backend's cap, when it sets one.

    The kernel keeps its own sizing logic and stays correct everywhere; a
    backend that needs a smaller tile, for register pressure or any other
    device limit, says so in its tune_configs.yaml.
    """
    from . import config_loader

    cap = config_loader.get_tile_limit(op_name, tile)
    return min(value, cap) if cap else value
