
# Copyright (c) 2026 FlagAttention contributors.
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

"""Load C550 candidates without changing correctness or benchmark coverage."""

import copy
import os
from functools import lru_cache
from pathlib import Path

import triton

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "tune_configs.yaml"
_FIELDS = {"META", "num_warps", "num_stages", "num_ctas", "maxnreg"}


@lru_cache(maxsize=1)
def _load_profiles():
    # Full tuning remains usable even if the optional profile cannot be read.
    import yaml

    data = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not isinstance(k, str) for k in data):
        raise ValueError(f"{_CONFIG_PATH}: expected a kernel-to-config-list mapping")
    return data


def get_tuned_config(
    kernel_name: str, fallback: list[triton.Config]
) -> list[triton.Config]:
    """Use measured C550 candidates, or all candidates when explicitly requested.

    FLAG_ATTN_GDN2_FULL_TUNING=1 restores the original search space only.
    It does not select test cases, shapes, precision, or implementation routes.
    The profile's historical winners are not guaranteed optimal for every shape.
    """
    if os.environ.get("FLAG_ATTN_GDN2_FULL_TUNING") == "1":
        return fallback
    profiles = _load_profiles()
    if kernel_name not in profiles:
        return fallback
    entries = profiles[kernel_name]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{kernel_name}: tuned candidates must be a nonempty list")

    selected = []
    for index, entry in enumerate(entries):
        label = f"{kernel_name}: candidate {index}"
        if not isinstance(entry, dict) or "META" not in entry:
            raise ValueError(f"{label}: expected a mapping containing META")
        unknown = set(entry) - _FIELDS
        if unknown:
            raise ValueError(f"{label}: unknown fields {sorted(map(str, unknown))}")
        meta = entry["META"]
        if not isinstance(meta, dict) or any(not isinstance(k, str) for k in meta):
            raise ValueError(f"{label}: META must be a mapping with string keys")
        options = {k: v for k, v in entry.items() if k != "META"}
        for key, value in options.items():
            if key == "maxnreg" and value is None:
                continue
            minimum = 0 if key == "num_stages" else 1
            if type(value) is not int or value < minimum:  # noqa: E721 - Exact int; reject bool/subclasses.
                raise ValueError(f"{label}: invalid {key}={value!r}")
        try:
            candidate = triton.Config(copy.deepcopy(meta), **options)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}: invalid Triton configuration: {error}") from error
        # Display strings omit fields such as pre_hook; compare complete state.
        if not any(vars(candidate) == vars(config) for config in fallback):
            raise ValueError(f"{label}: configuration is not in the full candidate list")
        if any(vars(candidate) == vars(config) for config in selected):
            raise ValueError(f"{label}: duplicate configuration")
        selected.append(candidate)
    return selected
