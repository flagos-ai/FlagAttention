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

"""Benchmark-only pytest options and NVIDIA baseline configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from benchmark.recording import configure_base_data
except ModuleNotFoundError:  # Execution from the benchmark directory.
    from recording import configure_base_data


def _load_base_data(path: str | None) -> dict | None:
    """Load the optional NVIDIA baseline without interrupting local benchmarks."""
    if not path:
        print("[base-data] no --base-data provided; comparison disabled")
        return None
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("baseline must be a JSON object")
        n_ops = sum(1 for key in data if not key.startswith("_"))
        has_scaling = "_scaling_factors" in data
        print(f"[base-data] loaded {path}: {n_ops} operators, scaling factors={has_scaling}")
        return data
    except (OSError, ValueError) as exc:
        print(f"[base-data] cannot load {path}: {exc}; comparison disabled")
        return None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--base-data",
        action="store",
        default=None,
        help="Path to an NVIDIA baseline JSON used for per-shape comparison.",
    )


def pytest_configure(config: pytest.Config) -> None:
    configure_base_data(_load_base_data(config.getoption("--base-data")))
