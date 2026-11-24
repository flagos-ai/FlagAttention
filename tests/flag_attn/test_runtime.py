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

"""Public metadata and backend selection regressions; no GPU is required."""

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import triton

import flag_attn
from flag_attn.runtime.backend import device_finder


@pytest.fixture
def automatic_vendor(monkeypatch):
    monkeypatch.delenv("FLAG_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("FLAG_ATTN_VENDOR", raising=False)


def test_public_device_metadata():
    assert isinstance(flag_attn.device, str)
    assert isinstance(flag_attn.vendor_name, str)
    assert flag_attn.vendor == flag_attn.vendor_name
    assert flag_attn.device == flag_attn.runtime.device.name
    assert flag_attn.vendor_name == flag_attn.runtime.device.vendor_name
    assert flag_attn.backend_info is flag_attn.runtime.device
    assert flag_attn.runtime.device.get_vendor_name() == flag_attn.vendor_name
    assert flag_attn.runtime.torch_device_fn is getattr(torch, flag_attn.device, None)
    assert {"device", "vendor", "vendor_name", "chunk_gdn2", "chunk_kda"}.issubset(
        flag_attn.__all__
    )


def test_unknown_attribute():
    with pytest.raises(AttributeError, match="has no attribute 'missing_operator'"):
        flag_attn.missing_operator


def _operator_inventory_names():
    yaml = pytest.importorskip("yaml")
    inventory = Path(__file__).resolve().parents[2] / "conf" / "operators.yaml"
    with open(inventory) as handle:
        catalog = yaml.safe_load(handle).get("ops", [])
    names, non_operator_ids = [], {"runtime"}
    for entry in catalog:
        if entry.get("id") in non_operator_ids:
            continue
        for name in entry.get("for", []):
            if name not in names and name not in non_operator_ids:
                names.append(name)
    return names


@pytest.mark.parametrize("name", _operator_inventory_names())
def test_operator_inventory_name_is_exposed_at_top_level(name):
    """Every conf/operators.yaml operator is reachable as flag_attn.<name>.

    A missing kernel dependency raises on resolution rather than reporting
    the name as absent, so only AttributeError is treated as a failure here.
    """
    assert name in flag_attn.__all__, f"{name} is in conf/operators.yaml but missing from flag_attn.__all__"
    try:
        resolved = getattr(flag_attn, name)
    except AttributeError as exc:
        pytest.fail(f"flag_attn.{name} is in conf/operators.yaml but not exposed: {exc}")
    except Exception as exc:  # unavailable kernel dependency in this environment
        pytest.skip(f"{name} is exposed but its implementation is unavailable here: {type(exc).__name__}: {exc}")
    assert callable(resolved) or not isinstance(resolved, ModuleType), (
        f"flag_attn.{name} should be an operator entry point, got {resolved!r}"
    )


def test_subpackage_paths_still_import_after_top_level_export():
    """Publishing a callable must not hide the same-named subpackage."""
    for module_name, attribute in (
        ("flag_attn.forgetting_attention.naive", "forgetting_attention"),
        ("flag_attn.forgetting_attention.parallel", "parallel_forgetting_attn"),
        ("flag_attn.sage_attention.quant_per_block", "per_block_int8"),
    ):
        module = importlib.import_module(module_name)
        assert hasattr(module, attribute), f"{module_name}.{attribute} is no longer importable"
    assert callable(flag_attn.forgetting_attention)
    assert flag_attn.forgetting_attention is flag_attn.parallel_forgetting_attn


@pytest.mark.parametrize(
    "vendor, device",
    [
        ("nvidia", "cuda"),
        ("amd", "cuda"),
        ("hygon", "cuda"),
        ("iluvatar", "cuda"),
        ("metax", "cuda"),
        ("thead", "cuda"),
        ("enflame", "gcu"),
        ("ascend", "npu"),
        ("cambricon", "mlu"),
        ("mthreads", "musa"),
        ("intel", "xpu"),
        ("cpu", "cpu"),
    ],
)
def test_explicit_vendor_device_mapping(vendor, device):
    detected = device_finder.DeviceDetector(vendor)
    assert detected.vendor_name == vendor
    assert detected.name == device


@pytest.mark.parametrize("variable", ["FLAG_ATTN_BACKEND", "FLAG_ATTN_VENDOR"])
def test_vendor_override(automatic_vendor, monkeypatch, variable):
    monkeypatch.setenv(variable, " _MeTaX ")
    assert device_finder.detect_vendor() == "metax"
    assert device_finder.DeviceDetector().name == "cuda"
    assert flag_attn.runtime.backend.is_metax_backend()


def test_backend_override_takes_precedence(automatic_vendor, monkeypatch):
    monkeypatch.setenv("FLAG_ATTN_BACKEND", "enflame")
    monkeypatch.setenv("FLAG_ATTN_VENDOR", "nvidia")
    assert device_finder.detect_vendor() == "enflame"


def test_invalid_vendor_override(automatic_vendor, monkeypatch):
    monkeypatch.setenv("FLAG_ATTN_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unknown FlagAttention backend 'typo'"):
        device_finder.detect_vendor()


def test_unavailable_cuda_wheel_is_not_a_gpu(automatic_vendor, monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(device_finder, "_is_available", lambda name: False)
    assert device_finder.detect_vendor() == "cpu"


@pytest.mark.parametrize(
    "available, vendor",
    [
        ("gcu", "enflame"),
        ("npu", "ascend"),
        ("mlu", "cambricon"),
        ("musa", "mthreads"),
        ("xpu", "intel"),
    ],
)
def test_available_native_backend(automatic_vendor, monkeypatch, available, vendor):
    monkeypatch.setattr(device_finder, "_is_available", lambda name: name == available)
    assert device_finder.detect_vendor() == vendor


@pytest.mark.parametrize(
    "device_name, triton_backend, hip_version, vendor",
    [
        ("NVIDIA H20", "cuda", None, "nvidia"),
        ("AMD Instinct MI300X", "hip", "6.3", "amd"),
        ("Hygon DCU", "hip", "6.3", "hygon"),
        ("MetaX C550", "cuda", None, "metax"),
        ("CUDA-compatible accelerator", "maca", None, "metax"),
        ("Iluvatar BI150", "cuda", None, "iluvatar"),
        ("PPU-ZW810E", "cuda", None, "thead"),
        ("Alixpu", "cuda", None, "thead"),
    ],
)
def test_cuda_compatible_vendors(
    automatic_vendor, monkeypatch, device_name, triton_backend, hip_version, vendor
):
    monkeypatch.setattr(device_finder, "_is_available", lambda name: name == "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: device_name)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    target = SimpleNamespace(backend=triton_backend)
    driver = SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target))
    monkeypatch.setattr(triton.runtime, "driver", driver)
    assert device_finder.detect_vendor() == vendor


def test_missing_triton_driver_does_not_break_metadata(automatic_vendor, monkeypatch):
    monkeypatch.setattr(device_finder, "_is_available", lambda name: name == "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA H20")
    monkeypatch.setattr(torch.version, "hip", None)

    def unavailable_target():
        raise RuntimeError("driver unavailable")

    driver = SimpleNamespace(
        active=SimpleNamespace(get_current_target=unavailable_target)
    )
    monkeypatch.setattr(triton.runtime, "driver", driver)
    assert device_finder.detect_vendor() == "nvidia"
