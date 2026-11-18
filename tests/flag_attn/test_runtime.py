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

from types import SimpleNamespace

import pytest
import torch
import triton

import flag_attn
from flag_attn.runtime.backend import arch_finder, device_finder, op_registry


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


@pytest.mark.parametrize(
    "capability, arch",
    [
        ((9, 0), "hopper"),
        ((10, 0), "blackwell"),
        ((12, 0), "blackwell"),
        ((8, 0), None),
        ((7, 5), None),
    ],
)
def test_nvidia_arch_detection(monkeypatch, capability, arch):
    monkeypatch.delenv("FLAG_ATTN_ARCH", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    arch_finder.detect_arch.cache_clear()
    assert arch_finder.detect_arch("nvidia") == arch


def test_non_nvidia_vendors_declare_no_arch(monkeypatch):
    monkeypatch.delenv("FLAG_ATTN_ARCH", raising=False)
    arch_finder.detect_arch.cache_clear()
    for vendor in ("metax", "hygon", "enflame", "ascend", "mthreads", "cpu"):
        assert arch_finder.detect_arch(vendor) is None


def test_arch_override(monkeypatch):
    monkeypatch.setenv("FLAG_ATTN_ARCH", "hopper")
    arch_finder.detect_arch.cache_clear()
    # The override applies without a matching GPU present, and to any vendor.
    assert arch_finder.detect_arch("nvidia") == "hopper"
    assert arch_finder.detect_arch("metax") == "hopper"


def test_arch_detection_survives_a_missing_device(monkeypatch):
    monkeypatch.delenv("FLAG_ATTN_ARCH", raising=False)

    def unavailable(*_):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    arch_finder.detect_arch.cache_clear()
    assert arch_finder.detect_arch("nvidia") is None


def test_arch_exports_overlay_vendor_exports():
    """An arch table wins for the names it declares and changes nothing else."""
    vendor_exports = op_registry._exports_of("flag_attn.runtime.backend._metax")
    assert vendor_exports, "expected the MetaX backend to declare operators"
    # No MetaX arch subpackage exists, so an arch argument must be a no-op.
    assert op_registry._backend_exports("metax", None) == vendor_exports
    assert op_registry._backend_exports("metax", "blackwell") == vendor_exports


def test_missing_arch_package_keeps_vendor_selection():
    exports = op_registry._backend_exports("nvidia", "does_not_exist")
    assert exports == op_registry._backend_exports("nvidia", None)


def test_invalid_arch_name_is_rejected():
    with pytest.raises(ValueError, match="Invalid arch name"):
        op_registry._backend_exports("nvidia", "not an identifier")


def test_generic_operator_resolves_without_an_arch_override():
    """chunk_kda must resolve to the generic pure-Triton module off Hopper."""
    resolved = op_registry.resolve_operator(
        "chunk_kda_fwd_infer",
        "flag_attn.FLA.kda.chunk_kda",
        vendor="nvidia",
        arch=None,
    )
    assert resolved.__module__ == "flag_attn.FLA.kda.chunk_kda"
