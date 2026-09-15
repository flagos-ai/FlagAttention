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

"""Vendor metadata with the same device/name contract as FlagGems."""

import os
from dataclasses import dataclass

import torch

from ..accelerator import _is_available, _load_backend

_VENDOR_DEVICES = {
    "nvidia": "cuda",
    "amd": "cuda",
    "hygon": "cuda",
    "iluvatar": "cuda",
    "metax": "cuda",
    "enflame": "gcu",
    "ascend": "npu",
    "cambricon": "mlu",
    "mthreads": "musa",
    "intel": "xpu",
    "cpu": "cpu",
}

_TRITON_VENDORS = {
    "maca": "metax",
    "metax": "metax",
    "hygon": "hygon",
    "dcu": "hygon",
    "iluvatar": "iluvatar",
}


def _normalize_vendor(value: str) -> str:
    vendor = value.strip().lower().removeprefix("_")
    if vendor not in _VENDOR_DEVICES:
        raise ValueError(
            f"Unknown FlagAttention backend {value!r}; "
            f"expected one of: {', '.join(_VENDOR_DEVICES)}"
        )
    return vendor


def _cuda_vendor() -> str:
    # Several vendors expose CUDA APIs. A CUDA-enabled wheel alone does not
    # identify NVIDIA; use the device name and active compiler target first.
    try:
        name = str(torch.cuda.get_device_name()).lower()
    except (AssertionError, OSError, RuntimeError):
        name = ""
    for vendor, names in (
        ("metax", ("metax", "曦云")),
        ("hygon", ("hygon", "dcu")),
        ("iluvatar", ("iluvatar", "天数")),
    ):
        if any(token in name for token in names):
            return vendor
    try:
        import triton

        target = triton.runtime.driver.active.get_current_target()
        vendor = _TRITON_VENDORS.get(str(target.backend).lower())
        if vendor:
            return vendor
    except Exception:
        # Driver initialization is optional for metadata inspection.
        pass
    return "amd" if getattr(torch.version, "hip", None) else "nvidia"


def detect_vendor() -> str:
    """Honor explicit selection, then discover an available Torch backend.

    ``FLAG_ATTN_BACKEND`` retains its existing precedence. The synonymous
    ``FLAG_ATTN_VENDOR`` accepts vendor names such as ``nvidia`` or ``metax``.
    CPU-only hosts can inspect the package without claiming a GPU is present.
    """
    for variable in ("FLAG_ATTN_BACKEND", "FLAG_ATTN_VENDOR"):
        value = os.environ.get(variable)
        if value:
            return _normalize_vendor(value)

    if _is_available("cuda"):
        return _cuda_vendor()
    for vendor in ("enflame", "ascend", "cambricon", "mthreads", "intel"):
        if _is_available(_VENDOR_DEVICES[vendor]):
            return vendor
    return "cpu"


@dataclass(frozen=True, init=False)
class DeviceDetector:
    """Selected vendor and Torch device type, following FlagGems' public API."""

    vendor_name: str
    name: str

    def __init__(self, vendor_name: str | None = None):
        vendor = (
            detect_vendor() if vendor_name is None else _normalize_vendor(vendor_name)
        )
        object.__setattr__(self, "vendor_name", vendor)
        object.__setattr__(self, "name", _VENDOR_DEVICES[vendor])

    @property
    def torch_device_fn(self):
        return _load_backend(self.name)

    def get_vendor_name(self) -> str:
        return self.vendor_name
