# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def full_precision_references():
    """Keep FP32 reference matmuls in FP32, including on TF32-default builds."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
