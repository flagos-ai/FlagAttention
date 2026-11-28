# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class GenericDecodePolicy:
    quant_type: str
    mtp: int

    @property
    def label(self):
        return "triton"
