# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Generic operators shared by the Ascend backend implementations."""

from .quantization import quant_per_block_int8

__all__ = ["quant_per_block_int8"]
