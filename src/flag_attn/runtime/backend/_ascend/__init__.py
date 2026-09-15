# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
from .attn_qk_int8_per_block import forward
from .quant_per_block import per_block_int8
from .ops import quant_per_block_int8

per_block_int8 = quant_per_block_int8

__all__ = ["forward", "quant_per_block_int8", "per_block_int8", "forward", "per_block_int8"]


def __getattr__(name):
    if name == "forward":
        from .sage_attention import forward

        return forward
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
