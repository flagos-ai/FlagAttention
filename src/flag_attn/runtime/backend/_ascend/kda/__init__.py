# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ascend Kimi Delta Attention inference."""

__all__ = ["chunk_kda"]


def __getattr__(name):
    if name == "chunk_kda":
        from .forward import chunk_kda

        globals()[name] = chunk_kda
        return chunk_kda
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
