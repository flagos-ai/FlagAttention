# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ascend gated linear attention."""

__all__ = ["chunk_gla"]


def __getattr__(name):
    if name == "chunk_gla":
        from .chunk import chunk_gla

        globals()[name] = chunk_gla
        return chunk_gla
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
