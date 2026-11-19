# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


def infllmv2_attention(*args, **kwargs):
    from .operator import infllmv2_attention as implementation

    return implementation(*args, **kwargs)


def infllmv2_decode(*args, **kwargs):
    from .operator import infllmv2_decode as implementation

    return implementation(*args, **kwargs)


__all__ = ["infllmv2_attention", "infllmv2_decode"]
