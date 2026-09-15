"""Ascend GDN2 implementation, separate from backend-wide common ops."""

def chunk_gdn2(*args, **kwargs):
    from .chunk_fwd_infer import chunk_gdn2_fwd_infer
    return chunk_gdn2_fwd_infer(*args, **kwargs)

__all__ = ["chunk_gdn2"]
