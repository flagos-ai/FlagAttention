from flag_attn.FLA.gated_delta_rule.api import chunk_gated_delta_rule
from flag_attn.FLA.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

__all__ = [
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
]


def fused_recurrent_gated_delta_rule_fwd(*args, **kwargs):
    from .fused_recurrent import fused_recurrent_gated_delta_rule_fwd as impl

    return impl(*args, **kwargs)


def fused_recurrent_gated_delta_rule_packed_decode(*args, **kwargs):
    from .packed_decode import fused_recurrent_gated_delta_rule_packed_decode as impl

    return impl(*args, **kwargs)
