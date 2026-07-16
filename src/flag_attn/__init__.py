try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "0.0.0"
    version_tuple = (0, 0, 0)


from flag_attn import testing  # noqa: F401
from flag_attn.flash import attention as flash_attention  # noqa: F401
from flag_attn.paged import attention as paged_attention  # noqa: F401
from flag_attn.piecewise import attention as piecewise_attention  # noqa: F401
from flag_attn.split_kv import attention as flash_attention_split_kv  # noqa: F401
