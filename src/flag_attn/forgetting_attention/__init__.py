"""Public Forgetting Attention exports."""
import importlib

from .parallel import forgetting_attention, parallel_forgetting_attn


def __getattr__(name):
    if name != "naive_forgetting_attention":
        raise AttributeError(name)
    value = importlib.import_module(".naive", __name__).forgetting_attention
    globals()[name] = value
    return value


__all__ = [
    "parallel_forgetting_attn", "forgetting_attention", "naive_forgetting_attention",
]
