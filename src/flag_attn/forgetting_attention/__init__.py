"""Public Forgetting Attention exports."""
from importlib import import_module


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "naive_forgetting_attention":
        value = import_module(".naive", __name__).forgetting_attention
    else:
        value = getattr(import_module(".parallel", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "parallel_forgetting_attn", "forgetting_attention", "naive_forgetting_attention",
]
