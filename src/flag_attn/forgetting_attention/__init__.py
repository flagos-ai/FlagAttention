"""Public Forgetting Attention exports."""
import importlib


def __getattr__(name):
    if name in ("parallel_forgetting_attn", "forgetting_attention"):
        from flag_attn import parallel_forgetting_attn

        value = parallel_forgetting_attn
    elif name == "naive_forgetting_attention":
        value = importlib.import_module(".naive", __name__).forgetting_attention
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "parallel_forgetting_attn", "forgetting_attention", "naive_forgetting_attention",
]
