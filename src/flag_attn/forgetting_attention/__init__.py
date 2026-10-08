"""Forgetting Attention / ACP: reference and H100 TLE implementations."""
import importlib
import importlib.util


def has_tle():
    try:
        return importlib.util.find_spec("triton.experimental.tle") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def __getattr__(name):
    if name == "forgetting_attention":
        if not has_tle():
            raise RuntimeError(
                "Forgetting Attention requires Triton 3.6 with compatible FlagTree/TLE"
            )
        module = ".parallel"
    elif name == "naive_forgetting_attention":
        module = ".naive"
    else:
        raise AttributeError(name)
    value = importlib.import_module(module, __name__).forgetting_attention
    globals()[name] = value
    return value


__all__ = ["forgetting_attention", "naive_forgetting_attention", "has_tle"]
