from flag_attn.runtime.backend import resolve_operator

_OPERATOR_EXPORTS = {"chunk_gdn2": ("chunk_gdn2", "flag_attn.gdn2.chunk", "chunk_gdn2")}


def __getattr__(name: str):
    try:
        operator, module, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = resolve_operator(operator, module, symbol)
    globals()[name] = value
    return value


__all__ = ["chunk_gdn2"]
