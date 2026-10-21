# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-capable dispatch and contract tests; these do not validate NPU kernels."""

import ast
import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flag_attn.runtime.backend import op_registry


@pytest.fixture(autouse=True)
def clear_registry():
    op_registry._backend_exports.cache_clear()
    op_registry._resolve_operator.cache_clear()
    yield
    op_registry._backend_exports.cache_clear()
    op_registry._resolve_operator.cache_clear()


@pytest.mark.parametrize("name", [
    "chunk_gdn2", "chunk_gla", "chunk_kda_fwd_infer",
    "sage_attention", "parallel_nsa_compression",
    "minimax_m3_sparse_attn", "minimax_m3_index_topk",
])
def test_ascend_declared_override_resolves_without_generic(monkeypatch, name):
    module_name, symbol = op_registry._backend_exports("ascend")[name]
    module = ModuleType(module_name)
    sentinel = Mock()
    setattr(module, symbol, sentinel)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert op_registry.resolve_operator(name, "missing_generic", vendor="ascend") is sentinel


def test_undeclared_operators_use_generic(monkeypatch):
    module = ModuleType("migration_generic")
    module.parallel_nsa = Mock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert "parallel_nsa" not in op_registry._backend_exports("ascend")
    assert "minimax_m3_sparse_attn_decode" not in op_registry._backend_exports("ascend")
    assert op_registry.resolve_operator("parallel_nsa", module.__name__, vendor="ascend") is module.parallel_nsa


def test_declared_dependency_error_is_not_silenced(monkeypatch):
    real_import = op_registry.import_module

    def fail_import(name):
        if name.endswith("gdn2.api"):
            raise ModuleNotFoundError("required Ascend dependency is missing", name="ascend_dependency")
        return real_import(name)

    monkeypatch.setattr(op_registry, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError, match="required Ascend dependency"):
        op_registry.resolve_operator("chunk_gdn2", "missing_generic", vendor="ascend")


@pytest.mark.parametrize("relative,name,generic", [
    ("gdn2.api", "chunk_gdn2", "gdn2/chunk.py"),
    ("kda.api", "chunk_kda_fwd_infer", "FLA/chunk_kda.py"),
    ("nsa.api", "parallel_nsa_compression", "parallel_nsa/parallel_nsa_compression.py"),
])
def test_adapters_keep_public_parameters(relative, name, generic):
    adapter = getattr(importlib.import_module(f"flag_attn.runtime.backend._ascend.{relative}"), name)
    root = Path(__file__).resolve().parents[2] / "src/flag_attn"
    tree = ast.parse((root / generic).read_text(encoding="utf-8"))
    signature = next(node.args for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assert list(inspect.signature(adapter).parameters) == [
        arg.arg for arg in signature.posonlyargs + signature.args + signature.kwonlyargs
    ]


def test_gdn2_training_rejected_before_loading_kernel():
    from flag_attn.runtime.backend._ascend.gdn2.api import chunk_gdn2

    q = torch.ones(1, requires_grad=True)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="only supports inference"):
        chunk_gdn2(q, q, q, q, q, q)


def test_gdn2_adapter_preserves_options(monkeypatch):
    from flag_attn.runtime.backend._ascend.gdn2.api import chunk_gdn2

    module_name = "flag_attn.runtime.backend._ascend.gdn2.chunk_fwd_infer"
    module = ModuleType(module_name)
    state = torch.arange(6).reshape(1, 1, 2, 3)
    history = state.unsqueeze(1)
    module.chunk_gdn2_fwd_infer = Mock(return_value=("output", state, history))
    monkeypatch.setitem(sys.modules, module_name, module)
    with torch.no_grad():
        result = chunk_gdn2(1, 2, 3, 4, 5, 6, state_v_first=True, return_intermediate_states=True)
    assert result[0] == "output"
    torch.testing.assert_close(result[1], state.transpose(-1, -2))
    torch.testing.assert_close(result[2], history.transpose(-1, -2))
    assert module.chunk_gdn2_fwd_infer.call_args.kwargs["state_v_first"] is False
    assert module.chunk_gdn2_fwd_infer.call_args.kwargs["return_intermediate_states"] is True


def test_kda_rejects_training_and_unsupported_chunk_indices():
    from flag_attn.runtime.backend._ascend.kda.api import chunk_kda_fwd_infer

    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference"):
        chunk_kda_fwd_infer(1, 2, 3, 4, 5)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="chunk_indices"):
        chunk_kda_fwd_infer(1, 2, 3, 4, 5, chunk_indices=object())


def test_shared_primitives_are_not_public_overrides():
    exports = op_registry._backend_exports("ascend")
    assert not {"chunk_local_cumsum", "mean_pooling", "parallel_attn_bwd_preprocess"} & exports.keys()


def test_ascend_forward_state_config_is_loadable():
    from flag_attn.runtime.configs_loader import TunedConfigLoader

    configs = TunedConfigLoader("ascend").get_tuned_config("chunk_fwd_kernel_h")
    assert configs, "Ascend forward state kernel requires explicit BK/BV configs"
    for config in configs:
        assert config.kwargs["BK"] in (32, 64)
        assert config.kwargs["BV"] in (32, 64)
        assert config.num_warps in (1, 2, 4, 8)
        assert config.num_stages > 0


@pytest.mark.parametrize("grad_input", [0, 1, 2])
def test_nsa_adapter_rejects_training_before_loading_kernel(grad_input):
    from flag_attn.runtime.backend._ascend.nsa.api import parallel_nsa_compression

    inputs = [torch.ones(1, requires_grad=index == grad_input) for index in range(3)]
    with torch.enable_grad(), pytest.raises(RuntimeError, match="forward-only"):
        parallel_nsa_compression(*inputs)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_nsa_public_facade_and_namespace_use_adapter(monkeypatch, context):
    import flag_attn
    from flag_attn.runtime.backend._ascend.nsa.api import parallel_nsa_compression as adapter

    namespace = importlib.import_module("flag_attn.parallel_nsa")
    for module in (namespace, flag_attn):
        monkeypatch.delattr(module, "parallel_nsa_compression", raising=False)
    implementation = ModuleType("flag_attn.runtime.backend._ascend.nsa.compression")
    implementation.parallel_nsa_compression = Mock(return_value=("output", "lse"))
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)
    monkeypatch.setattr(importlib.import_module("flag_attn.runtime"), "device", SimpleNamespace(vendor_name="ascend"))
    from flag_attn.parallel_nsa import parallel_nsa_compression

    assert parallel_nsa_compression is adapter
    assert flag_attn.parallel_nsa_compression is adapter
    inputs = [torch.ones(1, requires_grad=True) for _ in range(3)]
    with context():
        assert parallel_nsa_compression(*inputs) == ("output", "lse")
        assert flag_attn.parallel_nsa(*inputs, mode="compression") == ("output", "lse")
    assert implementation.parallel_nsa_compression.call_count == 2
    assert "parallel_nsa" not in op_registry._backend_exports("ascend")


def test_nsa_declared_dependency_error_propagates(monkeypatch):
    real_import = op_registry.import_module

    def fail_import(name):
        if name.endswith("nsa.api"):
            raise ModuleNotFoundError("required NSA dependency", name="nsa_dependency")
        return real_import(name)

    monkeypatch.setattr(op_registry, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError, match="required NSA dependency"):
        op_registry.resolve_operator("parallel_nsa_compression", "missing_generic", vendor="ascend")
