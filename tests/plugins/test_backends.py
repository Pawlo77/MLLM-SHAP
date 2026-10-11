"""Backend construction from the plugin context.

@see [[src/mllm_shap/plugins/backends.py]]
"""

import pytest

from mllm_shap import ExplainConfig, PluginContext, PluginGroup
from mllm_shap.backends.mock import MockBackend
from mllm_shap.errors import PluginNotFoundError
from mllm_shap.plugins import default_registry
from mllm_shap.plugins.backends import instantiate_backend


def _context(kind: str, model_id: str = "stub") -> PluginContext:
    return PluginContext(
        config=ExplainConfig.model_validate({
            "backend": {"kind": kind, "model_id": model_id}
        }),
        registry=default_registry(),
    )


def test_instantiate_backend_uses_config_kind_and_model() -> None:
    backend = instantiate_backend(_context("mock", model_id="tiny"))

    assert isinstance(backend, MockBackend)
    assert backend.model_id == "tiny"


def test_instantiate_backend_resolves_custom_registration() -> None:
    context = _context("custom")
    built: list[PluginContext] = []

    def factory(plugin_context: PluginContext) -> MockBackend:
        built.append(plugin_context)
        return MockBackend(model_id="custom-model")

    context.registry.register(PluginGroup.BACKENDS, "custom", factory)

    assert instantiate_backend(context).model_id == "custom-model"
    assert built[0].config is context.config


def test_instantiate_backend_unknown_kind_raises() -> None:
    with pytest.raises(PluginNotFoundError):
        instantiate_backend(_context("does-not-exist"))
