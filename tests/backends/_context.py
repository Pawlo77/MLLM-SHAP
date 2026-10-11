"""Helpers that build plugin contexts for backend tests."""

from mllm_shap.pipeline.config import BackendConfig, ExplainConfig
from mllm_shap.plugins import PluginContext


def backend_context(backend: BackendConfig) -> PluginContext:
    """Return a plugin context whose config uses ``backend``."""
    return PluginContext(config=ExplainConfig(backend=backend))
