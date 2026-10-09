"""Tests for LM Studio backend options carried in ``BackendConfig.options``.

@see [[src/nlp_shap/backends/lmstudio/options.py]]
"""

import pytest

from nlp_shap.backends.lmstudio import LmStudioBackend
from nlp_shap.backends.lmstudio.options import LmStudioOptions
from nlp_shap.errors import ConfigError
from nlp_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context


def test_lmstudio_options_defaults() -> None:
    options = LmStudioOptions()
    assert options.hub_search is None
    assert options.quantization == "Q4_K_M"
    assert options.auto_download is True
    assert options.serialize_generate is True


def test_lmstudio_backend_validates_options() -> None:
    backend = LmStudioBackend(
        backend_context(
            BackendConfig(
                kind="lmstudio",
                model_id="m",
                options={"serialize_generate": False, "quantization": "Q8_0"},
            )
        )
    )
    assert backend.options.serialize_generate is False
    assert backend.options.quantization == "Q8_0"


def test_lmstudio_backend_rejects_unknown_option() -> None:
    with pytest.raises(ConfigError, match="hub_serch"):
        LmStudioBackend(
            backend_context(
                BackendConfig(kind="lmstudio", model_id="m", options={"hub_serch": "x"})
            )
        )


def test_backend_config_rejects_lmstudio_fields_at_top_level() -> None:
    with pytest.raises(ValueError, match="serialize_generate"):
        BackendConfig.model_validate({
            "kind": "lmstudio",
            "model_id": "m",
            "serialize_generate": True,
        })


def test_backend_config_has_connection_defaults() -> None:
    config = BackendConfig(kind="api", model_id="m", api_host="http://mock/v1")
    assert config.timeout_s == 120.0
    assert config.max_retries == 3
    assert config.options == {}
