"""Tests for ExplainConfig YAML round-trip.

@see [[src/mllm_shap/pipeline/config.py]]
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mllm_shap.domain.enums import EmbeddingMode
from mllm_shap.domain.estimands import Estimand
from mllm_shap.errors import ConfigError
from mllm_shap.pipeline.config import (
    BackendConfig,
    NeymanConfig,
    explain_config_from_yaml,
    explain_config_to_yaml,
)

FIXTURE = Path(__file__).parent / "fixtures" / "explain_config.yaml"


def test_explain_config_yaml_round_trip_matches_fixture() -> None:
    """YAML load and dump preserve the Appendix A skeleton."""
    original_text = FIXTURE.read_text(encoding="utf-8")
    config = explain_config_from_yaml(original_text)
    restored = explain_config_from_yaml(explain_config_to_yaml(config))
    assert restored == config


def test_explain_config_parses_appendix_a_fields() -> None:
    """Nested sections deserialize to typed models."""
    config = explain_config_from_yaml(FIXTURE.read_text(encoding="utf-8"))
    assert config.backend.kind == "lmstudio"
    assert config.backend.model_id == "qwen2-500m-instruct"
    assert config.generation.max_new_tokens == 128
    assert config.explanation.estimand is Estimand.SHAPLEY
    assert config.explanation.budget.fraction == 0.3
    assert config.explanation.include_minimal_masks is False
    assert config.explanation.embedding_mode is EmbeddingMode.STATIC
    assert config.explanation.seed == 42


def test_explain_config_rejects_unknown_fields() -> None:
    """Extra YAML keys are rejected to keep the schema strict."""
    text = FIXTURE.read_text(encoding="utf-8") + "\nunknown_root: true\n"
    with pytest.raises(ValidationError, match="unknown_root"):
        explain_config_from_yaml(text)


def test_kv_cache_defaults_off_with_bounded_entries() -> None:
    from mllm_shap.pipeline.config import KvCacheConfig

    assert KvCacheConfig().enabled is False
    assert KvCacheConfig().max_entries == 8


def test_explain_config_from_yaml_raises_config_error_for_inconsistent_settings() -> (
    None
):
    """Cross-field checks surface as ConfigError, not pydantic's ValidationError."""
    text = (
        "backend: {kind: mock, model_id: stub}\n"
        "explanation: {estimator: exact, budget: {fraction: 0.5}}\n"
    )
    with pytest.raises(ConfigError, match="estimator 'exact'") as excinfo:
        explain_config_from_yaml(text)
    assert not isinstance(excinfo.value, ValidationError)


@pytest.mark.parametrize("estimator", ["complementary", "neyman_cc"])
def test_complementary_estimators_reject_banzhaf(estimator: str) -> None:
    text = (
        "backend: {kind: mock, model_id: stub}\n"
        f"explanation: {{estimator: {estimator}, estimand: banzhaf}}\n"
    )
    with pytest.raises(ConfigError, match="shapley"):
        explain_config_from_yaml(text)


@pytest.mark.parametrize(
    "neyman",
    [
        {"initial_fraction": -3},
        {"initial_fraction": 0.0},
        {"initial_fraction": 1.5},
        {"initial_num_samples": 0},
    ],
)
def test_neyman_options_are_range_checked(neyman: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="initial_"):
        NeymanConfig.model_validate(neyman)


def test_neyman_options_accept_valid_values() -> None:
    config = NeymanConfig(initial_fraction=1.0, initial_num_samples=1)
    assert (config.initial_fraction, config.initial_num_samples) == (1.0, 1)


@pytest.mark.parametrize("text", ["", "[1, 2]\n", "just text\n"])
def test_explain_config_from_yaml_refuses_non_mapping_root(text: str) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        explain_config_from_yaml(text)


def test_max_players_is_capped_at_sixty_two() -> None:
    """2^max_players must fit a signed 64-bit integer for the samplers' masks."""
    text = FIXTURE.read_text(encoding="utf-8").replace(
        "explanation:\n", "explanation:\n  max_players: 63\n", 1
    )
    with pytest.raises(ValidationError, match="max_players"):
        explain_config_from_yaml(text)


def test_api_backend_requires_api_host() -> None:
    """The ``api`` backend has no environment or localhost fallback for its host."""
    text = "backend: {kind: api, model_id: m}\n"
    with pytest.raises(ConfigError, match=r"backend\.api_host") as excinfo:
        explain_config_from_yaml(text)
    assert not isinstance(excinfo.value, ValidationError)
    assert BackendConfig(kind="mock", model_id="m").api_host is None
