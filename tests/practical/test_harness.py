"""Scenario catalogue, feature selection and model pins, without a model.

@see [[src/nlp_shap/pipeline/config.py]]
"""

import re

import pytest

from nlp_shap import ExplainConfig
from tests.practical._harness import (
    MODELS,
    SCENARIOS,
    WEIGHT_FILES,
    feature_selected,
    require_weights,
    tier_selected,
)


@pytest.mark.parametrize(
    ("spec", "feature", "selected"),
    [
        ("all", "backend.api", True),
        ("estimator.*", "estimator.neyman_cc", True),
        ("estimator.*", "estimand.banzhaf", False),
        ("conversation.multi_turn", "conversation.multi_turn", True),
        ("value.logprob,cli.*", "cli.reanalyze", True),
        ("value.logprob, cli.*", "value.logprob", True),
        ("", "value.logprob", False),
    ],
)
def test_feature_selection(spec: str, feature: str, selected: bool) -> None:
    """``--feature`` takes ``all`` or comma-separated fnmatch globs."""
    assert feature_selected(spec, feature) is selected


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_scenario_configs_validate(scenario_id: str) -> None:
    """Every scenario and variant builds a valid config and a snapshot."""
    scenario = SCENARIOS[scenario_id]
    model = MODELS[scenario.tier]
    configs = [scenario.config(model)] + [
        scenario.config(model, variant) for variant in scenario.variants
    ]
    assert all(isinstance(config, ExplainConfig) for config in configs)
    if configs[0].backend.kind == "transformers":
        assert configs[0].backend.options["revision"] == model.revision
    assert scenario.snapshot.messages()


@pytest.mark.parametrize(
    ("spec", "feature", "selected"),
    [
        ("quick", "estimator.exact", True),
        ("quick", "deep.estimator.reference", False),
        ("deep", "deep.estimator.reference", True),
        ("deep", "estimator.exact", False),
        ("all", "deep.cli.parity", True),
        ("all", "backend.api", True),
    ],
)
def test_tier_selection(spec: str, feature: str, selected: bool) -> None:
    """``--tier`` takes quick, deep or all; ``deep.*`` ids form the deep tier."""
    assert tier_selected(spec, feature) is selected


def test_models_are_pinned_small_and_typed() -> None:
    """Chat tiers load in float16 (quick < 4 GB, deep < 8 GB); the CTC model in
    float32 under 1 GB. Every model pins a commit sha."""
    assert set(MODELS) == {"default", "fast", "deep", "alignment"}
    for name, spec in MODELS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.dtype == ("float32" if name == "alignment" else "float16")
        assert spec.size_gb < {"deep": 8.0, "alignment": 1.0}.get(name, 4.0)


def test_deep_scenarios_use_the_deep_model() -> None:
    """Every ``deep.*`` scenario runs on the deep tier's model and no other does."""
    for sid, scenario in SCENARIOS.items():
        assert (scenario.tier == "deep") is sid.startswith("deep.")


def test_null_in_a_variant_deletes_the_key() -> None:
    """The long thread's token variant drops the messages partition's options."""
    scenario = SCENARIOS["deep.conversation.long"]
    config = scenario.config(MODELS["deep"], scenario.variants[0])
    assert config.explanation.plugin_options == {"focus_tokens": {"message_index": 5}}


def test_require_weights_checks_only_the_files_the_backend_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot without README or LICENSE (transformers skips them) is cached."""
    hub = pytest.importorskip("huggingface_hub")
    calls: list[object] = []

    def snapshot_download(*args: object, **kwargs: object) -> str:
        calls.append(kwargs.get("allow_patterns"))
        if kwargs.get("allow_patterns") is None:
            msg = "README.md, LICENSE missing"
            raise hub.errors.LocalEntryNotFoundError(msg)
        return "/cache"

    monkeypatch.delenv("NLP_SHAP_PRACTICAL_DOWNLOAD", raising=False)
    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    require_weights(MODELS["deep"])
    assert calls == [list(WEIGHT_FILES)]
