"""Scenario catalogue, feature selection and model pins, without a model.

@see [[src/mllm_shap/pipeline/config.py]]
"""

import re

import pytest

from mllm_shap import ExplainConfig
from tests.practical._harness import (
    LMS_HOST,
    MODELS,
    SCENARIOS,
    WEIGHT_FILES,
    feature_selected,
    require_lmstudio,
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
    """In-process tiers pin a commit sha: chat tiers float16 (quick < 4 GB, deep_hf
    < 8 GB), the CTC model float32 under 1 GB. The deep tier is an LM Studio key."""
    assert set(MODELS) == {
        "default",
        "fast",
        "deep",
        "deep_judge",
        "deep_hf",
        "alignment",
    }
    assert MODELS["deep"].kind == "api"
    for name, spec in MODELS.items():
        if spec.kind == "api":
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.dtype == ("float32" if name == "alignment" else "float16")
        assert spec.size_gb < {"deep_hf": 8.0, "alignment": 1.0}.get(name, 4.0)


def test_deep_scenarios_use_the_deep_tiers() -> None:
    """Every ``deep.*`` scenario runs on LM Studio, except the in-process KV cache."""
    for sid, scenario in SCENARIOS.items():
        expected = {"deep.runtime.kv_cache": "deep_hf"}.get(sid, "deep")
        if sid.startswith("deep."):
            assert scenario.tier == expected
        else:
            assert not scenario.tier.startswith("deep")


def test_deep_backend_is_the_lmstudio_server() -> None:
    """The deep tier posts to LM Studio's OpenAI-compatible server by model key."""
    config = SCENARIOS["deep.estimator.reference"].config(MODELS["deep"])
    assert config.backend.kind == "api"
    assert config.backend.api_host == LMS_HOST
    assert config.backend.model_id == MODELS["deep"].model_id


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

    monkeypatch.delenv("MLLM_SHAP_PRACTICAL_DOWNLOAD", raising=False)
    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    require_weights(MODELS["deep_hf"])
    assert calls == [list(WEIGHT_FILES)]


def test_deep_tier_skips_when_lmstudio_is_not_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No server on the port is a skip with the fix, not an error."""
    httpx = pytest.importorskip("httpx")

    def refuse(*args: object, **kwargs: object) -> None:
        msg = "connection refused"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(httpx, "get", refuse)
    with pytest.raises(pytest.skip.Exception, match="lms server start"):
        require_lmstudio(MODELS["deep"])
