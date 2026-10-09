"""Shapley axioms verified through the public explain pipeline.

@see [[src/nlp_shap/pipeline/__init__.py]]
"""

import pytest

from nlp_shap import ExplainConfig, ExplainRunner, PluginGroup, PluginRegistry
from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.estimation.estimands._weights import aggregate_shapley_exact
from nlp_shap.plugins import default_registry
from nlp_shap.protocols.backend import GenerationResult
from tests.estimation._games import full_table


class EchoBackend:
    """Return the rendered prompt text so coalition payoffs are inspectable."""

    def __init__(self, context: object | None = None) -> None:
        self._model_id = "echo"

    @property
    def model_id(self) -> str:
        return self._model_id

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        text = " ".join(
            message.text for turn in snapshot.turns for message in turn.messages
        )
        return GenerationRecord(text=text)


class CountValue:
    """v(S) = number of tokens from a fixed vocabulary present in the output."""

    vocabulary = frozenset({"alpha", "gamma"})

    def __init__(self, context: object | None = None) -> None:
        _ = context

    @property
    def name(self) -> str:
        return "count_test"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        return float(
            sum(1 for token in candidate.text.split() if token in self.vocabulary)
        )


def _registry() -> PluginRegistry:
    registry = default_registry()
    registry.register(PluginGroup.BACKENDS, "echo", EchoBackend)
    registry.register(PluginGroup.VALUE_FNS, "count_test", CountValue)
    return registry


def _config(estimator: str, fraction: float = 1.0) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "echo", "model_id": "echo"},
        "explanation": {
            "estimator": estimator,
            "value_fn": "count_test",
            "absence_policy": "delete",
            "budget": {"fraction": fraction},
            "archive": {"path": ""},
            "seed": 1,
        },
    })


def _snapshot(text: str) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


@pytest.mark.parametrize("estimator", ["exact", "mc", "complementary", "neyman_cc"])
def test_pipeline_recovers_additive_game(estimator: str) -> None:
    output = ExplainRunner(_config(estimator), registry=_registry()).explain_sync(
        _snapshot("alpha beta gamma delta")
    )
    assert output.result.values == pytest.approx([1.0, 0.0, 1.0, 0.0], abs=1e-9)
    assert output.result.base_value == pytest.approx(0.0)
    assert output.result.grand_value == pytest.approx(2.0)


@pytest.mark.parametrize("estimator", ["exact", "complementary", "neyman_cc"])
def test_pipeline_efficiency(estimator: str) -> None:
    output = ExplainRunner(_config(estimator), registry=_registry()).explain_sync(
        _snapshot("alpha beta gamma delta epsilon")
    )
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )


def test_pipeline_exact_executes_each_coalition_once() -> None:
    output = ExplainRunner(_config("exact"), registry=_registry()).explain_sync(
        _snapshot("alpha beta gamma")
    )
    assert output.metrics.requested == 8
    assert output.metrics.executed == 7
    assert output.metrics.cache_hits == 1


@pytest.mark.parametrize("estimator", ["exact", "mc", "complementary", "neyman_cc"])
def test_pipeline_single_player(estimator: str) -> None:
    output = ExplainRunner(_config(estimator), registry=_registry()).explain_sync(
        _snapshot("alpha")
    )
    assert len(output.result.values) == 1
    assert output.result.values[0] == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )


def test_pipeline_neyman_runs_second_phase() -> None:
    output = ExplainRunner(_config("neyman_cc"), registry=_registry()).explain_sync(
        _snapshot("a b c d e f")
    )
    assert 48 <= output.metrics.requested <= 64  # phase two ran; dedup may skip a few


class SynergyValue(CountValue):
    """Non-additive game: count plus a bonus when alpha and gamma co-occur."""

    @property
    def name(self) -> str:
        return "synergy_test"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        tokens = set(candidate.text.split())
        bonus = 3.0 if {"alpha", "gamma"} <= tokens else 0.0
        return super().score(base, candidate) + bonus


def _synergy_reference(words: list[str]) -> list[float]:
    """Exact Shapley values of the synergy game over the full table."""
    masks = full_table(len(words))
    payoffs = []
    for mask in masks:
        present = {word for word, keep in zip(words, mask, strict=True) if keep}
        count = len(present & {"alpha", "gamma"})
        payoffs.append(count + (3.0 if {"alpha", "gamma"} <= present else 0.0))
    return aggregate_shapley_exact(masks, payoffs)


@pytest.mark.parametrize("kv_cache", [True, False])
@pytest.mark.parametrize("estimator", ["exact", "mc", "complementary", "neyman_cc"])
def test_pipeline_matches_exact_reference_on_non_additive_game(
    estimator: str, kv_cache: bool
) -> None:
    registry = _registry()
    registry.register(PluginGroup.VALUE_FNS, "synergy_test", SynergyValue)
    payload = _config(estimator).model_dump(mode="python")
    payload["explanation"]["value_fn"] = "synergy_test"
    payload["explanation"]["kv_cache"] = {"enabled": kv_cache}
    words = ["alpha", "beta", "gamma", "delta"]
    output = ExplainRunner(
        ExplainConfig.model_validate(payload), registry=registry
    ).explain_sync(_snapshot(" ".join(words)))
    assert output.result.values == pytest.approx(_synergy_reference(words), abs=1e-9)
    assert output.result.values[0] == pytest.approx(2.5)
