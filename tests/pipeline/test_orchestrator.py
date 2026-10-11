"""Direct tests for the explain orchestrator.

@see [[src/mllm_shap/pipeline/orchestrator.py]]
"""

import asyncio
from collections.abc import Iterator, Sequence

import pytest

from mllm_shap import ExplainConfig, PluginGroup, PluginRegistry
from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.domain.players import PlayerSet
from mllm_shap.masking.partitions import TokenPartitioner
from mllm_shap.pipeline.context import ExplainContext
from mllm_shap.pipeline.orchestrator import ExplainOrchestrator
from mllm_shap.pipeline.result import ExplainRunOutput
from mllm_shap.plugins import default_registry
from mllm_shap.protocols.estimand import EstimandAggregator
from mllm_shap.protocols.estimator import TwoPhaseEstimator
from tests.pipeline._doubles import WordCountValue

_WORDS = ("alpha", "beta", "gamma")


class RecordingBackend:
    """Echo the rendered prompt and remember every prompt it received."""

    def __init__(self, context: object | None = None) -> None:
        _ = context
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "recording"

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        text = " ".join(message.text for message in snapshot.messages())
        self.prompts.append(text)
        return GenerationRecord(text=text)


class SingleMaskEstimator:
    """Sample one interior mask; record what the orchestrator aggregates."""

    def __init__(self, context: object | None = None) -> None:
        _ = context
        self.seen: dict[tuple[bool, ...], float] = {}

    @property
    def name(self) -> str:
        return "single"

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        _ = snapshot

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        yield CoalitionMask.from_sequence((True, False, False))

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        self.seen = dict(zip((mask.present for mask in masks), payoffs, strict=True))
        return [0.0] * len(masks[0].present)


class FakeTwoPhaseEstimator(SingleMaskEstimator):
    """Phase one samples one mask; phase two adds two more."""

    phase_two = ((False, True, False), (True, True, False))

    def __init__(self, context: object | None = None) -> None:
        super().__init__(context)
        self.phase_one: dict[tuple[bool, ...], float] = {}

    def begin_allocation(
        self, masks: Sequence[CoalitionMask], payoffs: Sequence[float]
    ) -> None:
        self.phase_one = dict(
            zip((mask.present for mask in masks), payoffs, strict=True)
        )

    def sample_allocation_masks(self) -> Iterator[CoalitionMask]:
        for present in self.phase_two:
            yield CoalitionMask.from_sequence(present)


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=" ".join(_WORDS)),)),
    ))


def _registry(estimator: SingleMaskEstimator) -> PluginRegistry:
    registry = default_registry()
    registry.register(PluginGroup.ESTIMATORS, "fake", lambda _context: estimator)
    registry.register(PluginGroup.VALUE_FNS, "word_count", WordCountValue)
    return registry


def _run(
    estimator_name: str,
    registry: PluginRegistry,
    backend: RecordingBackend,
) -> ExplainRunOutput:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "recording"},
        "explanation": {
            "estimator": estimator_name,
            "value_fn": "word_count",
            "absence_policy": "delete",
            "archive": {"path": ""},
        },
    })
    snapshot = _snapshot()
    context = ExplainContext(
        config=config,
        snapshot=snapshot,
        player_set=TokenPartitioner().partition(snapshot),
        registry=registry,
        run_id="orchestrator-test",
    )
    return asyncio.run(ExplainOrchestrator(context, backend).run())


def test_orchestrator_renders_one_job_per_exact_coalition() -> None:
    """Exact runs send every coalition but the cached grand one to the backend."""
    backend = RecordingBackend()
    output = _run("exact", default_registry_with_value(), backend)

    # First prompt is the base (grand) generation; the grand job is a cache hit.
    assert backend.prompts[0] == "alpha beta gamma"
    expected = {
        " ".join(word for word, keep in zip(_WORDS, mask, strict=True) if keep)
        for mask in _all_masks()
        if not all(mask)
    }
    assert set(backend.prompts[1:]) == expected
    assert len(backend.prompts) == 8
    assert output.metrics.requested == 8


def test_orchestrator_adds_grand_and_empty_masks() -> None:
    """A sampler that omits the boundary coalitions still gets v(N) and v(empty)."""
    estimator = SingleMaskEstimator()
    output = _run("fake", _registry(estimator), RecordingBackend())

    assert estimator.seen == {
        (True, False, False): 1.0,
        (True, True, True): 3.0,
        (False, False, False): 0.0,
    }
    assert output.result.grand_value == 3.0
    assert output.result.base_value == 0.0


def test_orchestrator_runs_second_phase_for_two_phase_estimators() -> None:
    """Phase-one payoffs feed begin_allocation; phase-two masks are aggregated."""
    estimator = FakeTwoPhaseEstimator()
    assert isinstance(estimator, TwoPhaseEstimator)
    output = _run("fake", _registry(estimator), RecordingBackend())

    assert estimator.phase_one == {
        (True, False, False): 1.0,
        (True, True, True): 3.0,
        (False, False, False): 0.0,
    }
    assert estimator.seen == {
        **estimator.phase_one,
        (False, True, False): 1.0,
        (True, True, False): 2.0,
    }
    assert output.metrics.requested == 5


def test_orchestrator_values_are_known_truth_for_additive_game() -> None:
    """Exact Shapley of the word-count game gives one unit per word."""
    output = _run("exact", default_registry_with_value(), RecordingBackend())

    assert output.result.values == pytest.approx([1.0, 1.0, 1.0])


def default_registry_with_value() -> PluginRegistry:
    registry = default_registry()
    registry.register(PluginGroup.VALUE_FNS, "word_count", WordCountValue)
    return registry


def _all_masks() -> list[tuple[bool, ...]]:
    return [
        tuple(bool((value >> index) & 1) for index in range(len(_WORDS)))
        for value in range(1 << len(_WORDS))
    ]
