"""Several sampled draws per coalition: config, keys and averaged values.

@see [[src/mllm_shap/pipeline/orchestrator.py]]
"""

import asyncio

import pytest
from pydantic import ValidationError

from mllm_shap import ExplainConfig
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.masking.partitions import TokenPartitioner
from mllm_shap.pipeline.context import ExplainContext
from mllm_shap.pipeline.orchestrator import ExplainOrchestrator
from mllm_shap.runtime.dedup import build_coalition_key
from tests.pipeline.test_orchestrator import (
    SingleMaskEstimator,
    _registry,
    _snapshot,
)


class CountingBackend:
    """The n-th generation of a prompt has n words, so draws differ."""

    def __init__(self, context: object | None = None) -> None:
        _ = context
        self.calls: dict[str, int] = {}

    @property
    def model_id(self) -> str:
        return "counting"

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        prompt = " ".join(message.text for message in snapshot.messages())
        self.calls[prompt] = self.calls.get(prompt, 0) + 1
        return GenerationRecord(text=" ".join(["w"] * self.calls[prompt]))


def test_draws_need_a_sampling_temperature() -> None:
    with pytest.raises(ValidationError):
        GenerationConfig(num_samples=4, temperature=0.0)
    assert GenerationConfig(num_samples=4, temperature=0.7, top_k=0).num_samples == 4
    assert GenerationConfig().num_samples == 1


def test_the_first_draw_keeps_the_key_of_a_single_draw() -> None:
    args = {
        "snapshot_id": "s",
        "player_ids": ("a", "b"),
        "mask_present": (True, False),
        "absence_policy": "delete",
        "model_id": "m",
        "generation": GenerationConfig(),
    }
    assert build_coalition_key(**args, draw=0) == build_coalition_key(**args)
    assert build_coalition_key(**args, draw=1) != build_coalition_key(**args)


def test_each_coalition_is_drawn_num_samples_times_and_scored_by_the_mean() -> None:
    estimator = SingleMaskEstimator()
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "counting"},
        "generation": {"temperature": 0.7, "top_k": 0, "num_samples": 4},
        "explanation": {
            "estimator": "fake",
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
        registry=_registry(estimator),
        run_id="draws-test",
    )
    backend = CountingBackend()
    output = asyncio.run(ExplainOrchestrator(context, backend).run())
    # The base answer is the grand coalition's first draw, so it is a cache hit.
    assert output.metrics.requested == 3 * 4
    assert output.metrics.executed == 3 * 4 - 1
    assert set(backend.calls.values()) == {4}
    assert set(estimator.seen.values()) == {2.5}
