"""Tests for the expected-grade judge value function.

@see [[src/mllm_shap/value/judge.py]]
"""

import asyncio
import math

import pytest

from mllm_shap.domain.generation import ForcedScore, GenerationRecord
from mllm_shap.errors import ConfigError
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import PluginContext, default_registry
from mllm_shap.value.generation_records import generation_record_from_text
from mllm_shap.value.judge import GRADES, JudgeGradeValue


class _LogitJudge:
    """Judge whose digit probabilities favour 5 for 'same' and 1 otherwise."""

    model_id = "judge"
    forces_continuations = True

    async def generate(self, snapshot, generation):  # type: ignore[no-untyped-def]
        msg = "a judge with logits must not generate"
        raise AssertionError(msg)

    async def score_continuations(self, snapshot, continuations, top_k=0):  # type: ignore[no-untyped-def]
        assert tuple(continuations) == GRADES
        text = snapshot.messages()[0].text
        same = "Candidate answer: ok\n" in text
        weights = [0.0, 0.0, 0.0, 0.0, 1.0] if same else [1.0, 1.0, 0.0, 0.0, 0.0]
        return [ForcedScore((math.log(w) if w else -math.inf,)) for w in weights]


class _SamplingJudge:
    """Judge without logits that answers 4, 2, 4, 2 ... and one unparsable reply."""

    model_id = "sampler"

    def __init__(self) -> None:
        self.replies = iter(["4", "2", "4", "2", "no idea"])

    async def generate(self, snapshot, generation):  # type: ignore[no-untyped-def]
        assert generation.temperature > 0
        return GenerationRecord(text=next(self.replies))


def _judge(judge, **options: object) -> JudgeGradeValue:  # type: ignore[no-untyped-def]
    values = {"backend": "mock", "model_id": "j", **options}
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "m"},
        "explanation": {"value_fn": "judge_grade"},
    })
    context = PluginContext(config=config, registry=default_registry(), options=values)
    return JudgeGradeValue(context, judge=judge)


def _records(*texts: str) -> list[GenerationRecord]:
    return [generation_record_from_text(t) for t in texts]


def test_expected_grade_is_the_probability_weighted_mean() -> None:
    value_fn = _judge(_LogitJudge())
    base, same, other = _records("ref", "ok", "bad")
    asyncio.run(value_fn.prepare(base, [base, same, other]))

    assert value_fn.score(base, same) == pytest.approx(5.0)
    assert value_fn.score(base, other) == pytest.approx(1.5)  # (1 + 2) / 2
    assert value_fn.score(base, base) == 5.0


def test_judge_without_logits_averages_sampled_grades() -> None:
    value_fn = _judge(_SamplingJudge(), samples=5)
    base, answer = _records("ref", "ok")
    asyncio.run(value_fn.prepare(base, [base, answer]))
    assert value_fn.score(base, answer) == pytest.approx(3.0)  # (4 + 2 + 4 + 2) / 4


def test_scoring_before_prepare_is_an_error() -> None:
    base, answer = _records("ref", "ok")
    with pytest.raises(ConfigError, match="prepare"):
        _judge(_LogitJudge()).score(base, answer)


def test_a_large_run_is_refused_unless_allowed() -> None:
    base = generation_record_from_text("ref")
    corpus = _records(*(f"answer {i}" for i in range(5)))
    small = _judge(_LogitJudge(), max_coalitions=3)
    with pytest.raises(ConfigError, match="allow_large_judge"):
        asyncio.run(small.prepare(base, corpus))
    allowed = _judge(_LogitJudge(), max_coalitions=3, allow_large_judge=True)
    asyncio.run(allowed.prepare(base, corpus))
    assert allowed.score(base, corpus[0]) == pytest.approx(1.5)


def test_judge_builds_its_backend_from_the_options() -> None:
    value_fn = _judge(None)
    assert value_fn.manifest_info() == {"backend": "mock", "model_id": "j"}
    base, answer = _records("ref", "ok")
    # The mock backend has no grade logits worth asserting; the grade is in [1, 5].
    asyncio.run(value_fn.prepare(base, [base, answer]))
    assert 1.0 <= value_fn.score(base, answer) <= 5.0
