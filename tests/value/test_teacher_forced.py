"""Tests for the teacher-forced value functions.

@see [[src/mllm_shap/value/teacher_forced.py]]
@see [[src/mllm_shap/value/contrastive.py]]
"""

import asyncio
import math

import pytest

from mllm_shap.backends.mock import MockBackend
from mllm_shap.backends.mock.backend import MOCK_VOCAB
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import ForcedScore, GenerationRecord
from mllm_shap.errors import BackendUnavailableError
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import PluginContext, PluginGroup, default_registry
from mllm_shap.protocols import RequiresTeacherForcing
from mllm_shap.value.contrastive import ContrastiveValue
from mllm_shap.value.teacher_forced import TfJsdValue, TfLogprobValue, _jsd

BASE = GenerationRecord(text="a b")


def _record(*logprobs: float, text: str = "a b") -> GenerationRecord:
    return GenerationRecord(text=text, forced={"a b": ForcedScore(tuple(logprobs))})


def _context(value_fn: str, **options: object) -> PluginContext:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "m"},
        "explanation": {
            "value_fn": value_fn,
            "plugin_options": {value_fn: options} if options else {},
        },
    })
    return PluginContext(config=config, registry=default_registry(), options=options)


def test_tf_logprob_averages_or_sums_the_forced_tokens() -> None:
    candidate = _record(-1.0, -3.0)
    assert TfLogprobValue().score(BASE, candidate) == pytest.approx(-2.0)
    summed = TfLogprobValue(_context("tf_logprob", reduce="sum"))
    assert summed.score(BASE, candidate) == pytest.approx(-4.0)
    assert TfLogprobValue().score(BASE, _record()) == 0.0


def test_tf_logprob_refuses_a_generation_without_forced_scores() -> None:
    with pytest.raises(BackendUnavailableError, match="teacher-forced"):
        TfLogprobValue().score(BASE, GenerationRecord(text="x"))


def test_teacher_forced_values_declare_what_the_pipeline_needs() -> None:
    value_fn = TfLogprobValue()
    assert isinstance(value_fn, RequiresTeacherForcing)
    assert value_fn.needs_teacher_forcing
    assert not value_fn.reads_candidate_text
    assert value_fn.forced_continuations(BASE) == ("a b",)
    assert TfJsdValue(_context("tf_jsd", top_k=7)).forced_top_k == 7


def test_jsd_of_known_distributions() -> None:
    half = math.log(0.5)
    # P = (1/2, 1/2) against Q = (1, 0): JSD = 0.215762 nats (hand computed).
    assert _jsd((0, 1), (half, half), (0, 1), (0.0, -math.inf)) == pytest.approx(
        0.215762, abs=1e-6
    )
    # Disjoint supports reach the log 2 bound; equal laws give 0.
    assert _jsd((0,), (0.0,), (1,), (0.0,)) == pytest.approx(math.log(2))
    assert _jsd((0, 1), (half, half), (1, 0), (half, half)) == pytest.approx(0.0)


def test_jsd_buckets_the_mass_outside_the_top_k() -> None:
    # Q lists only token 0 with mass 0.5; the other half is one rest bucket,
    # as is P's, so the laws are equal after bucketing.
    half = math.log(0.5)
    assert _jsd((0,), (half,), (0,), (half,)) == pytest.approx(0.0)


def _jsd_record(ids: tuple[int, ...], logs: tuple[float, ...]) -> GenerationRecord:
    return GenerationRecord(
        text="a b", forced={"a b": ForcedScore((0.0,), (ids,), (logs,))}
    )


def test_tf_jsd_is_zero_for_the_base_and_negative_for_a_shift() -> None:
    half = math.log(0.5)
    base = _jsd_record((0, 1), (half, half))
    shifted = _jsd_record((0, 1), (0.0, -math.inf))
    value_fn = TfJsdValue()
    assert value_fn.score(base, base) == 0.0
    assert value_fn.score(base, shifted) == pytest.approx(-0.215762, abs=1e-6)


def test_contrastive_is_target_minus_foil_log_probability() -> None:
    value_fn = ContrastiveValue(_context("contrastive", target="no", foil="yes"))
    candidate = GenerationRecord(
        text="",
        forced={"no": ForcedScore((-1.0, -0.5)), "yes": ForcedScore((-3.0,))},
    )
    assert value_fn.forced_continuations(BASE) == ("no", "yes")
    assert value_fn.score(BASE, candidate) == pytest.approx(1.5)


def test_contrastive_needs_target_and_foil() -> None:
    from mllm_shap.errors import ConfigError

    with pytest.raises(ConfigError, match="target"):
        default_registry().resolve(
            PluginGroup.VALUE_FNS, "contrastive", _context("contrastive")
        )


def _snapshot(text: str) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn((Message(role=Role.USER, text=text),)),
    ))


def test_mock_backend_forces_deterministic_normalised_distributions() -> None:
    backend = MockBackend()
    first = asyncio.run(
        backend.score_continuations(
            _snapshot("alpha beta"), ["x y z"], top_k=MOCK_VOCAB
        )
    )[0]
    again = asyncio.run(
        backend.score_continuations(
            _snapshot("alpha beta"), ["x y z"], top_k=MOCK_VOCAB
        )
    )[0]
    other = asyncio.run(
        backend.score_continuations(_snapshot("alpha"), ["x y z"], top_k=MOCK_VOCAB)
    )[0]

    assert first == again
    assert first.token_logprobs != other.token_logprobs
    assert len(first.token_logprobs) == 3
    for ids, logs in zip(first.top_ids, first.top_logprobs, strict=True):
        assert sorted(ids) == list(range(MOCK_VOCAB))
        assert math.fsum(math.exp(v) for v in logs) == pytest.approx(1.0)
    # The forced token's logprob is its entry in the position's distribution.
    assert all(
        lp in logs
        for lp, logs in zip(first.token_logprobs, first.top_logprobs, strict=True)
    )
    assert asyncio.run(backend.score_continuations(_snapshot("a"), [""]))[0] == (
        ForcedScore(())
    )
