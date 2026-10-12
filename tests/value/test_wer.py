"""Word error rate and the ``reference_wer`` value function.

@see [[src/mllm_shap/value/wer.py]]
"""

import pytest

from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import PluginContext
from mllm_shap.value.wer import ReferenceWerValue, word_error_rate


@pytest.mark.parametrize(
    ("reference", "hypothesis", "expected"),
    [
        ("the cat sat", "the cat sat", 0.0),
        ("The cat, sat.", "the cat sat", 0.0),  # case and punctuation ignored
        ("the cat sat", "the dog sat", 1 / 3),  # one substitution
        ("the cat sat", "the cat", 1 / 3),  # one deletion
        ("the cat sat", "the big cat sat down", 2 / 3),  # two insertions
        ("I'm here", "im here", 0.5),  # apostrophes stay inside words
        ("", "", 0.0),
        ("", "anything", 1.0),
    ],
)
def test_word_error_rate(reference: str, hypothesis: str, expected: float) -> None:
    assert word_error_rate(reference, hypothesis) == pytest.approx(expected)


def _value(**options: object) -> ReferenceWerValue:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"value_fn": "reference_wer"},
    })
    return ReferenceWerValue(PluginContext(config=config, options=options))


def test_reference_option_scores_against_the_given_text() -> None:
    value = _value(reference="i had that curiosity")
    base = GenerationRecord(text="unrelated full answer")
    assert value.score(base, GenerationRecord(text="I had that curiosity.")) == 1.0
    assert value.score(base, GenerationRecord(text="i had curiosity")) == 0.75


def test_without_reference_the_full_answer_is_the_target() -> None:
    value = _value()
    base = GenerationRecord(text="a b c d")
    assert value.score(base, base) == 1.0
    assert value.score(base, GenerationRecord(text="a b")) == 0.5


def test_more_errors_than_words_score_zero() -> None:
    value = _value(reference="a")
    assert value.score(GenerationRecord(text=""), GenerationRecord(text="x y z")) == 0.0


def test_defaults_without_a_context() -> None:
    value = ReferenceWerValue()
    assert (value.name, value.reference) == ("reference_wer", None)
