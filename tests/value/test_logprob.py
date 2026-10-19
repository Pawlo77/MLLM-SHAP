"""Tests for the logprob value function.

@see [[src/mllm_shap/value/logprob.py]]
"""

import pytest

from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import BackendUnavailableError, LogprobsUnavailableError
from mllm_shap.protocols import RequiresLogprobs
from mllm_shap.value.logprob import LogprobValue, SelfLogprobValue

_BASE = GenerationRecord(text="base", logprobs=(-5.0,))


def test_logprob_is_the_mean_generated_token_logprob() -> None:
    candidate = GenerationRecord(text="a b", logprobs=(-0.2, -0.4))
    assert SelfLogprobValue().score(_BASE, candidate) == pytest.approx(-0.3)


def test_logprob_of_an_empty_generation_is_zero() -> None:
    assert (
        SelfLogprobValue().score(_BASE, GenerationRecord(text="", logprobs=())) == 0.0
    )


def test_logprob_refuses_records_without_logprobs() -> None:
    with pytest.raises(LogprobsUnavailableError, match="logprobs"):
        SelfLogprobValue().score(_BASE, GenerationRecord(text="candidate"))
    assert issubclass(LogprobsUnavailableError, BackendUnavailableError)


def test_logprob_declares_it_needs_logprobs() -> None:
    assert isinstance(SelfLogprobValue(), RequiresLogprobs)
    assert SelfLogprobValue().needs_logprobs


def test_logprob_is_a_deprecated_alias_that_points_to_tf_logprob() -> None:
    candidate = GenerationRecord(text="a b", logprobs=(-0.2, -0.4))
    with pytest.warns(DeprecationWarning, match="tf_logprob"):
        alias = LogprobValue()
    assert alias.name == "logprob"
    assert SelfLogprobValue().name == "self_logprob"
    assert alias.score(_BASE, candidate) == SelfLogprobValue().score(_BASE, candidate)
