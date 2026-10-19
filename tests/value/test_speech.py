"""Tests for the speech value functions with injected encoders.

@see [[src/mllm_shap/value/speech.py]]
"""

from typing import ClassVar

import numpy as np
import pytest

from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import ConfigError
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import PluginContext, default_registry
from mllm_shap.value.speech import SpeechBertScoreValue, TranscribedValue


class _Frames:
    """Speech encoder: the path names the frames, ``a`` and ``b`` are axes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, path: str) -> np.ndarray:
        self.calls.append(path)
        axes = {"a": (1.0, 0.0), "b": (0.0, 1.0)}
        return np.array([axes[c] for c in path.split("/")[-1].removesuffix(".wav")])


def _spoken(name: str) -> GenerationRecord:
    return GenerationRecord(text="", audio_path=f"/audio/{name}.wav")


def test_speech_bertscore_is_precision_of_the_answer_frames() -> None:
    encoder = _Frames()
    value_fn = SpeechBertScoreValue(encoder=encoder)
    reference = _spoken("ab")
    # Answer frames {a, a, b}: every frame has a perfect match in {a, b}.
    assert value_fn.score(reference, _spoken("aab")) == pytest.approx(1.0)
    # Answer {a, a}: also perfect; reference {a} against answer {a, b}:
    # b has best cosine 0 to {a}, so precision is (1 + 0) / 2.
    assert value_fn.score(_spoken("a"), _spoken("ab")) == pytest.approx(0.5)
    assert encoder.calls.count("/audio/ab.wav") == 1  # each file is encoded once


def test_speech_bertscore_needs_the_audio_path() -> None:
    value_fn = SpeechBertScoreValue(encoder=_Frames())
    with pytest.raises(ConfigError, match="audio_path"):
        value_fn.score(GenerationRecord(text="x"), _spoken("a"))


class _Recognizer:
    TEXT: ClassVar = {"/audio/one.wav": "the cat sat", "/audio/two.wav": "the dog sat"}

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, path: str) -> str:
        self.calls += 1
        return self.TEXT[path]


def _transcribed(
    recognizer: _Recognizer, inner: str = "reference_wer"
) -> TranscribedValue:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "m"},
        "explanation": {
            "value_fn": "transcribed",
            "plugin_options": {"transcribed": {"value_fn": inner}},
        },
    })
    context = PluginContext(
        config=config,
        registry=default_registry(),
        options={"value_fn": inner},
    )
    return TranscribedValue(context, recognizer=recognizer)


def test_transcribed_applies_a_text_value_function_to_the_transcripts() -> None:
    recognizer = _Recognizer()
    value_fn = _transcribed(recognizer)
    one, two = _spoken("one"), _spoken("two")
    assert value_fn.score(one, one) == 1.0
    assert value_fn.score(one, two) == pytest.approx(1 - 1 / 3)  # word error rate
    assert recognizer.calls == 2  # each recording is transcribed once


def test_transcribed_fits_a_corpus_value_function_on_the_transcripts() -> None:
    value_fn = _transcribed(_Recognizer(), inner="tfidf_cosine")
    one, two = _spoken("one"), _spoken("two")
    value_fn.fit([one, two])
    assert value_fn.score(one, one) == pytest.approx(1.0)
    assert 0.0 < value_fn.score(one, two) < 1.0
