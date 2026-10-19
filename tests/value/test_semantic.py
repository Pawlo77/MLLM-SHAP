"""Tests for the encoder-based value functions with injected encoders.

@see [[src/mllm_shap/value/bertscore.py]]
@see [[src/mllm_shap/value/nli.py]]
@see [[src/mllm_shap/value/semascore.py]]
"""

import math
from typing import ClassVar

import numpy as np
import pytest

from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.protocols import FittableValueFunction
from mllm_shap.value.bertscore import BertScoreOptions, BertScoreValue
from mllm_shap.value.nli import NliLogOddsValue
from mllm_shap.value.semascore import SemaScoreValue, align_segments


def _rec(text: str) -> GenerationRecord:
    return GenerationRecord(text=text)


class _WordAxes:
    """Token encoder: each word is a fixed unit vector in a 3-d space."""

    VECTORS: ClassVar = {
        "cat": (1.0, 0.0, 0.0),
        "dog": (0.0, 1.0, 0.0),
        "pet": (0.6, 0.8, 0.0),
        "car": (0.0, 0.0, 1.0),
    }

    def __init__(self) -> None:
        self.batches: list[int] = []

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        self.batches.append(len(texts))
        return [
            np.array([self.VECTORS[w] for w in t.split()], dtype=float).reshape(-1, 3)
            for t in texts
        ]


def test_bertscore_matches_the_hand_computed_greedy_matching() -> None:
    value_fn = BertScoreValue(encoder=_WordAxes())
    # reference {cat, dog}; hypothesis {pet}: pet.cat = 0.6, pet.dog = 0.8.
    # recall = (0.6 + 0.8) / 2 = 0.7, precision = max(0.6, 0.8) = 0.8.
    f1 = 2 * 0.7 * 0.8 / (0.7 + 0.8)
    assert value_fn.score(_rec("cat dog"), _rec("pet")) == pytest.approx(f1)
    assert value_fn.score(_rec("cat dog"), _rec("cat dog")) == pytest.approx(1.0)
    assert value_fn.score(_rec("cat dog"), _rec("")) == 0.0
    assert value_fn.score(_rec("cat dog"), _rec("car")) == pytest.approx(0.0)


def test_bertscore_rescales_with_a_baseline() -> None:
    options = BertScoreOptions(baseline=0.5)
    assert options.baseline == 0.5
    value_fn = BertScoreValue(encoder=_WordAxes())
    value_fn._options = options
    assert value_fn.score(_rec("cat"), _rec("cat")) == pytest.approx(1.0)
    assert value_fn.score(_rec("cat"), _rec("car")) == pytest.approx(-1.0)


def test_bertscore_encodes_every_answer_of_an_instance_in_one_batch() -> None:
    encoder = _WordAxes()
    value_fn = BertScoreValue(encoder=encoder)
    assert isinstance(value_fn, FittableValueFunction)
    corpus = [_rec("cat"), _rec("dog"), _rec("pet"), _rec("cat"), _rec("car")]
    value_fn.fit(corpus)
    for record in corpus:
        value_fn.score(corpus[0], record)
    assert encoder.batches == [4]  # four distinct texts, one batch, no re-encoding


class _FakeNli:
    """Entailment probability: 0.9 for equal text or paraphrase, 0.05 for 'not'."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def entailment(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen.extend(pairs)
        return [
            0.05 if ("not" in a.split()) != ("not" in b.split()) else 0.9
            for a, b in pairs
        ]


def test_nli_log_odds_is_the_logit_of_the_geometric_mean() -> None:
    value_fn = NliLogOddsValue(model=_FakeNli())
    assert value_fn.score(_rec("it works"), _rec("it functions")) == pytest.approx(
        math.log(0.9 / 0.1)
    )
    negated = value_fn.score(_rec("it works"), _rec("it does not work"))
    assert negated == pytest.approx(math.log(0.05 / 0.95))
    assert negated < value_fn.score(_rec("it works"), _rec("it functions"))


def test_nli_prepends_the_question_to_both_answers() -> None:
    model = _FakeNli()
    value_fn = NliLogOddsValue(model=model)
    value_fn._options = value_fn._options.model_copy(update={"question": "Q?"})
    value_fn.score(_rec("a"), _rec("b"))
    assert set(model.seen) == {("Q? a", "Q? b"), ("Q? b", "Q? a")}


def test_semascore_alignment_groups_errors_into_segments() -> None:
    assert align_segments(["a", "b", "c"], ["a", "b", "c"]) == [
        ((0, 1), (0, 1)),
        ((1, 2), (1, 2)),
        ((2, 3), (2, 3)),
    ]
    # One substitution is a segment of its own between two matches.
    assert align_segments(["a", "x", "c"], ["a", "y", "c"]) == [
        ((0, 1), (0, 1)),
        ((1, 2), (1, 2)),
        ((2, 3), (2, 3)),
    ]
    # A deletion has an empty hypothesis span.
    assert align_segments(["a", "b"], ["a"]) == [((0, 1), (0, 1)), ((1, 2), (1, 1))]
    assert align_segments([], ["a"]) == [((0, 0), (0, 1))]


class _Words:
    """Word encoder: one-hot per distinct word, so similarity is exact match."""

    VOCAB = ("the", "cat", "sat", "dog", "ran")

    def encode(self, sentences: list[list[str]]) -> list[np.ndarray]:
        return [
            np.array([np.eye(5)[self.VOCAB.index(w)] for w in s]) for s in sentences
        ]


def test_semascore_is_one_for_a_perfect_transcript_and_drops_with_errors() -> None:
    value_fn = SemaScoreValue(encoder=_Words())
    reference = _rec("the cat sat")
    assert value_fn.score(reference, _rec("The cat sat.")) == pytest.approx(1.0)
    one_wrong = value_fn.score(reference, _rec("the dog sat"))
    two_wrong = value_fn.score(reference, _rec("the dog ran"))
    assert 0.0 < two_wrong < one_wrong < 1.0
    assert value_fn.score(reference, _rec("")) == 0.0


def test_semascore_hand_computed_value() -> None:
    value_fn = SemaScoreValue(encoder=_Words())
    # reference "the cat sat" vs "the dog sat": the mean reference vector is
    # (1, 1, 1) / 3 on the, cat, sat; alpha = cos(word, mean) = 1/sqrt(3) for each
    # word, so the weights are equal. Matches score 1; the substitution cat -> dog
    # has cosine 0 and scores 0. v = (1 + 0 + 1) / 3.
    assert value_fn.score(_rec("the cat sat"), _rec("the dog sat")) == pytest.approx(
        2 / 3
    )


def test_semascore_reference_option_replaces_the_base_answer() -> None:
    value_fn = SemaScoreValue(encoder=_Words())
    value_fn._options = value_fn._options.model_copy(
        update={"reference": "the dog ran"}
    )
    assert value_fn.score(_rec("the cat sat"), _rec("the dog ran")) == pytest.approx(
        1.0
    )


def test_semascore_empty_reference() -> None:
    value_fn = SemaScoreValue(encoder=_Words())
    assert value_fn.score(_rec(""), _rec("")) == 1.0
    assert value_fn.score(_rec(""), _rec("cat")) == 0.0


def test_value_functions_load_their_encoder_lazily_and_report_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mllm_shap.value import bertscore, nli, semascore

    loaded: list[str] = []

    def token_encoder(name: str, layer: int, revision: str) -> _WordAxes:
        loaded.append(f"{name}:{layer}:{revision}")
        return _WordAxes()

    def nli_model(name: str, revision: str) -> _FakeNli:
        loaded.append(f"{name}:{revision}")
        return _FakeNli()

    def word_encoder(name: str, revision: str) -> _Words:
        loaded.append(f"{name}:{revision}")
        return _Words()

    monkeypatch.setattr(bertscore, "load_token_encoder", token_encoder)
    monkeypatch.setattr(nli, "load_nli_model", nli_model)
    monkeypatch.setattr(semascore, "load_word_encoder", word_encoder)

    BertScoreValue().score(_rec("cat"), _rec("dog"))
    NliLogOddsValue().score(_rec("a"), _rec("b"))
    SemaScoreValue().score(_rec("the cat"), _rec("the dog"))

    assert loaded == [
        "roberta-large:17:main",
        "microsoft/deberta-large-mnli:main",
        "roberta-base:main",
    ]
    assert BertScoreValue().manifest_info()["layer"] == "17"
    assert NliLogOddsValue().manifest_info()["model"] == "microsoft/deberta-large-mnli"
    assert SemaScoreValue().manifest_info()["encoder"] == "roberta-base"
    assert BertScoreValue().name == "bertscore"
    assert NliLogOddsValue().name == "nli_logodds"
    assert SemaScoreValue().name == "semascore"
