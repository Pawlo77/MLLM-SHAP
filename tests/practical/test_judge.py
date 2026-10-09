"""Rubric checks and the logit judge's vote, without a model.

@see [[src/nlp_shap/pipeline/result.py]]
"""

import pytest

from tests.practical._judge import (
    efficiency_gap,
    logprob_margin,
    normalised_l1,
    top_by_magnitude,
    top_players,
    vote,
)


@pytest.mark.parametrize(
    ("margins", "outcome"),
    [
        ((2.0, 1.5, -3.0), "pass"),
        ((-1.2, -4.0, 2.0), "fail"),
        ((-1.2, 0.3, 0.9), "uncertain"),
        ((0.99, -0.99, 5.0), "uncertain"),
        ((1.0, 1.0, -1.0), "pass"),
    ],
)
def test_vote_needs_two_confident_answers(
    margins: tuple[float, ...], outcome: str
) -> None:
    """Two margins at or past 1 nat decide; anything less is uncertain."""
    assert vote(margins).outcome == outcome
    assert vote(margins).margins == margins


def test_vote_needs_three_prompts() -> None:
    """The judge asks exactly three paraphrases."""
    with pytest.raises(ValueError, match="three"):
        vote((2.0, 2.0))


def test_normalised_l1_against_reference() -> None:
    """Sum of absolute differences over the reference's absolute sum."""
    assert normalised_l1([1.0, -0.5, 0.0], [1.0, -1.0, 0.5]) == pytest.approx(0.4)


def test_top_players_and_magnitude_order() -> None:
    """Top players tie at the largest value; magnitude order ranks by |value|."""
    labels = ["a", "b", "c", "d"]
    assert top_players(labels, [0.2, -0.9, 0.5, 0.1]) == ["c"]
    assert top_players(labels, [0.5, -0.9, 0.5, 0.5 - 1e-12]) == ["a", "c", "d"]
    assert top_by_magnitude([0.2, -0.9, 0.5, 0.1], 2) == [1, 2]


def test_efficiency_gap() -> None:
    """Gap between the value sum and v(N) - v(empty)."""
    assert efficiency_gap([0.25, 0.5], base=0.1, grand=0.85) == pytest.approx(0.0)
    assert efficiency_gap([0.25, 0.5], base=0.0, grand=1.0) == pytest.approx(0.25)


def test_logprob_margin_reads_top_logprobs() -> None:
    """YES minus NO over stripped tokens; a missing word counts as the lowest entry."""
    top = [
        {"token": "YES", "logprob": -0.1},
        {"token": " NO", "logprob": -2.6},
        {"token": "Yes", "logprob": -5.0},
    ]
    assert logprob_margin(top) == pytest.approx(2.5)
    assert logprob_margin(top[:1] + top[2:]) == pytest.approx(4.9)
