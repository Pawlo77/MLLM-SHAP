"""Benchmark for TF-IDF value-function fitting and scoring.

@see [[src/nlp_shap/value/tfidf.py]]
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from nlp_shap.value.generation_records import generation_record_from_text
from nlp_shap.value.tfidf import TfIdfCosineValue

_RECORDS = [
    generation_record_from_text(
        " ".join(f"w{(index * 7 + offset) % 512}" for offset in range(16))
    )
    for index in range(4096)
]


def _fit_and_score() -> list[float]:
    value_fn = TfIdfCosineValue()
    value_fn.fit(_RECORDS)
    base = _RECORDS[0]
    return [value_fn.score(base, record) for record in _RECORDS]


@pytest.mark.bench
def test_tfidf_fit_and_score_4096(benchmark: BenchmarkFixture) -> None:
    """Fit TF-IDF on 4 096 records and score each against the base."""
    scores = benchmark(_fit_and_score)

    assert scores[0] == pytest.approx(1.0)
    assert all(0.0 <= score <= 1.0 + 1e-9 for score in scores)
