"""Tests for explain pipeline result types.

@see [[src/nlp_shap/pipeline/result.py]]
"""

import pytest

from nlp_shap.domain.estimands import Estimand
from nlp_shap.pipeline.result import ExplainResult


def test_explain_result_requires_estimand() -> None:
    """ExplainResult always carries an explicit estimand label."""
    result = ExplainResult(estimand=Estimand.SHAPLEY, values=(0.1, -0.2))
    assert result.estimand is Estimand.SHAPLEY
    assert result.values == (0.1, -0.2)


def test_explain_result_rejects_missing_estimand() -> None:
    """Construction fails when estimand is omitted."""
    with pytest.raises(TypeError):
        ExplainResult(values=(0.1,))  # type: ignore[call-arg]  # the omission is the test
