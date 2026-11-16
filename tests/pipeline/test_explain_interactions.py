"""Pairwise interactions of an explained input through the pipeline.

@see [[src/mllm_shap/pipeline/interactions.py]]
"""

import asyncio

import pytest

pytest.importorskip("shapiq")

from mllm_shap import ExplainConfig
from mllm_shap.pipeline.interactions import explain_interactions
from tests.pipeline.test_pipeline_axioms import _registry, _snapshot


def _run(words: str):
    config = ExplainConfig.model_validate({
        "backend": {"kind": "echo", "model_id": "echo"},
        "explanation": {"value_fn": "count_test", "archive": {"path": ""}},
    })
    return asyncio.run(
        explain_interactions(config, _snapshot(words), registry=_registry())
    )


def test_an_additive_game_has_no_pairwise_interaction() -> None:
    result = _run("alpha beta gamma delta")
    assert result.index == "k-SII"
    assert len(result.player_ids) == 4
    for row in result.values:
        assert row == pytest.approx([0.0] * 4, abs=1e-9)


def test_matrix_is_symmetric_and_pairs_are_ranked() -> None:
    result = _run("alpha beta gamma")
    matrix = result.values
    assert all(matrix[i][j] == matrix[j][i] for i in range(3) for j in range(3))
    assert len(result.top(2)) == 2
