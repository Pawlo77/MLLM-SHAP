"""Deletion curves: remove the top-k players at once and rescore.

@see [[src/mllm_shap/pipeline/faithfulness.py]]
"""

import asyncio
from collections.abc import Sequence

import pytest

from mllm_shap import ExplainConfig
from mllm_shap.pipeline import faithfulness
from mllm_shap.pipeline.faithfulness import DeletionCurve, deletion_curve
from tests.pipeline.test_pipeline_axioms import _registry, _snapshot


def _curve(
    attributions: Sequence[float], ks: Sequence[int] | None = None
) -> DeletionCurve:
    """v counts 'alpha' and 'gamma' in the echoed prompt."""
    config = ExplainConfig.model_validate({
        "backend": {"kind": "echo", "model_id": "echo"},
        "explanation": {"value_fn": "count_test", "archive": {"path": ""}},
    })
    snapshot = _snapshot("alpha beta gamma delta")
    return asyncio.run(
        deletion_curve(config, snapshot, attributions, ks=ks, registry=_registry())
    )


def test_curve_removes_the_highest_ranked_players_first() -> None:
    curve = _curve([0.9, 0.1, 0.5, -0.2])
    assert curve.removed == (0, 1, 2, 3, 4)
    assert curve.values == pytest.approx((2.0, 1.0, 0.0, 0.0, 0.0))
    assert curve.drops == pytest.approx((0.0, 1.0, 2.0, 2.0, 2.0))


def test_a_worse_ranking_has_a_larger_area() -> None:
    assert _curve([1, 0, 1, 0]).area < _curve([0, 1, 0, 1]).area


def test_chosen_k_values_only() -> None:
    curve = _curve([0.9, 0.1, 0.5, -0.2], ks=(2, 0))
    assert curve.removed == (0, 2)
    assert curve.values == pytest.approx((2.0, 0.0))


def test_the_sampler_is_a_private_estimator() -> None:
    assert faithfulness._DeletionSampler([1.0], None).name == "_deletion_curve"


def test_area_is_the_trapezoid_over_the_removed_fraction() -> None:
    curve = DeletionCurve(
        num_players=2, grand_value=2.0, removed=(0, 1, 2), values=(2.0, 1.0, 0.0)
    )
    assert curve.area == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("attributions", "ks", "match"),
    [
        ([1.0, 2.0], None, "one attribution per player"),
        ([1, 0, 1, 0], (5,), "0 to 4"),
    ],
)
def test_bad_inputs_are_refused(
    attributions: list[float], ks: tuple[int, ...] | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _curve(attributions, ks)
