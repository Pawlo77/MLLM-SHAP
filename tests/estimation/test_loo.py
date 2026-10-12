"""Leave-one-out attributions.

@see [[src/mllm_shap/estimation/loo.py]]
"""

import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.loo import LeaveOneOutEstimator
from tests.pipeline.test_pipeline_axioms import _registry, _snapshot


def _players(n: int) -> PlayerSet:
    return PlayerSet(player_ids=tuple(f"p{i}" for i in range(n)))


def test_samples_the_grand_coalition_and_each_leave_one_out() -> None:
    assert LeaveOneOutEstimator().name == "loo"
    masks = [
        m.present
        for m in LeaveOneOutEstimator().sample_masks(_players(3), 1.0, False, 0)
    ]
    assert masks == [
        (True, True, True),
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ]


def test_values_are_drops_from_the_grand_coalition() -> None:
    """An all-or-nothing game: every player is pivotal, so LOO gives each 1."""
    estimator = LeaveOneOutEstimator()
    masks = list(estimator.sample_masks(_players(3), 1.0, False, 0))
    payoffs = [1.0 if all(m.present) else 0.0 for m in masks]
    masks.append(CoalitionMask.from_sequence((False, False, False)))
    payoffs.append(0.0)
    assert estimator.estimate_attributions(masks, payoffs, aggregator=None) == [
        1.0,
        1.0,
        1.0,
    ]


def test_loo_runs_through_the_pipeline_with_n_plus_two_coalitions() -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "echo", "model_id": "echo"},
        "explanation": {
            "estimator": "loo",
            "value_fn": "count_test",
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config, registry=_registry()).explain_sync(
        _snapshot("alpha beta gamma delta")
    )
    assert output.result.values == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert output.metrics.requested == 6  # grand, four leave-one-outs, empty
