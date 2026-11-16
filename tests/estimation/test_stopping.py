"""Stopping rule: decisions, config checks and the pipeline run.

@see [[src/mllm_shap/estimation/stopping.py]]
"""

import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.errors import ConfigError
from mllm_shap.estimation.stopping import Z_95, check_stop, top_players
from mllm_shap.pipeline.config import explain_config_from_yaml


def test_top_players_rank_by_magnitude_and_break_ties_by_index() -> None:
    assert top_players([0.1, -0.9, 0.5, 0.5], 2) == (1, 2)
    assert top_players([1.0, 1.0, 1.0], 2) == (0, 1)


def test_narrow_intervals_stop_the_run() -> None:
    errors = [0.01, 0.02]
    wide = check_stop([1.0, 2.0], errors, 2 * Z_95 * 0.02 * 0.9, None, None)
    narrow = check_stop([1.0, 2.0], errors, 2 * Z_95 * 0.02 * 1.1, None, None)
    assert not wide.stop
    assert (narrow.stop, narrow.reason) == (True, "ci_width")


def test_a_stable_top_set_stops_on_the_second_equal_check() -> None:
    first = check_stop([0.9, 0.1, 0.5], None, None, 2, None)
    assert not first.stop
    same = check_stop([0.8, 0.2, 0.6], None, None, 2, first.top)
    assert (same.stop, same.reason) == (True, "top_k_stable")
    changed = check_stop([0.1, 0.9, 0.8], None, None, 2, first.top)
    assert not changed.stop


def test_config_needs_a_criterion_and_a_sampling_estimator() -> None:
    head = "backend: {kind: mock, model_id: stub}\n"
    with pytest.raises(ConfigError, match="max_ci_width or top_k_stable"):
        explain_config_from_yaml(
            f"{head}explanation: {{estimator: mc, "
            "budget: {fraction: 0.5, stop: {check_every: 4}}}\n"
        )
    for estimator, stop in (
        ("leverage", "{top_k_stable: 3}"),
        ("permutation", "{max_ci_width: 0.1}"),
    ):
        with pytest.raises(ConfigError, match=r"budget\.stop"):
            explain_config_from_yaml(
                f"{head}explanation: {{estimator: {estimator}, "
                f"budget: {{fraction: 0.5, stop: {stop}}}}}\n"
            )
    ok = explain_config_from_yaml(
        f"{head}explanation: {{estimator: mc, budget: {{fraction: 0.5, "
        "stop: {top_k_stable: 3}}}\n"
    )
    assert ok.explanation.budget.stop is not None


def _snapshot(words: int = 8) -> ConversationSnapshot:
    text = " ".join(f"w{i}" for i in range(words))
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _config(stop: dict[str, object] | None, estimator: str = "mc") -> ExplainConfig:
    budget: dict[str, object] = {"fraction": 1.0}
    if stop is not None:
        budget["stop"] = stop
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "estimator": estimator,
            "absence_policy": "pad",
            "value_fn": "tf_logprob",
            "budget": budget,
            "archive": {"path": ""},
        },
    })


def test_a_stable_top_set_ends_the_run_before_the_budget() -> None:
    full = ExplainRunner(_config(None)).explain_sync(_snapshot())
    stopped = ExplainRunner(
        _config({"top_k_stable": 3, "check_every": 8})
    ).explain_sync(_snapshot())
    assert stopped.metrics.requested < full.metrics.requested == 256
    assert len(stopped.result.values) == 8


def test_an_unreachable_width_runs_to_the_budget() -> None:
    config = _config({"max_ci_width": 1e-12, "check_every": 8})
    output = ExplainRunner(config).explain_sync(_snapshot())
    assert output.metrics.requested == 256


def test_a_budget_stopped_run_matches_the_full_run_at_the_same_masks() -> None:
    # A rule that never fires leaves the values of a plain run unchanged.
    plain = ExplainRunner(_config(None)).explain_sync(_snapshot())
    never = ExplainRunner(
        _config({"max_ci_width": 1e-12, "check_every": 16})
    ).explain_sync(_snapshot())
    assert list(never.result.values) == pytest.approx(list(plain.result.values))
