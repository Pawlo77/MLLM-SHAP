"""Absolute evaluation budgets: validation, exact counts and the player guard.

@see [[src/mllm_shap/pipeline/config.py]]
"""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.errors import ConfigError
from mllm_shap.pipeline.config import (
    BudgetConfig,
    explain_config_from_yaml,
    explain_config_to_yaml,
)


def _snapshot(num_words: int) -> ConversationSnapshot:
    text = " ".join(f"w{i}" for i in range(num_words))
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _config(
    budget: dict[str, float], estimator: str = "mc", **explanation: object
) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "estimator": estimator,
            "budget": budget,
            "absence_policy": "pad",
            "archive": {"path": ""},
            **explanation,
        },
    })


def _yaml(budget: str, estimator: str = "mc") -> str:
    return (
        "backend: {kind: mock, model_id: mock}\n"
        f"explanation: {{estimator: {estimator}, budget: {budget}}}\n"
    )


def test_budget_defaults_to_the_full_table() -> None:
    assert BudgetConfig() == BudgetConfig(fraction=1.0)
    assert (BudgetConfig().fraction, BudgetConfig().evaluations) == (1.0, None)
    absolute = BudgetConfig(evaluations=2000)
    assert (absolute.fraction, absolute.evaluations) == (None, 2000)


@pytest.mark.parametrize("num_players", [10, 100])
def test_per_player_budget_is_a_multiple_of_the_player_count(num_players: int) -> None:
    budget = BudgetConfig(per_player=4)
    assert (budget.fraction, budget.evaluations) == (None, None)
    assert budget.evaluations_for(num_players) == 4 * num_players
    assert BudgetConfig(per_player=2.5).evaluations_for(5) == 13
    output = ExplainRunner(_config({"per_player": 4})).explain_sync(
        _snapshot(num_players)
    )
    assert output.metrics.requested == 4 * num_players


def test_budget_keys_are_exclusive() -> None:
    with pytest.raises(ConfigError, match=r"exactly one of budget\.fraction"):
        explain_config_from_yaml(_yaml("{per_player: 4, evaluations: 10}"))
    with pytest.raises(ValidationError, match="per_player"):
        BudgetConfig.model_validate({"per_player": 0})


def test_fraction_and_evaluations_are_exclusive() -> None:
    with pytest.raises(ConfigError, match=r"exactly one of budget\.fraction"):
        explain_config_from_yaml(_yaml("{fraction: 0.5, evaluations: 10}"))


@pytest.mark.parametrize("evaluations", [0, -3, 2.5])
def test_evaluations_must_be_a_positive_integer(evaluations: float) -> None:
    with pytest.raises(ValidationError, match="evaluations"):
        BudgetConfig.model_validate({"evaluations": evaluations})


def test_exact_refuses_an_absolute_budget() -> None:
    with pytest.raises(ConfigError, match="estimator 'exact'"):
        explain_config_from_yaml(_yaml("{evaluations: 8}", estimator="exact"))


def test_yaml_writes_only_the_budget_that_is_set() -> None:
    """Configs without ``evaluations`` serialise as before, so run ids are stable."""
    default = explain_config_from_yaml(_yaml("{fraction: 0.5}"))
    assert "evaluations" not in explain_config_to_yaml(default)
    absolute = explain_config_from_yaml(_yaml("{evaluations: 10}"))
    text = explain_config_to_yaml(absolute)
    assert "  budget:\n    evaluations: 10\n" in text
    assert explain_config_from_yaml(text) == absolute


@pytest.mark.parametrize(
    ("estimator", "num_players", "evaluations", "requested"),
    [
        ("mc", 4, 1, 2),  # floor: the empty and grand coalitions
        ("mc", 4, 7, 7),
        ("mc", 4, 100, 16),  # ceiling 2^4
        ("mc", 6, 21, 21),
        ("complementary", 4, 1, 10),  # floor 2n + 2
        ("complementary", 4, 13, 12),  # whole pairs
        ("complementary", 6, 21, 20),
        ("complementary", 3, 100, 8),
        ("neyman_cc", 4, 13, 12),
        ("neyman_cc", 6, 30, 30),
    ],
)
def test_runs_request_the_absolute_budget(
    estimator: str, num_players: int, evaluations: int, requested: int
) -> None:
    output = ExplainRunner(
        _config({"evaluations": evaluations}, estimator=estimator)
    ).explain_sync(_snapshot(num_players))
    assert output.metrics.requested == requested


def test_max_players_guards_the_absolute_budget() -> None:
    # 40 players, 50 evaluations: about 2^5.6 coalitions, under 2^24.
    output = ExplainRunner(_config({"evaluations": 50})).explain_sync(_snapshot(40))
    assert output.metrics.requested == 50
    allowed = _config({"evaluations": 32}, max_players=5)
    assert ExplainRunner(allowed).explain_sync(_snapshot(8)).metrics.requested == 32
    refused = _config({"evaluations": 33}, max_players=5)
    with pytest.raises(ConfigError, match=r"budget\.evaluations 33 plans about 2\^5"):
        ExplainRunner(refused).explain_sync(_snapshot(8))


def test_reanalyze_of_a_full_absolute_budget(tmp_path: Path) -> None:
    config = _config({"evaluations": 8}, archive={"path": str(tmp_path / "{run_id}")})
    runner = ExplainRunner(config)
    first = runner.explain_sync(_snapshot(3))
    again = runner.reanalyze(tmp_path / first.run_id)
    assert first.metrics.requested == 8
    assert again.result.values == pytest.approx(first.result.values, abs=1e-12)


def _cli_check(tmp_path: Path, budget: str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "config.yaml"
    path.write_text(_yaml(budget), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "mllm_shap.cli", "config", "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )


def test_cli_config_check_accepts_an_absolute_budget(tmp_path: Path) -> None:
    checked = _cli_check(tmp_path, "{evaluations: 2000}")
    assert checked.returncode == 0
    assert "valid" in checked.stdout
    refused = _cli_check(tmp_path, "{fraction: 0.1, evaluations: 2000}")
    assert refused.returncode == 2
    assert "exactly one of budget.fraction" in refused.stderr


def test_additive_efficiency_makes_partial_budgets_efficient() -> None:
    plain = ExplainRunner(_config({"fraction": 0.5})).explain_sync(_snapshot(6))
    fixed = ExplainRunner(
        _config({"fraction": 0.5}, efficiency="additive")
    ).explain_sync(_snapshot(6))
    gap = fixed.result.grand_value - fixed.result.base_value
    assert sum(fixed.result.values) == pytest.approx(gap)
    shifts = [
        a - b for a, b in zip(fixed.result.values, plain.result.values, strict=True)
    ]
    assert shifts == pytest.approx([shifts[0]] * 6)
    assert "efficiency" not in explain_config_to_yaml(_config({"fraction": 0.5}))
    with pytest.raises(ValidationError, match="efficiency"):
        _config({"fraction": 0.5}, efficiency="additive", estimand="banzhaf")
