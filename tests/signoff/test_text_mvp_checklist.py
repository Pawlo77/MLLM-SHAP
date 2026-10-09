"""Text MVP sign-off: pipeline-level invariants only.

Unit-level checks live next to their modules (domain imports, scheduler, archive,
API backend); this file keeps what only the full pipeline can show.

@see [[src/nlp_shap/pipeline/runner.py]]
"""

import importlib
from pathlib import Path

import pytest

import nlp_shap
from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


def _snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="alpha beta gamma delta"),))
    return ConversationSnapshot.from_turns((turn,))


def _config(archive_path: str = "") -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "generation": {"max_new_tokens": 8, "temperature": 0.0},
        "explanation": {
            "estimator": "exact",
            "value_fn": "tfidf_cosine",
            "absence_policy": "pad",
            "budget": {"fraction": 1.0},
            "archive": {"path": archive_path, "flush_every": 1},
        },
    })


def test_every_public_export_resolves() -> None:
    """Each name in ``nlp_shap.__all__`` imports, lazy viz exports included."""
    module = importlib.import_module("nlp_shap")
    missing = [name for name in nlp_shap.__all__ if not hasattr(module, name)]
    assert missing == []


def test_exact_explain_is_efficient() -> None:
    """Exact Shapley through ExplainRunner sums to v(N) - v(empty)."""
    output = ExplainRunner(_config()).explain_sync(_snapshot())

    assert len(output.result.values) == 4
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )


def test_archive_and_reanalyze_round_trip(tmp_path: Path) -> None:
    """Reanalyze of an archived exact run reproduces explain without generation."""
    archive_template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_config(archive_template))
    explained = runner.explain_sync(_snapshot())

    reanalyzed = runner.reanalyze(tmp_path / explained.run_id)

    assert reanalyzed.metrics.executed == 0
    assert reanalyzed.result.values == pytest.approx(explained.result.values)
    assert reanalyzed.result.grand_value == pytest.approx(explained.result.grand_value)
    assert reanalyzed.result.base_value == pytest.approx(explained.result.base_value)
