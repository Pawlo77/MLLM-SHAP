"""Run timing summaries reported by explain and reanalyze.

@see [[src/nlp_shap/runtime/metrics.py]]
"""

from pathlib import Path

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.runtime.metrics import PerfSummary


def _config(archive_path: str) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "archive": {"path": archive_path},
        },
    })


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="a b c"),)),
    ))


def _stages(perf: PerfSummary) -> float:
    return perf.generation_ms + perf.scoring_ms + perf.aggregation_ms


def test_explain_perf_stages_fit_inside_total(tmp_path: Path) -> None:
    output = ExplainRunner(_config(str(tmp_path / "{run_id}"))).explain_sync(
        _snapshot()
    )

    assert output.perf is not None
    assert output.perf.generation_ms > 0.0
    assert 0.0 <= _stages(output.perf) <= output.perf.total_ms


def test_reanalyze_perf_reports_no_generation_time(tmp_path: Path) -> None:
    runner = ExplainRunner(_config(str(tmp_path / "{run_id}")))
    explained = runner.explain_sync(_snapshot())

    perf = runner.reanalyze(tmp_path / explained.run_id).perf

    assert perf is not None
    assert perf.generation_ms == 0.0
    assert 0.0 <= _stages(perf) <= perf.total_ms
