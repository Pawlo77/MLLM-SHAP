"""``explanation.stderr`` end to end: explain, the JSON payload and reanalyze.

@see [[src/mllm_shap/pipeline/orchestrator.py]]
"""

from pathlib import Path

import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role


def _snapshot() -> ConversationSnapshot:
    text = " ".join(f"w{i}" for i in range(8))
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _config(archive: str, estimator: str, **explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "estimator": estimator,
            "absence_policy": "pad",
            "archive": {"path": archive},
            "stderr": True,
            **explanation,
        },
    })


def test_mc_reports_errors_and_reanalyze_reproduces_them(tmp_path: Path) -> None:
    archive = tmp_path / "run"
    config = _config(str(archive), "mc", budget={"fraction": 0.3})
    runner = ExplainRunner(config)
    output = runner.explain_sync(_snapshot())

    stderr = output.result.stderr
    assert stderr is not None
    assert len(stderr) == 8
    assert all(value >= 0.0 for value in stderr)
    assert any(value > 0.0 for value in stderr)
    assert output.to_dict()["stderr"] == list(stderr)

    again = ExplainRunner(config).reanalyze(archive)
    assert again.result.stderr == pytest.approx(stderr)


def test_exact_reports_zero_errors(tmp_path: Path) -> None:
    config = _config(str(tmp_path / "exact"), "exact")
    output = ExplainRunner(config).explain_sync(_snapshot())
    assert output.result.stderr == (0.0,) * 8


def test_payload_omits_stderr_unless_requested(tmp_path: Path) -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "estimator": "exact",
            "absence_policy": "pad",
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config).explain_sync(_snapshot())
    assert output.result.stderr is None
    assert "stderr" not in output.to_dict()
