"""Tests for optional coalition progress on ExplainRunner.

@see [[src/nlp_shap/pipeline/runner.py]]
"""

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


def _snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="a b c"),))
    return ConversationSnapshot.from_turns((turn,))


def _config() -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "generation": {
            "max_new_tokens": 8,
            "temperature": 0.0,
        },
        "explanation": {
            "estimator": "exact",
            "estimand": "shapley",
            "value_fn": "tfidf_cosine",
            "normalizer": "identity",
            "players": "tokens",
            "absence_policy": "pad",
            "budget": {"fraction": 1.0},
            "max_inflight": 2,
            "seed": 0,
            "archive": {"path": ""},
        },
    })


def test_explain_runner_reports_coalition_progress(progress) -> None:
    output = ExplainRunner(_config(), progress=progress).explain_sync(_snapshot())
    assert progress.planned == [output.metrics.requested]
    assert progress.finished[-1] == (
        output.metrics.requested,
        output.metrics.requested,
    )
    assert len(progress.finished) == output.metrics.requested


def test_neyman_progress_reports_one_growing_plan(progress) -> None:
    """Phase two extends the plan; finished counts never restart."""
    payload = _config().model_dump(mode="python")
    payload["explanation"]["estimator"] = "neyman_cc"
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="a b c d e f"),)),
    ))
    output = ExplainRunner(
        ExplainConfig.model_validate(payload), progress=progress
    ).explain_sync(snapshot)
    assert progress.planned == sorted(progress.planned)
    assert progress.planned[-1] == output.metrics.requested
    done = [count for count, _ in progress.finished]
    assert done == list(range(1, output.metrics.requested + 1))
    assert progress.finished[-1] == (output.metrics.requested,) * 2
