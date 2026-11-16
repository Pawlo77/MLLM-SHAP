"""Replay backend: archived coalitions answer any estimator without the model.

@see [[src/mllm_shap/backends/replay/backend.py]]
"""

from pathlib import Path

import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.errors import ConfigError, ReplayMissError


def _snapshot(words: int = 6) -> ConversationSnapshot:
    text = " ".join(f"w{i}" for i in range(words))
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _live(archive: Path | str, **explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "absence_policy": "pad",
            "archive": {"path": str(archive)},
            **explanation,
        },
    })


def _replay(
    archive: Path, source: Path, fallback: bool = False, **explanation: object
) -> ExplainConfig:
    options: dict[str, object] = {"archives": [str(source)]}
    if fallback:
        options["fallback"] = {"kind": "mock", "model_id": "mock"}
    return ExplainConfig.model_validate({
        "backend": {"kind": "replay", "model_id": "mock", "options": options},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "absence_policy": "pad",
            "archive": {"path": str(archive)},
            **explanation,
        },
    })


@pytest.fixture
def exact_archive(tmp_path: Path) -> tuple[Path, list[float]]:
    root = tmp_path / "exact"
    output = ExplainRunner(_live(root, estimator="exact")).explain_sync(_snapshot())
    return root, list(output.result.values)


def test_exact_through_replay_equals_the_live_run_without_model_calls(
    tmp_path: Path, exact_archive: tuple[Path, list[float]]
) -> None:
    root, live = exact_archive
    config = _replay(tmp_path / "again", root, estimator="exact")
    output = ExplainRunner(config).explain_sync(_snapshot())
    assert list(output.result.values) == live
    assert output.metrics.executed == 0
    assert output.perf is not None
    assert output.perf.forward_sequences == 0


def test_any_estimator_and_seed_runs_on_the_archived_table(
    tmp_path: Path, exact_archive: tuple[Path, list[float]]
) -> None:
    root, _ = exact_archive
    sampled = {"estimator": "mc", "budget": {"fraction": 0.4}, "seed": 3}
    live = ExplainRunner(_live(tmp_path / "live", **sampled)).explain_sync(_snapshot())
    replayed = ExplainRunner(_replay(tmp_path / "re", root, **sampled)).explain_sync(
        _snapshot()
    )
    assert list(replayed.result.values) == list(live.result.values)
    assert replayed.metrics.executed == 0


def test_a_coalition_missing_from_the_archive_is_named(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    ExplainRunner(
        _live(partial, estimator="mc", budget={"fraction": 0.3})
    ).explain_sync(_snapshot())
    config = _replay(tmp_path / "out", partial, estimator="exact")
    with pytest.raises(ReplayMissError, match=r"coalition [01]{6}"):
        ExplainRunner(config).explain_sync(_snapshot())


def test_a_fallback_backend_computes_the_missing_coalitions(
    tmp_path: Path, exact_archive: tuple[Path, list[float]]
) -> None:
    _, live = exact_archive
    partial = tmp_path / "partial"
    ExplainRunner(
        _live(partial, estimator="mc", budget={"fraction": 0.3})
    ).explain_sync(_snapshot())
    config = _replay(tmp_path / "out", partial, fallback=True, estimator="exact")
    output = ExplainRunner(config).explain_sync(_snapshot())
    assert list(output.result.values) == pytest.approx(live)
    assert 0 < output.metrics.executed < 2**6


def test_teacher_forced_scores_replay_from_the_archive(tmp_path: Path) -> None:
    root = tmp_path / "forced"
    live = ExplainRunner(
        _live(root, estimator="exact", value_fn="tf_logprob")
    ).explain_sync(_snapshot(5))
    output = ExplainRunner(
        _replay(tmp_path / "again", root, estimator="exact", value_fn="tf_logprob")
    ).explain_sync(_snapshot(5))
    assert list(output.result.values) == list(live.result.values)
    assert output.metrics.executed == 0


def test_replay_needs_archives(tmp_path: Path) -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "replay", "model_id": "mock", "options": {"archives": []}},
        "explanation": {"archive": {"path": str(tmp_path / "x")}},
    })
    with pytest.raises(ConfigError, match="archives"):
        ExplainRunner(config).explain_sync(_snapshot())
