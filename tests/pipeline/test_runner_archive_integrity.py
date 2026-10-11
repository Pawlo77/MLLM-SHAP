"""Partial archives, streaming jobs and the player limit.

@see [[src/mllm_shap/pipeline/runner.py]]
"""

import sqlite3
from pathlib import Path

import pytest

from mllm_shap import ExplainConfig, ExplainRunner, PluginGroup
from mllm_shap.backends.mock import MockBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.estimands import Estimand
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import ArchiveIncompleteError, ConfigError
from mllm_shap.masking.codec import MaskCodec
from mllm_shap.pipeline.manifest import RunManifest
from mllm_shap.plugins import default_registry
from mllm_shap.runtime.archive import CoalitionRecordDraft, RunArchive
from mllm_shap.runtime.scheduler import InferenceScheduler


class FailingBackend(MockBackend):
    """Mock backend that raises on call number ``fail_on`` (sixth by default)."""

    def __init__(self, context: object | None = None, fail_on: int = 6) -> None:
        super().__init__(model_id="failing")
        self.calls = 0
        self.fail_on = fail_on

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        self.calls += 1
        if self.calls == self.fail_on:
            msg = "backend crashed"
            raise RuntimeError(msg)
        return await super().generate(snapshot, generation)


def _config(
    archive_path: str, kind: str = "mock", **explanation: object
) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": kind, "model_id": kind},
        "explanation": {
            "estimator": "exact",
            "absence_policy": "pad",
            "archive": {"path": archive_path},
            "max_inflight": 1,
            **explanation,
        },
    })


def _snapshot(text: str = "a b c d") -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def test_reanalyze_refuses_archive_of_failed_run(tmp_path: Path) -> None:
    registry = default_registry()
    registry.register(PluginGroup.BACKENDS, "failing", FailingBackend)
    template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_config(template, kind="failing"), registry=registry)

    with pytest.raises(RuntimeError, match="backend crashed"):
        runner.explain_sync(_snapshot())
    (archive_root,) = tmp_path.iterdir()

    with pytest.raises(ArchiveIncompleteError, match="did not finish"):
        runner.reanalyze(archive_root)


def test_reanalyze_refuses_exact_archive_missing_rows(tmp_path: Path) -> None:
    template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_config(template))
    first = runner.explain_sync(_snapshot("a b"))
    root = tmp_path / first.run_id
    with sqlite3.connect(root / "archive.sqlite") as conn:
        conn.execute(
            "DELETE FROM coalition_records WHERE record_id = "
            "(SELECT MIN(record_id) + 1 FROM coalition_records)"
        )
    with pytest.raises(ArchiveIncompleteError, match="coalitions"):
        runner.reanalyze(root)


def test_reanalyze_reports_crash_before_first_row_as_incomplete(
    tmp_path: Path,
) -> None:
    registry = default_registry()
    registry.register(
        PluginGroup.BACKENDS, "failing", lambda context: FailingBackend(fail_on=1)
    )
    template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_config(template, kind="failing"), registry=registry)

    with pytest.raises(RuntimeError, match="backend crashed"):
        runner.explain_sync(_snapshot())
    (archive_root,) = tmp_path.iterdir()

    with pytest.raises(ArchiveIncompleteError, match="did not finish"):
        runner.reanalyze(archive_root)


def test_reanalyze_refuses_full_budget_archive_missing_rows(tmp_path: Path) -> None:
    template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_config(template, estimator="mc"))
    first = runner.explain_sync(_snapshot("a b c"))
    root = tmp_path / first.run_id
    with sqlite3.connect(root / "archive.sqlite") as conn:
        conn.execute(
            "DELETE FROM coalition_records WHERE record_id = "
            "(SELECT MAX(record_id) FROM coalition_records)"
        )
    with pytest.raises(ArchiveIncompleteError, match="coalitions"):
        runner.reanalyze(root)


def test_reanalyze_refuses_archive_not_written_by_runner(tmp_path: Path) -> None:
    root = tmp_path / "hand-built"
    with RunArchive.open(root, RunManifest(Estimand.SHAPLEY, "x")) as archive:
        archive.write_base_generation("base")
        archive.append(
            CoalitionRecordDraft(
                snapshot_id="s",
                coalition_key="k",
                mask=MaskCodec.pack((True,)),
                absence_policy="pad",
                model_id="mock",
                generation_text="t",
                elapsed_ms=0.0,
                cache_hit=False,
            )
        )
        archive.mark_complete()
    with pytest.raises(ArchiveIncompleteError, match="not written by ExplainRunner"):
        ExplainRunner(_config("")).reanalyze(root)


def test_orchestrator_streams_jobs_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int | None]] = []
    original = InferenceScheduler.run_iter

    async def recording_run_iter(self, jobs, generate, total=None):
        seen.append((type(jobs).__name__, total))
        return await original(self, jobs, generate, total)

    monkeypatch.setattr(InferenceScheduler, "run_iter", recording_run_iter)
    ExplainRunner(_config("")).explain_sync(_snapshot("a b c"))
    assert seen == [("generator", 8)]


def test_runner_rejects_too_many_players_for_full_table() -> None:
    runner = ExplainRunner(_config("", max_players=3))
    with pytest.raises(ConfigError, match="max_players"):
        runner.explain_sync(_snapshot("a b c d"))


def test_max_players_does_not_limit_partial_budgets() -> None:
    config = _config("", max_players=3, estimator="mc", budget={"fraction": 0.25})
    output = ExplainRunner(config).explain_sync(_snapshot("a b c d"))
    assert len(output.result.values) == 4


def test_max_players_limits_planned_coalitions_of_partial_budgets() -> None:
    config = _config("", max_players=3, estimator="mc", budget={"fraction": 0.5})
    with pytest.raises(ConfigError, match="max_players"):
        ExplainRunner(config).explain_sync(_snapshot("a b c d e f"))
