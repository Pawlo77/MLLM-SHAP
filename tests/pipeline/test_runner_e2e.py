"""End-to-end explain pipeline tests with the mock backend.

@see [[src/nlp_shap/pipeline/runner.py]]
"""

from pathlib import Path

import pytest

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.errors import ArchiveExistsError, ConfigError, ReanalyzeMismatchError
from nlp_shap.estimation.exact import ExactEstimator
from nlp_shap.runtime import BASE_GENERATION_FILE, InMemoryObservabilitySink
from nlp_shap.runtime.archive import RunArchive


def _three_token_snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="a b c"),))
    return ConversationSnapshot.from_turns((turn,))


def _e2e_config(archive_path: str = "") -> ExplainConfig:
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
            "archive": {"path": archive_path, "flush_every": 1},
        },
    })


def test_runner_e2e_mock_exact_shapley() -> None:
    """Mock E2E returns attributions and scheduler metrics."""
    output = ExplainRunner(_e2e_config()).explain_sync(_three_token_snapshot())

    assert output.result.estimand is Estimand.SHAPLEY
    assert len(output.result.values) == 3
    assert output.metrics.requested == ExactEstimator.num_coalitions(3)
    assert output.metrics.executed == output.metrics.requested - 1
    assert output.metrics.cache_hits == 1
    assert output.run_id
    assert output.manifest.estimand is Estimand.SHAPLEY
    assert output.perf is not None
    assert output.perf.total_ms >= 0.0


def test_runner_e2e_mock_exact_with_delete_policy() -> None:
    """Exact SHAP completes under delete absence, including empty coalition."""
    payload = _e2e_config().model_dump(mode="python")
    payload["explanation"]["absence_policy"] = "delete"
    output = ExplainRunner(ExplainConfig.model_validate(payload)).explain_sync(
        _three_token_snapshot()
    )
    assert len(output.result.values) == 3
    assert output.metrics.requested == ExactEstimator.num_coalitions(3)
    assert output.metrics.executed == output.metrics.requested - 1
    assert output.metrics.cache_hits == 1


def test_runner_e2e_exact_shapley_beyond_default_hot_store_size() -> None:
    """Exact SHAP must retain all coalition outputs when 2^n exceeds store LRU default.

    Regression: HotResultStore maxsize=128 dropped early coalitions for n>=8,
    causing ``missing generation for coalition`` during scoring.
    """
    turn = Turn(
        messages=(Message(role=Role.USER, text="a b c d e f g h"),),
    )
    snapshot = ConversationSnapshot.from_turns((turn,))
    output = ExplainRunner(_e2e_config()).explain_sync(snapshot)

    assert len(output.result.values) == 8
    assert output.metrics.requested == ExactEstimator.num_coalitions(8)
    assert output.metrics.executed == output.metrics.requested - 1
    assert output.metrics.cache_hits == 1
    assert ExactEstimator.num_coalitions(8) > 128


def test_runner_archives_base_generation_before_coalitions(
    tmp_path: Path,
) -> None:
    """Grand-coalition text is persisted before coalition rows are appended."""
    archive_template = str(tmp_path / "{run_id}")
    telemetry = InMemoryObservabilitySink()
    output = ExplainRunner(
        _e2e_config(archive_path=archive_template),
        telemetry=telemetry,
    ).explain_sync(_three_token_snapshot())

    archive_root = tmp_path / output.run_id
    base_path = archive_root / BASE_GENERATION_FILE
    assert base_path.is_file()

    with RunArchive.load(archive_root) as archive:
        base_text = archive.read_base_generation()
        records = list(archive.history_lazy())

    assert base_text
    assert len(records) == output.metrics.requested
    span_names = [span.name for span in telemetry.spans]
    assert "orchestrator.generate_base" in span_names
    assert "orchestrator.schedule" in span_names
    assert "estimator.sample_masks" in span_names


def test_runner_reanalyze_rescores_without_backend_calls(tmp_path: Path) -> None:
    """Reanalyze swaps value functions with zero scheduler executions."""
    archive_template = str(tmp_path / "{run_id}")
    initial = ExplainRunner(_e2e_config(archive_path=archive_template)).explain_sync(
        _three_token_snapshot()
    )
    archive_root = tmp_path / initial.run_id

    rescored = ExplainRunner(
        _e2e_config(archive_path=archive_template).model_copy(
            update={
                "explanation": _e2e_config(
                    archive_path=archive_template
                ).explanation.model_copy(update={"value_fn": "embedding_cosine"})
            }
        )
    ).reanalyze(archive_root)

    assert rescored.metrics.executed == 0
    assert rescored.metrics.requested == initial.metrics.requested
    assert len(rescored.result.values) == len(initial.result.values)
    assert rescored.perf is not None
    assert rescored.perf.generation_ms == 0.0


def test_reanalyze_requires_base_generation(tmp_path: Path) -> None:
    """Reanalyze rejects a finished archive whose base generation was removed."""
    runner = ExplainRunner(_e2e_config(str(tmp_path / "{run_id}")))
    output = runner.explain_sync(_three_token_snapshot())
    root = tmp_path / output.run_id
    (root / BASE_GENERATION_FILE).unlink()

    with pytest.raises(ValueError, match="missing base generation"):
        runner.reanalyze(root)


def test_runner_duplicate_tokens_delete_policy() -> None:
    """Identical masked renderings must still produce one payoff per coalition."""
    payload = _e2e_config().model_dump(mode="python")
    payload["explanation"]["absence_policy"] = "delete"
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="a a a"),)),
    ))
    output = ExplainRunner(ExplainConfig.model_validate(payload)).explain_sync(snapshot)
    assert len(output.result.values) == 3
    assert output.metrics.requested == 8
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )


def test_rerun_into_existing_archive_raises_by_default(tmp_path: Path) -> None:
    template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_e2e_config(archive_path=template))
    first = runner.explain_sync(_three_token_snapshot())
    with pytest.raises(ArchiveExistsError):
        runner.explain_sync(_three_token_snapshot())
    with RunArchive.load(tmp_path / first.run_id) as archive:
        assert len(list(archive.history_lazy())) == first.metrics.requested


def test_rerun_with_overwrite_replaces_archive(tmp_path: Path) -> None:
    payload = _e2e_config(archive_path=str(tmp_path / "{run_id}")).model_dump(
        mode="python"
    )
    payload["explanation"]["archive"]["on_exists"] = "overwrite"
    runner = ExplainRunner(ExplainConfig.model_validate(payload))
    first = runner.explain_sync(_three_token_snapshot())
    second = runner.explain_sync(_three_token_snapshot())
    assert first.run_id == second.run_id
    with RunArchive.load(tmp_path / second.run_id) as archive:
        assert len(list(archive.history_lazy())) == second.metrics.requested


def test_run_id_changes_with_config() -> None:
    a = ExplainRunner(_e2e_config()).explain_sync(_three_token_snapshot()).run_id
    payload = _e2e_config().model_dump(mode="python")
    payload["explanation"]["absence_policy"] = "delete"
    b = (
        ExplainRunner(ExplainConfig.model_validate(payload))
        .explain_sync(_three_token_snapshot())
        .run_id
    )
    assert a != b


def test_run_id_ignores_archive_settings(tmp_path: Path) -> None:
    """Switching ``on_exists`` must target the archive of the same run."""
    a = ExplainRunner(_e2e_config()).explain_sync(_three_token_snapshot()).run_id
    payload = _e2e_config(archive_path=str(tmp_path / "{run_id}")).model_dump(
        mode="python"
    )
    payload["explanation"]["archive"]["on_exists"] = "overwrite"
    b = (
        ExplainRunner(ExplainConfig.model_validate(payload))
        .explain_sync(_three_token_snapshot())
        .run_id
    )
    assert a == b


def test_reanalyze_rejects_estimator_mismatch(tmp_path: Path) -> None:
    template = str(tmp_path / "{run_id}")
    first = ExplainRunner(_e2e_config(archive_path=template)).explain_sync(
        _three_token_snapshot()
    )
    payload = _e2e_config(archive_path=template).model_dump(mode="python")
    payload["explanation"]["estimator"] = "complementary"
    with pytest.raises(ReanalyzeMismatchError, match="estimator"):
        ExplainRunner(ExplainConfig.model_validate(payload)).reanalyze(
            tmp_path / first.run_id
        )


def test_reanalyze_matches_explain_values(tmp_path: Path) -> None:
    template = str(tmp_path / "{run_id}")
    runner = ExplainRunner(_e2e_config(archive_path=template))
    first = runner.explain_sync(_three_token_snapshot())
    again = runner.reanalyze(tmp_path / first.run_id)
    assert again.result.values == pytest.approx(first.result.values, abs=1e-9)
    assert again.result.grand_value == pytest.approx(first.result.grand_value)
    assert again.result.base_value == pytest.approx(first.result.base_value)


@pytest.mark.parametrize("value_fn", ["tfidf_cosine", "embedding_cosine"])
def test_reanalyze_scores_generated_text_like_explain(
    tmp_path: Path, value_fn: str
) -> None:
    """Explain and reanalyze build identical records from generated text."""
    payload = _e2e_config(archive_path=str(tmp_path / "{run_id}")).model_dump(
        mode="python"
    )
    payload["explanation"]["value_fn"] = value_fn
    runner = ExplainRunner(ExplainConfig.model_validate(payload))
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="a b c d"),)),
    ))
    first = runner.explain_sync(snapshot)
    again = runner.reanalyze(tmp_path / first.run_id)
    assert again.result.values == pytest.approx(first.result.values, abs=1e-9)


@pytest.mark.parametrize("embedding_mode", ["static", "contextual"])
def test_reanalyze_matches_explain_embedding_values(
    tmp_path: Path, embedding_mode: str
) -> None:
    """C15: mock embedding runs reanalyze to the explain values in both modes."""
    payload = _e2e_config(archive_path=str(tmp_path / "{run_id}")).model_dump(
        mode="python"
    )
    payload["explanation"]["value_fn"] = "embedding_cosine"
    payload["explanation"]["absence_policy"] = "delete"
    payload["explanation"]["embedding_mode"] = embedding_mode
    runner = ExplainRunner(ExplainConfig.model_validate(payload))
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="alpha beta gamma delta"),)),
    ))
    first = runner.explain_sync(snapshot)
    assert first.manifest.embedding_source == "hash"
    again = runner.reanalyze(tmp_path / first.run_id, strict=True)
    assert again.result.values == pytest.approx(first.result.values, abs=1e-12)
    assert again.result.grand_value == first.result.grand_value
    assert again.result.base_value == first.result.base_value


def test_runner_rejects_plugin_options_for_plugins_outside_the_run() -> None:
    payload = _e2e_config().model_dump(mode="python")
    payload["explanation"]["plugin_options"] = {"neutral": {"fill_char": "#"}}
    runner = ExplainRunner(ExplainConfig.model_validate(payload))
    with pytest.raises(ConfigError, match="'neutral'"):
        runner.explain_sync(_three_token_snapshot())


def test_reanalyze_rejects_unknown_plugin_option_keys(tmp_path: Path) -> None:
    runner = ExplainRunner(_e2e_config(str(tmp_path / "{run_id}")))
    root = tmp_path / runner.explain_sync(_three_token_snapshot()).run_id
    payload = _e2e_config().model_dump(mode="python")
    payload["explanation"]["plugin_options"] = {"identity": {"scale": 2}}
    with pytest.raises(ConfigError, match=r"normalizers/identity.*'scale'"):
        ExplainRunner(ExplainConfig.model_validate(payload)).reanalyze(root)
