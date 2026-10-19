"""Pipeline tests: teacher-forced value functions, rescoring and fixed payoffs.

@see [[src/mllm_shap/pipeline/orchestrator.py]]
@see [[src/mllm_shap/pipeline/rescore.py]]
"""

import asyncio
import random
from pathlib import Path

import pytest

from mllm_shap.backends.mock import MockBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import ConfigError, ReanalyzeMismatchError
from mllm_shap.pipeline import ExplainConfig, ExplainRunner
from mllm_shap.plugins import PluginGroup, default_registry
from mllm_shap.value.generation_records import generation_record_from_text


class CountingBackend(MockBackend):
    """Mock backend that counts generate and forced-scoring calls."""

    def __init__(self) -> None:
        super().__init__()
        self.generated = 0
        self.forced = 0

    async def generate(self, snapshot, generation):  # type: ignore[no-untyped-def]
        self.generated += 1
        return await super().generate(snapshot, generation)

    async def score_continuations(self, snapshot, continuations, top_k=0):  # type: ignore[no-untyped-def]
        self.forced += 1
        return await super().score_continuations(snapshot, continuations, top_k)


class NoForcingBackend:
    """Backend that only generates."""

    model_id = "plain"

    async def generate(self, snapshot, generation):  # type: ignore[no-untyped-def]
        return generation_record_from_text("one two")


class CoinBackend(MockBackend):
    """Mock backend whose generations differ on every call."""

    async def generate(self, snapshot, generation):  # type: ignore[no-untyped-def]
        _ = snapshot, generation
        return generation_record_from_text(f"w{random.random()}")


def _config(tmp_path: Path, value_fn: str, **explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "explanation": {
            "estimator": "exact",
            "value_fn": value_fn,
            "players": "tokens",
            "absence_policy": "delete",
            "archive": {"path": str(tmp_path / "{run_id}")},
            **explanation,
        },
    })


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn((Message(role=Role.USER, text="alpha beta gamma"),)),
    ))


@pytest.mark.parametrize("value_fn", ["tf_logprob", "tf_jsd"])
def test_teacher_forced_runs_skip_coalition_generation(
    tmp_path: Path, value_fn: str
) -> None:
    backend = CountingBackend()
    output = ExplainRunner(_config(tmp_path, value_fn), backend=backend).explain_sync(
        _snapshot()
    )

    assert backend.generated == 1  # only the full-input answer y_N
    assert backend.forced == 8  # one forward pass per coalition (2^3), base included
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value
    )
    if value_fn == "tf_jsd":
        assert output.result.grand_value == pytest.approx(0.0)
        assert output.result.base_value < 0.0


def test_tf_logprob_of_the_full_prompt_is_the_forced_score_of_the_base_answer(
    tmp_path: Path,
) -> None:
    backend = CountingBackend()
    output = ExplainRunner(
        _config(tmp_path, "tf_logprob"), backend=backend
    ).explain_sync(_snapshot())
    base = asyncio.run(backend.generate(_snapshot(), GenerationConfig()))
    expected = asyncio.run(backend.score_continuations(_snapshot(), [base.text]))[0]
    assert output.result.grand_value == pytest.approx(
        sum(expected.token_logprobs) / len(expected.token_logprobs)
    )


def test_a_backend_without_forcing_is_refused_before_generating(
    tmp_path: Path,
) -> None:
    runner = ExplainRunner(
        _config(tmp_path, "tf_logprob"),
        backend=NoForcingBackend(),  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigError, match="teacher-forced"):
        runner.explain_sync(_snapshot())


def test_reanalyze_reproduces_explain_from_archived_forced_scores(
    tmp_path: Path,
) -> None:
    runner = ExplainRunner(_config(tmp_path, "tf_jsd"), backend=CountingBackend())
    first = runner.explain_sync(_snapshot())
    again = runner.reanalyze(tmp_path / first.run_id)

    assert again.result.values == pytest.approx(first.result.values, abs=1e-12)
    assert again.result.base_value == pytest.approx(first.result.base_value)


def test_rescore_equals_a_fresh_explain_and_never_generates(tmp_path: Path) -> None:
    archive = tmp_path / "text"
    text_run = ExplainRunner(
        _config(tmp_path, "tfidf_cosine", archive={"path": f"{archive}/{{run_id}}"})
    ).explain_sync(_snapshot())
    root = archive / text_run.run_id
    fresh = ExplainRunner(
        _config(tmp_path, "tf_logprob"), backend=CountingBackend()
    ).explain_sync(_snapshot())

    with pytest.raises(ReanalyzeMismatchError, match="rescore"):
        ExplainRunner(_config(tmp_path, "tf_logprob")).reanalyze(root)

    backend = CountingBackend()
    rescoring = ExplainRunner(_config(tmp_path, "tf_logprob"), backend=backend)
    rescored = asyncio.run(rescoring.rescore(_snapshot(), root))

    assert backend.generated == 0
    assert backend.forced == 8
    assert rescored.result.values == pytest.approx(fresh.result.values, abs=1e-12)
    assert rescored.result.grand_value == pytest.approx(fresh.result.grand_value)


def test_rescore_refuses_a_backend_that_cannot_force(tmp_path: Path) -> None:
    text_run = ExplainRunner(_config(tmp_path, "tfidf_cosine")).explain_sync(
        _snapshot()
    )
    runner = ExplainRunner(
        _config(tmp_path, "tf_logprob"),
        backend=NoForcingBackend(),  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigError, match="continuations"):
        asyncio.run(runner.rescore(_snapshot(), tmp_path / text_run.run_id))


def test_rescore_refuses_a_conversation_with_other_players(tmp_path: Path) -> None:
    text_run = ExplainRunner(_config(tmp_path, "tfidf_cosine")).explain_sync(
        _snapshot()
    )
    other = ConversationSnapshot.from_turns((
        Turn((Message(role=Role.USER, text="alpha beta"),)),
    ))
    runner = ExplainRunner(_config(tmp_path, "tf_logprob"), backend=CountingBackend())
    with pytest.raises(ReanalyzeMismatchError, match="players"):
        asyncio.run(runner.rescore(other, tmp_path / text_run.run_id))


def test_efficiency_holds_under_sampled_decoding(tmp_path: Path) -> None:
    """Each coalition is generated once, so Shapley sums to v(N) - v(empty)."""
    config = _config(tmp_path, "tfidf_cosine", max_inflight=4).model_copy(
        update={"generation": GenerationConfig(temperature=0.7, top_k=5)}
    )
    runner = ExplainRunner(config, backend=CoinBackend())
    output = runner.explain_sync(_snapshot())

    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )
    assert output.manifest is not None
    assert output.manifest.decoding == {
        "temperature": 0.7,
        "top_k": 5,
        "seed": 42,
        "dedup": "auto",
    }
    again = runner.reanalyze(tmp_path / output.run_id)
    assert again.result.values == pytest.approx(output.result.values, abs=1e-9)


def test_repeated_masks_share_one_generation_at_positive_temperature(
    tmp_path: Path,
) -> None:
    backend = CountingBackend()
    config = _config(
        tmp_path, "tfidf_cosine", estimator="mc", budget={"evaluations": 40}
    ).model_copy(update={"generation": GenerationConfig(temperature=0.7, top_k=5)})
    ExplainRunner(config, backend=backend).explain_sync(_snapshot())

    assert backend.generated <= 8 + 1  # at most 2^3 coalitions plus the base


def test_tf_value_functions_are_registered_with_declared_metadata() -> None:
    registry = default_registry()
    for name in ("tf_logprob", "tf_jsd", "contrastive"):
        info = registry.resolve_type(PluginGroup.VALUE_FNS, name).info
        assert info.needs == ("teacher_forcing",)
        assert info.cost == "forward"


def test_generation_record_default_has_no_forced_scores() -> None:
    assert GenerationRecord(text="x").forced == {}
