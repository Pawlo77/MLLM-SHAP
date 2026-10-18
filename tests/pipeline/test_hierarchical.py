"""Hierarchical runs: call savings, agreement with the flat run, archives, config.

@see [[src/mllm_shap/pipeline/runner.py]]
"""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import ConfigError
from mllm_shap.value.generation_records import generation_record_from_text

PARAGRAPHS = 10
SENTENCES = 8
KEYS = {(2, 1), (2, 4), (2, 6), (7, 0), (7, 5)}


def _text() -> str:
    return "\n\n".join(
        " ".join(
            f"Point {p}.{k} reads KEY{p}{k} here."
            if (p, k) in KEYS
            else f"Point {p}.{k} says nothing."
            for k in range(SENTENCES)
        )
        for p in range(PARAGRAPHS)
    )


class MarkerBackend:
    """Answers with the key markers it can read, so only key sentences matter."""

    model_id = "marker"

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        _ = generation
        text = " ".join(m.text for m in snapshot.messages())
        return generation_record_from_text(
            "answer " + " ".join(re.findall(r"KEY\d\d", text))
        )


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=_text()),)),
    ))


def _config(seed: int = 0, archive: str = "", **explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 8},
        "explanation": {
            "estimator": "leverage",
            "players": "sentences",
            "budget": {"per_player": 4},
            "absence_policy": "delete",
            "archive": {"path": archive},
            "seed": seed,
            "hierarchical": {"keep_fraction": 0.2},
            **explanation,
        },
    })


def _top5(output_values: tuple[float, ...], ids: tuple[str, ...]) -> set[str]:
    ranked = sorted(range(len(ids)), key=lambda i: -abs(output_values[i]))
    return {ids[i] for i in ranked[:5]}


def test_hierarchy_saves_calls_and_finds_the_flat_top_sentences() -> None:
    agree = 0
    seeds = range(10)
    for seed in seeds:
        runner = ExplainRunner(_config(seed), backend=MarkerBackend())
        flat = runner.explain_sync(_snapshot())
        tiered = runner.explain_hierarchical_sync(_snapshot())
        assert flat.player_set is not None
        assert tiered.fine.player_set is not None
        calls = tiered.coarse.metrics.requested + tiered.fine.metrics.requested
        assert flat.metrics.requested >= 3 * calls
        agree += _top5(flat.result.values, flat.player_set.player_ids) == _top5(
            tiered.fine.result.values, tiered.fine.player_set.player_ids
        )
        assert tiered.kept_groups == (2, 7)
    assert agree >= 0.8 * len(seeds)


def test_both_games_are_archived_with_the_threshold(tmp_path: Path) -> None:
    config = _config(archive=str(tmp_path / "{run_id}"))
    output = ExplainRunner(config, backend=MarkerBackend()).explain_hierarchical_sync(
        _snapshot()
    )
    assert output.coarse.run_id != output.fine.run_id
    for level in (output.coarse, output.fine):
        assert (tmp_path / level.run_id).is_dir()
        hierarchical = level.manifest.config.explanation.hierarchical
        assert hierarchical is not None
        assert hierarchical.keep_fraction == 0.2
    assert output.coarse.player_set is not None
    assert output.coarse.player_set.num_players == PARAGRAPHS


def test_hierarchical_needs_its_settings_and_sentence_players() -> None:
    plain = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "explanation": {"players": "sentences"},
    })
    with pytest.raises(ConfigError, match=r"explanation\.hierarchical"):
        ExplainRunner(plain, backend=MarkerBackend()).explain_hierarchical_sync(
            _snapshot()
        )
    with pytest.raises(ValidationError, match="players: sentences"):
        _config(players="tokens")
