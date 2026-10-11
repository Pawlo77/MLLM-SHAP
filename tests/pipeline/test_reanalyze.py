"""Direct tests for reanalyzing hand-built run archives.

@see [[src/mllm_shap/pipeline/reanalyze.py]]
"""

from pathlib import Path

import pytest

from mllm_shap import ExplainConfig, PluginGroup, PluginRegistry
from mllm_shap.domain.estimands import Estimand
from mllm_shap.errors import ReanalyzeMismatchError
from mllm_shap.masking.codec import MaskCodec
from mllm_shap.pipeline.manifest import RunManifest
from mllm_shap.pipeline.reanalyze import reanalyze_sync
from mllm_shap.plugins import default_registry
from mllm_shap.runtime.archive import CoalitionRecordDraft, RunArchive
from tests.pipeline._doubles import WordCountValue

# (mask, generated text); the word count of the text is the coalition payoff.
_ROWS = (
    ((False, False), ""),
    ((True, False), "x"),
    ((False, True), "y y"),
    ((True, True), "x y y"),
    # Duplicate coalition key (a cache hit archived twice) with different text:
    # reanalyze must score the first row only.
    ((True, False), "x x x x x"),
)


def _config(**explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "explanation": {
            "estimator": "exact",
            "value_fn": "word_count",
            "absence_policy": "pad",
            "archive": {"path": ""},
            **explanation,
        },
    })


def _registry() -> PluginRegistry:
    registry = default_registry()
    registry.register(PluginGroup.VALUE_FNS, "word_count", WordCountValue)
    return registry


def _write_archive(root: Path) -> None:
    config = _config()
    manifest = RunManifest(
        estimand=Estimand.SHAPLEY,
        run_id="reanalyze-test",
        snapshot_id="snap",
        estimator="exact",
        value_fn="word_count",
        normalizer="identity",
        absence_policy="pad",
        players="tokens",
        backend_kind="mock",
        model_id="mock",
        player_ids=("p0", "p1"),
        config=config,
    )
    with RunArchive.open(root, manifest) as archive:
        archive.write_base_generation("x y y")
        for present, text in _ROWS:
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id="snap",
                    coalition_key="".join("1" if bit else "0" for bit in present),
                    mask=MaskCodec.pack(present),
                    absence_policy="pad",
                    model_id="mock",
                    generation_text=text,
                    elapsed_ms=0.0,
                    cache_hit=False,
                )
            )
        archive.mark_complete()


def test_reanalyze_scores_each_coalition_key_once(tmp_path: Path) -> None:
    """A repeated coalition key is scored from its first row only."""
    root = tmp_path / "run"
    _write_archive(root)

    output = reanalyze_sync(root, _config(), _registry())

    assert output.metrics.requested == 4
    assert output.metrics.executed == 0
    assert output.result.values == pytest.approx([1.0, 2.0])


def test_reanalyze_reports_base_and_grand_values(tmp_path: Path) -> None:
    """``base_value`` is v(empty) and ``grand_value`` is v(N)."""
    root = tmp_path / "run"
    _write_archive(root)

    output = reanalyze_sync(root, _config(), _registry())

    assert output.result.base_value == 0.0
    assert output.result.grand_value == 3.0
    assert output.run_id == "reanalyze-test"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("estimator", "mc"),
        ("absence_policy", "delete"),
        ("players", "messages"),
    ],
)
def test_reanalyze_rejects_mismatched_settings(
    tmp_path: Path, field: str, value: str
) -> None:
    """Changing how coalitions were sampled or rendered is refused."""
    root = tmp_path / "run"
    _write_archive(root)
    overrides: dict[str, object] = {field: value}
    if value == "mc":
        overrides["budget"] = {"fraction": 1.0}

    with pytest.raises(ReanalyzeMismatchError, match=field):
        reanalyze_sync(root, _config(**overrides), _registry())


def test_reanalyze_rejects_archive_without_base_generation(tmp_path: Path) -> None:
    """An archive with rows but no base generation cannot be rescored."""
    root = tmp_path / "run"
    _write_archive(root)
    (root / "base_generation.txt").unlink()

    with pytest.raises(ValueError, match="missing base generation"):
        reanalyze_sync(root, _config(), _registry())
