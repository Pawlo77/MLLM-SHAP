"""Real Wav2Vec2 alignment of a recorded sentence.

@see [[src/mllm_shap/alignment/sgpa.py]]
"""

import itertools
from pathlib import Path
from typing import Any

import pytest

from mllm_shap import AudioPart
from tests.practical._harness import MODELS, Scenario, require_weights, scenario_cases

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("scenario", scenario_cases("alignment.sgpa"))
def test_words_get_ordered_contiguous_confident_segments(
    scenario: Scenario, practical_record: dict[str, Any]
) -> None:
    """Every word is aligned, in order, without gaps, inside the clip."""
    pytest.importorskip("librosa")
    pytest.importorskip("transformers")
    from mllm_shap.alignment import SpectrogramGuidedAligner, Wav2Vec2Ctc

    spec = MODELS["alignment"]
    require_weights(spec)
    model = Wav2Vec2Ctc(spec.model_id, revision=spec.revision)
    audio = AudioPart(ROOT / scenario.expect["audio"], sample_rate=16000)
    words = scenario.snapshot.messages()[0].text.split()
    segments = SpectrogramGuidedAligner(model).align(words, audio)
    practical_record["numbers"] = {
        s.token: [round(s.start_time, 3), round(s.end_time, 3)] for s in segments
    }
    practical_record["properties"] = {
        "refined": sum(s.boundary_refined for s in segments),
        "min_confidence": min(s.confidence for s in segments),
    }

    assert [s.token for s in segments] == words
    for left, right in itertools.pairwise(segments):
        assert left.end_time == right.start_time
    assert 0.0 < segments[0].start_time < segments[-1].end_time <= 3.5
    assert min(s.confidence for s in segments) >= scenario.expect["min_confidence"]
    assert max(segments, key=lambda s: s.duration).token == scenario.expect["longest"]
    assert sum(s.boundary_refined for s in segments) >= len(segments) // 2
