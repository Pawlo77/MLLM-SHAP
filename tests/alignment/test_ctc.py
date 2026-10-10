"""CTC forced alignment with known paths.

@see [[src/nlp_shap/alignment/ctc.py]]
"""

import numpy as np
import pytest

from nlp_shap.alignment.ctc import forced_align, target_spans

LOW = -20.0


def _log_probs(frames: list[int], vocab: int = 4) -> np.ndarray:
    """One frame per entry; each frame puts almost all mass on its label."""
    lp = np.full((len(frames), vocab), LOW)
    lp[np.arange(len(frames)), frames] = 0.0
    return lp


def test_forced_align_follows_the_dominant_labels() -> None:
    # blank=0; frames say: blank A A blank B
    path = forced_align(_log_probs([0, 1, 1, 0, 2]), [1, 2], blank=0)
    assert path.tolist() == [-1, 0, 0, -1, 1]


def test_forced_align_separates_repeated_labels_with_a_blank() -> None:
    # "AA" needs a blank between the two A frames even if the model says A A A.
    path = forced_align(_log_probs([1, 1, 1]), [1, 1], blank=0)
    assert path.tolist()[0] == 0
    assert path.tolist()[-1] == 1
    assert -1 in path.tolist()


def test_forced_align_rejects_too_few_frames() -> None:
    with pytest.raises(ValueError, match="frames"):
        forced_align(_log_probs([1]), [1, 2], blank=0)


def test_target_spans_report_frames_and_confidence() -> None:
    lp = _log_probs([0, 1, 1, 0, 2])
    spans = target_spans(forced_align(lp, [1, 2], blank=0), lp, [1, 2])
    assert [(s.start, s.end) for s in spans] == [(1, 3), (4, 5)]
    assert [s.confidence for s in spans] == pytest.approx([1.0, 1.0])
