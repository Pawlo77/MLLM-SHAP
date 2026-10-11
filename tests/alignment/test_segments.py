"""Aligned audio segment records.

@see [[src/mllm_shap/alignment/segments.py]]
"""

import pytest

from mllm_shap.alignment.segments import AudioSegment


def _segment(**changes: object) -> AudioSegment:
    fields: dict[str, object] = {
        "token": "hello",
        "token_index": 0,
        "start_time": 0.5,
        "end_time": 0.75,
        "ctc_start_time": 0.52,
        "ctc_end_time": 0.74,
        "confidence": 0.9,
        "boundary_refined": True,
        "sample_rate": 16000,
        "start_sample": 8000,
        "end_sample": 12000,
    }
    fields.update(changes)
    return AudioSegment(**fields)  # type: ignore[arg-type]  # fields typed above


def test_duration_is_end_minus_start() -> None:
    assert _segment().duration == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"end_time": 0.5}, "end_time"),
        ({"confidence": 1.5}, "confidence"),
        ({"end_sample": 8000}, "end_sample"),
    ],
)
def test_invalid_segments_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _segment(**changes)
