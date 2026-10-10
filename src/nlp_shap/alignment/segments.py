"""Transcript tokens aligned to audio time ranges.

@domain alignment
@see [[knowledge/architecture/alignment]]
@tested-by [[tests/alignment/test_segments.py]]
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One transcript token and the stretch of audio that speaks it."""

    token: str
    """Transcript token as given to the aligner."""

    token_index: int
    """Position of ``token`` in the transcript's token list."""

    start_time: float
    """Start in seconds."""

    end_time: float
    """End in seconds; greater than ``start_time``."""

    confidence: float
    """Mean model probability of the token's characters, in ``[0, 1]``."""

    boundary_refined: bool
    """Whether both edges moved to a low-energy point of the waveform."""

    sample_rate: int
    """Sample rate of the source waveform."""

    start_sample: int
    """First sample of the segment in the source waveform."""

    end_sample: int
    """Sample after the last one; greater than ``start_sample``."""

    audio: bytes = b""
    """The segment as WAV bytes when the aligner attached it, else empty."""

    def __post_init__(self) -> None:
        """Refuse empty or inverted ranges and confidences outside ``[0, 1]``."""
        if self.end_time <= self.start_time:
            msg = "end_time must be greater than start_time"
            raise ValueError(msg)
        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must lie in [0, 1]"
            raise ValueError(msg)
        if self.end_sample <= self.start_sample:
            msg = "end_sample must be greater than start_sample"
            raise ValueError(msg)

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.end_time - self.start_time
