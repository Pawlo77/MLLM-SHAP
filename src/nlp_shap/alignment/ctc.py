"""CTC forced alignment (Viterbi over the blank-extended target sequence).

@domain alignment
@see [[knowledge/architecture/alignment]]
@tested-by [[tests/alignment/test_ctc.py]]
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BLANK_STATE: int = -1
"""Path value of a frame aligned to the blank label."""


@dataclass(frozen=True, slots=True)
class TargetSpan:
    """Frames ``[start, end)`` that one target label occupies."""

    start: int
    """First frame of the label."""

    end: int
    """Frame after the last one."""

    confidence: float
    """Mean probability of the label over its frames."""


def forced_align(
    log_probs: NDArray[np.floating], targets: Sequence[int], blank: int = 0
) -> NDArray[np.int64]:
    """Return, per frame, the index into ``targets`` it aligns to, or ``BLANK_STATE``.

    The most probable monotonic path over ``targets`` with optional blanks between
    labels and a mandatory blank between two equal neighbours (standard CTC).

    Raises:
        ValueError: If ``log_probs`` has too few frames for ``targets``.
    """
    frames, n = len(log_probs), len(targets)
    ext = np.full(2 * n + 1, blank)
    ext[1::2] = targets
    states = len(ext)
    # A label state may skip the blank before it unless it repeats the previous label.
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    score = np.full(states, -np.inf)
    score[:2] = log_probs[0, ext[:2]]
    back = np.zeros((frames, states), dtype=np.int64)
    for t in range(1, frames):
        stay, step = score, np.concatenate(([-np.inf], score[:-1]))
        jump = np.where(skip, np.concatenate(([-np.inf] * 2, score[:-2])), -np.inf)
        moves = np.stack([stay, step, jump])
        best = np.argmax(moves, axis=0)
        back[t] = np.arange(states) - best
        score = moves[best, np.arange(states)] + log_probs[t, ext]
    end = states - 1 if score[-1] >= score[-2] else states - 2
    if not np.isfinite(score[end]):
        msg = f"{frames} frames cannot hold {n} targets"
        raise ValueError(msg)
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = (end - 1) // 2 if end % 2 else BLANK_STATE
        end = back[t, end]
    return path


def target_spans(
    path: NDArray[np.int64], log_probs: NDArray[np.floating], targets: Sequence[int]
) -> list[TargetSpan]:
    """Return the frame span and mean probability of every target in ``path``."""
    spans: list[TargetSpan] = []
    for index, label in enumerate(targets):
        frames = np.flatnonzero(path == index)
        probability = np.exp(log_probs[frames, label]).mean()
        spans.append(
            TargetSpan(int(frames[0]), int(frames[-1]) + 1, float(probability))
        )
    return spans
