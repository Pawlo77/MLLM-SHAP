"""Generation payloads scored by value functions.

@domain domain
@see [[knowledge/architecture/domain]]
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ForcedScore:
    """Teacher-forced scores of one fixed continuation under one prompt."""

    token_logprobs: tuple[float, ...]
    """``log p(y_t | y_<t, prompt)`` for each continuation token."""

    top_ids: tuple[tuple[int, ...], ...] = ()
    """Per position, the ids of the ``k`` most probable next tokens (empty when
    the scorer was asked for no distributions)."""

    top_logprobs: tuple[tuple[float, ...], ...] = ()
    """Per position, the log probabilities of ``top_ids``; the mass outside them
    is ``1 - sum(exp(...))``."""


def forced_to_json(forced: Mapping[str, ForcedScore]) -> str:
    """Encode scores keyed by continuation text as a JSON object."""
    return json.dumps({
        text: [score.token_logprobs, score.top_ids, score.top_logprobs]
        for text, score in forced.items()
    })


def forced_from_json(column: str | None) -> dict[str, ForcedScore]:
    """Decode :func:`forced_to_json`; ``None`` gives an empty mapping."""
    if column is None:
        return {}
    return {
        text: ForcedScore(
            tuple(logprobs),
            tuple(tuple(row) for row in ids),
            tuple(tuple(row) for row in tops),
        )
        for text, (logprobs, ids, tops) in json.loads(column).items()
    }


@dataclass(frozen=True)
class GenerationRecord:
    """Concrete generation output consumed by built-in value functions."""

    text: str
    """Generated text payload."""

    text_token_rows: tuple[tuple[int, ...], ...] = ()
    """Token rows hashed for TF-IDF similarity (multimodal text stream)."""

    audio_token_rows: tuple[tuple[int, ...], ...] = ()
    """Token rows hashed for TF-IDF similarity (multimodal audio stream)."""

    logprobs: tuple[float, ...] | None = None
    """Natural-log probability of each generated token; ``None`` when the
    backend did not return them."""

    embedding: tuple[float, ...] = ()
    """Static embedding vector for embedding-based value functions."""

    contextual_embedding: tuple[float, ...] = ()
    """Contextual embedding vector for U2-style scoring."""

    forced: Mapping[str, ForcedScore] = field(default_factory=dict)
    """Teacher-forced scores keyed by continuation text, filled for value
    functions that need them."""

    audio_path: str | None = None
    """Waveform file of a spoken answer, for speech value functions."""
