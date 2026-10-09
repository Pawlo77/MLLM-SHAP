"""Build generation records from generated text for value-function scoring.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_generation_records.py]]
"""

import hashlib
import struct
import warnings
from typing import Final

from ..domain.generation import GenerationRecord

EMBED_DIM: Final[int] = 8
"""Dimension of the deterministic hash embedding used by mock and archive paths."""


def generation_record_from_text(
    text: str, context_id: str | None = None
) -> GenerationRecord:
    """Build a record whose token rows and embeddings derive from ``text`` only.

    ``explain`` and ``reanalyze`` both score records built here, so a value
    function sees the same inputs whether the text came from a backend or an
    archive. ``contextual_embedding`` hashes ``text`` with a separator so it
    differs from ``embedding``; it takes no prompt or snapshot id, which a
    backend sees for the rendered coalition and ``reanalyze`` never sees.

    Args:
        text: Generated text.
        context_id: Deprecated and ignored; passing it emits a
            ``DeprecationWarning``. It salted ``contextual_embedding`` until
            0.1.18, which made backend records differ from ``reanalyze``.
    """
    if context_id is not None:
        warnings.warn(
            "generation_record_from_text(context_id=...) is ignored and will be "
            "removed in 0.2.0",
            DeprecationWarning,
            stacklevel=2,
        )
    rows = tuple((stable_token_id(token),) for token in text.split())
    return GenerationRecord(
        text=text,
        text_token_rows=rows,
        embedding=hash_embedding(text),
        contextual_embedding=hash_embedding(f"{text}\0"),
    )


def stable_token_id(token: str) -> int:
    """Return a process-independent 56-bit id for ``token``."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=7).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def hash_embedding(payload: str) -> tuple[float, ...]:
    """Map ``payload`` to a fixed-size float vector in [-1, 1] via SHA-256."""
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return tuple(
        (struct.unpack_from("<I", digest, index * 4)[0] / 0xFFFFFFFF) * 2.0 - 1.0
        for index in range(EMBED_DIM)
    )
