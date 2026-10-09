"""SQLite cache of practical-test generations and judge verdicts.

Keys are sha256 digests of the model identity (id, revision, device, dtype) and
the exact input, so a pin or device change never reuses an old answer.
"""

import dataclasses
import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from nlp_shap.backends._chat import snapshot_to_chat_messages
from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import ConversationSnapshot
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.protocols.backend import GenerativeBackend

DEFAULT_PATH = Path.home() / ".cache" / "nlp-shap" / "practical" / "cache.sqlite"
"""Cache file shared by every practical run on this machine."""

STATS: Counter[str] = Counter()
"""Process-wide hit and miss counts, read by the results writer."""


def cache_key(*parts: object) -> str:
    """Return the sha256 of ``parts`` as canonical JSON."""
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PracticalCache:
    """Key-value store of JSON values, one table for every kind."""

    def __init__(self, path: Path) -> None:
        """Open or create the SQLite file at ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS entries "
            "(kind TEXT, key TEXT, value TEXT, PRIMARY KEY (kind, key))"
        )

    def get(self, kind: str, key: str) -> object | None:
        """Return the stored value, or ``None``."""
        row = self._db.execute(
            "SELECT value FROM entries WHERE kind = ? AND key = ?", (kind, key)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, kind: str, key: str, value: object) -> None:
        """Store ``value`` under ``(kind, key)``."""
        self._db.execute(
            "INSERT OR REPLACE INTO entries VALUES (?, ?, ?)",
            (kind, key, json.dumps(value)),
        )
        self._db.commit()


class CachedBackend:
    """Serve a logprob backend's generations from the cache when present."""

    def __init__(
        self,
        backend: GenerativeBackend,
        cache: PracticalCache,
        identity: Mapping[str, str],
    ) -> None:
        """Wrap ``backend``; ``identity`` names the model, revision and device."""
        self._backend = backend
        self._cache = cache
        self._identity = dict(identity)
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        """Return the wrapped backend's model id."""
        return self._backend.model_id

    @property
    def returns_logprobs(self) -> bool:
        """Records keep the logprobs the wrapped backend returned."""
        return True

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        """Return the cached record for this chat, else generate and store it."""
        key = cache_key(
            self._identity,
            snapshot_to_chat_messages(snapshot),
            generation.model_dump(mode="json"),
        )
        stored = self._cache.get("generation", key)
        if isinstance(stored, dict):
            self.hits += 1
            STATS["generation_hits"] += 1
            return _record(stored)
        self.misses += 1
        STATS["generation_misses"] += 1
        record = await self._backend.generate(snapshot, generation)
        assert isinstance(record, GenerationRecord)
        self._cache.put("generation", key, dataclasses.asdict(record))
        return record


def _record(payload: dict[str, list[object] | str | None]) -> GenerationRecord:
    logprobs = payload["logprobs"]
    return GenerationRecord(
        text=str(payload["text"]),
        text_token_rows=_rows(payload["text_token_rows"]),
        audio_token_rows=_rows(payload["audio_token_rows"]),
        logprobs=None if logprobs is None else tuple(_floats(logprobs)),
        embedding=tuple(_floats(payload["embedding"])),
        contextual_embedding=tuple(_floats(payload["contextual_embedding"])),
    )


def _rows(value: object) -> tuple[tuple[int, ...], ...]:
    assert isinstance(value, list)
    return tuple(tuple(int(i) for i in row) for row in value)


def _floats(value: object) -> list[float]:
    assert isinstance(value, list)
    return [float(v) for v in value]
