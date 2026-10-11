"""SQLite-backed run archive for coalition evaluation history.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_archive.py]]
"""

import json
import logging
import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, Self

from ..errors import (
    ArchiveCorruptError,
    ArchiveExistsError,
    ArchiveNotFoundError,
    ManifestVersionError,
)
from ..masking.codec import PackedMask

logger = logging.getLogger(__name__)

BASE_GENERATION_FILE = "base_generation.txt"
"""Filename for the grand-coalition reference generation at the archive root."""

ARCHIVE_SCHEMA_VERSION: Final[int] = 4
"""SQLite ``user_version`` written by this release and required on load."""

MANIFEST_SCHEMA_VERSION: Final[int] = 4
"""Manifest ``schema_version`` written by this release and required on load."""


class ManifestPayloadSource(Protocol):
    """Manifest that serializes itself to the JSON stored as ``manifest.json``."""

    def to_dict(self) -> Mapping[str, object]:
        """Return the JSON-compatible manifest payload."""


def check_manifest_schema_version(payload: Mapping[str, object]) -> None:
    """Refuse manifests written with a different schema version.

    Raises:
        ManifestVersionError: If ``schema_version`` is missing or differs from
            ``MANIFEST_SCHEMA_VERSION``.
    """
    schema_version = payload.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        msg = (
            f"unsupported manifest schema_version {schema_version!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}. Re-run explain to rebuild "
            "the archive."
        )
        raise ManifestVersionError(msg)


def read_manifest_payload(path: Path) -> Mapping[str, object]:
    """Read ``manifest.json`` and return its top-level JSON object.

    Raises:
        ArchiveCorruptError: If the file is not UTF-8 JSON or its root is not an
            object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"cannot parse archive manifest {path}: {exc}"
        raise ArchiveCorruptError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"archive manifest {path} must hold a JSON mapping"
        raise ArchiveCorruptError(msg)
    return payload


@dataclass(frozen=True, slots=True)
class CoalitionRecord:
    """One persisted coalition evaluation row."""

    record_id: int
    """Monotonic archive identifier for the coalition row."""

    snapshot_id: str
    """Conversation snapshot identifier evaluated for this coalition."""

    coalition_key: str
    """Stable deduplication key for the coalition evaluation."""

    mask: PackedMask
    """Packed coalition mask bytes and original bit length."""

    absence_policy: str
    """Registered absence-policy identifier used for rendering."""

    model_id: str
    """Backend model identifier used for generation."""

    generation_text: str
    """Generated model text for the coalition."""

    elapsed_ms: float
    """Wall-clock generation time in milliseconds."""

    cache_hit: bool
    """Whether the generation was served from an in-memory cache."""

    logprobs: tuple[float, ...] | None = None
    """Token logprobs of the generation; ``None`` when the backend gave none."""


@dataclass(frozen=True, slots=True)
class CoalitionRecordDraft:
    """Input payload used when appending a coalition record."""

    snapshot_id: str
    """Conversation snapshot identifier evaluated for this coalition."""

    coalition_key: str
    """Stable deduplication key for the coalition evaluation."""

    mask: PackedMask
    """Packed coalition mask bytes and original bit length."""

    absence_policy: str
    """Registered absence-policy identifier used for rendering."""

    model_id: str
    """Backend model identifier used for generation."""

    generation_text: str
    """Generated model text for the coalition."""

    elapsed_ms: float
    """Wall-clock generation time in milliseconds."""

    cache_hit: bool
    """Whether the generation was served from an in-memory cache."""

    logprobs: tuple[float, ...] | None = None
    """Token logprobs of the generation; ``None`` when the backend gave none."""


class RunArchive:
    """Persist coalition records, generation text included, to one SQLite file.

    The database uses WAL journaling with ``synchronous=NORMAL``; rows are
    committed every ``flush_every`` appends and on :meth:`close`.
    """

    def __init__(
        self,
        root: Path,
        flush_every: int = 50,
        read_only: bool = False,
    ) -> None:
        """Connect to ``root/archive.sqlite``; use :meth:`open` or :meth:`load`.

        With ``read_only`` the database is opened without writing to it (no
        schema setup), so archives on read-only storage can be read.
        """
        self._root = root
        self._flush_every = flush_every
        self._pending = 0
        self._db_path = root / "archive.sqlite"
        if read_only:
            uri = f"{self._db_path.resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        else:
            self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        if not read_only:
            self._initialize()

    @classmethod
    def open(
        cls,
        root: Path,
        manifest: ManifestPayloadSource,
        flush_every: int = 50,
        on_exists: Literal["error", "overwrite"] = "error",
    ) -> Self:
        """Create a run archive directory and open its SQLite database.

        An existing archive whose run did not finish (no completion marker) is
        replaced whatever ``on_exists`` says, so a crashed run can be re-run.

        Raises:
            ArchiveExistsError: If ``root`` already holds the archive of a
                finished run and ``on_exists`` is ``"error"``.
        """
        db_path = root / "archive.sqlite"
        if db_path.exists():
            if on_exists == "error" and _archive_is_complete(db_path):
                msg = (
                    f"run archive already exists at {root}; delete it or set "
                    "explanation.archive.on_exists to 'overwrite'"
                )
                raise ArchiveExistsError(msg)
            logger.info("replacing existing run archive at %s", root)
            for name in (
                "archive.sqlite",
                "archive.sqlite-wal",
                "archive.sqlite-shm",
                "manifest.json",
                BASE_GENERATION_FILE,
            ):
                (root / name).unlink(missing_ok=True)
            # Archives written before 0.1.18 kept generations in blobs/.
            shutil.rmtree(root / "blobs", ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        archive = cls(root, flush_every=flush_every)
        archive.write_manifest(manifest)
        return archive

    @classmethod
    def load(cls, root: Path) -> Self:
        """Open an existing run archive read-only, without writing any file.

        Raises:
            ArchiveNotFoundError: If the manifest or the SQLite database is
                missing (a ``FileNotFoundError``).
            ArchiveCorruptError: If the manifest is not a JSON object or the
                database cannot be read.
            ManifestVersionError: If the manifest or SQLite schema version is not
                the one written by this release.
        """
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            msg = f"archive manifest missing: {manifest_path}"
            raise ArchiveNotFoundError(msg)
        check_manifest_schema_version(read_manifest_payload(manifest_path))
        db_path = root / "archive.sqlite"
        if not db_path.is_file():
            msg = f"archive database missing: {db_path}"
            raise ArchiveNotFoundError(msg)
        archive = cls(root, read_only=True)
        try:
            user_version = int(
                archive._conn.execute("PRAGMA user_version").fetchone()[0]
            )
        except sqlite3.DatabaseError as exc:
            archive._conn.close()
            msg = f"cannot read archive database {db_path}: {exc}"
            raise ArchiveCorruptError(msg) from exc
        if user_version != ARCHIVE_SCHEMA_VERSION:
            archive._conn.close()
            msg = (
                f"archive {root} has SQLite user_version {user_version}; "
                f"expected {ARCHIVE_SCHEMA_VERSION}. Re-run explain to rebuild it."
            )
            raise ManifestVersionError(msg)
        return archive

    def write_manifest(self, manifest: ManifestPayloadSource) -> None:
        """Write ``manifest`` to ``manifest.json``, replacing the previous one."""
        (self._root / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_base_generation(self, text: str) -> None:
        """Persist the grand-coalition reference generation before coalition rows."""
        path = self._root / BASE_GENERATION_FILE
        path.write_text(text, encoding="utf-8")

    def read_base_generation(self) -> str | None:
        """Return the archived base generation when present."""
        path = self._root / BASE_GENERATION_FILE
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def append(self, draft: CoalitionRecordDraft) -> int:
        """Append one coalition record and return its archive identifier."""
        cursor = self._conn.execute(
            """
            INSERT INTO coalition_records (
                snapshot_id,
                coalition_key,
                mask_words,
                mask_n_bits,
                absence_policy,
                model_id,
                elapsed_ms,
                cache_hit,
                generation_text,
                logprobs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft.snapshot_id,
                draft.coalition_key,
                draft.mask.words,
                draft.mask.n_bits,
                draft.absence_policy,
                draft.model_id,
                draft.elapsed_ms,
                int(draft.cache_hit),
                draft.generation_text,
                None if draft.logprobs is None else json.dumps(draft.logprobs),
            ),
        )
        if cursor.lastrowid is None:
            msg = "SQLite insert did not return a row id"
            raise RuntimeError(msg)
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()
        return int(cursor.lastrowid)

    def mark_complete(self) -> None:
        """Record that every coalition of the run was archived."""
        self._conn.execute(
            "INSERT OR REPLACE INTO archive_meta (key, value) VALUES ('complete', '1')"
        )
        self.flush()

    @property
    def is_complete(self) -> bool:
        """Return whether :meth:`mark_complete` ran for this archive."""
        row = self._conn.execute(
            "SELECT value FROM archive_meta WHERE key = 'complete'"
        ).fetchone()
        return row is not None and row[0] == "1"

    def flush(self) -> None:
        """Commit pending archive writes to disk."""
        self._conn.commit()
        self._pending = 0

    def history_lazy(self) -> Iterator[CoalitionRecord]:
        """Iterate coalition records one row at a time without bulk preloading."""
        cursor = self._conn.execute(
            """
            SELECT
                record_id,
                snapshot_id,
                coalition_key,
                mask_words,
                mask_n_bits,
                absence_policy,
                model_id,
                elapsed_ms,
                cache_hit,
                generation_text,
                logprobs
            FROM coalition_records
            ORDER BY record_id
            """
        )
        for row in cursor:
            yield CoalitionRecord(
                record_id=int(row["record_id"]),
                snapshot_id=str(row["snapshot_id"]),
                coalition_key=str(row["coalition_key"]),
                mask=PackedMask(
                    words=bytes(row["mask_words"]),
                    n_bits=int(row["mask_n_bits"]),
                ),
                absence_policy=str(row["absence_policy"]),
                model_id=str(row["model_id"]),
                generation_text=str(row["generation_text"]),
                elapsed_ms=float(row["elapsed_ms"]),
                cache_hit=bool(row["cache_hit"]),
                logprobs=_logprobs_from_json(row["logprobs"]),
            )

    def close(self) -> None:
        """Flush and close the archive database connection."""
        self.flush()
        self._conn.close()

    def __enter__(self) -> Self:
        """Return the archive for use in a ``with`` block."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Flush and close the archive."""
        self.close()

    def _initialize(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coalition_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                coalition_key TEXT NOT NULL,
                mask_words BLOB NOT NULL,
                mask_n_bits INTEGER NOT NULL,
                absence_policy TEXT NOT NULL,
                model_id TEXT NOT NULL,
                elapsed_ms REAL NOT NULL,
                cache_hit INTEGER NOT NULL,
                generation_text TEXT NOT NULL,
                logprobs TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS archive_meta "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.execute(f"PRAGMA user_version = {ARCHIVE_SCHEMA_VERSION}")
        self._conn.commit()


def _archive_is_complete(db_path: Path) -> bool:
    """Return whether the archive at ``db_path`` holds the completion marker.

    Databases of another schema version count as complete, so they are never
    replaced without ``on_exists='overwrite'``.
    """
    conn = sqlite3.connect(db_path)
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if user_version != ARCHIVE_SCHEMA_VERSION:
            return True
        row = conn.execute(
            "SELECT value FROM archive_meta WHERE key = 'complete'"
        ).fetchone()
    except sqlite3.DatabaseError:
        return True
    finally:
        conn.close()
    return row is not None and row[0] == "1"


def _logprobs_from_json(column: str | None) -> tuple[float, ...] | None:
    """Decode the ``logprobs`` column: a JSON list of floats, or NULL."""
    if column is None:
        return None
    return tuple(float(value) for value in json.loads(column))
