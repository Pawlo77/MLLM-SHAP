"""Tests for run archive persistence.

@see [[src/mllm_shap/runtime/archive.py]]
"""

import dataclasses
import json
import sqlite3
from pathlib import Path

import pytest

from mllm_shap.domain.estimands import Estimand
from mllm_shap.errors import (
    ArchiveCorruptError,
    ArchiveExistsError,
    ArchiveNotFoundError,
    ManifestVersionError,
)
from mllm_shap.masking.codec import MaskCodec
from mllm_shap.pipeline.manifest import RunManifest, parse_manifest
from mllm_shap.runtime.archive import CoalitionRecordDraft, RunArchive


def test_run_archive_round_trips_records(
    tmp_path: Path,
    run_manifest: RunManifest,
    sample_snapshot,
) -> None:
    """One hundred coalition rows round-trip through SQLite and blob storage."""
    root = tmp_path / "run-1"
    packed = MaskCodec.pack((True, False, True))

    with RunArchive.open(root, run_manifest, flush_every=25) as archive:
        for index in range(100):
            archive.append(
                CoalitionRecordDraft(
                    snapshot_id=sample_snapshot.snapshot_id,
                    coalition_key=f"key-{index % 10}",
                    mask=packed,
                    absence_policy="delete",
                    model_id="mock",
                    generation_text=f"generation-{index}",
                    elapsed_ms=float(index),
                    cache_hit=index % 2 == 0,
                )
            )

    manifest_payload = parse_manifest(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    assert manifest_payload.estimand is Estimand.SHAPLEY
    assert manifest_payload.run_id == run_manifest.run_id

    with RunArchive.load(root) as archive:
        records = list(archive.history_lazy())

    assert len(records) == 100
    assert records[0].generation_text == "generation-0"
    assert records[-1].generation_text == "generation-99"
    assert records[10].elapsed_ms == 10.0


def test_archive_stores_generation_text_inline(
    tmp_path: Path,
    run_manifest: RunManifest,
    sample_snapshot,
) -> None:
    """Generation text lives in the SQLite row: no blob files, WAL journal."""
    root = tmp_path / "run-2"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("inline text"))

    assert not (root / "blobs").exists()
    with sqlite3.connect(root / "archive.sqlite") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
        (text,) = conn.execute(
            "SELECT generation_text FROM coalition_records"
        ).fetchone()
    assert text == "inline text"


def test_archive_round_trips_logprobs(
    tmp_path: Path, run_manifest: RunManifest
) -> None:
    """Token logprobs round-trip per row; rows without them read back as None."""
    root = tmp_path / "run-logprobs"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(dataclasses.replace(_draft("a"), logprobs=(-0.5, -1.25)))
        archive.append(dataclasses.replace(_draft("b"), logprobs=()))
        archive.append(_draft("c"))
    with RunArchive.load(root) as archive:
        logprobs = [record.logprobs for record in archive.history_lazy()]
    assert logprobs == [(-0.5, -1.25), (), None]


def test_archive_completion_marker(tmp_path: Path, run_manifest: RunManifest) -> None:
    root = tmp_path / "run-complete"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("a"))
        assert not archive.is_complete
        archive.mark_complete()
    with RunArchive.load(root) as archive:
        assert archive.is_complete


def test_run_archive_stores_base_generation(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """Base generation text round-trips through the archive root file."""
    root = tmp_path / "run-base"
    with RunArchive.open(root, run_manifest) as archive:
        archive.write_base_generation("grand coalition text")

    with RunArchive.load(root) as archive:
        assert archive.read_base_generation() == "grand coalition text"


def _draft(text: str) -> CoalitionRecordDraft:
    return CoalitionRecordDraft(
        snapshot_id="snap",
        coalition_key=f"key-{text}",
        mask=MaskCodec.pack((True, False)),
        absence_policy="pad",
        model_id="mock",
        generation_text=text,
        elapsed_ms=0.0,
        cache_hit=False,
    )


def test_open_existing_archive_raises_by_default(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """Opening the archive of a finished run refuses to append to it."""
    root = tmp_path / "run-exists"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("first"))
        archive.mark_complete()
    with pytest.raises(ArchiveExistsError, match="on_exists"):
        RunArchive.open(root, run_manifest)
    with RunArchive.load(root) as archive:
        assert [r.generation_text for r in archive.history_lazy()] == ["first"]


def test_open_existing_archive_overwrite_replaces_rows(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """``on_exists='overwrite'`` starts the archive from scratch."""
    root = tmp_path / "run-overwrite"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("first"))
        archive.write_base_generation("old base")
    with RunArchive.open(root, run_manifest, on_exists="overwrite") as archive:
        archive.append(_draft("second"))
    with RunArchive.load(root) as archive:
        assert [r.generation_text for r in archive.history_lazy()] == ["second"]
        assert archive.read_base_generation() is None


def test_load_rejects_version_two_archives(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """Archives written by 0.1.17 (user_version 2) are refused with a clear message."""
    root = tmp_path / "run-v2"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("old"))
    with sqlite3.connect(root / "archive.sqlite") as conn:
        conn.execute("PRAGMA user_version = 2")
    with pytest.raises(ManifestVersionError, match="Re-run explain"):
        RunArchive.load(root)


def test_load_rejects_unversioned_database(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """Archives written before SQLite ``user_version`` 2 are refused on load."""
    root = tmp_path / "run-old"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("old"))
    with sqlite3.connect(root / "archive.sqlite") as conn:
        conn.execute("PRAGMA user_version = 0")
    with pytest.raises(ManifestVersionError, match="user_version"):
        RunArchive.load(root)


def test_open_replaces_incomplete_archive_by_default(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """The archive of a crashed run is replaced even with ``on_exists='error'``."""
    root = tmp_path / "run-crashed"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("partial"))
        archive.write_base_generation("old base")
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("rerun"))
    with RunArchive.load(root) as archive:
        assert [r.generation_text for r in archive.history_lazy()] == ["rerun"]


def test_load_missing_database_raises_without_creating_it(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    root = tmp_path / "run-no-db"
    with RunArchive.open(root, run_manifest) as archive:
        archive.mark_complete()
    (root / "archive.sqlite").unlink()
    for suffix in ("-wal", "-shm"):
        (root / f"archive.sqlite{suffix}").unlink(missing_ok=True)

    with pytest.raises(FileNotFoundError, match=r"archive\.sqlite"):
        RunArchive.load(root)
    assert not (root / "archive.sqlite").exists()


def test_history_lazy_fetches_rows_on_demand(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """A partly consumed cursor has fetched only the rows it yielded."""
    root = tmp_path / "run-lazy"
    with RunArchive.open(root, run_manifest, flush_every=1000) as archive:
        for index in range(2000):
            archive.append(_draft(f"row-{index}"))

    fetched = 0

    class CountingCursor:
        def __init__(self, cursor: sqlite3.Cursor) -> None:
            self._cursor = cursor

        def __iter__(self) -> "CountingCursor":
            return self

        def __next__(self) -> sqlite3.Row:
            nonlocal fetched
            row = next(self._cursor)
            fetched += 1
            return row

    class CountingConnection:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, *args: object) -> object:
            cursor = self._conn.execute(sql, *args)
            return CountingCursor(cursor) if "coalition_records" in sql else cursor

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    with RunArchive.load(root) as archive:
        archive._conn = CountingConnection(archive._conn)  # type: ignore[assignment]  # test spy
        history = archive.history_lazy()
        texts = [next(history).generation_text for _ in range(3)]
        assert texts == ["row-0", "row-1", "row-2"]
        assert fetched == 3
        assert sum(1 for _ in history) == 1997
        assert fetched == 2000


def test_load_reads_a_read_only_archive_without_writing(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    """``load`` writes nothing, so archives on read-only storage can be read."""
    root = tmp_path / "run-read-only"
    with RunArchive.open(root, run_manifest) as archive:
        archive.append(_draft("kept"))
        archive.mark_complete()
    db_path = root / "archive.sqlite"
    before = db_path.read_bytes()
    db_path.chmod(0o444)
    try:
        with RunArchive.load(root) as archive:
            assert archive.is_complete
            assert [r.generation_text for r in archive.history_lazy()] == ["kept"]
    finally:
        db_path.chmod(0o644)
    assert db_path.read_bytes() == before


def test_load_refuses_unreadable_manifest_and_database(
    tmp_path: Path,
    run_manifest: RunManifest,
) -> None:
    root = tmp_path / "run-damaged"
    with RunArchive.open(root, run_manifest) as archive:
        archive.mark_complete()
    manifest_path = root / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")

    manifest_path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ArchiveCorruptError, match="mapping"):
        RunArchive.load(root)
    manifest_path.write_text("{oops", encoding="utf-8")
    with pytest.raises(ArchiveCorruptError, match="cannot parse"):
        RunArchive.load(root)

    manifest_path.write_text(manifest_text, encoding="utf-8")
    (root / "archive.sqlite").write_bytes(b"garbage" * 512)
    with pytest.raises(ArchiveCorruptError, match="archive database"):
        RunArchive.load(root)

    manifest_path.unlink()
    with pytest.raises(ArchiveNotFoundError, match="manifest"):
        RunArchive.load(root)
