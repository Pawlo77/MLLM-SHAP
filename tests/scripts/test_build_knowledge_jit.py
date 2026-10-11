"""Tests for the just-in-time context entries generated from the vault.

@see [[scripts/build_knowledge.py]]
"""

import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def builder(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/build_knowledge.py."""
    return script("build_knowledge")


@pytest.fixture
def vault_copy(tmp_path: Path) -> Path:
    """Copy the files the builder reads into a temporary root."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "src" / "mllm_shap", root / "src" / "mllm_shap")
    shutil.copytree(
        REPO / "knowledge",
        root / "knowledge",
        ignore=shutil.ignore_patterns("generated"),
    )
    for rel in ("pyproject.toml", "docs/release_notes.rst", "scripts/glossary.yaml"):
        source = REPO / rel
        if source.is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, root / rel)
    return root


def split(text: str) -> tuple[dict[str, object], list[str]]:
    """Return the frontmatter and body lines of an entry."""
    match = re.match(r"---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    assert match is not None
    meta = yaml.safe_load(match.group(1))
    assert isinstance(meta, dict)
    return meta, match.group(2).splitlines()


def test_paths_entry_per_architecture_note(
    builder: ModuleType, vault_copy: Path
) -> None:
    """Each architecture note yields a paths entry scoped to its package."""
    built = builder.build_jit(vault_copy)
    layer = vault_copy / builder.JIT / "paths" / builder.JIT_LAYER
    notes = sorted(
        p.stem for p in (vault_copy / "knowledge" / "architecture").glob("*.md")
    )
    assert sorted(p.stem for p in built if p.parent == layer and p.suffix == ".md") == (
        notes
    )
    meta, body = split(built[layer / "runtime.md"])
    assert meta["match"] == "src/mllm_shap/runtime/"
    assert meta["title"] == "runtime package (architecture note)"
    assert isinstance(meta["description"], str)
    assert "[[" not in meta["description"]
    assert "## Responsibility" in body
    assert "## Invariants" in body
    assert "## Known issues" in body
    assert "## Public surface" not in body
    text = "\n".join(body)
    assert "[[" not in text
    assert "(`knowledge/issues/Fixed issues.md`)" in text
    assert len(body) <= builder.JIT_MAX_LINES
    core, _ = split(built[layer / "core.md"])
    assert core["match"] == "src/mllm_shap/[^/]+[.]py$"
    assert re.search(str(core["match"]), "/abs/src/mllm_shap/errors.py")
    assert not re.search(str(core["match"]), "src/mllm_shap/runtime/store.py")


def test_vocabulary_entry_keys_term_and_aliases(
    builder: ModuleType, vault_copy: Path
) -> None:
    """Concept notes become vocabulary entries keyed on the term and its aliases;
    the everyday alias "cc" is dropped."""
    (
        vault_copy / "knowledge" / "concepts" / "Complementary contribution.md"
    ).write_text(
        "---\ntitle: Complementary contribution\ntype: concept\ndomain: estimation\n"
        "aliases: [CC]\ncreated: 2026-10-07\n---\n\n# Complementary contribution\n\n"
        "The CC of T is v(T) \u2212 v(N \u2216 T).\n"
    )
    built = builder.build_jit(vault_copy)
    layer = vault_copy / builder.JIT / "vocabulary" / builder.JIT_LAYER
    meta, body = split(built[layer / "complementary-contribution.md"])
    assert meta["keywords"] == "complementary contribution"
    assert body[1] == "# Complementary contribution"
    assert any("v(T) \u2212 v(N \u2216 T)" in line for line in body)
    concepts = list((vault_copy / "knowledge" / "concepts").glob("*.md"))
    entries = [p for p in built if p.parent == layer and p.suffix == ".md"]
    assert len(entries) == len(concepts)


def test_vocabulary_keywords_qualify_everyday_words(builder: ModuleType) -> None:
    """Generic single words never key an entry: they become qualified forms, and
    backticked class names containing the term are added."""
    body = "A `CoalitionMask` from `MaskCodec.decode`; see `PlayerSet` and `Mask`."
    assert builder.vocabulary_keywords("Mask", [], body) == [
        "coalition mask",
        "packed mask",
        "coalitionmask",
        "maskcodec",
    ]
    assert builder.vocabulary_keywords("Run", ["run"], "") == []
    assert builder.vocabulary_keywords("Player span", [], body) == ["player span"]
    assert "playerset" in builder.vocabulary_keywords("Player set", [], body)
    assert builder.vocabulary_keywords("Run archive", [], "") == ["run archive"]


def test_generated_vocabulary_has_no_everyday_keys(
    builder: ModuleType, vault_copy: Path
) -> None:
    """The vocabulary index never keys on the everyday words of the audit."""
    built = builder.build_jit(vault_copy)
    index = built[
        vault_copy / builder.JIT / "vocabulary" / builder.JIT_LAYER / "00-index.tsv"
    ]
    keys = {row.split("\t")[0] for row in index.splitlines()}
    assert not keys & set(builder.JIT_GENERIC_KEYWORDS)
    assert {"player span", "run manifest", "conversation snapshot"} <= keys


def test_tools_entry_for_git_push(builder: ModuleType, vault_copy: Path) -> None:
    """One tools entry reminds of the delivery protocol on git push."""
    built = builder.build_jit(vault_copy)
    path = vault_copy / builder.JIT / "tools" / builder.JIT_LAYER / "git-push.md"
    meta, body = split(built[path])
    assert meta["tool"] == "Bash"
    assert meta["match"] == "~@invocation git push"
    assert meta["mode"] == "remind"
    assert any("gh run watch" in line for line in body)
    assert "required `title` check" in " ".join(body)


def test_body_capped_at_max_lines(builder: ModuleType, vault_copy: Path) -> None:
    """A long architecture note is truncated with a pointer to the full note."""
    note = vault_copy / "knowledge" / "architecture" / "runtime.md"
    bullets = "".join(f"- invariant {index}\n" for index in range(100))
    note.write_text(
        note.read_text().replace("## Invariants\n", "## Invariants\n\n" + bullets)
    )
    built = builder.build_jit(vault_copy)
    path = vault_copy / builder.JIT / "paths" / builder.JIT_LAYER / "runtime.md"
    _, body = split(built[path])
    assert len(body) == builder.JIT_MAX_LINES
    assert body[-1] == "(truncated; read `knowledge/architecture/runtime.md`)"


def test_jit_flag_check_and_orphans(builder: ModuleType, vault_copy: Path) -> None:
    """--jit writes only the entries; --check covers them, including orphans."""
    assert builder.main(["--jit"], root=vault_copy) == 0
    assert not (vault_copy / builder.GENERATED).exists()
    assert builder.main(["--check"], root=vault_copy) == 1
    assert builder.main([], root=vault_copy) == 0
    assert builder.main(["--check"], root=vault_copy) == 0
    entry = vault_copy / builder.JIT / "paths" / builder.JIT_LAYER / "viz.md"
    entry.write_text(entry.read_text() + "edited\n")
    assert builder.main(["--check"], root=vault_copy) == 1
    assert builder.main(["--jit"], root=vault_copy) == 0
    assert builder.main(["--check"], root=vault_copy) == 0
    orphan = entry.with_name("gone.md")
    orphan.write_text("---\ntitle: x\n---\n")
    manual = vault_copy / builder.JIT / "paths" / "00-manual" / "mine.md"
    manual.parent.mkdir(parents=True)
    manual.write_text("---\ntitle: mine\nmatch: x\n---\nmine\n")
    assert builder.main(["--check"], root=vault_copy) == 1
    assert builder.main(["--jit"], root=vault_copy) == 0
    assert not orphan.exists()
    assert manual.is_file()
    assert builder.main(["--check"], root=vault_copy) == 0
    concept = vault_copy / "knowledge" / "concepts" / "Manifest.md"
    concept.write_text(concept.read_text().replace("aliases:", "aliases_old:"))
    concept.write_text(concept.read_text() + "\nMore text.\n")
    assert builder.main(["--check"], root=vault_copy) == 1


def test_committed_entries_are_current(builder: ModuleType) -> None:
    """The committed .claude/jit-context entries match the vault."""
    built = builder.build_jit(REPO)
    stale = [
        path.relative_to(REPO).as_posix()
        for path, content in built.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != content
    ]
    stale += [p.relative_to(REPO).as_posix() for p in builder.jit_orphans(REPO, built)]
    assert stale == []


def test_layer_indexes_follow_plugin_format(
    builder: ModuleType, vault_copy: Path
) -> None:
    """Each generated layer has a 00-index.tsv in the plugin's row format."""
    built = builder.build_jit(vault_copy)
    base = vault_copy / builder.JIT
    for dimension, columns in (("paths", 2), ("vocabulary", 3), ("tools", 7)):
        index = built[base / dimension / builder.JIT_LAYER / "00-index.tsv"]
        rows = index.splitlines()
        assert rows
        assert all(len(row.split("\t")) == columns for row in rows), dimension
        entries = {p.name for p in built if p.parent.parent.name == dimension}
        position = 2 if dimension == "tools" else 1
        assert {row.split("\t")[position] for row in rows} <= entries
    paths = built[base / "paths" / builder.JIT_LAYER / "00-index.tsv"]
    assert "src/mllm_shap/runtime/\truntime.md" in paths.splitlines()
    vocabulary = built[base / "vocabulary" / builder.JIT_LAYER / "00-index.tsv"]
    assert "prefix hash\tprefix-hash.md\t" in vocabulary.splitlines()


@pytest.mark.parametrize(
    ("command", "fires"),
    [
        ("git push", True),
        ("git -c x=y push origin feat", True),
        ("cd x && git push", True),
        ("env a=b git push", True),
        ("git stash push", False),
        ("git pushall", False),
        ('git commit -m "fix git push"', False),
    ],
)
def test_git_push_match_expands_like_the_plugin(
    builder: ModuleType, command: str, fires: bool
) -> None:
    """The expanded ~@invocation pattern matches as the plugin documents (awk ERE)."""
    pattern = builder.expand_invocation("~@invocation git push")
    assert pattern.startswith("~")
    result = subprocess.run(
        [
            "awk",
            "-v",
            f"re={pattern[1:]}",
            "-v",
            f"s={command}",
            "BEGIN { exit !(s ~ re) }",
        ],
        check=False,
    )
    assert (result.returncode == 0) == fires
    assert builder.expand_invocation("src/x/") == "src/x/"


def test_index_is_committed_not_ignored() -> None:
    """Only the plugin's .discovery/ is git-ignored, not the generated index."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".claude/jit-context/paths/10-auto/00-index.tsv"],
        cwd=REPO,
        check=False,
    )
    assert ignored.returncode == 1
