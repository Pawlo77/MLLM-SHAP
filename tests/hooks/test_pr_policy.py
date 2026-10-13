"""Tests for the delivery autopilot's branch classification and session marker.

@see [[scripts/pr_policy.py]]
"""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

CONFIG = """
[tool.agent-kit.pr-policy]
routine-types = "chore ci build docs test style"
review-paths = ["src/demo/*", "experiments/*/configs/*", "scripts/cluster/*"]
protected = [".github/workflows/publish.yaml"]
pin = "demo-lib"
release = true
"""
PYPROJECT_BASE = (
    '[project]\nname = "demo"\nversion = "1.0.0"\ndependencies = ["numpy>=2"]\n\n'
    '[dependency-groups]\ndev = ["pytest>=8"]\n'
)
PYPROJECT = PYPROJECT_BASE + CONFIG
LOCK = (
    'version = 1\n\n[[package]]\nname = "demo-lib"\nversion = "0.2.1"\n'
    'source = {{ git = "https://example.org/demo-lib.git#{sha}" }}\n\n'
    '[[package]]\nname = "numpy"\nversion = "{numpy}"\n'
)


@pytest.fixture(scope="module")
def policy() -> ModuleType:
    """Load scripts/pr_policy.py."""
    spec = importlib.util.spec_from_file_location(
        "pr_policy", ROOT / "scripts" / "pr_policy.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, subject: str, files: dict[str, str]) -> None:
    for rel, text in files.items():
        _write(repo, rel, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", subject)


def _make_repo(tmp_path: Path, pyproject: str) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _commit(
        root,
        "chore: base",
        {
            ".gitignore": ".remember/tmp/\n",
            "README.md": "# Demo\n",
            "pyproject.toml": pyproject,
            "uv.lock": LOCK.format(sha="a" * 40, numpy="2.0.0"),
            "src/demo/__init__.py": '"""Demo."""\n',
            "scripts/hooks/merge_gate.sh": "# Guard.\n",
        },
    )
    _git(root, "switch", "-qc", "feat/x")
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with a ``main`` base commit and a checked-out feature branch."""
    return _make_repo(tmp_path, PYPROJECT)


def test_docs_only_branch_is_routine(policy: ModuleType, repo: Path) -> None:
    """Allowed commit types, no src/ change and a small diff merge by themselves."""
    _commit(repo, "docs: explain the demo", {"README.md": "# Demo\n\nMore.\n"})
    _commit(repo, "chore: add a note", {"docs/note.md": "note\n"})
    result = policy.classify("main", repo)
    assert result.kind == "routine"
    assert result.files == ("README.md", "docs/note.md")


@pytest.mark.parametrize(
    ("subject", "files", "reason"),
    [
        ("fix: correct the readme", {"README.md": "# Fixed\n"}, "fix"),
        ("feat!: break it", {"README.md": "# Broken\n"}, "feat!"),
        ("chore: touch src", {"src/demo/x.py": '"""X."""\n'}, "src/demo/"),
        (
            "chore: retune a suite",
            {"experiments/tv/configs/suites/a.yaml": "seed: 1\n"},
            "experiments/*/configs/",
        ),
        (
            "chore: tweak a job",
            {"scripts/cluster/helios/run.sbatch": "#!/bin/bash\n"},
            "scripts/cluster/",
        ),
        (
            "build: move the library pin",
            {"uv.lock": LOCK.format(sha="b" * 40, numpy="2.0.0")},
            "demo-lib pin",
        ),
        (
            "ci: touch the publish workflow",
            {".github/workflows/publish.yaml": "name: x\n"},
            ".github/workflows/publish.yaml",
        ),
        (
            "ci: touch the template sync",
            {".github/workflows/template-sync.yaml": "name: x\n"},
            "template-sync.yaml",
        ),
        (
            "chore: tweak the hook config",
            {".pre-commit-config.yaml": "repos: []\n"},
            ".pre-commit-config.yaml",
        ),
        (
            "chore: tweak the shared make targets",
            {"kit.mk": "# x\n"},
            "kit.mk",
        ),
        (
            "chore: tweak the config generator",
            {"scripts/sync_configs.py": "# x\n"},
            "scripts/sync_configs.py",
        ),
        (
            "chore: tweak the shell guard",
            {"scripts/hooks/shell_guard.py": "# x\n"},
            "scripts/hooks/shell_guard.py",
        ),
        (
            "chore: tweak the guard",
            {"scripts/hooks/merge_gate.sh": "# Guard 2.\n"},
            "scripts/hooks/merge_gate.sh",
        ),
        ("docs: many lines", {"docs/big.md": "line\n" * 401}, "401 changed lines"),
        (
            "docs: many files",
            {f"docs/f{i}.md": "x\n" for i in range(26)},
            "26 files",
        ),
        (
            "build: add a package",
            {"pyproject.toml": PYPROJECT.replace('"numpy>=2"', '"numpy>=2", "scipy"')},
            "[project] dependencies",
        ),
        (
            "docs: bump numpy",
            {"pyproject.toml": PYPROJECT.replace("numpy>=2", "numpy>=3")},
            "[project] dependencies",
        ),
        (
            "chore: release v1.0.2",
            {"pyproject.toml": PYPROJECT.replace("1.0.0", "1.0.1")},
            "[project] version",
        ),
        (
            "build: bump version",
            {"pyproject.toml": PYPROJECT.replace("1.0.0", "1.0.1")},
            "[project]",
        ),
    ],
)
def test_review_cases(
    policy: ModuleType, repo: Path, subject: str, files: dict[str, str], reason: str
) -> None:
    """Each rule on its own turns the branch into a review PR, with the reason."""
    _commit(repo, subject, files)
    result = policy.classify("main", repo)
    assert result.kind == "review"
    assert any(reason in r for r in result.reasons), result.reasons


def test_lock_and_dev_group_changes_are_routine(policy: ModuleType, repo: Path) -> None:
    """A uv.lock refresh and a [dependency-groups] change are routine builds."""
    _commit(
        repo,
        "build: refresh the lock",
        {"uv.lock": LOCK.format(sha="a" * 40, numpy="2.1.0")},
    )
    _commit(
        repo,
        "build: bump pytest",
        {"pyproject.toml": PYPROJECT.replace("pytest>=8", "pytest>=9")},
    )
    assert policy.classify("main", repo).kind == "routine"


@pytest.mark.parametrize(
    ("subject", "pyproject"),
    [
        ("build: bump numpy", PYPROJECT.replace("numpy>=2", "numpy>=2.1,<3")),
        (
            "chore: widen an extra",
            PYPROJECT.replace(
                "\n\n[dependency-groups]",
                '\n\n[project.optional-dependencies]\napi = ["httpx>=0.29"]\n\n'
                "[dependency-groups]",
            ),
        ),
        ("chore: release v1.0.1", PYPROJECT.replace("1.0.0", "1.0.1")),
    ],
)
def test_specifier_and_release_changes_are_routine(
    policy: ModuleType, repo: Path, subject: str, pyproject: str
) -> None:
    """Same package set with new specifiers, or the version of a release commit."""
    if "optional" in pyproject:
        base = PYPROJECT.replace(
            "\n\n[dependency-groups]",
            '\n\n[project.optional-dependencies]\napi = ["httpx>=0.28"]\n\n'
            "[dependency-groups]",
        )
        _git(repo, "switch", "-q", "main")
        _commit(repo, "chore: add an extra", {"pyproject.toml": base})
        _git(repo, "switch", "-qC", "feat/x")
    _commit(repo, subject, {"pyproject.toml": pyproject})
    result = policy.classify("main", repo)
    assert result.kind == "routine", result.reasons


def test_lock_lines_do_not_count(policy: ModuleType, repo: Path) -> None:
    """uv.lock is generated: a large lock refresh stays within the line cap."""
    _commit(
        repo,
        "build: upgrade the lock",
        {"uv.lock": LOCK.format(sha="a" * 40, numpy="2.0.0") + "line\n" * 900},
    )
    assert policy.classify("main", repo).kind == "routine"


def _release(repo: Path, version: str) -> None:
    _write(repo, "pyproject.toml", PYPROJECT.replace("1.0.0", version))
    _git(repo, "add", "-A")
    _git(repo, "commit", "--allow-empty", "-qm", f"chore: release v{version}")
    _git(repo, "tag", "-a", f"v{version}", "-m", version)


@pytest.mark.parametrize(
    ("messages", "tagged", "expected"),
    [
        (["docs: explain", "chore: tidy"], "1.0.0", ("none", "1.0.0")),
        (["docs: explain", "perf: faster"], "1.0.0", ("patch", "1.0.1")),
        (["fix: correct", "feat: add"], "1.0.0", ("minor", "1.1.0")),
        (["feat!: drop the old API"], "1.0.0", ("major", "2.0.0")),
        (["fix: correct\n\nBREAKING CHANGE: new format"], "1.2.3", ("major", "2.0.0")),
        (["feat!: drop the old API"], "0.4.2", ("minor", "0.5.0")),
    ],
)
def test_suggest_bump(
    policy: ModuleType,
    repo: Path,
    messages: list[str],
    tagged: str,
    expected: tuple[str, str],
) -> None:
    """The highest commit type since the last v* tag decides the bump."""
    _release(repo, tagged)
    for message in messages:
        subject, _, body = message.partition("\n\n")
        _git(
            repo,
            "commit",
            "--allow-empty",
            "-qm",
            subject,
            *(["-m", body] * bool(body)),
        )
    assert policy.suggest_bump(repo) == expected


def test_suggest_bump_keeps_a_pending_version(
    policy: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A version already bumped past the computed one is kept; the CLI prints both."""
    _release(repo, "1.0.0")
    _commit(
        repo, "fix: correct", {"pyproject.toml": PYPROJECT.replace("1.0.0", "1.0.3")}
    )
    assert policy.main(["--root", str(repo), "--bump"]) == 0
    assert capsys.readouterr().out.strip() == "patch 1.0.3"


def test_branch_without_commits_is_review(policy: ModuleType, repo: Path) -> None:
    """Nothing to ship is never merged by the autopilot."""
    result = policy.classify("main", repo)
    assert result.kind == "review"
    assert any("no commits" in r for r in result.reasons)


def test_marker_write_update_and_validity(policy: ModuleType, repo: Path) -> None:
    """The marker records the classification and stays valid until the diff changes."""
    _commit(repo, "docs: explain", {"README.md": "# Demo\n\nMore.\n"})
    head = _git(repo, "rev-parse", "HEAD")
    marker = policy.write_marker("main", "s-1", repo)
    path = repo / ".remember" / "tmp" / "pr-autopilot.s-1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == marker
    assert (data["kind"], data["head"], data["branch"], data["pr"]) == (
        "routine",
        head,
        "feat/x",
        None,
    )
    assert data["base"] == "main"
    assert data["created"]

    policy.set_pr("s-1", 7, repo)
    found = policy.load_valid_marker("s-1", repo)
    assert found is not None
    assert found[0]["pr"] == 7
    assert found[1] is True  # marker head is the current HEAD

    # A later commit to the same file keeps the marker valid but not exact.
    _commit(repo, "docs: more", {"README.md": "# Demo\n\nEven more.\n"})
    found = policy.load_valid_marker("s-1", repo)
    assert found is not None
    assert found[1] is False

    # Rewriting keeps the recorded PR number.
    assert policy.write_marker("main", "s-1", repo)["pr"] == 7

    # A commit that touches a new path invalidates the marker.
    _commit(repo, "docs: new page", {"docs/page.md": "page\n"})
    assert policy.load_valid_marker("s-1", repo) is None


def test_marker_invalid_on_other_branch_or_rewritten_head(
    policy: ModuleType, repo: Path
) -> None:
    """A branch switch or a head that is no longer an ancestor drops the marker."""
    _commit(repo, "docs: explain", {"README.md": "# Demo\n\nMore.\n"})
    policy.write_marker("main", "s-1", repo)
    _git(repo, "switch", "-qc", "feat/y")
    assert policy.load_valid_marker("s-1", repo) is None
    _git(repo, "switch", "-q", "feat/x")
    assert policy.load_valid_marker("s-1", repo) is not None
    _git(repo, "reset", "-q", "--hard", "main")
    _commit(repo, "docs: explain again", {"README.md": "# Demo\n\nOther.\n"})
    assert policy.load_valid_marker("s-1", repo) is None


def test_marker_refused_on_main_and_unknown_session(
    policy: ModuleType, repo: Path
) -> None:
    """No marker is written for main or another session's key."""
    _git(repo, "switch", "-q", "main")
    with pytest.raises(policy.PolicyError, match="main"):
        policy.write_marker("main", "s-1", repo)
    assert policy.load_valid_marker("other", repo) is None


@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    [
        ("abc/../x", {}, "abcx"),
        (None, {"CLAUDE_CODE_SESSION_ID": "5bfc-69"}, "5bfc-69"),
        ("given", {"CLAUDE_CODE_SESSION_ID": "env"}, "given"),
    ],
)
def test_session_key(
    policy: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    env: dict[str, str],
    expected: str,
) -> None:
    """The key matches gate_lib.sh: [A-Za-z0-9_-] only, at most 64 characters."""
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert policy.session_key(explicit) == expected


def test_session_key_missing(
    policy: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --session or a Claude Code session id the marker cannot be keyed."""
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    with pytest.raises(policy.PolicyError, match="--session"):
        policy.session_key(None)


def test_cli_json_and_marker(
    policy: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints the kind and reasons, and writes and updates the marker."""
    _commit(repo, "fix: correct", {"README.md": "# Fixed\n"})
    assert policy.main(["--root", str(repo), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "review"
    assert out["reasons"]
    assert policy.main(["--root", str(repo), "--write-marker", "--session", "k1"]) == 0
    assert "review" in capsys.readouterr().out
    assert policy.main(["--root", str(repo), "--pr", "12", "--session", "k1"]) == 0
    marker = json.loads(
        (repo / ".remember" / "tmp" / "pr-autopilot.k1.json").read_text("utf-8")
    )
    assert (marker["kind"], marker["pr"]) == ("review", 12)


def test_merge_ok(
    policy: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--merge-ok prints ok only for a routine, exact marker with a PR number."""
    args = ["--root", str(repo), "--merge-ok", "--session", "k1"]
    _commit(repo, "docs: note", {"docs/note.md": "note\n"})
    assert policy.main(args) == 1  # no marker yet
    assert "no current marker" in capsys.readouterr().out
    policy.write_marker("main", "k1", repo)
    assert policy.main(args) == 1  # routine and exact, but no PR number
    assert "pr=None" in capsys.readouterr().out
    policy.write_marker("main", "k1", repo, pr=7)
    assert policy.main(args) == 0
    assert capsys.readouterr().out.strip() == "ok"
    _commit(repo, "docs: more", {"docs/note.md": "more\n"})  # HEAD moved: not exact
    assert policy.main(args) == 1
    assert "exact=False" in capsys.readouterr().out
    _commit(repo, "fix: change", {"README.md": "# Fixed\n"})
    policy.write_marker("main", "k2", repo, pr=7)  # a review branch
    assert policy.main(["--root", str(repo), "--merge-ok", "--session", "k2"]) == 1
    assert "marker is review" in capsys.readouterr().out


def test_style_commits_follow_the_configured_routine_types(
    policy: ModuleType, repo: Path, tmp_path: Path
) -> None:
    """``style`` is routine here; with the default types it needs the maintainer."""
    _commit(repo, "style: format the readme", {"README.md": "# Demo\n\nFormatted.\n"})
    assert policy.classify("main", repo).kind == "routine"
    strict = _make_repo(tmp_path / "strict", PYPROJECT_BASE)
    _commit(strict, "style: format the readme", {"README.md": "# Demo\n\nFormatted.\n"})
    result = policy.classify("main", strict)
    assert result.kind == "review"
    assert any("'style' is not routine" in r for r in result.reasons)


def test_version_change_needs_review_unless_releases_are_enabled(
    policy: ModuleType, tmp_path: Path
) -> None:
    """A release commit may change the version only when ``release = true``."""
    plain = PYPROJECT_BASE + CONFIG.replace("release = true", "release = false")
    repo = _make_repo(tmp_path / "plain", plain)
    _commit(
        repo,
        "chore: release v1.0.1",
        {"pyproject.toml": plain.replace("1.0.0", "1.0.1")},
    )
    result = policy.classify("main", repo)
    assert result.kind == "review"
    assert any("[project] version" in r for r in result.reasons)


def test_unconfigured_repository_has_no_pin_and_no_review_paths(
    policy: ModuleType, tmp_path: Path
) -> None:
    """Without settings, only the commit types, protected files and caps apply."""
    repo = _make_repo(tmp_path / "bare", PYPROJECT_BASE)
    _commit(
        repo,
        "chore: touch src and the lock",
        {
            "src/demo/x.py": '"""X."""\n',
            "uv.lock": LOCK.format(sha="b" * 40, numpy="2.0.0"),
        },
    )
    assert policy.classify("main", repo).kind == "routine"
    assert policy.settings(repo).pin == ""
