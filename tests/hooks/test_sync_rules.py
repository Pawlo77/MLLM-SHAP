"""Claude Code rules generated from the Cursor rules.

@see [[scripts/sync_rules.py]]
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "sync_rules.py"


@pytest.fixture(scope="module")
def sync_rules():
    """Load scripts/sync_rules.py without installing it as a package."""
    spec = importlib.util.spec_from_file_location("sync_rules", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_frontmatter_keeps_globs_raw(sync_rules) -> None:
    """Globs such as **/*.py survive (they are not valid plain YAML)."""
    text = (
        "---\ndescription: x\nglobs: **/*.py,tests/**\nalwaysApply: false\n---\n\n# T\n"
    )
    fields, body = sync_rules.split_frontmatter(text)
    assert sync_rules.globs_of(fields) == ["**/*.py", "tests/**"]
    assert body == "# T\n"


def test_render_scoped_and_always(
    sync_rules, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoped rules get paths; always-apply rules merge into always.md."""
    scoped = tmp_path / "types.mdc"
    scoped.write_text(
        "---\ndescription: t\nglobs: src/**/*.py\nalwaysApply: false\n---\n\n# Types\n",
        encoding="utf-8",
    )
    always = tmp_path / "style.mdc"
    always.write_text(
        "---\ndescription: c\nalwaysApply: true\n---\n\n# Always\n", encoding="utf-8"
    )
    monkeypatch.setattr(sync_rules, "ROOT", tmp_path)
    out = sync_rules.render([scoped, always])
    assert out["types.md"].startswith('---\npaths:\n  - "src/**/*.py"\n---\n')
    assert out["types.md"].endswith("# Types\n")
    assert "# Always" in out["always.md"]
    assert "types" not in out["always.md"]


def test_plugin_rule_has_no_claude_copy(
    sync_rules, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule the Claude Code plugin injects itself is not generated."""
    rule = tmp_path / "ponytail.mdc"
    rule.write_text("---\nalwaysApply: true\n---\n\n# P\n", encoding="utf-8")
    monkeypatch.setattr(sync_rules, "ROOT", tmp_path)
    assert sync_rules.render([rule]) == {}


def test_generated_rules_are_current() -> None:
    """.claude/rules/ matches .cursor/rules/ (run scripts/sync_rules.py)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
