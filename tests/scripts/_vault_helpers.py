"""Helpers shared by the script tests: temporary vault files and notes."""

from pathlib import Path

GLOSSARY = """\
CC:
  expansion: complementary contribution
MC:
  expansion: Monte Carlo
"""


def _vault_tables() -> str:
    """Return the ``[tool.agent-kit.vault*]`` tables of the repository's pyproject."""
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    blocks = text.split("\n[")
    keep = [
        b
        for b in blocks
        if b.startswith(("tool.agent-kit.vault", "[tool.agent-kit.vault"))
    ]
    return "\n".join(f"[{b.removeprefix('[')}" for b in keep) + "\n"


PYPROJECT = """\
[project]
name = "demo"
authors = [{ name = "Maintainer", email = "maintainer@example.org" }]

[project.entry-points."mllm_shap.estimators"]
exact = "mllm_shap.estimation.exact:ExactEstimator"
""" + _vault_tables()


def write(root: Path, rel: str, text: str) -> Path:
    """Write ``text`` to ``root / rel`` creating parents."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def note(
    title: str,
    kind: str,
    body: str,
    *,
    extra: str = "",
    created: bool = True,
) -> str:
    """Return a note with frontmatter."""
    created_line = "created: 2026-10-07\n" if created else ""
    parent = "" if kind in ("map", "generated") else 'parent: "[[Map]]"\n'
    return (
        f'---\ntitle: "{title}"\ntype: {kind}\ndomain: demo\n{parent}tags: [demo]\n'
        f"{created_line}{extra}---\n\n# {title}\n\n{body}"
    )


MODULE_BODY = (
    "## Responsibility\n"
    "\n"
    "- Exact enumeration.\n"
    "\n"
    "## Public surface\n"
    "\n"
    "- `ExactEstimator`.\n"
    "\n"
    "## Data flow\n"
    "\n"
    "- In and out.\n"
    "\n"
    "## Invariants\n"
    "\n"
    "- Efficiency.\n"
    "\n"
    "## Tests\n"
    "\n"
    "- [[tests/demo/test_exact.py]]\n"
)
