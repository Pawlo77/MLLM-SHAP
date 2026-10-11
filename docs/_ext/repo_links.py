"""Sphinx extension: point repository-relative links of included files at GitHub.

``docs/contributing.md`` includes ``CONTRIBUTING.md`` from the repository root,
whose links (``[AGENTS.md](AGENTS.md)``) are relative to the root
and have no page in the documentation. On the ``include-read`` event this
extension rewrites such Markdown links to their GitHub URL; in-page anchors and
absolute URLs are left alone.
"""

import re
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

REPOSITORY_BLOB = "https://github.com/Pawlo77/MLLM-SHAP/blob/main/"
ROOT = Path(__file__).resolve().parents[2]
_LINK = re.compile(r"\]\((?P<target>[^)\s#][^)\s]*)\)")


def rewrite_links(text: str) -> str:
    """Return ``text`` with repository-relative Markdown links made absolute."""

    def replace(match: re.Match[str]) -> str:
        target = match.group("target")
        if "://" in target or target.startswith("mailto:"):
            return match.group(0)
        return f"]({REPOSITORY_BLOB}{target})"

    return _LINK.sub(replace, text)


def _on_include_read(
    app: Sphinx,
    relative_path: Path,
    parent_docname: str,
    content: list[str],
) -> None:
    _ = parent_docname
    path = (Path(app.srcdir) / relative_path).resolve()
    if path.suffix == ".md" and path.parent == ROOT:
        content[0] = rewrite_links(content[0])


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the include-read handler."""
    app.connect("include-read", _on_include_read)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
