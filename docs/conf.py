"""Sphinx configuration for nlp-shap."""

import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DOCS / "_ext"))


def _footer_icon(filename: str) -> str:
    return (DOCS / "_static" / "icons" / filename).read_text(encoding="utf-8")


with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    _project = tomllib.load(pyproject_file)["project"]

project = _project["name"]
author = _project["authors"][0]["name"]
copyright = f"{datetime.now(tz=UTC).year}, Paweł Pozorski"
version = _project["version"]
release = version

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "guide_outputs",
    "public_exports",
    "config_table",
    "repo_links",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "AGENTS.md",
    "CLAUDE.md",
    "**/AGENTS.md",
    "**/CLAUDE.md",
]
html_static_path = ["_static"]
html_css_files = ["custom.css"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 2
nb_execution_mode = "off"
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Strict cross-references (``make docs`` runs with ``-n -W``). Each entry below is a
# name Sphinx cannot link and why; never add a blanket pattern.
nitpicky = True
nitpick_ignore: list[tuple[str, str]] = [
    # Type variable of ``PluginContext.option[T]``; type variables have no page.
    ("py:class", "T"),
    # matplotlib types imported under TYPE_CHECKING in nlp_shap.viz; the string
    # annotations carry no module, so intersphinx cannot resolve them.
    ("py:class", "Figure"),
    ("py:class", "Axes"),
    # numpy.typing.NDArray is rendered through its private alias module, which
    # the numpy inventory does not list.
    ("py:class", "numpy._typing._array_like.NDArray"),
    # numpy documents float32 and int64 as aliases of single and int_, without a
    # class entry of their own in its inventory.
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.int64"),
    # pydantic ``Field(gt=..., ge=..., le=...)`` constraints appear in rendered
    # signatures as annotated_types markers, which have no inventory.
    ("py:class", "annotated_types.Ge"),
    ("py:class", "annotated_types.Gt"),
    ("py:class", "annotated_types.Le"),
    # Private base of the built-in estimators; its methods are documented on each
    # estimator through ``:inherited-members:`` in ``api/estimation.rst``.
    ("py:class", "nlp_shap.estimation._shared._SnapshotEstimator"),
]
nitpick_ignore_regex: list[tuple[str, str]] = [
    # Numeric bounds of the same pydantic constraints (``0.0``, ``1.0``).
    ("py:class", r"\d+(\.\d+)?"),
    # Sphinx splits dataclass signatures whose generic field types contain a
    # comma (``OrderedDict[str, str]``, ``Mapping[str, object]``,
    # ``tuple[str, ...]``) and then looks up the fragment before the comma.
    ("py:class", r"(collections\.abc\.Mapping|collections\.OrderedDict|tuple)\[str"),
]

linkcheck_ignore = [
    # doi.org rate-limits CI runners; DOIs are checked by hand when references
    # change.
    r"https://doi\.org/.*",
    # Files in this repository are linked on ``main``; a branch that adds a file
    # would fail until it merges, and github.com rate-limits (HTTP 429) the
    # dozens of blob links. tests/docs checks that the linked files exist.
    r"https://github\.com/Pawlo77/nlp-shap/(blob|tree)/main/.*",
    # Linked from CONTRIBUTING.md; GitHub Discussions is not enabled on the
    # repository yet (404 until it is).
    r"https://github\.com/Pawlo77/nlp-shap/discussions",
]
linkcheck_timeout = 30
linkcheck_retries = 2

html_theme = "furo"
html_title = f"{project} {release}"
html_favicon = "_static/favicon.svg"
html_logo = "_static/logo.svg"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "source_repository": "https://github.com/Pawlo77/nlp-shap/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Pawlo77/nlp-shap",
            "html": _footer_icon("github.svg"),
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/nlp-shap/",
            "html": _footer_icon("pypi.svg"),
            "class": "",
        },
        {
            "name": "Issues",
            "url": "https://github.com/Pawlo77/nlp-shap/issues",
            "html": _footer_icon("issues.svg"),
            "class": "",
        },
    ],
}
