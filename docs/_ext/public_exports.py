"""Sphinx extension: the public-exports table and re-export cross-references.

``.. public-exports::`` imports :mod:`mllm_shap` (or the module given as argument),
walks its ``__all__`` and emits a table with one row per exported name: a
cross-reference to the object's canonical location and the first line of its
docstring. Dunder names such as ``__version__`` are skipped.

The extension also resolves references written against a re-export, for example
``:class:`~mllm_shap.ExplainRunner``` or ``:class:`mllm_shap.runtime.RunArchive```,
to the module that defines the object (``mllm_shap.pipeline.runner.ExplainRunner``),
so guides can use the short public paths under ``nitpicky``.
"""

import importlib
import inspect
import re
from dataclasses import dataclass
from types import ModuleType

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata

PACKAGE = "mllm_shap"


@dataclass(frozen=True)
class ExportRow:
    """One row of the public-exports table."""

    name: str
    target: str
    role: str
    summary: str


def _role_for(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "func"
    return "obj"


def first_doc_line(obj: object) -> str:
    """Return the first non-empty line of ``obj``'s own docstring, or ``""``."""
    if inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__")
    elif callable(obj):
        doc = getattr(obj, "__doc__", None)
    else:
        doc = None
    if not isinstance(doc, str):
        return ""
    for line in inspect.cleandoc(doc).splitlines():
        if line.strip():
            return line.strip()
    return ""


def canonical_target(obj: object, fallback: str) -> str:
    """Return ``module.qualname`` where ``obj`` is defined, else ``fallback``.

    Objects defined in a private module (``mllm_shap._logging``) are documented
    under their public path, so ``fallback`` is returned for them.
    """
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not (isinstance(module, str) and isinstance(qualname, str)):
        return fallback
    if any(part.startswith("_") for part in module.split(".")):
        return fallback
    return f"{module}.{qualname}"


_ROLE = re.compile(r":(?:py:)?(?P<role>\w+):`(?P<target>[^`<]+)`")


def qualify_roles(summary: str, obj: object) -> str:
    """Rewrite bare cross-reference targets in ``summary`` against ``obj``'s module.

    A docstring line such as ``Parse YAML into :class:`ExplainConfig`.`` refers to
    a name in the object's own module; on the API page that name is unknown, so
    it is replaced by the canonical dotted path (or by literal text when the
    name cannot be found).
    """
    module = inspect.getmodule(obj)

    def replace(match: re.Match[str]) -> str:
        role, target = match.group("role"), match.group("target").strip()
        bare = target.lstrip("~.")
        if "." in bare:
            return match.group(0)
        found = getattr(module, bare, None) if module is not None else None
        if found is None:
            return f"``{bare}``"
        return f":py:{role}:`~{canonical_target(found, f'{PACKAGE}.{bare}')}`"

    return _ROLE.sub(replace, summary)


def export_rows(module_name: str = PACKAGE) -> list[ExportRow]:
    """Return one row per non-dunder name in ``module_name.__all__``."""
    module = importlib.import_module(module_name)
    rows: list[ExportRow] = []
    for name in getattr(module, "__all__", ()):
        if name.startswith("__"):
            continue
        obj = getattr(module, name)
        target = canonical_target(obj, f"{module_name}.{name}")
        summary = qualify_roles(first_doc_line(obj), obj)
        rows.append(ExportRow(name, target, _role_for(obj), summary))
    return rows


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\")


class PublicExportsDirective(SphinxDirective):
    """Render ``__all__`` of a module as a two-column list-table."""

    optional_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Build the table from the imported module."""
        module_name = self.arguments[0] if self.arguments else PACKAGE
        try:
            rows = export_rows(module_name)
        except ImportError as error:
            msg = f"cannot import {module_name!r}: {error}"
            raise self.error(msg) from error
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 30 70",
            "",
            "   * - Symbol",
            "     - Summary",
        ]
        for row in rows:
            lines.append(f"   * - :py:{row.role}:`{row.name} <{row.target}>`")
            lines.append(f"     - {_escape(row.summary) or '(no docstring)'}")
        container = nodes.container()
        container["classes"] = ["public-exports"]
        self.state.nested_parse(StringList(lines, source=""), 0, container)
        return [container]


def _module_prefix(parts: list[str]) -> tuple[ModuleType, int] | None:
    for index in range(len(parts), 0, -1):
        try:
            return importlib.import_module(".".join(parts[:index])), index
        except ImportError:
            continue
    return None


def canonical_reference(target: str) -> str | None:
    """Map a ``mllm_shap`` re-export path to the defining module's path.

    Returns ``None`` when ``target`` is not under :mod:`mllm_shap`, cannot be
    imported, or already names the defining module.
    """
    parts = target.split(".")
    if parts[0] != PACKAGE:
        return None
    found = _module_prefix(parts)
    if found is None:
        return None
    obj: object = found[0]
    index = found[1]
    if index == len(parts):
        return None
    try:
        obj = getattr(obj, parts[index])
    except AttributeError:
        return None
    canonical = canonical_target(obj, "")
    if not canonical.startswith(f"{PACKAGE}."):
        return None
    resolved = ".".join([canonical, *parts[index + 1 :]])
    return resolved if resolved != target else None


def _resolve_reexport(
    app: Sphinx,
    env: BuildEnvironment,
    node: pending_xref,
    contnode: nodes.TextElement,
) -> nodes.reference | None:
    if node.get("refdomain") != "py":
        return None
    canonical = canonical_reference(node.get("reftarget", ""))
    if canonical is None:
        return None
    domain = env.get_domain("py")
    return domain.resolve_xref(
        env,
        node.get("refdoc", ""),
        app.builder,
        node["reftype"],
        canonical,
        node,
        contnode,
    )


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the directive and the re-export reference resolver."""
    app.add_directive("public-exports", PublicExportsDirective)
    app.connect("missing-reference", _resolve_reexport)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
