"""Sphinx extension: configuration reference tables from the pydantic models.

``.. config-table:: nlp_shap.pipeline.config.ExplanationConfig`` imports the model
and emits one row per field (nested models are expanded into dotted keys) with the
YAML key, the type, the default and the field's attribute docstring. Options:

``:prefix:``
    Key prefix such as ``explanation``; defaults to no prefix.
"""

import ast
import enum
import importlib
import inspect
import itertools
import json
import textwrap
import types
import typing
from dataclasses import dataclass
from functools import cache

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata


@dataclass(frozen=True)
class ConfigRow:
    """One documented configuration key."""

    key: str
    type: str
    default: str
    meaning: str


@cache
def attribute_docstrings(model: type) -> dict[str, str]:
    """Return ``{field: docstring}`` for attribute docstrings in ``model``'s body."""
    docs: dict[str, str] = {}
    for cls in reversed(model.__mro__):
        if cls is object or not issubclass(cls, BaseModel) or cls is BaseModel:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError):
            continue
        body = ast.parse(source).body[0]
        if not isinstance(body, ast.ClassDef):
            continue
        statements = body.body
        for current, following in itertools.pairwise(statements):
            if not (
                isinstance(current, ast.AnnAssign)
                and isinstance(current.target, ast.Name)
                and isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                continue
            docs[current.target.id] = inspect.cleandoc(following.value.value)
    return docs


def format_type(annotation: object) -> str:
    """Return a short, YAML-oriented rendering of a field annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return " | ".join(json.dumps(arg) for arg in args)
    if origin in (typing.Union, types.UnionType):
        return " | ".join(format_type(arg) for arg in args)
    if origin is not None:
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{', '.join(format_type(arg) for arg in args)}]"
    if annotation is type(None):
        return "null"
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return " | ".join(json.dumps(member.value) for member in annotation)
    if inspect.isclass(annotation):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def format_default(field: FieldInfo) -> str:
    """Return the field default as it would appear in YAML."""
    if field.is_required():
        return "required"
    default = field.get_default(call_default_factory=True)
    if isinstance(default, enum.Enum):
        default = default.value
    if isinstance(default, BaseModel):
        return "(section)"
    return json.dumps(default)


def _is_model(annotation: object) -> bool:
    return inspect.isclass(annotation) and issubclass(annotation, BaseModel)


def config_rows(model: type[BaseModel], prefix: str = "") -> list[ConfigRow]:
    """Return rows for every field of ``model``, expanding nested models."""
    docs = attribute_docstrings(model)
    rows: list[ConfigRow] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}.{name}" if prefix else name
        meaning = " ".join((field.description or docs.get(name, "")).split())
        annotation = field.annotation
        if _is_model(annotation):
            rows.append(ConfigRow(key, "section", "", meaning))
            rows.extend(config_rows(typing.cast(type[BaseModel], annotation), key))
            continue
        rows.append(
            ConfigRow(key, format_type(annotation), format_default(field), meaning)
        )
    return rows


def import_model(path: str) -> type[BaseModel]:
    """Import ``package.module.Model`` and check it is a pydantic model."""
    module_name, _, attribute = path.rpartition(".")
    model = getattr(importlib.import_module(module_name), attribute)
    if not _is_model(model):
        msg = f"{path} is not a pydantic model"
        raise TypeError(msg)
    return typing.cast(type[BaseModel], model)


def _literal(text: str) -> str:
    return f"``{text}``" if text else ""


class ConfigTableDirective(SphinxDirective):
    """Render a pydantic config model as a key/type/default/meaning table."""

    required_arguments = 1
    has_content = False
    option_spec: typing.ClassVar[dict[str, typing.Callable[[str], object]]] = {
        "prefix": directives.unchanged,
    }

    def run(self) -> list[nodes.Node]:
        """Build the table from the imported model."""
        try:
            model = import_model(self.arguments[0])
        except (ImportError, AttributeError, TypeError) as error:
            msg = f"config-table: {error}"
            raise self.error(msg) from error
        prefix = str(self.options.get("prefix", ""))
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 28 22 14 36",
            "",
            "   * - Key",
            "     - Type",
            "     - Default",
            "     - Meaning",
        ]
        for row in config_rows(model, prefix):
            lines.extend([
                f"   * - {_literal(row.key)}",
                f"     - {_literal(row.type)}",
                f"     - {_literal(row.default)}",
                f"     - {row.meaning}",
            ])
        container = nodes.container()
        container["classes"] = ["config-table"]
        self.state.nested_parse(StringList(lines, source=""), 0, container)
        return [container]


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the config-table directive."""
    app.add_directive("config-table", ConfigTableDirective)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
