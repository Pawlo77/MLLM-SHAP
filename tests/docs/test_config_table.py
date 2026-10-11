"""Tests for the ``config-table`` Sphinx directive helpers.

@see [[docs/_ext/config_table.py]]
"""

from typing import Literal

import pytest
from config_table import config_rows, format_type, import_model
from pydantic import BaseModel

from mllm_shap.pipeline.config import ExplainConfig


class _Inner(BaseModel):
    flag: bool = True
    """Inner flag."""


class _Outer(BaseModel):
    mode: Literal["a", "b"] = "a"
    """Mode switch."""

    inner: _Inner = _Inner()
    """Nested section."""


def test_nested_models_expand_to_dotted_keys() -> None:
    """Nested models become a section row followed by their fields."""
    rows = config_rows(_Outer, "root")
    assert [row.key for row in rows] == ["root.mode", "root.inner", "root.inner.flag"]
    assert rows[0].type == '"a" | "b"'
    assert rows[0].default == '"a"'
    assert rows[0].meaning == "Mode switch."
    assert rows[1].type == "section"
    assert rows[2].default == "true"


def test_every_config_key_is_documented() -> None:
    """Every key of the explain configuration has an attribute docstring."""
    rows = config_rows(ExplainConfig)
    undocumented = [row.key for row in rows if not row.meaning]
    assert not undocumented, f"config keys without a docstring: {undocumented}"


def test_required_fields_are_marked() -> None:
    """``backend.kind`` has no default and is reported as required."""
    rows = {row.key: row for row in config_rows(ExplainConfig)}
    assert rows["backend.kind"].default == "required"
    assert rows["explanation.archive.on_exists"].type == '"error" | "overwrite"'


def test_format_type_handles_optional() -> None:
    """Optional annotations render as a union with ``null``."""
    assert format_type(int | None) == "int | null"


def test_import_model_rejects_non_models() -> None:
    """Only pydantic models can be documented."""
    assert import_model("mllm_shap.pipeline.config.ExplainConfig") is ExplainConfig
    with pytest.raises(TypeError):
        import_model("mllm_shap.pipeline.config.explain_config_from_yaml")
