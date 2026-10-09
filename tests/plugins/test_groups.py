"""Plugin group identifiers match the packaging entry-point tables.

@see [[src/nlp_shap/plugins/groups.py]]
"""

import tomllib
from pathlib import Path

from nlp_shap.plugins import PluginGroup

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_every_group_has_an_entry_point_table_and_vice_versa() -> None:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    tables = set(data["project"]["entry-points"])

    assert {group.value for group in PluginGroup} == tables


def test_groups_are_strings_usable_as_entry_point_names() -> None:
    assert PluginGroup("nlp_shap.backends") is PluginGroup.BACKENDS
    assert f"{PluginGroup.ESTIMATORS}" == "nlp_shap.estimators"
