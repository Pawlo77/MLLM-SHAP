"""Tests for the ``public-exports`` Sphinx directive helpers.

@see [[docs/_ext/public_exports.py]]
"""

from public_exports import (
    canonical_reference,
    export_rows,
    first_doc_line,
    qualify_roles,
)

import nlp_shap


def test_every_export_has_a_docstring() -> None:
    """Each non-dunder name in ``nlp_shap.__all__`` has a summary line."""
    missing = [row.name for row in export_rows() if not row.summary]
    assert not missing, f"exports without a docstring: {missing}"


def test_rows_cover_all_exports() -> None:
    """The table lists every non-dunder export exactly once, in ``__all__`` order."""
    expected = [name for name in nlp_shap.__all__ if not name.startswith("__")]
    assert [row.name for row in export_rows()] == expected


def test_rows_point_at_defining_module() -> None:
    """Re-exported classes link to the module that defines them."""
    rows = {row.name: row for row in export_rows()}
    runner = rows["ExplainRunner"]
    assert runner.target == "nlp_shap.pipeline.runner.ExplainRunner"
    assert runner.role == "class"


def test_private_module_objects_keep_public_path() -> None:
    """Objects defined in ``nlp_shap._*`` modules are linked by their public name."""
    for row in export_rows():
        assert "._" not in row.target, row


def test_canonical_reference_maps_reexports() -> None:
    """Short public paths resolve to the defining module, other paths do not."""
    assert (
        canonical_reference("nlp_shap.ExplainRunner")
        == "nlp_shap.pipeline.runner.ExplainRunner"
    )
    assert (
        canonical_reference("nlp_shap.ExplainRunner.explain_sync")
        == "nlp_shap.pipeline.runner.ExplainRunner.explain_sync"
    )
    assert canonical_reference("nlp_shap.pipeline.runner.ExplainRunner") is None
    assert canonical_reference("nlp_shap.DoesNotExist") is None
    assert canonical_reference("numpy.ndarray") is None


def test_first_doc_line_ignores_inherited_docstrings() -> None:
    """A class without its own docstring yields an empty summary."""

    class Undocumented(nlp_shap.ExplainRunner):
        pass

    assert first_doc_line(Undocumented) == ""
    assert first_doc_line(nlp_shap.ExplainRunner)


def test_qualify_roles_rewrites_bare_targets() -> None:
    """Bare role targets are qualified against the object's module."""
    summary = qualify_roles(
        "Parse YAML into :class:`ExplainConfig` or :class:`Missing`.",
        nlp_shap.explain_config_from_yaml,
    )
    assert ":py:class:`~nlp_shap.pipeline.config.ExplainConfig`" in summary
    assert "``Missing``" in summary
