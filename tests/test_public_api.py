"""Public exports, error hierarchy and import layering.

@see [[knowledge/architecture/core]]
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import mllm_shap
from mllm_shap import errors
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import BUILTIN_PLUGINS

SRC = Path(__file__).resolve().parents[1] / "src" / "mllm_shap"


def test_every_builtin_plugin_class_is_exported() -> None:
    for factories in BUILTIN_PLUGINS.values():
        for factory in factories.values():
            assert getattr(mllm_shap, factory.__name__) is factory


@pytest.mark.parametrize(
    "name",
    [
        "TokenPartitioner",
        "DeletePolicy",
        "PadPolicy",
        "NeutralPolicy",
        "ObservabilitySink",
        "InMemoryObservabilitySink",
        "CoalitionProgress",
        "PluginContext",
        "configure_logging",
        "render_attribution",
        "display_attribution_html",
    ],
)
def test_named_exports(name: str) -> None:
    assert name in mllm_shap.__all__
    assert getattr(mllm_shap, name) is not None
    assert name in dir(mllm_shap)


def test_all_errors_are_exported_and_share_a_root() -> None:
    error_types = [
        value
        for value in vars(errors).values()
        if isinstance(value, type) and issubclass(value, Exception)
    ]
    assert len(error_types) == 13
    for error_type in error_types:
        assert issubclass(error_type, errors.MllmShapError)
        assert getattr(mllm_shap, error_type.__name__) is error_type
    assert issubclass(errors.PluginNotFoundError, LookupError)
    assert issubclass(errors.ConfigError, ValueError)
    assert issubclass(errors.BackendUnavailableError, RuntimeError)
    assert issubclass(errors.LogprobsUnavailableError, errors.BackendUnavailableError)
    assert issubclass(errors.ArchiveNotFoundError, FileNotFoundError)
    assert issubclass(errors.ArchiveCorruptError, ValueError)


def test_runtime_and_domain_do_not_import_the_pipeline() -> None:
    """``runtime`` and ``domain`` sit below ``pipeline``; no import goes upward."""
    violations: list[str] = []
    for package in ("runtime", "domain"):
        for path in sorted((SRC / package).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            violations.extend(
                f"{package}/{path.name}: {node.module}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("pipeline")
                and node.level == 2
            )
    assert violations == []


@pytest.mark.parametrize(
    "module",
    [
        "mllm_shap.runtime",
        "mllm_shap.backends.api",
        "mllm_shap.plugins",
        "mllm_shap.viz.token_text",
        "mllm_shap.masking.policies",
    ],
)
def test_submodules_import_in_a_fresh_interpreter(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _config(**explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "m"},
        "explanation": explanation,
    })


def test_exact_requires_full_budget() -> None:
    with pytest.raises(ValidationError, match=r"budget\.fraction"):
        _config(estimator="exact", budget={"fraction": 0.5})


def test_neyman_rejects_minimal_masks() -> None:
    with pytest.raises(ValidationError, match="include_minimal_masks"):
        _config(estimator="neyman_cc", include_minimal_masks=True)


def test_valid_combinations_pass() -> None:
    _config(estimator="exact")
    _config(estimator="mc", budget={"fraction": 0.5}, include_minimal_masks=True)
