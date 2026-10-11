"""Ensure domain modules remain backend-free.

@see [[knowledge/architecture/domain]]
"""

import ast
from pathlib import Path


def test_domain_modules_do_not_import_backends() -> None:
    """Domain code must not depend on torch or backend packages."""
    domain_root = Path(__file__).resolve().parents[2] / "src" / "mllm_shap" / "domain"
    forbidden_roots = {
        "torch",
        "transformers",
        "mllm_shap.backends",
        "mllm_shap.plugins",
    }
    violations: list[str] = []

    for path in sorted(domain_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path.name}: import {alias.name}"
                    for alias in node.names
                    if _is_forbidden(alias.name, forbidden_roots)
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and _is_forbidden(node.module, forbidden_roots)
            ):
                violations.append(f"{path.name}: from {node.module}")

    assert violations == []


def _is_forbidden(module: str, forbidden_roots: set[str]) -> bool:
    return any(
        module == root or module.startswith(f"{root}.") for root in forbidden_roots
    )
