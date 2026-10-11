---
paths:
  - "src/mllm_shap/**/*.py"
  - "tests/**/*.py"
  - "examples/**/*.ipynb"
---

<!-- Generated from .cursor/rules/python-types.mdc by scripts/sync_rules.py; do not edit. -->

# Python types

Match tooling in `pyproject.toml` — do not fight ruff or mypy.

- Imports at module top; no inline imports unless circular dependency is documented
- **Inside `src/mllm_shap/`:** relative imports only (`from .module import X`, `from ..pkg import Y`) — never `from mllm_shap...`
- **Outside the package** (`tests/`, `examples/`): absolute imports (`from mllm_shap...`)
- Annotate all public functions, methods, and module-level names in `src/mllm_shap`
- Prefer `pathlib.Path`, explicit returns, `X | Y` over `Optional`
- Do not use `from __future__ import annotations` unless required — prefer `typing.Self`, quoted forward refs (`"MyClass"`), or import reordering. Never add it as a default module header.
- Prefer `TypedDict` for structured dict payloads (manifests, config sections, wire formats) — required keys live in the type, not runtime `if key not in payload` checks
- Use `Literal[...]` for closed string unions on the wire; reserve `dict[str, Any]` for true dynamic bags only
- Optional behaviour is a capability protocol (`typing.Protocol`, `@runtime_checkable` when checked with `isinstance`), not `getattr`/`hasattr` probing
- Runtime validation belongs at external boundaries (`json.loads`, HTTP) — convert to a `TypedDict` once, then pass typed payloads inward
- No bare `except:`; no `# type: ignore` without a comment
- Ruff line length 88 — run the formatter
- Exhaustive `match` on unions: `never` in the default branch
- No bare `*` for keyword-only parameters unless required — e.g. multiple optional args where positional calls would be ambiguous, or a deliberate API boundary. Do not add `*` just to mark one trailing parameter as keyword-only.
- Do not use `...` as a stub function body when a docstring is already present — the docstring alone is a valid body for `Protocol` and abstract methods
