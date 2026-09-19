---
name: nlp-shap-development
description: >-
  Ordered implementation workflow for nlp-shap. Use for any code, test,
  packaging, hook, or CI change.
---

# nlp-shap development

Exit: `make agent-check` / `make check` (`AGENTS.md`). Rules: `.cursor/rules/*.mdc` (alwaysApply + matching globs).

## Conventions (pointers only)

- Logging bootstrap → `python-logging.mdc`
- Types / imports / no `__future__` → `python-types.mdc`
- Docstrings → `python-docstrings.mdc`
- Docs / release notes → `docs.mdc`
- Notebooks → `examples.mdc`
- Benches → `benchmarks.mdc` (`make bench`, not `make check`)
- Legacy port → skill `nlp-shap-port-legacy` (`../MLLM-Shap/mllm_shap/`)
- Audio rewrite → skill `nlp-shap-rewrite`

Details that grow: [reference.md](reference.md).

## Workflow

1. **Read** — task scope only; existing `src/` + `tests/`; rewrite → also `nlp-shap-rewrite`
2. **Red** — failing `tests/test_*.py` first (mirror `src/`); behavior not trivia
3. **Green** — smallest typed impl; reuse before new abstractions
4. **Document** — public API/algorithm change → `docs.mdc` + optional notebook (`examples.mdc`)
5. **Refactor** — simplify while green; no unrelated cleanup
6. **Validate** — re-read diff; `make agent-check`; `make docs` / `make notebooks` when applicable; perf review on hot paths; fix all issues found
7. **Hand off** — stop unless user asks commit/push/PR (`git-commits.mdc`)

```bash
make notebooks   # when examples/*.ipynb changed
make agent-check
make docs        # when API/docs changed
```

Packaging → `make build` · deps → `uv lock` · new Makefile target → `name: ## help` + `.PHONY`

After push: `gh run watch <run-id> --exit-status`.

## Pitfalls

`pre-commit` → **prek** · `pip install` → pyproject + `uv lock` · implement-before-test → back to Red · push without `make check` → remote Prek red · notebooks without outputs → do not commit · public feat without `docs/` → incomplete · phase/roadmap in `docs/` → forbidden
