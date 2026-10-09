---
name: development
description: >-
  Use for any code, test, packaging, hook or CI change in nlp-shap.
---

# nlp-shap development

Exit criterion: `make check` (`AGENTS.md`). Rules: `.cursor/rules/*.mdc` (always-apply rules
plus the ones whose globs match the files you touch).

## Conventions (pointers only)

- Logging bootstrap: `python-logging.mdc`
- Types, imports, no `__future__`: `python-types.mdc`
- Docstrings: `python-docstrings.mdc`
- Docs and release notes: `docs.mdc`
- Notebooks: `examples.mdc`
- Benchmarks: `benchmarks.mdc` (`make bench`, not part of `make check`)
- Legacy port: skill `port-legacy` (`../MLLM-Shap/mllm_shap/src/mllm_shap/`)
- Audio rewrite: skill `rewrite`

Details that grow: [reference.md](reference.md).

## Workflow

1. **Read**: task scope only; existing `src/` and `tests/`; for rewrite work also skill
   `rewrite`; for a `src/` change also `knowledge/architecture/<pkg>.md`.
2. **Red**: failing `tests/test_*.py` first (mirror `src/`); test behaviour, not trivia.
3. **Green**: smallest typed implementation; reuse before new abstractions.
4. **Document**: a public API or algorithm change follows `docs.mdc`, plus a notebook when
   `examples.mdc` requires one.
5. **Refactor**: simplify while green; no unrelated cleanup.
6. **Validate**: re-read the diff; update `knowledge/architecture/<pkg>.md` for every changed
   package (skill `notes`); run `make knowledge`, then `make check` (it includes
   `make vault-check`); `make docs` and `make notebooks` when the exit criteria require them;
   perf review on hot paths; fix every issue found.
7. **Hand off**: stop unless the user asks for a commit, push or PR (`git-commits.mdc`).

```bash
make knowledge
make check
make docs        # when the public API or docs/ changed
make notebooks   # notebooks or a public workflow changed, or a release PR
```

Packaging: `make build`. Dependencies: `uv lock`. A new Makefile target gets a
`name: ## help` line and a `.PHONY` entry.

After push: `gh run watch <run-id> --exit-status`.

## Pitfalls

- Use `prek`, not `pre-commit`; change dependencies in `pyproject.toml` and `uv lock`, not
  with `pip install`.
- Implementation before the test: go back to Red.
- A push without `make check` turns the remote Prek Checks red.
- Notebooks without outputs are not committed.
- A public feature without `docs/` is incomplete.
- Phase numbers or roadmap text in `docs/` are forbidden.
