# Development: MLLM-SHAP

## Conventions (pointers only)

- Logging bootstrap: `python-logging.mdc`
- Types, imports, no `__future__`: `python-types.mdc`
- Docstrings: `python-docstrings.mdc`
- Docs and release notes: `docs.mdc`
- Notebooks: `examples.mdc`
- Benchmarks: `benchmarks.mdc` (`make bench`, not part of `make check`)
- Legacy port: skill `port-legacy` (`../MLLM-Shap-legacy/mllm_shap/src/mllm_shap/`)
- Audio rewrite: skill `rewrite`

Details that grow: [reference.md](reference.md).

## Extra steps

- **Read**: for rewrite work also skill `rewrite`; for a `src/` change also
  `knowledge/architecture/<pkg>.md`.
- **Document**: a public API or algorithm change follows `docs.mdc`, plus a notebook when
  `examples.mdc` requires one.
- **Validate**: update `knowledge/architecture/<pkg>.md` for every changed package (skill
  `notes`); `make knowledge` before `make check`; `make docs` and `make notebooks` when the
  exit criteria require them; perf review on hot paths.

```bash
make knowledge
make check
make docs        # when the public API or docs/ changed
make notebooks   # notebooks or a public workflow changed, or a release PR
```

Packaging: `make build`.

## Pitfalls

- Notebooks without outputs are not committed.
- A public feature without `docs/` is incomplete.
- Phase numbers or roadmap text in `docs/` are forbidden.
