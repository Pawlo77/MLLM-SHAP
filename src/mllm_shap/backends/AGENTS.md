# Backend agents

- Extras only for torch/transformers/lmstudio/httpx
- CI = mock; local markers `lms` / `gpu`
- A new backend needs an entry point, tests, a benchmark under `tests/benchmarks/` and a
  minimal notebook in `examples/` (all required)
- Any backend change → update `knowledge/architecture/backends.md` (skill `notes`)

See `.cursor/rules/backends.mdc` and skill `plugin`.
