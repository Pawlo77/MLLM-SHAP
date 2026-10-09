# nlp-shap development reference

Load only when the task needs detail beyond `SKILL.md`.

## Docs deliverables

| Deliverable | Location |
|-------------|----------|
| Theory + formulas + paper links | `docs/theory/<topic>.rst` |
| Runnable usage | `docs/guides/<topic>.rst` |
| Module reference | `docs/api.rst` (`automodule`) |
| End-to-end notebook | `examples/<name>.ipynb` + catalog in `examples/README.md`, `docs/examples.rst`, `README.md` |
| Navigation | `docs/index.rst` toctree |
| Release notes | `docs/release_notes.rst` — section per version; `Unreleased` until tag |

User-facing `docs/` and `examples/`: **shipped API only** — no phases or rewrite progress.

## Notebook checklist

- Cell 1 = all imports (stdlib/third-party then `nlp_shap`)
- Full public usability scope for the feature; no stubs
- Dual-audience section commentary (technical + business)
- `make notebooks` before commit, after a `src/` change to a public workflow and before a release PR (outputs stored)

## Test markers

| Marker | Command | Notes |
|--------|---------|-------|
| (default unit) | `make tests` / `make agent-check` | Skips `lms` / `gpu` / `bench` |
| all markers | `make tests-all` | Includes optional markers (skip if deps missing) |
| `lms` | `pytest -m lms` | Needs local LM Studio |
| `gpu` | `pytest -m gpu` | Needs torch/transformers |
| `bench` | `make bench` | Local only; not in `make check` |

`make check` runs `make agent-check` (filtered pytest and `prek-all`), then `make vault-check`.

## Plugin / entry points

New estimator, estimand, backend, value_fn, normalizer, partition, absence_policy, or renderer → register in `pyproject.toml` `[project.entry-points."nlp_shap.*"]` and exercise via `PluginRegistry`. See rule `plugins.mdc` and skill `plugin`.

## Sibling repos

Clone as siblings of `nlp-shap/` (`CONTRIBUTING.md`, "Agent setup"; workspace file `nlp-shap.code-workspace`):

| Repo | Clone URL | Path from `nlp-shap/` |
|------|-----------|------------------------|
| Legacy MLLM-Shap | `https://github.com/Pawlo77/MLLM-Shap.git` | `../MLLM-Shap/mllm_shap/src/mllm_shap/` |

Open `<parent>/nlp-shap/nlp-shap.code-workspace` after cloning.
