# Contributing to mllm-shap

## Security

Report a vulnerability through GitHub's private reporting or by email to the maintainer.
Never commit secrets, tokens or personal data.

## What to contribute

- **Estimators** for coalition sampling (new approximation schemes, variance reduction) with a test against the exact reference.
- **Value functions** that score coalition outputs, and **estimands** that aggregate coalition payoffs.
- **Backends** for model inference. A new backend needs correctness tests, a benchmark under `tests/benchmarks/` and a minimal notebook in `examples/`.
- **Documentation**: guides, theory pages, API docs and notebooks.
- **Benchmarks** that cover hot runtime paths.
- **Bug reports and fixes**, ideally with a minimal configuration and snapshot that reproduces the problem.

Open an issue before a larger change to agree on its scope and design.

## Development setup

Prerequisites: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) (the project pins a minimum uv version in `pyproject.toml`; run `uv self update` if uv complains).

```bash
git clone https://github.com/Pawlo77/MLLM-SHAP.git
cd MLLM-SHAP
make install
```

`make install` installs Python 3.12, syncs all dependency groups and extras into `.venv` and installs the [prek](https://github.com/j178/prek) hooks (pre-commit, commit-msg, pre-push) in the main checkout; in a git worktree it only syncs the environment, because the hooks are shared and tied to the main checkout's venv. The pre-push hook runs `make agent-check`, the same checks CI runs. Change dependencies with `uv add` and `uv lock`, never `pip install`; a new runtime dependency needs a justification in `pyproject.toml`.

Jupyter kernel: `make kernel` (`mllm-shap (uv)`).

## Obsidian vault setup

Obsidian rewrites its settings on every open, so the live files under `.obsidian/` are not committed. The committed `*.example.json` files hold the community plugin list and the default settings. On a fresh clone, copy them once (existing files are kept):

```bash
find .obsidian -name '*.example.json' -exec sh -c 'cp -n "$1" "${1%.example.json}.json"' _ {} \;
```

Then open the folder as a vault and run `make obsidian-plugins`. To share a setting with the team, edit its `.example.json` by hand; never commit a live `.json` file.

## Quality gates

Run these before you open a pull request. CI runs the same checks.

| Command | When |
|---------|------|
| `make check` | Always. Filtered pytest, all prek hooks (Ruff, format, mypy strict, Bandit, codespell, hygiene) and `make vault-check` (vault notes, links, traceability tags). |
| `make docs` | When `docs/` or the public API changed. Strict Sphinx build plus a link check, so it needs network access. |
| `make notebooks` | When `examples/*.ipynb` changed, after a `src/` change to a public workflow, and before every release pull request; it re-executes the notebooks and stores outputs. |

Pytest markers select tests that need extra resources and are excluded from `make check`:

- `lms`: needs a local LM Studio server.
- `gpu`: needs `torch` and optionally a GPU.
- `bench`: local performance benchmarks (`make bench`).
- `slow`: builds a wheel or reaches the network.
- `practical`: real small models on a GPU (below).

`make tests-all` runs everything, including these markers.

### Practical tests

`make practical TIER=<quick|deep|all> FEATURE=<id|glob|all>` runs `tests/practical/scenarios.yaml` (quick: MPS or CUDA; deep: LM Studio). Pins live in `tests/practical/models.yaml`, results in `tests/practical/results/latest.json`.

| Tier | Models | Memory | Cold / warm (M4 Pro) |
|------|--------|--------|----------------------|
| `quick` (default) | `Qwen/Qwen2.5-1.5B-Instruct` 3.1 GB, `SmolLM2-360M-Instruct` 0.7 GB | under 4 GB | 130 s / 70 s |
| `deep` (`deep.*`, before a release) | LM Studio `qwen2.5-3b-instruct`, judge `gemma-3-4b-it`; `deep.runtime.kv_cache` loads `Qwen/Qwen2.5-3B-Instruct` 6.2 GB | 7.4 GB | 13 min / 4 min |

- Deep tier: `lms server start` (no server or key: skip); keep LM Studio idle otherwise. `MLLM_SHAP_PRACTICAL_DEEP_MODEL` and `_DEEP_JUDGE`: other keys.
- `MLLM_SHAP_PRACTICAL_DOWNLOAD=1`: download missing weights.
- `MLLM_SHAP_PRACTICAL_CPU=1`: run without a GPU (slow).
- `MLLM_SHAP_PRACTICAL_LMS_MODEL`: LM Studio key (`make practical` default `qwen2-500m-instruct`).
- `MLLM_SHAP_PRACTICAL_API_HOST` (default `http://localhost:1234/v1`) and `MLLM_SHAP_PRACTICAL_API_MODEL`; a remote host needs `MLLM_SHAP_PRACTICAL_REMOTE=1`.
- `MLLM_SHAP_PRACTICAL_JUDGE=lmstudio:<key>`: the quick tier's judge runs in LM Studio.

Write tests first for new behaviour and bug fixes.

## Project rules

- **Brevity:** one home per fact, link instead of restating; file budgets are enforced by `scripts/check_brevity.py`.
- **Generated files:** never edit `.claude/rules/`, `.codex/agents/` or `knowledge/generated/`; change the source and rerun `scripts/sync_rules.py`, `scripts/sync_agents.py` or `make knowledge`.
- **Template files:** files that come from [python-kit-template](https://github.com/Pawlo77/python-kit-template) change there, not here.

## Branches, commits and pull requests

- Branch from `main` with a type prefix (`feat/`, `fix/`, `docs/`, `test/`, `refactor/`, `chore/`, `ci/`, `perf/`); `main` is protected.
- Commit subjects are `type: summary` (`feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `build`, `perf`): imperative, no trailing period, at most 72 characters, one concern per commit. A `commit-msg` hook enforces it; never bypass hooks with `--no-verify` or force-push shared branches.
- The PR title uses the same format (workflow `pr-title`). Fill in the pull request template; open it ready for review (no draft or WIP) and keep it focused.
- The `assign-author.yaml` workflow assigns the author and labels the PR from the title type (`fix` is `bug`, `docs` is `documentation`, the rest `enhancement`); if it did not run: `gh pr edit <n> --add-assignee Pawlo77 --add-label <label>`.
- Required checks: `prek`, `tests-passed`, `title`; CodeQL also runs on every PR. Resolve every review thread before asking for a merge. The maintainer merges with a merge commit.

## Project layout

| Path | Role |
|------|------|
| `src/mllm_shap/domain/` | Cooperative-game types: players, coalitions, conversation snapshots, estimands. |
| `src/mllm_shap/protocols/` | Protocol contracts that plugins implement. |
| `src/mllm_shap/masking/` | Partitioning, absence policies, mask building and coalition codecs. |
| `src/mllm_shap/estimation/` | Estimators (exact, Monte Carlo, complementary, Neyman), estimands and normalizers. |
| `src/mllm_shap/value/` | Value functions that score coalition outputs. |
| `src/mllm_shap/backends/` | Model inference backends (mock, LM Studio, transformers, HTTP API). |
| `src/mllm_shap/runtime/` | Run archive, deduplication, caches, scheduler and metrics. |
| `src/mllm_shap/pipeline/` | Configuration, orchestration, runner and result objects. |
| `src/mllm_shap/plugins/` | Plugin registry and entry-point discovery. |
| `src/mllm_shap/viz/` | Token attribution renderers. |
| `tests/` | Unit, integration and benchmark tests, mirroring the package layout. |
| `docs/` | Sphinx documentation (guides, theory, API reference, release notes). |
| `examples/` | Executed Jupyter notebooks. |
| `config/` | Example pipeline configurations. |

## Adding a plugin

Estimators, estimands, value functions, normalizers, backends, partitions, absence policies and renderers are entry-point plugins. Implement the protocol, register the entry point and test through `PluginRegistry` as the [extending guide](docs/guides/extending.rst) describes; a backend also needs a benchmark and a notebook.

## Review automation

Cursor Bugbot reviews every pull request. CodeQL runs on every PR. Resolve or answer every review thread before asking for a merge.

## Releases

The next version comes from `scripts/pr_policy.py --bump` (commits since the last `v*` tag); the pull request carries the `chore: release vX.Y.Z` commit and `vX.Y.Z` is tagged on `main` after the merge. The Publish workflow uploads to PyPI and creates the GitHub Release. Failed releases are fixed forward; tags never move.

## Upgrades

`make upgrade-check` lists outdated packages, prek hooks, action pins, uv and Obsidian plugins; `make upgrade` applies them, except the protected `publish.yaml`. A weekly routine pull request ships them.

## AI-assisted contributions

AI tools are welcome. The contributor is responsible for the change: a human reviews every AI-written change, runs the quality gates and can explain the diff. Instructions for coding agents live in `AGENTS.md`.

## Environment variables

Copy `.env.example` to `.env` (git-ignored) and fill in only the variables you need: the HTTP API backend keys and a logging configuration path; never commit real values.

## Questions

Open a GitHub [issue](https://github.com/Pawlo77/MLLM-SHAP/issues) for usage questions, ideas, bugs and concrete feature requests. Documentation: <https://pawlo77.github.io/MLLM-SHAP/>.
