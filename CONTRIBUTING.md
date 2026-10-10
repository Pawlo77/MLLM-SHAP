# Contributing to nlp-shap

## Security

Report a vulnerability through GitHub's private reporting or by email to the maintainer.

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
git clone https://github.com/Pawlo77/nlp-shap.git
cd nlp-shap
make install
```

`make install` installs Python 3.12, syncs all dependency groups and extras into `.venv` and installs the [prek](https://github.com/j178/prek) hooks (pre-commit, commit-msg, pre-push). The pre-push hook runs `make agent-check`, the same checks CI runs. Change dependencies with `uv add` and `uv lock`, never `pip install`; a new runtime dependency needs a justification in `pyproject.toml`.

Jupyter kernel: `make kernel` (`nlp-shap (uv)`).

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

`make practical TIER=<quick|deep|all> FEATURE=<id|glob|all>` runs `tests/practical/scenarios.yaml` in float16 on MPS or CUDA. Pins live in `tests/practical/models.yaml`, results in `tests/practical/results/latest.json`.

| Tier | Models | Memory | Cold / warm (M4 Pro) |
|------|--------|--------|----------------------|
| `quick` (default) | `Qwen/Qwen2.5-1.5B-Instruct` 3.1 GB, `SmolLM2-360M-Instruct` 0.7 GB | under 4 GB | 130 s / 70 s |
| `deep` (`deep.*`, before a release) | `Qwen/Qwen2.5-3B-Instruct` 6.2 GB | 7.4 GB | 28 min / 4 min |

- `NLP_SHAP_PRACTICAL_DOWNLOAD=1`: download missing weights.
- `NLP_SHAP_PRACTICAL_CPU=1`: run without a GPU (slow).
- `NLP_SHAP_PRACTICAL_LMS_MODEL`: a downloaded LM Studio model key (`make practical` default `qwen2-500m-instruct`).
- `NLP_SHAP_PRACTICAL_API_HOST` (default `http://localhost:1234/v1`) and `NLP_SHAP_PRACTICAL_API_MODEL`; a remote host needs `NLP_SHAP_PRACTICAL_REMOTE=1`.
- `NLP_SHAP_PRACTICAL_JUDGE=lmstudio:<key>`: the quick tier's judge runs in LM Studio, for example `qwen2.5-7b-instruct` 4-bit (4.7 GB).

Code rules live in `.cursor/rules/python-*.mdc`; write tests first for new behaviour and bug fixes.

## Project layout

| Path | Role |
|------|------|
| `src/nlp_shap/domain/` | Cooperative-game types: players, coalitions, conversation snapshots, estimands. |
| `src/nlp_shap/protocols/` | Protocol contracts that plugins implement. |
| `src/nlp_shap/masking/` | Partitioning, absence policies, mask building and coalition codecs. |
| `src/nlp_shap/estimation/` | Estimators (exact, Monte Carlo, complementary, Neyman), estimands and normalizers. |
| `src/nlp_shap/value/` | Value functions that score coalition outputs. |
| `src/nlp_shap/backends/` | Model inference backends (mock, LM Studio, transformers, HTTP API). |
| `src/nlp_shap/runtime/` | Run archive, deduplication, caches, scheduler and metrics. |
| `src/nlp_shap/pipeline/` | Configuration, orchestration, runner and result objects. |
| `src/nlp_shap/plugins/` | Plugin registry and entry-point discovery. |
| `src/nlp_shap/viz/` | Token attribution renderers. |
| `tests/` | Unit, integration and benchmark tests, mirroring the package layout. |
| `docs/` | Sphinx documentation (guides, theory, API reference, release notes). |
| `examples/` | Executed Jupyter notebooks. |
| `config/` | Example pipeline configurations. |

## Adding a plugin

Estimators, estimands, value functions, normalizers, backends, partitions, absence policies and renderers are entry-point plugins. Implement the protocol, register the entry point and test through `PluginRegistry` as the [extending guide](docs/guides/extending.rst) describes; a backend also needs a benchmark and a notebook.

## Branches, commits and pull requests

- Branch from `main` with a type prefix (`feat/`, `fix/`, `docs/`, `test/`, `refactor/`, `chore/`, `ci/`, `perf/`); `main` is protected.
- Commit subjects are `type: summary` (`feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `build`, `perf`): imperative, no trailing period, at most 72 characters, one concern per commit. A `commit-msg` hook enforces it; never bypass hooks with `--no-verify` or force-push shared branches.
- The PR title uses the same format (workflow `pr-title`). Fill in the pull request template; open it ready for review (no draft or WIP) and keep it focused.
- The `assign-author.yaml` workflow assigns the author and labels the PR from the title type (`fix` is `bug`, `docs` is `documentation`, the rest `enhancement`); if it did not run: `gh pr edit <n> --add-assignee Pawlo77 --add-label <label>`.
- Required checks: `prek`, `tests-passed`, `CodeQL`, `title`. The maintainer merges with a merge commit.

## Review automation

Cursor Bugbot reviews every pull request with the rules in [.cursor/BUGBOT.md](.cursor/BUGBOT.md). CodeQL runs on every PR. Resolve or answer every review thread before asking for a merge.

## Releases

Skill `release` takes the next version from `scripts/pr_policy.py --bump` (commits since the last `v*` tag), commits `chore: release vX.Y.Z` and tags `vX.Y.Z` on `main` after the merge. The Publish workflow uploads to PyPI and creates the GitHub Release. Failed releases are fixed forward; tags never move.

## Upgrades

`make upgrade-check` lists outdated packages, prek hooks, action pins, uv and Obsidian plugins; `make upgrade` applies them, except the protected `publish.yaml`. Skill `upgrade` ships them weekly as a routine pull request.

## AI-assisted contributions

Coding agents are welcome as tools. Agent rules (exit criteria, commits, memory, vault) live in [AGENTS.md](AGENTS.md). The contributor is responsible for the change: a human reviews every agent-written change, runs the quality gates and can explain the diff. Agents never commit, push or open pull requests without explicit approval. Claude reviews its own work locally before a pull request is opened (skill `review`, agent `reviewer`); it does not review or comment on pull requests on GitHub, where Cursor Bugbot is the reviewer.

## Agent setup

Delivery runs through skill `ship`: routine changes (`scripts/pr_policy.py`) merge after Copilot's review, others stop before the merge. Copilot code review must be enabled on the repository (Settings → Copilot → Code review).

### Sibling repositories

One sibling repository is not vendored here: `MLLM-Shap` (the legacy implementation that skill `port-legacy` ports from). Clone it next to this repository under one parent directory:

```bash
mkdir -p <parent> && cd <parent>
for r in nlp-shap MLLM-Shap; do git clone https://github.com/Pawlo77/$r.git; done
```

Then open `<parent>/nlp-shap/nlp-shap.code-workspace` (File, Open Workspace from File). Paths in that file are relative to `nlp-shap/` (`../MLLM-Shap`).

### Tool matrix

One source per concern; every tool reads it through its own entry point.

| Concern | Source | Cursor | Claude Code | Codex |
|---------|--------|--------|-------------|-------|
| Instructions | `AGENTS.md` (plus nested `AGENTS.md`) | native | `CLAUDE.md` symlinks | native |
| Skills | `.agents/skills/` | `.cursor/skills` symlink | `.claude/skills` symlink | native |
| Commands | `.cursor/commands/` | native | `.claude/commands` symlink | none |
| Rules | `.cursor/rules/*.mdc` | native | `.claude/rules/*.md`, generated by `scripts/sync_rules.py` | skill `python-style` |
| Hooks | `scripts/hooks/*.sh` | `.cursor/hooks.json` | `.claude/settings.json` | `.codex/hooks.json` |
| Subagents | `.claude/agents/*.md` (read-only) | reads `.claude/agents/` | native | `.codex/agents/*.toml`, generated by `scripts/sync_agents.py` |
| Settings | per tool | `.cursor/hooks.json` | `.claude/settings.json` (permissions, status line, plugins) | `.codex/config.toml` |

Generated files are never edited by hand: change the source and rerun its generator (`AGENTS.md`, "Agent-only").

### Plugins

`.claude/settings.json` declares the marketplaces and the shared set in `enabledPlugins`; Claude Code offers to install it on first trust of the folder. Required: `superpowers`, `remember` (memory in `.remember/`) and `astral` (the uv, ruff and ty skills and the ty language server). The rest are optional; notes:

- `security-guidance`: the per-Stop model review is off (`ENABLE_STOP_REVIEW=0`); pattern warnings stay.
- `claude-jit-context@dpt-plugins` injects `.claude/jit-context/` entries when a matching file, word or command comes up. `make knowledge` writes them; never run the plugin's `rebuild-tsv.sh` over `10-auto/`.
- Trail of Bits skills (CC-BY-SA-4.0) are installed as plugins, never copied here. Vendored agents are listed in `.claude/agents/VENDORED.md`.

Install lines per tool:

- **Claude Code:** accept the plugin prompt, or `/plugin marketplace add` each marketplace in the settings and `/plugin install <name>@<marketplace>` each entry of `enabledPlugins`.
- **Codex:** `codex plugin marketplace add Digital-Process-Tools/claude-remember && codex plugin install remember`; `codex plugin marketplace add Digital-Process-Tools/codex-marketplace && codex plugin add jit-context@dpt-plugins`; `codex plugin install astral` for the uv, ruff and ty skills (or read them from the Claude Code plugin cache); Superpowers from the official Codex marketplace (`/plugins`).
- **Cursor:** `/add-plugin superpowers`; Cursor also reads `.claude/agents/`, `.claude/skills/` and `.cursor/hooks.json`, and its `sessionStart` hook injects the `.remember/` handoff.

### Literature hosts

Claude Code fetches from Semantic Scholar, arXiv and Crossref (plus `doi.org` redirects) without a prompt, so `paper-scout` and the literature watch run unattended. The accepted risk is exfiltration: an injected prompt could put file contents into a query string sent to those hosts. The Read deny list (secrets, keys, credential files, every `.cursorignore` entry) limits what could leak; the shell guard does not inspect network commands. Report a way around these limits as a vulnerability (section Security).

### Codex

Codex reads `AGENTS.md` and `.agents/skills/` natively. `.codex/config.toml` sets the sandbox and approval policy, `.codex/hooks.json` runs the shared `scripts/hooks/*.sh` (trust them once in Codex's hook review) and `.codex/agents/*.toml` mirror `.claude/agents/`. Codex has no `TaskCompleted`, `PostToolUseFailure` or `FileChanged` event, so those gates run in Claude Code only; run `make check` before hand-off and ask before any commit, push, tag or pull request.

### Environment variables

Copy `.env.example` to `.env` (git-ignored) and fill in only the variables you need: the HTTP API backend keys, a logging configuration path and the remember plugin switch; never commit real values.

## Questions

Open a GitHub [issue](https://github.com/Pawlo77/nlp-shap/issues) for usage questions, ideas, bugs and concrete feature requests. Documentation: <https://pawlo77.github.io/nlp-shap/>.
