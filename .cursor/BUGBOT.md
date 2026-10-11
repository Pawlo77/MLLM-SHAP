# Review rules

You review pull requests for mllm-shap, an open-source library for Shapley-value explanations of
conversational language models: estimators and estimand aggregators (`src/mllm_shap/estimation/`),
masking, an async runtime and archive, the explain pipeline, plugin backends, Sphinx docs,
example notebooks, and an Obsidian knowledge vault (`knowledge/`). Cite `AGENTS.md`, a
`.cursor/rules/*.mdc` file or the skill `review` checklist when you flag a rule.

This file is for Cursor Bugbot. Claude does not review pull requests on GitHub; it reviews its
own work locally before the pull request is opened (skill `review`).

Keep comments few, specific and fix-oriented. CI already runs prek (Ruff, mypy, bandit, codespell,
zizmor, the vault check) and pytest on Python 3.12–3.14; do not restate what they catch.

## Review order

1. **Secrets and supply chain:** tokens, keys or private hostnames anywhere (code, notes,
   `.remember/`), a checked-in `.env`, GitHub Actions not pinned to a commit SHA or exact
   release tag, new install or download steps, a new runtime dependency without a justification
   in `pyproject.toml`.
2. **Correctness against the estimand and the axioms:** efficiency
   (`sum(values) == grand_value - base_value` for Shapley), the empty and grand coalitions always
   evaluated, complementary pairs complementary (including `(N, ∅)`), Shapley weights
   `|S|! (n - |S| - 1)! / n!`, seeded randomness passed explicitly, scheduler errors propagated,
   archive and manifest versioning, `run_id` derived from snapshot and config.
3. **Tests loosened or missing:** a behaviour change without a known-truth test (additive,
   symmetric or random game with exact values). Deleting or loosening an assertion (wider
   tolerance, smaller sample, a skipped test) without a stated reason is a **Blocker**.
4. **Docs and vault contradicting code:** guides, theory pages, release notes, notebooks or
   `knowledge/` notes that disagree with what the code does; numbers not taken from executed
   output; phase numbers or roadmap text in user-facing docs.
5. **Maintainability:** a second implementation of something the package already has,
   `getattr` capability probing instead of a protocol, `dict[str, Any]` where a `TypedDict`
   exists, missing Google docstrings on public surfaces.

## Scope by path

| Path | Bar |
|------|-----|
| `src/mllm_shap/estimation/**` | Highest. Every change runs the axiom tests (`tests/estimation/test_axioms.py`, `tests/pipeline/test_pipeline_axioms.py`); no estimator or estimand name switches inside an aggregator; no `2^n` materialisation outside the exact estimator |
| `src/mllm_shap/pipeline/**` | Manifest fields and archive schema version bumped when the config or columns change; `archive.on_exists` honoured; errors propagate from the scheduler |
| `src/mllm_shap/backends/**` | Heavy dependencies only through optional extras; capabilities through protocols, not `getattr`; a benchmark and a minimal notebook for a new backend |
| `src/mllm_shap/**` (other) | Relative imports; mypy strict; Google docstrings; the architecture note `knowledge/architecture/<pkg>.md` updated in the same PR |
| `tests/**` | Tests assert values, not shapes; known-truth games for estimators; no loosened tolerances without a reason |
| `docs/**`, `knowledge/**` | Match the code; no phase numbers in `docs/`; no secrets, hostnames or personal data; generated notes (`generated: true`) never edited by hand |
| `.github/**` | Minimal `permissions`, `persist-credentials: false`, actions pinned by SHA or exact tag, no untrusted input interpolated into `run:` |
| `.claude/agents/**` | Agents stay read-only (no Edit or Write tools) and return their result as their final message |
| `.agents/skills/**`, `scripts/hooks/**` | Skills have valid frontmatter (name equals directory, description present); vendored skills (`VENDORED.md`) are not edited; hooks emit both the Cursor and the Claude Code output shape |
| `.obsidian/**`, `uv.lock` | Skip, unless a secret or a machine path appears |

## Severity

Use exactly these labels:

- **Blocker:** leaks a secret, breaks CI or a release, or makes a result wrong (an estimator no
  longer satisfies efficiency or no longer recovers a known-truth game, and its test was loosened).
- **High:** a likely failure in a supported workflow (`make check`, `make docs`, explain from the
  CLI or a notebook, publishing), or a wrong or misleading public name.
- **Medium:** brittle or misleading; it will bite the next change.
- **Low:** polish.

## Process rules

- **PR title:** `type: subject` or `type(scope): subject` with `type` in
  `feat|fix|docs|test|refactor|chore|ci|build|perf`; no `WIP` (`AGENTS.md`).
- **Assignee and label:** the PR is assigned to its author and labelled from the title type
  (`enhancement`, `bug` or `documentation`).
- **Never draft:** flag a draft PR or a `WIP` title or label.
- **Commits:** subjects follow the same `type: summary` rule, no trailing period, at most 72
  characters. Flag AI attribution lines (`Co-Authored-By: Claude`, `Generated with`) in
  commits, PR bodies or files.
