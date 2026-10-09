# AGENTS.md

## Exit criteria

```bash
make check
```

`make check` runs `agent-check` (filtered pytest and `prek-all`), then `vault-check` (vault
notes, links, traceability tags, generated notes, brevity budgets). Also run:

- `make docs` (strict Sphinx) when `docs/` or the public API change.
- `make notebooks` when `examples/*.ipynb` change, after a `src/` change to a public workflow
  and before every release PR.
- `make knowledge` after docstring or vault edits (prek runs it too).

Optional, not part of `make check`: `make tests-all`, `make bench`, `make practical` (skill `practical`).

## Agent-only

- Chat in caveman ultra (rule `.cursor/rules/caveman.mdc`; Codex reads that file once per
  session). Files, code, commits, PR bodies and subagent reports stay normal prose.
- Delivery goes through skill `ship` after `make check`: it commits, pushes, opens the PR,
  gets Copilot's review and addresses it; routine changes (`scripts/pr_policy.py`) merge
  by themselves, everything else stops before the merge with the PR link. Outside `ship` and `release`,
  no commit, push, tag or PR; never draft or WIP PRs, force pushes, amends or pushes to
  `main` (rule `git-commits`; enforced by each tool's permission settings and prek).
- Upgrades and releases go through skills `upgrade`, `release` and `ship`.
- Claude reviews its own work before a PR (skill `review`, agent `reviewer`) and answers
  Copilot's threads on its own PR; it never reviews other PRs on GitHub.
- Subagents default to Sonnet, Haiku for mechanical work; pass `model: opus` or `fable`
  when depth pays: estimator math, hard bugs, security or design review, a failed Sonnet run.
- No new markdown or doc expansion unless it is requested or crucial new information.
- No runtime dependency without a justification in `pyproject.toml`.
- Use `prek`, never the `pre-commit` CLI; change dependencies in `pyproject.toml` plus
  `uv lock`, never `pip install`.
- Write the fewest lines that keep the meaning: one home per fact, link instead of
  restating, budgets in rule `brevity` (checked by `scripts/check_brevity.py`).
- Never edit generated files: `.claude/rules/` (edit `.cursor/rules/*.mdc`, then
  `uv run python scripts/sync_rules.py`), `.codex/agents/` (`scripts/sync_agents.py`),
  `knowledge/generated/` and `.claude/jit-context/*/10-auto/` (`make knowledge`).

Keep this file general; detail lives in skills (`.agents/skills/`) and rules
(`.cursor/rules/*.mdc`). Tool matrix, plugins, sibling clones and Codex setup:
`CONTRIBUTING.md`, section "Agent setup".

## Siblings

| Sibling path (clone next to `nlp-shap/`) | Use |
|------|-----|
| `../MLLM-Shap/mllm_shap/src/mllm_shap/` | legacy code to port (skill `port-legacy`) |

## Memory and context

- `.remember/remember.md` is the handoff: read it first; write it with `/remember` before
  ending a session. `.remember/identity.md` says how to work here.
- Before a `src/` change read the package's `knowledge/architecture/<pkg>.md` (Claude Code
  injects its excerpt through `.claude/jit-context/`).

## Vault

- The repository root is an Obsidian vault; notes live in `knowledge/` only (entry:
  `knowledge/nlp-shap.md`).
- A change under `src/nlp_shap/<pkg>/` updates `knowledge/architecture/<pkg>.md` in the same
  change.
- A design choice appends a section to `knowledge/decisions/<Domain> decisions.md`; a bug
  found or fixed updates `knowledge/issues/`.
- Write notes only through skill `notes`.
- `make vault-check` (part of `make check`) fails on broken links, missing frontmatter and
  stale generated notes.

## Load order

1. This file and the nested `AGENTS.md` of the directory you work in; the exit criteria above.
2. Caveman ultra for chat (always-apply rules; Claude Code: `.claude/rules/always.md`).
3. Skills are chosen by their descriptions; scoped rules load by file path.
