<!-- Generated from .cursor/rules/brevity.mdc by scripts/sync_rules.py; do not edit. -->

# Brevity

Every line is read again by people and agents. Write the fewest lines that keep the meaning.

- Code: comments say why, never restate the next line. One-line docstrings; add `Args`/`Returns` only for a parameter that name and type do not explain. Functions ≤ 50 statements; module docstring ≤ 12 lines.
- Reuse before writing: a block that appears twice becomes a helper or a fixture.
- One home per fact. Link (`[[note]]`, `:doc:`, path) instead of restating it in another file.
- Reference material as tables or lists, not paragraphs. No "Overview", "Introduction" or "Summary" sections. No sentence over 35 words.
- Delete obsolete code and text; do not deprecate, archive or keep it "for history" (git has it).
- Budgets (`scripts/check_brevity.py`, run by `make vault-check`): README ≤ 900 words; docs page ≤ 1200; release-note bullet ≤ 60; issue note ≤ 350 (`Fixed issues` table ≤ 800); decision log ≤ 700 (≤ 120 per decision); architecture/handbook note ≤ 700; skill ≤ 100 lines and 500 words; rule ≤ 60 lines and 400 words; agent ≤ 40 lines; nested AGENTS.md ≤ 60 lines.
- Notebooks: markdown cell ≤ 150 words, code cell ≤ 40 lines, ≤ 900 markdown words per notebook; read a notebook with `jupyter nbconvert --to script --stdout`, never the raw file.
- A file at or over budget must not grow: cut or split before adding.
- Reports and PR bodies: findings first; no recap of the task or of code you only read.

<!-- Generated from .cursor/rules/caveman.mdc by scripts/sync_rules.py; do not edit. -->

# Caveman ultra (chat default)

Chat replies only:

- Drop articles, filler, pleasantries and hedging; fragments are fine.
- One idea per sentence.
- Exact technical terms; numbers, paths, commands and error text verbatim.
- No invented abbreviations; no arrows or symbols in place of words.
- Write plainly for security warnings and irreversible actions, then resume.
- Anything written to a file (code, comments, commits, PR bodies, notes, docs, plans, memory) and every subagent report stays normal prose.
- Off on "stop caveman" or "normal mode". Skill `caveman` is only for switching level with `/caveman <level>`.

<!-- Generated from .cursor/rules/git-commits.mdc by scripts/sync_rules.py; do not edit. -->

# Git commits and pull requests

- Commits, pushes and PRs happen inside skill `ship` (it classifies the branch and merges routine changes itself); outside it, never without the user's explicit permission.
- One logical change per commit; when a change spans unrelated concerns, propose a numbered commit plan.
- Message: `type: summary` (`feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`, `build`, `perf`), imperative, no period, ≤ 72 characters; an optional body says what changed and why.
- Before committing: the exit criteria in `AGENTS.md`, then `git status`, `git diff` and `git log -5`; stage only that commit's files.
- No AI attribution (`Co-Authored-By: Claude`, `Generated with …`) anywhere; `scripts/strip_watermarks.py` and `pr-title.yaml` enforce it.
- Never `--no-verify`, amend or force-push; push a feature branch, never `main`; never draft or WIP PRs (the shell guard denies them).
- After pushing, run `uv run python scripts/pr_follow.py` until it exits 0 (desktop app: `ccd_pr` `get_status` while checks run). It lists red checks, a stale branch, open threads and every comment (Codecov, Copilot): fix red checks in new commits, merge `origin/main` when behind, fix or answer each comment; never hand off on red.
- PRs open ready for review, titled in commit style, with the body of `.github/pull_request_template.md`.
- After `gh pr create`: `gh pr edit <n> --add-assignee Pawlo77 --add-label <label>`, with `bug` for `fix`, `documentation` for `docs` and `enhancement` otherwise.

<!-- Generated from .cursor/rules/minimal-diff.mdc by scripts/sync_rules.py; do not edit. -->

# Minimal diff

- Read the surrounding code first; match its naming, types and patterns.
- Solve the requested task only; no drive-by refactors.
- Reuse existing functions before adding abstractions; no one-line helpers; no handling for impossible cases.
- New behaviour needs tests first (`tdd.mdc`); no tests of the obvious.
- No new markdown files unless requested.
- If two approaches work, pick the simpler one that matches the repository.
- When porting from another repository, bring over only the slice the task needs.

<!-- Generated from .cursor/rules/package-boundaries.mdc by scripts/sync_rules.py; do not edit. -->

# Package boundaries

- `src/mllm_shap/` is the library; experiments, papers and GUIs live elsewhere.
- Inside `src/mllm_shap/` imports are relative; tests and examples use `from mllm_shap...`.
- User workflows are notebooks in `examples/` (`examples.mdc`).
- Add a module only with a real implementation; heavy stacks (torch, transformers) are optional extras.

<!-- Generated from .cursor/rules/uv-workflow.mdc by scripts/sync_rules.py; do not edit. -->

# uv workflow

- Install: `make install` (`uv sync --all-groups --all-extras`); run tools with `uv run`; build with `make build`.
- Dependencies: edit `pyproject.toml`, then `uv lock`; commit `uv.lock` with them. Never `pip install`.
- Hooks: `prek`, never the `pre-commit` CLI; `make prek-all` matches the remote Prek Checks.
- Python 3.12 (`.python-version`).
