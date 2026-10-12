# Upgrade: MLLM-SHAP

`make upgrade` runs `uv lock --upgrade`, `uv sync --all-groups --all-extras` and, for the
Obsidian manifests, `install_obsidian_plugins.py --force`.

`publish.yaml` is protected: its pins, and the uv version while it pins one, are reported
as `(protected: report only)`. List them in the PR body for the maintainer.

Extra checks: `make docs` after `make check`; `make notebooks` when a runtime dependency
(`[project] dependencies` or an extra) changed version in `uv.lock`.

## Housekeeping

Commit message `chore: prune completed handoff and stale notes`. Remove what finished work
left behind; git keeps history. Touch only the rows below.

| Target | Prune |
|--------|-------|
| `.remember/remember.md` | drop done items and merged-PR detail; compress `State` to current version, CI and schedules; keep `User-only` and open `Next` |
| `knowledge/issues/` | a fixed issue leaves its own note for one `Fixed issues` row; merge rows older than the last release into one line; keep it ≤ 800 words |
| `plans/`, `runs/` | delete finished plans and stale run output; keep `plans/AGENTS.md` |
| merged branches | `git branch --merged main` minus `main`: `git branch -d` each; no `-D`, no remote deletes |

Check each claim against `git log` and `gh pr list` first; when unsure, keep it and list it
in the PR body. Notes change only through skill `notes`; `make check` must still pass.

## Weekly cadence

Task id `mllm-shap-upgrade`, cron `0 8 * * 1`, working directory this repository, prompt
`Run /upgrade in the mllm-shap repository.` In Claude Code ask: "Create a local scheduled
task mllm-shap-upgrade, cron 0 8 * * 1, in this repository, prompt: Run /upgrade in the
mllm-shap repository."

## Never

- Edit `.github/workflows/publish.yaml` or bump `required-version` above its uv pin.
