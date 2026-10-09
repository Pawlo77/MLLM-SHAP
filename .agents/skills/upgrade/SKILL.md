---
name: upgrade
description: >-
  Use when asked to upgrade or update dependencies, prek hooks, workflow action pins, uv or
  the Obsidian plugins, on /upgrade, or for the weekly upgrade run.
---

# Upgrade: dependencies and pins on a routine branch

`scripts/upgrade.py` checks and applies; skill `ship` delivers. Invoking `/upgrade` (or
its scheduled run) is the request to commit, push and ship the upgrade branch.

| Target | Upgrades |
|--------|----------|
| `make upgrade-check` | lists outdated items, one line each; exit 1 when any |
| `make upgrade` | `uv lock --upgrade`, `uv sync --all-groups --all-extras`, `prek update`, action pins, uv (`required-version` with the workflows), Obsidian manifests plus `install_obsidian_plugins.py --force` |

`publish.yaml` is protected: its pins, and the uv version while it pins one, are reported
as `(protected: report only)`, never edited. List them in the PR body for the maintainer.

## Procedure

1. `git switch main && git pull --ff-only && git switch -c chore/upgrades-<YYYY-MM-DD>`.
2. `make upgrade-check`; nothing outdated: stop and say so. Else `make upgrade`.
3. `make check` and `make docs`. A failure caused by one package: pin it back in
   `pyproject.toml` with a one-line reason, for example
   `"numpy>=2.0.0,<2.6.0",  # 2.6 breaks the masking dtype; revisit monthly`, then
   `uv lock`, re-run the checks and name the pin in the PR body. Never add or remove a
   package here: that makes the branch a review PR.
4. `make notebooks` when a runtime dependency (`[project] dependencies` or an extra)
   changed version in `uv.lock`.
5. One commit, `build: upgrade dependencies and workflow pins`; when only workflows
   changed, `ci: upgrade workflow action pins` instead.
6. Run skill `ship`; `scripts/pr_policy.py` classifies it routine (`build:`/`ci:` commits,
   lock, hook revisions, pins, same package set) and it merges after Copilot and CI.

## Weekly cadence

The maintainer schedules it once: a **local** scheduled task of the Claude desktop app
(id `nlp-shap-upgrade`, cron `0 8 * * 1`, working directory this repository, prompt
`Run /upgrade in the nlp-shap repository.`). In Claude Code ask: "Create a local scheduled
task nlp-shap-upgrade, cron 0 8 * * 1, in this repository, prompt: Run /upgrade in the
nlp-shap repository." Creating it changes standing configuration, so only on the
maintainer's request.

Between runs, skill `ship` step 1 runs `make upgrade-check` and reports outdated items
without applying them.

## Never

- Edit `.github/workflows/publish.yaml` or bump `required-version` above its uv pin.
- Upgrade on a feature branch that carries other work.
- `pip install` or `uv add` outside `pyproject.toml` plus `uv lock`.
