---
name: upgrade
description: >-
  Use when asked to upgrade or update dependencies, prek hooks, workflow action pins or uv,
  on /upgrade, or for the weekly upgrade run.
---

# Upgrade: dependencies and pins on a routine branch

`scripts/upgrade.py` checks and applies; skill `ship` delivers. Invoking `/upgrade` (or
its scheduled run) is the request to commit, push and ship the upgrade branch.

| Target | Upgrades |
|--------|----------|
| `make upgrade-check` | lists outdated items, one line each; exit 1 when any |
| `make upgrade` | packages, `prek update`, action pins, uv (`required-version` with the workflows), pinned Obsidian plugins |

Protected items (`pin` and `protected-workflows` in `[tool.agent-kit.upgrade]`) are
reported as `(protected: report only)`, never edited. List them in the PR body for the
maintainer; how they are handled here: `local.md` in this folder.

## Procedure

1. `git switch main && git pull --ff-only && git switch -c chore/upgrades-<YYYY-MM-DD>`.
2. `make upgrade-check`; nothing outdated: stop and say so. Else `make upgrade`.
3. `make check` and the extra checks of `local.md`. A failure caused by one package: pin it
   back in `pyproject.toml` with a one-line reason, for example
   `"numpy>=2.0.0,<2.6.0",  # 2.6 breaks X; revisit monthly`, then `uv lock`, re-run the
   checks and name the pin in the PR body. Never add or remove a package here: that makes
   the branch a review PR.
4. One commit, `build: upgrade dependencies and workflow pins`; when only workflows
   changed, `ci: upgrade workflow action pins` instead.
5. Housekeeping (`local.md`) as its own commit, `chore: prune <what>`; skip it when
   nothing is stale.
6. Run skill `ship`; `scripts/pr_policy.py` classifies it routine (`build:`/`ci:` commits,
   lock, hook revisions, pins, same package set) and it merges after CI.

## Weekly cadence

A **local** scheduled task of the Claude desktop app (cron `0 8 * * 1`, working directory
this repository; its id and prompt are in `local.md`). Creating it changes standing
configuration, so only on the maintainer's request. Between runs, skill `ship` step 1 runs
`make upgrade-check` and reports without applying.

## Never

- Upgrade on a feature branch that carries other work.
- `pip install` or `uv add` outside `pyproject.toml` plus `uv lock`.
- Edit a protected workflow or move the pin on the routine branch (`local.md`).
