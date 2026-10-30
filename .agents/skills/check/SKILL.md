---
name: check
description: >-
  Use when asked to check, verify or run make check, and before hand-off.
---

# Check

## Default validate

```bash
make check
```

`make check` runs the tests, every prek hook and the vault check. What each part covers
here, and what else to run for the files you changed: `local.md` in this folder, when it exists.

## After push

Run `uv run python scripts/pr_follow.py` until it exits 0 (rule `git-commits`, skill `ci`).

## Stop

Do not commit, push or open a PR unless the user asked (`git-commits.mdc`).
