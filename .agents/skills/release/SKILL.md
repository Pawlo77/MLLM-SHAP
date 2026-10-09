---
name: release
description: >-
  Use when asked to release, publish, bump the version or tag vX.Y.Z, on /release, or
  when skill ship finds a version bump.
---

# nlp-shap release

The agent picks the bump and tags after the merge; the `Publish` workflow
(`.github/workflows/publish.yaml`, tags `v*.*.*`) uploads to PyPI by trusted publishing
and creates the GitHub Release. Invoking skill `ship` or `/release` is the request to
commit, tag and push the tag.

## Decide

`uv run python scripts/pr_policy.py --bump` prints `<bump> <next version>` from the
commits since the last `v*` tag (never below the `pyproject.toml` version):

| Commits since the tag | Bump |
|-----------------------|------|
| `type!:` or `BREAKING CHANGE` in a body | major (minor while below 1.0) |
| `feat` | minor |
| `fix`, `perf` | patch |
| anything else | none: no release |

## Before merge (on the branch being shipped)

Skip a step that is already done (the version already reads `X.Y.Z`).

1. Set `version` in `pyproject.toml` to `X.Y.Z`; `uv lock`.
2. `docs/release_notes.rst`: move the `Unreleased` bullets under a new
   `X.Y.Z (YYYY-MM-DD)` heading and leave an empty `Unreleased` heading above it.
3. `CITATION.cff`: `version` and `date-released`.
4. `make notebooks` when a public workflow changed (examples print the version); then
   `make docs`, `make check` and `make practical TIER=deep` (skill `practical`; red
   blocks the release, its summary line goes into the PR body).
5. Commit `chore: release vX.Y.Z`. `scripts/pr_policy.py` treats the version change as
   routine when it equals that subject's version.

## After merge (on `main`)

Skill `ship` runs this after a routine merge. For a review PR the maintainer merges, then
runs `/release` on `main` (or `/ship --after-merge`).

1. `git switch main && git pull --ff-only`; `pyproject.toml` reads `X.Y.Z` and
   `git tag -l vX.Y.Z` is empty.
2. Tag and push:

   ```bash
   git tag -a vX.Y.Z -m "nlp-shap vX.Y.Z"
   git push origin vX.Y.Z
   ```

3. `gh run list --workflow Publish --limit 3`, then `gh run watch <run-id> --exit-status`.
4. `pip index versions nlp-shap` lists `X.Y.Z` (PyPI can lag a minute; retry first) and
   `gh release view vX.Y.Z` shows the notes and artifacts.
5. On failure fix forward: a new branch with the fix, shipped and released as the next
   patch. Never delete, move or reuse a tag.
6. Write the `.remember/remember.md` handoff (`/remember`).

## Never

- Tag a commit that is not on `main`, or reuse a version (PyPI rejects re-uploads).
- Publish from a local machine (`uv publish`); the workflow is the only publisher.
- Edit `publish.yaml` without the maintainer.
