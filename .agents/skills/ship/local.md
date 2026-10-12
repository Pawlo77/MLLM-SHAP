# Ship: MLLM-SHAP steps

Before the PR (after `make check`):

- `make docs` and `make notebooks` when they apply (skill `check`).
- `uv run python scripts/practical_features.py --base main`: when it prints ids, run
  `/practical <ids>` (quick tier); red blocks; the summary goes in the body.
- After the commits: `uv run python scripts/check_vault.py --strict --strict-tags` (it sees
  stale notes only after the commit).
- `uv run python scripts/pr_policy.py --bump`: when not `none`, run skill `release`
  "Before merge" now (the PR carries the bump).

After merge: with a bump, skill `release` "After merge" tags it. `/ship --after-merge` (or
`/release` on `main`) runs only that step for a PR the maintainer merged.

Ready for your review, with a bump: add `; after you merge, run /release to tag vX.Y.Z`
to the final line.
