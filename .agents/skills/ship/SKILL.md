---
name: ship
description: >-
  Use when asked to ship, deliver, open the pull request or merge a finished change, or
  on /ship.
---

# Ship: commit, pull request, Copilot review, merge

Delivery autopilot for a finished branch: routine changes merge themselves, others stop
<<<<<<< before updating
before the merge. Needs `gh auth status` green and Copilot review.

## 1. Classify

- `make upgrade-check`: report outdated items.
- `git fetch origin`; `main` must be in the branch (`git merge-base --is-ancestor
  origin/main HEAD`), else `git merge origin/main` (never rebase), resolve, rerun checks.
- Exit criteria green (`make check`; `make docs`, `make notebooks` when they apply).
- `uv run python scripts/practical_features.py --base main`: when it prints ids, run
  `/practical <ids>` (quick tier); red blocks; the summary goes in the body.
- Commits per rule `git-commits`; then `uv run python scripts/check_vault.py --strict
  --strict-tags` (it sees stale notes only after the commit).
- `uv run python scripts/pr_policy.py --bump`: when not `none`, run skill `release`
  "Before merge" now (the PR carries the bump).
- `uv run python scripts/pr_policy.py --write-marker --session <key>` (key from the
  session-start banner); state the kind and why (routine rules: the docstring of
  `scripts/pr_policy.py`). `scripts/hooks/merge_gate.sh` checks the marker before a
  merge; re-run after every commit.
=======
before the merge. Needs `gh auth status` green and a remote `origin`. Steps specific to
this repository: `local.md` in this folder, when it exists.

## 1. Classify

- `git fetch origin`; `main` must be in the branch (`git merge-base --is-ancestor
  origin/main HEAD`), else `git merge origin/main` (never rebase), resolve, rerun checks.
- `make upgrade-check`: report outdated items (skill `upgrade` applies them).
- `make check` green, plus the pre-PR steps of `local.md`.
- Commits per rule `git-commits` (granular, `type: summary`, no AI attribution).
- `uv run python scripts/pr_policy.py --write-marker --session <key>` (key from the
  session-start banner); state the kind and why (rules: the docstring of
  `scripts/pr_policy.py`, settings `[tool.agent-kit.pr-policy]`). `scripts/hooks/merge_gate.sh`
  checks the marker before a merge; re-run after every commit.
>>>>>>> after updating

## 2. Open

```bash
git push -u origin HEAD
uv run python scripts/strip_watermarks.py <body>
gh pr create --title "<type>: <summary>" --body-file <body> --reviewer @copilot
<<<<<<< before updating
gh pr edit <n> --add-assignee Pawlo77 --add-label <enhancement|bug|documentation>
uv run python scripts/pr_policy.py --pr <n> --session <key>
```

Body from agent `pr-writer`: it names the kind (routine, or "Needs maintainer review")
and carries no AI attribution (`pr-title.yaml`).

## 3. Review loop (at most 5 rounds)

1. Wait for Copilot (poll every 60 s, 15 min at most):
   `gh pr view <n> --json reviews --jq '[.reviews[] | select(.author.login | startswith("copilot"))] | length'`;
   and watch CI to green on every PR after every push: `gh pr checks <n> --watch
   --fail-level error`. Fix a red check (skill `ci`) in a commit pushed to the same PR.
   Before the final line `gh pr view <n> --json mergeable` must say `MERGEABLE` and
   `origin/main` be in the branch; if `main` moved, merge it, push, watch again.
=======
gh pr edit <n> --add-assignee @me --add-label <enhancement|bug|documentation>
uv run python scripts/pr_policy.py --pr <n> --session <key>
```

Body from agent `pr-writer`: it names the kind (routine, or "Needs maintainer review") and
carries no AI attribution (`pr-title.yaml`). When `--reviewer @copilot` is refused (review
not enabled), create without it and say so.

## 3. Review loop (at most 5 rounds)

Green gate first: skill `ci`, then `scripts/pr_follow.py <n>` after every push.

1. Wait for Copilot (poll every 60 s, 15 min at most):
   `gh pr view <n> --json reviews --jq '[.reviews[] | select(.author.login | startswith("copilot"))] | length'`;
   and for CI: `gh pr checks <n> --watch --fail-level error`. Fix a red check (skill `ci`)
   in a commit pushed to the same PR.
>>>>>>> after updating
2. Unresolved threads, replies, resolving and re-requesting the review: the commands in
   [reference.md](reference.md).
3. Per thread: fix with skill `development` (tests first) or decide not to; `make check`;
   commit (`fix:`/`docs:`/`chore:` as fits); re-run `pr_policy.py --write-marker --pr <n>`
   and stop if the kind became review; push; reply; resolve; re-request the review.
   A declined thread gets the reason, stays unresolved and forces the review outcome.
<<<<<<< before updating
4. Success: no unresolved Copilot thread and CI green; a Copilot review that only
   reports its quota limit counts as done.
=======
4. Success: no unresolved Copilot thread, CI green, `gh pr view <n> --json mergeable` says
   `MERGEABLE` and `origin/main` is in the branch (if `main` moved, merge it, push, watch
   again). A Copilot review that only reports its quota limit, or none at all, counts as done.
>>>>>>> after updating

## 4. Outcome

- Routine and success: `gh pr merge <n> --auto --merge --delete-branch` (GitHub merges
  after the required checks); watch the `main` workflows (`gh run watch <id>
<<<<<<< before updating
  --exit-status`); with a bump, skill `release` "After merge" tags it; skill `vibe`
  Compound step for a non-obvious lesson; final line `Merged <url>` (plus `(no Copilot
  review: quota)` when skipped).
- Review, round limit or a declined thread: stop; final line `Ready for your review:
  <url>` (with a bump: `; after you merge, run /release to tag vX.Y.Z`) plus one line
  per open point. Never merge a review PR.
- `/ship --after-merge` (or `/release` on `main`): skill `release` "After merge".
=======
  --exit-status`); the "After merge" steps of `local.md`; skill `vibe` Compound step for a
  non-obvious lesson; final line `Merged <url>` (plus `(no Copilot review: quota)` or `(no
  Copilot review)` when skipped).
- Review, round limit or a declined thread: stop; final line `Ready for your review:
  <url>` plus one line per open point. Never merge a review PR.
>>>>>>> after updating

Never: force push, push to `main`, draft or WIP PRs, squash or rebase merges, `--admin`,
merging or reporting ready with red or pending CI, unresolved Copilot threads, or `main`
not merged into the branch.
