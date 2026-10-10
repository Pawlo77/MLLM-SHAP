---
name: ship
description: >-
  Use when asked to ship, deliver, open the pull request or merge a finished change, or
  on /ship.
---

# Ship: commit, pull request, Copilot review, merge

Delivery autopilot for a finished branch. Routine changes merge themselves; others stop
before the merge. Prerequisites: Copilot
code review enabled on the repository; `gh auth status` green; a remote `origin`.

## 1. Classify

- `make upgrade-check`: report outdated items; skill `upgrade` applies them.
- Exit criteria green (`make check`; `make docs` and `make notebooks` when they apply).
- `uv run python scripts/practical_features.py --base main`: when it prints ids, run
  `/practical <ids>` (quick tier); red blocks the PR; its summary goes in the body.
- Commits per rule `git-commits` (granular, `type: summary`, no AI attribution).
- `uv run python scripts/pr_policy.py --bump`: when not `none`, run skill `release`
  "Before merge" now, so the PR carries the version bump.
- `uv run python scripts/pr_policy.py --write-marker --session <key>` (key from the
  session-start banner). State the kind and its reasons. Routine: only
  `chore|ci|build|docs|test|style` commits, nothing under `src/`, `[project]` changes
  limited to dependency specifiers (`build`/`chore`/`ci`) and the release version, at
  most 400 lines and 25 files without `uv.lock`, no guard, settings or publish files.
  `scripts/hooks/merge_gate.sh` checks the marker before `gh pr merge`; re-run it after
  every commit.

## 2. Open

```bash
git push -u origin HEAD
uv run python scripts/strip_watermarks.py <body>
gh pr create --title "<type>: <summary>" --body-file <body> --reviewer @copilot
gh pr edit <n> --add-assignee Pawlo77 --add-label <enhancement|bug|documentation>
uv run python scripts/pr_policy.py --pr <n> --session <key>
```

Body from agent `pr-writer`: it names the kind (routine, merged after Copilot and green
CI, or "Needs maintainer review") and carries no AI attribution (`pr-title.yaml`).

## 3. Review loop (at most 5 rounds)

1. Wait for Copilot (poll every 60 s, 15 min at most):
   `gh pr view <n> --json reviews --jq '[.reviews[] | select(.author.login | startswith("copilot"))] | length'`;
   and for CI: `gh pr checks <n> --watch --fail-level error`.
2. Unresolved threads: `gh api graphql -f query='query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){pullRequest(number:$n){reviewThreads(first:100){nodes{id isResolved path line comments(first:10){nodes{id author{login} body}}}}}}}' -F o=Pawlo77 -F r=nlp-shap -F n=<n>`.
3. Per thread: fix with skill `development` (tests first) or decide not to; `make check`;
   commit (`fix:`/`docs:`/`chore:` as fits); re-run `pr_policy.py --write-marker --pr <n>`
   and stop with a message if the kind changed to review; push; reply
   (`gh api -X POST repos/Pawlo77/nlp-shap/pulls/<n>/comments/<id>/replies -f body=…`);
   resolve (`resolveReviewThread` mutation); re-request
   (`gh api -X POST repos/Pawlo77/nlp-shap/pulls/<n>/requested_reviewers -f 'reviewers[]=copilot-pull-request-reviewer[bot]'`).
   A declined thread gets the reason, stays unresolved and forces the review outcome.
4. Success: no unresolved Copilot thread and CI green; a Copilot review that only
   reports its quota limit counts as done. Red CI: fix in a follow-up commit (skill `ci`).

## 4. Outcome

- Routine and success: `gh pr merge <n> --auto --merge --delete-branch` (GitHub merges
  after the required checks); watch the `main` workflows (`gh run watch <id>
  --exit-status`); with a bump, skill `release` "After merge" tags it; skill `vibe`
  Compound step when something non-obvious was learned;
  final line `Merged <url>`, plus `(no Copilot review: quota)` when it was skipped.
- Review, round limit or a declined thread: stop; final line `Ready for your review:
  <url>` (with a bump: `; after you merge, run /release to tag vX.Y.Z`) plus one line
  per open point. Never merge a review PR.
- `/ship --after-merge` (or `/release` on `main`): skill `release` "After merge" for a
  PR the maintainer merged.

Never: force push, push to `main`, draft or WIP PRs, squash or rebase merges, `--admin`,
merging with red CI or unresolved Copilot threads.
