---
name: ship
description: >-
  Use when asked to ship, deliver, open the pull request or merge a finished change, or
  on /ship.
---

# Ship: commit, pull request, follow-up, merge

Delivery autopilot for a finished branch: routine changes merge themselves, others stop
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

## 2. Open

```bash
git push -u origin HEAD
uv run python scripts/strip_watermarks.py <body>
gh pr create --title "<type>: <summary>" --body-file <body>
gh pr edit <n> --add-assignee @me --add-label <enhancement|bug|documentation>
uv run python scripts/pr_policy.py --pr <n> --session <key>
```

Body from agent `pr-writer`: it names the kind (routine, or "Needs maintainer review") and
carries no AI attribution (`pr-title.yaml`). Copilot review is off by default. A repository
that wants it sets `copilot = true` in `[tool.agent-kit.ship]` of `pyproject.toml`: add
`--reviewer @copilot` to `gh pr create` (create without it when refused and say so) and
follow the "With Copilot on" notes below.

## 3. Follow-up loop (at most 5 rounds)

Green gate first: skill `ci`, then `scripts/pr_follow.py <n>` after every push.

1. Wait for CI: `gh pr checks <n> --watch --fail-level error`. Fix a red check (skill `ci`)
   in a commit pushed to the same PR. With Copilot on, also wait for its review (poll every
   60 s, 15 min at most):
   `gh pr view <n> --json reviews --jq '[.reviews[] | select(.author.login | startswith("copilot"))] | length'`.
2. Unresolved threads of any reviewer, replies, resolving and re-requesting: the commands in
   [reference.md](reference.md).
3. Per thread: fix with skill `development` (tests first) or decide not to; `make check`;
   commit (`fix:`/`docs:`/`chore:` as fits); re-run `pr_policy.py --write-marker --pr <n>`
   and stop if the kind became review; push; reply; resolve (with Copilot on, re-request its
   review). A declined thread gets the reason, stays unresolved and forces the review outcome.
4. Success: no unresolved thread, CI green, `gh pr view <n> --json mergeable` says
   `MERGEABLE` and `origin/main` is in the branch (if `main` moved, merge it, push, watch
   again). With Copilot on, a review that only reports its quota limit, or none at all,
   counts as done.

## 4. Outcome

- Routine and success: `gh pr merge <n> --auto --merge --delete-branch` (GitHub merges
  after the required checks); watch the `main` workflows (`gh run watch <id>
  --exit-status`); the "After merge" steps of `local.md`; final line `Merged <url>` (with
  Copilot on, plus `(no Copilot review)` when it was skipped).
- Review, round limit or a declined thread: stop; final line `Ready for your review:
  <url>` plus one line per open point. Never merge a review PR.

Never: force push, push to `main`, draft or WIP PRs, squash or rebase merges, `--admin`,
merging or reporting ready with red or pending CI, unresolved threads, or `main`
not merged into the branch.
