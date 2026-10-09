# Handoff

## State
Branch chore/audit-remediation holds every remediation phase and fix wave, all
uncommitted (2026-10-08). `make check`, `make docs` and `make notebooks` are green (11
notebooks with stored outputs). The plan files were deleted on 2026-10-08; the commit
plan below is the remaining record.

## Commit
One PR (maintainer decision 2026-10-09): themed commits of the final tree on
chore/audit-remediation; the historical COMMIT-PLAN.md could not be replayed.
Unreleased was folded into 0.1.18 (2026-10-09); 0.1.17 keeps its section but is
never tagged. Tag v0.1.18 after merge (skill release "After merge").

## User-only
- Approve the weekly literature watch task (.cursor/commands/lit-watch.md).
- Branch protection and repository settings (gh api commands in
  .cursor/rules/ci.mdc), after the title workflow is on main.
- Pages source: GitHub Actions, before the first docs deployment.
- Open the vault once in Obsidian: Code Graph colours and the Dashboard Dataview
  tables.

## Next
Practical layer: `make practical FEATURE=all` (Qwen2.5-1.5B fp16 on MPS, 27 scenarios,
~70 s warm); D04 open (SmolLM2 fp16 KV-cache drift on MPS). Upgrades: `/upgrade`
weekly schedule not yet created (maintainer decision). Releases: `/release` after merge.
Skill `ship` (delivery autopilot) is wired but unexercised: no remote `origin`
yet, no gh command run. First real use: enable Copilot code review on the repo,
add the remote, then `/ship` on a routine branch and watch merge_gate.sh.
No open vault issues; D02 (audio backend) and D03 (PR 4 rebase) stay deferred to
the audio track. The transformers-cpu CI job is non-gating until seen green once.
