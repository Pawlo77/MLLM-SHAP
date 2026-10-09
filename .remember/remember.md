# Handoff

## State
v0.1.18 released 2026-10-09: PR 5 (audit remediation, 23 commits) merged as
74f5a6e, tag pushed, Publish green, on PyPI and GitHub Releases. 0.1.17 has a
release-notes section but no tag. `main` CI green; Pages deploys from GitHub Actions
(https://pawlo77.github.io/nlp-shap/). CodeQL is skipped while the repository is private (code scanning needs paid Code
Security).

## User-only
- Copilot code review: quota was exhausted on PR 5, so skill `ship` had no review.
- Approve the weekly literature watch task (.cursor/commands/lit-watch.md).
- Branch protection and repository settings (.agents/skills/ci/reference.md).
- Open the vault once in Obsidian: Code Graph colours and the Dashboard Dataview
  tables.

## Next
Subagents default to Sonnet (`CLAUDE_CODE_SUBAGENT_MODEL`); AGENTS.md says when to
pass Opus or Fable. Practical: `make practical TIER=all FEATURE=all` passed (34, 3
skipped: LM Studio key unset, no audio backend; 72 min with downloads). D02 audio
backend deferred. `/upgrade`
weekly schedule not yet created; `make upgrade-check` lists pending bumps.
