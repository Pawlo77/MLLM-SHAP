---
paths:
  - "plans/**"
---

<!-- Generated from .cursor/rules/plans.mdc by scripts/sync_rules.py; do not edit. -->

# Plans

- Naming, header and status values: `plans/AGENTS.md`.
- A plan task is done only when its checkbox is ticked and `make check` passed after the change.
- Update the plan in the same pull request that executes it: tick boxes, bump `updated`, change
  the status line (`draft | in progress | done | superseded by <file>`).
- Never delete a plan; mark it `superseded by <file>` or `done`.
- Do not rewrite finished tasks; add a dated note under the task when reality diverged.
