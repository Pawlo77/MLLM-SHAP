# Plans agents

- A plan is written before multi-file work (skill `vibe`, step 2) and executed task by task; rules: `.cursor/rules/plans.mdc`.
- File `plans/YYYY-MM-DD-<slug>.md`: the day the plan was written, a lowercase hyphenated slug; supporting audits and research use the same pattern.
- Start from `knowledge/_templates/plan.md`: frontmatter (`title`, `type: plan`, `status`, `created`, `updated`, `tags`) and the writing-plans header (`**Goal:**`, `**Architecture:**`, `**Tech Stack:**`).
- `status` (also the status line under the title): `draft`, `in progress`, `done` or `superseded by <file>`.
- Tick a step (`- [ ]`) only when it is done and `make check` passed; update the plan in the pull request that executes it.
- Never delete a plan: mark it `done` or `superseded`.
- `make vault-check` validates plan frontmatter and headers; the Dashboard lists plans by status.
