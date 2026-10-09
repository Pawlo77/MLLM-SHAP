---
title: "Solutions"
type: handbook
domain: process
tags: [handbook, solution]
created: 2026-10-07
updated: 2026-10-07
---

# Solutions

Part of [[nlp-shap]]. One note per solved problem, written by step 9 ("Compound") of skill `vibe` after its pull request (PR) merges; ported from the compound step of the compound-engineering plugin.

## Rules

- **Path:** `knowledge/handbook/Solutions/<YYYY-MM-DD> - <title>.md`, `type: solution`, from template `_templates/solution.md` (skill `notes`).
- **Content:** the problem, what was tried and why it failed, what worked, and one reusable rule; `related-code` lists the files the fix changed; frontmatter `pr` and `plan-task` link the PR and the plan task.
- **Readers:** skill `review` and the brainstorming step of `vibe` read this folder before they start, so a solved problem is not solved twice.
- **Scope:** a design choice still gets a decision section and a bug an issue note; the solution note records how the work went, not what the code now does.

## Notes

```dataview
TABLE created AS "Date", domain AS "Area", pr AS "PR"
FROM "knowledge/handbook/Solutions"
WHERE type = "solution"
SORT created DESC
```
