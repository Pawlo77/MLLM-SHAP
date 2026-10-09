---
title: "Plans directory"
type: handbook
domain: process
tags: [plans, process]
created: 2026-10-07
---

# Plans

Implementation plans and the audits or research they come from. A plan is written before
multi-file work (skill `vibe`, step 2) and is executed task by task.

## Naming

`YYYY-MM-DD-<slug>.md`, dated the day the plan was written, slug in lowercase words joined by
hyphens (`2026-10-07-audit-remediation.md`). Supporting documents use the same pattern
(`2026-10-07-audit-report.md`).

## Header

Every plan starts with frontmatter (`title`, `type: plan`, `status`, `created`, `updated`,
`tags`) and then the writing-plans header:

```markdown
# <Feature> Implementation Plan

**Status:** draft

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** <one sentence>

**Architecture:** <two or three sentences>

**Tech Stack:** <key technologies>
```

## Status

The status line at the top (and the frontmatter `status`) is one of:

- `draft`: being written; not ready to execute.
- `in progress`: execution has started.
- `done`: every task is ticked and merged.
- `superseded by <file>`: replaced by a newer plan; kept for history.

## Tracking

- Each task step is a checkbox (`- [ ]`). A step is ticked only when it is done and
  `make check` passed (`.cursor/rules/plans.mdc`).
- A plan is updated in the same pull request that executes it: tick the boxes, update
  `updated`, and change the status when it changes.
- Plans are never deleted. A plan that no longer applies is marked
  `superseded by <file>` or `done`.
