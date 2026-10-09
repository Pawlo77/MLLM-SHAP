---
title: "D03 - PR 4 rebase"
type: issue
domain: backends
status: deferred
severity: minor
found: 2026-10-07
created: 2026-10-07
fixed_in: ""
tags: [issue, deferred, audio, process]
---

# D03 - PR 4 rebase

Related: [[D02 - Audio backend]], [[Roadmap]].

## Symptom

- Pull request (PR) 4 (audio track versions 0.2.0 to 0.2.2, 39 files, 2714 added lines) has been open since 2026-09-19 with green continuous integration (CI) and no review.

## Cause

- It bundles three planned releases into one PR, against the "one phase, one tag, one publish" rule of skill `rewrite`.
- It builds on the text-only abstractions that [[D02 - Audio backend]] lists.

## Repro

`gh pr view 4 --json files,additions,createdAt`.

## Fix

- User decision, deferred: land 0.1.17 (correctness) first, rebase after plan Phase 3 merges, then split into one PR per phase tag.
