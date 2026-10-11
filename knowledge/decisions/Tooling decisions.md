---
title: "Tooling decisions"
type: decision-log
domain: tooling
created: 2026-10-07
updated: 2026-10-08
tags: [decisions, tooling]
---

# Tooling decisions

Part of [[mllm-shap]]; handbook: [[Onboarding]].

## 2026-10-07 — Agent skills canonical directory

Decision: `.agents/skills/` (the Agent Skills open standard location) is the single source of skills. `.cursor/skills` and `.claude/skills` are symlinks to it; `.claude/commands` links to `.cursor/commands`.

Why: Skills lived in `.cursor/skills/`, which only Cursor reads (Claude Code reads `.claude/skills/`, Codex `.agents/skills/`); the skill frontmatter already suits all three tools.

Consequences: One edit updates every tool; rules and skills point at `.agents/skills/`. Windows clones need symlink support (`core.symlinks`). Rejected: per-tool copies, which the audit found drifting (the vendored caveman skill).

## 2026-10-07 — Memory committed under remember

Decision: Commit the compressed memory layers, `identity.md`, `remember.md` (single handoff), `config.json` and `README.md`; ignore `logs/`, `tmp/`, `run/`, locks, `now.md` and per-session handoffs. `config.json` pins `data_dir: .remember`, `handoff_mode: single` and `prompt_stamp: stable`, and turns plugin promotions off. The prek hook `memory-content-guard` rejects secrets, private e-mail addresses and non-loopback IP addresses in `.remember/*.md` and `knowledge/**/*.md`.

Why: The remember plugin wrote a `.remember/.gitignore` of `*`, so memory never left one machine and Cursor and Codex sessions started without it. The plugin writes that file only when it is absent, so a committed one is honoured.

Consequences: Every tool starts from the same handoff (`.remember/remember.md`); Cursor injects it through its session-start hook. Memory diffs appear in pull requests and need review like any file. Rejected: external storage (`~/.remember`), which stays per machine.

## 2026-10-07 — Plans directory

Decision: Implementation plans live in `plans/YYYY-MM-DD-<slug>.md` with the writing-plans header, `type: plan` and a `status`. A plan is updated (checkboxes, status) in the pull request that executes it; plans are never deleted, only marked superseded.

Why: Plans lived in chat or outside this repository, where agents here could not read or tick them.

Consequences: `make vault-check` validates plan frontmatter and headers; the [[Dashboard]] lists plans by status.

## 2026-10-07 — Python upper bound lifted

Decision: `requires-python = ">=3.12"`; continuous integration (CI) tests 3.12, 3.13 and 3.14.

Why: `">=3.12,<3.13"` blocked Python 3.13 and 3.14 users of a pure-Python package, with no recorded reason.

Consequences: Optional extras (torch, transformers, lmstudio) may lag new Python versions; their tests stay behind markers ([[Release history]]). Rejected: keeping the cap until every extra supports the newest Python, which penalises the base install.
