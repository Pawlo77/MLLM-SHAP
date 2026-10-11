---
title: "Vault decisions"
type: decision-log
domain: vault
created: 2026-10-07
updated: 2026-10-08
tags: [decisions, vault]
related-code:
  - "[[scripts/check_vault.py]]"
  - "[[scripts/build_knowledge.py]]"
  - "[[scripts/hooks/session_start.sh]]"
---

# Vault decisions

Part of [[mllm-shap]]; rules: [[knowledge/README|Vault rules]].

## 2026-10-08 — One log per domain and one papers file

Decision: Decisions live as dated sections of one `knowledge/decisions/<Domain> decisions.md` log per domain (`type: decision-log`, sections Decision, Why, Consequences, newest first); papers live as sections of [[Papers]] (`type: paper-list`). `scripts/check_vault.py` checks the section fields and heading links; `scripts/build_knowledge.py` lists every decision in [[Decisions]].

Why: 23 decision and 14 paper notes repeated frontmatter and Context, Alternatives and Math boilerplate: 6,447 counted words, against about 4,000 for the same decisions as log sections and the papers as one list.

Consequences: Links point at sections (`[[Estimation decisions#2026-10-07 — Full coalition table|…]]`). A log is named `<Domain> decisions`, not `<domain>`, because architecture notes already own the package names. The site export drops log sections that name a plan phase.

## 2026-10-07 — Vault hook is a ratchet (superseded 2026-10-08)

Decision: The editor hook ran `scripts/check_vault.py --quiet` after each agent edit under `knowledge/`, `plans/`, `src/`, `tests/` or `scripts/glossary.yaml`. It cached the previous issue list in `.remember/tmp/` and blocked (Claude Code: exit 2; Cursor: additional context) only when an edit raised the issue count, reporting the new lines. Each session start cleared the cache; prek `vault-check`, `make check` and the strict continuous integration (CI) step stayed absolute.

Why: Blocking on any issue would stop unrelated edits whenever the vault was already red, for example while another change deletes a module a note links.

Consequences: An agent could work in a red vault but not make it worse unnoticed; commits still needed a green vault. Rejected: absolute blocking (stalls every edit on someone else's breakage); warning only (broken links slip through until commit). Superseded on 2026-10-08: the hook was removed with the other custom gates in the [[Process decisions#2026-10-08 — Delivery autopilot merges routine changes|delivery autopilot]] clean-up, leaving `make vault-check` in prek and CI as the only check.

## 2026-10-07 — Repository root as Obsidian vault

Decision: The repository root is the Obsidian vault `mllm-shap`: notes in `knowledge/`, plus `plans/` and `.remember/*.md`. `.obsidian/` settings are committed; workspace state and plugin code are git-ignored. `scripts/build_knowledge.py` rebuilds generated notes from the package; `scripts/check_vault.py` checks links, frontmatter, isolation, abbreviations, secrets and traceability tags in prek and CI. Handbook, architecture, decision and concept notes are exported into the Sphinx site (`docs/handbook/`).

Why: Design knowledge lived in chat logs and the audit report. The attribution research vault showed that a repository-rooted vault lets the Code Graph plugin draw code, tests and notes as one graph.

Consequences: Every change under `src/mllm_shap/<pkg>/` updates `knowledge/architecture/<pkg>.md`, which reviewers check. No PDFs are committed; papers link to digital object identifiers (DOIs) or arXiv. Rejected: a separate notes repository, which drifts from the code and is invisible to agents here.
