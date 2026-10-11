---
title: "Process decisions"
type: decision-log
domain: process
created: 2026-10-07
updated: 2026-10-09
tags: [decisions, process]
related-code:
  - "[[scripts/pr_policy.py]]"
  - "[[scripts/upgrade.py]]"
  - "[[scripts/hooks/merge_gate.sh]]"
  - "[[scripts/check_brevity.py]]"
  - "[[scripts/practical_features.py]]"
  - "[[scripts/strip_watermarks.py]]"
---

# Process decisions

Part of [[mllm-shap]]; handbook: [[Onboarding]].

## 2026-10-09 — AI attribution stripped

Decision: No `Co-Authored-By` or "Generated with" line naming an assistant; `scripts/strip_watermarks.py` strips them (prek, edit hooks, skill `ship`), `pr-title.yaml` rejects them.

Why: Banners add noise, not information (reverses 2026-10-07).

Consequences: Agent work is invisible in `git log`. Rejected: check-only hook (agents re-add them).

## 2026-10-08 — Deep practical tier

Decision: `make practical TIER=deep` runs `deep.*` scenarios on pinned `Qwen2.5-3B-Instruct` float16 (generator and judge) under an 8 GB VRAM cap. A cached 10-player exact reference (1024 generations, `results/reference-<hash>.json`) feeds every sampled run. Skill `release` runs it; `ship` keeps the quick tier, whose judge may run in LM Studio.

Why: Five-player prompts cannot separate estimators by error versus budget or test long threads.

Consequences: 28 minutes cold, 4 warm (M4 Pro). Rejected: a 7B in-process model (over the cap).

## 2026-10-08 — Practical tests on real small models

Decision: Marker `practical(<feature>)` runs `tests/practical/scenarios.yaml` on pinned `Qwen2.5-1.5B-Instruct` (generator and judge) and `SmolLM2-360M-Instruct` on a GPU; skill `ship` runs the features `scripts/practical_features.py` maps from the diff. Rubric checks assert; a logit judge (YES−NO margin ≥ 1 nat in two of three paraphrases) answers semantic questions only.

Why: Mock and random-weight models cannot show whether attributions make sense.

Consequences: Not in `make check` or CI; a generation cache makes warm runs take seconds. Rejected: a hosted judge (keys, cost).

## 2026-10-08 — Automatic upgrades and releases

Decision: Skill `upgrade` runs `scripts/upgrade.py` weekly on a `chore/upgrades-<date>` branch: lock, prek revisions, action pins, uv and Obsidian manifests, never `publish.yaml`. `pr_policy.py --bump` bumps by the commits since the last `v*` tag; skill `release` commits `chore: release vX.Y.Z` in the PR and tags after the merge. Version-specifier, lock, pin and release-version changes are routine.

Why: The maintainer asked for current dependencies and hands-free releases.

Consequences: New packages, `publish.yaml` pins and uv bumps wait for the maintainer; failed releases fix forward. Rejected: `release-please`, `python-semantic-release` (own the changelog or publish unreviewed).

## 2026-10-08 — Delivery autopilot merges routine changes

Decision: Skill `ship` is the only delivery path. `scripts/pr_policy.py` calls a branch routine (rules in its docstring) or review. Both open the pull request (PR) with a Copilot review and resolve its threads in at most five rounds; routine merges on green continuous integration (CI), review hands the maintainer the link. `scripts/hooks/merge_gate.sh` merges only on a routine marker for `HEAD`; permissions deny force pushes, pushes to `main`, drafts and squash or rebase merges.

Why: The maintainer asked routine changes to merge after Copilot's review.

Consequences: Security-relevant changes never auto-merge, even as `chore`. Claude answers Copilot on its own PR and never reviews other PRs.

## 2026-10-08 — Claude reviews before the pull request

Decision: Claude reviews its own work locally before a PR opens (skill `review`, agent `reviewer`; input `git diff main...HEAD` or a path) and never reviews or comments on a PR on GitHub (`AGENTS.md`).

Why: Cursor Bugbot already reviews every PR.

Consequences: `.cursor/BUGBOT.md` holds the severities skill `review` quotes; the `code-review` plugin serves local diffs. Rejected: a claude-review workflow (duplicates Bugbot, spends CI model calls); reviewing open PRs on request (conflicting threads).

## 2026-10-08 — Brevity budgets

Decision: The always-apply rule `brevity` asks for one home per fact, lists over paragraphs, no sentence over 35 words, and deletion over archiving. `scripts/check_brevity.py` enforces its per-file budgets in `make vault-check` and prek; ruff enforces `PLR0915` and `C901`. A file at budget must not grow; old large functions are not split for it.

Why: Facts repeated across README, docs, vault and rules drifted; the maintainer asked for less.

Consequences: Growth moves detail to another file; vendored skills and agents are exempt. Rejected: an unchecked guideline (drifts); compressed prose (hard to read).

## 2026-10-07 — Single release 0.1.18 for Phases 3 and 2

Decision: Phases 3 and 2 ship together as 0.1.18; there is no 0.1.19.

Why: Phase 3 (plugin contracts) ran first in one tree, and Phase 2 (backend runtime) needs its contracts.

Consequences: One upgrade carries every plugin-contract and archive-schema-3 break. Rejected: two tags from one tree.

## 2026-10-07 — Merge strategy

Decision: PRs merge with a merge commit; head branches are deleted on merge.

Why: A plan phase lands as one PR with one commit per task.

Consequences: Task commits stay on `main` for `git bisect`. Rejected: squash (loses task history); rebase (rewrites hashes review comments cite).
