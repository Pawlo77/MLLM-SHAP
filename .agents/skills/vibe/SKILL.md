---
name: vibe
description: >-
  Use when the user describes a feature or fix to build end to end, says "vibe", "just
  build" or "ship", or invokes /vibe.
---

# mllm-shap vibe loop

Read `.remember/remember.md`, `knowledge/mllm-shap.md`, the open notes under
`knowledge/issues/` and the notes in `knowledge/handbook/Solutions/` (problems
already solved, with their reusable rules) before anything else.

1. **Brainstorm** (superpowers:brainstorming): restate the ask in three lines,
   list the files it touches, name the invariant it must keep (efficiency,
   complementary pairs, manifest version, protocol signatures). Ask at most one
   question; otherwise state the assumption and continue.
2. **Plan**: for more than one file or any `src/estimation`/`pipeline` change,
   write `plans/YYYY-MM-DD-<slug>.md` with the writing-plans header and one task
   per test cycle. Smaller changes skip the file and list the steps in chat.
3. **Branch**: `git switch -c <type>/<slug>` from `main`.
4. **Implement** with skill `development` (red → green → refactor).
   Run the narrowest pytest target after each step; `make agent-check` before
   moving to the next task.
5. **Record**: append a `knowledge/decisions/<Domain> decisions.md` section for any design choice
   (skill `notes`); set `status: fixed` and `fixed_in` on the
   `knowledge/issues/` note when a listed bug is closed; update the touched
   `knowledge/architecture/<pkg>.md`; add the release-notes bullet; run
   `make vault-check`.
6. **Review**: run skill `review` on `git diff main...HEAD`; fix every
   Blocker and High finding before continuing. When the diff touches
   `src/mllm_shap/estimation/` or `src/mllm_shap/runtime/scheduler.py` and `codex`
   runs (`codex --version`), also run the Trail of Bits `second-opinion` skill
   (Codex through `codex exec`) on the same diff. List the findings where the two
   reviews disagree under "Second opinion: disagreements" for the user; do not
   merge or average them, and fix only what you can confirm with a test.
7. **Bug-hunt**: run skill `bugspot` probes when `estimation/`,
   `pipeline/`, `runtime/` or a backend changed.
8. **Ship**: run skill `ship` (commits, PR, Copilot review loop, merge or hand-off).
   Finish with `/remember`.
9. **Compound** (after the merge, inside `ship` for routine changes): write
   `knowledge/handbook/Solutions/<YYYY-MM-DD> - <title>.md` with skill
   `notes` (`type: solution`): the problem, what was tried and why it
   failed, what worked, and one reusable rule; `related-code` for the files the
   fix changed, frontmatter `pr` (the PR URL) and `plan-task` (the plan file and
   task). Skip it only when nothing was tried that failed and no rule emerged;
   say so in the hand-off.

Stop conditions: red `make agent-check` you cannot fix in two attempts; a change
that needs a new runtime dependency; anything touching publish, tags, or
branch protection. Report and wait.
