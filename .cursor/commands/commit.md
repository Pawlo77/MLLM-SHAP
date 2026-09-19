# Commit

Propose granular commits only — do **not** commit until the user approves.

1. Run validate first (`/check` / skill `nlp-shap-check`)
2. `git status` · `git diff` · `git log -5` (parallel)
3. Split unrelated concerns; propose a numbered commit plan
4. Ask which commits to run (all / subset / squash)
5. After approval: stage per commit, HEREDOC message, verify `git status`

Follow `.cursor/rules/git-commits.mdc`. Never `--no-verify`, amend, or force-push unless user rules allow.
