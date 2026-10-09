# Literature watch

Weekly shortlist of new papers, written into the local vault for the user to triage.

Schedule: a **local** scheduled task of the Claude desktop app (id `nlp-shap-lit-watch`, cron `0 9 * * 1`, working directory this repository, prompt `Run /lit-watch in the nlp-shap repository.`). Never a remote routine: it cannot write this machine's `knowledge/papers/_watch/`. Creating the task changes standing configuration, so ask the user first.

1. Run the `paper-scout` agent on: Shapley and Banzhaf attribution for LLM prompts; complementary contributions; Neyman allocation for Shapley sampling; token-level explainability of language models. $ARGUMENTS narrows or replaces the topics. Papers from the last 60 days only.
2. Write its report unchanged to `knowledge/papers/_watch/<YYYY-MM-DD>.md` (git-ignored; skipped by the vault scripts) with frontmatter `title: "Literature watch <YYYY-MM-DD>"`, `type: handbook`, `domain: papers`, `tags: [watch]`, `created: <YYYY-MM-DD>`.
3. Do not edit `knowledge/papers/Papers.md`, commit or push. Reply with the path and the top three entries; the user picks, then skill `notes` appends their sections to `Papers.md`.
