# Handoff

## State
v0.1.19 released 2026-10-10 (PR 6: D04 half-precision KV-cache warning, C16 LM
Studio repeated keys, brevity RST-fence fix, practical weight check); v0.1.18 on
2026-10-09 (PR 5, audit remediation). 0.1.17 has notes but no tag. The repository is
public since 2026-10-10; `main` is protected (`prek`, `tests-passed`, `CodeQL`,
`title`), merge commits only; secret scanning and push protection on. Pages deploys from GitHub Actions.
Weekly local tasks: `nlp-shap-upgrade` (Mon 08:00) and `nlp-shap-lit-watch` (Mon 09:00).

## User-only
- Copilot code review reports "quota limit" (PRs 5 to 7); skill `ship` merges routine
  PRs without it and says so. Optional: `CODECOV_TOKEN` secret.

## Next
- `make practical` defaults `NLP_SHAP_PRACTICAL_LMS_MODEL=qwen2-500m-instruct`; start
  LM Studio with `lms server start` for the connector scenarios.
- Open PR 4 (`feat/audio-track-v0.2`): D02 audio backend and D03 rebase, deferred.
