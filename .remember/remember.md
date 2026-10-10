# Handoff

## State
v0.2.1 released 2026-10-10: the audio track's Phases 16 (SGPA alignment, 0.2.0, PR 11)
and 17 (`sgpa_segments` audio players and audio absence rendering, 0.2.1, PR 13), ported
from the closed PR 4 onto the 0.1.18 message model; Phase 15 was superseded by 0.1.18.
The repository is public; `main` is protected (`prek`, `tests-passed`, `CodeQL`,
`title`), merge commits only, auto-merge on for routine PRs; secret scanning on.
Weekly local tasks: `nlp-shap-upgrade` (Mon 08:00), `nlp-shap-lit-watch` (Mon 09:00).

## User-only
- Copilot code review reports "quota limit"; skill `ship` merges routine PRs without it
  and says so.

## Next
- Audio track Phase 18 (mock audio backend, 0.2.2), then 19 to 21 (skill `rewrite`);
  D02 stays open until a backend generates from audio.
- D05: the deep prefix-cache speed-up check.
- D06: add the SGPA paper to Papers.md once its conference review ends.
- Guide figures from notebooks: tag a cell `docs-figure:<name>`; the docs build extracts
  it (`docs/scripts/notebook_figures.py`).
