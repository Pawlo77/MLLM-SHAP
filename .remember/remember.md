# Handoff

## State
The rewrite plan is complete: text Phases 0–14 and audio Phases 15–21, released
through v0.6.0 (2026-10-12). The audio track added `mock_audio` (0.4.0, never on
PyPI), audio input for `transformers` (processor path, Granite Speech), `api`
(`input_audio`) and `lmstudio` (`audio: transcript`) (0.4.1), snapshot JSON audio and
`sgpa_segments` `transcript: align_only` (0.5.0), and `loo`, `reference_wer` and
`deletion_curve` (0.6.0). PyPI skipped 0.3.1 and 0.4.0.
The repository is public; `main` is protected (`prek`, `tests-passed`, `CodeQL`,
`title`), merge commits only, auto-merge on for routine PRs; secret scanning on.
Weekly local task: `mllm-shap-upgrade` (Mon 08:00).

## User-only
- Copilot review is off by default (`[tool.agent-kit.ship] copilot`); skill `ship` follows
  CI and the threads of any reviewer.
- Done means every CI check green, the Codecov report comment green (CI has no torch,
  so test new transformers code with stubs) and the PR mergeable with `main`.

## Next
- SGPA Audio work moves to `mllm-xai`: the Phase 0 game definition and model matrix,
  then the runner factory and smoke experiment on top of `mllm-shap` 0.6.0.
- D06: add the SGPA paper to Papers.md once its conference review ends.
- `tests/benchmarks/baseline.json` has no `test_mock_audio_bench` entry; record it with
  `make bench-baseline` on the baseline machine.
- Guide figures from notebooks: tag a cell `docs-figure:<name>`; the docs build extracts
  it (`docs/scripts/notebook_figures.py`).
