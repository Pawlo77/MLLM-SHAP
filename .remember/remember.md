# Handoff

## State
The rewrite plan is complete: text Phases 0–14 and audio Phases 15–21, released
through v0.6.0 (2026-10-12). The approximation plan (mllm-xai) shipped in v0.7.1
(2026-10-18): `leverage`, `kernel_banzhaf` and `proxy` (extra `proxy`), `budget.per_player`,
`explanation.efficiency`, the `sentences` partition, `explain_hierarchical`, stratified `mc`
aggregation and corrected Neyman weights; attributions of `mc`, `complementary` and
`neyman_cc` at partial budgets differ from 0.6.0. The v0.7.0 tag was pushed without its
version bump and never published; it was not reused. The audio track added `mock_audio` (0.4.0, never on
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
  then the runner factory and smoke experiment on top of `mllm-shap` 0.7.1.
- `make practical TIER=deep` did not run for 0.7.1 (LM Studio missing); the practical
  scenarios `estimator.leverage`, `estimator.proxy`, `estimator.kernel_banzhaf` and
  `partition.sentences` carry unmeasured thresholds. Run them and record the numbers.
- Before tagging, check the release commit holds the bump:
  `git show HEAD:pyproject.toml | sed -n 3p`. A hook once dropped unstaged edits.
- D06: add the SGPA paper to Papers.md once its conference review ends.
- `tests/benchmarks/baseline.json` has no `test_mock_audio_bench` entry; record it with
  `make bench-baseline` on the baseline machine.
- Guide figures from notebooks: tag a cell `docs-figure:<name>`; the docs build extracts
  it (`docs/scripts/notebook_figures.py`).
