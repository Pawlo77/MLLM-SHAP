---
title: "Fixed issues"
type: issue
domain: index
status: fixed
found: 2026-10-07
created: 2026-10-08
fixed_in: "0.1.18"
tags: [issue, fixed]
---

# Fixed issues

Part of [[mllm-shap]]. Fixed bugs, one row each; open and deferred issues keep their own notes ([[Open issues]]). Release dates: [[Release history]].

| ID | Title | Package | Cause | Fixed in | Pinned by |
|---|---|---|---|---|---|
| C01 | Grand coalition omitted | [[estimation]] | `ExactEstimator.iter_mask_ints` skipped the all-present mask and the orchestrator never seeded `v(N)`, so values broke efficiency. | 0.1.17 | `test_pipeline_efficiency` in [[tests/pipeline/test_pipeline_axioms.py]] |
| C02 | Archive append on re-run | [[runtime]] | `run_id` ignored the configuration and `RunArchive` inserted into an existing database. | 0.1.17 | `test_rerun_into_existing_archive_raises_by_default` in [[tests/pipeline/test_runner_e2e.py]] |
| C03 | Complementary pair omitted | [[estimation]] | `is_empty_or_grand` rejected `(N, ∅)`, the only pair that reaches contribution column `n`. | 0.1.17 | `test_complementary_full_table_matches_exact_shapley` in [[tests/estimation/test_complementary.py]] |
| C04 | Neyman phase two never runs | [[estimation]] | Phase one demanded more samples than distinct masks, and the orchestrator never called the allocation phase. | 0.1.17 | `test_orchestrator_runs_second_phase_for_two_phase_estimators` in [[tests/pipeline/test_orchestrator.py]] |
| C05 | Monte Carlo aggregation biased | [[estimation]] | `aggregate_from_marginals` summed the observed marginals only, so missing pairs counted as zero. | 0.1.17 | `test_partial_sample_scale_matches_full_table` in [[tests/estimation/test_axioms.py]] |
| C06 | KV cache write-only and quadratic | [[backends]] | The key-value (KV) cache ran one forward pass per prefix length into an unbounded dict that `generate` never read. | 0.1.18 | `test_transformers_backend_reuses_prefix_kv_state` in [[tests/backends/test_transformers_text.py]] |
| C07 | Scheduler swallows early failures | [[runtime]] | `run_iter` discarded the done set of `asyncio.wait` without reading its exceptions. | 0.1.17 | `test_scheduler_raises_first_failure_and_cancels_rest` in [[tests/runtime/test_scheduler.py]] |
| C08 | Explain and reanalyze value functions differ | [[pipeline]] | `explain` built token rows from the rendered prompt, `reanalyze` from the generated text. | 0.1.17 | `test_reanalyze_matches_explain_values` in [[tests/pipeline/test_runner_e2e.py]] |
| C09 | Mask hash not stable | [[masking]] | `hash_mask` used Python's per-process salted `hash()`. | 0.1.17 | `test_hash_mask_is_stable_across_processes` in [[tests/masking/test_codec.py]] |
| C10 | Neyman complement index | [[estimation]] | `_estimate_m_hat` paired column `k` with `n - k - 1` instead of the complement size `n - k`. | 0.1.17 | `test_neyman_complement_column_pairs_size_j_with_n_minus_j` in [[tests/estimation/test_neyman.py]] |
| C11 | KV cache grouping misaligns payoffs | [[pipeline]] | Jobs sorted by prefix hash returned payoffs in sorted order while aggregation kept the original mask order. | 0.1.17 | `test_pipeline_matches_exact_reference_on_non_additive_game` (`kv_cache=True`) in [[tests/pipeline/test_pipeline_axioms.py]] |
| C12 | Complementary size-1 stratum starvation | [[estimation]] | Random pairs rarely drew `({i}, N ∖ {i})`, and zero-filled empty strata over-weighted the size-`n` stratum. | 0.1.18 | `test_complementary_covers_every_size_one_stratum_at_half_budget` in [[tests/estimation/test_complementary.py]] |
| C13 | Neyman truncated phase-one walk | [[estimation]] | The phase-one cursor walked the grid player by player, so truncation skipped whole players. | 0.1.18 | `test_neyman_truncated_phase_one_spreads_empty_cells` in [[tests/estimation/test_neyman.py]] |
| C14 | Complementary budget floor drops a singleton pair | [[estimation]] | The `2n` floor counted `(N, ∅)`, leaving room for only `n - 1` singleton pairs. | 0.1.18 | `test_complementary_floor_budget_is_symmetric_and_efficient` in [[tests/estimation/test_axioms.py]] |
| C15 | Contextual embeddings differ between explain and reanalyze | [[pipeline]] | Backends salted the contextual hash with the coalition's `snapshot_id`, `reanalyze` with the explained snapshot's id. | 0.1.18 | `test_reanalyze_matches_explain_embedding_values` in [[tests/pipeline/test_runner_e2e.py]] |
| C16 | LM Studio key listed twice is ambiguous | [[backends]] | LM Studio lists a model key once per downloaded quantisation, and `match_downloaded_model_key` counted the repeats as candidates. | unreleased | `test_match_accepts_a_key_listed_twice` in [[tests/backends/test_lmstudio_resolve.py]] |
| D01 | Absolute evaluation budget | [[pipeline]] | `BudgetConfig` had only `fraction`, so a budget could not be set as an evaluation count. | 0.1.18 | `test_max_players_guards_the_absolute_budget` in [[tests/pipeline/test_budget_evaluations.py]] |
| D02 | No backend generates from audio | [[backends]] | Every backend refused `AudioPart` content. Fixed by the transformers processor path, `input_audio` parts in `api` and `mock_audio`. | 0.4.1 | `test_generate_passes_audio_features_and_skips_the_prefix_cache` in [[tests/backends/test_transformers_audio.py]] |
| D03 | Audio-track pull request 4 needed a rebase | process | It bundled three planned releases and built on the text-only message model that 0.1.18 replaced with content parts. | 0.2.1 | Ported per phase in pull requests 11 (0.2.0) and 13 (0.2.1); Phase 15 superseded by 0.1.18 |
| D04 | KV prefix cache changes half-precision greedy outputs | [[backends]] | Cached prefixes run in separate forward passes that round differently in float16 and bfloat16, and the guide claimed equality. | unreleased | `test_transformers_backend_warns_once_on_half_precision_prefix_cache` in [[tests/backends/test_transformers_text.py]] |
| D05 | Prefix cache speed-up fell below its threshold | [[backends]] | The deep test timed the cache-on arm first on a model the generation cache had left cold, so that arm paid the accelerator warm-up (1.01 against 1.21 warm on Qwen 1.5B). | unreleased | `test_prefix_cache_is_faster_and_equal` in [[tests/practical/deep/test_runtime.py]] warms up first: 1.17 on Qwen 3B |
| D07 | Empty reply has no logprobs | [[backends]] | LM Studio sends `logprobs: null` for a zero-token reply, so `ApiBackend` returned `None` and `reanalyze` refused the archive. | unreleased | `test_api_backend_scores_an_empty_completion_with_no_logprobs` in [[tests/backends/test_api.py]] |
| M01 | Neyman internals untested in isolation | [[estimation]] | Tests checked budgets and end-to-end values only, leaving the allocation arithmetic unpinned. | 0.1.18 | [[tests/estimation/test_neyman_internals.py]] |
| M02 | Stratified marginal aggregation lacks boundary tests | [[estimation]] | Partial-table tests used additive games, where renormalisation mistakes cancel. | 0.1.18 | [[tests/estimation/estimands/test_weights.py]] |
| M03 | Sampler budget helpers lack exact-count tests | [[estimation]] | Tests checked that budgets were even and capped, not the exact counts at small `n`. | 0.1.18 | [[tests/estimation/test_shared.py]] |
| N01 | Wrong DOIs in theory references | docs | `docs/theory/*.rst` cited one wrong and two unresolvable digital object identifiers (DOIs). | 0.1.18 | none; `linkcheck` skips doi.org, so DOIs are checked by hand |
| T01 | Agent tool quirks from the session-start smoke test | tooling | Each tool wraps the shared hooks in its own payload; Codex `apply_patch` names files only inside the patch. | 0.1.18 | `test_edited_file_codex_patch` in [[tests/hooks/test_lib_sh.py]] |
| V01 | Renderer font weight warning and label overlaps | [[viz]] | DejaVu Sans has no `semibold`, and label positions were estimated from character counts. | 0.1.18 | `test_token_text_boxes_do_not_overlap` in [[tests/viz/test_render.py]] |
