---
title: "Pipeline decisions"
type: decision-log
domain: pipeline
created: 2026-10-07
updated: 2026-10-08
tags: [decisions, pipeline]
related-code:
  - "[[src/nlp_shap/pipeline/config.py]]"
  - "[[src/nlp_shap/pipeline/manifest.py]]"
  - "[[src/nlp_shap/pipeline/orchestrator.py]]"
  - "[[src/nlp_shap/pipeline/reanalyze.py]]"
  - "[[src/nlp_shap/pipeline/runner.py]]"
  - "[[src/nlp_shap/estimation/_shared.py]]"
  - "[[src/nlp_shap/runtime/archive.py]]"
---

# Pipeline decisions

Part of [[nlp-shap]]; architecture: [[pipeline]]. Concepts: [[Manifest]], [[Run id]], [[Run archive]].

## 2026-10-08 — Absolute budget passed to samplers as an exact fraction

Decision: `BudgetConfig.evaluations` (an integer of at least 1) is an alternative to `fraction`. Setting both raises `ConfigError`, neither gives `fraction: 1.0`, and only the set key is serialised. `estimation._shared.budget_fraction(n, fraction, evaluations)` is the one conversion: `evaluations / 2^n`, capped at 1. The orchestrator passes it to `sample_masks`, `reanalyze` treats `1.0` as a full-table run, and the runner's `max_players` guard compares `min(log2(evaluations), n)`. `exact` accepts only `fraction: 1.0`.

Why: `fraction` scales `2^n`, so above about 25 players no typed fraction gives a servable number of coalitions ([[Fixed issues|D01]]).

Consequences: Division by a power of two is exact, so `int(2^n * fraction)` in `compute_mc_num_samples` and `compute_complementary_num_splits` returns exactly `evaluations` (tested to `n = 1000`). Each sampler's floor (2 for Monte Carlo (MC); `min(2n + 2, 2^n)` and whole pairs for complementary and Neyman) and the ceiling `2^n` still apply. The estimator protocol is unchanged, so third-party estimators get an equivalent fraction. Configurations without `evaluations` keep their [[Run id]]. Above `n = 1023` the count helpers overflow, as for `fraction`. Rejected: an `evaluations` argument on `sample_masks` (changes every plugin's protocol); reading it from the plugin context in each sampler (branches everywhere, meaningless fraction for third parties).

## 2026-10-07 — Run id config digest and archive policy

Decision: `run_id = f"{snapshot_id[:16]}-{config_digest}"`. `config_digest` is the first 12 hex characters of a SHA-256 over the YAML dump of the configuration without `archive`, plus the effective model id (the injected backend's since 0.1.18). `ArchiveConfig.on_exists` is `error` (default; raises `ArchiveExistsError` for a finished run's archive) or `overwrite`; since 0.1.18 an unfinished run's archive is replaced either way.

Why: `f"{snapshot_id}-{seed}"` collided across configurations and the archive appended to existing rows, double-counting them in [[Reanalyze]] ([[Fixed issues|C02]]).

Consequences: Re-running a configuration into the same path fails loudly instead of corrupting the archive. Archive directory names changed in 0.1.17, and [[Reanalyze]] rejects older archives ([[Pipeline decisions#2026-10-07 — Manifest schema version 2|manifest schema 2]]).

## 2026-10-07 — Manifest schema version 2

Decision: `MANIFEST_SCHEMA_VERSION = 2`. The payload records `schema_version`, `nlp_shap_version`, `run_id`, `snapshot_id`, `estimand`, `estimator`, `value_fn`, `normalizer`, `absence_policy`, `players`, `backend_kind`, `model_id`, `player_ids` and the full configuration. The archive database sets `PRAGMA user_version = 2` and indexes `coalition_key`. `parse_manifest` and `RunArchive.load` raise `ManifestVersionError` for any other version. `reanalyze` resolves the archived estimator, raises `ReanalyzeMismatchError` when the configuration names another estimator, absence policy or partition, and aggregates through `estimator.estimate_attributions`.

Why: The manifest held only `estimand` and `run_id` and SQLite had no `user_version`, so [[Reanalyze]] could not tell which estimator or configuration wrote an archive and always used the marginal aggregator.

Consequences: 0.1.17 archives are self-describing; 0.1.16 and older are rejected (they carry [[Fixed issues|C01]] anyway). 0.1.18 moved generation text into the SQLite row, switched to WAL journaling and bumped `ARCHIVE_SCHEMA_VERSION` and `MANIFEST_SCHEMA_VERSION` to 3 under the same policy; 0.1.17 archives are rebuilt by re-running explain. Rejected: the two-field manifest, with which `reanalyze` re-aggregates a complementary contribution archive as exact.
