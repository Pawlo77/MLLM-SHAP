---
title: "mllm-shap"
type: map
domain: index
tags: [moc, mllm-shap]
created: 2026-10-07
updated: 2026-10-10
---

# mllm-shap

mllm-shap explains the output of a conversational language model. The prompt's tokens are players of a cooperative game, and each player gets a share of the change in a value function between masked and unmasked generations. The library separates what is measured (the estimand: Shapley value or Banzhaf index), how coalitions are sampled (the estimator) and how a generation is scored (the value function). Every coalition evaluation goes into a run archive that can be re-scored later.

Start here: [[Onboarding]], then [[How a run works]]. Live tables: [[Dashboard]].

> [!note] State of the code
> The architecture notes describe release 0.1.18. Open items: [[Open issues]]; all decisions: [[Decisions]].

## Packages

| Package | Responsibility |
|---|---|
| [[core]] | Public exports, `mllm-shap` command line, logging, typed errors |
| [[domain]] | Frozen value types: snapshots with text and audio parts, players and spans, masks, generation and runtime settings |
| [[protocols]] | Typing contracts every plugin kind implements, capability protocols, plugin options |
| [[masking]] | Token partition with player spans, span-based absence policies, mask codec |
| [[estimation]] | Coalition samplers, estimand aggregators, normalisers |
| [[value]] | Value functions that score a coalition generation against the base generation |
| [[runtime]] | TaskGroup scheduler, deduplication, result store, run archive, prefix cache, telemetry |
| [[pipeline]] | Configuration, runner, orchestrator, reanalyze, manifest, results |
| [[plugins]] | Registry, entry-point groups, built-in table, plugin context and options |
| [[backends]] | Mock, LM Studio, Hugging Face transformers and OpenAI-compatible HTTP application programming interface (API) backends |
| [[viz]] | Token attribution renderers |
| [[alignment]] | Spectrogram-guided alignment of transcript tokens to speech segments |

Generated from the code: [[Modules]], [[Public API]], [[Config keys]], [[Plugins]], [[Release history]], [[Benchmarks]].

## Plugin groups

Groups, config keys and built-in names: [[Plugins]]. `BUILTIN_PLUGINS` registers all eight groups; `default_registry()` adds third-party entry points that do not reuse a built-in name ([[plugins]], [[Plugin options]]).

## Invariants

- **Boundary coalitions**: the empty coalition and the grand coalition are always evaluated, for every estimator ([[Fixed issues]] C01, C03).
- **Efficiency**: for the Shapley estimand on the full coalition table, `sum(values) + base_value == grand_value` (empty- and full-coalition payoffs); partial budgets converge to it.
- **Complementary pairs**: complementary contribution sampling evaluates a coalition together with its complement, starting with `(N, ∅)` and the singleton pairs ([[Estimation decisions#2026-10-07 — Singleton-first complementary sampling|Singleton-first complementary sampling]]).
- **Run id**: identifies one snapshot under one configuration: the snapshot id plus a configuration digest ([[Run id]]).
- **Manifest**: every run archive carries a versioned manifest (schema 4) with the full configuration ([[Pipeline decisions#2026-10-07 — Manifest schema version 2|Manifest schema version 2]]).

## Concepts

[[Base generation]] · [[Run archive]] · [[Reanalyze]] · [[Manifest]] · [[Run id]] · [[Prefix hash]] · [[SGPA]] · [[Snapshot]] · [[Message]] · [[Player span]] · [[Plugin options]] · [[Capability protocol]] · [[Solution note]]

## Decisions

One log per domain, newest first: [[Domain decisions|domain]] · [[Masking decisions|masking]] · [[Estimation decisions|estimation]] · [[Pipeline decisions|pipeline]] · [[Backends decisions|backends]] · [[Runtime decisions|runtime]] · [[Value decisions|value]] · [[Alignment decisions|alignment]]

## Papers

Full citations: [[Papers]].

| Topic | Papers |
|---|---|
| Shapley value | [[Papers#Shapley 1953 — A value for n-person games\|Shapley 1953]], [[Papers#Young 1985 — Monotonic solutions of cooperative games\|Young 1985]], [[Papers#Owen 1972 — Multilinear extensions of games\|Owen 1972]] |
| Banzhaf index | [[Papers#Banzhaf 1965 — Weighted voting doesn't work\|Banzhaf 1965]], [[Papers#Dubey 1979 — Mathematical properties of the Banzhaf power index\|Dubey 1979]] |
| Sampling | [[Papers#Castro 2009 — Polynomial calculation of the Shapley value\|Castro 2009]], [[Papers#Mitchell 2022 — Sampling permutations for Shapley value estimation\|Mitchell 2022]], [[Papers#Strumbelj 2014 — Explaining prediction models with feature contributions\|Strumbelj 2014]], [[Papers#Neyman 1934 — Two aspects of the representative method\|Neyman 1934]] |
| Complementary contribution | [[Papers#Zhang 2023 — Efficient sampling approaches to Shapley value approximation\|Zhang 2023]] |
| Attribution for models and text | [[Papers#Lundberg 2017 — A unified approach to interpreting model predictions\|Lundberg 2017]], [[Papers#Fryer 2021 — Shapley values for feature selection\|Fryer 2021]], [[Papers#Ethayarajh 2021 — Attention flows are Shapley value explanations\|Ethayarajh 2021]], [[Papers#Goldshmidt 2024 — TokenSHAP\|Goldshmidt 2024]] |

## Handbook

[[Onboarding]] · [[How a run works]] · [[Debugging a wrong attribution]] · [[Roadmap]] · [[Python SHAP visualization reference]]
