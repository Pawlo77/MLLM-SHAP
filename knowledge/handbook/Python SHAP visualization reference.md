---
title: "Python SHAP visualization reference"
type: handbook
domain: viz
parent: "[[mllm-shap]]"
tags: [handbook, viz, shap]
created: 2026-10-10
updated: 2026-10-10
---

# Python SHAP visualization reference

Module note: [[viz]].

Read: `shap` 0.53.0 (tag `v0.53.0`, PyPI latest on 2026-10-10), source `shap/plots/`. SHAP means SHapley Additive exPlanations.

## Steps

1. Before adding a figure to [[viz]], find the matching `shap` plot below and reuse its question and encoding.
2. Take work from "Gaps worth closing" from the top; add the release version to a gap once it ships.
3. When a `shap` release changes `shap/plots/`, re-read the source and update the version line and `updated`.

## shap plots

Input is a `shap.Explanation`: `values`, `base_values`, `data`, `feature_names`, `output_names`, optional `clustering`. Rows are samples; a third axis is the output.

| Plot | Question it answers | Input | Text |
|---|---|---|---|
| `bar` | Which features matter, locally (one row) or globally (mean absolute value)? Optional `clustering` and `clustering_cutoff` merge correlated features | one or many rows, cohorts | yes, via cutoff |
| `waterfall` | How do features move the output from the expected value to the prediction? | one row | no |
| `force` | The same, as one stacked push and pull bar (JavaScript, or matplotlib) | base value, one or many rows | no |
| `decision` | How do cumulative contributions differ between predictions? | many rows, interactions | no |
| `beeswarm`, `violin` | How do values spread across samples, coloured by feature value? | many rows | no |
| `heatmap` | Which samples explain alike? Supervised clustering of rows | many rows | no |
| `scatter`, `partial_dependence` | How does one feature drive its value, and what interacts with it? | feature columns | no |
| `embedding`, `monitoring`, `group_difference` | Sample map, drift over time, difference between two groups | many rows | no |
| `image`, `image_to_text` | Pixel and region attributions | image rows | no |
| `benchmark` | Which explainer scores better? | benchmark results | no |
| `text` | Which input tokens drive each output? | token rows with `data` | yes |

`text` behaviour:

- Colours each input token by value; clicking a token shows its value.
- With `output_names`, it draws one panel per output and a clickable header switches between them.
- Tokens merge into one chunk when group effects fall below `grouping_threshold`, using the `clustering` tree.
- `num_starting_labels` sets how many top tokens start uncovered.
- Colour and axis limits are shared across rows.
- `text_to_text` (heatmap and saliency switch) exists in the source, but `text` has its call commented out.

## What mllm-shap has

- `token_text`: coloured tokens, matplotlib or HTML fragment.
- `token_bar`: sorted horizontal bars with value labels.
- Both take one attribution vector per player ([[viz]]).
- Labels come from `PlayerSet.spans`, so message and custom players already work.

## Gaps worth closing

Data: `ExplainResult` holds `values`, `base_value`, `grand_value`; spans hold `message_index`, `speaker`, `samples`. The [[Run archive]] stores generated text and logprobs per coalition, not payoffs; [[Reanalyze]] re-scores them.

| Rank | Gap | User question | Data | Effort | Priority |
|---|---|---|---|---|---|
| 1 | Waterfall | How does each player move the score from `base_value` to `grand_value`? | present | small | high |
| 2 | Grouping by message | Which message matters, then which token inside it? | `message_index`, `speaker` | small | high |
| 3 | Shapley and Banzhaf side by side | Do the two estimands agree? | two runs, labelled `estimand` | small | medium |
| 4 | Interactive HTML | Show a token's value on click, as `shap.plots.text` does | present | small | medium |
| 5 | Many-conversation bar or beeswarm | Which speakers or roles matter across a dataset? | many outputs; players differ per run, so aggregate by role | medium | medium |
| 6 | Coalition and interaction views | Which pairs of players act together? | coalition masks and archived text, payoffs by `reanalyze` | large | medium |
| 7 | Force plot | Same answer as the waterfall | present | small | low |
| 8 | Per-generated-token attributions, output switching | Which input token drives each output token? | missing: the value function scores whole generations | large | blocked |

## Not worth porting

- `image`, `image_to_text`: mllm-shap has no pixel players.
- `scatter`, `partial_dependence`, `beeswarm` coloured by feature value: players are tokens without numeric feature values.
- `monitoring`, `embedding`, `group_difference`: tabular sample populations.
- `benchmark`: mllm-shap benchmarks live in `make bench`.
- Token clustering tree: partitioners emit flat players, so there is no tree to cut.

## Sources

- https://shap.readthedocs.io/en/latest/api.html
- https://github.com/shap/shap/tree/v0.53.0/shap/plots (`_bar.py`, `_text.py`, `_waterfall.py`, `_force.py`, `_decision.py`, `_heatmap.py`, `_scatter.py`, `_beeswarm.py`, `_violin.py`, `_embedding.py`, `_monitoring.py`, `_group_difference.py`, `_image.py`, `_benchmark.py`)
- https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html
- https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html
- https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/text_generation/Open%20Ended%20GPT2%20Text%20Generation%20Explanations.html
- https://pypi.org/project/shap/
