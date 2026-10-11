---
title: "viz"
type: module
domain: viz
tags: [architecture, viz]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/viz/__init__.py]]"
  - "[[src/mllm_shap/viz/audio.py]]"
  - "[[src/mllm_shap/viz/colors.py]]"
  - "[[src/mllm_shap/viz/labels.py]]"
  - "[[src/mllm_shap/viz/render.py]]"
  - "[[src/mllm_shap/viz/style.py]]"
  - "[[src/mllm_shap/viz/token_bar.py]]"
  - "[[src/mllm_shap/viz/token_text.py]]"
---

# viz

Part of [[mllm-shap]].

## Responsibility

- Render per-player attributions for people: inline coloured text and a sorted bar chart.

## Public surface

- `render_attribution(output, snapshot, player_set, renderer="token_text", title=None, registry=None)`: resolves a renderer from the registry (default `default_registry()`; pass `ExplainRunner.registry` to reuse custom renderers) and returns a matplotlib figure.
- `render_attribution_html(output, snapshot, player_set, title=None)`: HTML fragment for Jupyter; `display_attribution_html` wraps it in `IPython.display.HTML`.
- `TokenTextRenderer` (`token_text`, also `to_html`), `TokenBarRenderer` (`token_bar`); both take a `PluginContext`.
- `token_labels(snapshot, player_set)`: each span's `display_label` (the partitioner's label, else the text with a `"<speaker>: "` prefix) from `PlayerSet.spans`; raises without spans or when spans do not match the snapshot.
- `check_attribution(labels, values)`: the `ValueError` both renderers raise for unequal lengths or an empty attribution.
- `plot_audio_segments(audio, segments, values=None)`: a two-panel figure of [[alignment]] segments, with the waveform, segment shading (by `values` on the diverging scale when given), refined boundaries, dashed CTC edges with arrows to where refinement moved them, and the RMS energy below; not a renderer plugin, since it draws audio, not a player attribution table.
- `colors.py` (red-blue diverging scale), `style.py` (seaborn theme).

## Data flow

- In: `ExplainRunOutput` from the [[pipeline]] (its `player_set` holds the explained players), the snapshot.
- Out: figure or HTML string; matplotlib, seaborn and IPython are imported lazily (`viz` extra); `mllm_shap/__init__.py` exposes the three helpers lazily.

## Invariants

- Labels and values have equal length; empty attributions raise.
- Labels work for any partitioner that fills spans.

- `token_text` measures each token's rendered box (`Text.get_window_extent`), wraps lines at the axes width and sizes the figure height to the line count; `token_bar` offsets value labels in points on the outer side of each bar, widens the x range to hold them and puts the sign legend outside the axes on the right.
- Titles use `fontweight="bold"`; the fallback DejaVu Sans has no `semibold`.

## Known issues

- Fixed: [[Fixed issues|V01]].

## Tests

- [[tests/viz/test_render.py]]
- [[tests/viz/test_labels.py]]
- [[tests/viz/test_audio.py]]
