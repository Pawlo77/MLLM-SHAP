---
title: "Player span"
type: concept
domain: domain
tags: [concept]
created: 2026-10-08
related-code:
  - "[[src/nlp_shap/domain/players.py]]"
  - "[[src/nlp_shap/masking/tokens.py]]"
---

# Player span

A player span (`PlayerSpan(message_index, start, end, text, speaker=None, label=None)`) records where one [[domain|Player]] sits: the index of its message in the flattened snapshot and its character range in that message's text. `display_label` is the partitioner's `label`, else the text with a `"<speaker>: "` prefix. The [[masking|Partition]] fills one span per player in `PlayerSet.spans`; absence policies replace exactly those ranges and keep the rest of the text, and the renderers read labels from them, so nothing re-tokenises.

Package: [[domain]].
