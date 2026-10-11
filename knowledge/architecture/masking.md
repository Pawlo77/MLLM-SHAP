---
title: "masking"
type: module
domain: masking
tags: [architecture, masking]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/masking/__init__.py]]"
  - "[[src/mllm_shap/masking/audio.py]]"
  - "[[src/mllm_shap/masking/builder.py]]"
  - "[[src/mllm_shap/masking/codec.py]]"
  - "[[src/mllm_shap/masking/partitions.py]]"
  - "[[src/mllm_shap/masking/policies.py]]"
  - "[[src/mllm_shap/masking/space.py]]"
  - "[[src/mllm_shap/masking/tokens.py]]"
---

# masking

Part of [[mllm-shap]].

## Responsibility

- Turn a snapshot into players with spans (the token Partition) and a coalition mask into a rendered snapshot (the Absence policy).
- Pack masks into bytes and hash them.

## Public surface

- `TokenPartitioner` (`tokens`): one player per whitespace token of every message; returns `PlayerSet(player_ids, spans)` built by `tokens.token_spans` (spans carry the message's speaker); player ids `"{snapshot_id}:{turn}:{message}:{token}"`.
- `MessagePartitioner` (`messages`): one player per non-empty message of the roles in `MessagesOptions.roles` (default all); span = whole text, label `"<role>[/<speaker>] <n>: "` plus the first `LABEL_PREVIEW_CHARS` (30) characters; ids `"{snapshot_id}:{turn}:{message}"` ([[Masking decisions#2026-10-08 — Message players keep their slot when deleted|message players]]).
- `DeletePolicy` (`delete`): removes absent spans together with the whitespace that separated them.
- `PadPolicy` (`pad`): replaces absent spans with `placeholder` (default `[MASK]`, plugin option `PadOptions.placeholder`).
- `NeutralPolicy` (`neutral`): replaces each absent span with `fill_char` (default `_`, plugin option `NeutralOptions.fill_char`) repeated to the span's width.
- All three take a `PluginContext` first and read only `PlayerSet.spans`; a player set without spans raises. For spans with `samples` (audio players, [[alignment]]) `audio.render_audio` also edits the message's first `AudioPart`: `delete` cuts absent samples, `pad` and `neutral` replace them with silence of the same length ([[Masking decisions#2026-10-10 — Audio players are sample ranges on the player span|audio players]]).
- `tokens.py`: `token_spans(snapshot)`, `render_spans(snapshot, spans, replacements)` (replaces character ranges per message, `None` deletes, text outside every span is kept).
- `MaskBuilder(policy)`: `view(snapshot, players, mask) -> MaskedSnapshot` (no copy), `render(masked) -> ConversationSnapshot`.
- `MaskCodec`: `pack` to little-endian bytes, `unpack`, `hash_mask` (64-bit `blake2b` over the packed bytes and bit count, stable across processes).
- `MaskSpace`: projects a mask over explainable positions into a full mask; not used by the pipeline.

## Data flow

- In: `ConversationSnapshot` and `CoalitionMask` from [[domain]]; the `PlayerSet` with spans from the partitioner.
- Out: one rendered `ConversationSnapshot` per coalition, rendered lazily by the [[pipeline]] orchestrator as the scheduler consumes jobs.
- Policies never re-tokenise: any partitioner that fills spans (messages, sentences, custom units) works with every policy and with `viz` labels; under `delete` a message player leaves its message with empty text.

## Invariants

- Rendered snapshots keep the turn and message structure and the original whitespace outside deleted spans; only span text changes.
- A rendered snapshot gets a new `snapshot_id` (rebuilt through `ConversationSnapshot.from_turns`).
- Under `delete`, different masks can render the same text (`"a a a"` with one token kept); coalition keys are built from the mask, not the rendered text.

## Known issues

- None open. [[Fixed issues|C09]] was fixed in 0.1.17.

## Tests

- [[tests/masking/test_partitions.py]]
- [[tests/masking/test_custom_partitioner.py]]
- [[tests/masking/test_policies.py]]
- [[tests/masking/test_tokens.py]]
- [[tests/masking/test_builder.py]]
- [[tests/masking/test_codec.py]]
- [[tests/masking/test_space.py]]
