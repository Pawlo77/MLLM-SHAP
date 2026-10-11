---
title: "domain"
type: module
domain: domain
parent: "[[mllm-shap]]"
tags: [architecture, domain]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/domain/__init__.py]]"
  - "[[src/mllm_shap/domain/coalition.py]]"
  - "[[src/mllm_shap/domain/config.py]]"
  - "[[src/mllm_shap/domain/conversation.py]]"
  - "[[src/mllm_shap/domain/enums.py]]"
  - "[[src/mllm_shap/domain/estimands.py]]"
  - "[[src/mllm_shap/domain/game.py]]"
  - "[[src/mllm_shap/domain/generation.py]]"
  - "[[src/mllm_shap/domain/players.py]]"
---

# domain

## Responsibility

- Immutable value types shared by every other package: conversation input, players and their spans, coalition masks, generations, estimand labels, and the generation and runtime settings backends and the runtime read.
- No input or output and no backend.

## Public surface

- `TextPart(text)`, `AudioPart(source, sample_rate)` (`source` is a path or bytes; `data()` reads it), `ContentPart` union ([[Message]]).
- `Message(role, text=... | content=..., speaker=None)`: `content` is a tuple of parts; `text` joins the text parts with spaces; `with_text(text)` replaces the text parts and keeps audio and speaker; `with_audio(audio)` replaces the first audio part.
- `Message.speaker` must be a non-empty single line ([[Domain decisions#2026-10-08 — Speaker rendered into the message content|speaker]]).
- `Turn(messages)`; `ConversationSnapshot(turns, snapshot_id)` ([[Snapshot]]): `from_turns` derives `snapshot_id` (first 16 hex characters of a SHA-256 over every message; text-only messages hash role and text as before 0.1.18, other messages hash every part, audio by the SHA-256 of its bytes; a speaker is hashed only when set); `messages()` flattens the turns.
- `ConversationSnapshot.from_json(text)` / `to_json(indent=2)` with the `SnapshotPayload`, `TurnPayload`, `MessagePayload` TypedDicts: `{"turns": [{"messages": [{"role", "text", "speaker"?}]}]}` plus an optional `snapshot_id` when it is not the content-derived one; multi-part and audio messages are refused; shape errors raise `TypeError`, bad JSON, roles or empty turns raise `ValueError`.
- `PlayerSpan(message_index, start, end, text, speaker=None, label=None, samples=None)` with `display_label`; `samples` is the `[start, end)` sample range of an audio player ([[Player span]]); `PlayerSet(player_ids, spans=())` with `num_players`.
- `CoalitionMask(present)`: `from_sequence`, `coalition_size()`, `validate_against(player_set)` ([[masking]]).
- `GenerationRecord`: `text`, `text_token_rows`, `audio_token_rows`, `logprobs` (`None` when the backend returned none), `embedding`, `contextual_embedding`.
- `domain/config.py`: `GenerationConfig(max_new_tokens=128, temperature=0.0, top_k=1, compute_embeddings=False)`, `DedupConfig(enabled="auto")`, `KvCacheConfig(enabled=False, max_entries=8)`; also importable from `mllm_shap.pipeline.config`.
- `Estimand` (`shapley`, `banzhaf`), `EstimandWire`, `estimand_to_wire` ([[estimation|Estimand]]); `Role` (`user`, `assistant`, `system`) and `RoleName`, the same values as a `Literal` for plugin options (`roles`, `text_roles`); `EmbeddingMode` (`static`, `contextual`); `CooperativeGame(player_set)`.

## Data flow

- Imports nothing from the package except pydantic models in `config.py`; every other package imports it, and [[runtime]] reads its settings from here instead of from [[pipeline]].

## Invariants

- Every type is frozen; snapshots, turns, player sets and masks reject empty construction.
- Snapshot ids of text-only conversations did not change in 0.1.18; two snapshots with the same text and different audio get different ids.
- `PlayerSet.spans`, when present, has one span per player id, in player order.

## Known issues

- Audio parts reach the `transformers` audio models, the `api` backend and `mock_audio` ([[backends]]).

## Tests

- [[tests/domain/test_conversation.py]]
- [[tests/domain/test_message_content.py]]
- [[tests/domain/test_players.py]]
- [[tests/domain/test_coalition.py]]
- [[tests/domain/test_game.py]]
- [[tests/domain/test_estimands.py]]
- [[tests/domain/test_enums.py]]
- [[tests/domain/test_imports.py]]
