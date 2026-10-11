---
title: "Masking decisions"
type: decision-log
domain: masking
parent: "[[mllm-shap]]"
created: 2026-10-08
updated: 2026-10-10
tags: [decisions, masking]
related-code:
  - "[[src/mllm_shap/masking/partitions.py]]"
  - "[[src/mllm_shap/masking/policies.py]]"
---

# Masking decisions

Architecture: [[masking]]. Related: [[Player span]].


## 2026-10-10 — Audio players are sample ranges on the player span

Decision: An audio player is a `PlayerSpan` with `samples` set: the word's text range in the transcript plus its sample range in the message's first `AudioPart`. The existing absence policies edit both; `delete` cuts samples, `pad` and `neutral` silence them. The earlier branch's `TokenFilter` classes were dropped, since words the model cannot spell get no player anyway.

Why: The audio-track branch built players by hand outside the plugin registry and rebuilt messages with its own audio model, so audio players could not be configured or archived like text players.

Consequences: `explanation.players: sgpa_segments` works with every policy, estimator and the run archive; transcripts and audio stay masked together. Rejected: a separate audio absence policy (duplicates the delete, pad and neutral choice).
## 2026-10-08 — Message players keep their slot when deleted

Decision: The `messages` partition makes each non-empty message one player whose span is its whole text. Under `delete` an absent message keeps its place with empty text; `pad` and `neutral` replace the text. Which roles become players is the plugin option `roles`, not a top-level `players_roles` key.

Why: Message-level attribution reuses every absence policy and label through `PlayerSet.spans`. Removing absent messages leaves the empty coalition with no message, which `ConversationSnapshot` rejects and neither the `api` backend nor the plain-text prompt fallback can send.

Consequences: The model sees an empty turn of that role, as it does for a message whose tokens are all deleted. A new top-level key would enter the configuration digest and change every run id; a plugin option does not.
