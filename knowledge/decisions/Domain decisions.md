---
title: "Domain decisions"
type: decision-log
domain: domain
parent: "[[mllm-shap]]"
created: 2026-10-08
updated: 2026-10-08
tags: [decisions, domain]
related-code:
  - "[[src/mllm_shap/domain/conversation.py]]"
  - "[[src/mllm_shap/domain/players.py]]"
  - "[[src/mllm_shap/backends/_chat.py]]"
---
# Domain decisions

Architecture: [[domain]]. Related: [[backends]], [[Message]].

## 2026-10-08 — Speaker rendered into the message content

Decision: `Message.speaker` is an optional non-empty single-line name. It is hashed into the snapshot id and written as `"speaker"` in snapshot JSON only when set. Backends send `"<speaker>: <text>"` under the unchanged role (`message_chat_text`); a message emptied by `delete` is sent empty. `PlayerSpan.speaker` carries it to labels and `to_dict`.

Why: Conversations where several people share the `user` role need to say who wrote what, and chat templates have no speaker slot.

Consequences: Snapshot ids and outputs without speakers do not change. Rejected: one role per speaker (templates accept only `user`, `assistant` and `system`); the OpenAI message `name` field (Hugging Face templates and the LM Studio SDK drop it, so backends would see different prompts).
