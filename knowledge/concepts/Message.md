---
title: "Message"
type: concept
domain: domain
parent: "[[mllm-shap]]"
tags: [concept]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/domain/conversation.py]]"
  - "[[src/mllm_shap/domain/enums.py]]"
---
# Message

A message is one turn element with a role (`user`, `assistant` or `system`), an optional `speaker` name and ordered content parts: `TextPart` and `AudioPart`. `Message.text` joins the text parts; the token partition splits it into [[domain|Player]] objects, the `messages` partition makes it one. Backends send a speaker as `"<speaker>: <text>"` ([[Domain decisions#2026-10-08 — Speaker rendered into the message content|speaker rendering]]). Audio parts reach the backends that hear audio ([[backends]]).

Package: [[domain]].
