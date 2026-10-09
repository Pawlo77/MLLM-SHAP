---
title: "Snapshot"
type: concept
domain: domain
tags: [concept]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/nlp_shap/domain/conversation.py]]"
---

# Snapshot

A snapshot (`ConversationSnapshot`) is the frozen conversation under explanation: an ordered tuple of [[domain|Turn]] objects plus a `snapshot_id`. `from_turns` derives the id from a SHA-256 over every [[Message]] (role and text for text-only messages, every part otherwise), so rendering a coalition produces a new snapshot with a new id. `from_json` and `to_json` read and write the text form the command line uses.

Package: [[domain]].
