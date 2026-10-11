---
title: "<% tp.file.title %>"
type: module
domain: <% tp.file.title %>
parent: "[[mllm-shap]]"
tags: [architecture, <% tp.file.title %>]
created: <% tp.date.now("YYYY-MM-DD") %>
updated: <% tp.date.now("YYYY-MM-DD") %>
related-code:
  - "<% '[' + '[src/mllm_shap/' + tp.file.title + '/__init__.py]' + ']' %>"
---

# <% tp.file.title %>

## Responsibility

- <% tp.file.cursor() %>

## Public surface

- `Name(args)`: one line per class or function users or other packages call.

## Data flow

- In: what comes in and from which package.
- Out: what goes out and to which package.

## Invariants

- What must stay true after every change.

## Known issues

- <% "[" + "[<issue note>]" + "]" %>

## Tests

- <% '[' + '[tests/' + tp.file.title + '/test_<module>.py]' + ']' %>
