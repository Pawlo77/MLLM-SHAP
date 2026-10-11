---
title: "<% tp.file.title %>"
type: issue
domain: <% tp.system.prompt("Package?") %>
parent: "[[mllm-shap]]"
status: open
severity: <% tp.system.suggester(["blocker", "major", "minor"], ["blocker", "major", "minor"]) %>
found: <% tp.date.now("YYYY-MM-DD") %>
created: <% tp.date.now("YYYY-MM-DD") %>
fixed_in: ""
tags: [issue]
related-code: []
---

# <% tp.file.title %>

Package: <% "[" + "[<package architecture note>]" + "]" %>.

## Symptom

- <% tp.file.cursor() %>

## Cause

- The mechanism, with the file and function.

## Repro

```python
# smallest snippet that shows the symptom
```

## Fix

- Plan task or pull request; release version once fixed.
