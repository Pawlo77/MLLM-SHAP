---
title: "<% tp.file.title.split(' - ').slice(1).join(' - ') %>"
type: solution
domain: <% tp.system.prompt("Package or area?") %>
parent: "[[mllm-shap]]"
created: <% tp.date.now("YYYY-MM-DD") %>
tags: [solution]
pr: "<% tp.system.prompt("Pull request URL?") %>"
plan-task: "<% tp.system.prompt("Plan task (plans/<file>.md, Task n.m)?") %>"
related-code: []
---

# <% tp.file.title.split(' - ').slice(1).join(' - ') %>

Solved in the pull request above; plan: <% "[" + "[<plan file>]" + "]" %>.

## Problem

- <% tp.file.cursor() %>

## What was tried

- Attempt: why it failed, with the command or test that showed it.

## What worked

- The change that fixed it, with the file and function.

## Reusable rule

- One rule a future change can apply without reading this note.
