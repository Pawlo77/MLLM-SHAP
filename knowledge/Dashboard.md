---
title: "Dashboard"
type: dashboard
domain: index
parent: "[[mllm-shap]]"
tags: [moc, dashboard]
created: 2026-10-07
updated: 2026-10-08
---

# Dashboard

Live tables built by the Dataview plugin from note frontmatter; they render only in Obsidian. The embedded generated notes at the end render everywhere.

## Issues by status and severity

```dataview
TABLE WITHOUT ID file.link AS Issue, severity AS Severity, status AS Status, fixed_in AS "Fixed in"
FROM "knowledge/issues"
WHERE type = "issue"
SORT status ASC, severity ASC, file.name ASC
```

## Open issues by package

```dataview
TABLE WITHOUT ID domain AS Package, length(rows) AS Open, rows.file.link AS Issues
FROM "knowledge/issues"
WHERE type = "issue" AND (status = "open" OR status = "deferred")
GROUP BY domain
SORT domain ASC
```

## Recently fixed

```dataview
TABLE WITHOUT ID file.link AS Issue, fixed_in AS "Fixed in", severity AS Severity
FROM "knowledge/issues"
WHERE type = "issue" AND status = "fixed"
SORT fixed_in DESC, file.name ASC
```

## Decision logs

```dataview
TABLE WITHOUT ID file.link AS Log, domain AS Domain, updated AS Updated
FROM "knowledge/decisions"
WHERE type = "decision-log"
SORT updated DESC, domain ASC
```

## Modules by domain

```dataview
TABLE WITHOUT ID domain AS Domain, file.link AS "Architecture note", length(row["related-code"]) AS Modules
FROM "knowledge/architecture"
WHERE type = "module"
SORT domain ASC
```

## Concepts without code

```dataview
LIST
FROM "knowledge/concepts"
WHERE type = "concept" AND !related-code
SORT file.name ASC
```

## Plans by status

```dataview
TABLE WITHOUT ID file.link AS Plan, status AS Status, created AS Created
FROM "plans"
WHERE type = "plan"
SORT created DESC
```

## Solved problems

```dataview
TABLE created AS "Date", domain AS "Area", pr AS "PR"
FROM "knowledge/handbook/Solutions"
WHERE type = "solution"
SORT created DESC
```

## Generated

![[Open issues]]

![[Decisions]]

![[Release history]]

![[Benchmarks]]

Code tables: [[Modules]], [[Public API]], [[Config keys]], [[Plugins]]. Cited papers: [[Papers]].
