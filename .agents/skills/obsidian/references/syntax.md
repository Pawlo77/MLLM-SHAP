# Syntax reference

Write from these patterns; scoping rules are in SKILL.md.

## Markdown

`[[Note#Heading]]`, `[[Note#^id]]` (block: ` ^id` ends the paragraph), `[[Note|text]]`,
`![[Note#Section]]`. Callouts `> [!abstract|example|warning|note|tip]`,
`-` suffix folds. Frontmatter: quote values with `:`, lists as YAML lists, `aliases:`,
`tags:` without `#`. `%% comment %%`, `==highlight==`, `$…$`/`$$…$$` (MathJax).

## Dataview (DQL)

No DataviewJS or inline JS.

```dataview
TABLE WITHOUT ID file.link AS Log, domain AS Domain, updated AS Updated
FROM "notes/decisions"
WHERE type = "decision-log" AND domain != "process"
SORT updated DESC
```

- `[[Note]]` in FROM = notes linking *to* it; `outgoing([[Note]])` = its targets.
- After `GROUP BY`, pages are `rows` (`rows.file.link`, `length(rows)`); `FLATTEN` = one row
  per list element.
- Dashed keys: bare (`!related-code`) or `row["related-code"]`; subtraction needs spaces.
- Missing field = `null`: `WHERE !field`. Inline: `` `= this.year` ``.
- Backlink count: `LIST length(file.inlinks) … SORT length(file.inlinks) ASC`.
- Logs per module: `TABLE WITHOUT ID code AS Module, rows.file.link AS Logs …
  FLATTEN row["related-code"] AS code GROUP BY code`.

## Bases

`.base` YAML file or ```` ```base ```` block; filters are expression strings.

```yaml
filters:
  and:
    - file.inFolder("notes/decisions")
    - 'type == "decision-log"'
properties: {domain: {displayName: Domain}}
views:
  - type: table
    name: Decision logs
    order: [file.name, domain, updated]
    sort: [{property: updated, direction: DESC}]
  - type: cards
    name: Cards
```

- Expressions: `file.hasTag/hasLink("…")`, bare property or `note.<name>`, `== != && ||`.
  `formulas:` → `formula.<name>` in `order`.
- Table edits by the user land in frontmatter. If a view won't render, fall back to Dataview.

## Canvas

JSON Canvas; pixels, `y` down. Grid (400 px columns, 200 px rows), no overlaps.

```json
{
  "nodes": [
    {"id": "g1", "type": "group", "x": -20, "y": -60, "width": 440, "height": 300, "label": "Estimators"},
    {"id": "n1", "type": "file", "file": "notes/decisions/<Domain> decisions.md", "x": 0, "y": 0, "width": 380, "height": 200},
    {"id": "t1", "type": "text", "text": "**Complementary contribution**", "x": 0, "y": 320, "width": 380, "height": 100, "color": "4"}
  ],
  "edges": [{"id": "e1", "fromNode": "n1", "fromSide": "bottom", "toNode": "t1", "toSide": "top", "label": "→"}]
}
```

- Types: `text`, `file` (+ `subpath` `"#<heading>"`), `link` (`url`), `group` (list groups
  first). `color` `"1"`–`"6"` (red→purple) or `"#rrggbb"`. `toEnd: "none"` = undirected.
- Renaming a note outside Obsidian breaks file nodes: update `file` paths too.

## Mermaid

- `flowchart LR|TD`; dashed `-.->` for optional or side paths (the run archive); `-- label -->`.
- Plain-word ids, quote labels with spaces/symbols. No LaTeX (`beta_1`, not `$\beta_1$`).
- Also fit: `sequenceDiagram` (scheduler and backend calls), `stateDiagram-v2`, `timeline`.

## Templater

`<% tp.file.title %>`,
`<% tp.date.now("YYYY-MM-DD") %>`, `<% tp.file.cursor() %>`, `<% tp.system.prompt("Package?") %>`;
`<%* … %>` runs JS, `tR += "text"` appends.
