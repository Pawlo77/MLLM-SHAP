---
title: "{{SHORT NAME}}"
aliases: ["{{OFFICIAL TITLE}}"]
authors: ["{{FULL NAME}}", "{{FULL NAME}}"]
year: {{YEAR}}
venue: "{{VENUE WITH VOLUME AND PAGES}}"
arxiv: "{{ARXIV ID}}"
doi: "{{DOI}}"
url: "{{URL OF PUBLISHED VERSION, ELSE ARXIV}}"
code: "{{CODE URL, OMIT IF NOT RELEASED}}"
supplement: "{{SUPPLEMENT URL, OMIT IF NONE}}"
citekey: {{firstauthor}}{{year}}{{firstword}}
verified: {{YYYY-MM-DD}}
reviewed: {{YYYY-MM-DD, SET BY THE REVIEWER ONLY}}
theme: "[[{{THEME}}]]"
tags: [type/paper, theme/{{theme-slug}}, topic/{{topic}}]
---
# {{SHORT NAME}}

> [!info] TL;DR
> {{What the paper changes and its main result, 3–5 plain sentences.}}

## Motivation

- {{The gap, with the prior work linked: [Author et al., YEAR](verified URL) or a vault note.}}
- {{Why it matters.}}

## Method

{{Notation sentence, renamed to the notation in reference.md.}}

**{{Definition or step}}.** {{Statement.}}

$$
{{key equation}}
$$

Intuition: {{one sentence in plain words}}.

**{{Theorem or proposition}}.** {{Statement with its assumptions.}}

## Implementation

- **Code:** {{URL or "not released"}}.
- **Software and hardware:** {{frameworks, versions if given, GPUs or chips}}.
- **Architecture:** {{layers, widths, parameters, neuron model and parameters, surrogate, time steps}}.
- **Training:** {{optimiser, learning rate and schedule, batch size, epochs, budget}}.

## Data

- {{Dataset}}: {{what it is}}, {{size and splits}}, {{encoding or preprocessing}}.

## Experiments

- **{{Name}}:** {{setup, baselines and metrics}}; {{what was found}}.

## Results

- {{Result in numbers}} ({{setting}}; {{baseline}}).

## Limitations

- {{Failure mode}} (authors).
- {{Failure mode}} (ours).

## Relation to our project

- {{What we take, test or avoid}}; {{link to the project note}}.

## Future work

- **Done:** {{direction the paper names}}; [{{Later author}} et al., {{YEAR}}]({{verified URL}}).
- **Open:** {{gap no later work closes}} (searched {{YYYY-MM}}).
