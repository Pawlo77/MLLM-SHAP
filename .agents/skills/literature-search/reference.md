# Literature search reference

## Sources

| Source | Use | Limits |
|--------|-----|--------|
| OpenAlex | keyword search with a date filter; forward citations (`cites:`) of seeds | polite pool: set `LITERATURE_MAILTO` |
| arXiv API | field queries (`abs:`, `ti:`) newest first; `arxiv:comment` often names the venue | one request every 3 s |
| Semantic Scholar | forward citations of seeds; venue names | frequent 429s without a key; back off 30 s and more |
| Crossref | published title, authors, year, volume and pages of a DOI | none in practice |

Forward citations of 15–20 seeds find most core papers; keyword queries find the rest.
Add workshop proceedings (ACL Anthology, PMLR, CEUR) by hand.

## Query design

- One query per method family and domain pair.
- Include groups: method terms AND domain terms; method papers without a domain term
  need a pattern of their own.
- Excludes for known false-positive fields of the area.

## Full text

1. arXiv HTML (`https://arxiv.org/html/<id>`); it can be truncated, so check the section
   count and fall back to the PDF.
2. arXiv or open-access publisher PDF through `pypdf`.
3. Publisher pages that refuse scripts open in a browser.
4. Nothing readable: record the paper as unread; never write from the abstract.

## Metadata pitfalls

- arXiv and published titles differ: keep the published title as the first alias, the
  old name as a second alias, and drop `arxiv` when the titles disagree.
- Curly and straight apostrophes normalise differently; copy the exact title.
- The Crossref year can differ from the issue year printed in the PDF.
- Venues come from the arXiv comment, the proceedings page or Crossref, never memory.
- Search metadata truncates author lists; take authors from the arXiv record.

## Pacing

- About 50 notes fit one session: read the method, results and limitations, not every
  appendix; re-read the source for every number written.
- Report progress to the user every few notes.
