# Vendored skills

Copied unedited from upstream; update by re-fetching the files at a newer commit and changing
the commit below. Only MIT or Apache-2.0 sources are vendored. The uv, ruff and ty skills come
from the required `astral` plugin, not from this directory.

| Skill | Upstream path | Commit | Licence |
|-------|---------------|--------|---------|
| caveman | [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) `skills/caveman/SKILL.md` | `0574b85a19ab7b2b7f42cc3b3838e20f24ab85eb` | MIT (skill surface at that commit) |
| ponytail, ponytail-review, ponytail-audit, ponytail-debt, ponytail-gain, ponytail-help | python-kit-template `.agents/skills/ponytail*` and `.cursor/rules/ponytail.mdc` | `09c434ab73ca43761c5d8b0c5d341926c2fba710` | see `ponytail/LICENSE` |

The caveman MIT notice (Copyright (c) 2026 Julius Brussee) is kept in `caveman/LICENSE`.

Local edits, the only exceptions to "unedited":

- `caveman/SKILL.md`: the frontmatter `description` is trigger-only (use only on /caveman, a level
  change or a question about the mode) so the body does not load every session; the always-on ultra
  rules live in `.cursor/rules/caveman.mdc`. The default level line says `ultra` instead of `full`
  (`AGENTS.md`).
