# Knowledge agents

- The repository root is the Obsidian vault `mllm-shap`; its map is `mllm-shap.md`. Write notes only through skill `notes` (paths, frontmatter, writing rules: its `reference.md`); vault machinery: skill `obsidian`.
- `generated/` is rebuilt by `make knowledge`, never edited; `_templates/` holds Templater templates, not notes.
- A change under `src/mllm_shap/<pkg>/` updates `architecture/<pkg>.md` (bump `updated`) in the same change.
- A design choice appends a section to `decisions/<Domain> decisions.md`; a bug found or fixed adds or updates its note in `issues/`.
- A paper is one section of `Papers.md`: DOI or arXiv links, never PDFs.
- No secrets, tokens, private hostnames or personal data; spell out each abbreviation at its first use (`scripts/glossary.yaml`).
- `make vault-check` passes before hand-off.
