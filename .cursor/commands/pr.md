# PR

Propose a pull request — do **not** create or push until the user approves.

1. Ensure `make agent-check` is green (docs/notebooks if needed)
2. `git status` · `git diff main...HEAD` · `git log main..HEAD`
3. Draft title (commit style) + body:

```markdown
## Summary
- …

## Test plan
- [ ] make agent-check
- [ ] make docs (if API/docs changed)
- [ ] make notebooks (if examples changed)
- [ ] gh run watch — remote workflows green
```

4. Ask user to approve create; only then `git push -u` + `gh pr create`
   - **Never WIP:** no `--draft`, no `WIP`/`[WIP]` in title, no WIP label — ready-for-review only
5. **Always** assign + label (never skip):

```bash
gh pr edit <n> --add-assignee Pawlo77 --add-label <label>
```

| Title prefix | Label |
|--------------|-------|
| `feat:` / `perf:` | `enhancement` |
| `fix:` | `bug` |
| `docs:` | `documentation` |
| `chore:` / `ci:` / `build:` / `refactor:` / `test:` | `enhancement` |
