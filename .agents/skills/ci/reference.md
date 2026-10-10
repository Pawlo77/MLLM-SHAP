# Repository settings

GitHub settings the maintainer applies; agents propose these commands and never run them.
Branch protection is the real gate outside the agent tools.

## Branch protection on `main`

Required checks (job names): `prek`, `tests-passed`, `CodeQL`, `title`. Pull request and
conversation resolution required; no force pushes or deletions; linear history off (merge
commits). Run it after the `title` workflow is on `main`, or merges wait for a check that
never reports.

```bash
gh api -X PUT repos/Pawlo77/nlp-shap/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {
    "strict": false,
    "checks": [
      {"context": "prek"},
      {"context": "tests-passed"},
      {"context": "CodeQL"},
      {"context": "title"}
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 0,
    "dismiss_stale_reviews": false,
    "require_code_owner_reviews": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
JSON
gh api repos/Pawlo77/nlp-shap/branches/main/protection --jq '.required_status_checks.checks[].context'
```

## Other settings

```bash
gh api -X PATCH repos/Pawlo77/nlp-shap -F delete_branch_on_merge=true -F has_wiki=false \
  -F allow_squash_merge=false -F allow_rebase_merge=false
gh api -X PUT repos/Pawlo77/nlp-shap/pages -f build_type=workflow
gh api -X PUT repos/Pawlo77/nlp-shap/vulnerability-alerts
gh api -X PATCH repos/Pawlo77/nlp-shap --input - <<'JSON'
{"security_and_analysis": {"secret_scanning": {"status": "enabled"},
 "secret_scanning_push_protection": {"status": "enabled"}}}
JSON
gh secret set CODECOV_TOKEN
```

- No inline `#` comments: interactive zsh passes them to `gh` as arguments.
- Pages takes `-X POST` when it is off. `CODECOV_TOKEN` is optional.
- Code scanning runs from `codeql.yaml` (advanced setup) and needs a public repository
  or GitHub Code Security; leave the default setup off.
- Discussions stay off (`has_discussions=true` enables them).
