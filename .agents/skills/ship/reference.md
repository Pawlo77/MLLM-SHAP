# Ship reference

Review-thread commands (owner `Pawlo77`, repository `MLLM-SHAP`, pull request `<n>`).

Unresolved threads:

```bash
gh api graphql -f query='query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){pullRequest(number:$n){reviewThreads(first:100){nodes{id isResolved path line comments(first:10){nodes{id author{login} body}}}}}}}' -F o=Pawlo77 -F r=MLLM-SHAP -F n=<n>
```

Reply to a comment, resolve a thread (`resolveReviewThread` mutation with the thread `id`) and re-request Copilot:

```bash
gh api -X POST repos/Pawlo77/MLLM-SHAP/pulls/<n>/comments/<id>/replies -f body=…
gh api -X POST repos/Pawlo77/MLLM-SHAP/pulls/<n>/requested_reviewers -f 'reviewers[]=copilot-pull-request-reviewer[bot]'
```
