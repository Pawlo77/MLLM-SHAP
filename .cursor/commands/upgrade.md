# Upgrade

Run skill `upgrade`: branch `chore/upgrades-<date>` from `main`, `make upgrade`, checks,
one `build:` commit, then skill `ship` (routine, merges itself). $ARGUMENTS `--check`
only runs `make upgrade-check` and reports.
