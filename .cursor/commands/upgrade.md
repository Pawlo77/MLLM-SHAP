# Upgrade

Run skill `upgrade`: branch `chore/upgrades-<date>` from `main`, `make upgrade`, checks,
one `build:` commit, then skill `ship` (routine, merges itself). $ARGUMENTS `--check`
<<<<<<< before updating
only runs `make upgrade-check` and reports.
=======
only runs `make upgrade-check` and reports; `--pin` moves the configured library pin on
its own review branch (see the skill's `local.md`).
>>>>>>> after updating
