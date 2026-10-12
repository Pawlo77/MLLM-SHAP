# Shared make targets of python-kit-template. Set project variables in the Makefile
# before `include kit.mk`; add project targets after it. Never edit this file.
#
#   PYTHON_VERSION   uv python install            (default 3.12)
#   SYNC_ARGS        uv sync arguments            (default --all-groups)
#   TEST_ARGS        pytest arguments for `tests` (default: whole suite)
#   SKILLS_TESTS     tests behind `skills-check`
#   VAULT_CHECK_EXTRA  recipe lines `vault-check` runs after the shared checks

.DEFAULT_GOAL := help

PYTHON_VERSION ?= 3.12
SYNC_ARGS ?= --all-groups
HOOK_TYPES ?= --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
TEST_ARGS ?=
SKILLS_TESTS ?= tests/hooks/test_skill_frontmatter.py tests/hooks/test_agents.py
VAULT_ARGS ?= $(shell bash -c '. scripts/hooks/kit_conf.sh; kit_conf vault-check-args --quiet')

.PHONY: help install upgrade-check upgrade upgrade-pin clean prek prek-all tests agent-check \
	check vault-check knowledge rules agents configs skills-check skills-validate \
	obsidian-plugins

help: ## Show available targets
	@printf "Available targets:\n"
	@grep -hE '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Set up the development environment
	uv python install $(PYTHON_VERSION)
	uv sync $(SYNC_ARGS)
	@if [ "$$(git rev-parse --git-dir)" = "$$(git rev-parse --git-common-dir)" ]; then \
		uv run prek install $(HOOK_TYPES); \
		echo "prek hooks installed. Re-run after cloning."; \
	else \
		echo "Secondary worktree: git hooks are shared and tied to one venv; run 'make install' in the main checkout."; \
	fi

upgrade-check: ## List outdated packages, action pins, uv and Obsidian plugins
	uv run python scripts/upgrade.py --check

upgrade: ## Upgrade them all except the protected items
	uv run python scripts/upgrade.py --apply

upgrade-pin: ## Move the configured library pin to its branch head (a review PR)
	uv run python scripts/upgrade.py --pin

clean: ## Remove virtual environment and cache files; uninstalls the git hooks in the main checkout
	@if [ "$$(git rev-parse --git-dir)" = "$$(git rev-parse --git-common-dir)" ]; then \
		uv run prek uninstall || true; \
	fi
	rm -rf .venv/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d \( -name ".cache" -o -name "*__pycache__" -o -name ".*_cache" \) -exec rm -rf {} + 2>/dev/null || true

prek: ## Run git hook checks on changed files
	uv run prek run --fail-fast

prek-all: ## Run git hook checks on all files
	uv run prek run --all-files

tests: ## Run the test suite
	uv run pytest $(TEST_ARGS)

agent-check: ## Agent validate: tests + prek-all
	$(MAKE) tests
	$(MAKE) prek-all

check: ## Tests, all quality hooks and the vault check
	$(MAKE) agent-check
	$(MAKE) vault-check

vault-check: ## Vault consistency: notes, links, generated notes, brevity
	uv run python scripts/check_vault.py $(VAULT_ARGS)
	uv run python scripts/check_brevity.py
	@test ! -f scripts/build_knowledge.py || uv run python scripts/build_knowledge.py --check
	$(VAULT_CHECK_EXTRA)

knowledge: ## Rebuild the generated notes from the code
	@test ! -f scripts/build_knowledge.py || uv run python scripts/build_knowledge.py

rules: ## Regenerate .claude/rules/ from .cursor/rules/
	uv run python scripts/sync_rules.py

agents: ## Regenerate .codex/agents/*.toml from .claude/agents/*.md
	uv run python scripts/sync_agents.py

configs: ## Regenerate the agent tool configs (.claude, .codex, .cursor)
	uv run python scripts/sync_configs.py

skills-check: ## Skill and subagent frontmatter follows the Agent Skills spec
	uv run pytest -q -p no:cacheprovider $(SKILLS_TESTS)

skills-validate: ## Upstream Agent Skills validator on every skill (advisory)
	@for d in .agents/skills/*/; do uvx --from skills-ref==0.1.1 agentskills validate "$$d" || true; done

obsidian-plugins: ## Download the pinned community plugin releases into .obsidian/plugins
	uv run python scripts/install_obsidian_plugins.py
