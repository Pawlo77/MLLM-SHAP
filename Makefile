.DEFAULT_GOAL := help

.PHONY: help install upgrade-check upgrade clean prek prek-all check agent-check tests tests-all coverage bench bench-regression docs notebooks build knowledge vault-check graph obsidian-plugins rules kernel bench-baseline mutate agents skills-validate practical

BENCH_ARGS := -m bench -p no:cacheprovider --benchmark-min-rounds=5 \
	--benchmark-warmup=on --benchmark-warmup-iterations=10

help: ## Show available targets
	@printf "Available targets:\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-17s %s\n", $$1, $$2}'

install: ## Set up the development environment
	uv python install 3.12
	uv sync --all-groups --all-extras
	uv run prek install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
	@echo "prek hooks installed (pre-commit, commit-msg, pre-push). Re-run after cloning."

kernel: ## Register .venv as the Jupyter kernel "mllm-shap (uv)"
	uv run python -m ipykernel install --user --name mllm-shap --display-name "mllm-shap (uv)"

upgrade-check: ## List outdated packages, hooks, action pins, uv and Obsidian plugins
	uv run python scripts/upgrade.py --check

upgrade: ## Upgrade them all except the protected publish.yaml pins
	uv run python scripts/upgrade.py --apply

clean: ## Remove virtual environment and cache files
	rm -rf .venv/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d \( -name ".cache" -o -name "*__pycache__" -o -name ".*_cache" \) -exec rm -rf {} + 2>/dev/null || true

prek: ## Run git hook checks on changed files
	uv run prek run --fail-fast

prek-all: ## Run git hook checks on all files
	uv run prek run --all-files

tests: ## Unit tests (skip lms/gpu/bench/slow/practical)
	uv run pytest -m "not lms and not gpu and not bench and not slow and not practical" tests/

FEATURE ?= all
TIER ?= quick
# LM Studio key for the backend.lmstudio and backend.api scenarios (environment wins).
MLLM_SHAP_PRACTICAL_LMS_MODEL ?= qwen2-500m-instruct
export MLLM_SHAP_PRACTICAL_LMS_MODEL

practical: ## Practical tests on real models (GPU); TIER=quick|deep|all FEATURE=<id|glob|all>
	uv run pytest tests/practical -m practical --tier='$(TIER)' --feature='$(FEATURE)' -rs

tests-all: ## All tests including optional markers
	uv run pytest tests/

agent-check: ## Agent validate: filtered pytest + prek-all
	$(MAKE) tests
	$(MAKE) prek-all

check: ## Filtered tests, all quality hooks and the vault check
	$(MAKE) agent-check
	$(MAKE) vault-check

coverage: ## Run tests with coverage report (fails under 85%, see pyproject)
	uv run pytest --cov=./src/mllm_shap --cov-report=term-missing -m "not lms and not gpu and not bench and not slow and not practical" tests/

bench: ## Run benchmarks; JSON report in .pytest_benchmark/benchmark.json
	uv run pytest tests/benchmarks/ $(BENCH_ARGS) --benchmark-json=.pytest_benchmark/benchmark.json

bench-regression: ## Run benchmarks and fail when a median is >25% slower than baseline.json
	uv run pytest tests/benchmarks/ $(BENCH_ARGS) --benchmark-json=.pytest_benchmark/benchmark.json \
		--benchmark-compare=tests/benchmarks/baseline.json --benchmark-compare-fail=median:25%

bench-baseline: ## Re-record tests/benchmarks/baseline.json on this machine
	rm -rf .pytest_benchmark/storage
	uv run pytest tests/benchmarks/ $(BENCH_ARGS) \
		--benchmark-storage=file://.pytest_benchmark/storage --benchmark-save=baseline
	cp .pytest_benchmark/storage/*/0001_baseline.json tests/benchmarks/baseline.json

docs: ## Build the documentation (strict) and check external links
	uv run make -C docs html
	uv run make -C docs linkcheck

notebooks: ## Execute every example notebook in place (outputs are committed)
	@set -e; for nb in examples/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute "$$nb" --inplace; \
	done

mutate: ## Mutation testing of the estimation package (slow, on demand, not CI)
	uv run mutmut run
	uv run mutmut results

build: ## Build source and wheel distributions
	uv build

knowledge: ## Rebuild knowledge/generated and the .claude/jit-context entries from the package
	uv run python scripts/build_knowledge.py

vault-check: ## Vault consistency: notes, links, tags, generated notes, brevity
	uv run python scripts/check_vault.py --strict-tags
	uv run python scripts/check_brevity.py
	uv run python scripts/build_knowledge.py --check
	uv run python docs/scripts/check_docs.py

graph: ## Offline Code Graph report; ARGS="--codemap" or ARGS="--neighbors <path|note>"
	uv run python scripts/build_knowledge.py --graph $(ARGS)

obsidian-plugins: ## Download the pinned community plugin releases into .obsidian/plugins
	uv run python scripts/install_obsidian_plugins.py

rules: ## Regenerate .claude/rules/ from .cursor/rules/
	uv run python scripts/sync_rules.py

agents: ## Regenerate .codex/agents/*.toml from .claude/agents/*.md
	uv run python scripts/sync_agents.py

skills-validate: ## Upstream Agent Skills validator on every skill (advisory)
	@for d in .agents/skills/*/; do uvx --from skills-ref==0.1.1 agentskills validate "$$d" || true; done
