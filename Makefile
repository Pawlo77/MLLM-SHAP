# MLLM-SHAP. Shared targets: kit.mk (python-kit-template). This file sets its variables
# and adds the targets that only this repository has.
SYNC_ARGS := --all-groups --all-extras
TEST_ARGS := -m "not lms and not gpu and not bench and not slow and not practical" tests/
VAULT_CHECK_EXTRA := uv run python docs/scripts/check_docs.py

include kit.mk

.PHONY: kernel practical tests-all coverage bench bench-regression bench-baseline docs notebooks mutate build graph

BENCH_ARGS := -m bench -p no:cacheprovider --benchmark-min-rounds=5 \
	--benchmark-warmup=on --benchmark-warmup-iterations=10

kernel: ## Register .venv as the Jupyter kernel "mllm-shap (uv)"
	uv run python -m ipykernel install --user --name mllm-shap --display-name "mllm-shap (uv)"

FEATURE ?= all
TIER ?= quick
# LM Studio key for the backend.lmstudio and backend.api scenarios (environment wins).
MLLM_SHAP_PRACTICAL_LMS_MODEL ?= qwen2-500m-instruct
export MLLM_SHAP_PRACTICAL_LMS_MODEL

practical: ## Practical tests on real models (GPU); TIER=quick|deep|all FEATURE=<id|glob|all>
	uv run pytest tests/practical -m practical --tier='$(TIER)' --feature='$(FEATURE)' -rs

tests-all: ## All tests including optional markers
	uv run pytest tests/

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

graph: ## Offline Code Graph report; ARGS="--codemap" or ARGS="--neighbors <path|note>"
	uv run python scripts/build_knowledge.py --graph $(ARGS)
