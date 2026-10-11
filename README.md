<div align="center">
  <h1>mllm-shap</h1>
  <p><strong>Shapley-value attributions for conversational language models.</strong></p>
  <p>Explicit estimands (Shapley vs Banzhaf), reproducible run archives, pluggable backends (Hugging Face, LM Studio, OpenAI-compatible APIs).</p>
  <p>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/pypi/v/mllm-shap.svg" alt="PyPI"></a>
    <a href="https://github.com/Pawlo77/MLLM-SHAP/actions/workflows/test.yaml"><img src="https://github.com/Pawlo77/MLLM-SHAP/actions/workflows/test.yaml/badge.svg" alt="Pytest"></a>
    <a href="https://github.com/Pawlo77/MLLM-SHAP/actions/workflows/prek.yaml"><img src="https://github.com/Pawlo77/MLLM-SHAP/actions/workflows/prek.yaml/badge.svg" alt="Prek Checks"></a>
    <a href="https://codecov.io/gh/Pawlo77/MLLM-SHAP"><img src="https://codecov.io/gh/Pawlo77/MLLM-SHAP/graph/badge.svg" alt="Codecov"></a>
    <a href="https://pypi.org/project/mllm-shap/"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg" alt="Python 3.12-3.14"></a>
    <a href="https://pawlo77.github.io/MLLM-SHAP/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Documentation"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  </p>
</div>

## Installation

```bash
pip install mllm-shap
```

Optional backends and visualization:

```bash
pip install "mllm-shap[transformers]"   # Hugging Face text backend
pip install "mllm-shap[lmstudio]"       # LM Studio SDK
pip install "mllm-shap[api]"            # OpenAI-compatible HTTP API
pip install "mllm-shap[viz]"            # matplotlib token charts
```

The core package does not install PyTorch. Python 3.12 or newer; [uv](https://docs.astral.sh/uv/) for development.

## Why mllm-shap

- **Honest estimands.** Shapley and Banzhaf are separate, labelled aggregators; every result and archive records which one produced it, and results report `base_value` and `grand_value` so efficiency (`sum(values) == v(N) - v(empty)`) can be verified; axiom tests enforce it.
- **Reproducible runs.** One validated configuration, a run id derived from it, and an archive of every model generation, so attributions can be recomputed with another value function without calling the model again.
- **Pluggable everything.** Partitions, absence policies, estimators, value functions, backends and renderers are entry-point plugins; exact, Monte Carlo, complementary and Neyman estimators ship built in.

## Quickstart

```python
import math

from mllm_shap import (
    ConversationSnapshot,
    ExplainConfig,
    ExplainRunner,
    Message,
    Role,
    Turn,
)

snapshot = ConversationSnapshot.from_turns((
    Turn(messages=(Message(role=Role.USER, text="refund my order"),)),
))
config = ExplainConfig.model_validate({
    "backend": {"kind": "mock", "model_id": "stub"},
    "explanation": {"estimator": "exact", "value_fn": "tfidf_cosine"},
})
result = ExplainRunner(config).explain_sync(snapshot).result

print(result.values)  # one attribution per token
print(result.base_value)  # v(empty): payoff of the fully masked prompt
print(result.grand_value)  # v(N): payoff of the full prompt
# Efficiency: Shapley values sum to v(N) - v(empty).
assert math.isclose(sum(result.values), result.grand_value - result.base_value)
```

The `mock` backend needs no model; swap `backend.kind` for `transformers`, `lmstudio` or `api` to explain a real model. Runs are archived under `./runs/{run_id}`; running the same configuration twice raises `ArchiveExistsError` unless `explanation.archive.on_exists` is `overwrite` (or `archive.path` is `""`).

### Command line

The same run without Python: write a snapshot as JSON, start from the commented config skeleton, and get the result as JSON (`--render token_bar --png out.png` also saves a chart).

```bash
echo '{"turns": [{"messages": [{"role": "user", "text": "refund my order"}]}]}' > snapshot.json
mllm-shap config --example > config.yaml
mllm-shap explain --config config.yaml --snapshot snapshot.json --out result.json
mllm-shap reanalyze --config config.yaml --archive runs/<run_id>   # rescore, no model calls
mllm-shap plugins
```

See the [command-line guide](https://pawlo77.github.io/MLLM-SHAP/guides/cli.html).

## Documentation

Full documentation: **[pawlo77.github.io/MLLM-SHAP](https://pawlo77.github.io/MLLM-SHAP/)**

- [Getting started](https://pawlo77.github.io/MLLM-SHAP/guides/getting_started.html)
- [How a run works](https://pawlo77.github.io/MLLM-SHAP/guides/pipeline.html)
- [Configuration](https://pawlo77.github.io/MLLM-SHAP/guides/config.html)
- [Choosing an estimator](https://pawlo77.github.io/MLLM-SHAP/guides/choosing_an_estimator.html)
- [Backends](https://pawlo77.github.io/MLLM-SHAP/guides/backends.html)
- [Extending](https://pawlo77.github.io/MLLM-SHAP/guides/extending.html)
- [Theory](https://pawlo77.github.io/MLLM-SHAP/theory/shapley_values.html)
- [API reference](https://pawlo77.github.io/MLLM-SHAP/api.html)

## Examples, vault and development

- Runnable notebooks: [`examples/`](examples/README.md), also embedded in the [docs](https://pawlo77.github.io/MLLM-SHAP/examples.html).
- Architecture notes, decisions, known issues and papers: the Obsidian vault in [`knowledge/`](knowledge/mllm-shap.md).
- Development setup, checks and pull requests: [`CONTRIBUTING.md`](CONTRIBUTING.md); coding-agent rules: [`AGENTS.md`](AGENTS.md).

## Citing

If you use mllm-shap in research, please cite it ([`CITATION.cff`](CITATION.cff); GitHub's "Cite this repository" button renders it as APA or BibTeX):

```yaml
cff-version: 1.2.0
title: "mllm-shap: Shapley-value explainability for conversational language models"
type: software
authors:
  - family-names: Pozorski
    given-names: Paweł
repository-code: "https://github.com/Pawlo77/MLLM-SHAP"
license: Apache-2.0
```

## License

Apache License 2.0. See [LICENSE](LICENSE).

Copyright 2026 Paweł Pozorski.
