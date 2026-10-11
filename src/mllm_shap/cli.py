"""Command-line interface: ``mllm-shap explain|reanalyze|plugins|config``.

Exit codes: 0 success, 2 invalid configuration, snapshot, archive or arguments,
3 backend unavailable (missing extra, unreachable server).

@domain core
@see [[knowledge/architecture/core]]
@tested-by [[tests/test_cli.py]]
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import yaml
from pydantic import ValidationError

from ._logging import configure_logging
from .domain.conversation import ConversationSnapshot
from .errors import BackendUnavailableError, ConfigError, MllmShapError
from .pipeline.config import ExplainConfig, explain_config_from_yaml
from .pipeline.result import ExplainRunOutput
from .pipeline.runner import ExplainRunner
from .plugins.builtin import default_registry
from .plugins.groups import PluginGroup
from .viz.render import render_attribution

EXIT_OK: Final[int] = 0
"""The command succeeded."""

EXIT_INVALID: Final[int] = 2
"""Invalid arguments, configuration, snapshot or archive (argparse uses 2 too)."""

EXIT_BACKEND_UNAVAILABLE: Final[int] = 3
"""The configured backend cannot be built or reached."""

LOG_LEVELS: Final[tuple[str, ...]] = ("debug", "info", "warning", "error")
"""Values accepted by ``--log-level`` (case-insensitive)."""

RENDERERS: Final[tuple[str, ...]] = ("token_text", "token_bar")
"""Built-in renderer names accepted by ``explain --render``."""

EXAMPLE_CONFIG: Final[str] = """\
# mllm-shap explain configuration. Validate with: mllm-shap config --check FILE
backend:
  kind: mock                # mock | api | lmstudio | transformers (or a plugin)
  model_id: mock            # model name passed to the backend
  # api_host: http://localhost:1234/v1   # api backend only
  timeout_s: 120.0          # per-request timeout of remote backends
  max_retries: 3            # retries after transient backend failures
  options: {}               # backend-specific settings (LM Studio: see docs)

generation:
  max_new_tokens: 32        # tokens generated per coalition
  temperature: 0.0          # 0.0 makes generations deterministic (enables dedup)
  top_k: 1
  compute_embeddings: false # set automatically for embedding value functions

explanation:
  estimand: shapley         # shapley | banzhaf
  estimator: exact          # exact | mc | complementary | neyman_cc
  value_fn: tfidf_cosine    # tfidf_cosine | embedding_cosine | logprob | ...
  normalizer: identity      # identity | abs_sum | power_shift | min_max
  players: tokens           # tokens | messages (one player per message)
  absence_policy: delete    # delete | pad | neutral
  budget:
    fraction: 1.0           # share of the 2^n coalitions; exact needs 1.0
    # evaluations: 2000     # or an absolute count instead of fraction (not exact)
  include_minimal_masks: false
  max_inflight: 2           # concurrent backend calls
  max_players: 24           # refuse runs planning more than 2^max_players coalitions
  seed: 0
  archive:
    path: ./runs/{run_id}   # "" disables the archive (reanalyze needs one)
    flush_every: 50
    on_exists: error        # error | overwrite
  dedup:
    enabled: auto           # auto | on | off
  kv_cache:
    enabled: false          # prefix KV reuse (transformers backend)
    max_entries: 8
  plugin_options: {}        # per-plugin settings keyed by <group>/<name> or name
"""
"""Commented YAML skeleton printed by ``mllm-shap config --example``."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.log_level is not None:
        configure_logging(level=args.log_level.upper())
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (BackendUnavailableError, ConnectionError) as exc:
        _error(f"backend unavailable: {exc}")
        return EXIT_BACKEND_UNAVAILABLE
    except (MllmShapError, ValidationError, yaml.YAMLError) as exc:
        _error(str(exc))
        return EXIT_INVALID


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mllm-shap",
        description="Shapley-value attributions for conversational language models.",
    )
    parser.add_argument(
        "--log-level",
        type=str.lower,
        choices=LOG_LEVELS,
        help="log mllm_shap events at this level to stderr",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    explain = commands.add_parser("explain", help="explain one snapshot")
    explain.add_argument("--config", type=Path, required=True, help="YAML config")
    explain.add_argument(
        "--snapshot", type=Path, required=True, help="snapshot JSON file"
    )
    explain.add_argument("--out", type=Path, help="write the result JSON here")
    explain.add_argument("--render", choices=RENDERERS, help="renderer for --png")
    explain.add_argument("--png", type=Path, help="save the rendered figure here")
    explain.set_defaults(handler=_explain)

    reanalyze = commands.add_parser(
        "reanalyze", help="rescore an archived run without backend calls"
    )
    reanalyze.add_argument("--config", type=Path, required=True, help="YAML config")
    reanalyze.add_argument(
        "--archive", type=Path, required=True, help="run archive directory"
    )
    reanalyze.add_argument("--out", type=Path, help="write the result JSON here")
    reanalyze.add_argument(
        "--strict",
        action="store_true",
        help="refuse runs scored with backend embeddings instead of warning",
    )
    reanalyze.set_defaults(handler=_reanalyze)

    plugins = commands.add_parser("plugins", help="list registered plugins")
    plugins.set_defaults(handler=_plugins)

    config = commands.add_parser("config", help="print or check a configuration")
    action = config.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--example", action="store_true", help="print a commented YAML skeleton"
    )
    action.add_argument("--check", type=Path, metavar="FILE", help="validate FILE")
    config.set_defaults(handler=_config)
    return parser


def _explain(args: argparse.Namespace) -> int:
    if (args.render is None) != (args.png is None):
        msg = "--render and --png must be given together"
        raise ConfigError(msg)
    config = _load_config(args.config)
    snapshot_text = _read_input(args.snapshot, "snapshot")
    try:
        snapshot = ConversationSnapshot.from_json(snapshot_text)
    except (ValueError, TypeError) as exc:
        msg = f"invalid snapshot {args.snapshot}: {exc}"
        raise ConfigError(msg) from exc
    try:
        output = ExplainRunner(config).explain_sync(snapshot)
    except ImportError as exc:
        raise BackendUnavailableError(str(exc)) from exc
    _write_result(output, args.out)
    if args.png is not None:
        _render_png(output, snapshot, args.render, args.png)
    return EXIT_OK


def _reanalyze(args: argparse.Namespace) -> int:
    runner = ExplainRunner(_load_config(args.config))
    try:
        output = runner.reanalyze(args.archive, strict=args.strict)
    except FileNotFoundError as exc:
        msg = f"cannot read archive {args.archive}: {exc}"
        raise ConfigError(msg) from exc
    _write_result(output, args.out)
    return EXIT_OK


def _plugins(_args: argparse.Namespace) -> int:
    registry = default_registry()
    for group in PluginGroup:
        print(f"{group.value}: {', '.join(registry.names(group))}")
    return EXIT_OK


def _config(args: argparse.Namespace) -> int:
    if args.example:
        sys.stdout.write(EXAMPLE_CONFIG)
        return EXIT_OK
    _load_config(args.check)
    print(f"{args.check}: valid")
    return EXIT_OK


def _load_config(path: Path) -> ExplainConfig:
    return explain_config_from_yaml(_read_input(path, "config"))


def _read_input(path: Path, what: str) -> str:
    """Read an input file named on the command line.

    Raises:
        ConfigError: If the file cannot be read.
    """
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read {what} {path}: {exc.strerror or exc}"
        raise ConfigError(msg) from exc


def _write_result(output: ExplainRunOutput, out: Path | None) -> None:
    text = json.dumps(output.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")


def _render_png(
    output: ExplainRunOutput,
    snapshot: ConversationSnapshot,
    renderer: str,
    png: Path,
) -> None:
    if output.player_set is None:
        msg = "the explain output has no player set to label the figure"
        raise ConfigError(msg)
    try:
        figure = render_attribution(
            output, snapshot, output.player_set, renderer=renderer
        )
    except ImportError as exc:
        msg = f"rendering needs the viz extra (pip install 'mllm-shap[viz]'): {exc}"
        raise ConfigError(msg) from exc
    figure.savefig(png, bbox_inches="tight")


def _error(message: str) -> None:
    print(f"mllm-shap: error: {message}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
