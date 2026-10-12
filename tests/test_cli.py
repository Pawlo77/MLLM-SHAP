"""The ``mllm-shap`` command line, run in a subprocess with the mock backend.

@see [[src/mllm_shap/cli.py]]
"""

import io
import json
import logging
import subprocess
import sys
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from mllm_shap.cli import EXAMPLE_CONFIG, main
from mllm_shap.pipeline.config import explain_config_from_yaml

_SNAPSHOT = {"turns": [{"messages": [{"role": "user", "text": "Why is the sky blue"}]}]}


def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mllm_shap.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_inputs(tmp_path: Path, **backend: object) -> tuple[Path, Path]:
    config = explain_config_from_yaml(EXAMPLE_CONFIG).model_dump(mode="json")
    config["backend"].update(backend)
    config["explanation"]["archive"]["path"] = str(tmp_path / "runs" / "{run_id}")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_SNAPSHOT), encoding="utf-8")
    return config_path, snapshot_path


def test_example_config_is_valid() -> None:
    config = explain_config_from_yaml(EXAMPLE_CONFIG)
    assert config.backend.kind == "mock"
    assert config.explanation.estimator == "exact"


def test_cli_explain_writes_result_json_and_png(tmp_path: Path) -> None:
    config_path, snapshot_path = _write_inputs(tmp_path)
    out = tmp_path / "result.json"
    png = tmp_path / "result.png"

    completed = _cli(
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
        "--out",
        str(out),
        "--render",
        "token_bar",
        "--png",
        str(png),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert list(result) == [
        "values",
        "labels",
        "base_value",
        "grand_value",
        "estimand",
        "metrics",
        "manifest",
        "perf",
    ]
    assert [label["text"] for label in result["labels"]] == [
        "Why",
        "is",
        "the",
        "sky",
        "blue",
    ]
    assert sum(result["values"]) == pytest.approx(
        result["grand_value"] - result["base_value"], abs=1e-9
    )
    assert result["metrics"]["requested"] == 32
    assert png.read_bytes().startswith(b"\x89PNG")


def test_cli_reanalyze_matches_explain(tmp_path: Path) -> None:
    config_path, snapshot_path = _write_inputs(tmp_path)
    explained = _cli(
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
        cwd=tmp_path,
    )
    assert explained.returncode == 0, explained.stderr
    first = json.loads(explained.stdout)
    archive = tmp_path / "runs" / first["manifest"]["run_id"]

    rescored = _cli(
        "reanalyze",
        "--config",
        str(config_path),
        "--archive",
        str(archive),
        cwd=tmp_path,
    )

    assert rescored.returncode == 0, rescored.stderr
    second = json.loads(rescored.stdout)
    assert second["values"] == pytest.approx(first["values"])
    assert second["metrics"]["executed"] == 0
    assert [label["text"] for label in second["labels"]] == [None] * 5


def test_cli_plugins_lists_every_group(tmp_path: Path) -> None:
    completed = _cli("plugins", cwd=tmp_path)

    assert completed.returncode == 0
    assert "mllm_shap.estimators: complementary, exact, loo, mc, neyman_cc" in (
        completed.stdout
    )
    assert (
        "mllm_shap.backends: api, lmstudio, mock, mock_audio, transformers"
        in completed.stdout
    )


def test_cli_config_example_and_check(tmp_path: Path) -> None:
    example = _cli("config", "--example", cwd=tmp_path)
    assert example.returncode == 0
    assert example.stdout == EXAMPLE_CONFIG
    path = tmp_path / "example.yaml"
    path.write_text(example.stdout, encoding="utf-8")

    checked = _cli("config", "--check", str(path), cwd=tmp_path)

    assert checked.returncode == 0
    assert "valid" in checked.stdout


def test_cli_invalid_config_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "backend: {kind: mock, model_id: m}\n"
        "explanation: {estimator: exact, budget: {fraction: 0.5}}\n",
        encoding="utf-8",
    )

    completed = _cli("config", "--check", str(path), cwd=tmp_path)

    assert completed.returncode == 2
    assert "budget.fraction" in completed.stderr


def test_cli_bad_snapshot_exits_2(tmp_path: Path) -> None:
    config_path, snapshot_path = _write_inputs(tmp_path)
    snapshot_path.write_text('{"turns": "nope"}', encoding="utf-8")

    completed = _cli(
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
        cwd=tmp_path,
    )

    assert completed.returncode == 2
    assert "turns" in completed.stderr


def test_cli_unreachable_backend_exits_3(tmp_path: Path) -> None:
    config_path, snapshot_path = _write_inputs(
        tmp_path, kind="api", api_host="http://127.0.0.1:9/v1", max_retries=0
    )

    completed = _cli(
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
        cwd=tmp_path,
    )

    assert completed.returncode == 3
    assert "backend unavailable" in completed.stderr


def test_main_in_process_explain_render_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() maps validation errors to 2 and renders charts in-process."""
    package_logger = logging.getLogger("mllm_shap")
    monkeypatch.setattr(package_logger, "handlers", list(package_logger.handlers))
    monkeypatch.setattr(package_logger, "level", package_logger.level)
    monkeypatch.setattr(package_logger, "propagate", package_logger.propagate)
    monkeypatch.chdir(tmp_path)
    config_path, snapshot_path = _write_inputs(tmp_path)
    base = ["explain", "--config", str(config_path), "--snapshot", str(snapshot_path)]

    assert main([*base, "--render", "token_text"]) == 2
    assert "--render and --png" in capsys.readouterr().err

    png = tmp_path / "chart.png"
    assert (
        main([
            "--log-level",
            "info",
            *base,
            "--render",
            "token_text",
            "--png",
            str(png),
        ])
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["estimand"] == "shapley"
    assert png.is_file()

    assert main(["config", "--check", str(tmp_path / "missing.yaml")]) == 2


def test_main_rejects_unknown_log_level(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--log-level", "verbose", "plugins"])
    assert excinfo.value.code == 2
    assert "verbose" in capsys.readouterr().err


def test_main_accepts_upper_case_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    levels: list[str] = []
    monkeypatch.setattr(
        "mllm_shap.cli.configure_logging", lambda level: levels.append(level)
    )
    assert main(["--log-level", "INFO", "plugins"]) == 0
    assert levels == ["INFO"]


def test_main_maps_connection_errors_to_exit_3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, snapshot_path = _write_inputs(tmp_path)

    def refuse(self: object, snapshot: object) -> None:
        msg = "connection refused"
        raise ConnectionRefusedError(msg)

    monkeypatch.setattr("mllm_shap.cli.ExplainRunner.explain_sync", refuse)
    code = main([
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
    ])
    assert code == 3
    assert "connection refused" in capsys.readouterr().err


def test_main_lets_unexpected_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> None:
        msg = "bug"
        raise ValueError(msg)

    monkeypatch.setattr("mllm_shap.cli.default_registry", broken)
    with pytest.raises(ValueError, match="bug"):
        main(["plugins"])


@pytest.mark.parametrize("missing", ["snapshot", "config"])
def test_main_missing_input_file_exits_2(
    tmp_path: Path, missing: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, snapshot_path = _write_inputs(tmp_path)
    paths = {"config": config_path, "snapshot": snapshot_path}
    paths[missing].unlink()
    code = main([
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
    ])
    assert code == 2
    assert str(paths[missing]) in capsys.readouterr().err


def test_main_reanalyze_missing_archive_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = _write_inputs(tmp_path)
    code = main([
        "reanalyze",
        "--config",
        str(config_path),
        "--archive",
        str(tmp_path / "nowhere"),
    ])
    assert code == 2
    assert "nowhere" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["", "- backend\n- explanation\n"])
def test_cli_config_check_non_mapping_root_exits_2(
    tmp_path: Path, text: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    assert main(["config", "--check", str(path)]) == 2
    assert "must be a mapping" in capsys.readouterr().err


def _explained_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    config_path, snapshot_path = _write_inputs(tmp_path)
    code = main([
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
    ])
    assert code == 0
    run_id = json.loads(capsys.readouterr().out)["manifest"]["run_id"]
    return tmp_path / "runs" / run_id


def _reanalyze(tmp_path: Path, archive: Path, *extra: str) -> int:
    return main([
        "reanalyze",
        "--config",
        str(tmp_path / "config.yaml"),
        "--archive",
        str(archive),
        *extra,
    ])


def test_cli_reanalyze_missing_manifest_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _explained_archive(tmp_path, capsys)
    (archive / "manifest.json").unlink()

    assert _reanalyze(tmp_path, archive) == 2
    assert "manifest.json" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda manifest: {**manifest, "schema_version": 2}, "schema_version 2"),
        (lambda manifest: "{not json", "manifest"),
        (lambda manifest: [manifest], "mapping"),
        (lambda manifest: {**manifest, "player_ids": "p0"}, "player_ids"),
        (lambda manifest: {**manifest, "estimand": "owen"}, "estimand"),
        (lambda manifest: {**manifest, "config": {"backend": 1}}, "config"),
    ],
)
def test_cli_reanalyze_damaged_manifest_exits_2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    damage: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    archive = _explained_archive(tmp_path, capsys)
    path = archive / "manifest.json"
    damaged = damage(json.loads(path.read_text(encoding="utf-8")))
    path.write_text(
        damaged if isinstance(damaged, str) else json.dumps(damaged),
        encoding="utf-8",
    )

    assert _reanalyze(tmp_path, archive) == 2
    assert message in capsys.readouterr().err


def test_cli_reanalyze_corrupt_database_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = _explained_archive(tmp_path, capsys)
    (archive / "archive.sqlite").write_bytes(b"not a database" * 100)

    assert _reanalyze(tmp_path, archive) == 2
    assert "archive.sqlite" in capsys.readouterr().err


def test_cli_reanalyze_strict_refuses_backend_embedding_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    archive = _explained_archive(tmp_path, capsys)
    path = archive / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["embedding_source"] = "backend"
    manifest["embedding_model"] = "some/model"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["explanation"]["value_fn"] = "embedding_cosine"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="mllm_shap"):
        assert _reanalyze(tmp_path, archive) == 0
    assert "hash embeddings" in caplog.text
    capsys.readouterr()
    assert _reanalyze(tmp_path, archive, "--strict") == 2
    assert "some/model" in capsys.readouterr().err


def test_main_explains_messages_with_speaker_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, snapshot_path = _write_inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["explanation"]["players"] = "messages"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    messages = [
        {"role": "user", "text": "Is the sky blue?", "speaker": "alice"},
        {"role": "user", "text": "Yes, mostly.", "speaker": "bob"},
        {"role": "user", "text": "Why?", "speaker": "alice"},
    ]
    snapshot_path.write_text(json.dumps({"turns": [{"messages": messages}]}))

    assert main(["config", "--check", str(config_path)]) == 0
    capsys.readouterr()
    assert (
        main([
            "explain",
            "--config",
            str(config_path),
            "--snapshot",
            str(snapshot_path),
        ])
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert len(result["values"]) == 3
    assert [label["speaker"] for label in result["labels"]] == ["alice", "bob", "alice"]
    assert [label["label"] for label in result["labels"]] == [
        "user/alice 1: Is the sky blue?",
        "user/bob 2: Yes, mostly.",
        "user/alice 3: Why?",
    ]


def test_cli_explain_reads_audio_paths_relative_to_the_snapshot(tmp_path: Path) -> None:
    """A clip next to the snapshot file loads whatever the working directory."""
    config_path, _ = _write_inputs(tmp_path, kind="mock_audio")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "clip.wav").write_bytes(_tone_wav())
    snapshot_path = inputs / "spoken.json"
    content = [
        {"type": "text", "text": "say it"},
        {"type": "audio", "path": "clip.wav", "sample_rate": 8000},
    ]
    snapshot = {"turns": [{"messages": [{"role": "user", "content": content}]}]}
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    out = tmp_path / "result.json"

    completed = _cli(
        "explain",
        "--config",
        str(config_path),
        "--snapshot",
        str(snapshot_path),
        "--out",
        str(out),
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert [label["text"] for label in result["labels"]] == ["say", "it"]


def _tone_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x00\x40" * 800)
    return buffer.getvalue()
