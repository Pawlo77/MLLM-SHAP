"""Compute next audio-track rewrite phase from the package version."""

import sys
from pathlib import Path

# (phase, tag without v, short focus)
_AUDIO_PHASES: tuple[tuple[int, str, str], ...] = (
    (15, "0.2.0", "multimodal domain + enums"),
    (16, "0.2.1", "SGPA alignment (Wav2Vec2)"),
    (17, "0.2.2", "audio masking / SGPA partition"),
    (18, "0.2.3", "mock audio backend"),
    (19, "0.2.4", "liquid-audio backend"),
    (20, "0.2.5", "multimodal E2E pipeline"),
    (21, "0.2.6", "audio value fns + SGPA smoke"),
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLAN_DIR = (
    _REPO_ROOT.parent / "nlp-shap-research" / "docs" / "plans" / "infrastructure"
)


def _parse(version: str) -> tuple[int, ...]:
    """Parse a dotted version into an int tuple (non-digits ignored)."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if digits:
            parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def next_phase_hint(version: str) -> str:
    """Return a one-line next-phase hint for sessionStart context."""
    current = _parse(version)
    for phase, tag, focus in _AUDIO_PHASES:
        if current < _parse(tag):
            plan_note = (
                " Plan sibling present."
                if _PLAN_DIR.is_dir()
                else (
                    " Clone nlp-shap-research as sibling of nlp-shap/; "
                    "open nlp-shap.code-workspace (see AGENTS.md)."
                )
            )
            return (
                f"Next audio phase {phase} (v{tag}): {focus}."
                f" Skill nlp-shap-rewrite.{plan_note}"
            )
    return "Audio track phases 15-21 complete or package ahead of plan."


def main() -> None:
    """Print the next-phase hint for the given version argument."""
    version = sys.argv[1] if len(sys.argv) > 1 else "0.0.0"
    print(next_phase_hint(version))


if __name__ == "__main__":
    main()
