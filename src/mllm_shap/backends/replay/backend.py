"""Answer coalitions from run archives instead of a model.

Rows are looked up by coalition key, which fixes the snapshot, players, absence
policy, model id and generation settings, so a replay run must use the config of the
archived run apart from the estimator, budget, seed and value function. A coalition
that is not archived raises :class:`~mllm_shap.errors.ReplayMissError`, unless a
``fallback`` backend is configured to compute it.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_replay.py]]
"""

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import ForcedScore
from ...errors import ConfigError, ReplayMissError
from ...masking.codec import MaskCodec
from ...protocols.backend import (
    GenerationResult,
    GenerativeBackend,
    SupportsTeacherForcing,
)
from ...protocols.plugin import PluginOptions
from ...runtime.archive import CoalitionRecord, RunArchive, read_manifest_payload

if TYPE_CHECKING:
    from ...plugins.context import PluginContext


class ReplayOptions(PluginOptions):
    """Options of the ``replay`` backend."""

    archives: list[str]
    """Run archive folders to read; rows of earlier folders win on equal keys."""

    fallback: dict[str, object] | None = None
    """``backend`` settings (``kind``, ``model_id``, ...) of a live backend that
    computes coalitions missing from the archives."""


class ReplayBackend:
    """Serve archived coalition generations; compute misses with a fallback."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``; settings are ``backend.options``."""

    def __init__(self, context: "PluginContext") -> None:
        """Index the rows of every archive in ``backend.options.archives``."""
        options = ReplayOptions.model_validate(context.config.backend.options)
        if not options.archives:
            msg = "backend.options.archives must name at least one run archive"
            raise ConfigError(msg)
        self._rows: dict[str, CoalitionRecord] = {}
        self._base: dict[str, tuple[str, CoalitionRecord | None]] = {}
        self._model_id = ""
        for folder in options.archives:
            self._index(Path(folder))
        self._fallback: GenerativeBackend | None = None
        if options.fallback is not None:
            # Imported here: plugins.builtin imports this module at package import.
            from ...plugins.groups import PluginGroup

            live = type(context.config.backend).model_validate(options.fallback)
            sub = dataclasses.replace(
                context, config=context.config.model_copy(update={"backend": live})
            )
            self._fallback = cast(
                GenerativeBackend,
                context.registry.resolve(PluginGroup.BACKENDS, live.kind, sub),
            )

    def _index(self, root: Path) -> None:
        manifest = read_manifest_payload(root / "manifest.json")
        snapshot_id = str(manifest["snapshot_id"])
        model_id = str(manifest["model_id"])
        if self._model_id and model_id != self._model_id:
            msg = (
                "archives were written by different models: "
                f"{self._model_id!r} and {model_id!r}"
            )
            raise ConfigError(msg)
        self._model_id = model_id
        grand: CoalitionRecord | None = None
        with RunArchive.load(root) as archive:
            for row in archive.history_lazy():
                self._rows.setdefault(row.coalition_key, row)
                if all(MaskCodec.unpack(row.mask)):
                    grand = grand or row
            text = archive.read_base_generation()
        if text is not None:
            self._base.setdefault(snapshot_id, (text, grand))

    @property
    def model_id(self) -> str:
        """Return the model id recorded in the archives."""
        return self._model_id

    @property
    def has_fallback(self) -> bool:
        """Return whether missing coalitions are computed by a live backend."""
        return self._fallback is not None

    def archived_row(self, coalition_key: str) -> CoalitionRecord | None:
        """Return the archived row of ``coalition_key``, if any."""
        return self._rows.get(coalition_key)

    def archived_base(
        self, snapshot_id: str
    ) -> tuple[str, CoalitionRecord | None] | None:
        """Return the archived base text and grand-coalition row of a snapshot."""
        return self._base.get(snapshot_id)

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationResult:
        """Delegate to the fallback; archived coalitions never reach this call."""
        if self._fallback is None:
            msg = (
                "coalition is not in the replay archives (prompt "
                f"{snapshot.snapshot_id!r}); archive it first or set "
                "backend.options.fallback"
            )
            raise ReplayMissError(msg)
        return await self._fallback.generate(snapshot, generation)

    @property
    def forces_continuations(self) -> bool:
        """Always true; archived rows carry their teacher-forced scores."""
        return True

    async def score_continuations(
        self,
        snapshot: ConversationSnapshot,
        continuations: Sequence[str],
        top_k: int = 0,
    ) -> list[ForcedScore]:
        """Delegate to a teacher-forcing fallback; archived rows are read directly."""
        if isinstance(self._fallback, SupportsTeacherForcing):
            return await self._fallback.score_continuations(
                snapshot, continuations, top_k
            )
        msg = "teacher-forced scores of this coalition are not in the replay archives"
        raise ReplayMissError(msg)

    async def aclose(self) -> None:
        """Close the fallback backend, if any."""
        close = getattr(self._fallback, "aclose", None)
        if callable(close):
            await close()
