"""Public explain pipeline entry point.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
"""

import asyncio
import hashlib
import math
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from ..domain.conversation import ConversationSnapshot
from ..errors import ConfigError
from ..plugins import (
    PluginContext,
    PluginGroup,
    PluginRegistry,
    check_plugin_options,
    default_registry,
)
from ..plugins.backends import instantiate_backend
from ..protocols.backend import GenerativeBackend, SupportsAclose
from ..protocols.partition import PlayerPartitioner
from ..runtime.progress import CoalitionProgress
from ..runtime.telemetry import NullObservabilitySink, ObservabilitySink
from .config import ArchiveConfig, ExplainConfig, explain_config_to_yaml
from .context import ExplainContext
from .orchestrator import ExplainOrchestrator
from .reanalyze import reanalyze_sync
from .result import ExplainRunOutput


class ExplainRunner:
    """Run the explain pipeline for a conversation snapshot.

    With an injected backend, :meth:`explain_sync` runs every call on one event
    loop owned by the runner, so backends holding loop-bound resources (HTTP
    clients, locks) work across calls. :meth:`close`, :meth:`aclose` or a
    ``with`` / ``async with`` block close the backend and that loop.
    """

    def __init__(
        self,
        config: ExplainConfig,
        registry: PluginRegistry | None = None,
        telemetry: ObservabilitySink | None = None,
        progress: CoalitionProgress | None = None,
        backend: GenerativeBackend | None = None,
    ) -> None:
        """Configure the runner.

        Args:
            config: Explain configuration.
            registry: Plugin registry; defaults to :func:`default_registry`.
            telemetry: Optional span sink.
            progress: Optional coalition progress callbacks.
            backend: Backend reused by every :meth:`explain` call and closed by
                :meth:`aclose`. Without it, each call builds the backend named in
                ``config.backend`` and closes it when the call ends.
        """
        self._config = config
        self._backend = backend
        self._registry = registry or default_registry()
        self._telemetry = telemetry or NullObservabilitySink()
        self._progress = progress
        self._loop_runner: asyncio.Runner | None = None

    @property
    def registry(self) -> PluginRegistry:
        """Return the plugin registry used by this runner."""
        return self._registry

    async def explain(self, snapshot: ConversationSnapshot) -> ExplainRunOutput:
        """Execute the async explain pipeline."""
        context = self._build_context(snapshot)
        if self._backend is not None:
            return await ExplainOrchestrator(context, self._backend).run()
        backend = instantiate_backend(context.plugin_context())
        try:
            orchestrator = ExplainOrchestrator(context, backend)
            return await orchestrator.run()
        finally:
            if isinstance(backend, SupportsAclose):
                await backend.aclose()

    async def aclose(self) -> None:
        """Close the injected backend and the loop of :meth:`explain_sync`.

        When :meth:`explain_sync` already ran, the backend is closed on its own
        loop (in a worker thread), since its resources are bound to that loop.
        """
        if self._loop_runner is not None:
            await asyncio.to_thread(self.close)
            return
        if isinstance(self._backend, SupportsAclose):
            await self._backend.aclose()

    def close(self) -> None:
        """Close the injected backend and the loop of :meth:`explain_sync`.

        Raises:
            RuntimeError: If called from a running event loop before
                :meth:`explain_sync` created its loop; use ``await
                runner.aclose()`` there.
        """
        loop_runner = self._loop_runner
        if loop_runner is None:
            if isinstance(self._backend, SupportsAclose):
                _raise_inside_running_loop("close", "aclose")
                asyncio.run(self._backend.aclose())
            return
        try:
            if isinstance(self._backend, SupportsAclose):
                loop_runner.run(self._backend.aclose())
        finally:
            loop_runner.close()
            self._loop_runner = None

    def explain_sync(self, snapshot: ConversationSnapshot) -> ExplainRunOutput:
        """Run :meth:`explain` to completion from synchronous code.

        Without an injected backend each call uses a fresh event loop. With one,
        every call reuses a loop the runner creates on first use and closes in
        :meth:`close` / :meth:`aclose`.

        Raises:
            RuntimeError: If called from a running event loop (for example Jupyter);
                use ``await runner.explain(snapshot)`` there.
        """
        _raise_inside_running_loop("explain_sync", "explain")
        if self._backend is None:
            return asyncio.run(self.explain(snapshot))
        if self._loop_runner is None:
            self._loop_runner = asyncio.Runner()
        return self._loop_runner.run(self.explain(snapshot))

    def __enter__(self) -> Self:
        """Return the runner for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the injected backend and the runner's event loop."""
        self.close()

    async def __aenter__(self) -> Self:
        """Return the runner for use in an ``async with`` block."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the injected backend and the runner's event loop."""
        await self.aclose()

    def reanalyze(
        self, archive_root: Path | str, strict: bool = False
    ) -> ExplainRunOutput:
        """Rescore an archived run with the runner's current explain settings.

        Embedding value functions score hash embeddings of the archived text.
        When the archived run scored backend embeddings, this logs a WARNING, or
        with ``strict`` raises :exc:`~mllm_shap.errors.ReanalyzeMismatchError`.
        """
        return reanalyze_sync(
            Path(archive_root),
            self._config,
            self._registry,
            telemetry=self._telemetry,
            strict=strict,
        )

    def _build_context(self, snapshot: ConversationSnapshot) -> ExplainContext:
        check_plugin_options(self._config, self._registry)
        partitioner = cast(
            PlayerPartitioner,
            self._registry.resolve(
                PluginGroup.PARTITIONS,
                self._config.explanation.players,
                PluginContext(config=self._config, registry=self._registry),
            ),
        )
        player_set = partitioner.partition(snapshot)
        if not player_set.spans:
            msg = (
                f"partitioner {self._config.explanation.players!r} returned a "
                "player set without spans"
            )
            raise ConfigError(msg)
        explanation = self._config.explanation
        num_players = player_set.num_players
        budget = explanation.budget
        # Every builtin estimator plans about fraction * 2^n coalitions, or the
        # absolute count capped at 2^n; compare in log2 so huge player counts do
        # not overflow a float.
        if budget.evaluations is None:
            planned_log2 = num_players + math.log2(budget.fraction or 1.0)
            setting = f"budget.fraction {budget.fraction}"
        else:
            planned_log2 = min(math.log2(budget.evaluations), num_players)
            setting = f"budget.evaluations {budget.evaluations}"
        if planned_log2 > explanation.max_players:
            msg = (
                f"snapshot has {num_players} players; {setting} plans about "
                f"2^{planned_log2:.1f} coalitions, more than the "
                f"2^{explanation.max_players} allowed by "
                f"explanation.max_players={explanation.max_players}. Lower the "
                "budget, use coarser players, or raise max_players."
            )
            raise ConfigError(msg)
        # The digest covers every config field except where the archive is
        # written, so a changed estimator, policy, value function or seed never
        # shares an archive with an older run, while ``archive.on_exists``
        # still targets the run it is meant to overwrite.
        identity = self._config.model_copy(
            update={
                "explanation": self._config.explanation.model_copy(
                    update={"archive": ArchiveConfig()}
                )
            }
        )
        model_id = (
            self._backend.model_id
            if self._backend is not None
            else self._config.backend.model_id
        )
        digest = hashlib.sha256(
            (explain_config_to_yaml(identity) + model_id).encode("utf-8")
        ).hexdigest()[:12]
        run_id = f"{snapshot.snapshot_id[:16]}-{digest}"
        return ExplainContext(
            config=self._config,
            snapshot=snapshot,
            player_set=player_set,
            registry=self._registry,
            run_id=run_id,
            archive_root=self._resolve_archive_root(run_id),
            telemetry=self._telemetry,
            progress=self._progress,
        )

    def _resolve_archive_root(self, run_id: str) -> Path | None:
        template = self._config.explanation.archive.path.strip()
        if not template:
            return None
        return Path(template.format(run_id=run_id))


def _raise_inside_running_loop(method: str, async_method: str) -> None:
    """Refuse a blocking call made from a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    msg = (
        f"{method} cannot run inside an event loop; "
        f"use 'await runner.{async_method}(...)'"
    )
    raise RuntimeError(msg)
