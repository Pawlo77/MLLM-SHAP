"""Teacher-forced value functions: log probability and Jensen-Shannon shift of ``y_N``.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_teacher_forced.py]]
"""

import math
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from ..domain.generation import ForcedScore, GenerationRecord
from ..errors import BackendUnavailableError
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


def forced_for(
    result: GenerationResult, continuation: str, value_fn: str
) -> ForcedScore:
    """Return the teacher-forced score of ``continuation`` held by ``result``.

    Raises:
        BackendUnavailableError: If the generation carries no such score.
    """
    forced = (
        result.forced.get(continuation)
        if isinstance(result, GenerationRecord)
        else None
    )
    if forced is None:
        msg = (
            f"value_fn {value_fn!r} needs the teacher-forced score of "
            f"{continuation[:40]!r}, which the generation does not carry"
        )
        raise BackendUnavailableError(msg)
    return forced


class TeacherForcedValue:
    """Shared declarations of value functions that score ``y_N`` by forcing."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    registered_name: ClassVar[str]
    """Registered value-function identifier."""

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return self.registered_name

    @property
    def needs_teacher_forcing(self) -> bool:
        """Always true; scoring reads ``GenerationRecord.forced``."""
        return True

    @property
    def reads_candidate_text(self) -> bool:
        """Always false; coalition answers are never generated."""
        return False

    @property
    def forced_top_k(self) -> int:
        """Return how many next-token candidates per position to request."""
        return 0

    def forced_continuations(self, base: GenerationResult) -> tuple[str, ...]:
        """Return the base answer :math:`y_N` as the continuation to score."""
        return (base.text,)


class TfLogprobOptions(PluginOptions):
    """Options of the ``tf_logprob`` value function."""

    reduce: Literal["mean", "sum"] = "mean"
    """Average (default) or add the log probabilities of the answer tokens."""


class TfLogprobValue(TeacherForcedValue):
    r"""Score :math:`v(S) = \frac1\ell\sum_t \log p(y_{N,t} \mid y_{N,<t}, x_S)`.

    The full-input answer :math:`y_N` is scored under the coalition prompt
    :math:`x_S` by one forward pass; no coalition answer is generated. An empty
    :math:`y_N` scores 0. ``reduce: sum`` drops the :math:`1/\ell` factor.
    """

    options_model: ClassVar[type[PluginOptions]] = TfLogprobOptions
    """Options accepted from ``explanation.plugin_options``."""

    registered_name = "tf_logprob"

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-math.inf, 0.0),
        reads_base=True,
        needs=("teacher_forcing",),
        cost="forward",
    )

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read ``reduce`` from the plugin options in ``context``."""
        options = TfLogprobOptions()
        if context is not None:
            options = TfLogprobOptions.model_validate(context.options)
        self.reduce = options.reduce
        """Whether token log probabilities are averaged or added."""

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the teacher-forced log probability of ``base`` under ``candidate``."""
        logprobs = forced_for(candidate, base.text, self.name).token_logprobs
        if not logprobs:
            return 0.0
        total = math.fsum(logprobs)
        return total / len(logprobs) if self.reduce == "mean" else total


class TfJsdOptions(PluginOptions):
    """Options of the ``tf_jsd`` value function."""

    top_k: int = 20
    """Next-token candidates kept per position; the rest is one bucket."""


class TfJsdValue(TeacherForcedValue):
    r"""Score :math:`v(S) = -\sum_t \mathrm{JSD}(P_N(\cdot) \Vert P_S(\cdot))`.

    :math:`P_N` and :math:`P_S` are the next-token distributions at each position
    of :math:`y_N` under the full prompt and the coalition prompt, so
    :math:`v(N) = 0` and each position costs at most :math:`\log 2`. Each
    distribution is its ``top_k`` tokens plus one bucket for the remaining mass,
    so the divergence is exact once ``top_k`` covers the vocabulary and a lower
    bound otherwise.
    """

    options_model: ClassVar[type[PluginOptions]] = TfJsdOptions
    """Options accepted from ``explanation.plugin_options``."""

    registered_name = "tf_jsd"

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-math.inf, 0.0),
        reads_base=True,
        needs=("teacher_forcing",),
        cost="forward",
    )

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read ``top_k`` from the plugin options in ``context``."""
        options = TfJsdOptions()
        if context is not None:
            options = TfJsdOptions.model_validate(context.options)
        self._top_k = options.top_k

    @property
    def forced_top_k(self) -> int:
        """Return the configured ``top_k``."""
        return self._top_k

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return minus the summed Jensen-Shannon divergence over the answer."""
        reference = forced_for(base, base.text, self.name)
        shifted = forced_for(candidate, base.text, self.name)
        return -math.fsum(
            _jsd(ids_n, lp_n, ids_s, lp_s)
            for ids_n, lp_n, ids_s, lp_s in zip(
                reference.top_ids,
                reference.top_logprobs,
                shifted.top_ids,
                shifted.top_logprobs,
                strict=True,
            )
        )


def _jsd(
    ids_p: tuple[int, ...],
    logp: tuple[float, ...],
    ids_q: tuple[int, ...],
    logq: tuple[float, ...],
) -> float:
    """Jensen-Shannon divergence in nats of two top-k laws with a rest bucket."""
    support = sorted(set(ids_p) | set(ids_q))
    index = {token: i for i, token in enumerate(support)}
    p = np.zeros(len(support) + 1)
    q = np.zeros(len(support) + 1)
    for law, ids, logs in ((p, ids_p, logp), (q, ids_q, logq)):
        for token, value in zip(ids, logs, strict=True):
            law[index[token]] = math.exp(value)
        law[-1] = max(0.0, 1.0 - law[:-1].sum())
    mid = (p + q) / 2.0
    return float(0.5 * _kl(p, mid) + 0.5 * _kl(q, mid))


def _kl(p: "np.ndarray", q: "np.ndarray") -> float:
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
