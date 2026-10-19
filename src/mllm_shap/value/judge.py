"""Expected-grade value function from a judge model's score probabilities.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_judge.py]]
"""

import dataclasses
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from ..domain.config import GenerationConfig
from ..domain.conversation import ConversationSnapshot, Message, Turn
from ..domain.enums import Role
from ..domain.generation import GenerationRecord
from ..errors import ConfigError
from ..protocols.backend import (
    GenerationResult,
    GenerativeBackend,
    SupportsAclose,
    SupportsTeacherForcing,
)
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

GRADES = ("1", "2", "3", "4", "5")
"""Scores the judge may give."""

_PROMPT = (
    "You compare two answers to the same question.\n"
    "Reference answer: {reference}\n"
    "Candidate answer: {candidate}\n"
    "Grade from 1 (different meaning) to 5 (same meaning) how well the candidate "
    "preserves the meaning of the reference. Reply with the digit only."
)


class JudgeOptions(PluginOptions):
    """Options of the ``judge_grade`` value function."""

    backend: str
    """Registered backend kind of the judge, such as ``transformers``."""

    model_id: str
    """Judge model name passed to that backend."""

    backend_options: dict[str, object] = Field(default_factory=dict)
    """``backend.options`` of the judge backend."""

    samples: int = 20
    """Graded generations averaged when the judge backend returns no logits."""

    allow_large_judge: bool = False
    """Allow more than ``max_coalitions`` distinct answers per run."""

    max_coalitions: int = 2000
    """Distinct answers above which the judge refuses to run."""


class JudgeGradeValue:
    r"""Score :math:`v(S) = \sum_g g\,p(g)` for a judge's grade distribution.

    A judge model is asked how well :math:`y_S` preserves the meaning of
    :math:`y_N` on a 1 to 5 scale. With a judge backend that scores continuations,
    :math:`p(g)` is its probability of each digit, renormalised over 1 to 5;
    otherwise ``samples`` graded generations are averaged. The result is
    continuous, so it has few ties (G-Eval), but it carries the judge's position
    and length biases and costs a model call per coalition: it is a reference for
    small subsets, guarded at ``max_coalitions`` distinct answers.
    """

    options_model: ClassVar[type[PluginOptions]] = JudgeOptions
    """Options accepted from ``explanation.plugin_options``."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(1.0, 5.0),
        reads_base=True,
        needs=("judge",),
        extra_model="judge model (any backend; logits preferred)",
        cost="judge",
    )

    def __init__(
        self, context: "PluginContext", judge: GenerativeBackend | None = None
    ) -> None:
        """Read the options in ``context``; ``judge`` injects a built backend."""
        self._options = JudgeOptions.model_validate(context.options)
        self._context = context
        self._judge = judge
        self._grades: dict[tuple[str, str], float] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "judge_grade"

    def manifest_info(self) -> dict[str, str]:
        """Return the judge backend and model for the run manifest."""
        return {"backend": self._options.backend, "model_id": self._options.model_id}

    async def prepare(
        self, base: GenerationRecord, corpus: Sequence[GenerationRecord]
    ) -> None:
        """Grade every distinct answer of the run against the base answer.

        Raises:
            ConfigError: If the run has more distinct answers than
                ``max_coalitions`` and ``allow_large_judge`` is false.
        """
        texts = sorted({record.text for record in corpus} - {base.text})
        if len(texts) > self._options.max_coalitions and not (
            self._options.allow_large_judge
        ):
            msg = (
                f"judge_grade would grade {len(texts)} answers, above "
                f"max_coalitions={self._options.max_coalitions}; score a subsample "
                "or set plugin_options.judge_grade.allow_large_judge"
            )
            raise ConfigError(msg)
        judge = self._backend()
        for text in texts:
            if (base.text, text) not in self._grades:
                self._grades[base.text, text] = await self._grade(
                    judge, base.text, text
                )
        self._grades[base.text, base.text] = 5.0 if base.text else 0.0

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the cached expected grade of ``candidate``.

        Raises:
            ConfigError: If ``prepare`` has not graded this pair.
        """
        try:
            return self._grades[base.text, candidate.text]
        except KeyError:
            msg = "judge_grade needs the pipeline to await prepare() before scoring"
            raise ConfigError(msg) from None

    def _backend(self) -> GenerativeBackend:
        if self._judge is None:
            # Imported here: the pipeline and plugin packages import this module.
            from ..pipeline.config import BackendConfig
            from ..plugins.backends import instantiate_backend

            config = self._context.config.model_copy(
                update={
                    "backend": BackendConfig(
                        kind=self._options.backend,
                        model_id=self._options.model_id,
                        options=self._options.backend_options,
                    )
                }
            )
            self._judge = instantiate_backend(
                dataclasses.replace(self._context, config=config, options={})
            )
        return self._judge

    async def aclose(self) -> None:
        """Release the judge backend."""
        if isinstance(self._judge, SupportsAclose):
            await self._judge.aclose()

    async def _grade(
        self, judge: GenerativeBackend, reference: str, answer: str
    ) -> float:
        snapshot = ConversationSnapshot.from_turns((
            Turn((
                Message(
                    Role.USER,
                    text=_PROMPT.format(reference=reference, candidate=answer),
                ),
            )),
        ))
        if isinstance(judge, SupportsTeacherForcing) and judge.forces_continuations:
            scores = await judge.score_continuations(snapshot, GRADES)
            logits = [math.fsum(s.token_logprobs) for s in scores]
            top = max(logits)
            weights = [math.exp(x - top) for x in logits]
            return math.fsum(
                int(g) * w for g, w in zip(GRADES, weights, strict=True)
            ) / math.fsum(weights)
        sampled = GenerationConfig(max_new_tokens=4, temperature=1.0, top_k=0)
        grades = []
        for _ in range(self._options.samples):
            reply = await judge.generate(snapshot, sampled)
            match = re.search(r"[1-5]", reply.text)
            if match:
                grades.append(int(match.group()))
        if not grades:
            msg = "the judge returned no grade from 1 to 5"
            raise ConfigError(msg)
        return math.fsum(grades) / len(grades)
