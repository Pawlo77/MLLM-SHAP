"""Rubric checks (hard asserts) and the logit judge (semantic questions).

The judge runs one forward pass per prompt and compares the next-token logits of
YES and NO; the difference is a log-odds margin in nats. Three paraphrases vote:
two margins of at least ``MARGIN`` decide, anything else is "uncertain". With
``NLP_SHAP_PRACTICAL_JUDGE=lmstudio:<key>`` the quick tier asks an LM Studio model
instead, reading the same margin from its top logprobs.
"""

import asyncio
import json
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from nlp_shap.backends.transformers._torch import import_torch
from nlp_shap.backends.transformers.text import TransformersTextBackend

from ._cache import STATS, PracticalCache, cache_key

MARGIN = 1.0
"""Smallest YES-minus-NO logit margin, in nats, that counts as an answer."""

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
"""LM Studio's local chat endpoint, used by :class:`LmStudioJudge`."""

SYSTEM = "You are a strict grader. Reply with exactly one word: YES or NO."
"""System prompt of every judge question."""

Outcome = Literal["pass", "fail", "uncertain"]
"""Result of a vote: a confident YES majority, a confident NO majority, or neither."""


@dataclass(frozen=True)
class Verdict:
    """Outcome of three paraphrased judge prompts."""

    outcome: Outcome
    """Majority answer, or ``uncertain``."""

    margins: tuple[float, ...]
    """YES-minus-NO logit margin of each prompt."""


def vote(margins: Sequence[float]) -> Verdict:
    """Return the majority of three margins; two confident NO answers fail."""
    if len(margins) != 3:
        msg = f"the judge needs three prompts, got {len(margins)}"
        raise ValueError(msg)
    yes = sum(m >= MARGIN for m in margins)
    no = sum(m <= -MARGIN for m in margins)
    outcome: Outcome = "fail" if no >= 2 else "pass" if yes >= 2 else "uncertain"
    return Verdict(outcome, tuple(margins))


def answer_prompts(request: str, answer: str) -> list[str]:
    """Return three paraphrases of "is ``answer`` a correct, on-topic reply"."""
    return [
        f'Request: "{request}"\nReply: "{answer}"\nIs the reply correct and on topic?',
        f'Someone wrote "{request}" and got the reply "{answer}". Is that reply right?',
        f'Is "{answer}" a factually correct response to "{request}"?',
    ]


def driver_prompts(conversation: str, answer: str, span: str) -> list[str]:
    """Return three paraphrases of "is ``span`` what the reply ``answer`` rests on"."""
    return [
        (
            f'Conversation:\n{conversation}\nReply: "{answer}"\n'
            f'Is the message "{span}" the main reason for that reply?'
        ),
        (
            f'In this chat:\n{conversation}\nthe reply was "{answer}". '
            f'Does the reply depend mostly on the message "{span}"?'
        ),
        (
            f'Given the chat below, is "{span}" the message that makes the reply '
            f'"{answer}" possible?\n{conversation}'
        ),
    ]


def normalised_l1(estimate: Sequence[float], reference: Sequence[float]) -> float:
    """Return ``sum |estimate - reference| / sum |reference|``."""
    gap = sum(abs(e - r) for e, r in zip(estimate, reference, strict=True))
    return gap / sum(abs(r) for r in reference)


def top_players(
    labels: Sequence[str], values: Sequence[float], tol: float = 1e-9
) -> list[str]:
    """Return the labels within ``tol`` of the largest value (symmetric players tie)."""
    best = max(values)
    return [label for label, v in zip(labels, values, strict=True) if v >= best - tol]


def top_by_magnitude(values: Sequence[float], k: int) -> list[int]:
    """Return the indices of the ``k`` largest absolute values, largest first."""
    return sorted(range(len(values)), key=lambda i: -abs(values[i]))[:k]


def efficiency_gap(values: Sequence[float], base: float, grand: float) -> float:
    """Return ``|sum(values) - (grand - base)|``."""
    return abs(sum(values) - (grand - base))


def logprob_margin(top: Sequence[Mapping[str, Any]]) -> float:
    """Return YES-minus-NO from top logprobs; a missing word takes the lowest entry."""
    floor = min(float(entry["logprob"]) for entry in top)

    def best(word: str) -> float:
        found = [float(e["logprob"]) for e in top if str(e["token"]).strip() == word]
        return max(found, default=floor)

    return best("YES") - best("NO")


class Judge:
    """Cache-backed vote over three paraphrases; subclasses compute one margin."""

    def __init__(self, cache: PracticalCache, identity: Mapping[str, str]) -> None:
        """Cache verdicts per judge identity."""
        self._cache = cache
        self._identity = dict(identity)

    @property
    def name(self) -> str:
        """Return the judge model, for the results file."""
        return self._identity["model"]

    def _margin(self, prompt: str) -> float:
        raise NotImplementedError

    def ask(self, prompts: Sequence[str]) -> Verdict:
        """Return the vote over three paraphrases of one question."""
        key = cache_key(self._identity, SYSTEM, list(prompts))
        stored = self._cache.get("verdict", key)
        if isinstance(stored, list):
            STATS["verdict_hits"] += 1
            return vote([float(m) for m in stored])
        STATS["verdict_misses"] += 1
        margins = [self._margin(prompt) for prompt in prompts]
        self._cache.put("verdict", key, margins)
        return vote(margins)


class LogitJudge(Judge):
    """Ask yes/no questions of the practical model itself."""

    def __init__(
        self,
        backend: TransformersTextBackend,
        cache: PracticalCache,
        identity: Mapping[str, str],
    ) -> None:
        """Share ``backend``'s weights; cache verdicts per model identity."""
        super().__init__(cache, identity)
        self._backend = backend

    def _margin(self, prompt: str) -> float:
        backend = self._backend
        asyncio.run(backend._ensure_loaded())
        tokenizer: Any = backend._tokenizer
        model: Any = backend._model
        torch = import_torch()
        chat = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(backend._device)
        with backend._model_lock, torch.no_grad():
            logits = model(ids).logits[0, -1].float()

        def best(*words: str) -> float:
            first = {tokenizer.encode(w, add_special_tokens=False)[0] for w in words}
            return max(float(logits[i]) for i in first)

        return best("YES", " YES") - best("NO", " NO")


class LmStudioJudge(Judge):
    """Ask yes/no questions of a model LM Studio serves on this machine."""

    def __init__(self, key: str, cache: PracticalCache) -> None:
        """Judge with LM Studio model ``key``; verdicts are cached under it."""
        super().__init__(cache, {"model": f"lmstudio:{key}"})
        self._key = key

    def _margin(self, prompt: str) -> float:
        body = {
            "model": self._key,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 2,  # LM Studio returns no logprobs for max_tokens 1
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 10,  # LM Studio maximum
        }
        request = urllib.request.Request(
            LMSTUDIO_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
            payload = json.load(response)
        first = payload["choices"][0]["logprobs"]["content"][0]
        return logprob_margin(first["top_logprobs"])
