"""Token counting and monotonic generation metric collection."""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from modeltop.chat.models import ChatMessage, GenerationMetrics

type Clock = Callable[[], float]


class TokenCounter(Protocol):
    """Replaceable token counter without model-framework coupling."""

    @property
    def name(self) -> str:
        """Stable display name for count provenance."""
        ...

    @property
    def exact(self) -> bool:
        """Whether counts are tokenizer-exact rather than estimates."""
        ...

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...


@dataclass(frozen=True, slots=True)
class MessageTokenMeasurement:
    """One complete chat-message token measurement and its provenance."""

    total_tokens: int
    template_overhead_tokens: int | None
    estimated: bool


@runtime_checkable
class MessageTokenCounter(TokenCounter, Protocol):
    """Counter that can account for a backend chat template directly."""

    def count_messages(
        self, messages: Sequence[ChatMessage]
    ) -> MessageTokenMeasurement:
        """Count complete role/content messages including template overhead."""
        ...


def token_counter_name(counter: TokenCounter) -> str:
    """Return a stable, nonblank token-counter provenance label."""
    name = counter.name.strip()
    return name or type(counter).__name__


def count_chat_messages(
    counter: TokenCounter, messages: Sequence[ChatMessage]
) -> MessageTokenMeasurement:
    """Count messages without constructing a potentially huge joined prompt."""
    if isinstance(counter, MessageTokenCounter):
        return counter.count_messages(messages)
    content_tokens = sum(counter.count(message.content) for message in messages)
    template_overhead = 4 * len(messages) + 3
    return MessageTokenMeasurement(
        total_tokens=content_tokens + template_overhead,
        template_overhead_tokens=template_overhead,
        estimated=True,
    )


class CharacterTokenCounter:
    """Dependency-free character approximation: ceil(non-whitespace / 4)."""

    @property
    def name(self) -> str:
        return "character-estimate"

    @property
    def exact(self) -> bool:
        return False

    def count(self, text: str) -> int:
        characters = sum(not character.isspace() for character in text)
        return math.ceil(characters / 4)

    def count_messages(
        self, messages: Sequence[ChatMessage]
    ) -> MessageTokenMeasurement:
        characters = sum(
            sum(not character.isspace() for character in message.content)
            for message in messages
        )
        content_tokens = math.ceil(characters / 4)
        template_overhead = 4 * len(messages) + 3
        return MessageTokenMeasurement(
            total_tokens=content_tokens + template_overhead,
            template_overhead_tokens=template_overhead,
            estimated=True,
        )


class MetricCollector:
    """Collect one generation using server, exact, then approximate precedence."""

    def __init__(
        self,
        messages: Sequence[ChatMessage],
        *,
        clock: Clock = time.perf_counter,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._clock = clock
        self._counter = token_counter or CharacterTokenCounter()
        self._prompt_measurement = count_chat_messages(self._counter, messages)
        self._completion = ""
        self._prompt_usage: int | None = None
        self._completion_usage: int | None = None
        self._total_usage: int | None = None
        self._draft_usage: int | None = None
        self._accepted_usage: int | None = None
        self._acceptance_rate_usage: float | None = None
        self._request_started_at = clock()
        self._first_token_at: float | None = None
        self._completed_at: float | None = None
        self._finish_reason: str | None = None
        self._streamed = True
        self._cancelled = False

    @property
    def request_started_at(self) -> float:
        return self._request_started_at

    def add_content(self, text: str) -> None:
        if text and self._first_token_at is None and self._streamed:
            self._first_token_at = self._clock()
        self._completion += text

    def update_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        draft_tokens: int | None = None,
        accepted_tokens: int | None = None,
        acceptance_rate: float | None = None,
    ) -> None:
        if prompt_tokens is not None:
            self._prompt_usage = prompt_tokens
        if completion_tokens is not None:
            self._completion_usage = completion_tokens
        if total_tokens is not None:
            self._total_usage = total_tokens
        if draft_tokens is not None:
            self._draft_usage = draft_tokens
        if accepted_tokens is not None:
            self._accepted_usage = accepted_tokens
        if acceptance_rate is not None:
            self._acceptance_rate_usage = acceptance_rate

    def mark_fallback(self) -> None:
        self._streamed = False
        self._first_token_at = None

    def finish(
        self,
        *,
        finish_reason: str | None = None,
        cancelled: bool = False,
    ) -> GenerationMetrics:
        self._finish_reason = finish_reason
        self._cancelled = cancelled
        self._completed_at = self._clock()
        return self.snapshot()

    def snapshot(self) -> GenerationMetrics:
        now = self._completed_at if self._completed_at is not None else self._clock()
        prompt_tokens, prompt_estimated = self._resolved_prompt_count()
        completion_tokens, completion_estimated = self._resolved_count(
            self._completion_usage, self._completion
        )
        if self._total_usage is not None:
            total_tokens = self._total_usage
            total_estimated = False
        else:
            total_tokens = prompt_tokens + completion_tokens
            total_estimated = prompt_estimated or completion_estimated

        draft_tokens = self._draft_usage
        accepted_tokens = self._accepted_usage
        acceptance_rate = self._acceptance_rate_usage
        if (
            acceptance_rate is None
            and draft_tokens is not None
            and draft_tokens > 0
            and accepted_tokens is not None
        ):
            acceptance_rate = accepted_tokens / draft_tokens

        ttft_ms: float | None = None
        active_duration: float | None = None
        output_speed: float | None = None
        inter_token_latency_ms: float | None = None
        if self._streamed and self._first_token_at is not None:
            ttft_ms = (self._first_token_at - self._request_started_at) * 1000
            active_duration = now - self._first_token_at
            if completion_tokens > 0 and active_duration > 0:
                output_speed = completion_tokens / active_duration
                inter_token_latency_ms = 1000 / output_speed

        return GenerationMetrics(
            request_started_at=self._request_started_at,
            first_token_at=self._first_token_at,
            completed_at=self._completed_at,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_estimated=prompt_estimated,
            completion_tokens_estimated=completion_estimated,
            total_tokens_estimated=total_estimated,
            draft_tokens=draft_tokens,
            accepted_tokens=accepted_tokens,
            acceptance_rate=acceptance_rate,
            ttft_ms=ttft_ms,
            active_generation_duration_s=active_duration,
            total_duration_s=now - self._request_started_at,
            output_tokens_per_second=output_speed,
            inter_token_latency_ms=inter_token_latency_ms,
            finish_reason=self._finish_reason,
            streamed=self._streamed,
            cancelled=self._cancelled,
        )

    def _resolved_prompt_count(self) -> tuple[int, bool]:
        if self._prompt_usage is not None:
            return self._prompt_usage, False
        return (
            self._prompt_measurement.total_tokens,
            self._prompt_measurement.estimated,
        )

    def _resolved_count(self, usage: int | None, text: str) -> tuple[int, bool]:
        if usage is not None:
            return usage, False
        return self._counter.count(text), not self._counter.exact
