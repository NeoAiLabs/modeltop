"""State-independent reusable chat generation runner."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from modeltop.api.chat import (
    ChatStreamEvent,
    ContentDelta,
    GenerationFinished,
    ResponseStarted,
    StreamingFallback,
    UsageUpdate,
)
from modeltop.api.errors import APIClientError
from modeltop.chat.metrics import (
    CharacterTokenCounter,
    Clock,
    MetricCollector,
    TokenCounter,
)
from modeltop.chat.models import ChatMessage, GenerationMetrics, GenerationSettings

logger = logging.getLogger(__name__)


class GenerationClient(Protocol):
    """Minimal client contract required by generation runners."""

    def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]: ...


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Frozen request snapshot independent from dashboard state."""

    server_id: str
    model_id: str
    messages: tuple[ChatMessage, ...]
    settings: GenerationSettings
    request_timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationProgress:
    """Coalesced complete output and metric snapshot."""

    content: str
    metrics: GenerationMetrics
    notice: str | None = None
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """Terminal output returned by the runner."""

    content: str
    metrics: GenerationMetrics
    notice: str | None = None
    status_code: int | None = None


class GenerationFailed(Exception):
    """Typed API failure carrying safe partial generation state."""

    def __init__(self, error: APIClientError, outcome: GenerationOutcome) -> None:
        super().__init__(error.user_message)
        self.error = error
        self.outcome = outcome


class GenerationCancelled(asyncio.CancelledError):
    """Cancellation carrying finalized partial generation state."""

    def __init__(self, outcome: GenerationOutcome) -> None:
        super().__init__()
        self.outcome = outcome


type ProgressCallback = Callable[[GenerationProgress], None]


class GenerationService:
    """Consume typed API events and calculate safe generation progress."""

    def __init__(
        self,
        client: GenerationClient,
        *,
        clock: Clock = time.perf_counter,
        token_counter: TokenCounter | None = None,
        publish_interval_seconds: float = 0.05,
    ) -> None:
        self._client = client
        self._clock = clock
        self._token_counter = token_counter or CharacterTokenCounter()
        self._publish_interval_seconds = publish_interval_seconds

    @property
    def token_counter(self) -> TokenCounter:
        """Return the counter shared by prompt building and generation metrics."""
        return self._token_counter

    async def run(
        self,
        request: GenerationRequest,
        on_progress: ProgressCallback,
    ) -> GenerationOutcome:
        """Run one request, preserving partial state in typed terminal failures."""
        collector = MetricCollector(
            request.messages,
            clock=self._clock,
            token_counter=self._token_counter,
        )
        content_parts: list[str] = []
        notice: str | None = None
        finish_reason: str | None = None
        status_code: int | None = None
        last_publish_at: float | None = None
        published_content = False
        logger.info(
            "Chat generation starting server=%s model=%s messages=%d chars=%s",
            request.server_id,
            request.model_id,
            len(request.messages),
            tuple(len(message.content) for message in request.messages),
        )
        try:
            async for event in self._client.stream_chat_completion(
                request.model_id,
                request.messages,
                request.settings,
                timeout_seconds=request.request_timeout_seconds,
            ):
                force_publish = False
                if isinstance(event, ResponseStarted):
                    status_code = event.status_code
                    force_publish = True
                elif isinstance(event, ContentDelta):
                    collector.add_content(event.text)
                    content_parts.append(event.text)
                elif isinstance(event, UsageUpdate):
                    collector.update_usage(
                        event.prompt_tokens,
                        event.completion_tokens,
                        event.total_tokens,
                        event.draft_tokens,
                        event.accepted_tokens,
                        event.acceptance_rate,
                    )
                    force_publish = True
                elif isinstance(event, GenerationFinished):
                    finish_reason = event.finish_reason
                    if not event.streamed:
                        collector.mark_fallback()
                    force_publish = True
                elif isinstance(event, StreamingFallback):
                    notice = event.reason
                    collector.mark_fallback()
                    force_publish = True
                    logger.info(
                        "Chat generation fallback server=%s model=%s",
                        request.server_id,
                        request.model_id,
                    )
                else:
                    force_publish = True

                content = "".join(content_parts)
                now = self._clock()
                first_content = bool(content) and not published_content
                due = (
                    last_publish_at is None
                    or now - last_publish_at >= self._publish_interval_seconds
                )
                if force_publish or first_content or (content and due):
                    on_progress(
                        GenerationProgress(
                            content,
                            collector.snapshot(),
                            notice,
                            status_code,
                        )
                    )
                    last_publish_at = now
                    published_content = bool(content)
        except asyncio.CancelledError as error:
            metrics = collector.finish(finish_reason=finish_reason, cancelled=True)
            outcome = GenerationOutcome(
                "".join(content_parts), metrics, notice, status_code
            )
            logger.info(
                "Chat generation cancelled server=%s model=%s chars=%d",
                request.server_id,
                request.model_id,
                len(outcome.content),
            )
            raise GenerationCancelled(outcome) from error
        except APIClientError as error:
            if status_code is None:
                status_code = error.status_code
            metrics = collector.finish(finish_reason=finish_reason)
            outcome = GenerationOutcome(
                "".join(content_parts), metrics, notice, status_code
            )
            logger.warning(
                "Chat generation failed server=%s model=%s error=%s status=%s",
                request.server_id,
                request.model_id,
                type(error).__name__,
                status_code,
            )
            raise GenerationFailed(error, outcome) from error

        metrics = collector.finish(finish_reason=finish_reason)
        outcome = GenerationOutcome(
            "".join(content_parts), metrics, notice, status_code
        )
        on_progress(GenerationProgress(outcome.content, metrics, notice, status_code))
        logger.info(
            "Chat generation completed server=%s model=%s chars=%d "
            "prompt_tokens=%s completion_tokens=%s finish=%s",
            request.server_id,
            request.model_id,
            len(outcome.content),
            metrics.prompt_tokens,
            metrics.completion_tokens,
            finish_reason,
        )
        return outcome
