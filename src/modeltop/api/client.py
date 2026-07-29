"""Minimal asynchronous OpenAI-compatible API client."""

import errno
import json
import logging
import socket
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import httpx

from modeltop.api.chat import (
    ChatStreamEvent,
    ResponseStarted,
    SSEDecoder,
    StreamingFallback,
    build_chat_payload,
    parse_non_stream_completion,
)
from modeltop.api.errors import (
    APIClientError,
    AuthenticationError,
    ChatEndpointNotFoundError,
    ContextLimitError,
    HTTPResponseError,
    ModelNotFoundError,
    ModelsEndpointNotFoundError,
    ProtocolError,
    RateLimitError,
    RequestRejectedError,
    RequestTimeoutError,
    ServerConnectionError,
)
from modeltop.chat.models import ChatMessage, GenerationSettings

logger = logging.getLogger(__name__)

type Clock = Callable[[], float]


def normalize_base_url(base_url: str) -> str:
    """Normalize a configured API root to a terminal ``/v1``."""
    normalized = base_url.strip().rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


@dataclass(frozen=True, slots=True)
class RawModelsResponse:
    """Validated models response envelope before item parsing."""

    data: tuple[Mapping[str, object], ...]
    latency_ms: float
    status_code: int


def _causes(error: BaseException) -> tuple[BaseException, ...]:
    causes: list[BaseException] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        causes.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(causes)


def _connection_error(
    error: httpx.ConnectError, *, operation: str = "GET models"
) -> ServerConnectionError:
    causes = _causes(error)
    if any(isinstance(cause, socket.gaierror) for cause in causes):
        return ServerConnectionError(
            "Unable to resolve server",
            f"{operation} failed because hostname resolution failed",
        )
    if any(
        isinstance(cause, OSError) and cause.errno == errno.ECONNREFUSED
        for cause in causes
    ):
        return ServerConnectionError(
            "Connection refused",
            f"{operation} failed because the connection was refused",
        )
    return ServerConnectionError(
        "Unable to connect to server",
        f"{operation} failed with {type(error).__name__}",
    )


def _http_error(status_code: int) -> APIClientError:
    detail = f"GET models returned HTTP {status_code}"
    if status_code in {401, 403}:
        return AuthenticationError(
            "Authentication failed", detail, status_code=status_code
        )
    if status_code == 404:
        return ModelsEndpointNotFoundError(
            "Models endpoint not found", detail, status_code=status_code
        )
    if 500 <= status_code <= 599:
        return HTTPResponseError(
            f"Server error (HTTP {status_code})",
            detail,
            status_code=status_code,
        )
    return HTTPResponseError(
        f"Server returned HTTP {status_code}", detail, status_code=status_code
    )


def _chat_http_error(status_code: int, classification: str | None) -> APIClientError:
    detail = f"POST chat/completions returned HTTP {status_code}"
    if status_code in {401, 403}:
        return AuthenticationError(
            "Authentication failed", detail, status_code=status_code
        )
    if status_code == 404:
        if classification == "model_not_found":
            return ModelNotFoundError(
                "Selected model was not found", detail, status_code=status_code
            )
        return ChatEndpointNotFoundError(
            "Chat endpoint not found", detail, status_code=status_code
        )
    if status_code == 429:
        return RateLimitError(
            "Server rate limit reached", detail, status_code=status_code
        )
    if status_code in {400, 413, 422} and classification == "context_limit":
        return ContextLimitError(
            "Conversation exceeds the server context limit",
            detail,
            status_code=status_code,
        )
    if status_code in {400, 413, 422}:
        return RequestRejectedError(
            "Chat request was rejected", detail, status_code=status_code
        )
    if 500 <= status_code <= 599:
        return HTTPResponseError(
            f"Server error (HTTP {status_code})",
            detail,
            status_code=status_code,
        )
    return HTTPResponseError(
        f"Server returned HTTP {status_code}", detail, status_code=status_code
    )


def _classify_error_payload(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    root = cast(Mapping[object, object], payload)
    error = root.get("error", root)
    if not isinstance(error, Mapping):
        return None
    fields = cast(Mapping[object, object], error)
    values = [
        value.casefold()
        for key in ("type", "code", "param", "message")
        if isinstance((value := fields.get(key)), str)
    ]
    signal = " ".join(values)
    if any(
        marker in signal
        for marker in (
            "model_not_found",
            "model not found",
            "unknown model",
            "does not exist",
        )
    ):
        return "model_not_found"
    unsupported = any(
        marker in signal
        for marker in (
            "unsupported",
            "not support",
            "unknown field",
            "unrecognized",
            "extra inputs",
        )
    )
    if unsupported and ("stream_options" in signal or "include_usage" in signal):
        return "stream_options"
    if unsupported and "stream" in signal:
        return "streaming"
    if any(
        marker in signal
        for marker in (
            "context length",
            "context_length",
            "maximum context",
            "max context",
            "context limit",
            "too many tokens",
            "prompt is too long",
            "input tokens exceed",
            "context window exceeded",
            "requested tokens exceed context",
            "sequence length exceeds maximum",
            "max model length",
            "maximum sequence length",
        )
    ):
        return "context_limit"
    return None


class OpenAICompatibleClient:
    """Pooled client for the generic ``GET /v1/models`` endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key is not None and (trimmed_key := api_key.strip()):
            headers["Authorization"] = f"Bearer {trimmed_key}"
        self._clock = clock
        self._timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(
            base_url=f"{normalize_base_url(base_url)}/",
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            headers=headers,
            transport=transport,
        )

    async def list_models(self) -> RawModelsResponse:
        """Fetch and structurally validate the models response envelope."""
        logger.info("Requesting models endpoint")
        started_at = self._clock()
        try:
            response = await self._client.get("models")
        except httpx.TimeoutException as error:
            raise RequestTimeoutError(
                "Request timed out",
                f"GET models failed with {type(error).__name__}",
            ) from error
        except httpx.ConnectError as error:
            raise _connection_error(error) from error
        except (httpx.RemoteProtocolError, httpx.DecodingError) as error:
            raise ProtocolError(
                "Invalid response from server",
                f"GET models failed with {type(error).__name__}",
            ) from error
        except httpx.RequestError as error:
            raise ServerConnectionError(
                "Unable to connect to server",
                f"GET models failed with {type(error).__name__}",
            ) from error
        latency_ms = (self._clock() - started_at) * 1000
        logger.info("Models endpoint returned HTTP %d", response.status_code)

        if not 200 <= response.status_code <= 299:
            raise _http_error(response.status_code)

        try:
            payload: object = response.json()
        except (ValueError, httpx.DecodingError) as error:
            raise ProtocolError(
                "Invalid response from server",
                f"GET models returned malformed JSON ({type(error).__name__})",
            ) from error

        if not isinstance(payload, Mapping):
            raise ProtocolError(
                "Invalid response from server",
                "GET models response root is not an object",
            )
        envelope = cast(Mapping[object, object], payload)
        data = envelope.get("data")
        if not isinstance(data, list):
            raise ProtocolError(
                "Invalid response from server",
                "GET models response data is not a list",
            )
        items: list[Mapping[str, object]] = []
        for item in cast(list[object], data):
            if not isinstance(item, Mapping):
                raise ProtocolError(
                    "Invalid response from server",
                    "GET models response data contains a non-object item",
                )
            items.append(cast(Mapping[str, object], item))

        return RawModelsResponse(
            data=tuple(items),
            latency_ms=latency_ms,
            status_code=response.status_code,
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        """Stream typed chat events with conservative pre-content fallbacks."""
        effective_timeout = (
            self._timeout_seconds if timeout_seconds is None else timeout_seconds
        )
        include_usage = True
        usage_retry_used = False
        while True:
            payload = build_chat_payload(
                model,
                messages,
                settings,
                stream=True,
                include_usage=include_usage,
            )
            try:
                async with self._client.stream(
                    "POST",
                    "chat/completions",
                    headers={"Accept": "text/event-stream"},
                    json=payload,
                    timeout=effective_timeout,
                ) as response:
                    if not 200 <= response.status_code <= 299:
                        classification = await self._error_classification(response)
                        if classification == "stream_options" and not usage_retry_used:
                            include_usage = False
                            usage_retry_used = True
                            logger.info("Retrying chat without optional stream usage")
                            continue
                        if classification == "streaming":
                            async for event in self._non_stream_chat(
                                model,
                                messages,
                                settings,
                                timeout_seconds=effective_timeout,
                            ):
                                yield event
                            return
                        raise _chat_http_error(response.status_code, classification)

                    yield ResponseStarted(response.status_code)
                    content_type = response.headers.get("content-type", "").casefold()
                    if "application/json" in content_type:
                        try:
                            await response.aread()
                            response_payload: object = response.json()
                        except (ValueError, httpx.DecodingError) as error:
                            raise ProtocolError(
                                "Invalid chat response from server",
                                "Chat completion returned malformed JSON",
                            ) from error
                        yield StreamingFallback(
                            "Streaming unavailable; response received "
                            "in non-streaming mode"
                        )
                        for event in parse_non_stream_completion(response_payload):
                            yield event
                        return
                    if "text/event-stream" not in content_type:
                        raise ProtocolError(
                            "Invalid chat response from server",
                            "Chat stream returned an unexpected content type",
                        )

                    decoder = SSEDecoder()
                    async for chunk in response.aiter_bytes():
                        for event in decoder.feed(chunk):
                            yield event
                    for event in decoder.finish():
                        yield event
                    if not decoder.is_complete:
                        raise ProtocolError(
                            "Connection lost during generation",
                            "Chat stream ended before finish or DONE",
                        )
                    return
            except APIClientError:
                raise
            except httpx.TimeoutException as error:
                raise RequestTimeoutError(
                    "Generation timed out",
                    f"POST chat/completions failed with {type(error).__name__}",
                ) from error
            except httpx.ConnectError as error:
                raise _connection_error(
                    error, operation="POST chat/completions"
                ) from error
            except httpx.RemoteProtocolError as error:
                raise ServerConnectionError(
                    "Connection lost during generation",
                    "POST chat/completions stream ended with a protocol error",
                ) from error
            except httpx.RequestError as error:
                raise ServerConnectionError(
                    "Connection lost during generation",
                    f"POST chat/completions failed with {type(error).__name__}",
                ) from error

    async def _non_stream_chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[ChatStreamEvent]:
        payload = build_chat_payload(
            model,
            messages,
            settings,
            stream=False,
            include_usage=False,
        )
        async with self._client.stream(
            "POST",
            "chat/completions",
            headers={"Accept": "application/json"},
            json=payload,
            timeout=timeout_seconds,
        ) as response:
            if not 200 <= response.status_code <= 299:
                classification = await self._error_classification(response)
                raise _chat_http_error(response.status_code, classification)
            yield ResponseStarted(response.status_code)
            content_type = response.headers.get("content-type", "").casefold()
            if "application/json" not in content_type:
                raise ProtocolError(
                    "Invalid chat response from server",
                    "Non-stream chat completion returned an unexpected content type",
                )
            try:
                await response.aread()
                payload_object: object = response.json()
            except (ValueError, httpx.DecodingError) as error:
                raise ProtocolError(
                    "Invalid chat response from server",
                    "Non-stream chat completion returned malformed JSON",
                ) from error
            yield StreamingFallback(
                "Streaming unavailable; response received in non-streaming mode"
            )
            for event in parse_non_stream_completion(payload_object):
                yield event

    @staticmethod
    async def _error_classification(response: httpx.Response) -> str | None:
        limit = 8192
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > limit:
                return None
            body.extend(chunk)
        try:
            payload: object = json.loads(body)
        except (TypeError, ValueError):
            return None
        return _classify_error_payload(payload)

    async def aclose(self) -> None:
        """Close the owned connection pool."""
        await self._client.aclose()
