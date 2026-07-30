"""Typed chat payloads and incremental server-sent-event decoding."""

import codecs
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from modeltop.api.errors import ProtocolError
from modeltop.chat.models import ChatMessage, GenerationSettings


@dataclass(frozen=True, slots=True)
class ResponseStarted:
    status_code: int


@dataclass(frozen=True, slots=True)
class ContentDelta:
    text: str


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    draft_tokens: int | None = None
    accepted_tokens: int | None = None
    acceptance_rate: float | None = None


@dataclass(frozen=True, slots=True)
class GenerationFinished:
    finish_reason: str
    streamed: bool


@dataclass(frozen=True, slots=True)
class StreamingFallback:
    reason: str


@dataclass(frozen=True, slots=True)
class StreamDone:
    pass


type ChatStreamEvent = (
    ResponseStarted
    | ContentDelta
    | UsageUpdate
    | GenerationFinished
    | StreamingFallback
    | StreamDone
)


def build_chat_payload(
    model: str,
    messages: Sequence[ChatMessage],
    settings: GenerationSettings,
    *,
    stream: bool = True,
    include_usage: bool = True,
) -> dict[str, object]:
    """Build an ordered generic OpenAI-compatible request payload."""
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": message.role, "content": message.content} for message in messages
        ],
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "stream": stream,
    }
    if settings.seed is not None:
        payload["seed"] = settings.seed
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    if settings.enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": settings.enable_thinking}
    return payload


def _stream_protocol(detail: str) -> ProtocolError:
    return ProtocolError("Invalid chat response from server", detail)


def _token_count(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _stream_protocol(f"Chat usage {field} is not a non-negative integer")
    return value


def _acceptance_rate(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _stream_protocol(f"Chat usage {field} is not a finite number in [0, 1]")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise _stream_protocol(f"Chat usage {field} is not a finite number in [0, 1]")
    return rate


def _first_present(usage: Mapping[object, object], *keys: str) -> object:
    for key in keys:
        if key in usage:
            return usage[key]
    return None


def _completion_token_details(
    usage: Mapping[object, object],
) -> tuple[int | None, int | None]:
    if "completion_tokens_details" not in usage:
        return None, None
    value = usage["completion_tokens_details"]
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        raise _stream_protocol("Chat usage completion_tokens_details is not an object")
    details = cast(Mapping[object, object], value)
    return (
        _token_count(
            details.get("accepted_prediction_tokens"),
            "completion_tokens_details.accepted_prediction_tokens",
        ),
        _token_count(
            details.get("rejected_prediction_tokens"),
            "completion_tokens_details.rejected_prediction_tokens",
        ),
    )


def _parse_usage(value: object) -> UsageUpdate:
    if not isinstance(value, Mapping):
        raise _stream_protocol("Chat usage is not an object")
    usage = cast(Mapping[object, object], value)
    draft_fields = ("draft_tokens", "num_draft_tokens", "speculative_tokens")
    accepted_fields = (
        "accepted_tokens",
        "num_accepted_tokens",
        "spec_accepted_tokens",
    )
    detail_accepted, detail_rejected = _completion_token_details(usage)
    if any(field in usage for field in draft_fields):
        draft_tokens = _token_count(
            _first_present(usage, *draft_fields),
            "draft_tokens",
        )
    elif detail_accepted is not None and detail_rejected is not None:
        draft_tokens = detail_accepted + detail_rejected
    else:
        draft_tokens = None
    accepted_tokens = (
        _token_count(
            _first_present(usage, *accepted_fields),
            "accepted_tokens",
        )
        if any(field in usage for field in accepted_fields)
        else detail_accepted
    )
    return UsageUpdate(
        _token_count(usage.get("prompt_tokens"), "prompt_tokens"),
        _token_count(usage.get("completion_tokens"), "completion_tokens"),
        _token_count(usage.get("total_tokens"), "total_tokens"),
        draft_tokens,
        accepted_tokens,
        _acceptance_rate(
            _first_present(usage, "acceptance_rate", "spec_token_acceptance_rate"),
            "acceptance_rate",
        ),
    )


def parse_stream_json(data: str) -> tuple[ChatStreamEvent, ...]:
    """Validate one independent SSE JSON event and return typed values."""
    try:
        payload: object = json.loads(data)
    except (TypeError, ValueError) as error:
        raise _stream_protocol("Chat stream contained malformed JSON") from error
    if not isinstance(payload, Mapping):
        raise _stream_protocol("Chat stream JSON root is not an object")
    root = cast(Mapping[object, object], payload)
    events: list[ChatStreamEvent] = []

    choices_value = root.get("choices", ())
    if not isinstance(choices_value, list):
        raise _stream_protocol("Chat stream choices is not a list")
    for choice_value in cast(list[object], choices_value):
        if not isinstance(choice_value, Mapping):
            raise _stream_protocol("Chat stream choice is not an object")
        choice = cast(Mapping[object, object], choice_value)
        delta_value = choice.get("delta")
        if not isinstance(delta_value, Mapping):
            raise _stream_protocol("Chat stream delta is not an object")
        delta = cast(Mapping[object, object], delta_value)
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise _stream_protocol("Chat stream content delta is not text")
        if isinstance(content, str) and content:
            events.append(ContentDelta(content))
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise _stream_protocol("Chat stream finish reason is not text")
        if isinstance(finish_reason, str):
            events.append(GenerationFinished(finish_reason, streamed=True))

    if "usage" in root and root["usage"] is not None:
        events.append(_parse_usage(root["usage"]))
    return tuple(events)


def parse_non_stream_completion(payload: object) -> tuple[ChatStreamEvent, ...]:
    """Validate a non-stream completion and expose it as typed events."""
    if not isinstance(payload, Mapping):
        raise _stream_protocol("Chat completion JSON root is not an object")
    root = cast(Mapping[object, object], payload)
    choices_value = root.get("choices")
    if not isinstance(choices_value, list) or not choices_value:
        raise _stream_protocol("Chat completion choices is not a non-empty list")
    choice_value = cast(list[object], choices_value)[0]
    if not isinstance(choice_value, Mapping):
        raise _stream_protocol("Chat completion choice is not an object")
    choice = cast(Mapping[object, object], choice_value)
    message_value = choice.get("message")
    if not isinstance(message_value, Mapping):
        raise _stream_protocol("Chat completion message is not an object")
    message = cast(Mapping[object, object], message_value)
    content = message.get("content")
    if not isinstance(content, str):
        raise _stream_protocol("Chat completion content is not text")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise _stream_protocol("Chat completion finish reason is not text")

    events: list[ChatStreamEvent] = []
    if content:
        events.append(ContentDelta(content))
    if "usage" in root and root["usage"] is not None:
        events.append(_parse_usage(root["usage"]))
    events.append(GenerationFinished(finish_reason or "stop", streamed=False))
    events.append(StreamDone())
    return tuple(events)


class SSEDecoder:
    """Stateful byte decoder for OpenAI-compatible SSE response bodies."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._text = ""
        self._data_lines: list[str] = []
        self.saw_done = False
        self.saw_finish = False

    @property
    def is_complete(self) -> bool:
        return self.saw_done or self.saw_finish

    def feed(self, chunk: bytes) -> tuple[ChatStreamEvent, ...]:
        """Decode one arbitrary byte fragment."""
        try:
            self._text += self._utf8.decode(chunk, final=False)
        except UnicodeDecodeError as error:
            raise _stream_protocol("Chat stream is not valid UTF-8") from error
        return self._drain_lines(final=False)

    def finish(self) -> tuple[ChatStreamEvent, ...]:
        """Flush valid final text and dispatch pending event data at EOF."""
        try:
            self._text += self._utf8.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise _stream_protocol(
                "Chat stream ended inside a UTF-8 code point"
            ) from error
        events = list(self._drain_lines(final=True))
        if self._data_lines:
            events.extend(self._dispatch())
        return tuple(events)

    def _drain_lines(self, *, final: bool) -> tuple[ChatStreamEvent, ...]:
        events: list[ChatStreamEvent] = []
        while self._text:
            lf = self._text.find("\n")
            cr = self._text.find("\r")
            positions = [position for position in (lf, cr) if position >= 0]
            if not positions:
                if final:
                    line, self._text = self._text, ""
                    events.extend(self._process_line(line))
                break
            end = min(positions)
            if self._text[end] == "\r" and end + 1 == len(self._text) and not final:
                break
            delimiter_length = 2 if self._text[end : end + 2] == "\r\n" else 1
            line = self._text[:end]
            self._text = self._text[end + delimiter_length :]
            events.extend(self._process_line(line))
        return tuple(events)

    def _process_line(self, line: str) -> tuple[ChatStreamEvent, ...]:
        if not line:
            return self._dispatch()
        if line.startswith(":"):
            return ()
        field, separator, value = line.partition(":")
        if field != "data":
            return ()
        if separator and value.startswith(" "):
            value = value[1:]
        self._data_lines.append(value)
        return ()

    def _dispatch(self) -> tuple[ChatStreamEvent, ...]:
        if not self._data_lines:
            return ()
        data = "\n".join(self._data_lines)
        self._data_lines.clear()
        if data.strip() == "[DONE]":
            self.saw_done = True
            return (StreamDone(),)
        events = parse_stream_json(data)
        if any(isinstance(event, GenerationFinished) for event in events):
            self.saw_finish = True
        return events
