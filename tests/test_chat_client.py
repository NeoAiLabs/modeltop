"""OpenAI-compatible chat payload, SSE, retry, and privacy tests."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast

import httpx
import pytest

from modeltop.api.chat import (
    ContentDelta,
    GenerationFinished,
    ResponseStarted,
    SSEDecoder,
    StreamDone,
    StreamingFallback,
    UsageUpdate,
)
from modeltop.api.client import OpenAICompatibleClient
from modeltop.api.errors import (
    AuthenticationError,
    ChatEndpointNotFoundError,
    ContextLimitError,
    HTTPResponseError,
    ModelNotFoundError,
    ProtocolError,
    RateLimitError,
    RequestRejectedError,
)
from modeltop.api.metrics import (
    SpeculativeCounters,
    parse_vllm_speculative_counters,
)
from modeltop.chat.models import ChatMessage, GenerationSettings


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _sse_response(*chunks: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        stream=_Chunks(list(chunks)),
    )


async def _collect(client: OpenAICompatibleClient) -> list[object]:
    return [
        event
        async for event in client.stream_chat_completion(
            "model",
            (ChatMessage("system", "sys"), ChatMessage("user", "hello")),
            GenerationSettings(seed=7),
        )
    ]


def test_vllm_speculative_counter_parser_sums_selected_model_engines() -> None:
    payload = """
# HELP vllm:spec_decode_num_draft_tokens_total Draft tokens
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="selected"} 12
vllm:spec_decode_num_draft_tokens_total{engine="1",model_name="selected"} 8.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="selected"} 9
vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="selected"} 6
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="other"} 100
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="other"} 100
"""
    assert parse_vllm_speculative_counters(payload, "selected") == SpeculativeCounters(
        draft_tokens=20, accepted_tokens=15
    )


@pytest.mark.parametrize(
    "payload",
    [
        'vllm:spec_decode_num_draft_tokens_total{model_name="selected"} 12',
        (
            'vllm:spec_decode_num_draft_tokens_total{model_name="other"} 12\n'
            'vllm:spec_decode_num_accepted_tokens_total{model_name="other"} 9'
        ),
        (
            'vllm:spec_decode_num_draft_tokens_total{model_name="selected"} 12.5\n'
            'vllm:spec_decode_num_accepted_tokens_total{model_name="selected"} 9'
        ),
        'vllm:spec_decode_num_draft_tokens_total{model_name="selected" 12',
    ],
)
def test_vllm_speculative_counter_parser_returns_none_when_unavailable(
    payload: str,
) -> None:
    assert parse_vllm_speculative_counters(payload, "selected") is None


def test_vllm_speculative_counter_request_is_optional_and_authenticated() -> None:
    async def scenario() -> None:
        requests: list[httpx.Request] = []
        responses = [
            httpx.Response(
                200,
                text=(
                    "vllm:spec_decode_num_draft_tokens_total"
                    '{model_name="selected"} 20\n'
                    "vllm:spec_decode_num_accepted_tokens_total"
                    '{model_name="selected"} 15'
                ),
            ),
            httpx.Response(503),
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return responses.pop(0)

        client = OpenAICompatibleClient(
            "http://server/prefix/v1",
            "unique-key",
            3,
            transport=httpx.MockTransport(handler),
        )
        assert await client.get_vllm_speculative_counters(
            "selected", timeout_seconds=2
        ) == SpeculativeCounters(draft_tokens=20, accepted_tokens=15)
        assert (
            await client.get_vllm_speculative_counters("selected", timeout_seconds=2)
            is None
        )
        assert str(requests[0].url) == "http://server/prefix/metrics"
        assert requests[0].headers["authorization"] == "Bearer unique-key"
        await client.aclose()

    asyncio.run(scenario())


def test_exact_endpoint_headers_ordered_payload_and_seed() -> None:
    async def scenario() -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return _sse_response(
                b'data: {"choices":[{"delta":{"content":"ok"},',
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

        client = OpenAICompatibleClient(
            "http://server/prefix/v1",
            "unique-key",
            3,
            transport=httpx.MockTransport(handler),
        )
        events = await _collect(client)
        request = requests[0]
        assert str(request.url) == "http://server/prefix/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer unique-key"
        assert request.headers["accept"] == "text/event-stream"
        payload = cast(dict[str, object], json.loads(request.content))
        assert payload == {
            "model": "model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024,
            "seed": 7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert events == [
            ResponseStarted(200),
            ContentDelta("ok"),
            GenerationFinished("stop", True),
            StreamDone(),
        ]
        await client.aclose()

    asyncio.run(scenario())


def test_decoder_handles_every_byte_boundary_crlf_comments_and_usage() -> None:
    event = (
        ": keepalive\r\n"
        "event: message\r\n"
        'data: {"ignored":1,\r\n'
        'data: "choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n'
        'data: {"choices":[{"delta":{"content":"hé"},'
        '"finish_reason":null}],"unknown":true}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":2,'
        '"completion_tokens":1,"total_tokens":3},"x":0}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    decoder = SSEDecoder()
    events: list[object] = []
    for byte in event:
        events.extend(decoder.feed(bytes([byte])))
    events.extend(decoder.finish())
    assert events == [
        ContentDelta("hé"),
        UsageUpdate(2, 1, 3),
        GenerationFinished("stop", True),
        StreamDone(),
    ]
    assert decoder.is_complete


@pytest.mark.parametrize(
    "payload",
    [
        b"data: {\n\n",
        b"data: []\n\n",
        b'data: {"choices":{}}\n\n',
        b'data: {"choices":[1]}\n\n',
        b'data: {"choices":[{"delta":1}]}\n\n',
        b'data: {"choices":[{"delta":{"content":1}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":true}}\n\n',
    ],
)
def test_decoder_rejects_malformed_json_and_shapes(payload: bytes) -> None:
    decoder = SSEDecoder()
    with pytest.raises(ProtocolError):
        decoder.feed(payload)


def test_clean_finish_without_done_and_early_eof_policy() -> None:
    async def scenario() -> None:
        responses = [
            _sse_response(
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
            ),
            _sse_response(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'),
        ]
        client = OpenAICompatibleClient(
            "http://server",
            None,
            3,
            transport=httpx.MockTransport(lambda request: responses.pop(0)),
        )
        first = await _collect(client)
        assert first == [ResponseStarted(200), GenerationFinished("length", True)]
        with pytest.raises(ProtocolError, match="Connection lost"):
            await _collect(client)
        await client.aclose()

    asyncio.run(scenario())


def test_stream_options_retry_and_non_stream_fallback_are_single_shot() -> None:
    async def scenario() -> None:
        payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = cast(dict[str, object], json.loads(request.content))
            payloads.append(payload)
            if len(payloads) == 1:
                return httpx.Response(
                    422,
                    json={
                        "error": {"message": "unsupported stream_options include_usage"}
                    },
                )
            if len(payloads) == 2:
                return httpx.Response(
                    400,
                    json={"error": {"message": "stream is not supported"}},
                )
            return httpx.Response(
                201,
                json={
                    "choices": [
                        {"message": {"content": "fallback"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        events = await _collect(client)
        assert len(payloads) == 3
        assert "stream_options" in payloads[0]
        assert "stream_options" not in payloads[1]
        assert payloads[2]["stream"] is False
        assert events == [
            ResponseStarted(201),
            StreamingFallback(
                "Streaming unavailable; response received in non-streaming mode"
            ),
            ContentDelta("fallback"),
            UsageUpdate(3, 2, 5),
            GenerationFinished("stop", False),
            StreamDone(),
        ]
        await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("status", "body", "error_type"),
    [
        (400, {"error": {"message": "bad request"}}, RequestRejectedError),
        (
            400,
            {"error": {"message": "prompt is too long"}},
            ContextLimitError,
        ),
        (
            413,
            {"error": {"code": "context_length_exceeded"}},
            ContextLimitError,
        ),
        (413, {"error": {"message": "payload rejected"}}, RequestRejectedError),
        (401, {}, AuthenticationError),
        (403, {}, AuthenticationError),
        (404, {}, ChatEndpointNotFoundError),
        (
            422,
            {"error": {"message": "maximum context length exceeded"}},
            ContextLimitError,
        ),
        (
            404,
            {"error": {"code": "model_not_found"}},
            ModelNotFoundError,
        ),
        (
            500,
            {"error": {"message": "maximum context length exceeded"}},
            HTTPResponseError,
        ),
        (429, {}, RateLimitError),
        (500, {}, HTTPResponseError),
    ],
)
def test_chat_error_mapping_and_sanitization(
    status: int, body: object, error_type: type[Exception]
) -> None:
    async def scenario() -> None:
        client = OpenAICompatibleClient(
            "http://server",
            "unique-secret-key",
            3,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=body)
            ),
        )
        with pytest.raises(error_type) as caught:
            await _collect(client)
        combined = f"{caught.value!s} {getattr(caught.value, 'detail', '')}"
        assert getattr(caught.value, "status_code", None) == status
        assert "unique-secret-key" not in combined
        assert "maximum context length exceeded" not in combined
        await client.aclose()

    asyncio.run(scenario())


def test_json_response_ignoring_stream_falls_back_without_second_request() -> None:
    async def scenario() -> None:
        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "one"}}]},
            )

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        events = await _collect(client)
        assert request_count == 1
        assert events[:2] == [
            ResponseStarted(200),
            StreamingFallback(
                "Streaming unavailable; response received in non-streaming mode"
            ),
        ]
        assert ContentDelta("one") in events
        await client.aclose()

    asyncio.run(scenario())


def test_cancellation_closes_stream_and_client_remains_reusable() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        streams: list[_Chunks] = []

        class GatedStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
                await release.wait()

            async def aclose(self) -> None:
                self.closed = True

        def handler(request: httpx.Request) -> httpx.Response:
            if not streams:
                stream = GatedStream()
                streams.append(cast(_Chunks, stream))
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=stream,
                )
            return _sse_response(b"data: [DONE]\n\n")

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        iterator = client.stream_chat_completion(
            "model", (ChatMessage("user", "prompt"),), GenerationSettings()
        )
        assert await anext(iterator) == ResponseStarted(200)
        assert await anext(iterator) == ContentDelta("first")
        await cast(AsyncGenerator[object, None], iterator).aclose()
        assert streams[0].closed
        assert await _collect(client) == [ResponseStarted(200), StreamDone()]
        await client.aclose()

    asyncio.run(scenario())


def test_request_timeout_override_reaches_stream_and_fallback() -> None:
    async def scenario() -> None:
        timeouts: list[dict[str, float]] = []
        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            timeouts.append(cast(dict[str, float], request.extensions["timeout"]))
            if request_count == 1:
                return _sse_response(b"data: [DONE]\n\n")
            if request_count == 2:
                return httpx.Response(
                    400,
                    json={"error": {"message": "stream is not supported"}},
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "fallback"}}]},
            )

        client = OpenAICompatibleClient(
            "http://server", None, 3, transport=httpx.MockTransport(handler)
        )
        await _collect(client)
        override_events = [
            event
            async for event in client.stream_chat_completion(
                "model",
                (ChatMessage("user", "hello"),),
                GenerationSettings(),
                timeout_seconds=9,
            )
        ]

        assert all(value == 3 for value in timeouts[0].values())
        assert all(value == 9 for timeout in timeouts[1:] for value in timeout.values())
        assert ContentDelta("fallback") in override_events
        await client.aclose()

    asyncio.run(scenario())


def test_usage_parses_speculative_fields_and_aliases() -> None:
    canonical = (
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,'
        b'"total_tokens":14,"draft_tokens":6,"accepted_tokens":4,'
        b'"acceptance_rate":0.6666666666666666}}\n\n'
    )
    alias = (
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"num_draft_tokens":8,"num_accepted_tokens":5,'
        b'"spec_token_acceptance_rate":0.625}}\n\n'
    )
    omitted = (
        b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,'
        b'"total_tokens":3}}\n\n'
    )
    nested = (
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,'
        b'"total_tokens":14,"completion_tokens_details":'
        b'{"accepted_prediction_tokens":4,"rejected_prediction_tokens":2}}}\n\n'
    )
    flat_wins = (
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,'
        b'"total_tokens":14,"draft_tokens":8,"accepted_tokens":5,'
        b'"completion_tokens_details":'
        b'{"accepted_prediction_tokens":4,"rejected_prediction_tokens":2}}}\n\n'
    )
    decoder = SSEDecoder()
    assert decoder.feed(canonical) == (
        UsageUpdate(
            10,
            4,
            14,
            draft_tokens=6,
            accepted_tokens=4,
            acceptance_rate=0.6666666666666666,
        ),
    )
    assert decoder.feed(alias) == (
        UsageUpdate(1, 1, 2, draft_tokens=8, accepted_tokens=5, acceptance_rate=0.625),
    )
    assert decoder.feed(omitted) == (UsageUpdate(2, 1, 3),)
    assert decoder.feed(nested) == (
        UsageUpdate(10, 4, 14, draft_tokens=6, accepted_tokens=4),
    )
    assert decoder.feed(flat_wins) == (
        UsageUpdate(10, 4, 14, draft_tokens=8, accepted_tokens=5),
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"draft_tokens":true}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"accepted_tokens":-1}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"acceptance_rate":true}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"acceptance_rate":1.5}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"acceptance_rate":"0.5"}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"completion_tokens_details":[]}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"completion_tokens_details":'
        b'{"accepted_prediction_tokens":true,"rejected_prediction_tokens":1}}}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,'
        b'"total_tokens":2,"completion_tokens_details":'
        b'{"accepted_prediction_tokens":1,"rejected_prediction_tokens":-1}}}\n\n',
    ],
)
def test_decoder_rejects_malformed_speculative_usage(payload: bytes) -> None:
    decoder = SSEDecoder()
    with pytest.raises(ProtocolError):
        decoder.feed(payload)
