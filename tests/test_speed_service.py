"""Sequential orchestration, failure, cancellation, and reservation contracts."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest

from modeltop.api.chat import (
    ChatStreamEvent,
    ContentDelta,
    GenerationFinished,
    StreamDone,
    UsageUpdate,
)
from modeltop.api.client import OpenAICompatibleClient
from modeltop.api.errors import HTTPResponseError
from modeltop.benchmarks.models import (
    SpeedTestConfig,
    SpeedTestResult,
    SpeedTestStatus,
)
from modeltop.chat.models import ChatMessage, GenerationSettings
from modeltop.models import DiscoveredModel, ServerConfig
from modeltop.services.chat import ChatOperationError, DashboardChatService
from modeltop.services.generation import GenerationService
from modeltop.services.speed_test import SpeedTestOperationError, SpeedTestService
from modeltop.state import (
    ApplicationState,
    ApplicationStateStore,
    ServerStatus,
    initial_application_state,
)


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self) -> None:
        pass


def _sse_response(*chunks: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        stream=_Chunks(list(chunks)),
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.1
        return self.value


class _ScriptedClient:
    def __init__(self, scripts: list[list[ChatStreamEvent | Exception]]) -> None:
        self.scripts = scripts
        self.requests: list[
            tuple[str, Sequence[ChatMessage], GenerationSettings, float | None]
        ] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        self.requests.append((model, messages, settings, timeout_seconds))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for item in self.scripts.pop(0):
                await asyncio.sleep(0)
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.in_flight -= 1


class _GatedClient:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = False

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        try:
            yield ContentDelta("partial")
            self.entered.set()
            await asyncio.Event().wait()
        finally:
            self.closed = True


def _success(text: str = "answer") -> list[ChatStreamEvent | Exception]:
    return [
        ContentDelta(text),
        UsageUpdate(10, 4, 14),
        GenerationFinished("stop", True),
        StreamDone(),
    ]


def _store(*, online: bool = True, model: bool = True) -> ApplicationStateStore:
    state = initial_application_state("server", hardware_enabled=False)
    return ApplicationStateStore(
        replace(
            state,
            server_status=ServerStatus.ONLINE if online else ServerStatus.OFFLINE,
            selected_model_id="model" if model else None,
            available_models=(
                (DiscoveredModel(id="model", owned_by="vllm"),) if model else ()
            ),
        )
    )


def _service(
    client: _ScriptedClient | _GatedClient | OpenAICompatibleClient,
    store: ApplicationStateStore,
) -> SpeedTestService:
    server = ServerConfig(
        id="server",
        name="Local",
        base_url="http://localhost:8000/v1",
        backend_hint=None,
    )
    return SpeedTestService(
        GenerationService(client, clock=_Clock()),
        store,
        server,
        lambda state: None,
        utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
        run_id_factory=lambda started_at: f"speed-test-{id(client):x}",
    )


def test_warmups_then_measured_are_strictly_sequential_and_aggregated() -> None:
    async def scenario() -> None:
        client = _ScriptedClient([_success("warm"), _success("one"), _success("two")])
        store = _store()
        states: list[ApplicationState] = []
        service = SpeedTestService(
            GenerationService(client, clock=_Clock()),
            store,
            ServerConfig(id="server", name="Local", base_url="http://localhost:8000"),
            states.append,
            utc_now=lambda: datetime(2026, 7, 27, tzinfo=UTC),
            run_id_factory=lambda started_at: "speed-test-fixed",
        )
        config = SpeedTestConfig(
            prompt="benchmark prompt",
            warmup_runs=1,
            measured_runs=2,
            max_tokens=77,
            temperature=0.2,
            top_p=0.8,
            seed=9,
            request_timeout_seconds=12,
            thinking_mode="disabled",
        )
        result = await service.run_test(service.begin_test(config))

        assert client.max_in_flight == 1
        assert [run.warmup for run in result.run_results] == [True, False, False]
        assert [run.run_number for run in result.run_results] == [1, 1, 2]
        assert result.status is SpeedTestStatus.COMPLETED
        assert result.successful_runs == 2
        assert result.output_tokens_per_second.count == 2
        assert result.prompt_tokens.mean == 10
        assert all(request[0] == "model" for request in client.requests)
        assert all(
            request[1] == (ChatMessage("user", "benchmark prompt"),)
            for request in client.requests
        )
        assert all(request[2].max_tokens == 77 for request in client.requests)
        assert all(request[2].enable_thinking is False for request in client.requests)
        assert all(request[3] == 12 for request in client.requests)
        assert result.backend == "vLLM"
        assert service.state.speed_test.results == (result,)
        assert service.state.speed_test.run_id is None
        assert any(state.speed_test.live_output_preview == "one" for state in states)

    asyncio.run(scenario())


def test_stop_on_error_and_continue_on_error_results() -> None:
    async def scenario() -> None:
        failure = HTTPResponseError("Readable failure", "safe detail")
        stop_client = _ScriptedClient([_success(), [failure], _success()])
        stop_service = _service(stop_client, _store())
        stopped = await stop_service.run_test(
            stop_service.begin_test(SpeedTestConfig(warmup_runs=1, measured_runs=2))
        )
        assert stopped.status is SpeedTestStatus.FAILED
        assert len(stopped.run_results) == 2
        assert stopped.error == "Readable failure"
        assert len(stop_client.requests) == 2

        continue_client = _ScriptedClient([[failure], _success(), [failure]])
        continue_service = _service(continue_client, _store())
        continued = await continue_service.run_test(
            continue_service.begin_test(
                SpeedTestConfig(
                    warmup_runs=0,
                    measured_runs=3,
                    continue_on_error=True,
                )
            )
        )
        assert continued.status is SpeedTestStatus.COMPLETED_WITH_ERRORS
        assert continued.successful_runs == 1
        assert continued.failed_runs == 2
        assert continued.output_tokens_per_second.count == 1
        assert len(continue_client.requests) == 3

        all_failed_client = _ScriptedClient([[failure], [failure]])
        all_failed_service = _service(all_failed_client, _store())
        all_failed = await all_failed_service.run_test(
            all_failed_service.begin_test(
                SpeedTestConfig(
                    warmup_runs=0,
                    measured_runs=2,
                    continue_on_error=True,
                )
            )
        )
        assert all_failed.status is SpeedTestStatus.COMPLETED_WITH_ERRORS
        assert all_failed.successful_runs == 0
        assert all_failed.output_tokens_per_second.count == 0
        assert all_failed.output_tokens_per_second.mean is None

    asyncio.run(scenario())


def test_cancellation_retains_partial_and_allows_rerun() -> None:
    async def scenario() -> None:
        client = _GatedClient()
        store = _store()
        service = _service(client, store)
        pending = service.begin_test(SpeedTestConfig(warmup_runs=0, measured_runs=1))
        task = asyncio.create_task(service.run_test(pending))
        await client.entered.wait()
        assert service.request_cancellation(pending.run_id)
        assert service.state.speed_test.status is SpeedTestStatus.CANCELLING
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.closed
        result = service.state.speed_test.latest_result
        assert result is not None
        assert result.status is SpeedTestStatus.CANCELLED
        assert result.cancelled_runs == 1
        assert result.run_results[0].response_character_count == len("partial")

        scripted = _ScriptedClient([_success()])
        rerun_service = _service(scripted, store)
        rerun = await rerun_service.run_test(
            rerun_service.begin_test(SpeedTestConfig(warmup_runs=0, measured_runs=1))
        )
        assert rerun.status is SpeedTestStatus.COMPLETED
        assert len(rerun_service.state.speed_test.results) == 2
        assert rerun_service.state.speed_test.results[0] is result
        assert result.status is SpeedTestStatus.CANCELLED

    asyncio.run(scenario())


def test_preflight_guards_cancel_before_entry_and_stale_ids() -> None:
    client = _ScriptedClient([])
    for store, message in (
        (_store(online=False), "offline"),
        (_store(model=False), "Select"),
    ):
        with pytest.raises(SpeedTestOperationError, match=message):
            _service(client, store).begin_test(SpeedTestConfig())
    assert client.requests == []

    chat_active_store = _store()
    chat_active_store.update(
        lambda state: replace(state, active_generation_id=state.generation_id + 1)
    )
    with pytest.raises(SpeedTestOperationError, match="Chat generation"):
        _service(client, chat_active_store).begin_test(SpeedTestConfig())

    store = _store()
    service = _service(client, store)
    pending = service.begin_test(SpeedTestConfig())
    with pytest.raises(SpeedTestOperationError, match="already"):
        service.begin_test(SpeedTestConfig())
    chat = DashboardChatService(GenerationService(client), store, lambda state: None)
    with pytest.raises(ChatOperationError, match="Speed Test"):
        chat.begin_generation("hello")
    cancelled = service.cancel_reservation(pending)
    assert cancelled is not None
    assert cancelled.status is SpeedTestStatus.CANCELLED
    assert service.cancel_reservation(pending) is None
    assert not service.request_cancellation("stale")


def test_estimated_usage_and_unexpected_failure_cleanup() -> None:
    async def scenario() -> None:
        estimated_client = _ScriptedClient(
            [[ContentDelta("estimated output"), GenerationFinished("stop", True)]]
        )
        estimated_service = _service(estimated_client, _store())
        estimated = await estimated_service.run_test(
            estimated_service.begin_test(
                SpeedTestConfig(warmup_runs=0, measured_runs=1)
            )
        )
        run = estimated.run_results[0]
        assert run.prompt_tokens_estimated
        assert run.completion_tokens_estimated
        assert run.total_tokens_estimated
        assert estimated.estimated_measured_metrics

        broken_client = _ScriptedClient(
            [[ContentDelta("safe partial"), RuntimeError("private failure")]]
        )
        broken_service = _service(broken_client, _store())
        with pytest.raises(RuntimeError, match="private failure"):
            await broken_service.run_test(
                broken_service.begin_test(
                    SpeedTestConfig(warmup_runs=0, measured_runs=1)
                )
            )
        failed = broken_service.state.speed_test.latest_result
        assert failed is not None
        assert failed.status is SpeedTestStatus.FAILED
        assert failed.error == "Speed Test failed"
        assert failed.run_results[0].response_character_count == len("safe partial")
        assert broken_service.state.speed_test.run_id is None

    asyncio.run(scenario())


def test_openai_compatible_streaming_metric_boundaries() -> None:
    async def scenario() -> None:
        responses = [
            _sse_response(
                b'data: {"choices":[{"delta":{"content":"estimated output"},'
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            ),
            httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "choices": [
                        {
                            "message": {"content": "non-stream response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                },
            ),
            _sse_response(
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":10,'
                b'"completion_tokens":3,"total_tokens":13}}\n\n'
                b"data: [DONE]\n\n"
            ),
        ]
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return responses.pop(0)

        client = OpenAICompatibleClient(
            "http://server/v1",
            None,
            3,
            transport=httpx.MockTransport(handler),
        )
        results: list[SpeedTestResult] = []
        for _ in range(3):
            service = _service(client, _store())
            pending = service.begin_test(
                SpeedTestConfig(warmup_runs=0, measured_runs=1)
            )
            results.append(await service.run_test(pending))

        for request in requests:
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["accept"] == "text/event-stream"
            assert json.loads(request.content)["stream"] is True
            assert json.loads(request.content)["stream_options"] == {
                "include_usage": True
            }

        textual = results[0]
        assert textual.run_results[0].streamed
        assert textual.run_results[0].completion_tokens_estimated
        assert textual.run_results[0].output_tokens_per_second is not None
        assert textual.output_tokens_per_second.count == 1

        fallback = results[1]
        assert not fallback.run_results[0].streamed
        assert fallback.run_results[0].response_character_count > 0
        assert fallback.run_results[0].completion_tokens == 3
        assert fallback.run_results[0].output_tokens_per_second is None
        assert fallback.output_tokens_per_second.count == 0
        contentless = results[2]
        assert contentless.run_results[0].streamed
        assert contentless.run_results[0].response_character_count == 0
        assert contentless.run_results[0].completion_tokens == 3
        assert contentless.run_results[0].output_tokens_per_second is None
        assert contentless.output_tokens_per_second.count == 0
        await client.aclose()

    asyncio.run(scenario())
