"""Reusable runner and dashboard chat state-transition tests."""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace

import pytest

from modeltop.api.chat import (
    ChatStreamEvent,
    ContentDelta,
    GenerationFinished,
    ResponseStarted,
    StreamDone,
    StreamingFallback,
    UsageUpdate,
)
from modeltop.api.errors import HTTPResponseError, ProtocolError
from modeltop.chat.models import ChatMessage, GenerationSettings, GenerationStatus
from modeltop.models import DiscoveredModel
from modeltop.services.chat import ChatOperationError, DashboardChatService
from modeltop.services.generation import (
    GenerationCancelled,
    GenerationFailed,
    GenerationProgress,
    GenerationRequest,
    GenerationService,
)
from modeltop.state import (
    ApplicationState,
    ApplicationStateStore,
    HardwareStatus,
    ServerStatus,
    initial_application_state,
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
        self.requests: list[tuple[str, Sequence[ChatMessage], GenerationSettings]] = []

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        self.requests.append((model, messages, settings))
        for item in self.scripts.pop(0):
            await asyncio.sleep(0)
            if isinstance(item, Exception):
                raise item
            yield item


class _GatedClient:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.requests = 0

    async def stream_chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[ChatStreamEvent]:
        self.requests += 1
        try:
            yield ResponseStarted(200)
            yield ContentDelta("partial")
            self.entered.set()
            await self.release.wait()
            yield GenerationFinished("stop", True)
            yield StreamDone()
        finally:
            self.closed = True


def _store(*, online: bool = True, model: bool = True) -> ApplicationStateStore:
    state = initial_application_state("server", hardware_enabled=True)
    state = replace(
        state,
        server_status=ServerStatus.ONLINE if online else ServerStatus.OFFLINE,
        selected_model_id="model" if model else None,
        available_models=(DiscoveredModel(id="model"),) if model else (),
        hardware_status=HardwareStatus.DEGRADED,
        hardware_last_error="hardware marker",
        last_error="server marker",
    )
    return ApplicationStateStore(state)


def test_blank_offline_no_model_and_overlap_reject_without_requests() -> None:
    client = _ScriptedClient([])
    for store, prompt, message in (
        (_store(), " \n", "nonblank"),
        (_store(online=False), "hello", "offline"),
        (_store(model=False), "hello", "Select"),
    ):
        service = DashboardChatService(
            GenerationService(client), store, lambda state: None
        )
        with pytest.raises(ChatOperationError, match=message):
            service.begin_generation(prompt)
    assert client.requests == []

    service = DashboardChatService(
        GenerationService(client), _store(), lambda state: None
    )
    service.begin_generation("first")
    with pytest.raises(ChatOperationError, match="already"):
        service.begin_generation("second")
    assert client.requests == []


def test_success_resends_full_history_and_commits_one_assistant() -> None:
    async def scenario() -> None:
        client = _ScriptedClient(
            [
                [
                    ContentDelta("answer"),
                    UsageUpdate(5, 2, 7),
                    GenerationFinished("stop", True),
                    StreamDone(),
                ],
                [ContentDelta("next"), GenerationFinished("stop", True), StreamDone()],
            ]
        )
        store = _store()
        states: list[ApplicationState] = []
        service = DashboardChatService(
            GenerationService(client, clock=_Clock()), store, states.append
        )
        first = service.begin_generation("question")
        assert service.state.generation_status is GenerationStatus.STARTING
        assert service.state.chat_session.messages == (ChatMessage("user", "question"),)
        await service.generate(first)
        assert service.state.generation_status is GenerationStatus.COMPLETED
        assert service.state.active_generation_id is None
        assert service.state.chat_session.messages == (
            ChatMessage("user", "question"),
            ChatMessage("assistant", "answer"),
        )
        assert service.state.generation_metrics is not None
        assert service.state.generation_metrics.total_tokens == 7

        second = service.begin_generation("follow-up")
        await service.generate(second)
        sent = client.requests[1][1]
        assert sent == (
            ChatMessage("system", "You are a helpful assistant."),
            ChatMessage("user", "question"),
            ChatMessage("assistant", "answer"),
            ChatMessage("user", "follow-up"),
        )
        assert service.state.hardware_last_error == "hardware marker"
        assert service.state.last_error == "server marker"
        assert any(state.current_response == "answer" for state in states)

    asyncio.run(scenario())


def test_fallback_notice_error_partial_recovery_and_safe_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        caplog.set_level(logging.INFO)
        client = _ScriptedClient(
            [
                [
                    StreamingFallback("fallback notice"),
                    ContentDelta("unique-completion"),
                    GenerationFinished("stop", False),
                    StreamDone(),
                ],
                [
                    ContentDelta("safe partial"),
                    HTTPResponseError("Readable failure", "safe typed detail"),
                ],
                [
                    ContentDelta("recovered"),
                    GenerationFinished("stop", True),
                    StreamDone(),
                ],
            ]
        )
        service = DashboardChatService(
            GenerationService(client, clock=_Clock()),
            _store(),
            lambda state: None,
        )
        await service.generate(service.begin_generation("unique-prompt"))
        assert service.state.generation_notice == "fallback notice"
        assert service.state.generation_metrics is not None
        assert not service.state.generation_metrics.streamed

        with pytest.raises(GenerationFailed):
            await service.generate(service.begin_generation("fail"))
        assert service.state.generation_status is GenerationStatus.ERROR
        assert service.state.generation_error == "Readable failure"
        assert service.state.chat_session.messages[-1] == ChatMessage(
            "assistant", "safe partial"
        )
        await service.generate(service.begin_generation("again"))
        assert service.state.generation_status is GenerationStatus.COMPLETED
        assert "unique-prompt" not in caplog.text
        assert "unique-completion" not in caplog.text

    asyncio.run(scenario())


def test_cancellation_commits_partial_and_allows_immediate_next_request() -> None:
    async def scenario() -> None:
        client = _GatedClient()
        service = DashboardChatService(
            GenerationService(client, clock=_Clock()),
            _store(),
            lambda state: None,
        )
        pending = service.begin_generation("cancel me")
        task = asyncio.create_task(service.generate(pending))
        await client.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.closed
        assert service.state.generation_status is GenerationStatus.CANCELLED
        assert service.state.active_generation_id is None
        assert service.state.chat_session.messages[-1] == ChatMessage(
            "assistant", "partial"
        )
        assert service.state.generation_metrics is not None
        assert service.state.generation_metrics.cancelled
        next_pending = service.begin_generation("next")
        assert next_pending.generation_id == pending.generation_id + 1

    asyncio.run(scenario())


def test_clear_and_preferences_preserve_other_runtime_lanes() -> None:
    store = _store()
    client = _ScriptedClient([])
    service = DashboardChatService(GenerationService(client), store, lambda state: None)
    settings = GenerationSettings(temperature=1.1, seed=8)
    service.update_preferences(settings, "custom", True)
    state = service.state
    assert state.chat_session.settings == settings
    assert state.chat_session.system_prompt == "custom"
    assert state.chat_session.show_system_prompt
    assert state.hardware_last_error == "hardware marker"
    assert state.last_error == "server marker"

    service.clear_conversation()
    assert service.state.chat_session.settings == settings
    pending = service.begin_generation("active")
    with pytest.raises(ChatOperationError):
        service.clear_conversation()
    with pytest.raises(ChatOperationError):
        service.update_preferences(GenerationSettings(), "other", False)
    assert service.state.active_generation_id == pending.generation_id


def test_unexpected_client_and_callback_failures_release_reservation() -> None:
    async def client_failure() -> None:
        client = _ScriptedClient(
            [[ContentDelta("safe partial"), RuntimeError("private failure")]]
        )
        service = DashboardChatService(
            GenerationService(client, clock=_Clock()),
            _store(),
            lambda state: None,
        )
        with pytest.raises(RuntimeError, match="private failure"):
            await service.generate(service.begin_generation("private prompt"))
        assert service.state.active_generation_id is None
        assert service.state.generation_status is GenerationStatus.ERROR
        assert service.state.generation_error == "Generation failed"
        assert service.state.chat_session.messages[-1] == ChatMessage(
            "assistant", "safe partial"
        )

    async def callback_failure() -> None:
        def callback(state: ApplicationState) -> None:
            if state.generation_status is GenerationStatus.STREAMING:
                raise RuntimeError("render failed")

        client = _ScriptedClient(
            [[ContentDelta("safe partial"), GenerationFinished("stop", True)]]
        )
        service = DashboardChatService(
            GenerationService(client, clock=_Clock()),
            _store(),
            callback,
        )
        with pytest.raises(RuntimeError, match="render failed"):
            await service.generate(service.begin_generation("prompt"))
        assert service.state.active_generation_id is None
        assert service.state.generation_error == "Generation failed"

    asyncio.run(client_failure())
    asyncio.run(callback_failure())


def test_cancel_before_run_and_stale_generation_ids_are_noops() -> None:
    client = _ScriptedClient([])
    service = DashboardChatService(
        GenerationService(client), _store(), lambda state: None
    )
    pending = service.begin_generation("reserved")
    assert not service.cancel_reservation(pending.generation_id + 1)
    assert service.state.active_generation_id == pending.generation_id
    assert service.cancel_reservation(pending.generation_id)
    assert service.state.active_generation_id is None
    assert service.state.generation_status is GenerationStatus.CANCELLED
    assert not service.cancel_reservation(pending.generation_id)
    assert client.requests == []


def test_generation_status_code_survives_progress_and_safe_failures() -> None:
    async def scenario() -> None:
        client = _ScriptedClient(
            [
                [
                    ResponseStarted(201),
                    ContentDelta("complete"),
                    GenerationFinished("stop", True),
                    StreamDone(),
                ],
                [
                    HTTPResponseError(
                        "Rate limited",
                        "safe detail",
                        status_code=429,
                    )
                ],
                [
                    ResponseStarted(206),
                    ContentDelta("partial"),
                    ProtocolError("Connection lost", "safe protocol detail"),
                ],
            ]
        )
        service = GenerationService(client, clock=_Clock())
        request = GenerationRequest(
            "server",
            "model",
            (ChatMessage("user", "prompt"),),
            GenerationSettings(),
        )

        progress: list[GenerationProgress] = []
        outcome = await service.run(request, progress.append)
        assert outcome.status_code == 201
        assert progress
        assert all(update.status_code == 201 for update in progress)

        with pytest.raises(GenerationFailed) as rejected:
            await service.run(request, lambda update: None)
        assert rejected.value.error.status_code == 429
        assert rejected.value.outcome.status_code == 429

        protocol_progress: list[GenerationProgress] = []
        with pytest.raises(GenerationFailed) as interrupted:
            await service.run(request, protocol_progress.append)
        assert interrupted.value.error.status_code is None
        assert interrupted.value.outcome.status_code == 206
        assert all(update.status_code == 206 for update in protocol_progress)

    asyncio.run(scenario())


def test_generation_cancellation_retains_accepted_response_status() -> None:
    async def scenario() -> None:
        client = _GatedClient()
        service = GenerationService(client, clock=_Clock())
        request = GenerationRequest(
            "server",
            "model",
            (ChatMessage("user", "prompt"),),
            GenerationSettings(),
        )
        task = asyncio.create_task(service.run(request, lambda update: None))
        await client.entered.wait()
        task.cancel()
        with pytest.raises(GenerationCancelled) as cancelled:
            await task
        assert cancelled.value.outcome.status_code == 200
        assert cancelled.value.outcome.content == "partial"
        assert client.closed

    asyncio.run(scenario())
