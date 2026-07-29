"""Textual Pilot coverage for inline chat, streaming, cancellation, and focus."""
# pyright: reportPrivateUsage=false

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

import httpx
from textual.widgets import (
    ContentSwitcher,
    Input,
    Markdown,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets._toast import Toast

from modeltop.api.client import OpenAICompatibleClient
from modeltop.app import ModelTopApp
from modeltop.chat.models import GenerationStatus
from modeltop.screens.chat import ChatView
from modeltop.state import ServerStatus
from modeltop.widgets.chat_history import ChatHistory
from modeltop.widgets.chat_input import PromptTextArea
from modeltop.widgets.generation_metrics import GenerationMetricsView
from modeltop.widgets.generation_settings import GenerationSettingsPanel
from tests.test_app import (
    _config,
    _plain_render,
    _ScriptedHardwareProvider,
    _wait_for_status,
)


class _GatedChatStream(httpx.AsyncByteStream):
    def __init__(self, first: str, rest: str) -> None:
        self.first = first
        self.rest = rest
        self.first_sent = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _delta(self.first)
        self.first_sent.set()
        await self.release.wait()
        yield _delta(self.rest)
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":{"prompt_tokens":4,'
            b'"completion_tokens":2,"total_tokens":6}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True



class _SteppedChatStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[str, ...]) -> None:
        self.chunks = chunks
        self.sent = tuple(asyncio.Event() for _ in chunks)
        self.advance = tuple(asyncio.Event() for _ in chunks[:-1])

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            yield _delta(chunk)
            self.sent[index].set()
            if index < len(self.advance):
                await self.advance[index].wait()
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":{"prompt_tokens":4,'
            b'"completion_tokens":2,"total_tokens":6}}\n\n'
        )
        yield b"data: [DONE]\n\n"



def _delta(text: str) -> bytes:
    payload = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


class _ChatTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        streams: list[_GatedChatStream | _SteppedChatStream] | None = None,
        *,
        fail_model_request: int | None = None,
        gate_model_request: int | None = None,
    ) -> None:
        self.streams = streams or []
        self.fail_model_request = fail_model_request
        self.gate_model_request = gate_model_request
        self.model_requests = 0
        self.model_request_started = asyncio.Event()
        self.release_model_request = asyncio.Event()
        self.chat_requests: list[dict[str, object]] = []
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.model_requests += 1
            if self.model_requests == self.fail_model_request:
                raise httpx.ReadError("broken", request=request)
            if self.model_requests == self.gate_model_request:
                self.model_request_started.set()
                await self.release_model_request.wait()
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        payload = cast(dict[str, object], json.loads(request.content))
        self.chat_requests.append(payload)
        if self.streams:
            stream = self.streams.pop(0)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )
        stream = _GatedChatStream("complete", "")
        stream.release.set()
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def aclose(self) -> None:
        self.close_count += 1


def _app(
    transport: _ChatTransport,
    *,
    refresh_interval: float = 3600,
) -> ModelTopApp:
    client = OpenAICompatibleClient(
        "http://server/prefix/v1", None, 5, transport=transport
    )
    return ModelTopApp(
        _config(refresh_interval=refresh_interval),
        client=client,
        hardware_provider=_ScriptedHardwareProvider(),
    )


def test_chat_remains_usable_after_automatic_refresh_failure() -> None:
    async def scenario() -> None:
        transport = _ChatTransport(
            fail_model_request=2,
            gate_model_request=3,
        )
        app = _app(transport, refresh_interval=0.02)
        async with app.run_test(
            size=(100, 30),
            notifications=True,
        ) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await pilot.press("down", "enter")
            await asyncio.wait_for(
                transport.model_request_started.wait(),
                timeout=3,
            )
            refresh_worker = app._refresh_worker
            assert refresh_worker is not None
            assert refresh_worker.is_running
            assert app._refresh_timer is not None
            app._refresh_timer.pause()
            await pilot.pause()
            await pilot.pause()

            state = app.dashboard_state
            assert state is not None
            assert state.server_status is ServerStatus.ONLINE
            assert state.is_refreshing
            assert state.selected_model_id == "model"
            assert state.last_error == "Unable to connect to server"
            chat_status = _plain_render(app.query_one("#chat-status", Static))
            assert "READY · Enter sends" in chat_status
            assert "OFFLINE" not in chat_status
            assert len(app.query(Toast)) == 0

            editor = app.query_one("#chat-prompt", PromptTextArea)
            editor.text = "hello"
            editor.action_send()
            for _ in range(100):
                await asyncio.sleep(0.005)
                latest = app.dashboard_state
                if (
                    latest is not None
                    and latest.generation_status is GenerationStatus.COMPLETED
                ):
                    break
            completed = app.dashboard_state
            assert completed is not None
            assert completed.generation_status is GenerationStatus.COMPLETED
            assert len(transport.chat_requests) == 1
            assert completed.chat_session.messages[-1].content == "complete"
            assert refresh_worker.is_running

            transport.release_model_request.set()
            recovery = await asyncio.wait_for(refresh_worker.wait(), timeout=3)
            assert recovery.success
            recovered = app.dashboard_state
            assert recovered is not None
            assert recovered.server_status is ServerStatus.ONLINE
            assert not recovered.is_refreshing
            assert recovered.last_error is None
            assert len(app.query(Toast)) == 0
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_sidebar_ids_inline_navigation_focus_and_geometry() -> None:
    async def scenario() -> None:
        transport = _ChatTransport()
        app = _app(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            menu = app.query_one("#sidebar-menu", OptionList)
            assert [
                menu.get_option_at_index(i).id for i in range(menu.option_count)
            ] == [
                "overview",
                "chat",
                "speed-test",
                "concurrency",
                "context",
                "tool-calling",
                "drafter",
                "results",
                "settings",
            ]
            screens = tuple(app.screen_stack)
            await pilot.press("down", "enter")
            await pilot.pause()
            assert tuple(app.screen_stack) == screens
            assert app.query_one(ContentSwitcher).current == "chat-workspace"
            assert _plain_render(app.query_one("#header-subtitle", Static)) == (
                "LOCAL HARDWARE · CHAT READY"
            )
            assert isinstance(app.focused, TextArea)
            chat = app.query_one(ChatView)
            assert chat.region.width > 0 and chat.region.height > 0
            assert app.query_one(ChatHistory).region.height >= 4
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert chat.region.width > 0 and chat.region.height > 0
            menu.focus()
            menu.highlighted = 0
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one(ContentSwitcher).current == "overview-workspace"
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_gated_stream_is_incremental_history_survives_and_metrics_finalize() -> None:
    async def scenario() -> None:
        stream = _GatedChatStream("partial ", "answer")
        follow_stream = _GatedChatStream("complete", "")
        transport = _ChatTransport([stream, follow_stream])
        app = _app(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await pilot.press("down", "enter")
            await pilot.pause()
            await pilot.press("h", "e", "l", "l", "o", "enter")
            await asyncio.wait_for(stream.first_sent.wait(), timeout=3)
            for _ in range(10):
                await pilot.pause()
                state = app.dashboard_state
                if state is not None and state.current_response == "partial ":
                    break
            assert app.dashboard_state is not None
            assert app.dashboard_state.current_response == "partial "
            assert app.dashboard_state.generation_status is GenerationStatus.STREAMING
            markdown = tuple(app.query(Markdown))[-1]
            assert markdown.source == "partial "
            metrics = app.query_one(GenerationMetricsView)
            assert metrics.has_class("visible")

            menu = app.query_one("#sidebar-menu", OptionList)
            menu.focus()
            menu.highlighted = 0
            await pilot.press("enter")
            stream.release.set()
            for _ in range(30):
                await pilot.pause()
                latest = app.dashboard_state
                if latest.generation_status is GenerationStatus.COMPLETED:
                    break
            assert app.query_one(ContentSwitcher).current == "overview-workspace"
            assert app.focused is menu
            menu.highlighted = 1
            await pilot.press("enter")
            await pilot.pause()
            completed_state = app.dashboard_state
            assert completed_state is not None
            assert completed_state.chat_session.messages[-1].content == "partial answer"
            assert completed_state.generation_metrics is not None
            assert completed_state.generation_metrics.total_tokens == 6
            assert transport.chat_requests[0]["messages"] == [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ]
            await pilot.press("f", "o", "l", "l", "o", "w", "enter")
            await asyncio.wait_for(follow_stream.first_sent.wait(), timeout=3)
            for _ in range(30):
                await pilot.pause()
                latest = app.dashboard_state
                if latest.active_generation_id is None:
                    break
            second_messages = cast(list[object], transport.chat_requests[1]["messages"])
            assert second_messages[1:4] == [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "partial answer"},
                {"role": "user", "content": "follow"},
            ]
            follow_stream.release.set()
        assert stream.closed

    asyncio.run(scenario())


def test_streamed_chat_follows_bottom_until_reader_scrolls_away() -> None:
    async def scenario() -> None:
        chunks = tuple(
            f"fragment {index} {'x' * 60}\n" * 16 for index in range(1, 5)
        )
        stream = _SteppedChatStream(chunks)
        app = _app(_ChatTransport([stream]))

        async def wait_for_rendered(expected: str) -> None:
            for _ in range(30):
                await pilot.pause()
                state = app.dashboard_state
                markdown = tuple(history.query(Markdown))[-1]
                response_rendered = state is not None and (
                    state.current_response == expected
                    or (
                        bool(state.chat_session.messages)
                        and state.chat_session.messages[-1].content == expected
                    )
                )
                if response_rendered and markdown.source == expected:
                    return
            raise AssertionError("streamed response was not rendered")

        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await pilot.press("down", "enter")
            await pilot.press("f", "o", "l", "l", "o", "w", "enter")
            history = app.query_one(ChatHistory)

            await asyncio.wait_for(stream.sent[0].wait(), timeout=3)
            await wait_for_rendered(chunks[0])
            assert history.is_vertical_scroll_end

            stream.advance[0].set()
            await asyncio.wait_for(stream.sent[1].wait(), timeout=3)
            await wait_for_rendered("".join(chunks[:2]))
            assert history.is_vertical_scroll_end

            history.scroll_home(animate=False)
            await pilot.pause()
            stream.advance[1].set()
            await asyncio.wait_for(stream.sent[2].wait(), timeout=3)
            await wait_for_rendered("".join(chunks[:3]))
            assert history.scroll_offset.y == 0
            assert not history.is_vertical_scroll_end

            history.scroll_end(animate=False)
            await pilot.pause()
            stream.advance[2].set()
            await asyncio.wait_for(stream.sent[3].wait(), timeout=3)
            await wait_for_rendered("".join(chunks))
            assert history.is_vertical_scroll_end

    asyncio.run(scenario())


def test_cancel_clear_settings_multiline_and_focused_keys() -> None:
    async def scenario() -> None:
        first = _GatedChatStream("partial", " ignored")
        second = _GatedChatStream("again", " done")
        transport = _ChatTransport([first, second])
        app = _app(transport)
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await pilot.press("down", "enter")
            await pilot.pause()
            editor = app.query_one("#chat-prompt", TextArea)
            await pilot.press("q", "r", "?")
            assert editor.text == "qr?"
            editor.clear()
            await pilot.press("a", "shift+enter", "b")
            assert editor.text == "a\nb"
            await pilot.press("enter")
            await asyncio.wait_for(first.first_sent.wait(), timeout=3)
            await pilot.press("escape")
            for _ in range(30):
                await pilot.pause()
                latest = app.dashboard_state
                if latest is not None and latest.active_generation_id is None:
                    break
            cancelled_state = app.dashboard_state
            assert cancelled_state is not None
            assert first.closed
            assert cancelled_state.generation_status is GenerationStatus.CANCELLED
            assert cancelled_state.chat_session.messages[-1].content == "partial"
            assert isinstance(app.focused, TextArea)

            await pilot.press("n", "e", "x", "t", "enter")
            await asyncio.wait_for(second.first_sent.wait(), timeout=3)
            second.release.set()
            for _ in range(30):
                await pilot.pause()
                latest = app.dashboard_state
                if latest is not None and latest.active_generation_id is None:
                    break
            await pilot.press("ctrl+k")
            await pilot.pause()
            cleared_state = app.dashboard_state
            assert cleared_state is not None
            assert cleared_state.chat_session.messages == ()

            await pilot.press("ctrl+g")
            panel = app.query_one(GenerationSettingsPanel)
            assert panel.is_open
            app.query_one("#temperature", Input).value = "3"
            panel.apply_settings()
            await pilot.pause()
            unchanged_state = app.dashboard_state
            assert unchanged_state is not None
            assert unchanged_state.chat_session.settings.temperature == 0.7
            app.query_one("#temperature", Input).value = "1.2"
            app.query_one("#seed", Input).value = "42"
            panel.apply_settings()
            await pilot.pause()
            updated_state = app.dashboard_state
            assert updated_state is not None
            assert updated_state.chat_session.settings.temperature == 1.2
            assert updated_state.chat_session.settings.seed == 42
        assert transport.close_count == 1

    asyncio.run(scenario())
