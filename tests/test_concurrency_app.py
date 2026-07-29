"""Textual Pilot coverage for the Concurrency benchmark workspace."""
# pyright: reportPrivateUsage=false

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
from textual.widgets import Button, Input, OptionList, Static

from modeltop.api.client import OpenAICompatibleClient
from modeltop.app import ModelTopApp
from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkConfig,
    ConcurrencyBenchmarkStatus,
)
from modeltop.screens.concurrency import ConcurrencyView
from modeltop.state import ServerStatus
from modeltop.widgets.benchmark_configuration import BenchmarkConfigurationPanel
from modeltop.widgets.request_table import RequestTable
from tests.test_app import (
    _config,
    _plain_render,
    _ScriptedHardwareProvider,
    _wait_for_status,
)


class _GateStream(httpx.AsyncByteStream):
    def __init__(self, transport: "_ConcurrencyTransport", number: int) -> None:
        self.transport = transport
        self.number = number
        self.closed = False
        self._active = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self._active = True
        self.transport.active += 1
        self.transport.peak_active = max(
            self.transport.peak_active, self.transport.active
        )
        if self.transport.peak_active >= self.transport.expected_peak:
            self.transport.reached_peak.set()
        try:
            payload = {
                "choices": [{"delta": {"content": "first"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
            await self.transport.release.wait()
            yield (
                b'data: {"choices":[{"delta":{"content":" done"},'
                b'"finish_reason":null}]}\n\n'
            )
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield (
                b'data: {"choices":[],"usage":{"prompt_tokens":5,'
                b'"completion_tokens":2,"total_tokens":7}}\n\n'
            )
            yield b"data: [DONE]\n\n"
        finally:
            if self._active:
                self._active = False
                self.transport.active -= 1

    async def aclose(self) -> None:
        self.closed = True
        if self._active:
            self._active = False
            self.transport.active -= 1


class _ConcurrencyTransport(httpx.AsyncBaseTransport):
    def __init__(self, expected_peak: int = 2) -> None:
        self.expected_peak = expected_peak
        self.active = 0
        self.peak_active = 0
        self.total_chat_requests = 0
        self.model_requests = 0
        self.reached_peak = asyncio.Event()
        self.release = asyncio.Event()
        self.streams: list[_GateStream] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.model_requests += 1
            return httpx.Response(200, json={"data": [{"id": "model"}]})
        self.total_chat_requests += 1
        stream = _GateStream(self, self.total_chat_requests)
        self.streams.append(stream)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    async def aclose(self) -> None:
        return None


def _app(transport: _ConcurrencyTransport) -> ModelTopApp:
    client = OpenAICompatibleClient(
        "http://server/prefix/v1", None, 5, transport=transport
    )
    return ModelTopApp(
        _config(refresh_interval=3600),
        client=client,
        hardware_provider=_ScriptedHardwareProvider(),
    )


def test_navigation_controls_invalid_input_and_narrow_resize() -> None:
    async def scenario() -> None:
        transport = _ConcurrencyTransport()
        app = _app(transport)
        async with app.run_test(size=(60, 20), notifications=True) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 3
            await pilot.press("enter")
            await pilot.pause()
            assert app.dashboard_state is not None
            assert app.dashboard_state.active_view == "concurrency"
            view = app.query_one("#concurrency-workspace", ConcurrencyView)
            assert len(view.query(BenchmarkConfigurationPanel)) == 1
            for identifier in (
                "concurrency-fixed",
                "concurrency-levels",
                "concurrency-requests",
                "concurrency-warmups",
                "concurrency-max-tokens",
                "concurrency-temperature",
                "concurrency-top-p",
                "concurrency-seed",
                "concurrency-timeout",
                "concurrency-delay",
                "concurrency-run",
            ):
                assert len(view.query(f"#{identifier}")) == 1
            view.config_panel.load_config(
                ConcurrencyBenchmarkConfig(
                    mode="sweep",
                    concurrency_levels=(1, 2),
                    requests_per_level=2,
                    warmup_requests=0,
                )
            )
            view.query_one("#concurrency-levels", Input).value = "1, 1"
            view.query_one("#concurrency-run", Button).press()
            await pilot.pause()
            assert transport.total_chat_requests == 0
            await pilot.resize_terminal(80, 24)
            await pilot.resize_terminal(60, 20)

    asyncio.run(scenario())


def test_gated_fixed_run_isolates_chat_cancels_and_recovers_polling() -> None:
    async def scenario() -> None:
        transport = _ConcurrencyTransport(expected_peak=2)
        app = _app(transport)
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            initial_models = transport.model_requests
            app._set_active_view("concurrency")
            view = app.query_one(ConcurrencyView)
            config = ConcurrencyBenchmarkConfig(
                mode="fixed",
                concurrency_levels=(2,),
                requests_per_level=6,
                warmup_requests=0,
                request_timeout_seconds=10.0,
                delay_between_levels_seconds=0.0,
            )
            view.show_config(config)
            view.query_one("#concurrency-run", Button).press()
            await asyncio.wait_for(transport.reached_peak.wait(), timeout=3)
            await pilot.pause()
            state = app.dashboard_state
            assert state is not None
            assert (
                state.concurrency_benchmark.status is ConcurrencyBenchmarkStatus.RUNNING
            )
            assert transport.peak_active == 2
            assert transport.peak_active <= config.concurrency_levels[0]
            assert transport.model_requests == initial_models
            progress_table = view.query_one("#concurrency-request-table", RequestTable)
            assert progress_table.row_count == 6

            app._set_active_view("chat")
            await pilot.pause()
            chat_text = _plain_render(app.query_one("#chat-status", Static))
            assert "Chat is unavailable while a benchmark is running" in chat_text
            assert app.query_one("#chat-prompt").disabled
            await pilot.press("escape")
            for _ in range(200):
                await asyncio.sleep(0.005)
                latest = app.dashboard_state
                if (
                    latest is not None
                    and latest.concurrency_benchmark.status
                    is ConcurrencyBenchmarkStatus.CANCELLED
                ):
                    break
            terminal = app.dashboard_state
            assert terminal is not None
            assert (
                terminal.concurrency_benchmark.status
                is ConcurrencyBenchmarkStatus.CANCELLED
            )
            assert terminal.concurrency_benchmark.latest_result is not None
            assert terminal.concurrency_benchmark.latest_result.cancelled
            assert transport.active == 0
            assert all(stream.closed for stream in transport.streams)
            for _ in range(100):
                if transport.model_requests > initial_models:
                    break
                await asyncio.sleep(0.005)
            assert transport.model_requests > initial_models
            assert not app.query_one("#chat-prompt").disabled

    asyncio.run(scenario())
