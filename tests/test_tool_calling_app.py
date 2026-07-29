"""Behavioral Tool Calling workspace and lifecycle tests."""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import httpx
from rich.text import Text
from textual.pilot import Pilot
from textual.widgets import ContentSwitcher, OptionList, Static

from modeltop.api.client import OpenAICompatibleClient
from modeltop.app import FullToolCallingConfirmation, ModelTopApp
from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkStatus,
)
from modeltop.benchmarks.tool_calling import (
    ScenarioStartCallback,
    UpstreamBenchmarkRunner,
    suite_registry,
)
from modeltop.models import ModelTopConfig
from modeltop.screens.tool_calling import ToolCallingView
from modeltop.state import ServerStatus
from modeltop.widgets.footer import StatusFooter
from tests.test_tool_calling_benchmark import successful_upstream_runner


class _ModelsTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests = 0
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        del request
        self.requests += 1
        return httpx.Response(200, json={"data": [{"id": "model"}]})

    async def aclose(self) -> None:
        self.close_count += 1


def _app_config() -> ModelTopConfig:
    return ModelTopConfig.model_validate(
        {
            "application": {
                "refresh_interval_seconds": 3600,
                "request_timeout_seconds": 5,
                "default_server": "server",
            },
            "hardware": {"enabled": False},
            "servers": [
                {
                    "id": "server",
                    "name": "Test Server",
                    "base_url": "http://server/prefix/v1",
                    "backend_hint": "vllm",
                }
            ],
        }
    )


def _app(
    transport: _ModelsTransport,
    runner: UpstreamBenchmarkRunner,
) -> ModelTopApp:
    return ModelTopApp(
        _app_config(),
        client=OpenAICompatibleClient(
            "http://server/prefix/v1",
            None,
            5,
            transport=transport,
        ),
        tool_calling_runner=runner,
    )


def _plain(widget: Static) -> str:
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


async def _wait_for_server_online(
    app: ModelTopApp,
    pilot: Pilot[None],
) -> None:
    for _ in range(30):
        state = app.dashboard_state
        if state is not None and state.server_status is ServerStatus.ONLINE:
            return
        await pilot.pause()
    raise AssertionError("dashboard did not reach online")


async def _wait_for_tool_status(
    app: ModelTopApp,
    status: ToolCallingBenchmarkStatus,
) -> None:
    for _ in range(80):
        state = app.dashboard_state
        if state is not None and state.tool_calling_benchmark.status is status:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Tool Calling did not reach {status}")


def test_full_confirmation_then_core_run_rerun_and_edit() -> None:
    async def scenario() -> None:
        transport = _ModelsTransport()
        app = _app(transport, successful_upstream_runner())
        async with app.run_test(size=(120, 38)) as pilot:
            await _wait_for_server_online(app, pilot)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 5
            menu.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.dashboard_state is not None
            assert app.dashboard_state.active_view == "tool-calling"
            panel = app.query_one(ToolCallingView).config_panel
            assert panel.parse_config() == ToolCallingBenchmarkConfig()

            app.action_run_or_rerun()
            await pilot.pause()
            assert isinstance(app.screen, FullToolCallingConfirmation)
            assert not app.dashboard_state.tool_calling_benchmark.is_active
            await pilot.press("escape")

            panel.load_config(ToolCallingBenchmarkConfig(suite="core"))
            app.action_run_or_rerun()
            await _wait_for_tool_status(app, ToolCallingBenchmarkStatus.COMPLETED)
            await pilot.pause()
            result = app.dashboard_state.tool_calling_benchmark.latest_result
            assert result is not None
            assert result.attempted_count == len(result.scenarios) == 15
            assert result.final_score == 100
            switcher = app.query_one("#tool-calling-view-switcher", ContentSwitcher)
            assert switcher.current == "tool-calling-result-panel"
            assert "TOOL CALLING 100%" in _plain(
                app.query_one("#header-subtitle", Static)
            )
            assert "100% score" in _plain(app.query_one(StatusFooter))

            app.action_export_result()
            await pilot.pause()
            assert switcher.current == "tool-calling-config-panel"
            edited = panel.parse_config()
            assert edited is not None and edited.suite == "core"

            app.action_run_or_rerun()
            await _wait_for_tool_status(app, ToolCallingBenchmarkStatus.COMPLETED)
            assert (
                app.dashboard_state.tool_calling_benchmark.latest_result is not result
            )

        assert transport.close_count == 1

    asyncio.run(scenario())


def test_active_tool_calling_wins_status_and_escape_cleans_up() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def gated(**kwargs: Any) -> dict[str, Any]:
            identity = suite_registry("core")[0]
            definition = SimpleNamespace(
                id=identity[0],
                category=SimpleNamespace(value=identity[1]),
                title=identity[2],
            )
            callback = cast(ScenarioStartCallback, kwargs["on_scenario_start"])
            await callback(definition, 0, 15)
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("gate unexpectedly released")
            finally:
                cleaned.set()

        transport = _ModelsTransport()
        app = _app(transport, cast(UpstreamBenchmarkRunner, gated))
        async with app.run_test(size=(120, 38)) as pilot:
            await _wait_for_server_online(app, pilot)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 5
            menu.focus()
            await pilot.press("enter")
            app.query_one(ToolCallingView).config_panel.load_config(
                ToolCallingBenchmarkConfig(suite="core")
            )
            app.action_run_or_rerun()
            await started.wait()
            await pilot.pause()

            menu.highlighted = 1
            menu.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.dashboard_state is not None
            assert app.dashboard_state.active_view == "chat"
            assert "TOOL CALLING" in _plain(app.query_one("#header-subtitle", Static))
            assert "0/15" in _plain(app.query_one(StatusFooter))

            await pilot.press("escape")
            await _wait_for_tool_status(app, ToolCallingBenchmarkStatus.CANCELLED)
            await pilot.pause()
            assert cleaned.is_set()
            lane = app.dashboard_state.tool_calling_benchmark
            assert lane.active_benchmark_id is None
            assert lane.latest_result is not None
            assert lane.latest_result.scenarios == ()
            assert not app.dashboard_state.benchmark_is_active

        assert transport.close_count == 1

    asyncio.run(scenario())
