"""Behavioral tests for the live ModelTop dashboard."""

import asyncio
import errno
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path

import httpx
import pytest
from rich.console import Console
from rich.text import Text
from textual import constants as textual_constants
from textual.containers import Horizontal
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import DataTable, OptionList, Static
from textual.widgets._toast import Toast

import modeltop
from modeltop.api.client import OpenAICompatibleClient
from modeltop.app import ConfigurationErrorScreen, ModelTopApp, ShortcutHelp
from modeltop.constants import APP_NAME, PACKAGE_VERSION
from modeltop.hardware.base import (
    HardwareProvider,
    HardwareProviderUnavailable,
)
from modeltop.hardware.models import (
    CpuMetrics,
    GpuMetrics,
    HardwareSnapshot,
    MemoryMetrics,
)
from modeltop.models import ModelTopConfig
from modeltop.state import HardwareStatus, ServerStatus
from modeltop.widgets.footer import StatusFooter
from modeltop.widgets.header import HeaderBar
from modeltop.widgets.sidebar import BenchmarkSidebar
from modeltop.widgets.workspace import Workspace


def _plain_render(widget: Widget) -> str:
    if isinstance(widget, DataTable):
        return "\n".join(
            widget.render_line(line).text for line in range(widget.size.height)
        )
    rendered = widget.render()
    if isinstance(rendered, Text):
        return rendered.plain
    output = StringIO()
    console = Console(file=output, width=widget.size.width or 120, color_system=None)
    console.print(rendered)
    return output.getvalue()


def _plain_prompt(prompt: object) -> str:
    return prompt.plain if isinstance(prompt, Text) else str(prompt)


def _config(
    *,
    refresh_interval: float = 3600,
    hardware_enabled: bool = True,
    backend_hint: str | None = "vllm",
) -> ModelTopConfig:
    return ModelTopConfig.model_validate(
        {
            "application": {
                "refresh_interval_seconds": refresh_interval,
                "request_timeout_seconds": 5,
                "default_server": "server",
            },
            "hardware": {
                "enabled": hardware_enabled,
                "refresh_interval_seconds": 3600,
                "preferred_provider": "auto",
            },
            "servers": [
                {
                    "id": "server",
                    "name": "Test Server",
                    "base_url": "http://server/prefix/v1",
                    "backend_hint": backend_hint,
                }
            ],
        }
    )


def _hardware_snapshot(
    names: tuple[str, ...] = ("NVIDIA Fixture",),
    *,
    error: str | None = None,
    second: int = 0,
) -> HardwareSnapshot:
    return HardwareSnapshot(
        provider_name="fixture",
        gpus=tuple(
            GpuMetrics(
                index,
                name,
                f"gpu-{index}",
                40.0 + index * 20,
                (index + 1) * 1024**3,
                8 * 1024**3,
                50.0 + index * 10,
                100.0 + index * 25,
                200.0,
                120.0,
            )
            for index, name in enumerate(names)
        ),
        cpu=CpuMetrics(25.0, 8, 4, 1.0, 2.0, 3.0),
        memory=MemoryMetrics(4 * 1024**3, 32 * 1024**3, 12.5),
        collected_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=second),
        error=error,
    )


class _ScriptedHardwareProvider(HardwareProvider):
    name = "fixture"

    def __init__(
        self,
        outcomes: Sequence[HardwareSnapshot | BaseException] | None = None,
        *,
        gate_call: int | None = None,
    ) -> None:
        self.outcomes = list(outcomes or [_hardware_snapshot()])
        self.gate_call = gate_call
        self.collect_count = 0
        self.close_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def collect(self) -> HardwareSnapshot:
        self.collect_count += 1
        if self.collect_count == self.gate_call:
            self.started.set()
            await self.release.wait()
        outcome = self.outcomes[min(self.collect_count - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.close_count += 1


class _ScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, outcomes: Sequence[httpx.Response | str]) -> None:
        self.outcomes = list(outcomes)
        self.requests = 0
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        outcome = self.outcomes.pop(0)
        if outcome == "refused":
            try:
                raise OSError(errno.ECONNREFUSED, "refused")
            except OSError as cause:
                raise httpx.ConnectError("failed", request=request) from cause
        assert isinstance(outcome, httpx.Response)
        return outcome

    async def aclose(self) -> None:
        self.close_count += 1


class _GatedTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, gate_first: bool) -> None:
        self.gate_first = gate_first
        self.requests = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        should_gate = self.requests == 1 if self.gate_first else self.requests == 2
        if should_gate:
            self.started.set()
            await self.release.wait()
        return httpx.Response(
            200,
            json={"data": [{"id": "org/beta"}, {"id": "org/alpha"}]},
        )

    async def aclose(self) -> None:
        self.close_count += 1


def _app_with_transport(
    transport: httpx.AsyncBaseTransport,
    *,
    refresh_interval: float = 3600,
    hardware_provider: HardwareProvider | None = None,
    hardware_enabled: bool = True,
    backend_hint: str | None = "vllm",
) -> ModelTopApp:
    client = OpenAICompatibleClient(
        "http://server/prefix/v1",
        None,
        5,
        transport=transport,
    )
    provider = hardware_provider or _ScriptedHardwareProvider()
    return ModelTopApp(
        _config(
            refresh_interval=refresh_interval,
            hardware_enabled=hardware_enabled,
            backend_hint=backend_hint,
        ),
        client=client,
        hardware_provider=provider,
    )


async def _wait_for_status(
    app: ModelTopApp,
    pilot: Pilot[None],
    status: ServerStatus,
) -> None:
    for _ in range(30):
        state = app.dashboard_state
        if state is not None and state.server_status is status:
            return
        await pilot.pause()
    raise AssertionError(f"dashboard did not reach {status}")


async def _wait_for_hardware(app: ModelTopApp, pilot: Pilot[None]) -> None:
    for _ in range(30):
        state = app.dashboard_state
        if state is not None and state.hardware_snapshot is not None:
            return
        await pilot.pause()
    raise AssertionError("dashboard did not collect hardware")


def test_package_metadata_and_application_class() -> None:
    """Package metadata and the Textual entry class remain public."""
    assert modeltop.APP_NAME == "ModelTop"
    assert modeltop.__version__ == "0.1.0"
    assert ModelTopApp.__name__ == "ModelTopApp"
    assert APP_NAME == "ModelTop"
    assert PACKAGE_VERSION == "0.1.0"


def test_dashboard_uses_catppuccin_mocha_visual_theme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered dashboard surfaces resolve the Catppuccin Mocha palette."""

    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(textual_constants, "COLOR_SYSTEM", "truecolor")

    async def scenario() -> None:
        transport = _ScriptedTransport(
            [
                httpx.Response(
                    200,
                    json={"data": [{"id": "org/beta"}, {"id": "org/alpha"}]},
                )
            ]
        )
        app = _app_with_transport(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.focus()
            await pilot.press("down", "enter")
            await pilot.pause()
            svg = app.export_screenshot(simplify=True).lower()
            rendered_colors = set(re.findall(r"#[0-9a-f]{6}", svg))
            assert app.theme == "catppuccin-mocha"
            for color in (
                "#181825",
                "#1e1e2e",
                "#313244",
                "#585b70",
                "#f5c2e7",
                "#abe9b3",
            ):
                assert color in rendered_colors

    asyncio.run(scenario())


def test_initial_online_rendering_selector_sidebar_and_selection() -> None:
    """Initial discovery renders live regions and enables two-model selection."""

    async def scenario() -> None:
        transport = _ScriptedTransport(
            [
                httpx.Response(
                    200,
                    json={"data": [{"id": "org/beta"}, {"id": "org/alpha"}]},
                )
            ]
        )
        app = _app_with_transport(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await _wait_for_hardware(app, pilot)
            assert len(app.query(HeaderBar)) == 1
            assert len(app.query(BenchmarkSidebar)) == 1
            assert len(app.query(Workspace)) == 1
            assert len(app.query(StatusFooter)) == 1

            menu = app.query_one("#sidebar-menu", OptionList)
            assert [
                _plain_prompt(menu.get_option_at_index(index).prompt)
                for index in range(menu.option_count)
            ] == [
                "Overview",
                "Chat",
                "Speed Test",
                "Concurrency",
                "Context Length",
                "Tool Calling",
                "Drafter",
                "Results",
                "Settings",
            ]
            assert not menu.get_option_at_index(3).disabled
            assert not menu.get_option_at_index(4).disabled
            assert not menu.get_option_at_index(5).disabled
            assert not menu.get_option_at_index(6).disabled
            assert app.focused is menu
            screen_stack = tuple(app.screen_stack)
            await pilot.press("enter")
            assert tuple(app.screen_stack) == screen_stack

            selector = app.query_one("#model-selector", OptionList)
            assert selector.display
            assert [
                _plain_prompt(selector.get_option_at_index(index).prompt)
                for index in range(selector.option_count)
            ] == ["org/alpha", "org/beta"]
            assert selector.highlighted == 0

            rendered = "\n".join(_plain_render(widget) for widget in app.query(Static))
            for expected in (
                "MODELTOP",
                "Test Server",
                "org/alpha",
                "server/prefix",
                "vLLM",
                "ONLINE",
                "LOCAL HARDWARE",
                "NVIDIA Fixture",
                "1.0/8.0 GB",
                "40%",
                "50°C",
                "100/200 W",
                "FAN 120%",
                "CPU 25%",
                "RAM 4.0/32.0 GB",
                "PROVIDER fixture",
                "DISCOVERED MODELS",
                "2 models",
                "R Refresh",
                "Q Quit",
            ):
                assert expected in rendered
            assert rendered.count("GPU\nNVIDIA Fixture") == 1
            assert rendered.count("VRAM\n1.0/8.0 GB") == 1
            assert "CONTEXT\n--" not in rendered
            assert "QUANT\n--" not in rendered

            await pilot.press("tab")
            assert app.focused is selector
            await pilot.press("down", "enter")
            assert app.dashboard_state is not None
            assert app.dashboard_state.selected_model_id == "org/beta"
            assert "org/beta" in _plain_render(app.query_one("#metric-model", Static))
            assert "beta" in _plain_render(app.query_one(StatusFooter))
            await pilot.press("tab")
            assert app.focused is menu
            await pilot.press("shift+tab")
            assert app.focused is selector
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_header_falls_back_to_selected_model_owner_for_backend() -> None:
    """The models response identifies the backend when config has no hint."""

    async def scenario() -> None:
        transport = _ScriptedTransport(
            [
                httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "model",
                                "owned_by": "vllm",
                            }
                        ]
                    },
                )
            ]
        )
        app = _app_with_transport(transport, backend_hint=None)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert _plain_render(app.query_one("#metric-backend", Static)) == (
                "BACKEND\nvLLM"
            )
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_one_and_no_model_states_collapse_focus() -> None:
    """One or zero models hide the selector and return focus to the sidebar."""

    async def scenario() -> None:
        transport = _ScriptedTransport(
            [
                httpx.Response(200, json={"data": [{"id": "one"}, {"id": "two"}]}),
                httpx.Response(200, json={"data": [{"id": "one"}]}),
                httpx.Response(200, json={"data": []}),
            ]
        )
        app = _app_with_transport(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            selector = app.query_one("#model-selector", OptionList)
            menu = app.query_one("#sidebar-menu", OptionList)
            await pilot.press("tab")
            assert app.focused is selector

            await pilot.press("ctrl+l")
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert not selector.display
            assert app.focused is menu
            assert _plain_render(app.query_one("#models-state", Static)) == "one"
            await pilot.press("tab", "shift+tab")
            assert app.focused is menu

            await pilot.press("ctrl+l")
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert not selector.display
            assert _plain_render(app.query_one("#models-state", Static)) == (
                "No models discovered"
            )
            assert "No models" in _plain_render(app.query_one(StatusFooter))

    asyncio.run(scenario())


def test_manual_refresh_stays_online_until_completion() -> None:
    """Manual refresh keeps confirmed online state until discovery completes."""

    async def scenario() -> None:
        transport = _GatedTransport(gate_first=False)
        app = _app_with_transport(transport)
        async with app.run_test(
            size=(100, 30),
            notifications=True,
        ) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert app.dashboard_state is not None
            selected_model_id = app.dashboard_state.selected_model_id
            selector = app.query_one("#model-selector", OptionList)
            assert selected_model_id == "org/alpha"
            assert selector.display
            assert selector.highlighted == 0

            await pilot.press("ctrl+l")
            await transport.started.wait()
            await pilot.pause()
            assert app.dashboard_state.server_status is ServerStatus.ONLINE
            assert app.dashboard_state.is_refreshing
            assert app.dashboard_state.selected_model_id == selected_model_id
            assert selector.display
            assert selector.highlighted == 0
            assert _plain_render(app.query_one("#metric-status", Static)) == (
                "STATUS\nONLINE"
            )
            assert selected_model_id in _plain_render(
                app.query_one("#metric-model", Static)
            )
            footer = _plain_render(app.query_one(StatusFooter))
            assert "ONLINE" in footer
            assert "Refreshing..." in footer
            assert "OFFLINE" not in footer
            assert "CONNECTING" not in footer
            chat_status = _plain_render(app.query_one("#chat-status", Static))
            assert "READY · Enter sends" in chat_status
            assert "OFFLINE" not in chat_status

            transport.release.set()
            for _ in range(30):
                await pilot.pause()
                if not app.dashboard_state.is_refreshing:
                    break
            else:
                raise AssertionError("manual refresh did not complete")

            footer = _plain_render(app.query_one(StatusFooter))
            assert "R Refresh" in footer
            assert "Refreshing..." not in footer
            toast = app.query_one(Toast)
            assert _plain_render(toast).startswith("Refresh\nDiscovered 2 models in ")
            assert _plain_render(toast).endswith(" ms.")
            assert toast.has_class("-information")
            assert transport.requests == 2

    asyncio.run(scenario())


def test_overlapping_manual_refresh_issues_one_request() -> None:
    """An overlapping keypress is rejected without a second HTTP request."""

    async def scenario() -> None:
        transport = _GatedTransport(gate_first=True)
        app = _app_with_transport(transport)
        async with app.run_test(
            size=(100, 30),
            notifications=True,
        ) as pilot:
            await transport.started.wait()
            assert app.dashboard_state is not None
            assert app.dashboard_state.server_status is ServerStatus.CONNECTING
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert transport.requests == 1
            toast = app.query_one(Toast)
            assert _plain_render(toast) == "Refresh\nRefresh already in progress."
            transport.release.set()
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert transport.requests == 1

    asyncio.run(scenario())


def test_automatic_refresh_is_silent() -> None:
    """The interval refreshes live state without creating a notification."""

    async def scenario() -> None:
        outcomes = [
            httpx.Response(200, json={"data": [{"id": "model"}]}) for _ in range(20)
        ]
        transport = _ScriptedTransport(outcomes)
        app = _app_with_transport(transport, refresh_interval=0.02)
        async with app.run_test(
            size=(100, 30),
            notifications=True,
        ) as pilot:
            for _ in range(50):
                if transport.requests >= 2:
                    break
                await pilot.pause()
                await asyncio.sleep(0.01)
            assert transport.requests >= 2
            assert len(app.query(Toast)) == 0

    asyncio.run(scenario())


def test_offline_error_toast_and_recovery_retain_selection() -> None:
    """Online to offline to online clears latency and restores user selection."""

    async def scenario() -> None:
        models = {"data": [{"id": "org/alpha"}, {"id": "org/beta"}]}
        transport = _ScriptedTransport(
            [
                httpx.Response(200, json=models),
                "refused",
                httpx.Response(200, json=models),
            ]
        )
        app = _app_with_transport(transport)
        async with app.run_test(
            size=(100, 30),
            notifications=True,
        ) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await pilot.press("tab", "down", "enter")
            assert app.dashboard_state is not None
            assert app.dashboard_state.selected_model_id == "org/beta"

            await pilot.press("ctrl+l")
            await _wait_for_status(app, pilot, ServerStatus.OFFLINE)
            assert app.dashboard_state.connection_latency_ms is None
            assert app.dashboard_state.selected_model_id == "org/beta"
            assert not app.query_one("#model-selector", OptionList).display
            assert _plain_render(app.query_one("#models-state", Static)) == (
                "Models unavailable"
            )
            toast = app.query_one(Toast)
            assert _plain_render(toast) == "Refresh\nConnection refused"
            assert toast.has_class("-error")

            await pilot.press("ctrl+l")
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert app.dashboard_state.selected_model_id == "org/beta"
            assert app.dashboard_state.connection_latency_ms is not None

    asyncio.run(scenario())


def test_help_quit_and_layout_geometry() -> None:
    """Help, quit bindings, and 100x30 to 80x24 geometry remain stable."""

    async def scenario() -> None:
        transport = _ScriptedTransport(
            [httpx.Response(200, json={"data": [{"id": "model"}]})]
        )
        app = _app_with_transport(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)

            def assert_layout() -> None:
                header = app.query_one(HeaderBar)
                body = app.query_one("#dashboard-body", Horizontal)
                sidebar = app.query_one(BenchmarkSidebar)
                workspace = app.query_one(Workspace)
                footer = app.query_one(StatusFooter)
                screen = app.screen
                assert sidebar.region.width == 24
                assert header.region.width == screen.region.width
                assert footer.region.width == screen.region.width
                assert workspace.region.x == sidebar.region.right
                assert footer.region.bottom == screen.region.bottom
                assert body.region.y == header.region.bottom
                assert body.region.bottom == footer.region.y
                assert workspace.region.width > 0

            assert_layout()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert_layout()

            await pilot.press("?")
            assert isinstance(app.screen, ShortcutHelp)
            assert _plain_render(app.screen.query_one("#shortcut-help", Static)) == (
                "KEYBOARD SHORTCUTS\n\n"
                "↑ / ↓  Move selection · Tab cycle focus\n"
                "Enter  Select / start / send · Shift/Alt+Enter newline\n"
                "Esc  Cancel active work / back\n"
                "R  Run or Run Again in benchmark workspaces\n"
                "E  Edit Tool/Context/Concurrency · export Speed result\n"
                "C  Copy Speed result summary\n"
                "Ctrl+K  Clear chat · Ctrl+G chat settings\n"
                "?  Toggle help\n"
                "Q  Quit outside inputs · Ctrl+Q always quit\n"
            )
            await pilot.press("escape", "?", "q")
        assert app.return_code == 0
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_invalid_configuration_screen_has_no_dashboard_or_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing authoritative config opens guidance and creates no dashboard."""

    async def scenario() -> None:
        missing = tmp_path / "missing.yaml"
        monkeypatch.setenv("MODELTOP_CONFIG", str(missing))
        app = ModelTopApp()
        async with app.run_test(size=(80, 24)) as pilot:
            assert isinstance(app.screen, ConfigurationErrorScreen)
            content = _plain_render(
                app.screen.query_one("#configuration-error", Static)
            )
            assert content == (
                "CONFIGURATION ERROR\n\n"
                f"Could not load configuration from {missing}: file does not exist\n\n"
                "Set MODELTOP_CONFIG or fix the selected YAML file.\n\n"
                "Q Quit"
            )
            assert len(app.query(HeaderBar)) == 0
            assert len(app.query(BenchmarkSidebar)) == 0
            assert len(app.query(Workspace)) == 0
            assert len(app.query(StatusFooter)) == 0
            assert app.dashboard_state is None
            await pilot.press("q")
        assert app.return_code == 0

    asyncio.run(scenario())


def test_hardware_completes_while_server_remains_connecting() -> None:
    async def scenario() -> None:
        transport = _GatedTransport(gate_first=True)
        hardware = _ScriptedHardwareProvider()
        app = _app_with_transport(transport, hardware_provider=hardware)
        async with app.run_test(size=(100, 30)) as pilot:
            await transport.started.wait()
            await _wait_for_hardware(app, pilot)
            assert app.dashboard_state is not None
            assert app.dashboard_state.server_status is ServerStatus.CONNECTING
            assert app.dashboard_state.hardware_status is HardwareStatus.AVAILABLE
            assert "NVIDIA Fixture" in _plain_render(
                app.query_one("#metric-gpu", Static)
            )
            transport.release.set()
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
        assert hardware.close_count == 1
        assert transport.close_count == 1

    asyncio.run(scenario())


def test_unavailable_and_degraded_hardware_render_readably() -> None:
    async def scenario() -> None:
        unavailable_transport = _ScriptedTransport(
            [httpx.Response(200, json={"data": []})]
        )
        unavailable_provider = _ScriptedHardwareProvider(
            [HardwareProviderUnavailable("No NVIDIA GPU detected", "private")]
        )
        unavailable_app = _app_with_transport(
            unavailable_transport,
            hardware_provider=unavailable_provider,
        )
        async with unavailable_app.run_test(size=(100, 30)) as pilot:
            for _ in range(30):
                state = unavailable_app.dashboard_state
                if (
                    state is not None
                    and state.hardware_status is HardwareStatus.UNAVAILABLE
                    and not state.hardware_is_refreshing
                ):
                    break
                await pilot.pause()
            rendered = "\n".join(
                _plain_render(widget) for widget in unavailable_app.query(Static)
            )
            assert "Unavailable" in rendered
            assert "No NVIDIA GPU detected" in rendered
            assert "private" not in rendered

        degraded_transport = _ScriptedTransport(
            [httpx.Response(200, json={"data": []})]
        )
        degraded_provider = _ScriptedHardwareProvider(
            [_hardware_snapshot(error="Partial GPU metrics available")]
        )
        degraded_app = _app_with_transport(
            degraded_transport,
            hardware_provider=degraded_provider,
        )
        async with degraded_app.run_test(size=(100, 30)) as pilot:
            await _wait_for_hardware(degraded_app, pilot)
            assert degraded_app.dashboard_state is not None
            assert (
                degraded_app.dashboard_state.hardware_status is HardwareStatus.DEGRADED
            )
            rendered = "\n".join(
                _plain_render(widget) for widget in degraded_app.query(Static)
            )
            assert "NVIDIA Fixture" in rendered
            assert "DEGRADED" in rendered
            assert "Partial GPU metrics available" in rendered

    asyncio.run(scenario())


def test_identical_and_mixed_multi_gpu_rendering_and_scroll() -> None:
    async def scenario() -> None:
        names = tuple(f"GPU Model {index}" for index in range(10))
        hardware = _ScriptedHardwareProvider(
            [
                _hardware_snapshot(("Same GPU", "Same GPU")),
                _hardware_snapshot(("GPU A", "GPU B"), second=1),
                _hardware_snapshot(names, second=2),
            ]
        )
        transport = _ScriptedTransport(
            [
                httpx.Response(200, json={"data": [{"id": "model"}]}),
                httpx.Response(200, json={"data": [{"id": "model"}]}),
                httpx.Response(200, json={"data": [{"id": "model"}]}),
            ]
        )
        app = _app_with_transport(transport, hardware_provider=hardware)
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_hardware(app, pilot)
            assert "2 × Same GPU" in _plain_render(  # noqa: RUF001
                app.query_one("#metric-gpu", Static)
            )
            gpu_rows = _plain_render(app.query_one("#hardware-gpus", Static))
            assert len(gpu_rows.splitlines()) == 2

            await pilot.press("ctrl+l")
            for _ in range(30):
                state = app.dashboard_state
                if (
                    hardware.collect_count == 2
                    and state is not None
                    and not state.hardware_is_refreshing
                ):
                    break
                await pilot.pause()
            assert "2 GPUs" in _plain_render(app.query_one("#metric-gpu", Static))
            gpu_rows = _plain_render(app.query_one("#hardware-gpus", Static))
            assert "GPU 0  GPU A" in gpu_rows
            assert "GPU 1  GPU B" in gpu_rows

            await pilot.press("ctrl+l")
            for _ in range(30):
                state = app.dashboard_state
                if (
                    hardware.collect_count == 3
                    and state is not None
                    and not state.hardware_is_refreshing
                ):
                    break
                await pilot.pause()
            assert "10 GPUs" in _plain_render(app.query_one("#metric-gpu", Static))
            gpu_rows = _plain_render(app.query_one("#hardware-gpus", Static))
            assert len(gpu_rows.splitlines()) == 10
            workspace = app.query_one(Workspace)
            assert not workspace.can_focus
            workspace.scroll_end(animate=False)
            await pilot.pause()
            assert workspace.max_scroll_y > 0
            assert workspace.scroll_y > 0

    asyncio.run(scenario())


def test_manual_refresh_starts_one_operation_per_lane_and_overlap_skips() -> None:
    async def scenario() -> None:
        transport = _ScriptedTransport(
            [
                httpx.Response(200, json={"data": [{"id": "model"}]}),
                httpx.Response(200, json={"data": [{"id": "model"}]}),
                httpx.Response(200, json={"data": [{"id": "model"}]}),
            ]
        )
        hardware = _ScriptedHardwareProvider(
            [_hardware_snapshot(), _hardware_snapshot(second=1)]
        )
        app = _app_with_transport(transport, hardware_provider=hardware)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            await _wait_for_hardware(app, pilot)
            await pilot.press("ctrl+l")
            for _ in range(30):
                if transport.requests == 2 and hardware.collect_count == 2:
                    break
                await pilot.pause()
            assert transport.requests == 2
            assert hardware.collect_count == 2

        gated_transport = _ScriptedTransport(
            [
                httpx.Response(200, json={"data": [{"id": "model"}]}),
                httpx.Response(200, json={"data": [{"id": "model"}]}),
            ]
        )
        gated_hardware = _ScriptedHardwareProvider(gate_call=1)
        gated_app = _app_with_transport(
            gated_transport,
            hardware_provider=gated_hardware,
        )
        async with gated_app.run_test(size=(100, 30)) as pilot:
            await gated_hardware.started.wait()
            await _wait_for_status(gated_app, pilot, ServerStatus.ONLINE)
            await pilot.press("ctrl+l")
            for _ in range(30):
                if gated_transport.requests == 2:
                    break
                await pilot.pause()
            assert gated_transport.requests == 2
            assert gated_hardware.collect_count == 1
            gated_hardware.release.set()
            await _wait_for_hardware(gated_app, pilot)
        assert gated_hardware.close_count == 1

    asyncio.run(scenario())


def test_disabled_hardware_creates_no_lane_and_leaves_server_functional() -> None:
    async def scenario() -> None:
        transport = _ScriptedTransport(
            [httpx.Response(200, json={"data": [{"id": "model"}]})]
        )
        hardware = _ScriptedHardwareProvider()
        app = _app_with_transport(
            transport,
            hardware_provider=hardware,
            hardware_enabled=False,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            assert app.dashboard_state is not None
            assert (
                app.dashboard_state.hardware_last_error
                == "Hardware monitoring disabled"
            )
            assert "Hardware disabled" in _plain_render(app.query_one(StatusFooter))
            assert app.__dict__["_hardware_monitor"] is None
            assert app.__dict__["_hardware_timer"] is None
            assert app.__dict__["_hardware_worker"] is None
            assert hardware.collect_count == 0
        assert hardware.close_count == 0
        assert transport.close_count == 1

    asyncio.run(scenario())
