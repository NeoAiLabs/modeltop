"""Textual Pilot coverage for Speed Test workspaces and session Results."""
# pyright: reportPrivateUsage=false

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, ContentSwitcher, Input, OptionList, Static

import modeltop.app as app_module
from modeltop.benchmarks.models import (
    SpeedTestConfig,
    SpeedTestResult,
    SpeedTestStatus,
)
from modeltop.screens.results import ResultsView
from modeltop.screens.settings import SettingsView
from modeltop.screens.speed_test import SpeedTestView
from modeltop.services.result_export import ResultExportError
from modeltop.state import ServerStatus
from modeltop.widgets.footer import StatusFooter
from modeltop.widgets.speed_test_config import SpeedTestConfigPanel
from modeltop.widgets.speed_test_progress import SpeedTestProgressPanel
from modeltop.widgets.speed_test_results import SpeedTestResultsPanel
from tests.test_app import _plain_render, _wait_for_status
from tests.test_chat_app import _app, _ChatTransport, _GatedChatStream


def test_speed_workflow_results_reopen_and_settings_navigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        transport = _ChatTransport()
        app = _app(transport)
        async with app.run_test(size=(100, 30)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 2
            await pilot.press("enter")
            await pilot.pause()

            switcher = app.query_one("#workspace-switcher", ContentSwitcher)
            assert switcher.current == "speed-test-workspace"
            view = app.query_one(SpeedTestView)
            panel = view.query_one(SpeedTestConfigPanel)
            assert panel.query_one("#speed-warmups", Input).value == "1"
            assert panel.query_one("#speed-measured", Input).value == "5"
            assert panel.query_one("#speed-max-tokens", Input).value == "256"

            panel.query_one("#speed-measured", Input).value = "0"
            panel.query_one("#speed-start", Button).press()
            await pilot.pause()
            assert transport.chat_requests == []
            assert app.dashboard_state is not None
            assert app.dashboard_state.speed_test.status is SpeedTestStatus.IDLE

            config = SpeedTestConfig(
                preset="custom",
                prompt="benchmark-only prompt",
                warmup_runs=1,
                measured_runs=3,
                max_tokens=32,
            )
            panel.load_config(config)
            panel.query_one("#speed-start", Button).press()
            await pilot.pause()
            for _ in range(200):
                await asyncio.sleep(0.005)
                state = app.dashboard_state
                if state.speed_test.status.is_terminal:
                    break
            state = app.dashboard_state
            assert state is not None
            assert state.speed_test.status is SpeedTestStatus.COMPLETED
            assert len(transport.chat_requests) == 4
            assert all(
                request["messages"]
                == [{"role": "user", "content": "benchmark-only prompt"}]
                for request in transport.chat_requests
            )
            assert state.chat_session.messages == ()
            assert view.results_panel.result is state.speed_test.latest_result
            rendered = "\n".join(
                _plain_render(widget)
                for widget in view.query_one(SpeedTestResultsPanel).query(Static)
            )
            assert "3/3 successful" in rendered
            assert "Warm-up 1" in rendered
            assert "Run 3" in rendered
            app.action_copy_summary()
            assert app.clipboard is not None
            assert "ModelTop Speed Test" in app.clipboard
            exported = tmp_path / "result.json"
            exported.write_text("{}\n", encoding="utf-8")

            def successful_export(_result: SpeedTestResult) -> Path:
                return exported

            monkeypatch.setattr(
                app_module,
                "export_speed_test_result",
                successful_export,
            )
            app.action_export_result()

            def fail_export(_result: SpeedTestResult) -> Path:
                raise ResultExportError("Result export failed: denied")

            monkeypatch.setattr(app_module, "export_speed_test_result", fail_export)
            app.action_export_result()
            await pilot.pause()

            menu.focus()
            menu.highlighted = 7
            await pilot.press("enter")
            await pilot.pause()
            assert switcher.current == "results-workspace"
            results = app.query_one(ResultsView)
            history = results.query_one("#results-list", OptionList)
            assert history.option_count == 1
            history.focus()
            history.highlighted = 0
            await pilot.press("enter")
            await pilot.pause()
            assert results.showing_detail
            await pilot.press("escape")
            await pilot.pause()
            assert not results.showing_detail
            await pilot.press("escape")
            await pilot.pause()
            assert switcher.current == "speed-test-workspace"

            menu.focus()
            menu.highlighted = 8
            await pilot.press("enter")
            await pilot.pause()
            assert switcher.current == "settings-workspace"
            settings = "\n".join(
                _plain_render(widget)
                for widget in app.query_one(SettingsView).query(Static)
            )
            assert "SETTINGS · READ ONLY" in settings
            assert "Request timeout" in settings

    asyncio.run(scenario())


def test_cancelled_stream_retains_partial_and_run_again_works() -> None:
    async def scenario() -> None:
        first_chunk = "\n".join(f"line {index:02d}" for index in range(20))
        stream = _GatedChatStream(first_chunk, "rest")
        transport = _ChatTransport([stream])
        app = _app(transport)
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_for_status(app, pilot, ServerStatus.ONLINE)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 2
            await pilot.press("enter")
            panel = app.query_one(SpeedTestConfigPanel)
            panel.load_config(
                SpeedTestConfig(
                    preset="custom",
                    prompt="cancel benchmark",
                    warmup_runs=0,
                    measured_runs=1,
                )
            )
            panel.query_one("#speed-start", Button).press()
            await asyncio.wait_for(stream.first_sent.wait(), timeout=3)
            await pilot.pause()

            running = app.dashboard_state
            assert running is not None
            assert running.speed_test.status is SpeedTestStatus.RUNNING
            assert running.speed_test.live_output_preview == first_chunk
            assert running.speed_test.latest_metrics is not None
            assert running.speed_test.latest_metrics.ttft_ms is not None
            assert "SPEED RUN 1/1" in _plain_render(
                app.query_one("#header-subtitle", Static)
            )
            assert "RUNNING" in _plain_render(app.query_one(StatusFooter))

            live_output = app.query_one("#speed-live-output", VerticalScroll)
            assert live_output.max_scroll_y > 0
            assert live_output.is_vertical_scroll_end
            first_max_scroll_y = live_output.max_scroll_y
            extended = (
                first_chunk
                + "\n"
                + "\n".join(f"line {index:02d}" for index in range(20, 30))
            )
            app.query_one(SpeedTestProgressPanel).update_state(
                replace(running.speed_test, live_output_preview=extended)
            )
            await pilot.pause()
            assert live_output.max_scroll_y > first_max_scroll_y
            assert live_output.is_vertical_scroll_end

            live_output.scroll_home(animate=False, immediate=True)
            assert live_output.scroll_y == 0
            app.query_one(SpeedTestProgressPanel).update_state(
                replace(
                    running.speed_test,
                    live_output_preview=extended + "\nline 30",
                )
            )
            await pilot.pause()
            assert live_output.scroll_y == 0

            await pilot.press("escape")
            for _ in range(200):
                await asyncio.sleep(0.005)
                state = app.dashboard_state
                if state is not None and state.speed_test.status.is_terminal:
                    break
            cancelled = app.dashboard_state
            assert cancelled is not None
            assert cancelled.speed_test.status is SpeedTestStatus.CANCELLED
            result = cancelled.speed_test.latest_result
            assert result is not None
            assert result.cancelled_runs == 1
            assert result.run_results[0].response_character_count == len(first_chunk)
            assert stream.closed
            assert cancelled.chat_session.messages == ()
            assert "SPEED CANCELLED" in _plain_render(
                app.query_one("#header-subtitle", Static)
            )

            await pilot.press("r")
            for _ in range(200):
                await asyncio.sleep(0.005)
                state = app.dashboard_state
                if (
                    state is not None
                    and state.speed_test.status is SpeedTestStatus.COMPLETED
                    and len(state.speed_test.results) == 2
                ):
                    break
            rerun = app.dashboard_state
            assert rerun is not None
            assert rerun.speed_test.status is SpeedTestStatus.COMPLETED
            assert len(rerun.speed_test.results) == 2
            assert rerun.chat_session.messages == ()
            assert len(transport.chat_requests) == 2

    asyncio.run(scenario())
