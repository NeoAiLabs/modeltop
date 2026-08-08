"""Selectable r0b0bench widget and end-to-end app behavior tests."""
# pyright: reportPrivateUsage=false

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.pilot import Pilot
from textual.widgets import (
    Checkbox,
    ContentSwitcher,
    DataTable,
    Input,
    OptionList,
    Static,
)

import modeltop.widgets.r0b0bench_configuration as configuration_widget_module
from modeltop.api.client import OpenAICompatibleClient
from modeltop.app import ModelTopApp
from modeltop.benchmarks.models import (
    R0b0benchBenchmarkConfig,
    R0b0benchBenchmarkProgress,
    R0b0benchBenchmarkResult,
    R0b0benchBenchmarkStatus,
    R0b0benchLaneResult,
    R0b0benchLaneStatus,
    R0b0benchMetric,
)
from modeltop.benchmarks.r0b0bench import (
    R0B0BENCH_COMMIT,
    R0B0BENCH_REPORT_SCHEMA,
    R0B0BENCH_VERSION,
    R0b0benchPreparedRun,
    R0b0benchRunner,
    R0b0benchRunnerReport,
    R0b0benchRunnerRequest,
)
from modeltop.models import ModelTopConfig
from modeltop.screens.r0b0bench import R0b0benchView
from modeltop.services.r0b0bench_datasets import R0b0benchAssetStatus
from modeltop.state import ServerStatus, initial_application_state
from modeltop.widgets.footer import StatusFooter
from modeltop.widgets.r0b0bench_configuration import R0b0benchConfigurationPanel
from modeltop.widgets.r0b0bench_progress import R0b0benchProgressPanel
from modeltop.widgets.r0b0bench_results import R0b0benchResultsPanel

_NOW = datetime(2026, 8, 4, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolate_dataset_registry(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_paths() -> dict[str, str]:
        return {}

    def no_statuses() -> tuple[R0b0benchAssetStatus, ...]:
        return ()

    monkeypatch.setattr(
        configuration_widget_module, "r0b0bench_installed_paths", no_paths
    )
    monkeypatch.setattr(
        configuration_widget_module, "r0b0bench_asset_status", no_statuses
    )


def _plain(widget: Static) -> str:
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


class _ConfigurationApp(App[None]):
    def compose(self) -> ComposeResult:
        yield R0b0benchConfigurationPanel()


class _ProgressApp(App[None]):
    def compose(self) -> ComposeResult:
        yield R0b0benchProgressPanel()


class _ResultsApp(App[None]):
    def compose(self) -> ComposeResult:
        yield R0b0benchResultsPanel()


def _result(tmp_path: Path) -> R0b0benchBenchmarkResult:
    row = R0b0benchLaneResult(
        "canary",
        R0b0benchLaneStatus.PASS,
        0,
        0.25,
        (
            R0b0benchMetric("passed", True, None),
            R0b0benchMetric("cases", 1, "count"),
        ),
    )
    return R0b0benchBenchmarkResult(
        benchmark_id="r0b0bench-20260804T120000Z-deadbeef",
        upstream_run_id="r0b0bench-20260804T120000Z-deadbeef",
        upstream_version=R0B0BENCH_VERSION,
        upstream_schema_version=R0B0BENCH_REPORT_SCHEMA,
        upstream_commit=R0B0BENCH_COMMIT,
        config=R0b0benchBenchmarkConfig(selected_lanes=("canary",)),
        server_id="server",
        server_name="Server",
        server_endpoint="127.0.0.1:8000",
        model_id="org/model",
        backend="vLLM",
        started_at=_NOW,
        completed_at=_NOW,
        status=R0b0benchBenchmarkStatus.COMPLETED,
        cancelled=False,
        error_code=None,
        error_message=None,
        selected_count=1,
        completed_count=1,
        unstarted_lanes=(),
        lanes=(row,),
        infra_errors_total=0,
        invalid_for_publish=True,
        warning_codes=("filtered_selection",),
        hardware_summary=None,
        run_directory=tmp_path / "PRIVATE_EVIDENCE",
    )


def test_configuration_selects_exact_tests_and_validates_inputs() -> None:
    async def scenario() -> None:
        app = _ConfigurationApp()
        async with app.run_test(size=(120, 50)) as pilot:
            panel = app.query_one(R0b0benchConfigurationPanel)
            assert panel.parse_config() == R0b0benchBenchmarkConfig()

            panel.query_one("#r0b0bench-lane-perf", Checkbox).value = True
            await pilot.pause()
            assert panel.query_one("#r0b0bench-lane-latency", Checkbox).disabled
            assert panel.query_one("#r0b0bench-lane-concurrency", Checkbox).disabled
            assert panel.query_one("#r0b0bench-lane-throughput", Checkbox).disabled
            parsed = panel.parse_config()
            assert parsed is not None and parsed.selected_lanes == ("canary", "perf")

            profile = panel.query_one("#r0b0bench-profile", OptionList)
            profile.highlighted = 2
            profile.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert panel.query_one("#r0b0bench-lane-qa", Checkbox).disabled
            assert not panel.query_one("#r0b0bench-lane-qa", Checkbox).value

            timeout = panel.query_one("#r0b0bench-timeout", Input)
            timeout.value = "not-a-number"
            assert panel.parse_config(notify=True) is None
            assert "finite positive" in _plain(
                panel.query_one("#r0b0bench-error", Static)
            )

    asyncio.run(scenario())


def test_configuration_prefills_validated_installed_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text("{}\n")
    monkeypatch.setattr(
        configuration_widget_module,
        "r0b0bench_installed_paths",
        lambda: {"qa_data_path": str(qa_path)},
    )
    monkeypatch.setattr(
        configuration_widget_module,
        "r0b0bench_asset_status",
        lambda: (R0b0benchAssetStatus("qa", "QA / ARC-Easy", "installed", qa_path),),
    )

    async def scenario() -> None:
        app = _ConfigurationApp()
        async with app.run_test(size=(120, 50)):
            panel = app.query_one(R0b0benchConfigurationPanel)
            assert panel.query_one("#r0b0bench-qa-data", Input).value == str(qa_path)
            status = _plain(panel.query_one("#r0b0bench-assets", Static))
            assert "QA / ARC-Easy: INSTALLED" in status

    asyncio.run(scenario())


def test_progress_and_result_panels_render_normalized_payload_free_state(
    tmp_path: Path,
) -> None:
    async def progress_scenario() -> None:
        app = _ProgressApp()
        async with app.run_test(size=(100, 25)):
            state = replace(
                initial_application_state("server", hardware_enabled=False),
                selected_model_id="org/model",
            )
            progress = R0b0benchBenchmarkProgress(
                configured_count=4,
                completed_count=2,
                pass_count=1,
                fail_count=1,
                skip_count=0,
                error_count=0,
                not_implemented_count=0,
                current_lane="concurrency",
                elapsed_seconds=12.5,
                cached_hardware=None,
            )
            state = replace(
                state,
                r0b0bench_benchmark=replace(
                    state.r0b0bench_benchmark,
                    status=R0b0benchBenchmarkStatus.RUNNING,
                    active_benchmark_id="r0b0bench-active",
                    progress=progress,
                ),
            )
            panel = app.query_one(R0b0benchProgressPanel)
            panel.update_state(state)
            text = "\n".join(_plain(widget) for widget in panel.query(Static))
            assert "2/4 completed" in text
            assert "pass 1" in text and "fail 1" in text
            assert "3/4 · concurrency" in text
            assert "12.5s" in text
            assert "Esc cancels" in text

    async def result_scenario() -> None:
        app = _ResultsApp()
        async with app.run_test(size=(120, 35)):
            panel = app.query_one(R0b0benchResultsPanel)
            panel.update_result(_result(tmp_path))
            text = "\n".join(_plain(widget) for widget in panel.query(Static))
            assert "DIAGNOSTIC · INVALID FOR PUBLISH" in text
            assert R0B0BENCH_COMMIT in text
            assert "Filtered test selection" in text
            assert str(tmp_path / "PRIVATE_EVIDENCE") in text
            assert "prompt" not in text.lower()
            table = cast(DataTable[str], panel.query_one(DataTable))
            assert table.row_count == 1

    asyncio.run(progress_scenario())
    asyncio.run(result_scenario())


class _ModelsTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"data": [{"id": "org/model"}]})

    async def aclose(self) -> None:
        self.close_count += 1


class _AppRunner:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def prepare(self, request: R0b0benchRunnerRequest) -> R0b0benchPreparedRun:
        run_directory = self.root / request.benchmark_id
        run_directory.mkdir(mode=0o700, parents=True)
        return R0b0benchPreparedRun(request, {}, run_directory)

    async def run(
        self,
        prepared: R0b0benchPreparedRun,
        on_start: Any,
        on_finish: Any,
    ) -> R0b0benchRunnerReport:
        rows: list[R0b0benchLaneResult] = []
        for index, lane_id in enumerate(prepared.request.config.selected_lanes, 1):
            metrics = (
                (
                    R0b0benchMetric("passed", True, None),
                    R0b0benchMetric("cases", 1, "count"),
                )
                if lane_id == "canary"
                else ()
            )
            row = R0b0benchLaneResult(
                lane_id,
                R0b0benchLaneStatus.PASS,
                0,
                0.1,
                metrics,
            )
            await on_start(lane_id, index, len(prepared.request.config.selected_lanes))
            await on_finish(row, index, len(prepared.request.config.selected_lanes))
            rows.append(row)
        return R0b0benchRunnerReport(
            upstream_run_id=prepared.request.benchmark_id,
            upstream_version=R0B0BENCH_VERSION,
            schema_version=R0B0BENCH_REPORT_SCHEMA,
            profile=prepared.request.config.profile,
            model_id=prepared.request.model_id,
            elapsed_seconds=0.4,
            invalid_for_publish=True,
            infra_errors_total=0,
            lanes=tuple(rows),
            unstarted_lanes=(),
            run_directory=prepared.run_directory,
            cancelled=False,
        )


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
                    "api_key": "EMPTY",
                    "backend_hint": "vllm",
                }
            ],
        }
    )


async def _wait_for_server_online(app: ModelTopApp, pilot: Pilot[None]) -> None:
    for _ in range(30):
        state = app.dashboard_state
        if state is not None and state.server_status is ServerStatus.ONLINE:
            return
        await pilot.pause()
    raise AssertionError("dashboard did not reach online")


async def _wait_for_r0b0bench(
    app: ModelTopApp, status: R0b0benchBenchmarkStatus
) -> None:
    for _ in range(100):
        state = app.dashboard_state
        if state is not None and state.r0b0bench_benchmark.status is status:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"r0b0bench did not reach {status}")


def test_app_runs_selectable_workspace_archives_and_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            "modeltop.services.result_archive._DEFAULT_HISTORY_DIRECTORY",
            tmp_path / "history",
        )
        transport = _ModelsTransport()
        app = ModelTopApp(
            _app_config(),
            client=OpenAICompatibleClient(
                "http://server/prefix/v1", "EMPTY", 5, transport=transport
            ),
            r0b0bench_runner=cast(R0b0benchRunner, _AppRunner(tmp_path / "runs")),
        )
        async with app.run_test(size=(125, 42)) as pilot:
            await _wait_for_server_online(app, pilot)
            menu = app.query_one("#sidebar-menu", OptionList)
            menu.highlighted = 6
            menu.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert app.dashboard_state is not None
            assert app.dashboard_state.active_view == "r0b0bench"
            assert app.query_one(R0b0benchView).config_panel.parse_config() == (
                R0b0benchBenchmarkConfig()
            )

            app.action_run_or_rerun()
            await _wait_for_r0b0bench(app, R0b0benchBenchmarkStatus.COMPLETED)
            await pilot.pause()
            state = app.dashboard_state
            assert state is not None
            result = state.r0b0bench_benchmark.latest_result
            assert result is not None and result.completed_count == 4
            assert result.invalid_for_publish
            switcher = app.query_one("#r0b0bench-view-switcher", ContentSwitcher)
            assert switcher.current == "r0b0bench-result-panel"
            assert "R0B0BENCH COMPLETE" in _plain(
                app.query_one("#header-subtitle", Static)
            )
            assert "pass 4/4" in _plain(app.query_one(StatusFooter))
            assert any(
                entry.kind == "r0b0bench" for entry in state.result_archive.entries
            )

            app.action_run_or_rerun()
            await _wait_for_r0b0bench(app, R0b0benchBenchmarkStatus.COMPLETED)
            assert app.dashboard_state.r0b0bench_benchmark.latest_result is not result

            app.action_export_result()
            await pilot.pause()
            assert switcher.current == "r0b0bench-config-panel"

        assert transport.close_count == 1

    asyncio.run(scenario())
