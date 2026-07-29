"""Keyboard and payload-free rendering coverage for Tool Calling widgets."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input, OptionList, Static

from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingBenchmarkProgress,
    ToolCallingBenchmarkResult,
    ToolCallingBenchmarkStatus,
    ToolCallingCategoryScore,
    ToolCallingScenarioProgress,
    ToolCallingScenarioResult,
    ToolCallingScenarioStatus,
)
from modeltop.state import initial_application_state
from modeltop.widgets.tool_calling_configuration import (
    ToolCallingConfigurationPanel,
)
from modeltop.widgets.tool_calling_progress import ToolCallingProgressPanel
from modeltop.widgets.tool_calling_results import ToolCallingResultsPanel

_NOW = datetime(2026, 3, 20, tzinfo=UTC)


def _plain(widget: Static) -> str:
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Text) else str(rendered)


class _ConfigurationApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ToolCallingConfigurationPanel()


class _ProgressApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ToolCallingProgressPanel()


class _ResultsApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ToolCallingResultsPanel()


def _scenario() -> ToolCallingScenarioResult:
    return ToolCallingScenarioResult(
        scenario_id="single_tool",
        category="A",
        title="Single tool",
        status=ToolCallingScenarioStatus.PASS,
        points=2,
        failure_kind=None,
        duration_seconds=1.0,
        ttft_ms=25.0,
        turn_count=1,
        prompt_tokens=10,
        completion_tokens=5,
        infrastructure_excluded=False,
    )


def _result() -> ToolCallingBenchmarkResult:
    return ToolCallingBenchmarkResult(
        benchmark_id="tool-call-widget",
        upstream_run_id="opaque-run",
        config_fingerprint="opaque-fingerprint",
        server_id="server",
        server_name="Server",
        server_endpoint="127.0.0.1:8000",
        model_id="model",
        backend="vLLM",
        integration_commit="7ec8fcf33943020349ff6df339834a7ef984da00",
        upstream_version="2.3.0",
        schema_version="1",
        config=ToolCallingBenchmarkConfig(suite="core"),
        started_at=_NOW,
        completed_at=_NOW,
        status=ToolCallingBenchmarkStatus.COMPLETED,
        cancelled=False,
        error_code=None,
        error_message=None,
        attempted_count=15,
        gradable_count=1,
        excluded_count=14,
        completion_rate_percent=6.7,
        final_score=100,
        total_points=2,
        max_points=2,
        rating="★★★★★ Excellent",
        category_k_gradable=False,
        safety_gate_passed=True,
        deployability=91,
        responsiveness=80,
        median_turn_ms=25.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        categories=(
            ToolCallingCategoryScore(
                category="A",
                label="Tool Selection",
                earned_points=2,
                max_points=2,
                percent=100.0,
                pass_count=1,
                partial_count=0,
                fail_count=0,
            ),
        ),
        scenarios=(_scenario(),),
        warnings=("Fixed adapter-family fallback warning.",),
        hardware_summary=None,
    )


def test_configuration_labels_parse_strictly_and_preserve_focus() -> None:
    async def scenario() -> None:
        app = _ConfigurationApp()
        async with app.run_test(size=(100, 30)) as pilot:
            panel = app.query_one(ToolCallingConfigurationPanel)
            suite = panel.query_one("#tool-calling-suite", OptionList)
            assert suite.option_count == 2
            assert "Full · 69 scenarios" in str(suite.get_option_at_index(0).prompt)
            assert "Core · 15 scenarios" in str(suite.get_option_at_index(1).prompt)
            full = panel.parse_config()
            assert full is not None and full.suite == "full"

            suite.focus()
            await pilot.press("down", "enter")
            await pilot.pause()
            core = panel.parse_config()
            assert core is not None and core.suite == "core"
            assert app.focused is suite

            timeout = panel.query_one("#tool-calling-timeout", Input)
            timeout.value = "inf"
            assert panel.parse_config() is None
            assert _plain(panel.query_one("#tool-calling-error", Static))
            timeout.value = "42.5"
            parsed = panel.parse_config()
            assert parsed is not None
            assert parsed.request_timeout_seconds == 42.5

    asyncio.run(scenario())


def test_progress_renders_counts_current_scenario_and_cancel_hint() -> None:
    async def scenario() -> None:
        app = _ProgressApp()
        async with app.run_test(size=(100, 25)):
            state = initial_application_state("server", hardware_enabled=False)
            state = replace(state, selected_model_id="model")
            progress = ToolCallingBenchmarkProgress(
                configured_count=15,
                completed_count=2,
                gradable_count=1,
                excluded_count=1,
                pass_count=1,
                partial_count=0,
                fail_count=0,
                current_scenario=ToolCallingScenarioProgress(
                    scenario_id="third_case",
                    category="C",
                    title="Third case",
                    source_index=2,
                    started_at_monotonic=10.0,
                ),
                elapsed_seconds=12.5,
                cached_hardware=None,
            )
            state = replace(
                state,
                tool_calling_benchmark=replace(
                    state.tool_calling_benchmark,
                    config=ToolCallingBenchmarkConfig(suite="core"),
                    status=ToolCallingBenchmarkStatus.RUNNING,
                    active_benchmark_id="tool-call-widget",
                    progress=progress,
                ),
            )
            panel = app.query_one(ToolCallingProgressPanel)
            panel.update_state(state)
            text = "\n".join(_plain(widget) for widget in panel.query(Static))
            assert "2/15" in text
            assert "pass 1" in text
            assert "excluded 1" in text
            assert "third_case" in text
            assert "12.5s" in text
            assert "Esc cancels" in text

    asyncio.run(scenario())


def test_results_render_official_tables_without_payload_fields() -> None:
    async def scenario() -> None:
        app = _ResultsApp()
        async with app.run_test(size=(120, 36)):
            panel = app.query_one(ToolCallingResultsPanel)
            result = _result()
            panel.update_result(result)
            summary = _plain(panel.query_one("#tool-calling-result-summary", Static))
            assert "Official score: 100%" in summary
            assert "Completion: 6.7%" in summary
            assert "Safety: NOT COVERED" in summary
            assert "Infrastructure failures" in summary
            assert "2.3.0" in summary
            assert "SENSITIVE" not in summary
            categories = cast(
                DataTable[str],
                app.query_one("#tool-calling-category-table", DataTable),
            )
            scenarios = cast(
                DataTable[str],
                app.query_one("#tool-calling-scenario-table", DataTable),
            )
            assert categories.row_count == 1
            assert scenarios.row_count == 1

    asyncio.run(scenario())
