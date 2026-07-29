"""Official aggregate and payload-free scenario Tool Calling results."""

from typing import cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Static

from modeltop.benchmarks.models import ToolCallingBenchmarkResult
from modeltop.messages import (
    ToolCallingBenchmarkEditRequested,
    ToolCallingBenchmarkRunAgainRequested,
)

_FAILURE_LABELS = {
    "wrong_tool": "Wrong tool",
    "wrong_args": "Wrong arguments",
    "missing_step": "Missing step",
    "forbidden_action": "Forbidden action",
    "budget_exceeded": "Turn budget exceeded",
    "timeout": "Timeout (excluded)",
    "connection_error": "Connection error (excluded)",
    "server_error": "Server error (excluded)",
    "model_crash": "Model crash",
    "evaluator_error": "Evaluator error",
    "partial": "Partial",
}


def _value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


class ToolCallingResultsPanel(VerticalScroll):
    """Render one normalized result without trace, export, or history controls."""

    DEFAULT_CSS = """
    ToolCallingResultsPanel { width: 1fr; height: 1fr; padding: 0 1; }
    ToolCallingResultsPanel .section-title {
        height: 1; color: #5da9e9; text-style: bold;
    }
    ToolCallingResultsPanel #tool-calling-result-summary { height: auto; }
    ToolCallingResultsPanel DataTable { height: auto; min-height: 5; max-height: 18; }
    ToolCallingResultsPanel #tool-calling-result-actions { height: 3; }
    """

    def compose(self) -> ComposeResult:
        yield Static("TOOL CALLING RESULT", classes="section-title")
        yield Static("", id="tool-calling-result-summary", markup=False)
        yield Static("CATEGORY SCORES", classes="section-title")
        yield DataTable(id="tool-calling-category-table", zebra_stripes=True)
        yield Static("SCENARIO OUTCOMES", classes="section-title")
        yield DataTable(id="tool-calling-scenario-table", zebra_stripes=True)
        yield Horizontal(
            Button("Run Again", id="tool-calling-run-again", variant="primary"),
            Button("Edit", id="tool-calling-edit"),
            id="tool-calling-result-actions",
        )

    def on_mount(self) -> None:
        categories = cast(
            DataTable[str],
            self.query_one("#tool-calling-category-table", DataTable),
        )
        categories.cursor_type = "row"
        for key, label in (
            ("category", "Category"),
            ("points", "Points"),
            ("percent", "Percent"),
            ("pass", "Pass"),
            ("partial", "Partial"),
            ("fail", "Fail"),
        ):
            categories.add_column(label, key=key)
        scenarios = cast(
            DataTable[str],
            self.query_one("#tool-calling-scenario-table", DataTable),
        )
        scenarios.cursor_type = "row"
        for key, label in (
            ("id", "Scenario"),
            ("category", "Cat"),
            ("status", "Status"),
            ("points", "Pts"),
            ("duration", "Duration"),
            ("ttft", "TTFT"),
            ("turns", "Turns"),
            ("failure", "Failure kind"),
        ):
            scenarios.add_column(label, key=key)

    def update_result(self, result: ToolCallingBenchmarkResult) -> None:
        """Render only fields explicitly retained by ModelTop's boundary."""
        categories = cast(
            DataTable[str],
            self.query_one("#tool-calling-category-table", DataTable),
        )
        categories.clear()
        for category in result.categories:
            categories.add_row(
                f"{category.category} {category.label}",
                f"{category.earned_points}/{category.max_points}",
                f"{category.percent:.0f}%",
                str(category.pass_count),
                str(category.partial_count),
                str(category.fail_count),
                key=category.category,
            )
        scenarios = cast(
            DataTable[str],
            self.query_one("#tool-calling-scenario-table", DataTable),
        )
        scenarios.clear()
        for row in result.scenarios:
            scenarios.add_row(
                row.scenario_id,
                row.category,
                row.status.value.upper(),
                str(row.points),
                f"{row.duration_seconds:.1f}s",
                _value(row.ttft_ms, "ms"),
                str(row.turn_count),
                _FAILURE_LABELS[row.failure_kind]
                if row.failure_kind is not None
                else "--",
                key=row.scenario_id,
            )

        if result.config.suite == "core" or not result.category_k_gradable:
            safety = "NOT COVERED"
        elif result.safety_gate_passed:
            safety = "PASSED"
        else:
            safety = "FAILED"
        summary = [
            f"Status: {result.status.value.upper()} · "
            f"Suite: {result.config.suite.title()} · "
            f"Wall time: {result.wall_time_seconds:.1f}s",
            f"Official score: {_value(result.final_score, '%')} · "
            f"Rating: {result.rating or '--'} · "
            f"Points: {_value(result.total_points)}/{_value(result.max_points)}",
            f"Completion: {_value(result.completion_rate_percent, '%')} · "
            f"Gradable: {result.gradable_count}/{result.attempted_count} · "
            f"Excluded: {result.excluded_count}",
            f"Safety: {safety} · Deployability: {_value(result.deployability, '%')} · "
            f"Responsiveness: {_value(result.responsiveness, '%')}",
            f"Median turn latency: {_value(result.median_turn_ms, 'ms')} · "
            f"Tokens: {result.total_tokens} "
            f"({result.prompt_tokens} prompt / {result.completion_tokens} completion)",
            f"Upstream: {result.upstream_version or '--'} · "
            f"Schema: {result.schema_version or '--'}",
        ]
        if result.excluded_count:
            summary.append(
                "Infrastructure failures were excluded from score points and capacity."
            )
        if result.cancelled:
            summary.append(
                "Cancelled result contains completed callbacks only; official "
                "aggregates "
                "are unavailable."
            )
        if result.error_message is not None:
            summary.append(result.error_message)
        summary.extend(result.warnings)
        self.query_one("#tool-calling-result-summary", Static).update(
            "\n".join(summary)
        )

    def focus_actions(self) -> None:
        self.query_one("#tool-calling-run-again", Button).focus()

    @on(Button.Pressed, "#tool-calling-run-again")
    def run_again(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ToolCallingBenchmarkRunAgainRequested())

    @on(Button.Pressed, "#tool-calling-edit")
    def edit(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ToolCallingBenchmarkEditRequested())
