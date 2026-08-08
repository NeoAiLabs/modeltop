"""Normalized, provenance-rich r0b0bench result rendering."""

from typing import cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Static

from modeltop.benchmarks.models import (
    R0b0benchBenchmarkResult,
    R0b0benchMetric,
    R0b0benchWarningCode,
)
from modeltop.messages import (
    R0b0benchBenchmarkEditRequested,
    R0b0benchBenchmarkRunAgainRequested,
)

_WARNING_TEXT: dict[R0b0benchWarningCode, str] = {
    "filtered_selection": "Filtered test selection; this run is diagnostic.",
    "perf_composite": "Perf is a diagnostic composite, not a publishable lane.",
    "canary_infrastructure_stop": (
        "Canary infrastructure failure stopped remaining tests."
    ),
    "cancelled_partial": "Cancellation retained only completed validated tests.",
}


def _metric(metric: R0b0benchMetric) -> str:
    if isinstance(metric.value, bool):
        value = "true" if metric.value else "false"
    elif isinstance(metric.value, float):
        value = f"{metric.value:.3f}".rstrip("0").rstrip(".")
    else:
        value = str(metric.value)
    return f"{metric.name}={value}{'' if metric.unit is None else f' {metric.unit}'}"


class R0b0benchResultsPanel(VerticalScroll):
    """Render normalized lanes without a composite ModelTop score."""

    DEFAULT_CSS = """
    R0b0benchResultsPanel { width: 1fr; height: 1fr; padding: 0 1; }
    R0b0benchResultsPanel .section-title {
        height: 1; color: $primary; text-style: bold;
    }
    R0b0benchResultsPanel .result-line { height: auto; min-height: 1; }
    R0b0benchResultsPanel #r0b0bench-validity {
        height: auto; text-style: bold;
    }
    R0b0benchResultsPanel DataTable { height: auto; min-height: 6; }
    R0b0benchResultsPanel #r0b0bench-result-actions { height: 3; }
    """

    def compose(self) -> ComposeResult:
        yield Static("R0B0BENCH RESULT", classes="section-title")
        yield Static("", id="r0b0bench-result-summary", classes="result-line")
        yield Static("", id="r0b0bench-validity")
        yield Static("", id="r0b0bench-provenance", classes="result-line")
        yield Static("", id="r0b0bench-counts", classes="result-line")
        yield Static("", id="r0b0bench-warnings", classes="result-line")
        yield DataTable(id="r0b0bench-lanes", cursor_type="row")
        yield Static("", id="r0b0bench-unstarted", classes="result-line")
        yield Static("", id="r0b0bench-evidence", classes="result-line")
        yield Horizontal(
            Button("Run Again", id="r0b0bench-run-again", variant="primary"),
            Button("Edit", id="r0b0bench-edit"),
            id="r0b0bench-result-actions",
        )

    def on_mount(self) -> None:
        table = cast(DataTable[str], self.query_one("#r0b0bench-lanes", DataTable))
        table.add_column("Test", key="lane")
        table.add_column("Status", key="status")
        table.add_column("Infra", key="infra")
        table.add_column("Elapsed", key="elapsed")
        table.add_column("Normalized metrics", key="metrics")

    def update_result(self, result: R0b0benchBenchmarkResult) -> None:
        """Render only normalized and explicitly safe retained fields."""
        self.query_one("#r0b0bench-result-summary", Static).update(
            f"Status: {result.status.value.replace('_', ' ').upper()} · "
            f"Profile: {result.config.profile} · Model: {result.model_id}"
        )
        validity = (
            "DIAGNOSTIC · INVALID FOR PUBLISH"
            if result.invalid_for_publish
            else "PUBLISH-VALID"
        )
        self.query_one("#r0b0bench-validity", Static).update(validity)
        self.query_one("#r0b0bench-provenance", Static).update(
            f"Upstream: {result.upstream_version or '--'} · "
            f"commit {result.upstream_commit} · "
            f"schema {result.upstream_schema_version or '--'}"
        )
        pass_count = sum(row.status.value == "PASS" for row in result.lanes)
        fail_count = sum(row.status.value == "FAIL" for row in result.lanes)
        error_count = sum(
            row.status.value in {"ERROR", "NOT_IMPLEMENTED"} for row in result.lanes
        )
        self.query_one("#r0b0bench-counts", Static).update(
            f"Selected: {result.selected_count} · "
            f"Completed: {result.completed_count} · "
            f"Pass: {pass_count} · Fail: {fail_count} · Error: {error_count} · "
            f"Infra: {result.infra_errors_total} · "
            f"Time: {result.wall_time_seconds:.1f}s"
        )
        warnings = "\n".join(_WARNING_TEXT[code] for code in result.warning_codes)
        if result.error_message:
            warnings = f"{warnings}\n{result.error_message}".strip()
        self.query_one("#r0b0bench-warnings", Static).update(warnings)
        table = cast(DataTable[str], self.query_one("#r0b0bench-lanes", DataTable))
        table.clear()
        for row in result.lanes:
            table.add_row(
                row.lane_id,
                row.status.value,
                str(row.infra_errors),
                "--" if row.elapsed_seconds is None else f"{row.elapsed_seconds:.2f}s",
                " · ".join(_metric(metric) for metric in row.metrics) or "--",
                key=row.lane_id,
            )
        self.query_one("#r0b0bench-unstarted", Static).update(
            "Unstarted tests: "
            + (", ".join(result.unstarted_lanes) if result.unstarted_lanes else "none")
        )
        self.query_one("#r0b0bench-evidence", Static).update(
            "Private evidence: "
            + (str(result.run_directory) if result.run_directory is not None else "--")
        )

    def focus_actions(self) -> None:
        self.query_one("#r0b0bench-run-again", Button).focus()

    @on(Button.Pressed, "#r0b0bench-run-again")
    def run_again(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(R0b0benchBenchmarkRunAgainRequested())

    @on(Button.Pressed, "#r0b0bench-edit")
    def edit(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(R0b0benchBenchmarkEditRequested())
