"""Live immutable-state rendering for a Concurrency benchmark."""

from typing import cast

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Static

from modeltop.hardware.models import format_percentage, summarize_gpus
from modeltop.state import ApplicationState
from modeltop.widgets.percentile_summary import PercentileSummary
from modeltop.widgets.request_table import RequestTable


class BenchmarkProgressPanel(VerticalScroll):
    """Render live level, rates, hardware, warnings, and request rows."""

    DEFAULT_CSS = """
    BenchmarkProgressPanel { width: 1fr; height: 1fr; padding: 1 2; }
    BenchmarkProgressPanel #concurrency-progress-summary { height: auto; }
    BenchmarkProgressPanel #concurrency-level-table { height: 8; }
    BenchmarkProgressPanel #concurrency-request-table { height: 1fr; min-height: 8; }
    BenchmarkProgressPanel .section-title { color: #5da9e9; text-style: bold; }
    """

    def compose(self) -> ComposeResult:
        yield Static("CONCURRENCY BENCHMARK", classes="section-title")
        yield Static("Validating benchmark…", id="concurrency-progress-summary")
        yield Static(
            "LOCAL HARDWARE · Hardware metrics unavailable",
            id="concurrency-local-hardware",
        )
        yield DataTable(id="concurrency-level-table", zebra_stripes=True)
        yield Static("", id="concurrency-progress-warnings", classes="warning")
        yield PercentileSummary(id="concurrency-percentiles")
        yield RequestTable(id="concurrency-request-table")

    def on_mount(self) -> None:
        table = cast(
            DataTable[str],
            self.query_one("#concurrency-level-table", DataTable),
        )
        for column in ("CONC", "SUCCESS", "REQ/S", "TOK/S", "TTFT P50", "LAT P95"):
            table.add_column(column, key=column)

    def update_state(self, state: ApplicationState) -> None:
        lane = state.concurrency_benchmark
        progress = lane.progress
        if progress is None:
            self.query_one("#concurrency-progress-summary", Static).update(
                lane.status.value.replace("_", " ").upper()
            )
            return
        level = progress.active_concurrency_level
        phase = (
            "BETWEEN LEVELS"
            if lane.status.value == "between_levels"
            else (progress.phase or lane.status.value).replace("_", " ").upper()
        )
        summary = Text()
        summary.append(f"{phase}", style="#5da9e9 bold")
        if level is not None:
            summary.append(f" · CONC {level}")
        if progress.next_concurrency_level is not None:
            summary.append(f" · next {progress.next_concurrency_level}")
        if progress.delay_remaining_seconds is not None:
            summary.append(f" · {progress.delay_remaining_seconds:.1f}s")
        summary.append(
            f"\n{progress.completed_request_count}/"
            f"{progress.configured_requests} complete"
            f" · {progress.active_request_count} active"
            f" · {progress.queued_request_count} queued"
            f" · {progress.successful_request_count} success"
            f" · {progress.failed_request_count} failed"
            f" · {progress.timed_out_request_count} timeout"
            f" · {progress.cancelled_request_count} cancelled"
        )
        ttft = (
            "--"
            if progress.median_ttft_ms is None
            else f"{progress.median_ttft_ms:.1f} ms"
        )
        summary.append(
            f"\nElapsed {progress.elapsed_seconds:.1f}s · "
            f"{progress.aggregate_output_tokens_per_second:.1f} aggregate tok/s · "
            f"{progress.requests_per_second:.2f} req/s · median TTFT {ttft}"
        )
        self.query_one("#concurrency-progress-summary", Static).update(summary)

        snapshot = state.hardware_snapshot
        hardware = self.query_one("#concurrency-local-hardware", Static)
        if snapshot is None:
            hardware.update("LOCAL HARDWARE · Hardware metrics unavailable")
        else:
            gpu = summarize_gpus(snapshot.gpus)
            hardware.update(
                "LOCAL HARDWARE · may not represent a remote model server · "
                f"GPU {format_percentage(gpu.utilisation_percent)} · "
                f"CPU {format_percentage(snapshot.cpu.utilisation_percent)}"
            )

        table = cast(
            DataTable[str],
            self.query_one("#concurrency-level-table", DataTable),
        )
        for completed in progress.completed_levels:
            key = str(completed.concurrency)
            values = (
                str(completed.concurrency),
                f"{completed.successful_requests}/{completed.attempted_requests}",
                f"{completed.requests_per_second:.2f}",
                f"{completed.aggregate_output_tokens_per_second:.1f}",
                "--"
                if completed.ttft_ms.p50 is None
                else f"{completed.ttft_ms.p50:.1f}",
                "--"
                if completed.latency_seconds.p95 is None
                else f"{completed.latency_seconds.p95:.3f}",
            )
            if key not in table.rows:
                table.add_row(*values, key=key)
            else:
                for column, value in zip(table.columns, values, strict=True):
                    table.update_cell(key, column, value)

        warnings = "\n".join(progress.warnings)
        self.query_one("#concurrency-progress-warnings", Static).update(warnings)
        highest = progress.completed_levels[-1] if progress.completed_levels else None
        self.query_one(PercentileSummary).update_level(
            highest, requested_max_tokens=lane.config.max_tokens
        )
        self.query_one(RequestTable).update_requests(progress.request_rows)
