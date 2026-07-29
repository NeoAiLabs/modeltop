"""Terminal Concurrency benchmark result rendering."""

from typing import cast

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Static

from modeltop.benchmarks.models import ConcurrencyBenchmarkResult
from modeltop.hardware.models import format_bytes, format_percentage
from modeltop.messages import (
    ConcurrencyBenchmarkEditRequested,
    ConcurrencyBenchmarkRunAgainRequested,
)
from modeltop.theme import ERROR, PRIMARY, WARNING
from modeltop.widgets.percentile_summary import PercentileSummary
from modeltop.widgets.request_table import RequestTable


class ConcurrencyResultsPanel(VerticalScroll):
    """Show retained terminal scaling, hardware, observations, and requests."""

    DEFAULT_CSS = """
    ConcurrencyResultsPanel { width: 1fr; height: 1fr; padding: 1 2; }
    ConcurrencyResultsPanel #concurrency-result-summary { height: auto; }
    ConcurrencyResultsPanel #concurrency-scaling-table { height: 10; }
    ConcurrencyResultsPanel #concurrency-result-requests { height: 1fr; min-height: 9; }
    ConcurrencyResultsPanel #concurrency-result-actions { height: 3; }
    ConcurrencyResultsPanel .section-title { color: $primary; text-style: bold; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._result: ConcurrencyBenchmarkResult | None = None
        self._rendered_benchmark_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("CONCURRENCY RESULT", classes="section-title")
        yield Static("", id="concurrency-result-summary")
        yield DataTable(id="concurrency-scaling-table", zebra_stripes=True)
        yield PercentileSummary(id="concurrency-result-percentiles")
        yield Static("", id="concurrency-result-hardware")
        yield Static("", id="concurrency-result-observations")
        yield RequestTable(id="concurrency-result-requests")
        with Horizontal(id="concurrency-result-actions"):
            yield Button("Run Again", id="concurrency-run-again", variant="primary")
            yield Button("Edit Configuration", id="concurrency-edit")

    def on_mount(self) -> None:
        table = cast(
            DataTable[str],
            self.query_one("#concurrency-scaling-table", DataTable),
        )
        for column in (
            "CONC",
            "SUCCESS",
            "REQ/S",
            "TOK/S",
            "TTFT P50",
            "TTFT P95",
            "LAT P50",
            "LAT P95",
            "REQ TOK/S",
            "GPU AVG",
        ):
            table.add_column(column, key=column)

    def update_result(self, result: ConcurrencyBenchmarkResult) -> None:
        self._result = result
        summary = Text()
        summary.append(result.status.value.upper(), style=f"{PRIMARY} bold")
        summary.append(
            f" · {result.server_name} · {result.model_id} · "
            f"{result.wall_time_seconds:.2f}s total"
        )
        if result.error:
            summary.append(f"\n{result.error}", style=ERROR)
        if any(level.partial for level in result.levels):
            summary.append("\nPartial measured results retained.", style=WARNING)
        highest = result.levels[-1] if result.levels else None
        if highest is not None:
            summary.append(
                f"\nHighest attempted CONC {highest.concurrency} · "
                f"{highest.successful_requests}/{highest.attempted_requests} success · "
                f"{highest.aggregate_output_tokens_per_second:.1f} aggregate tok/s · "
                f"{highest.requests_per_second:.2f} req/s"
            )
            summary.append(
                f"\nConfigured {highest.configured_requests} · "
                f"attempted {highest.attempted_requests} · "
                f"failed {highest.failed_requests} · "
                f"timeout {highest.timed_out_requests} · "
                f"cancelled {highest.cancelled_requests} · "
                f"{highest.total_completion_tokens} completion tokens · "
                f"{highest.token_count_mode.title()} counts"
            )
        self.query_one("#concurrency-result-summary", Static).update(summary)

        table = cast(
            DataTable[str],
            self.query_one("#concurrency-scaling-table", DataTable),
        )
        table.clear(columns=False)
        for level in result.levels:
            hardware = level.hardware_summary
            gpu = (
                hardware.average_gpu_utilisation_percent
                if hardware is not None
                else None
            )
            values = (
                str(level.concurrency),
                f"{level.successful_requests}/{level.attempted_requests}",
                f"{level.requests_per_second:.2f}",
                f"{level.aggregate_output_tokens_per_second:.1f}",
                "--" if level.ttft_ms.p50 is None else f"{level.ttft_ms.p50:.1f}",
                "--" if level.ttft_ms.p95 is None else f"{level.ttft_ms.p95:.1f}",
                "--"
                if level.latency_seconds.p50 is None
                else f"{level.latency_seconds.p50:.3f}",
                "--"
                if level.latency_seconds.p95 is None
                else f"{level.latency_seconds.p95:.3f}",
                "--"
                if level.request_output_tokens_per_second.median is None
                else f"{level.request_output_tokens_per_second.median:.1f}",
                format_percentage(gpu),
            )
            table.add_row(*values, key=str(level.concurrency))

        self.query_one(PercentileSummary).update_level(
            highest, requested_max_tokens=result.config.max_tokens
        )
        hardware_text = Text("LOCAL HARDWARE", style=f"{PRIMARY} bold")
        hardware_text.append(" · may not represent a remote model server\n")
        if highest is None or highest.hardware_summary is None:
            hardware_text.append("Hardware metrics unavailable")
        else:
            hardware = highest.hardware_summary
            maximum_gpu = hardware.maximum_gpu_utilisation_percent
            average_vram = hardware.average_vram_used_bytes
            maximum_gpu_label = format_percentage(
                float(maximum_gpu) if maximum_gpu is not None else None
            )
            average_vram_label = format_bytes(
                int(average_vram) if average_vram is not None else None
            )
            hardware_text.append(
                f"{hardware.sample_count} samples · GPU average "
                f"{format_percentage(hardware.average_gpu_utilisation_percent)} · "
                f"GPU max "
                f"{maximum_gpu_label} · "
                f"VRAM average "
                f"{average_vram_label} · "
                f"CPU average "
                f"{format_percentage(hardware.average_cpu_utilisation_percent)}"
            )
        self.query_one("#concurrency-result-hardware", Static).update(hardware_text)

        observations = Text("OBSERVATIONS", style=f"{PRIMARY} bold")
        if result.observations:
            for observation in result.observations:
                observations.append(f"\n· {observation.message}")
        else:
            observations.append("\nNo cross-level scaling observation available.")
        for warning in result.warnings:
            observations.append(f"\nWarning: {warning}", style=WARNING)
        self.query_one("#concurrency-result-observations", Static).update(observations)

        request_table = self.query_one("#concurrency-result-requests", RequestTable)
        if self._rendered_benchmark_id != result.benchmark_id:
            request_table.reset_requests()
            self._rendered_benchmark_id = result.benchmark_id
        request_table.update_requests(
            request for level in result.levels for request in level.requests
        )

    def focus_actions(self) -> None:
        self.query_one("#concurrency-run-again", Button).focus()

    @on(Button.Pressed, "#concurrency-run-again")
    def run_again(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ConcurrencyBenchmarkRunAgainRequested())

    @on(Button.Pressed, "#concurrency-edit")
    def edit_configuration(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ConcurrencyBenchmarkEditRequested())
