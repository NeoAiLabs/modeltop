"""Reusable terminal Speed Test result detail panel."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Static

from modeltop.benchmarks.models import (
    MetricStatistics,
    SpeedTestResult,
    SpeedTestRunResult,
)
from modeltop.hardware.models import HardwareSnapshot, summarize_gpus
from modeltop.messages import (
    SpeedTestCopySummaryRequested,
    SpeedTestExportRequested,
    SpeedTestRunAgainRequested,
)


class SpeedTestResultsPanel(VerticalScroll):
    """Render complete or partial result metadata, aggregates, and request rows."""

    DEFAULT_CSS = """
    SpeedTestResultsPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: $catppuccin-base;
        overflow-x: hidden;
    }
    SpeedTestResultsPanel .result-heading {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    SpeedTestResultsPanel #result-status,
    SpeedTestResultsPanel #result-meta,
    SpeedTestResultsPanel #result-config,
    SpeedTestResultsPanel #result-hardware,
    SpeedTestResultsPanel #result-summary,
    SpeedTestResultsPanel #result-runs {
        height: auto;
        min-height: 1;
    }
    SpeedTestResultsPanel #result-actions { height: 3; }
    SpeedTestResultsPanel #result-actions Button { margin-right: 1; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._result: SpeedTestResult | None = None

    def compose(self) -> ComposeResult:
        yield Static("SPEED TEST RESULT", classes="result-heading")
        yield Static("No result selected.", id="result-status", markup=False)
        yield Static("", id="result-meta", markup=False)
        yield Static("CONFIGURATION", classes="result-heading")
        yield Static("", id="result-config", markup=False)
        yield Static("AGGREGATES · SUCCESSFUL MEASURED RUNS", classes="result-heading")
        yield Static("", id="result-summary", markup=False)
        yield Static("INDIVIDUAL REQUESTS", classes="result-heading")
        yield Static("", id="result-runs", markup=False)
        yield Static("HARDWARE SNAPSHOTS", classes="result-heading")
        yield Static("", id="result-hardware", markup=False)
        with Horizontal(id="result-actions"):
            yield Button("Run Again", id="result-run-again", variant="primary")
            yield Button("Export JSON", id="result-export")
            yield Button("Copy Summary", id="result-copy")
        yield Static("Esc Back", markup=False)

    @property
    def result(self) -> SpeedTestResult | None:
        return self._result

    def update_result(self, result: SpeedTestResult) -> None:
        self._result = result
        attempted = result.attempted_measured_runs
        self.query_one("#result-status", Static).update(
            f"{result.status.value.upper().replace('_', ' ')} · "
            f"{result.successful_runs}/{result.measured_runs} successful · "
            f"{attempted}/{result.measured_runs} attempted · "
            f"{result.failed_runs} failed · {result.cancelled_runs} cancelled"
        )
        self.query_one("#result-meta", Static).update(
            f"RUN {result.run_id}\n"
            f"STARTED {result.started_at.isoformat()} · "
            f"COMPLETED {result.completed_at.isoformat()}\n"
            f"SERVER {result.server_name} ({result.server_id}) · "
            f"ENDPOINT {result.server_endpoint} · MODEL {result.model_id} · "
            f"BACKEND {result.backend}"
        )
        config = result.config
        seed = "--" if config.seed is None else str(config.seed)
        thinking = (
            "DISABLED" if config.thinking_mode == "disabled" else "SERVER DEFAULT"
        )
        self.query_one("#result-config", Static).update(
            f"PRESET {config.preset.upper()} · WARM-UPS {config.warmup_runs} · "
            f"RUNS {config.measured_runs} · MAX TOKENS {config.max_tokens}\n"
            f"TEMPERATURE {config.temperature:g} · TOP-P {config.top_p:g} · "
            f"SEED {seed} · TIMEOUT {config.request_timeout_seconds:g}s · "
            f"THINKING {thinking} · "
            f"CONTINUE ON ERROR {'YES' if config.continue_on_error else 'NO'}"
        )
        estimated = result.estimated_measured_metrics
        summary = "\n".join(
            (
                self._stats_row(
                    "Output speed", result.output_tokens_per_second, "tok/s", estimated
                ),
                self._stats_row("TTFT", result.ttft_ms, "ms"),
                self._stats_row("Total duration", result.total_duration_s, "s"),
                self._stats_row(
                    "Generation duration", result.generation_duration_s, "s"
                ),
                self._stats_row("Prompt tokens", result.prompt_tokens, "", estimated),
                self._stats_row(
                    "Completion tokens", result.completion_tokens, "", estimated
                ),
            )
        )
        self.query_one("#result-summary", Static).update(summary)
        rows = "\n".join(self._run_row(run) for run in result.run_results)
        self.query_one("#result-runs", Static).update(rows or "No requests attempted.")
        self.query_one("#result-hardware", Static).update(
            f"BEFORE {self._hardware_label(result.hardware_before)}\n"
            f"AFTER  {self._hardware_label(result.hardware_after)}"
        )

    def focus_actions(self) -> None:
        self.query_one("#result-run-again", Button).focus()

    @on(Button.Pressed, "#result-run-again")
    def run_again(self, event: Button.Pressed) -> None:
        event.stop()
        if self._result is not None:
            self.post_message(SpeedTestRunAgainRequested(self._result.run_id))

    @on(Button.Pressed, "#result-export")
    def export(self, event: Button.Pressed) -> None:
        event.stop()
        if self._result is not None:
            self.post_message(SpeedTestExportRequested(self._result.run_id))

    @on(Button.Pressed, "#result-copy")
    def copy_summary(self, event: Button.Pressed) -> None:
        event.stop()
        if self._result is not None:
            self.post_message(SpeedTestCopySummaryRequested(self._result.run_id))

    @staticmethod
    def _stats_row(
        label: str,
        stats: MetricStatistics,
        unit: str,
        estimated: bool = False,
    ) -> str:
        prefix = "~" if estimated and stats.count else ""

        def value(number: float | None) -> str:
            if number is None:
                return "--"
            suffix = f" {unit}" if unit else ""
            return f"{prefix}{number:.2f}{suffix}"

        return (
            f"{label}: N {stats.count} · MEAN {value(stats.mean)} · "
            f"MEDIAN {value(stats.median)} · MIN {value(stats.minimum)} · "
            f"MAX {value(stats.maximum)} · P95 {value(stats.p95)} · "
            f"SD {value(stats.standard_deviation)}"
        )

    @staticmethod
    def _run_row(run: SpeedTestRunResult) -> str:
        phase = "Warm-up" if run.warmup else "Run"
        state = "CANCELLED" if run.cancelled else "OK" if run.success else "ERROR"
        estimate = "~" if run.tokens_estimated else ""
        ttft = "--" if run.ttft_ms is None else f"{run.ttft_ms:.1f}ms"
        speed = (
            "--"
            if run.output_tokens_per_second is None
            else f"{estimate}{run.output_tokens_per_second:.1f}tok/s"
        )
        tokens = (
            "--"
            if run.completion_tokens is None
            else f"{estimate}{run.completion_tokens}"
        )
        duration = (
            "--" if run.total_duration_s is None else f"{run.total_duration_s:.2f}s"
        )
        finish = run.finish_reason or "--"
        error = f" · {run.error}" if run.error else ""
        return (
            f"{phase} {run.run_number} · {state} · TTFT {ttft} · SPEED {speed} · "
            f"TOKENS {tokens} · TOTAL {duration} · FINISH {finish} · "
            f"CHARS {run.response_character_count}{error}"
        )

    @staticmethod
    def _hardware_label(snapshot: HardwareSnapshot | None) -> str:
        if snapshot is None:
            return "-- (unavailable)"
        gpu = summarize_gpus(snapshot.gpus).display_name
        error = f" · {snapshot.error}" if snapshot.error else ""
        return (
            f"{snapshot.collected_at.isoformat()} · {gpu} · "
            f"{snapshot.provider_name}{error}"
        )
