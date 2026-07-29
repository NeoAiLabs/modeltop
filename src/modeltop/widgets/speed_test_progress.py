"""Live sequential Speed Test progress panel."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import ProgressBar, Static

from modeltop.benchmarks.models import SpeedTestRunResult, SpeedTestState


class SpeedTestProgressPanel(VerticalScroll):
    """Render coalesced live metrics and immutable completed request rows."""

    DEFAULT_CSS = """
    SpeedTestProgressPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: $catppuccin-base;
        overflow-x: hidden;
    }
    SpeedTestProgressPanel .speed-heading {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    SpeedTestProgressPanel #speed-progress-bar { height: 1; }
    SpeedTestProgressPanel #speed-live-metrics { height: 2; }
    SpeedTestProgressPanel #speed-live-output {
        width: 1fr;
        height: 6;
        border: solid $border-blurred;
        background: $catppuccin-base;
        padding: 0 1;
    }
    SpeedTestProgressPanel #speed-live-output-text {
        width: 1fr;
        height: auto;
    }
    SpeedTestProgressPanel #speed-run-log { height: auto; min-height: 2; }
    SpeedTestProgressPanel #speed-cancel-hint { height: 1; color: $warning; }
    """

    def compose(self) -> ComposeResult:
        yield Static("SPEED TEST RUNNING", classes="speed-heading")
        yield Static("Preparing...", id="speed-phase", markup=False)
        yield ProgressBar(id="speed-progress-bar", show_eta=False)
        yield Static("", id="speed-live-metrics", markup=False)
        yield Static("LIVE OUTPUT", classes="speed-heading")
        with VerticalScroll(id="speed-live-output", can_focus=False):
            yield Static("", id="speed-live-output-text", markup=False)
        yield Static("COMPLETED REQUESTS", classes="speed-heading")
        yield Static("No requests completed yet.", id="speed-run-log", markup=False)
        yield Static("Esc Cancel", id="speed-cancel-hint", markup=False)

    def update_state(self, speed: SpeedTestState) -> None:
        phase = "PREPARING"
        if speed.current_phase == "warmup":
            phase = f"WARM-UP {speed.current_run}/{speed.phase_total}"
        elif speed.current_phase == "measured":
            phase = f"RUN {speed.current_run}/{speed.phase_total}"
        if speed.status.value == "cancelling":
            phase = "CANCELLING · closing active stream"
        self.query_one("#speed-phase", Static).update(phase)

        config = speed.config
        completed = len(speed.run_results)
        total = config.warmup_runs + config.measured_runs
        self.query_one("#speed-progress-bar", ProgressBar).update(
            total=total, progress=completed
        )
        metrics = speed.latest_metrics
        if metrics is None:
            metric_line = (
                f"TTFT -- · SPEED -- tok/s · PROMPT -- · "
                f"OUTPUT --/{config.max_tokens} · ELAPSED --"
            )
        else:
            estimated = metrics.completion_tokens_estimated
            completion = self._token(metrics.completion_tokens, estimated)
            prompt_tokens = self._token(
                metrics.prompt_tokens, metrics.prompt_tokens_estimated
            )
            metric_line = (
                f"TTFT {self._duration_ms(metrics.ttft_ms)} · "
                f"SPEED {self._speed(metrics.output_tokens_per_second, estimated)} · "
                f"PROMPT {prompt_tokens} "
                "· "
                f"OUTPUT {completion}/{config.max_tokens} · "
                f"ELAPSED {self._seconds(metrics.total_duration_s)}"
            )
        self.query_one("#speed-live-metrics", Static).update(metric_line)
        live_output = self.query_one("#speed-live-output", VerticalScroll)
        live_output_text = self.query_one("#speed-live-output-text", Static)
        if live_output.is_vertical_scroll_end:
            live_output.anchor()
        else:
            live_output.anchor(False)
        live_output_text.update(speed.live_output_preview or "Waiting for output...")
        rows = "\n".join(self._run_row(run) for run in speed.run_results)
        self.query_one("#speed-run-log", Static).update(
            rows or "No requests completed yet."
        )

    @classmethod
    def _run_row(cls, run: SpeedTestRunResult) -> str:
        label = f"Warm-up {run.run_number}" if run.warmup else f"Run {run.run_number}"
        if run.cancelled:
            state = "CANCELLED"
        elif run.success:
            state = "OK"
        else:
            state = f"ERROR {run.error or 'request failed'}"
        return (
            f"{label} · {state} · TTFT {cls._duration_ms(run.ttft_ms)} · "
            f"{cls._speed(run.output_tokens_per_second, run.tokens_estimated)} · "
            f"TOKENS {cls._token(run.completion_tokens, run.tokens_estimated)} · "
            f"TOTAL {cls._seconds(run.total_duration_s)}"
        )

    @staticmethod
    def _token(value: int | None, estimated: bool) -> str:
        if value is None:
            return "--"
        return f"{'~' if estimated else ''}{value}"

    @staticmethod
    def _speed(value: float | None, estimated: bool) -> str:
        if value is None:
            return "-- tok/s"
        return f"{'~' if estimated else ''}{value:.1f} tok/s"

    @staticmethod
    def _duration_ms(value: float | None) -> str:
        return "--" if value is None else f"{value:.1f} ms"

    @staticmethod
    def _seconds(value: float | None) -> str:
        return "--" if value is None else f"{value:.2f} s"
