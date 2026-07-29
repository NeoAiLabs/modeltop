"""Live sequential Drafter benchmark progress panel."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from modeltop.benchmarks.models import DrafterBenchmarkState


class DrafterProgressPanel(VerticalScroll):
    """Render phase, live generation metrics, and speculative telemetry."""

    DEFAULT_CSS = """
    DrafterProgressPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: #0f151d;
        overflow-x: hidden;
    }
    DrafterProgressPanel .section-title {
        height: 1;
        color: #5da9e9;
        text-style: bold;
    }
    DrafterProgressPanel .progress-row { height: auto; min-height: 1; }
    DrafterProgressPanel #drafter-progress-cancel { color: #e5c07b; }
    """

    def compose(self) -> ComposeResult:
        yield Static("DRAFTER PROGRESS", classes="section-title")
        yield Static("Status: --", id="drafter-progress-status", classes="progress-row")
        yield Static("Phase: --", id="drafter-progress-phase", classes="progress-row")
        yield Static("Run: --", id="drafter-progress-run", classes="progress-row")
        yield Static("TTFT: --", id="drafter-progress-ttft", classes="progress-row")
        yield Static("Output: --", id="drafter-progress-output", classes="progress-row")
        yield Static(
            "Draft tokens: --", id="drafter-progress-draft", classes="progress-row"
        )
        yield Static(
            "Accepted tokens: --",
            id="drafter-progress-accepted",
            classes="progress-row",
        )
        yield Static(
            "Acceptance rate: --", id="drafter-progress-rate", classes="progress-row"
        )
        yield Static(
            "Last error: --", id="drafter-progress-error", classes="progress-row"
        )
        yield Static(
            "Esc cancels", id="drafter-progress-cancel", classes="progress-row"
        )

    def update_state(self, state: DrafterBenchmarkState) -> None:
        progress = state.progress
        metrics = progress.latest_metrics if progress is not None else None

        status_label = state.status.value.replace("_", " ").upper()
        self.query_one("#drafter-progress-status", Static).update(
            f"Status: {status_label}"
        )

        if progress is None or progress.current_phase is None:
            phase_label = "--"
            run_label = "--"
        else:
            phase_label = progress.current_phase.upper()
            run_label = f"{progress.current_run}/{progress.phase_total}"
            if progress.configured_measured_runs:
                run_label = (
                    f"{run_label} · measured "
                    f"{progress.completed_measured_runs}/"
                    f"{progress.configured_measured_runs}"
                )
        self.query_one("#drafter-progress-phase", Static).update(
            f"Phase: {phase_label}"
        )
        self.query_one("#drafter-progress-run", Static).update(f"Run: {run_label}")

        ttft = (
            f"{metrics.ttft_ms:.1f} ms"
            if metrics is not None and metrics.ttft_ms is not None
            else "--"
        )
        if metrics is None:
            output = "--"
        else:
            tokens = (
                "--"
                if metrics.completion_tokens is None
                else f"{'~' if metrics.completion_tokens_estimated else ''}"
                f"{metrics.completion_tokens}"
            )
            speed = (
                "-- tok/s"
                if metrics.output_tokens_per_second is None
                else f"{'~' if metrics.completion_tokens_estimated else ''}"
                f"{metrics.output_tokens_per_second:.1f} tok/s"
            )
            output = f"{tokens} · {speed}"
        self.query_one("#drafter-progress-ttft", Static).update(f"TTFT: {ttft}")
        self.query_one("#drafter-progress-output", Static).update(f"Output: {output}")

        draft = (
            "--"
            if metrics is None or metrics.draft_tokens is None
            else str(metrics.draft_tokens)
        )
        accepted = (
            "--"
            if metrics is None or metrics.accepted_tokens is None
            else str(metrics.accepted_tokens)
        )
        rate = (
            "--"
            if metrics is None or metrics.acceptance_rate is None
            else f"{metrics.acceptance_rate:.2f}"
        )
        self.query_one("#drafter-progress-draft", Static).update(
            f"Draft tokens: {draft}"
        )
        self.query_one("#drafter-progress-accepted", Static).update(
            f"Accepted tokens: {accepted}"
        )
        self.query_one("#drafter-progress-rate", Static).update(
            f"Acceptance rate: {rate}"
        )

        error = "--"
        if progress is not None and progress.last_error:
            error = progress.last_error
        elif state.benchmark_error:
            error = state.benchmark_error
        self.query_one("#drafter-progress-error", Static).update(f"Last error: {error}")
