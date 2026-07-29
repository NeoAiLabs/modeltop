"""Compact stable-row Context benchmark progress rendering."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from modeltop.state import ApplicationState


class ContextProgressPanel(VerticalScroll):
    """Render compact immutable progress without prompt or generated content."""

    DEFAULT_CSS = """
    ContextProgressPanel { width: 1fr; height: 1fr; padding: 0 1; }
    ContextProgressPanel .section-title { height: 1; color: #5da9e9; text-style: bold; }
    ContextProgressPanel .progress-row { height: auto; min-height: 1; }
    ContextProgressPanel #context-progress-warning { color: #e5c07b; }
    """

    def compose(self) -> ComposeResult:
        yield Static("CONTEXT BENCHMARK PROGRESS", classes="section-title")
        yield Static("Status: --", id="context-progress-status", classes="progress-row")
        yield Static("Prompt: --", id="context-progress-build", classes="progress-row")
        yield Static("Target: --", id="context-progress-target", classes="progress-row")
        yield Static("Probe: --", id="context-progress-probe", classes="progress-row")
        yield Static("Run: --", id="context-progress-run", classes="progress-row")
        yield Static(
            "Request: --", id="context-progress-request", classes="progress-row"
        )
        yield Static(
            "Metrics: --", id="context-progress-metrics", classes="progress-row"
        )
        yield Static("Delay: --", id="context-progress-delay", classes="progress-row")
        yield Static(
            "Hardware: --", id="context-progress-hardware", classes="progress-row"
        )
        yield Static("", id="context-progress-warning", classes="progress-row")

    def update_state(self, state: ApplicationState) -> None:
        lane = state.context_benchmark
        progress = lane.progress
        self.query_one("#context-progress-status", Static).update(
            f"Status: {lane.status.value.replace('_', ' ').title()}"
        )
        if progress is None:
            return
        build = progress.build
        self.query_one("#context-progress-build", Static).update(
            "Prompt: --"
            if build is None
            else (
                f"Building prompt: {build.measured_count} / {build.target_length} "
                f"{build.context_unit} · fragments {build.fragment_count} · "
                f"iteration {build.iteration} · {build.percentage:.1f}%"
            )
        )
        self.query_one("#context-progress-target", Static).update(
            f"Target: {progress.active_target_length or '--'} · "
            f"{progress.target_index}/{progress.target_count or '--'} · "
            f"next {progress.next_target_length or '--'}"
        )
        bounds = progress.probe_bounds
        self.query_one("#context-progress-probe", Static).update(
            "Probe: --"
            if bounds is None
            else (
                f"Probe {bounds.stage} · success "
                f"{bounds.highest_confirmed_success or '--'} · "
                f"rejection {bounds.first_confirmed_rejection or '--'} · "
                f"resolution {bounds.resolution_tokens}"
            )
        )
        self.query_one("#context-progress-run", Static).update(
            f"Run: {progress.run_number}/{progress.configured_runs or '--'} · "
            f"completed lengths {len(progress.completed_lengths)}"
        )
        request = progress.active_request
        self.query_one("#context-progress-request", Static).update(
            "Request: --"
            if request is None
            else (
                f"Request: {request.state} · target {request.target_length} · "
                f"run {request.run_number} · output "
                f"{request.response_character_count} chars"
            )
        )
        metrics = request.latest_metrics if request is not None else None
        self.query_one("#context-progress-metrics", Static).update(
            "Metrics: --"
            if metrics is None
            else (
                f"TTFT {metrics.ttft_ms if metrics.ttft_ms is not None else '--'} ms · "
                f"output {metrics.completion_tokens or 0} tok · "
                f"{metrics.output_tokens_per_second or 0:.1f} tok/s"
            )
        )
        self.query_one("#context-progress-delay", Static).update(
            "Delay: --"
            if progress.delay_remaining_seconds is None
            else f"Delay: {progress.delay_remaining_seconds:.1f}s"
        )
        hardware = progress.cached_hardware
        self.query_one("#context-progress-hardware", Static).update(
            "Hardware: unavailable"
            if hardware is None
            else f"Hardware: cached {hardware.collected_at.isoformat()}"
        )
        self.query_one("#context-progress-warning", Static).update(
            "\n".join(progress.warnings)
        )
