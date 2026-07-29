"""Payload-free progress rendering for Tool Calling benchmarks."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from modeltop.state import ApplicationState


class ToolCallingProgressPanel(VerticalScroll):
    """Render official-suite coverage and cached local hardware only."""

    DEFAULT_CSS = """
    ToolCallingProgressPanel { width: 1fr; height: 1fr; padding: 0 1; }
    ToolCallingProgressPanel .section-title {
        height: 1; color: #5da9e9; text-style: bold;
    }
    ToolCallingProgressPanel .progress-row { height: auto; min-height: 1; }
    ToolCallingProgressPanel #tool-calling-progress-cancel { color: #e5c07b; }
    """

    def compose(self) -> ComposeResult:
        yield Static("TOOL CALLING PROGRESS", classes="section-title")
        yield Static(
            "Status: --", id="tool-calling-progress-status", classes="progress-row"
        )
        yield Static(
            "Suite: --", id="tool-calling-progress-suite", classes="progress-row"
        )
        yield Static(
            "Model: --", id="tool-calling-progress-model", classes="progress-row"
        )
        yield Static(
            "Scenarios: --", id="tool-calling-progress-count", classes="progress-row"
        )
        yield Static(
            "Outcomes: --", id="tool-calling-progress-outcomes", classes="progress-row"
        )
        yield Static(
            "Coverage: --", id="tool-calling-progress-coverage", classes="progress-row"
        )
        yield Static(
            "Current: --", id="tool-calling-progress-current", classes="progress-row"
        )
        yield Static(
            "Elapsed: --", id="tool-calling-progress-elapsed", classes="progress-row"
        )
        yield Static(
            "Hardware: --", id="tool-calling-progress-hardware", classes="progress-row"
        )
        yield Static(
            "Esc cancels", id="tool-calling-progress-cancel", classes="progress-row"
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.tool_calling_benchmark
        progress = lane.progress
        self.query_one("#tool-calling-progress-status", Static).update(
            f"Status: {lane.status.value.replace('_', ' ').title()}"
        )
        self.query_one("#tool-calling-progress-suite", Static).update(
            f"Suite: {lane.config.suite.title()} · "
            f"{lane.config.scenario_count} scenarios"
        )
        self.query_one("#tool-calling-progress-model", Static).update(
            f"Model: {state.selected_model_id or '--'}"
        )
        if progress is None:
            return
        self.query_one("#tool-calling-progress-count", Static).update(
            f"Scenarios: {progress.completed_count}/{progress.configured_count}"
        )
        self.query_one("#tool-calling-progress-outcomes", Static).update(
            f"Outcomes: pass {progress.pass_count} · "
            f"partial {progress.partial_count} · "
            f"fail {progress.fail_count} · excluded {progress.excluded_count}"
        )
        self.query_one("#tool-calling-progress-coverage", Static).update(
            f"Completion coverage: {progress.completion_rate_percent:.1f}% "
            f"({progress.gradable_count}/{progress.completed_count or '--'} completed)"
        )
        current = progress.current_scenario
        self.query_one("#tool-calling-progress-current", Static).update(
            "Current: --"
            if current is None
            else (
                f"Current: {current.source_index + 1}/{progress.configured_count} · "
                f"{current.scenario_id} · {current.category} {current.title}"
            )
        )
        self.query_one("#tool-calling-progress-elapsed", Static).update(
            f"Elapsed: {progress.elapsed_seconds:.1f}s"
        )
        hardware = progress.cached_hardware
        self.query_one("#tool-calling-progress-hardware", Static).update(
            "Hardware: unavailable"
            if hardware is None
            else f"Hardware: cached {hardware.collected_at.isoformat()}"
        )
