"""Payload-free progress rendering for r0b0bench."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from modeltop.benchmarks.r0b0bench_contract import r0b0bench_ordered_selection
from modeltop.state import ApplicationState


class R0b0benchProgressPanel(VerticalScroll):
    """Show lane progress, bounded outcomes, and cached local hardware."""

    DEFAULT_CSS = """
    R0b0benchProgressPanel { width: 1fr; height: 1fr; padding: 0 1; }
    R0b0benchProgressPanel .section-title {
        height: 1; color: $primary; text-style: bold;
    }
    R0b0benchProgressPanel .progress-row { height: auto; min-height: 1; }
    R0b0benchProgressPanel #r0b0bench-progress-cancel { color: $warning; }
    """

    def compose(self) -> ComposeResult:
        yield Static("R0B0BENCH PROGRESS", classes="section-title")
        for identifier, label in (
            ("status", "Status: --"),
            ("profile", "Profile: --"),
            ("model", "Model: --"),
            ("count", "Tests: --"),
            ("outcomes", "Outcomes: --"),
            ("current", "Current: --"),
            ("elapsed", "Elapsed: --"),
            ("hardware", "Hardware: --"),
        ):
            yield Static(
                label,
                id=f"r0b0bench-progress-{identifier}",
                classes="progress-row",
            )
        yield Static(
            "Esc cancels", id="r0b0bench-progress-cancel", classes="progress-row"
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.r0b0bench_benchmark
        progress = lane.progress
        self.query_one("#r0b0bench-progress-status", Static).update(
            f"Status: {lane.status.value.replace('_', ' ').title()}"
        )
        self.query_one("#r0b0bench-progress-profile", Static).update(
            f"Profile: {lane.config.profile} · "
            f"{len(lane.config.selected_lanes)} selected"
        )
        self.query_one("#r0b0bench-progress-model", Static).update(
            f"Model: {state.selected_model_id or '--'}"
        )
        if progress is None:
            return
        self.query_one("#r0b0bench-progress-count", Static).update(
            f"Tests: {progress.completed_count}/{progress.configured_count} completed"
        )
        self.query_one("#r0b0bench-progress-outcomes", Static).update(
            f"Outcomes: pass {progress.pass_count} · fail {progress.fail_count} · "
            f"skip {progress.skip_count} · error {progress.error_count} · "
            f"not implemented {progress.not_implemented_count}"
        )
        current = progress.current_lane
        ordered = r0b0bench_ordered_selection(
            lane.config.profile, lane.config.selected_lanes
        )
        current_text = "Current: --"
        if current is not None:
            current_text = (
                f"Current: {ordered.index(current) + 1}/{len(ordered)} · {current}"
            )
        self.query_one("#r0b0bench-progress-current", Static).update(current_text)
        self.query_one("#r0b0bench-progress-elapsed", Static).update(
            f"Elapsed: {progress.elapsed_seconds:.1f}s"
        )
        hardware = progress.cached_hardware
        self.query_one("#r0b0bench-progress-hardware", Static).update(
            "Hardware: unavailable"
            if hardware is None
            else f"Hardware: cached {hardware.collected_at.isoformat()}"
        )
