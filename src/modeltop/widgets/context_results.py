"""Session-only Context benchmark comparison and retrieval results."""

from typing import cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Static

from modeltop.benchmarks.models import ContextBenchmarkResult
from modeltop.messages import (
    ContextBenchmarkEditRequested,
    ContextBenchmarkRunAgainRequested,
)


def _value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


class ContextResultsPanel(VerticalScroll):
    """Render one latest in-memory Context result without export affordances."""

    DEFAULT_CSS = """
    ContextResultsPanel { width: 1fr; height: 1fr; padding: 0 1; }
    ContextResultsPanel .section-title { height: 1; color: $primary; text-style: bold; }
    ContextResultsPanel DataTable { height: auto; min-height: 5; max-height: 16; }
    ContextResultsPanel #context-result-summary { height: auto; }
    ContextResultsPanel #context-retrieval-detail { height: auto; }
    ContextResultsPanel #context-result-actions { height: 3; }
    """

    def compose(self) -> ComposeResult:
        yield Static("CONTEXT LENGTH RESULT", classes="section-title")
        yield Static("", id="context-result-summary", markup=False)
        yield DataTable(id="context-comparison-table", zebra_stripes=True)
        yield Static("RETRIEVAL DETAILS", classes="section-title")
        yield Static("", id="context-retrieval-detail", markup=False)
        yield Horizontal(
            Button("Run Again", id="context-run-again", variant="primary"),
            Button("Edit", id="context-edit"),
            id="context-result-actions",
        )

    def on_mount(self) -> None:
        table = cast(
            DataTable[str],
            self.query_one("#context-comparison-table", DataTable),
        )
        table.cursor_type = "row"
        for key, label in (
            ("target", "Target"),
            ("prompt", "Prompt tok"),
            ("accepted", "Accepted"),
            ("attempts", "Attempts"),
            ("success", "Success"),
            ("ttft", "TTFT med/p95"),
            ("input", "Estimated prefill ~tok/s"),
            ("output", "Output tok/s"),
            ("latency", "Latency med"),
            ("vram", "Max VRAM"),
            ("gpu", "Avg GPU"),
            ("retrieval", "Retrieval"),
        ):
            table.add_column(label, key=key)

    def update_result(self, result: ContextBenchmarkResult) -> None:
        table = cast(
            DataTable[str],
            self.query_one("#context-comparison-table", DataTable),
        )
        table.clear()
        details: list[str] = []
        for length in result.lengths:
            prompt_source = (
                "server"
                if any(
                    request.measurement.server_prompt_tokens is not None
                    for request in length.requests
                )
                else (
                    "estimated"
                    if any(request.measurement.estimated for request in length.requests)
                    else "local"
                )
            )
            accepted = (
                "REJECTED"
                if length.context_rejected_requests == length.configured_requests
                and length.configured_requests > 0
                else f"{length.accepted_requests}/{length.attempted_requests}"
            )
            retrieval_attempts = sum(
                value for _, value in length.retrieval_attempts_by_position
            )
            retrieval_successes = sum(
                value for _, value in length.retrieval_successes_by_position
            )
            hardware = length.hardware_summary
            table.add_row(
                f"{length.target_length} {length.context_unit}",
                f"{_value(length.prompt_tokens.median)} ({prompt_source})",
                accepted,
                f"{length.attempted_requests}/{length.configured_requests}",
                f"{length.success_rate_percent:.1f}%",
                f"{_value(length.ttft_ms.median)}/{_value(length.ttft_ms.p95)} ms",
                _value(length.estimated_input_tokens_per_second.median, " ~tok/s"),
                _value(length.output_tokens_per_second.median),
                _value(length.latency_seconds.median, "s"),
                _value(hardware.maximum_vram_used_bytes if hardware else None),
                _value(
                    hardware.average_gpu_utilisation_percent if hardware else None, "%"
                ),
                f"{retrieval_successes}/{retrieval_attempts}"
                if retrieval_attempts
                else "--",
                key=str(length.target_length),
            )
            for request in length.requests:
                for score in request.retrieval_results:
                    truncation = (
                        " [preview truncated]" if score.preview_truncated else ""
                    )
                    details.append(
                        f"{length.target_length} · {score.marker} · "
                        f"{score.position} · {score.status.upper()}{truncation}"
                    )
        bounds = result.probe_bounds
        provenance = sorted(
            {
                request.measurement.counter_name
                + (" (estimated)" if request.measurement.estimated else " (exact)")
                for length in result.lengths
                for request in length.requests
            }
        )
        summary = [
            f"Status: {result.status.value.upper()} · "
            f"Wall time: {result.wall_time_seconds:.1f}s",
            "Highest successful prompt: "
            f"{_value(result.highest_successful_prompt_tokens)} tokens",
            "First fully rejected prompt: "
            f"{_value(result.first_fully_rejected_prompt_tokens)} tokens",
            f"Token provenance: {', '.join(provenance) if provenance else '--'}",
        ]
        if bounds is not None:
            summary.append(
                "Probe: success "
                f"{_value(bounds.highest_confirmed_success)} · rejection "
                f"{_value(bounds.first_confirmed_rejection)} · resolution "
                f"{bounds.resolution_tokens} · {bounds.stage}"
            )
        if result.cancelled or any(length.partial for length in result.lengths):
            summary.append("Partial result retained; unstarted work is absent.")
        summary.extend(observation.message for observation in result.observations)
        summary.extend(result.warnings)
        self.query_one("#context-result-summary", Static).update("\n".join(summary))
        self.query_one("#context-retrieval-detail", Static).update(
            "\n".join(details) if details else "No retrieval rows."
        )

    def focus_actions(self) -> None:
        self.query_one("#context-run-again", Button).focus()

    @on(Button.Pressed, "#context-run-again")
    def run_again(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ContextBenchmarkRunAgainRequested())

    @on(Button.Pressed, "#context-edit")
    def edit(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(ContextBenchmarkEditRequested())
