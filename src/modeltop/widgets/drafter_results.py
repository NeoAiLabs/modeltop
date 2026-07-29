"""Terminal Drafter benchmark result rendering."""

from typing import cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Static

from modeltop.benchmarks.models import (
    DrafterBenchmarkResult,
    DrafterRunResult,
    MetricStatistics,
)
from modeltop.messages import (
    DrafterBenchmarkEditRequested,
    DrafterBenchmarkRunAgainRequested,
)


def _value(value: float | int | None, *, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def _mean(stats: MetricStatistics, *, suffix: str = "", digits: int = 1) -> str:
    return _value(stats.mean, suffix=suffix, digits=digits)


class DrafterResultsPanel(VerticalScroll):
    """Render one latest-only Drafter result without export or history."""

    DEFAULT_CSS = """
    DrafterResultsPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: $catppuccin-base;
    }
    DrafterResultsPanel .section-title {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    DrafterResultsPanel #drafter-result-summary { height: auto; }
    DrafterResultsPanel #drafter-result-headline { height: auto; }
    DrafterResultsPanel #drafter-result-observations { height: auto; }
    DrafterResultsPanel DataTable { height: auto; min-height: 5; max-height: 18; }
    DrafterResultsPanel #drafter-result-actions { height: 3; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._result: DrafterBenchmarkResult | None = None

    def compose(self) -> ComposeResult:
        yield Static("DRAFTER RESULT", classes="section-title")
        yield Static("", id="drafter-result-summary", markup=False)
        yield Static("", id="drafter-result-headline", markup=False)
        yield Static("MEASURED RUNS", classes="section-title")
        yield DataTable(id="drafter-result-runs", zebra_stripes=True)
        yield Static("OBSERVATIONS", classes="section-title")
        yield Static("", id="drafter-result-observations", markup=False)
        yield Horizontal(
            Button("Run Again", id="drafter-run-again", variant="primary"),
            Button("Edit", id="drafter-edit"),
            id="drafter-result-actions",
        )

    def on_mount(self) -> None:
        table = cast(
            DataTable[str],
            self.query_one("#drafter-result-runs", DataTable),
        )
        table.cursor_type = "row"
        for key, label in (
            ("run", "RUN"),
            ("ok", "OK"),
            ("ttft", "TTFT"),
            ("toks", "TOK/S"),
            ("draft", "DRAFT"),
            ("acc", "ACC"),
            ("rate", "RATE"),
            ("comp", "COMP TOK"),
        ):
            table.add_column(label, key=key)

    def update_result(self, result: DrafterBenchmarkResult) -> None:
        self._result = result
        wall = (result.completed_at - result.started_at).total_seconds()
        summary = [
            f"Status: {result.status.value.replace('_', ' ').upper()} · "
            f"{result.server_name} · {result.model_id}",
            f"Wall time: {wall:.1f}s · "
            f"Successful: {result.successful_runs}/{result.measured_runs} measured · "
            f"Failed: {result.failed_runs} · Cancelled: {result.cancelled_runs}",
        ]
        if result.error:
            summary.append(result.error)
        self.query_one("#drafter-result-summary", Static).update("\n".join(summary))

        acceptance = (
            "UNAVAILABLE"
            if result.acceptance_rate.mean is None
            else f"{result.acceptance_rate.mean:.2f}"
        )
        headline = (
            f"Mean TTFT: {_mean(result.ttft_ms, suffix=' ms')} · "
            f"Mean tok/s: {_mean(result.output_tokens_per_second, suffix=' tok/s')} · "
            f"Mean acceptance: {acceptance}\n"
            f"Mean draft tokens: {_mean(result.draft_tokens, digits=1)} · "
            f"Mean accepted tokens: {_mean(result.accepted_tokens, digits=1)}"
        )
        self.query_one("#drafter-result-headline", Static).update(headline)

        table = cast(
            DataTable[str],
            self.query_one("#drafter-result-runs", DataTable),
        )
        table.clear()
        for run in result.run_results:
            if run.warmup:
                continue
            table.add_row(
                str(run.run_number),
                self._ok_label(run),
                _value(run.ttft_ms, suffix=" ms"),
                _value(run.output_tokens_per_second, suffix=" tok/s"),
                _value(run.draft_tokens),
                _value(run.accepted_tokens),
                _value(run.acceptance_rate, digits=2),
                _value(run.completion_tokens),
                key=f"run-{run.run_number}",
            )

        if result.observations:
            observations = "\n".join(
                f"· {observation.message}" for observation in result.observations
            )
        else:
            observations = "None"
        self.query_one("#drafter-result-observations", Static).update(observations)

    def focus_actions(self) -> None:
        self.query_one("#drafter-run-again", Button).focus()

    @on(Button.Pressed, "#drafter-run-again")
    def run_again(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(DrafterBenchmarkRunAgainRequested())

    @on(Button.Pressed, "#drafter-edit")
    def edit(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(DrafterBenchmarkEditRequested())

    @staticmethod
    def _ok_label(run: DrafterRunResult) -> str:
        if run.success:
            return "Y"
        if run.cancelled:
            return "C"
        return "N"
