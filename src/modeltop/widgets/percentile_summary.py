"""Compact percentile and output-length summary."""

from rich.text import Text
from textual.widgets import Static

from modeltop.benchmarks.models import ConcurrencyLevelResult
from modeltop.theme import PRIMARY


class PercentileSummary(Static):
    """Render percentiles for the current highest completed level."""

    DEFAULT_CSS = """
    PercentileSummary {
        width: 1fr;
        height: auto;
        border: solid $border-blurred;
        padding: 0 1;
    }
    """

    @staticmethod
    def _value(value: float | None, suffix: str = "") -> str:
        return "--" if value is None else f"{value:.1f}{suffix}"

    def update_level(
        self, level: ConcurrencyLevelResult | None, *, requested_max_tokens: int
    ) -> None:
        if level is None:
            self.update("Percentiles unavailable")
            return
        mode = level.token_count_mode.title()
        content = Text()
        content.append(
            f"PERCENTILES · CONC {level.concurrency}\n", style=f"{PRIMARY} bold"
        )
        content.append(
            "TTFT ms  "
            f"p50 {self._value(level.ttft_ms.p50)}  "
            f"p90 {self._value(level.ttft_ms.p90)}  "
            f"p95 {self._value(level.ttft_ms.p95)}  "
            f"p99 {self._value(level.ttft_ms.p99)}\n"
        )
        content.append(
            "Latency s  "
            f"p50 {self._value(level.latency_seconds.p50)}  "
            f"p90 {self._value(level.latency_seconds.p90)}  "
            f"p95 {self._value(level.latency_seconds.p95)}  "
            f"p99 {self._value(level.latency_seconds.p99)}\n"
        )
        content.append(
            "Request tok/s  "
            f"mean {self._value(level.request_output_tokens_per_second.mean)}  "
            f"median {self._value(level.request_output_tokens_per_second.median)}\n"
        )
        content.append(
            f"Output requested max {requested_max_tokens} · actual mean "
            f"{self._value(level.completion_tokens.mean)} / min "
            f"{self._value(level.completion_tokens.minimum)} / max "
            f"{self._value(level.completion_tokens.maximum)} · {mode} counts"
        )
        self.update(content)
