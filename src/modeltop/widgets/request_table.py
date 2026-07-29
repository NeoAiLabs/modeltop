"""Incrementally rendered per-request Concurrency benchmark table."""

from collections.abc import Iterable

from textual.widgets import DataTable

from modeltop.benchmarks.models import (
    ConcurrencyRequestProgress,
    ConcurrencyRequestResult,
)

RequestRow = ConcurrencyRequestProgress | ConcurrencyRequestResult


class RequestTable(DataTable[str]):
    """Retain every request row and update only cells whose values changed."""

    COLUMNS = (
        "ID",
        "CONC",
        "STATE",
        "TTFT",
        "LATENCY",
        "TOKENS",
        "TOK/S",
        "FINISH",
        "ERROR",
    )

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, zebra_stripes=True, cursor_type="row")
        self._rendered: dict[str, tuple[str, ...]] = {}

    def on_mount(self) -> None:
        for column in self.COLUMNS:
            self.add_column(column, key=column)

    @staticmethod
    def _compact_id(row: RequestRow) -> str:
        return f"c{row.concurrency_level}-r{row.sequence_number:04d}"

    @staticmethod
    def _truncate(text: str | None, limit: int = 48) -> str:
        if not text:
            return "--"
        safe = " ".join(text.split())
        return safe if len(safe) <= limit else f"{safe[: limit - 1]}…"

    @staticmethod
    def _render_row(row: RequestRow) -> tuple[str, ...]:
        if isinstance(row, ConcurrencyRequestProgress):
            metrics = row.latest_metrics
            estimated = bool(
                metrics is not None and metrics.completion_tokens_estimated
            )
            tokens = metrics.completion_tokens if metrics is not None else None
            speed = metrics.output_tokens_per_second if metrics is not None else None
            ttft = metrics.ttft_ms if metrics is not None else None
            latency = metrics.total_duration_s if metrics is not None else None
            finish = metrics.finish_reason if metrics is not None else None
            state = row.state.upper()
            error = row.error
        else:
            estimated = row.completion_tokens_estimated
            tokens = row.completion_tokens
            speed = row.output_tokens_per_second
            ttft = row.ttft_ms
            latency = row.total_latency_seconds
            finish = row.finish_reason
            state = (
                "DONE"
                if row.success
                else "TIMEOUT"
                if row.timed_out
                else "CANCELLED"
                if row.cancelled
                else "ERROR"
            )
            error = row.error_message
        prefix = "~" if estimated else ""
        return (
            RequestTable._compact_id(row),
            str(row.concurrency_level),
            state,
            "--" if ttft is None else f"{ttft:.1f} ms",
            "--" if latency is None else f"{latency:.3f} s",
            "--" if tokens is None else f"{prefix}{tokens}",
            "--" if speed is None else f"{prefix}{speed:.1f}",
            finish or "--",
            RequestTable._truncate(error),
        )

    def reset_requests(self) -> None:
        """Clear rows only when switching to a different terminal benchmark."""
        self.clear(columns=False)
        self._rendered.clear()

    def update_requests(self, requests: Iterable[RequestRow]) -> None:
        """Append unseen IDs and update changed cells without clearing rows."""
        was_at_bottom = self.is_vertical_scroll_end
        rows_appended = False
        for request in requests:
            key = request.request_id
            rendered = self._render_row(request)
            previous = self._rendered.get(key)
            if previous is None:
                self.add_row(*rendered, key=key)
                rows_appended = True
            elif previous != rendered:
                for column, old, new in zip(
                    self.COLUMNS, previous, rendered, strict=True
                ):
                    if old != new:
                        self.update_cell(key, column, new)
            self._rendered[key] = rendered
        if rows_appended and was_at_bottom:
            self.anchor()
