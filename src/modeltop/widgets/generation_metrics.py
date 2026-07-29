"""Compact exact-versus-estimated generation metrics."""

from textual.widgets import Static

from modeltop.chat.models import GenerationMetrics
from modeltop.state import ApplicationState


def _count(value: int | None, estimated: bool) -> str:
    if value is None:
        return "--"
    return f"{'~' if estimated else ''}{value}"


def _duration(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}s"


def _milliseconds(value: float | None) -> str:
    return "--" if value is None else f"{value:.0f}ms"


def _speed(value: float | None, estimated: bool) -> str:
    if value is None:
        return "--"
    return f"{'~' if estimated else ''}{value:.1f} tok/s"


class GenerationMetricsView(Static):
    """Render live and terminal generation metrics without fabricated values."""

    DEFAULT_CSS = """
    GenerationMetricsView {
        display: none;
        width: 1fr;
        height: 3;
        border: solid $border-blurred;
        background: $catppuccin-base;
        padding: 0 1;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    GenerationMetricsView.visible { display: block; }
    """

    def update_state(self, state: ApplicationState) -> None:
        metrics = state.generation_metrics
        self.set_class(metrics is not None, "visible")
        if metrics is None:
            self.update("")
            return
        status = state.generation_status.value.upper()
        if state.generation_error:
            status = f"ERROR · {state.generation_error}"
        elif metrics.cancelled:
            status = "CANCELLED"
        elif state.generation_notice:
            status = f"{status} · NON-STREAM FALLBACK"
        self.update(self._format_metrics(status, metrics))

    @staticmethod
    def _format_metrics(status: str, metrics: GenerationMetrics) -> str:
        output_tokens = _count(
            metrics.completion_tokens,
            metrics.completion_tokens_estimated,
        )
        output_speed = _speed(
            metrics.output_tokens_per_second,
            metrics.completion_tokens_estimated,
        )
        counts = (
            f"P {_count(metrics.prompt_tokens, metrics.prompt_tokens_estimated)} · "
            f"O {output_tokens} · "
            f"T {_count(metrics.total_tokens, metrics.total_tokens_estimated)}"
        )
        first = (
            f"{status}  ELAPSED {_duration(metrics.total_duration_s)} · "
            f"TTFT {_milliseconds(metrics.ttft_ms)} · {counts}"
        )
        second = (
            f"GEN {_duration(metrics.active_generation_duration_s)} · "
            f"TOTAL {_duration(metrics.total_duration_s)} · "
            f"{output_speed} · "
            f"ITL {_milliseconds(metrics.inter_token_latency_ms)} · "
            f"FINISH {metrics.finish_reason or '--'}"
        )
        return f"{first}\n{second}"
