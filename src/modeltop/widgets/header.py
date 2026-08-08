"""Dashboard header with live server and model metrics."""

from typing import cast

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkStatus,
    ContextBenchmarkStatus,
    DrafterBenchmarkStatus,
    R0b0benchBenchmarkStatus,
    SpeedTestStatus,
    ToolCallingBenchmarkStatus,
)
from modeltop.chat.models import GenerationStatus
from modeltop.hardware.models import (
    format_byte_pair,
    format_percentage,
    format_temperature,
    summarize_gpus,
    truncate_device_name,
)
from modeltop.models import ServerConfig, format_backend_label
from modeltop.state import ApplicationState, ServerStatus
from modeltop.theme import CatppuccinPalette, CatppuccinTheme, palette_for


class HeaderBar(Vertical):
    """Six-row dashboard title and metric header."""

    DEFAULT_CSS = """
    HeaderBar {
        width: 1fr;
        height: 6;
        background: $surface;
        border: solid $border-blurred;
    }

    HeaderBar #header-title-row {
        width: 1fr;
        height: 1;
    }

    HeaderBar #header-title {
        width: 1fr;
        height: 1;
        color: $primary;
        text-style: bold;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    HeaderBar #header-subtitle {
        width: auto;
        height: 1;
        color: $catppuccin-muted;
        text-align: right;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    HeaderBar #header-metrics {
        width: 1fr;
        height: 3;
    }

    HeaderBar .metric {
        height: 3;
        content-align: center middle;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: clip;
        border-right: solid $border-blurred;
    }

    HeaderBar #metric-server {
        width: 2fr;
    }

    HeaderBar #metric-model {
        width: 4fr;
    }

    HeaderBar #metric-endpoint {
        width: 2fr;
    }

    HeaderBar #metric-gpu {
        width: 4fr;
    }

    HeaderBar #metric-vram {
        width: 3fr;
    }

    HeaderBar #metric-util,
    HeaderBar #metric-temp,
    HeaderBar #metric-latency {
        width: 1fr;
    }

    HeaderBar #metric-power,
    HeaderBar #metric-backend,
    HeaderBar #metric-status {
        width: 2fr;
    }

    HeaderBar #metric-status {
        border-right: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the title and ten live metric cells."""
        with Horizontal(id="header-title-row"):
            yield Static("MODELTOP", id="header-title")
            yield Static(
                "LOCAL HARDWARE",
                id="header-subtitle",
            )

        palette = palette_for(
            cast(CatppuccinTheme, self.app.theme)  # pyright: ignore[reportUnknownMemberType]
        )
        with Horizontal(id="header-metrics"):
            for label, identifier in (
                ("SERVER", "server"),
                ("MODEL", "model"),
                ("ENDPOINT", "endpoint"),
                ("GPU", "gpu"),
                ("VRAM", "vram"),
                ("UTIL", "util"),
                ("TEMP", "temp"),
                ("POWER", "power"),
                ("BACKEND", "backend"),
                ("LATENCY", "latency"),
                ("STATUS", "status"),
            ):
                yield Static(
                    self._metric(label, "--", palette, palette.text),
                    classes="metric",
                    id=f"metric-{identifier}",
                )

    @staticmethod
    def _metric(
        label: str,
        value: str,
        palette: CatppuccinPalette,
        value_color: str,
    ) -> Text:
        return Text.assemble((label, palette.primary), "\n", (value, value_color))

    def update_state(
        self,
        state: ApplicationState,
        server: ServerConfig,
    ) -> None:
        """Update existing metric cells from one coherent state snapshot."""
        subtitle = "LOCAL HARDWARE"
        tool_calling = state.tool_calling_benchmark
        r0b0bench = state.r0b0bench_benchmark
        drafter = state.drafter_benchmark
        concurrency = state.concurrency_benchmark
        context = state.context_benchmark
        speed = state.speed_test
        speed_label: str | None = None
        if tool_calling.is_active:
            speed_label = self._tool_calling_status(state)
        elif r0b0bench.is_active:
            speed_label = self._r0b0bench_status(state)
        elif drafter.is_active:
            speed_label = self._drafter_status(state)
        elif context.is_active:
            speed_label = self._context_status(state)
        elif concurrency.is_active:
            speed_label = self._concurrency_status(state)
        elif speed.is_active:
            speed_label = self._speed_status(
                speed.status, speed.current_run, speed.phase_total
            )
        elif state.active_view == "tool-calling":
            speed_label = self._tool_calling_status(state)
        elif state.active_view == "r0b0bench":
            speed_label = self._r0b0bench_status(state)
        elif state.active_view == "drafter":
            speed_label = self._drafter_status(state)
        elif state.active_view == "context":
            speed_label = self._context_status(state)
        elif state.active_view == "concurrency":
            speed_label = self._concurrency_status(state)
        elif state.active_view == "speed-test":
            speed_label = self._speed_status(speed.status, 0, 0)
        elif state.active_view == "results":
            selected = (
                speed.result_by_id(speed.selected_result_id)
                if speed.selected_result_id is not None
                else None
            )
            speed_label = (
                self._speed_status(selected.status, 0, 0)
                if selected is not None
                else "RESULTS"
            )
        elif state.active_view == "settings":
            speed_label = "SETTINGS"
        elif state.active_view == "chat":
            if state.generation_status is GenerationStatus.STARTING:
                speed_label = "CHAT STARTING"
            elif state.generation_status is GenerationStatus.STREAMING:
                speed_label = "CHAT GENERATING"
            else:
                speed_label = "CHAT READY"
        if speed_label is not None:
            subtitle = f"{subtitle} · {speed_label}"
        self.query_one("#header-subtitle", Static).update(subtitle)
        online = state.server_status is ServerStatus.ONLINE
        if online and state.selected_model_id is not None:
            model = state.selected_model_id
        elif online:
            model = "No models"
        else:
            model = "--"
        latency = (
            f"{state.connection_latency_ms:.0f} ms"
            if online and state.connection_latency_ms is not None
            else "--"
        )
        palette = palette_for(
            cast(CatppuccinTheme, self.app.theme)  # pyright: ignore[reportUnknownMemberType]
        )
        status_colors = {
            ServerStatus.CONNECTING: palette.warning,
            ServerStatus.ONLINE: palette.success,
            ServerStatus.OFFLINE: palette.error,
            ServerStatus.ERROR: palette.error,
        }
        snapshot = state.hardware_snapshot
        if snapshot is not None and snapshot.gpus:
            summary = summarize_gpus(snapshot.gpus)
            gpu = truncate_device_name(summary.display_name, 24)
            vram = format_byte_pair(
                summary.memory_used_bytes,
                summary.memory_total_bytes,
                compact=True,
            )
            utilisation = format_percentage(summary.utilisation_percent)
            temperature = format_temperature(summary.temperature_celsius)
            if summary.count > 1:
                if utilisation != "--":
                    utilisation = f"{utilisation} avg"
                if temperature != "--":
                    temperature = f"{temperature} max"
            power = self._power_pair(
                summary.power_draw_watts, summary.power_limit_watts
            )
        else:
            if snapshot is None and state.hardware_status.value == "initialising":
                gpu = "Initialising"
            elif state.hardware_last_error == "Hardware monitoring disabled":
                gpu = "Disabled"
            else:
                gpu = "Unavailable"
            vram = utilisation = temperature = power = "--"

        backend = server.backend_label
        if backend == "--" and state.selected_model_id is not None:
            selected_model = next(
                (
                    discovered
                    for discovered in state.available_models
                    if discovered.id == state.selected_model_id
                ),
                None,
            )
            if selected_model is not None:
                backend = format_backend_label(selected_model.owned_by)

        values = {
            "server": server.name,
            "model": model,
            "endpoint": server.endpoint_label,
            "gpu": gpu,
            "vram": vram,
            "util": utilisation,
            "temp": temperature,
            "power": power,
            "backend": backend,
            "latency": latency,
            "status": state.server_status.value.upper(),
        }
        for identifier, value in values.items():
            color = (
                status_colors[state.server_status]
                if identifier == "status"
                else palette.text
            )
            label = identifier.upper()
            self.query_one(f"#metric-{identifier}", Static).update(
                self._metric(label, value, palette, color)
            )

    @staticmethod
    def _speed_status(
        status: SpeedTestStatus, current_run: int, phase_total: int
    ) -> str:
        if status is SpeedTestStatus.WARMING_UP:
            return f"SPEED WARM-UP {current_run}/{phase_total}"
        if status is SpeedTestStatus.RUNNING:
            return f"SPEED RUN {current_run}/{phase_total}"
        labels = {
            SpeedTestStatus.IDLE: "SPEED READY",
            SpeedTestStatus.PREPARING: "SPEED READY",
            SpeedTestStatus.CANCELLING: "SPEED CANCELLING",
            SpeedTestStatus.COMPLETED: "SPEED COMPLETE",
            SpeedTestStatus.COMPLETED_WITH_ERRORS: "SPEED COMPLETE",
            SpeedTestStatus.CANCELLED: "SPEED CANCELLED",
            SpeedTestStatus.FAILED: "SPEED FAILED",
        }
        return labels[status]

    @staticmethod
    def _tool_calling_status(state: ApplicationState) -> str:
        lane = state.tool_calling_benchmark
        progress = lane.progress
        if lane.status is ToolCallingBenchmarkStatus.RUNNING:
            completed = progress.completed_count if progress is not None else 0
            return f"TOOL CALLING {completed}/{lane.config.scenario_count}"
        labels = {
            ToolCallingBenchmarkStatus.IDLE: "TOOL CALLING READY",
            ToolCallingBenchmarkStatus.VALIDATING: "TOOL CALLING VALIDATING",
            ToolCallingBenchmarkStatus.CANCELLING: "TOOL CALLING CANCELLING",
            ToolCallingBenchmarkStatus.CANCELLED: "TOOL CALLING CANCELLED",
            ToolCallingBenchmarkStatus.ERROR: "TOOL CALLING ERROR",
        }
        if lane.status is ToolCallingBenchmarkStatus.COMPLETED:
            score = (
                lane.latest_result.final_score
                if lane.latest_result is not None
                else None
            )
            return (
                "TOOL CALLING COMPLETE" if score is None else f"TOOL CALLING {score}%"
            )
        return labels[lane.status]

    @staticmethod
    def _r0b0bench_status(state: ApplicationState) -> str:
        lane = state.r0b0bench_benchmark
        progress = lane.progress
        if lane.status is R0b0benchBenchmarkStatus.RUNNING:
            completed = progress.completed_count if progress is not None else 0
            return f"R0B0BENCH {completed}/{len(lane.config.selected_lanes)}"
        labels = {
            R0b0benchBenchmarkStatus.IDLE: "R0B0BENCH READY",
            R0b0benchBenchmarkStatus.VALIDATING: "R0B0BENCH VALIDATING",
            R0b0benchBenchmarkStatus.CANCELLING: "R0B0BENCH CANCELLING",
            R0b0benchBenchmarkStatus.COMPLETED: "R0B0BENCH COMPLETE",
            R0b0benchBenchmarkStatus.COMPLETED_WITH_ERRORS: "R0B0BENCH ERRORS",
            R0b0benchBenchmarkStatus.CANCELLED: "R0B0BENCH CANCELLED",
            R0b0benchBenchmarkStatus.ERROR: "R0B0BENCH ERROR",
        }
        return labels[lane.status]

    @staticmethod
    def _drafter_status(state: ApplicationState) -> str:
        lane = state.drafter_benchmark
        progress = lane.progress
        current_run = progress.current_run if progress is not None else 0
        phase_total = progress.phase_total if progress is not None else 0
        if lane.status is DrafterBenchmarkStatus.WARMING_UP:
            return f"DRAFTER WARM-UP {current_run}/{phase_total}"
        if lane.status is DrafterBenchmarkStatus.RUNNING:
            return f"DRAFTER RUN {current_run}/{phase_total}"
        labels = {
            DrafterBenchmarkStatus.IDLE: "DRAFTER READY",
            DrafterBenchmarkStatus.PREPARING: "DRAFTER READY",
            DrafterBenchmarkStatus.CANCELLING: "DRAFTER CANCELLING",
            DrafterBenchmarkStatus.COMPLETED: "DRAFTER COMPLETE",
            DrafterBenchmarkStatus.COMPLETED_WITH_ERRORS: "DRAFTER COMPLETE",
            DrafterBenchmarkStatus.CANCELLED: "DRAFTER CANCELLED",
            DrafterBenchmarkStatus.FAILED: "DRAFTER FAILED",
        }
        return labels[lane.status]

    @staticmethod
    def _concurrency_status(state: ApplicationState) -> str:
        lane = state.concurrency_benchmark
        level = (
            lane.progress.active_concurrency_level
            if lane.progress is not None
            else None
        )
        suffix = "" if level is None else f" {level}"
        labels = {
            ConcurrencyBenchmarkStatus.IDLE: "BENCH READY",
            ConcurrencyBenchmarkStatus.VALIDATING: "BENCH READY",
            ConcurrencyBenchmarkStatus.WARMING_UP: f"BENCH WARM-UP{suffix}",
            ConcurrencyBenchmarkStatus.RUNNING: f"BENCH RUNNING{suffix}",
            ConcurrencyBenchmarkStatus.BETWEEN_LEVELS: "BENCH BETWEEN LEVELS",
            ConcurrencyBenchmarkStatus.CANCELLING: "BENCH CANCELLING",
            ConcurrencyBenchmarkStatus.COMPLETED: "BENCH COMPLETED",
            ConcurrencyBenchmarkStatus.CANCELLED: "BENCH CANCELLED",
            ConcurrencyBenchmarkStatus.ERROR: "BENCH ERROR",
        }
        return labels[lane.status]

    @staticmethod
    def _context_status(state: ApplicationState) -> str:
        lane = state.context_benchmark
        target = (
            lane.progress.active_target_length if lane.progress is not None else None
        )
        suffix = "" if target is None else f" {target}"
        labels = {
            ContextBenchmarkStatus.IDLE: "CONTEXT READY",
            ContextBenchmarkStatus.VALIDATING: "CONTEXT READY",
            ContextBenchmarkStatus.BUILDING_PROMPT: f"CONTEXT BUILDING{suffix}",
            ContextBenchmarkStatus.WARMING_UP: f"CONTEXT WARM-UP{suffix}",
            ContextBenchmarkStatus.RUNNING: f"CONTEXT RUNNING{suffix}",
            ContextBenchmarkStatus.PROBING: f"CONTEXT PROBING{suffix}",
            ContextBenchmarkStatus.BETWEEN_LENGTHS: "CONTEXT BETWEEN LENGTHS",
            ContextBenchmarkStatus.CANCELLING: "CONTEXT CANCELLING",
            ContextBenchmarkStatus.COMPLETED: "CONTEXT COMPLETED",
            ContextBenchmarkStatus.CANCELLED: "CONTEXT CANCELLED",
            ContextBenchmarkStatus.ERROR: "CONTEXT ERROR",
        }
        return labels[lane.status]

    @staticmethod
    def _power_pair(draw: float | None, limit: float | None) -> str:
        if draw is None and limit is None:
            return "--"
        draw_text = "--" if draw is None else f"{draw:.0f}"
        limit_text = "--" if limit is None else f"{limit:.0f}"
        return f"{draw_text}/{limit_text} W"
