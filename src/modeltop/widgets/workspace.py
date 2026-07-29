"""Live model-discovery workspace."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import OptionList, Static

from modeltop.hardware.models import (
    GpuMetrics,
    format_byte_pair,
    format_elapsed,
    format_percentage,
    format_temperature,
)
from modeltop.state import ApplicationState, ServerStatus


class Workspace(VerticalScroll):
    """Scrollable workspace with conditional model-selection focus."""

    DEFAULT_CSS = """
    Workspace {
        width: 1fr;
        height: 1fr;
        background: $catppuccin-base;
        overflow-x: hidden;
        border-top: solid $border-blurred;
        border-right: solid $border-blurred;
        border-bottom: solid $border-blurred;
        content-align: center middle;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: clip;
        align: center middle;
    }

    Workspace #hardware-title,
    Workspace #hardware-gpus,
    Workspace #hardware-system,
    Workspace #hardware-status {
        width: 60;
        max-width: 90%;
        height: auto;
        min-height: 1;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: clip;
        overflow-x: hidden;
    }

    Workspace #hardware-title {
        color: $primary;
        text-style: bold;
    }

    Workspace #hardware-gpus,
    Workspace #hardware-system {
        color: $foreground;
    }

    Workspace #hardware-status {
        color: $catppuccin-muted;
        margin-bottom: 1;
    }

    Workspace #models-title {
        width: 60;
        max-width: 90%;
        height: 1;
        color: $primary;
        text-style: bold;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    Workspace #model-selector {
        width: 60;
        max-width: 90%;
        height: 8;
        padding: 0;
        border: solid $border-blurred;
        background: $surface;
        color: $foreground;
    }

    Workspace #model-selector:focus {
        border: solid $border-blurred;
        background-tint: transparent;
    }

    Workspace #model-selector > .option-list--option-highlighted {
        color: $catppuccin-crust;
        background: $secondary;
        text-style: none;
    }

    Workspace #models-state {
        width: 60;
        max-width: 90%;
        height: 1;
        color: $foreground;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: clip;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(can_focus=False, id=id)

    def compose(self) -> ComposeResult:
        """Compose hardware metrics above the existing model selector."""
        yield Static("LOCAL HARDWARE", id="hardware-title", markup=False)
        yield Static("GPU --", id="hardware-gpus", markup=False)
        yield Static(
            "CPU -- │ RAM -- │ LOAD -- -- --", id="hardware-system", markup=False
        )
        yield Static(
            "PROVIDER -- │ INITIALISING │ UPDATED --",
            id="hardware-status",
            markup=False,
        )
        yield Static("DISCOVERED MODELS", id="models-title")
        yield OptionList(
            id="model-selector",
            compact=True,
            markup=False,
        )
        yield Static("Connecting...", id="models-state")

    def update_state(self, state: ApplicationState) -> None:
        """Render model choices while keeping focus valid."""
        selector = self.query_one("#model-selector", OptionList)
        state_text = self.query_one("#models-state", Static)
        models = state.available_models
        selector.set_options(model.id for model in models)

        multiple_online = (
            state.server_status is ServerStatus.ONLINE and len(models) >= 2
        )
        if multiple_online:
            selector.display = True
            selector.disabled = False
            state_text.display = False
            selector.highlighted = next(
                (
                    index
                    for index, model in enumerate(models)
                    if model.id == state.selected_model_id
                ),
                None,
            )
            return

        if selector.has_focus:
            self.screen.query_one("#sidebar-menu", OptionList).focus()
        selector.display = False
        selector.disabled = True
        selector.highlighted = None
        state_text.display = True
        if state.server_status is ServerStatus.ONLINE:
            message = models[0].id if models else "No models discovered"
        elif state.server_status is ServerStatus.CONNECTING:
            message = "Connecting..."
        else:
            message = "Models unavailable"
        state_text.update(message)

    def update_hardware_state(self, state: ApplicationState) -> None:
        """Update only stable hardware rows without changing selector or focus."""
        title = self.query_one("#hardware-title", Static)
        gpu_rows = self.query_one("#hardware-gpus", Static)
        system_row = self.query_one("#hardware-system", Static)
        status_row = self.query_one("#hardware-status", Static)
        title.update("LOCAL HARDWARE")

        snapshot = state.hardware_snapshot
        if snapshot is None:
            gpu_rows.update("GPU -- │ UTIL -- │ VRAM -- │ TEMP -- │ POWER -- │ FAN --")
            system_row.update("CPU -- │ RAM -- │ LOAD -- -- --")
            provider = "--"
            updated_at = state.hardware_last_refresh_time
        else:
            if snapshot.gpus:
                gpu_rows.update("\n".join(self._gpu_line(gpu) for gpu in snapshot.gpus))
            else:
                gpu_rows.update(
                    "GPU -- │ UTIL -- │ VRAM -- │ TEMP -- │ POWER -- │ FAN --"
                )
            cpu = format_percentage(snapshot.cpu.utilisation_percent)
            ram = format_byte_pair(
                snapshot.memory.used_bytes,
                snapshot.memory.total_bytes,
                compact=True,
            )
            loads = " ".join(
                self._format_load(value)
                for value in (
                    snapshot.cpu.load_average_1m,
                    snapshot.cpu.load_average_5m,
                    snapshot.cpu.load_average_15m,
                )
            )
            system_row.update(f"CPU {cpu} │ RAM {ram} │ LOAD {loads}")
            provider = snapshot.provider_name
            updated_at = snapshot.collected_at

        status = state.hardware_status.value.upper()
        error = state.hardware_last_error
        error_text = f" │ {error}" if error is not None else ""
        status_row.update(
            f"PROVIDER {provider} │ {status}{error_text} │ "
            f"UPDATED {format_elapsed(updated_at)}"
        )

    @classmethod
    def _gpu_line(cls, gpu: GpuMetrics) -> str:
        vram = format_byte_pair(
            gpu.memory_used_bytes, gpu.memory_total_bytes, compact=True
        )
        return (
            f"GPU {gpu.index}  {gpu.name} │ "
            f"UTIL {format_percentage(gpu.utilisation_percent)} │ "
            f"VRAM {vram} │ "
            f"TEMP {format_temperature(gpu.temperature_celsius)} │ "
            f"POWER {cls._power_pair(gpu.power_draw_watts, gpu.power_limit_watts)} │ "
            f"FAN {format_percentage(gpu.fan_speed_percent)}"
        )

    @staticmethod
    def _format_load(value: float | None) -> str:
        return "--" if value is None else f"{value:.2f}"

    @staticmethod
    def _power_pair(draw: float | None, limit: float | None) -> str:
        if draw is None and limit is None:
            return "--"
        draw_text = "--" if draw is None else f"{draw:.0f}"
        limit_text = "--" if limit is None else f"{limit:.0f}"
        return f"{draw_text}/{limit_text} W"
