"""Selectable r0b0bench workspace and safety confirmation."""

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, ContentSwitcher, Static

from modeltop.benchmarks.models import R0b0benchBenchmarkConfig
from modeltop.benchmarks.r0b0bench_contract import R0B0BENCH_QUALITY_ORDER
from modeltop.state import ApplicationState
from modeltop.widgets.r0b0bench_configuration import R0b0benchConfigurationPanel
from modeltop.widgets.r0b0bench_progress import R0b0benchProgressPanel
from modeltop.widgets.r0b0bench_results import R0b0benchResultsPanel


def r0b0bench_requires_confirmation(config: R0b0benchBenchmarkConfig) -> bool:
    """Return whether this run includes long or local-code capabilities."""
    selected = set(config.selected_lanes)
    return bool(
        selected.intersection({"humaneval", "niah", "bfcl_mt", "bfcl_ast"})
        or (config.profile == "core" and selected.intersection(R0B0BENCH_QUALITY_ORDER))
    )


class R0b0benchRunConfirmation(ModalScreen[bool]):
    """Confirm long/high-context and local generated-code execution."""

    DEFAULT_CSS = """
    R0b0benchRunConfirmation {
        align: center middle; background: $catppuccin-crust 80%;
    }
    R0b0benchRunConfirmation #r0b0bench-confirm-dialog {
        width: 72; max-width: 90%; height: auto; padding: 1 2;
        border: solid $warning; background: $catppuccin-base;
    }
    R0b0benchRunConfirmation #r0b0bench-confirm-actions {
        height: 3; align-horizontal: center;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    def __init__(self, config: R0b0benchBenchmarkConfig) -> None:
        super().__init__()
        self._config = config

    def compose(self) -> ComposeResult:
        selected = set(self._config.selected_lanes)
        facts = [
            "Selected r0b0bench tests may run many requests and take substantial time."
        ]
        if "niah" in selected:
            facts.append(
                "NIAH uses the endpoint's advertised maximum context and executes "
                "local tokenizer code."
            )
        if selected.intersection({"bfcl_mt", "bfcl_ast", "humaneval"}):
            facts.append(
                "Selected tests execute local generated-code or evaluator processes."
            )
        yield Static(" ".join(facts), id="r0b0bench-confirm-dialog", markup=False)
        yield Horizontal(
            Button("Continue", id="r0b0bench-confirm", variant="warning"),
            Button("Cancel", id="r0b0bench-confirm-cancel"),
            id="r0b0bench-confirm-actions",
        )

    def on_mount(self) -> None:
        self.query_one("#r0b0bench-confirm", Button).focus()

    @on(Button.Pressed, "#r0b0bench-confirm")
    def confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)

    @on(Button.Pressed, "#r0b0bench-confirm-cancel")
    def cancel_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class R0b0benchView(Vertical):
    """Switch among editable configuration, progress, and latest result."""

    DEFAULT_CSS = """
    R0b0benchView { width: 1fr; height: 1fr; background: $catppuccin-base; }
    R0b0benchView #r0b0bench-view-switcher { width: 1fr; height: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_ready = True
        self._was_active = False

    def compose(self) -> ComposeResult:
        yield ContentSwitcher(
            R0b0benchConfigurationPanel(id="r0b0bench-config-panel"),
            R0b0benchProgressPanel(id="r0b0bench-progress-panel"),
            R0b0benchResultsPanel(id="r0b0bench-result-panel"),
            id="r0b0bench-view-switcher",
            initial="r0b0bench-config-panel",
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.r0b0bench_benchmark
        switcher = self.query_one("#r0b0bench-view-switcher", ContentSwitcher)
        if lane.is_active:
            self._was_active = True
            self._show_ready = False
            switcher.current = "r0b0bench-progress-panel"
            self.progress_panel.update_state(state)
            return
        if lane.is_terminal and not self._show_ready and lane.latest_result is not None:
            switcher.current = "r0b0bench-result-panel"
            self.results_panel.update_result(lane.latest_result)
            if self._was_active and state.active_view == "r0b0bench":
                self.call_after_refresh(self.results_panel.focus_actions)
            self._was_active = False
            return
        switcher.current = "r0b0bench-config-panel"

    def show_config(self, config: R0b0benchBenchmarkConfig | None = None) -> None:
        self._show_ready = True
        self._was_active = False
        if config is not None:
            self.config_panel.load_config(config)
        self.query_one(
            "#r0b0bench-view-switcher", ContentSwitcher
        ).current = "r0b0bench-config-panel"
        self.call_after_refresh(self.config_panel.focus_profile)

    def prepare_run(self, config: R0b0benchBenchmarkConfig) -> None:
        self._show_ready = False
        self.config_panel.load_config(config)

    @property
    def config_panel(self) -> R0b0benchConfigurationPanel:
        return self.query_one("#r0b0bench-config-panel", R0b0benchConfigurationPanel)

    @property
    def progress_panel(self) -> R0b0benchProgressPanel:
        return self.query_one("#r0b0bench-progress-panel", R0b0benchProgressPanel)

    @property
    def results_panel(self) -> R0b0benchResultsPanel:
        return self.query_one("#r0b0bench-result-panel", R0b0benchResultsPanel)
