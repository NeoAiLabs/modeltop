"""Inline keyboard-first Tool Calling benchmark workspace."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher

from modeltop.benchmarks.models import ToolCallingBenchmarkConfig
from modeltop.state import ApplicationState
from modeltop.widgets.tool_calling_configuration import (
    ToolCallingConfigurationPanel,
)
from modeltop.widgets.tool_calling_progress import ToolCallingProgressPanel
from modeltop.widgets.tool_calling_results import ToolCallingResultsPanel


class ToolCallingView(Vertical):
    """Switch between configuration, progress, and one latest result."""

    DEFAULT_CSS = """
    ToolCallingView { width: 1fr; height: 1fr; background: $catppuccin-base; }
    ToolCallingView #tool-calling-view-switcher { width: 1fr; height: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_ready = True
        self._was_active = False

    def compose(self) -> ComposeResult:
        yield ContentSwitcher(
            ToolCallingConfigurationPanel(id="tool-calling-config-panel"),
            ToolCallingProgressPanel(id="tool-calling-progress-panel"),
            ToolCallingResultsPanel(id="tool-calling-result-panel"),
            id="tool-calling-view-switcher",
            initial="tool-calling-config-panel",
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.tool_calling_benchmark
        switcher = self.query_one("#tool-calling-view-switcher", ContentSwitcher)
        if lane.is_active:
            self._was_active = True
            self._show_ready = False
            switcher.current = "tool-calling-progress-panel"
            self.progress_panel.update_state(state)
            return
        if lane.is_terminal and not self._show_ready and lane.latest_result is not None:
            switcher.current = "tool-calling-result-panel"
            self.results_panel.update_result(lane.latest_result)
            if self._was_active and state.active_view == "tool-calling":
                self.call_after_refresh(self.results_panel.focus_actions)
            self._was_active = False
            return
        switcher.current = "tool-calling-config-panel"

    def show_config(self, config: ToolCallingBenchmarkConfig | None = None) -> None:
        self._show_ready = True
        self._was_active = False
        if config is not None:
            self.config_panel.load_config(config)
        self.query_one(
            "#tool-calling-view-switcher", ContentSwitcher
        ).current = "tool-calling-config-panel"
        self.call_after_refresh(self.config_panel.focus_suite)

    def prepare_run(self, config: ToolCallingBenchmarkConfig) -> None:
        self._show_ready = False
        self.config_panel.load_config(config)

    @property
    def config_panel(self) -> ToolCallingConfigurationPanel:
        return self.query_one(
            "#tool-calling-config-panel", ToolCallingConfigurationPanel
        )

    @property
    def progress_panel(self) -> ToolCallingProgressPanel:
        return self.query_one("#tool-calling-progress-panel", ToolCallingProgressPanel)

    @property
    def results_panel(self) -> ToolCallingResultsPanel:
        return self.query_one("#tool-calling-result-panel", ToolCallingResultsPanel)
