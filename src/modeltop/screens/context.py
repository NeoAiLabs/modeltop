"""Inline keyboard-first Context Length benchmark workspace."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher

from modeltop.benchmarks.models import ContextBenchmarkConfig
from modeltop.state import ApplicationState
from modeltop.widgets.context_configuration import ContextConfigurationPanel
from modeltop.widgets.context_progress import ContextProgressPanel
from modeltop.widgets.context_results import ContextResultsPanel


class ContextView(Vertical):
    """Switch between Context configuration, compact progress, and latest result."""

    DEFAULT_CSS = """
    ContextView { width: 1fr; height: 1fr; background: $catppuccin-base; }
    ContextView #context-view-switcher { width: 1fr; height: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_ready = True
        self._was_active = False

    def compose(self) -> ComposeResult:
        yield ContentSwitcher(
            ContextConfigurationPanel(id="context-config-panel"),
            ContextProgressPanel(id="context-progress-panel"),
            ContextResultsPanel(id="context-result-panel"),
            id="context-view-switcher",
            initial="context-config-panel",
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.context_benchmark
        switcher = self.query_one("#context-view-switcher", ContentSwitcher)
        if lane.is_active:
            self._was_active = True
            self._show_ready = False
            switcher.current = "context-progress-panel"
            self.progress_panel.update_state(state)
            return
        if lane.is_terminal and not self._show_ready and lane.latest_result is not None:
            switcher.current = "context-result-panel"
            self.results_panel.update_result(lane.latest_result)
            if self._was_active and state.active_view == "context":
                self.call_after_refresh(self.results_panel.focus_actions)
            self._was_active = False
            return
        switcher.current = "context-config-panel"

    def show_config(self, config: ContextBenchmarkConfig | None = None) -> None:
        self._show_ready = True
        self._was_active = False
        if config is not None:
            self.config_panel.load_config(config)
        self.query_one(
            "#context-view-switcher", ContentSwitcher
        ).current = "context-config-panel"
        self.call_after_refresh(self.config_panel.focus_mode)

    def prepare_run(self, config: ContextBenchmarkConfig) -> None:
        self._show_ready = False
        self.config_panel.load_config(config)

    @property
    def config_panel(self) -> ContextConfigurationPanel:
        return self.query_one("#context-config-panel", ContextConfigurationPanel)

    @property
    def progress_panel(self) -> ContextProgressPanel:
        return self.query_one("#context-progress-panel", ContextProgressPanel)

    @property
    def results_panel(self) -> ContextResultsPanel:
        return self.query_one("#context-result-panel", ContextResultsPanel)
