"""Inline keyboard-first Drafter benchmark workspace."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher

from modeltop.benchmarks.models import DrafterBenchmarkConfig
from modeltop.state import ApplicationState
from modeltop.widgets.drafter_configuration import DrafterConfigurationPanel
from modeltop.widgets.drafter_progress import DrafterProgressPanel
from modeltop.widgets.drafter_results import DrafterResultsPanel


class DrafterView(Vertical):
    """Switch between configuration, live progress, and latest terminal result."""

    DEFAULT_CSS = """
    DrafterView { width: 1fr; height: 1fr; background: $catppuccin-base; }
    DrafterView #drafter-view-switcher { width: 1fr; height: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_ready = True
        self._was_active = False

    def compose(self) -> ComposeResult:
        yield ContentSwitcher(
            DrafterConfigurationPanel(id="drafter-config-panel"),
            DrafterProgressPanel(id="drafter-progress-panel"),
            DrafterResultsPanel(id="drafter-result-panel"),
            id="drafter-view-switcher",
            initial="drafter-config-panel",
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.drafter_benchmark
        switcher = self.query_one("#drafter-view-switcher", ContentSwitcher)
        if lane.is_active:
            self._was_active = True
            self._show_ready = False
            switcher.current = "drafter-progress-panel"
            self.progress_panel.update_state(lane)
            return
        if lane.is_terminal and not self._show_ready and lane.latest_result is not None:
            switcher.current = "drafter-result-panel"
            self.results_panel.update_result(lane.latest_result)
            if self._was_active and state.active_view == "drafter":
                self.call_after_refresh(self.results_panel.focus_actions)
            self._was_active = False
            return
        switcher.current = "drafter-config-panel"

    def show_config(self, config: DrafterBenchmarkConfig | None = None) -> None:
        self._show_ready = True
        self._was_active = False
        if config is not None:
            self.config_panel.load_config(config)
        self.query_one(
            "#drafter-view-switcher", ContentSwitcher
        ).current = "drafter-config-panel"
        self.call_after_refresh(self.config_panel.focus_prompt)

    def prepare_run(self, config: DrafterBenchmarkConfig) -> None:
        self._show_ready = False
        self.config_panel.load_config(config)

    @property
    def config_panel(self) -> DrafterConfigurationPanel:
        return self.query_one("#drafter-config-panel", DrafterConfigurationPanel)

    @property
    def progress_panel(self) -> DrafterProgressPanel:
        return self.query_one("#drafter-progress-panel", DrafterProgressPanel)

    @property
    def results_panel(self) -> DrafterResultsPanel:
        return self.query_one("#drafter-result-panel", DrafterResultsPanel)
