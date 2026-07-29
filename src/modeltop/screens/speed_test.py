"""Inline keyboard-first Speed Test workspace."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher

from modeltop.benchmarks.models import SpeedTestConfig
from modeltop.state import ApplicationState
from modeltop.widgets.speed_test_config import SpeedTestConfigPanel
from modeltop.widgets.speed_test_progress import SpeedTestProgressPanel
from modeltop.widgets.speed_test_results import SpeedTestResultsPanel


class SpeedTestView(Vertical):
    """Switch between validated configuration, live progress, and latest result."""

    DEFAULT_CSS = """
    SpeedTestView { width: 1fr; height: 1fr; background: $catppuccin-base; }
    SpeedTestView #speed-view-switcher { width: 1fr; height: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_ready = True
        self._was_active = False

    def compose(self) -> ComposeResult:
        yield ContentSwitcher(
            SpeedTestConfigPanel(id="speed-config-panel"),
            SpeedTestProgressPanel(id="speed-progress-panel"),
            SpeedTestResultsPanel(id="speed-result-panel"),
            id="speed-view-switcher",
            initial="speed-config-panel",
        )

    def update_state(self, state: ApplicationState) -> None:
        speed = state.speed_test
        switcher = self.query_one("#speed-view-switcher", ContentSwitcher)
        if speed.is_active:
            self._was_active = True
            self._show_ready = False
            switcher.current = "speed-progress-panel"
            self.progress_panel.update_state(speed)
            return
        if (
            speed.is_terminal
            and not self._show_ready
            and speed.latest_result is not None
        ):
            switcher.current = "speed-result-panel"
            self.results_panel.update_result(speed.latest_result)
            if self._was_active and state.active_view == "speed-test":
                self.call_after_refresh(self.results_panel.focus_actions)
            self._was_active = False
            return
        switcher.current = "speed-config-panel"

    def show_config(self, config: SpeedTestConfig | None = None) -> None:
        self._show_ready = True
        self._was_active = False
        if config is not None:
            self.config_panel.load_config(config)
        self.query_one(
            "#speed-view-switcher", ContentSwitcher
        ).current = "speed-config-panel"
        self.call_after_refresh(self.config_panel.focus_presets)

    def prepare_run(self, config: SpeedTestConfig) -> None:
        self._show_ready = False
        self.config_panel.load_config(config)

    @property
    def config_panel(self) -> SpeedTestConfigPanel:
        return self.query_one("#speed-config-panel", SpeedTestConfigPanel)

    @property
    def progress_panel(self) -> SpeedTestProgressPanel:
        return self.query_one("#speed-progress-panel", SpeedTestProgressPanel)

    @property
    def results_panel(self) -> SpeedTestResultsPanel:
        return self.query_one("#speed-result-panel", SpeedTestResultsPanel)
