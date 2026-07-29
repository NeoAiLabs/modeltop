"""Inline keyboard-first Concurrency benchmark workspace."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher

from modeltop.benchmarks.models import ConcurrencyBenchmarkConfig
from modeltop.state import ApplicationState
from modeltop.widgets.benchmark_configuration import BenchmarkConfigurationPanel
from modeltop.widgets.benchmark_progress import BenchmarkProgressPanel
from modeltop.widgets.concurrency_results import ConcurrencyResultsPanel


class ConcurrencyView(Vertical):
    """Switch between configuration, live progress, and latest terminal result."""

    DEFAULT_CSS = """
    ConcurrencyView { width: 1fr; height: 1fr; background: #0f151d; }
    ConcurrencyView #concurrency-view-switcher { width: 1fr; height: 1fr; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_ready = True
        self._was_active = False

    def compose(self) -> ComposeResult:
        yield ContentSwitcher(
            BenchmarkConfigurationPanel(id="concurrency-config-panel"),
            BenchmarkProgressPanel(id="concurrency-progress-panel"),
            ConcurrencyResultsPanel(id="concurrency-result-panel"),
            id="concurrency-view-switcher",
            initial="concurrency-config-panel",
        )

    def update_state(self, state: ApplicationState) -> None:
        lane = state.concurrency_benchmark
        switcher = self.query_one("#concurrency-view-switcher", ContentSwitcher)
        if lane.is_active:
            self._was_active = True
            self._show_ready = False
            switcher.current = "concurrency-progress-panel"
            self.progress_panel.update_state(state)
            return
        if lane.is_terminal and not self._show_ready and lane.latest_result is not None:
            switcher.current = "concurrency-result-panel"
            self.results_panel.update_result(lane.latest_result)
            if self._was_active and state.active_view == "concurrency":
                self.call_after_refresh(self.results_panel.focus_actions)
            self._was_active = False
            return
        switcher.current = "concurrency-config-panel"

    def show_config(self, config: ConcurrencyBenchmarkConfig | None = None) -> None:
        self._show_ready = True
        self._was_active = False
        if config is not None:
            self.config_panel.load_config(config)
        self.query_one(
            "#concurrency-view-switcher", ContentSwitcher
        ).current = "concurrency-config-panel"
        self.call_after_refresh(self.config_panel.focus_mode)

    def prepare_run(self, config: ConcurrencyBenchmarkConfig) -> None:
        self._show_ready = False
        self.config_panel.load_config(config)

    @property
    def config_panel(self) -> BenchmarkConfigurationPanel:
        return self.query_one("#concurrency-config-panel", BenchmarkConfigurationPanel)

    @property
    def progress_panel(self) -> BenchmarkProgressPanel:
        return self.query_one("#concurrency-progress-panel", BenchmarkProgressPanel)

    @property
    def results_panel(self) -> ConcurrencyResultsPanel:
        return self.query_one("#concurrency-result-panel", ConcurrencyResultsPanel)
