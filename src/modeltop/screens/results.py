"""In-process Speed Test result history and detail workspace."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import ContentSwitcher, OptionList, Static
from textual.widgets.option_list import Option

from modeltop.messages import SpeedTestResultSelected
from modeltop.state import ApplicationState
from modeltop.widgets.speed_test_results import SpeedTestResultsPanel


class ResultsView(Vertical):
    """Show newest-first session history and reusable terminal detail."""

    DEFAULT_CSS = """
    ResultsView { width: 1fr; height: 1fr; padding: 0 1; background: #0f151d; }
    ResultsView #results-title { height: 1; color: #5da9e9; text-style: bold; }
    ResultsView #results-switcher { width: 1fr; height: 1fr; }
    ResultsView #results-list { width: 1fr; height: 1fr; border: solid #2d3b49; }
    ResultsView #results-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("SESSION SPEED TEST RESULTS", id="results-title")
        yield ContentSwitcher(
            OptionList(id="results-list", compact=True, markup=False),
            Static(
                "No Speed Test results in this session.",
                id="results-empty",
                markup=False,
            ),
            SpeedTestResultsPanel(id="history-result-panel"),
            id="results-switcher",
            initial="results-empty",
        )

    def update_state(self, state: ApplicationState) -> None:
        speed = state.speed_test
        switcher = self.query_one("#results-switcher", ContentSwitcher)
        if speed.selected_result_id is not None:
            selected = speed.result_by_id(speed.selected_result_id)
            if selected is not None:
                self.detail_panel.update_result(selected)
                switcher.current = "history-result-panel"
                return
        if not speed.results:
            switcher.current = "results-empty"
            return
        options: list[Option] = []
        for result in reversed(speed.results):
            mean = result.output_tokens_per_second.mean
            speed_label = "--" if mean is None else f"{mean:.1f} tok/s"
            options.append(
                Option(
                    f"{result.completed_at.isoformat()} · {result.model_id} · "
                    f"{result.status.value.replace('_', ' ').upper()} · "
                    f"{speed_label} · "
                    f"{result.run_id[-12:]}",
                    id=result.run_id,
                )
            )
        self.query_one("#results-list", OptionList).set_options(options)
        switcher.current = "results-list"

    def focus_list(self) -> None:
        switcher = self.query_one("#results-switcher", ContentSwitcher)
        if switcher.current == "results-list":
            self.query_one("#results-list", OptionList).focus()

    @property
    def showing_detail(self) -> bool:
        return (
            self.query_one("#results-switcher", ContentSwitcher).current
            == "history-result-panel"
        )

    @property
    def detail_panel(self) -> SpeedTestResultsPanel:
        return self.query_one("#history-result-panel", SpeedTestResultsPanel)

    @on(OptionList.OptionSelected, "#results-list")
    def result_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id is not None:
            self.post_message(SpeedTestResultSelected(event.option.id))
