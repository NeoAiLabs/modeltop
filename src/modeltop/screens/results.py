"""Durable benchmark history and family-safe comparison workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.widgets import ContentSwitcher, OptionList, Static
from textual.widgets.option_list import Option

from modeltop.messages import (
    ArchivedResultsComparisonRequested,
    ArchivedResultsSelectionChanged,
    SpeedTestResultSelected,
)
from modeltop.services.result_archive import ArchiveEntry
from modeltop.state import ApplicationState
from modeltop.widgets.archived_comparison import ArchivedComparisonPanel
from modeltop.widgets.speed_test_results import SpeedTestResultsPanel

if TYPE_CHECKING:
    from modeltop.app import ModelTopApp


class ResultsView(Vertical):
    """Show persisted benchmark history and compare two runs of one family."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("space", "toggle_selection", "Select", show=False),
        Binding("escape", "back", "Back", show=False),
    ]

    DEFAULT_CSS = """
    ResultsView { width: 1fr; height: 1fr; padding: 0 1; background: $catppuccin-base; }
    ResultsView #results-title { height: 1; color: $primary; text-style: bold; }
    ResultsView #results-switcher { width: 1fr; height: 1fr; }
    ResultsView #results-list {
        width: 1fr; height: 1fr; border: solid $border-blurred;
    }
    ResultsView #results-empty {
        width: 1fr; height: 1fr; content-align: center middle;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("BENCHMARK HISTORY", id="results-title")
        yield ContentSwitcher(
            OptionList(id="results-list", compact=True, markup=False),
            Static("No archived benchmark results.", id="results-empty", markup=False),
            SpeedTestResultsPanel(id="history-result-panel"),
            ArchivedComparisonPanel(id="archived-comparison-panel", markup=False),
            id="results-switcher",
            initial="results-empty",
        )

    def update_state(self, state: ApplicationState) -> None:
        archive = state.result_archive
        switcher = self.query_one("#results-switcher", ContentSwitcher)
        if state.speed_test.selected_result_id is not None:
            live = state.speed_test.result_by_id(state.speed_test.selected_result_id)
            if live is not None:
                self.query_one(SpeedTestResultsPanel).update_result(live)
                switcher.current = "history-result-panel"
                return

        selected = archive.archive_selection
        if len(selected) == 2:
            first = archive.documents.get(selected[0])
            second = archive.documents.get(selected[1])
            if (
                first is not None
                and second is not None
                and first.entry.kind == second.entry.kind
            ):
                self.query_one(ArchivedComparisonPanel).update_documents(first, second)
                switcher.current = "archived-comparison-panel"
                return
        if not archive.entries:
            switcher.current = "results-empty"
            return
        options: list[Option] = []
        for entry in archive.entries:
            timestamp = entry.completed_at.isoformat()
            prefix = self._prefix(entry.result_id, selected)
            prompt = (
                f"{prefix}{timestamp} · {entry.kind} · {entry.model_id} · "
                f"{entry.status.upper()} · {self._headline(entry)} · "
                f"{entry.result_id[-12:]}"
            )
            options.append(Option(prompt, id=entry.result_id))
        self.query_one("#results-list", OptionList).set_options(options)
        switcher.current = "results-list"

    def focus_list(self) -> None:
        if (
            self.query_one("#results-switcher", ContentSwitcher).current
            == "results-list"
        ):
            self.query_one("#results-list", OptionList).focus()

    @property
    def showing_detail(self) -> bool:
        return (
            self.query_one("#results-switcher", ContentSwitcher).current
            == "history-result-panel"
        )

    def action_toggle_selection(self) -> None:
        option_list = self.query_one("#results-list", OptionList)
        option = option_list.get_option_at_index(option_list.highlighted or 0)
        if option.id is not None:
            self._toggle(str(option.id))

    def _toggle(self, result_id: str) -> None:
        app = cast("ModelTopApp", cast(object, self.app))  # pyright: ignore[reportUnknownMemberType]
        state = app.dashboard_state
        if state is None:
            return
        selected = state.result_archive.archive_selection
        if result_id in selected:
            self.post_message(
                ArchivedResultsSelectionChanged(
                    tuple(x for x in selected if x != result_id)
                )
            )
            return
        entry = state.result_archive.documents.get(result_id)
        if entry is None:
            return
        if selected:
            first = state.result_archive.documents.get(selected[0])
            if first is not None and first.entry.kind != entry.entry.kind:
                app.notify(
                    "Select two runs from the same benchmark type.", severity="warning"
                )
                return
        self.post_message(ArchivedResultsSelectionChanged((*selected, result_id)[-2:]))

    def action_back(self) -> None:
        app = cast("ModelTopApp", cast(object, self.app))  # pyright: ignore[reportUnknownMemberType]
        if self.showing_detail:
            app._select_result(None)  # pyright: ignore[reportPrivateUsage]
            return
        state = app.dashboard_state
        if state is not None and state.result_archive.archive_selection:
            self.post_message(ArchivedResultsSelectionChanged(()))
            return
        app._set_active_view("speed-test")  # pyright: ignore[reportPrivateUsage]

    @staticmethod
    def _prefix(result_id: str, selected: tuple[str, ...]) -> str:
        try:
            return f"[{selected.index(result_id) + 1}] "
        except ValueError:
            return ""

    @staticmethod
    def _headline(entry: ArchiveEntry) -> str:
        summary = entry.summary
        if entry.kind in {"speed-test", "drafter"}:
            value = summary.get("mean_output_tokens_per_second")
            return "--" if value is None else f"{value:.1f} tok/s"
        if entry.kind == "concurrency":
            value = summary.get("peak_aggregate_output_tokens_per_second")
            return "--" if value is None else f"{value:.1f} tok/s"
        if entry.kind == "context":
            return f"{summary.get('highest_successful_prompt_tokens') or '--'} tokens"
        if entry.kind == "r0b0bench":
            validity = "DIAGNOSTIC" if summary.get("invalid_for_publish") else "VALID"
            return (
                f"{summary.get('pass_count', 0)}/{summary.get('selected_count', 0)} "
                f"pass · {summary.get('infra_errors_total', 0)} infra · {validity}"
            )
        score = summary.get("final_score")
        return f"{score if score is not None else '--'} score"

    @on(OptionList.OptionSelected, "#results-list")
    def result_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        state = cast("ModelTopApp", cast(object, self.app)).dashboard_state  # pyright: ignore[reportUnknownMemberType]
        if state is None:
            return
        selected = state.result_archive.archive_selection
        if len(selected) == 2:
            self.post_message(
                ArchivedResultsComparisonRequested((selected[0], selected[1]))
            )
            return
        if event.option.id is None:
            return
        result_id = str(event.option.id)
        if state.speed_test.result_by_id(result_id):
            self.post_message(SpeedTestResultSelected(result_id))
            return
        if result_id in state.result_archive.documents:
            self._toggle(result_id)
