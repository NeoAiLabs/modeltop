"""Keyboard-navigable benchmark sidebar."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option


class BenchmarkSidebar(Vertical):
    """Fixed-width benchmark navigation panel."""

    DEFAULT_CSS = """
    BenchmarkSidebar {
        width: 24;
        height: 1fr;
        background: #111820;
        border: solid #2d3b49;
    }

    BenchmarkSidebar #sidebar-title {
        width: 1fr;
        height: 1;
        color: #5da9e9;
        text-style: bold;
        text-wrap: nowrap;
        text-overflow: clip;
    }

    BenchmarkSidebar #sidebar-menu {
        width: 1fr;
        height: 1fr;
        padding: 0;
        border: none;
        background: transparent;
        color: #d8dee9;
    }

    BenchmarkSidebar #sidebar-menu:focus {
        border: none;
        background-tint: transparent;
    }

    BenchmarkSidebar #sidebar-menu > .option-list--option-highlighted {
        color: #ffffff;
        background: #5da9e9;
        text-style: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the benchmark label and sole focusable menu."""
        yield Static("BENCHMARKS", id="sidebar-title")
        yield OptionList(
            Option("Overview", id="overview"),
            Option("Chat", id="chat"),
            Option("Speed Test", id="speed-test"),
            Option("Concurrency", id="concurrency"),
            Option("Context Length", id="context"),
            Option("Tool Calling", id="tool-calling"),
            Option("Drafter", id="drafter"),
            Option("Results", id="results"),
            Option("Settings", id="settings"),
            id="sidebar-menu",
            compact=True,
            markup=False,
        )
