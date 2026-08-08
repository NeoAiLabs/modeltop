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
        background: $surface;
        border: solid $border-blurred;
    }

    BenchmarkSidebar #sidebar-title {
        width: 1fr;
        height: 1;
        color: $primary;
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
        color: $foreground;
    }

    BenchmarkSidebar #sidebar-menu:focus {
        border: none;
        background-tint: transparent;
    }

    BenchmarkSidebar #sidebar-menu > .option-list--option-highlighted {
        color: $catppuccin-crust;
        background: $secondary;
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
            Option("r0b0bench", id="r0b0bench"),
            Option("Drafter", id="drafter"),
            Option("Results", id="results"),
            Option("Settings", id="settings"),
            id="sidebar-menu",
            compact=True,
            markup=False,
        )
