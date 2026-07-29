"""Keyboard-first configuration for the official Tool Calling suites."""

from typing import cast

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from modeltop.benchmarks.models import (
    ToolCallingBenchmarkConfig,
    ToolCallingSuite,
)
from modeltop.messages import ToolCallingBenchmarkStartRequested


class ToolCallingConfigurationPanel(VerticalScroll):
    """Parse the two operational inputs into one immutable runtime config."""

    DEFAULT_CSS = """
    ToolCallingConfigurationPanel { width: 1fr; height: 1fr; padding: 0 1; }
    ToolCallingConfigurationPanel .section-title {
        height: 1; color: $primary; text-style: bold;
    }
    ToolCallingConfigurationPanel .field-row { height: 3; width: 1fr; }
    ToolCallingConfigurationPanel .field-row Label { width: 29; padding-top: 1; }
    ToolCallingConfigurationPanel .field-row Input { width: 1fr; }
    ToolCallingConfigurationPanel OptionList {
        height: 4;
        border: solid $border-blurred;
    }
    ToolCallingConfigurationPanel .facts { height: auto; color: $catppuccin-muted; }
    ToolCallingConfigurationPanel #tool-calling-error {
        height: auto; color: $error;
    }
    ToolCallingConfigurationPanel #tool-calling-run { margin: 1 0; }
    """

    def compose(self) -> ComposeResult:
        yield Static("TOOL CALLING CONFIGURATION", classes="section-title")
        yield Label("Official suite")
        yield OptionList(
            Option("Full · 69 scenarios", id="full"),
            Option("Core · 15 scenarios", id="core"),
            id="tool-calling-suite",
            compact=True,
        )
        yield Horizontal(
            Label("Per-request timeout seconds"),
            Input(id="tool-calling-timeout"),
            classes="field-row",
        )
        yield Static(
            "Full is the official 69-scenario Categories A-O suite and includes "
            "Category O structured-output cases. Core runs 15 scenarios.\n"
            "Canonical scoring controls are fixed: temperature 0, 8 turns, "
            "sequential execution, no injected errors, and no extra payload.",
            classes="facts",
            markup=False,
        )
        yield Static("", id="tool-calling-error", markup=False)
        yield Button(
            "Run Tool Calling Benchmark",
            id="tool-calling-run",
            variant="primary",
        )

    def on_mount(self) -> None:
        self.load_config(ToolCallingBenchmarkConfig())

    @staticmethod
    def _select(option_list: OptionList, option_id: str) -> None:
        for index in range(option_list.option_count):
            option = option_list.get_option_at_index(index)
            if option.id == option_id:
                option_list.highlighted = index
                return

    def load_config(self, config: ToolCallingBenchmarkConfig) -> None:
        """Restore an immutable config into the editable draft."""
        self._select(
            self.query_one("#tool-calling-suite", OptionList),
            config.suite,
        )
        self.query_one("#tool-calling-timeout", Input).value = str(
            config.request_timeout_seconds
        )
        self.query_one("#tool-calling-error", Static).update("")

    def parse_config(
        self,
        *,
        notify: bool = False,
    ) -> ToolCallingBenchmarkConfig | None:
        """Strictly parse the current draft through the runtime model."""
        option_list = self.query_one("#tool-calling-suite", OptionList)
        option = option_list.get_option_at_index(option_list.highlighted or 0)
        try:
            timeout = float(
                self.query_one("#tool-calling-timeout", Input).value.strip()
            )
            config = ToolCallingBenchmarkConfig(
                suite=cast(ToolCallingSuite, str(option.id)),
                request_timeout_seconds=timeout,
            )
        except (ValueError, ValidationError) as error:
            message = str(error).splitlines()[0]
            self.query_one("#tool-calling-error", Static).update(message)
            if notify:
                self.notify(
                    message,
                    title="Tool Calling configuration",
                    severity="error",
                )
            return None
        self.query_one("#tool-calling-error", Static).update("")
        return config

    def focus_suite(self) -> None:
        self.query_one("#tool-calling-suite", OptionList).focus()

    @on(Button.Pressed, "#tool-calling-run")
    def run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        config = self.parse_config(notify=True)
        if config is not None:
            self.post_message(ToolCallingBenchmarkStartRequested(config))
