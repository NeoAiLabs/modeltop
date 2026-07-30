"""Editable, validated Speed Test configuration panel."""

from typing import ClassVar, cast

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, OptionList, Static, Switch, TextArea
from textual.widgets.option_list import Option

from modeltop.benchmarks.models import (
    SpeedTestConfig,
    SpeedTestPreset,
    speed_test_config_for_preset,
)
from modeltop.messages import SpeedTestStartRequested


class SpeedTestConfigPanel(VerticalScroll):
    """Keep editable draft text local until Start validates it once."""

    DEFAULT_CSS = """
    SpeedTestConfigPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        overflow-x: hidden;
        background: $catppuccin-base;
    }
    SpeedTestConfigPanel .section-title {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    SpeedTestConfigPanel #speed-preset-list {
        height: 4;
        width: 100%;
        border: solid $border-blurred;
        padding: 0;
        background: $catppuccin-base;
    }
    SpeedTestConfigPanel #speed-prompt { height: 5; }
    SpeedTestConfigPanel .config-row { height: 3; }
    SpeedTestConfigPanel .config-row Label { width: 18; padding-top: 1; }
    SpeedTestConfigPanel .config-row Input { width: 1fr; }
    SpeedTestConfigPanel #continue-row { height: 2; }
    SpeedTestConfigPanel #speed-run-plan { height: 2; color: $catppuccin-muted; }
    SpeedTestConfigPanel #speed-start { width: 18; }
    """

    _FIELD_WIDGETS: ClassVar[dict[str, str]] = {
        "prompt": "#speed-prompt",
        "warmup_runs": "#speed-warmups",
        "measured_runs": "#speed-measured",
        "max_tokens": "#speed-max-tokens",
        "temperature": "#speed-temperature",
        "top_p": "#speed-top-p",
        "seed": "#speed-seed",
        "request_timeout_seconds": "#speed-timeout",
        "continue_on_error": "#speed-continue",
        "thinking_mode": "#speed-disable-thinking",
    }

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._preset: SpeedTestPreset = "standard"
        self._loaded = False

    def compose(self) -> ComposeResult:
        yield Static("SPEED TEST CONFIGURATION", classes="section-title")
        yield OptionList(
            Option("Quick · 1 warm-up / 3 runs / 128 tokens", id="quick"),
            Option("Standard · 1 warm-up / 5 runs / 256 tokens", id="standard"),
            Option("Long · 1 warm-up / 3 runs / 1024 tokens", id="long"),
            Option("Custom · preserve displayed values", id="custom"),
            id="speed-preset-list",
            compact=True,
            markup=False,
        )
        yield Label("Prompt", classes="section-title")
        yield TextArea(id="speed-prompt", soft_wrap=True, show_line_numbers=False)
        for label, identifier in (
            ("Warm-up runs", "speed-warmups"),
            ("Measured runs", "speed-measured"),
            ("Max tokens", "speed-max-tokens"),
            ("Temperature", "speed-temperature"),
            ("Top-p", "speed-top-p"),
            ("Seed (optional)", "speed-seed"),
            ("Timeout seconds", "speed-timeout"),
        ):
            with Horizontal(classes="config-row"):
                yield Label(label)
                yield Input(id=identifier)
        with Horizontal(id="continue-row", classes="config-row"):
            yield Label("Continue on error")
            yield Switch(id="speed-continue")
        with Horizontal(id="thinking-row", classes="config-row"):
            yield Label("Disable thinking (Qwen/vLLM)")
            yield Switch(id="speed-disable-thinking")
        yield Static("", id="speed-run-plan", markup=False)
        yield Button("Start Speed Test", id="speed-start", variant="primary")

    def on_mount(self) -> None:
        self.load_config(SpeedTestConfig())

    def load_config(self, config: SpeedTestConfig) -> None:
        """Replace the displayed draft without starting a benchmark."""
        self._preset = config.preset
        self.query_one("#speed-prompt", TextArea).text = config.prompt
        values = {
            "#speed-warmups": str(config.warmup_runs),
            "#speed-measured": str(config.measured_runs),
            "#speed-max-tokens": str(config.max_tokens),
            "#speed-temperature": str(config.temperature),
            "#speed-top-p": str(config.top_p),
            "#speed-seed": "" if config.seed is None else str(config.seed),
            "#speed-timeout": str(config.request_timeout_seconds),
        }
        for selector, value in values.items():
            self.query_one(selector, Input).value = value
        self.query_one("#speed-continue", Switch).value = config.continue_on_error
        self.query_one("#speed-disable-thinking", Switch).value = (
            config.thinking_mode == "disabled"
        )
        presets = self.query_one("#speed-preset-list", OptionList)
        presets.highlighted = ("quick", "standard", "long", "custom").index(
            config.preset
        )
        self._loaded = True
        self._update_plan()

    @on(OptionList.OptionSelected, "#speed-preset-list")
    def select_preset(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if not self._loaded or event.option.id is None:
            return
        preset = cast(SpeedTestPreset, event.option.id)
        current = self._parse_config(notify=False)
        if preset == "custom":
            if current is not None:
                self.load_config(speed_test_config_for_preset("custom", current))
            else:
                self._preset = "custom"
            return
        self.load_config(speed_test_config_for_preset(preset))

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Switch.Changed)
    def draft_changed(self) -> None:
        if self._loaded:
            self._update_plan()

    @on(Button.Pressed, "#speed-start")
    def start_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        config = self._parse_config(notify=True)
        if config is not None:
            self.post_message(SpeedTestStartRequested(config))

    def focus_presets(self) -> None:
        self.query_one("#speed-preset-list", OptionList).focus()

    def _parse_config(self, *, notify: bool) -> SpeedTestConfig | None:
        seed_text = self.query_one("#speed-seed", Input).value.strip()
        data: dict[str, object] = {
            "preset": self._preset,
            "prompt": self.query_one("#speed-prompt", TextArea).text,
            "warmup_runs": self.query_one("#speed-warmups", Input).value,
            "measured_runs": self.query_one("#speed-measured", Input).value,
            "max_tokens": self.query_one("#speed-max-tokens", Input).value,
            "temperature": self.query_one("#speed-temperature", Input).value,
            "top_p": self.query_one("#speed-top-p", Input).value,
            "seed": seed_text if seed_text else None,
            "request_timeout_seconds": self.query_one("#speed-timeout", Input).value,
            "continue_on_error": self.query_one("#speed-continue", Switch).value,
            "thinking_mode": (
                "disabled"
                if self.query_one("#speed-disable-thinking", Switch).value
                else "server_default"
            ),
        }
        try:
            return SpeedTestConfig.model_validate(data)
        except ValidationError as error:
            if notify:
                issue = error.errors()[0]
                field = str(issue["loc"][0])
                message = str(issue["msg"]).removeprefix("Value error, ")
                self.notify(
                    f"{field.replace('_', ' ').title()}: {message}",
                    title="Invalid Speed Test configuration",
                    severity="error",
                )
                selector = self._FIELD_WIDGETS.get(field)
                if selector is not None:
                    self.query_one(selector).focus()
            return None

    def _update_plan(self) -> None:
        config = self._parse_config(notify=False)
        plan = self.query_one("#speed-run-plan", Static)
        if config is None:
            plan.update("Run plan: fix invalid fields before starting")
            return
        plan.update(
            f"Run plan: {config.warmup_runs} warm-up + {config.measured_runs} "
            f"measured · sequential · {config.max_tokens} max tokens"
        )
