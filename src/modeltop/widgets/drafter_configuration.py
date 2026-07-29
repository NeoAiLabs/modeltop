"""Editable, validated Drafter benchmark configuration panel."""

from typing import ClassVar

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, Static, Switch, TextArea

from modeltop.benchmarks.models import DrafterBenchmarkConfig
from modeltop.messages import DrafterBenchmarkStartRequested


class DrafterConfigurationPanel(VerticalScroll):
    """Keep editable draft text local until Run validates it once."""

    DEFAULT_CSS = """
    DrafterConfigurationPanel {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        overflow-x: hidden;
        background: #0f151d;
    }
    DrafterConfigurationPanel .section-title {
        height: 1;
        color: #5da9e9;
        text-style: bold;
    }
    DrafterConfigurationPanel #drafter-prompt { height: 5; }
    DrafterConfigurationPanel .config-row { height: 3; }
    DrafterConfigurationPanel .config-row Label { width: 18; padding-top: 1; }
    DrafterConfigurationPanel .config-row Input { width: 1fr; }
    DrafterConfigurationPanel #continue-row { height: 2; }
    DrafterConfigurationPanel #drafter-run-plan { height: 2; color: #7f8c9a; }
    DrafterConfigurationPanel #drafter-run { width: 18; }
    """

    _FIELD_WIDGETS: ClassVar[dict[str, str]] = {
        "prompt": "#drafter-prompt",
        "warmup_runs": "#drafter-warmups",
        "measured_runs": "#drafter-measured",
        "max_tokens": "#drafter-max-tokens",
        "temperature": "#drafter-temperature",
        "top_p": "#drafter-top-p",
        "seed": "#drafter-seed",
        "request_timeout_seconds": "#drafter-timeout",
        "continue_on_error": "#drafter-continue",
    }

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._loaded = False

    def compose(self) -> ComposeResult:
        yield Static("DRAFTER CONFIGURATION", classes="section-title")
        yield Label("Prompt", classes="section-title")
        yield TextArea(id="drafter-prompt", soft_wrap=True, show_line_numbers=False)
        for label, identifier in (
            ("Warm-up runs", "drafter-warmups"),
            ("Measured runs", "drafter-measured"),
            ("Max tokens", "drafter-max-tokens"),
            ("Temperature", "drafter-temperature"),
            ("Top-p", "drafter-top-p"),
            ("Seed (optional)", "drafter-seed"),
            ("Timeout seconds", "drafter-timeout"),
        ):
            with Horizontal(classes="config-row"):
                yield Label(label)
                yield Input(id=identifier)
        with Horizontal(id="continue-row", classes="config-row"):
            yield Label("Continue on error")
            yield Switch(id="drafter-continue")
        yield Static("", id="drafter-run-plan", markup=False)
        yield Button("Run Drafter", id="drafter-run", variant="primary")

    def on_mount(self) -> None:
        self.load_config(DrafterBenchmarkConfig())

    def load_config(self, config: DrafterBenchmarkConfig) -> None:
        """Replace the displayed draft without starting a benchmark."""
        self.query_one("#drafter-prompt", TextArea).text = config.prompt
        values = {
            "#drafter-warmups": str(config.warmup_runs),
            "#drafter-measured": str(config.measured_runs),
            "#drafter-max-tokens": str(config.max_tokens),
            "#drafter-temperature": str(config.temperature),
            "#drafter-top-p": str(config.top_p),
            "#drafter-seed": "" if config.seed is None else str(config.seed),
            "#drafter-timeout": str(config.request_timeout_seconds),
        }
        for selector, value in values.items():
            self.query_one(selector, Input).value = value
        self.query_one("#drafter-continue", Switch).value = config.continue_on_error
        self._loaded = True
        self._update_plan()

    def parse_config(self, *, notify: bool = False) -> DrafterBenchmarkConfig | None:
        """Strictly parse the current draft through the runtime model."""
        return self._parse_config(notify=notify)

    def focus_prompt(self) -> None:
        self.query_one("#drafter-prompt", TextArea).focus()

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Switch.Changed)
    def draft_changed(self) -> None:
        if self._loaded:
            self._update_plan()

    @on(Button.Pressed, "#drafter-run")
    def run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        config = self.parse_config(notify=True)
        if config is not None:
            self.post_message(DrafterBenchmarkStartRequested(config))

    def _parse_config(self, *, notify: bool) -> DrafterBenchmarkConfig | None:
        seed_text = self.query_one("#drafter-seed", Input).value.strip()
        data: dict[str, object] = {
            "prompt": self.query_one("#drafter-prompt", TextArea).text,
            "warmup_runs": self.query_one("#drafter-warmups", Input).value,
            "measured_runs": self.query_one("#drafter-measured", Input).value,
            "max_tokens": self.query_one("#drafter-max-tokens", Input).value,
            "temperature": self.query_one("#drafter-temperature", Input).value,
            "top_p": self.query_one("#drafter-top-p", Input).value,
            "seed": seed_text if seed_text else None,
            "request_timeout_seconds": self.query_one("#drafter-timeout", Input).value,
            "continue_on_error": self.query_one("#drafter-continue", Switch).value,
        }
        try:
            return DrafterBenchmarkConfig.model_validate(data)
        except ValidationError as error:
            if notify:
                issue = error.errors()[0]
                field = str(issue["loc"][0])
                message = str(issue["msg"]).removeprefix("Value error, ")
                self.notify(
                    f"{field.replace('_', ' ').title()}: {message}",
                    title="Invalid Drafter configuration",
                    severity="error",
                )
                selector = self._FIELD_WIDGETS.get(field)
                if selector is not None:
                    self.query_one(selector).focus()
            return None

    def _update_plan(self) -> None:
        config = self._parse_config(notify=False)
        plan = self.query_one("#drafter-run-plan", Static)
        if config is None:
            plan.update("Run plan: fix invalid fields before starting")
            return
        plan.update(
            f"Run plan: {config.warmup_runs} warm-up + {config.measured_runs} "
            f"measured · sequential · {config.max_tokens} max tokens"
        )
