"""Editable validated configuration for Concurrency benchmarks."""

from typing import ClassVar, cast

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, OptionList, Static, Switch, TextArea
from textual.widgets.option_list import Option

from modeltop.benchmarks.models import ConcurrencyBenchmarkConfig, ConcurrencyMode
from modeltop.messages import ConcurrencyBenchmarkStartRequested


class BenchmarkConfigurationPanel(VerticalScroll):
    """Keep fixed and sweep drafts local until one validated Start message."""

    DEFAULT_CSS = """
    BenchmarkConfigurationPanel { width: 1fr; height: 1fr; padding: 1 2; }
    BenchmarkConfigurationPanel .section-title { color: $primary; text-style: bold; }
    BenchmarkConfigurationPanel .config-row { height: 3; width: 1fr; }
    BenchmarkConfigurationPanel .config-row Label { width: 26; padding-top: 1; }
    BenchmarkConfigurationPanel .config-row Input { width: 1fr; }
    BenchmarkConfigurationPanel TextArea { height: 5; border: solid $border-blurred; }
    BenchmarkConfigurationPanel #concurrency-mode { height: 4; }
    BenchmarkConfigurationPanel #concurrency-run-plan { margin: 1 0; }
    BenchmarkConfigurationPanel .hidden { display: none; }
    """

    _FIELD_WIDGETS: ClassVar[dict[str, str]] = {
        "prompt": "#concurrency-prompt",
        "concurrency_levels": "#concurrency-levels",
        "requests_per_level": "#concurrency-requests",
        "warmup_requests": "#concurrency-warmups",
        "max_tokens": "#concurrency-max-tokens",
        "temperature": "#concurrency-temperature",
        "top_p": "#concurrency-top-p",
        "seed": "#concurrency-seed",
        "request_timeout_seconds": "#concurrency-timeout",
        "delay_between_levels_seconds": "#concurrency-delay",
    }

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._mode: ConcurrencyMode = "sweep"
        self._fixed_draft = "1"
        self._levels_draft = "1, 2, 4, 8"
        self._loaded = False
        self._maximum_concurrency = 128

    def compose(self) -> ComposeResult:
        yield Static("CONCURRENCY BENCHMARK CONFIGURATION", classes="section-title")
        yield OptionList(
            Option("Fixed", id="fixed"),
            Option("Sweep", id="sweep"),
            id="concurrency-mode",
            compact=True,
            markup=False,
        )
        with Horizontal(id="concurrency-fixed-row", classes="config-row hidden"):
            yield Label("Maximum simultaneous requests")
            yield Input(id="concurrency-fixed")
        with Horizontal(id="concurrency-levels-row", classes="config-row"):
            yield Label("Concurrency levels")
            yield Input(id="concurrency-levels")
        for label, identifier in (
            ("Requests per level", "concurrency-requests"),
            ("Warm-up requests", "concurrency-warmups"),
            ("Max tokens", "concurrency-max-tokens"),
            ("Temperature", "concurrency-temperature"),
            ("Top-p", "concurrency-top-p"),
            ("Seed (optional)", "concurrency-seed"),
            ("Request timeout seconds", "concurrency-timeout"),
            ("Delay between levels", "concurrency-delay"),
        ):
            with Horizontal(classes="config-row"):
                yield Label(label, id=f"{identifier}-label")
                yield Input(id=identifier)
        with Horizontal(classes="config-row"):
            yield Label("Disable thinking (Qwen/vLLM)")
            yield Switch(id="concurrency-disable-thinking")
        yield Label("Optional system prompt", classes="section-title")
        yield TextArea(
            id="concurrency-system-prompt", soft_wrap=True, show_line_numbers=False
        )
        yield Label("Fixed user prompt", classes="section-title")
        yield TextArea(id="concurrency-prompt", soft_wrap=True, show_line_numbers=False)
        yield Static("", id="concurrency-run-plan", markup=False)
        yield Button(
            "Run Concurrency Benchmark", id="concurrency-run", variant="primary"
        )

    def on_mount(self) -> None:
        self.load_config(ConcurrencyBenchmarkConfig())

    def load_config(self, config: ConcurrencyBenchmarkConfig) -> None:
        """Replace the displayed draft while preserving inactive mode text."""
        self._loaded = False
        self._mode = config.mode
        self._maximum_concurrency = config.maximum_concurrency
        if config.mode == "fixed":
            self._fixed_draft = str(config.concurrency_levels[0])
        else:
            self._levels_draft = ", ".join(map(str, config.concurrency_levels))
        self.query_one("#concurrency-fixed", Input).value = self._fixed_draft
        self.query_one("#concurrency-levels", Input).value = self._levels_draft
        values = {
            "#concurrency-requests": str(config.requests_per_level),
            "#concurrency-warmups": str(config.warmup_requests),
            "#concurrency-max-tokens": str(config.max_tokens),
            "#concurrency-temperature": str(config.temperature),
            "#concurrency-top-p": str(config.top_p),
            "#concurrency-seed": "" if config.seed is None else str(config.seed),
            "#concurrency-timeout": str(config.request_timeout_seconds),
            "#concurrency-delay": str(config.delay_between_levels_seconds),
        }
        for selector, value in values.items():
            self.query_one(selector, Input).value = value
        self.query_one("#concurrency-system-prompt", TextArea).text = (
            config.system_prompt or ""
        )
        self.query_one("#concurrency-prompt", TextArea).text = config.prompt
        self.query_one("#concurrency-disable-thinking", Switch).value = (
            config.thinking_mode == "disabled"
        )
        self.query_one("#concurrency-mode", OptionList).highlighted = (
            0 if config.mode == "fixed" else 1
        )
        self._loaded = True
        self._update_mode_visibility()
        self._update_plan()

    @on(OptionList.OptionSelected, "#concurrency-mode")
    def select_mode(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id not in {"fixed", "sweep"}:
            return
        self._remember_level_drafts()
        self._mode = cast(ConcurrencyMode, event.option.id)
        self._restore_level_draft()
        self._update_mode_visibility()
        self._update_plan()

    @on(Input.Changed)
    @on(TextArea.Changed)
    @on(Switch.Changed)
    def draft_changed(self) -> None:
        if self._loaded:
            self._remember_level_drafts()
            self._update_plan()

    @on(Button.Pressed, "#concurrency-run")
    def start_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        config = self.parse_config(notify=True)
        if config is not None:
            self.post_message(ConcurrencyBenchmarkStartRequested(config))

    def focus_mode(self) -> None:
        self.query_one("#concurrency-mode", OptionList).focus()

    def _remember_level_drafts(self) -> None:
        if not self._loaded:
            return
        self._fixed_draft = self.query_one("#concurrency-fixed", Input).value
        self._levels_draft = self.query_one("#concurrency-levels", Input).value

    def _restore_level_draft(self) -> None:
        self.query_one("#concurrency-fixed", Input).value = self._fixed_draft
        self.query_one("#concurrency-levels", Input).value = self._levels_draft

    def _update_mode_visibility(self) -> None:
        fixed = self.query_one("#concurrency-fixed-row", Horizontal)
        levels = self.query_one("#concurrency-levels-row", Horizontal)
        fixed.set_class(self._mode != "fixed", "hidden")
        levels.set_class(self._mode != "sweep", "hidden")
        label = self.query_one("#concurrency-requests-label", Label)
        label.update(
            "Total requests" if self._mode == "fixed" else "Requests per level"
        )

    @staticmethod
    def _base10_integer(text: str, field: str, *, optional: bool = False) -> int | None:
        value = text.strip()
        if optional and not value:
            return None
        signless = value[1:] if value.startswith(("+", "-")) else value
        if not signless or not signless.isascii() or not signless.isdecimal():
            raise ValueError(f"{field}: must be a base-10 integer")
        return int(value, 10)

    def _parse_levels(self) -> tuple[int, ...]:
        if self._mode == "fixed":
            value = self._base10_integer(
                self.query_one("#concurrency-fixed", Input).value,
                "Concurrency",
            )
            assert value is not None
            return (value,)
        text = self.query_one("#concurrency-levels", Input).value
        components = [component.strip() for component in text.split(",")]
        if not components or any(not component for component in components):
            raise ValueError("Concurrency levels: use comma-separated integers")
        values = [
            self._base10_integer(component, "Concurrency levels")
            for component in components
        ]
        return tuple(cast(int, value) for value in values)

    def parse_config(
        self, *, notify: bool = False
    ) -> ConcurrencyBenchmarkConfig | None:
        """Parse strict base-10 drafts and validate the complete Pydantic model."""
        try:
            data: dict[str, object] = {
                "mode": self._mode,
                "prompt": self.query_one("#concurrency-prompt", TextArea).text,
                "system_prompt": self.query_one(
                    "#concurrency-system-prompt", TextArea
                ).text,
                "concurrency_levels": self._parse_levels(),
                "requests_per_level": self._base10_integer(
                    self.query_one("#concurrency-requests", Input).value,
                    "Requests",
                ),
                "warmup_requests": self._base10_integer(
                    self.query_one("#concurrency-warmups", Input).value,
                    "Warm-up requests",
                ),
                "max_tokens": self._base10_integer(
                    self.query_one("#concurrency-max-tokens", Input).value,
                    "Max tokens",
                ),
                "temperature": float(
                    self.query_one("#concurrency-temperature", Input).value
                ),
                "top_p": float(self.query_one("#concurrency-top-p", Input).value),
                "seed": self._base10_integer(
                    self.query_one("#concurrency-seed", Input).value,
                    "Seed",
                    optional=True,
                ),
                "request_timeout_seconds": float(
                    self.query_one("#concurrency-timeout", Input).value
                ),
                "delay_between_levels_seconds": float(
                    self.query_one("#concurrency-delay", Input).value
                ),
                "stream": True,
                "maximum_concurrency": self._maximum_concurrency,
                "thinking_mode": (
                    "disabled"
                    if self.query_one("#concurrency-disable-thinking", Switch).value
                    else "server_default"
                ),
            }
            return ConcurrencyBenchmarkConfig.model_validate(data)
        except (ValueError, ValidationError) as error:
            if notify:
                selector: str | None = None
                if isinstance(error, ValidationError):
                    issue = error.errors()[0]
                    field = str(issue["loc"][0])
                    message = str(issue["msg"]).removeprefix("Value error, ")
                    detail = f"{field.replace('_', ' ').title()}: {message}"
                    selector = self._FIELD_WIDGETS.get(field)
                else:
                    detail = str(error)
                self.notify(
                    detail,
                    title="Invalid Concurrency configuration",
                    severity="error",
                )
                if selector is not None:
                    self.query_one(selector).focus()
            return None

    def _update_plan(self) -> None:
        plan = self.query_one("#concurrency-run-plan", Static)
        config = self.parse_config(notify=False)
        if config is None:
            plan.update("Run plan: fix invalid fields before starting")
            return
        levels = config.concurrency_levels
        measured = config.requests_per_level * len(levels)
        warmups = config.warmup_requests * len(levels)
        total = measured + warmups
        warnings: list[str] = []
        if max(levels) >= config.maximum_concurrency * 0.75:
            warnings.append("high concurrency")
        if config.requests_per_level >= 500:
            warnings.append("high request count")
        warning = f" · Warning: {', '.join(warnings)}" if warnings else ""
        plan.update(
            f"Run plan: maximum {max(levels)} simultaneous requests · "
            f"{measured} measured + {warmups} warm-up = {total} total · "
            f"{config.max_tokens} max tokens · safety maximum "
            f"{config.maximum_concurrency}{warning}"
        )
