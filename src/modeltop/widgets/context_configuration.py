"""Keyboard-first validated Context benchmark configuration editor."""

from typing import Literal, cast, overload

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from modeltop.benchmarks.context_builder import (
    MAX_BUILD_FRAGMENTS,
    MAX_BUILD_ITERATIONS,
    RETRIEVAL_PREVIEW_CHARACTERS,
)
from modeltop.benchmarks.models import (
    ContextBenchmarkConfig,
    ContextContentSource,
    ContextMode,
    ContextUnit,
    RetrievalPosition,
)
from modeltop.messages import ContextBenchmarkStartRequested


class ContextConfigurationPanel(VerticalScroll):
    """Own mode-specific drafts until one complete immutable config validates."""

    DEFAULT_CSS = """
    ContextConfigurationPanel { width: 1fr; height: 1fr; padding: 0 1; }
    ContextConfigurationPanel .section-title {
        height: 1; color: #5da9e9; text-style: bold;
    }
    ContextConfigurationPanel .field-row { height: 3; width: 1fr; }
    ContextConfigurationPanel .field-row Label { width: 27; padding-top: 1; }
    ContextConfigurationPanel .field-row Input { width: 1fr; }
    ContextConfigurationPanel OptionList { height: 5; border: solid #2d3b49; }
    ContextConfigurationPanel #context-base-text { height: 5; }
    ContextConfigurationPanel .facts { height: auto; color: #8fa3b8; }
    ContextConfigurationPanel #context-error { height: auto; color: #ef6b73; }
    ContextConfigurationPanel #context-run { margin: 1 0; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._mode: ContextMode = "sweep"
        self._length_drafts: dict[ContextMode, str] = {
            "fixed": "1024",
            "sweep": "1024, 4096, 8192, 16384, 32768",
            "probe": "1024, 4096",
            "retrieval": "1024, 4096, 8192",
        }
        self._loaded = False

    @staticmethod
    def _row(label: str, widget: Input) -> Horizontal:
        return Horizontal(Label(label), widget, classes="field-row")

    def compose(self) -> ComposeResult:
        yield Static("CONTEXT LENGTH CONFIGURATION", classes="section-title")
        yield Label("Mode")
        yield OptionList(
            Option("Fixed", id="fixed"),
            Option("Sweep", id="sweep"),
            Option("Probe", id="probe"),
            Option("Retrieval", id="retrieval"),
            id="context-mode",
            compact=True,
        )
        yield Label("Unit")
        yield OptionList(
            Option("Tokens", id="tokens"),
            Option("Characters", id="characters"),
            id="context-unit",
            compact=True,
        )
        yield self._row("Target lengths", Input(id="context-lengths"))
        yield self._row("Repetitions / length", Input(id="context-repetitions"))
        yield self._row("Warm-up requests", Input(id="context-warmups"))
        yield Label("Content source")
        yield OptionList(
            Option("Synthetic", id="synthetic"),
            Option("Repeated text", id="repeated_text"),
            Option("Built-in corpus", id="built_in_corpus"),
            id="context-source",
            compact=True,
        )
        yield Label("Repeated base text")
        yield TextArea(id="context-base-text")
        yield self._row("Content random seed", Input(id="context-random-seed"))
        yield self._row("Generation seed (blank none)", Input(id="context-seed"))
        yield self._row("Performance output tokens", Input(id="context-output"))
        yield self._row("Temperature", Input(id="context-temperature"))
        yield self._row("Top-p", Input(id="context-top-p"))
        yield self._row("Request timeout seconds", Input(id="context-timeout"))
        yield self._row("Delay between lengths", Input(id="context-delay"))
        yield Static("PROBE", classes="section-title")
        yield self._row("Probe start tokens", Input(id="context-probe-start"))
        yield self._row("Probe maximum tokens", Input(id="context-probe-maximum"))
        yield self._row("Probe resolution tokens", Input(id="context-probe-resolution"))
        yield Static("RETRIEVAL", classes="section-title")
        yield self._row("Positions (comma separated)", Input(id="context-positions"))
        yield self._row("Manual key (optional)", Input(id="context-key"))
        yield self._row("Retrieval output tokens", Input(id="context-retrieval-output"))
        yield Checkbox("Case-insensitive match", id="context-case-insensitive")
        yield Checkbox("Allow unambiguous containment", id="context-containment")
        yield Checkbox("Tri-marker truncation detection", id="context-truncation")
        yield Checkbox("Regenerate auto-key per run", id="context-regenerate")
        yield Static("EXECUTION", classes="section-title")
        yield Checkbox("Reuse byte-identical prompt", id="context-reuse")
        yield Checkbox("Unique deterministic RUN suffix", id="context-unique")
        yield Checkbox("Show estimated input rate", id="context-input-rate")
        yield Checkbox("Early stop at rejection boundary", id="context-early-stop")
        yield Checkbox("Continue after timeout", id="context-continue-timeout")
        yield Static("", id="context-plan-facts", classes="facts", markup=False)
        yield Static("", id="context-error", markup=False)
        yield Button("Run Context Benchmark", id="context-run", variant="primary")

    def on_mount(self) -> None:
        self.load_config(ContextBenchmarkConfig())

    @staticmethod
    def _select(option_list: OptionList, option_id: str) -> None:
        for index in range(option_list.option_count):
            option = option_list.get_option_at_index(index)
            if option.id == option_id:
                option_list.highlighted = index
                return

    def load_config(self, config: ContextBenchmarkConfig) -> None:
        """Load one validated config while preserving all mode draft slots."""
        self._loaded = False
        self._mode = config.mode
        self._length_drafts[config.mode] = ", ".join(
            str(value) for value in config.target_lengths
        )
        self._select(self.query_one("#context-mode", OptionList), config.mode)
        self._select(self.query_one("#context-unit", OptionList), config.context_unit)
        self._select(
            self.query_one("#context-source", OptionList), config.content_source
        )
        values = {
            "#context-lengths": self._length_drafts[config.mode],
            "#context-repetitions": str(config.repetitions_per_length),
            "#context-warmups": str(config.warmup_requests),
            "#context-random-seed": str(config.random_seed),
            "#context-seed": "" if config.seed is None else str(config.seed),
            "#context-output": str(config.maximum_output_tokens),
            "#context-temperature": str(config.temperature),
            "#context-top-p": str(config.top_p),
            "#context-timeout": str(config.request_timeout_seconds),
            "#context-delay": str(config.delay_between_lengths_seconds),
            "#context-probe-start": str(config.probe_start_tokens),
            "#context-probe-maximum": str(config.probe_maximum_tokens),
            "#context-probe-resolution": str(config.probe_resolution_tokens),
            "#context-positions": ", ".join(config.retrieval_positions),
            "#context-key": config.retrieval_key or "",
            "#context-retrieval-output": str(config.retrieval_maximum_output_tokens),
        }
        for selector, value in values.items():
            self.query_one(selector, Input).value = value
        self.query_one("#context-base-text", TextArea).text = config.base_text or ""
        checks = {
            "#context-case-insensitive": config.retrieval_case_insensitive_match,
            "#context-containment": config.retrieval_containment_match,
            "#context-truncation": config.retrieval_truncation_detection,
            "#context-regenerate": config.retrieval_regenerate_per_run,
            "#context-reuse": config.reuse_prompt,
            "#context-unique": config.unique_prompt_suffix_per_run,
            "#context-input-rate": config.estimated_input_rate_enabled,
            "#context-early-stop": config.early_stop_enabled,
            "#context-continue-timeout": config.continue_after_timeout,
        }
        for selector, value in checks.items():
            self.query_one(selector, Checkbox).value = value
        self._maximum_context = config.maximum_context_test_tokens
        self._warning_threshold = config.warning_threshold_tokens
        self._tolerance = config.prompt_target_tolerance_percent
        self._hardware_interval = config.hardware_sample_interval_seconds
        self._loaded = True
        self._update_visibility()
        self._update_facts()

    @on(OptionList.OptionSelected, "#context-mode")
    def mode_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if not self._loaded or event.option.id is None:
            return
        self._length_drafts[self._mode] = self.query_one(
            "#context-lengths", Input
        ).value
        self._mode = event.option.id  # type: ignore[assignment]
        self.query_one("#context-lengths", Input).value = self._length_drafts[
            self._mode
        ]
        if self._mode == "probe":
            self._select(self.query_one("#context-unit", OptionList), "tokens")
        self._update_visibility()

    @on(OptionList.OptionSelected, "#context-source")
    def source_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._update_visibility()

    @on(Button.Pressed, "#context-run")
    def run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        config = self.parse_config(notify=True)
        if config is not None:
            self.post_message(ContextBenchmarkStartRequested(config))

    def focus_mode(self) -> None:
        self.query_one("#context-mode", OptionList).focus()

    @staticmethod
    @overload
    def _integer(
        value: str, field: str, *, optional: Literal[False] = False
    ) -> int: ...

    @staticmethod
    @overload
    def _integer(value: str, field: str, *, optional: Literal[True]) -> int | None: ...

    @staticmethod
    def _integer(value: str, field: str, *, optional: bool = False) -> int | None:
        stripped = value.strip()
        if optional and not stripped:
            return None
        if not stripped or not stripped.isascii() or not stripped.lstrip("-").isdigit():
            raise ValueError(f"{field} must be a base-10 integer")
        return int(stripped, 10)

    @staticmethod
    def _number(value: str, field: str) -> float:
        try:
            return float(value.strip())
        except ValueError as error:
            raise ValueError(f"{field} must be a number") from error

    def _option(self, selector: str) -> str:
        option_list = self.query_one(selector, OptionList)
        option = option_list.get_option_at_index(option_list.highlighted or 0)
        return str(option.id)

    def _lengths(self) -> tuple[int, ...]:
        values = [
            part.strip()
            for part in self.query_one("#context-lengths", Input).value.split(",")
        ]
        if any(
            not value or not value.isascii() or not value.isdigit() for value in values
        ):
            raise ValueError("Target lengths must be comma-separated base-10 integers")
        parsed = [int(value, 10) for value in values]
        if len(parsed) != len(set(parsed)):
            raise ValueError("Target lengths must not contain duplicates")
        return tuple(sorted(parsed))

    def parse_config(self, *, notify: bool = False) -> ContextBenchmarkConfig | None:
        """Parse the current draft through the complete runtime model."""
        try:
            source = self._option("#context-source")
            key = self.query_one("#context-key", Input).value.strip() or None
            config = ContextBenchmarkConfig(
                mode=self._mode,
                target_lengths=self._lengths(),
                context_unit=cast(ContextUnit, self._option("#context-unit")),
                repetitions_per_length=self._integer(
                    self.query_one("#context-repetitions", Input).value, "Repetitions"
                ),
                warmup_requests=self._integer(
                    self.query_one("#context-warmups", Input).value, "Warm-ups"
                ),
                content_source=cast(ContextContentSource, source),
                base_text=self.query_one("#context-base-text", TextArea).text
                if source == "repeated_text"
                else None,
                random_seed=self._integer(
                    self.query_one("#context-random-seed", Input).value, "Random seed"
                ),
                maximum_output_tokens=self._integer(
                    self.query_one("#context-output", Input).value, "Output tokens"
                ),
                temperature=self._number(
                    self.query_one("#context-temperature", Input).value, "Temperature"
                ),
                top_p=self._number(
                    self.query_one("#context-top-p", Input).value, "Top-p"
                ),
                seed=self._integer(
                    self.query_one("#context-seed", Input).value,
                    "Generation seed",
                    optional=True,
                ),
                request_timeout_seconds=self._number(
                    self.query_one("#context-timeout", Input).value, "Timeout"
                ),
                delay_between_lengths_seconds=self._number(
                    self.query_one("#context-delay", Input).value, "Delay"
                ),
                maximum_context_test_tokens=self._maximum_context,
                warning_threshold_tokens=self._warning_threshold,
                prompt_target_tolerance_percent=self._tolerance,
                hardware_sample_interval_seconds=self._hardware_interval,
                estimated_input_rate_enabled=self.query_one(
                    "#context-input-rate", Checkbox
                ).value,
                reuse_prompt=self.query_one("#context-reuse", Checkbox).value,
                unique_prompt_suffix_per_run=self.query_one(
                    "#context-unique", Checkbox
                ).value,
                early_stop_enabled=self.query_one(
                    "#context-early-stop", Checkbox
                ).value,
                continue_after_timeout=self.query_one(
                    "#context-continue-timeout", Checkbox
                ).value,
                probe_start_tokens=self._integer(
                    self.query_one("#context-probe-start", Input).value, "Probe start"
                ),
                probe_maximum_tokens=self._integer(
                    self.query_one("#context-probe-maximum", Input).value,
                    "Probe maximum",
                ),
                probe_resolution_tokens=self._integer(
                    self.query_one("#context-probe-resolution", Input).value,
                    "Probe resolution",
                ),
                retrieval_enabled=self._mode == "retrieval",
                retrieval_positions=cast(
                    tuple[RetrievalPosition, ...],
                    tuple(
                        part.strip()
                        for part in self.query_one(
                            "#context-positions", Input
                        ).value.split(",")
                        if part.strip()
                    ),
                ),
                retrieval_key=key if self._mode == "retrieval" else None,
                retrieval_maximum_output_tokens=self._integer(
                    self.query_one("#context-retrieval-output", Input).value,
                    "Retrieval output",
                ),
                retrieval_case_insensitive_match=self.query_one(
                    "#context-case-insensitive", Checkbox
                ).value,
                retrieval_containment_match=self.query_one(
                    "#context-containment", Checkbox
                ).value,
                retrieval_truncation_detection=self.query_one(
                    "#context-truncation", Checkbox
                ).value,
                retrieval_regenerate_per_run=self.query_one(
                    "#context-regenerate", Checkbox
                ).value,
            )
        except (ValueError, ValidationError) as error:
            message = str(error).splitlines()[0]
            self.query_one("#context-error", Static).update(message)
            if notify:
                self.notify(message, title="Context configuration", severity="error")
            return None
        self.query_one("#context-error", Static).update("")
        return config

    def _update_visibility(self) -> None:
        probe = self._mode == "probe"
        retrieval = self._mode == "retrieval"
        for selector in (
            "#context-probe-start",
            "#context-probe-maximum",
            "#context-probe-resolution",
        ):
            widget = self.query_one(selector)
            assert widget.parent is not None
            widget.parent.display = probe
        for selector in (
            "#context-positions",
            "#context-key",
            "#context-retrieval-output",
        ):
            widget = self.query_one(selector)
            assert widget.parent is not None
            widget.parent.display = retrieval
        for selector in (
            "#context-case-insensitive",
            "#context-containment",
            "#context-truncation",
            "#context-regenerate",
        ):
            self.query_one(selector).display = retrieval
        self.query_one("#context-base-text").display = (
            self._option("#context-source") == "repeated_text"
        )

    def _update_facts(self) -> None:
        self.query_one("#context-plan-facts", Static).update(
            f"Safety maximum: {self._maximum_context} tokens · "
            f"Warning: {self._warning_threshold} tokens\n"
            f"Target tolerance: {self._tolerance:g}% · "
            f"Hardware interval: {self._hardware_interval:g}s\n"
            f"Builder caps: {MAX_BUILD_ITERATIONS} iterations / "
            f"{MAX_BUILD_FRAGMENTS} fragments · "
            f"Retrieval preview: {RETRIEVAL_PREVIEW_CHARACTERS} characters"
        )
