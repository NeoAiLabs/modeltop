"""Selectable, prerequisite-aware configuration for pinned r0b0bench rc2."""

from __future__ import annotations

from typing import cast

from pydantic import ValidationError
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from modeltop.benchmarks.models import R0b0benchBenchmarkConfig
from modeltop.benchmarks.r0b0bench import resolve_r0b0bench_prerequisites
from modeltop.benchmarks.r0b0bench_contract import (
    R0B0BENCH_QUALITY_ORDER,
    R0B0BENCH_SYSTEMS_ORDER,
    R0b0benchLaneId,
    R0b0benchProfile,
    r0b0bench_profile_lanes,
)
from modeltop.messages import R0b0benchBenchmarkStartRequested
from modeltop.services.r0b0bench_datasets import (
    r0b0bench_asset_status,
    r0b0bench_installed_paths,
)

_PROFILE_LABELS: tuple[tuple[str, R0b0benchProfile], ...] = (
    ("Core Subset", "core-subset"),
    ("Core", "core"),
    ("Systems", "systems"),
)
_LANE_LABELS: dict[R0b0benchLaneId, str] = {
    "canary": "Canary",
    "bfcl_mt": "BFCL multi-turn",
    "bfcl_ast": "BFCL AST",
    "latency": "Latency",
    "concurrency": "Concurrency",
    "throughput": "Throughput",
    "niah": "Needle in a Haystack (NIAH)",
    "qa": "QA (ARC-Easy)",
    "ifeval": "IFEval (lightweight scorer)",
    "humaneval": "HumanEval",
    "gsm8k": "GSM8K",
    "perf": "Perf composite (diagnostic)",
}
_PATH_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("tokenizer_path", "Local tokenizer/model path", "r0b0bench-tokenizer"),
    ("bfcl_python", "BFCL Python", "r0b0bench-bfcl-python"),
    (
        "bfcl_scripts_directory",
        "BFCL scripts directory",
        "r0b0bench-bfcl-scripts",
    ),
    ("qa_data_path", "QA JSONL", "r0b0bench-qa-data"),
    ("ifeval_data_path", "IFEval JSONL", "r0b0bench-ifeval-data"),
    ("humaneval_data_path", "HumanEval JSONL", "r0b0bench-humaneval-data"),
    ("gsm8k_data_path", "GSM8K JSONL", "r0b0bench-gsm8k-data"),
)
_FIELD_LABELS = {field: label for field, label, _ in _PATH_FIELDS} | {
    "allow_unsafe_humaneval": "HumanEval acknowledgment",
}


class R0b0benchConfigurationPanel(VerticalScroll):
    """Edit one immutable profile and exact lane selection."""

    DEFAULT_CSS = """
    R0b0benchConfigurationPanel { width: 1fr; height: 1fr; padding: 0 1; }
    R0b0benchConfigurationPanel .section-title {
        height: 1; color: $primary; text-style: bold;
    }
    R0b0benchConfigurationPanel .config-row { height: 3; }
    R0b0benchConfigurationPanel .config-label { width: 28; padding: 1 1 0 0; }
    R0b0benchConfigurationPanel Input { width: 1fr; }
    R0b0benchConfigurationPanel OptionList { height: 5; }
    R0b0benchConfigurationPanel .facts { height: auto; color: $text-muted; }
    R0b0benchConfigurationPanel #r0b0bench-error {
        height: auto; min-height: 1; color: $error;
    }
    R0b0benchConfigurationPanel #r0b0bench-actions { height: 3; }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._profile: R0b0benchProfile = "core-subset"
        self._quality_memory: set[R0b0benchLaneId] = set()
        self._updating = False

    @staticmethod
    def _row(label: str, control: Input) -> Horizontal:
        return Horizontal(
            Label(label, classes="config-label"), control, classes="config-row"
        )

    def compose(self) -> ComposeResult:
        yield Static("R0B0BENCH CONFIGURATION", classes="section-title")
        yield Static("PROFILE", classes="section-title")
        yield OptionList(
            *(Option(label, id=profile) for label, profile in _PROFILE_LABELS),
            id="r0b0bench-profile",
        )
        yield Static("SYSTEMS TESTS", classes="section-title")
        for lane in R0B0BENCH_SYSTEMS_ORDER:
            yield Checkbox(_LANE_LABELS[lane], id=f"r0b0bench-lane-{lane}")
        yield Static("QUALITY TESTS", classes="section-title")
        for lane in R0B0BENCH_QUALITY_ORDER:
            yield Checkbox(_LANE_LABELS[lane], id=f"r0b0bench-lane-{lane}")
        yield Static("DIAGNOSTIC EXTRA", classes="section-title")
        yield Checkbox(_LANE_LABELS["perf"], id="r0b0bench-lane-perf")
        yield Button(
            "Select all profile tests", id="r0b0bench-select-all", variant="default"
        )
        yield Static("EXECUTION", classes="section-title")
        yield self._row("Request timeout (seconds)", Input(id="r0b0bench-timeout"))
        for _, label, identifier in _PATH_FIELDS:
            yield self._row(label, Input(id=identifier))
        yield Checkbox(
            "Allow HumanEval to execute model-generated Python outside a "
            "hardened sandbox",
            id="r0b0bench-allow-humaneval",
        )
        yield Static("", id="r0b0bench-assets", classes="facts", markup=False)
        yield Static(
            "core and core-subset use upstream rc2 semantics. IFEval is the rc2 "
            "lightweight scorer; QA is ARC-Easy-only. Filtered and perf runs are "
            "diagnostic. NIAH uses the endpoint's advertised maximum context and "
            "local tokenizer code. Raw private traces may contain prompts, responses, "
            "or generated code.",
            classes="facts",
            markup=False,
        )
        yield Static("", id="r0b0bench-error", markup=False)
        yield Horizontal(
            Button("Run r0b0bench", id="r0b0bench-run", variant="primary"),
            id="r0b0bench-actions",
        )

    def on_mount(self) -> None:
        self.load_config(R0b0benchBenchmarkConfig())

    @staticmethod
    def _select_profile(option_list: OptionList, profile: str) -> None:
        for index in range(option_list.option_count):
            option = option_list.get_option_at_index(index)
            if option.id == profile:
                option_list.highlighted = index
                return

    def _checkbox(self, lane: R0b0benchLaneId) -> Checkbox:
        return self.query_one(f"#r0b0bench-lane-{lane}", Checkbox)

    def _selected_lanes(self) -> tuple[R0b0benchLaneId, ...]:
        order = (*R0B0BENCH_SYSTEMS_ORDER, *R0B0BENCH_QUALITY_ORDER, "perf")
        return tuple(lane for lane in order if self._checkbox(lane).value)

    def load_config(self, config: R0b0benchBenchmarkConfig) -> None:
        """Restore every draft field and exact lane selection."""
        self._updating = True
        self._profile = config.profile
        self._select_profile(
            self.query_one("#r0b0bench-profile", OptionList), config.profile
        )
        selected = set(config.selected_lanes)
        self._quality_memory = selected.intersection(R0B0BENCH_QUALITY_ORDER)
        for lane in (*R0B0BENCH_SYSTEMS_ORDER, *R0B0BENCH_QUALITY_ORDER, "perf"):
            self._checkbox(lane).value = lane in selected
        self.query_one("#r0b0bench-timeout", Input).value = str(
            config.request_timeout_seconds
        )
        installed = r0b0bench_installed_paths()
        for field_name, _, identifier in _PATH_FIELDS:
            value = cast(str | None, getattr(config, field_name)) or installed.get(
                field_name
            )
            self.query_one(f"#{identifier}", Input).value = value or ""
        self.query_one(
            "#r0b0bench-allow-humaneval", Checkbox
        ).value = config.allow_unsafe_humaneval
        self._apply_profile_state(restore_quality=False)
        self._apply_exclusivity()
        self.query_one("#r0b0bench-error", Static).update("")
        self.query_one("#r0b0bench-assets", Static).update(
            "LOCAL ASSETS · "
            + " · ".join(
                f"{row.label}: {row.state.upper()}" for row in r0b0bench_asset_status()
            )
        )
        self._updating = False

    def _apply_profile_state(self, *, restore_quality: bool) -> None:
        systems_only = self._profile == "systems"
        for lane in R0B0BENCH_QUALITY_ORDER:
            checkbox = self._checkbox(lane)
            if systems_only:
                if checkbox.value:
                    self._quality_memory.add(lane)
                checkbox.value = False
                checkbox.disabled = True
            else:
                checkbox.disabled = False
                if restore_quality:
                    checkbox.value = lane in self._quality_memory
        self._update_humaneval_ack()

    def _apply_exclusivity(self) -> None:
        perf = self._checkbox("perf")
        components = tuple(
            self._checkbox(lane) for lane in ("latency", "concurrency", "throughput")
        )
        if perf.value:
            for checkbox in components:
                checkbox.value = False
                checkbox.disabled = True
        else:
            for checkbox in components:
                checkbox.disabled = False

    def _update_humaneval_ack(self) -> None:
        acknowledgment = self.query_one("#r0b0bench-allow-humaneval", Checkbox)
        acknowledgment.disabled = not self._checkbox("humaneval").value

    @on(OptionList.OptionSelected, "#r0b0bench-profile")
    def profile_changed(self, event: OptionList.OptionSelected) -> None:
        if self._updating or event.option.id is None:
            return
        previous = self._profile
        self._profile = cast(R0b0benchProfile, str(event.option.id))
        self._updating = True
        self._apply_profile_state(restore_quality=previous == "systems")
        self._updating = False

    @on(Checkbox.Changed)
    def checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self._updating or not event.checkbox.id:
            return
        identifier = event.checkbox.id
        if not identifier.startswith("r0b0bench-lane-"):
            return
        lane = cast(R0b0benchLaneId, identifier.removeprefix("r0b0bench-lane-"))
        self._updating = True
        if lane in R0B0BENCH_QUALITY_ORDER and event.value:
            self._quality_memory.add(lane)
        elif lane in R0B0BENCH_QUALITY_ORDER:
            self._quality_memory.discard(lane)
        if lane == "perf" and event.value:
            for component in ("latency", "concurrency", "throughput"):
                self._checkbox(component).value = False
        elif lane in {"latency", "concurrency", "throughput"} and event.value:
            self._checkbox("perf").value = False
        self._apply_exclusivity()
        self._update_humaneval_ack()
        self._updating = False

    @on(Button.Pressed, "#r0b0bench-select-all")
    def select_all(self, event: Button.Pressed) -> None:
        event.stop()
        self._updating = True
        allowed = set(r0b0bench_profile_lanes(self._profile))
        self._checkbox("perf").value = False
        for lane in (*R0B0BENCH_SYSTEMS_ORDER, *R0B0BENCH_QUALITY_ORDER):
            self._checkbox(lane).value = lane in allowed
        self._quality_memory = allowed.intersection(R0B0BENCH_QUALITY_ORDER)
        self._apply_profile_state(restore_quality=False)
        self._apply_exclusivity()
        self._updating = False

    def parse_config(self, *, notify: bool = False) -> R0b0benchBenchmarkConfig | None:
        """Parse the draft and report one bounded validation/prerequisite issue."""
        error = self.query_one("#r0b0bench-error", Static)
        timeout_text = self.query_one("#r0b0bench-timeout", Input).value.strip()
        try:
            timeout = float(timeout_text)
        except ValueError:
            if notify:
                error.update("Request timeout must be a finite positive number.")
            return None
        values: dict[str, object] = {
            "profile": self._profile,
            "selected_lanes": self._selected_lanes(),
            "request_timeout_seconds": timeout,
            "allow_unsafe_humaneval": self.query_one(
                "#r0b0bench-allow-humaneval", Checkbox
            ).value,
        }
        for field_name, _, identifier in _PATH_FIELDS:
            text = self.query_one(f"#{identifier}", Input).value.strip()
            values[field_name] = text or None
        try:
            config = R0b0benchBenchmarkConfig.model_validate(values)
        except ValidationError as validation_error:
            message = validation_error.errors()[0].get("msg", "Invalid configuration")
            if notify:
                error.update(str(message).removeprefix("Value error, "))
            return None
        check = resolve_r0b0bench_prerequisites(config)
        if check.issues:
            issue = check.issues[0]
            lane = "r0b0bench" if issue.lane_id is None else _LANE_LABELS[issue.lane_id]
            label = _FIELD_LABELS.get(issue.field_name, issue.field_name)
            if notify:
                error.update(f"{lane}: configure {label}.")
            return None
        error.update("")
        return config

    def focus_profile(self) -> None:
        self.query_one("#r0b0bench-profile", OptionList).focus()

    @on(Button.Pressed, "#r0b0bench-run")
    def run_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        config = self.parse_config(notify=True)
        if config is not None:
            self.post_message(R0b0benchBenchmarkStartRequested(config))
