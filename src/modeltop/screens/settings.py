"""Read-only application configuration workspace."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from modeltop.models import ModelTopConfig, ServerConfig
from modeltop.state import ApplicationState


class SettingsView(VerticalScroll):
    """Describe YAML-backed runtime configuration without duplicating editors."""

    DEFAULT_CSS = """
    SettingsView {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        background: $catppuccin-base;
        overflow-x: hidden;
    }
    SettingsView .settings-heading { height: 1; color: $primary; text-style: bold; }
    SettingsView #settings-content { height: auto; }
    """

    def __init__(
        self,
        config: ModelTopConfig,
        server: ServerConfig,
        source: Path | None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._config = config
        self._server = server
        self._source = source

    def compose(self) -> ComposeResult:
        yield Static("SETTINGS · READ ONLY", classes="settings-heading")
        yield Static("", id="settings-content", markup=False)

    def update_state(self, state: ApplicationState) -> None:
        source = str(self._source) if self._source is not None else "provided in memory"
        hardware = "enabled" if self._config.hardware.enabled else "disabled"
        selected = state.selected_model_id or "--"
        concurrency = self._config.benchmarks.concurrency
        context = self._config.benchmarks.context
        tool_calling = self._config.benchmarks.tool_calling
        r0b0bench = self._config.benchmarks.r0b0bench
        levels = ", ".join(str(level) for level in concurrency.default_levels)
        lengths = ", ".join(str(length) for length in context.default_lengths)
        positions = ", ".join(context.retrieval.positions)
        r0b0bench_tests = ", ".join(r0b0bench.default_tests)
        base_text = (
            "unset"
            if context.base_text is None
            else f"configured ({len(context.base_text)} characters)"
        )
        retrieval_key = (
            "unset"
            if context.retrieval.key is None
            else f"configured ({len(context.retrieval.key)} characters)"
        )
        self.query_one("#settings-content", Static).update(
            f"CONFIGURATION SOURCE\n{source}\n\n"
            f"SELECTED SERVER\n{self._server.name} ({self._server.id})\n"
            f"Endpoint: {self._server.endpoint_label}\n"
            f"Backend hint: {self._server.backend_label}\n"
            f"Selected model: {selected}\n\n"
            f"RUNTIME\nRefresh interval: "
            f"{self._config.application.refresh_interval_seconds:g}s\n"
            f"Request timeout: {self._config.application.request_timeout_seconds:g}s\n"
            f"Chat request timeout: "
            f"{self._config.application.chat_request_timeout_seconds:g}s\n"
            f"THEME\n{self._config.application.theme}\n"
            f"Hardware monitoring: {hardware}\n"
            f"Hardware refresh: {self._config.hardware.refresh_interval_seconds:g}s\n\n"
            "CONCURRENCY BENCHMARK DEFAULTS\n"
            f"Levels: {levels}\n"
            f"Requests per level: {concurrency.requests_per_level}\n"
            f"Warm-up requests per level: {concurrency.warmup_requests}\n"
            f"Maximum output tokens: {concurrency.max_tokens}\n"
            f"Request timeout: {concurrency.request_timeout_seconds:g}s\n"
            f"Delay between levels: {concurrency.delay_between_levels_seconds:g}s\n"
            f"Safety maximum concurrency: {concurrency.maximum_concurrency}\n"
            f"Unique prompt suffix per request: "
            f"{concurrency.unique_prompt_suffix_per_request}\n\n"
            "CONTEXT BENCHMARK DEFAULTS\n"
            f"Mode: {context.default_mode}\n"
            f"Lengths: {lengths}\n"
            f"Unit: {context.context_unit}\n"
            f"Repetitions per length: {context.repetitions_per_length}\n"
            f"Warm-up requests: {context.warmup_requests}\n"
            f"Content source: {context.content_source}\n"
            f"Base text: {base_text}\n"
            f"Content random seed: {context.random_seed}\n"
            f"Maximum output tokens: {context.maximum_output_tokens}\n"
            f"Temperature: {context.temperature:g}\n"
            f"Top-p: {context.top_p:g}\n"
            f"Generation seed: {context.seed}\n"
            f"Request timeout: {context.request_timeout_seconds:g}s\n"
            f"Delay between lengths: {context.delay_between_lengths_seconds:g}s\n"
            f"Safety maximum tokens: {context.maximum_context_test_tokens}\n"
            f"Warning threshold tokens: {context.warning_threshold_tokens}\n"
            f"Target tolerance: {context.prompt_target_tolerance_percent:g}%\n"
            f"Hardware sample interval: "
            f"{context.hardware_sample_interval_seconds:g}s\n"
            f"Estimated input rate: {context.estimated_input_rate_enabled}\n"
            f"Reuse prompt: {context.reuse_prompt}\n"
            f"Unique per-run suffix: {context.unique_prompt_suffix_per_run}\n"
            f"Early stop: {context.early_stop_enabled}\n"
            f"Continue after timeout: {context.continue_after_timeout}\n"
            f"Probe start/maximum/resolution: {context.probe.start_tokens} / "
            f"{context.probe.maximum_tokens} / {context.probe.resolution_tokens}\n"
            f"Retrieval enabled: {context.retrieval.enabled}\n"
            f"Retrieval positions: {positions}\n"
            f"Retrieval key: {retrieval_key}\n"
            f"Retrieval output tokens: {context.retrieval.maximum_output_tokens}\n"
            f"Case-insensitive match: {context.retrieval.case_insensitive_match}\n"
            f"Containment match: {context.retrieval.containment_match}\n"
            f"Truncation detection: {context.retrieval.truncation_detection}\n"
            f"Regenerate retrieval key: {context.retrieval.regenerate_per_run}\n\n"
            "TOOL CALLING BENCHMARK DEFAULTS\n"
            f"Suite: {tool_calling.default_suite}\n"
            f"Per-request timeout: {tool_calling.request_timeout_seconds:g}s\n"
            "Scoring/generation controls: fixed by the benchmark integration\n\n"
            "R0B0BENCH DEFAULTS\n"
            f"Profile: {r0b0bench.default_profile}\n"
            f"Selected tests: {r0b0bench_tests}\n"
            f"Per-request timeout: {r0b0bench.request_timeout_seconds:g}s\n"
            "Upstream: r0b0bench 1.0.0rc2\n"
            "Commit: d5ed83d8499a952546cf458e090be42ee4a48eef\n"
            "Report schema: 2\n"
            "Profiles core-subset/core: systems plus quality.\n"
            "Profile systems: the seven systems tests.\n"
            "Filtered or perf runs are diagnostic; publishing is invalid.\n"
            "BFCL needs pinned Python and official adapter scripts.\n"
            "Quality tests need operator-provided local JSONL datasets.\n"
            "NIAH needs a local tokenizer and advertised maximum context.\n"
            "HumanEval runs generated Python without a hardened sandbox.\n"
            "Authenticated endpoints are unsupported by upstream rc2.\n"
            "Raw evidence: ~/.local/share/modeltop/r0b0bench\n"
            "Evidence may contain prompts, responses, or generated code.\n\n"
            "Configuration editing remains YAML-based.\n"
            "Edit YAML and restart ModelTop to apply a theme change.\n"
            "Use Chat · Ctrl+G for in-memory generation settings."
        )
