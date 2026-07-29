"""Compact live dashboard status footer."""

from rich.text import Text
from textual.widgets import Static

from modeltop.benchmarks.models import (
    ConcurrencyBenchmarkStatus,
    ContextBenchmarkStatus,
    DrafterBenchmarkStatus,
    SpeedTestResult,
    SpeedTestStatus,
    ToolCallingBenchmarkStatus,
)
from modeltop.chat.models import GenerationStatus
from modeltop.hardware.models import (
    format_byte_pair,
    format_percentage,
    summarize_gpus,
)
from modeltop.models import ServerConfig
from modeltop.state import ApplicationState, ServerStatus


class StatusFooter(Static):
    """Three-row clipped server status bar."""

    DEFAULT_CSS = """
    StatusFooter {
        width: 1fr;
        height: 3;
        background: #111820;
        border: solid #2d3b49;
        content-align: center middle;
        text-align: center;
        text-wrap: nowrap;
        text-overflow: clip;
    }
    """

    def __init__(self) -> None:
        """Initialize an empty status line for the first state render."""
        super().__init__("")
        self._state: ApplicationState | None = None
        self._server: ServerConfig | None = None

    def on_resize(self) -> None:
        if self._state is not None and self._server is not None:
            self.update_state(self._state, self._server)

    @staticmethod
    def _line(status: str, color: str, parts: tuple[str, ...]) -> Text:
        line = Text(status, style=color)
        for part in parts:
            line.append(" │ ", style="#7f8c9a")
            line.append(part, style="#d8dee9")
        return line

    def update_state(
        self,
        state: ApplicationState,
        server: ServerConfig,
    ) -> None:
        """Render a clipped summary for the current state."""
        self._state = state
        self._server = server
        if state.tool_calling_benchmark.is_active:
            self.update(self._tool_calling_line(state))
            return
        if state.drafter_benchmark.is_active:
            self.update(self._drafter_line(state))
            return
        if state.context_benchmark.is_active:
            self.update(self._context_line(state))
            return
        if state.concurrency_benchmark.is_active:
            self.update(self._concurrency_line(state))
            return
        if state.speed_test.is_active:
            self.update(self._speed_line(state, None))
            return
        if state.active_view == "tool-calling":
            self.update(self._tool_calling_line(state))
            return
        if state.active_view == "drafter":
            self.update(self._drafter_line(state))
            return
        if state.active_view == "context":
            self.update(self._context_line(state))
            return
        if state.active_view == "concurrency":
            self.update(self._concurrency_line(state))
            return
        if state.active_view == "speed-test":
            self.update(self._speed_line(state, state.speed_test.latest_result))
            return
        if state.active_view == "results":
            selected = (
                state.speed_test.result_by_id(state.speed_test.selected_result_id)
                if state.speed_test.selected_result_id is not None
                else None
            )
            if selected is not None:
                self.update(self._speed_line(state, selected))
            else:
                count = len(state.speed_test.results)
                self.update(
                    self._line(
                        "RESULTS",
                        "#5da9e9",
                        (
                            f"{count} session result{'s' if count != 1 else ''}",
                            "Enter Open",
                            "Esc Speed Test",
                            "Ctrl+Q Quit",
                        ),
                    )
                )
            return
        if state.active_view == "settings":
            self.update(
                self._line(
                    "SETTINGS",
                    "#5da9e9",
                    ("Read only", "Edit YAML to persist changes", "Ctrl+Q Quit"),
                )
            )
            return
        if state.active_view == "chat":
            self.update(self._chat_line(state, server))
            return
        hardware = self._hardware_parts(state)
        status = state.server_status.value.upper()
        if state.server_status is ServerStatus.ONLINE:
            model = (
                state.selected_model_id.rsplit("/", 1)[-1]
                if state.selected_model_id is not None
                else "No models"
            )
            latency = (
                f"{state.connection_latency_ms:.0f} ms"
                if state.connection_latency_ms is not None
                else "-- ms"
            )
            count = len(state.available_models)
            noun = "model" if count == 1 else "models"
            parts = (
                server.name,
                model,
                latency,
                *hardware,
                f"{count} {noun}",
                "Refreshing..." if state.is_refreshing else "R Refresh",
                "Q Quit",
            )
            color = "#5fd38d"
        elif state.server_status is ServerStatus.CONNECTING:
            parts = (
                server.name,
                server.endpoint_label,
                *hardware,
                "Refreshing...",
                "Q Quit",
            )
            color = "#f2cc60"
        else:
            parts = (
                server.name,
                state.last_error or "--",
                *hardware,
                "R Retry",
                "Q Quit",
            )
            color = "#ef6b73"
        self.update(self._line(status, color, parts))

    def _tool_calling_line(self, state: ApplicationState) -> Text:
        lane = state.tool_calling_benchmark
        progress = lane.progress
        if lane.is_active:
            completed = progress.completed_count if progress is not None else 0
            configured = lane.config.scenario_count
            if progress is None:
                outcomes = "pass 0 · partial 0 · fail 0 · excluded 0"
                coverage = "-- coverage"
            else:
                outcomes = (
                    f"pass {progress.pass_count} · partial {progress.partial_count} · "
                    f"fail {progress.fail_count} · excluded {progress.excluded_count}"
                )
                coverage = f"{progress.completion_rate_percent:.1f}% coverage"
            status = (
                "CANCELLING"
                if lane.status is ToolCallingBenchmarkStatus.CANCELLING
                else lane.status.value.upper()
            )
            return self._line(
                status,
                "#f2cc60",
                (
                    f"{lane.config.suite} {completed}/{configured}",
                    outcomes,
                    coverage,
                    "Esc Cancel",
                ),
            )
        if lane.is_terminal:
            statuses = {
                ToolCallingBenchmarkStatus.COMPLETED: ("COMPLETE", "#5fd38d"),
                ToolCallingBenchmarkStatus.CANCELLED: ("CANCELLED", "#f2cc60"),
                ToolCallingBenchmarkStatus.ERROR: ("ERROR", "#ef6b73"),
            }
            status, color = statuses[lane.status]
            result = lane.latest_result
            score = (
                "-- score"
                if result is None or result.final_score is None
                else f"{result.final_score}% score"
            )
            completion = (
                "-- coverage"
                if result is None or result.completion_rate_percent is None
                else f"{result.completion_rate_percent:.1f}% coverage"
            )
            return self._line(
                status,
                color,
                (score, completion, "R Run Again", "E Edit"),
            )
        return self._line(
            "TOOL CALLING READY",
            "#5da9e9",
            (
                f"{lane.config.suite} · {lane.config.scenario_count} scenarios",
                f"{lane.config.request_timeout_seconds:g}s timeout",
                "R Run",
            ),
        )

    def _drafter_line(self, state: ApplicationState) -> Text:
        lane = state.drafter_benchmark
        progress = lane.progress
        if lane.is_active:
            if lane.status is DrafterBenchmarkStatus.WARMING_UP:
                status = "WARM-UP"
            elif lane.status is DrafterBenchmarkStatus.RUNNING:
                status = "RUNNING"
            elif lane.status is DrafterBenchmarkStatus.CANCELLING:
                status = "CANCELLING"
            else:
                status = "PREPARING"
            if progress is None or progress.current_phase is None:
                phase = "--"
            else:
                phase = f"{progress.current_run}/{progress.phase_total}"
            metrics = progress.latest_metrics if progress is not None else None
            ttft = (
                f"{metrics.ttft_ms:.1f} ms"
                if metrics is not None and metrics.ttft_ms is not None
                else "--"
            )
            speed_label = (
                self._speed_label(
                    metrics.output_tokens_per_second,
                    metrics.completion_tokens_estimated,
                )
                if metrics is not None
                else "-- tok/s"
            )
            rate = (
                f"acc {metrics.acceptance_rate:.2f}"
                if metrics is not None and metrics.acceptance_rate is not None
                else "acc --"
            )
            return self._line(
                status,
                "#f2cc60",
                (
                    phase,
                    f"TTFT {ttft}",
                    speed_label,
                    rate,
                    "Esc Cancel",
                    "Ctrl+Q Quit",
                ),
            )
        if lane.is_terminal and lane.latest_result is not None:
            result = lane.latest_result
            status = result.status.value.replace("_", " ").upper()
            mean = result.acceptance_rate.mean
            mean_label = "acc UNAVAILABLE" if mean is None else f"acc {mean:.2f}"
            tok = result.output_tokens_per_second.mean
            tok_label = "-- tok/s" if tok is None else f"{tok:.1f} tok/s"
            return self._line(
                status,
                (
                    "#5fd38d"
                    if result.status
                    in {
                        DrafterBenchmarkStatus.COMPLETED,
                        DrafterBenchmarkStatus.COMPLETED_WITH_ERRORS,
                    }
                    else "#f2cc60"
                ),
                (
                    f"{result.successful_runs}/{result.measured_runs} successful",
                    tok_label,
                    mean_label,
                    "R Run Again",
                    "E Edit",
                ),
            )
        return self._line(
            "DRAFTER READY",
            "#5da9e9",
            (
                f"{lane.config.measured_runs} measured · "
                f"{lane.config.max_tokens} max tokens",
                f"{lane.config.request_timeout_seconds:g}s timeout",
                "R Run",
            ),
        )

    def _context_line(self, state: ApplicationState) -> Text:
        lane = state.context_benchmark
        progress = lane.progress
        if lane.is_active:
            target = (
                "--"
                if progress is None or progress.active_target_length is None
                else str(progress.active_target_length)
            )
            run = progress.run_number if progress is not None else 0
            configured = progress.configured_runs if progress is not None else 0
            if lane.status is ContextBenchmarkStatus.BETWEEN_LENGTHS:
                remaining = (
                    "--"
                    if progress is None or progress.delay_remaining_seconds is None
                    else f"{progress.delay_remaining_seconds:.1f}s"
                )
                return self._line(
                    "BETWEEN",
                    "#f2cc60",
                    (
                        f"next {progress.next_target_length if progress else '--'}",
                        remaining,
                        "Esc Cancel",
                    ),
                )
            return self._line(
                lane.status.value.replace("_", " ").upper(),
                "#f2cc60",
                (f"target {target}", f"run {run}/{configured}", "Esc Cancel"),
            )
        if lane.is_terminal:
            statuses = {
                ContextBenchmarkStatus.COMPLETED: ("COMPLETE", "#5fd38d"),
                ContextBenchmarkStatus.CANCELLED: ("CANCELLED", "#f2cc60"),
                ContextBenchmarkStatus.ERROR: ("ERROR", "#ef6b73"),
            }
            status, color = statuses[lane.status]
            result = lane.latest_result
            highest = (
                "--"
                if result is None or result.highest_successful_prompt_tokens is None
                else str(result.highest_successful_prompt_tokens)
            )
            return self._line(
                status,
                color,
                (f"highest accepted {highest} tok", "R Run Again", "E Edit"),
            )
        lengths = ",".join(map(str, lane.config.target_lengths))
        return self._line(
            "CONTEXT READY",
            "#5da9e9",
            (f"{lane.config.mode} {lengths}", "R Run", "E Edit"),
        )

    def _concurrency_line(self, state: ApplicationState) -> Text:
        lane = state.concurrency_benchmark
        progress = lane.progress
        if lane.is_active:
            if lane.status is ConcurrencyBenchmarkStatus.BETWEEN_LEVELS:
                remaining = (
                    "--"
                    if progress is None or progress.delay_remaining_seconds is None
                    else f"{progress.delay_remaining_seconds:.1f}s"
                )
                next_level = (
                    "--"
                    if progress is None or progress.next_concurrency_level is None
                    else str(progress.next_concurrency_level)
                )
                return self._line(
                    "BETWEEN",
                    "#f2cc60",
                    (
                        f"next CONC {next_level}",
                        remaining,
                        "Esc Cancel",
                    ),
                )
            level = (
                "--"
                if progress is None or progress.active_concurrency_level is None
                else str(progress.active_concurrency_level)
            )
            completed = progress.completed_request_count if progress else 0
            configured = progress.configured_requests if progress else 0
            active = progress.active_request_count if progress else 0
            throughput = (
                "-- tok/s"
                if progress is None
                else f"{progress.aggregate_output_tokens_per_second:.1f} tok/s"
            )
            status = (
                "CANCELLING"
                if lane.status is ConcurrencyBenchmarkStatus.CANCELLING
                else "RUNNING"
            )
            return self._line(
                status,
                "#f2cc60",
                (
                    f"CONC {level}",
                    f"{completed}/{configured} complete",
                    f"{active} active",
                    throughput,
                    "Esc Cancel",
                ),
            )
        if lane.is_terminal:
            result = lane.latest_result
            peak = (
                max(
                    (
                        level.aggregate_output_tokens_per_second
                        for level in result.levels
                    ),
                    default=None,
                )
                if result is not None
                else None
            )
            peak_text = "--" if peak is None else f"peak {peak:.1f} tok/s"
            statuses = {
                ConcurrencyBenchmarkStatus.COMPLETED: ("COMPLETE", "#5fd38d"),
                ConcurrencyBenchmarkStatus.CANCELLED: ("CANCELLED", "#f2cc60"),
                ConcurrencyBenchmarkStatus.ERROR: ("ERROR", "#ef6b73"),
            }
            status, color = statuses[lane.status]
            return self._line(
                status,
                color,
                (peak_text, "R Run Again", "E Edit"),
            )
        levels = ",".join(map(str, lane.config.concurrency_levels))
        return self._line(
            "READY",
            "#5da9e9",
            (
                f"Concurrency {levels}",
                f"{lane.config.requests_per_level} requests/level",
                "R Run",
            ),
        )

    def _speed_line(
        self, state: ApplicationState, result: SpeedTestResult | None
    ) -> Text:
        speed = state.speed_test
        if speed.is_active:
            metrics = speed.latest_metrics
            if speed.status is SpeedTestStatus.WARMING_UP:
                status = "WARM-UP"
            elif speed.status is SpeedTestStatus.RUNNING:
                status = "RUNNING"
            elif speed.status is SpeedTestStatus.CANCELLING:
                status = "CANCELLING"
            else:
                status = "PREPARING"
            phase = (
                f"{speed.current_run}/{speed.phase_total}"
                if speed.current_phase is not None
                else "--"
            )
            ttft = (
                f"{metrics.ttft_ms:.1f} ms"
                if metrics is not None and metrics.ttft_ms is not None
                else "--"
            )
            output = (
                self._token_label(
                    metrics.completion_tokens,
                    metrics.completion_tokens_estimated,
                )
                if metrics is not None
                else "--"
            )
            speed_label = (
                self._speed_label(
                    metrics.output_tokens_per_second,
                    metrics.completion_tokens_estimated,
                )
                if metrics is not None
                else "-- tok/s"
            )
            return self._line(
                status,
                "#f2cc60",
                (
                    phase,
                    f"TTFT {ttft}",
                    f"{output}/{speed.config.max_tokens} tokens",
                    speed_label,
                    "Esc Cancel",
                    "Ctrl+Q Quit",
                ),
            )
        if result is not None:
            status = result.status.value.replace("_", " ").upper()
            mean = result.output_tokens_per_second.mean
            mean_label = "-- tok/s" if mean is None else f"{mean:.1f} tok/s"
            return self._line(
                status,
                (
                    "#5fd38d"
                    if result.status
                    in {
                        SpeedTestStatus.COMPLETED,
                        SpeedTestStatus.COMPLETED_WITH_ERRORS,
                    }
                    else "#f2cc60"
                ),
                (
                    f"{result.successful_runs}/{result.measured_runs} successful",
                    mean_label,
                    "R Run Again",
                    "E Export",
                    "C Copy",
                    "Esc Back",
                ),
            )
        return self._line(
            "SPEED READY",
            "#5fd38d",
            (
                "Select preset and edit fields",
                "Enter Start",
                "R Refresh",
                "Ctrl+Q Quit",
            ),
        )

    def _chat_line(self, state: ApplicationState, server: ServerConfig) -> Text:
        metrics = state.generation_metrics
        model = (
            state.selected_model_id.rsplit("/", 1)[-1]
            if state.selected_model_id is not None
            else "No model"
        )
        if state.active_generation_id is not None:
            ttft = (
                f"{metrics.ttft_ms:.0f} ms"
                if metrics is not None and metrics.ttft_ms is not None
                else "--"
            )
            tokens = (
                self._token_label(
                    metrics.completion_tokens,
                    metrics.completion_tokens_estimated,
                )
                if metrics is not None
                else "--"
            )
            speed = (
                self._speed_label(
                    metrics.output_tokens_per_second,
                    metrics.completion_tokens_estimated,
                )
                if metrics is not None
                else "--"
            )
            return self._line(
                "GENERATING",
                "#f2cc60",
                (
                    f"TTFT {ttft}",
                    f"{tokens} tokens",
                    speed,
                    "Esc Cancel",
                    "Ctrl+Q Quit",
                ),
            )
        if state.generation_status is GenerationStatus.ERROR:
            return self._line(
                "ERROR",
                "#ef6b73",
                (
                    state.generation_error or "Generation failed",
                    "Enter Retry",
                    "Ctrl+Q Quit",
                ),
            )
        if state.generation_status is GenerationStatus.CANCELLED:
            return self._line(
                "CANCELLED",
                "#f2cc60",
                ("Partial response kept", "Enter Send", "Ctrl+Q Quit"),
            )
        if state.server_status is not ServerStatus.ONLINE:
            return self._line(
                "OFFLINE",
                "#ef6b73",
                (
                    server.name,
                    state.last_error or "Server unavailable",
                    *self._hardware_parts(state),
                    "R Retry",
                    "Ctrl+Q Quit",
                ),
            )
        if state.selected_model_id is None:
            return self._line(
                "NO MODEL",
                "#f2cc60",
                (
                    server.name,
                    *self._hardware_parts(state),
                    "Select a model before sending",
                    "Ctrl+Q Quit",
                ),
            )
        if metrics is not None:
            speed = self._speed_label(
                metrics.output_tokens_per_second,
                metrics.completion_tokens_estimated,
            )
            duration = (
                f"{metrics.total_duration_s:.2f}s"
                if metrics.total_duration_s is not None
                else "--"
            )
            status = "FALLBACK" if state.generation_notice else "READY"
            return self._line(
                status,
                "#5fd38d",
                (
                    server.name,
                    model,
                    speed,
                    duration,
                    "Enter Send",
                    "Ctrl+Q Quit",
                ),
            )
        return self._line(
            "READY",
            "#5fd38d",
            (
                server.name,
                model,
                *self._hardware_parts(state),
                "Enter Send",
                "Ctrl+Q Quit",
            ),
        )

    @staticmethod
    def _token_label(value: int | None, estimated: bool) -> str:
        if value is None:
            return "--"
        return f"{'~' if estimated else ''}{value}"

    @staticmethod
    def _speed_label(value: float | None, estimated: bool) -> str:
        if value is None:
            return "-- tok/s"
        return f"{'~' if estimated else ''}{value:.1f} tok/s"

    @staticmethod
    def _hardware_parts(state: ApplicationState) -> tuple[str, ...]:
        if state.hardware_last_error == "Hardware monitoring disabled":
            return ("Hardware disabled",)
        snapshot = state.hardware_snapshot
        if snapshot is None:
            return ("Hardware unavailable",)
        cpu = f"CPU {format_percentage(snapshot.cpu.utilisation_percent)}"
        if not snapshot.gpus:
            return ("Hardware unavailable", cpu)
        summary = summarize_gpus(snapshot.gpus)
        return (
            summary.display_name,
            f"GPU {format_percentage(summary.utilisation_percent)}",
            "VRAM "
            + format_byte_pair(
                summary.memory_used_bytes,
                summary.memory_total_bytes,
                compact=True,
            ),
            cpu,
        )
