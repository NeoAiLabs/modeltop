"""Immutable domain models for sequential and concurrency benchmarks."""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeltop.chat.models import GenerationMetrics, ThinkingMode
from modeltop.hardware.models import HardwareSnapshot
from modeltop.models import (
    ABSOLUTE_CONTEXT_TEST_TOKENS,
    MAX_BASE_TEXT_CHARACTERS,
    ConcurrencyBenchmarkDefaultsConfig,
    ContextBenchmarkDefaultsConfig,
    DrafterBenchmarkDefaultsConfig,
    ToolCallingBenchmarkDefaultsConfig,
)

from .prompts import DEFAULT_CONCURRENCY_PROMPT, DEFAULT_DRAFTER_PROMPT

type SpeedTestPreset = Literal["quick", "standard", "long", "custom"]
type SpeedTestPhase = Literal["warmup", "measured"]

DEFAULT_SPEED_TEST_PROMPT = (
    "Explain how a modern operating system schedules work across multiple CPU cores. "
    "Cover processes, threads, priorities, preemption, load balancing, and "
    "the tradeoffs between throughput, latency, fairness, and power efficiency "
    "in a detailed, coherent "
    "response."
)


class SpeedTestConfig(BaseModel):
    """Validated immutable input for one sequential benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preset: SpeedTestPreset = "standard"
    prompt: str = DEFAULT_SPEED_TEST_PROMPT
    warmup_runs: int = Field(default=1, ge=0, le=20)
    measured_runs: int = Field(default=5, ge=1, le=100)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = 42
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    continue_on_error: bool = False
    thinking_mode: ThinkingMode = "server_default"

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("temperature", "top_p", "request_timeout_seconds")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value


def speed_test_config_for_preset(
    preset: SpeedTestPreset,
    current: SpeedTestConfig | None = None,
) -> SpeedTestConfig:
    """Return exact preset values; Custom preserves the current validated draft."""
    if preset == "custom":
        return (current or SpeedTestConfig()).model_copy(update={"preset": "custom"})

    values = {
        "quick": (1, 3, 128),
        "standard": (1, 5, 256),
        "long": (1, 3, 1024),
    }
    warmup_runs, measured_runs, max_tokens = values[preset]
    return SpeedTestConfig(
        preset=preset,
        prompt=DEFAULT_SPEED_TEST_PROMPT,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        request_timeout_seconds=300.0,
        continue_on_error=False,
    )


class SpeedTestStatus(StrEnum):
    """Lifecycle status for the latest Speed Test."""

    IDLE = "idle"
    PREPARING = "preparing"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        return self in {
            self.PREPARING,
            self.WARMING_UP,
            self.RUNNING,
            self.CANCELLING,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.CANCELLED,
            self.COMPLETED,
            self.COMPLETED_WITH_ERRORS,
            self.FAILED,
        }


@dataclass(frozen=True, slots=True)
class SpeedTestRunResult:
    """Terminal metrics for one warm-up or measured request."""

    run_number: int
    warmup: bool
    success: bool
    cancelled: bool
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_tokens_estimated: bool
    completion_tokens_estimated: bool
    total_tokens_estimated: bool
    ttft_ms: float | None
    generation_duration_s: float | None
    total_duration_s: float | None
    output_tokens_per_second: float | None
    finish_reason: str | None
    streamed: bool
    response_character_count: int

    @property
    def tokens_estimated(self) -> bool:
        return (
            self.prompt_tokens_estimated
            or self.completion_tokens_estimated
            or self.total_tokens_estimated
        )


@dataclass(frozen=True, slots=True)
class MetricStatistics:
    """Available-value summary for one benchmark metric."""

    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    p95: float | None
    standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class SpeedTestAggregates:
    """Statistics calculated from successful measured requests only."""

    ttft_ms: MetricStatistics
    output_tokens_per_second: MetricStatistics
    total_duration_s: MetricStatistics
    generation_duration_s: MetricStatistics
    prompt_tokens: MetricStatistics
    completion_tokens: MetricStatistics


@dataclass(frozen=True, slots=True)
class SpeedTestResult:
    """One immutable terminal Speed Test retained for this process."""

    run_id: str
    status: SpeedTestStatus
    started_at: datetime
    completed_at: datetime
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    config: SpeedTestConfig
    run_results: tuple[SpeedTestRunResult, ...]
    ttft_ms: MetricStatistics
    output_tokens_per_second: MetricStatistics
    total_duration_s: MetricStatistics
    generation_duration_s: MetricStatistics
    prompt_tokens: MetricStatistics
    completion_tokens: MetricStatistics
    hardware_before: HardwareSnapshot | None
    hardware_after: HardwareSnapshot | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.status.is_terminal:
            raise ValueError("Speed Test results require a terminal status")
        for name, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")

    @property
    def warmup_runs(self) -> int:
        return self.config.warmup_runs

    @property
    def measured_runs(self) -> int:
        return self.config.measured_runs

    @property
    def attempted_warmup_runs(self) -> int:
        return sum(run.warmup for run in self.run_results)

    @property
    def attempted_measured_runs(self) -> int:
        return sum(not run.warmup for run in self.run_results)

    @property
    def successful_runs(self) -> int:
        return sum(not run.warmup and run.success for run in self.run_results)

    @property
    def failed_runs(self) -> int:
        return sum(
            not run.warmup and not run.success and not run.cancelled
            for run in self.run_results
        )

    @property
    def cancelled_runs(self) -> int:
        return sum(not run.warmup and run.cancelled for run in self.run_results)

    @property
    def estimated_measured_metrics(self) -> bool:
        return any(
            not run.warmup and run.success and run.tokens_estimated
            for run in self.run_results
        )


@dataclass(frozen=True, slots=True)
class SpeedTestState:
    """All benchmark state published through the application snapshot."""

    config: SpeedTestConfig
    status: SpeedTestStatus
    run_id: str | None
    current_phase: SpeedTestPhase | None
    current_run: int
    phase_total: int
    latest_metrics: GenerationMetrics | None
    live_output_preview: str
    run_results: tuple[SpeedTestRunResult, ...]
    last_error: str | None
    results: tuple[SpeedTestResult, ...]
    selected_result_id: str | None

    def __post_init__(self) -> None:
        if len(self.live_output_preview) > 2000:
            raise ValueError("live_output_preview exceeds 2,000 characters")

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def latest_result(self) -> SpeedTestResult | None:
        return self.results[-1] if self.results else None

    def result_by_id(self, run_id: str) -> SpeedTestResult | None:
        return next(
            (result for result in self.results if result.run_id == run_id), None
        )


def initial_speed_test_state() -> SpeedTestState:
    """Build the ready benchmark lane with the Standard preset."""
    return SpeedTestState(
        config=SpeedTestConfig(),
        status=SpeedTestStatus.IDLE,
        run_id=None,
        current_phase=None,
        current_run=0,
        phase_total=0,
        latest_metrics=None,
        live_output_preview="",
        run_results=(),
        last_error=None,
        results=(),
        selected_result_id=None,
    )


type ConcurrencyMode = Literal["fixed", "sweep"]
type ConcurrencyPhase = Literal["warmup", "measured"]
type TokenCountMode = Literal["exact", "estimated", "mixed", "unavailable"]
type ConcurrencyRequestState = Literal[
    "queued", "running", "done", "error", "timeout", "cancelled"
]
type SaturationObservationCode = Literal[
    "gpu_saturation",
    "throughput_plateau",
    "latency_degradation",
    "ttft_degradation",
    "request_speed_degradation",
    "reliability_degradation",
    "peak_throughput",
    "lowest_median_ttft",
    "first_failures",
    "output_length_variance",
]
type ContextMode = Literal["fixed", "sweep", "probe", "retrieval"]
type ContextUnit = Literal["tokens", "characters"]
type ContextContentSource = Literal["synthetic", "repeated_text", "built_in_corpus"]
type RetrievalPosition = Literal[
    "beginning", "quarter", "middle", "three_quarters", "end", "random"
]
type RetrievalStatus = Literal["pass", "fail", "ambiguous", "error"]
type ContextRequestState = Literal[
    "queued",
    "building",
    "running",
    "done",
    "rejected",
    "timeout",
    "cancelled",
    "error",
]
type ContextObservationCode = Literal[
    "first_context_rejection",
    "sharp_ttft_increase",
    "output_speed_degradation",
    "vram_growth",
    "possible_prompt_caching",
    "retrieval_degradation",
    "possible_left_truncation",
    "possible_right_truncation",
]

_CONCURRENCY_DEFAULTS = ConcurrencyBenchmarkDefaultsConfig()


class ConcurrencyBenchmarkConfig(BaseModel):
    """Validated immutable input for one concurrency benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ConcurrencyMode = "sweep"
    prompt: str = DEFAULT_CONCURRENCY_PROMPT
    system_prompt: str | None = None
    concurrency_levels: tuple[int, ...] = _CONCURRENCY_DEFAULTS.default_levels
    requests_per_level: int = Field(
        default=_CONCURRENCY_DEFAULTS.requests_per_level, ge=1, le=1000
    )
    warmup_requests: int = Field(
        default=_CONCURRENCY_DEFAULTS.warmup_requests, ge=0, le=1000
    )
    max_tokens: int = Field(default=_CONCURRENCY_DEFAULTS.max_tokens, ge=1)
    temperature: float = Field(
        default=_CONCURRENCY_DEFAULTS.temperature, ge=0.0, le=2.0
    )
    top_p: float = Field(default=_CONCURRENCY_DEFAULTS.top_p, gt=0.0, le=1.0)
    seed: int | None = 42
    request_timeout_seconds: float = Field(
        default=_CONCURRENCY_DEFAULTS.request_timeout_seconds, gt=0.0
    )
    delay_between_levels_seconds: float = Field(
        default=_CONCURRENCY_DEFAULTS.delay_between_levels_seconds, ge=0.0
    )
    stream: Literal[True] = True
    maximum_concurrency: int = Field(
        default=_CONCURRENCY_DEFAULTS.maximum_concurrency, ge=1
    )
    thinking_mode: ThinkingMode = "server_default"

    @field_validator("concurrency_levels", mode="before")
    @classmethod
    def validate_level_input(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("must be a list of integers")
        levels = cast(list[object] | tuple[object, ...], value)
        if any(
            isinstance(level, bool) or not isinstance(level, int) for level in levels
        ):
            raise ValueError("levels must be integers")
        return levels

    @field_validator(
        "requests_per_level",
        "warmup_requests",
        "max_tokens",
        "maximum_concurrency",
        mode="before",
    )
    @classmethod
    def validate_integer_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("must be an integer or None")
        return value

    @field_validator("stream", mode="before")
    @classmethod
    def validate_stream(cls, value: object) -> object:
        if value is not True:
            raise ValueError("must be true")
        return value

    @field_validator("concurrency_levels")
    @classmethod
    def validate_levels(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(level <= 0 for level in value):
            raise ValueError("levels must be greater than zero")
        if len(value) != len(set(value)):
            raise ValueError("levels must be unique")
        return tuple(sorted(value))

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("system_prompt")
    @classmethod
    def normalize_system_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator(
        "temperature",
        "top_p",
        "request_timeout_seconds",
        "delay_between_levels_seconds",
        mode="before",
    )
    @classmethod
    def validate_numeric_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value

    @field_validator(
        "temperature",
        "top_p",
        "request_timeout_seconds",
        "delay_between_levels_seconds",
    )
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @model_validator(mode="after")
    def validate_mode_and_safety_maximum(self) -> Self:
        level_count = len(self.concurrency_levels)
        if self.mode == "fixed" and level_count != 1:
            raise ValueError("fixed mode requires exactly one concurrency level")
        if self.mode == "sweep" and level_count < 2:
            raise ValueError("sweep mode requires at least two concurrency levels")
        if any(level > self.maximum_concurrency for level in self.concurrency_levels):
            raise ValueError("concurrency levels must not exceed maximum_concurrency")
        return self


def concurrency_benchmark_config_from_defaults(
    defaults: ConcurrencyBenchmarkDefaultsConfig,
) -> ConcurrencyBenchmarkConfig:
    """Build the runtime concurrency config from effective YAML defaults."""
    return ConcurrencyBenchmarkConfig(
        mode="sweep",
        prompt=DEFAULT_CONCURRENCY_PROMPT,
        system_prompt=None,
        concurrency_levels=defaults.default_levels,
        requests_per_level=defaults.requests_per_level,
        warmup_requests=defaults.warmup_requests,
        max_tokens=defaults.max_tokens,
        temperature=defaults.temperature,
        top_p=defaults.top_p,
        seed=42,
        request_timeout_seconds=defaults.request_timeout_seconds,
        delay_between_levels_seconds=defaults.delay_between_levels_seconds,
        stream=True,
        maximum_concurrency=defaults.maximum_concurrency,
    )


_CONTEXT_DEFAULTS = ContextBenchmarkDefaultsConfig()
_CONTEXT_RETRIEVAL_KEY = re.compile(r"[A-Za-z0-9-]{1,128}\Z")


class ContextBenchmarkConfig(BaseModel):
    """Validated immutable input for one Context Length benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ContextMode = _CONTEXT_DEFAULTS.default_mode
    target_lengths: tuple[int, ...] = _CONTEXT_DEFAULTS.default_lengths
    context_unit: ContextUnit = _CONTEXT_DEFAULTS.context_unit
    repetitions_per_length: int = Field(
        default=_CONTEXT_DEFAULTS.repetitions_per_length, ge=1, le=100
    )
    warmup_requests: int = Field(
        default=_CONTEXT_DEFAULTS.warmup_requests, ge=0, le=100
    )
    content_source: ContextContentSource = _CONTEXT_DEFAULTS.content_source
    base_text: str | None = _CONTEXT_DEFAULTS.base_text
    random_seed: int = _CONTEXT_DEFAULTS.random_seed
    maximum_output_tokens: int = Field(
        default=_CONTEXT_DEFAULTS.maximum_output_tokens, ge=1, le=32768
    )
    temperature: float = Field(default=_CONTEXT_DEFAULTS.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default=_CONTEXT_DEFAULTS.top_p, gt=0.0, le=1.0)
    seed: int | None = _CONTEXT_DEFAULTS.seed
    request_timeout_seconds: float = Field(
        default=_CONTEXT_DEFAULTS.request_timeout_seconds, gt=0.0
    )
    delay_between_lengths_seconds: float = Field(
        default=_CONTEXT_DEFAULTS.delay_between_lengths_seconds, ge=0.0
    )
    maximum_context_test_tokens: int = Field(
        default=_CONTEXT_DEFAULTS.maximum_context_test_tokens,
        ge=1,
        le=ABSOLUTE_CONTEXT_TEST_TOKENS,
    )
    warning_threshold_tokens: int = Field(
        default=_CONTEXT_DEFAULTS.warning_threshold_tokens, ge=1
    )
    prompt_target_tolerance_percent: float = Field(
        default=_CONTEXT_DEFAULTS.prompt_target_tolerance_percent,
        gt=0.0,
        le=10.0,
    )
    hardware_sample_interval_seconds: float = Field(
        default=_CONTEXT_DEFAULTS.hardware_sample_interval_seconds,
        ge=0.1,
        le=60.0,
    )
    estimated_input_rate_enabled: bool = _CONTEXT_DEFAULTS.estimated_input_rate_enabled
    reuse_prompt: bool = _CONTEXT_DEFAULTS.reuse_prompt
    unique_prompt_suffix_per_run: bool = _CONTEXT_DEFAULTS.unique_prompt_suffix_per_run
    early_stop_enabled: bool = _CONTEXT_DEFAULTS.early_stop_enabled
    continue_after_timeout: bool = _CONTEXT_DEFAULTS.continue_after_timeout
    probe_start_tokens: int = Field(default=_CONTEXT_DEFAULTS.probe.start_tokens, ge=1)
    probe_maximum_tokens: int = Field(
        default=_CONTEXT_DEFAULTS.probe.maximum_tokens, ge=1
    )
    probe_resolution_tokens: int = Field(
        default=_CONTEXT_DEFAULTS.probe.resolution_tokens, ge=1
    )
    retrieval_enabled: bool = _CONTEXT_DEFAULTS.retrieval.enabled
    retrieval_positions: tuple[RetrievalPosition, ...] = (
        _CONTEXT_DEFAULTS.retrieval.positions
    )
    retrieval_key: str | None = _CONTEXT_DEFAULTS.retrieval.key
    retrieval_maximum_output_tokens: int = Field(
        default=_CONTEXT_DEFAULTS.retrieval.maximum_output_tokens,
        ge=1,
        le=32768,
    )
    retrieval_case_insensitive_match: bool = (
        _CONTEXT_DEFAULTS.retrieval.case_insensitive_match
    )
    retrieval_containment_match: bool = _CONTEXT_DEFAULTS.retrieval.containment_match
    retrieval_truncation_detection: bool = (
        _CONTEXT_DEFAULTS.retrieval.truncation_detection
    )
    retrieval_regenerate_per_run: bool = _CONTEXT_DEFAULTS.retrieval.regenerate_per_run
    thinking_mode: ThinkingMode = "server_default"

    @field_validator("target_lengths", mode="before")
    @classmethod
    def validate_target_input(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("target_lengths must be a list of integers")
        targets = cast(list[object] | tuple[object, ...], value)
        if any(
            isinstance(target, bool) or not isinstance(target, int)
            for target in targets
        ):
            raise ValueError("target lengths must be integers")
        return targets

    @field_validator("retrieval_positions", mode="before")
    @classmethod
    def validate_retrieval_positions_input(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("retrieval_positions must be a list")
        return cast(list[object] | tuple[object, ...], value)

    @field_validator(
        "repetitions_per_length",
        "warmup_requests",
        "random_seed",
        "maximum_output_tokens",
        "maximum_context_test_tokens",
        "warning_threshold_tokens",
        "probe_start_tokens",
        "probe_maximum_tokens",
        "probe_resolution_tokens",
        "retrieval_maximum_output_tokens",
        mode="before",
    )
    @classmethod
    def validate_integer_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("must be an integer or None")
        return value

    @field_validator(
        "temperature",
        "top_p",
        "request_timeout_seconds",
        "delay_between_lengths_seconds",
        "prompt_target_tolerance_percent",
        "hardware_sample_interval_seconds",
        mode="before",
    )
    @classmethod
    def validate_numeric_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value

    @field_validator(
        "temperature",
        "top_p",
        "request_timeout_seconds",
        "delay_between_lengths_seconds",
        "prompt_target_tolerance_percent",
        "hardware_sample_interval_seconds",
    )
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @field_validator("retrieval_key")
    @classmethod
    def validate_retrieval_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if _CONTEXT_RETRIEVAL_KEY.fullmatch(normalized) is None:
            raise ValueError(
                "retrieval_key must contain only ASCII letters, digits, and hyphens"
            )
        return normalized

    @field_validator("target_lengths")
    @classmethod
    def validate_targets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one target length is required")
        if any(target <= 0 for target in value):
            raise ValueError("target lengths must be greater than zero")
        if len(value) != len(set(value)):
            raise ValueError("target lengths must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("target lengths must be sorted in ascending order")
        return value

    @field_validator("retrieval_positions")
    @classmethod
    def validate_retrieval_positions(
        cls, value: tuple[RetrievalPosition, ...]
    ) -> tuple[RetrievalPosition, ...]:
        if not value:
            raise ValueError("at least one retrieval position is required")
        if len(value) != len(set(value)):
            raise ValueError("retrieval positions must be unique")
        return value

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> Self:
        target_count = len(self.target_lengths)
        if self.mode == "fixed" and target_count != 1:
            raise ValueError("fixed mode requires exactly one target length")
        if self.mode == "sweep" and target_count < 2:
            raise ValueError("sweep mode requires at least two target lengths")
        if self.mode == "retrieval" and target_count < 1:
            raise ValueError("retrieval mode requires at least one target length")
        if self.retrieval_enabled != (self.mode == "retrieval"):
            raise ValueError("retrieval_enabled must match retrieval mode")
        if self.content_source == "repeated_text":
            if self.base_text is None or not self.base_text.strip():
                raise ValueError("repeated_text requires nonblank base_text")
            if len(self.base_text) > MAX_BASE_TEXT_CHARACTERS:
                raise ValueError("base_text exceeds the character limit")
        elif self.base_text is not None:
            raise ValueError("base_text must be None unless repeated_text is active")
        if self.mode != "retrieval" and self.retrieval_key is not None:
            raise ValueError("retrieval_key must be None outside retrieval mode")
        if self.warning_threshold_tokens > self.maximum_context_test_tokens:
            raise ValueError(
                "warning_threshold_tokens must not exceed the safety maximum"
            )
        if self.probe_maximum_tokens <= self.probe_start_tokens:
            raise ValueError("probe maximum must be greater than probe start")
        if self.probe_start_tokens % self.probe_resolution_tokens:
            raise ValueError("probe start must be a resolution multiple")
        if self.probe_maximum_tokens % self.probe_resolution_tokens:
            raise ValueError("probe maximum must be a resolution multiple")
        if (
            self.probe_resolution_tokens
            >= self.probe_maximum_tokens - self.probe_start_tokens
        ):
            raise ValueError("probe resolution must be smaller than the probe range")
        if self.probe_maximum_tokens > self.maximum_context_test_tokens:
            raise ValueError("probe maximum exceeds the safety maximum")
        output_budget = (
            self.retrieval_maximum_output_tokens
            if self.mode == "retrieval"
            else self.maximum_output_tokens
        )
        if self.mode == "probe":
            if self.context_unit != "tokens":
                raise ValueError("probe mode requires token units")
            if self.probe_maximum_tokens + output_budget > (
                self.maximum_context_test_tokens
            ):
                raise ValueError("probe maximum plus output reserve exceeds safety")
        if self.context_unit == "tokens" and any(
            target + output_budget > self.maximum_context_test_tokens
            for target in self.target_lengths
        ):
            raise ValueError("prompt plus reserved output exceeds safety")
        if self.context_unit == "characters" and any(
            target > self.maximum_context_test_tokens * 4
            for target in self.target_lengths
        ):
            raise ValueError("character target exceeds the configured safety maximum")
        return self


def context_benchmark_config_from_defaults(
    defaults: ContextBenchmarkDefaultsConfig,
) -> ContextBenchmarkConfig:
    """Build the runtime Context config from effective YAML defaults."""
    return ContextBenchmarkConfig(
        mode=defaults.default_mode,
        target_lengths=defaults.default_lengths,
        context_unit=defaults.context_unit,
        repetitions_per_length=defaults.repetitions_per_length,
        warmup_requests=defaults.warmup_requests,
        content_source=defaults.content_source,
        base_text=defaults.base_text
        if defaults.content_source == "repeated_text"
        else None,
        random_seed=defaults.random_seed,
        maximum_output_tokens=defaults.maximum_output_tokens,
        temperature=defaults.temperature,
        top_p=defaults.top_p,
        seed=defaults.seed,
        request_timeout_seconds=defaults.request_timeout_seconds,
        delay_between_lengths_seconds=defaults.delay_between_lengths_seconds,
        maximum_context_test_tokens=defaults.maximum_context_test_tokens,
        warning_threshold_tokens=defaults.warning_threshold_tokens,
        prompt_target_tolerance_percent=defaults.prompt_target_tolerance_percent,
        hardware_sample_interval_seconds=defaults.hardware_sample_interval_seconds,
        estimated_input_rate_enabled=defaults.estimated_input_rate_enabled,
        reuse_prompt=defaults.reuse_prompt,
        unique_prompt_suffix_per_run=defaults.unique_prompt_suffix_per_run,
        early_stop_enabled=defaults.early_stop_enabled,
        continue_after_timeout=defaults.continue_after_timeout,
        probe_start_tokens=defaults.probe.start_tokens,
        probe_maximum_tokens=defaults.probe.maximum_tokens,
        probe_resolution_tokens=defaults.probe.resolution_tokens,
        retrieval_enabled=defaults.retrieval.enabled,
        retrieval_positions=defaults.retrieval.positions,
        retrieval_key=defaults.retrieval.key if defaults.retrieval.enabled else None,
        retrieval_maximum_output_tokens=defaults.retrieval.maximum_output_tokens,
        retrieval_case_insensitive_match=defaults.retrieval.case_insensitive_match,
        retrieval_containment_match=defaults.retrieval.containment_match,
        retrieval_truncation_detection=defaults.retrieval.truncation_detection,
        retrieval_regenerate_per_run=defaults.retrieval.regenerate_per_run,
    )


@dataclass(frozen=True, slots=True)
class PercentileStatistics:
    """Available-value statistics using nearest-rank percentiles."""

    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    standard_deviation: float | None


@dataclass(frozen=True, slots=True)
class ConcurrencyRequestResult:
    """One terminal measured request."""

    request_id: str
    concurrency_level: int
    sequence_number: int
    started_at: float
    first_token_at: float | None
    completed_at: float | None
    queue_wait_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_tokens_estimated: bool
    completion_tokens_estimated: bool
    total_tokens_estimated: bool
    ttft_ms: float | None
    total_latency_seconds: float | None
    generation_duration_seconds: float | None
    output_tokens_per_second: float | None
    finish_reason: str | None
    status_code: int | None
    success: bool
    cancelled: bool
    timed_out: bool
    streamed: bool
    response_character_count: int
    error_type: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if self.concurrency_level <= 0:
            raise ValueError("concurrency_level must be greater than zero")
        if self.sequence_number <= 0:
            raise ValueError("sequence_number must be greater than zero")
        if not math.isfinite(self.started_at):
            raise ValueError("started_at must be finite")
        if not math.isfinite(self.queue_wait_seconds) or self.queue_wait_seconds < 0:
            raise ValueError("queue_wait_seconds must be finite and non-negative")
        for name, value in (
            ("first_token_at", self.first_token_at),
            ("completed_at", self.completed_at),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value is not None and value < self.started_at:
                raise ValueError(f"{name} must not precede started_at")
        if (
            self.first_token_at is not None
            and self.completed_at is not None
            and self.completed_at < self.first_token_at
        ):
            raise ValueError("completed_at must not precede first_token_at")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("ttft_ms", self.ttft_ms),
            ("total_latency_seconds", self.total_latency_seconds),
            ("generation_duration_seconds", self.generation_duration_seconds),
            ("output_tokens_per_second", self.output_tokens_per_second),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.response_character_count < 0:
            raise ValueError("response_character_count must be non-negative")
        if sum((self.success, self.cancelled, self.timed_out)) > 1:
            raise ValueError(
                "success, cancelled, and timed_out must be mutually exclusive"
            )

    @property
    def token_count_is_estimated(self) -> bool:
        """Return whether the completion-token count was estimated."""
        return self.completion_tokens_estimated


@dataclass(frozen=True, slots=True)
class HardwareBenchmarkSummary:
    """Aggregate local-hardware metrics across one measured level."""

    sample_count: int
    average_gpu_utilisation_percent: float | None
    maximum_gpu_utilisation_percent: float | int | None
    average_vram_used_bytes: float | None
    maximum_vram_used_bytes: float | int | None
    average_temperature_celsius: float | None
    maximum_temperature_celsius: float | int | None
    average_power_draw_watts: float | None
    maximum_power_draw_watts: float | int | None
    average_cpu_utilisation_percent: float | None
    maximum_cpu_utilisation_percent: float | int | None
    average_memory_used_bytes: float | None
    maximum_memory_used_bytes: float | int | None


@dataclass(frozen=True, slots=True)
class SaturationObservation:
    """One typed, workload-specific scaling observation."""

    code: SaturationObservationCode
    concurrency: int
    message: str


@dataclass(frozen=True, slots=True)
class ConcurrencyLevelResult:
    """Terminal measured-load summary for one concurrency level."""

    concurrency: int
    configured_requests: int
    attempted_requests: int
    completed_requests: int
    successful_requests: int
    failed_requests: int
    cancelled_requests: int
    timed_out_requests: int
    wall_time_seconds: float
    success_rate_percent: float
    total_prompt_tokens: int
    total_completion_tokens: int
    token_count_mode: TokenCountMode
    requests_per_second: float
    aggregate_output_tokens_per_second: float
    ttft_ms: PercentileStatistics
    latency_seconds: PercentileStatistics
    generation_duration_seconds: PercentileStatistics
    request_output_tokens_per_second: PercentileStatistics
    completion_tokens: PercentileStatistics
    hardware_samples: tuple[HardwareSnapshot, ...]
    hardware_summary: HardwareBenchmarkSummary | None
    output_length_warning: str | None
    partial: bool
    early_stopped: bool
    observations: tuple[SaturationObservation, ...]
    requests: tuple[ConcurrencyRequestResult, ...]

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("concurrency must be greater than zero")
        counts = (
            self.configured_requests,
            self.attempted_requests,
            self.completed_requests,
            self.successful_requests,
            self.failed_requests,
            self.cancelled_requests,
            self.timed_out_requests,
        )
        if any(count < 0 for count in counts):
            raise ValueError("request counts must be non-negative")
        if self.attempted_requests > self.configured_requests:
            raise ValueError("attempted_requests must not exceed configured_requests")
        if self.completed_requests > self.attempted_requests:
            raise ValueError("completed_requests must not exceed attempted_requests")
        if (
            self.successful_requests
            + self.failed_requests
            + self.cancelled_requests
            + self.timed_out_requests
            != self.completed_requests
        ):
            raise ValueError("terminal outcome counts must sum to completed_requests")
        if len(self.requests) != self.completed_requests:
            raise ValueError("requests must contain every terminal attempted row")
        expected_outcomes = (
            sum(request.success for request in self.requests),
            sum(
                not request.success and not request.cancelled and not request.timed_out
                for request in self.requests
            ),
            sum(request.cancelled for request in self.requests),
            sum(request.timed_out for request in self.requests),
        )
        actual_outcomes = (
            self.successful_requests,
            self.failed_requests,
            self.cancelled_requests,
            self.timed_out_requests,
        )
        if actual_outcomes != expected_outcomes:
            raise ValueError("outcome counts must match terminal request rows")
        if any(
            request.concurrency_level != self.concurrency for request in self.requests
        ):
            raise ValueError("request concurrency must match the level")
        request_ids = tuple(request.request_id for request in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique within a level")
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0:
            raise ValueError("wall_time_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.success_rate_percent)
            or not 0 <= self.success_rate_percent <= 100
        ):
            raise ValueError("success_rate_percent must be finite and within 0..100")
        for name, value in (
            ("requests_per_second", self.requests_per_second),
            (
                "aggregate_output_tokens_per_second",
                self.aggregate_output_tokens_per_second,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.total_prompt_tokens < 0 or self.total_completion_tokens < 0:
            raise ValueError("token totals must be non-negative")


class ConcurrencyBenchmarkStatus(StrEnum):
    """Lifecycle status for the latest Concurrency benchmark."""

    IDLE = "idle"
    VALIDATING = "validating"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    BETWEEN_LEVELS = "between_levels"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_active(self) -> bool:
        return self in {
            self.VALIDATING,
            self.WARMING_UP,
            self.RUNNING,
            self.BETWEEN_LEVELS,
            self.CANCELLING,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.CANCELLED, self.ERROR}


@dataclass(frozen=True, slots=True)
class ConcurrencyBenchmarkResult:
    """One terminal, session-only concurrency result."""

    benchmark_id: str
    status: ConcurrencyBenchmarkStatus
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime
    completed_at: datetime
    config: ConcurrencyBenchmarkConfig
    levels: tuple[ConcurrencyLevelResult, ...]
    cancelled: bool
    error: str | None
    warnings: tuple[str, ...]
    observations: tuple[SaturationObservation, ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must not be blank")
        if not self.status.is_terminal:
            raise ValueError("Concurrency benchmark results require a terminal status")
        for name, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.cancelled != (self.status is ConcurrencyBenchmarkStatus.CANCELLED):
            raise ValueError("cancelled must match the terminal status")
        level_values = tuple(level.concurrency for level in self.levels)
        if level_values != tuple(sorted(level_values)):
            raise ValueError("levels must be ordered by concurrency")
        if len(level_values) != len(set(level_values)):
            raise ValueError("levels must have unique concurrency values")

    @property
    def wall_time_seconds(self) -> float:
        """Return total elapsed UTC wall time, including warm-ups and delays."""
        return (self.completed_at - self.started_at).total_seconds()


@dataclass(frozen=True, slots=True)
class ConcurrencyRequestProgress:
    """Latest immutable progress for one configured request."""

    request_id: str
    concurrency_level: int
    sequence_number: int
    state: ConcurrencyRequestState
    queued_at: float
    started_at: float | None
    latest_metrics: GenerationMetrics | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ConcurrencyBenchmarkProgress:
    """Rate-limited immutable progress for the active benchmark."""

    phase: ConcurrencyPhase | None
    active_concurrency_level: int | None
    next_concurrency_level: int | None
    delay_remaining_seconds: float | None
    configured_requests: int
    active_request_count: int
    queued_request_count: int
    completed_request_count: int
    successful_request_count: int
    failed_request_count: int
    timed_out_request_count: int
    cancelled_request_count: int
    elapsed_seconds: float
    aggregate_output_tokens_per_second: float
    requests_per_second: float
    median_ttft_ms: float | None
    request_rows: tuple[ConcurrencyRequestProgress, ...]
    completed_levels: tuple[ConcurrencyLevelResult, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConcurrencyBenchmarkState:
    """Independent Concurrency benchmark lane in the application state."""

    config: ConcurrencyBenchmarkConfig
    status: ConcurrencyBenchmarkStatus
    active_benchmark_id: str | None
    progress: ConcurrencyBenchmarkProgress | None
    benchmark_started_at: datetime | None
    latest_result: ConcurrencyBenchmarkResult | None
    benchmark_error: str | None

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


def initial_concurrency_benchmark_state(
    default_config: ConcurrencyBenchmarkConfig,
) -> ConcurrencyBenchmarkState:
    """Build the ready, independent Concurrency benchmark lane."""
    return ConcurrencyBenchmarkState(
        config=default_config,
        status=ConcurrencyBenchmarkStatus.IDLE,
        active_benchmark_id=None,
        progress=None,
        benchmark_started_at=None,
        latest_result=None,
        benchmark_error=None,
    )


class ContextBenchmarkStatus(StrEnum):
    """Lifecycle status for the latest Context Length benchmark."""

    IDLE = "idle"
    VALIDATING = "validating"
    BUILDING_PROMPT = "building_prompt"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    PROBING = "probing"
    BETWEEN_LENGTHS = "between_lengths"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_active(self) -> bool:
        return self in {
            self.VALIDATING,
            self.BUILDING_PROMPT,
            self.WARMING_UP,
            self.RUNNING,
            self.PROBING,
            self.BETWEEN_LENGTHS,
            self.CANCELLING,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.CANCELLED, self.ERROR}


@dataclass(frozen=True, slots=True)
class ContextPromptMeasurement:
    """Bounded prompt-size provenance for one Context request."""

    requested_length: int
    requested_unit: ContextUnit
    visible_content_characters: int
    body_tokens: int
    system_tokens: int
    instruction_tokens: int
    template_overhead_tokens: int | None
    local_prompt_tokens: int
    server_prompt_tokens: int | None
    builder_difference: int
    server_token_difference: int | None
    server_token_difference_percent: float | None
    counter_name: str
    estimated: bool

    def __post_init__(self) -> None:
        if self.requested_length <= 0:
            raise ValueError("requested_length must be greater than zero")
        for name, value in (
            ("visible_content_characters", self.visible_content_characters),
            ("body_tokens", self.body_tokens),
            ("system_tokens", self.system_tokens),
            ("instruction_tokens", self.instruction_tokens),
            ("local_prompt_tokens", self.local_prompt_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.template_overhead_tokens is not None and (
            self.template_overhead_tokens < 0
        ):
            raise ValueError("template_overhead_tokens must be non-negative")
        if self.server_prompt_tokens is not None and self.server_prompt_tokens < 0:
            raise ValueError("server_prompt_tokens must be non-negative")
        if not self.counter_name.strip():
            raise ValueError("counter_name must not be blank")
        if self.server_token_difference_percent is not None and not math.isfinite(
            self.server_token_difference_percent
        ):
            raise ValueError("server token difference percent must be finite")

    @property
    def effective_prompt_tokens(self) -> int:
        """Prefer server usage while retaining local preflight provenance."""
        return (
            self.server_prompt_tokens
            if self.server_prompt_tokens is not None
            else self.local_prompt_tokens
        )


@dataclass(frozen=True, slots=True)
class ContextRetrievalResult:
    """One bounded deterministic retrieval score."""

    marker: str
    position: RetrievalPosition
    realised_placement_percent: float
    expected_value: str
    raw_preview: str
    normalized_preview: str
    preview_truncated: bool
    status: RetrievalStatus

    def __post_init__(self) -> None:
        if not self.marker.strip():
            raise ValueError("marker must not be blank")
        if not math.isfinite(self.realised_placement_percent) or not (
            0 <= self.realised_placement_percent <= 100
        ):
            raise ValueError("realised placement must be within 0..100")
        if len(self.raw_preview) > 512 or len(self.normalized_preview) > 512:
            raise ValueError("retrieval previews must not exceed 512 characters")


@dataclass(frozen=True, slots=True)
class ContextRequestResult:
    """One terminal measured Context request without retained performance output."""

    request_id: str
    target_length: int
    run_number: int
    sequence_number: int
    measurement: ContextPromptMeasurement
    requested_at: float
    first_token_at: float | None
    completed_at: float | None
    ttft_ms: float | None
    total_latency_seconds: float | None
    generation_duration_seconds: float | None
    completion_tokens: int | None
    completion_tokens_estimated: bool
    output_tokens_per_second: float | None
    estimated_input_tokens_per_second: float | None
    finish_reason: str | None
    status_code: int | None
    streamed: bool
    success: bool
    state: ContextRequestState
    accepted: bool | None
    context_rejected: bool
    timed_out: bool
    cancelled: bool
    retrieval_results: tuple[ContextRetrievalResult, ...]
    response_character_count: int
    error_type: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be blank")
        if self.target_length <= 0 or self.run_number <= 0 or self.sequence_number <= 0:
            raise ValueError("target and sequence values must be positive")
        if self.measurement.requested_length != self.target_length:
            raise ValueError("measurement target must match request target")
        if not math.isfinite(self.requested_at):
            raise ValueError("requested_at must be finite")
        for name, value in (
            ("first_token_at", self.first_token_at),
            ("completed_at", self.completed_at),
        ):
            if value is not None and (
                not math.isfinite(value) or value < self.requested_at
            ):
                raise ValueError(f"{name} must be finite and ordered")
        if (
            self.first_token_at is not None
            and self.completed_at is not None
            and self.completed_at < self.first_token_at
        ):
            raise ValueError("completed_at must not precede first_token_at")
        for name, value in (
            ("ttft_ms", self.ttft_ms),
            ("total_latency_seconds", self.total_latency_seconds),
            ("generation_duration_seconds", self.generation_duration_seconds),
            ("output_tokens_per_second", self.output_tokens_per_second),
            (
                "estimated_input_tokens_per_second",
                self.estimated_input_tokens_per_second,
            ),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.completion_tokens is not None and self.completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if self.response_character_count < 0:
            raise ValueError("response_character_count must be non-negative")
        if self.context_rejected != (self.accepted is False):
            raise ValueError("context_rejected must match rejected acceptance")
        if (
            sum((self.success, self.context_rejected, self.timed_out, self.cancelled))
            > 1
        ):
            raise ValueError("terminal outcome flags must be mutually exclusive")
        expected_state = {
            True: "done",
            self.context_rejected: "rejected",
            self.timed_out: "timeout",
            self.cancelled: "cancelled",
        }
        if self.success and self.state != expected_state[True]:
            raise ValueError("successful requests must use done state")


@dataclass(frozen=True, slots=True)
class ContextObservation:
    """One closed, display-safe Context observation."""

    code: ContextObservationCode
    target_length: int | None
    message: str


@dataclass(frozen=True, slots=True)
class ContextLengthResult:
    """Terminal measured summary for one requested context length."""

    target_length: int
    context_unit: ContextUnit
    effective_total_budget_tokens: int
    configured_requests: int
    attempted_requests: int
    completed_requests: int
    accepted_requests: int
    successful_requests: int
    failed_requests: int
    timed_out_requests: int
    cancelled_requests: int
    context_rejected_requests: int
    success_rate_percent: float
    prompt_tokens: PercentileStatistics
    ttft_ms: PercentileStatistics
    latency_seconds: PercentileStatistics
    output_tokens_per_second: PercentileStatistics
    estimated_input_tokens_per_second: PercentileStatistics
    completion_tokens: PercentileStatistics
    retrieval_attempts_by_position: tuple[tuple[RetrievalPosition, int], ...]
    retrieval_successes_by_position: tuple[tuple[RetrievalPosition, int], ...]
    retrieval_rate_by_position: tuple[tuple[RetrievalPosition, float], ...]
    hardware_samples: tuple[HardwareSnapshot, ...]
    hardware_summary: HardwareBenchmarkSummary | None
    partial: bool
    early_stopped: bool
    observations: tuple[ContextObservation, ...]
    requests: tuple[ContextRequestResult, ...]

    def __post_init__(self) -> None:
        if self.target_length <= 0 or self.effective_total_budget_tokens <= 0:
            raise ValueError("target and effective budget must be positive")
        counts = (
            self.configured_requests,
            self.attempted_requests,
            self.completed_requests,
            self.accepted_requests,
            self.successful_requests,
            self.failed_requests,
            self.timed_out_requests,
            self.cancelled_requests,
            self.context_rejected_requests,
        )
        if any(count < 0 for count in counts):
            raise ValueError("request counts must be non-negative")
        if self.attempted_requests > self.configured_requests:
            raise ValueError("attempted requests exceed configured requests")
        if self.completed_requests != len(self.requests):
            raise ValueError("completed request count must match request rows")
        if self.completed_requests > self.attempted_requests:
            raise ValueError("completed requests exceed attempted requests")
        if not math.isfinite(self.success_rate_percent) or not (
            0 <= self.success_rate_percent <= 100
        ):
            raise ValueError("success rate must be within 0..100")
        sequences = tuple(request.sequence_number for request in self.requests)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(
            set(sequences)
        ):
            raise ValueError("request sequences must be ordered and unique")
        if any(
            request.target_length != self.target_length for request in self.requests
        ):
            raise ValueError("request target must match its length result")


@dataclass(frozen=True, slots=True)
class ContextProbeBounds:
    """Latest deterministic probe bounds and planner stage."""

    highest_confirmed_success: int | None
    first_confirmed_rejection: int | None
    last_candidate: int | None
    resolution_tokens: int
    attempted_targets: tuple[int, ...]
    stage: Literal["exponential", "binary", "complete", "inconclusive"]
    inconclusive_reason: str | None

    def __post_init__(self) -> None:
        if self.resolution_tokens <= 0:
            raise ValueError("probe resolution must be positive")
        if len(self.attempted_targets) != len(set(self.attempted_targets)):
            raise ValueError("probe targets must not be duplicated")
        if self.last_candidate is not None and (
            not self.attempted_targets
            or self.last_candidate != self.attempted_targets[-1]
        ):
            raise ValueError("last candidate must match the attempted target tail")
        if (
            self.highest_confirmed_success is not None
            and self.first_confirmed_rejection is not None
            and self.highest_confirmed_success >= self.first_confirmed_rejection
        ):
            raise ValueError("probe success bound must precede rejection bound")


@dataclass(frozen=True, slots=True)
class ContextBenchmarkResult:
    """One terminal, session-only Context Length benchmark."""

    benchmark_id: str
    status: ContextBenchmarkStatus
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    started_at: datetime
    completed_at: datetime
    config: ContextBenchmarkConfig
    lengths: tuple[ContextLengthResult, ...]
    highest_successful_prompt_tokens: int | None
    first_fully_rejected_prompt_tokens: int | None
    probe_bounds: ContextProbeBounds | None
    possible_truncation: bool
    cancelled: bool
    error: str | None
    warnings: tuple[str, ...]
    observations: tuple[ContextObservation, ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id.strip():
            raise ValueError("benchmark_id must not be blank")
        if not self.status.is_terminal:
            raise ValueError("Context benchmark results require a terminal status")
        for name, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.cancelled != (self.status is ContextBenchmarkStatus.CANCELLED):
            raise ValueError("cancelled must match the terminal status")
        targets = tuple(length.target_length for length in self.lengths)
        if self.config.mode != "probe" and targets != tuple(sorted(targets)):
            raise ValueError("non-probe length results must be target ordered")
        if len(targets) != len(set(targets)):
            raise ValueError("length results must have unique targets")

    @property
    def wall_time_seconds(self) -> float:
        """Return total wall time including builds, warm-ups, and delays."""
        return (self.completed_at - self.started_at).total_seconds()


@dataclass(frozen=True, slots=True)
class ContextPromptBuildProgress:
    """Compact prompt-construction progress without retained content."""

    target_length: int
    context_unit: ContextUnit
    measured_count: int
    fragment_count: int
    iteration: int
    percentage: float


@dataclass(frozen=True, slots=True)
class ContextRequestProgress:
    """Latest active Context request metrics without generated content."""

    request_id: str
    target_length: int
    run_number: int
    sequence_number: int
    state: ContextRequestState
    latest_metrics: GenerationMetrics | None
    response_character_count: int
    error: str | None


@dataclass(frozen=True, slots=True)
class ContextBenchmarkProgress:
    """Rate-limited immutable progress for an active Context benchmark."""

    build: ContextPromptBuildProgress | None
    active_target_length: int | None
    next_target_length: int | None
    target_index: int
    target_count: int
    run_number: int
    configured_runs: int
    delay_remaining_seconds: float | None
    active_request: ContextRequestProgress | None
    completed_lengths: tuple[ContextLengthResult, ...]
    probe_bounds: ContextProbeBounds | None
    probe_stage: Literal["exponential", "binary", "complete", "inconclusive"] | None
    warnings: tuple[str, ...]
    cached_hardware: HardwareSnapshot | None


@dataclass(frozen=True, slots=True)
class ContextBenchmarkState:
    """Independent Context benchmark lane in application state."""

    config: ContextBenchmarkConfig
    status: ContextBenchmarkStatus
    active_benchmark_id: str | None
    progress: ContextBenchmarkProgress | None
    benchmark_started_at: datetime | None
    latest_result: ContextBenchmarkResult | None
    benchmark_error: str | None

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


def initial_context_benchmark_state(
    default_config: ContextBenchmarkConfig,
) -> ContextBenchmarkState:
    """Build the ready, independent Context benchmark lane."""
    return ContextBenchmarkState(
        config=default_config,
        status=ContextBenchmarkStatus.IDLE,
        active_benchmark_id=None,
        progress=None,
        benchmark_started_at=None,
        latest_result=None,
        benchmark_error=None,
    )


type ToolCallingSuite = Literal["core", "full"]
type ToolCallingFailureKind = Literal[
    "wrong_tool",
    "wrong_args",
    "missing_step",
    "forbidden_action",
    "budget_exceeded",
    "timeout",
    "connection_error",
    "server_error",
    "model_crash",
    "evaluator_error",
    "partial",
]
type ToolCallingErrorCode = Literal[
    "dependency_unavailable",
    "upstream_failure",
    "invalid_upstream_result",
]


class ToolCallingBenchmarkConfig(BaseModel):
    """Validated immutable input for one Tool Calling benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite: ToolCallingSuite = "full"
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @field_validator("request_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @property
    def scenario_count(self) -> int:
        return 15 if self.suite == "core" else 69


def tool_calling_benchmark_config_from_defaults(
    defaults: ToolCallingBenchmarkDefaultsConfig,
) -> ToolCallingBenchmarkConfig:
    """Build the runtime Tool Calling config from effective YAML defaults."""
    return ToolCallingBenchmarkConfig(
        suite=defaults.default_suite,
        request_timeout_seconds=defaults.request_timeout_seconds,
    )


class ToolCallingBenchmarkStatus(StrEnum):
    """Lifecycle status for the latest Tool Calling benchmark."""

    IDLE = "idle"
    VALIDATING = "validating"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_active(self) -> bool:
        return self in {
            self.VALIDATING,
            self.RUNNING,
            self.CANCELLING,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {self.COMPLETED, self.CANCELLED, self.ERROR}


class ToolCallingScenarioStatus(StrEnum):
    """Payload-free scenario outcome retained by ModelTop."""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


def _validate_bounded_text(name: str, value: str, maximum: int) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")


def _validate_non_negative_metric(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ToolCallingScenarioProgress:
    """Bounded identity for the currently running upstream scenario."""

    scenario_id: str
    category: str
    title: str
    source_index: int
    started_at_monotonic: float

    def __post_init__(self) -> None:
        _validate_bounded_text("scenario_id", self.scenario_id, 128)
        _validate_bounded_text("category", self.category, 1)
        _validate_bounded_text("title", self.title, 256)
        if self.category not in "ABCDEFGHIJKLMNO":
            raise ValueError("category must be within A..O")
        if not 0 <= self.source_index < 69:
            raise ValueError("source_index must be within 0..68")
        if (
            not math.isfinite(self.started_at_monotonic)
            or self.started_at_monotonic < 0
        ):
            raise ValueError("started_at_monotonic must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ToolCallingScenarioResult:
    """One normalized scenario result with no model-generated payload."""

    scenario_id: str
    category: str
    title: str
    status: ToolCallingScenarioStatus
    points: int
    failure_kind: ToolCallingFailureKind | None
    duration_seconds: float
    ttft_ms: float | None
    turn_count: int
    prompt_tokens: int
    completion_tokens: int
    infrastructure_excluded: bool

    def __post_init__(self) -> None:
        _validate_bounded_text("scenario_id", self.scenario_id, 128)
        _validate_bounded_text("category", self.category, 1)
        _validate_bounded_text("title", self.title, 256)
        if self.category not in "ABCDEFGHIJKLMNO":
            raise ValueError("category must be within A..O")
        expected_points = {
            ToolCallingScenarioStatus.PASS: 2,
            ToolCallingScenarioStatus.PARTIAL: 1,
            ToolCallingScenarioStatus.FAIL: 0,
        }
        if self.points not in {0, 1, 2} or expected_points[self.status] != self.points:
            raise ValueError("scenario status and points are inconsistent")
        _validate_non_negative_metric("duration_seconds", self.duration_seconds)
        _validate_non_negative_metric("ttft_ms", self.ttft_ms)
        if self.turn_count < 0:
            raise ValueError("turn_count must be non-negative")
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")
        infrastructure_kinds = {"timeout", "connection_error", "server_error"}
        if self.infrastructure_excluded != (
            self.status is ToolCallingScenarioStatus.FAIL
            and self.failure_kind in infrastructure_kinds
        ):
            raise ValueError(
                "infrastructure_excluded must match the infrastructure failure kind"
            )


@dataclass(frozen=True, slots=True)
class ToolCallingCategoryScore:
    """Official upstream category aggregate after structural validation."""

    category: str
    label: str
    earned_points: int
    max_points: int
    percent: float
    pass_count: int
    partial_count: int
    fail_count: int

    def __post_init__(self) -> None:
        _validate_bounded_text("category", self.category, 1)
        _validate_bounded_text("label", self.label, 128)
        if self.category not in "ABCDEFGHIJKLMNO":
            raise ValueError("category must be within A..O")
        counts = (self.pass_count, self.partial_count, self.fail_count)
        if (
            self.earned_points < 0
            or self.max_points < 0
            or any(count < 0 for count in counts)
        ):
            raise ValueError("category points and counts must be non-negative")
        if self.earned_points > self.max_points:
            raise ValueError("category earned points exceed capacity")
        if self.max_points != sum(counts) * 2:
            raise ValueError("category capacity must match gradable status counts")
        if self.earned_points != self.pass_count * 2 + self.partial_count:
            raise ValueError("category points must match status counts")
        if not math.isfinite(self.percent) or not 0 <= self.percent <= 100:
            raise ValueError("category percent must be within 0..100")


@dataclass(frozen=True, slots=True)
class ToolCallingBenchmarkProgress:
    """Rate-limited payload-free progress for an active Tool Calling run."""

    configured_count: int
    completed_count: int
    gradable_count: int
    excluded_count: int
    pass_count: int
    partial_count: int
    fail_count: int
    current_scenario: ToolCallingScenarioProgress | None
    elapsed_seconds: float
    cached_hardware: HardwareSnapshot | None

    def __post_init__(self) -> None:
        counts = (
            self.configured_count,
            self.completed_count,
            self.gradable_count,
            self.excluded_count,
            self.pass_count,
            self.partial_count,
            self.fail_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("progress counts must be non-negative")
        if self.configured_count not in {15, 69}:
            raise ValueError("configured_count must be 15 or 69")
        if self.completed_count > self.configured_count:
            raise ValueError("completed_count exceeds configured_count")
        if self.gradable_count + self.excluded_count != self.completed_count:
            raise ValueError("gradable and excluded counts must match completed count")
        if self.pass_count + self.partial_count + self.fail_count != (
            self.gradable_count
        ):
            raise ValueError("status counts must match gradable count")
        _validate_non_negative_metric("elapsed_seconds", self.elapsed_seconds)

    @property
    def completion_rate_percent(self) -> float:
        if not self.completed_count:
            return 0
        return self.gradable_count / self.completed_count * 100


@dataclass(frozen=True, slots=True)
class ToolCallingBenchmarkResult:
    """One bounded normalized terminal result retained only in memory."""

    benchmark_id: str
    upstream_run_id: str | None
    config_fingerprint: str | None
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    integration_commit: str
    upstream_version: str | None
    schema_version: str | None
    config: ToolCallingBenchmarkConfig
    started_at: datetime
    completed_at: datetime
    status: ToolCallingBenchmarkStatus
    cancelled: bool
    error_code: ToolCallingErrorCode | None
    error_message: str | None
    attempted_count: int
    gradable_count: int
    excluded_count: int
    completion_rate_percent: float | None
    final_score: int | None
    total_points: int | None
    max_points: int | None
    rating: str | None
    category_k_gradable: bool
    safety_gate_passed: bool | None
    deployability: int | None
    responsiveness: int | None
    median_turn_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    categories: tuple[ToolCallingCategoryScore, ...]
    scenarios: tuple[ToolCallingScenarioResult, ...]
    warnings: tuple[str, ...]
    hardware_summary: HardwareBenchmarkSummary | None

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("benchmark_id", self.benchmark_id, 128),
            ("server_id", self.server_id, 128),
            ("server_name", self.server_name, 256),
            ("server_endpoint", self.server_endpoint, 512),
            ("model_id", self.model_id, 512),
            ("backend", self.backend, 128),
            ("integration_commit", self.integration_commit, 40),
        ):
            _validate_bounded_text(name, value, maximum)
        for name, value in (
            ("upstream_run_id", self.upstream_run_id),
            ("config_fingerprint", self.config_fingerprint),
            ("upstream_version", self.upstream_version),
            ("schema_version", self.schema_version),
            ("rating", self.rating),
            ("error_message", self.error_message),
        ):
            if value is not None and len(value) > 256:
                raise ValueError(f"{name} exceeds 256 characters")
        if not self.status.is_terminal:
            raise ValueError("Tool Calling results require a terminal status")
        if self.cancelled != (self.status is ToolCallingBenchmarkStatus.CANCELLED):
            raise ValueError("cancelled must match terminal status")
        if (self.error_code is None) != (self.error_message is None):
            raise ValueError("error code and message must appear together")
        if self.status is ToolCallingBenchmarkStatus.ERROR and self.error_code is None:
            raise ValueError("error results require a bounded error")
        if self.status is not ToolCallingBenchmarkStatus.ERROR and (
            self.error_code is not None
        ):
            raise ValueError("only error results may retain an error")
        for name, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.attempted_count not in {15, 69}:
            raise ValueError("attempted_count must be 15 or 69")
        if self.gradable_count < 0 or self.excluded_count < 0:
            raise ValueError("coverage counts must be non-negative")
        if self.gradable_count + self.excluded_count > self.attempted_count:
            raise ValueError("coverage exceeds attempted scenarios")
        if len(self.scenarios) > self.attempted_count or len(self.categories) > 15:
            raise ValueError("result tuples exceed official suite bounds")
        scenario_ids = tuple(row.scenario_id for row in self.scenarios)
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario IDs must be unique")
        category_ids = tuple(row.category for row in self.categories)
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("category IDs must be unique")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
            ("total_tokens", self.total_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt plus completion tokens")
        if self.completion_rate_percent is not None and (
            not math.isfinite(self.completion_rate_percent)
            or not 0 <= self.completion_rate_percent <= 100
        ):
            raise ValueError("completion rate must be within 0..100")
        if self.final_score is not None and not 0 <= self.final_score <= 100:
            raise ValueError("final score must be within 0..100")
        if self.total_points is not None and self.total_points < 0:
            raise ValueError("total points must be non-negative")
        if self.max_points is not None and self.max_points < 0:
            raise ValueError("max points must be non-negative")
        if (
            self.total_points is not None
            and self.max_points is not None
            and self.total_points > self.max_points
        ):
            raise ValueError("total points exceed maximum")
        for name, value in (
            ("deployability", self.deployability),
            ("responsiveness", self.responsiveness),
        ):
            if value is not None and not 0 <= value <= 100:
                raise ValueError(f"{name} must be within 0..100")
        _validate_non_negative_metric("median_turn_ms", self.median_turn_ms)
        if len(self.warnings) > 4 or any(
            len(warning) > 256 for warning in self.warnings
        ):
            raise ValueError("warnings exceed bounded retention limits")

    @property
    def wall_time_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()


@dataclass(frozen=True, slots=True)
class ToolCallingBenchmarkState:
    """Independent Tool Calling lane in the application state."""

    config: ToolCallingBenchmarkConfig
    status: ToolCallingBenchmarkStatus
    active_benchmark_id: str | None
    progress: ToolCallingBenchmarkProgress | None
    benchmark_started_at: datetime | None
    latest_result: ToolCallingBenchmarkResult | None
    benchmark_error: str | None

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


def initial_tool_calling_benchmark_state(
    default_config: ToolCallingBenchmarkConfig,
) -> ToolCallingBenchmarkState:
    """Build the ready, independent Tool Calling benchmark lane."""
    return ToolCallingBenchmarkState(
        config=default_config,
        status=ToolCallingBenchmarkStatus.IDLE,
        active_benchmark_id=None,
        progress=None,
        benchmark_started_at=None,
        latest_result=None,
        benchmark_error=None,
    )


type DrafterPhase = Literal["warmup", "measured"]


class DrafterBenchmarkConfig(BaseModel):
    """Validated immutable input for one Drafter benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = DEFAULT_DRAFTER_PROMPT
    warmup_runs: int = Field(default=1, ge=0, le=20)
    measured_runs: int = Field(default=5, ge=1, le=100)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = 42
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    continue_on_error: bool = False

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("temperature", "top_p", "request_timeout_seconds")
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value


def drafter_benchmark_config_from_defaults(
    defaults: DrafterBenchmarkDefaultsConfig,
) -> DrafterBenchmarkConfig:
    """Build the runtime Drafter config from effective YAML defaults."""
    return DrafterBenchmarkConfig(
        prompt=DEFAULT_DRAFTER_PROMPT,
        warmup_runs=defaults.warmup_runs,
        measured_runs=defaults.measured_runs,
        max_tokens=defaults.max_tokens,
        temperature=defaults.temperature,
        top_p=defaults.top_p,
        seed=defaults.seed,
        request_timeout_seconds=defaults.request_timeout_seconds,
        continue_on_error=defaults.continue_on_error,
    )


class DrafterBenchmarkStatus(StrEnum):
    """Lifecycle status for the latest Drafter benchmark."""

    IDLE = "idle"
    PREPARING = "preparing"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"

    @property
    def is_active(self) -> bool:
        return self in {
            self.PREPARING,
            self.WARMING_UP,
            self.RUNNING,
            self.CANCELLING,
        }

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.CANCELLED,
            self.COMPLETED,
            self.COMPLETED_WITH_ERRORS,
            self.FAILED,
        }


@dataclass(frozen=True, slots=True)
class DrafterRunResult:
    """Terminal metrics for one warm-up or measured Drafter request."""

    run_number: int
    warmup: bool
    success: bool
    cancelled: bool
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_tokens_estimated: bool
    completion_tokens_estimated: bool
    total_tokens_estimated: bool
    draft_tokens: int | None
    accepted_tokens: int | None
    acceptance_rate: float | None
    ttft_ms: float | None
    generation_duration_s: float | None
    total_duration_s: float | None
    output_tokens_per_second: float | None
    finish_reason: str | None
    streamed: bool

    @property
    def tokens_estimated(self) -> bool:
        return (
            self.prompt_tokens_estimated
            or self.completion_tokens_estimated
            or self.total_tokens_estimated
        )

    @property
    def speculative_telemetry_present(self) -> bool:
        return (
            self.draft_tokens is not None
            or self.accepted_tokens is not None
            or self.acceptance_rate is not None
        )


@dataclass(frozen=True, slots=True)
class DrafterAggregates:
    """Statistics calculated from successful measured Drafter requests only."""

    ttft_ms: MetricStatistics
    output_tokens_per_second: MetricStatistics
    total_duration_s: MetricStatistics
    generation_duration_s: MetricStatistics
    prompt_tokens: MetricStatistics
    completion_tokens: MetricStatistics
    draft_tokens: MetricStatistics
    accepted_tokens: MetricStatistics
    acceptance_rate: MetricStatistics


type DrafterObservationCode = Literal[
    "speculative_telemetry_unavailable",
    "partial_speculative_telemetry",
    "low_mean_acceptance_rate",
]


@dataclass(frozen=True, slots=True)
class DrafterObservation:
    """Closed-set observation derived from Drafter run outcomes."""

    code: DrafterObservationCode
    message: str


@dataclass(frozen=True, slots=True)
class DrafterBenchmarkResult:
    """One immutable terminal Drafter benchmark retained for this session."""

    benchmark_id: str
    status: DrafterBenchmarkStatus
    started_at: datetime
    completed_at: datetime
    server_id: str
    server_name: str
    server_endpoint: str
    model_id: str
    backend: str
    config: DrafterBenchmarkConfig
    run_results: tuple[DrafterRunResult, ...]
    ttft_ms: MetricStatistics
    output_tokens_per_second: MetricStatistics
    total_duration_s: MetricStatistics
    generation_duration_s: MetricStatistics
    prompt_tokens: MetricStatistics
    completion_tokens: MetricStatistics
    draft_tokens: MetricStatistics
    accepted_tokens: MetricStatistics
    acceptance_rate: MetricStatistics
    observations: tuple[DrafterObservation, ...]
    hardware_before: HardwareSnapshot | None
    hardware_after: HardwareSnapshot | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.status.is_terminal:
            raise ValueError("Drafter results require a terminal status")
        for name, value in (
            ("started_at", self.started_at),
            ("completed_at", self.completed_at),
        ):
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")

    @property
    def warmup_runs(self) -> int:
        return self.config.warmup_runs

    @property
    def measured_runs(self) -> int:
        return self.config.measured_runs

    @property
    def attempted_warmup_runs(self) -> int:
        return sum(run.warmup for run in self.run_results)

    @property
    def attempted_measured_runs(self) -> int:
        return sum(not run.warmup for run in self.run_results)

    @property
    def successful_runs(self) -> int:
        return sum(not run.warmup and run.success for run in self.run_results)

    @property
    def failed_runs(self) -> int:
        return sum(
            not run.warmup and not run.success and not run.cancelled
            for run in self.run_results
        )

    @property
    def cancelled_runs(self) -> int:
        return sum(not run.warmup and run.cancelled for run in self.run_results)

    @property
    def speculative_telemetry_available(self) -> bool:
        return any(
            not run.warmup and run.success and run.speculative_telemetry_present
            for run in self.run_results
        )


@dataclass(frozen=True, slots=True)
class DrafterBenchmarkProgress:
    """Live progress for an active Drafter benchmark."""

    current_phase: DrafterPhase | None
    current_run: int
    phase_total: int
    completed_measured_runs: int
    configured_measured_runs: int
    latest_metrics: GenerationMetrics | None
    last_error: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DrafterBenchmarkState:
    """Independent latest-only Drafter lane in application state."""

    config: DrafterBenchmarkConfig
    status: DrafterBenchmarkStatus
    active_benchmark_id: str | None
    progress: DrafterBenchmarkProgress | None
    started_at: datetime | None
    latest_result: DrafterBenchmarkResult | None
    benchmark_error: str | None

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


def initial_drafter_benchmark_state(
    default_config: DrafterBenchmarkConfig,
) -> DrafterBenchmarkState:
    """Build the ready, independent Drafter benchmark lane."""
    return DrafterBenchmarkState(
        config=default_config,
        status=DrafterBenchmarkStatus.IDLE,
        active_benchmark_id=None,
        progress=None,
        started_at=None,
        latest_result=None,
        benchmark_error=None,
    )
