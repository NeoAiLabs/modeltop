"""Validated configuration and API domain models."""

import math
import re
from typing import Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modeltop.benchmarks.r0b0bench_contract import (
    R0b0benchLaneId,
    R0b0benchProfile,
    validate_r0b0bench_selection,
)
from modeltop.theme import DEFAULT_CATPPUCCIN_THEME, CatppuccinTheme


def _validate_positive_finite_interval(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("must be finite and greater than zero")
    return value


def format_backend_label(value: str | None) -> str:
    """Normalize a configured hint or discovered model owner for display."""
    if value is None:
        return "--"
    label = value.strip()
    if not label:
        return "--"
    if label.casefold() == "vllm":
        return "vLLM"
    return label


class ApplicationConfig(BaseModel):
    """Application-level runtime settings."""

    model_config = ConfigDict(extra="forbid")

    refresh_interval_seconds: float = 5.0
    request_timeout_seconds: float = 5.0
    default_server: str | None = None
    theme: CatppuccinTheme = DEFAULT_CATPPUCCIN_THEME

    @field_validator("refresh_interval_seconds", "request_timeout_seconds")
    @classmethod
    def validate_positive_finite_interval(cls, value: float) -> float:
        return _validate_positive_finite_interval(value)


class HardwareConfig(BaseModel):
    """Local hardware monitoring settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    refresh_interval_seconds: float = 2.0
    preferred_provider: Literal["auto", "nvml", "nvidia-smi", "disabled"] = "auto"

    @field_validator("refresh_interval_seconds")
    @classmethod
    def validate_positive_finite_interval(cls, value: float) -> float:
        return _validate_positive_finite_interval(value)


class ConcurrencyBenchmarkDefaultsConfig(BaseModel):
    """Validated YAML defaults for the Concurrency benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_levels: tuple[int, ...] = (1, 2, 4, 8)
    requests_per_level: int = Field(default=16, ge=1, le=1000)
    warmup_requests: int = Field(default=2, ge=0, le=1000)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)
    delay_between_levels_seconds: float = Field(default=3.0, ge=0.0)
    maximum_concurrency: int = Field(default=128, ge=1)
    unique_prompt_suffix_per_request: bool = True

    @field_validator("default_levels", mode="before")
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

    @field_validator("default_levels")
    @classmethod
    def validate_levels(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) < 2:
            raise ValueError("at least two default concurrency levels are required")
        if any(level <= 0 for level in value):
            raise ValueError("levels must be greater than zero")
        if len(value) != len(set(value)):
            raise ValueError("levels must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("levels must be sorted in ascending order")
        return value

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
    def validate_safety_maximum(self) -> Self:
        if any(level > self.maximum_concurrency for level in self.default_levels):
            raise ValueError("default levels must not exceed maximum_concurrency")
        return self


ABSOLUTE_CONTEXT_TEST_TOKENS = 1_048_576
MAX_BASE_TEXT_CHARACTERS = 1_048_576
_RETRIEVAL_POSITIONS = {
    "beginning",
    "quarter",
    "middle",
    "three_quarters",
    "end",
    "random",
}
_RETRIEVAL_KEY_PATTERN = re.compile(r"[A-Za-z0-9-]{1,128}\Z")


class ContextProbeDefaultsConfig(BaseModel):
    """Validated YAML defaults for bounded context probing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_tokens: int = Field(default=4096, ge=1)
    maximum_tokens: int = Field(default=131072, ge=1)
    resolution_tokens: int = Field(default=1024, ge=1)

    @field_validator(
        "start_tokens", "maximum_tokens", "resolution_tokens", mode="before"
    )
    @classmethod
    def validate_integer_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @model_validator(mode="after")
    def validate_probe_range(self) -> Self:
        if self.maximum_tokens <= self.start_tokens:
            raise ValueError("maximum_tokens must be greater than start_tokens")
        if self.start_tokens % self.resolution_tokens:
            raise ValueError("start_tokens must be a resolution multiple")
        if self.maximum_tokens % self.resolution_tokens:
            raise ValueError("maximum_tokens must be a resolution multiple")
        if self.resolution_tokens >= self.maximum_tokens - self.start_tokens:
            raise ValueError("resolution_tokens must be smaller than the probe range")
        return self


class ContextRetrievalDefaultsConfig(BaseModel):
    """Validated YAML defaults for deterministic retrieval checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    positions: tuple[
        Literal[
            "beginning",
            "quarter",
            "middle",
            "three_quarters",
            "end",
            "random",
        ],
        ...,
    ] = ("beginning", "middle", "end")
    key: str | None = None
    maximum_output_tokens: int = Field(default=32, ge=1, le=32768)
    case_insensitive_match: bool = False
    containment_match: bool = False
    truncation_detection: bool = True
    regenerate_per_run: bool = True

    @field_validator("maximum_output_tokens", mode="before")
    @classmethod
    def validate_integer_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator("positions", mode="before")
    @classmethod
    def validate_positions_input(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("positions must be a list")
        positions = cast(list[object] | tuple[object, ...], value)
        if not positions:
            raise ValueError("at least one retrieval position is required")
        if any(not isinstance(position, str) for position in positions):
            raise ValueError("retrieval positions must be strings")
        if any(position not in _RETRIEVAL_POSITIONS for position in positions):
            raise ValueError("unknown retrieval position")
        if len(positions) != len(set(positions)):
            raise ValueError("retrieval positions must be unique")
        return positions

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if _RETRIEVAL_KEY_PATTERN.fullmatch(normalized) is None:
            raise ValueError("key must contain only ASCII letters, digits, and hyphens")
        return normalized


class ContextBenchmarkDefaultsConfig(BaseModel):
    """Validated YAML defaults for Context Length benchmarks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_mode: Literal["fixed", "sweep", "probe", "retrieval"] = "sweep"
    default_lengths: tuple[int, ...] = (1024, 4096, 8192, 16384, 32768)
    context_unit: Literal["tokens", "characters"] = "tokens"
    repetitions_per_length: int = Field(default=3, ge=1, le=100)
    warmup_requests: int = Field(default=1, ge=0, le=100)
    content_source: Literal["synthetic", "repeated_text", "built_in_corpus"] = (
        "synthetic"
    )
    base_text: str | None = None
    random_seed: int = 42
    maximum_output_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = 42
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    delay_between_lengths_seconds: float = Field(default=3.0, ge=0.0)
    maximum_context_test_tokens: int = Field(
        default=262144, ge=1, le=ABSOLUTE_CONTEXT_TEST_TOKENS
    )
    warning_threshold_tokens: int = Field(default=65536, ge=1)
    prompt_target_tolerance_percent: float = Field(default=1.0, gt=0.0, le=10.0)
    hardware_sample_interval_seconds: float = Field(default=0.5, ge=0.1, le=60.0)
    estimated_input_rate_enabled: bool = True
    reuse_prompt: bool = True
    unique_prompt_suffix_per_run: bool = False
    early_stop_enabled: bool = True
    continue_after_timeout: bool = True
    probe: ContextProbeDefaultsConfig = Field(
        default_factory=ContextProbeDefaultsConfig
    )
    retrieval: ContextRetrievalDefaultsConfig = Field(
        default_factory=ContextRetrievalDefaultsConfig
    )

    @field_validator("default_lengths", mode="before")
    @classmethod
    def validate_lengths_input(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise ValueError("default_lengths must be a list of integers")
        lengths = cast(list[object] | tuple[object, ...], value)
        if any(
            isinstance(length, bool) or not isinstance(length, int)
            for length in lengths
        ):
            raise ValueError("default_lengths must contain only integers")
        return lengths

    @field_validator(
        "repetitions_per_length",
        "warmup_requests",
        "random_seed",
        "maximum_output_tokens",
        "maximum_context_test_tokens",
        "warning_threshold_tokens",
        mode="before",
    )
    @classmethod
    def validate_integer_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def validate_optional_integer_input(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("must be an integer or null")
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

    @field_validator("base_text")
    @classmethod
    def validate_base_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) > MAX_BASE_TEXT_CHARACTERS:
            raise ValueError("base_text exceeds the character limit")
        return value

    @field_validator("default_lengths")
    @classmethod
    def validate_lengths(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("at least one default context length is required")
        if any(length <= 0 for length in value):
            raise ValueError("default lengths must be greater than zero")
        if len(value) != len(set(value)):
            raise ValueError("default lengths must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("default lengths must be sorted in ascending order")
        return value

    @model_validator(mode="after")
    def validate_context_defaults(self) -> Self:
        if self.warning_threshold_tokens > self.maximum_context_test_tokens:
            raise ValueError(
                "warning_threshold_tokens must not exceed maximum_context_test_tokens"
            )
        if self.probe.maximum_tokens > self.maximum_context_test_tokens:
            raise ValueError(
                "probe.maximum_tokens must not exceed maximum_context_test_tokens"
            )
        if self.context_unit == "tokens" and any(
            length + self.maximum_output_tokens > self.maximum_context_test_tokens
            for length in self.default_lengths
        ):
            raise ValueError(
                "default token length plus output reserve exceeds "
                "maximum_context_test_tokens"
            )
        return self


class R0b0benchBenchmarkDefaultsConfig(BaseModel):
    """Validated YAML defaults for the r0b0bench workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_profile: R0b0benchProfile = "core-subset"
    default_tests: tuple[R0b0benchLaneId, ...] = (
        "canary",
        "latency",
        "concurrency",
        "throughput",
    )
    request_timeout_seconds: float = 600.0
    tokenizer_path: str | None = Field(default=None, repr=False)
    bfcl_python: str | None = Field(default=None, repr=False)
    bfcl_scripts_directory: str | None = Field(default=None, repr=False)
    qa_data_path: str | None = Field(default=None, repr=False)
    ifeval_data_path: str | None = Field(default=None, repr=False)
    humaneval_data_path: str | None = Field(default=None, repr=False)
    gsm8k_data_path: str | None = Field(default=None, repr=False)

    @field_validator("request_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value

    @field_validator("request_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        return _validate_positive_finite_interval(value)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        validate_r0b0bench_selection(self.default_profile, self.default_tests)
        return self


class ToolCallingBenchmarkDefaultsConfig(BaseModel):
    """Validated YAML defaults for the Tool Calling benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_suite: Literal["core", "full"] = "full"
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
        return _validate_positive_finite_interval(value)


class DrafterBenchmarkDefaultsConfig(BaseModel):
    """Validated YAML defaults for the Drafter benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    warmup_runs: int = Field(default=1, ge=0, le=20)
    measured_runs: int = Field(default=5, ge=1, le=100)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = 42
    request_timeout_seconds: float = Field(default=300.0, gt=0.0)
    continue_on_error: bool = False

    @field_validator(
        "warmup_runs",
        "measured_runs",
        "max_tokens",
        mode="before",
    )
    @classmethod
    def validate_integer_input(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed_input(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("must be an integer")
        return value

    @field_validator(
        "temperature",
        "top_p",
        "request_timeout_seconds",
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
    )
    @classmethod
    def validate_finite_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("must be finite")
        return value


class BenchmarksConfig(BaseModel):
    """Validated benchmark defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: ConcurrencyBenchmarkDefaultsConfig = Field(
        default_factory=ConcurrencyBenchmarkDefaultsConfig
    )
    context: ContextBenchmarkDefaultsConfig = Field(
        default_factory=ContextBenchmarkDefaultsConfig
    )
    tool_calling: ToolCallingBenchmarkDefaultsConfig = Field(
        default_factory=ToolCallingBenchmarkDefaultsConfig
    )
    drafter: DrafterBenchmarkDefaultsConfig = Field(
        default_factory=DrafterBenchmarkDefaultsConfig
    )
    r0b0bench: R0b0benchBenchmarkDefaultsConfig = Field(
        default_factory=R0b0benchBenchmarkDefaultsConfig
    )


class ServerConfig(BaseModel):
    """One generic OpenAI-compatible server definition."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    base_url: str
    api_key: str | None = Field(default=None, repr=False)
    backend_hint: str | None = None
    default_model: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip()
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError as error:
            raise ValueError("must be a valid HTTP(S) URL") from error
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("must be an absolute HTTP(S) URL with a host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("must not contain username or password")
        if parsed.query or parsed.fragment:
            raise ValueError("must not contain query or fragment")
        return value

    @property
    def endpoint_label(self) -> str:
        """Return a credential-free, compact endpoint label for display."""
        parsed = urlsplit(self.base_url)
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return f"{parsed.netloc}{path}"

    @property
    def backend_label(self) -> str:
        """Return the normalized configured backend hint."""
        return format_backend_label(self.backend_hint)


class ModelTopConfig(BaseModel):
    """Complete validated ModelTop configuration."""

    model_config = ConfigDict(extra="forbid")

    application: ApplicationConfig
    servers: list[ServerConfig]
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)

    @model_validator(mode="after")
    def validate_server_references(self) -> Self:
        if not self.servers:
            raise ValueError("at least one server is required")
        server_ids = [server.id for server in self.servers]
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("server IDs must be unique")
        default_server = self.application.default_server
        if default_server is not None and default_server not in server_ids:
            raise ValueError("default_server must reference a configured server")
        return self


class DiscoveredModel(BaseModel):
    """One model returned by an OpenAI-compatible models endpoint."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    id: str
    owned_by: str | None = None
    created: int | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value
